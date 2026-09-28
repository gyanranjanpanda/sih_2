"""The coupled well twin: reservoir, wellbore, rod string, pump and surface in one loop.

This is where the integration the problem statement asks for actually happens.
One day of the model runs like this:

1. The reservoir sets the average heated-zone temperature and the oil viscosity
   at the sandface, and with them the deliverability against flowing pressure.
2. The wellbore carries the produced fluid up the tubing and sets the
   temperature, and therefore the viscosity, at every depth along the rods.
3. That viscosity profile sets the drag on the rods, which sets the damping in
   the wave solver, the loads on the card, the float margin and the power.
4. The pump converts the plunger stroke and speed into a liquid rate, reduced
   by slippage and by whatever fillage the reservoir can support.
5. The operating point is solved so the inflow and the pump agree on one
   flowing bottomhole pressure, and the produced fluid carries heat out of the
   heated zone, which feeds back into step 1 tomorrow.

The rate appears on both sides of steps 2 to 5, so each day is closed with a
short fixed-point iteration seeded from the previous day.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.core.logging import get_logger
from app.core.numerics import require_finite
from app.core.units import M3_PER_BBL, STANDARD_GRAVITY_M_PER_S2
from app.twin.fluid import FluidModel
from app.twin.reservoir import (
    CyclePhase,
    InjectionPlan,
    ReservoirModel,
    ReservoirParameters,
)
from app.twin.srp.cards import Diagnosis, DynamometerCard, classify_by_rules, extract_features
from app.twin.srp.floating import FloatAnalysis, analyse_float, build_drag_profile
from app.twin.srp.kinematics import Kinematics, SpeedProfile, StrokeMotion, build_kinematics
from app.twin.srp.power import PowerReport, evaluate_power
from app.twin.srp.pump import PumpPerformance, evaluate_pump, fillage_from_inflow
from app.twin.srp.stress import FatigueCounter, StressReport, analyse_stress
from app.twin.srp.wave import PumpBoundary, RodTaper, WaveSolution, build_taper, solve_forward
from app.twin.surface import SurfaceModel, TankHeatingResult
from app.twin.wellbore import ProductionResult, WellboreModel

LOGGER = get_logger(__name__)

CARD_SAMPLES = 360
"""Samples per cycle used for the dynamometer card."""


class RunMode(StrEnum):
    """How the engine is being driven."""

    SIMULATE = "simulate"
    ASSIMILATE = "assimilate"


@dataclass(frozen=True)
class PumpSetpoint:
    """The controllable settings of the lift system."""

    spm: float
    stroke_length_m: float
    speed_profile: SpeedProfile = field(default_factory=SpeedProfile)

    @classmethod
    def from_config(cls, config: FieldConfig) -> PumpSetpoint:
        """Default setpoint from ``config/field.yaml``."""
        profile = SpeedProfile(
            upstroke_speed_frac=config.srp.hydraulic_upstroke_speed_frac
            if config.srp.unit_type == "hydraulic"
            else 1.0,
            downstroke_speed_frac=config.srp.hydraulic_downstroke_speed_frac
            if config.srp.unit_type == "hydraulic"
            else 1.0,
        )
        return cls(
            spm=config.srp.spm_setpoint,
            stroke_length_m=config.srp.stroke_length_m,
            speed_profile=profile,
        )

    def as_dict(self) -> dict[str, float]:
        """Flat serialisable form."""
        return {
            "spm": self.spm,
            "stroke_length_m": self.stroke_length_m,
            "upstroke_speed_frac": self.speed_profile.upstroke_speed_frac,
            "downstroke_speed_frac": self.speed_profile.downstroke_speed_frac,
            "top_of_downstroke_decel_frac": self.speed_profile.top_of_downstroke_decel_frac,
            "decel_window_frac": self.speed_profile.decel_window_frac,
        }


@dataclass(frozen=True)
class TwinParameters:
    """Everything calibration and assimilation may adjust, in one place.

    Keeping the adjustable set small and explicit is what makes the Ensemble
    Kalman Filter in :mod:`app.twin.assimilation` tractable and what makes the
    reported parameter uncertainty meaningful.
    """

    permeability_md: float
    skin_dimensionless: float
    net_pay_thickness_m: float
    thermal_loss_multiplier: float
    vit_heat_transfer_w_per_m2_k: float
    rod_damping_factor: float
    cycle_energy_retention_frac: float
    productivity_decline_per_cycle_frac: float
    water_cut_initial_frac: float
    water_cut_growth_per_cycle_frac: float

    @classmethod
    def from_config(cls, config: FieldConfig) -> TwinParameters:
        """Defaults taken from configuration."""
        return cls(
            permeability_md=config.reservoir.permeability_md,
            skin_dimensionless=config.reservoir.skin_dimensionless,
            net_pay_thickness_m=config.reservoir.net_pay_thickness_m,
            thermal_loss_multiplier=1.0,
            vit_heat_transfer_w_per_m2_k=config.wellbore.overall_heat_transfer_w_per_m2_k,
            rod_damping_factor=config.srp.damping_factor_dimensionless,
            cycle_energy_retention_frac=config.reservoir.cycle_energy_retention_frac,
            productivity_decline_per_cycle_frac=(
                config.reservoir.productivity_decline_per_cycle_frac
            ),
            water_cut_initial_frac=config.reservoir.water_cut_initial_frac,
            water_cut_growth_per_cycle_frac=config.reservoir.water_cut_growth_per_cycle_frac,
        )

    def to_reservoir_parameters(self) -> ReservoirParameters:
        """The subset the reservoir model consumes."""
        return ReservoirParameters(
            permeability_md=self.permeability_md,
            skin_dimensionless=self.skin_dimensionless,
            net_pay_thickness_m=self.net_pay_thickness_m,
            thermal_loss_multiplier=self.thermal_loss_multiplier,
            cycle_energy_retention_frac=self.cycle_energy_retention_frac,
            productivity_decline_per_cycle_frac=self.productivity_decline_per_cycle_frac,
            water_cut_initial_frac=self.water_cut_initial_frac,
            water_cut_growth_per_cycle_frac=self.water_cut_growth_per_cycle_frac,
        )

    def as_dict(self) -> dict[str, float]:
        """Flat serialisable form."""
        return asdict(self)

    @classmethod
    def from_vector(
        cls, names: list[str], values: NDArray[np.float64], base: TwinParameters
    ) -> TwinParameters:
        """Rebuild from an ordered vector, leaving unlisted fields at their base value."""
        payload = base.as_dict()
        for name, value in zip(names, values, strict=True):
            if name not in payload:
                raise PhysicsDomainError(f"Unknown twin parameter '{name}'.", parameter=name)
            payload[name] = float(value)
        return cls(**payload)

    def to_vector(self, names: list[str]) -> NDArray[np.float64]:
        """Extract an ordered vector of the named fields."""
        payload = self.as_dict()
        return np.asarray([payload[name] for name in names], dtype=float)


@dataclass(frozen=True)
class OperatingPoint:
    """Where the inflow performance curve and the outflow capacity meet."""

    bottomhole_pressure_kpa: float
    liquid_rate_m3_per_day: float
    is_pumped_off: bool
    is_flowing: bool


@dataclass(frozen=True)
class LiftState:
    """Everything the lift system is doing at one operating point."""

    wave: WaveSolution
    motion: StrokeMotion
    float_analysis: FloatAnalysis
    stress: StressReport
    power: PowerReport
    pump: PumpPerformance
    diagnosis: Diagnosis
    production_profile: ProductionResult
    viscosity_at_pump_pa_s: float
    viscosity_profile_pa_s: NDArray[np.float64]
    fluid_load_n: float
    bottomhole_pressure_kpa: float
    liquid_rate_m3_per_day: float
    rate_limit_m3_per_day: float
    is_pumped_off: bool
    is_flowing: bool


@dataclass
class DayRecord:
    """One simulated day, flattened for storage, the API and the dashboard."""

    day: float
    cycle_number: int
    phase: str
    values: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str]:
        """Flat dictionary including the identifying fields."""
        payload: dict[str, float | str] = {
            "day": self.day,
            "cycle_number": float(self.cycle_number),
            "phase": self.phase,
        }
        payload.update(self.values)
        return payload


@dataclass
class CycleResult:
    """Outcome of one complete CSS cycle."""

    cycle_number: int
    days: list[DayRecord]
    oil_m3: float
    water_m3: float
    steam_m3_cwe: float
    steam_oil_ratio: float
    injection_energy_j: float
    lift_energy_kwh: float
    surface_energy_kwh: float
    energy_kwh_per_m3: float
    energy_kwh_per_bbl: float
    float_event_days: int
    mean_float_margin: float
    peak_load_n: float
    mean_fillage_frac: float
    max_stress_utilisation_frac: float
    fatigue_damage: float
    cutoff_day: float
    cutoff_reason: str
    constraint_violations: list[str]
    revenue_usd: float
    energy_cost_usd: float

    @property
    def is_feasible(self) -> bool:
        """Whether the cycle ran without violating any hard constraint."""
        return not self.constraint_violations

    def summary(self) -> dict[str, float | str | bool]:
        """Compact summary for the KPI table and the API."""
        return {
            "cycle_number": self.cycle_number,
            "oil_m3": self.oil_m3,
            "water_m3": self.water_m3,
            "steam_m3_cwe": self.steam_m3_cwe,
            "steam_oil_ratio": self.steam_oil_ratio,
            "energy_kwh_per_bbl": self.energy_kwh_per_bbl,
            "float_event_days": self.float_event_days,
            "mean_float_margin": self.mean_float_margin,
            "peak_load_kn": self.peak_load_n / 1000.0,
            "mean_fillage_frac": self.mean_fillage_frac,
            "max_stress_utilisation_frac": self.max_stress_utilisation_frac,
            "fatigue_damage": self.fatigue_damage,
            "cutoff_day": self.cutoff_day,
            "cutoff_reason": self.cutoff_reason,
            "feasible": self.is_feasible,
            "revenue_usd": self.revenue_usd,
            "energy_cost_usd": self.energy_cost_usd,
        }


class CoupledWellTwin:
    """Daily-step coupled model of one well.

    Args:
        config: Well configuration.
        parameters: Calibratable parameter set. Defaults to the configuration.
        rod_solver_interval_days: How often the rod wave equation is re-solved
            during a cycle. The rod state changes only as fast as the viscosity
            at the pump does, which is a slow drift, so re-solving every few
            days is accurate and much faster. Set to 1 for a full-resolution run.
        wave_cycles: Pumping cycles the wave solver runs before it reports.
    """

    def __init__(
        self,
        config: FieldConfig,
        parameters: TwinParameters | None = None,
        rod_solver_interval_days: int = 5,
        wave_cycles: int = 5,
    ) -> None:
        self.config = config
        self.parameters = parameters or TwinParameters.from_config(config)
        self.rod_solver_interval_days = max(int(rod_solver_interval_days), 1)
        self.wave_cycles = max(int(wave_cycles), 2)

        self.fluid = FluidModel(config.fluid)
        self.reservoir = ReservoirModel(
            config, self.fluid, self.parameters.to_reservoir_parameters()
        )
        self.wellbore = WellboreModel(
            config,
            self.fluid,
            heat_transfer_override_w_per_m2_k=(self.parameters.vit_heat_transfer_w_per_m2_k),
        )
        self.surface = SurfaceModel(config, self.fluid)
        # A coarser wellbore grid for the fast cycle path, where only the
        # wellhead temperature is needed rather than the whole rod profile.
        self._fast_wellbore = WellboreModel(
            config,
            self.fluid,
            heat_transfer_override_w_per_m2_k=self.parameters.vit_heat_transfer_w_per_m2_k,
            node_count=21,
        )
        self.taper: RodTaper = build_taper(config.srp)
        self.kinematics: Kinematics = build_kinematics(config.srp)
        self.fatigue = FatigueCounter()
        self._last_lift: LiftState | None = None
        self._last_rate_m3_per_day = 5.0

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Return the twin to virgin reservoir conditions and zero fatigue."""
        self.reservoir.reset()
        self.fatigue = FatigueCounter()
        self._last_lift = None
        self._last_rate_m3_per_day = 5.0

    def with_parameters(self, parameters: TwinParameters) -> CoupledWellTwin:
        """A fresh twin with the same configuration and a different parameter set."""
        return CoupledWellTwin(
            self.config,
            parameters,
            rod_solver_interval_days=self.rod_solver_interval_days,
            wave_cycles=self.wave_cycles,
        )

    @property
    def minimum_intake_pressure_kpa(self) -> float:
        """Pump intake pressure when the annulus is pumped down to the pump intake."""
        return self.config.wellbore.wellhead_pressure_kpa

    def maximum_intake_pressure_kpa(self, temperature_c: float, water_cut_frac: float) -> float:
        """Pump intake pressure when the annulus is full to surface.

        Equation: p = p_casing + rho g L_pump.
        Units: kPa.
        A rod pump cannot hold a higher intake pressure than this: once the
        annulus is full the level cannot rise any further, and any additional
        deliverability leaves the well by flowing rather than by being pumped.
        Leaving this bound out lets the operating-point solver return a flowing
        pressure above the reservoir pressure, which then reads as zero inflow
        on the following day.
        """
        density = float(self.fluid.mixture_density_kg_per_m3(temperature_c, water_cut_frac))
        column_kpa = density * STANDARD_GRAVITY_M_PER_S2 * self.config.srp.pump_depth_m / 1000.0
        return self.config.wellbore.wellhead_pressure_kpa + column_kpa

    # ------------------------------------------------------------ lift system
    def _fluid_load_n(
        self, bottomhole_pressure_kpa: float, temperature_c: float, water_cut_frac: float
    ) -> float:
        """Load the plunger carries with the travelling valve shut.

        Equation: F = A_plunger (p_discharge - p_intake), with the discharge
        pressure being the wellhead pressure plus the static head of the tubing
        fluid column down to the pump.
        Units: N.
        """
        density = float(self.fluid.mixture_density_kg_per_m3(temperature_c, water_cut_frac))
        head_kpa = density * STANDARD_GRAVITY_M_PER_S2 * self.config.srp.pump_depth_m / 1000.0
        discharge_kpa = self.config.wellbore.wellhead_pressure_kpa + head_kpa
        difference_pa = max(discharge_kpa - bottomhole_pressure_kpa, 0.0) * 1000.0
        return self.config.srp.plunger_area_m2 * difference_pa

    def solve_operating_point(
        self,
        pump_capacity_m3_per_day: float,
        production_days: float,
        temperature_c: float,
        water_cut_frac: float,
    ) -> OperatingPoint:
        """Find the flowing bottomhole pressure where inflow matches outflow.

        Three regimes are possible and the solver picks between them:

        * **Flowing.** Right after a soak the heated zone is still at close to
          the injection pressure. Even with the annulus full to surface the
          reservoir delivers more than the pump can take, so the surplus leaves
          the well by natural flow. The intake pressure sits at the full-column
          value and the rate is the deliverability there, capped by what the
          well site can actually handle: the crude goes into a tank and leaves
          by road tanker, so the site has a real throughput limit.
        * **Pumped, level between.** The reservoir can supply the pump but not
          more. The annulus level settles where inflow equals pump capacity.
        * **Pumped off.** The reservoir cannot keep up even with the level
          drawn down to the pump intake. The pump runs partly empty, which is
          what produces fluid pound.
        """
        low_kpa = self.minimum_intake_pressure_kpa
        high_kpa = min(
            self.maximum_intake_pressure_kpa(temperature_c, water_cut_frac),
            self.reservoir.state.reservoir_pressure_kpa,
        )
        facility_limit = self.config.surface.max_well_rate_m3_per_day

        if high_kpa <= low_kpa:
            return OperatingPoint(low_kpa, 0.0, True, False)

        # The inflow model is exactly linear in drawdown: every term other than
        # the pressure difference is independent of the flowing pressure. One
        # evaluation therefore gives the whole inflow performance line, and the
        # operating point follows in closed form. Searching for it would be both
        # slower and less accurate.
        reservoir_pressure_kpa = self.reservoir.state.reservoir_pressure_kpa
        rate_at_pump_intake = self.reservoir.deliverability_m3_per_day(low_kpa, production_days)
        drawdown_at_pump_intake = reservoir_pressure_kpa - low_kpa
        if drawdown_at_pump_intake <= 0.0 or rate_at_pump_intake <= 0.0:
            return OperatingPoint(low_kpa, 0.0, True, False)
        productivity_m3_per_day_kpa = rate_at_pump_intake / drawdown_at_pump_intake

        rate_at_full_column = max(
            productivity_m3_per_day_kpa * (reservoir_pressure_kpa - high_kpa), 0.0
        )
        if rate_at_full_column > pump_capacity_m3_per_day:
            return OperatingPoint(
                bottomhole_pressure_kpa=high_kpa,
                liquid_rate_m3_per_day=min(rate_at_full_column, facility_limit),
                is_pumped_off=False,
                is_flowing=True,
            )
        if rate_at_pump_intake <= pump_capacity_m3_per_day:
            return OperatingPoint(
                bottomhole_pressure_kpa=low_kpa,
                liquid_rate_m3_per_day=min(rate_at_pump_intake, facility_limit),
                is_pumped_off=True,
                is_flowing=False,
            )
        pressure_kpa = reservoir_pressure_kpa - pump_capacity_m3_per_day / (
            productivity_m3_per_day_kpa
        )
        pressure_kpa = float(min(max(pressure_kpa, low_kpa), high_kpa))
        return OperatingPoint(
            bottomhole_pressure_kpa=pressure_kpa,
            liquid_rate_m3_per_day=min(pump_capacity_m3_per_day, facility_limit),
            is_pumped_off=False,
            is_flowing=False,
        )

    def evaluate_lift(
        self,
        setpoint: PumpSetpoint,
        sandface_temp_c: float,
        water_cut_frac: float,
        elapsed_days: float,
        production_days: float,
        gas_interference_frac: float | None = None,
        clearance_multiplier: float = 1.0,
        iterations: int = 3,
    ) -> LiftState:
        """Solve the lift system and the inflow for one consistent operating point.

        The liquid rate appears in the wellbore heat balance, in the drag on the
        rods and in the pump fillage, so it is closed by a short fixed-point
        iteration seeded from the previous day. Three passes are enough: the
        rate changes by well under a percent after the second.
        """
        srp = self.config.srp
        motion = self.kinematics.motion(
            setpoint.spm, setpoint.stroke_length_m, setpoint.speed_profile, CARD_SAMPLES
        )
        gas_frac = (
            srp.gas_interference_frac if gas_interference_frac is None else gas_interference_frac
        )

        rate_m3_per_day = max(self._last_rate_m3_per_day, 0.05)
        plunger_stroke_m = 0.9 * setpoint.stroke_length_m
        solution: WaveSolution | None = None
        production: ProductionResult | None = None
        viscosity_profile = np.full(self.taper.node_count, 0.05)
        performance: PumpPerformance | None = None
        bottomhole_kpa = self.minimum_intake_pressure_kpa
        pumped_off = True
        flowing = False
        deliverability = 0.0
        fillage = 1.0
        fluid_load_n = 0.0

        for _ in range(max(iterations, 1)):
            displacement = srp.plunger_area_m2 * plunger_stroke_m * motion.spm * 1440.0
            operating = self.solve_operating_point(
                pump_capacity_m3_per_day=displacement,
                production_days=production_days + 0.5,
                temperature_c=sandface_temp_c,
                water_cut_frac=water_cut_frac,
            )
            bottomhole_kpa = operating.bottomhole_pressure_kpa
            deliverability = operating.liquid_rate_m3_per_day
            pumped_off = operating.is_pumped_off
            flowing = operating.is_flowing
            fillage = fillage_from_inflow(deliverability, displacement)

            production = self.wellbore.produce(
                sandface_temp_c=sandface_temp_c,
                liquid_rate_m3_per_day=max(rate_m3_per_day, 0.05),
                water_cut_frac=water_cut_frac,
                elapsed_days=max(elapsed_days, 0.1),
            )
            viscosity_profile = np.interp(
                self.taper.depth_m, production.depth_m, production.viscosity_pa_s
            )
            drag = build_drag_profile(
                self.taper,
                self.config.wellbore.tubing_inner_diameter_m,
                viscosity_profile,
                max(rate_m3_per_day, 0.05),
            )
            fluid_load_n = self._fluid_load_n(
                bottomhole_kpa, production.pump_intake_temp_c, water_cut_frac
            )
            fluid_density = float(
                self.fluid.mixture_density_kg_per_m3(production.pump_intake_temp_c, water_cut_frac)
            )
            solution = solve_forward(
                motion=motion,
                taper=self.taper,
                config=srp,
                pump=PumpBoundary(
                    fluid_load_n=fluid_load_n,
                    fillage_frac=fillage,
                    gas_interference_frac=gas_frac,
                    stroke_length_m=plunger_stroke_m,
                ),
                damping_factor_dimensionless=self.parameters.rod_damping_factor,
                fluid_density_kg_per_m3=fluid_density,
                cycles=self.wave_cycles,
                drag=drag,
            )
            plunger_stroke_m = max(solution.pump_stroke_m, 0.05)
            performance = evaluate_pump(
                config=srp,
                plunger_stroke_m=plunger_stroke_m,
                spm=motion.spm,
                viscosity_at_pump_pa_s=float(viscosity_profile[-1]),
                pressure_difference_pa=fluid_load_n / srp.plunger_area_m2,
                fillage_frac=fillage,
                gas_interference_frac=gas_frac,
                clearance_multiplier=clearance_multiplier,
            )
            rate_m3_per_day = (
                deliverability
                if flowing
                else min(performance.liquid_rate_m3_per_day, deliverability)
            )

        assert solution is not None and production is not None and performance is not None
        self._last_rate_m3_per_day = rate_m3_per_day

        float_analysis = analyse_float(
            taper=self.taper,
            config=srp,
            motion=motion,
            viscosity_profile_pa_s=viscosity_profile,
            fluid_density_kg_per_m3=float(
                self.fluid.mixture_density_kg_per_m3(production.pump_intake_temp_c, water_cut_frac)
            ),
            tubing_inner_diameter_m=self.config.wellbore.tubing_inner_diameter_m,
            liquid_rate_m3_per_day=max(rate_m3_per_day, 0.05),
            surface_load_n=solution.surface_load_n,
        )
        stress = analyse_stress(solution, self.taper, srp)
        power = evaluate_power(srp, motion, solution.surface_load_n, max(rate_m3_per_day, 1e-6))
        card = DynamometerCard(
            solution.pump_position_m, solution.pump_load_n, False, motion.cycle_time_s
        )
        diagnosis = classify_by_rules(
            extract_features(card),
            expected_fluid_load_n=fluid_load_n,
            minimum_fillage_frac=srp.minimum_fillage_frac,
            minimum_load_threshold_n=srp.minimum_polished_rod_load_n,
            surface_float_index=float_analysis.float_index,
        )
        return LiftState(
            wave=solution,
            motion=motion,
            float_analysis=float_analysis,
            stress=stress,
            power=power,
            pump=performance,
            diagnosis=diagnosis,
            production_profile=production,
            viscosity_at_pump_pa_s=float(viscosity_profile[-1]),
            viscosity_profile_pa_s=viscosity_profile,
            fluid_load_n=fluid_load_n,
            bottomhole_pressure_kpa=bottomhole_kpa,
            liquid_rate_m3_per_day=rate_m3_per_day,
            rate_limit_m3_per_day=max(deliverability, 0.0),
            is_pumped_off=pumped_off,
            is_flowing=flowing,
        )

    def static_plunger_stroke_m(self, stroke_length_m: float, fluid_load_n: float) -> float:
        """Plunger stroke from the static rod stretch, without solving the wave equation.

        Equation: S_plunger = S_surface - F_fluid * sum(L_i / (E A_i)).
        Units: m.
        The stretch term is exact for a static string and it is what dominates
        the difference between the surface and the downhole stroke at the slow
        speeds a heavy-oil well runs at. The full wave solver adds the dynamic
        overtravel on top; a test checks the two agree to within a few percent
        at the operating speeds used here.
        """
        compliance_m_per_n = sum(
            section.length_m / (self.config.srp.steel_youngs_modulus_pa * section.area_m2)
            for section in self.config.srp.rod_sections
        )
        return max(stroke_length_m - fluid_load_n * compliance_m_per_n, 0.05)

    def _string_drag_coefficient_n_s_per_m(
        self, production: ProductionResult, liquid_rate_m3_per_day: float
    ) -> float:
        """Total linear drag coefficient of the rod string, in N.s/m.

        Integrates the node-wise annular drag linearisation over the string.
        Used by the fast cycle path so its energy estimate includes the viscous
        term without solving the wave equation.
        """
        viscosity = np.interp(self.taper.depth_m, production.depth_m, production.viscosity_pa_s)
        drag = build_drag_profile(
            self.taper,
            self.config.wellbore.tubing_inner_diameter_m,
            viscosity,
            liquid_rate_m3_per_day,
        )
        return float(np.sum(drag.linear_coefficient_n_s_per_m2) * self.taper.step_m)

    def run_cycle_fast(
        self,
        plan: InjectionPlan,
        setpoint: PumpSetpoint,
        cycle_number: int = 1,
        max_production_days: float | None = None,
    ) -> CycleResult:
        """Run a cycle with the reservoir, the wellbore and an analytic pump only.

        The rod wave equation is not solved. The pump capacity comes from the
        static plunger stroke, which is accurate at these speeds, and the rod
        diagnostics are left out entirely.

        This exists because calibration and the CSS surrogate need hundreds of
        cycle evaluations of reservoir quantities, and solving the rod string
        every few days for each one would make them unusable. The quantities it
        does not compute are not reported, so nothing downstream can mistake a
        fast run for a full one: the float margin, the stress utilisation and
        the peak load all come back as zero and the result is marked infeasible
        only on constraints it genuinely evaluated.
        """
        config = self.config
        days_limit = max_production_days or config.css.production_days_max
        violations: list[str] = []

        self.reservoir.begin_cycle(cycle_number)
        injection = self.wellbore.inject_steam(
            wellhead_pressure_kpa=plan.injection_pressure_kpa,
            wellhead_quality_frac=plan.steam_quality_frac,
            rate_m3_per_day_cwe=plan.injection_rate_m3_per_day_cwe,
            elapsed_days=max(plan.injection_days, 0.5),
        )
        sandface_injection_pressure_kpa = float(injection.pressure_kpa[-1])
        if sandface_injection_pressure_kpa > config.reservoir.fracture_pressure_kpa:
            violations.append(
                f"Sandface injection pressure {sandface_injection_pressure_kpa:.0f} kPa exceeds "
                f"the fracture pressure {config.reservoir.fracture_pressure_kpa:.0f} kPa."
            )
        generator_limit = self.surface.generator_rate_limit_m3_per_day_cwe(
            plan.injection_pressure_kpa, plan.steam_quality_frac
        )
        if plan.injection_rate_m3_per_day_cwe > generator_limit * 1.001:
            violations.append(
                f"Injection rate {plan.injection_rate_m3_per_day_cwe:.0f} m3/day exceeds the "
                f"steam generator capacity of {generator_limit:.0f} m3/day."
            )
        self.reservoir.step_injection(
            plan=plan,
            days=plan.injection_days,
            sandface_heat_rate_w=injection.sandface_heat_rate_w,
            sandface_steam_temp_c=injection.sandface_temp_c,
        )
        injection_energy_j = self.surface.steam_generation_fuel_energy_j(
            plan.steam_volume_m3_cwe, plan.injection_pressure_kpa, plan.steam_quality_frac
        )
        self.reservoir.step_soak(plan.soak_days)
        self.reservoir.state.phase_day = 0.0
        self.reservoir.state.phase = CyclePhase.PRODUCTION

        motion = self.kinematics.motion(
            setpoint.spm, setpoint.stroke_length_m, setpoint.speed_profile, 64
        )
        surface_energy_kwh = 0.0
        lift_energy_kwh = 0.0
        fillages: list[float] = []
        cutoff_day = 0.0
        cutoff_reason = "production day limit reached"
        economics = config.economics
        marginal_reference: float | None = None

        for day_index in range(int(days_limit)):
            state = self.reservoir.state
            fluid_load_n = self._fluid_load_n(
                self.minimum_intake_pressure_kpa,
                state.average_heated_temp_c,
                state.water_cut_frac,
            )
            plunger_stroke_m = self.static_plunger_stroke_m(setpoint.stroke_length_m, fluid_load_n)
            displacement = config.srp.plunger_area_m2 * plunger_stroke_m * motion.spm * 1440.0
            operating = self.solve_operating_point(
                pump_capacity_m3_per_day=displacement,
                production_days=float(day_index) + 0.5,
                temperature_c=state.average_heated_temp_c,
                water_cut_frac=state.water_cut_frac,
            )
            outflow = (
                operating.liquid_rate_m3_per_day
                if operating.is_flowing
                else min(displacement, config.surface.max_well_rate_m3_per_day)
            )
            step = self.reservoir.step_production(
                days=1.0,
                bottomhole_pressure_kpa=operating.bottomhole_pressure_kpa,
                pump_capacity_m3_per_day=outflow,
            )
            fillages.append(fillage_from_inflow(step["total_liquid_rate_m3_per_day"], displacement))
            oil_rate = step["oil_rate_m3_per_day"]
            production_profile = self._fast_wellbore.produce(
                sandface_temp_c=state.average_heated_temp_c,
                liquid_rate_m3_per_day=max(step["total_liquid_rate_m3_per_day"], 0.05),
                water_cut_frac=state.water_cut_frac,
                elapsed_days=float(day_index) + 1.0,
            )
            wellhead_temp_c = production_profile.wellhead_temp_c
            tank = self.surface.tank_heating_demand(
                oil_rate_m3_per_day=oil_rate,
                water_rate_m3_per_day=step["water_rate_m3_per_day"],
                wellhead_temp_c=wellhead_temp_c,
            )
            surface_energy_kwh += tank.total_heat_kwh_per_day

            # Analytic lift energy. Two terms: the plunger lifting the fluid
            # load through the plunger stroke once per cycle, and the viscous
            # drag the rods push through. In heavy oil the drag term is the
            # larger of the two, so leaving it out would let the fast path run
            # cycles far past the point where they stop paying for themselves.
            #
            # Drag power: with the annular drag linearised as lambda(x), the
            # dissipated power is the integral of lambda v^2 over the string,
            # averaged over the cycle. For a stroke S at N strokes per minute
            # the mean square velocity is (pi S N / 60)^2 / 2.
            drag_coefficient_n_s_per_m = self._string_drag_coefficient_n_s_per_m(
                production_profile, max(step["total_liquid_rate_m3_per_day"], 0.05)
            )
            speed_amplitude_m_per_s = math.pi * setpoint.stroke_length_m * motion.spm / 60.0
            drag_power_w = 0.5 * drag_coefficient_n_s_per_m * speed_amplitude_m_per_s**2 / 2.0
            lifting_power_w = fluid_load_n * plunger_stroke_m * motion.spm / 60.0
            polished_rod_power_w = lifting_power_w + drag_power_w
            shaft_w = polished_rod_power_w / max(
                config.srp.gearbox_efficiency_frac * config.srp.belt_efficiency_frac, 0.1
            )
            load_fraction = min(max(shaft_w / config.srp.motor_rated_power_w, 0.02), 1.2)
            efficiency = max(
                config.srp.motor_efficiency_peak_frac * (1.0 - 0.5 * (1.0 - load_fraction) ** 2),
                0.15,
            )
            daily_lift_kwh = shaft_w / efficiency * 24.0 / 1000.0
            lift_energy_kwh += daily_lift_kwh

            cutoff_day = float(day_index) + 1.0
            if oil_rate < config.css.cutoff_oil_rate_m3_per_day:
                cutoff_reason = f"oil rate fell to {oil_rate:.2f} m3/day, below the economic limit"
                break
            daily_revenue_usd = oil_rate * economics.oil_price_usd_per_m3
            daily_cost_usd = (
                daily_lift_kwh * economics.electricity_cost_usd_per_kwh + tank.fuel_cost_usd_per_day
            )
            if daily_revenue_usd < daily_cost_usd:
                cutoff_reason = (
                    f"a further day would earn {daily_revenue_usd:.0f} USD of oil against "
                    f"{daily_cost_usd:.0f} USD of energy"
                )
                break

            # The same marginal oil per unit energy rule the full engine uses,
            # so the two agree on when a cycle should end. Without it the fast
            # path runs every cycle to the day limit and overstates cycle oil.
            daily_energy_kwh = daily_lift_kwh + tank.total_heat_kwh_per_day
            marginal_ratio = oil_rate / max(daily_energy_kwh, 1.0e-6) * 1000.0
            if marginal_reference is None and day_index >= 5:
                marginal_reference = marginal_ratio
            if (
                marginal_reference is not None
                and marginal_ratio < plan.cutoff_marginal_energy_ratio * marginal_reference
            ):
                cutoff_reason = (
                    "marginal oil per unit energy fell to "
                    f"{marginal_ratio / marginal_reference:.2f} of its early-cycle value"
                )
                break

        state = self.reservoir.state
        oil_m3 = state.cycle_oil_m3
        injection_energy_kwh = injection_energy_j / 3.6e6
        total_energy_kwh = lift_energy_kwh + surface_energy_kwh + injection_energy_kwh
        energy_per_m3 = total_energy_kwh / oil_m3 if oil_m3 > 0.0 else float("inf")
        return CycleResult(
            cycle_number=cycle_number,
            days=[],
            oil_m3=oil_m3,
            water_m3=max(state.cycle_water_m3, 0.0),
            steam_m3_cwe=state.cycle_steam_m3_cwe,
            steam_oil_ratio=state.steam_oil_ratio,
            injection_energy_j=injection_energy_j,
            lift_energy_kwh=lift_energy_kwh,
            surface_energy_kwh=surface_energy_kwh,
            energy_kwh_per_m3=energy_per_m3,
            energy_kwh_per_bbl=energy_per_m3 * M3_PER_BBL,
            float_event_days=0,
            mean_float_margin=0.0,
            peak_load_n=0.0,
            mean_fillage_frac=float(np.mean(fillages)) if fillages else 0.0,
            max_stress_utilisation_frac=0.0,
            fatigue_damage=0.0,
            cutoff_day=cutoff_day,
            cutoff_reason=cutoff_reason,
            constraint_violations=violations,
            revenue_usd=oil_m3 * economics.oil_price_usd_per_m3,
            energy_cost_usd=(
                injection_energy_j / 1.0e9 * economics.fuel_cost_usd_per_gj
                + lift_energy_kwh * economics.electricity_cost_usd_per_kwh
                + surface_energy_kwh / 277.778 * economics.fuel_cost_usd_per_gj
            ),
        )

    # ------------------------------------------------------------- cycle loop
    def run_cycle(
        self,
        plan: InjectionPlan,
        setpoint: PumpSetpoint,
        cycle_number: int = 1,
        max_production_days: float | None = None,
        record_days: bool = True,
        setpoint_schedule: dict[int, PumpSetpoint] | None = None,
    ) -> CycleResult:
        """Run injection, soak and production for one complete CSS cycle.

        Args:
            plan: Steam volume, rate, pressure, quality, soak and cut-off rule.
            setpoint: Pump settings used unless a schedule overrides them.
            cycle_number: Which cycle this is, which drives the decline terms.
            max_production_days: Upper bound on the production phase.
            record_days: Keep the per-day records. Turned off inside optimizer
                loops where only the totals are needed.
            setpoint_schedule: Optional mapping from production day to a new
                setpoint, used to replay what a controller decided.
        """
        config = self.config
        srp = config.srp
        days_limit = max_production_days or config.css.production_days_max
        violations: list[str] = []
        records: list[DayRecord] = []

        # ---- injection -----------------------------------------------------
        self.reservoir.begin_cycle(cycle_number)
        injection_days = plan.injection_days
        if injection_days > 60.0:
            violations.append(
                f"Injection would take {injection_days:.0f} days, beyond any practical window."
            )
        injection = self.wellbore.inject_steam(
            wellhead_pressure_kpa=plan.injection_pressure_kpa,
            wellhead_quality_frac=plan.steam_quality_frac,
            rate_m3_per_day_cwe=plan.injection_rate_m3_per_day_cwe,
            elapsed_days=max(injection_days, 0.5),
        )
        sandface_injection_pressure_kpa = float(injection.pressure_kpa[-1])
        fracture_kpa = config.reservoir.fracture_pressure_kpa
        if sandface_injection_pressure_kpa > fracture_kpa:
            violations.append(
                f"Sandface injection pressure {sandface_injection_pressure_kpa:.0f} kPa exceeds "
                f"the fracture pressure {fracture_kpa:.0f} kPa."
            )
        generator_limit = self.surface.generator_rate_limit_m3_per_day_cwe(
            plan.injection_pressure_kpa, plan.steam_quality_frac
        )
        if plan.injection_rate_m3_per_day_cwe > generator_limit * 1.001:
            violations.append(
                f"Injection rate {plan.injection_rate_m3_per_day_cwe:.0f} m3/day exceeds the "
                f"steam generator capacity of {generator_limit:.0f} m3/day."
            )

        self.reservoir.step_injection(
            plan=plan,
            days=injection_days,
            sandface_heat_rate_w=injection.sandface_heat_rate_w,
            sandface_steam_temp_c=injection.sandface_temp_c,
        )
        injection_energy_j = self.surface.steam_generation_fuel_energy_j(
            plan.steam_volume_m3_cwe, plan.injection_pressure_kpa, plan.steam_quality_frac
        )

        # ---- soak ----------------------------------------------------------
        self.reservoir.step_soak(plan.soak_days)

        # ---- production ----------------------------------------------------
        self.reservoir.state.phase_day = 0.0
        self.reservoir.state.phase = CyclePhase.PRODUCTION
        lift: LiftState | None = None
        lift_energy_kwh = 0.0
        surface_energy_kwh = 0.0
        float_event_days = 0
        float_margins: list[float] = []
        fillages: list[float] = []
        peak_load_n = 0.0
        max_stress = 0.0
        cutoff_day = 0.0
        cutoff_reason = "production day limit reached"
        active_setpoint = setpoint
        marginal_reference: float | None = None
        economics_prices = config.economics

        for day_index in range(int(days_limit)):
            production_day = float(day_index)
            if setpoint_schedule and day_index in setpoint_schedule:
                active_setpoint = setpoint_schedule[day_index]

            if day_index % self.rod_solver_interval_days == 0 or lift is None:
                lift = self.evaluate_lift(
                    setpoint=active_setpoint,
                    sandface_temp_c=self.reservoir.state.average_heated_temp_c,
                    water_cut_frac=self.reservoir.state.water_cut_frac,
                    elapsed_days=production_day + 1.0,
                    production_days=production_day,
                )

            outflow_capacity_m3_per_day = (
                lift.rate_limit_m3_per_day
                if lift.is_flowing
                else min(
                    lift.pump.displacement_m3_per_day,
                    self.config.surface.max_well_rate_m3_per_day,
                )
            )
            step = self.reservoir.step_production(
                days=1.0,
                bottomhole_pressure_kpa=lift.bottomhole_pressure_kpa,
                pump_capacity_m3_per_day=outflow_capacity_m3_per_day,
            )
            oil_rate = step["oil_rate_m3_per_day"]
            water_rate = step["water_rate_m3_per_day"]

            tank = self.surface.tank_heating_demand(
                oil_rate_m3_per_day=oil_rate,
                water_rate_m3_per_day=water_rate,
                wellhead_temp_c=lift.production_profile.wellhead_temp_c,
            )
            lift_energy_kwh += lift.power.daily_energy_kwh
            surface_energy_kwh += tank.total_heat_kwh_per_day

            cycles_today = lift.motion.spm * 1440.0
            self.fatigue.accumulate(lift.stress, cycles_today)
            if lift.float_analysis.is_floating:
                float_event_days += 1
                self.fatigue.accumulate_impact(
                    lift.float_analysis.estimated_impact_load_n,
                    lift.stress,
                    self.taper,
                    cycles_today * lift.float_analysis.float_index,
                )
            float_margins.append(lift.float_analysis.minimum_section_margin)
            fillages.append(lift.pump.fillage_frac)
            peak_load_n = max(peak_load_n, lift.wave.peak_polished_rod_load_n)
            max_stress = max(max_stress, lift.stress.maximum_utilisation_frac)

            if record_days:
                records.append(
                    self._build_record(
                        production_day, cycle_number, lift, step, tank, active_setpoint
                    )
                )

            cutoff_day = production_day + 1.0
            daily_energy_kwh = lift.power.daily_energy_kwh + tank.total_heat_kwh_per_day
            marginal_ratio = oil_rate / max(daily_energy_kwh, 1.0e-6) * 1000.0
            if marginal_reference is None and production_day >= 5.0:
                marginal_reference = marginal_ratio

            if oil_rate < config.css.cutoff_oil_rate_m3_per_day:
                cutoff_reason = (
                    f"oil rate fell to {oil_rate:.2f} m3/day, below the economic limit of "
                    f"{config.css.cutoff_oil_rate_m3_per_day:.2f} m3/day"
                )
                break

            # Primary rule: stop when another day of pumping costs more energy
            # than the oil it brings up is worth. This is the marginal oil per
            # unit energy criterion the brief asks for, expressed in money so it
            # can be explained to an operator.
            daily_revenue_usd = oil_rate * economics_prices.oil_price_usd_per_m3
            daily_cost_usd = (
                lift.power.daily_energy_kwh * economics_prices.electricity_cost_usd_per_kwh
                + tank.fuel_cost_usd_per_day
            )
            if daily_revenue_usd < daily_cost_usd:
                cutoff_reason = (
                    f"a further day of production would earn {daily_revenue_usd:.0f} USD of oil "
                    f"against {daily_cost_usd:.0f} USD of lift and tank heating energy"
                )
                break

            # Secondary rule: the configured fraction of the early-cycle
            # marginal oil per unit energy, which the CSS optimizer can tune.
            if (
                marginal_reference is not None
                and marginal_ratio < plan.cutoff_marginal_energy_ratio * marginal_reference
            ):
                cutoff_reason = (
                    "marginal oil per unit energy fell to "
                    f"{marginal_ratio / marginal_reference:.2f} of its early-cycle value"
                )
                break

        # ---- roll up -------------------------------------------------------
        state = self.reservoir.state
        oil_m3 = state.cycle_oil_m3
        water_m3 = max(state.cycle_water_m3, 0.0)
        steam_m3 = state.cycle_steam_m3_cwe
        injection_energy_kwh = injection_energy_j / 3.6e6
        total_energy_kwh = lift_energy_kwh + surface_energy_kwh + injection_energy_kwh
        energy_per_m3 = total_energy_kwh / oil_m3 if oil_m3 > 0.0 else float("inf")

        if peak_load_n > srp.structural_load_rating_n:
            violations.append(
                f"Peak polished rod load {peak_load_n / 1000.0:.1f} kN exceeds the "
                f"{srp.structural_load_rating_n / 1000.0:.1f} kN structural rating."
            )
        if max_stress > 1.0:
            violations.append(
                f"Rod stress reached {max_stress * 100.0:.0f} percent of the modified "
                "Goodman allowable."
            )

        economics = config.economics
        revenue_usd = oil_m3 * economics.oil_price_usd_per_m3
        energy_cost_usd = (
            injection_energy_j / 1.0e9 * economics.fuel_cost_usd_per_gj
            + (lift_energy_kwh) * economics.electricity_cost_usd_per_kwh
            + surface_energy_kwh / 277.778 * economics.fuel_cost_usd_per_gj
        )

        return CycleResult(
            cycle_number=cycle_number,
            days=records,
            oil_m3=oil_m3,
            water_m3=water_m3,
            steam_m3_cwe=steam_m3,
            steam_oil_ratio=steam_m3 / oil_m3 if oil_m3 > 0.0 else float("inf"),
            injection_energy_j=injection_energy_j,
            lift_energy_kwh=lift_energy_kwh,
            surface_energy_kwh=surface_energy_kwh,
            energy_kwh_per_m3=energy_per_m3,
            energy_kwh_per_bbl=energy_per_m3 * M3_PER_BBL,
            float_event_days=float_event_days,
            mean_float_margin=float(np.mean(float_margins)) if float_margins else 0.0,
            peak_load_n=peak_load_n,
            mean_fillage_frac=float(np.mean(fillages)) if fillages else 0.0,
            max_stress_utilisation_frac=max_stress,
            fatigue_damage=self.fatigue.damage,
            cutoff_day=cutoff_day,
            cutoff_reason=cutoff_reason,
            constraint_violations=violations,
            revenue_usd=revenue_usd,
            energy_cost_usd=energy_cost_usd,
        )

    def run_cycles(
        self,
        plans: list[InjectionPlan],
        setpoint: PumpSetpoint,
        start_cycle: int = 1,
        max_production_days: float | None = None,
        record_days: bool = True,
    ) -> list[CycleResult]:
        """Run several CSS cycles in sequence on the same well."""
        results: list[CycleResult] = []
        for offset, plan in enumerate(plans):
            results.append(
                self.run_cycle(
                    plan=plan,
                    setpoint=setpoint,
                    cycle_number=start_cycle + offset,
                    max_production_days=max_production_days,
                    record_days=record_days,
                )
            )
        return results

    # -------------------------------------------------------------- recording
    def _build_record(
        self,
        production_day: float,
        cycle_number: int,
        lift: LiftState,
        step: dict[str, float],
        tank: TankHeatingResult,
        setpoint: PumpSetpoint,
    ) -> DayRecord:
        """Flatten one day of state into a record."""
        reservoir = self.reservoir.snapshot()
        values: dict[str, float] = {
            "production_day": production_day,
            "heated_radius_m": reservoir["heated_radius_m"],
            "average_heated_temp_c": reservoir["average_heated_temp_c"],
            "sandface_viscosity_cp": reservoir["sandface_viscosity_cp"],
            "reservoir_pressure_kpa": reservoir["reservoir_pressure_kpa"],
            "bottomhole_pressure_kpa": lift.bottomhole_pressure_kpa,
            "oil_rate_m3_per_day": step["oil_rate_m3_per_day"],
            "water_rate_m3_per_day": step["water_rate_m3_per_day"],
            "water_cut_frac": reservoir["water_cut_frac"],
            "deliverability_m3_per_day": step["deliverability_m3_per_day"],
            "cumulative_oil_m3": reservoir["cumulative_oil_m3"],
            "cycle_oil_m3": reservoir["cycle_oil_m3"],
            "steam_oil_ratio": reservoir["steam_oil_ratio"],
            "wellhead_temp_c": lift.production_profile.wellhead_temp_c,
            "pump_intake_temp_c": lift.production_profile.pump_intake_temp_c,
            "viscosity_at_pump_cp": lift.viscosity_at_pump_pa_s * 1000.0,
            "viscosity_mid_string_cp": float(
                lift.viscosity_profile_pa_s[len(lift.viscosity_profile_pa_s) // 2]
            )
            * 1000.0,
            "spm": setpoint.spm,
            "achieved_spm": lift.motion.spm,
            "stroke_length_m": setpoint.stroke_length_m,
            "downstroke_speed_frac": setpoint.speed_profile.downstroke_speed_frac,
            "upstroke_speed_frac": setpoint.speed_profile.upstroke_speed_frac,
            "peak_polished_rod_load_n": lift.wave.peak_polished_rod_load_n,
            "minimum_polished_rod_load_n": lift.wave.minimum_polished_rod_load_n,
            "load_range_n": lift.wave.load_range_n,
            "plunger_stroke_m": lift.wave.pump_stroke_m,
            "fluid_load_n": lift.fluid_load_n,
            "float_margin_index": lift.float_analysis.minimum_section_margin,
            "float_index": lift.float_analysis.float_index,
            "impact_load_n": lift.float_analysis.estimated_impact_load_n,
            "fillage_frac": lift.pump.fillage_frac,
            "volumetric_efficiency_frac": lift.pump.volumetric_efficiency_frac,
            "slippage_m3_per_day": lift.pump.slippage_m3_per_day,
            "stress_utilisation_frac": lift.stress.maximum_utilisation_frac,
            "fatigue_damage": self.fatigue.damage,
            "polished_rod_power_w": lift.power.polished_rod_power_w,
            "motor_electrical_power_w": lift.power.motor_electrical_power_w,
            "gearbox_torque_utilisation_frac": lift.power.gearbox_torque_utilisation_frac,
            "lift_energy_kwh_per_day": lift.power.daily_energy_kwh,
            "surface_energy_kwh_per_day": tank.total_heat_kwh_per_day,
            "energy_kwh_per_bbl": lift.power.energy_kwh_per_bbl
            if np.isfinite(lift.power.energy_kwh_per_bbl)
            else 0.0,
            "is_pumped_off": float(lift.is_pumped_off),
            "is_flowing": float(lift.is_flowing),
        }
        require_finite(np.asarray(list(values.values()), dtype=float), "day record")
        values["diagnosis_confidence"] = lift.diagnosis.confidence
        return DayRecord(
            day=self.reservoir.state.day,
            cycle_number=cycle_number,
            phase=str(self.reservoir.state.phase),
            values=values,
        )

    def latest_card(self) -> DynamometerCard | None:
        """Most recent pump card, for the diagnostics page."""
        if self._last_lift is None:
            return None
        return DynamometerCard(
            self._last_lift.wave.pump_position_m,
            self._last_lift.wave.pump_load_n,
            False,
            self._last_lift.motion.cycle_time_s,
        )
