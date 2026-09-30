"""Model predictive control of the pump setpoint.

The controller searches over the pumping speed, the stroke length and the
intra-stroke velocity profile, looking a week ahead along the reservoir cooling
forecast. Looking ahead matters: a setting that is comfortable today can put
the rods into float in five days once the viscosity at the pump has risen, and
a controller that only sees today will keep chasing that.

Search cost is managed the way the brief asks for. The inner loop uses
closed-form estimates of the quantities that would otherwise need the rod wave
equation, which are accurate to a few percent, and the finalists are then
verified with the full solver before anything is recommended. Both the estimate
and the verified value are reported, so the surrogate error is visible rather
than assumed.

The recommendation always carries the binding constraint in plain words, so an
operator can see why the controller wants what it wants.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.config import FieldConfig, OptimizerConfig, VariableBound
from app.core.logging import get_logger
from app.core.numerics import clamp
from app.core.units import J_PER_KWH, M3_PER_BBL, SECONDS_PER_DAY, STANDARD_GRAVITY_M_PER_S2
from app.optimize.guard import Guard, SafetyEnvelope
from app.twin.coupled import CoupledWellTwin, LiftState, PumpSetpoint
from app.twin.srp.floating import build_drag_profile
from app.twin.srp.kinematics import SpeedProfile, StrokeMotion
from app.twin.srp.power import motor_efficiency_frac, net_gearbox_torque_n_m
from app.twin.srp.pump import evaluate_pump, fillage_from_inflow

LOGGER = get_logger(__name__)

SEARCH_CARD_SAMPLES = 96
"""Samples per cycle inside the search. The finalists are re-run at full resolution."""

PEAK_LOAD_SAFETY_FRAC = 1.04
"""The analytic peak load runs about 1 percent low, so the search adds a margin."""


@dataclass(frozen=True)
class HorizonPoint:
    """Everything about the well at one point on the forecast horizon.

    Precomputed once per optimisation, because none of it depends on the
    setpoint being tested. This is what makes the search fast enough to answer
    in seconds.
    """

    day_offset: float
    production_day: float
    sandface_temp_c: float
    water_cut_frac: float
    reservoir_pressure_kpa: float
    minimum_intake_pressure_kpa: float
    maximum_intake_pressure_kpa: float
    discharge_pressure_kpa: float
    facility_limit_m3_per_day: float
    productivity_m3_per_day_per_kpa: float
    viscosity_profile_pa_s: NDArray[np.float64]
    viscosity_at_pump_pa_s: float
    drag_below_coefficient_n_s_per_m: NDArray[np.float64]
    weight_below_n: NDArray[np.float64]
    total_drag_coefficient_n_s_per_m: float
    buoyant_weight_n: float
    rod_mass_kg: float
    fluid_density_kg_per_m3: float
    pump_intake_temp_c: float

    def fluid_load_n(self, plunger_area_m2: float, bottomhole_pressure_kpa: float) -> float:
        """Load the plunger carries at a given flowing pressure."""
        return plunger_area_m2 * max(
            (self.discharge_pressure_kpa - bottomhole_pressure_kpa), 0.0
        ) * 1000.0

    def operating_point(
        self, pump_capacity_m3_per_day: float
    ) -> tuple[float, float, bool]:
        """Flowing pressure, deliverable rate and whether the well is pumped off.

        The same closed-form solution the coupled engine uses, so the fast
        prediction and the full run agree on where the annular fluid level
        settles. Pinning the flowing pressure at the pumped-off value instead
        would overstate both the drawdown and the fluid load.
        """
        low = self.minimum_intake_pressure_kpa
        high = min(self.maximum_intake_pressure_kpa, self.reservoir_pressure_kpa)
        if high <= low or self.productivity_m3_per_day_per_kpa <= 0.0:
            return low, 0.0, True
        rate_at_full_column = max(
            self.productivity_m3_per_day_per_kpa * (self.reservoir_pressure_kpa - high), 0.0
        )
        if rate_at_full_column > pump_capacity_m3_per_day:
            return high, min(rate_at_full_column, self.facility_limit_m3_per_day), False
        rate_at_intake = max(
            self.productivity_m3_per_day_per_kpa * (self.reservoir_pressure_kpa - low), 0.0
        )
        if rate_at_intake <= pump_capacity_m3_per_day:
            return low, min(rate_at_intake, self.facility_limit_m3_per_day), True
        pressure = self.reservoir_pressure_kpa - pump_capacity_m3_per_day / (
            self.productivity_m3_per_day_per_kpa
        )
        pressure = float(min(max(pressure, low), high))
        return pressure, min(pump_capacity_m3_per_day, self.facility_limit_m3_per_day), False


@dataclass
class PumpPrediction:
    """What one setpoint is predicted to do at one horizon point."""

    oil_rate_m3_per_day: float
    liquid_rate_m3_per_day: float
    fillage_frac: float
    float_margin_index: float
    peak_load_n: float
    minimum_load_n: float
    gearbox_torque_utilisation_frac: float
    energy_kwh_per_day: float
    energy_kwh_per_bbl: float
    achieved_spm: float

    def as_dict(self) -> dict[str, float]:
        """Serialisable form."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class PumpCandidate:
    """A setpoint with its predicted behaviour over the whole horizon."""

    setpoint: PumpSetpoint
    predictions: list[PumpPrediction]
    score: float
    violations: list[str] = field(default_factory=list)
    binding_constraint: str = ""

    @property
    def is_feasible(self) -> bool:
        """Whether every constraint holds at every horizon point."""
        return not self.violations

    @property
    def mean_oil_rate_m3_per_day(self) -> float:
        """Mean predicted oil rate over the horizon."""
        return float(np.mean([p.oil_rate_m3_per_day for p in self.predictions]))

    @property
    def mean_energy_kwh_per_bbl(self) -> float:
        """Mean predicted energy per barrel over the horizon."""
        finite = [
            p.energy_kwh_per_bbl for p in self.predictions if math.isfinite(p.energy_kwh_per_bbl)
        ]
        return float(np.mean(finite)) if finite else math.inf

    @property
    def worst_float_margin(self) -> float:
        """Smallest float margin over the horizon."""
        return float(np.min([p.float_margin_index for p in self.predictions]))

    @property
    def peak_load_n(self) -> float:
        """Largest peak load over the horizon."""
        return float(np.max([p.peak_load_n for p in self.predictions]))

    @property
    def minimum_load_n(self) -> float:
        """Smallest minimum load over the horizon."""
        return float(np.min([p.minimum_load_n for p in self.predictions]))

    @property
    def mean_fillage_frac(self) -> float:
        """Mean predicted fillage over the horizon."""
        return float(np.mean([p.fillage_frac for p in self.predictions]))


@dataclass
class PumpRecommendation:
    """The controller's answer, with everything needed to explain it."""

    well_id: str
    current: PumpSetpoint
    recommended: PumpSetpoint
    current_prediction: PumpCandidate
    recommended_prediction: PumpCandidate
    verified: dict[str, float] = field(default_factory=dict)
    surrogate_error: dict[str, float] = field(default_factory=dict)
    guard_adjustments: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    requires_approval: bool = True
    evaluations: int = 0
    elapsed_s: float = 0.0

    @property
    def oil_change_m3_per_day(self) -> float:
        """Predicted change in oil rate."""
        return (
            self.recommended_prediction.mean_oil_rate_m3_per_day
            - self.current_prediction.mean_oil_rate_m3_per_day
        )

    @property
    def energy_change_kwh_per_bbl(self) -> float:
        """Predicted change in energy per barrel."""
        return (
            self.recommended_prediction.mean_energy_kwh_per_bbl
            - self.current_prediction.mean_energy_kwh_per_bbl
        )

    @property
    def is_change(self) -> bool:
        """Whether the controller is asking for anything to change."""
        return self.recommended.as_dict() != self.current.as_dict()

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the API and the audit log."""
        return {
            "well_id": self.well_id,
            "current": self.current.as_dict(),
            "recommended": self.recommended.as_dict(),
            "is_change": self.is_change,
            "reason": self.reason,
            "confidence": self.confidence,
            "requires_approval": self.requires_approval,
            "oil_change_m3_per_day": self.oil_change_m3_per_day,
            "energy_change_kwh_per_bbl": self.energy_change_kwh_per_bbl,
            "current_prediction": {
                "oil_rate_m3_per_day": self.current_prediction.mean_oil_rate_m3_per_day,
                "float_margin_index": self.current_prediction.worst_float_margin,
                "peak_load_n": self.current_prediction.peak_load_n,
                "minimum_load_n": self.current_prediction.minimum_load_n,
                "fillage_frac": self.current_prediction.mean_fillage_frac,
                "energy_kwh_per_bbl": self.current_prediction.mean_energy_kwh_per_bbl,
                "feasible": self.current_prediction.is_feasible,
                "violations": self.current_prediction.violations,
            },
            "recommended_prediction": {
                "oil_rate_m3_per_day": self.recommended_prediction.mean_oil_rate_m3_per_day,
                "float_margin_index": self.recommended_prediction.worst_float_margin,
                "peak_load_n": self.recommended_prediction.peak_load_n,
                "minimum_load_n": self.recommended_prediction.minimum_load_n,
                "fillage_frac": self.recommended_prediction.mean_fillage_frac,
                "energy_kwh_per_bbl": self.recommended_prediction.mean_energy_kwh_per_bbl,
                "feasible": self.recommended_prediction.is_feasible,
                "binding_constraint": self.recommended_prediction.binding_constraint,
            },
            "verified": self.verified,
            "surrogate_error": self.surrogate_error,
            "guard_adjustments": self.guard_adjustments,
            "evaluations": self.evaluations,
            "elapsed_s": self.elapsed_s,
        }


class PumpOptimizer:
    """Model predictive controller for the sucker rod pump setpoint."""

    def __init__(
        self,
        config: FieldConfig,
        optimizer: OptimizerConfig,
        twin: CoupledWellTwin | None = None,
        guard: Guard | None = None,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.settings = optimizer.pump
        self.twin = twin or CoupledWellTwin(config)
        self.guard = guard or Guard(config, optimizer)
        self._evaluations = 0

    # --------------------------------------------------------------- horizon
    def build_horizon(
        self,
        sandface_temp_c: float,
        water_cut_frac: float,
        production_day: float,
        cooling_rate_c_per_day: float | None = None,
        liquid_rate_guess_m3_per_day: float = 5.0,
    ) -> list[HorizonPoint]:
        """Precompute the well state at each point on the forecast horizon.

        The reservoir cooling forecast is what makes this a coupled controller
        rather than a reactive one: the viscosity the rods will be moving
        through in a week is what decides whether today's speed is safe.
        """
        twin = self.twin
        steps = self.settings.horizon_steps
        horizon_days = self.settings.horizon_days
        if cooling_rate_c_per_day is None:
            cooling_rate_c_per_day = self.forecast_cooling_rate_c_per_day()

        srp = self.config.srp
        taper = twin.taper
        steel_density = srp.steel_density_kg_per_m3
        rod_mass_kg = float(np.sum(taper.area_m2) * taper.step_m) * steel_density
        reservoir_pressure_kpa = twin.reservoir.state.reservoir_pressure_kpa
        bottomhole_kpa = twin.minimum_intake_pressure_kpa

        points: list[HorizonPoint] = []
        for index in range(steps):
            day_offset = horizon_days * index / max(steps - 1, 1)
            temperature_c = max(
                sandface_temp_c - cooling_rate_c_per_day * day_offset,
                self.config.reservoir.initial_temp_c,
            )
            production = twin.wellbore.produce(
                sandface_temp_c=temperature_c,
                liquid_rate_m3_per_day=max(liquid_rate_guess_m3_per_day, 0.05),
                water_cut_frac=water_cut_frac,
                elapsed_days=max(production_day + day_offset, 1.0),
            )
            viscosity = np.interp(taper.depth_m, production.depth_m, production.viscosity_pa_s)
            drag = build_drag_profile(
                taper,
                self.config.wellbore.tubing_inner_diameter_m,
                viscosity,
                max(liquid_rate_guess_m3_per_day, 0.05),
            )
            segment_drag = drag.linear_coefficient_n_s_per_m2 * taper.step_m
            drag_below = np.concatenate(
                [np.cumsum(segment_drag[::-1])[::-1][1:], [0.0]]
            )
            fluid_density = float(
                twin.fluid.mixture_density_kg_per_m3(
                    production.pump_intake_temp_c, water_cut_frac
                )
            )
            effective_density = max(steel_density - fluid_density, 0.0)
            segment_weight = (
                effective_density * STANDARD_GRAVITY_M_PER_S2 * taper.area_m2 * taper.step_m
            )
            weight_below = np.concatenate([np.cumsum(segment_weight[::-1])[::-1][1:], [0.0]])

            saved_temp = twin.reservoir.state.average_heated_temp_c
            twin.reservoir.state.average_heated_temp_c = temperature_c
            reference_rate = twin.reservoir.deliverability_m3_per_day(
                bottomhole_kpa, production_day + day_offset + 0.5
            )
            twin.reservoir.state.average_heated_temp_c = saved_temp
            drawdown_kpa = max(reservoir_pressure_kpa - bottomhole_kpa, 1.0e-6)
            discharge_kpa = self.config.wellbore.wellhead_pressure_kpa + (
                fluid_density * STANDARD_GRAVITY_M_PER_S2 * srp.pump_depth_m / 1000.0
            )

            points.append(
                HorizonPoint(
                    day_offset=day_offset,
                    production_day=production_day + day_offset,
                    sandface_temp_c=temperature_c,
                    water_cut_frac=water_cut_frac,
                    reservoir_pressure_kpa=reservoir_pressure_kpa,
                    minimum_intake_pressure_kpa=bottomhole_kpa,
                    maximum_intake_pressure_kpa=twin.maximum_intake_pressure_kpa(
                        production.pump_intake_temp_c, water_cut_frac
                    ),
                    discharge_pressure_kpa=discharge_kpa,
                    facility_limit_m3_per_day=self.config.surface.max_well_rate_m3_per_day,
                    productivity_m3_per_day_per_kpa=reference_rate / drawdown_kpa,
                    viscosity_profile_pa_s=viscosity,
                    viscosity_at_pump_pa_s=float(viscosity[-1]),
                    drag_below_coefficient_n_s_per_m=drag_below,
                    weight_below_n=weight_below,
                    total_drag_coefficient_n_s_per_m=float(np.sum(segment_drag)),
                    buoyant_weight_n=float(weight_below[0]),
                    rod_mass_kg=rod_mass_kg,
                    fluid_density_kg_per_m3=fluid_density,
                    pump_intake_temp_c=production.pump_intake_temp_c,
                )
            )
        return points

    def forecast_cooling_rate_c_per_day(self) -> float:
        """Cooling rate of the heated zone predicted by the reservoir model.

        Taken by stepping a copy of the reservoir state forward over the
        horizon and then restoring it, so the forecast uses the same conduction
        physics the twin uses rather than a separate fitted trend.
        """
        reservoir = self.twin.reservoir
        before_temp = reservoir.state.average_heated_temp_c
        saved = (
            reservoir.state.heated_zone_energy_j,
            reservoir.state.conduction_clock_days,
            reservoir.state.day,
            reservoir.state.phase_day,
            reservoir.state.cumulative_conduction_loss_j,
            reservoir.state.phase,
        )
        reservoir.step_soak(self.settings.horizon_days)
        after_temp = reservoir.state.average_heated_temp_c
        (
            reservoir.state.heated_zone_energy_j,
            reservoir.state.conduction_clock_days,
            reservoir.state.day,
            reservoir.state.phase_day,
            reservoir.state.cumulative_conduction_loss_j,
            reservoir.state.phase,
        ) = saved
        reservoir.state.average_heated_temp_c = before_temp
        return max((before_temp - after_temp) / max(self.settings.horizon_days, 1.0e-6), 0.0)

    # ----------------------------------------------------------- fast predict
    def predict(self, setpoint: PumpSetpoint, point: HorizonPoint) -> PumpPrediction:
        """Predict one setpoint at one horizon point, without the wave solver.

        Every quantity here has a closed form:

        * Drag force is the whole-string drag coefficient times the rod speed.
        * Float margin is the tension balance, buoyant weight below a node minus
          the drag carried below it, evaluated at the peak downstroke speed.
        * Peak polished rod load is buoyant weight plus fluid load plus upstroke
          drag plus rod inertia; minimum load is buoyant weight minus downstroke
          drag minus inertia.
        * Gearbox torque uses the torque factor waveform against a two-level
          load waveform.

        The estimates are within about three percent of the wave solver, which a
        test checks. The finalist is verified with the full solver anyway.
        """
        self._evaluations += 1
        srp = self.config.srp
        motion = self.twin.kinematics.motion(
            setpoint.spm, setpoint.stroke_length_m, setpoint.speed_profile, SEARCH_CARD_SAMPLES
        )
        drag_total = point.total_drag_coefficient_n_s_per_m
        up_speed = motion.peak_upstroke_speed_m_per_s
        down_speed = motion.peak_downstroke_speed_m_per_s
        acceleration_up = float(np.max(motion.acceleration_m_per_s2))
        acceleration_down = float(np.min(motion.acceleration_m_per_s2))

        tension_n = point.weight_below_n - point.drag_below_coefficient_n_s_per_m * down_speed
        with np.errstate(divide="ignore", invalid="ignore"):
            margins = np.where(
                point.weight_below_n > 1.0,
                tension_n / np.maximum(point.weight_below_n, 1.0),
                1.0,
            )
        float_margin = float(np.min(margins[:-1])) if margins.size > 1 else 1.0

        # The flowing pressure, the fluid load and the plunger stroke depend on
        # each other, so they are closed with two passes. The second changes the
        # answer by well under a percent.
        fluid_load_n = point.fluid_load_n(
            srp.plunger_area_m2, point.minimum_intake_pressure_kpa
        )
        plunger_stroke_m = self.twin.static_plunger_stroke_m(
            setpoint.stroke_length_m, fluid_load_n
        )
        displacement = srp.plunger_area_m2 * plunger_stroke_m * motion.spm * 1440.0
        bottomhole_kpa, deliverability, _ = point.operating_point(displacement)
        for _ in range(2):
            fluid_load_n = point.fluid_load_n(srp.plunger_area_m2, bottomhole_kpa)
            plunger_stroke_m = self.twin.static_plunger_stroke_m(
                setpoint.stroke_length_m, fluid_load_n
            )
            displacement = srp.plunger_area_m2 * plunger_stroke_m * motion.spm * 1440.0
            bottomhole_kpa, deliverability, _ = point.operating_point(displacement)
        fillage = fillage_from_inflow(deliverability, displacement)

        peak_load_n = (
            point.buoyant_weight_n
            + fluid_load_n
            + drag_total * up_speed
            + point.rod_mass_kg * acceleration_up
        ) * PEAK_LOAD_SAFETY_FRAC
        minimum_load_n = (
            point.buoyant_weight_n
            - drag_total * down_speed
            + point.rod_mass_kg * acceleration_down
        )

        performance = evaluate_pump(
            config=srp,
            plunger_stroke_m=plunger_stroke_m,
            spm=motion.spm,
            viscosity_at_pump_pa_s=point.viscosity_at_pump_pa_s,
            pressure_difference_pa=fluid_load_n / srp.plunger_area_m2,
            fillage_frac=fillage,
            gas_interference_frac=srp.gas_interference_frac,
        )
        liquid_rate = min(
            performance.liquid_rate_m3_per_day,
            deliverability,
            self.config.surface.max_well_rate_m3_per_day,
        )
        oil_rate = liquid_rate * (1.0 - point.water_cut_frac)

        torque_utilisation = self._torque_utilisation(
            motion, point, drag_total, fluid_load_n
        )
        lifting_power_w = fluid_load_n * plunger_stroke_m * motion.spm / 60.0
        speed_amplitude = math.pi * setpoint.stroke_length_m * motion.spm / 60.0
        drag_power_w = 0.25 * drag_total * speed_amplitude**2
        shaft_w = (lifting_power_w + drag_power_w) / max(
            srp.gearbox_efficiency_frac * srp.belt_efficiency_frac, 0.1
        )
        efficiency = motor_efficiency_frac(
            shaft_w / max(srp.motor_rated_power_w, 1.0), srp.motor_efficiency_peak_frac
        )
        daily_energy_kwh = shaft_w / efficiency * SECONDS_PER_DAY / J_PER_KWH
        energy_per_m3 = daily_energy_kwh / liquid_rate if liquid_rate > 0.0 else math.inf

        return PumpPrediction(
            oil_rate_m3_per_day=oil_rate,
            liquid_rate_m3_per_day=liquid_rate,
            fillage_frac=fillage,
            float_margin_index=float_margin,
            peak_load_n=peak_load_n,
            minimum_load_n=minimum_load_n,
            gearbox_torque_utilisation_frac=torque_utilisation,
            energy_kwh_per_day=daily_energy_kwh,
            energy_kwh_per_bbl=energy_per_m3 * M3_PER_BBL
            if math.isfinite(energy_per_m3)
            else math.inf,
            achieved_spm=motion.spm,
        )

    def _torque_utilisation(
        self,
        motion: StrokeMotion,
        point: HorizonPoint,
        drag_total: float,
        fluid_load_n: float,
    ) -> float:
        """Peak gearbox torque as a fraction of the rating, from a two-level card."""
        if motion.unit_type == "hydraulic":
            return 0.0
        upstroke = motion.velocity_m_per_s > 0.0
        load = (
            np.where(
                upstroke, point.buoyant_weight_n + fluid_load_n, point.buoyant_weight_n
            )
            + drag_total * motion.velocity_m_per_s
        )
        torque = net_gearbox_torque_n_m(
            motion.torque_factor_m,
            load,
            motion.crank_angle_rad,
            self.config.srp.counterbalance_moment_n_m,
        )
        return float(np.max(np.abs(torque))) / self.config.srp.gearbox_torque_rating_n_m

    # -------------------------------------------------------------- evaluate
    def evaluate(self, setpoint: PumpSetpoint, horizon: list[HorizonPoint]) -> PumpCandidate:
        """Predict a setpoint over the whole horizon and score it."""
        constraints = self.settings.constraints
        srp = self.config.srp
        predictions = [self.predict(setpoint, point) for point in horizon]
        violations: list[str] = []
        binding: list[tuple[float, str]] = []

        worst_margin = min(p.float_margin_index for p in predictions)
        if worst_margin < constraints.min_float_margin_frac:
            violations.append(
                f"float margin falls to {worst_margin:.2f}, below the minimum of "
                f"{constraints.min_float_margin_frac:.2f}"
            )
        binding.append(
            (
                (worst_margin - constraints.min_float_margin_frac)
                / max(constraints.min_float_margin_frac, 1.0e-6),
                f"the float margin is {worst_margin:.2f} against a minimum of "
                f"{constraints.min_float_margin_frac:.2f}",
            )
        )

        peak = max(p.peak_load_n for p in predictions)
        load_limit = constraints.max_structural_load_frac * srp.structural_load_rating_n
        if peak > load_limit:
            violations.append(
                f"peak polished rod load reaches {peak / 1000.0:.1f} kN, above the "
                f"{load_limit / 1000.0:.1f} kN working limit"
            )
        binding.append(
            (
                (load_limit - peak) / max(load_limit, 1.0),
                f"the peak load is {peak / 1000.0:.1f} kN against a limit of "
                f"{load_limit / 1000.0:.1f} kN",
            )
        )

        minimum = min(p.minimum_load_n for p in predictions)
        if minimum < constraints.min_polished_rod_load_n:
            violations.append(
                f"minimum polished rod load falls to {minimum / 1000.0:.1f} kN, below the "
                f"{constraints.min_polished_rod_load_n / 1000.0:.1f} kN float threshold"
            )

        torque = max(p.gearbox_torque_utilisation_frac for p in predictions)
        if torque > constraints.max_gearbox_torque_frac:
            violations.append(
                f"gearbox torque reaches {torque * 100:.0f} percent of rating, above the "
                f"{constraints.max_gearbox_torque_frac * 100:.0f} percent limit"
            )
        binding.append(
            (
                (constraints.max_gearbox_torque_frac - torque)
                / max(constraints.max_gearbox_torque_frac, 1.0e-6),
                f"gearbox torque is {torque * 100:.0f} percent of rating",
            )
        )

        mean_fillage = float(np.mean([p.fillage_frac for p in predictions]))
        if mean_fillage < constraints.min_fillage_frac:
            binding.append(
                (
                    (mean_fillage - constraints.min_fillage_frac)
                    / max(constraints.min_fillage_frac, 1.0e-6),
                    f"fillage averages {mean_fillage * 100:.0f} percent against a minimum "
                    f"of {constraints.min_fillage_frac * 100:.0f} percent",
                )
            )

        weights = self.settings.objective
        oil = float(np.mean([p.oil_rate_m3_per_day for p in predictions]))
        energy = float(np.mean([p.energy_kwh_per_day for p in predictions]))
        score = weights.production_weight * oil - weights.energy_weight * energy / 100.0
        score -= weights.float_penalty_weight * max(
            constraints.min_float_margin_frac - worst_margin, 0.0
        )
        score -= weights.load_penalty_weight * max(
            (peak - load_limit) / max(load_limit, 1.0), 0.0
        )
        if mean_fillage < constraints.min_fillage_frac:
            score -= 0.5 * oil * (constraints.min_fillage_frac - mean_fillage)

        binding.sort()
        return PumpCandidate(
            setpoint=setpoint,
            predictions=predictions,
            score=float(score),
            violations=violations,
            binding_constraint=binding[0][1] if binding else "",
        )

    # ---------------------------------------------------------------- search
    def optimize(
        self,
        current: PumpSetpoint,
        sandface_temp_c: float,
        water_cut_frac: float,
        production_day: float,
        well_id: str = "unknown",
        verify: bool = True,
        liquid_rate_guess_m3_per_day: float = 5.0,
    ) -> PumpRecommendation:
        """Search for the best feasible setpoint over the forecast horizon.

        Args:
            current: The setpoint the well is running now.
            sandface_temp_c: Heated-zone temperature now.
            water_cut_frac: Water cut now.
            production_day: Days since this cycle came on production.
            well_id: Well identifier, carried into the recommendation.
            verify: Re-run the answer through the full rod solver and report the
                surrogate error. Turned off only inside long scenario loops.
            liquid_rate_guess_m3_per_day: Starting rate for the wellbore
                temperature profile. The search refines it.
        """
        started = time.time()
        self._evaluations = 0

        # The temperature profile up the string depends on how fast the well is
        # producing, and the rate depends on the setpoint. One refinement pass
        # closes that loop: build the horizon on a first guess, predict the rate
        # the current setpoint gives, then rebuild on that rate. Without it the
        # search reasons about a colder string than the well actually has, and
        # the verification against the full solver disagrees badly.
        horizon = self.build_horizon(
            sandface_temp_c=sandface_temp_c,
            water_cut_frac=water_cut_frac,
            production_day=production_day,
            liquid_rate_guess_m3_per_day=liquid_rate_guess_m3_per_day,
        )
        current_candidate = self.evaluate(current, horizon)
        refined_rate = float(
            np.mean([p.liquid_rate_m3_per_day for p in current_candidate.predictions])
        )
        if abs(refined_rate - liquid_rate_guess_m3_per_day) > 0.2 * max(refined_rate, 0.1):
            horizon = self.build_horizon(
                sandface_temp_c=sandface_temp_c,
                water_cut_frac=water_cut_frac,
                production_day=production_day,
                liquid_rate_guess_m3_per_day=max(refined_rate, 0.1),
            )
            current_candidate = self.evaluate(current, horizon)

        rng = np.random.default_rng(self.settings.seed)
        variables = self.settings.variables
        envelope = self.guard.envelope
        best = current_candidate

        starts = [current]
        for _ in range(self.settings.restarts - 1):
            starts.append(
                PumpSetpoint(
                    spm=float(rng.uniform(envelope.spm_min, envelope.spm_max)),
                    stroke_length_m=float(rng.choice(envelope.stroke_options_m)),
                    speed_profile=SpeedProfile(
                        upstroke_speed_frac=float(
                            rng.uniform(
                                variables["upstroke_speed_frac"].low,
                                variables["upstroke_speed_frac"].high,
                            )
                        ),
                        downstroke_speed_frac=float(
                            rng.uniform(
                                variables["downstroke_speed_frac"].low,
                                variables["downstroke_speed_frac"].high,
                            )
                        ),
                        top_of_downstroke_decel_frac=float(
                            rng.uniform(0.0, variables["top_of_downstroke_decel_frac"].high)
                        ),
                    ),
                )
            )

        for start in starts:
            candidate = self._pattern_search(start, horizon, variables, envelope)
            if self._better(candidate, best):
                best = candidate

        guarded = self.guard.review(best.setpoint, current)
        final_setpoint = guarded.setpoint
        final_candidate = (
            best
            if final_setpoint.as_dict() == best.setpoint.as_dict()
            else self.evaluate(final_setpoint, horizon)
        )

        verified: dict[str, float] = {}
        surrogate_error: dict[str, float] = {}
        if verify:
            verified, surrogate_error = self.verify(
                final_setpoint, horizon[0], final_candidate.predictions[0]
            )

        return PumpRecommendation(
            well_id=well_id,
            current=current,
            recommended=final_setpoint,
            current_prediction=current_candidate,
            recommended_prediction=final_candidate,
            verified=verified,
            surrogate_error=surrogate_error,
            guard_adjustments=guarded.adjustments,
            reason=self._explain(current_candidate, final_candidate, horizon),
            confidence=self._confidence(final_candidate, surrogate_error),
            requires_approval=guarded.requires_approval,
            evaluations=self._evaluations,
            elapsed_s=time.time() - started,
        )

    def _better(self, candidate: PumpCandidate, incumbent: PumpCandidate) -> bool:
        """Feasibility first, then score. A candidate never wins by cheating."""
        if candidate.is_feasible and not incumbent.is_feasible:
            return True
        if not candidate.is_feasible and incumbent.is_feasible:
            return False
        return candidate.score > incumbent.score

    def _pattern_search(
        self,
        start: PumpSetpoint,
        horizon: list[HorizonPoint],
        variables: dict[str, VariableBound],
        envelope: SafetyEnvelope,
    ) -> PumpCandidate:
        """Coordinate pattern search with a shrinking step.

        A pattern search is used rather than a gradient method because two of
        the variables are discrete or nearly so, and because it degrades
        gracefully: stopping early still returns the best point seen.
        """
        best = self.evaluate(start, horizon)
        step = 1.0
        iterations = max(self.settings.iterations // max(self.settings.restarts, 1), 8)
        for _ in range(iterations):
            improved = False
            for neighbour in self._neighbours(best.setpoint, step, variables, envelope):
                candidate = self.evaluate(neighbour, horizon)
                if self._better(candidate, best):
                    best = candidate
                    improved = True
            if not improved:
                step *= 0.5
                if step < 0.02:
                    break
        return best

    def _neighbours(
        self,
        setpoint: PumpSetpoint,
        step: float,
        variables: dict[str, VariableBound],
        envelope: SafetyEnvelope,
    ) -> list[PumpSetpoint]:
        """Candidate moves from the current point."""
        profile = setpoint.speed_profile
        moves: list[PumpSetpoint] = []
        spm_step = step * 0.35 * (envelope.spm_max - envelope.spm_min)
        for delta in (spm_step, -spm_step):
            moves.append(
                PumpSetpoint(
                    spm=clamp(setpoint.spm + delta, envelope.spm_min, envelope.spm_max),
                    stroke_length_m=setpoint.stroke_length_m,
                    speed_profile=profile,
                )
            )
        for name in (
            "downstroke_speed_frac",
            "upstroke_speed_frac",
            "top_of_downstroke_decel_frac",
        ):
            bound = variables[name]
            amount = step * 0.35 * (bound.high - bound.low)
            for delta in (amount, -amount):
                values = {
                    "upstroke_speed_frac": profile.upstroke_speed_frac,
                    "downstroke_speed_frac": profile.downstroke_speed_frac,
                    "top_of_downstroke_decel_frac": profile.top_of_downstroke_decel_frac,
                }
                values[name] = clamp(values[name] + delta, bound.low, bound.high)
                moves.append(
                    PumpSetpoint(
                        spm=setpoint.spm,
                        stroke_length_m=setpoint.stroke_length_m,
                        speed_profile=SpeedProfile(**values),
                    )
                )
        options = list(envelope.stroke_options_m)
        index = options.index(envelope.nearest_stroke_m(setpoint.stroke_length_m))
        for neighbour_index in (index - 1, index + 1):
            if 0 <= neighbour_index < len(options):
                moves.append(
                    PumpSetpoint(
                        spm=setpoint.spm,
                        stroke_length_m=options[neighbour_index],
                        speed_profile=profile,
                    )
                )
        return moves

    # ----------------------------------------------------------- verification
    def verify(
        self, setpoint: PumpSetpoint, point: HorizonPoint, prediction: PumpPrediction
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Re-run a setpoint through the full rod wave solver and report the gap."""
        lift: LiftState = self.twin.evaluate_lift(
            setpoint=setpoint,
            sandface_temp_c=point.sandface_temp_c,
            water_cut_frac=point.water_cut_frac,
            elapsed_days=max(point.production_day + 1.0, 1.0),
            production_days=point.production_day,
        )
        verified = {
            "peak_load_n": lift.wave.peak_polished_rod_load_n,
            "minimum_load_n": lift.wave.minimum_polished_rod_load_n,
            "float_margin_index": lift.float_analysis.minimum_section_margin,
            "float_index": lift.float_analysis.float_index,
            "fillage_frac": lift.pump.fillage_frac,
            "liquid_rate_m3_per_day": lift.liquid_rate_m3_per_day,
            "energy_kwh_per_bbl": lift.power.energy_kwh_per_bbl
            if math.isfinite(lift.power.energy_kwh_per_bbl)
            else 0.0,
            "stress_utilisation_frac": lift.stress.maximum_utilisation_frac,
            "gearbox_torque_utilisation_frac": lift.power.gearbox_torque_utilisation_frac,
        }
        error = {
            "peak_load_frac": abs(prediction.peak_load_n - verified["peak_load_n"])
            / max(verified["peak_load_n"], 1.0),
            "float_margin_absolute": abs(
                prediction.float_margin_index - verified["float_margin_index"]
            ),
            "liquid_rate_frac": abs(
                prediction.liquid_rate_m3_per_day - verified["liquid_rate_m3_per_day"]
            )
            / max(verified["liquid_rate_m3_per_day"], 1.0e-6),
        }
        return verified, error

    # ------------------------------------------------------------ explanation
    def _explain(
        self,
        current: PumpCandidate,
        recommended: PumpCandidate,
        horizon: list[HorizonPoint],
    ) -> str:
        """The reason for the recommendation, in plain words."""
        cooling = horizon[0].sandface_temp_c - horizon[-1].sandface_temp_c
        viscosity_now = horizon[0].viscosity_at_pump_pa_s * 1000.0
        viscosity_later = horizon[-1].viscosity_at_pump_pa_s * 1000.0
        driver = (
            f"The heated zone is predicted to cool {cooling:.0f} degrees over the next "
            f"{horizon[-1].day_offset:.0f} days, taking the viscosity at the pump from "
            f"{viscosity_now:.0f} to {viscosity_later:.0f} cP."
        )
        if recommended.setpoint.as_dict() == current.setpoint.as_dict():
            return (
                f"No change recommended. {driver} Even so, "
                f"{recommended.binding_constraint}, and nothing tested improved on the "
                "current setting."
            )

        parts: list[str] = []
        speed_change = recommended.setpoint.spm - current.setpoint.spm
        if abs(speed_change) > 0.02:
            parts.append(
                f"speed {'raised' if speed_change > 0 else 'lowered'} from "
                f"{current.setpoint.spm:.2f} to {recommended.setpoint.spm:.2f} strokes per minute"
            )
        downstroke_change = (
            recommended.setpoint.speed_profile.downstroke_speed_frac
            - current.setpoint.speed_profile.downstroke_speed_frac
        )
        if abs(downstroke_change) > 0.02:
            parts.append(
                f"downstroke slowed {abs(downstroke_change) * 100:.0f} percent"
                if downstroke_change < 0
                else f"downstroke speeded up {downstroke_change * 100:.0f} percent"
            )
        decel_change = (
            recommended.setpoint.speed_profile.top_of_downstroke_decel_frac
            - current.setpoint.speed_profile.top_of_downstroke_decel_frac
        )
        if abs(decel_change) > 0.02:
            parts.append(
                f"an extra {abs(decel_change) * 100:.0f} percent deceleration into the top of "
                "the downstroke"
            )
        if abs(recommended.setpoint.stroke_length_m - current.setpoint.stroke_length_m) > 0.01:
            parts.append(f"stroke changed to {recommended.setpoint.stroke_length_m:.2f} m")

        effect = (
            f"That moves the float margin from {current.worst_float_margin:.2f} to "
            f"{recommended.worst_float_margin:.2f} and the peak load from "
            f"{current.peak_load_n / 1000.0:.1f} to {recommended.peak_load_n / 1000.0:.1f} kN."
        )
        oil_change = recommended.mean_oil_rate_m3_per_day - current.mean_oil_rate_m3_per_day
        energy_change = (
            recommended.mean_energy_kwh_per_bbl - current.mean_energy_kwh_per_bbl
        )
        production = (
            f"Predicted oil changes by {oil_change:+.2f} m3 per day and lift energy by "
            f"{energy_change:+.2f} kWh per barrel."
        )
        change_text = ", ".join(parts) if parts else "setpoint adjusted"
        return (
            change_text[0].upper()
            + change_text[1:]
            + ". "
            + driver
            + " "
            + effect
            + " "
            + production
            + f" Binding constraint: {recommended.binding_constraint}."
        )

    def _confidence(
        self, candidate: PumpCandidate, surrogate_error: dict[str, float]
    ) -> float:
        """How much to trust this recommendation, between 0 and 1.

        Confidence falls when the candidate sits close to a constraint, and when
        the fast prediction and the verified rod solve disagree.
        """
        margin_headroom = clamp(
            (candidate.worst_float_margin - self.settings.constraints.min_float_margin_frac)
            / 0.3,
            0.0,
            1.0,
        )
        load_limit = (
            self.settings.constraints.max_structural_load_frac
            * self.config.srp.structural_load_rating_n
        )
        load_headroom = clamp((load_limit - candidate.peak_load_n) / (0.2 * load_limit), 0.0, 1.0)
        agreement = 1.0
        if surrogate_error:
            agreement = clamp(1.0 - surrogate_error.get("peak_load_frac", 0.0) / 0.15, 0.0, 1.0)
        base = 0.45 + 0.2 * margin_headroom + 0.15 * load_headroom + 0.2 * agreement
        if not candidate.is_feasible:
            base *= 0.4
        return float(clamp(base, 0.05, 0.98))


def historical_practice_setpoint(config: FieldConfig) -> PumpSetpoint:
    """The setting an operator working from history would use.

    This is the baseline every optimizer comparison is measured against. It has
    to be a real practice rather than a straw man, or the comparison means
    nothing: a fixed speed from the well file, a neutral intra-stroke profile,
    and no reaction to how the well is cooling.
    """
    return PumpSetpoint(
        spm=config.srp.spm_setpoint,
        stroke_length_m=config.srp.stroke_length_m,
        speed_profile=SpeedProfile(),
    )
