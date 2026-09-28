"""One synthetic well: the higher-fidelity simulator running a full CSS history.

This assembles the axisymmetric thermal reservoir, the transient wellbore, the
fine rod solver and the sensor model into a well that can be run for several
cycles. What comes out is what the ingestion layer would receive from the
field: daily production, cycle records, telemetry and dynamometer cards, all
with measurement error and injected faults.

The twin never sees this module, only its noisy output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.config import FieldConfig
from app.core.logging import get_logger
from app.core.units import SECONDS_PER_DAY, STANDARD_GRAVITY_M_PER_S2
from app.simulate.truth.priors import HiddenParameters
from app.simulate.truth.reservoir_fv import AxisymmetricThermalReservoir
from app.simulate.truth.rods_fine import refine_taper, solve_fine
from app.simulate.truth.sensors import SensorModel, SensorSettings
from app.simulate.truth.wellbore_fd import TransientWellbore
from app.twin.reservoir import InjectionPlan
from app.twin.srp.cards import CardClass
from app.twin.srp.floating import annular_drag_per_length_n_per_m
from app.twin.srp.kinematics import SpeedProfile, build_kinematics
from app.twin.srp.wave import DragProfile, PumpBoundary, build_taper

LOGGER = get_logger(__name__)

CARD_SAMPLES = 180
"""Samples per card in the generated telemetry."""


@dataclass(frozen=True)
class TruthSetpoint:
    """Pump settings used by the truth simulator, mirroring the twin's setpoint."""

    spm: float
    stroke_length_m: float
    upstroke_speed_frac: float = 1.0
    downstroke_speed_frac: float = 1.0
    top_of_downstroke_decel_frac: float = 0.0

    def profile(self) -> SpeedProfile:
        """As a speed profile for the kinematics."""
        return SpeedProfile(
            upstroke_speed_frac=self.upstroke_speed_frac,
            downstroke_speed_frac=self.downstroke_speed_frac,
            top_of_downstroke_decel_frac=self.top_of_downstroke_decel_frac,
        )


@dataclass
class TruthWellResult:
    """Everything one simulated well produced."""

    well_id: str
    daily_rows: list[dict[str, Any]] = field(default_factory=list)
    cycle_rows: list[dict[str, Any]] = field(default_factory=list)
    telemetry_rows: list[dict[str, Any]] = field(default_factory=list)
    card_rows: list[dict[str, Any]] = field(default_factory=list)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)
    sensor_fault_log: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    def cycle_count(self) -> int:
        """Number of completed cycles."""
        return len(self.cycle_rows)


class TruthWell:
    """A single synthetic well under cyclic steam stimulation.

    Args:
        config: Field configuration, for the parts that are not hidden.
        hidden: The hidden per-well parameters.
        sensor_settings: Measurement error settings.
        fault_rates: Per-cycle probabilities of each mechanical fault.
        telemetry_interval_minutes: Sampling interval of the high-rate channels.
        rod_update_interval_days: How often the fine rod solver is run.
    """

    def __init__(
        self,
        config: FieldConfig,
        hidden: HiddenParameters,
        sensor_settings: SensorSettings,
        fault_rates: dict[str, dict[str, float]],
        telemetry_interval_minutes: int = 15,
        rod_update_interval_days: int = 10,
    ) -> None:
        self.config = config
        self.hidden = hidden
        self.fault_rates = fault_rates
        self.telemetry_interval_minutes = telemetry_interval_minutes
        self.rod_update_interval_days = max(rod_update_interval_days, 1)
        self.rng = np.random.default_rng(hidden.seed + 7)
        self.sensors = SensorModel(sensor_settings, seed=hidden.seed + 11)

        self.reservoir = AxisymmetricThermalReservoir(config, hidden)
        self.wellbore = TransientWellbore(
            config,
            base_heat_transfer_w_per_m2_k=hidden.vit_overall_heat_transfer_w_per_m2_k,
            degradation_per_year_frac=hidden.vit_degradation_per_year_frac,
        )
        srp_config = (
            config.srp
            if hidden.unit_type == "conventional"
            else config.with_overrides({"srp": {"unit_type": "hydraulic"}}).srp
        )
        self.srp_config = srp_config
        self.kinematics = build_kinematics(srp_config)
        self.taper = refine_taper(build_taper(srp_config), refinement=2)
        self.clearance_multiplier = 1.0
        self.cumulative_impact_events = 0.0
        self._pending_card: dict[str, Any] | None = None
        self._last_rate = 5.0

    # ------------------------------------------------------------------ faults
    def _draw_cycle_faults(self, cycle_number: int, mean_viscosity_pa_s: float) -> set[str]:
        """Decide which mechanical faults are active in this cycle."""
        active: set[str] = set()
        for name, spec in self.fault_rates.items():
            rate = float(spec.get("base_rate_per_cycle", 0.0))
            sensitivity = float(spec.get("viscosity_sensitivity", 0.0))
            if sensitivity > 0.0:
                rate *= 1.0 + sensitivity * min(mean_viscosity_pa_s / 0.5, 3.0)
            rate *= 1.0 + 0.08 * (cycle_number - 1)
            if self.rng.random() < min(rate, 0.95):
                active.add(name)
        return active

    # -------------------------------------------------------------- rod solve
    def _drag_profile(
        self, viscosity_profile_pa_s: NDArray[np.float64], liquid_rate_m3_per_day: float
    ) -> DragProfile:
        """Linearise the annular drag on the fine grid."""
        tubing_radius_m = 0.5 * self.config.wellbore.tubing_inner_diameter_m
        net_flow = liquid_rate_m3_per_day / SECONDS_PER_DAY
        linear = np.zeros(self.taper.node_count)
        static = np.zeros(self.taper.node_count)
        for index in range(self.taper.node_count):
            radius_m = math.sqrt(self.taper.area_m2[index] / math.pi)
            viscosity = float(viscosity_profile_pa_s[index])
            at_rest = annular_drag_per_length_n_per_m(
                radius_m, tubing_radius_m, 0.0, net_flow, viscosity
            ).drag_per_length_n_per_m
            moving = annular_drag_per_length_n_per_m(
                radius_m, tubing_radius_m, 1.0, net_flow, viscosity
            ).drag_per_length_n_per_m
            linear[index] = max(at_rest - moving, 0.0)
            static[index] = at_rest
        return DragProfile(linear_coefficient_n_s_per_m2=linear, static_force_n_per_m=static)

    def _oil_viscosity_pa_s(self, temperature_c: NDArray[np.float64]) -> NDArray[np.float64]:
        """Produced-fluid viscosity at the given temperatures, with emulsion uplift."""
        base = self.reservoir.oil_viscosity_pa_s(temperature_c)
        water_cut = self.reservoir.water_cut_frac()
        inversion = self.config.fluid.emulsion_inversion_water_cut_frac
        uplift = math.exp(self.config.fluid.emulsion_richardson_k * min(water_cut, inversion))
        return base * uplift

    # ----------------------------------------------------------------- cycles
    def run_history(
        self,
        cycles: int,
        plan: InjectionPlan,
        setpoint: TruthSetpoint,
        max_production_days: int = 150,
    ) -> TruthWellResult:
        """Run a full multi-cycle history for this well."""
        result = TruthWellResult(well_id=self.hidden.well_id)
        calendar_day = 0.0

        for cycle_number in range(1, cycles + 1):
            self.reservoir.begin_cycle(cycle_number)
            elapsed_years = calendar_day / 365.25

            injection_days = plan.injection_days
            injection = self.wellbore.inject_steam(
                wellhead_pressure_kpa=plan.injection_pressure_kpa,
                wellhead_quality_frac=plan.steam_quality_frac,
                rate_m3_per_day_cwe=plan.injection_rate_m3_per_day_cwe,
                step_days=max(injection_days, 0.5),
                elapsed_years=elapsed_years,
            )
            steps = max(int(math.ceil(injection_days)), 1)
            for _ in range(steps):
                self.reservoir.step_injection(
                    step_days=injection_days / steps,
                    sandface_heat_rate_w=injection.sandface_heat_rate_w,
                    sandface_steam_temp_c=injection.sandface_temp_c,
                    steam_rate_m3_per_day_cwe=plan.injection_rate_m3_per_day_cwe,
                )
            calendar_day += injection_days

            soak_days = max(plan.soak_days, 0.0)
            soak_steps = max(int(math.ceil(soak_days)), 1)
            for _ in range(soak_steps):
                self.reservoir.step_conduction(soak_days / soak_steps)
            calendar_day += soak_days

            cycle_faults = self._draw_cycle_faults(
                cycle_number,
                float(np.mean(self._oil_viscosity_pa_s(self.reservoir.temperature_c[0, :]))),
            )
            if "worn_pump" in cycle_faults:
                self.clearance_multiplier = min(self.clearance_multiplier * 1.8, 6.0)

            cycle_oil = 0.0
            cycle_water = 0.0
            cycle_lift_kwh = 0.0
            production_day = 0
            rod_state: dict[str, Any] | None = None
            float_days = 0

            for production_day in range(max_production_days):
                elapsed_years = calendar_day / 365.25
                if production_day % self.rod_update_interval_days == 0 or rod_state is None:
                    rod_state = self._solve_rods(
                        setpoint=setpoint,
                        elapsed_years=elapsed_years,
                        cycle_faults=cycle_faults,
                        production_day=production_day,
                    )
                step = self.reservoir.step_production(
                    step_days=1.0,
                    bottomhole_pressure_kpa=rod_state["bottomhole_pressure_kpa"],
                    rate_limit_m3_per_day=rod_state["pump_capacity_m3_per_day"],
                )
                oil_rate = step["oil_rate_m3_per_day"]
                water_rate = step["water_rate_m3_per_day"]
                cycle_oil += oil_rate
                cycle_water += water_rate
                cycle_lift_kwh += rod_state["daily_energy_kwh"]
                if rod_state["is_floating"]:
                    float_days += 1
                    self.cumulative_impact_events += rod_state["float_index"] * 1440.0

                pending_card = getattr(self, "_pending_card", None)
                if pending_card is not None:
                    pending_card["day"] = round(calendar_day, 3)
                    result.card_rows.append(pending_card)
                    self._pending_card = None

                samples_per_day = int(round(1440 / self.telemetry_interval_minutes))
                for sample in range(samples_per_day):
                    if sample % max(samples_per_day // 4, 1) != 0:
                        continue
                    result.telemetry_rows.append(
                        {
                            "well_id": self.hidden.well_id,
                            "timestamp_day": round(
                                calendar_day + sample * self.telemetry_interval_minutes / 1440.0,
                                5,
                            ),
                            "cycle_number": cycle_number,
                            "spm": setpoint.spm,
                            "stroke_length_m": setpoint.stroke_length_m,
                            "vfd_frequency_hz": setpoint.spm
                            / max(self.config.srp.spm_setpoint, 1e-6)
                            * self.config.srp.vfd_base_frequency_hz,
                            "motor_power_w": rod_state["motor_power_w"],
                            "peak_polished_rod_load_n": rod_state["peak_load_n"],
                            "minimum_polished_rod_load_n": rod_state["minimum_load_n"],
                            "fillage_frac": rod_state["fillage_frac"],
                            "pump_intake_temp_c": rod_state["pump_intake_temp_c"],
                            "wellhead_temp_c": rod_state["wellhead_temp_c"],
                            "casing_pressure_kpa": self.config.wellbore.wellhead_pressure_kpa,
                        }
                    )

                result.daily_rows.append(
                    {
                        "well_id": self.hidden.well_id,
                        "day": round(calendar_day, 3),
                        "cycle_number": cycle_number,
                        "production_day": production_day,
                        "oil_rate_m3_per_day": oil_rate,
                        "water_rate_m3_per_day": water_rate,
                        "water_cut_frac": step["water_cut_frac"],
                        "near_well_temp_c": self.reservoir.near_well_temperature_c(),
                        "average_pay_temp_c": self.reservoir.average_pay_temperature_c(),
                        "heated_radius_m": self.reservoir.heated_radius_m(),
                        "reservoir_pressure_kpa": self.reservoir.pressure_kpa,
                        "bottomhole_pressure_kpa": rod_state["bottomhole_pressure_kpa"],
                        "wellhead_temp_c": rod_state["wellhead_temp_c"],
                        "pump_intake_temp_c": rod_state["pump_intake_temp_c"],
                        "spm": setpoint.spm,
                        "stroke_length_m": setpoint.stroke_length_m,
                        "peak_polished_rod_load_n": rod_state["peak_load_n"],
                        "minimum_polished_rod_load_n": rod_state["minimum_load_n"],
                        "fillage_frac": rod_state["fillage_frac"],
                        "float_index": rod_state["float_index"],
                        "motor_power_w": rod_state["motor_power_w"],
                        "card_class": rod_state["card_class"],
                    }
                )
                calendar_day += 1.0
                if oil_rate < self.config.css.cutoff_oil_rate_m3_per_day:
                    break

            result.cycle_rows.append(
                {
                    "well_id": self.hidden.well_id,
                    "cycle_number": cycle_number,
                    "steam_volume_m3_cwe": plan.steam_volume_m3_cwe,
                    "injection_rate_m3_per_day_cwe": plan.injection_rate_m3_per_day_cwe,
                    "injection_pressure_kpa": plan.injection_pressure_kpa,
                    "steam_quality_frac": plan.steam_quality_frac,
                    "sandface_quality_frac": injection.sandface_quality_frac,
                    "soak_days": plan.soak_days,
                    "production_days": production_day + 1,
                    "oil_m3": cycle_oil,
                    "water_m3": cycle_water,
                    "steam_oil_ratio": plan.steam_volume_m3_cwe / max(cycle_oil, 1.0e-6),
                    "lift_energy_kwh": cycle_lift_kwh,
                    "float_event_days": float_days,
                    "cutoff_reason": "economic rate limit",
                    "faults": ";".join(sorted(cycle_faults)) if cycle_faults else "",
                }
            )
            for fault in cycle_faults:
                if fault in {"rod_parting", "pump_unseating"}:
                    result.failure_rows.append(
                        {
                            "well_id": self.hidden.well_id,
                            "day": round(calendar_day, 3),
                            "cycle_number": cycle_number,
                            "failure_type": fault,
                            "depth_m": float(
                                self.rng.uniform(0.3, 0.95) * self.config.srp.pump_depth_m
                            ),
                        }
                    )
                    if fault == "worn_pump":
                        self.clearance_multiplier = 1.0

        result.sensor_fault_log = self.sensors.log.as_dict()
        return result

    # ------------------------------------------------------------- rod solve
    def _solve_rods(
        self,
        setpoint: TruthSetpoint,
        elapsed_years: float,
        cycle_faults: set[str],
        production_day: int,
    ) -> dict[str, Any]:
        """Run the fine rod solver and the pump for the current well state."""
        srp = self.srp_config
        motion = self.kinematics.motion(
            setpoint.spm, setpoint.stroke_length_m, setpoint.profile(), CARD_SAMPLES
        )
        water_cut = self.reservoir.water_cut_frac()
        sandface_temp_c = self.reservoir.near_well_temperature_c()

        production = self.wellbore.produce(
            sandface_temp_c=sandface_temp_c,
            liquid_rate_m3_per_day=max(getattr(self, "_last_rate", 5.0), 0.1),
            water_cut_frac=water_cut,
            step_days=1.0,
            pump_depth_m=srp.pump_depth_m,
            elapsed_years=elapsed_years,
        )
        temperature_profile = np.interp(
            self.taper.depth_m, production.depth_m, production.temperature_c
        )
        viscosity_profile = self._oil_viscosity_pa_s(temperature_profile)

        density = 900.0 * (1.0 - water_cut) + 1000.0 * water_cut
        head_kpa = density * STANDARD_GRAVITY_M_PER_S2 * srp.pump_depth_m / 1000.0
        maximum_intake_kpa = self.config.wellbore.wellhead_pressure_kpa + head_kpa
        minimum_intake_kpa = self.config.wellbore.wellhead_pressure_kpa

        plunger_stroke_m = 0.9 * setpoint.stroke_length_m
        rate_m3_per_day = max(getattr(self, "_last_rate", 5.0), 0.1)
        bottomhole_kpa = minimum_intake_kpa
        fillage = 1.0
        solution = None
        capacity = 0.0

        for _ in range(2):
            capacity = srp.plunger_area_m2 * plunger_stroke_m * motion.spm * 1440.0
            upper_kpa = min(maximum_intake_kpa, self.reservoir.pressure_kpa)
            if self.reservoir.deliverability_m3_per_day(upper_kpa) > capacity:
                bottomhole_kpa = upper_kpa
                available = min(
                    self.reservoir.deliverability_m3_per_day(upper_kpa),
                    self.config.surface.max_well_rate_m3_per_day,
                )
            elif self.reservoir.deliverability_m3_per_day(minimum_intake_kpa) <= capacity:
                bottomhole_kpa = minimum_intake_kpa
                available = self.reservoir.deliverability_m3_per_day(minimum_intake_kpa)
            else:
                low, high = minimum_intake_kpa, upper_kpa
                for _ in range(40):
                    mid = 0.5 * (low + high)
                    if self.reservoir.deliverability_m3_per_day(mid) > capacity:
                        low = mid
                    else:
                        high = mid
                bottomhole_kpa = 0.5 * (low + high)
                available = capacity
            fillage = float(min(max(available / max(capacity, 1.0e-9), 0.0), 1.0))
            if "gas_interference" in cycle_faults:
                fillage = min(fillage, 0.85)

            fluid_load_n = (
                srp.plunger_area_m2
                * max((self.config.wellbore.wellhead_pressure_kpa + head_kpa - bottomhole_kpa), 0.0)
                * 1000.0
            )
            if "worn_pump" in cycle_faults:
                fluid_load_n *= 0.6
            if "pump_unseating" in cycle_faults:
                fluid_load_n *= 0.15

            drag = self._drag_profile(viscosity_profile, rate_m3_per_day)
            boundary = PumpBoundary(
                fluid_load_n=fluid_load_n,
                fillage_frac=fillage,
                gas_interference_frac=0.3 if "gas_interference" in cycle_faults else 0.0,
                plunger_friction_n=900.0 if "sticking" in cycle_faults else 0.0,
                stroke_length_m=plunger_stroke_m,
                tagging_contact_travel_m=(
                    0.9 * plunger_stroke_m if "tagging" in cycle_faults else 0.0
                ),
                sticking_load_n=6000.0 if "sticking" in cycle_faults else 0.0,
            )
            solution = solve_fine(
                motion=motion,
                taper=self.taper,
                config=srp,
                pump=boundary,
                viscous_coefficient_n_s_per_m2=drag.linear_coefficient_n_s_per_m2,
                static_drag_n_per_m=drag.static_force_n_per_m,
                fluid_density_kg_per_m3=density,
                cycles=3,
                courant=0.9,
            )
            plunger_stroke_m = max(solution.pump_stroke_m, 0.05)
            rate_m3_per_day = min(capacity * fillage, available)

        assert solution is not None
        self._last_rate = rate_m3_per_day

        threshold_n = srp.minimum_polished_rod_load_n
        float_index = float(np.mean(solution.surface_load_n < threshold_n))
        is_floating = float_index > 0.02

        work_j = float(
            np.trapezoid(
                np.maximum(solution.surface_load_n * motion.velocity_m_per_s, 0.0),
                dx=motion.cycle_time_s / solution.surface_load_n.size,
            )
        )
        polished_rod_power_w = work_j / motion.cycle_time_s
        shaft_w = polished_rod_power_w / (srp.gearbox_efficiency_frac * srp.belt_efficiency_frac)
        load_fraction = min(max(shaft_w / srp.motor_rated_power_w, 0.02), 1.2)
        efficiency = srp.motor_efficiency_peak_frac * (1.0 - 0.5 * (1.0 - load_fraction) ** 2)
        motor_power_w = shaft_w / max(efficiency, 0.15)

        card_class = self._label_card(cycle_faults, fillage, is_floating)
        if production_day % max(self.rod_update_interval_days, 1) == 0:
            self._record_card(solution, motion, card_class, production_day)

        return {
            "bottomhole_pressure_kpa": bottomhole_kpa,
            "pump_capacity_m3_per_day": capacity,
            "fillage_frac": fillage,
            "peak_load_n": solution.peak_polished_rod_load_n,
            "minimum_load_n": solution.minimum_polished_rod_load_n,
            "float_index": float_index,
            "is_floating": is_floating,
            "daily_energy_kwh": motor_power_w * 24.0 / 1000.0,
            "motor_power_w": motor_power_w,
            "wellhead_temp_c": production.wellhead_temp_c,
            "pump_intake_temp_c": production.pump_intake_temp_c,
            "card_class": card_class,
            "solution": solution,
            "motion": motion,
        }

    @staticmethod
    def _label_card(cycle_faults: set[str], fillage: float, is_floating: bool) -> str:
        """The true label of the card, for training and for scoring."""
        if "rod_parting" in cycle_faults:
            return str(CardClass.PARTED_ROD)
        if "pump_unseating" in cycle_faults:
            return str(CardClass.UNSEATING)
        if is_floating:
            return str(CardClass.ROD_FLOAT)
        if "tagging" in cycle_faults:
            return str(CardClass.TAGGING)
        if "sticking" in cycle_faults:
            return str(CardClass.STICKING)
        if "gas_interference" in cycle_faults:
            return str(CardClass.GAS_INTERFERENCE)
        if "worn_pump" in cycle_faults:
            return str(CardClass.WORN_PUMP)
        if fillage < 0.7:
            return str(CardClass.FLUID_POUND)
        return str(CardClass.NORMAL)

    def _record_card(
        self, solution: Any, motion: Any, card_class: str, production_day: int
    ) -> None:
        """Store a measured surface card for the telemetry table."""
        load, position = self.sensors.apply_card(
            f"{self.hidden.well_id}_card", solution.surface_load_n, solution.surface_position_m
        )
        self._pending_card = {
            "well_id": self.hidden.well_id,
            "production_day": production_day,
            "cycle_number": self.reservoir.cycle_number,
            "card_class": card_class,
            "spm": motion.spm,
            "stroke_length_m": motion.stroke_length_m,
            "position_m": np.round(position, 5).tolist(),
            "load_n": np.round(load, 1).tolist(),
            "pump_position_m": np.round(solution.pump_position_m, 5).tolist(),
            "pump_load_n": np.round(solution.pump_load_n, 1).tolist(),
        }
