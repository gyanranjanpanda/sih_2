"""Configuration models and loaders.

Every field constant used by the twin, the optimizers and the truth simulator
comes from YAML under ``config/``. Nothing physical is hard coded in logic.
The models below are pydantic v2 so a malformed or out-of-range value produces
a readable message naming the file, the key and the allowed range.
"""

from __future__ import annotations

import copy
import functools
import itertools
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.errors import ConfigError

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = _BACKEND_DIR.parent


def config_dir() -> Path:
    """Directory holding the YAML configuration, overridable with WELL_TWIN_CONFIG_DIR."""
    override = os.environ.get("WELL_TWIN_CONFIG_DIR")
    return Path(override).resolve() if override else REPO_ROOT / "config"


def data_dir() -> Path:
    """Directory holding generated and ingested data, overridable with WELL_TWIN_DATA_DIR."""
    override = os.environ.get("WELL_TWIN_DATA_DIR")
    return Path(override).resolve() if override else REPO_ROOT / "data"


def reports_dir() -> Path:
    """Directory holding metric reports, overridable with WELL_TWIN_REPORTS_DIR."""
    override = os.environ.get("WELL_TWIN_REPORTS_DIR")
    return Path(override).resolve() if override else REPO_ROOT / "reports"


def models_dir() -> Path:
    """Directory holding trained model artifacts, overridable with WELL_TWIN_MODELS_DIR."""
    override = os.environ.get("WELL_TWIN_MODELS_DIR")
    return Path(override).resolve() if override else REPO_ROOT / "models"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)


# --------------------------------------------------------------------------------------
# Field configuration
# --------------------------------------------------------------------------------------
class FieldInfo(_Base):
    """Identification of the field and the provenance mode of the data in use."""

    name: str
    operator: str
    basin: str
    formation: str
    data_mode: Literal["SYNTHETIC", "REAL"] = "SYNTHETIC"


class ViscosityAnchor(_Base):
    """One measured or assumed viscosity point used to fit the temperature model."""

    temp_c: float = Field(gt=-273.15, lt=500.0)
    viscosity_cp: float = Field(gt=0.0, lt=1.0e8)


class FluidConfig(_Base):
    """Crude, water and emulsion property settings."""

    api_gravity_deg: float = Field(gt=-10.0, lt=100.0)
    viscosity_anchors: list[ViscosityAnchor] = Field(min_length=2)
    viscosity_model: Literal["walther", "arrhenius"] = "walther"
    emulsion_model: Literal["richardson", "brinkman", "none"] = "richardson"
    emulsion_richardson_k: float = Field(ge=0.0, le=20.0)
    emulsion_inversion_water_cut_frac: float = Field(gt=0.0, lt=1.0)
    asphaltene_multiplier: float = Field(ge=1.0, le=10.0)
    asphaltene_enabled: bool = False
    non_newtonian_enabled: bool = False
    power_law_index: float = Field(gt=0.0, le=1.5)
    oil_specific_heat_j_per_kg_k: float = Field(gt=500.0, lt=5000.0)
    water_specific_heat_j_per_kg_k: float = Field(gt=3000.0, lt=5000.0)
    oil_thermal_expansion_per_k: float = Field(ge=0.0, lt=0.01)
    water_density_kg_per_m3: float = Field(gt=800.0, lt=1200.0)
    reference_temp_c: float = Field(gt=-273.15, lt=100.0)

    @model_validator(mode="after")
    def _anchors_must_be_distinct_and_monotone(self) -> FluidConfig:
        temps = [a.temp_c for a in self.viscosity_anchors]
        if len(set(temps)) != len(temps):
            raise ValueError("viscosity_anchors must be at distinct temperatures")
        ordered = sorted(self.viscosity_anchors, key=lambda a: a.temp_c)
        for lower, upper in itertools.pairwise(ordered):
            if upper.viscosity_cp >= lower.viscosity_cp:
                raise ValueError(
                    "viscosity must decrease as temperature rises: "
                    f"{lower.viscosity_cp} cP at {lower.temp_c} C then "
                    f"{upper.viscosity_cp} cP at {upper.temp_c} C"
                )
        return self


class ReservoirConfig(_Base):
    """Static reservoir and thermal rock properties for the reduced-order twin."""

    depth_m: float = Field(gt=0.0, lt=8000.0)
    net_pay_thickness_m: float = Field(gt=0.0, lt=500.0)
    porosity_frac: float = Field(gt=0.0, lt=0.6)
    initial_temp_c: float = Field(gt=-273.15, lt=400.0)
    initial_pressure_kpa: float = Field(gt=0.0, lt=100000.0)
    abandonment_pressure_kpa: float = Field(gt=0.0, lt=100000.0)
    permeability_md: float = Field(gt=0.0, lt=100000.0)
    skin_dimensionless: float = Field(gt=-5.0, lt=50.0)
    drainage_radius_m: float = Field(gt=0.0, lt=5000.0)
    wellbore_radius_m: float = Field(gt=0.0, lt=2.0)
    rock_volumetric_heat_capacity_j_per_m3_k: float = Field(gt=1.0e5, lt=1.0e7)
    overburden_thermal_conductivity_w_per_m_k: float = Field(gt=0.0, lt=20.0)
    overburden_volumetric_heat_capacity_j_per_m3_k: float = Field(gt=1.0e5, lt=1.0e7)
    reservoir_thermal_conductivity_w_per_m_k: float = Field(gt=0.0, lt=20.0)
    oil_saturation_initial_frac: float = Field(gt=0.0, le=1.0)
    residual_oil_saturation_frac: float = Field(ge=0.0, lt=1.0)
    total_compressibility_per_kpa: float = Field(gt=0.0, lt=1.0e-2)
    fracture_pressure_gradient_kpa_per_m: float = Field(gt=0.0, lt=50.0)
    fracture_enhancement_enabled: bool = False
    fracture_area_multiplier: float = Field(ge=1.0, le=5.0)
    dilation_enabled: bool = False
    water_cut_initial_frac: float = Field(ge=0.0, lt=1.0)
    water_cut_growth_per_cycle_frac: float = Field(ge=0.0, lt=0.5)
    water_cut_max_frac: float = Field(gt=0.0, lt=1.0)
    cycle_energy_retention_frac: float = Field(gt=0.0, le=1.0)
    productivity_decline_per_cycle_frac: float = Field(ge=0.0, lt=0.9)

    @model_validator(mode="after")
    def _check_ordering(self) -> ReservoirConfig:
        if self.residual_oil_saturation_frac >= self.oil_saturation_initial_frac:
            raise ValueError(
                "residual_oil_saturation_frac must be below oil_saturation_initial_frac"
            )
        if self.abandonment_pressure_kpa >= self.initial_pressure_kpa:
            raise ValueError("abandonment_pressure_kpa must be below initial_pressure_kpa")
        if self.drainage_radius_m <= self.wellbore_radius_m:
            raise ValueError("drainage_radius_m must exceed wellbore_radius_m")
        if self.water_cut_initial_frac >= self.water_cut_max_frac:
            raise ValueError("water_cut_initial_frac must be below water_cut_max_frac")
        return self

    @property
    def fracture_pressure_kpa(self) -> float:
        """Formation fracture pressure at reservoir depth."""
        return self.fracture_pressure_gradient_kpa_per_m * self.depth_m


class WellboreConfig(_Base):
    """Tubing, casing and heat transfer settings for the wellbore."""

    tubing_outer_diameter_m: float = Field(gt=0.0, lt=1.0)
    tubing_inner_diameter_m: float = Field(gt=0.0, lt=1.0)
    casing_inner_diameter_m: float = Field(gt=0.0, lt=1.0)
    insulation_type: Literal["vit", "bare"] = "vit"
    vit_overall_heat_transfer_w_per_m2_k: float = Field(gt=0.0, lt=50.0)
    bare_overall_heat_transfer_w_per_m2_k: float = Field(gt=0.0, lt=200.0)
    formation_thermal_conductivity_w_per_m_k: float = Field(gt=0.0, lt=20.0)
    formation_thermal_diffusivity_m2_per_s: float = Field(gt=0.0, lt=1.0e-4)
    surface_temp_c: float = Field(gt=-60.0, lt=80.0)
    geothermal_gradient_c_per_m: float = Field(ge=0.0, lt=0.2)
    tubing_roughness_m: float = Field(gt=0.0, lt=0.01)
    wellhead_pressure_kpa: float = Field(ge=0.0, lt=20000.0)

    @model_validator(mode="after")
    def _check_diameters(self) -> WellboreConfig:
        if self.tubing_inner_diameter_m >= self.tubing_outer_diameter_m:
            raise ValueError("tubing_inner_diameter_m must be below tubing_outer_diameter_m")
        if self.casing_inner_diameter_m <= self.tubing_outer_diameter_m:
            raise ValueError("casing_inner_diameter_m must exceed tubing_outer_diameter_m")
        return self

    @property
    def overall_heat_transfer_w_per_m2_k(self) -> float:
        """Active overall heat transfer coefficient for the configured completion."""
        if self.insulation_type == "vit":
            return self.vit_overall_heat_transfer_w_per_m2_k
        return self.bare_overall_heat_transfer_w_per_m2_k


class RodSection(_Base):
    """One taper section of the rod string, ordered from surface downwards."""

    diameter_m: float = Field(gt=0.0, lt=0.2)
    length_m: float = Field(gt=0.0, lt=5000.0)
    grade: str
    minimum_tensile_strength_pa: float = Field(gt=1.0e8, lt=2.0e9)

    @property
    def area_m2(self) -> float:
        """Cross-sectional steel area of the section."""
        import math

        return math.pi * 0.25 * self.diameter_m**2


class SrpConfig(_Base):
    """Sucker rod pump unit, rod string, drive and pump settings."""

    unit_type: Literal["conventional", "hydraulic"] = "conventional"
    pump_depth_m: float = Field(gt=0.0, lt=6000.0)
    plunger_diameter_m: float = Field(gt=0.0, lt=0.3)
    pump_clearance_m: float = Field(gt=0.0, lt=0.01)
    stroke_length_m: float = Field(gt=0.0, lt=15.0)
    stroke_length_options_m: list[float] = Field(min_length=1)
    spm_setpoint: float = Field(gt=0.0, lt=30.0)
    spm_min: float = Field(gt=0.0, lt=30.0)
    spm_max: float = Field(gt=0.0, lt=30.0)
    spm_rate_limit_per_step: float = Field(gt=0.0, lt=10.0)
    acoustic_velocity_m_per_s: float = Field(gt=1000.0, lt=10000.0)
    steel_density_kg_per_m3: float = Field(gt=5000.0, lt=10000.0)
    steel_youngs_modulus_pa: float = Field(gt=1.0e11, lt=4.0e11)
    damping_factor_dimensionless: float = Field(gt=0.0, lt=5.0)
    damping_viscosity_exponent: float = Field(ge=0.0, le=1.0)
    damping_factor_max: float = Field(gt=0.0, lt=5.0)
    wave_grid_nodes_per_section: int = Field(ge=4, le=400)
    wave_time_steps_per_cycle: int = Field(ge=100, le=20000)
    rod_sections: list[RodSection] = Field(min_length=1)
    service_factor_dimensionless: float = Field(gt=0.0, le=1.5)
    structural_load_rating_n: float = Field(gt=0.0, lt=1.0e7)
    gearbox_torque_rating_n_m: float = Field(gt=0.0, lt=1.0e7)
    counterbalance_moment_n_m: float = Field(ge=0.0, lt=1.0e7)
    pitman_length_m: float = Field(gt=0.0, lt=20.0)
    crank_to_saddle_distance_m: float = Field(gt=0.0, lt=30.0)
    walking_beam_rear_arm_m: float = Field(gt=0.0, lt=20.0)
    walking_beam_front_arm_m: float = Field(gt=0.0, lt=20.0)
    motor_rated_power_w: float = Field(gt=0.0, lt=1.0e6)
    motor_efficiency_peak_frac: float = Field(gt=0.0, le=1.0)
    gearbox_efficiency_frac: float = Field(gt=0.0, le=1.0)
    belt_efficiency_frac: float = Field(gt=0.0, le=1.0)
    vfd_frequency_min_hz: float = Field(gt=0.0, lt=200.0)
    vfd_frequency_max_hz: float = Field(gt=0.0, lt=200.0)
    vfd_base_frequency_hz: float = Field(gt=0.0, lt=200.0)
    hydraulic_max_pressure_kpa: float = Field(gt=0.0, lt=100000.0)
    hydraulic_max_flow_m3_per_s: float = Field(gt=0.0, lt=1.0)
    hydraulic_cylinder_area_m2: float = Field(gt=0.0, lt=1.0)
    hydraulic_upstroke_speed_frac: float = Field(gt=0.0, le=3.0)
    hydraulic_downstroke_speed_frac: float = Field(gt=0.0, le=3.0)
    hydraulic_dwell_s: float = Field(ge=0.0, lt=60.0)
    float_margin_minimum_frac: float = Field(ge=0.0, lt=1.0)
    minimum_polished_rod_load_n: float = Field(ge=0.0, lt=1.0e5)
    minimum_fillage_frac: float = Field(gt=0.0, le=1.0)
    gas_interference_frac: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_ranges(self) -> SrpConfig:
        if self.spm_min >= self.spm_max:
            raise ValueError("spm_min must be below spm_max")
        if not self.spm_min <= self.spm_setpoint <= self.spm_max:
            raise ValueError("spm_setpoint must lie between spm_min and spm_max")
        if self.vfd_frequency_min_hz >= self.vfd_frequency_max_hz:
            raise ValueError("vfd_frequency_min_hz must be below vfd_frequency_max_hz")
        if any(value <= 0.0 for value in self.stroke_length_options_m):
            raise ValueError("stroke_length_options_m entries must be positive")
        total_rod_length_m = sum(section.length_m for section in self.rod_sections)
        if abs(total_rod_length_m - self.pump_depth_m) > 0.02 * self.pump_depth_m:
            raise ValueError(
                "rod_sections total length "
                f"({total_rod_length_m:.1f} m) must match pump_depth_m "
                f"({self.pump_depth_m:.1f} m) within 2 percent"
            )
        return self

    @property
    def total_rod_length_m(self) -> float:
        """Sum of the taper section lengths."""
        return sum(section.length_m for section in self.rod_sections)

    @property
    def plunger_area_m2(self) -> float:
        """Plunger cross-sectional area."""
        import math

        return math.pi * 0.25 * self.plunger_diameter_m**2


class SurfaceConfig(_Base):
    """Tank heating, bowser logistics and mobile steam generator settings."""

    tank_volume_m3: float = Field(gt=0.0, lt=10000.0)
    tank_target_temp_c: float = Field(gt=-273.15, lt=200.0)
    tank_ambient_temp_c: float = Field(gt=-60.0, lt=80.0)
    tank_heat_loss_coefficient_w_per_m2_k: float = Field(gt=0.0, lt=100.0)
    tank_surface_area_m2: float = Field(gt=0.0, lt=10000.0)
    bowser_capacity_m3: float = Field(gt=0.0, lt=1000.0)
    steam_generator_thermal_power_w: float = Field(gt=0.0, lt=1.0e9)
    steam_generator_efficiency_frac: float = Field(gt=0.0, le=1.0)
    steam_generator_count: int = Field(ge=1, le=50)
    rig_move_days: float = Field(ge=0.0, lt=30.0)


class CssConfig(_Base):
    """Cyclic steam stimulation design defaults for the next cycle."""

    steam_volume_m3_cwe: float = Field(gt=0.0, lt=100000.0)
    injection_rate_m3_per_day_cwe: float = Field(gt=0.0, lt=5000.0)
    injection_pressure_kpa: float = Field(gt=0.0, lt=50000.0)
    steam_quality_frac: float = Field(gt=0.0, le=1.0)
    soak_days: float = Field(ge=0.0, lt=120.0)
    production_days_max: float = Field(gt=0.0, lt=2000.0)
    cutoff_oil_rate_m3_per_day: float = Field(ge=0.0, lt=1000.0)
    cutoff_marginal_energy_ratio: float = Field(gt=0.0, lt=10.0)
    cycles_planned: int = Field(ge=1, le=50)

    @property
    def injection_days(self) -> float:
        """Days required to inject the design steam volume at the design rate."""
        return self.steam_volume_m3_cwe / self.injection_rate_m3_per_day_cwe


class EconomicsConfig(_Base):
    """Illustrative prices and costs. Not Oil India commercial data."""

    oil_price_usd_per_m3: float = Field(gt=0.0, lt=10000.0)
    fuel_cost_usd_per_gj: float = Field(gt=0.0, lt=1000.0)
    electricity_cost_usd_per_kwh: float = Field(gt=0.0, lt=100.0)
    rod_job_cost_usd: float = Field(ge=0.0, lt=1.0e7)
    pump_unseating_cost_usd: float = Field(ge=0.0, lt=1.0e7)
    deferred_oil_penalty_usd_per_m3_day: float = Field(ge=0.0, lt=1000.0)


class ProvenanceEntry(_Base):
    """Where one configured value came from and how much to trust it."""

    assumed: bool
    source: str
    confidence: Literal["high", "medium", "low"]


class FieldConfig(_Base):
    """Root configuration object assembled from ``config/field.yaml`` plus well overlays."""

    field: FieldInfo
    fluid: FluidConfig
    reservoir: ReservoirConfig
    wellbore: WellboreConfig
    srp: SrpConfig
    surface: SurfaceConfig
    css: CssConfig
    economics: EconomicsConfig
    provenance: dict[str, ProvenanceEntry] = Field(default_factory=dict)
    well_id: str = "TEMPLATE"

    def assumed_keys(self) -> list[str]:
        """Dotted keys whose values are engineering placeholders, not field data."""
        return sorted(key for key, entry in self.provenance.items() if entry.assumed)

    def with_overrides(self, overrides: dict[str, Any]) -> FieldConfig:
        """Return a new config with a nested dictionary of overrides merged in."""
        merged = _deep_merge(self.model_dump(), overrides)
        return _build_field_config(merged, source="override")


# --------------------------------------------------------------------------------------
# Optimizer configuration
# --------------------------------------------------------------------------------------
class VariableBound(_Base):
    """Inclusive lower and upper bound for a decision variable."""

    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> VariableBound:
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be below high ({self.high})")
        return self

    def clamp(self, value: float) -> float:
        """Clamp a value into the bound."""
        return min(max(value, self.low), self.high)


class CssObjectiveWeights(_Base):
    """Scalarisation weights used to pick one point off the Pareto front."""

    cycle_oil_m3: float = Field(ge=0.0)
    steam_oil_ratio: float = Field(ge=0.0)
    energy_cost_usd: float = Field(ge=0.0)


class CssConstraints(_Base):
    """Hard constraints applied to every CSS candidate."""

    fracture_pressure_safety_frac: float = Field(gt=0.0, le=1.0)
    min_heated_radius_m: float = Field(gt=0.0, lt=1000.0)
    max_injection_days: float = Field(gt=0.0, lt=365.0)
    min_float_margin_frac: float = Field(ge=0.0, lt=1.0)
    max_rod_stress_utilization_frac: float = Field(gt=0.0, le=1.0)


class RobustConfig(_Base):
    """Settings for evaluating candidates across a parameter ensemble."""

    enabled: bool = True
    ensemble_size: int = Field(ge=2, le=200)
    percentile: float = Field(ge=0.0, le=100.0)
    parameter_spread_frac: float = Field(gt=0.0, lt=1.0)


class SurrogateConfig(_Base):
    """Settings for the optional surrogate used inside the CSS search."""

    enabled: bool = True
    training_samples: int = Field(ge=10, le=10000)
    revalidate_every_n_calls: int = Field(ge=1, le=100000)
    max_acceptable_mape_frac: float = Field(gt=0.0, lt=1.0)


class CssOptimizerConfig(_Base):
    """CSS cycle optimizer settings."""

    algorithm: Literal["nsga2"] = "nsga2"
    population_size: int = Field(ge=8, le=1000)
    generations: int = Field(ge=1, le=1000)
    seed: int
    variables: dict[str, VariableBound]
    objective_weights: CssObjectiveWeights
    constraints: CssConstraints
    robust: RobustConfig
    surrogate: SurrogateConfig


class PumpObjectiveWeights(_Base):
    """Objective weights for the pump controller."""

    production_weight: float = Field(ge=0.0)
    energy_weight: float = Field(ge=0.0)
    float_penalty_weight: float = Field(ge=0.0)
    load_penalty_weight: float = Field(ge=0.0)


class PumpConstraints(_Base):
    """Hard constraints applied to every pump setpoint candidate."""

    min_float_margin_frac: float = Field(ge=0.0, lt=1.0)
    max_structural_load_frac: float = Field(gt=0.0, le=1.0)
    max_gearbox_torque_frac: float = Field(gt=0.0, le=1.0)
    min_fillage_frac: float = Field(gt=0.0, le=1.0)
    min_polished_rod_load_n: float = Field(ge=0.0, lt=1.0e5)
    max_stroke_change_per_step_m: float = Field(gt=0.0, lt=10.0)
    max_spm_change_per_step: float = Field(gt=0.0, lt=20.0)


class PumpOptimizerConfig(_Base):
    """Pump control optimizer settings."""

    algorithm: Literal["mpc_local_search"] = "mpc_local_search"
    horizon_days: float = Field(gt=0.0, lt=365.0)
    horizon_steps: int = Field(ge=1, le=200)
    seed: int
    restarts: int = Field(ge=1, le=100)
    iterations: int = Field(ge=1, le=10000)
    variables: dict[str, VariableBound]
    objective: PumpObjectiveWeights
    constraints: PumpConstraints


class FleetOptimizerConfig(_Base):
    """Fleet steam scheduler settings."""

    solver: Literal["cp_sat"] = "cp_sat"
    max_solve_seconds: float = Field(gt=0.0, lt=3600.0)
    horizon_days: int = Field(ge=1, le=3650)
    baseline_rule: Literal["earliest_due_first"] = "earliest_due_first"
    deferred_oil_weight: float = Field(ge=0.0)
    makespan_weight: float = Field(ge=0.0)


class GuardConfig(_Base):
    """Supervisory guard settings."""

    mode: Literal["advisory", "auto"] = "advisory"
    clamp_to_config_bounds: bool = True
    max_relative_change_per_step_frac: float = Field(gt=0.0, le=1.0)
    reject_on_nan: bool = True


class OptimizerConfig(_Base):
    """Root optimizer configuration from ``config/optimizer.yaml``."""

    css: CssOptimizerConfig
    pump: PumpOptimizerConfig
    fleet: FleetOptimizerConfig
    guard: GuardConfig


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, raising ConfigError with the path on any problem."""
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}", path=str(path))
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}", path=str(path)) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Expected a mapping at the top level of {path}, found {type(loaded).__name__}.",
            path=str(path),
        )
    return loaded


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"{len(exc.errors())} configuration problem(s) in {source}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def _build_field_config(payload: dict[str, Any], source: str) -> FieldConfig:
    try:
        return FieldConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source), source=source) from exc


def load_field_config(path: Path | None = None) -> FieldConfig:
    """Load and validate the field-level configuration."""
    target = path or (config_dir() / "field.yaml")
    return _build_field_config(read_yaml(target), source=str(target))


def load_well_config(well_id: str, base: FieldConfig | None = None) -> FieldConfig:
    """Load the field configuration with the overlay for one well merged on top.

    A well overlay is ``config/wells/<well_id>.yaml`` and may override any subset
    of the field configuration. Missing overlays are not an error: the field
    template is returned with the well id stamped on it.
    """
    field_config = base or load_field_config()
    overlay_path = config_dir() / "wells" / f"{well_id}.yaml"
    payload = field_config.model_dump()
    payload["well_id"] = well_id
    if overlay_path.exists():
        overlay = read_yaml(overlay_path)
        overlay.pop("well_id", None)
        payload = _deep_merge(payload, overlay)
        payload["well_id"] = well_id
    return _build_field_config(payload, source=str(overlay_path))


def list_well_ids() -> list[str]:
    """Well ids that have an overlay file under ``config/wells``."""
    wells_path = config_dir() / "wells"
    if not wells_path.exists():
        return []
    return sorted(p.stem for p in wells_path.glob("*.yaml"))


def load_optimizer_config(path: Path | None = None) -> OptimizerConfig:
    """Load and validate the optimizer configuration."""
    target = path or (config_dir() / "optimizer.yaml")
    payload = read_yaml(target)
    try:
        return OptimizerConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, str(target)), source=str(target)) from exc


@functools.lru_cache(maxsize=1)
def get_field_config() -> FieldConfig:
    """Process-wide cached field configuration."""
    return load_field_config()


@functools.lru_cache(maxsize=1)
def get_optimizer_config() -> OptimizerConfig:
    """Process-wide cached optimizer configuration."""
    return load_optimizer_config()


def clear_config_cache() -> None:
    """Drop cached configuration. Used by tests and by the calibration workflow."""
    get_field_config.cache_clear()
    get_optimizer_config.cache_clear()
