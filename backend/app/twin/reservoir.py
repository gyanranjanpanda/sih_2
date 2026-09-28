"""Reduced-order thermal reservoir model for a cyclic steam stimulation well.

Structure of the model, in the order the physics is applied:

1. Injection. Marx and Langenheim (1959) gives the growth of the heated area
   from the net heat rate arriving at the sandface.
2. Soak and production. A lumped energy balance on the heated zone, with the
   conduction decay taken from the product of the vertical and radial unit
   solutions in the manner of Boberg and Lantz (1966), and with the enthalpy
   carried away by produced fluid subtracted explicitly.
3. Inflow. Steady-state composite radial flow through a hot inner region and a
   cold outer region, with a pressure decline term for the depleted reservoir.
4. Cycle decline. Fitted multipliers for the loss of productivity and the rise
   in water cut from one cycle to the next.

Every equation below states its units and source. The twin never reads the
truth simulator: all parameters come from configuration or from calibration
against measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf, erfcx

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import require_finite, require_positive
from app.core.steam import injected_heat_rate_w, saturation_temperature_c
from app.core.units import SECONDS_PER_DAY
from app.twin.fluid import FluidModel

MILLIDARCY_TO_M2 = 9.869233e-16
"""Exact conversion from millidarcy to square metres."""

MIN_SOAK_CLOCK_DAYS = 0.25
"""Lower bound on the conduction clock, so the decay rate stays finite at t = 0."""


class CyclePhase(StrEnum):
    """Phase of the cyclic steam stimulation cycle."""

    INJECTION = "injection"
    SOAK = "soak"
    PRODUCTION = "production"
    IDLE = "idle"


# --------------------------------------------------------------------------------------
# Analytic building blocks
# --------------------------------------------------------------------------------------
def marx_langenheim_g(dimensionless_time: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Marx and Langenheim heated-area function G(t_D).

    Equation: G(t_D) = exp(t_D) erfc(sqrt(t_D)) + 2 sqrt(t_D / pi) - 1.
    Units: dimensionless in, dimensionless out.
    Implementation note: ``exp(t_D) erfc(sqrt(t_D))`` is evaluated with the
    scaled complementary error function erfcx, so the product stays accurate for
    large t_D where exp and erfc individually overflow and underflow.
    Limiting case: G(t_D) approaches t_D as t_D goes to zero, which is the
    no-conduction-loss case used in the tests.
    Source: Marx, J. W. and Langenheim, R. H. (1959), Reservoir heating by hot
    fluid injection, Trans. AIME 216, 312-315.
    """
    t_d = np.asarray(dimensionless_time, dtype=float)
    if np.any(t_d < 0.0):
        raise PhysicsDomainError("Dimensionless time cannot be negative.")
    root = np.sqrt(t_d)
    value = erfcx(root) + 2.0 * root / math.sqrt(math.pi) - 1.0
    return np.maximum(np.asarray(value, dtype=float), 0.0)


def marx_langenheim_area_m2(
    net_heat_rate_w: float,
    elapsed_days: float,
    net_pay_thickness_m: float,
    temperature_rise_k: float,
    rock_volumetric_heat_capacity_j_per_m3_k: float,
    overburden_thermal_conductivity_w_per_m_k: float,
    overburden_volumetric_heat_capacity_j_per_m3_k: float,
) -> float:
    """Heated area after a period of constant-rate heat injection.

    Equation:
        t_D = 4 K_ob M_ob t / (M_R^2 h^2)
        A_s = Q_i M_R h / (4 K_ob M_ob dT) * G(t_D)
    Units: Q_i in W, t in days (converted to seconds inside), h in m, dT in K,
    M in J/m3/K, K in W/m/K. Returns square metres.
    Assumptions: constant injection rate and steam temperature, a sharp steam
    front with a step temperature profile, conduction loss only to the
    over- and underburden, no gravity override, no pressure dependence.
    Limiting case: with zero overburden conductivity the area reduces to
    Q_i t / (M_R h dT), which the tests check.
    Source: Marx and Langenheim (1959); Butler, Thermal Recovery of Oil and
    Bitumen (1991), chapter 7.
    """
    require_positive(net_pay_thickness_m, "net_pay_thickness_m")
    require_positive(temperature_rise_k, "temperature_rise_k")
    require_positive(
        rock_volumetric_heat_capacity_j_per_m3_k, "rock_volumetric_heat_capacity_j_per_m3_k"
    )
    if net_heat_rate_w < 0.0 or elapsed_days < 0.0:
        raise PhysicsDomainError(
            "Heat rate and elapsed time must be non-negative.",
            net_heat_rate_w=net_heat_rate_w,
            elapsed_days=elapsed_days,
        )
    elapsed_s = elapsed_days * SECONDS_PER_DAY
    conduction_group = (
        overburden_thermal_conductivity_w_per_m_k * overburden_volumetric_heat_capacity_j_per_m3_k
    )
    if conduction_group <= 0.0:
        # No conduction loss: all injected heat is stored in the pay.
        return (
            net_heat_rate_w
            * elapsed_s
            / (rock_volumetric_heat_capacity_j_per_m3_k * net_pay_thickness_m * temperature_rise_k)
        )
    dimensionless_time = (
        4.0
        * conduction_group
        * elapsed_s
        / (rock_volumetric_heat_capacity_j_per_m3_k**2 * net_pay_thickness_m**2)
    )
    prefactor = (
        net_heat_rate_w
        * rock_volumetric_heat_capacity_j_per_m3_k
        * net_pay_thickness_m
        / (4.0 * conduction_group * temperature_rise_k)
    )
    area_m2 = float(prefactor * marx_langenheim_g(dimensionless_time))
    return require_finite(area_m2, "marx_langenheim_area_m2")


def vertical_conduction_unit_solution(
    elapsed_days: float, net_pay_thickness_m: float, thermal_diffusivity_m2_per_s: float
) -> float:
    """Fraction of the initial excess heat still inside a cooling slab.

    Equation: with X = h / (2 sqrt(alpha t)),
        f_v = erf(X) + (exp(-X^2) - 1) / (X sqrt(pi)).
    This is the exact average over a slab of thickness h that starts at a
    uniform excess temperature inside an infinite medium of the same
    diffusivity, obtained by integrating the two-error-function solution.
    Units: dimensionless. Time in days, thickness in m, diffusivity in m2/s.
    Limiting cases: f_v goes to 1 as t goes to 0 and to 0 as t goes to infinity.
    Source: Carslaw and Jaeger, Conduction of Heat in Solids, section 2.4;
    used in the Boberg and Lantz (1966) sense as the vertical unit solution.
    """
    require_positive(net_pay_thickness_m, "net_pay_thickness_m")
    require_positive(thermal_diffusivity_m2_per_s, "thermal_diffusivity_m2_per_s")
    elapsed_s = max(elapsed_days, 0.0) * SECONDS_PER_DAY
    if elapsed_s <= 0.0:
        return 1.0
    x = net_pay_thickness_m / (2.0 * math.sqrt(thermal_diffusivity_m2_per_s * elapsed_s))
    if x > 30.0:
        return 1.0
    if x < 1.0e-8:
        return 0.0
    value = float(erf(x)) + (math.exp(-(x**2)) - 1.0) / (x * math.sqrt(math.pi))
    return min(max(value, 0.0), 1.0)


def radial_conduction_unit_solution(
    elapsed_days: float, heated_radius_m: float, thermal_diffusivity_m2_per_s: float
) -> float:
    """Fraction of the initial excess heat still inside a cooling cylinder.

    Equation: f_r = 1 / (1 + 4 alpha t / r_h^2).
    Units: dimensionless. Time in days, radius in m, diffusivity in m2/s.
    Assumptions: this is the standard two-parameter approximation to the radial
    unit solution. It is exact in both limits: it equals 1 at t = 0, and for
    large t it reproduces the r_h^2 / (4 alpha t) decay of a line-source
    Gaussian averaged over the original cylinder. It is used in place of the
    tabulated Boberg and Lantz radial solution so the model stays analytic.
    Source: Boberg, T. C. and Lantz, R. B. (1966), Calculation of the production
    rate of a thermally stimulated well, J. Pet. Tech. 18(12), 1613-1623;
    asymptote from Carslaw and Jaeger, section 10.3.
    """
    require_positive(heated_radius_m, "heated_radius_m")
    require_positive(thermal_diffusivity_m2_per_s, "thermal_diffusivity_m2_per_s")
    elapsed_s = max(elapsed_days, 0.0) * SECONDS_PER_DAY
    return 1.0 / (1.0 + 4.0 * thermal_diffusivity_m2_per_s * elapsed_s / heated_radius_m**2)


def conduction_retention(
    elapsed_days: float,
    net_pay_thickness_m: float,
    heated_radius_m: float,
    thermal_diffusivity_m2_per_s: float,
) -> float:
    """Product of the vertical and radial unit solutions.

    This is the fraction of the heat placed in the heated zone that is still
    there after ``elapsed_days`` if nothing is produced. The Boberg and Lantz
    heat balance uses exactly this product.
    """
    return vertical_conduction_unit_solution(
        elapsed_days, net_pay_thickness_m, thermal_diffusivity_m2_per_s
    ) * radial_conduction_unit_solution(elapsed_days, heated_radius_m, thermal_diffusivity_m2_per_s)


def conduction_decay_rate_per_s(
    elapsed_days: float,
    net_pay_thickness_m: float,
    heated_radius_m: float,
    thermal_diffusivity_m2_per_s: float,
) -> float:
    """Instantaneous fractional heat loss rate of the heated zone.

    Equation: lambda(t) = -d ln(f_v f_r) / dt, evaluated by a centred finite
    difference on the analytic unit solutions.
    Units: 1/s. Returns a non-negative value.
    """
    step_days = max(0.02 * max(elapsed_days, MIN_SOAK_CLOCK_DAYS), 1.0e-3)
    clock = max(elapsed_days, MIN_SOAK_CLOCK_DAYS)
    ahead = conduction_retention(
        clock + step_days, net_pay_thickness_m, heated_radius_m, thermal_diffusivity_m2_per_s
    )
    behind = conduction_retention(
        max(clock - step_days, 1.0e-4),
        net_pay_thickness_m,
        heated_radius_m,
        thermal_diffusivity_m2_per_s,
    )
    ahead = max(ahead, 1.0e-12)
    behind = max(behind, 1.0e-12)
    derivative_per_day = (math.log(ahead) - math.log(behind)) / (2.0 * step_days)
    return max(-derivative_per_day, 0.0) / SECONDS_PER_DAY


def composite_radial_productivity_ratio(
    heated_radius_m: float,
    wellbore_radius_m: float,
    drainage_radius_m: float,
    hot_viscosity_pa_s: float,
    cold_viscosity_pa_s: float,
    skin_dimensionless: float = 0.0,
) -> float:
    """Ratio of stimulated to unstimulated inflow at the same drawdown.

    Equation:
        q_h / q_c = mu_c (ln(r_e / r_w) + S)
                    / (mu_h (ln(r_h / r_w) + S) + mu_c ln(r_e / r_h))
    Units: dimensionless. Radii in m, viscosities in Pa.s.
    Assumptions: steady-state radial flow, two concentric regions with a sharp
    viscosity contrast, the same absolute permeability in both regions, the
    skin acting inside the hot region.
    Limiting cases: the ratio is exactly 1 when r_h equals r_w, and tends to
    mu_c / mu_h as r_h approaches r_e. Both are covered by tests.
    Source: standard composite radial inflow, see Prats, Thermal Recovery,
    SPE Monograph 7, chapter 5.
    """
    require_positive(wellbore_radius_m, "wellbore_radius_m")
    require_positive(hot_viscosity_pa_s, "hot_viscosity_pa_s")
    require_positive(cold_viscosity_pa_s, "cold_viscosity_pa_s")
    if drainage_radius_m <= wellbore_radius_m:
        raise PhysicsDomainError("Drainage radius must exceed wellbore radius.")
    heated_radius_m = min(max(heated_radius_m, wellbore_radius_m), drainage_radius_m)
    outer_term = math.log(drainage_radius_m / wellbore_radius_m) + skin_dimensionless
    if outer_term <= 0.0:
        raise PhysicsDomainError("Geometry and skin give a non-positive cold flow resistance.")
    inner = hot_viscosity_pa_s * (
        math.log(heated_radius_m / wellbore_radius_m) + skin_dimensionless
    )
    outer = cold_viscosity_pa_s * math.log(drainage_radius_m / heated_radius_m)
    denominator = inner + outer
    if denominator <= 0.0:
        raise PhysicsDomainError("Composite inflow denominator is non-positive.")
    return float(cold_viscosity_pa_s * outer_term / denominator)


def darcy_liquid_rate_m3_per_s(
    permeability_m2: float,
    net_pay_thickness_m: float,
    drawdown_pa: float,
    viscosity_pa_s: float,
    drainage_radius_m: float,
    wellbore_radius_m: float,
    skin_dimensionless: float = 0.0,
) -> float:
    """Steady-state single-region radial Darcy inflow.

    Equation: q = 2 pi k h dp / (mu (ln(r_e / r_w) + S)).
    Units: k in m2, h in m, dp in Pa, mu in Pa.s, radii in m. Returns m3/s.
    Assumptions: incompressible single-phase steady radial flow, no turbulence.
    Source: Darcy's law in radial coordinates, Dake, Fundamentals of Reservoir
    Engineering, chapter 4.
    """
    require_positive(viscosity_pa_s, "viscosity_pa_s")
    if drawdown_pa <= 0.0:
        return 0.0
    resistance = math.log(drainage_radius_m / wellbore_radius_m) + skin_dimensionless
    if resistance <= 0.0:
        raise PhysicsDomainError("Flow resistance term is non-positive.")
    return (
        2.0
        * math.pi
        * permeability_m2
        * net_pay_thickness_m
        * drawdown_pa
        / (viscosity_pa_s * resistance)
    )


def corey_oil_relative_permeability(
    oil_saturation_frac: float,
    residual_oil_saturation_frac: float,
    initial_oil_saturation_frac: float,
    exponent: float = 2.0,
) -> float:
    """Corey style relative permeability to oil.

    Equation: k_ro = ((S_o - S_or) / (S_oi - S_or))^n.
    Units: dimensionless. Saturations as fractions.
    Assumptions: two-phase oil and water, end-point relative permeability of 1
    at the initial oil saturation, Corey exponent from config.
    Source: Corey (1954), as summarised in Dake chapter 4.
    """
    span = initial_oil_saturation_frac - residual_oil_saturation_frac
    if span <= 0.0:
        raise PhysicsDomainError("Initial oil saturation must exceed residual.")
    normalised = (oil_saturation_frac - residual_oil_saturation_frac) / span
    return float(min(max(normalised, 0.0), 1.0) ** exponent)


# --------------------------------------------------------------------------------------
# Calibratable parameters and state
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ReservoirParameters:
    """The subset of reservoir properties that calibration and assimilation adjust."""

    permeability_md: float
    skin_dimensionless: float
    net_pay_thickness_m: float
    thermal_loss_multiplier: float = 1.0
    cycle_energy_retention_frac: float = 0.86
    productivity_decline_per_cycle_frac: float = 0.08
    water_cut_initial_frac: float = 0.12
    water_cut_growth_per_cycle_frac: float = 0.06

    @classmethod
    def from_config(cls, config: FieldConfig) -> ReservoirParameters:
        """Build the default parameter set from configuration."""
        reservoir = config.reservoir
        return cls(
            permeability_md=reservoir.permeability_md,
            skin_dimensionless=reservoir.skin_dimensionless,
            net_pay_thickness_m=reservoir.net_pay_thickness_m,
            thermal_loss_multiplier=1.0,
            cycle_energy_retention_frac=reservoir.cycle_energy_retention_frac,
            productivity_decline_per_cycle_frac=reservoir.productivity_decline_per_cycle_frac,
            water_cut_initial_frac=reservoir.water_cut_initial_frac,
            water_cut_growth_per_cycle_frac=reservoir.water_cut_growth_per_cycle_frac,
        )

    def perturbed(self, **updates: float) -> ReservoirParameters:
        """Return a copy with the named fields replaced."""
        return replace(self, **updates)


@dataclass
class ReservoirState:
    """Mutable state of the reduced-order reservoir model."""

    day: float = 0.0
    cycle_number: int = 1
    phase: CyclePhase = CyclePhase.IDLE
    phase_day: float = 0.0
    conduction_clock_days: float = MIN_SOAK_CLOCK_DAYS
    heated_area_m2: float = 0.0
    heated_radius_m: float = 0.0
    steam_zone_temp_c: float = 0.0
    heated_zone_energy_j: float = 0.0
    average_heated_temp_c: float = 0.0
    reservoir_pressure_kpa: float = 0.0
    oil_saturation_frac: float = 0.0
    water_cut_frac: float = 0.0
    oil_rate_m3_per_day: float = 0.0
    water_rate_m3_per_day: float = 0.0
    cumulative_oil_m3: float = 0.0
    cumulative_water_m3: float = 0.0
    cumulative_steam_m3_cwe: float = 0.0
    cumulative_injected_heat_j: float = 0.0
    cumulative_conduction_loss_j: float = 0.0
    cumulative_produced_heat_j: float = 0.0
    cycle_oil_m3: float = 0.0
    cycle_steam_m3_cwe: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def steam_oil_ratio(self) -> float:
        """Cycle steam-to-oil ratio in cold water equivalent m3 per m3 of oil."""
        if self.cycle_oil_m3 <= 0.0:
            return float("inf")
        return self.cycle_steam_m3_cwe / self.cycle_oil_m3

    @property
    def cumulative_steam_oil_ratio(self) -> float:
        """Life-to-date steam-to-oil ratio."""
        if self.cumulative_oil_m3 <= 0.0:
            return float("inf")
        return self.cumulative_steam_m3_cwe / self.cumulative_oil_m3


@dataclass(frozen=True)
class InjectionPlan:
    """The controllable part of one CSS cycle."""

    steam_volume_m3_cwe: float
    injection_rate_m3_per_day_cwe: float
    injection_pressure_kpa: float
    steam_quality_frac: float
    soak_days: float
    cutoff_marginal_energy_ratio: float

    @classmethod
    def from_config(cls, config: FieldConfig) -> InjectionPlan:
        """Default plan taken from ``config/field.yaml``."""
        css = config.css
        return cls(
            steam_volume_m3_cwe=css.steam_volume_m3_cwe,
            injection_rate_m3_per_day_cwe=css.injection_rate_m3_per_day_cwe,
            injection_pressure_kpa=css.injection_pressure_kpa,
            steam_quality_frac=css.steam_quality_frac,
            soak_days=css.soak_days,
            cutoff_marginal_energy_ratio=css.cutoff_marginal_energy_ratio,
        )

    @property
    def injection_days(self) -> float:
        """Days of injection implied by the volume and the rate."""
        return self.steam_volume_m3_cwe / max(self.injection_rate_m3_per_day_cwe, 1.0e-9)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class ReservoirModel:
    """Daily-step thermal and inflow model of the near-well region.

    The model is deliberately reduced order. It resolves one heated zone, not a
    temperature field, and it uses analytic conduction solutions rather than a
    grid. The higher-fidelity axisymmetric finite-volume model that generates
    the synthetic data lives in ``app.simulate.truth`` and is never imported
    here.
    """

    def __init__(
        self,
        config: FieldConfig,
        fluid: FluidModel | None = None,
        parameters: ReservoirParameters | None = None,
    ) -> None:
        self.config = config
        self.fluid = fluid or FluidModel(config.fluid)
        self.parameters = parameters or ReservoirParameters.from_config(config)
        self.state = ReservoirState()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Return the model to virgin reservoir conditions."""
        reservoir = self.config.reservoir
        self.state = ReservoirState(
            reservoir_pressure_kpa=reservoir.initial_pressure_kpa,
            oil_saturation_frac=reservoir.oil_saturation_initial_frac,
            water_cut_frac=self.parameters.water_cut_initial_frac,
            average_heated_temp_c=reservoir.initial_temp_c,
            heated_radius_m=reservoir.wellbore_radius_m,
            heated_area_m2=math.pi * reservoir.wellbore_radius_m**2,
            steam_zone_temp_c=reservoir.initial_temp_c,
        )

    @property
    def initial_temp_c(self) -> float:
        """Virgin reservoir temperature."""
        return self.config.reservoir.initial_temp_c

    @property
    def permeability_m2(self) -> float:
        """Absolute permeability in SI units."""
        return self.parameters.permeability_md * MILLIDARCY_TO_M2

    @property
    def thermal_diffusivity_m2_per_s(self) -> float:
        """Thermal diffusivity of the reservoir rock.

        Equation: alpha = K / M, conductivity over volumetric heat capacity.
        The calibratable ``thermal_loss_multiplier`` scales it so assimilation
        can correct for unmodelled loss paths such as gravity override.
        """
        reservoir = self.config.reservoir
        base = (
            reservoir.overburden_thermal_conductivity_w_per_m_k
            / reservoir.overburden_volumetric_heat_capacity_j_per_m3_k
        )
        return base * max(self.parameters.thermal_loss_multiplier, 1.0e-3)

    @property
    def pore_volume_m3(self) -> float:
        """Drainage pore volume used for the material balance pressure decline."""
        reservoir = self.config.reservoir
        area = math.pi * (reservoir.drainage_radius_m**2 - reservoir.wellbore_radius_m**2)
        return area * self.parameters.net_pay_thickness_m * reservoir.porosity_frac

    def fracture_pressure_kpa(self) -> float:
        """Formation fracture pressure at the reservoir depth."""
        return self.config.reservoir.fracture_pressure_kpa

    # -------------------------------------------------------------- injection
    def sandface_steam_temperature_c(self, sandface_pressure_kpa: float) -> float:
        """Saturation temperature of the steam arriving at the sandface."""
        clamped = min(max(sandface_pressure_kpa, 120.0), 19500.0)
        return saturation_temperature_c(clamped)

    def step_injection(
        self,
        plan: InjectionPlan,
        days: float,
        sandface_heat_rate_w: float,
        sandface_steam_temp_c: float,
    ) -> None:
        """Advance the injection phase by ``days`` at a constant heat rate.

        The heat rate is supplied by the wellbore model, which has already
        removed the heat lost through the vacuum insulated tubing. Marx and
        Langenheim is applied to the cumulative injected heat so the heated area
        is path independent for a constant rate.
        """
        if days <= 0.0:
            return
        reservoir = self.config.reservoir
        state = self.state
        temperature_rise_k = max(sandface_steam_temp_c - reservoir.initial_temp_c, 1.0)

        state.phase = CyclePhase.INJECTION
        state.steam_zone_temp_c = sandface_steam_temp_c
        elapsed_before = state.phase_day
        elapsed_after = elapsed_before + days

        area_after = marx_langenheim_area_m2(
            net_heat_rate_w=sandface_heat_rate_w,
            elapsed_days=elapsed_after,
            net_pay_thickness_m=self.parameters.net_pay_thickness_m,
            temperature_rise_k=temperature_rise_k,
            rock_volumetric_heat_capacity_j_per_m3_k=(
                reservoir.rock_volumetric_heat_capacity_j_per_m3_k
            ),
            overburden_thermal_conductivity_w_per_m_k=(
                reservoir.overburden_thermal_conductivity_w_per_m_k
                * max(self.parameters.thermal_loss_multiplier, 1.0e-3)
            ),
            overburden_volumetric_heat_capacity_j_per_m3_k=(
                reservoir.overburden_volumetric_heat_capacity_j_per_m3_k
            ),
        )
        if reservoir.fracture_enhancement_enabled:
            area_after *= reservoir.fracture_area_multiplier

        heat_in_j = sandface_heat_rate_w * days * SECONDS_PER_DAY
        stored_j = (
            reservoir.rock_volumetric_heat_capacity_j_per_m3_k
            * area_after
            * self.parameters.net_pay_thickness_m
            * temperature_rise_k
        )
        state.cumulative_injected_heat_j += heat_in_j
        conduction_loss_j = max(
            state.cumulative_injected_heat_j - stored_j - state.cumulative_produced_heat_j, 0.0
        )
        state.cumulative_conduction_loss_j = conduction_loss_j

        state.heated_area_m2 = max(area_after, math.pi * reservoir.wellbore_radius_m**2)
        state.heated_radius_m = math.sqrt(state.heated_area_m2 / math.pi)
        state.heated_zone_energy_j = stored_j
        state.average_heated_temp_c = reservoir.initial_temp_c + temperature_rise_k

        injected_volume = plan.injection_rate_m3_per_day_cwe * days
        state.cumulative_steam_m3_cwe += injected_volume
        state.cycle_steam_m3_cwe += injected_volume

        # Injection repressurises the drainage volume.
        pressure_rise_kpa = injected_volume / (
            reservoir.total_compressibility_per_kpa * max(self.pore_volume_m3, 1.0)
        )
        state.reservoir_pressure_kpa = min(
            state.reservoir_pressure_kpa + pressure_rise_kpa, self.fracture_pressure_kpa()
        )

        state.phase_day = elapsed_after
        state.day += days
        state.conduction_clock_days = max(0.5 * elapsed_after, MIN_SOAK_CLOCK_DAYS)

    # ------------------------------------------------------------ soak, cool
    def _apply_conduction_loss(self, days: float) -> float:
        """Remove conduction heat loss for ``days`` and return the loss in joules."""
        state = self.state
        decay_per_s = conduction_decay_rate_per_s(
            elapsed_days=state.conduction_clock_days,
            net_pay_thickness_m=self.parameters.net_pay_thickness_m,
            heated_radius_m=max(state.heated_radius_m, self.config.reservoir.wellbore_radius_m),
            thermal_diffusivity_m2_per_s=self.thermal_diffusivity_m2_per_s,
        )
        retained = math.exp(-decay_per_s * days * SECONDS_PER_DAY)
        loss_j = state.heated_zone_energy_j * (1.0 - retained)
        state.heated_zone_energy_j -= loss_j
        state.cumulative_conduction_loss_j += loss_j
        state.conduction_clock_days += days
        return loss_j

    def _refresh_temperature(self) -> None:
        """Recompute the average heated-zone temperature from the stored energy."""
        state = self.state
        reservoir = self.config.reservoir
        volume_m3 = max(state.heated_area_m2 * self.parameters.net_pay_thickness_m, 1.0e-6)
        heat_capacity = reservoir.rock_volumetric_heat_capacity_j_per_m3_k * volume_m3
        excess_k = state.heated_zone_energy_j / heat_capacity
        state.average_heated_temp_c = reservoir.initial_temp_c + max(excess_k, 0.0)
        if state.steam_zone_temp_c > 0.0:
            state.average_heated_temp_c = min(state.average_heated_temp_c, state.steam_zone_temp_c)

    def step_soak(self, days: float) -> None:
        """Advance the soak phase, losing heat by conduction only."""
        if days <= 0.0:
            return
        self.state.phase = CyclePhase.SOAK
        self._apply_conduction_loss(days)
        self._refresh_temperature()
        self.state.day += days
        self.state.phase_day += days

    # ------------------------------------------------------------- production
    def sandface_viscosity_pa_s(self) -> float:
        """Oil viscosity at the sandface, at the average heated-zone temperature.

        In-situ flow is two-phase oil and water moving under their own relative
        permeabilities, not a dispersion, so the emulsion uplift is deliberately
        not applied here. The uplift is applied in the wellbore and the pump,
        where shear actually creates the water-in-oil emulsion. This split is
        recorded in ``docs/ASSUMPTIONS.md``.
        """
        return float(self.fluid.dead_oil_viscosity_pa_s(self.state.average_heated_temp_c))

    def cold_viscosity_pa_s(self) -> float:
        """Oil viscosity in the unheated part of the drainage area."""
        return float(self.fluid.dead_oil_viscosity_pa_s(self.initial_temp_c))

    def hydraulic_diffusivity_m2_per_s(self) -> float:
        """Pressure diffusivity of the cold outer region.

        Equation: eta = k / (phi mu c_t).
        Units: m2/s. Permeability in m2, viscosity in Pa.s, compressibility in
        1/Pa. It sets how fast the pressure transient reaches out after the well
        is put back on production.
        Source: Dake, Fundamentals of Reservoir Engineering, chapter 5.
        """
        reservoir = self.config.reservoir
        compressibility_per_pa = reservoir.total_compressibility_per_kpa / 1000.0
        return self.permeability_m2 / (
            reservoir.porosity_frac * self.cold_viscosity_pa_s() * compressibility_per_pa
        )

    def transient_drainage_radius_m(self, production_days: float) -> float:
        """Radius of investigation of the pressure transient since the well came on.

        Equation: r_inv = sqrt(4 eta t), bounded below by the heated radius and
        above by the drainage radius.
        Units: m. Time in days.
        Rationale: a CSS well is reopened after every soak, so the flow is
        transient for much of the cycle. Holding the outer radius at the full
        drainage radius from day one would overstate the cold-oil resistance and
        understate the early-cycle rate. This term is what produces the
        characteristic CSS shape of a peak right after the soak followed by a
        decline as the transient reaches further into cold oil.
        Source: Dake chapter 5; Lee, Well Testing, SPE Textbook 1, section 1.3.
        """
        reservoir = self.config.reservoir
        floor_m = max(self.state.heated_radius_m, reservoir.wellbore_radius_m * 2.0)
        if production_days <= 0.0:
            return floor_m
        radius_m = math.sqrt(
            4.0 * self.hydraulic_diffusivity_m2_per_s() * production_days * SECONDS_PER_DAY
        )
        return float(min(max(radius_m, floor_m), reservoir.drainage_radius_m))

    def deliverability_m3_per_day(
        self, bottomhole_pressure_kpa: float, production_days: float | None = None
    ) -> float:
        """Total liquid the reservoir can deliver at the given flowing pressure.

        Combines the composite radial productivity ratio with a single-region
        Darcy rate evaluated at the cold viscosity, so the ratio carries the
        whole effect of heating. The outer radius is the transient radius of
        investigation rather than the full drainage radius. Relative
        permeability, the cycle productivity decline and the optional fracture
        switch are applied as multipliers.
        """
        reservoir = self.config.reservoir
        state = self.state
        drawdown_kpa = state.reservoir_pressure_kpa - bottomhole_pressure_kpa
        if drawdown_kpa <= 0.0:
            return 0.0
        elapsed_days = state.phase_day if production_days is None else production_days
        outer_radius_m = self.transient_drainage_radius_m(elapsed_days)
        cold_viscosity = self.cold_viscosity_pa_s()
        hot_viscosity = self.sandface_viscosity_pa_s()
        cold_rate_m3_per_s = darcy_liquid_rate_m3_per_s(
            permeability_m2=self.permeability_m2,
            net_pay_thickness_m=self.parameters.net_pay_thickness_m,
            drawdown_pa=drawdown_kpa * 1000.0,
            viscosity_pa_s=cold_viscosity,
            drainage_radius_m=outer_radius_m,
            wellbore_radius_m=reservoir.wellbore_radius_m,
            skin_dimensionless=self.parameters.skin_dimensionless,
        )
        ratio = composite_radial_productivity_ratio(
            heated_radius_m=state.heated_radius_m,
            wellbore_radius_m=reservoir.wellbore_radius_m,
            drainage_radius_m=outer_radius_m,
            hot_viscosity_pa_s=hot_viscosity,
            cold_viscosity_pa_s=cold_viscosity,
            skin_dimensionless=self.parameters.skin_dimensionless,
        )
        relative_permeability = corey_oil_relative_permeability(
            oil_saturation_frac=state.oil_saturation_frac,
            residual_oil_saturation_frac=reservoir.residual_oil_saturation_frac,
            initial_oil_saturation_frac=reservoir.oil_saturation_initial_frac,
        )
        cycle_multiplier = (1.0 - self.parameters.productivity_decline_per_cycle_frac) ** (
            state.cycle_number - 1
        )
        rate_m3_per_day = (
            cold_rate_m3_per_s * SECONDS_PER_DAY * ratio * relative_permeability * cycle_multiplier
        )
        if reservoir.fracture_enhancement_enabled:
            rate_m3_per_day *= reservoir.fracture_area_multiplier
        return require_finite(max(rate_m3_per_day, 0.0), "deliverability_m3_per_day")

    def step_production(
        self,
        days: float,
        bottomhole_pressure_kpa: float,
        pump_capacity_m3_per_day: float | None = None,
    ) -> dict[str, float]:
        """Advance the production phase by ``days``.

        Args:
            days: Length of the step in days.
            bottomhole_pressure_kpa: Flowing pressure the pump holds at the sandface.
            pump_capacity_m3_per_day: Upper bound set by the pump. ``None`` means
                the reservoir is unconstrained, which is only used for testing.

        Returns:
            A dictionary of the rates and energies over the step.
        """
        if days <= 0.0:
            return {}
        reservoir = self.config.reservoir
        state = self.state
        state.phase = CyclePhase.PRODUCTION

        deliverability = self.deliverability_m3_per_day(bottomhole_pressure_kpa)
        total_rate = deliverability
        limited_by_pump = False
        if pump_capacity_m3_per_day is not None:
            total_rate = min(total_rate, max(pump_capacity_m3_per_day, 0.0))
            limited_by_pump = total_rate < deliverability - 1.0e-9

        oil_rate = total_rate * (1.0 - state.water_cut_frac)
        water_rate = total_rate * state.water_cut_frac

        # Enthalpy removed by the produced fluid, relative to the virgin reservoir.
        excess_temp_k = max(state.average_heated_temp_c - reservoir.initial_temp_c, 0.0)
        density = float(
            self.fluid.mixture_density_kg_per_m3(state.average_heated_temp_c, state.water_cut_frac)
        )
        specific_heat = float(self.fluid.mixture_specific_heat_j_per_kg_k(state.water_cut_frac))
        produced_heat_j = total_rate * days * density * specific_heat * excess_temp_k

        conduction_loss_j = self._apply_conduction_loss(days)
        state.heated_zone_energy_j = max(state.heated_zone_energy_j - produced_heat_j, 0.0)
        state.cumulative_produced_heat_j += produced_heat_j
        self._refresh_temperature()

        # Material balance pressure decline.
        withdrawn_m3 = total_rate * days
        pressure_drop_kpa = withdrawn_m3 / (
            reservoir.total_compressibility_per_kpa * max(self.pore_volume_m3, 1.0)
        )
        state.reservoir_pressure_kpa = max(
            state.reservoir_pressure_kpa - pressure_drop_kpa,
            reservoir.abandonment_pressure_kpa,
        )

        # Saturation and water cut evolution inside the drainage volume.
        oil_volume_m3 = state.oil_saturation_frac * self.pore_volume_m3
        oil_volume_m3 = max(oil_volume_m3 - oil_rate * days, 0.0)
        state.oil_saturation_frac = max(
            oil_volume_m3 / max(self.pore_volume_m3, 1.0e-6),
            reservoir.residual_oil_saturation_frac,
        )
        state.water_cut_frac = min(
            state.water_cut_frac
            + self.parameters.water_cut_growth_per_cycle_frac
            * days
            / max(self.config.css.production_days_max, 1.0),
            reservoir.water_cut_max_frac,
        )

        state.oil_rate_m3_per_day = oil_rate
        state.water_rate_m3_per_day = water_rate
        state.cumulative_oil_m3 += oil_rate * days
        state.cumulative_water_m3 += water_rate * days
        state.cycle_oil_m3 += oil_rate * days
        state.day += days
        state.phase_day += days

        return {
            "deliverability_m3_per_day": deliverability,
            "total_liquid_rate_m3_per_day": total_rate,
            "oil_rate_m3_per_day": oil_rate,
            "water_rate_m3_per_day": water_rate,
            "produced_heat_j": produced_heat_j,
            "conduction_loss_j": conduction_loss_j,
            "limited_by_pump": float(limited_by_pump),
        }

    # ------------------------------------------------------------ cycle logic
    def begin_cycle(self, cycle_number: int) -> None:
        """Start a new CSS cycle, carrying over the retained heat."""
        state = self.state
        state.cycle_number = cycle_number
        state.phase = CyclePhase.INJECTION
        state.phase_day = 0.0
        state.cycle_oil_m3 = 0.0
        state.cycle_steam_m3_cwe = 0.0
        state.heated_zone_energy_j *= self.parameters.cycle_energy_retention_frac
        state.water_cut_frac = min(
            self.parameters.water_cut_initial_frac
            + self.parameters.water_cut_growth_per_cycle_frac * (cycle_number - 1),
            self.config.reservoir.water_cut_max_frac,
        )
        state.conduction_clock_days = MIN_SOAK_CLOCK_DAYS

    def snapshot(self) -> dict[str, float]:
        """Flat dictionary of the current state, for logging and for the API."""
        state = self.state
        return {
            "day": state.day,
            "cycle_number": float(state.cycle_number),
            "phase_day": state.phase_day,
            "heated_radius_m": state.heated_radius_m,
            "heated_area_m2": state.heated_area_m2,
            "average_heated_temp_c": state.average_heated_temp_c,
            "steam_zone_temp_c": state.steam_zone_temp_c,
            "sandface_viscosity_cp": self.sandface_viscosity_pa_s() * 1000.0,
            "reservoir_pressure_kpa": state.reservoir_pressure_kpa,
            "oil_rate_m3_per_day": state.oil_rate_m3_per_day,
            "water_rate_m3_per_day": state.water_rate_m3_per_day,
            "water_cut_frac": state.water_cut_frac,
            "oil_saturation_frac": state.oil_saturation_frac,
            "cumulative_oil_m3": state.cumulative_oil_m3,
            "cumulative_steam_m3_cwe": state.cumulative_steam_m3_cwe,
            "cycle_oil_m3": state.cycle_oil_m3,
            "cycle_steam_m3_cwe": state.cycle_steam_m3_cwe,
            "steam_oil_ratio": state.steam_oil_ratio,
            "heated_zone_energy_j": state.heated_zone_energy_j,
        }


def sandface_heat_rate_w(
    plan: InjectionPlan, config: FieldConfig, wellbore_heat_loss_fraction: float = 0.0
) -> float:
    """Net heat rate reaching the sandface for an injection plan.

    The wellbore loss fraction comes from :mod:`app.twin.wellbore`. Keeping the
    call here lets the reservoir model be tested on its own with a zero loss.
    """
    surface_rate_w = injected_heat_rate_w(
        rate_m3_per_day_cwe=plan.injection_rate_m3_per_day_cwe,
        pressure_kpa=plan.injection_pressure_kpa,
        quality_frac=plan.steam_quality_frac,
        reference_temp_c=config.reservoir.initial_temp_c,
        water_density_kg_per_m3=config.fluid.water_density_kg_per_m3,
        water_specific_heat_j_per_kg_k=config.fluid.water_specific_heat_j_per_kg_k,
    )
    return surface_rate_w * max(1.0 - wellbore_heat_loss_fraction, 0.0)
