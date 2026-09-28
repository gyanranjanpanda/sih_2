"""Wellbore heat transmission and pressure for injection and production.

Injection is treated as saturated steam: the temperature is pinned to the
saturation temperature of the local pressure and the heat lost through the
tubing wall is taken out of the latent heat, so the steam arrives at the
sandface with a lower quality. Production is treated as single-phase liquid:
the Ramey (1962) relaxation-distance solution gives the temperature profile up
the string, which sets the viscosity at pump depth and along the rods.

The Baghewala wells are completed with vacuum insulated tubing, so the overall
heat transfer coefficient is one to two orders of magnitude smaller than for
bare tubing. The model carries both and the tests check that VIT delivers
hotter fluid than bare tubing under otherwise identical conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import FieldConfig, WellboreConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import require_finite, require_positive
from app.core.steam import latent_heat_j_per_kg, saturation_temperature_c
from app.core.units import SECONDS_PER_DAY, STANDARD_GRAVITY_M_PER_S2
from app.twin.fluid import FluidModel

RAMEY_LATE_TIME_CONSTANT = 0.290
"""Constant in the Ramey late-time approximation to the earth time function."""


def ramey_time_function(
    elapsed_days: float, tubing_outer_radius_m: float, thermal_diffusivity_m2_per_s: float
) -> float:
    """Dimensionless earth transient time function f(t) in the Ramey solution.

    Equation: with t_D = alpha t / r_to^2,
        f(t) = ln(2 sqrt(t_D)) - 0.290 for t_D greater than 1.5,
        f(t) = 1.1281 sqrt(t_D) (1 - 0.3 sqrt(t_D)) otherwise.
    Units: dimensionless. Time in days, radius in m, diffusivity in m2/s.
    Assumptions: radial conduction into an infinite formation at the undisturbed
    geothermal temperature, constant wellbore heat flux.
    Source: Ramey, H. J. (1962), Wellbore heat transmission, J. Pet. Tech.
    14(4), 427-435; short-time form from Hasan and Kabir (1991).
    """
    require_positive(tubing_outer_radius_m, "tubing_outer_radius_m")
    require_positive(thermal_diffusivity_m2_per_s, "thermal_diffusivity_m2_per_s")
    elapsed_s = max(elapsed_days, 1.0e-4) * SECONDS_PER_DAY
    dimensionless_time = thermal_diffusivity_m2_per_s * elapsed_s / tubing_outer_radius_m**2
    if dimensionless_time > 1.5:
        return math.log(2.0 * math.sqrt(dimensionless_time)) - RAMEY_LATE_TIME_CONSTANT
    root = math.sqrt(dimensionless_time)
    return max(1.1281 * root * (1.0 - 0.3 * root), 1.0e-6)


def overall_resistance_coefficient_w_per_m_k(
    overall_heat_transfer_w_per_m2_k: float,
    tubing_outer_radius_m: float,
    formation_conductivity_w_per_m_k: float,
    time_function: float,
) -> float:
    """Heat transfer per unit length per kelvin, tubing and formation in series.

    Equation: U_A = 2 pi r_to U_to k_e / (k_e + r_to U_to f(t)).
    Units: W/m/K. Radius in m, U_to in W/m2/K, k_e in W/m/K.
    Assumptions: steady resistance through the completion, transient conduction
    in the formation captured by f(t), no natural convection in the annulus
    beyond what U_to already represents.
    Source: Ramey (1962); Willhite (1967) for the overall coefficient.
    """
    require_positive(overall_heat_transfer_w_per_m2_k, "overall_heat_transfer_w_per_m2_k")
    require_positive(formation_conductivity_w_per_m_k, "formation_conductivity_w_per_m_k")
    denominator = formation_conductivity_w_per_m_k + (
        tubing_outer_radius_m * overall_heat_transfer_w_per_m2_k * time_function
    )
    return (
        2.0
        * math.pi
        * tubing_outer_radius_m
        * overall_heat_transfer_w_per_m2_k
        * formation_conductivity_w_per_m_k
        / denominator
    )


def geothermal_temperature_c(
    depth_m: float | NDArray[np.float64], surface_temp_c: float, gradient_c_per_m: float
) -> float | NDArray[np.float64]:
    """Undisturbed formation temperature at depth.

    Equation: T_ei(z) = T_surface + g z.
    Units: degrees Celsius, depth in m, gradient in C/m.
    """
    return surface_temp_c + gradient_c_per_m * np.asarray(depth_m, dtype=float)


def ramey_downward_temperature_c(
    depth_m: float | NDArray[np.float64],
    inlet_temp_c: float,
    relaxation_distance_m: float,
    surface_temp_c: float,
    gradient_c_per_m: float,
) -> float | NDArray[np.float64]:
    """Fluid temperature flowing down the tubing.

    Equation: T_f(z) = T_s - g A + g z + (T_in - T_s + g A) exp(-z / A),
    with the relaxation distance A = w c_p / U_A.
    Units: degrees Celsius, depth in m, A in m.
    Limiting cases: A to infinity gives T_f = T_in at every depth, which is the
    zero heat loss case; A to zero gives T_f equal to the geothermal profile.
    Both are covered by tests.
    Source: Ramey (1962), equation 7.
    """
    require_positive(relaxation_distance_m, "relaxation_distance_m")
    z = np.asarray(depth_m, dtype=float)
    exponent = np.clip(-z / relaxation_distance_m, -700.0, 0.0)
    value = (
        surface_temp_c
        - gradient_c_per_m * relaxation_distance_m
        + gradient_c_per_m * z
        + (inlet_temp_c - surface_temp_c + gradient_c_per_m * relaxation_distance_m)
        * np.exp(exponent)
    )
    return float(value) if np.isscalar(depth_m) else np.asarray(value, dtype=float)


def ramey_upward_temperature_c(
    depth_m: float | NDArray[np.float64],
    intake_temp_c: float,
    intake_depth_m: float,
    relaxation_distance_m: float,
    surface_temp_c: float,
    gradient_c_per_m: float,
) -> float | NDArray[np.float64]:
    """Fluid temperature flowing up the tubing from the pump intake.

    Equation: T_f(z) = T_s + g A + g z
                       + (T_intake - T_s - g A - g L) exp((z - L) / A),
    with L the intake depth and A the relaxation distance.
    Units: degrees Celsius, depths in m, A in m.
    Limiting cases: A to infinity gives T_f = T_intake everywhere, A to zero
    gives the geothermal profile. Both are covered by tests.
    Source: Ramey (1962), production form.
    """
    require_positive(relaxation_distance_m, "relaxation_distance_m")
    z = np.asarray(depth_m, dtype=float)
    exponent = np.clip((z - intake_depth_m) / relaxation_distance_m, -700.0, 0.0)
    offset = (
        intake_temp_c
        - surface_temp_c
        - gradient_c_per_m * relaxation_distance_m
        - gradient_c_per_m * intake_depth_m
    )
    value = (
        surface_temp_c
        + gradient_c_per_m * relaxation_distance_m
        + gradient_c_per_m * z
        + offset * np.exp(exponent)
    )
    return float(value) if np.isscalar(depth_m) else np.asarray(value, dtype=float)


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of steam injection down the tubing for one set of conditions."""

    depth_m: NDArray[np.float64]
    temperature_c: NDArray[np.float64]
    quality_frac: NDArray[np.float64]
    pressure_kpa: NDArray[np.float64]
    heat_loss_w: float
    surface_heat_rate_w: float
    sandface_heat_rate_w: float
    sandface_temp_c: float
    sandface_quality_frac: float

    @property
    def heat_loss_fraction(self) -> float:
        """Fraction of the surface heat rate lost to the formation."""
        if self.surface_heat_rate_w <= 0.0:
            return 0.0
        return float(min(max(self.heat_loss_w / self.surface_heat_rate_w, 0.0), 1.0))


@dataclass(frozen=True)
class ProductionResult:
    """Outcome of lifting produced fluid up the tubing."""

    depth_m: NDArray[np.float64]
    temperature_c: NDArray[np.float64]
    viscosity_pa_s: NDArray[np.float64]
    wellhead_temp_c: float
    pump_intake_temp_c: float
    relaxation_distance_m: float

    def viscosity_at_depth_pa_s(self, depth_m: float) -> float:
        """Interpolated apparent viscosity at one depth along the rod string."""
        return float(np.interp(depth_m, self.depth_m, self.viscosity_pa_s))

    def temperature_at_depth_c(self, depth_m: float) -> float:
        """Interpolated fluid temperature at one depth along the rod string."""
        return float(np.interp(depth_m, self.depth_m, self.temperature_c))


class WellboreModel:
    """Heat transmission and pressure along the tubing for one well."""

    def __init__(
        self,
        config: FieldConfig,
        fluid: FluidModel | None = None,
        heat_transfer_override_w_per_m2_k: float | None = None,
        node_count: int = 121,
    ) -> None:
        self.config = config
        self.wellbore: WellboreConfig = config.wellbore
        self.fluid = fluid or FluidModel(config.fluid)
        self.node_count = max(node_count, 11)
        self._heat_transfer_override = heat_transfer_override_w_per_m2_k

    @property
    def overall_heat_transfer_w_per_m2_k(self) -> float:
        """Active overall heat transfer coefficient, including any calibration override."""
        if self._heat_transfer_override is not None:
            return self._heat_transfer_override
        return self.wellbore.overall_heat_transfer_w_per_m2_k

    @property
    def tubing_outer_radius_m(self) -> float:
        """Outer radius of the tubing, the reference radius for the heat balance."""
        return 0.5 * self.wellbore.tubing_outer_diameter_m

    def with_heat_transfer(self, coefficient_w_per_m2_k: float) -> WellboreModel:
        """Return a copy of the model with a different overall heat transfer coefficient."""
        return WellboreModel(
            self.config,
            fluid=self.fluid,
            heat_transfer_override_w_per_m2_k=coefficient_w_per_m2_k,
            node_count=self.node_count,
        )

    # -------------------------------------------------------------- injection
    def inject_steam(
        self,
        wellhead_pressure_kpa: float,
        wellhead_quality_frac: float,
        rate_m3_per_day_cwe: float,
        elapsed_days: float,
        depth_m: float | None = None,
    ) -> InjectionResult:
        """March saturated steam down the tubing, losing latent heat to the formation.

        Equation: per unit length, dQ/dz = U_A (T_sat(p) - T_ei(z)), and the
        quality falls as dx/dz = -(dQ/dz) / (w h_fg).
        Units: rate in m3/day cold water equivalent, pressures in kPa absolute.
        Assumptions: saturated two-phase flow with no slip, pressure along the
        string from the mixture static head only (friction in the large tubing
        bore is small compared with the head for these rates), and a switch to
        single-phase hot water if the quality reaches zero.
        Source: Ramey (1962) for the earth coupling; standard steam quality
        balance as in Prats, Thermal Recovery, chapter 4.
        """
        require_positive(rate_m3_per_day_cwe, "rate_m3_per_day_cwe")
        total_depth_m = depth_m if depth_m is not None else self.config.reservoir.depth_m
        require_positive(total_depth_m, "depth_m")
        if not 0.0 <= wellhead_quality_frac <= 1.0:
            raise PhysicsDomainError(
                "Steam quality must lie in [0, 1].", quality_frac=wellhead_quality_frac
            )

        depths = np.linspace(0.0, total_depth_m, self.node_count)
        step_m = float(depths[1] - depths[0])
        mass_rate_kg_per_s = (
            rate_m3_per_day_cwe * self.config.fluid.water_density_kg_per_m3 / SECONDS_PER_DAY
        )
        time_function = ramey_time_function(
            elapsed_days,
            self.tubing_outer_radius_m,
            self.wellbore.formation_thermal_diffusivity_m2_per_s,
        )
        conductance_w_per_m_k = overall_resistance_coefficient_w_per_m_k(
            self.overall_heat_transfer_w_per_m2_k,
            self.tubing_outer_radius_m,
            self.wellbore.formation_thermal_conductivity_w_per_m_k,
            time_function,
        )

        temperatures = np.zeros_like(depths)
        qualities = np.zeros_like(depths)
        pressures = np.zeros_like(depths)

        pressure_kpa = wellhead_pressure_kpa
        quality = wellhead_quality_frac
        temperature_c = saturation_temperature_c(pressure_kpa)
        total_loss_w = 0.0
        specific_heat = self.config.fluid.water_specific_heat_j_per_kg_k

        for index, depth in enumerate(depths):
            geothermal_c = float(
                geothermal_temperature_c(
                    depth, self.wellbore.surface_temp_c, self.wellbore.geothermal_gradient_c_per_m
                )
            )
            temperatures[index] = temperature_c
            qualities[index] = quality
            pressures[index] = pressure_kpa
            if index == len(depths) - 1:
                break

            loss_w_per_m = conductance_w_per_m_k * max(temperature_c - geothermal_c, 0.0)
            segment_loss_w = loss_w_per_m * step_m
            total_loss_w += segment_loss_w

            if quality > 1.0e-6:
                latent_j_per_kg = latent_heat_j_per_kg(min(max(pressure_kpa, 120.0), 19500.0))
                quality = max(
                    quality - segment_loss_w / (mass_rate_kg_per_s * latent_j_per_kg), 0.0
                )
            else:
                temperature_c = max(
                    temperature_c - segment_loss_w / (mass_rate_kg_per_s * specific_heat),
                    geothermal_c,
                )

            # Mixture static head. Steam is light, condensate is not.
            mixture_density = self._two_phase_density_kg_per_m3(pressure_kpa, quality)
            pressure_kpa += mixture_density * STANDARD_GRAVITY_M_PER_S2 * step_m / 1000.0
            pressure_kpa = min(pressure_kpa, 19500.0)
            if quality > 1.0e-6:
                temperature_c = saturation_temperature_c(max(pressure_kpa, 120.0))

        surface_heat_rate_w = self._steam_heat_rate_w(
            mass_rate_kg_per_s, wellhead_pressure_kpa, wellhead_quality_frac
        )
        sandface_heat_rate_w = self._steam_heat_rate_w(
            mass_rate_kg_per_s, float(pressures[-1]), float(qualities[-1]), float(temperatures[-1])
        )
        require_finite(temperatures, "injection temperature profile")
        return InjectionResult(
            depth_m=depths,
            temperature_c=temperatures,
            quality_frac=qualities,
            pressure_kpa=pressures,
            heat_loss_w=float(total_loss_w),
            surface_heat_rate_w=float(surface_heat_rate_w),
            sandface_heat_rate_w=float(max(sandface_heat_rate_w, 0.0)),
            sandface_temp_c=float(temperatures[-1]),
            sandface_quality_frac=float(qualities[-1]),
        )

    def _two_phase_density_kg_per_m3(self, pressure_kpa: float, quality_frac: float) -> float:
        """No-slip mixture density of wet steam, from a simple steam density fit.

        Assumption: the vapour behaves as an ideal gas with the water molar mass
        at the saturation temperature, which is accurate enough for a static
        head term that contributes a few percent of the injection pressure.
        """
        temperature_k = saturation_temperature_c(max(pressure_kpa, 120.0)) + 273.15
        vapour_density = (pressure_kpa * 1000.0) * 0.018015 / (8.314 * temperature_k)
        liquid_density = self.config.fluid.water_density_kg_per_m3 * 0.85
        specific_volume = (
            quality_frac / max(vapour_density, 1.0e-3) + (1.0 - quality_frac) / liquid_density
        )
        return 1.0 / max(specific_volume, 1.0e-6)

    def _steam_heat_rate_w(
        self,
        mass_rate_kg_per_s: float,
        pressure_kpa: float,
        quality_frac: float,
        temperature_c: float | None = None,
    ) -> float:
        """Heat rate carried by the steam relative to the reservoir datum temperature."""
        from app.core.steam import liquid_enthalpy_j_per_kg

        clamped_kpa = min(max(pressure_kpa, 120.0), 19500.0)
        datum_j_per_kg = (
            self.config.fluid.water_specific_heat_j_per_kg_k * self.config.reservoir.initial_temp_c
        )
        if quality_frac > 1.0e-6 or temperature_c is None:
            enthalpy_j_per_kg = liquid_enthalpy_j_per_kg(
                clamped_kpa
            ) + quality_frac * latent_heat_j_per_kg(clamped_kpa)
        else:
            enthalpy_j_per_kg = self.config.fluid.water_specific_heat_j_per_kg_k * temperature_c
        return mass_rate_kg_per_s * max(enthalpy_j_per_kg - datum_j_per_kg, 0.0)

    # ------------------------------------------------------------- production
    def produce(
        self,
        sandface_temp_c: float,
        liquid_rate_m3_per_day: float,
        water_cut_frac: float,
        elapsed_days: float,
        pump_depth_m: float | None = None,
    ) -> ProductionResult:
        """Temperature and viscosity profile of the produced fluid up the tubing.

        The produced fluid enters at the pump intake at the sandface temperature
        and cools on the way up. The resulting profile is what the rod string
        sees, so it drives viscous drag and damping.

        Assumptions: single-phase liquid in the tubing, constant specific heat
        over the step, and the emulsion viscosity uplift applied here because
        the water-in-oil dispersion is created by shear in the pump and tubing
        rather than in the formation.
        Source: Ramey (1962) production form; Hasan and Kabir (2002) for the
        use of the relaxation distance in artificial-lift wells.
        """
        intake_depth_m = pump_depth_m if pump_depth_m is not None else self.config.srp.pump_depth_m
        require_positive(intake_depth_m, "pump_depth_m")
        depths = np.linspace(0.0, intake_depth_m, self.node_count)

        density_kg_per_m3 = float(
            self.fluid.mixture_density_kg_per_m3(sandface_temp_c, water_cut_frac)
        )
        specific_heat = float(self.fluid.mixture_specific_heat_j_per_kg_k(water_cut_frac))
        mass_rate_kg_per_s = max(
            liquid_rate_m3_per_day * density_kg_per_m3 / SECONDS_PER_DAY, 1.0e-6
        )
        time_function = ramey_time_function(
            elapsed_days,
            self.tubing_outer_radius_m,
            self.wellbore.formation_thermal_diffusivity_m2_per_s,
        )
        conductance_w_per_m_k = overall_resistance_coefficient_w_per_m_k(
            self.overall_heat_transfer_w_per_m2_k,
            self.tubing_outer_radius_m,
            self.wellbore.formation_thermal_conductivity_w_per_m_k,
            time_function,
        )
        relaxation_distance_m = max(
            mass_rate_kg_per_s * specific_heat / max(conductance_w_per_m_k, 1.0e-9), 1.0e-3
        )
        temperatures = np.asarray(
            ramey_upward_temperature_c(
                depths,
                intake_temp_c=sandface_temp_c,
                intake_depth_m=intake_depth_m,
                relaxation_distance_m=relaxation_distance_m,
                surface_temp_c=self.wellbore.surface_temp_c,
                gradient_c_per_m=self.wellbore.geothermal_gradient_c_per_m,
            ),
            dtype=float,
        )
        # The produced fluid cannot be hotter than it was at the intake, nor
        # colder than the surrounding formation.
        geothermal = np.asarray(
            geothermal_temperature_c(
                depths, self.wellbore.surface_temp_c, self.wellbore.geothermal_gradient_c_per_m
            ),
            dtype=float,
        )
        temperatures = np.clip(
            temperatures, np.minimum(geothermal, sandface_temp_c), sandface_temp_c
        )
        viscosities = np.asarray(
            self.fluid.viscosity_pa_s(temperatures, water_cut_frac), dtype=float
        )
        require_finite(temperatures, "production temperature profile")
        return ProductionResult(
            depth_m=depths,
            temperature_c=temperatures,
            viscosity_pa_s=viscosities,
            wellhead_temp_c=float(temperatures[0]),
            pump_intake_temp_c=float(temperatures[-1]),
            relaxation_distance_m=float(relaxation_distance_m),
        )

    # --------------------------------------------------------------- pressure
    def bottomhole_pressure_kpa(
        self,
        fluid_level_depth_m: float,
        temperature_c: float,
        water_cut_frac: float,
        casing_head_pressure_kpa: float | None = None,
    ) -> float:
        """Flowing bottomhole pressure from the annular fluid level.

        Equation: p_wf = p_casing + rho g (L_pump - L_level).
        Units: kPa. Depths in m, temperature in degrees Celsius.
        Assumption: a static annular liquid column above the pump intake, which
        is the standard acoustic fluid-level interpretation.
        Source: Podio and McCoy, acoustic well sounding practice.
        """
        casing_kpa = (
            casing_head_pressure_kpa
            if casing_head_pressure_kpa is not None
            else self.wellbore.wellhead_pressure_kpa
        )
        density = float(self.fluid.mixture_density_kg_per_m3(temperature_c, water_cut_frac))
        column_m = max(self.config.srp.pump_depth_m - fluid_level_depth_m, 0.0)
        return casing_kpa + density * STANDARD_GRAVITY_M_PER_S2 * column_m / 1000.0

    def tubing_friction_pressure_kpa(
        self, liquid_rate_m3_per_day: float, viscosity_pa_s: float, density_kg_per_m3: float
    ) -> float:
        """Frictional pressure drop for liquid flowing up the tubing.

        Equation: laminar Hagen-Poiseuille when Re is below 2300,
        dp = 32 mu v L / d^2; otherwise the Colebrook friction factor is
        approximated by the Swamee-Jain explicit form.
        Units: kPa. Rate in m3/day, viscosity in Pa.s, density in kg/m3.
        Note: for 18 degree API crude at these rates the flow is deeply laminar,
        so the laminar branch is the one that matters. The turbulent branch is
        kept for hot, high water cut cases.
        Source: White, Fluid Mechanics, chapter 6; Swamee and Jain (1976).
        """
        diameter_m = self.wellbore.tubing_inner_diameter_m
        area_m2 = math.pi * 0.25 * diameter_m**2
        velocity_m_per_s = liquid_rate_m3_per_day / SECONDS_PER_DAY / area_m2
        if velocity_m_per_s <= 0.0:
            return 0.0
        length_m = self.config.srp.pump_depth_m
        reynolds = density_kg_per_m3 * velocity_m_per_s * diameter_m / max(viscosity_pa_s, 1.0e-9)
        if reynolds < 2300.0:
            drop_pa = 32.0 * viscosity_pa_s * velocity_m_per_s * length_m / diameter_m**2
        else:
            relative_roughness = self.wellbore.tubing_roughness_m / diameter_m
            friction_factor = 0.25 / (
                math.log10(relative_roughness / 3.7 + 5.74 / reynolds**0.9) ** 2
            )
            drop_pa = (
                friction_factor
                * length_m
                / diameter_m
                * 0.5
                * density_kg_per_m3
                * velocity_m_per_s**2
            )
        return drop_pa / 1000.0

    def injection_pressure_is_safe(self, sandface_pressure_kpa: float, safety_frac: float) -> bool:
        """Whether a sandface injection pressure stays below the fracture limit."""
        return sandface_pressure_kpa <= safety_frac * self.config.reservoir.fracture_pressure_kpa
