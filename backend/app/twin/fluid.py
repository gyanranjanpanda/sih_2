"""Fluid properties for Baghewala heavy crude.

All correlations here are published petroleum engineering relations. Each
function states its equation, units, assumptions and source. The parameters
are fitted from the anchors in ``config/field.yaml`` and can be refitted from
uploaded measurements by the calibration workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from app.core.config import FluidConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import require_finite
from app.core.units import (
    ABSOLUTE_ZERO_C,
    api_to_specific_gravity,
    celsius_to_kelvin,
)

Number = float | NDArray[np.float64]

WALTHER_OFFSET = 0.7
"""Constant in ASTM D341: log10(log10(nu + 0.7)) = A - B log10(T)."""

MIN_KINEMATIC_VISCOSITY_CST = 0.3
"""Lower clamp. Below the viscosity of hot water, the Walther form loses meaning."""

MAX_KINEMATIC_VISCOSITY_CST = 5.0e7
"""Upper clamp, well above any measured crude, used to keep solvers finite."""


def water_viscosity_pa_s(temp_c: Number) -> Number:
    """Dynamic viscosity of liquid water against temperature.

    Equation: mu = 2.414e-5 * 10^(247.8 / (T_K - 140)), the Vogel form.
    Units: Pa.s, temperature in degrees Celsius.
    Assumptions: liquid water at moderate pressure, 0 to 370 degrees C.
    Source: Al-Shemmeri, Engineering Fluid Mechanics (2012), water property table fit.
    """
    temp_k = np.asarray(celsius_to_kelvin(temp_c), dtype=float)
    if np.any(temp_k <= 140.5):
        raise PhysicsDomainError(
            "Water viscosity correlation is not valid at or below about -132 degrees C.",
            temp_c=float(np.min(np.asarray(temp_c, dtype=float))),
        )
    value = 2.414e-5 * np.power(10.0, 247.8 / (temp_k - 140.0))
    return float(value) if np.isscalar(temp_c) else np.asarray(value, dtype=float)


@dataclass(frozen=True)
class WaltherFit:
    """Coefficients of the ASTM D341 viscosity-temperature line."""

    coefficient_a: float
    coefficient_b: float

    def kinematic_viscosity_cst(self, temp_c: Number) -> Number:
        """Kinematic viscosity from the fitted line.

        Equation: nu = 10^(10^(A - B log10(T_K))) - 0.7, ASTM D341.
        Units: cSt, temperature in degrees Celsius.
        """
        temp_k = np.asarray(celsius_to_kelvin(temp_c), dtype=float)
        if np.any(temp_k <= 0.0):
            raise PhysicsDomainError(
                "Temperature below absolute zero passed to the viscosity model.",
                temp_c=float(np.min(np.asarray(temp_c, dtype=float))),
            )
        inner = self.coefficient_a - self.coefficient_b * np.log10(temp_k)
        # Guard the double exponential: clip the inner exponent to a range that
        # brackets everything from a hot light oil to a cold bitumen.
        inner = np.clip(inner, -1.2, 1.4)
        value = np.power(10.0, np.power(10.0, inner)) - WALTHER_OFFSET
        value = np.clip(value, MIN_KINEMATIC_VISCOSITY_CST, MAX_KINEMATIC_VISCOSITY_CST)
        return float(value) if np.isscalar(temp_c) else np.asarray(value, dtype=float)


def fit_walther(anchor_temps_c: list[float], anchor_kinematic_cst: list[float]) -> WaltherFit:
    """Least-squares fit of the ASTM D341 line through two or more anchors.

    Equation: Z = log10(log10(nu + 0.7)) is linear in log10(T_K) with slope -B.
    Units: temperatures in degrees Celsius, viscosities in cSt.
    Assumptions: single-phase Newtonian liquid, no wax or asphaltene
    precipitation inside the fitted range.
    Source: ASTM D341, Standard Practice for Viscosity-Temperature Charts.
    """
    if len(anchor_temps_c) != len(anchor_kinematic_cst):
        raise PhysicsDomainError("Anchor temperature and viscosity lists differ in length.")
    if len(anchor_temps_c) < 2:
        raise PhysicsDomainError("At least two anchors are needed to fit the Walther line.")
    temps_k = np.asarray([celsius_to_kelvin(t) for t in anchor_temps_c], dtype=float)
    viscosities = np.asarray(anchor_kinematic_cst, dtype=float)
    if np.any(temps_k <= 0.0):
        raise PhysicsDomainError("Anchor temperature below absolute zero.")
    if np.any(viscosities <= 0.0):
        raise PhysicsDomainError("Anchor viscosity must be strictly positive.")
    z_values = np.log10(np.log10(viscosities + WALTHER_OFFSET))
    x_values = np.log10(temps_k)
    slope, intercept = np.polyfit(x_values, z_values, 1)
    return WaltherFit(coefficient_a=float(intercept), coefficient_b=float(-slope))


@dataclass(frozen=True)
class ArrheniusFit:
    """Coefficients of the Arrhenius viscosity form mu = a exp(b / T_K)."""

    coefficient_a_pa_s: float
    coefficient_b_k: float

    def dynamic_viscosity_pa_s(self, temp_c: Number) -> Number:
        """Dynamic viscosity from the fitted Arrhenius line.

        Equation: mu = a exp(b / T_K).
        Units: Pa.s, temperature in degrees Celsius.
        """
        temp_k = np.asarray(celsius_to_kelvin(temp_c), dtype=float)
        if np.any(temp_k <= 0.0):
            raise PhysicsDomainError("Temperature below absolute zero in the Arrhenius model.")
        value = self.coefficient_a_pa_s * np.exp(
            np.clip(self.coefficient_b_k / temp_k, -50.0, 50.0)
        )
        return float(value) if np.isscalar(temp_c) else np.asarray(value, dtype=float)


def fit_arrhenius(anchor_temps_c: list[float], anchor_dynamic_pa_s: list[float]) -> ArrheniusFit:
    """Least-squares fit of ln(mu) against 1 / T_K.

    Equation: ln(mu) = ln(a) + b / T_K.
    Units: temperatures in degrees Celsius, viscosities in Pa.s.
    Source: standard Arrhenius (Andrade) viscosity form for liquids.
    """
    if len(anchor_temps_c) < 2:
        raise PhysicsDomainError("At least two anchors are needed to fit the Arrhenius line.")
    temps_k = np.asarray([celsius_to_kelvin(t) for t in anchor_temps_c], dtype=float)
    viscosities = np.asarray(anchor_dynamic_pa_s, dtype=float)
    if np.any(viscosities <= 0.0):
        raise PhysicsDomainError("Anchor viscosity must be strictly positive.")
    slope, intercept = np.polyfit(1.0 / temps_k, np.log(viscosities), 1)
    return ArrheniusFit(
        coefficient_a_pa_s=float(math.exp(intercept)), coefficient_b_k=float(slope * 0.0 + slope)
    ).__class__(coefficient_a_pa_s=float(math.exp(intercept)), coefficient_b_k=float(slope))


class FluidModel:
    """Temperature, water cut and shear dependent properties of the produced fluid.

    The model is built once per well from :class:`FluidConfig` and is then cheap
    to evaluate for scalars or numpy arrays.
    """

    def __init__(self, config: FluidConfig) -> None:
        self.config = config
        self.specific_gravity = api_to_specific_gravity(config.api_gravity_deg)
        self.reference_density_kg_per_m3 = self.specific_gravity * config.water_density_kg_per_m3

    # ---------------------------------------------------------------- density
    def oil_density_kg_per_m3(self, temp_c: Number) -> Number:
        """Oil density against temperature.

        Equation: rho(T) = rho_ref / (1 + beta (T - T_ref)).
        Units: kg/m3, temperature in degrees Celsius.
        Assumptions: constant volumetric thermal expansion coefficient beta over
        the CSS temperature range. This is adequate for a heavy crude between
        ambient and about 320 degrees C.
        Source: API MPMS chapter 11 volume correction, linearised.
        """
        temp = np.asarray(temp_c, dtype=float)
        if np.any(temp <= ABSOLUTE_ZERO_C):
            raise PhysicsDomainError("Temperature below absolute zero passed to oil density.")
        denominator = 1.0 + self.config.oil_thermal_expansion_per_k * (
            temp - self.config.reference_temp_c
        )
        denominator = np.clip(denominator, 0.5, 2.0)
        value = self.reference_density_kg_per_m3 / denominator
        return float(value) if np.isscalar(temp_c) else np.asarray(value, dtype=float)

    def mixture_density_kg_per_m3(self, temp_c: Number, water_cut_frac: Number) -> Number:
        """Volume-weighted density of the produced oil and water mixture."""
        water_cut = self._checked_water_cut(water_cut_frac)
        oil_density = np.asarray(self.oil_density_kg_per_m3(temp_c), dtype=float)
        value = (1.0 - water_cut) * oil_density + water_cut * self.config.water_density_kg_per_m3
        return float(value) if np.isscalar(temp_c) and np.isscalar(water_cut_frac) else value

    # ------------------------------------------------------------- viscosity
    @cached_property
    def walther(self) -> WaltherFit:
        """Walther fit built from the configured anchors."""
        temps = [anchor.temp_c for anchor in self.config.viscosity_anchors]
        kinematic = [
            self._dynamic_cp_to_kinematic_cst(anchor.viscosity_cp, anchor.temp_c)
            for anchor in self.config.viscosity_anchors
        ]
        return fit_walther(temps, kinematic)

    @cached_property
    def arrhenius(self) -> ArrheniusFit:
        """Arrhenius fit built from the configured anchors."""
        temps = [anchor.temp_c for anchor in self.config.viscosity_anchors]
        dynamic = [anchor.viscosity_cp * 1.0e-3 for anchor in self.config.viscosity_anchors]
        return fit_arrhenius(temps, dynamic)

    def _dynamic_cp_to_kinematic_cst(self, viscosity_cp: float, temp_c: float) -> float:
        density = float(self.oil_density_kg_per_m3(temp_c))
        return viscosity_cp * 1000.0 / density

    def dead_oil_viscosity_pa_s(self, temp_c: Number) -> Number:
        """Viscosity of the water-free crude at the given temperature.

        Equation: ASTM D341 (Walther) by default, Arrhenius when selected in
        config. Kinematic viscosity is converted to dynamic with the
        temperature-corrected oil density.
        Units: Pa.s, temperature in degrees Celsius.
        Assumptions: dead oil, no dissolved gas, Newtonian.
        Source: ASTM D341; Beggs and Robinson (1975) for the general approach.
        """
        if self.config.viscosity_model == "arrhenius":
            value = np.asarray(self.arrhenius.dynamic_viscosity_pa_s(temp_c), dtype=float)
        else:
            kinematic_cst = np.asarray(self.walther.kinematic_viscosity_cst(temp_c), dtype=float)
            density = np.asarray(self.oil_density_kg_per_m3(temp_c), dtype=float)
            value = kinematic_cst * 1.0e-6 * density
        if self.config.asphaltene_enabled:
            value = value * self.config.asphaltene_multiplier
        require_finite(value, "dead_oil_viscosity_pa_s")
        return float(value) if np.isscalar(temp_c) else np.asarray(value, dtype=float)

    def emulsion_multiplier(self, water_cut_frac: Number) -> Number:
        """Viscosity uplift of a water-in-oil emulsion against water cut.

        Equations:
          Richardson: mu_e / mu_o = exp(k * WC) below the inversion point.
          Brinkman:   mu_e / mu_o = (1 - WC)^(-2.5) below the inversion point.
        Above the inversion water cut the emulsion inverts to oil-in-water and
        the apparent viscosity collapses towards the water value; the model
        blends linearly from the inversion multiplier down to 1.0 at a water
        cut of 1.0, which is a deliberate simplification.
        Units: dimensionless multiplier, water cut as a fraction in [0, 1).
        Assumptions: no surfactant chemistry, isothermal inversion point.
        Source: Richardson (1933); Brinkman (1952); Pal, Rheology of Emulsions.
        """
        water_cut = self._checked_water_cut(water_cut_frac)
        if self.config.emulsion_model == "none":
            multiplier = np.ones_like(water_cut)
        else:
            inversion = self.config.emulsion_inversion_water_cut_frac
            below = np.minimum(water_cut, inversion)
            if self.config.emulsion_model == "richardson":
                rising = np.exp(self.config.emulsion_richardson_k * below)
            else:
                rising = np.power(np.clip(1.0 - below, 1.0e-3, 1.0), -2.5)
            peak = (
                math.exp(self.config.emulsion_richardson_k * inversion)
                if self.config.emulsion_model == "richardson"
                else (1.0 - inversion) ** -2.5
            )
            above_fraction = np.clip(
                (water_cut - inversion) / max(1.0 - inversion, 1.0e-6), 0.0, 1.0
            )
            falling = peak + (1.0 - peak) * above_fraction
            multiplier = np.where(water_cut <= inversion, rising, falling)
        multiplier = np.clip(multiplier, 1.0e-3, 1.0e3)
        return (
            float(multiplier)
            if np.isscalar(water_cut_frac)
            else np.asarray(multiplier, dtype=float)
        )

    def viscosity_pa_s(
        self,
        temp_c: Number,
        water_cut_frac: Number = 0.0,
        shear_rate_per_s: Number | None = None,
    ) -> Number:
        """Apparent viscosity of the produced fluid.

        Equation: mu = mu_dead(T) * f_emulsion(WC) * f_shear(gamma), where the
        shear factor is unity unless the power-law option is enabled.
        Units: Pa.s, temperature in degrees Celsius, water cut as a fraction,
        shear rate in 1/s.
        Assumptions: the emulsion multiplier is temperature independent, which
        is conservative for a heated well because real emulsions break as the
        temperature rises.
        Source: ASTM D341 combined with Richardson or Brinkman emulsion uplift.
        """
        base = np.asarray(self.dead_oil_viscosity_pa_s(temp_c), dtype=float)
        value = base * np.asarray(self.emulsion_multiplier(water_cut_frac), dtype=float)
        if self.config.non_newtonian_enabled and shear_rate_per_s is not None:
            shear = np.clip(np.asarray(shear_rate_per_s, dtype=float), 1.0e-4, 1.0e5)
            value = value * np.power(shear, self.config.power_law_index - 1.0)
        value = np.clip(value, 1.0e-6, 1.0e6)
        require_finite(value, "viscosity_pa_s")
        scalar = np.isscalar(temp_c) and np.isscalar(water_cut_frac)
        return float(value) if scalar else np.asarray(value, dtype=float)

    def viscosity_cp(self, temp_c: Number, water_cut_frac: Number = 0.0) -> Number:
        """Apparent viscosity in centipoise, for reporting at the edges."""
        return np.asarray(self.viscosity_pa_s(temp_c, water_cut_frac), dtype=float) * 1000.0

    def temperature_for_viscosity_c(
        self, target_viscosity_pa_s: float, water_cut_frac: float = 0.0
    ) -> float:
        """Temperature at which the fluid reaches a target viscosity.

        Solved by bisection on the monotone viscosity-temperature relation over
        0 to 350 degrees C. Used to size tank heating and to explain how much
        heating a pumpable viscosity needs.
        """
        if target_viscosity_pa_s <= 0.0:
            raise PhysicsDomainError("Target viscosity must be positive.")
        low, high = 0.0, 350.0
        if self.viscosity_pa_s(low, water_cut_frac) < target_viscosity_pa_s:
            return low
        if self.viscosity_pa_s(high, water_cut_frac) > target_viscosity_pa_s:
            return high
        for _ in range(80):
            mid = 0.5 * (low + high)
            if float(self.viscosity_pa_s(mid, water_cut_frac)) > target_viscosity_pa_s:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    # -------------------------------------------------------------- thermal
    def mixture_specific_heat_j_per_kg_k(self, water_cut_frac: Number) -> Number:
        """Mass-weighted specific heat of the produced mixture.

        Assumption: the water cut is a volume fraction at the measurement
        temperature; the mass weighting uses the configured reference densities.
        """
        water_cut = self._checked_water_cut(water_cut_frac)
        water_mass = water_cut * self.config.water_density_kg_per_m3
        oil_mass = (1.0 - water_cut) * self.reference_density_kg_per_m3
        total = water_mass + oil_mass
        value = (
            water_mass * self.config.water_specific_heat_j_per_kg_k
            + oil_mass * self.config.oil_specific_heat_j_per_kg_k
        ) / np.clip(total, 1.0e-6, None)
        return float(value) if np.isscalar(water_cut_frac) else np.asarray(value, dtype=float)

    # -------------------------------------------------------------- internal
    @staticmethod
    def _checked_water_cut(water_cut_frac: Number) -> NDArray[np.float64]:
        water_cut = np.asarray(water_cut_frac, dtype=float)
        if np.any(water_cut < 0.0) or np.any(water_cut > 1.0):
            raise PhysicsDomainError(
                "Water cut must lie in [0, 1].",
                water_cut_frac=float(np.max(np.abs(water_cut))),
            )
        return water_cut
