"""Transient numerical wellbore heat transmission for the truth simulator.

The twin uses the analytic Ramey relaxation-distance solution with a
correlated earth time function. Here the formation around the well is
discretised radially at every depth node and the conduction into it is solved
in time, so the wellbore and the formation are genuinely coupled rather than
the formation being represented by a correlation. The steam quality balance
down the tubing is marched on the same grid.

The differences from the twin are deliberate: a resolved formation instead of
f(t), a coupled rather than sequential solution, and a vacuum insulated tubing
coefficient that degrades with time because real VIT loses vacuum.

The radial systems at all depth nodes share the same tridiagonal structure, so
they are solved together with a vectorised Thomas algorithm. That keeps a
model with a resolved formation fast enough to generate two years of telemetry
for a dozen wells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import FieldConfig
from app.core.errors import NumericalError
from app.core.numerics import require_finite, require_positive
from app.core.steam import latent_heat_j_per_kg, liquid_enthalpy_j_per_kg, saturation_temperature_c
from app.core.units import SECONDS_PER_DAY, STANDARD_GRAVITY_M_PER_S2

FORMATION_OUTER_RADIUS_M = 15.0
"""How far into the formation the transient grid reaches."""


def solve_tridiagonal_batch(
    lower: NDArray[np.float64],
    diagonal: NDArray[np.float64],
    upper: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve many tridiagonal systems that share a structure, by the Thomas algorithm.

    Args:
        lower: Sub-diagonal, shape (systems, n). Entry 0 is unused.
        diagonal: Main diagonal, shape (systems, n).
        upper: Super-diagonal, shape (systems, n). Entry n-1 is unused.
        right: Right hand sides, shape (systems, n).

    Returns:
        The solutions, shape (systems, n).
    """
    count = diagonal.shape[1]
    modified_upper = np.zeros_like(diagonal)
    modified_right = np.zeros_like(right)
    pivot = diagonal[:, 0]
    if np.any(np.abs(pivot) < 1.0e-300):
        raise NumericalError("Singular tridiagonal system in the transient wellbore solve.")
    modified_upper[:, 0] = upper[:, 0] / pivot
    modified_right[:, 0] = right[:, 0] / pivot
    for index in range(1, count):
        pivot = diagonal[:, index] - lower[:, index] * modified_upper[:, index - 1]
        if np.any(np.abs(pivot) < 1.0e-300):
            raise NumericalError("Singular tridiagonal system in the transient wellbore solve.")
        modified_upper[:, index] = upper[:, index] / pivot
        modified_right[:, index] = (
            right[:, index] - lower[:, index] * modified_right[:, index - 1]
        ) / pivot
    solution = np.zeros_like(right)
    solution[:, -1] = modified_right[:, -1]
    for index in range(count - 2, -1, -1):
        solution[:, index] = (
            modified_right[:, index] - modified_upper[:, index] * solution[:, index + 1]
        )
    return solution


@dataclass(frozen=True)
class WellboreInjectionResult:
    """Steam state arriving at the sandface."""

    depth_m: NDArray[np.float64]
    temperature_c: NDArray[np.float64]
    quality_frac: NDArray[np.float64]
    pressure_kpa: NDArray[np.float64]
    sandface_temp_c: float
    sandface_quality_frac: float
    sandface_heat_rate_w: float
    surface_heat_rate_w: float
    heat_loss_w: float

    @property
    def heat_loss_fraction(self) -> float:
        """Fraction of the surface heat rate lost through the tubing wall."""
        if self.surface_heat_rate_w <= 0.0:
            return 0.0
        return float(min(max(self.heat_loss_w / self.surface_heat_rate_w, 0.0), 1.0))


@dataclass(frozen=True)
class WellboreProductionResult:
    """Produced fluid temperature along the string."""

    depth_m: NDArray[np.float64]
    temperature_c: NDArray[np.float64]
    wellhead_temp_c: float
    pump_intake_temp_c: float


class TransientWellbore:
    """Wellbore with a radially discretised formation at every depth, solved in time."""

    def __init__(
        self,
        config: FieldConfig,
        base_heat_transfer_w_per_m2_k: float,
        degradation_per_year_frac: float = 0.0,
        depth_nodes: int = 31,
        radial_nodes: int = 12,
    ) -> None:
        self.config = config
        self.base_heat_transfer = base_heat_transfer_w_per_m2_k
        self.degradation_per_year_frac = degradation_per_year_frac
        self.depth_m = np.linspace(0.0, config.reservoir.depth_m, depth_nodes)
        self.tubing_outer_radius_m = 0.5 * config.wellbore.tubing_outer_diameter_m
        self.radius_face_m = np.geomspace(
            self.tubing_outer_radius_m, FORMATION_OUTER_RADIUS_M, radial_nodes + 1
        )
        self.radius_centre_m = np.sqrt(self.radius_face_m[:-1] * self.radius_face_m[1:])
        self.formation_temp_c = self._geothermal_profile()
        self.formation_field_c = np.repeat(self.formation_temp_c[:, None], radial_nodes, axis=1)
        self.elapsed_years = 0.0
        self._cell_volume_m2 = math.pi * (
            self.radius_face_m[1:] ** 2 - self.radius_face_m[:-1] ** 2
        )
        self._inner_conductance_base = 2.0 * math.pi * self.tubing_outer_radius_m
        self._build_static_conductances()

    def _geothermal_profile(self) -> NDArray[np.float64]:
        wellbore = self.config.wellbore
        return wellbore.surface_temp_c + wellbore.geothermal_gradient_c_per_m * self.depth_m

    def _build_static_conductances(self) -> None:
        """Face conductances of the radial grid, in W/K per metre of well."""
        conductivity = self.config.wellbore.formation_thermal_conductivity_w_per_m_k
        nodes = self.radius_centre_m.size
        self._right_conductance = np.zeros(nodes)
        self._left_conductance = np.zeros(nodes)
        for index in range(nodes - 1):
            area = 2.0 * math.pi * self.radius_face_m[index + 1]
            distance = self.radius_centre_m[index + 1] - self.radius_centre_m[index]
            self._right_conductance[index] = conductivity * area / distance
        self._left_conductance[1:] = self._right_conductance[:-1]

    def heat_transfer_coefficient(self, elapsed_years: float) -> float:
        """Overall tubing coefficient after ``elapsed_years`` of vacuum loss."""
        return self.base_heat_transfer * (
            1.0 + self.degradation_per_year_frac * max(elapsed_years, 0.0)
        )

    def wall_temperature_c(self) -> NDArray[np.float64]:
        """Formation temperature at the first cell out from the tubing.

        The fluid march uses this to compute its heat loss, which makes the
        coupling explicit in the formation and implicit in the fluid. That is
        stable. Guessing a fluid profile, advancing the formation against it and
        then re-marching is not: with a bare completion the two passes oscillate
        between a fully cooled and a fully hot string.
        """
        return self.formation_field_c[:, 0].copy()

    def heat_loss_w_per_m(
        self, fluid_temperature_c: NDArray[np.float64] | float
    ) -> NDArray[np.float64]:
        """Heat loss per unit length for a given fluid temperature."""
        inner_conductance = (
            self.heat_transfer_coefficient(self.elapsed_years) * self._inner_conductance_base
        )
        return inner_conductance * (
            np.asarray(fluid_temperature_c, dtype=float) - self.wall_temperature_c()
        )

    def advance_formation(
        self, fluid_temperature_c: NDArray[np.float64], step_days: float
    ) -> NDArray[np.float64]:
        """Advance the formation one step and return the heat loss per unit length.

        Backward Euler in the radial direction at every depth node at once. The
        inner boundary couples to the fluid through the overall heat transfer
        coefficient; the outer boundary is pinned to the undisturbed geothermal
        temperature.
        """
        wellbore = self.config.wellbore
        conductivity = wellbore.formation_thermal_conductivity_w_per_m_k
        capacity = conductivity / wellbore.formation_thermal_diffusivity_m2_per_s
        step_s = max(step_days, 1.0e-4) * SECONDS_PER_DAY
        depths, nodes = self.formation_field_c.shape

        storage = capacity * self._cell_volume_m2 / step_s
        inner_conductance = (
            self.heat_transfer_coefficient(self.elapsed_years) * self._inner_conductance_base
        )

        lower = np.tile(-self._left_conductance, (depths, 1))
        upper = np.tile(-self._right_conductance, (depths, 1))
        diagonal = np.tile(storage + self._left_conductance + self._right_conductance, (depths, 1))
        right = storage * self.formation_field_c

        diagonal[:, 0] += inner_conductance
        right[:, 0] += inner_conductance * np.asarray(fluid_temperature_c, dtype=float)
        # Dirichlet outer boundary at the undisturbed temperature.
        diagonal[:, -1] = 1.0
        lower[:, -1] = 0.0
        upper[:, -1] = 0.0
        right[:, -1] = self.formation_temp_c

        solution = solve_tridiagonal_batch(lower, diagonal, upper, right)
        require_finite(solution, "transient wellbore formation field")
        self.formation_field_c = solution
        return inner_conductance * (np.asarray(fluid_temperature_c, dtype=float) - solution[:, 0])

    # -------------------------------------------------------------- injection
    def inject_steam(
        self,
        wellhead_pressure_kpa: float,
        wellhead_quality_frac: float,
        rate_m3_per_day_cwe: float,
        step_days: float,
        elapsed_years: float = 0.0,
    ) -> WellboreInjectionResult:
        """March saturated steam down the tubing, coupled to the formation solve.

        The steam is marched using the heat loss implied by the formation state
        at the start of the step, and the formation is then advanced against the
        resulting fluid temperature profile. Making the coupling explicit in the
        formation and implicit in the fluid keeps it stable; iterating a guessed
        fluid profile against the formation does not.
        """
        require_positive(rate_m3_per_day_cwe, "rate_m3_per_day_cwe")
        self.elapsed_years = elapsed_years
        mass_rate = (
            rate_m3_per_day_cwe * self.config.fluid.water_density_kg_per_m3 / SECONDS_PER_DAY
        )
        step_m = float(self.depth_m[1] - self.depth_m[0])
        temperature = np.zeros(self.depth_m.size)
        quality = np.zeros(self.depth_m.size)
        pressure = np.zeros(self.depth_m.size)
        total_loss_w = 0.0

        inner_conductance = (
            self.heat_transfer_coefficient(self.elapsed_years) * self._inner_conductance_base
        )
        wall_temperature = self.wall_temperature_c()

        pressure_kpa = wellhead_pressure_kpa
        quality_frac = wellhead_quality_frac
        temperature_c = saturation_temperature_c(min(max(pressure_kpa, 120.0), 19500.0))
        for index in range(self.depth_m.size):
            temperature[index] = temperature_c
            quality[index] = quality_frac
            pressure[index] = pressure_kpa
            if index == self.depth_m.size - 1:
                break
            loss_w = (
                max(inner_conductance * (temperature_c - float(wall_temperature[index])), 0.0)
                * step_m
            )
            total_loss_w += loss_w
            latent = latent_heat_j_per_kg(min(max(pressure_kpa, 120.0), 19500.0))
            if quality_frac > 1.0e-6:
                quality_frac = max(quality_frac - loss_w / (mass_rate * latent), 0.0)
            else:
                temperature_c = max(
                    temperature_c
                    - loss_w / (mass_rate * self.config.fluid.water_specific_heat_j_per_kg_k),
                    float(self.formation_temp_c[index]),
                )
            vapour_density = pressure_kpa * 1000.0 * 0.018015 / (8.314 * (temperature_c + 273.15))
            specific_volume = (
                quality_frac / max(vapour_density, 1.0e-3) + (1.0 - quality_frac) / 850.0
            )
            mixture_density = 1.0 / max(specific_volume, 1.0e-6)
            pressure_kpa = min(
                pressure_kpa + mixture_density * STANDARD_GRAVITY_M_PER_S2 * step_m / 1000.0,
                19500.0,
            )
            if quality_frac > 1.0e-6:
                temperature_c = saturation_temperature_c(max(pressure_kpa, 120.0))

        self.advance_formation(temperature, step_days)

        datum_j_per_kg = (
            self.config.fluid.water_specific_heat_j_per_kg_k * self.config.reservoir.initial_temp_c
        )

        def heat_rate(
            pressure_value: float, quality_value: float, temperature_value: float
        ) -> float:
            clamped = min(max(pressure_value, 120.0), 19500.0)
            if quality_value > 1.0e-6:
                enthalpy = liquid_enthalpy_j_per_kg(clamped) + quality_value * latent_heat_j_per_kg(
                    clamped
                )
            else:
                enthalpy = self.config.fluid.water_specific_heat_j_per_kg_k * temperature_value
            return mass_rate * max(enthalpy - datum_j_per_kg, 0.0)

        surface_w = heat_rate(wellhead_pressure_kpa, wellhead_quality_frac, float(temperature[0]))
        sandface_w = heat_rate(float(pressure[-1]), float(quality[-1]), float(temperature[-1]))
        return WellboreInjectionResult(
            depth_m=self.depth_m,
            temperature_c=temperature,
            quality_frac=quality,
            pressure_kpa=pressure,
            sandface_temp_c=float(temperature[-1]),
            sandface_quality_frac=float(quality[-1]),
            sandface_heat_rate_w=float(max(sandface_w, 0.0)),
            surface_heat_rate_w=float(surface_w),
            heat_loss_w=float(total_loss_w),
        )

    # ------------------------------------------------------------- production
    def produce(
        self,
        sandface_temp_c: float,
        liquid_rate_m3_per_day: float,
        water_cut_frac: float,
        step_days: float,
        pump_depth_m: float,
        elapsed_years: float = 0.0,
    ) -> WellboreProductionResult:
        """March produced fluid up the tubing, coupled to the formation solve."""
        self.elapsed_years = elapsed_years
        density = 900.0 * (1.0 - water_cut_frac) + 1000.0 * water_cut_frac
        specific_heat = 2100.0 * (1.0 - water_cut_frac) + 4186.0 * water_cut_frac
        mass_rate = max(liquid_rate_m3_per_day * density / SECONDS_PER_DAY, 1.0e-6)
        step_m = float(self.depth_m[1] - self.depth_m[0])
        temperature = np.zeros(self.depth_m.size)
        inner_conductance = (
            self.heat_transfer_coefficient(self.elapsed_years) * self._inner_conductance_base
        )
        wall_temperature = self.wall_temperature_c()

        temperature_c = sandface_temp_c
        for position in reversed(range(self.depth_m.size)):
            temperature[position] = temperature_c
            if position == 0:
                break
            loss_w = (
                inner_conductance * (temperature_c - float(wall_temperature[position])) * step_m
            )
            temperature_c = temperature_c - loss_w / (mass_rate * specific_heat)
            temperature_c = min(
                max(temperature_c, float(self.formation_temp_c[position - 1])),
                sandface_temp_c,
            )

        self.advance_formation(temperature, step_days)

        intake_index = int(np.argmin(np.abs(self.depth_m - pump_depth_m)))
        require_finite(temperature, "transient wellbore production profile")
        return WellboreProductionResult(
            depth_m=self.depth_m,
            temperature_c=temperature,
            wellhead_temp_c=float(temperature[0]),
            pump_intake_temp_c=float(temperature[intake_index]),
        )
