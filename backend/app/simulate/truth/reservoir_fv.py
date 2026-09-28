"""Axisymmetric finite-volume thermal reservoir model for the truth simulator.

This is deliberately a different model from the one in ``app.twin.reservoir``.
The twin uses analytic lumped solutions: Marx and Langenheim for the heated
area and a pair of conduction unit solutions for the cooling. Here the
temperature field is solved on an (r, z) grid with several layers of different
permeability, an implicit conduction solve, explicit advection of the injected
enthalpy distributed between layers by mobility, optional gravity override and
an optional high permeability fracture streak. Inflow is computed by summing
series radial resistances cell by cell through the resolved temperature field
rather than from a two-region formula.

That difference is the whole point. If the same equations generated the data
and powered the twin, every validation number would be meaningless.

Discretisation: logarithmic radial grid from the wellbore to the drainage
radius, uniform vertical grid covering the pay plus a buffer of over and
underburden so the heat loss upward and downward is resolved rather than
correlated. Conduction is advanced with backward Euler on a sparse operator, so
the daily step is unconditionally stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from numpy.typing import NDArray

from app.core.config import FieldConfig
from app.core.errors import NumericalError, PhysicsDomainError
from app.core.numerics import require_finite
from app.core.steam import latent_heat_j_per_kg, saturation_temperature_c
from app.core.units import SECONDS_PER_DAY
from app.simulate.truth.priors import HiddenParameters

MILLIDARCY_TO_M2 = 9.869233e-16
BURDEN_BUFFER_M = 30.0
"""Thickness of over and underburden carried in the grid, each side."""


@dataclass(frozen=True)
class GridGeometry:
    """Cell centres, faces and volumes of the axisymmetric grid."""

    radius_face_m: NDArray[np.float64]
    radius_centre_m: NDArray[np.float64]
    depth_face_m: NDArray[np.float64]
    depth_centre_m: NDArray[np.float64]
    volume_m3: NDArray[np.float64]
    is_pay: NDArray[np.bool_]
    layer_index: NDArray[np.int_]

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as (radial cells, vertical cells)."""
        return (self.radius_centre_m.size, self.depth_centre_m.size)

    @property
    def cell_count(self) -> int:
        """Total number of cells."""
        return int(self.radius_centre_m.size * self.depth_centre_m.size)


def build_grid(
    wellbore_radius_m: float,
    drainage_radius_m: float,
    layer_thicknesses_m: NDArray[np.float64],
    radial_cells: int = 36,
    cells_per_layer: int = 3,
    burden_cells: int = 8,
) -> GridGeometry:
    """Build a logarithmic radial by uniform vertical grid.

    A logarithmic radial grid puts resolution where the gradients are, which is
    at the well. The vertical grid resolves each reservoir layer and carries a
    buffer of over and underburden so the conduction loss out of the pay is
    computed rather than correlated.
    """
    if radial_cells < 8 or cells_per_layer < 1 or burden_cells < 2:
        raise PhysicsDomainError("The truth grid is too coarse to be meaningful.")
    radius_face = np.geomspace(wellbore_radius_m, drainage_radius_m, radial_cells + 1)
    radius_centre = np.sqrt(radius_face[:-1] * radius_face[1:])

    pay_thickness_m = float(np.sum(layer_thicknesses_m))
    burden_step_m = BURDEN_BUFFER_M / burden_cells
    upper = np.arange(burden_cells, 0, -1) * -burden_step_m
    pay_faces: list[float] = [0.0]
    layer_of_cell: list[int] = []
    cursor = 0.0
    for index, thickness in enumerate(layer_thicknesses_m):
        step = float(thickness) / cells_per_layer
        for _ in range(cells_per_layer):
            cursor += step
            pay_faces.append(cursor)
            layer_of_cell.append(index)
    lower = pay_thickness_m + np.arange(1, burden_cells + 1) * burden_step_m

    depth_face = np.concatenate([upper - burden_step_m * 0.0, np.asarray(pay_faces), lower])
    depth_face = np.unique(np.round(depth_face, 9))
    depth_centre = 0.5 * (depth_face[:-1] + depth_face[1:])

    is_pay = (depth_centre > 0.0) & (depth_centre < pay_thickness_m)
    layer_index = np.full(depth_centre.size, -1, dtype=int)
    pay_positions = np.nonzero(is_pay)[0]
    if pay_positions.size != len(layer_of_cell):
        # Assign by depth if rounding merged a face.
        cumulative = np.cumsum(layer_thicknesses_m)
        for position in pay_positions:
            layer_index[position] = int(np.searchsorted(cumulative, depth_centre[position]))
        layer_index = np.clip(layer_index, -1, len(layer_thicknesses_m) - 1)
    else:
        layer_index[pay_positions] = layer_of_cell

    annulus_area = math.pi * (radius_face[1:] ** 2 - radius_face[:-1] ** 2)
    cell_height = np.diff(depth_face)
    volume = annulus_area[:, None] * cell_height[None, :]
    return GridGeometry(
        radius_face_m=radius_face,
        radius_centre_m=radius_centre,
        depth_face_m=depth_face,
        depth_centre_m=depth_centre,
        volume_m3=volume,
        is_pay=is_pay,
        layer_index=layer_index,
    )


class AxisymmetricThermalReservoir:
    """Temperature field, pressure and inflow of one well, resolved in (r, z).

    Args:
        config: Field configuration, for the properties that are not hidden.
        hidden: The hidden per-well parameters.
        radial_cells: Radial resolution.
        cells_per_layer: Vertical cells inside each reservoir layer.
    """

    def __init__(
        self,
        config: FieldConfig,
        hidden: HiddenParameters,
        radial_cells: int = 36,
        cells_per_layer: int = 3,
    ) -> None:
        self.config = config
        self.hidden = hidden
        self.grid = build_grid(
            wellbore_radius_m=config.reservoir.wellbore_radius_m,
            drainage_radius_m=config.reservoir.drainage_radius_m,
            layer_thicknesses_m=hidden.layer_thicknesses_m(),
            radial_cells=radial_cells,
            cells_per_layer=cells_per_layer,
        )
        self.initial_temp_c = config.reservoir.initial_temp_c
        self.temperature_c = np.full(self.grid.shape, self.initial_temp_c)
        self.pressure_kpa = hidden.initial_pressure_kpa
        self.oil_saturation_frac = config.reservoir.oil_saturation_initial_frac
        self.cumulative_oil_m3 = 0.0
        self.cumulative_water_m3 = 0.0
        self.cumulative_steam_m3_cwe = 0.0
        self.cycle_number = 1
        self.day = 0.0
        self.production_day = 0.0
        self._conduction_operator: sparse.csc_matrix | None = None
        self._operator_step_days: float | None = None

        self.thermal_conductivity = self._build_field(
            pay_value=config.reservoir.reservoir_thermal_conductivity_w_per_m_k,
            burden_value=hidden.overburden_thermal_conductivity_w_per_m_k,
        )
        self.volumetric_heat_capacity = self._build_field(
            pay_value=config.reservoir.rock_volumetric_heat_capacity_j_per_m3_k,
            burden_value=config.reservoir.overburden_volumetric_heat_capacity_j_per_m3_k,
        )
        self.permeability_m2 = self._build_permeability_field()

    # ------------------------------------------------------------------ setup
    def _build_field(self, pay_value: float, burden_value: float) -> NDArray[np.float64]:
        """A cell field that takes one value in the pay and another in the burden."""
        field = np.full(self.grid.shape, burden_value)
        field[:, self.grid.is_pay] = pay_value
        return field

    def _build_permeability_field(self) -> NDArray[np.float64]:
        """Permeability per cell, taken from the layer it belongs to."""
        field = np.zeros(self.grid.shape)
        for column, layer in enumerate(self.grid.layer_index):
            if layer < 0:
                continue
            field[:, column] = self.hidden.layers[layer].permeability_md * MILLIDARCY_TO_M2
        return field

    @property
    def pay_thickness_m(self) -> float:
        """Total net pay thickness."""
        return float(np.sum(self.hidden.layer_thicknesses_m()))

    def pay_cell_heights_m(self) -> NDArray[np.float64]:
        """Height of every vertical cell."""
        return np.diff(self.grid.depth_face_m)

    # ------------------------------------------------------------- conduction
    def _build_conduction_operator(self, step_days: float) -> sparse.csc_matrix:
        """Backward Euler operator for two-dimensional conduction.

        Equation: (I - dt L) T^{n+1} = T^n, with L the finite-volume Laplacian
        divided by the volumetric heat capacity. Harmonic means are used at the
        faces so a contrast in conductivity is handled correctly.
        """
        radial_cells, vertical_cells = self.grid.shape
        step_s = step_days * SECONDS_PER_DAY
        heights = self.pay_cell_heights_m()
        volume = self.grid.volume_m3
        capacity = self.volumetric_heat_capacity * volume

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []

        def index(i: int, j: int) -> int:
            return i * vertical_cells + j

        def harmonic(a: float, b: float) -> float:
            return 2.0 * a * b / (a + b) if (a + b) > 0.0 else 0.0

        for i in range(radial_cells):
            for j in range(vertical_cells):
                centre = index(i, j)
                diagonal = capacity[i, j] / step_s
                # Radial faces.
                if i + 1 < radial_cells:
                    area = 2.0 * math.pi * self.grid.radius_face_m[i + 1] * heights[j]
                    distance = self.grid.radius_centre_m[i + 1] - self.grid.radius_centre_m[i]
                    conductance = (
                        harmonic(
                            self.thermal_conductivity[i, j],
                            self.thermal_conductivity[i + 1, j],
                        )
                        * area
                        / distance
                    )
                    diagonal += conductance
                    rows.append(centre)
                    columns.append(index(i + 1, j))
                    values.append(-conductance)
                if i > 0:
                    area = 2.0 * math.pi * self.grid.radius_face_m[i] * heights[j]
                    distance = self.grid.radius_centre_m[i] - self.grid.radius_centre_m[i - 1]
                    conductance = (
                        harmonic(
                            self.thermal_conductivity[i, j],
                            self.thermal_conductivity[i - 1, j],
                        )
                        * area
                        / distance
                    )
                    diagonal += conductance
                    rows.append(centre)
                    columns.append(index(i - 1, j))
                    values.append(-conductance)
                # Vertical faces.
                annulus = math.pi * (
                    self.grid.radius_face_m[i + 1] ** 2 - self.grid.radius_face_m[i] ** 2
                )
                if j + 1 < vertical_cells:
                    distance = self.grid.depth_centre_m[j + 1] - self.grid.depth_centre_m[j]
                    conductance = (
                        harmonic(
                            self.thermal_conductivity[i, j],
                            self.thermal_conductivity[i, j + 1],
                        )
                        * annulus
                        / distance
                    )
                    diagonal += conductance
                    rows.append(centre)
                    columns.append(index(i, j + 1))
                    values.append(-conductance)
                if j > 0:
                    distance = self.grid.depth_centre_m[j] - self.grid.depth_centre_m[j - 1]
                    conductance = (
                        harmonic(
                            self.thermal_conductivity[i, j],
                            self.thermal_conductivity[i, j - 1],
                        )
                        * annulus
                        / distance
                    )
                    diagonal += conductance
                    rows.append(centre)
                    columns.append(index(i, j - 1))
                    values.append(-conductance)
                rows.append(centre)
                columns.append(centre)
                values.append(diagonal)

        size = radial_cells * vertical_cells
        return sparse.csc_matrix(sparse.coo_matrix((values, (rows, columns)), shape=(size, size)))

    def step_conduction(self, step_days: float) -> None:
        """Advance conduction by one step with a backward Euler solve."""
        if step_days <= 0.0:
            return
        if self._conduction_operator is None or self._operator_step_days != step_days:
            self._conduction_operator = self._build_conduction_operator(step_days)
            self._operator_step_days = step_days
        step_s = step_days * SECONDS_PER_DAY
        capacity = self.volumetric_heat_capacity * self.grid.volume_m3
        right_hand_side = (capacity / step_s * self.temperature_c).reshape(-1)
        # Far-field and outer-burden boundaries are held at the virgin temperature.
        solution = sparse_linalg.spsolve(self._conduction_operator, right_hand_side)
        if not np.all(np.isfinite(solution)):
            raise NumericalError("The truth conduction solve produced non-finite values.")
        self.temperature_c = solution.reshape(self.grid.shape)
        self.temperature_c[-1, :] = self.initial_temp_c
        self.temperature_c[:, 0] = self.initial_temp_c
        self.temperature_c[:, -1] = self.initial_temp_c

    # -------------------------------------------------------------- injection
    def layer_mobility_weights(self) -> NDArray[np.float64]:
        """How the injected steam divides between layers.

        Weighted by kh, which is the standard injectivity split, and then biased
        upward by the gravity override strength because steam rises. The twin
        has no vertical resolution at all, so it cannot represent this, and that
        is one of the discrepancies the validation report has to be honest about.
        """
        layers = self.hidden.layers
        weights = np.asarray(
            [layer.permeability_md * layer.thickness_m for layer in layers], dtype=float
        )
        weights = np.maximum(weights, 1.0e-9)
        weights /= weights.sum()
        override = self.hidden.gravity_override_strength_frac
        if override > 0.0:
            # Layer zero is the top of the pay.
            bias = np.linspace(1.0 + override, 1.0 - override, len(layers))
            bias = np.maximum(bias, 0.05)
            weights = weights * bias
            weights /= weights.sum()
        return weights

    def step_injection(
        self,
        step_days: float,
        sandface_heat_rate_w: float,
        sandface_steam_temp_c: float,
        steam_rate_m3_per_day_cwe: float,
    ) -> None:
        """Advance injection by one step.

        The injected enthalpy is divided between layers by mobility, and within
        each layer it is deposited outward from the well: the cells nearest the
        well are raised to the steam temperature first, and the remainder of the
        heat moves to the next cell out. This is an explicit front advance
        rather than the analytic Marx and Langenheim area, and it produces a
        different, layer-dependent heated shape.
        """
        if step_days <= 0.0:
            return
        heat_j = sandface_heat_rate_w * step_days * SECONDS_PER_DAY
        weights = self.layer_mobility_weights()
        radial_cells, vertical_cells = self.grid.shape

        for layer_position, weight in enumerate(weights):
            columns = np.nonzero(self.grid.layer_index == layer_position)[0]
            if columns.size == 0:
                continue
            layer_heat_j = heat_j * float(weight)
            per_column_j = layer_heat_j / columns.size
            for column in columns:
                remaining_j = per_column_j
                for ring in range(radial_cells):
                    if remaining_j <= 0.0:
                        break
                    capacity_j = (
                        self.volumetric_heat_capacity[ring, column]
                        * self.grid.volume_m3[ring, column]
                    )
                    deficit_k = sandface_steam_temp_c - self.temperature_c[ring, column]
                    if deficit_k <= 0.0:
                        continue
                    take_j = min(remaining_j, capacity_j * deficit_k)
                    self.temperature_c[ring, column] += take_j / capacity_j
                    remaining_j -= take_j

        self.cumulative_steam_m3_cwe += steam_rate_m3_per_day_cwe * step_days
        self.step_conduction(step_days)

        pore_volume = self.pore_volume_m3()
        compressibility = self.config.reservoir.total_compressibility_per_kpa
        self.pressure_kpa = min(
            self.pressure_kpa
            + steam_rate_m3_per_day_cwe * step_days / max(compressibility * pore_volume, 1.0e-9),
            self.config.reservoir.fracture_pressure_kpa,
        )
        self.day += step_days
        require_finite(self.temperature_c, "truth temperature field")

    # ------------------------------------------------------------- production
    def pore_volume_m3(self) -> float:
        """Pore volume of the pay inside the drainage radius."""
        total = 0.0
        for layer_position, layer in enumerate(self.hidden.layers):
            columns = np.nonzero(self.grid.layer_index == layer_position)[0]
            total += float(np.sum(self.grid.volume_m3[:, columns])) * layer.porosity_frac
        return total

    def oil_viscosity_pa_s(self, temperature_c: NDArray[np.float64]) -> NDArray[np.float64]:
        """Oil viscosity at the resolved cell temperatures.

        The truth simulator uses its own two-point Walther fit anchored on the
        hidden viscosity at 50 degrees C, which differs from the anchor the twin
        is configured with. That mismatch is deliberate.
        """
        anchor_hot_cp = 12.0 * (self.hidden.viscosity_50c_cp / 11500.0) ** 0.35
        temps_k = np.asarray([50.0, 200.0]) + 273.15
        viscosities = np.asarray([self.hidden.viscosity_50c_cp, anchor_hot_cp])
        z_values = np.log10(np.log10(viscosities + 0.7))
        slope, intercept = np.polyfit(np.log10(temps_k), z_values, 1)
        temp_k = np.clip(np.asarray(temperature_c, dtype=float) + 273.15, 1.0, None)
        inner = np.clip(intercept + slope * np.log10(temp_k), -1.2, 1.4)
        centipoise = np.clip(np.power(10.0, np.power(10.0, inner)) - 0.7, 0.3, 5.0e7)
        return centipoise * 1.0e-3

    def _layer_ring_viscosity_pa_s(self) -> NDArray[np.float64]:
        """Mean oil viscosity in each (ring, layer) cell group."""
        viscosity = self.oil_viscosity_pa_s(self.temperature_c)
        layers = len(self.hidden.layers)
        result = np.zeros((self.grid.shape[0], layers))
        for layer_position in range(layers):
            columns = np.nonzero(self.grid.layer_index == layer_position)[0]
            if columns.size == 0:
                result[:, layer_position] = viscosity[:, 0]
            else:
                heights = self.pay_cell_heights_m()[columns]
                result[:, layer_position] = np.average(
                    viscosity[:, columns], axis=1, weights=heights
                )
        return result

    def transient_ring_count(self) -> NDArray[np.int_]:
        """How many radial rings the pressure transient has reached, per layer.

        The travel time of a radial pressure transient through a cell is
        r dr / (2 eta) with eta = k / (phi mu c_t). Summing that outward gives
        the time at which the transient reaches each ring, and inverting it
        gives the radius of investigation. For a constant diffusivity this
        reduces exactly to r = sqrt(4 eta t).

        The twin uses one diffusivity for the whole cold region. Here the
        diffusivity is evaluated cell by cell against the resolved temperature
        field, so a layer whose heated zone extends further develops its
        drainage faster. That is a real difference between the two models and it
        shows up in the early-cycle rate.
        """
        viscosity = self._layer_ring_viscosity_pa_s()
        compressibility_per_pa = self.config.reservoir.total_compressibility_per_kpa / 1000.0
        radii = self.grid.radius_centre_m
        widths = np.diff(self.grid.radius_face_m)
        counts = np.zeros(len(self.hidden.layers), dtype=int)
        elapsed_s = max(self.production_day, 0.0) * SECONDS_PER_DAY
        for layer_position, layer in enumerate(self.hidden.layers):
            permeability = layer.permeability_md * MILLIDARCY_TO_M2
            diffusivity = permeability / (
                layer.porosity_frac * viscosity[:, layer_position] * compressibility_per_pa
            )
            travel_time_s = np.cumsum(radii * widths / (2.0 * np.maximum(diffusivity, 1.0e-12)))
            reached = np.nonzero(travel_time_s <= elapsed_s)[0]
            counts[layer_position] = int(reached[-1] + 1) if reached.size else 1
        return np.clip(counts, 1, self.grid.shape[0])

    def layer_productivity_index(self) -> NDArray[np.float64]:
        """Productivity index of each layer, by summing series radial resistances.

        Equation: for layer L, 1 / J = sum over radial cells inside the radius
        of investigation of mu(T_cell) ln(r_out / r_in) / (2 pi k h), plus the
        skin term.
        Units: m3/s per Pa.
        This resolves the actual radial temperature profile rather than assuming
        two regions with a sharp boundary, which is what the twin does.
        """
        viscosity = self._layer_ring_viscosity_pa_s()
        log_ratios = np.log(self.grid.radius_face_m[1:] / self.grid.radius_face_m[:-1])
        counts = self.transient_ring_count()
        indices = np.zeros(len(self.hidden.layers))
        for layer_position, layer in enumerate(self.hidden.layers):
            permeability = layer.permeability_md * MILLIDARCY_TO_M2
            geometry = 2.0 * math.pi * permeability * layer.thickness_m
            reach = counts[layer_position]
            resistance = float(
                np.sum(viscosity[:reach, layer_position] * log_ratios[:reach]) / geometry
            )
            resistance += float(
                viscosity[0, layer_position] * self.hidden.skin_dimensionless / geometry
            )
            indices[layer_position] = 1.0 / max(resistance, 1.0e-30)
        return indices

    def deliverability_m3_per_day(self, bottomhole_pressure_kpa: float) -> float:
        """Total liquid the reservoir can deliver against a flowing pressure."""
        drawdown_kpa = self.pressure_kpa - bottomhole_pressure_kpa
        if drawdown_kpa <= 0.0:
            return 0.0
        relative_permeability = self._relative_permeability()
        decline = (1.0 - self.hidden.productivity_decline_per_cycle_frac) ** (self.cycle_number - 1)
        total_index = float(np.sum(self.layer_productivity_index()))
        rate_m3_per_s = total_index * drawdown_kpa * 1000.0 * relative_permeability * decline
        return float(max(rate_m3_per_s * SECONDS_PER_DAY, 0.0))

    def _relative_permeability(self) -> float:
        reservoir = self.config.reservoir
        span = reservoir.oil_saturation_initial_frac - reservoir.residual_oil_saturation_frac
        normalised = (self.oil_saturation_frac - reservoir.residual_oil_saturation_frac) / span
        return float(min(max(normalised, 0.0), 1.0) ** 2)

    def water_cut_frac(self) -> float:
        """Water cut for the current cycle."""
        return float(
            min(
                self.hidden.water_cut_initial_frac
                + self.hidden.water_cut_growth_per_cycle_frac * (self.cycle_number - 1),
                self.config.reservoir.water_cut_max_frac,
            )
        )

    def step_production(
        self, step_days: float, bottomhole_pressure_kpa: float, rate_limit_m3_per_day: float
    ) -> dict[str, float]:
        """Advance production by one step and cool the near-well cells by advection."""
        if step_days <= 0.0:
            return {}
        self.production_day += 0.5 * step_days
        deliverability = self.deliverability_m3_per_day(bottomhole_pressure_kpa)
        self.production_day -= 0.5 * step_days
        total_rate = min(deliverability, max(rate_limit_m3_per_day, 0.0))
        water_cut = self.water_cut_frac()
        oil_rate = total_rate * (1.0 - water_cut)
        water_rate = total_rate * water_cut

        # Advective cooling: the produced fluid leaves at the near-well
        # temperature, and cold fluid enters at the outer boundary. Each layer
        # is drawn down in proportion to its productivity index, so a high
        # permeability streak cools faster than the layers around it.
        indices = self.layer_productivity_index()
        share = indices / max(float(np.sum(indices)), 1.0e-30)
        density = 900.0
        specific_heat = 2100.0
        for layer_position, layer in enumerate(self.hidden.layers):
            columns = np.nonzero(self.grid.layer_index == layer_position)[0]
            if columns.size == 0:
                continue
            layer_volume_m3 = total_rate * step_days * float(share[layer_position])
            if layer_volume_m3 <= 0.0:
                continue
            swept_volume_m3 = layer_volume_m3 / max(layer.porosity_frac, 1.0e-6)
            remaining_m3 = swept_volume_m3
            for ring in range(self.grid.shape[0]):
                if remaining_m3 <= 0.0:
                    break
                cell_volume = float(np.sum(self.grid.volume_m3[ring, columns]))
                fraction = min(remaining_m3 / max(cell_volume, 1.0e-9), 1.0)
                outer = min(ring + 1, self.grid.shape[0] - 1)
                mixed = (1.0 - fraction) * self.temperature_c[
                    ring, columns
                ] + fraction * self.temperature_c[outer, columns]
                self.temperature_c[ring, columns] = mixed
                remaining_m3 -= cell_volume
        _ = density * specific_heat  # properties kept explicit for readability

        self.step_conduction(step_days)

        pore_volume = self.pore_volume_m3()
        compressibility = self.config.reservoir.total_compressibility_per_kpa
        self.pressure_kpa = max(
            self.pressure_kpa - total_rate * step_days / max(compressibility * pore_volume, 1.0e-9),
            self.config.reservoir.abandonment_pressure_kpa,
        )
        oil_in_place = self.oil_saturation_frac * pore_volume
        self.oil_saturation_frac = max(
            (oil_in_place - oil_rate * step_days) / max(pore_volume, 1.0e-9),
            self.config.reservoir.residual_oil_saturation_frac,
        )
        self.cumulative_oil_m3 += oil_rate * step_days
        self.cumulative_water_m3 += water_rate * step_days
        self.day += step_days
        self.production_day += step_days
        require_finite(self.temperature_c, "truth temperature field")
        return {
            "deliverability_m3_per_day": deliverability,
            "total_liquid_rate_m3_per_day": total_rate,
            "oil_rate_m3_per_day": oil_rate,
            "water_rate_m3_per_day": water_rate,
            "water_cut_frac": water_cut,
        }

    # ---------------------------------------------------------------- summary
    def near_well_temperature_c(self) -> float:
        """Pay-weighted temperature of the first radial ring, what a gauge would read."""
        pay_columns = np.nonzero(self.grid.layer_index >= 0)[0]
        heights = self.pay_cell_heights_m()[pay_columns]
        return float(np.average(self.temperature_c[0, pay_columns], weights=heights))

    def average_pay_temperature_c(self) -> float:
        """Volume-weighted mean temperature over the whole pay."""
        pay_columns = np.nonzero(self.grid.layer_index >= 0)[0]
        volume = self.grid.volume_m3[:, pay_columns]
        return float(np.average(self.temperature_c[:, pay_columns], weights=volume))

    def heated_radius_m(self, threshold_k: float = 20.0) -> float:
        """Outermost radius where the pay is more than ``threshold_k`` above virgin."""
        pay_columns = np.nonzero(self.grid.layer_index >= 0)[0]
        excess = self.temperature_c[:, pay_columns] - self.initial_temp_c
        heated_rings = np.nonzero(np.max(excess, axis=1) > threshold_k)[0]
        if heated_rings.size == 0:
            return float(self.config.reservoir.wellbore_radius_m)
        return float(self.grid.radius_face_m[heated_rings[-1] + 1])

    def stored_heat_j(self) -> float:
        """Heat stored in the whole grid above the virgin temperature."""
        excess = self.temperature_c - self.initial_temp_c
        return float(np.sum(excess * self.volumetric_heat_capacity * self.grid.volume_m3))

    def begin_cycle(self, cycle_number: int) -> None:
        """Start a new cycle. The temperature field carries over as it stands.

        The production clock is reset because the well is shut in for injection
        and the soak, so the pressure transient restarts when it is reopened.
        """
        self.cycle_number = cycle_number
        self.production_day = 0.0


def steam_mass_rate_kg_per_s(rate_m3_per_day_cwe: float, water_density_kg_per_m3: float) -> float:
    """Mass rate of injected steam from the cold water equivalent volume rate."""
    return rate_m3_per_day_cwe * water_density_kg_per_m3 / SECONDS_PER_DAY


def sandface_steam_temperature_c(pressure_kpa: float) -> float:
    """Saturation temperature at the sandface injection pressure."""
    return saturation_temperature_c(min(max(pressure_kpa, 120.0), 19500.0))


def latent_heat_at_pressure_j_per_kg(pressure_kpa: float) -> float:
    """Latent heat at the sandface injection pressure."""
    return latent_heat_j_per_kg(min(max(pressure_kpa, 120.0), 19500.0))
