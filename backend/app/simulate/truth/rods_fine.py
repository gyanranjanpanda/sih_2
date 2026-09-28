"""Finer rod dynamics for the truth simulator.

Three deliberate differences from the twin's solver in ``app.twin.srp.wave``:

1. A finer spatial grid and a smaller time step, so the wave content the twin
   smooths over is present in the generated cards.
2. Nonlinear Coulomb plus viscous friction rather than a purely linear damping
   term. Real rod strings have a stick-slip component from the tubing contact
   that no linear damping law reproduces.
3. A different damping law. The twin scales a Gibbs factor with viscosity to a
   fitted exponent; here the viscous part comes straight from the annular
   solution and the Coulomb part is proportional to the local rod weight.

The interface matches the twin's solver so the same card machinery can read the
result, but nothing in the twin imports this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import SrpConfig
from app.core.errors import NumericalError
from app.core.numerics import require_finite
from app.core.units import STANDARD_GRAVITY_M_PER_S2
from app.twin.srp.kinematics import StrokeMotion
from app.twin.srp.wave import PumpBoundary, RodTaper

COULOMB_FRICTION_FRACTION = 0.012
"""Coulomb friction per unit rod weight, from tubing contact. A hidden property."""


@dataclass(frozen=True)
class FineRodSolution:
    """Result of one converged fine-grid rod run."""

    time_s: NDArray[np.float64]
    surface_position_m: NDArray[np.float64]
    surface_load_n: NDArray[np.float64]
    pump_position_m: NDArray[np.float64]
    pump_load_n: NDArray[np.float64]
    node_peak_load_n: NDArray[np.float64]
    node_min_load_n: NDArray[np.float64]

    @property
    def peak_polished_rod_load_n(self) -> float:
        """Maximum polished rod load over the cycle."""
        return float(np.max(self.surface_load_n))

    @property
    def minimum_polished_rod_load_n(self) -> float:
        """Minimum polished rod load over the cycle."""
        return float(np.min(self.surface_load_n))

    @property
    def pump_stroke_m(self) -> float:
        """Net plunger travel."""
        return float(np.ptp(self.pump_position_m))


def refine_taper(taper: RodTaper, refinement: int = 3) -> RodTaper:
    """Return the same rod string on a finer uniform grid."""
    nodes = (taper.node_count - 1) * refinement + 1
    depth = np.linspace(0.0, taper.length_m, nodes)
    area = np.interp(depth, taper.depth_m, taper.area_m2)
    section_index = np.round(
        np.interp(depth, taper.depth_m, taper.section_index.astype(float))
    ).astype(int)
    # Snap areas back onto the exact section values so the taper stays sharp.
    for index, section in enumerate(taper.sections):
        area[section_index == index] = section.area_m2
    return RodTaper(
        depth_m=depth,
        area_m2=area,
        section_index=section_index,
        sections=taper.sections,
        length_m=taper.length_m,
    )


def solve_fine(
    motion: StrokeMotion,
    taper: RodTaper,
    config: SrpConfig,
    pump: PumpBoundary,
    viscous_coefficient_n_s_per_m2: NDArray[np.float64],
    static_drag_n_per_m: NDArray[np.float64],
    fluid_density_kg_per_m3: float,
    coulomb_fraction: float = COULOMB_FRICTION_FRACTION,
    cycles: int = 6,
    courant: float = 0.8,
) -> FineRodSolution:
    """Solve the rod string with nonlinear friction on a fine grid.

    Equation, per unit length:

        rho A u_tt = d/dx (E A u_x) + (rho - rho_f) A g
                     - lambda(x) u_t - d0(x) - F_coulomb(x) sign(u_t)

    The Coulomb term is what the twin's linear damping cannot represent. It
    flattens the load transitions at the stroke ends and adds a small hysteresis
    to the card, which is visible in the generated data and is one of the things
    the twin has to live with.
    """
    acoustic = config.acoustic_velocity_m_per_s
    step_m = taper.step_m
    time_step_s = courant * step_m / acoustic
    steps_per_cycle = max(int(math.ceil(motion.cycle_time_s / time_step_s)), motion.time_s.size * 2)
    time_step_s = motion.cycle_time_s / steps_per_cycle
    if acoustic * time_step_s / step_m > 1.0:
        raise NumericalError("Courant condition violated in the fine rod solver.")

    node_count = taper.node_count
    areas = taper.area_m2
    face_areas = taper.face_areas_m2()
    youngs = config.steel_youngs_modulus_pa
    steel_density = config.steel_density_kg_per_m3
    mass_per_length = steel_density * areas
    buoyant_acceleration = (
        max(steel_density - fluid_density_kg_per_m3, 0.0) / steel_density
    ) * STANDARD_GRAVITY_M_PER_S2
    coulomb_force_n_per_m = coulomb_fraction * mass_per_length * STANDARD_GRAVITY_M_PER_S2

    cycle_time_grid = np.append(motion.time_s, motion.cycle_time_s)
    surface_cycle = np.append(motion.position_m, motion.position_m[0])
    surface_u_cycle = motion.stroke_length_m - surface_cycle
    total_steps = steps_per_cycle * cycles
    solver_time = np.arange(total_steps + 1) * time_step_s
    surface_u = np.interp(
        np.mod(solver_time, motion.cycle_time_s), cycle_time_grid, surface_u_cycle
    )

    segment_weight = steel_density * buoyant_acceleration * areas * step_m
    weight_below = np.concatenate([np.cumsum(segment_weight[::-1])[::-1][1:], [0.0]])
    static_tension = pump.fluid_load_n + weight_below
    stretch = np.concatenate(
        [[0.0], np.cumsum(static_tension[:-1] / (youngs * areas[:-1]) * step_m)]
    )
    previous = surface_u[0] + stretch
    current = previous.copy()

    damping = viscous_coefficient_n_s_per_m2 / np.maximum(mass_per_length, 1.0e-9)
    body_acceleration = buoyant_acceleration - static_drag_n_per_m / np.maximum(
        mass_per_length, 1.0e-9
    )
    damping_half = 0.5 * damping * time_step_s
    surface_damping_half = float(damping_half[0])
    wave_coefficient = youngs * time_step_s**2 / (steel_density * step_m**2)

    plunger_moving_up = False
    turnaround_top_u = float(current[-1])
    turnaround_bottom_u = float(current[-1])
    filtered_velocity = 0.0
    filter_weight = time_step_s / (motion.cycle_time_s / 20.0 + time_step_s)
    velocity_threshold = 0.02 * max(
        motion.peak_upstroke_speed_m_per_s, motion.peak_downstroke_speed_m_per_s, 1.0e-3
    )

    surface_load_history = np.zeros(total_steps + 1)
    pump_position_history = np.zeros(total_steps + 1)
    pump_load_history = np.zeros(total_steps + 1)
    node_peak = np.full(node_count, -np.inf)
    node_min = np.full(node_count, np.inf)

    for step in range(total_steps + 1):
        plunger_m = float(current[-1])
        raw_velocity = (current[-1] - previous[-1]) / time_step_s
        filtered_velocity += filter_weight * (raw_velocity - filtered_velocity)
        if plunger_moving_up and filtered_velocity > velocity_threshold:
            plunger_moving_up = False
            turnaround_top_u = plunger_m
        elif not plunger_moving_up and filtered_velocity < -velocity_threshold:
            plunger_moving_up = True
            turnaround_bottom_u = plunger_m
        pump_load = pump.load_n(
            travel_from_top_m=max(plunger_m - turnaround_top_u, 0.0),
            travel_from_bottom_m=max(turnaround_bottom_u - plunger_m, 0.0),
            moving_up=plunger_moving_up,
        )

        velocity = (current - previous) / time_step_s
        coulomb_acceleration = (
            -coulomb_force_n_per_m * np.tanh(velocity / 0.01) / np.maximum(mass_per_length, 1.0e-9)
        )

        next_surface_u = surface_u[min(step + 1, total_steps)]
        required_laplacian = (
            (1.0 + surface_damping_half) * next_surface_u
            - 2.0 * current[0]
            + (1.0 - surface_damping_half) * previous[0]
            - (body_acceleration[0] + coulomb_acceleration[0]) * time_step_s**2
        ) / wave_coefficient
        interior_flux = face_areas[0] * (current[1] - current[0])
        ghost_above = current[0] - (interior_flux - required_laplacian * areas[0]) / areas[0]
        surface_load = youngs * areas[0] * (current[1] - ghost_above) / (2.0 * step_m)

        surface_load_history[step] = surface_load
        pump_position_history[step] = plunger_m
        pump_load_history[step] = pump_load

        if step >= total_steps - steps_per_cycle:
            node_forces = np.zeros(node_count)
            node_forces[1:-1] = youngs * areas[1:-1] * (current[2:] - current[:-2]) / (2.0 * step_m)
            node_forces[0] = surface_load
            node_forces[-1] = pump_load
            node_peak = np.maximum(node_peak, node_forces)
            node_min = np.minimum(node_min, node_forces)

        if step == total_steps:
            break

        laplacian = np.zeros(node_count)
        laplacian[1:-1] = (
            face_areas[1:] * (current[2:] - current[1:-1])
            - face_areas[:-1] * (current[1:-1] - current[:-2])
        ) / areas[1:-1]
        ghost_last = current[-2] + 2.0 * step_m * pump_load / (youngs * areas[-1])
        laplacian[-1] = (
            face_areas[-1] * (ghost_last - current[-1])
            - face_areas[-1] * (current[-1] - current[-2])
        ) / areas[-1]

        nxt = (
            2.0 * current
            - (1.0 - damping_half) * previous
            + wave_coefficient * laplacian
            + (body_acceleration + coulomb_acceleration) * time_step_s**2
        ) / (1.0 + damping_half)
        nxt[0] = surface_u[step + 1]
        previous, current = current, nxt

    require_finite(surface_load_history, "fine rod surface load")
    window = slice(total_steps - steps_per_cycle, total_steps + 1)
    window_time = solver_time[window] - solver_time[total_steps - steps_per_cycle]
    surface_load_cycle = np.interp(motion.time_s, window_time, surface_load_history[window])
    pump_position_cycle = np.interp(motion.time_s, window_time, pump_position_history[window])
    pump_load_cycle = np.interp(motion.time_s, window_time, pump_load_history[window])
    pump_position_up = float(np.max(pump_position_cycle)) - pump_position_cycle

    return FineRodSolution(
        time_s=motion.time_s,
        surface_position_m=motion.position_m,
        surface_load_n=surface_load_cycle,
        pump_position_m=pump_position_up,
        pump_load_n=pump_load_cycle,
        node_peak_load_n=node_peak,
        node_min_load_n=node_min,
    )
