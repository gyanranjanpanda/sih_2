"""Rod string dynamics: the damped wave equation on a tapered string.

Two solvers live here and both solve the same equation

    u_tt = a^2 u_xx - c u_t

with x measured downward from the polished rod, u the axial displacement
positive downward, a the acoustic velocity in steel and c the Gibbs damping
coefficient.

* :func:`solve_forward` is a time-domain finite-volume solver. It takes the
  surface motion as a boundary condition and a nonlinear pump boundary that
  switches with the travelling valve, so it can produce fluid pound, gas
  interference and incomplete fillage. This is the predictive solver the twin
  and the optimizers use.
* :func:`downhole_from_surface` is the classical Gibbs frequency-domain
  solution. It takes a measured surface card and marches each harmonic down the
  taper with an exact transfer matrix to give the downhole pump card. This is
  the diagnostic direction.

A test runs the forward solver, feeds its surface card into the inverse solver
and checks that the downhole card comes back, which validates both.

Source: Gibbs, S. G. (1963), Predicting the behaviour of sucker-rod pumping
systems, J. Pet. Tech. 15(7); Gibbs, Rod Pumping: Modern Methods of Design,
Diagnosis and Surveillance, chapters 3 and 5; Everitt and Jennings (1992) for
the finite-difference treatment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import RodSection, SrpConfig
from app.core.errors import NumericalError, PhysicsDomainError
from app.core.numerics import require_finite, require_positive
from app.core.units import STANDARD_GRAVITY_M_PER_S2
from app.twin.srp.kinematics import StrokeMotion

MAX_COURANT_NUMBER = 0.95
"""Stability limit enforced on the explicit time-domain solver."""


@dataclass(frozen=True)
class RodTaper:
    """Node geometry of a tapered rod string on a uniform spatial grid."""

    depth_m: NDArray[np.float64]
    area_m2: NDArray[np.float64]
    section_index: NDArray[np.int_]
    sections: tuple[RodSection, ...]
    length_m: float

    @property
    def node_count(self) -> int:
        """Number of grid nodes."""
        return int(self.depth_m.size)

    @property
    def step_m(self) -> float:
        """Uniform node spacing."""
        return float(self.depth_m[1] - self.depth_m[0])

    def face_areas_m2(self) -> NDArray[np.float64]:
        """Areas at the faces between nodes, by arithmetic mean of the neighbours."""
        return 0.5 * (self.area_m2[:-1] + self.area_m2[1:])

    def buoyant_weight_n(
        self, fluid_density_kg_per_m3: float, steel_density_kg_per_m3: float
    ) -> float:
        """Total buoyant weight of the rod string in the produced fluid.

        Equation: W_b = sum over sections of (rho_steel - rho_fluid) g A L.
        Units: N. Densities in kg/m3, areas in m2, lengths in m.
        Assumption: the whole string is submerged, which holds once the annulus
        is covered above the pump.
        """
        effective_density = max(steel_density_kg_per_m3 - fluid_density_kg_per_m3, 0.0)
        return sum(
            effective_density * STANDARD_GRAVITY_M_PER_S2 * section.area_m2 * section.length_m
            for section in self.sections
        )

    def dry_weight_n(self, steel_density_kg_per_m3: float) -> float:
        """Total dry weight of the rod string in air."""
        return sum(
            steel_density_kg_per_m3 * STANDARD_GRAVITY_M_PER_S2 * section.area_m2 * section.length_m
            for section in self.sections
        )

    def section_boundaries_m(self) -> list[float]:
        """Cumulative depths at which the taper changes."""
        depths: list[float] = []
        cursor = 0.0
        for section in self.sections:
            cursor += section.length_m
            depths.append(cursor)
        return depths


def build_taper(config: SrpConfig, nodes_per_section: int | None = None) -> RodTaper:
    """Discretise the configured rod string onto a uniform grid.

    A uniform spacing across the whole string keeps one stable time step for the
    explicit solver. Each node is assigned the area of the section it falls in,
    and the finite-volume face areas handle the impedance change at the taper.
    """
    per_section = nodes_per_section or config.wave_grid_nodes_per_section
    total_nodes = max(per_section * len(config.rod_sections), 12)
    length_m = config.total_rod_length_m
    depth_m = np.linspace(0.0, length_m, total_nodes)
    area_m2 = np.zeros_like(depth_m)
    section_index = np.zeros(depth_m.size, dtype=int)
    cursor = 0.0
    boundaries: list[tuple[float, float, int]] = []
    for index, section in enumerate(config.rod_sections):
        boundaries.append((cursor, cursor + section.length_m, index))
        cursor += section.length_m
    for node, depth in enumerate(depth_m):
        chosen = len(config.rod_sections) - 1
        for lower, upper, index in boundaries:
            if lower <= depth < upper:
                chosen = index
                break
        area_m2[node] = config.rod_sections[chosen].area_m2
        section_index[node] = chosen
    return RodTaper(
        depth_m=depth_m,
        area_m2=area_m2,
        section_index=section_index,
        sections=tuple(config.rod_sections),
        length_m=length_m,
    )


def gibbs_damping_coefficient_per_s(
    damping_factor_dimensionless: float, acoustic_velocity_m_per_s: float, length_m: float
) -> float:
    """Gibbs damping coefficient from the dimensionless damping factor.

    Equation: c = pi v a / (2 L).
    Units: 1/s. Acoustic velocity in m/s, length in m.
    Assumptions: the damping is uniform along the string and represents the
    viscous drag of the produced fluid on the rods. The dimensionless factor v
    is typically 0.05 to 0.5 and is raised for heavy oil by
    :func:`viscous_damping_factor`.
    Source: Gibbs, Rod Pumping, chapter 3.
    """
    require_positive(length_m, "length_m")
    return math.pi * damping_factor_dimensionless * acoustic_velocity_m_per_s / (2.0 * length_m)


def viscous_damping_factor(
    config: SrpConfig, average_viscosity_pa_s: float, reference_viscosity_pa_s: float = 0.05
) -> float:
    """Dimensionless damping factor scaled by the viscosity the rods move through.

    Equation: v = v0 (mu / mu_ref)^p, bounded above by the configured maximum.
    Units: dimensionless. Viscosities in Pa.s.
    Assumptions: the configured base factor v0 applies at the reference
    viscosity of 50 cP. The exponent p is a calibration parameter, not a
    measured constant, and assimilation adjusts it per well. Bounding the result
    keeps the explicit solver stable when the well is very cold.
    Source: the functional form follows the brief; Gibbs (1963) notes that the
    damping factor rises strongly with fluid viscosity but gives no correlation.
    """
    require_positive(reference_viscosity_pa_s, "reference_viscosity_pa_s")
    ratio = max(average_viscosity_pa_s, 1.0e-6) / reference_viscosity_pa_s
    factor = config.damping_factor_dimensionless * ratio**config.damping_viscosity_exponent
    return float(min(max(factor, 1.0e-3), config.damping_factor_max))


@dataclass(frozen=True)
class DragProfile:
    """Node-wise linearisation of the viscous drag on the rod string.

    The annular drag solution in :mod:`app.twin.srp.floating` is exactly linear
    in the rod velocity for a fixed net throughput, so it can be written as

        drag_up(x, V) = -lambda(x) V + d0(x)

    with V the rod velocity measured upward. Substituting into the momentum
    equation for the downward displacement u turns lambda into a node-varying
    damping coefficient and d0 into a node-varying body force.

    Using this in place of a single Gibbs damping factor matters for heavy oil.
    The drag is one to two orders of magnitude larger at the cold top of the
    string than at the hot pump, and a single scalar cannot represent that. It
    also removes the double counting that comes from having both a Gibbs
    damping term and a separate explicit drag calculation.

    Attributes:
        linear_coefficient_n_s_per_m2: lambda at each node, in N.s/m per metre
            of rod, so the units are N.s/m2.
        static_force_n_per_m: d0 at each node, the drag present at zero rod
            velocity because the produced fluid is flowing past, in N/m.
    """

    linear_coefficient_n_s_per_m2: NDArray[np.float64]
    static_force_n_per_m: NDArray[np.float64]

    def equivalent_damping_per_s(self, taper: RodTaper, steel_density_kg_per_m3: float) -> float:
        """Mass-weighted mean damping coefficient, for reporting and for the inverse solution.

        The Gibbs inverse solution needs one scalar damping value. Averaging the
        node-wise coefficients by rod mass gives the value that dissipates the
        same energy for uniform motion.
        """
        mass_per_length = steel_density_kg_per_m3 * taper.area_m2
        node_damping = self.linear_coefficient_n_s_per_m2 / np.maximum(mass_per_length, 1.0e-9)
        return float(np.average(node_damping, weights=mass_per_length))


@dataclass(frozen=True)
class PumpBoundary:
    """Nonlinear downhole boundary condition set by the pump valves.

    Attributes:
        fluid_load_n: Load the plunger carries with the travelling valve closed,
            the plunger area times the pressure difference across it.
        fillage_frac: Fraction of the pump barrel filled with liquid. Below 1
            the travelling valve stays shut for part of the downstroke and the
            load drops abruptly when the plunger reaches liquid, which is fluid
            pound.
        gas_interference_frac: Fraction of the barrel occupied by gas. Gas
            compresses, so the load falls gradually rather than abruptly.
        plunger_friction_n: Mechanical friction between plunger and barrel,
            always opposing motion.
        stroke_length_m: Surface stroke length, used to scale the fillage
            travel on the downstroke.
        tagging_contact_travel_m: Downward plunger travel at which the plunger
            strikes the bottom of the barrel. Zero disables tagging. It is
            expressed as a travel rather than a clearance below the stroke so it
            does not depend on the surface stroke, which differs from the
            plunger stroke by the rod stretch.
        sticking_load_n: Extra load from a sticking plunger, applied on the
            upstroke only. Zero disables it.
        valve_transition_m: Plunger travel over which the load transfers when a
            valve changes state. This is not a numerical smoothing knob: it is
            the travel needed to compress or expand the liquid in the barrel
            from intake to discharge pressure, plus the valve ball travel. For
            a 2.3 m plunger stroke, an oil compressibility near 7e-10 per Pa and
            a 10 MPa pressure difference, the compressibility part alone is
            about 16 mm, so 20 mm is the default.
    """

    fluid_load_n: float
    fillage_frac: float = 1.0
    gas_interference_frac: float = 0.0
    plunger_friction_n: float = 0.0
    stroke_length_m: float = 2.54
    tagging_contact_travel_m: float = 0.0
    sticking_load_n: float = 0.0
    valve_transition_m: float = 0.02

    @staticmethod
    def _smoothstep(fraction: float) -> float:
        """Cubic smoothstep, zero slope at both ends."""
        clamped = min(max(fraction, 0.0), 1.0)
        return clamped * clamped * (3.0 - 2.0 * clamped)

    def load_n(
        self, travel_from_top_m: float, travel_from_bottom_m: float, moving_up: bool
    ) -> float:
        """Rod tension at the pump for the current plunger state.

        Args:
            travel_from_top_m: Downward plunger travel since the last turnaround
                at the top of the plunger stroke.
            travel_from_bottom_m: Upward plunger travel since the last
                turnaround at the bottom of the plunger stroke.
            moving_up: Whether the plunger is on the upstroke. The caller
                determines this from a filtered velocity with hysteresis, not
                from the raw finite-difference velocity, because the rod string
                rings numerically at the pump node and an unfiltered sign test
                makes the travelling valve chatter.
        """
        transition_m = max(self.valve_transition_m, 1.0e-4)
        friction = self.plunger_friction_n * (1.0 if moving_up else -1.0)

        if moving_up:
            # The standing valve opens and the travelling valve closes only
            # once the barrel pressure has risen to the discharge pressure.
            pickup = self._smoothstep(travel_from_bottom_m / transition_m)
            return max(self.fluid_load_n * pickup + friction + self.sticking_load_n, 0.0)

        # Downstroke. The travelling valve cannot open until the plunger reaches
        # liquid, so an underfilled pump holds the load and then drops it
        # abruptly: that is fluid pound.
        void_travel_m = (1.0 - self.fillage_frac) * self.stroke_length_m
        gas_travel_m = self.gas_interference_frac * self.stroke_length_m
        beyond_m = travel_from_top_m - void_travel_m

        if beyond_m <= 0.0:
            carried = self.fluid_load_n
        elif gas_travel_m > 0.0 and beyond_m < gas_travel_m:
            # Gas compresses, so the load falls gradually rather than abruptly.
            carried = self.fluid_load_n * (1.0 - beyond_m / gas_travel_m)
        else:
            release_from_m = beyond_m - (gas_travel_m if gas_travel_m > 0.0 else 0.0)
            carried = self.fluid_load_n * (1.0 - self._smoothstep(release_from_m / transition_m))

        tagging_load = 0.0
        if self.tagging_contact_travel_m > 0.0:
            overtravel_m = travel_from_top_m - self.tagging_contact_travel_m
            if overtravel_m > 0.0:
                # A stiff contact spring, capped so the solver stays finite. The
                # sign is negative because the barrel pushes up on the plunger,
                # putting the bottom of the rod string into compression.
                tagging_load = -min(
                    overtravel_m / max(0.5 * transition_m, 1.0e-4) * self.fluid_load_n,
                    4.0 * max(self.fluid_load_n, 1.0e3),
                )
        return max(carried + friction, -abs(self.fluid_load_n)) + tagging_load


@dataclass(frozen=True)
class WaveSolution:
    """Result of one converged rod dynamics run over a single cycle."""

    time_s: NDArray[np.float64]
    surface_position_m: NDArray[np.float64]
    surface_load_n: NDArray[np.float64]
    pump_position_m: NDArray[np.float64]
    pump_load_n: NDArray[np.float64]
    node_depth_m: NDArray[np.float64]
    node_peak_load_n: NDArray[np.float64]
    node_min_load_n: NDArray[np.float64]
    damping_coefficient_per_s: float
    courant_number: float
    solver_time_step_s: float

    @property
    def peak_polished_rod_load_n(self) -> float:
        """Maximum polished rod load over the cycle."""
        return float(np.max(self.surface_load_n))

    @property
    def minimum_polished_rod_load_n(self) -> float:
        """Minimum polished rod load over the cycle."""
        return float(np.min(self.surface_load_n))

    @property
    def load_range_n(self) -> float:
        """Peak minus minimum polished rod load."""
        return self.peak_polished_rod_load_n - self.minimum_polished_rod_load_n

    @property
    def pump_stroke_m(self) -> float:
        """Net plunger travel over the cycle."""
        return float(np.max(self.pump_position_m) - np.min(self.pump_position_m))


def solve_forward(
    motion: StrokeMotion,
    taper: RodTaper,
    config: SrpConfig,
    pump: PumpBoundary,
    damping_factor_dimensionless: float,
    fluid_density_kg_per_m3: float,
    cycles: int = 4,
    drag: DragProfile | None = None,
) -> WaveSolution:
    """Time-domain solution of the damped wave equation with a nonlinear pump.

    The surface position is imposed from the unit kinematics. The pump end
    carries the valve-dependent load. The solver runs several cycles from rest
    and returns the last one, by which point the transient from the initial
    condition has decayed.

    Equation, in finite-volume form on a uniform grid:

        rho A_i u_tt = d/dx (E A u_x) + (rho - rho_f) A_i g - c rho A_i u_t

    Args:
        motion: Polished rod motion over one cycle.
        taper: Discretised rod string.
        config: Rod and unit settings.
        pump: Downhole boundary condition.
        damping_factor_dimensionless: Gibbs damping factor v. Used only when
            ``drag`` is not supplied.
        fluid_density_kg_per_m3: Density of the fluid the rods are submerged in.
        cycles: Number of cycles to run before the result is taken.
        drag: Node-wise drag linearisation. When supplied it replaces the single
            Gibbs damping factor, so the card reflects the actual viscosity
            profile along the string and rod float appears in the surface card
            rather than only in the separate mechanical check.

    Raises:
        NumericalError: if the Courant condition cannot be satisfied or the
            solution becomes non-finite.
    """
    acoustic_velocity = config.acoustic_velocity_m_per_s
    step_m = taper.step_m
    require_positive(step_m, "rod grid step")
    if cycles < 1:
        raise PhysicsDomainError("At least one cycle must be solved.", cycles=cycles)

    stable_step_s = MAX_COURANT_NUMBER * step_m / acoustic_velocity
    steps_per_cycle = int(math.ceil(motion.cycle_time_s / stable_step_s))
    steps_per_cycle = max(steps_per_cycle, motion.time_s.size * 2)
    time_step_s = motion.cycle_time_s / steps_per_cycle
    courant = acoustic_velocity * time_step_s / step_m
    if courant > 1.0:
        raise NumericalError(
            "Courant condition violated in the rod wave solver.",
            courant_number=courant,
            time_step_s=time_step_s,
            grid_step_m=step_m,
        )

    node_count = taper.node_count
    areas = taper.area_m2
    mass_per_length = config.steel_density_kg_per_m3 * areas
    if drag is None:
        scalar_damping_per_s = gibbs_damping_coefficient_per_s(
            damping_factor_dimensionless, acoustic_velocity, taper.length_m
        )
        node_damping_per_s = np.full(node_count, scalar_damping_per_s)
        drag_acceleration_m_per_s2 = np.zeros(node_count)
        reported_damping_per_s = scalar_damping_per_s
    else:
        if drag.linear_coefficient_n_s_per_m2.size != node_count:
            raise PhysicsDomainError(
                "Drag profile must have one value per rod node.",
                expected=node_count,
                received=int(drag.linear_coefficient_n_s_per_m2.size),
            )
        node_damping_per_s = np.maximum(
            drag.linear_coefficient_n_s_per_m2 / np.maximum(mass_per_length, 1.0e-9), 0.0
        )
        # d0 acts upward on the rods, so it decelerates the downward coordinate.
        drag_acceleration_m_per_s2 = -drag.static_force_n_per_m / np.maximum(
            mass_per_length, 1.0e-9
        )
        reported_damping_per_s = drag.equivalent_damping_per_s(
            taper, config.steel_density_kg_per_m3
        )
    face_areas = taper.face_areas_m2()
    youngs = config.steel_youngs_modulus_pa
    steel_density = config.steel_density_kg_per_m3
    buoyant_acceleration = (
        max(steel_density - fluid_density_kg_per_m3, 0.0) / steel_density
    ) * STANDARD_GRAVITY_M_PER_S2

    # Surface boundary: u is positive downward, so it is the stroke length minus
    # the polished rod position measured up from the bottom of the stroke.
    cycle_time_grid = np.append(motion.time_s, motion.cycle_time_s)
    surface_cycle = np.append(motion.position_m, motion.position_m[0])
    surface_u_cycle = motion.stroke_length_m - surface_cycle

    total_steps = steps_per_cycle * cycles
    solver_time = np.arange(total_steps + 1) * time_step_s
    phase_time = np.mod(solver_time, motion.cycle_time_s)
    surface_u = np.interp(phase_time, cycle_time_grid, surface_u_cycle)

    # Static initial condition: rod stretch under its own buoyant weight plus
    # the standing fluid load. Starting from the static solution removes most of
    # the start-up transient.
    segment_weight_n = steel_density * buoyant_acceleration * areas * step_m
    weight_below_n = np.concatenate([np.cumsum(segment_weight_n[::-1])[::-1][1:], [0.0]])
    static_tension_n = pump.fluid_load_n + weight_below_n
    stretch_m = np.concatenate(
        [[0.0], np.cumsum(static_tension_n[:-1] / (youngs * areas[:-1]) * step_m)]
    )
    previous = surface_u[0] + stretch_m
    current = previous.copy()

    # Valve state machine. The plunger velocity is low-pass filtered over about
    # one twentieth of a cycle so that the numerical ringing of the rod string
    # at the pump node does not flip the travelling valve.
    plunger_moving_up = False
    turnaround_top_u_m = float(current[-1])
    turnaround_bottom_u_m = float(current[-1])
    filtered_velocity_m_per_s = 0.0
    filter_time_constant_s = max(motion.cycle_time_s / 20.0, 4.0 * time_step_s)
    filter_weight = time_step_s / (filter_time_constant_s + time_step_s)
    velocity_threshold_m_per_s = 0.02 * max(
        motion.peak_upstroke_speed_m_per_s, motion.peak_downstroke_speed_m_per_s, 1.0e-3
    )

    surface_load_history = np.zeros(total_steps + 1)
    pump_position_history = np.zeros(total_steps + 1)
    pump_load_history = np.zeros(total_steps + 1)
    node_peak = np.full(node_count, -np.inf)
    node_min = np.full(node_count, np.inf)

    damping_half = 0.5 * node_damping_per_s * time_step_s
    surface_damping_half = float(damping_half[0])
    body_acceleration = buoyant_acceleration + drag_acceleration_m_per_s2
    wave_coefficient = youngs * time_step_s**2 / (steel_density * step_m**2)

    for step in range(total_steps + 1):
        plunger_m = float(current[-1])
        raw_velocity = (current[-1] - previous[-1]) / time_step_s
        filtered_velocity_m_per_s += filter_weight * (raw_velocity - filtered_velocity_m_per_s)
        if plunger_moving_up and filtered_velocity_m_per_s > velocity_threshold_m_per_s:
            plunger_moving_up = False
            turnaround_top_u_m = plunger_m
        elif not plunger_moving_up and filtered_velocity_m_per_s < -velocity_threshold_m_per_s:
            plunger_moving_up = True
            turnaround_bottom_u_m = plunger_m
        pump_load = pump.load_n(
            travel_from_top_m=max(plunger_m - turnaround_top_u_m, 0.0),
            travel_from_bottom_m=max(turnaround_bottom_u_m - plunger_m, 0.0),
            moving_up=plunger_moving_up,
        )

        # Surface reaction force. The displacement at node zero is imposed, so
        # the momentum balance at that node is solved for the fictitious node
        # just above the polished rod, and the reaction is the tension there.
        next_surface_u = surface_u[min(step + 1, total_steps)]
        required_laplacian = (
            (1.0 + surface_damping_half) * next_surface_u
            - 2.0 * current[0]
            + (1.0 - surface_damping_half) * previous[0]
            - body_acceleration[0] * time_step_s**2
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
            + body_acceleration * time_step_s**2
        ) / (1.0 + damping_half)
        nxt[0] = surface_u[step + 1]
        previous, current = current, nxt

    require_finite(surface_load_history, "surface load history")

    last = slice(total_steps - steps_per_cycle, total_steps + 1)
    last_time = solver_time[last] - solver_time[total_steps - steps_per_cycle]
    sample_time = motion.time_s
    surface_load_cycle = np.interp(sample_time, last_time, surface_load_history[last])
    pump_position_cycle = np.interp(sample_time, last_time, pump_position_history[last])
    pump_load_cycle = np.interp(sample_time, last_time, pump_load_history[last])

    # Report the plunger position measured upward from its lowest point, so it
    # is directly comparable with the surface position.
    pump_position_up_m = float(np.max(pump_position_cycle)) - pump_position_cycle

    return WaveSolution(
        time_s=sample_time,
        surface_position_m=motion.position_m,
        surface_load_n=surface_load_cycle,
        pump_position_m=pump_position_up_m,
        pump_load_n=pump_load_cycle,
        node_depth_m=taper.depth_m,
        node_peak_load_n=node_peak,
        node_min_load_n=node_min,
        damping_coefficient_per_s=float(reported_damping_per_s),
        courant_number=courant,
        solver_time_step_s=time_step_s,
    )


def _harmonic_wavenumbers(
    harmonics: NDArray[np.float64],
    angular_frequency: float,
    acoustic_velocity_m_per_s: float,
    damping_per_s: float,
) -> NDArray[np.complex128]:
    """Complex wavenumber k(n) for each harmonic of the damped wave equation.

    Equation: substituting u = U(x) exp(i n omega t) gives
        U'' = k^2 U with k^2 = (-(n omega)^2 + i c n omega) / a^2.
    """
    omega_n = harmonics * angular_frequency
    k_squared = (-(omega_n**2) + 1j * damping_per_s * omega_n) / acoustic_velocity_m_per_s**2
    return np.sqrt(k_squared.astype(np.complex128))


def _transfer_section(
    displacement: NDArray[np.complex128],
    force: NDArray[np.complex128],
    wavenumber: NDArray[np.complex128],
    length_m: float,
    area_m2: float,
    youngs_modulus_pa: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """March one uniform section with the exact transfer matrix.

        [u(L)]   [ cosh(kL)           sinh(kL) / (k E A) ] [u(0)]
        [F(L)] = [ E A k sinh(kL)     cosh(kL)           ] [F(0)]

    The static harmonic, where k is zero, reduces to the rod stretch relation
    u(L) = u(0) + F L / (E A), which is handled explicitly.
    """
    stiffness = youngs_modulus_pa * area_m2
    argument = wavenumber * length_m
    cosh_term = np.cosh(argument)
    sinh_term = np.sinh(argument)
    small = np.abs(argument) < 1.0e-9
    sinh_over_k = np.where(small, length_m, sinh_term / np.where(small, 1.0, wavenumber))
    k_sinh = np.where(small, 0.0, wavenumber * sinh_term)
    new_displacement = displacement * cosh_term + force * sinh_over_k / stiffness
    new_force = displacement * stiffness * k_sinh + force * cosh_term
    return new_displacement, new_force


def downhole_from_surface(
    surface_position_m: NDArray[np.float64],
    surface_load_n: NDArray[np.float64],
    cycle_time_s: float,
    taper: RodTaper,
    config: SrpConfig,
    damping_factor_dimensionless: float,
    fluid_density_kg_per_m3: float,
    retained_harmonics: int = 24,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gibbs inverse solution: downhole pump card from a measured surface card.

    Each Fourier harmonic of the measured surface displacement and load is
    marched down the taper with the exact transfer matrix of the damped wave
    equation. Only the first ``retained_harmonics`` harmonics are kept, which
    is the standard regularisation: the higher harmonics grow exponentially
    with depth and would amplify measurement noise without adding signal.

    The buoyant rod weight is removed from the static component before the
    march and is not added back, because the pump card is the load the plunger
    sees, not the load the polished rod sees.

    Args:
        surface_position_m: Polished rod position over one cycle, measured up.
        surface_load_n: Polished rod load over the same cycle.
        cycle_time_s: Duration of the cycle.
        taper: Discretised rod string.
        config: Rod and unit settings.
        damping_factor_dimensionless: Gibbs damping factor v.
        fluid_density_kg_per_m3: Density of the fluid the rods are submerged in.
        retained_harmonics: Number of Fourier harmonics kept.

    Returns:
        Plunger position measured up from its lowest point, and pump load.
    """
    samples = surface_position_m.size
    if surface_load_n.size != samples:
        raise PhysicsDomainError("Position and load series must have the same length.")
    if samples < 8:
        raise PhysicsDomainError("At least eight samples are needed for the inverse solution.")
    require_positive(cycle_time_s, "cycle_time_s")

    angular_frequency = 2.0 * math.pi / cycle_time_s
    damping_per_s = gibbs_damping_coefficient_per_s(
        damping_factor_dimensionless, config.acoustic_velocity_m_per_s, taper.length_m
    )
    # Displacement positive downward, to match the forward solver.
    displacement_series = -surface_position_m

    keep = min(retained_harmonics, samples // 2)
    position_spectrum = np.fft.rfft(displacement_series)
    load_spectrum = np.fft.rfft(surface_load_n)
    position_spectrum[keep + 1 :] = 0.0
    load_spectrum[keep + 1 :] = 0.0

    harmonics = np.arange(position_spectrum.size, dtype=float)
    wavenumbers = _harmonic_wavenumbers(
        harmonics, angular_frequency, config.acoustic_velocity_m_per_s, damping_per_s
    )

    displacement = position_spectrum.astype(np.complex128)
    force = load_spectrum.astype(np.complex128)

    steel_density = config.steel_density_kg_per_m3
    buoyant_unit_weight = (
        max(steel_density - fluid_density_kg_per_m3, 0.0) * STANDARD_GRAVITY_M_PER_S2
    )
    for section in taper.sections:
        # The transfer matrix carries every harmonic. Its static harmonic gives
        # the rod stretch; the buoyant weight of the section is then removed
        # from that static component so the result is the load the plunger sees.
        displacement, force = _transfer_section(
            displacement,
            force,
            wavenumbers,
            section.length_m,
            section.area_m2,
            config.steel_youngs_modulus_pa,
        )
        # Remove the buoyant weight of this section from the static harmonic.
        force[0] = force[0] - buoyant_unit_weight * section.area_m2 * section.length_m * samples

    pump_displacement = np.fft.irfft(displacement, n=samples)
    pump_load = np.fft.irfft(force, n=samples)
    pump_position_up_m = float(np.max(-pump_displacement)) - (-pump_displacement)
    require_finite(pump_load, "inverse solution pump load")
    return np.asarray(pump_position_up_m, dtype=float), np.asarray(pump_load, dtype=float)
