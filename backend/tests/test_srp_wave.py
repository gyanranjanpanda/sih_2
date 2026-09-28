"""Rod dynamics tests: analytic wave case, stability, taper and the inverse solution."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from app.core.config import FieldConfig, RodSection, SrpConfig
from app.core.errors import NumericalError, PhysicsDomainError
from app.twin.srp.floating import build_drag_profile
from app.twin.srp.kinematics import ConventionalKinematics, SpeedProfile
from app.twin.srp.wave import (
    MAX_COURANT_NUMBER,
    PumpBoundary,
    RodTaper,
    build_taper,
    downhole_from_surface,
    gibbs_damping_coefficient_per_s,
    solve_forward,
    viscous_damping_factor,
)


def _uniform_taper(length_m: float, diameter_m: float, nodes: int) -> RodTaper:
    section = RodSection(
        diameter_m=diameter_m,
        length_m=length_m,
        grade="D",
        minimum_tensile_strength_pa=7.93e8,
    )
    depth = np.linspace(0.0, length_m, nodes)
    area = np.full(nodes, section.area_m2)
    return RodTaper(
        depth_m=depth,
        area_m2=area,
        section_index=np.zeros(nodes, dtype=int),
        sections=(section,),
        length_m=length_m,
    )


def test_taper_covers_the_string(taper: RodTaper, field_config: FieldConfig) -> None:
    assert taper.length_m == pytest.approx(field_config.srp.total_rod_length_m)
    assert taper.depth_m[0] == 0.0
    assert taper.depth_m[-1] == pytest.approx(taper.length_m)
    assert len(set(np.round(taper.area_m2, 9))) == len(field_config.srp.rod_sections)


def test_taper_areas_decrease_with_depth(taper: RodTaper) -> None:
    """A correctly designed taper is heaviest at the top."""
    assert taper.area_m2[0] >= taper.area_m2[-1]


def test_buoyant_weight_is_below_dry_weight(taper: RodTaper, field_config: FieldConfig) -> None:
    dry = taper.dry_weight_n(field_config.srp.steel_density_kg_per_m3)
    buoyant = taper.buoyant_weight_n(900.0, field_config.srp.steel_density_kg_per_m3)
    assert 0.0 < buoyant < dry
    assert buoyant == pytest.approx(dry * (7850.0 - 900.0) / 7850.0, rel=1e-6)


def test_gibbs_damping_matches_its_definition() -> None:
    assert gibbs_damping_coefficient_per_s(0.25, 4990.0, 1100.0) == pytest.approx(
        math.pi * 0.25 * 4990.0 / (2.0 * 1100.0)
    )


def test_viscous_damping_factor_rises_with_viscosity(field_config: FieldConfig) -> None:
    low = viscous_damping_factor(field_config.srp, 0.005)
    high = viscous_damping_factor(field_config.srp, 5.0)
    assert high > low
    assert high <= field_config.srp.damping_factor_max


def _uniform_rod_config(
    length_m: float, acoustic_m_per_s: float, section: RodSection, nodes: int, spm: float
) -> SrpConfig:
    """A minimal configuration describing a bare uniform rod, for analytic tests."""
    return SrpConfig(
        unit_type="conventional",
        pump_depth_m=length_m,
        plunger_diameter_m=0.04445,
        pump_clearance_m=0.000127,
        stroke_length_m=1.0,
        stroke_length_options_m=[1.0],
        spm_setpoint=spm,
        spm_min=0.5,
        spm_max=29.0,
        spm_rate_limit_per_step=1.0,
        acoustic_velocity_m_per_s=acoustic_m_per_s,
        steel_density_kg_per_m3=7850.0,
        steel_youngs_modulus_pa=7850.0 * acoustic_m_per_s**2,
        damping_factor_dimensionless=0.3,
        damping_viscosity_exponent=0.0,
        damping_factor_max=1.0,
        wave_grid_nodes_per_section=nodes,
        wave_time_steps_per_cycle=2000,
        rod_sections=[section],
        service_factor_dimensionless=1.0,
        structural_load_rating_n=9.0e6,
        gearbox_torque_rating_n_m=9.0e6,
        counterbalance_moment_n_m=0.0,
        pitman_length_m=3.05,
        crank_to_saddle_distance_m=3.66,
        walking_beam_rear_arm_m=2.44,
        walking_beam_front_arm_m=3.05,
        motor_rated_power_w=56000.0,
        motor_efficiency_peak_frac=0.92,
        gearbox_efficiency_frac=0.9,
        belt_efficiency_frac=0.96,
        vfd_frequency_min_hz=15.0,
        vfd_frequency_max_hz=60.0,
        vfd_base_frequency_hz=50.0,
        hydraulic_max_pressure_kpa=21000.0,
        hydraulic_max_flow_m3_per_s=0.0022,
        hydraulic_cylinder_area_m2=0.0182,
        hydraulic_upstroke_speed_frac=1.0,
        hydraulic_downstroke_speed_frac=1.0,
        hydraulic_dwell_s=0.0,
        float_margin_minimum_frac=0.15,
        minimum_polished_rod_load_n=1000.0,
        minimum_fillage_frac=0.7,
        gas_interference_frac=0.0,
    )


def _sinusoidal_motion(amplitude_m: float, frequency_hz: float, samples: int):  # type: ignore[no-untyped-def]
    """Pure sinusoidal polished rod motion, for analytic comparisons."""
    from app.twin.srp.kinematics import StrokeMotion

    omega = 2.0 * math.pi * frequency_hz
    cycle_time_s = 1.0 / frequency_hz
    time_s = np.linspace(0.0, cycle_time_s, samples, endpoint=False)
    position_m = amplitude_m * (1.0 - np.cos(omega * time_s))
    return StrokeMotion(
        time_s=time_s,
        position_m=position_m,
        velocity_m_per_s=np.gradient(position_m, time_s[1] - time_s[0]),
        acceleration_m_per_s2=np.zeros(samples),
        crank_angle_rad=omega * time_s,
        torque_factor_m=np.zeros(samples),
        cycle_time_s=cycle_time_s,
        stroke_length_m=2.0 * amplitude_m,
        unit_type="conventional",
    )


def test_uniform_rod_matches_the_analytic_damped_standing_wave() -> None:
    """The solver must reproduce the exact solution of the damped wave equation.

    For a uniform rod of length L driven at the surface with a free lower end,
    substituting u = U exp(i omega t) into u_tt = a^2 u_xx - c u_t gives
    U'' = k^2 U with the complex wavenumber k^2 = (-omega^2 + i c omega) / a^2.
    Imposing zero force at the lower end on the exact transfer matrix gives the
    surface reaction in closed form:

        F(0) = -E A k tanh(k L) u(0).

    At low frequency this reduces to -rho A L omega^2 u(0) plus the damping
    term, which is the expected inertial reaction: pushing the top down while
    the rod below lags puts the top of the string into compression.

    The test drives the surface with a pure sinusoid, extracts the fundamental
    harmonic of the computed surface load, and compares it with that closed
    form in both magnitude and phase. Damping is set to a realistic 0.3 so the
    start-up transient has decayed by the time the reported cycle is taken; an
    undamped rod would keep its start-up transient forever and could not be
    compared with a steady-state solution.
    """
    length_m = 900.0
    acoustic = 4990.0
    frequency_hz = 0.20
    omega = 2.0 * math.pi * frequency_hz
    amplitude_m = 0.5
    damping_factor = 0.3
    samples = 720
    nodes = 181

    section = RodSection(
        diameter_m=0.02223, length_m=length_m, grade="D", minimum_tensile_strength_pa=7.93e8
    )
    taper = _uniform_taper(length_m, section.diameter_m, nodes)
    config = _uniform_rod_config(length_m, acoustic, section, nodes, frequency_hz * 60.0)
    motion = _sinusoidal_motion(amplitude_m, frequency_hz, samples)

    solution = solve_forward(
        motion=motion,
        taper=taper,
        config=config,
        pump=PumpBoundary(fluid_load_n=0.0, fillage_frac=1.0, stroke_length_m=1.0),
        damping_factor_dimensionless=damping_factor,
        fluid_density_kg_per_m3=7850.0,  # neutral buoyancy removes the gravity term
        cycles=8,
    )

    damping_per_s = gibbs_damping_coefficient_per_s(damping_factor, acoustic, length_m)
    wavenumber = np.sqrt(complex(-(omega**2), damping_per_s * omega)) / acoustic
    stiffness = config.steel_youngs_modulus_pa * section.area_m2
    analytic_gain = -stiffness * wavenumber * np.tanh(wavenumber * length_m)

    # Surface displacement is positive downward, so it is the negative of the
    # polished rod position measured up.
    displacement_spectrum = np.fft.rfft(-motion.position_m)
    load_spectrum = np.fft.rfft(solution.surface_load_n)
    numerical_gain = load_spectrum[1] / displacement_spectrum[1]

    magnitude_error = abs(abs(numerical_gain) - abs(analytic_gain)) / abs(analytic_gain)
    phase_error = abs(np.angle(numerical_gain / analytic_gain))
    assert magnitude_error < 0.03, f"gain magnitude error {magnitude_error:.4f}"
    assert phase_error < 0.05, f"gain phase error {phase_error:.4f} rad"


def test_energy_does_not_grow_without_an_input() -> None:
    """A rod released from a stretched state must lose energy, never gain it."""
    length_m = 900.0
    acoustic = 4990.0
    samples = 720
    nodes = 91
    section = RodSection(
        diameter_m=0.02223, length_m=length_m, grade="D", minimum_tensile_strength_pa=7.93e8
    )
    taper = _uniform_taper(length_m, section.diameter_m, nodes)
    config = _uniform_rod_config(length_m, acoustic, section, nodes, 6.0)
    # Hold the surface still: the only energy present is the start-up transient.
    from app.twin.srp.kinematics import StrokeMotion

    time_s = np.linspace(0.0, 10.0, samples, endpoint=False)
    motion = StrokeMotion(
        time_s=time_s,
        position_m=np.zeros(samples),
        velocity_m_per_s=np.zeros(samples),
        acceleration_m_per_s2=np.zeros(samples),
        crank_angle_rad=np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False),
        torque_factor_m=np.zeros(samples),
        cycle_time_s=10.0,
        stroke_length_m=1.0,
        unit_type="conventional",
    )
    early = solve_forward(motion, taper, config, PumpBoundary(20000.0), 0.05, 7850.0, cycles=1)
    late = solve_forward(motion, taper, config, PumpBoundary(20000.0), 0.05, 7850.0, cycles=12)
    assert float(np.ptp(late.surface_load_n)) <= float(np.ptp(early.surface_load_n)) + 1.0


def test_courant_condition_is_respected(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion, taper, field_config.srp, PumpBoundary(15000.0), 0.25, 900.0, cycles=3
    )
    assert solution.courant_number <= MAX_COURANT_NUMBER + 1e-9


def test_courant_violation_raises_a_clear_error(
    field_config: FieldConfig, kinematics: ConventionalKinematics, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.twin.srp.wave as wave_module

    monkeypatch.setattr(wave_module, "MAX_COURANT_NUMBER", 1.5)
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 8)
    taper = build_taper(field_config.srp)
    with pytest.raises(NumericalError, match="Courant"):
        solve_forward(motion, taper, field_config.srp, PumpBoundary(15000.0), 0.25, 900.0, cycles=1)


def test_zero_cycles_is_rejected(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    with pytest.raises(PhysicsDomainError):
        solve_forward(motion, taper, field_config.srp, PumpBoundary(15000.0), 0.25, 900.0, cycles=0)


def test_static_card_brackets_the_rod_weight_and_fluid_load(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    """At a very low pumping speed the card should approach the static limits."""
    fluid_load_n = 15000.0
    buoyant_n = taper.buoyant_weight_n(900.0, field_config.srp.steel_density_kg_per_m3)
    motion = kinematics.motion(1.5, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion,
        taper,
        field_config.srp,
        PumpBoundary(fluid_load_n, stroke_length_m=2.3),
        0.05,
        900.0,
        cycles=6,
    )
    assert solution.peak_polished_rod_load_n == pytest.approx(buoyant_n + fluid_load_n, rel=0.25)
    assert solution.minimum_polished_rod_load_n == pytest.approx(buoyant_n, rel=0.25)


def test_load_range_grows_with_pumping_speed(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    ranges = []
    for spm in (2.0, 5.0, 8.0):
        motion = kinematics.motion(spm, 2.54, SpeedProfile(), 360)
        solution = solve_forward(
            motion,
            taper,
            field_config.srp,
            PumpBoundary(15000.0, stroke_length_m=2.3),
            0.25,
            900.0,
            cycles=5,
        )
        ranges.append(solution.load_range_n)
    assert all(later > earlier for earlier, later in itertools.pairwise(ranges))


def test_plunger_stroke_is_shorter_than_the_surface_stroke(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    """Rod stretch under the fluid load always costs some plunger travel."""
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion,
        taper,
        field_config.srp,
        PumpBoundary(15000.0, stroke_length_m=2.3),
        0.25,
        900.0,
        cycles=5,
    )
    assert solution.pump_stroke_m < 2.54
    assert solution.pump_stroke_m > 0.5 * 2.54


def test_solution_has_no_non_finite_values(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion, taper, field_config.srp, PumpBoundary(15000.0), 0.25, 900.0, cycles=4
    )
    for array in (
        solution.surface_load_n,
        solution.pump_load_n,
        solution.pump_position_m,
        solution.node_peak_load_n,
        solution.node_min_load_n,
    ):
        assert np.all(np.isfinite(array))


def test_inverse_solution_recovers_the_pump_card(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    """Cross-validation: forward solver, then Gibbs inverse, must close the loop."""
    fluid_load_n = 15000.0
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion,
        taper,
        field_config.srp,
        PumpBoundary(fluid_load_n, stroke_length_m=2.3),
        0.25,
        900.0,
        cycles=6,
    )
    position_m, load_n = downhole_from_surface(
        solution.surface_position_m,
        solution.surface_load_n,
        motion.cycle_time_s,
        taper,
        field_config.srp,
        0.25,
        900.0,
        retained_harmonics=24,
    )
    recovered_range = float(np.ptp(load_n))
    true_range = float(np.ptp(solution.pump_load_n))
    assert recovered_range == pytest.approx(true_range, rel=0.10)
    assert float(np.ptp(position_m)) == pytest.approx(solution.pump_stroke_m, rel=0.05)


def test_inverse_solution_rejects_mismatched_series(
    taper: RodTaper, field_config: FieldConfig
) -> None:
    with pytest.raises(PhysicsDomainError, match="same length"):
        downhole_from_surface(np.zeros(10), np.zeros(9), 12.0, taper, field_config.srp, 0.25, 900.0)


def test_incomplete_fillage_shortens_the_loaded_part_of_the_downstroke(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    full = solve_forward(
        motion,
        taper,
        field_config.srp,
        PumpBoundary(15000.0, fillage_frac=1.0, stroke_length_m=2.3),
        0.25,
        900.0,
        cycles=5,
    )
    partial = solve_forward(
        motion,
        taper,
        field_config.srp,
        PumpBoundary(15000.0, fillage_frac=0.5, stroke_length_m=2.3),
        0.25,
        900.0,
        cycles=5,
    )
    assert float(np.mean(partial.pump_load_n)) > float(np.mean(full.pump_load_n))


def test_drag_profile_replaces_the_scalar_damping(
    taper: RodTaper,
    field_config: FieldConfig,
    kinematics: ConventionalKinematics,
    viscosity_profile: np.ndarray,
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    drag = build_drag_profile(
        taper, field_config.wellbore.tubing_inner_diameter_m, viscosity_profile, 5.0
    )
    solution = solve_forward(
        motion,
        taper,
        field_config.srp,
        PumpBoundary(15000.0, stroke_length_m=2.3),
        0.25,
        900.0,
        cycles=5,
        drag=drag,
    )
    assert solution.damping_coefficient_per_s > 0.0
    assert np.all(np.isfinite(solution.surface_load_n))


def test_drag_profile_length_is_validated(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    from app.twin.srp.wave import DragProfile

    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    bad = DragProfile(linear_coefficient_n_s_per_m2=np.zeros(3), static_force_n_per_m=np.zeros(3))
    with pytest.raises(PhysicsDomainError, match="one value per rod node"):
        solve_forward(
            motion,
            taper,
            field_config.srp,
            PumpBoundary(15000.0),
            0.25,
            900.0,
            cycles=2,
            drag=bad,
        )
