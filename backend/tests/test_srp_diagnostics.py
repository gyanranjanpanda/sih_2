"""Drag, rod float, stress, pump and power tests, plus the card rule baseline."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.core.units import STANDARD_GRAVITY_M_PER_S2
from app.twin.fluid import FluidModel
from app.twin.srp.cards import (
    CARD_CLASS_ORDER,
    CardClass,
    DynamometerCard,
    classify_by_rules,
    extract_features,
)
from app.twin.srp.floating import (
    analyse_float,
    annular_drag_per_length_n_per_m,
    build_drag_profile,
    coupling_drag_n,
    string_drag_n,
)
from app.twin.srp.kinematics import ConventionalKinematics, SpeedProfile
from app.twin.srp.power import (
    evaluate_power,
    motor_efficiency_frac,
    net_gearbox_torque_n_m,
    polished_rod_power_w,
    spm_from_vfd_frequency,
    vfd_frequency_from_spm,
)
from app.twin.srp.pump import (
    evaluate_pump,
    fillage_from_inflow,
    pump_displacement_m3_per_day,
    slippage_rate_m3_per_day,
)
from app.twin.srp.stress import (
    FatigueCounter,
    analyse_stress,
    cycles_to_failure,
    modified_goodman_allowable_pa,
    stress_profile_mpa,
)
from app.twin.srp.wave import PumpBoundary, RodTaper, solve_forward
from app.twin.wellbore import WellboreModel


# ------------------------------------------------------------------------------ drag
def test_pure_couette_drag_matches_the_thin_gap_limit() -> None:
    """With a narrow gap and no net flow the shear should approach mu V / gap."""
    rod_radius_m = 0.0500
    tubing_radius_m = 0.0505
    velocity = 1.0
    viscosity = 1.0
    solution = annular_drag_per_length_n_per_m(
        rod_radius_m, tubing_radius_m, velocity, 0.0, viscosity
    )
    gap = tubing_radius_m - rod_radius_m
    couette_only = -2.0 * math.pi * rod_radius_m * viscosity * velocity / gap
    # The return flow raises the drag above the pure Couette value, but for a
    # narrow gap the two must be the same order and the same sign.
    assert solution.drag_per_length_n_per_m < 0.0
    assert abs(solution.drag_per_length_n_per_m) > abs(couette_only)
    assert abs(solution.drag_per_length_n_per_m) < 5.0 * abs(couette_only)


def test_drag_opposes_motion_in_both_directions() -> None:
    up = annular_drag_per_length_n_per_m(0.0111, 0.031, 0.6, 0.0, 1.0)
    down = annular_drag_per_length_n_per_m(0.0111, 0.031, -0.6, 0.0, 1.0)
    assert up.drag_per_length_n_per_m < 0.0
    assert down.drag_per_length_n_per_m > 0.0
    assert up.drag_per_length_n_per_m == pytest.approx(-down.drag_per_length_n_per_m, rel=1e-9)


def test_drag_is_linear_in_viscosity_and_velocity() -> None:
    base = annular_drag_per_length_n_per_m(0.0111, 0.031, 0.5, 0.0, 1.0)
    twice_viscosity = annular_drag_per_length_n_per_m(0.0111, 0.031, 0.5, 0.0, 2.0)
    twice_velocity = annular_drag_per_length_n_per_m(0.0111, 0.031, 1.0, 0.0, 1.0)
    assert twice_viscosity.drag_per_length_n_per_m == pytest.approx(
        2.0 * base.drag_per_length_n_per_m, rel=1e-9
    )
    assert twice_velocity.drag_per_length_n_per_m == pytest.approx(
        2.0 * base.drag_per_length_n_per_m, rel=1e-9
    )


def test_drag_rejects_a_rod_larger_than_the_tubing() -> None:
    with pytest.raises(PhysicsDomainError):
        annular_drag_per_length_n_per_m(0.05, 0.03, 0.5, 0.0, 1.0)


def test_coupling_drag_adds_to_the_body_drag() -> None:
    value = coupling_drag_n(120, 0.0413, 0.10, 0.031, 0.6, 1.0)
    assert value < 0.0


def test_coupling_drag_rejects_an_oversized_coupling() -> None:
    with pytest.raises(PhysicsDomainError):
        coupling_drag_n(120, 0.08, 0.10, 0.031, 0.6, 1.0)


def test_cumulative_string_drag_is_largest_at_the_surface(
    taper: RodTaper, field_config: FieldConfig, viscosity_profile: np.ndarray
) -> None:
    drag = string_drag_n(
        taper,
        field_config.wellbore.tubing_inner_diameter_m,
        -0.6,
        5.0 / 86400.0,
        viscosity_profile,
    )
    assert drag[0] > drag[-1]
    assert drag[-1] == pytest.approx(0.0, abs=1.0)


def test_drag_profile_linearisation_reproduces_the_annulus_solution(
    taper: RodTaper, field_config: FieldConfig, viscosity_profile: np.ndarray
) -> None:
    """The annular solution is exactly linear in velocity, so the fit is exact."""
    profile = build_drag_profile(
        taper, field_config.wellbore.tubing_inner_diameter_m, viscosity_profile, 5.0
    )
    velocity = -0.45
    for index in (0, taper.node_count // 2, taper.node_count - 1):
        radius_m = math.sqrt(taper.area_m2[index] / math.pi)
        direct = annular_drag_per_length_n_per_m(
            radius_m,
            0.5 * field_config.wellbore.tubing_inner_diameter_m,
            velocity,
            5.0 / 86400.0,
            float(viscosity_profile[index]),
        ).drag_per_length_n_per_m
        linearised = (
            -profile.linear_coefficient_n_s_per_m2[index] * velocity
            + profile.static_force_n_per_m[index]
        )
        assert linearised == pytest.approx(direct, rel=1e-9)


# ------------------------------------------------------------------------ rod float
def _float_case(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
    sandface_temp_c: float,
    downstroke_frac: float = 1.0,
    upstroke_frac: float = 1.0,
):  # type: ignore[no-untyped-def]
    """Run one operating point and return the float analysis and the wave solution."""
    production = WellboreModel(field_config, fluid).produce(sandface_temp_c, 5.0, 0.15, 30.0)
    viscosity = np.interp(taper.depth_m, production.depth_m, production.viscosity_pa_s)
    drag = build_drag_profile(taper, field_config.wellbore.tubing_inner_diameter_m, viscosity, 5.0)
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(upstroke_frac, downstroke_frac, 0.0), 360)
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
    analysis = analyse_float(
        taper,
        field_config.srp,
        motion,
        viscosity,
        900.0,
        field_config.wellbore.tubing_inner_diameter_m,
        5.0,
        solution.surface_load_n,
    )
    return analysis, solution, motion


def test_float_margin_is_higher_for_a_hot_well(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    """Acceptance criterion for milestone 3: heavy cold oil floats, hot oil does not."""
    hot, _, _ = _float_case(field_config, fluid, taper, kinematics, 300.0)
    cold, _, _ = _float_case(field_config, fluid, taper, kinematics, 120.0)
    assert hot.minimum_section_margin > cold.minimum_section_margin
    assert hot.minimum_section_margin > 0.0
    assert cold.minimum_section_margin < 0.0


def test_slowing_the_downstroke_raises_the_float_margin(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    fast, _, _ = _float_case(field_config, fluid, taper, kinematics, 120.0, 1.0)
    slow, _, _ = _float_case(field_config, fluid, taper, kinematics, 120.0, 0.5)
    assert slow.minimum_section_margin > fast.minimum_section_margin


def test_slowing_only_the_downstroke_beats_slowing_both_strokes_equally(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    """Acceptance criterion for milestone 3.

    Two settings are compared at the same pumping rate: slowing only the
    downstroke, and slowing both strokes by the same amount. Both reach the same
    strokes per minute, but only the first reduces the peak downstroke velocity,
    which is what sets the drag and therefore the float margin.
    """
    downstroke_only, _, motion_a = _float_case(
        field_config, fluid, taper, kinematics, 120.0, downstroke_frac=0.5, upstroke_frac=1.0
    )
    both_strokes, _, motion_b = _float_case(
        field_config, fluid, taper, kinematics, 120.0, downstroke_frac=0.667, upstroke_frac=0.667
    )
    assert motion_a.spm == pytest.approx(motion_b.spm, rel=0.12)
    assert downstroke_only.minimum_section_margin > both_strokes.minimum_section_margin


def test_float_shows_in_the_surface_card_and_gives_an_impact_load(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    analysis, solution, _ = _float_case(field_config, fluid, taper, kinematics, 120.0)
    assert analysis.float_index > 0.0
    assert analysis.downstroke_fraction_affected > analysis.float_index
    assert analysis.estimated_impact_load_n > 0.0
    assert solution.minimum_polished_rod_load_n < field_config.srp.minimum_polished_rod_load_n
    assert analysis.is_floating


def test_float_margin_matches_a_hand_calculation(
    field_config: FieldConfig, taper: RodTaper, kinematics: ConventionalKinematics
) -> None:
    """With a uniform viscosity the margin is (W_b - F_drag) / W_b exactly."""
    viscosity = np.full(taper.node_count, 0.05)
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    analysis = analyse_float(
        taper,
        field_config.srp,
        motion,
        viscosity,
        900.0,
        field_config.wellbore.tubing_inner_diameter_m,
        5.0,
    )
    expected_weight_n = sum(
        (field_config.srp.steel_density_kg_per_m3 - 900.0)
        * STANDARD_GRAVITY_M_PER_S2
        * section.area_m2
        * section.length_m
        for section in taper.sections
    )
    assert analysis.buoyant_weight_n == pytest.approx(expected_weight_n, rel=0.02)
    hand_margin = (analysis.buoyant_weight_n - analysis.peak_drag_n) / analysis.buoyant_weight_n
    assert analysis.float_margin_index == pytest.approx(hand_margin, rel=1e-9)


def test_float_analysis_validates_the_viscosity_profile_length(
    field_config: FieldConfig, taper: RodTaper, kinematics: ConventionalKinematics
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    with pytest.raises(PhysicsDomainError, match="one value per rod node"):
        analyse_float(
            taper,
            field_config.srp,
            motion,
            np.zeros(5) + 0.05,
            900.0,
            field_config.wellbore.tubing_inner_diameter_m,
            5.0,
        )


# --------------------------------------------------------------------------- stress
def test_goodman_allowable_matches_the_api_formula() -> None:
    allowable = modified_goodman_allowable_pa(1.0e8, 7.93e8, 1.0)
    assert allowable == pytest.approx(0.25 * 7.93e8 + 0.5625 * 1.0e8)


def test_service_factor_scales_the_allowable() -> None:
    full = modified_goodman_allowable_pa(1.0e8, 7.93e8, 1.0)
    derated = modified_goodman_allowable_pa(1.0e8, 7.93e8, 0.8)
    assert derated == pytest.approx(0.8 * full)


def test_fatigue_life_falls_as_the_amplitude_rises() -> None:
    lives = [cycles_to_failure(amplitude, 2.0e8) for amplitude in (1.0e8, 1.5e8, 2.5e8)]
    assert all(later < earlier for earlier, later in itertools.pairwise(lives))
    assert cycles_to_failure(0.0, 2.0e8) == math.inf


def test_stress_report_identifies_the_limiting_section(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
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
    report = analyse_stress(solution, taper, field_config.srp)
    assert len(report.sections) == len(field_config.srp.rod_sections)
    assert 0.0 < report.maximum_utilisation_frac < 2.0
    assert report.sections[report.limiting_section_index].utilisation_frac == pytest.approx(
        report.maximum_utilisation_frac
    )
    depth, peak, minimum = stress_profile_mpa(solution, taper)
    assert depth.size == peak.size == minimum.size == taper.node_count
    assert np.all(peak >= minimum)


def test_fatigue_counter_accumulates_and_impacts_hurt_more(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
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
    report = analyse_stress(solution, taper, field_config.srp)
    counter = FatigueCounter()
    running = counter.accumulate(report, 10000.0)
    impact = counter.accumulate_impact(20000.0, report, taper, 10000.0)
    assert impact > running
    assert counter.damage == pytest.approx(running + impact)
    assert 0.0 <= counter.remaining_life_fraction <= 1.0


def test_fatigue_counter_rejects_negative_cycles(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion, taper, field_config.srp, PumpBoundary(15000.0), 0.25, 900.0, cycles=3
    )
    report = analyse_stress(solution, taper, field_config.srp)
    with pytest.raises(PhysicsDomainError):
        FatigueCounter().accumulate(report, -1.0)


# ----------------------------------------------------------------------------- pump
def test_displacement_matches_its_definition(field_config: FieldConfig) -> None:
    area = field_config.srp.plunger_area_m2
    assert pump_displacement_m3_per_day(area, 2.3, 5.0) == pytest.approx(area * 2.3 * 5.0 * 1440.0)


def test_pump_rate_never_exceeds_displacement(field_config: FieldConfig) -> None:
    """Sanity check from the brief."""
    for spm in (2.0, 5.0, 9.0):
        performance = evaluate_pump(field_config.srp, 2.3, spm, 0.01, 9.0e6)
        assert performance.liquid_rate_m3_per_day <= performance.displacement_m3_per_day
        assert performance.volumetric_efficiency_frac <= 1.0


def test_slippage_rises_as_viscosity_falls() -> None:
    hot = slippage_rate_m3_per_day(0.04445, 0.000127, 1.2, 9.0e6, 0.003)
    cold = slippage_rate_m3_per_day(0.04445, 0.000127, 1.2, 9.0e6, 1.0)
    assert hot > cold


def test_slippage_scales_with_the_cube_of_the_clearance() -> None:
    base = slippage_rate_m3_per_day(0.04445, 0.000127, 1.2, 9.0e6, 0.01)
    doubled = slippage_rate_m3_per_day(0.04445, 0.000254, 1.2, 9.0e6, 0.01)
    assert doubled / base == pytest.approx(8.0, rel=0.05)


def test_worn_pump_loses_volumetric_efficiency(field_config: FieldConfig) -> None:
    healthy = evaluate_pump(field_config.srp, 2.3, 5.0, 0.01, 9.0e6)
    worn = evaluate_pump(field_config.srp, 2.3, 5.0, 0.01, 9.0e6, clearance_multiplier=2.5)
    assert worn.volumetric_efficiency_frac < healthy.volumetric_efficiency_frac


def test_fillage_and_gas_cut_the_delivered_rate(field_config: FieldConfig) -> None:
    full = evaluate_pump(field_config.srp, 2.3, 5.0, 0.01, 9.0e6)
    partial = evaluate_pump(
        field_config.srp, 2.3, 5.0, 0.01, 9.0e6, fillage_frac=0.6, gas_interference_frac=0.1
    )
    assert partial.liquid_rate_m3_per_day < full.liquid_rate_m3_per_day


def test_zero_flow_edge_case(field_config: FieldConfig) -> None:
    performance = evaluate_pump(field_config.srp, 0.0, 0.0, 0.01, 9.0e6)
    assert performance.displacement_m3_per_day == 0.0
    assert performance.liquid_rate_m3_per_day == 0.0
    assert performance.volumetric_efficiency_frac == 0.0


def test_invalid_fillage_is_rejected(field_config: FieldConfig) -> None:
    with pytest.raises(PhysicsDomainError, match="Fillage"):
        evaluate_pump(field_config.srp, 2.3, 5.0, 0.01, 9.0e6, fillage_frac=1.5)


def test_fillage_from_inflow_is_bounded() -> None:
    assert fillage_from_inflow(50.0, 20.0) == pytest.approx(1.0)
    assert fillage_from_inflow(5.0, 20.0) == pytest.approx(0.25)
    assert fillage_from_inflow(5.0, 0.0) == 0.0


# ---------------------------------------------------------------------------- power
def test_polished_rod_power_of_a_pure_lift() -> None:
    """A constant load lifted at a constant speed for half a cycle."""
    samples = 1000
    load = np.full(samples, 10000.0)
    velocity = np.concatenate([np.full(samples // 2, 0.5), np.full(samples // 2, -0.5)])
    power = polished_rod_power_w(load, velocity, 10.0)
    assert power == pytest.approx(10000.0 * 0.5 * 0.5, rel=0.02)


def test_power_rejects_mismatched_series() -> None:
    with pytest.raises(PhysicsDomainError):
        polished_rod_power_w(np.zeros(4), np.zeros(5), 1.0)


def test_motor_efficiency_peaks_near_rated_load() -> None:
    assert motor_efficiency_frac(1.0, 0.92) == pytest.approx(0.92)
    assert motor_efficiency_frac(0.1, 0.92) < motor_efficiency_frac(0.8, 0.92)


def test_vfd_frequency_maps_to_speed(field_config: FieldConfig) -> None:
    spm = spm_from_vfd_frequency(field_config.srp, 50.0)
    assert spm == pytest.approx(field_config.srp.spm_setpoint)
    assert vfd_frequency_from_spm(field_config.srp, spm) == pytest.approx(50.0)


def test_vfd_frequency_outside_limits_is_rejected(field_config: FieldConfig) -> None:
    with pytest.raises(PhysicsDomainError, match="outside the configured limits"):
        spm_from_vfd_frequency(field_config.srp, 5.0)


def test_counterbalance_reduces_peak_gearbox_torque() -> None:
    """A counterbalance only helps when the load is high on the upstroke.

    The idealised case here has a sinusoidal torque factor, rod weight carried
    all the way round and the fluid load carried only on the upstroke, which is
    what a real card looks like. Setting the counterbalance moment halfway
    between the two peaks should roughly halve the peak torque.
    """
    angle = np.linspace(0.0, 2.0 * math.pi, 3600, endpoint=False)
    crank_radius_m = 1.0
    torque_factor = crank_radius_m * np.sin(angle)
    rod_weight_n = 25000.0
    fluid_load_n = 15000.0
    load = rod_weight_n + fluid_load_n * (np.sin(angle) > 0.0)

    without = net_gearbox_torque_n_m(torque_factor, load, angle, 0.0)
    ideal_moment = crank_radius_m * (rod_weight_n + 0.5 * fluid_load_n)
    balanced = net_gearbox_torque_n_m(torque_factor, load, angle, ideal_moment)

    assert float(np.max(np.abs(without))) == pytest.approx(
        crank_radius_m * (rod_weight_n + fluid_load_n), rel=0.01
    )
    assert float(np.max(np.abs(balanced))) == pytest.approx(
        crank_radius_m * 0.5 * fluid_load_n, rel=0.01
    )
    assert float(np.max(np.abs(balanced))) < float(np.max(np.abs(without)))


def test_over_counterbalancing_raises_peak_torque_again() -> None:
    angle = np.linspace(0.0, 2.0 * math.pi, 3600, endpoint=False)
    torque_factor = np.sin(angle)
    load = 25000.0 + 15000.0 * (np.sin(angle) > 0.0)
    ideal = net_gearbox_torque_n_m(torque_factor, load, angle, 32500.0)
    excessive = net_gearbox_torque_n_m(torque_factor, load, angle, 70000.0)
    assert float(np.max(np.abs(excessive))) > float(np.max(np.abs(ideal)))


def test_power_report_flags_an_overloaded_unit(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    derated = field_config.with_overrides(
        {"srp": {"structural_load_rating_n": 20000.0, "gearbox_torque_rating_n_m": 5000.0}}
    )
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
    report = evaluate_power(derated.srp, motion, solution.surface_load_n, 5.0)
    assert not report.within_limits
    assert "rating" in report.binding


def test_energy_per_barrel_is_finite_and_positive(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
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
    report = evaluate_power(field_config.srp, motion, solution.surface_load_n, 24.0)
    assert 0.0 < report.energy_kwh_per_bbl < 50.0
    assert report.within_limits


def test_zero_rate_gives_infinite_energy_per_barrel(
    taper: RodTaper, field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion, taper, field_config.srp, PumpBoundary(15000.0), 0.25, 900.0, cycles=3
    )
    report = evaluate_power(field_config.srp, motion, solution.surface_load_n, 0.0)
    assert math.isinf(report.energy_kwh_per_m3)


def test_hydraulic_unit_is_checked_against_its_own_limits(
    taper: RodTaper, field_config: FieldConfig
) -> None:
    from app.twin.srp.kinematics import HydraulicKinematics

    hydraulic_config = field_config.with_overrides({"srp": {"unit_type": "hydraulic"}})
    unit = HydraulicKinematics(hydraulic_config.srp)
    motion = unit.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion,
        taper,
        hydraulic_config.srp,
        PumpBoundary(15000.0, stroke_length_m=2.3),
        0.25,
        900.0,
        cycles=5,
    )
    report = evaluate_power(hydraulic_config.srp, motion, solution.surface_load_n, 20.0)
    assert report.peak_gearbox_torque_n_m == 0.0
    assert report.hydraulic_pressure_kpa > 0.0
    assert report.hydraulic_flow_m3_per_s > 0.0


# ---------------------------------------------------------------------------- cards
def _card_for(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
    sandface_temp_c: float = 250.0,
    fillage: float = 1.0,
    gas: float = 0.0,
    tagging: float = 0.0,
    load_multiplier: float = 1.0,
    sticking_n: float = 0.0,
):  # type: ignore[no-untyped-def]
    """Build one pump card plus its float index, for the classifier tests."""
    fluid_load_n = 15071.0
    production = WellboreModel(field_config, fluid).produce(sandface_temp_c, 5.0, 0.15, 30.0)
    viscosity = np.interp(taper.depth_m, production.depth_m, production.viscosity_pa_s)
    drag = build_drag_profile(taper, field_config.wellbore.tubing_inner_diameter_m, viscosity, 5.0)
    motion = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    solution = solve_forward(
        motion,
        taper,
        field_config.srp,
        PumpBoundary(
            fluid_load_n * load_multiplier,
            fillage_frac=fillage,
            gas_interference_frac=gas,
            stroke_length_m=2.31,
            tagging_contact_travel_m=tagging,
            sticking_load_n=sticking_n,
        ),
        0.25,
        900.0,
        cycles=6,
        drag=drag,
    )
    analysis = analyse_float(
        taper,
        field_config.srp,
        motion,
        viscosity,
        900.0,
        field_config.wellbore.tubing_inner_diameter_m,
        5.0,
        solution.surface_load_n,
    )
    card = DynamometerCard(
        solution.pump_position_m, solution.pump_load_n, False, motion.cycle_time_s
    )
    return card, analysis, fluid_load_n


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        (CardClass.NORMAL, {}),
        (CardClass.FLUID_POUND, {"fillage": 0.5}),
        (CardClass.GAS_INTERFERENCE, {"fillage": 0.75, "gas": 0.35}),
        (CardClass.WORN_PUMP, {"load_multiplier": 0.45}),
        (CardClass.TAGGING, {"tagging": 2.1}),
        (CardClass.ROD_FLOAT, {"sandface_temp_c": 120.0}),
        (CardClass.UNSEATING, {"load_multiplier": 0.12}),
        (CardClass.STICKING, {"sticking_n": 9000.0}),
        (CardClass.PARTED_ROD, {"load_multiplier": 0.01}),
    ],
)
def test_rule_classifier_recovers_each_injected_fault(
    label: CardClass,
    kwargs: dict[str, float],
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    card, analysis, expected_load = _card_for(field_config, fluid, taper, kinematics, **kwargs)
    features = extract_features(card)
    diagnosis = classify_by_rules(features, expected_load, surface_float_index=analysis.float_index)
    assert diagnosis.card_class == label
    assert 0.0 < diagnosis.confidence <= 1.0
    assert len(diagnosis.reason) > 30


def test_card_features_are_finite_and_serialisable(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    card, _, _ = _card_for(field_config, fluid, taper, kinematics)
    features = extract_features(card)
    for name, value in features.as_dict().items():
        assert np.isfinite(value), name


def test_normalised_card_is_scaled_to_the_unit_square(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    card, _, _ = _card_for(field_config, fluid, taper, kinematics)
    normalised = card.normalised(64)
    assert normalised.shape == (2, 64)
    assert normalised.min() == pytest.approx(0.0, abs=1e-9)
    assert normalised.max() == pytest.approx(1.0, abs=1e-9)


def test_card_work_is_positive(
    field_config: FieldConfig,
    fluid: FluidModel,
    taper: RodTaper,
    kinematics: ConventionalKinematics,
) -> None:
    card, _, _ = _card_for(field_config, fluid, taper, kinematics)
    assert card.work_per_cycle_j() > 0.0


def test_card_rejects_mismatched_arrays() -> None:
    with pytest.raises(PhysicsDomainError):
        DynamometerCard(np.zeros(10), np.zeros(9), False, 12.0)


def test_card_rejects_a_short_series() -> None:
    with pytest.raises(PhysicsDomainError):
        DynamometerCard(np.zeros(4), np.zeros(4), False, 12.0)


def test_class_order_is_stable() -> None:
    assert CARD_CLASS_ORDER[0] == CardClass.NORMAL
    assert CardClass.UNKNOWN not in CARD_CLASS_ORDER
    assert len(set(CARD_CLASS_ORDER)) == len(CARD_CLASS_ORDER)
