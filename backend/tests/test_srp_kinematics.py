"""Pumping unit kinematics tests for both unit types."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.twin.srp.kinematics import (
    ConventionalKinematics,
    HydraulicKinematics,
    SpeedProfile,
    build_kinematics,
)


def test_stroke_range_matches_the_requested_stroke(
    kinematics: ConventionalKinematics, neutral_profile: SpeedProfile
) -> None:
    for stroke_m in (1.68, 2.54, 3.66):
        motion = kinematics.motion(5.0, stroke_m, neutral_profile, 360)
        assert float(np.ptp(motion.position_m)) == pytest.approx(stroke_m, rel=0.01)
        assert float(np.min(motion.position_m)) == pytest.approx(0.0, abs=1e-6)


def test_cycle_starts_and_ends_at_the_bottom_of_the_stroke(
    kinematics: ConventionalKinematics, neutral_profile: SpeedProfile
) -> None:
    motion = kinematics.motion(5.0, 2.54, neutral_profile, 360)
    assert motion.position_m[0] == pytest.approx(0.0, abs=1e-6)


def test_neutral_profile_gives_the_commanded_speed(
    kinematics: ConventionalKinematics, neutral_profile: SpeedProfile
) -> None:
    motion = kinematics.motion(6.0, 2.54, neutral_profile, 360)
    assert motion.spm == pytest.approx(6.0, rel=1e-6)
    assert motion.cycle_time_s == pytest.approx(10.0, rel=1e-6)


def test_four_bar_linkage_is_asymmetric(
    kinematics: ConventionalKinematics, neutral_profile: SpeedProfile
) -> None:
    """A real beam unit does not reach the same peak speed up and down.

    An in-line slider crank would be exactly symmetric, which is why the API
    Spec 11E four-bar geometry is used instead.
    """
    motion = kinematics.motion(5.0, 2.54, neutral_profile, 720)
    up = motion.peak_upstroke_speed_m_per_s
    down = motion.peak_downstroke_speed_m_per_s
    assert abs(up - down) / up > 0.03


def test_crank_radius_grows_with_stroke(kinematics: ConventionalKinematics) -> None:
    radii = [kinematics.crank_radius_m(s) for s in (1.68, 2.13, 2.54, 3.05, 3.66)]
    assert all(later > earlier for earlier, later in itertools.pairwise(radii))


def test_unreachable_stroke_raises_a_clear_error(kinematics: ConventionalKinematics) -> None:
    with pytest.raises(PhysicsDomainError, match="cannot reach"):
        kinematics.crank_radius_m(14.0)


def test_slowing_the_downstroke_reduces_the_peak_downstroke_speed(
    kinematics: ConventionalKinematics,
) -> None:
    base = kinematics.motion(5.0, 2.54, SpeedProfile(), 360)
    slowed = kinematics.motion(5.0, 2.54, SpeedProfile(1.0, 0.5, 0.0), 360)
    assert slowed.peak_downstroke_speed_m_per_s < base.peak_downstroke_speed_m_per_s
    assert slowed.peak_upstroke_speed_m_per_s == pytest.approx(
        base.peak_upstroke_speed_m_per_s, rel=0.05
    )
    assert slowed.spm < base.spm


def test_top_of_downstroke_deceleration_slows_the_early_downstroke_most(
    kinematics: ConventionalKinematics,
) -> None:
    """The published control art says the top of the downstroke is what matters."""
    base = kinematics.motion(5.0, 2.54, SpeedProfile(), 720)
    shaped = kinematics.motion(5.0, 2.54, SpeedProfile(1.0, 1.0, 0.5, 0.35), 720)

    def early_downstroke_speed(motion) -> float:  # type: ignore[no-untyped-def]
        relative = motion.position_m / motion.stroke_length_m
        mask = motion.downstroke_mask & (relative > 0.75)
        return float(np.max(np.abs(motion.velocity_m_per_s[mask])))

    def late_downstroke_speed(motion) -> float:  # type: ignore[no-untyped-def]
        relative = motion.position_m / motion.stroke_length_m
        mask = motion.downstroke_mask & (relative < 0.25)
        return float(np.max(np.abs(motion.velocity_m_per_s[mask])))

    early_reduction = 1.0 - early_downstroke_speed(shaped) / early_downstroke_speed(base)
    late_reduction = 1.0 - late_downstroke_speed(shaped) / late_downstroke_speed(base)
    assert early_reduction > late_reduction


def test_speed_profile_rejects_out_of_range_values() -> None:
    with pytest.raises(PhysicsDomainError):
        SpeedProfile(upstroke_speed_frac=0.0)
    with pytest.raises(PhysicsDomainError):
        SpeedProfile(top_of_downstroke_decel_frac=1.0)
    with pytest.raises(PhysicsDomainError):
        SpeedProfile(decel_window_frac=0.0)


def test_neutral_profile_is_detected() -> None:
    assert SpeedProfile().is_neutral
    assert not SpeedProfile(downstroke_speed_frac=0.8).is_neutral


def test_torque_factor_integrates_to_the_stroke(
    kinematics: ConventionalKinematics, neutral_profile: SpeedProfile
) -> None:
    """The torque factor is the position derivative, so its positive part is the stroke."""
    motion = kinematics.motion(5.0, 2.54, neutral_profile, 1440)
    step_rad = float(np.mean(np.diff(motion.crank_angle_rad)))
    rise_m = float(np.sum(np.maximum(motion.torque_factor_m, 0.0)) * step_rad)
    assert rise_m == pytest.approx(2.54, rel=0.05)


def test_hydraulic_unit_covers_the_stroke(field_config: FieldConfig) -> None:
    unit = HydraulicKinematics(field_config.srp)
    motion = unit.motion(5.0, 2.54, SpeedProfile(), 360)
    assert float(np.ptp(motion.position_m)) == pytest.approx(2.54, rel=0.02)
    assert motion.unit_type == "hydraulic"
    assert np.all(motion.torque_factor_m == 0.0)


def test_hydraulic_unit_supports_different_up_and_down_speeds(
    field_config: FieldConfig,
) -> None:
    unit = HydraulicKinematics(field_config.srp)
    motion = unit.motion(5.0, 2.54, SpeedProfile(1.2, 0.5, 0.0), 720)
    assert motion.peak_upstroke_speed_m_per_s > motion.peak_downstroke_speed_m_per_s


def test_hydraulic_dwell_extends_the_cycle(field_config: FieldConfig) -> None:
    no_dwell = HydraulicKinematics(
        field_config.with_overrides({"srp": {"hydraulic_dwell_s": 0.0}}).srp
    ).motion(5.0, 2.54, SpeedProfile(), 360)
    with_dwell = HydraulicKinematics(
        field_config.with_overrides({"srp": {"hydraulic_dwell_s": 2.0}}).srp
    ).motion(5.0, 2.54, SpeedProfile(), 360)
    assert with_dwell.peak_upstroke_speed_m_per_s > no_dwell.peak_upstroke_speed_m_per_s


def test_hydraulic_unit_holds_a_flatter_velocity_than_a_crank_unit(
    field_config: FieldConfig, kinematics: ConventionalKinematics
) -> None:
    """A hydraulic cylinder can hold near constant speed; a crank unit cannot."""
    crank = kinematics.motion(5.0, 2.54, SpeedProfile(), 720)
    hydraulic = HydraulicKinematics(
        field_config.with_overrides({"srp": {"hydraulic_dwell_s": 0.0}}).srp
    ).motion(5.0, 2.54, SpeedProfile(), 720)

    def peak_to_mean(motion) -> float:  # type: ignore[no-untyped-def]
        up = motion.velocity_m_per_s[motion.velocity_m_per_s > 0.0]
        return float(np.max(up) / np.mean(up))

    assert peak_to_mean(hydraulic) < peak_to_mean(crank)


def test_build_kinematics_selects_the_configured_unit(field_config: FieldConfig) -> None:
    assert build_kinematics(field_config.srp).unit_type == "conventional"
    hydraulic_config = field_config.with_overrides({"srp": {"unit_type": "hydraulic"}})
    assert build_kinematics(hydraulic_config.srp).unit_type == "hydraulic"


def test_motion_is_periodic(
    kinematics: ConventionalKinematics, neutral_profile: SpeedProfile
) -> None:
    motion = kinematics.motion(5.0, 2.54, neutral_profile, 360)
    assert motion.position_m[0] == pytest.approx(motion.position_m[-1], abs=0.05)


def test_negative_speed_is_rejected(kinematics: ConventionalKinematics) -> None:
    with pytest.raises(PhysicsDomainError):
        kinematics.motion(-1.0, 2.54, SpeedProfile(), 360)
