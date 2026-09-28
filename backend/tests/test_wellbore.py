"""Wellbore heat transmission tests, including the VIT versus bare comparison."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.twin.fluid import FluidModel
from app.twin.wellbore import (
    WellboreModel,
    geothermal_temperature_c,
    overall_resistance_coefficient_w_per_m_k,
    ramey_downward_temperature_c,
    ramey_time_function,
    ramey_upward_temperature_c,
)


def test_ramey_time_function_grows_with_time() -> None:
    values = [ramey_time_function(days, 0.0365, 7.2e-7) for days in (0.1, 1.0, 10.0, 100.0)]
    assert all(later > earlier for earlier, later in itertools.pairwise(values))
    assert all(value > 0.0 for value in values)


def test_overall_conductance_rises_with_the_completion_coefficient() -> None:
    low = overall_resistance_coefficient_w_per_m_k(0.6, 0.0365, 1.7, 2.0)
    high = overall_resistance_coefficient_w_per_m_k(17.0, 0.0365, 1.7, 2.0)
    assert high > low


def test_geothermal_profile_is_linear() -> None:
    assert geothermal_temperature_c(0.0, 32.0, 0.013) == pytest.approx(32.0)
    assert geothermal_temperature_c(1150.0, 32.0, 0.013) == pytest.approx(32.0 + 14.95)


def test_downward_solution_has_no_heat_loss_at_infinite_relaxation_distance() -> None:
    """Limiting case: with no heat loss the fluid keeps its inlet temperature."""
    value = ramey_downward_temperature_c(1100.0, 320.0, 1.0e12, 32.0, 0.013)
    assert float(value) == pytest.approx(320.0, rel=1e-6)


def test_downward_solution_reaches_the_formation_at_zero_relaxation_distance() -> None:
    """Limiting case: with very large heat loss the fluid equals the formation."""
    depth_m = 1100.0
    value = ramey_downward_temperature_c(depth_m, 320.0, 1.0e-6, 32.0, 0.013)
    assert float(value) == pytest.approx(
        float(geothermal_temperature_c(depth_m, 32.0, 0.013)), rel=1e-6
    )


def test_upward_solution_has_no_heat_loss_at_infinite_relaxation_distance() -> None:
    value = ramey_upward_temperature_c(0.0, 280.0, 1100.0, 1.0e12, 32.0, 0.013)
    assert float(value) == pytest.approx(280.0, rel=1e-5)


def test_upward_solution_reaches_the_formation_at_zero_relaxation_distance() -> None:
    value = ramey_upward_temperature_c(0.0, 280.0, 1100.0, 1.0e-6, 32.0, 0.013)
    assert float(value) == pytest.approx(32.0, rel=1e-6)


def test_upward_profile_cools_towards_the_surface(wellbore: WellboreModel) -> None:
    result = wellbore.produce(280.0, 5.0, 0.15, 30.0)
    assert result.wellhead_temp_c < result.pump_intake_temp_c
    assert np.all(np.diff(result.temperature_c) >= -1e-9)


def test_vit_delivers_hotter_steam_than_bare_tubing(wellbore: WellboreModel) -> None:
    """Acceptance criterion for milestone 2."""
    vit = wellbore.with_heat_transfer(0.6).inject_steam(11000.0, 0.78, 160.0, 5.0)
    bare = wellbore.with_heat_transfer(17.0).inject_steam(11000.0, 0.78, 160.0, 5.0)
    assert vit.sandface_quality_frac > bare.sandface_quality_frac
    assert vit.sandface_heat_rate_w > bare.sandface_heat_rate_w
    assert vit.heat_loss_fraction < bare.heat_loss_fraction


def test_vit_keeps_produced_fluid_hotter_up_the_string(wellbore: WellboreModel) -> None:
    vit = wellbore.with_heat_transfer(0.6).produce(280.0, 5.0, 0.15, 30.0)
    bare = wellbore.with_heat_transfer(17.0).produce(280.0, 5.0, 0.15, 30.0)
    assert vit.wellhead_temp_c > bare.wellhead_temp_c
    mid_depth_m = 550.0
    assert vit.viscosity_at_depth_pa_s(mid_depth_m) < bare.viscosity_at_depth_pa_s(mid_depth_m)


def test_injection_quality_falls_monotonically_with_depth(wellbore: WellboreModel) -> None:
    result = wellbore.inject_steam(11000.0, 0.80, 160.0, 5.0)
    assert np.all(np.diff(result.quality_frac) <= 1e-12)
    assert result.sandface_quality_frac < 0.80


def test_zero_heat_loss_preserves_steam_quality(wellbore: WellboreModel) -> None:
    """Limiting case: an ideal insulator loses no quality."""
    result = wellbore.with_heat_transfer(1.0e-9).inject_steam(11000.0, 0.78, 160.0, 5.0)
    assert result.sandface_quality_frac == pytest.approx(0.78, rel=1e-4)
    assert result.heat_loss_fraction == pytest.approx(0.0, abs=1e-4)


def test_higher_injection_rate_lowers_the_heat_loss_fraction(wellbore: WellboreModel) -> None:
    """More mass carries the same wall loss, so the fractional loss falls."""
    slow = wellbore.inject_steam(11000.0, 0.78, 80.0, 5.0)
    fast = wellbore.inject_steam(11000.0, 0.78, 260.0, 5.0)
    assert fast.heat_loss_fraction < slow.heat_loss_fraction


def test_heat_loss_fraction_falls_as_the_formation_warms_up(wellbore: WellboreModel) -> None:
    early = wellbore.inject_steam(11000.0, 0.78, 160.0, 0.5)
    late = wellbore.inject_steam(11000.0, 0.78, 160.0, 60.0)
    assert late.heat_loss_fraction < early.heat_loss_fraction


def test_invalid_steam_quality_is_rejected(wellbore: WellboreModel) -> None:
    with pytest.raises(PhysicsDomainError, match="quality"):
        wellbore.inject_steam(11000.0, 1.4, 160.0, 5.0)


def test_negative_injection_rate_is_rejected(wellbore: WellboreModel) -> None:
    with pytest.raises(PhysicsDomainError):
        wellbore.inject_steam(11000.0, 0.78, -10.0, 5.0)


def test_bottomhole_pressure_rises_with_the_fluid_column(wellbore: WellboreModel) -> None:
    high_level = wellbore.bottomhole_pressure_kpa(200.0, 80.0, 0.2)
    low_level = wellbore.bottomhole_pressure_kpa(900.0, 80.0, 0.2)
    assert high_level > low_level


def test_friction_drop_is_zero_at_zero_rate(wellbore: WellboreModel) -> None:
    assert wellbore.tubing_friction_pressure_kpa(0.0, 1.0, 900.0) == 0.0


def test_friction_drop_rises_with_viscosity(wellbore: WellboreModel) -> None:
    thin = wellbore.tubing_friction_pressure_kpa(10.0, 0.01, 900.0)
    thick = wellbore.tubing_friction_pressure_kpa(10.0, 1.0, 900.0)
    assert thick > thin


def test_injection_pressure_safety_check(
    wellbore: WellboreModel, field_config: FieldConfig
) -> None:
    fracture_kpa = field_config.reservoir.fracture_pressure_kpa
    assert wellbore.injection_pressure_is_safe(0.5 * fracture_kpa, 0.92)
    assert not wellbore.injection_pressure_is_safe(0.99 * fracture_kpa, 0.92)


def test_produced_temperature_never_exceeds_the_intake(wellbore: WellboreModel) -> None:
    result = wellbore.produce(180.0, 3.0, 0.3, 10.0)
    assert np.all(result.temperature_c <= 180.0 + 1e-9)


def test_very_low_rate_still_gives_a_finite_profile(wellbore: WellboreModel) -> None:
    result = wellbore.produce(180.0, 0.01, 0.3, 10.0)
    assert np.all(np.isfinite(result.temperature_c))
    assert np.all(np.isfinite(result.viscosity_pa_s))


def test_bare_completion_selected_from_config(field_config: FieldConfig, fluid: FluidModel) -> None:
    bare_config = field_config.with_overrides({"wellbore": {"insulation_type": "bare"}})
    model = WellboreModel(bare_config, fluid)
    assert model.overall_heat_transfer_w_per_m2_k == pytest.approx(
        field_config.wellbore.bare_overall_heat_transfer_w_per_m2_k
    )
