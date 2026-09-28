"""Reservoir thermal and inflow tests against analytic and limiting cases."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.core.steam import saturation_temperature_c
from app.core.units import SECONDS_PER_DAY
from app.twin.fluid import FluidModel
from app.twin.reservoir import (
    CyclePhase,
    InjectionPlan,
    ReservoirModel,
    composite_radial_productivity_ratio,
    conduction_retention,
    corey_oil_relative_permeability,
    darcy_liquid_rate_m3_per_s,
    marx_langenheim_area_m2,
    marx_langenheim_g,
    radial_conduction_unit_solution,
    sandface_heat_rate_w,
    vertical_conduction_unit_solution,
)


# --------------------------------------------------------------------------- Marx-Langenheim
def test_marx_langenheim_g_tends_to_dimensionless_time_at_short_times() -> None:
    """Limiting case: G(t_D) approaches t_D as t_D goes to zero."""
    for t_d in (1.0e-6, 1.0e-5, 1.0e-4):
        assert float(marx_langenheim_g(t_d)) == pytest.approx(t_d, rel=0.01)


def test_marx_langenheim_g_is_monotone_and_finite_at_large_times() -> None:
    t_d = np.logspace(-6, 4, 200)
    values = marx_langenheim_g(t_d)
    assert np.all(np.diff(values) > 0.0)
    assert np.all(np.isfinite(values))


def test_marx_langenheim_area_without_conduction_equals_the_energy_balance() -> None:
    """With zero overburden conductivity, A = Q t / (M_R h dT) exactly."""
    heat_rate_w = 4.0e6
    days = 10.0
    thickness_m = 12.0
    temperature_rise_k = 270.0
    heat_capacity = 2.35e6
    area = marx_langenheim_area_m2(
        net_heat_rate_w=heat_rate_w,
        elapsed_days=days,
        net_pay_thickness_m=thickness_m,
        temperature_rise_k=temperature_rise_k,
        rock_volumetric_heat_capacity_j_per_m3_k=heat_capacity,
        overburden_thermal_conductivity_w_per_m_k=0.0,
        overburden_volumetric_heat_capacity_j_per_m3_k=heat_capacity,
    )
    expected = (
        heat_rate_w * days * SECONDS_PER_DAY / (heat_capacity * thickness_m * temperature_rise_k)
    )
    assert area == pytest.approx(expected, rel=1e-9)


def test_marx_langenheim_area_grows_monotonically_with_time() -> None:
    areas = [
        marx_langenheim_area_m2(4.0e6, days, 12.0, 270.0, 2.35e6, 1.7, 2.35e6)
        for days in (1.0, 3.0, 6.0, 11.0, 20.0)
    ]
    assert all(later > earlier for earlier, later in itertools.pairwise(areas))


def test_conduction_loss_reduces_the_heated_area() -> None:
    with_loss = marx_langenheim_area_m2(4.0e6, 11.0, 12.0, 270.0, 2.35e6, 1.7, 2.35e6)
    without_loss = marx_langenheim_area_m2(4.0e6, 11.0, 12.0, 270.0, 2.35e6, 0.0, 2.35e6)
    assert with_loss < without_loss


def test_marx_langenheim_rejects_negative_inputs() -> None:
    with pytest.raises(PhysicsDomainError):
        marx_langenheim_area_m2(-1.0, 5.0, 12.0, 270.0, 2.35e6, 1.7, 2.35e6)


# ----------------------------------------------------------------------- conduction decay
def test_vertical_unit_solution_limits() -> None:
    assert vertical_conduction_unit_solution(0.0, 12.0, 7.2e-7) == pytest.approx(1.0)
    assert vertical_conduction_unit_solution(1.0e9, 12.0, 7.2e-7) == pytest.approx(0.0, abs=1e-3)


def test_radial_unit_solution_limits_and_asymptote() -> None:
    assert radial_conduction_unit_solution(0.0, 12.0, 7.2e-7) == pytest.approx(1.0)
    days = 1.0e5
    alpha = 7.2e-7
    radius = 12.0
    expected = radius**2 / (4.0 * alpha * days * SECONDS_PER_DAY)
    assert radial_conduction_unit_solution(days, radius, alpha) == pytest.approx(expected, rel=0.01)


def test_conduction_retention_decreases_with_time() -> None:
    values = [conduction_retention(d, 12.0, 12.0, 7.2e-7) for d in (0.0, 10.0, 50.0, 200.0)]
    assert all(later < earlier for earlier, later in itertools.pairwise(values))
    assert values[0] == pytest.approx(1.0)


def test_a_thicker_and_wider_zone_retains_heat_longer() -> None:
    small = conduction_retention(60.0, 6.0, 6.0, 7.2e-7)
    large = conduction_retention(60.0, 20.0, 20.0, 7.2e-7)
    assert large > small


# ------------------------------------------------------------------- composite inflow
def test_ratio_is_one_when_the_heated_radius_equals_the_wellbore_radius() -> None:
    ratio = composite_radial_productivity_ratio(
        heated_radius_m=0.108,
        wellbore_radius_m=0.108,
        drainage_radius_m=75.0,
        hot_viscosity_pa_s=0.003,
        cold_viscosity_pa_s=15.0,
    )
    assert ratio == pytest.approx(1.0, rel=1e-12)


def test_ratio_tends_to_the_viscosity_ratio_as_the_heated_zone_fills_the_drainage_area() -> None:
    """The limit is exact only when r_h equals r_e, so test the convergence."""
    cold, hot = 15.0, 0.003
    errors = []
    for gap_m in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        ratio = composite_radial_productivity_ratio(
            heated_radius_m=75.0 - gap_m,
            wellbore_radius_m=0.108,
            drainage_radius_m=75.0,
            hot_viscosity_pa_s=hot,
            cold_viscosity_pa_s=cold,
        )
        errors.append(abs(ratio - cold / hot) / (cold / hot))
    assert all(
        later < earlier for earlier, later in itertools.pairwise(errors)
    ), "the ratio must converge on the viscosity ratio as the gap closes"
    # Convergence is first order in the remaining cold annulus thickness.
    assert errors[-2] / errors[-1] == pytest.approx(10.0, rel=0.2)
    assert errors[-1] < 1.0e-4


def test_ratio_rises_with_heated_radius() -> None:
    ratios = [
        composite_radial_productivity_ratio(radius, 0.108, 75.0, 0.003, 15.0)
        for radius in (1.0, 5.0, 12.0, 30.0)
    ]
    assert all(later > earlier for earlier, later in itertools.pairwise(ratios))


def test_inflow_ratio_rejects_bad_geometry() -> None:
    with pytest.raises(PhysicsDomainError):
        composite_radial_productivity_ratio(5.0, 10.0, 5.0, 0.003, 15.0)


def test_darcy_rate_scales_linearly_with_drawdown() -> None:
    base = darcy_liquid_rate_m3_per_s(1.0e-12, 12.0, 1.0e6, 15.0, 75.0, 0.108)
    doubled = darcy_liquid_rate_m3_per_s(1.0e-12, 12.0, 2.0e6, 15.0, 75.0, 0.108)
    assert doubled == pytest.approx(2.0 * base)


def test_darcy_rate_is_zero_without_drawdown() -> None:
    assert darcy_liquid_rate_m3_per_s(1.0e-12, 12.0, -1.0, 15.0, 75.0, 0.108) == 0.0


def test_corey_relative_permeability_end_points() -> None:
    assert corey_oil_relative_permeability(0.78, 0.22, 0.78) == pytest.approx(1.0)
    assert corey_oil_relative_permeability(0.22, 0.22, 0.78) == pytest.approx(0.0)
    assert 0.0 < corey_oil_relative_permeability(0.5, 0.22, 0.78) < 1.0


# --------------------------------------------------------------------------- full cycle
def _run_one_cycle(
    model: ReservoirModel, config: FieldConfig, production_days: int = 150
) -> list[dict[str, float]]:
    plan = InjectionPlan.from_config(config)
    heat_rate_w = sandface_heat_rate_w(plan, config, wellbore_heat_loss_fraction=0.02)
    steam_temp_c = model.sandface_steam_temperature_c(plan.injection_pressure_kpa)
    model.begin_cycle(1)
    model.step_injection(plan, plan.injection_days, heat_rate_w, steam_temp_c)
    model.step_soak(plan.soak_days)
    model.state.phase_day = 0.0
    history: list[dict[str, float]] = []
    for _ in range(production_days):
        result = model.step_production(1.0, 3000.0, pump_capacity_m3_per_day=30.0)
        snapshot = model.snapshot()
        snapshot.update(result)
        history.append(snapshot)
    return history


def test_one_cycle_shows_the_expected_qualitative_behaviour(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    """Temperature decays, viscosity rises, rate decays and SOR falls then settles."""
    history = _run_one_cycle(reservoir, field_config)
    temperatures = [row["average_heated_temp_c"] for row in history]
    viscosities = [row["sandface_viscosity_cp"] for row in history]
    rates = [row["oil_rate_m3_per_day"] for row in history]
    assert temperatures[-1] < temperatures[0]
    assert all(later <= earlier + 1e-9 for earlier, later in itertools.pairwise(temperatures))
    assert viscosities[-1] > viscosities[0]
    assert rates[-1] < rates[10]
    assert reservoir.state.cycle_oil_m3 > 0.0


def test_heated_radius_is_tens_of_metres(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    """Sanity check from the brief: heated radius in the tens of metres."""
    _run_one_cycle(reservoir, field_config, production_days=5)
    assert 3.0 <= reservoir.state.heated_radius_m <= 60.0


def test_steam_oil_ratio_is_plausible(reservoir: ReservoirModel, field_config: FieldConfig) -> None:
    """Sanity check from the brief: SOR of a few m3 of steam per m3 of oil."""
    _run_one_cycle(reservoir, field_config)
    assert 1.0 <= reservoir.state.steam_oil_ratio <= 12.0


def test_no_negative_flows_or_non_finite_values(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    history = _run_one_cycle(reservoir, field_config)
    for row in history:
        for key, value in row.items():
            assert np.isfinite(value) or key.endswith("ratio"), f"{key} is not finite"
        assert row["oil_rate_m3_per_day"] >= 0.0
        assert row["water_rate_m3_per_day"] >= 0.0


def test_later_cycles_are_weaker(reservoir: ReservoirModel, field_config: FieldConfig) -> None:
    plan = InjectionPlan.from_config(field_config)
    heat_rate_w = sandface_heat_rate_w(plan, field_config, 0.02)
    steam_temp_c = reservoir.sandface_steam_temperature_c(plan.injection_pressure_kpa)
    cycle_oil: list[float] = []
    for cycle in (1, 2, 3, 4):
        reservoir.begin_cycle(cycle)
        reservoir.step_injection(plan, plan.injection_days, heat_rate_w, steam_temp_c)
        reservoir.step_soak(plan.soak_days)
        reservoir.state.phase_day = 0.0
        for _ in range(120):
            reservoir.step_production(1.0, 3000.0, 30.0)
        cycle_oil.append(reservoir.state.cycle_oil_m3)
    assert cycle_oil[-1] < cycle_oil[0]


def test_water_cut_rises_across_cycles(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    reservoir.begin_cycle(1)
    first = reservoir.state.water_cut_frac
    reservoir.begin_cycle(4)
    assert reservoir.state.water_cut_frac > first


def test_energy_balance_closes_within_tolerance(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    """Injected heat equals heat stored plus heat lost plus heat produced."""
    _run_one_cycle(reservoir, field_config, production_days=60)
    state = reservoir.state
    accounted = (
        state.heated_zone_energy_j
        + state.cumulative_conduction_loss_j
        + state.cumulative_produced_heat_j
    )
    assert accounted == pytest.approx(state.cumulative_injected_heat_j, rel=0.02)


def test_transient_drainage_radius_grows_and_is_bounded(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    _run_one_cycle(reservoir, field_config, production_days=5)
    radii = [reservoir.transient_drainage_radius_m(d) for d in (0.0, 1.0, 10.0, 60.0, 400.0)]
    assert all(later >= earlier - 1e-9 for earlier, later in itertools.pairwise(radii))
    assert radii[-1] == pytest.approx(field_config.reservoir.drainage_radius_m)
    assert radii[0] >= reservoir.state.heated_radius_m


def test_pump_capacity_limits_production(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    plan = InjectionPlan.from_config(field_config)
    heat_rate_w = sandface_heat_rate_w(plan, field_config, 0.02)
    steam_temp_c = reservoir.sandface_steam_temperature_c(plan.injection_pressure_kpa)
    reservoir.begin_cycle(1)
    reservoir.step_injection(plan, plan.injection_days, heat_rate_w, steam_temp_c)
    reservoir.step_soak(plan.soak_days)
    reservoir.state.phase_day = 0.0
    result = reservoir.step_production(1.0, 3000.0, pump_capacity_m3_per_day=2.0)
    assert result["total_liquid_rate_m3_per_day"] == pytest.approx(2.0)
    assert result["limited_by_pump"] == pytest.approx(1.0)


def test_no_drawdown_gives_no_flow(reservoir: ReservoirModel) -> None:
    pressure = reservoir.state.reservoir_pressure_kpa
    assert reservoir.deliverability_m3_per_day(pressure + 100.0) == 0.0


def test_steam_temperature_follows_the_saturation_curve(reservoir: ReservoirModel) -> None:
    for pressure_kpa in (5000.0, 11000.0, 15000.0):
        assert reservoir.sandface_steam_temperature_c(pressure_kpa) == pytest.approx(
            saturation_temperature_c(pressure_kpa)
        )


def test_reset_returns_to_virgin_conditions(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    _run_one_cycle(reservoir, field_config, production_days=20)
    reservoir.reset()
    assert reservoir.state.cumulative_oil_m3 == 0.0
    assert reservoir.state.phase == CyclePhase.IDLE
    assert reservoir.state.reservoir_pressure_kpa == pytest.approx(
        field_config.reservoir.initial_pressure_kpa
    )


def test_fracture_switch_raises_the_heated_area(
    field_config: FieldConfig, fluid: FluidModel
) -> None:
    plain = ReservoirModel(field_config, fluid)
    fractured = ReservoirModel(
        field_config.with_overrides({"reservoir": {"fracture_enhancement_enabled": True}}), fluid
    )
    plan = InjectionPlan.from_config(field_config)
    heat_rate_w = sandface_heat_rate_w(plan, field_config, 0.02)
    steam_temp_c = plain.sandface_steam_temperature_c(plan.injection_pressure_kpa)
    for model in (plain, fractured):
        model.begin_cycle(1)
        model.step_injection(plan, plan.injection_days, heat_rate_w, steam_temp_c)
    assert fractured.state.heated_area_m2 > plain.state.heated_area_m2


def test_injection_pressure_is_capped_at_the_fracture_pressure(
    reservoir: ReservoirModel, field_config: FieldConfig
) -> None:
    plan = InjectionPlan.from_config(field_config)
    heat_rate_w = sandface_heat_rate_w(plan, field_config, 0.02)
    steam_temp_c = reservoir.sandface_steam_temperature_c(plan.injection_pressure_kpa)
    reservoir.begin_cycle(1)
    reservoir.step_injection(plan, 200.0, heat_rate_w, steam_temp_c)
    assert reservoir.state.reservoir_pressure_kpa <= reservoir.fracture_pressure_kpa() + 1e-6


def test_higher_steam_volume_gives_a_larger_heated_zone(
    field_config: FieldConfig, fluid: FluidModel
) -> None:
    radii: list[float] = []
    for volume_m3 in (1200.0, 1800.0, 2600.0):
        model = ReservoirModel(field_config, fluid)
        plan = InjectionPlan.from_config(field_config)
        plan = InjectionPlan(
            steam_volume_m3_cwe=volume_m3,
            injection_rate_m3_per_day_cwe=plan.injection_rate_m3_per_day_cwe,
            injection_pressure_kpa=plan.injection_pressure_kpa,
            steam_quality_frac=plan.steam_quality_frac,
            soak_days=plan.soak_days,
            cutoff_marginal_energy_ratio=plan.cutoff_marginal_energy_ratio,
        )
        heat_rate_w = sandface_heat_rate_w(plan, field_config, 0.02)
        steam_temp_c = model.sandface_steam_temperature_c(plan.injection_pressure_kpa)
        model.begin_cycle(1)
        model.step_injection(plan, plan.injection_days, heat_rate_w, steam_temp_c)
        radii.append(model.state.heated_radius_m)
    assert all(later > earlier for earlier, later in itertools.pairwise(radii))
    assert math.isfinite(radii[-1])
