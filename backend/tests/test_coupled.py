"""Coupled engine, surface model and assimilation tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.twin.assimilation import (
    DEFAULT_PARAMETER_BOUNDS,
    EnsembleKalmanAssimilator,
    Observation,
    build_prior_ensemble,
    ensemble_kalman_update,
    rolling_least_squares_refit,
)
from app.twin.coupled import CoupledWellTwin, PumpSetpoint, TwinParameters
from app.twin.fluid import FluidModel
from app.twin.reservoir import InjectionPlan
from app.twin.srp.kinematics import SpeedProfile
from app.twin.surface import SurfaceModel


# --------------------------------------------------------------------- surface
def test_pumpable_temperature_is_inside_the_heating_range(
    field_config: FieldConfig, fluid: FluidModel
) -> None:
    surface = SurfaceModel(field_config, fluid)
    temperature_c = surface.temperature_for_pumpable_crude_c()
    assert 40.0 < temperature_c < 200.0
    assert float(fluid.viscosity_pa_s(temperature_c)) == pytest.approx(0.5, rel=0.02)


def test_hot_wellhead_fluid_needs_no_sensible_heating(
    field_config: FieldConfig, fluid: FluidModel
) -> None:
    """A well with vacuum insulated tubing can deliver fluid above the tank target."""
    surface = SurfaceModel(field_config, fluid)
    hot = surface.tank_heating_demand(4.0, 0.7, wellhead_temp_c=120.0)
    cold = surface.tank_heating_demand(4.0, 0.7, wellhead_temp_c=35.0)
    assert hot.sensible_heat_j_per_day == pytest.approx(0.0)
    assert cold.sensible_heat_j_per_day > 0.0
    assert hot.total_heat_j_per_day < cold.total_heat_j_per_day


def test_standing_loss_is_always_present(field_config: FieldConfig, fluid: FluidModel) -> None:
    surface = SurfaceModel(field_config, fluid)
    result = surface.tank_heating_demand(0.0, 0.0, wellhead_temp_c=200.0)
    assert result.standing_loss_j_per_day > 0.0
    assert result.total_heat_j_per_day == pytest.approx(result.standing_loss_j_per_day)


def test_steam_generation_energy_scales_with_volume(
    field_config: FieldConfig, fluid: FluidModel
) -> None:
    surface = SurfaceModel(field_config, fluid)
    single = surface.steam_generation_fuel_energy_j(1000.0, 11000.0, 0.78)
    double = surface.steam_generation_fuel_energy_j(2000.0, 11000.0, 0.78)
    assert double == pytest.approx(2.0 * single)


def test_generator_rate_limit_is_consistent_with_the_configured_rate(
    field_config: FieldConfig, fluid: FluidModel
) -> None:
    """The default injection rate must be one a single mobile unit can raise."""
    surface = SurfaceModel(field_config, fluid)
    limit = surface.generator_rate_limit_m3_per_day_cwe(
        field_config.css.injection_pressure_kpa, field_config.css.steam_quality_frac
    )
    assert field_config.css.injection_rate_m3_per_day_cwe <= limit


def test_generator_occupancy_includes_the_rig_move(
    field_config: FieldConfig, fluid: FluidModel
) -> None:
    surface = SurfaceModel(field_config, fluid)
    days = surface.steam_generator_days_required(1800.0, 160.0)
    assert days == pytest.approx(1800.0 / 160.0 + field_config.surface.rig_move_days)


def test_zero_injection_rate_is_rejected(field_config: FieldConfig, fluid: FluidModel) -> None:
    with pytest.raises(PhysicsDomainError):
        SurfaceModel(field_config, fluid).steam_generator_days_required(1800.0, 0.0)


# ---------------------------------------------------------------- coupled engine
def _post_soak_twin(field_config: FieldConfig, **kwargs) -> CoupledWellTwin:  # type: ignore[no-untyped-def]
    """A twin that has been through injection and a soak, so the zone is hot.

    Several tests need a realistic operating state. Calling evaluate_lift on a
    virgin reservoir with a hot sandface temperature is not one: the reservoir
    would deliver almost nothing, the string would cool to the geothermal
    profile and the rods would simply be stuck.
    """
    twin = CoupledWellTwin(field_config, **kwargs)
    plan = InjectionPlan.from_config(field_config)
    injection = twin.wellbore.inject_steam(
        wellhead_pressure_kpa=plan.injection_pressure_kpa,
        wellhead_quality_frac=plan.steam_quality_frac,
        rate_m3_per_day_cwe=plan.injection_rate_m3_per_day_cwe,
        elapsed_days=plan.injection_days,
    )
    twin.reservoir.begin_cycle(1)
    twin.reservoir.step_injection(
        plan, plan.injection_days, injection.sandface_heat_rate_w, injection.sandface_temp_c
    )
    twin.reservoir.step_soak(plan.soak_days)
    twin.reservoir.state.phase_day = 0.0
    return twin


@pytest.fixture(scope="module")
def cycle_result(field_config: FieldConfig):  # type: ignore[no-untyped-def]
    """One full coupled cycle, shared by several tests because it is slow."""
    twin = CoupledWellTwin(field_config, rod_solver_interval_days=10)
    return twin, twin.run_cycle(
        InjectionPlan.from_config(field_config),
        PumpSetpoint.from_config(field_config),
        cycle_number=1,
        max_production_days=120,
    )


def test_cooling_to_viscosity_to_load_chain_is_visible(cycle_result) -> None:  # type: ignore[no-untyped-def]
    """Acceptance criterion for milestone 5.

    One run must show the whole chain: the heated zone cools, the viscosity at
    the pump and along the rods rises, the drag rises with it, the float margin
    falls and the peak polished rod load rises.
    """
    _, result = cycle_result
    assert len(result.days) > 40
    first = result.days[5].values
    last = result.days[-1].values
    assert last["average_heated_temp_c"] < first["average_heated_temp_c"]
    assert last["viscosity_at_pump_cp"] > first["viscosity_at_pump_cp"]
    assert last["viscosity_mid_string_cp"] > first["viscosity_mid_string_cp"]
    assert last["float_margin_index"] < first["float_margin_index"]
    assert last["peak_polished_rod_load_n"] > first["peak_polished_rod_load_n"]
    assert last["oil_rate_m3_per_day"] < first["oil_rate_m3_per_day"]


def test_cycle_kpis_are_plausible(cycle_result) -> None:  # type: ignore[no-untyped-def]
    _, result = cycle_result
    assert result.oil_m3 > 0.0
    assert 1.0 <= result.steam_oil_ratio <= 12.0
    assert result.is_feasible
    assert 0.0 < result.mean_fillage_frac <= 1.0
    assert result.energy_kwh_per_bbl > 0.0
    assert result.revenue_usd > 0.0


def test_every_recorded_value_is_finite(cycle_result) -> None:  # type: ignore[no-untyped-def]
    _, result = cycle_result
    for record in result.days:
        for key, value in record.values.items():
            assert np.isfinite(value), f"{key} is not finite on day {record.day}"


def test_operating_point_regimes(field_config: FieldConfig) -> None:
    """Flowing, pumped with a level, and pumped off must all be reachable."""
    twin = _post_soak_twin(field_config)
    temperature_c = twin.reservoir.state.average_heated_temp_c
    flowing = twin.solve_operating_point(2.0, 1.0, temperature_c, 0.1)
    assert flowing.is_flowing

    pumped = twin.solve_operating_point(1.0e6, 1.0, temperature_c, 0.1)
    assert pumped.is_pumped_off
    assert pumped.bottomhole_pressure_kpa == pytest.approx(twin.minimum_intake_pressure_kpa)

    moderate = twin.solve_operating_point(
        0.5 * twin.reservoir.deliverability_m3_per_day(twin.minimum_intake_pressure_kpa, 1.0),
        1.0,
        temperature_c,
        0.1,
    )
    assert not moderate.is_flowing
    assert not moderate.is_pumped_off
    assert (
        twin.minimum_intake_pressure_kpa
        < moderate.bottomhole_pressure_kpa
        <= twin.maximum_intake_pressure_kpa(temperature_c, 0.1)
    )


def test_intake_pressure_cannot_exceed_a_full_annular_column(
    field_config: FieldConfig,
) -> None:
    twin = CoupledWellTwin(field_config)
    ceiling = twin.maximum_intake_pressure_kpa(200.0, 0.2)
    assert ceiling > twin.minimum_intake_pressure_kpa
    twin.reservoir.state.reservoir_pressure_kpa = 30000.0
    point = twin.solve_operating_point(1.0, 5.0, 200.0, 0.2)
    assert point.bottomhole_pressure_kpa <= ceiling + 1e-6


def test_static_plunger_stroke_matches_the_wave_solver(field_config: FieldConfig) -> None:
    """The closed-form stretch used by the fast path must match the full solver."""
    twin = _post_soak_twin(field_config)
    setpoint = PumpSetpoint.from_config(field_config)
    lift = twin.evaluate_lift(
        setpoint=setpoint,
        sandface_temp_c=twin.reservoir.state.average_heated_temp_c,
        water_cut_frac=twin.reservoir.state.water_cut_frac,
        elapsed_days=5.0,
        production_days=5.0,
    )
    analytic = twin.static_plunger_stroke_m(setpoint.stroke_length_m, lift.fluid_load_n)
    assert analytic == pytest.approx(lift.wave.pump_stroke_m, rel=0.10)


def test_fast_cycle_tracks_the_full_engine(field_config: FieldConfig) -> None:
    """The surrogate used by calibration and the optimizer must stay close.

    The brief requires the surrogate error to be reported rather than assumed.
    """
    plan = InjectionPlan.from_config(field_config)
    setpoint = PumpSetpoint.from_config(field_config)
    twin = CoupledWellTwin(field_config, rod_solver_interval_days=10)
    fast = twin.run_cycle_fast(plan, setpoint, 1, max_production_days=120)
    twin.reset()
    full = twin.run_cycle(plan, setpoint, 1, max_production_days=120, record_days=False)
    assert abs(fast.oil_m3 - full.oil_m3) / full.oil_m3 < 0.12
    assert abs(fast.steam_oil_ratio - full.steam_oil_ratio) / full.steam_oil_ratio < 0.12


def test_slowing_the_downstroke_raises_the_float_margin_in_the_coupled_run(
    field_config: FieldConfig,
) -> None:
    twin = _post_soak_twin(field_config)
    base = twin.evaluate_lift(
        setpoint=PumpSetpoint(3.0, 2.54, SpeedProfile()),
        sandface_temp_c=150.0,
        water_cut_frac=0.2,
        elapsed_days=60.0,
        production_days=60.0,
    )
    slowed = twin.evaluate_lift(
        setpoint=PumpSetpoint(3.0, 2.54, SpeedProfile(1.0, 0.5, 0.0)),
        sandface_temp_c=150.0,
        water_cut_frac=0.2,
        elapsed_days=60.0,
        production_days=60.0,
    )
    assert slowed.float_analysis.minimum_section_margin > base.float_analysis.minimum_section_margin


def test_later_cycles_produce_less(field_config: FieldConfig) -> None:
    twin = CoupledWellTwin(field_config, rod_solver_interval_days=30)
    plan = InjectionPlan.from_config(field_config)
    setpoint = PumpSetpoint.from_config(field_config)
    volumes = [
        twin.run_cycle_fast(plan, setpoint, cycle, max_production_days=90).oil_m3
        for cycle in (1, 2, 3, 4)
    ]
    assert volumes[-1] < volumes[0]


def test_infeasible_plan_is_reported_not_silently_accepted(
    field_config: FieldConfig,
) -> None:
    twin = CoupledWellTwin(field_config)
    plan = InjectionPlan(
        steam_volume_m3_cwe=1800.0,
        injection_rate_m3_per_day_cwe=1500.0,
        injection_pressure_kpa=19000.0,
        steam_quality_frac=0.78,
        soak_days=6.0,
        cutoff_marginal_energy_ratio=0.35,
    )
    result = twin.run_cycle_fast(plan, PumpSetpoint.from_config(field_config), 1, 20)
    assert not result.is_feasible
    assert any("fracture" in note or "generator" in note for note in result.constraint_violations)


def test_twin_parameter_vector_round_trip(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    names = ["permeability_md", "rod_damping_factor"]
    vector = base.to_vector(names)
    rebuilt = TwinParameters.from_vector(names, vector * 2.0, base)
    assert rebuilt.permeability_md == pytest.approx(2.0 * base.permeability_md)
    assert rebuilt.skin_dimensionless == pytest.approx(base.skin_dimensionless)


def test_unknown_twin_parameter_is_rejected(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    with pytest.raises(PhysicsDomainError, match="Unknown twin parameter"):
        TwinParameters.from_vector(["not_a_parameter"], np.array([1.0]), base)


# ----------------------------------------------------------------- assimilation
def test_prior_ensemble_respects_bounds(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    names = ["permeability_md", "skin_dimensionless", "rod_damping_factor"]
    ensemble = build_prior_ensemble(base, names, 64, spread_frac=1.0, seed=1)
    for index, name in enumerate(names):
        low, high = DEFAULT_PARAMETER_BOUNDS[name]
        assert float(np.min(ensemble[:, index])) >= low - 1e-9
        assert float(np.max(ensemble[:, index])) <= high + 1e-9


def test_kalman_update_moves_towards_the_observation() -> None:
    rng = np.random.default_rng(0)
    members = 64
    ensemble = rng.normal(0.0, 1.0, (members, 1))
    predictions = 3.0 * ensemble
    observation = np.array([6.0])
    sigma = np.array([0.1])
    posterior = ensemble_kalman_update(ensemble, predictions, observation, sigma, seed=0)
    assert abs(float(np.mean(posterior)) - 2.0) < abs(float(np.mean(ensemble)) - 2.0)


def test_kalman_update_validates_its_inputs() -> None:
    with pytest.raises(PhysicsDomainError, match="same number of members"):
        ensemble_kalman_update(np.zeros((4, 2)), np.zeros((3, 1)), np.zeros(1), np.ones(1))
    with pytest.raises(PhysicsDomainError, match="counts differ"):
        ensemble_kalman_update(np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(1), np.ones(1))
    with pytest.raises(PhysicsDomainError, match="Inflation"):
        ensemble_kalman_update(
            np.zeros((4, 2)), np.zeros((4, 1)), np.zeros(1), np.ones(1), inflation=0.5
        )


def test_assimilation_reduces_parameter_error(field_config: FieldConfig) -> None:
    """The filter must find identifiable parameters from noisy observations."""
    base = TwinParameters.from_config(field_config)
    names = ["thermal_loss_multiplier", "rod_damping_factor"]
    truth = TwinParameters.from_vector(names, np.array([1.8, 0.55]), base)

    def predict(parameters: TwinParameters) -> list[float]:
        return [
            300.0 / parameters.thermal_loss_multiplier,
            50.0 * parameters.rod_damping_factor,
        ]

    targets = predict(truth)
    observations = [
        Observation("temperature", targets[0], 5.0),
        Observation("damping", targets[1], 1.0),
    ]
    assimilator = EnsembleKalmanAssimilator(base, names, ensemble_size=48, spread_frac=0.5, seed=4)
    prior_error = float(
        np.mean(np.abs(assimilator.prior_mean - truth.to_vector(names)) / truth.to_vector(names))
    )
    for _ in range(5):
        result = assimilator.update(predict, observations)
    posterior_error = float(
        np.mean(np.abs(result.mean - truth.to_vector(names)) / truth.to_vector(names))
    )
    assert posterior_error < 0.25 * prior_error


def test_assimilation_reports_uncertainty_reduction(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    names = ["thermal_loss_multiplier"]

    def predict(parameters: TwinParameters) -> list[float]:
        return [300.0 / parameters.thermal_loss_multiplier]

    observations = [Observation("temperature", 200.0, 4.0)]
    assimilator = EnsembleKalmanAssimilator(base, names, ensemble_size=40, seed=2)
    for _ in range(4):
        result = assimilator.update(predict, observations)
    report = result.uncertainty()["thermal_loss_multiplier"]
    assert report["uncertainty_reduction_frac"] > 0.3
    assert report["standard_deviation"] < report["prior_standard_deviation"]


def test_percentile_parameters_are_ordered(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    names = ["permeability_md"]

    def predict(parameters: TwinParameters) -> list[float]:
        return [parameters.permeability_md * 0.01]

    assimilator = EnsembleKalmanAssimilator(base, names, ensemble_size=40, seed=6)
    result = assimilator.update(predict, [Observation("rate", 12.0, 2.0)])
    low = result.percentile_parameters(base, 10.0).permeability_md
    high = result.percentile_parameters(base, 90.0).permeability_md
    assert low < high


def test_least_squares_fallback_finds_the_answer(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    names = ["thermal_loss_multiplier"]

    def predict(parameters: TwinParameters) -> list[float]:
        return [300.0 / parameters.thermal_loss_multiplier]

    result = rolling_least_squares_refit(
        base, names, predict, [Observation("temperature", 150.0, 2.0)]
    )
    assert result.method == "rolling_least_squares"
    assert float(result.mean[0]) == pytest.approx(2.0, rel=0.05)


def test_assimilator_rejects_unknown_parameters(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    with pytest.raises(PhysicsDomainError, match="Unknown twin parameters"):
        EnsembleKalmanAssimilator(base, ["not_a_parameter"])


def test_assimilator_requires_observations(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    assimilator = EnsembleKalmanAssimilator(base, ["rod_damping_factor"], ensemble_size=8)
    with pytest.raises(PhysicsDomainError, match="At least one observation"):
        assimilator.update(lambda _parameters: [0.0], [])


def test_observation_rejects_a_non_positive_sigma() -> None:
    with pytest.raises(PhysicsDomainError):
        Observation("a", 1.0, 0.0)


def test_parameters_never_leave_their_bounds(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    names = ["permeability_md"]

    def predict(parameters: TwinParameters) -> list[float]:
        return [parameters.permeability_md * 1.0e6]

    assimilator = EnsembleKalmanAssimilator(base, names, ensemble_size=32, seed=9)
    for _ in range(6):
        result = assimilator.update(predict, [Observation("silly", 1.0e12, 1.0)])
    low, high = DEFAULT_PARAMETER_BOUNDS["permeability_md"]
    assert low <= float(result.mean[0]) <= high


def test_monotone_ensemble_spread_shrinks(field_config: FieldConfig) -> None:
    base = TwinParameters.from_config(field_config)
    names = ["rod_damping_factor"]

    def predict(parameters: TwinParameters) -> list[float]:
        return [parameters.rod_damping_factor * 10.0]

    assimilator = EnsembleKalmanAssimilator(base, names, ensemble_size=48, seed=3)
    spreads = []
    for _ in range(4):
        result = assimilator.update(predict, [Observation("x", 4.0, 0.2)])
        spreads.append(float(result.standard_deviation[0]))
    assert all(later <= earlier * 1.05 for earlier, later in itertools.pairwise(spreads))
