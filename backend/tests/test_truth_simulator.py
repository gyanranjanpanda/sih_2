"""Truth simulator tests: reproducibility, physics, and difference from the twin."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.simulate.truth.priors import (
    HiddenParameters,
    draw_hidden_parameters,
    load_truth_priors,
    read_sealed_parameters,
    seal_parameters,
)
from app.simulate.truth.reservoir_fv import AxisymmetricThermalReservoir, build_grid
from app.simulate.truth.sensors import SensorModel, SensorSettings
from app.simulate.truth.well import TruthSetpoint, TruthWell
from app.simulate.truth.wellbore_fd import TransientWellbore, solve_tridiagonal_batch
from app.twin.reservoir import InjectionPlan


@pytest.fixture(scope="module")
def priors() -> dict:
    """The truth priors file."""
    return load_truth_priors()


@pytest.fixture
def hidden(priors: dict) -> HiddenParameters:
    """Hidden parameters for one well."""
    return draw_hidden_parameters("BGW-TEST", 0, priors)


# ------------------------------------------------------------------------ priors
def test_hidden_parameters_are_reproducible(priors: dict) -> None:
    first = draw_hidden_parameters("BGW-01", 0, priors)
    second = draw_hidden_parameters("BGW-01", 0, priors)
    assert first.to_dict() == second.to_dict()


def test_different_wells_get_different_parameters(priors: dict) -> None:
    first = draw_hidden_parameters("BGW-01", 0, priors)
    second = draw_hidden_parameters("BGW-02", 1, priors)
    assert first.permeability_md != second.permeability_md


def test_adding_a_well_does_not_change_earlier_wells(priors: dict) -> None:
    """The seed is per well index, so the dataset is stable as it grows."""
    before = draw_hidden_parameters("BGW-03", 2, priors).to_dict()
    _ = draw_hidden_parameters("BGW-09", 8, priors)
    after = draw_hidden_parameters("BGW-03", 2, priors).to_dict()
    assert before == after


def test_priors_respect_their_bounds(priors: dict) -> None:
    for index in range(12):
        drawn = draw_hidden_parameters(f"BGW-{index:02d}", index, priors)
        assert 10000.0 <= drawn.viscosity_50c_cp <= 13000.0
        assert 120.0 <= drawn.permeability_md <= 2200.0
        assert 5.0 <= drawn.net_pay_thickness_m <= 22.0
        assert drawn.unit_type in {"conventional", "hydraulic"}


def test_layer_thicknesses_sum_to_the_net_pay(hidden: HiddenParameters) -> None:
    assert float(np.sum(hidden.layer_thicknesses_m())) == pytest.approx(
        hidden.net_pay_thickness_m, rel=1e-9
    )


def test_fracture_streak_raises_one_layer_permeability(priors: dict) -> None:
    fractured = [draw_hidden_parameters(f"BGW-{index:02d}", index, priors) for index in range(24)]
    with_streak = [item for item in fractured if item.fracture_streak_present]
    assert with_streak, "no well drew a fracture streak in 24 draws"
    for item in with_streak:
        permeabilities = item.layer_permeabilities_md()
        assert float(np.max(permeabilities)) > 3.0 * float(np.median(permeabilities))


def test_sealed_parameters_round_trip(hidden: HiddenParameters, tmp_path) -> None:
    seal_parameters([hidden], tmp_path)
    recovered = read_sealed_parameters(tmp_path)
    assert len(recovered) == 1
    assert recovered[0].permeability_md == pytest.approx(hidden.permeability_md)
    assert len(recovered[0].layers) == len(hidden.layers)


# -------------------------------------------------------------------------- grid
def test_grid_covers_the_pay_and_the_burden(hidden: HiddenParameters) -> None:
    grid = build_grid(0.108, 150.0, hidden.layer_thicknesses_m())
    assert grid.radius_face_m[0] == pytest.approx(0.108)
    assert grid.radius_face_m[-1] == pytest.approx(150.0)
    assert float(np.sum(grid.is_pay)) > 0
    assert float(np.min(grid.depth_centre_m)) < 0.0
    assert float(np.max(grid.depth_centre_m)) > float(np.sum(hidden.layer_thicknesses_m()))


def test_grid_volumes_sum_to_the_cylinder(hidden: HiddenParameters) -> None:
    grid = build_grid(0.108, 150.0, hidden.layer_thicknesses_m())
    height = float(np.max(grid.depth_face_m) - np.min(grid.depth_face_m))
    expected = np.pi * (150.0**2 - 0.108**2) * height
    assert float(np.sum(grid.volume_m3)) == pytest.approx(expected, rel=1e-9)


def test_grid_rejects_a_useless_resolution(hidden: HiddenParameters) -> None:
    with pytest.raises(PhysicsDomainError):
        build_grid(0.108, 150.0, hidden.layer_thicknesses_m(), radial_cells=4)


# ---------------------------------------------------------------------- reservoir
def test_injection_heats_the_near_well_region(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    reservoir = AxisymmetricThermalReservoir(field_config, hidden)
    before = reservoir.near_well_temperature_c()
    reservoir.step_injection(5.0, 4.0e6, 310.0, 160.0)
    assert reservoir.near_well_temperature_c() > before
    assert reservoir.heated_radius_m() > field_config.reservoir.wellbore_radius_m


def test_conduction_conserves_energy_to_the_far_field(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    """With Dirichlet outer boundaries heat leaves, so stored energy must fall."""
    reservoir = AxisymmetricThermalReservoir(field_config, hidden)
    reservoir.step_injection(8.0, 4.0e6, 310.0, 160.0)
    stored = [reservoir.stored_heat_j()]
    for _ in range(4):
        reservoir.step_conduction(10.0)
        stored.append(reservoir.stored_heat_j())
    assert all(later < earlier for earlier, later in itertools.pairwise(stored))


def test_production_cools_the_near_well_region(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    reservoir = AxisymmetricThermalReservoir(field_config, hidden)
    reservoir.step_injection(10.0, 4.0e6, 310.0, 160.0)
    hot = reservoir.near_well_temperature_c()
    for _ in range(20):
        reservoir.step_production(1.0, 400.0, 6.0)
    assert reservoir.near_well_temperature_c() < hot


def test_deliverability_falls_through_a_production_cycle(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    """Deliverability decays as the cycle runs.

    Note that the criterion has to be over a production period, not a pure soak.
    Conduction alone can raise deliverability for a while, because it carries
    heat from a small very hot zone into the cold annulus where the flow
    resistance actually is. That is why soaking helps, and it is a real
    behaviour of the resolved model rather than a defect.
    """
    reservoir = AxisymmetricThermalReservoir(field_config, hidden)
    reservoir.step_injection(10.0, 4.0e6, 310.0, 160.0)
    rates = []
    for day in range(90):
        step = reservoir.step_production(1.0, 400.0, 8.0)
        if day in {5, 30, 60, 89}:
            rates.append(step["total_liquid_rate_m3_per_day"])
    assert rates[-1] < rates[0]
    assert all(later <= earlier * 1.05 for earlier, later in itertools.pairwise(rates))


def test_soaking_can_raise_deliverability_before_it_falls(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    """A soak spreads heat outward into the resistance, which is why it helps."""
    reservoir = AxisymmetricThermalReservoir(field_config, hidden)
    reservoir.step_injection(10.0, 4.0e6, 310.0, 160.0)
    reservoir.production_day = 10.0
    before_soak = reservoir.deliverability_m3_per_day(400.0)
    for _ in range(7):
        reservoir.step_conduction(1.0)
    assert reservoir.deliverability_m3_per_day(400.0) > before_soak


def test_transient_ring_count_grows_with_production_time(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    reservoir = AxisymmetricThermalReservoir(field_config, hidden)
    reservoir.step_injection(10.0, 4.0e6, 310.0, 160.0)
    counts = []
    for days in (0.5, 5.0, 40.0, 200.0):
        reservoir.production_day = days
        counts.append(int(np.sum(reservoir.transient_ring_count())))
    assert all(later >= earlier for earlier, later in itertools.pairwise(counts))
    assert counts[-1] > counts[0]


def test_gravity_override_biases_steam_upward(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    from dataclasses import replace

    neutral = replace(hidden, gravity_override_strength_frac=0.0)
    biased = replace(hidden, gravity_override_strength_frac=0.3)
    neutral_weights = AxisymmetricThermalReservoir(field_config, neutral).layer_mobility_weights()
    biased_weights = AxisymmetricThermalReservoir(field_config, biased).layer_mobility_weights()
    assert biased_weights[0] > neutral_weights[0]
    assert biased_weights[-1] < neutral_weights[-1]
    assert float(np.sum(biased_weights)) == pytest.approx(1.0)


def test_no_negative_rates_or_non_finite_values(
    field_config: FieldConfig, hidden: HiddenParameters
) -> None:
    reservoir = AxisymmetricThermalReservoir(field_config, hidden)
    reservoir.step_injection(10.0, 4.0e6, 310.0, 160.0)
    for _ in range(30):
        step = reservoir.step_production(1.0, 400.0, 8.0)
        assert step["oil_rate_m3_per_day"] >= 0.0
        assert np.all(np.isfinite(reservoir.temperature_c))


# ----------------------------------------------------------------------- wellbore
def test_tridiagonal_batch_matches_a_dense_solve() -> None:
    rng = np.random.default_rng(3)
    systems, size = 5, 9
    lower = np.zeros((systems, size))
    upper = np.zeros((systems, size))
    diagonal = rng.uniform(4.0, 6.0, (systems, size))
    lower[:, 1:] = rng.uniform(-1.0, -0.4, (systems, size - 1))
    upper[:, :-1] = rng.uniform(-1.0, -0.4, (systems, size - 1))
    right = rng.normal(0.0, 1.0, (systems, size))
    solution = solve_tridiagonal_batch(lower, diagonal, upper, right)
    for index in range(systems):
        dense = (
            np.diag(diagonal[index]) + np.diag(lower[index, 1:], -1) + np.diag(upper[index, :-1], 1)
        )
        assert np.allclose(dense @ solution[index], right[index], atol=1e-9)


def test_transient_wellbore_vit_beats_bare(field_config: FieldConfig) -> None:
    vit = TransientWellbore(field_config, 0.6).inject_steam(11000.0, 0.78, 160.0, 11.0)
    bare = TransientWellbore(field_config, 17.0).inject_steam(11000.0, 0.78, 160.0, 11.0)
    assert vit.sandface_quality_frac > bare.sandface_quality_frac
    assert vit.heat_loss_fraction < bare.heat_loss_fraction


def test_transient_production_profile_is_monotone(field_config: FieldConfig) -> None:
    wellbore = TransientWellbore(field_config, 0.6)
    for _ in range(10):
        result = wellbore.produce(300.0, 5.0, 0.15, 1.0, 1100.0)
    assert np.all(np.diff(result.temperature_c) >= -1e-9)
    assert result.wellhead_temp_c <= result.pump_intake_temp_c + 1e-9


def test_vit_degradation_raises_heat_loss(field_config: FieldConfig) -> None:
    wellbore = TransientWellbore(field_config, 0.6, degradation_per_year_frac=0.3)
    assert wellbore.heat_transfer_coefficient(2.0) > wellbore.heat_transfer_coefficient(0.0)


def test_formation_warms_up_and_the_loss_falls(field_config: FieldConfig) -> None:
    wellbore = TransientWellbore(field_config, 3.0)
    first = float(np.sum(wellbore.heat_loss_w_per_m(np.full(wellbore.depth_m.size, 250.0))))
    for _ in range(40):
        wellbore.produce(250.0, 5.0, 0.15, 1.0, 1100.0)
    later = float(np.sum(wellbore.heat_loss_w_per_m(np.full(wellbore.depth_m.size, 250.0))))
    assert later < first


# --------------------------------------------------------------------- sensors
def test_sensor_model_is_reproducible(priors: dict) -> None:
    settings = SensorSettings.from_dict(priors["sensors"])
    clean = np.linspace(10.0, 20.0, 400)
    first = SensorModel(settings, seed=5).apply("a", "rate", clean)
    second = SensorModel(settings, seed=5).apply("a", "rate", clean)
    assert np.allclose(first, second, equal_nan=True)


def test_sensor_model_injects_faults_and_logs_them(priors: dict) -> None:
    settings = SensorSettings.from_dict(priors["sensors"])
    model = SensorModel(settings, seed=11)
    clean = np.linspace(10.0, 20.0, 5000)
    measured = model.apply("channel", "rate", clean)
    assert measured.shape == clean.shape
    assert model.log.total() > 0
    assert int(np.count_nonzero(np.isnan(measured))) > 0


def test_clean_channels_get_noise_but_no_faults(priors: dict) -> None:
    settings = SensorSettings.from_dict(priors["sensors"])
    model = SensorModel(settings, seed=13)
    clean = np.linspace(10.0, 20.0, 2000)
    measured = model.apply("clean", "rate", clean, allow_faults=False)
    assert not np.any(np.isnan(measured))
    assert not np.allclose(measured, clean)


def test_drift_grows_with_age(priors: dict) -> None:
    settings = SensorSettings.from_dict(priors["sensors"])
    clean = np.full(500, 100.0)
    young = SensorModel(settings, seed=2).apply(
        "a", "rate", clean, elapsed_years=np.zeros(500), allow_faults=False
    )
    old = SensorModel(settings, seed=2).apply(
        "a", "rate", clean, elapsed_years=np.full(500, 5.0), allow_faults=False
    )
    assert float(np.nanmean(old)) > float(np.nanmean(young))


def test_unknown_channel_kind_is_rejected(priors: dict) -> None:
    settings = SensorSettings.from_dict(priors["sensors"])
    with pytest.raises(PhysicsDomainError):
        SensorModel(settings, seed=1).apply("a", "not_a_channel", np.zeros(10))


# ------------------------------------------------------------------------- well
@pytest.mark.slow
def test_truth_well_produces_a_plausible_cycle(
    field_config: FieldConfig, hidden: HiddenParameters, priors: dict
) -> None:
    well = TruthWell(
        field_config,
        hidden,
        SensorSettings.from_dict(priors["sensors"]),
        priors["faults"],
    )
    result = well.run_history(
        cycles=1,
        plan=InjectionPlan.from_config(field_config),
        setpoint=TruthSetpoint(spm=2.0, stroke_length_m=2.54),
        max_production_days=60,
    )
    assert result.cycle_count() == 1
    cycle = result.cycle_rows[0]
    assert cycle["oil_m3"] > 0.0
    assert 0.5 <= cycle["steam_oil_ratio"] <= 40.0
    rates = [row["oil_rate_m3_per_day"] for row in result.daily_rows]
    assert rates[-1] < max(rates)
    assert result.card_rows
    assert result.telemetry_rows


@pytest.mark.slow
def test_slowing_the_pump_reduces_float_days(
    field_config: FieldConfig, hidden: HiddenParameters, priors: dict
) -> None:
    """The headline result the pump optimizer is built to exploit."""
    plan = InjectionPlan.from_config(field_config)

    def float_days(spm: float) -> int:
        well = TruthWell(
            field_config,
            hidden,
            SensorSettings.from_dict(priors["sensors"]),
            priors["faults"],
        )
        outcome = well.run_history(
            cycles=1,
            plan=plan,
            setpoint=TruthSetpoint(spm=spm, stroke_length_m=2.54),
            max_production_days=45,
        )
        return int(outcome.cycle_rows[0]["float_event_days"])

    assert float_days(1.2) <= float_days(3.0)
