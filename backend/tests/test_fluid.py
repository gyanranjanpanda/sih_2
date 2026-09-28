"""Fluid property tests against analytic and limiting cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.twin.fluid import (
    WALTHER_OFFSET,
    FluidModel,
    fit_arrhenius,
    fit_walther,
    water_viscosity_pa_s,
)


def test_anchor_viscosity_is_reproduced_exactly(fluid: FluidModel) -> None:
    """The 50 degree C anchor must return the configured value."""
    for anchor in fluid.config.viscosity_anchors:
        recovered_cp = float(fluid.dead_oil_viscosity_pa_s(anchor.temp_c)) * 1000.0
        assert recovered_cp == pytest.approx(anchor.viscosity_cp, rel=1.0e-6)


def test_walther_fit_reproduces_a_synthetic_table_within_one_percent() -> None:
    """A table generated from a known Walther line must be recovered to 1 percent."""
    truth_a, truth_b = 9.1, 3.4
    temps_c = [20.0, 60.0, 110.0, 180.0, 260.0]
    kinematic = []
    for temp_c in temps_c:
        inner = truth_a - truth_b * math.log10(temp_c + 273.15)
        kinematic.append(10.0 ** (10.0**inner) - WALTHER_OFFSET)
    fitted = fit_walther(temps_c, kinematic)
    for temp_c, expected in zip(temps_c, kinematic, strict=True):
        predicted = float(fitted.kinematic_viscosity_cst(temp_c))
        assert predicted == pytest.approx(expected, rel=0.01)


def test_viscosity_decreases_strictly_with_temperature(fluid: FluidModel) -> None:
    temps = np.linspace(20.0, 320.0, 60)
    viscosity = np.asarray(fluid.dead_oil_viscosity_pa_s(temps))
    assert np.all(np.diff(viscosity) < 0.0)


def test_viscosity_at_fifty_degrees_is_inside_the_published_range(fluid: FluidModel) -> None:
    """Sanity check from the brief: 10000 to 13000 cP at 50 degrees C."""
    viscosity_cp = float(fluid.dead_oil_viscosity_pa_s(50.0)) * 1000.0
    assert 10000.0 <= viscosity_cp <= 13000.0


def test_viscosity_at_one_hundred_fifty_degrees_is_two_orders_lower(fluid: FluidModel) -> None:
    """Sanity check from the brief."""
    cold = float(fluid.dead_oil_viscosity_pa_s(50.0))
    hot = float(fluid.dead_oil_viscosity_pa_s(150.0))
    assert cold / hot >= 100.0


def test_temperature_below_absolute_zero_is_rejected(fluid: FluidModel) -> None:
    with pytest.raises(PhysicsDomainError):
        fluid.dead_oil_viscosity_pa_s(-300.0)


def test_negative_water_cut_is_rejected(fluid: FluidModel) -> None:
    with pytest.raises(PhysicsDomainError, match="Water cut"):
        fluid.viscosity_pa_s(100.0, -0.1)


def test_water_cut_above_one_is_rejected(fluid: FluidModel) -> None:
    with pytest.raises(PhysicsDomainError, match="Water cut"):
        fluid.viscosity_pa_s(100.0, 1.4)


def test_emulsion_uplift_rises_to_the_inversion_point_then_falls(fluid: FluidModel) -> None:
    inversion = fluid.config.emulsion_inversion_water_cut_frac
    below = np.linspace(0.0, inversion, 25)
    multipliers = np.asarray(fluid.emulsion_multiplier(below))
    assert np.all(np.diff(multipliers) > 0.0)
    assert float(fluid.emulsion_multiplier(0.0)) == pytest.approx(1.0)
    assert float(fluid.emulsion_multiplier(0.95)) < float(fluid.emulsion_multiplier(inversion))
    assert float(fluid.emulsion_multiplier(1.0)) == pytest.approx(1.0, rel=1e-6)


def test_zero_water_cut_leaves_viscosity_unchanged(fluid: FluidModel) -> None:
    assert float(fluid.viscosity_pa_s(120.0, 0.0)) == pytest.approx(
        float(fluid.dead_oil_viscosity_pa_s(120.0))
    )


def test_specific_gravity_matches_api_correlation(
    fluid: FluidModel, field_config: FieldConfig
) -> None:
    expected = 141.5 / (131.5 + field_config.fluid.api_gravity_deg)
    assert fluid.specific_gravity == pytest.approx(expected)


def test_density_falls_as_temperature_rises(fluid: FluidModel) -> None:
    assert float(fluid.oil_density_kg_per_m3(250.0)) < float(fluid.oil_density_kg_per_m3(20.0))


def test_mixture_density_is_bounded_by_the_two_phases(fluid: FluidModel) -> None:
    oil = float(fluid.oil_density_kg_per_m3(80.0))
    water = fluid.config.water_density_kg_per_m3
    mixed = float(fluid.mixture_density_kg_per_m3(80.0, 0.4))
    assert min(oil, water) <= mixed <= max(oil, water)


def test_water_viscosity_matches_reference_values() -> None:
    """The Vogel fit should be within a few percent of tabulated water viscosity."""
    assert float(water_viscosity_pa_s(20.0)) == pytest.approx(1.002e-3, rel=0.05)
    assert float(water_viscosity_pa_s(100.0)) == pytest.approx(0.282e-3, rel=0.10)


def test_arrhenius_fit_reproduces_its_anchors() -> None:
    temps = [50.0, 200.0]
    dynamic = [11.5, 0.012]
    fitted = fit_arrhenius(temps, dynamic)
    for temp_c, expected in zip(temps, dynamic, strict=True):
        assert float(fitted.dynamic_viscosity_pa_s(temp_c)) == pytest.approx(expected, rel=1e-6)


def test_arrhenius_model_is_selectable(field_config: FieldConfig) -> None:
    arrhenius_config = field_config.with_overrides({"fluid": {"viscosity_model": "arrhenius"}})
    model = FluidModel(arrhenius_config.fluid)
    for anchor in arrhenius_config.fluid.viscosity_anchors:
        assert float(model.dead_oil_viscosity_pa_s(anchor.temp_c)) * 1000.0 == pytest.approx(
            anchor.viscosity_cp, rel=1e-6
        )


def test_asphaltene_multiplier_is_off_by_default_and_applies_when_enabled(
    field_config: FieldConfig,
) -> None:
    base = FluidModel(field_config.fluid)
    enabled = FluidModel(
        field_config.with_overrides(
            {"fluid": {"asphaltene_enabled": True, "asphaltene_multiplier": 2.0}}
        ).fluid
    )
    assert float(enabled.dead_oil_viscosity_pa_s(100.0)) == pytest.approx(
        2.0 * float(base.dead_oil_viscosity_pa_s(100.0))
    )


def test_fit_requires_at_least_two_anchors() -> None:
    with pytest.raises(PhysicsDomainError, match="At least two anchors"):
        fit_walther([50.0], [12000.0])


def test_fit_rejects_non_positive_viscosity() -> None:
    with pytest.raises(PhysicsDomainError, match="strictly positive"):
        fit_walther([50.0, 200.0], [12000.0, 0.0])


def test_temperature_for_viscosity_inverts_the_model(fluid: FluidModel) -> None:
    target_pa_s = 0.5
    temp_c = fluid.temperature_for_viscosity_c(target_pa_s)
    assert float(fluid.viscosity_pa_s(temp_c)) == pytest.approx(target_pa_s, rel=0.01)


def test_mixture_specific_heat_lies_between_oil_and_water(fluid: FluidModel) -> None:
    value = float(fluid.mixture_specific_heat_j_per_kg_k(0.5))
    assert (
        fluid.config.oil_specific_heat_j_per_kg_k
        < value
        < fluid.config.water_specific_heat_j_per_kg_k
    )


# The fluid model is stateless and immutable, so reusing one instance across
# generated examples cannot leak state between them.
@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    cold=st.floats(min_value=10.0, max_value=150.0),
    delta=st.floats(min_value=1.0, max_value=150.0),
)
def test_viscosity_monotonicity_property(fluid: FluidModel, cold: float, delta: float) -> None:
    """Property: viscosity is strictly monotone decreasing in temperature."""
    hot = cold + delta
    assert float(fluid.dead_oil_viscosity_pa_s(hot)) < float(fluid.dead_oil_viscosity_pa_s(cold))
