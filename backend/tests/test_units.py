"""Unit conversion tests, including property tests for round trips."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.errors import UnitError
from app.core.units import (
    api_to_specific_gravity,
    celsius_to_kelvin,
    centipoise_to_centistokes,
    centistokes_to_centipoise,
    convert,
    dimension_of,
    kelvin_to_celsius,
    pressure_gradient_kpa_per_m,
    specific_gravity_to_api,
)


def test_known_conversions_match_published_values() -> None:
    assert convert(1.0, "bbl", "m3") == pytest.approx(0.158987294928, rel=1e-12)
    assert convert(1.0, "psi", "kpa") == pytest.approx(6.894757, rel=1e-6)
    assert convert(1.0, "ft", "m") == pytest.approx(0.3048, rel=1e-12)
    assert convert(1.0, "kwh", "j") == pytest.approx(3.6e6, rel=1e-12)
    assert convert(1.0, "hp", "w") == pytest.approx(745.6998716, rel=1e-6)
    assert convert(1.0, "cp", "pa_s") == pytest.approx(1.0e-3, rel=1e-12)


def test_temperature_conversions_are_affine() -> None:
    assert convert(0.0, "c", "k") == pytest.approx(273.15)
    assert convert(32.0, "f", "c") == pytest.approx(0.0, abs=1e-9)
    assert convert(212.0, "f", "c") == pytest.approx(100.0, abs=1e-9)
    assert convert(-40.0, "c", "f") == pytest.approx(-40.0, abs=1e-9)
    assert kelvin_to_celsius(celsius_to_kelvin(47.0)) == pytest.approx(47.0)


def test_indian_field_units_are_supported() -> None:
    """kg/cm2 is still used in Indian field practice and must convert explicitly."""
    assert convert(1.0, "kgf_per_cm2", "kpa") == pytest.approx(98.0665, rel=1e-9)


def test_mismatched_dimensions_are_rejected() -> None:
    with pytest.raises(UnitError, match="Cannot convert"):
        convert(1.0, "m", "kpa")


def test_unknown_unit_is_rejected() -> None:
    with pytest.raises(UnitError, match="Unknown unit"):
        convert(1.0, "furlong", "m")


def test_dimension_lookup() -> None:
    assert dimension_of("kpa") == "pressure"
    assert dimension_of("C") == "temperature"
    assert dimension_of("m3_per_day") == "rate"


def test_api_gravity_round_trip() -> None:
    for api in (10.0, 17.0, 19.0, 35.0):
        assert specific_gravity_to_api(api_to_specific_gravity(api)) == pytest.approx(api)


def test_api_gravity_of_water_is_ten() -> None:
    assert api_to_specific_gravity(10.0) == pytest.approx(1.0, rel=1e-3)


def test_api_gravity_domain_guard() -> None:
    with pytest.raises(UnitError):
        api_to_specific_gravity(-140.0)


def test_viscosity_conversion_round_trip() -> None:
    density = 946.5
    assert centistokes_to_centipoise(
        centipoise_to_centistokes(11500.0, density), density
    ) == pytest.approx(11500.0)


def test_hydrostatic_gradient_of_water() -> None:
    assert pressure_gradient_kpa_per_m(1000.0) == pytest.approx(9.80665, rel=1e-9)


def test_array_conversion_is_elementwise() -> None:
    values = np.array([1.0, 2.0, 3.0])
    converted = convert(values, "bbl", "m3")
    assert isinstance(converted, np.ndarray)
    assert converted[1] == pytest.approx(2.0 * 0.158987294928)


@settings(max_examples=200, deadline=None)
@given(
    value=st.floats(min_value=-1.0e6, max_value=1.0e6, allow_nan=False, allow_infinity=False),
    pair=st.sampled_from(
        [("kpa", "psi"), ("m3", "bbl"), ("m", "ft"), ("j", "kwh"), ("n", "lbf"), ("c", "f")]
    ),
)
def test_round_trip_is_identity(value: float, pair: tuple[str, str]) -> None:
    """Property: converting there and back returns the original value."""
    source, target = pair
    result = convert(convert(value, source, target), target, source)
    assert math.isclose(result, value, rel_tol=1e-9, abs_tol=1e-6)


@settings(max_examples=100, deadline=None)
@given(
    low=st.floats(min_value=1.0, max_value=1.0e4),
    span=st.floats(min_value=0.1, max_value=1.0e4),
)
def test_conversion_is_monotone(low: float, span: float) -> None:
    """Property: a positive-factor conversion preserves ordering."""
    high = low + span
    assert convert(low, "kpa", "psi") < convert(high, "kpa", "psi")
