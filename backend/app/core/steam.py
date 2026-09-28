"""Saturated steam properties.

Primary source is the IAPWS-IF97 formulation through the ``iapws`` package.
A compact built-in saturation table is kept as a fallback so the twin still
runs if the optional dependency is absent, and so the tests can check the two
paths agree. All functions take and return SI values with units in the names.

Source: IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
Thermodynamic Properties of Water and Steam (IF97).
"""

from __future__ import annotations

import bisect
from functools import lru_cache

from app.core.errors import PhysicsDomainError
from app.core.numerics import require_range

try:  # pragma: no cover - import guard, both paths are exercised by tests
    from iapws import IAPWS97 as _IAPWS97

    _HAVE_IAPWS = True
except ImportError:  # pragma: no cover
    _IAPWS97 = None  # type: ignore[assignment]
    _HAVE_IAPWS = False

# Fallback saturation table: pressure (kPa), saturation temperature (C),
# liquid enthalpy (kJ/kg), latent heat of vaporisation (kJ/kg).
# Values from standard steam tables at the listed pressures.
_SAT_TABLE: tuple[tuple[float, float, float, float], ...] = (
    (100.0, 99.61, 417.4, 2257.5),
    (200.0, 120.21, 504.7, 2201.6),
    (500.0, 151.84, 640.1, 2108.0),
    (1000.0, 179.89, 762.5, 2014.6),
    (2000.0, 212.38, 908.6, 1889.8),
    (3000.0, 233.86, 1008.3, 1794.9),
    (4000.0, 250.36, 1087.4, 1713.5),
    (5000.0, 263.94, 1154.5, 1639.7),
    (6000.0, 275.59, 1213.7, 1570.9),
    (8000.0, 295.01, 1317.1, 1441.4),
    (10000.0, 311.00, 1407.6, 1317.1),
    (12000.0, 324.68, 1491.3, 1193.6),
    (14000.0, 336.67, 1571.6, 1066.5),
    (16000.0, 347.36, 1650.5, 930.7),
    (18000.0, 356.99, 1732.1, 777.2),
    (20000.0, 365.75, 1827.1, 583.6),
)

MIN_PRESSURE_KPA = 100.0
MAX_PRESSURE_KPA = 20000.0


def _interpolate(pressure_kpa: float, column: int) -> float:
    pressures = [row[0] for row in _SAT_TABLE]
    index = bisect.bisect_left(pressures, pressure_kpa)
    if index == 0:
        return _SAT_TABLE[0][column]
    if index >= len(_SAT_TABLE):
        return _SAT_TABLE[-1][column]
    low, high = _SAT_TABLE[index - 1], _SAT_TABLE[index]
    weight = (pressure_kpa - low[0]) / (high[0] - low[0])
    return low[column] + weight * (high[column] - low[column])


@lru_cache(maxsize=2048)
def saturation_temperature_c(pressure_kpa: float) -> float:
    """Saturation temperature of water at the given absolute pressure.

    Args:
        pressure_kpa: Absolute pressure, 100 to 20000 kPa.

    Returns:
        Saturation temperature in degrees Celsius.
    """
    require_range(pressure_kpa, "pressure_kpa", MIN_PRESSURE_KPA, MAX_PRESSURE_KPA)
    if _HAVE_IAPWS:
        state = _IAPWS97(P=pressure_kpa / 1000.0, x=0.5)
        return float(state.T) - 273.15
    return _interpolate(pressure_kpa, 1)


@lru_cache(maxsize=2048)
def latent_heat_j_per_kg(pressure_kpa: float) -> float:
    """Latent heat of vaporisation at the given absolute pressure."""
    require_range(pressure_kpa, "pressure_kpa", MIN_PRESSURE_KPA, MAX_PRESSURE_KPA)
    if _HAVE_IAPWS:
        pressure_mpa = pressure_kpa / 1000.0
        liquid = _IAPWS97(P=pressure_mpa, x=0.0)
        vapour = _IAPWS97(P=pressure_mpa, x=1.0)
        return float(vapour.h - liquid.h) * 1000.0
    return _interpolate(pressure_kpa, 3) * 1000.0


@lru_cache(maxsize=2048)
def liquid_enthalpy_j_per_kg(pressure_kpa: float) -> float:
    """Saturated liquid enthalpy at the given absolute pressure, relative to 0 C."""
    require_range(pressure_kpa, "pressure_kpa", MIN_PRESSURE_KPA, MAX_PRESSURE_KPA)
    if _HAVE_IAPWS:
        return float(_IAPWS97(P=pressure_kpa / 1000.0, x=0.0).h) * 1000.0
    return _interpolate(pressure_kpa, 2) * 1000.0


def steam_enthalpy_j_per_kg(pressure_kpa: float, quality_frac: float) -> float:
    """Specific enthalpy of wet steam of the given quality.

    Equation: h = h_f + x * h_fg.
    Units: J/kg, pressure in kPa absolute, quality dimensionless in [0, 1].
    Assumption: the steam is at saturation, which is the usual state at the
    outlet of a once-through mobile steam generator.
    """
    require_range(quality_frac, "quality_frac", 0.0, 1.0)
    return liquid_enthalpy_j_per_kg(pressure_kpa) + quality_frac * latent_heat_j_per_kg(
        pressure_kpa
    )


def injected_heat_rate_w(
    rate_m3_per_day_cwe: float,
    pressure_kpa: float,
    quality_frac: float,
    reference_temp_c: float,
    water_density_kg_per_m3: float = 1000.0,
    water_specific_heat_j_per_kg_k: float = 4186.0,
) -> float:
    """Net heat rate carried by injected steam above the reservoir datum.

    Equation: Q = m_dot * (h_f + x * h_fg - c_w * (T_ref - 0 C)), with the
    mass rate taken from the cold water equivalent volume rate.
    Units: W. Rates in m3/day cold water equivalent, pressure in kPa absolute.
    Assumption: the enthalpy datum is liquid water at 0 C, consistent with the
    IF97 reference state, and the reservoir datum temperature is the initial
    reservoir temperature.
    Source: Marx and Langenheim (1959); Butler, Thermal Recovery of Oil and
    Bitumen (1991), chapter 7.
    """
    if rate_m3_per_day_cwe < 0.0:
        raise PhysicsDomainError(
            "Injection rate cannot be negative.", rate_m3_per_day_cwe=rate_m3_per_day_cwe
        )
    mass_rate_kg_per_s = rate_m3_per_day_cwe * water_density_kg_per_m3 / 86400.0
    enthalpy_j_per_kg = steam_enthalpy_j_per_kg(pressure_kpa, quality_frac)
    datum_enthalpy_j_per_kg = water_specific_heat_j_per_kg_k * reference_temp_c
    return mass_rate_kg_per_s * max(enthalpy_j_per_kg - datum_enthalpy_j_per_kg, 0.0)


def iapws_available() -> bool:
    """Whether the IF97 implementation is installed."""
    return _HAVE_IAPWS
