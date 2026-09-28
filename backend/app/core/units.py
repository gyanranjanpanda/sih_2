"""Unit conversions for the edges of the system.

Everything inside the twin, the optimizers and the truth simulator is SI:
metres, seconds, kilograms, kelvin (temperatures are carried in degrees Celsius
where that is the engineering convention, and converted explicitly), pascals,
cubic metres and newtons. Conversions live here and are applied only in the
ingestion layer, the API serialisation layer and the user interface.

Naming rule used across the repository: every variable that carries a physical
quantity states its unit in the name, for example ``pressure_kpa``,
``rate_m3_per_day``, ``viscosity_pa_s``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.core.errors import UnitError

# --------------------------------------------------------------------------------------
# Exact or standards-defined constants
# --------------------------------------------------------------------------------------
ABSOLUTE_ZERO_C: Final[float] = -273.15
STANDARD_GRAVITY_M_PER_S2: Final[float] = 9.80665
"""Standard acceleration of free fall, CGPM 1901."""

M_PER_FT: Final[float] = 0.3048
"""Exact by definition of the international foot."""
M_PER_IN: Final[float] = 0.0254
M3_PER_BBL: Final[float] = 0.158987294928
"""Exact definition of the US petroleum barrel, 42 US gallons."""
KG_PER_LB: Final[float] = 0.45359237
"""Exact by definition of the international avoirdupois pound."""
N_PER_LBF: Final[float] = KG_PER_LB * STANDARD_GRAVITY_M_PER_S2
PA_PER_PSI: Final[float] = N_PER_LBF / (M_PER_IN**2)
PA_PER_BAR: Final[float] = 1.0e5
PA_PER_ATM: Final[float] = 101325.0
PA_PER_KGF_PER_CM2: Final[float] = STANDARD_GRAVITY_M_PER_S2 * 1.0e4
"""kg/cm2 gauge is still common in Indian field practice; convert explicitly."""
J_PER_KWH: Final[float] = 3.6e6
J_PER_BTU: Final[float] = 1055.05585262
"""International Table Btu."""
W_PER_HP: Final[float] = 745.699871582
SECONDS_PER_DAY: Final[float] = 86400.0
SECONDS_PER_HOUR: Final[float] = 3600.0
DAYS_PER_YEAR: Final[float] = 365.25

Number = float | NDArray[np.float64]

# --------------------------------------------------------------------------------------
# Multiplicative registry: value_in_si = value * factor
# Temperature is affine, so it is handled separately.
# --------------------------------------------------------------------------------------
_TO_SI: Final[dict[str, tuple[str, float]]] = {
    # length -> m
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "ft": ("length", M_PER_FT),
    "in": ("length", M_PER_IN),
    # pressure -> Pa
    "pa": ("pressure", 1.0),
    "kpa": ("pressure", 1.0e3),
    "mpa": ("pressure", 1.0e6),
    "bar": ("pressure", PA_PER_BAR),
    "psi": ("pressure", PA_PER_PSI),
    "atm": ("pressure", PA_PER_ATM),
    "kgf_per_cm2": ("pressure", PA_PER_KGF_PER_CM2),
    # volume -> m3
    "m3": ("volume", 1.0),
    "l": ("volume", 1.0e-3),
    "bbl": ("volume", M3_PER_BBL),
    "ft3": ("volume", M_PER_FT**3),
    # volumetric rate -> m3/s
    "m3_per_s": ("rate", 1.0),
    "m3_per_day": ("rate", 1.0 / SECONDS_PER_DAY),
    "m3_per_hour": ("rate", 1.0 / SECONDS_PER_HOUR),
    "bbl_per_day": ("rate", M3_PER_BBL / SECONDS_PER_DAY),
    # mass -> kg
    "kg": ("mass", 1.0),
    "g": ("mass", 1.0e-3),
    "tonne": ("mass", 1000.0),
    "lb": ("mass", KG_PER_LB),
    # force -> N
    "n": ("force", 1.0),
    "kn": ("force", 1000.0),
    "lbf": ("force", N_PER_LBF),
    "klbf": ("force", 1000.0 * N_PER_LBF),
    # torque -> N.m
    "n_m": ("torque", 1.0),
    "kn_m": ("torque", 1000.0),
    "in_lbf": ("torque", N_PER_LBF * M_PER_IN),
    "ft_lbf": ("torque", N_PER_LBF * M_PER_FT),
    # energy -> J
    "j": ("energy", 1.0),
    "kj": ("energy", 1.0e3),
    "mj": ("energy", 1.0e6),
    "gj": ("energy", 1.0e9),
    "kwh": ("energy", J_PER_KWH),
    "btu": ("energy", J_PER_BTU),
    "mmbtu": ("energy", 1.0e6 * J_PER_BTU),
    # power -> W
    "w": ("power", 1.0),
    "kw": ("power", 1.0e3),
    "mw": ("power", 1.0e6),
    "hp": ("power", W_PER_HP),
    # dynamic viscosity -> Pa.s
    "pa_s": ("dynamic_viscosity", 1.0),
    "cp": ("dynamic_viscosity", 1.0e-3),
    "p": ("dynamic_viscosity", 0.1),
    # kinematic viscosity -> m2/s
    "m2_per_s": ("kinematic_viscosity", 1.0),
    "cst": ("kinematic_viscosity", 1.0e-6),
    "st": ("kinematic_viscosity", 1.0e-4),
    # density -> kg/m3
    "kg_per_m3": ("density", 1.0),
    "g_per_cm3": ("density", 1000.0),
    # time -> s
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "hour": ("time", SECONDS_PER_HOUR),
    "day": ("time", SECONDS_PER_DAY),
    "year": ("time", SECONDS_PER_DAY * DAYS_PER_YEAR),
    # dimensionless
    "frac": ("dimensionless", 1.0),
    "percent": ("dimensionless", 0.01),
}

_TEMPERATURE_UNITS: Final[frozenset[str]] = frozenset({"c", "k", "f", "r"})


def _normalise(unit: str) -> str:
    return unit.strip().lower().replace("-", "_").replace(".", "_").replace("/", "_per_")


def dimension_of(unit: str) -> str:
    """Return the dimension name of a unit, for example ``pressure``.

    Raises:
        UnitError: if the unit is not in the registry.
    """
    key = _normalise(unit)
    if key in _TEMPERATURE_UNITS:
        return "temperature"
    if key not in _TO_SI:
        raise UnitError(f"Unknown unit '{unit}'.", unit=unit)
    return _TO_SI[key][0]


def convert(value: Number, from_unit: str, to_unit: str) -> Number:
    """Convert a value between two units of the same dimension.

    Args:
        value: Magnitude in ``from_unit``. Scalars and numpy arrays both work.
        from_unit: Source unit key, for example ``"psi"``.
        to_unit: Target unit key, for example ``"kpa"``.

    Returns:
        The magnitude expressed in ``to_unit``.

    Raises:
        UnitError: if either unit is unknown or the dimensions differ.
    """
    src, dst = _normalise(from_unit), _normalise(to_unit)
    if src == dst:
        return value
    src_dim, dst_dim = dimension_of(src), dimension_of(dst)
    if src_dim != dst_dim:
        raise UnitError(
            f"Cannot convert {src_dim} '{from_unit}' to {dst_dim} '{to_unit}'.",
            from_unit=from_unit,
            to_unit=to_unit,
        )
    if src_dim == "temperature":
        return _from_kelvin(_to_kelvin(value, src), dst)
    return value * (_TO_SI[src][1] / _TO_SI[dst][1])


def _to_kelvin(value: Number, unit: str) -> Number:
    if unit == "k":
        return value
    if unit == "c":
        return value - ABSOLUTE_ZERO_C
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0 - ABSOLUTE_ZERO_C
    if unit == "r":
        return value * 5.0 / 9.0
    raise UnitError(f"Unknown temperature unit '{unit}'.", unit=unit)


def _from_kelvin(value_k: Number, unit: str) -> Number:
    if unit == "k":
        return value_k
    if unit == "c":
        return value_k + ABSOLUTE_ZERO_C
    if unit == "f":
        return (value_k + ABSOLUTE_ZERO_C) * 9.0 / 5.0 + 32.0
    if unit == "r":
        return value_k * 9.0 / 5.0
    raise UnitError(f"Unknown temperature unit '{unit}'.", unit=unit)


# --------------------------------------------------------------------------------------
# Named shorthands used often enough to deserve a function
# --------------------------------------------------------------------------------------
def celsius_to_kelvin(temp_c: Number) -> Number:
    """Convert degrees Celsius to kelvin."""
    return temp_c - ABSOLUTE_ZERO_C


def kelvin_to_celsius(temp_k: Number) -> Number:
    """Convert kelvin to degrees Celsius."""
    return temp_k + ABSOLUTE_ZERO_C


def api_to_specific_gravity(api_gravity_deg: float) -> float:
    """Specific gravity at 60 deg F from API gravity.

    Equation: SG = 141.5 / (131.5 + API).
    Units: API gravity in degrees, SG dimensionless relative to water at 60 deg F.
    Source: API Manual of Petroleum Measurement Standards, chapter 11.
    """
    denominator = 131.5 + api_gravity_deg
    if denominator <= 0.0:
        raise UnitError(
            "API gravity of -131.5 degrees or below has no physical meaning.",
            api_gravity_deg=api_gravity_deg,
        )
    return 141.5 / denominator


def specific_gravity_to_api(specific_gravity: float) -> float:
    """API gravity in degrees from specific gravity at 60 deg F."""
    if specific_gravity <= 0.0:
        raise UnitError("Specific gravity must be positive.", specific_gravity=specific_gravity)
    return 141.5 / specific_gravity - 131.5


def centistokes_to_centipoise(viscosity_cst: Number, density_kg_per_m3: Number) -> Number:
    """Dynamic viscosity in cP from kinematic viscosity in cSt and density.

    Equation: mu [cP] = nu [cSt] * rho [g/cm3] = nu [cSt] * rho [kg/m3] / 1000.
    """
    return viscosity_cst * density_kg_per_m3 / 1000.0


def centipoise_to_centistokes(viscosity_cp: Number, density_kg_per_m3: Number) -> Number:
    """Kinematic viscosity in cSt from dynamic viscosity in cP and density."""
    density = np.asarray(density_kg_per_m3, dtype=float)
    if np.any(density <= 0.0):
        raise UnitError("Density must be positive to convert cP to cSt.")
    return viscosity_cp * 1000.0 / density_kg_per_m3


def pressure_gradient_kpa_per_m(density_kg_per_m3: Number) -> Number:
    """Hydrostatic gradient in kPa/m for a fluid of the given density."""
    return density_kg_per_m3 * STANDARD_GRAVITY_M_PER_S2 / 1000.0
