"""Small numerical helpers with loud failure on non-finite values.

The brief requires that no NaN or infinity ever passes silently through the
twin. Every solver in this repository routes its outputs through
:func:`require_finite` so a numerical failure is reported at the place it
happened rather than surfacing as a strange chart later.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.core.errors import NumericalError, PhysicsDomainError

T = TypeVar("T", float, NDArray[np.float64])


def require_finite(value: T, name: str, **context: Any) -> T:
    """Return ``value`` unchanged, or raise if it holds a NaN or an infinity."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise NumericalError(
            f"{name} contains {bad} non-finite value(s).",
            quantity=name,
            non_finite_count=bad,
            **context,
        )
    return value


def require_positive(value: float, name: str, **context: Any) -> float:
    """Return ``value`` unchanged, or raise if it is not strictly positive."""
    if not np.isfinite(value) or value <= 0.0:
        raise PhysicsDomainError(
            f"{name} must be strictly positive, received {value}.",
            quantity=name,
            value=value,
            **context,
        )
    return value


def require_range(value: float, name: str, low: float, high: float, **context: Any) -> float:
    """Return ``value`` unchanged, or raise if it falls outside ``[low, high]``."""
    if not np.isfinite(value) or not (low <= value <= high):
        raise PhysicsDomainError(
            f"{name} must lie in [{low}, {high}], received {value}.",
            quantity=name,
            value=value,
            low=low,
            high=high,
            **context,
        )
    return value


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar into a closed interval."""
    if low > high:
        raise PhysicsDomainError(
            f"clamp received an inverted interval [{low}, {high}].", low=low, high=high
        )
    return min(max(value, low), high)


def safe_divide(
    numerator: ArrayLike, denominator: ArrayLike, fallback: float = 0.0
) -> NDArray[np.float64]:
    """Element-wise division that yields ``fallback`` where the denominator is zero."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, fallback, dtype=float)
    nonzero = den != 0.0
    np.divide(num, den, out=out, where=nonzero)
    return out


def trapezoid(values: ArrayLike, step: float) -> float:
    """Trapezoidal integral of evenly spaced samples."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    return float(np.trapezoid(array, dx=step))


def moving_average(values: ArrayLike, window: int) -> NDArray[np.float64]:
    """Centred moving average with edge padding, used for smoothing telemetry."""
    array = np.asarray(values, dtype=float)
    if window <= 1 or array.size == 0:
        return array.astype(float)
    window = min(window, array.size)
    pad = window // 2
    padded = np.pad(array, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="same")
    return np.asarray(smoothed[pad : pad + array.size], dtype=float)
