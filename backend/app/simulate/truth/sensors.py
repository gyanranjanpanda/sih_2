"""Sensor model: noise, drift, dropouts, stuck values, spikes and unit errors.

Real field telemetry is not clean. If the twin is only ever tested on clean
data it will look better than it is, and the quality-assurance layer in
``app.ingestion`` will never be exercised. Every fault here is seeded, so a
dataset is reproducible, and every fault is recorded so the anomaly detector
can be scored against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.errors import PhysicsDomainError


@dataclass(frozen=True)
class SensorSettings:
    """Fault rates and magnitudes, loaded from ``config/truth_priors.yaml``."""

    load_noise_n_sigma: float
    position_noise_m_sigma: float
    temperature_noise_c_sigma: float
    pressure_noise_kpa_sigma: float
    rate_noise_frac_sigma: float
    drift_per_year_frac: float
    dropout_probability: float
    stuck_probability: float
    stuck_mean_duration_steps: int
    spike_probability: float
    spike_magnitude_sigma: float
    unit_error_probability: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensorSettings:
        """Build from the ``sensors`` block of the priors file."""
        return cls(**payload)


@dataclass
class SensorFaultLog:
    """Where each injected sensor fault landed, by channel."""

    dropouts: dict[str, list[int]]
    stuck: dict[str, list[int]]
    spikes: dict[str, list[int]]
    unit_errors: dict[str, list[int]]

    def total(self) -> int:
        """Total number of faulted samples across all channels."""
        return sum(
            len(indices)
            for group in (self.dropouts, self.stuck, self.spikes, self.unit_errors)
            for indices in group.values()
        )

    def as_dict(self) -> dict[str, dict[str, list[int]]]:
        """Serialisable form."""
        return {
            "dropouts": self.dropouts,
            "stuck": self.stuck,
            "spikes": self.spikes,
            "unit_errors": self.unit_errors,
        }


class SensorModel:
    """Applies measurement error to clean simulated channels.

    Args:
        settings: Fault rates and magnitudes.
        seed: Seed, so a dataset is reproducible.
        unit_error_factors: Multipliers applied when a unit error is injected,
            keyed by channel. The defaults are the mistakes that actually happen
            in this industry: barrels reported as cubic metres, pressure in
            kg/cm2 reported as kPa, and pounds force reported as newtons.
    """

    DEFAULT_UNIT_ERROR_FACTORS = {
        "rate": 1.0 / 0.158987294928,
        "pressure": 98.0665,
        "load": 4.4482216152605,
        "temperature": 1.8,
    }

    def __init__(
        self,
        settings: SensorSettings,
        seed: int,
        unit_error_factors: dict[str, float] | None = None,
    ) -> None:
        self.settings = settings
        self.rng = np.random.default_rng(seed)
        self.unit_error_factors = unit_error_factors or dict(self.DEFAULT_UNIT_ERROR_FACTORS)
        self.log = SensorFaultLog({}, {}, {}, {})

    def _noise_sigma(self, kind: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        settings = self.settings
        if kind == "load":
            return np.full(values.shape, settings.load_noise_n_sigma)
        if kind == "position":
            return np.full(values.shape, settings.position_noise_m_sigma)
        if kind == "temperature":
            return np.full(values.shape, settings.temperature_noise_c_sigma)
        if kind == "pressure":
            return np.full(values.shape, settings.pressure_noise_kpa_sigma)
        if kind == "rate":
            return np.abs(values) * settings.rate_noise_frac_sigma
        raise PhysicsDomainError(f"Unknown sensor channel kind '{kind}'.", kind=kind)

    def apply(
        self,
        channel: str,
        kind: str,
        values: NDArray[np.float64],
        elapsed_years: NDArray[np.float64] | None = None,
        allow_faults: bool = True,
    ) -> NDArray[np.float64]:
        """Return a measured version of a clean channel.

        Args:
            channel: Name used in the fault log.
            kind: One of load, position, temperature, pressure, rate.
            values: Clean values.
            elapsed_years: Age of each sample, used for calibration drift.
            allow_faults: Set to False for channels that should stay clean, such
                as the reference copy kept for scoring the anomaly detector.
        """
        clean = np.asarray(values, dtype=float)
        measured = clean.copy()
        settings = self.settings

        measured = measured + self.rng.normal(0.0, 1.0, clean.shape) * self._noise_sigma(
            kind, clean
        )
        if elapsed_years is not None:
            measured = measured * (
                1.0 + settings.drift_per_year_frac * np.asarray(elapsed_years, dtype=float)
            )
        if not allow_faults:
            return measured

        size = measured.size
        dropout_mask = self.rng.random(size) < settings.dropout_probability
        if np.any(dropout_mask):
            measured[dropout_mask] = np.nan
            self.log.dropouts[channel] = np.nonzero(dropout_mask)[0].tolist()

        stuck_indices: list[int] = []
        position = 0
        while position < size:
            if self.rng.random() < settings.stuck_probability:
                duration = int(max(self.rng.exponential(settings.stuck_mean_duration_steps), 2))
                end = min(position + duration, size)
                measured[position:end] = measured[position]
                stuck_indices.extend(range(position, end))
                position = end
            else:
                position += 1
        if stuck_indices:
            self.log.stuck[channel] = stuck_indices

        spike_mask = self.rng.random(size) < settings.spike_probability
        if np.any(spike_mask):
            magnitude = self._noise_sigma(kind, clean) * settings.spike_magnitude_sigma
            signs = self.rng.choice([-1.0, 1.0], size=size)
            measured[spike_mask] = measured[spike_mask] + (magnitude * signs)[spike_mask]
            self.log.spikes[channel] = np.nonzero(spike_mask)[0].tolist()

        if self.rng.random() < settings.unit_error_probability * max(size, 1) / 1000.0:
            factor = self.unit_error_factors.get(kind)
            if factor is not None:
                start = int(self.rng.integers(0, max(size - 1, 1)))
                end = min(start + int(max(size * 0.05, 3)), size)
                measured[start:end] = measured[start:end] * factor
                self.log.unit_errors[channel] = list(range(start, end))

        return measured

    def apply_card(
        self, channel: str, load_n: NDArray[np.float64], position_m: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Measure a dynamometer card, which has its own load and position channels."""
        measured_load = self.apply(f"{channel}_load", "load", load_n, allow_faults=False)
        measured_position = self.apply(
            f"{channel}_position", "position", position_m, allow_faults=False
        )
        return measured_load, measured_position
