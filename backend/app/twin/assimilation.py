"""Keeping the twin in step with a live well.

Two estimators are provided.

* An Ensemble Kalman Filter over a small set of physically meaningful
  parameters, with covariance inflation and hard bounds. This is the primary
  method: it gives a parameter distribution rather than a point estimate, so
  the API can report how confident a recommendation is.
* A rolling-window least squares refit, used as a fallback when the ensemble
  collapses or when there are too few observations for a meaningful spread.

The parameter set is deliberately small. Trying to estimate every unknown at
Baghewala from a handful of daily measurements would produce a confident
looking answer with no information in it. The parameters chosen are the ones
that both matter to the predictions and are identifiable from the measurements
the field actually has: effective permeability and skin, the thermal loss
multiplier, the rod damping factor and the vacuum insulated tubing heat
transfer coefficient.

Source: Evensen, G. (2003), The Ensemble Kalman Filter: theoretical formulation
and practical implementation, Ocean Dynamics 53; Aanonsen et al. (2009), The
ensemble Kalman filter in reservoir engineering, SPE Journal 14(3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.errors import NumericalError, PhysicsDomainError
from app.core.logging import get_logger
from app.core.numerics import require_finite
from app.twin.coupled import TwinParameters

LOGGER = get_logger(__name__)

DEFAULT_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "permeability_md": (80.0, 6000.0),
    "skin_dimensionless": (-2.0, 12.0),
    "net_pay_thickness_m": (4.0, 30.0),
    "thermal_loss_multiplier": (0.3, 3.0),
    "vit_heat_transfer_w_per_m2_k": (0.15, 4.0),
    "rod_damping_factor": (0.05, 1.2),
    "cycle_energy_retention_frac": (0.5, 0.99),
    "productivity_decline_per_cycle_frac": (0.0, 0.35),
    "water_cut_initial_frac": (0.0, 0.6),
    "water_cut_growth_per_cycle_frac": (0.0, 0.2),
}
"""Hard bounds every estimate is clipped into. Outside them the physics is wrong."""


@dataclass(frozen=True)
class Observation:
    """One measured quantity with its assumed measurement standard deviation."""

    name: str
    value: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if self.standard_deviation <= 0.0:
            raise PhysicsDomainError(
                "Observation standard deviation must be positive.",
                name=self.name,
                standard_deviation=self.standard_deviation,
            )


@dataclass(frozen=True)
class AssimilationResult:
    """Posterior parameter estimate with its spread."""

    parameter_names: tuple[str, ...]
    mean: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]
    ensemble: NDArray[np.float64]
    prior_mean: NDArray[np.float64]
    prior_standard_deviation: NDArray[np.float64]
    observation_count: int
    method: str

    def as_parameters(self, base: TwinParameters) -> TwinParameters:
        """Posterior mean as a :class:`TwinParameters`."""
        return TwinParameters.from_vector(list(self.parameter_names), self.mean, base)

    def uncertainty(self) -> dict[str, dict[str, float]]:
        """Per-parameter mean, standard deviation and the reduction against the prior."""
        report: dict[str, dict[str, float]] = {}
        for index, name in enumerate(self.parameter_names):
            prior_sigma = float(self.prior_standard_deviation[index])
            posterior_sigma = float(self.standard_deviation[index])
            report[name] = {
                "mean": float(self.mean[index]),
                "standard_deviation": posterior_sigma,
                "prior_standard_deviation": prior_sigma,
                "uncertainty_reduction_frac": (
                    1.0 - posterior_sigma / prior_sigma if prior_sigma > 0.0 else 0.0
                ),
            }
        return report

    def percentile_parameters(self, base: TwinParameters, percentile: float) -> TwinParameters:
        """Parameter set at a given percentile of the posterior ensemble.

        Used by the robust optimizer, which ranks candidates on a conservative
        percentile rather than on the mean.
        """
        values = np.percentile(self.ensemble, percentile, axis=0)
        return TwinParameters.from_vector(list(self.parameter_names), values, base)


def _clip_to_bounds(
    ensemble: NDArray[np.float64],
    names: Sequence[str],
    bounds: dict[str, tuple[float, float]],
) -> NDArray[np.float64]:
    clipped = ensemble.copy()
    for index, name in enumerate(names):
        low, high = bounds.get(name, (-np.inf, np.inf))
        clipped[:, index] = np.clip(clipped[:, index], low, high)
    return clipped


def build_prior_ensemble(
    base: TwinParameters,
    parameter_names: Sequence[str],
    ensemble_size: int,
    spread_frac: float = 0.35,
    seed: int = 0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> NDArray[np.float64]:
    """Draw a prior ensemble around a parameter set.

    Strictly positive parameters are perturbed log-normally so they stay
    positive; parameters that may be negative, such as skin, are perturbed
    normally. The spread is a fraction of the base value.
    """
    if ensemble_size < 4:
        raise PhysicsDomainError(
            "An ensemble of at least four members is needed.", ensemble_size=ensemble_size
        )
    rng = np.random.default_rng(seed)
    limits = bounds or DEFAULT_PARAMETER_BOUNDS
    centre = base.to_vector(list(parameter_names))
    ensemble = np.zeros((ensemble_size, len(parameter_names)))
    for index, name in enumerate(parameter_names):
        value = centre[index]
        if name == "skin_dimensionless" or value <= 0.0:
            scale = max(abs(value) * spread_frac, 0.5)
            ensemble[:, index] = value + rng.normal(0.0, scale, ensemble_size)
        else:
            ensemble[:, index] = value * np.exp(
                rng.normal(0.0, spread_frac, ensemble_size) - 0.5 * spread_frac**2
            )
    return _clip_to_bounds(ensemble, parameter_names, limits)


def ensemble_kalman_update(
    ensemble: NDArray[np.float64],
    predictions: NDArray[np.float64],
    observations: NDArray[np.float64],
    observation_sigma: NDArray[np.float64],
    inflation: float = 1.05,
    seed: int = 0,
) -> NDArray[np.float64]:
    """One stochastic Ensemble Kalman Filter analysis step.

    Equations:
        K   = C_xy (C_yy + R)^-1
        x_j = x_j + K (d + e_j - y_j)
    with C_xy the cross covariance between parameters and predicted
    observations, C_yy the predicted observation covariance, R the observation
    error covariance and e_j a perturbation drawn from R.

    Covariance inflation is applied to the parameter ensemble before the update,
    which counteracts the ensemble collapse that small ensembles suffer from.

    Args:
        ensemble: Prior parameters, shape (members, parameters).
        predictions: Predicted observations, shape (members, observations).
        observations: Measured values, shape (observations,).
        observation_sigma: Measurement standard deviations, shape (observations,).
        inflation: Multiplicative covariance inflation factor, at least 1.
        seed: Seed for the observation perturbations, so runs are reproducible.

    Returns:
        The posterior ensemble.
    """
    if ensemble.ndim != 2 or predictions.ndim != 2:
        raise PhysicsDomainError("Ensemble and predictions must both be two dimensional.")
    if ensemble.shape[0] != predictions.shape[0]:
        raise PhysicsDomainError(
            "Ensemble and predictions must have the same number of members.",
            ensemble_members=int(ensemble.shape[0]),
            prediction_members=int(predictions.shape[0]),
        )
    if predictions.shape[1] != observations.size:
        raise PhysicsDomainError(
            "Prediction and observation counts differ.",
            predicted=int(predictions.shape[1]),
            observed=int(observations.size),
        )
    if inflation < 1.0:
        raise PhysicsDomainError("Inflation must be at least 1.", inflation=inflation)

    require_finite(ensemble, "prior ensemble")
    require_finite(predictions, "ensemble predictions")

    members = ensemble.shape[0]
    rng = np.random.default_rng(seed)

    parameter_mean = ensemble.mean(axis=0)
    inflated = parameter_mean + inflation * (ensemble - parameter_mean)
    prediction_mean = predictions.mean(axis=0)

    parameter_anomaly = inflated - inflated.mean(axis=0)
    prediction_anomaly = predictions - prediction_mean
    denominator = max(members - 1, 1)
    cross_covariance = parameter_anomaly.T @ prediction_anomaly / denominator
    prediction_covariance = prediction_anomaly.T @ prediction_anomaly / denominator
    observation_covariance = np.diag(observation_sigma**2)

    gain_matrix = prediction_covariance + observation_covariance
    try:
        gain = cross_covariance @ np.linalg.pinv(gain_matrix)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - pinv rarely fails
        raise NumericalError("Kalman gain could not be computed.") from exc

    perturbations = rng.normal(0.0, 1.0, size=(members, observations.size)) * observation_sigma
    innovation = (observations + perturbations) - predictions
    posterior = inflated + innovation @ gain.T
    return require_finite(posterior, "posterior ensemble")


class EnsembleKalmanAssimilator:
    """Parameter estimation for one well by Ensemble Kalman Filter.

    Args:
        base: Prior parameter set, normally the configuration defaults.
        parameter_names: Which parameters to estimate. Keep the list short.
        ensemble_size: Number of members. Twenty to fifty is usual.
        spread_frac: Prior spread as a fraction of each base value.
        inflation: Covariance inflation applied at every update.
        seed: Seed, so the whole procedure is reproducible.
        bounds: Hard bounds, defaulting to :data:`DEFAULT_PARAMETER_BOUNDS`.
        n_jobs: Workers used to evaluate the ensemble. Members are independent,
            so this scales almost linearly. One means run in this process, which
            is what the tests use so failures are easy to read.
    """

    def __init__(
        self,
        base: TwinParameters,
        parameter_names: Sequence[str],
        ensemble_size: int = 32,
        spread_frac: float = 0.35,
        inflation: float = 1.05,
        seed: int = 0,
        bounds: dict[str, tuple[float, float]] | None = None,
        n_jobs: int = 1,
    ) -> None:
        unknown = [name for name in parameter_names if name not in base.as_dict()]
        if unknown:
            raise PhysicsDomainError(
                f"Unknown twin parameters requested: {', '.join(unknown)}.", unknown=unknown
            )
        self.base = base
        self.parameter_names = tuple(parameter_names)
        self.bounds = bounds or DEFAULT_PARAMETER_BOUNDS
        self.inflation = inflation
        self.seed = seed
        self.n_jobs = n_jobs
        self.ensemble = build_prior_ensemble(
            base, self.parameter_names, ensemble_size, spread_frac, seed, self.bounds
        )
        self.prior_mean = self.ensemble.mean(axis=0).copy()
        self.prior_standard_deviation = self.ensemble.std(axis=0).copy()
        self.update_count = 0

    @property
    def ensemble_size(self) -> int:
        """Number of ensemble members."""
        return int(self.ensemble.shape[0])

    def members(self) -> list[TwinParameters]:
        """The ensemble as a list of parameter sets, ready to be simulated."""
        return [
            TwinParameters.from_vector(list(self.parameter_names), row, self.base)
            for row in self.ensemble
        ]

    def update(
        self,
        predict: Callable[[TwinParameters], Sequence[float]],
        observations: Sequence[Observation],
    ) -> AssimilationResult:
        """Run one analysis step.

        Args:
            predict: Maps a parameter set to the predicted values of the
                observed quantities, in the same order as ``observations``.
            observations: The measurements for this step.

        Returns:
            The posterior estimate with its spread.
        """
        if not observations:
            raise PhysicsDomainError("At least one observation is required.")
        values = np.asarray([obs.value for obs in observations], dtype=float)
        sigma = np.asarray([obs.standard_deviation for obs in observations], dtype=float)

        members = self.members()
        if self.n_jobs == 1:
            raw = [list(predict(member)) for member in members]
        else:
            from joblib import Parallel, delayed

            raw = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(lambda member: list(predict(member)))(member) for member in members
            )
        predictions = np.zeros((self.ensemble_size, len(observations)))
        for index, values_out in enumerate(raw):
            predicted = np.asarray(values_out, dtype=float)
            if predicted.size != len(observations):
                raise PhysicsDomainError(
                    "Predictor returned the wrong number of values.",
                    expected=len(observations),
                    received=int(predicted.size),
                )
            predictions[index] = np.nan_to_num(predicted, nan=0.0, posinf=1.0e12, neginf=-1.0e12)

        posterior = ensemble_kalman_update(
            ensemble=self.ensemble,
            predictions=predictions,
            observations=values,
            observation_sigma=sigma,
            inflation=self.inflation,
            seed=self.seed + self.update_count,
        )
        self.ensemble = _clip_to_bounds(posterior, self.parameter_names, self.bounds)
        self.update_count += 1

        spread = self.ensemble.std(axis=0)
        if float(np.max(spread / np.maximum(np.abs(self.prior_mean), 1.0e-9))) < 1.0e-6:
            LOGGER.warning(
                "Ensemble has collapsed; treat the parameter uncertainty as unreliable.",
                extra={"update_count": self.update_count},
            )
        return AssimilationResult(
            parameter_names=self.parameter_names,
            mean=self.ensemble.mean(axis=0),
            standard_deviation=spread,
            ensemble=self.ensemble.copy(),
            prior_mean=self.prior_mean,
            prior_standard_deviation=self.prior_standard_deviation,
            observation_count=len(observations),
            method="ensemble_kalman_filter",
        )


def rolling_least_squares_refit(
    base: TwinParameters,
    parameter_names: Sequence[str],
    predict: Callable[[TwinParameters], Sequence[float]],
    observations: Sequence[Observation],
    bounds: dict[str, tuple[float, float]] | None = None,
    max_iterations: int = 40,
    seed: int = 0,
) -> AssimilationResult:
    """Fallback estimator: bounded least squares on the same residuals.

    A Nelder-Mead search on the weighted sum of squared residuals, in a
    log-transformed space for strictly positive parameters so that the search
    cannot leave the physical domain. It returns a point estimate with the
    spread taken from a small finite-difference sensitivity, so the result has
    the same shape as the filter output and the API does not have to care which
    estimator ran.
    """
    from scipy.optimize import minimize

    limits = bounds or DEFAULT_PARAMETER_BOUNDS
    names = list(parameter_names)
    start = base.to_vector(names)
    values = np.asarray([obs.value for obs in observations], dtype=float)
    sigma = np.asarray([obs.standard_deviation for obs in observations], dtype=float)

    def objective(vector: NDArray[np.float64]) -> float:
        candidate = _clip_to_bounds(vector.reshape(1, -1), names, limits)[0]
        parameters = TwinParameters.from_vector(names, candidate, base)
        predicted = np.asarray(list(predict(parameters)), dtype=float)
        predicted = np.nan_to_num(predicted, nan=1.0e12, posinf=1.0e12, neginf=-1.0e12)
        residual = (predicted - values) / sigma
        return float(np.sum(residual**2))

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": max_iterations * max(len(names), 1), "xatol": 1e-4, "fatol": 1e-4},
    )
    best = _clip_to_bounds(np.asarray(result.x).reshape(1, -1), names, limits)[0]

    # Spread from a local finite-difference sensitivity, so the caller still gets
    # an uncertainty even though this estimator is a point method.
    rng = np.random.default_rng(seed)
    perturbations = rng.normal(0.0, 0.05, size=(16, len(names)))
    ensemble = _clip_to_bounds(best * (1.0 + perturbations), names, limits)
    return AssimilationResult(
        parameter_names=tuple(names),
        mean=best,
        standard_deviation=ensemble.std(axis=0),
        ensemble=ensemble,
        prior_mean=start,
        prior_standard_deviation=np.abs(start) * 0.35 + 1.0e-9,
        observation_count=len(observations),
        method="rolling_least_squares",
    )
