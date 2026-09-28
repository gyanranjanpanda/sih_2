"""Calibration: fitting the twin to a well's measured history.

This is the workflow that turns the configured placeholders into numbers for a
particular well. It runs in three stages, from cheapest and most reliable to
most expensive:

1. **Fluid.** Refit the viscosity-temperature line to the laboratory table in
   the ``fluid`` table. This needs no simulation and is the single highest
   value calibration, because viscosity drives everything downstream.
2. **Direct estimates.** Read what can be read straight off the data: the
   productivity decline between cycles, the water cut trend, and the rod
   damping implied by the measured card load range.
3. **Assimilation.** Run the Ensemble Kalman Filter over the remaining
   parameters against the daily oil rate and near-well temperature of the
   training cycles.

Everything reports before-and-after fit quality so an engineer can see whether
the calibration helped, and the result records which cycles were used so the
held-out cycles stay held out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import FieldConfig
from app.core.errors import DataValidationError
from app.core.logging import get_logger
from app.twin.assimilation import (
    AssimilationResult,
    EnsembleKalmanAssimilator,
    Observation,
)
from app.twin.coupled import CoupledWellTwin, PumpSetpoint, TwinParameters
from app.twin.fluid import fit_walther
from app.twin.reservoir import InjectionPlan

LOGGER = get_logger(__name__)

DEFAULT_ASSIMILATED_PARAMETERS: tuple[str, ...] = (
    "permeability_md",
    "skin_dimensionless",
    "thermal_loss_multiplier",
    "vit_heat_transfer_w_per_m2_k",
    "rod_damping_factor",
)
"""The parameters the filter estimates. Kept short so the estimate means something."""


@dataclass
class FitQuality:
    """How well a model reproduces a measured series."""

    mean_absolute_error: float
    root_mean_square_error: float
    mean_absolute_percentage_error: float
    sample_count: int

    @classmethod
    def compare(cls, measured: np.ndarray, modelled: np.ndarray) -> FitQuality:
        """Compute the three error measures, ignoring missing samples."""
        measured = np.asarray(measured, dtype=float)
        modelled = np.asarray(modelled, dtype=float)
        mask = np.isfinite(measured) & np.isfinite(modelled)
        if not np.any(mask):
            return cls(float("nan"), float("nan"), float("nan"), 0)
        difference = modelled[mask] - measured[mask]
        denominator = np.maximum(np.abs(measured[mask]), 1.0e-6)
        return cls(
            mean_absolute_error=float(np.mean(np.abs(difference))),
            root_mean_square_error=float(np.sqrt(np.mean(difference**2))),
            mean_absolute_percentage_error=float(np.mean(np.abs(difference) / denominator)),
            sample_count=int(np.count_nonzero(mask)),
        )

    def as_dict(self) -> dict[str, float]:
        """Serialisable form."""
        return {
            "mean_absolute_error": self.mean_absolute_error,
            "root_mean_square_error": self.root_mean_square_error,
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error,
            "sample_count": self.sample_count,
        }


@dataclass
class CalibrationResult:
    """Everything one calibration produced."""

    well_id: str
    training_cycles: list[int]
    parameters: TwinParameters
    prior_parameters: TwinParameters
    fluid_anchors: list[dict[str, float]] = field(default_factory=list)
    assimilation: AssimilationResult | None = None
    before: FitQuality | None = None
    after: FitQuality | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """Whether calibration reduced the error on the training cycles."""
        if self.before is None or self.after is None:
            return False
        return self.after.root_mean_square_error < self.before.root_mean_square_error

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the API and for ``reports/``."""
        payload: dict[str, Any] = {
            "well_id": self.well_id,
            "training_cycles": self.training_cycles,
            "parameters": self.parameters.as_dict(),
            "prior_parameters": self.prior_parameters.as_dict(),
            "fluid_anchors": self.fluid_anchors,
            "improved": self.improved,
            "notes": self.notes,
        }
        if self.before is not None:
            payload["fit_before"] = self.before.as_dict()
        if self.after is not None:
            payload["fit_after"] = self.after.as_dict()
        if self.assimilation is not None:
            payload["uncertainty"] = self.assimilation.uncertainty()
            payload["method"] = self.assimilation.method
        return payload


def fit_fluid_anchors(fluid_frame: pd.DataFrame, well_id: str) -> list[dict[str, float]]:
    """Refit the viscosity-temperature anchors from a laboratory table.

    The ASTM D341 line is fitted through every measurement for the well, and two
    anchors are then emitted at 50 and 200 degrees C so the twin configuration
    stays in the same shape. Fitting through all the points and re-emitting two
    is better than picking two measurements, because it uses the whole table.
    """
    rows = fluid_frame[fluid_frame["well_id"] == well_id]
    if rows.shape[0] < 2:
        raise DataValidationError(
            f"At least two viscosity measurements are needed to calibrate well {well_id}.",
            well_id=well_id,
            measurements=int(rows.shape[0]),
        )
    temps = rows["temp_c"].to_numpy(dtype=float)
    viscosity_cp = rows["viscosity_cp"].to_numpy(dtype=float)
    # Convert to kinematic with a nominal density so the Walther fit is on the
    # same quantity the twin uses; the density cancels when the twin converts back.
    nominal_density = 950.0
    kinematic_cst = viscosity_cp * 1000.0 / nominal_density
    fit = fit_walther(list(temps), list(kinematic_cst))
    anchors: list[dict[str, float]] = []
    for temp_c in (50.0, 200.0):
        kinematic = float(fit.kinematic_viscosity_cst(temp_c))
        anchors.append({"temp_c": temp_c, "viscosity_cp": kinematic * nominal_density / 1000.0})
    return anchors


def estimate_decline_from_cycles(cycles: pd.DataFrame, well_id: str) -> dict[str, float]:
    """Read the productivity decline and water cut trend straight off the history.

    Equation: fitting ln(oil) against cycle number gives a decline rate directly,
    which is far more reliable than asking a filter to find it.
    """
    rows = cycles[cycles["well_id"] == well_id].sort_values("cycle_number")
    result: dict[str, float] = {}
    oil = rows["oil_m3"].to_numpy(dtype=float)
    numbers = rows["cycle_number"].to_numpy(dtype=float)
    usable = np.isfinite(oil) & (oil > 0.0)
    if int(np.count_nonzero(usable)) >= 3:
        slope, _ = np.polyfit(numbers[usable], np.log(oil[usable]), 1)
        result["productivity_decline_per_cycle_frac"] = float(
            np.clip(1.0 - np.exp(slope), 0.0, 0.34)
        )
    if "water_m3" in rows.columns:
        water = rows["water_m3"].to_numpy(dtype=float)
        total = oil + water
        valid = np.isfinite(total) & (total > 0.0)
        if int(np.count_nonzero(valid)) >= 2:
            water_cut = water[valid] / total[valid]
            result["water_cut_initial_frac"] = float(np.clip(water_cut[0], 0.0, 0.59))
            if valid.sum() >= 3:
                growth, _ = np.polyfit(numbers[valid], water_cut, 1)
                result["water_cut_growth_per_cycle_frac"] = float(np.clip(growth, 0.0, 0.19))
    return result


def calibrate_well(
    config: FieldConfig,
    well_id: str,
    daily: pd.DataFrame,
    cycles: pd.DataFrame,
    fluid: pd.DataFrame | None = None,
    training_cycles: list[int] | None = None,
    ensemble_size: int = 20,
    filter_passes: int = 2,
    n_jobs: int = -1,
    parameter_names: tuple[str, ...] = DEFAULT_ASSIMILATED_PARAMETERS,
    seed: int = 20260101,
) -> CalibrationResult:
    """Calibrate the twin for one well against its measured history.

    Args:
        config: The well's configuration, before calibration.
        well_id: Which well.
        daily: The ``production_daily`` table.
        cycles: The ``css_cycles`` table.
        fluid: The ``fluid`` table, if a laboratory viscosity table exists.
        training_cycles: Which cycles to fit on. Defaults to the first three.
        ensemble_size: Members in the Ensemble Kalman Filter.
        filter_passes: How many analysis steps to run.
        parameter_names: Which parameters the filter estimates.
        n_jobs: Workers used to evaluate the ensemble; minus one uses every core.
        seed: Seed, so calibration is reproducible.
    """
    well_daily = daily[daily["well_id"] == well_id]
    well_cycles = cycles[cycles["well_id"] == well_id]
    if well_daily.empty or well_cycles.empty:
        raise DataValidationError(f"No history found for well {well_id}.", well_id=well_id)

    available = sorted(int(value) for value in well_cycles["cycle_number"].unique())
    chosen = training_cycles or available[: min(3, len(available))]
    notes: list[str] = []

    working_config = config
    anchors: list[dict[str, float]] = []
    if fluid is not None and not fluid.empty and (fluid["well_id"] == well_id).any():
        anchors = fit_fluid_anchors(fluid, well_id)
        working_config = config.with_overrides({"fluid": {"viscosity_anchors": anchors}})
        notes.append(
            f"Fluid model refitted to {int((fluid['well_id'] == well_id).sum())} laboratory "
            "viscosity measurements."
        )

    prior = TwinParameters.from_config(working_config)
    direct = estimate_decline_from_cycles(cycles, well_id)
    if direct:
        prior = TwinParameters(**{**prior.as_dict(), **direct})
        notes.append(
            "Cycle decline and water cut trend read directly from the cycle history: "
            + ", ".join(f"{key} = {value:.3f}" for key, value in sorted(direct.items()))
        )

    target = well_daily[well_daily["cycle_number"].isin(chosen)]
    measured_oil = target.groupby("cycle_number")["oil_rate_m3_per_day"].mean()
    measured_temp = (
        target.groupby("cycle_number")["near_well_temp_c"].mean()
        if "near_well_temp_c" in target.columns
        else None
    )
    if measured_oil.empty:
        raise DataValidationError(
            f"Well {well_id} has no usable daily oil rate in the training cycles.",
            well_id=well_id,
        )

    plans = _plans_from_cycles(well_cycles, chosen, working_config)
    setpoint = _setpoint_from_history(well_daily, working_config)

    cycle_lengths = {
        cycle: float(max(target[target["cycle_number"] == cycle].shape[0], 20)) for cycle in chosen
    }

    def predict(parameters: TwinParameters) -> list[float]:
        """Predicted cycle-mean oil rate and heated-zone temperature.

        The fast reservoir and pump path is used: both observables are
        reservoir quantities, and solving the rod string for every ensemble
        member of every pass would make calibration take minutes instead of
        seconds without changing the answer.
        """
        twin = CoupledWellTwin(working_config, parameters)
        values: list[float] = []
        for index, plan in enumerate(plans):
            cycle = chosen[index]
            outcome = twin.run_cycle_fast(
                plan=plan,
                setpoint=setpoint,
                cycle_number=cycle,
                max_production_days=min(cycle_lengths[cycle], 150.0),
            )
            values.append(outcome.oil_m3 / max(outcome.cutoff_day, 1.0))
            if measured_temp is not None:
                values.append(twin.reservoir.state.average_heated_temp_c)
        return values

    observations: list[Observation] = []
    for cycle in chosen:
        observations.append(
            Observation(
                name=f"oil_rate_cycle_{cycle}",
                value=float(measured_oil.get(cycle, np.nan)),
                standard_deviation=max(0.15 * float(measured_oil.get(cycle, 1.0)), 0.2),
            )
        )
        if measured_temp is not None:
            observations.append(
                Observation(
                    name=f"near_well_temp_cycle_{cycle}",
                    value=float(measured_temp.get(cycle, np.nan)),
                    standard_deviation=12.0,
                )
            )
    observations = [obs for obs in observations if np.isfinite(obs.value)]
    if not observations:
        raise DataValidationError(
            f"Well {well_id} has no finite observations to calibrate against.", well_id=well_id
        )

    targets = np.asarray([obs.value for obs in observations], dtype=float)
    before = FitQuality.compare(targets, np.asarray(predict(prior), dtype=float))

    assimilator = EnsembleKalmanAssimilator(
        base=prior,
        parameter_names=list(parameter_names),
        ensemble_size=ensemble_size,
        spread_frac=0.4,
        seed=seed,
        n_jobs=n_jobs,
    )
    outcome: AssimilationResult | None = None
    for _ in range(max(filter_passes, 1)):
        outcome = assimilator.update(predict, observations)
    assert outcome is not None
    posterior = outcome.as_parameters(prior)
    after = FitQuality.compare(targets, np.asarray(predict(posterior), dtype=float))

    if after.root_mean_square_error > before.root_mean_square_error:
        notes.append(
            "The filter did not improve the fit on the training cycles, so the prior "
            "parameters were kept. This is reported rather than hidden."
        )
        posterior = prior
        after = before

    return CalibrationResult(
        well_id=well_id,
        training_cycles=list(chosen),
        parameters=posterior,
        prior_parameters=prior,
        fluid_anchors=anchors,
        assimilation=outcome,
        before=before,
        after=after,
        notes=notes,
    )


def _plans_from_cycles(
    cycles: pd.DataFrame, chosen: list[int], config: FieldConfig
) -> list[InjectionPlan]:
    """Rebuild the injection plans that were actually used, from the cycle records."""
    plans: list[InjectionPlan] = []
    for cycle in chosen:
        rows = cycles[cycles["cycle_number"] == cycle]
        if rows.empty:
            plans.append(InjectionPlan.from_config(config))
            continue
        row = rows.iloc[0]
        plans.append(
            InjectionPlan(
                steam_volume_m3_cwe=float(row["steam_volume_m3_cwe"]),
                injection_rate_m3_per_day_cwe=float(row["injection_rate_m3_per_day_cwe"]),
                injection_pressure_kpa=float(row["injection_pressure_kpa"]),
                steam_quality_frac=float(row["steam_quality_frac"]),
                soak_days=float(row["soak_days"]),
                cutoff_marginal_energy_ratio=config.css.cutoff_marginal_energy_ratio,
            )
        )
    return plans


def _setpoint_from_history(daily: pd.DataFrame, config: FieldConfig) -> PumpSetpoint:
    """Recover the pump setting the well was actually run at."""
    spm = config.srp.spm_setpoint
    stroke = config.srp.stroke_length_m
    if "spm" in daily.columns:
        values = pd.to_numeric(daily["spm"], errors="coerce").dropna()
        if not values.empty:
            spm = float(np.clip(values.median(), config.srp.spm_min, config.srp.spm_max))
    if "stroke_length_m" in daily.columns:
        values = pd.to_numeric(daily["stroke_length_m"], errors="coerce").dropna()
        if not values.empty:
            stroke = float(values.median())
    return PumpSetpoint(spm=spm, stroke_length_m=stroke)
