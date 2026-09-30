"""Cyclic steam stimulation cycle design.

The optimizer chooses the steam volume, the injection rate and pressure, the
steam quality, the soak length and the production cut-off rule for the next
cycle on one well. It returns a Pareto front over cycle oil, steam-to-oil ratio
and energy cost, and picks one recommendation from it with the weights in
``config/optimizer.yaml``.

Two things make this more than a steam calculator.

**The pump-compatibility constraint.** A steam design that produces a lot of
oil but lets the heated zone cool to the point where the rods float is not a
good design. Every candidate is therefore checked against the lift system at
the coldest point of the cycle it would produce: the float margin and the rod
stress at the operating pump setting have to stay inside their limits. This is
the coupling the problem statement asks for, expressed as a constraint rather
than as a claim.

**Robustness.** Almost every reservoir parameter at Baghewala is unknown. A
candidate that is only good for one parameter set is not a recommendation, so
candidates are scored on a conservative percentile across an ensemble of
plausible parameter sets rather than on a single best case.

Cost is managed with a surrogate trained on the fast cycle path, and the
surrogate error is measured on a held-out split and reported. Every point on
the returned front is then re-evaluated with the real model, so nothing is
recommended on the strength of a fit alone.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.config import FieldConfig, OptimizerConfig
from app.core.errors import PhysicsDomainError
from app.core.logging import get_logger
from app.core.numerics import clamp
from app.core.units import M3_PER_BBL
from app.optimize.pump_optimizer import PumpOptimizer
from app.twin.coupled import CoupledWellTwin, PumpSetpoint, TwinParameters
from app.twin.reservoir import InjectionPlan

LOGGER = get_logger(__name__)

VARIABLE_ORDER: tuple[str, ...] = (
    "steam_volume_m3_cwe",
    "injection_rate_m3_per_day_cwe",
    "injection_pressure_kpa",
    "steam_quality_frac",
    "soak_days",
    "cutoff_marginal_energy_ratio",
)
"""Fixed decision variable order, so vectors and reports stay comparable."""


@dataclass
class CycleOutcome:
    """What one steam design is predicted to do."""

    oil_m3: float
    steam_m3_cwe: float
    steam_oil_ratio: float
    energy_cost_usd: float
    revenue_usd: float
    energy_kwh_per_bbl: float
    production_days: float
    heated_radius_m: float
    end_of_cycle_temp_c: float
    float_margin_index: float
    rod_stress_utilisation_frac: float
    peak_load_n: float
    violations: list[str] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        """Whether every hard constraint holds."""
        return not self.violations

    @property
    def net_value_usd(self) -> float:
        """Revenue less energy cost for the cycle."""
        return self.revenue_usd - self.energy_cost_usd

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        payload = asdict(self)
        payload["feasible"] = self.is_feasible
        payload["net_value_usd"] = self.net_value_usd
        return payload


@dataclass
class CssCandidate:
    """One steam design with its predicted outcome and score."""

    plan: InjectionPlan
    outcome: CycleOutcome
    score: float
    robust_percentile_oil_m3: float | None = None
    robust_violation_rate: float = 0.0

    def plan_dict(self) -> dict[str, float]:
        """The decision variables as a flat dictionary."""
        return {name: getattr(self.plan, name) for name in VARIABLE_ORDER}

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "plan": self.plan_dict(),
            "outcome": self.outcome.as_dict(),
            "score": self.score,
            "robust_percentile_oil_m3": self.robust_percentile_oil_m3,
            "robust_violation_rate": self.robust_violation_rate,
        }


@dataclass
class CssRecommendation:
    """The optimizer's answer for one well."""

    well_id: str
    cycle_number: int
    recommended: CssCandidate
    baseline: CssCandidate
    pareto_front: list[CssCandidate]
    surrogate_error: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    evaluations: int = 0
    elapsed_s: float = 0.0
    recommended_cutoff_day: float = 0.0

    @property
    def oil_change_m3(self) -> float:
        """Predicted change in cycle oil against the historical rule."""
        return self.recommended.outcome.oil_m3 - self.baseline.outcome.oil_m3

    @property
    def sor_change(self) -> float:
        """Predicted change in steam-to-oil ratio."""
        return (
            self.recommended.outcome.steam_oil_ratio - self.baseline.outcome.steam_oil_ratio
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the API."""
        return {
            "well_id": self.well_id,
            "cycle_number": self.cycle_number,
            "recommended": self.recommended.as_dict(),
            "baseline": self.baseline.as_dict(),
            "pareto_front": [candidate.as_dict() for candidate in self.pareto_front],
            "surrogate_error": self.surrogate_error,
            "reason": self.reason,
            "confidence": self.confidence,
            "evaluations": self.evaluations,
            "elapsed_s": self.elapsed_s,
            "oil_change_m3": self.oil_change_m3,
            "steam_oil_ratio_change": self.sor_change,
            "recommended_cutoff_day": self.recommended_cutoff_day,
        }


def historical_practice_plan(config: FieldConfig, recent_cycles: list[dict[str, float]] | None = None
                             ) -> InjectionPlan:
    """The steam design an engineer working from history would choose.

    The rule is the one the brief names: the mean of the last three cycles. With
    no history the field defaults are used. This is the baseline every
    comparison is measured against and it has to be a real practice, not a straw
    man, or the comparison means nothing.
    """
    default = InjectionPlan.from_config(config)
    if not recent_cycles:
        return default
    recent = recent_cycles[-3:]

    def mean_of(name: str, fallback: float) -> float:
        values = [
            float(row[name]) for row in recent if name in row and np.isfinite(row[name])
        ]
        return float(np.mean(values)) if values else fallback

    return InjectionPlan(
        steam_volume_m3_cwe=mean_of("steam_volume_m3_cwe", default.steam_volume_m3_cwe),
        injection_rate_m3_per_day_cwe=mean_of(
            "injection_rate_m3_per_day_cwe", default.injection_rate_m3_per_day_cwe
        ),
        injection_pressure_kpa=mean_of(
            "injection_pressure_kpa", default.injection_pressure_kpa
        ),
        steam_quality_frac=mean_of("steam_quality_frac", default.steam_quality_frac),
        soak_days=mean_of("soak_days", default.soak_days),
        cutoff_marginal_energy_ratio=default.cutoff_marginal_energy_ratio,
    )


class CssOptimizer:
    """Multi-objective design of the next steam cycle."""

    def __init__(
        self,
        config: FieldConfig,
        optimizer: OptimizerConfig,
        parameters: TwinParameters | None = None,
        pump_setpoint: PumpSetpoint | None = None,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.settings = optimizer.css
        self.parameters = parameters or TwinParameters.from_config(config)
        self.pump_setpoint = pump_setpoint or PumpSetpoint.from_config(config)
        self._evaluations = 0

    # ------------------------------------------------------------- variables
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Lower and upper bounds of the decision vector."""
        low = np.asarray(
            [self.settings.variables[name].low for name in VARIABLE_ORDER], dtype=float
        )
        high = np.asarray(
            [self.settings.variables[name].high for name in VARIABLE_ORDER], dtype=float
        )
        return low, high

    def plan_from_vector(self, vector: NDArray[np.float64]) -> InjectionPlan:
        """Build an injection plan from a decision vector."""
        low, high = self.bounds()
        clipped = np.clip(np.asarray(vector, dtype=float), low, high)
        values = dict(zip(VARIABLE_ORDER, clipped, strict=True))
        return InjectionPlan(**{name: float(value) for name, value in values.items()})

    def vector_from_plan(self, plan: InjectionPlan) -> NDArray[np.float64]:
        """Decision vector of a plan, clipped into the bounds."""
        low, high = self.bounds()
        return np.clip(
            np.asarray([getattr(plan, name) for name in VARIABLE_ORDER], dtype=float), low, high
        )

    # ------------------------------------------------------------- evaluation
    def evaluate(
        self,
        plan: InjectionPlan,
        cycle_number: int = 1,
        parameters: TwinParameters | None = None,
    ) -> CycleOutcome:
        """Run one candidate design through the twin and check every constraint."""
        self._evaluations += 1
        twin = CoupledWellTwin(self.config, parameters or self.parameters)
        result = twin.run_cycle_fast(plan, self.pump_setpoint, cycle_number)
        state = twin.reservoir.state
        violations = list(result.constraint_violations)
        constraints = self.settings.constraints

        fracture_limit = (
            constraints.fracture_pressure_safety_frac
            * self.config.reservoir.fracture_pressure_kpa
        )
        if plan.injection_pressure_kpa > fracture_limit:
            violations.append(
                f"injection pressure {plan.injection_pressure_kpa:.0f} kPa is above the "
                f"{constraints.fracture_pressure_safety_frac * 100:.0f} percent fracture "
                f"safety limit of {fracture_limit:.0f} kPa"
            )
        if plan.injection_days > constraints.max_injection_days:
            violations.append(
                f"injection would take {plan.injection_days:.0f} days, above the "
                f"{constraints.max_injection_days:.0f} day limit"
            )
        if state.heated_radius_m < constraints.min_heated_radius_m:
            violations.append(
                f"heated radius of {state.heated_radius_m:.1f} m is below the "
                f"{constraints.min_heated_radius_m:.1f} m minimum"
            )

        lift = self._pump_compatibility(twin, state.average_heated_temp_c, state.water_cut_frac)
        if lift["float_margin_index"] < constraints.min_float_margin_frac:
            violations.append(
                f"at the end of the cycle the float margin falls to "
                f"{lift['float_margin_index']:.2f}, below the "
                f"{constraints.min_float_margin_frac:.2f} minimum, so the rods would float "
                "before the cycle is over"
            )
        if lift["rod_stress_utilisation_frac"] > constraints.max_rod_stress_utilization_frac:
            violations.append(
                f"rod stress reaches {lift['rod_stress_utilisation_frac'] * 100:.0f} percent "
                f"of the Goodman allowable, above the "
                f"{constraints.max_rod_stress_utilization_frac * 100:.0f} percent limit"
            )

        return CycleOutcome(
            oil_m3=result.oil_m3,
            steam_m3_cwe=result.steam_m3_cwe,
            steam_oil_ratio=result.steam_oil_ratio
            if math.isfinite(result.steam_oil_ratio)
            else 1.0e6,
            energy_cost_usd=result.energy_cost_usd,
            revenue_usd=result.revenue_usd,
            energy_kwh_per_bbl=result.energy_kwh_per_bbl
            if math.isfinite(result.energy_kwh_per_bbl)
            else 1.0e6,
            production_days=result.cutoff_day,
            heated_radius_m=state.heated_radius_m,
            end_of_cycle_temp_c=state.average_heated_temp_c,
            float_margin_index=lift["float_margin_index"],
            rod_stress_utilisation_frac=lift["rod_stress_utilisation_frac"],
            peak_load_n=lift["peak_load_n"],
            violations=violations,
        )

    def _pump_compatibility(
        self, twin: CoupledWellTwin, end_temp_c: float, water_cut_frac: float
    ) -> dict[str, float]:
        """Check the lift system at the coldest point of the predicted cycle.

        The end of the cycle is the binding case: that is when the heated zone
        is coolest, the viscosity along the rods is highest and the float margin
        is lowest. Evaluating there is what turns the coupling into a constraint
        the CSS optimizer actually has to respect.
        """
        pump = PumpOptimizer(self.config, self.optimizer, twin)
        horizon = pump.build_horizon(
            sandface_temp_c=end_temp_c,
            water_cut_frac=water_cut_frac,
            production_day=1.0,
            cooling_rate_c_per_day=0.0,
            liquid_rate_guess_m3_per_day=max(twin.reservoir.state.oil_rate_m3_per_day, 0.5),
        )
        prediction = pump.predict(self.pump_setpoint, horizon[0])
        section = self.config.srp.rod_sections[0]
        allowable_pa = 0.25 * section.minimum_tensile_strength_pa + 0.5625 * max(
            prediction.minimum_load_n / section.area_m2, 0.0
        )
        utilisation = (prediction.peak_load_n / section.area_m2) / max(allowable_pa, 1.0)
        return {
            "float_margin_index": prediction.float_margin_index,
            "rod_stress_utilisation_frac": float(utilisation),
            "peak_load_n": prediction.peak_load_n,
        }

    def score(self, outcome: CycleOutcome) -> float:
        """Scalarise an outcome with the configured weights, higher is better."""
        weights = self.settings.objective_weights
        oil_term = weights.cycle_oil_m3 * outcome.oil_m3 / 500.0
        sor_term = weights.steam_oil_ratio * min(outcome.steam_oil_ratio, 50.0) / 5.0
        cost_term = weights.energy_cost_usd * outcome.energy_cost_usd / 50000.0
        value = oil_term - sor_term - cost_term
        if not outcome.is_feasible:
            value -= 10.0 + len(outcome.violations)
        return float(value)

    # -------------------------------------------------------------- surrogate
    def _train_surrogate(
        self, cycle_number: int, n_jobs: int = -1
    ) -> tuple[Any, dict[str, float]]:
        """Fit a surrogate on samples from the fast cycle path.

        A gradient boosted tree per objective, trained on a Latin hypercube
        sample and scored on a held-out fifth of it. The held-out error is
        reported with the recommendation: a surrogate whose error is not
        measured is not a surrogate, it is a guess.
        """
        from joblib import Parallel, delayed
        from sklearn.ensemble import HistGradientBoostingRegressor

        rng = np.random.default_rng(self.settings.seed)
        low, high = self.bounds()
        samples = self.settings.surrogate.training_samples
        dimensions = low.size
        # Latin hypercube, which covers the space far better than uniform
        # sampling at these sample counts.
        grid = (rng.permuted(np.tile(np.arange(samples), (dimensions, 1)), axis=1).T + 0.5)
        design = low + (high - low) * grid / samples

        outcomes = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.evaluate)(self.plan_from_vector(row), cycle_number)
            for row in design
        )
        self._evaluations += len(design)

        targets = {
            "oil_m3": np.asarray([o.oil_m3 for o in outcomes]),
            "steam_oil_ratio": np.asarray(
                [min(o.steam_oil_ratio, 50.0) for o in outcomes]
            ),
            "energy_cost_usd": np.asarray([o.energy_cost_usd for o in outcomes]),
            "float_margin_index": np.asarray([o.float_margin_index for o in outcomes]),
            "rod_stress_utilisation_frac": np.asarray(
                [o.rod_stress_utilisation_frac for o in outcomes]
            ),
            "heated_radius_m": np.asarray([o.heated_radius_m for o in outcomes]),
        }
        split = max(int(0.8 * samples), 8)
        models: dict[str, Any] = {}
        errors: dict[str, float] = {}
        for name, values in targets.items():
            model = HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.08, random_state=self.settings.seed
            )
            model.fit(design[:split], values[:split])
            models[name] = model
            held_out = values[split:]
            if held_out.size:
                predicted = model.predict(design[split:])
                denominator = np.maximum(np.abs(held_out), 1.0e-6)
                errors[name] = float(np.mean(np.abs(predicted - held_out) / denominator))
            model.fit(design, values)
        return models, errors

    # ----------------------------------------------------------------- search
    def optimize(
        self,
        cycle_number: int = 1,
        baseline: InjectionPlan | None = None,
        well_id: str = "unknown",
        n_jobs: int = -1,
        front_size: int = 12,
    ) -> CssRecommendation:
        """Design the next cycle.

        The search runs NSGA-II on the surrogate, then re-evaluates the whole
        returned front with the real model, so the recommendation is never made
        on the strength of a fit alone.
        """
        started = time.time()
        self._evaluations = 0
        baseline_plan = baseline or InjectionPlan.from_config(self.config)
        baseline_outcome = self.evaluate(baseline_plan, cycle_number)
        baseline_candidate = CssCandidate(
            plan=baseline_plan, outcome=baseline_outcome, score=self.score(baseline_outcome)
        )

        models, surrogate_error = self._train_surrogate(cycle_number, n_jobs=n_jobs)
        worst_error = max(surrogate_error.values()) if surrogate_error else 0.0
        if worst_error > self.settings.surrogate.max_acceptable_mape_frac:
            LOGGER.warning(
                "Surrogate error is above the configured limit; the front is still "
                "re-evaluated with the real model, so the recommendation stays valid.",
                extra={"surrogate_error": surrogate_error},
            )

        raw_front = self._run_nsga2(models)
        candidates: list[CssCandidate] = []
        for vector in raw_front[:front_size]:
            plan = self.plan_from_vector(vector)
            outcome = self.evaluate(plan, cycle_number)
            candidates.append(
                CssCandidate(plan=plan, outcome=outcome, score=self.score(outcome))
            )
        candidates.append(baseline_candidate)

        if self.settings.robust.enabled:
            self._apply_robustness(candidates, cycle_number, n_jobs=n_jobs)

        feasible = [candidate for candidate in candidates if candidate.outcome.is_feasible]
        pool = feasible or candidates
        best = max(pool, key=lambda candidate: candidate.score)

        pareto = self._pareto_filter(candidates)
        recommendation = CssRecommendation(
            well_id=well_id,
            cycle_number=cycle_number,
            recommended=best,
            baseline=baseline_candidate,
            pareto_front=pareto,
            surrogate_error=surrogate_error,
            evaluations=self._evaluations,
            elapsed_s=time.time() - started,
            recommended_cutoff_day=best.outcome.production_days,
        )
        recommendation.reason = self._explain(recommendation)
        recommendation.confidence = self._confidence(recommendation, worst_error)
        return recommendation

    def _run_nsga2(self, models: dict[str, Any]) -> list[NDArray[np.float64]]:
        """Run NSGA-II on the surrogate and return the non-dominated set."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize

        low, high = self.bounds()
        constraints = self.settings.constraints
        fracture_limit = (
            constraints.fracture_pressure_safety_frac
            * self.config.reservoir.fracture_pressure_kpa
        )
        settings = self.settings

        class SurrogateProblem(Problem):
            """Three objectives and four constraints, all from the surrogate."""

            def __init__(self) -> None:
                super().__init__(n_var=low.size, n_obj=3, n_ieq_constr=4, xl=low, xu=high)

            def _evaluate(self, x: NDArray[np.float64], out: dict[str, Any], *args: Any,
                          **kwargs: Any) -> None:
                oil = models["oil_m3"].predict(x)
                sor = models["steam_oil_ratio"].predict(x)
                cost = models["energy_cost_usd"].predict(x)
                margin = models["float_margin_index"].predict(x)
                stress = models["rod_stress_utilisation_frac"].predict(x)
                radius = models["heated_radius_m"].predict(x)
                out["F"] = np.column_stack([-oil, sor, cost])
                injection_days = x[:, 0] / np.maximum(x[:, 1], 1.0e-6)
                out["G"] = np.column_stack(
                    [
                        x[:, 2] - fracture_limit,
                        injection_days - constraints.max_injection_days,
                        constraints.min_float_margin_frac - margin,
                        stress - constraints.max_rod_stress_utilization_frac,
                    ]
                )
                _ = radius

        result = minimize(
            SurrogateProblem(),
            NSGA2(pop_size=settings.population_size),
            ("n_gen", settings.generations),
            seed=settings.seed,
            verbose=False,
        )
        if result.X is None:
            return []
        population = np.atleast_2d(result.X)
        return [np.asarray(row, dtype=float) for row in population]

    def _apply_robustness(
        self, candidates: list[CssCandidate], cycle_number: int, n_jobs: int = -1
    ) -> None:
        """Re-score candidates across an ensemble of plausible parameter sets.

        A design that only works for one guess at the permeability is not a
        recommendation. Each candidate is run against an ensemble drawn around
        the current parameter estimate and ranked on a conservative percentile
        of cycle oil, with any candidate that violates a constraint anywhere in
        the ensemble penalised in proportion.
        """
        from joblib import Parallel, delayed

        from app.twin.assimilation import build_prior_ensemble

        robust = self.settings.robust
        names = [
            "permeability_md",
            "skin_dimensionless",
            "net_pay_thickness_m",
            "thermal_loss_multiplier",
        ]
        ensemble = build_prior_ensemble(
            self.parameters,
            names,
            robust.ensemble_size,
            spread_frac=robust.parameter_spread_frac,
            seed=self.settings.seed,
        )
        members = [
            TwinParameters.from_vector(names, row, self.parameters) for row in ensemble
        ]
        for candidate in candidates:
            outcomes = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self.evaluate)(candidate.plan, cycle_number, member)
                for member in members
            )
            self._evaluations += len(members)
            oil = np.asarray([outcome.oil_m3 for outcome in outcomes])
            violations = float(np.mean([not o.is_feasible for o in outcomes]))
            candidate.robust_percentile_oil_m3 = float(
                np.percentile(oil, robust.percentile)
            )
            candidate.robust_violation_rate = violations
            # Rank on the conservative percentile rather than the mean, and
            # penalise designs that break somewhere in the ensemble.
            candidate.score = self.score(candidate.outcome) * (
                candidate.robust_percentile_oil_m3 / max(candidate.outcome.oil_m3, 1.0e-6)
            ) - 5.0 * violations

    @staticmethod
    def _pareto_filter(candidates: list[CssCandidate]) -> list[CssCandidate]:
        """Non-dominated set over more oil, lower steam-to-oil ratio, lower cost."""
        points = np.asarray(
            [
                [
                    -candidate.outcome.oil_m3,
                    min(candidate.outcome.steam_oil_ratio, 1.0e6),
                    candidate.outcome.energy_cost_usd,
                ]
                for candidate in candidates
            ],
            dtype=float,
        )
        keep: list[CssCandidate] = []
        for index, candidate in enumerate(candidates):
            dominated = np.all(points <= points[index], axis=1) & np.any(
                points < points[index], axis=1
            )
            if not bool(np.any(dominated)):
                keep.append(candidate)
        return keep

    # ------------------------------------------------------------ explanation
    def _explain(self, recommendation: CssRecommendation) -> str:
        """The reason for the recommendation, in plain words."""
        best = recommendation.recommended
        baseline = recommendation.baseline
        plan = best.plan
        base_plan = baseline.plan
        parts: list[str] = []
        for label, new, old, unit in (
            ("steam volume", plan.steam_volume_m3_cwe, base_plan.steam_volume_m3_cwe, "m3"),
            (
                "injection rate",
                plan.injection_rate_m3_per_day_cwe,
                base_plan.injection_rate_m3_per_day_cwe,
                "m3 per day",
            ),
            (
                "injection pressure",
                plan.injection_pressure_kpa,
                base_plan.injection_pressure_kpa,
                "kPa",
            ),
            ("soak", plan.soak_days, base_plan.soak_days, "days"),
        ):
            if abs(new - old) > 0.02 * max(abs(old), 1.0):
                parts.append(
                    f"{label} {'raised' if new > old else 'lowered'} from {old:.0f} to "
                    f"{new:.0f} {unit}"
                )
        if abs(plan.steam_quality_frac - base_plan.steam_quality_frac) > 0.01:
            parts.append(
                f"steam quality set to {plan.steam_quality_frac:.2f} from "
                f"{base_plan.steam_quality_frac:.2f}"
            )

        change = ", ".join(parts) if parts else "the historical design is already the best found"
        effect = (
            f"Predicted cycle oil {baseline.outcome.oil_m3:.0f} to "
            f"{best.outcome.oil_m3:.0f} m3, steam to oil ratio "
            f"{baseline.outcome.steam_oil_ratio:.2f} to {best.outcome.steam_oil_ratio:.2f}, "
            f"energy cost {baseline.outcome.energy_cost_usd:,.0f} to "
            f"{best.outcome.energy_cost_usd:,.0f} USD."
        )
        coupling = (
            f"At the end of the cycle the heated zone would be at "
            f"{best.outcome.end_of_cycle_temp_c:.0f} degrees, which leaves the rods a float "
            f"margin of {best.outcome.float_margin_index:.2f} and a rod stress of "
            f"{best.outcome.rod_stress_utilisation_frac * 100:.0f} percent of the Goodman "
            "allowable at the current pump setting. That is the pump-compatibility "
            "constraint, and it is what stops the optimizer choosing a design that makes oil "
            "and then floats the rods."
        )
        cutoff = (
            f"Recommended production cut-off is day {best.outcome.production_days:.0f}, where "
            "another day of pumping stops paying for its own energy."
        )
        robustness = ""
        if best.robust_percentile_oil_m3 is not None:
            robustness = (
                f" Across the parameter ensemble the conservative "
                f"{self.settings.robust.percentile:.0f}th percentile of cycle oil is "
                f"{best.robust_percentile_oil_m3:.0f} m3, and "
                f"{best.robust_violation_rate * 100:.0f} percent of the ensemble members "
                "violate a constraint."
            )
        return (
            change[0].upper() + change[1:] + ". " + effect + " " + coupling + " " + cutoff
            + robustness
        )

    def _confidence(self, recommendation: CssRecommendation, surrogate_error: float) -> float:
        """How much to trust this recommendation, between 0 and 1."""
        best = recommendation.recommended
        agreement = clamp(1.0 - surrogate_error / 0.25, 0.0, 1.0)
        robustness = 1.0 - best.robust_violation_rate
        margin_headroom = clamp(
            (best.outcome.float_margin_index - self.settings.constraints.min_float_margin_frac)
            / 0.3,
            0.0,
            1.0,
        )
        base = 0.35 + 0.25 * agreement + 0.2 * robustness + 0.2 * margin_headroom
        if not best.outcome.is_feasible:
            base *= 0.4
        return float(clamp(base, 0.05, 0.95))


def marginal_cutoff_day(
    config: FieldConfig,
    daily_oil_m3: list[float],
    daily_energy_kwh: list[float],
) -> tuple[float, str]:
    """The day another day of pumping stops paying for its own energy.

    Equation: stop at the first day where oil rate times the oil price falls
    below the daily energy cost.
    Units: days.
    This is the marginal oil per unit energy rule the brief asks for, expressed
    in money so it can be explained to an operator without a chart.
    """
    if len(daily_oil_m3) != len(daily_energy_kwh):
        raise PhysicsDomainError("Oil and energy series must have the same length.")
    economics = config.economics
    for day, (oil, energy) in enumerate(zip(daily_oil_m3, daily_energy_kwh, strict=True)):
        revenue = oil * economics.oil_price_usd_per_m3
        cost = energy * economics.electricity_cost_usd_per_kwh
        if revenue < cost:
            return float(day), (
                f"on day {day} the oil is worth {revenue:.0f} USD against {cost:.0f} USD "
                "of energy"
            )
    total_oil = float(np.sum(daily_oil_m3))
    return float(len(daily_oil_m3)), (
        f"the cycle stayed economic for the whole {len(daily_oil_m3)} days, producing "
        f"{total_oil:.0f} m3 of oil, and {M3_PER_BBL:.3f} m3 per barrel was used for the "
        "barrel conversion"
    )
