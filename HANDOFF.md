# HANDOFF: continuing the SIH26120 well twin

Paste the block in "The prompt" below as your first message to Claude Code, from the
repository root. Everything above and below it is context for you, the human.

---

## The prompt

```
Read BRIEF.md, CLAUDE.md, PLAN.md and HANDOFF.md completely before touching anything.

This repository is a partly built entry for SIH26120: a well-to-surface digital twin for
cyclic steam stimulation and sucker rod pump operations at Baghewala. Milestones M1 to M5
are finished and committed. M7 is about half done. M6, M8, M9 and M10 have not been started.
HANDOFF.md has the exact status, the known open problems and the order I want you to work in.

Before you write any code:
1. Run `make install` if the virtualenv is missing, then `make test`. All 296 tests must
   pass and `make lint` and `make typecheck` must be clean. If they are not, fix that first
   and tell me what was broken.
2. Run `.venv/bin/python scripts/generate_data.py --wells 3 --days 60 --jobs 3` to get a
   small synthetic dataset locally. The committed dataset is gitignored.
3. Print a short plan for the next milestone before starting it.

Then work through the milestones in the order given in HANDOFF.md, one at a time. After each
milestone: run `make lint`, `make typecheck`, `make test`, update PLAN.md and CHANGELOG.md,
and commit with a specific message. Never start a milestone with a failing test.

Follow every rule in CLAUDE.md without exception. In particular:
- No field constant in logic; everything goes in config/*.yaml with a provenance entry, and
  a test enforces that.
- SI units inside the code, units in every variable name, conversion only at the edges.
- Every physics function gets a docstring with equation, units, assumptions and source, plus
  a test against an analytic or limiting case.
- app/twin, app/optimize and app/ml must never import app.simulate. tests/test_boundaries.py
  enforces this. Do not weaken it.
- Never loosen a test tolerance to make something pass. If a tolerance is genuinely wrong,
  change it and write down why in docs/VALIDATION.md.
- Label every synthetic output SYNTHETIC. Never present a simulated number as a field result.
- No tilde character anywhere in code, comments, docs or UI text.

Where you find a real problem with something I built, say so and fix it rather than working
around it. Where a decision is cheap to reverse, make it, write the assumption into
docs/ASSUMPTIONS.md and keep going. Only stop and ask me if a choice is expensive to reverse.
```

---

## Status as of this handoff

Three commits are on `master`:

| Commit | Contents |
| --- | --- |
| `c943477` | M1 to M3: foundations, fluid, reservoir, wellbore, rod string and pump |
| `06d1b8d` | M4 to M5: truth simulator, generator, ingestion, coupled engine, assimilation |
| `f88dd10` | M6 label but actually M7 part one: CSS and pump optimizers, guard |

Gates right now: **296 tests pass, ruff clean, mypy clean on 40 source files.**

Note the last commit message says M6; it is mislabelled. It contains optimizer work, not the
machine learning layer. Do not go looking for ML code that is not there.

### What works and is tested

- **Config layer.** `config/field.yaml` carries every field constant with a `provenance`
  block. A test fails if any configured value has no recorded source, and another test fails
  if a value taken from Oil India or from the problem statement is marked `assumed`.
- **Fluid.** ASTM D341 Walther with a selectable Arrhenius form, emulsion uplift with an
  inversion guard, density and specific heat against temperature.
- **Reservoir.** Marx and Langenheim heated area evaluated with `erfcx` so it stays accurate
  at large times, Boberg and Lantz style cooling from exact conduction unit solutions,
  composite radial inflow with a transient radius of investigation.
- **Wellbore.** Ramey heat transmission both directions, steam quality loss down the string,
  vacuum insulated versus bare tubing. VIT loses 1 percent of injected heat against 14 percent
  for bare tubing, and keeps the produced fluid at 109 C at the wellhead against 33 C.
- **Rod string.** API Spec 11E four-bar kinematics (not an in-line slider crank, which would
  be exactly symmetric and would erase the upstroke and downstroke asymmetry). Damped wave
  solver validated against the exact analytic damped standing wave to 3 percent in magnitude
  and 0.05 rad in phase. Gibbs frequency-domain inverse solution cross-validated against the
  forward solver.
- **Rod float.** Annular Couette plus pressure-driven drag, node-wise drag linearisation fed
  into the wave solver, float margin index per taper section and from the surface card, and an
  impact load estimate. A cold well floats and a hot one does not; slowing only the downstroke
  beats slowing both strokes equally at the same pumping rate. Both are tested.
- **Cards.** Feature extraction and a rule-based classifier that recovers all nine injected
  fault classes on the generated data. This is the baseline the learned classifier must beat.
- **Truth simulator.** Axisymmetric finite-volume thermal reservoir with layers, gravity
  override and an optional fracture streak; transient numerical wellbore with a resolved
  formation; fine-grid rod solver with nonlinear Coulomb friction; seeded sensor model with
  noise, drift, dropouts, stuck values, spikes and unit errors. Genuinely different equations
  from the twin, which is the whole point.
- **Ingestion.** Schema plus validator that names the suspected unit when a range looks wrong,
  and a cross-table check that catches a barrel to cubic metre swap that range checks cannot
  see on their own.
- **Coupled engine.** Closes reservoir to wellbore to rods to pump and back. One run shows the
  full chain: the zone cools 308 to 217 C, viscosity at mid-string rises 7 to 534 cP, float
  margin falls 1.00 to 0.52, peak polished rod load rises 26 to 46 kN.
- **Assimilation.** Ensemble Kalman Filter with bounds, inflation and parallel members.
  Calibration cuts permeability error from 106 to 368 percent down to 12 to 94 percent against
  the sealed truth parameters, and fits the hidden viscosity to within 0.05 percent.

### What exists but is not finished

- `app/optimize/guard.py` works and is not tested.
- `app/optimize/pump_optimizer.py` works. It answers in about 2.2 s with roughly 4000
  candidate evaluations, and its fast surrogate agrees with the full rod solver to 0.4 percent
  on float margin and about 2.4 percent on peak load. Not tested.
- `app/optimize/css_optimizer.py` runs NSGA-II on a trained surrogate, re-evaluates the whole
  front with the real model, and applies the pump-compatibility constraint and robust mode.
  It completes in about 38 s. Not tested, and it has an open problem, below.

### What does not exist at all

- `app/optimize/fleet_scheduler.py` (CP-SAT), `app/optimize/controller.py`
- `app/simulate/scenarios.py`
- `app/ml/` in its entirety
- `app/api/` in its entirety
- `frontend/` beyond an empty `src` directory
- `scripts/run_scenario.py`, `scripts/train_models.py`, `scripts/calibrate.py`,
  `scripts/smoke_api.py`. The Makefile already references all four, so `make verify` fails
  until they exist.
- Everything in `docs/`. The directory is empty. `README.md`, `docker-compose.yml` and
  `.env.example` do not exist either.

---

## Open problems to fix, in priority order

1. **The CSS optimizer surrogate is not accurate enough on the float margin.** On the last
   run the held-out error was 0.348 against a configured limit of 0.12
   (`config/optimizer.yaml`, `css.surrogate.max_acceptable_mape_frac`). It logs a warning and
   still re-evaluates the front with the real model, so the recommendation is not wrong, but
   NSGA-II is searching a poor approximation of the constraint that matters most. Likely
   causes worth checking: the float margin is close to a step function in some corners of the
   design space, and 120 Latin hypercube samples across six dimensions is thin. Options are
   more samples, a classifier for feasibility plus a regressor for the objectives, or
   evaluating the float margin directly rather than through the surrogate since the closed
   form is already cheap.

2. **On the one well tried, the CSS optimizer returned the historical design unchanged.** That
   may well be the right answer for that well, but it has not been checked against a well
   where a change is clearly needed. Confirm on a well with a thin pay or a low permeability
   before trusting it.

3. **`make verify` is broken** because three of the scripts it calls do not exist yet.

---

## Suggested order of work

**First, M7 to completion.** It is the highest-value milestone left and the brief says so.

- Write `backend/tests/test_optimize.py` covering the guard (clamps, rate limits, rejection of
  non-finite values, advisory versus auto), the pump optimizer (the fast surrogate agrees with
  the full solver; a cold well gets a slower downstroke; a candidate never wins by violating a
  constraint the baseline respected) and the CSS optimizer (the pump-compatibility constraint
  actually rejects designs that would float the rods; the robust mode never picks a candidate
  that breaks across the ensemble).
- Fix the surrogate problem in item 1 above.
- Write `fleet_scheduler.py` with OR-Tools CP-SAT: sequence CSS injections across wells given
  the mobile steam generator count from config, rig move times, injection durations, due dates
  from decline forecasts and the cost of deferred oil. Compare against an earliest-due-first
  rule. `data/synthetic/steam_generator_log.csv` already has the shape the output should take.
- Write `controller.py`: read, QA, update twin, diagnose, optimize, guard, publish, log.
- Write `app/simulate/scenarios.py` and `scripts/run_scenario.py`. This is the acceptance
  criterion: run a held-out well through the same hidden-truth scenario twice, once with
  historical-practice controls and once with optimizer controls, same seeds and same
  disturbances, and print the KPI table. **Check constraint status for both runs.** The brief
  is explicit that an optimizer which wins by violating a constraint the baseline respected is
  a failure, not a result. Include the ablations: pump only, CSS only, and joint, so the value
  of coupling is measured rather than claimed.

**Then M6, the machine learning layer.** Split by well and by time, never randomly across
time. Every model is compared with the stated baseline and the comparison goes in `reports/`.
The rule classifier in `app/twin/srp/cards.py` is the baseline for the card model, and it is
a strong one, so be honest if the learned model does not beat it. If time is short the brief
says to cut, in order: anomaly detector, failure risk, then card classifier.

**Then M8, the API**, M9 the dashboard, and M10 the documentation and submission pack. The
endpoint list is in BRIEF.md section 12 and the dashboard pages are in section 13.

Docs are not optional. `docs/PHYSICS.md`, `docs/ASSUMPTIONS.md`, `docs/DATA_SCHEMA.md`,
`docs/VALIDATION.md`, `docs/SOURCES.md`, `docs/DEMO_SCRIPT.md` and `docs/SIH_SUBMISSION.md`
are all in the definition of done. Much of PHYSICS and ASSUMPTIONS can be lifted from the
docstrings and the `provenance` block, which were written with that in mind.

---

## Things that will bite you if you do not know them

- **The rod wave grid resolution is load dependent.** An earlier configuration used 16 nodes
  per section, which looked converged against one light load case and was 32 percent wrong on
  a heavier one. It is now 28. `test_wave_grid_resolution_is_converged` compares against a
  grid with three times the resolution across a spread of speeds, loads and fillages. Do not
  lower it to make something faster.
- **Use `run_cycle_fast` for anything that needs hundreds of cycle evaluations.** It skips the
  rod wave equation and agrees with the full `run_cycle` to about 3 percent on cycle oil and
  steam-to-oil ratio, at roughly 20 times the speed. The full engine takes about 20 s for a
  150 day cycle.
- **Two rules end a cycle and both must match between the fast and the full path.** There is
  an economic rule and a marginal oil per unit energy rule. When only one path had the second
  rule the fast path ran every cycle to the day limit and overstated cycle oil by 34 percent.
- **The pump optimizer horizon has to be refined once.** The temperature profile up the string
  depends on the rate, and the rate depends on the setpoint. `optimize` builds the horizon on
  a first guess, predicts the rate, then rebuilds. Without that the verification against the
  full solver disagrees by 39 percent.
- **Conduction alone can raise deliverability for a while.** It carries heat from a small very
  hot zone out into the cold annulus where the flow resistance actually is. That is why a soak
  helps, and it is why the deliverability decay test has to be over a production period rather
  than a shut-in.
- **`data/synthetic/` is gitignored.** Regenerate it. Generation is parallel across wells and
  takes a few minutes for the full 12 wells at 150 days.
- **`config/wells/*.yaml` are generated** by the data generator and only contain what an
  engineer could know without a well test: the completion record and a laboratory viscosity
  table. Everything else is left to calibration. Keep it that way.

## Commands

```
make install        create the virtualenv and install pinned dependencies
make test           full test suite
make test-fast      skip the slow tests
make lint           ruff
make typecheck      mypy
make coverage       coverage on twin/ and optimize/, fails under 80 percent
make data           generate the synthetic dataset
make verify         the gate before every commit, currently broken, see open problem 3
```

macOS note: LightGBM needs `brew install libomp`. It is already installed on the machine this
was built on.
