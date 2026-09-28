# SIH26120 Build Brief (v2, research-backed)

## Digital Twin for Well-to-Surface Optimization of CSS and SRP Operations, Baghewala Heavy Oil Field (Oil India Limited)

How to use this file: save it as `BRIEF.md` in an empty repository, copy Appendix A into `CLAUDE.md` in the same folder, then paste Appendix B as your first message to Claude Code. Everything below is written to Claude Code, so it says "you".

---

## 1. Mission and definition of done

Build a working, tested, demonstrable digital twin that couples three things in one loop: (1) the thermal reservoir around a CSS well, (2) the wellbore and rod string, and (3) the surface pumping unit and its drive. On top of the twin, build forecasting, diagnostics, and two optimizers: one for CSS cycle design and one for real-time pump operation.

The work is done only when all of these are true:

1. A fresh clone plus `docker compose up` gives a running backend and dashboard with seeded demo data in under 5 minutes.
2. `pytest` passes, coverage on `twin/` and `optimize/` is at least 80 percent, `ruff` and `mypy` are clean.
3. `scripts/run_scenario.py` prints a baseline-versus-optimized KPI table on a held-out synthetic well, and the optimized case violates no safety constraint.
4. Every model and every number shown in the UI is either labelled SYNTHETIC or traceable to a cited source in `docs/SOURCES.md`.
5. `docs/PHYSICS.md`, `docs/ASSUMPTIONS.md`, `docs/DATA_SCHEMA.md`, `docs/VALIDATION.md`, `docs/DEMO_SCRIPT.md`, and the README exist and match the code.

---

## 2. Working rules (non-negotiable)

1. Read this whole file first. Write `PLAN.md` (architecture, order of work, open questions), then start Milestone 1.
2. Work one milestone at a time. After each: run tests and linters, update `PLAN.md`, commit. Never start the next milestone with a failing test.
3. Never invent field data and never present simulated results as field results. Use the labels defined in section 9.
4. Physics first, machine learning second. Every ML model must be compared with a physics or rule baseline, and the comparison goes in `reports/`.
5. No field constant is hard-coded in logic. Depth, viscosity constants, rod string, pump size, pressure limits and so on live in `config/*.yaml` with units in the key names (`pressure_kpa`, `temp_c`, `rate_m3_per_day`).
6. SI units internally. Convert only at the edges (API, UI, CSV import).
7. Every physics function has a docstring stating equation, units, assumptions, and source, plus at least one test against an analytic or limiting case.
8. When something is ambiguous, choose the most defensible engineering reading, write it in `docs/ASSUMPTIONS.md` with the reason, and continue. Ask the user only for choices that are expensive to reverse.
9. Do not use the tilde character anywhere (code comments, docs, UI text). Write "about" or "approximately".
10. Write like a professional engineer: plain, specific, no hype words in docs or UI copy.
11. If a test fails, fix the cause. Never loosen a tolerance or delete a test just to pass, unless you document why the tolerance was wrong.
12. Keep a running `CHANGELOG.md`. Keep commits small and messages specific.

---

## 3. Verified field facts (researched) and what is still unknown

These come from Oil India's own site and press coverage. Use them to anchor defaults. Where a source is weak it is marked. Where sources disagree, keep the value configurable and note the conflict in `docs/SOURCES.md`.

| Fact | Value | Source and confidence |
| --- | --- | --- |
| Reservoir | Jodhpur Sandstone, heavy oil, average depth 1150 m | Oil India, Rajasthan Fields page. High. |
| Crude viscosity | 10,000 to 13,000 cP at 50 degrees C | Oil India, Rajasthan Fields page. High. |
| API gravity and reservoir temperature | 17 to 19 degrees API, 46 to 48 degrees C | The SIH problem statement itself. High for the task. |
| Basin | Bikaner-Nagaur, Rajasthan | Oil India. High. |
| History | Heavy oil discovered 1991; first pilot CSS 2006; commercial production and first successful CSS 2017 (Oil India page). An older trade report dates "India's first CSS" at well BGW-8 to Dec 2018. | Sources conflict on the "first CSS" date. Do not state one in the UI. |
| Lift | Produced with sucker rod pumps, both conventional and hydraulic SRP units | Oil India. High. |
| Completion | Most wells thermally completed with thermal wellhead and vacuum insulated tubing (VIT) | Oil India. High. |
| Surface | Crude stored in tanks at each well site, heated with steam and hot water from a mobile steam generator, then moved by bowser to ONGC's North Santhal CTF at Mehsana | Oil India. High. |
| Well counts | Oil India page says 56 drilled and 34 producing, and elsewhere on the same page 35 drilled; press in April 2026 said 52 drilled, 33 operational, CSS done on 19 wells. | Inconsistent. Treat as "several tens of wells". |
| Field rate | Above 1100 barrels per day on the Oil India page (older snapshot said above 600) | Medium. Do not quote as current. |
| Newer techniques | Fishbone drilling (first in well BGW#40), barefoot completion, electric downhole heaters, diluent injection, high-temperature thermal wellheads, SAGD planned | Oil India and BusinessToday, Apr 2026. Medium to high. |
| CSS pilot response | A reported five to six fold production increase after steam on the pilot well | Scribd copy of an Oil India note. Low. Use only as an order-of-magnitude sanity check, never as a target. |
| Fractured reservoir | A trade article calls the Jodhpur Sandstone naturally fractured | Low (secondary article). Make fracture-enhanced steam distribution an optional switch, off by default. |

### Unknowns you must parameterize (never guess silently)

Rod string taper and grade, pump depth and plunger diameter, tubing size, VIT overall heat transfer coefficient, steam quality and temperature actually delivered, cycle steam volumes and injection rates, soak durations, fracture pressure, pumping unit type and rating per well, VFD limits, hydraulic unit pressure limits, thermal properties of the rock, relative permeability, water cut behavior. Put engineering-plausible defaults in `config/` and mark every one `assumed: true` with a reason. Provide a calibration workflow so real numbers can replace them without code changes.

Plausible default anchors (all flagged as assumptions, to be replaced by real data): viscosity anchor of 11,500 cP at 50 degrees C (midpoint of the published range), a second anchor for the Walther fit in the low tens of cP near 200 degrees C, tubing 2-7/8 in, rods 7/8 in tapered, pump depth about 1100 m, steam saturation temperature computed from injection pressure with a steam table (use `iapws` or a small table). Do not present these as Baghewala facts.

---

## 4. The problem statement (source of truth)

Heavy crude (17 to 19 degrees API) from the Jodhpur Sandstone: high viscosity, high asphaltene content, low reservoir pressure, low reservoir temperature (46 to 48 degrees C), poor mobility under primary recovery. CSS and SRP are both essential, but today they are designed separately from historical experience. After steam injection the reservoir cools, viscosity rises, and pump efficiency falls, energy use rises, rods float, rods fail, and recovery drops.

Current challenges: CSS parameters (steam volume, injection pressure, soak time, production cut-off) rest on history; SRP settings (stroke length, SPM, VFD) are adjusted manually and reactively; heavy crude causes rod floating, impact loading, pump unsetting, rod failures, maintenance; reservoir, wellbore, and pump are not optimized together; no predictive analytics, so SOR and energy per barrel are higher than needed.

Required: an AI-enabled Well-to-Surface Digital Twin integrating reservoir, wellbore, and surface systems for real-time monitoring, prediction, and optimization that will: optimize CSS cycle parameters; predict reservoir heating, cooling, and production; continuously optimize SRP stroke speed and SPM to well conditions; detect rod floating and minimize impact loading; improve pump efficiency and equipment reliability; optimize steam and energy use and cut operating cost.

Expected benefits: more oil and recovery, lower SOR, lower energy per barrel, fewer rod failures and pump unsettings, longer equipment life, predictive decision making.

Data the field has (so the system must ingest it): production history, CSS cycle records, steam injection parameters, VFD and SRP operating data, rod failure and pump unsetting history, completion and reservoir data, fluid properties and pressure data.

---

## 5. Competitive landscape and how we differ

Public repositories for this same problem statement already exist. Read them for ideas, but do not copy code. What they do, and where the gaps are:

| Project | Approach | Gap we can fill |
| --- | --- | --- |
| ThermoLift (gauravchahal20/120-SIH) | Browser-only JavaScript, ISA-101 styled operator screens, Marx-Langenheim heating, Ramey wellbore heat loss, Andrade viscosity, Gibbs wave dynamometer cards, rod-fall limit, steam-unit scheduler, ridge regression cycle model with R squared about 0.80, honest synthetic-data note | No real backend, no data assimilation, no separate ground-truth simulator, single-run ML on the same generator |
| USHNA (Zaraar21-cloud/USHNA) | Physics-first design: Marx-Langenheim, Boberg-Lantz, Ramey, Walther, Gibbs, float margin index; planned EnKF, MPC, Bayesian optimization | Late phases (assimilation, MPC) still marked pending; no dashboard evidence |
| viveky1621 | Lumped single-well reservoir plus surrogate ML | Explicitly one lumped volume and no steam-front model; ML learns its own simulator |
| SIHWinners/SIH2 | Sensor dashboard, IsolationForest anomaly detection, random-forest SPM recommendation | Statistical, little physics, no CSS-to-pump coupling |

Our differentiators (build these deliberately, they are what to show the judges):

1. **A ground-truth simulator that is different from the twin.** Test the twin against a higher-fidelity hidden model, not against itself (section 7). Most entries skip this.
2. **True coupling with a shared constraint.** The CSS optimizer is constrained by predicted pump behavior (float margin, rod load, viscosity at pump), and the pump optimizer is informed by predicted reservoir cooling. Show this chain on one screen.
3. **Field-specific hardware.** Baghewala uses vacuum insulated tubing, thermal wellheads, and both conventional and hydraulic SRP units. Model VIT heat loss and support a hydraulic unit type with arbitrary intra-stroke velocity profiles.
4. **Intra-stroke speed control for rod float**, not just a lower SPM. Published control patents note that slowing the motor after float is detected does not prevent float because the rods may already be in the high-speed part of the stroke. Optimize the downstroke velocity profile.
5. **Data assimilation and calibration** (Ensemble Kalman Filter or rolling re-fit) so the twin stays in sync with a live well, plus a CSV calibration wizard for real Oil India data.
6. **Fleet layer.** Mobile steam generators are shared across wells, so scheduling injection across wells is a real decision. Include a CP-SAT scheduler.
7. **Uncertainty everywhere.** Prediction intervals on forecasts, robust optimization against parameter uncertainty, and a confidence label on every recommendation.
8. **Explainability for operators.** Every recommendation shows the reason in plain words and the constraint that was binding.
9. **Engineering rigor visible in the repo:** tests, validation report, reproducible runs, honest limitations.

---

## 6. Product concept and architecture

One paragraph: a Python backend holds a coupled model of one or more wells. Live or replayed sensor data keeps the model in sync. Three intelligence layers sit on top: forecasting (near-wellbore temperature, viscosity, production), diagnostics (dynamometer card classification, rod float, fluid pound, unseating and rod failure risk), and optimization (CSS cycle parameters per well, pump settings now, fleet steam scheduling). A React dashboard shows state, recommendations with reasons and confidence, and the expected savings.

```
Dashboard (React + TypeScript + Vite, Plotly)
        | REST + WebSocket
API (FastAPI, pydantic v2)
   |-- Ingestion + QA        (CSV adapters, validators, unit checks, gap and spike rules)
   |-- Twin engine           (reservoir + wellbore + rod/pump + surface, coupled)
   |-- Assimilation          (EnKF or rolling parameter re-fit)
   |-- Analytics (ML)        (forecasts, card classifier, failure risk)
   |-- Optimizers            (CSS cycle, pump control, fleet scheduler, supervisory guard)
   |-- Truth simulator       (high fidelity, used only to generate synthetic data and to test the twin)
Storage: SQLite for the demo (Postgres-compatible\ schema via SQLAlchemy), Parquet for bulk time series
```

Stack: Python 3.11, numpy, scipy, numba (only where profiling proves it helps), pandas, pydantic v2, FastAPI, SQLAlchemy, scikit-learn, LightGBM, PyTorch (small 1D CNN), pymoo (NSGA-II), Optuna, OR-Tools (CP-SAT), iapws or a steam table module, pytest, hypothesis, ruff, mypy, pre-commit. Frontend: React, TypeScript, Vite, Plotly.js, TanStack Query. Docker Compose for one-command run.

---

## 7. Truth simulator versus twin (avoid the "inverse crime")

If the same equations generate the data and also power the twin, every result looks perfect and means nothing. So build two levels:

**Truth simulator (`simulate/truth/`)**, used only to make synthetic data and to judge the twin:

- Axisymmetric (r, z) finite-volume thermal model around the well, several layers with different permeability (a heterogeneity switch, including an optional high-permeability fracture streak).
- Energy balance with conduction, advection of injected hot fluid distributed by layer mobility, heat loss to over and underburden, gravity override as an option.
- Oil flow by Darcy law with temperature-dependent mobility, pressure decline, and water cut that grows over cycles.
- Wellbore heat loss by a transient numerical (finite difference) solution instead of the analytic Ramey form.
- Rod string dynamics by a finer wave-equation grid with nonlinear friction and a slightly different damping law.
- Sensor model: noise, drift, dropouts, stuck values, spikes, and occasional unit errors, seeded.
- Hidden parameters are drawn per well from priors and stored in a sealed file that the twin code cannot read.

**Twin (`twin/`)**, the reduced-order model with analytic or lumped physics described in section 8, calibrated from the noisy observations only.

Acceptance: after calibration on cycles 1 to 3 of a well, the twin must predict cycle 4 and 5 oil rate and temperature with errors reported in `docs/VALIDATION.md`. Report honestly where it fails (for example fracture-dominated wells).

---

## 8. Physics specifications (twin layer)

Document each equation, source, unit, and assumption in `docs/PHYSICS.md`. Every function gets tests (see section 17).

### 8.1 Fluid (`twin/fluid.py`)

- Viscosity versus temperature: ASTM D341 (Walther): log10(log10(nu + 0.7)) = A - B *log10(T_K), with nu in cSt. Convert between cP and cSt with oil density (specific gravity from API: SG = 141.5 / (131.5 + API)); include a density-versus-temperature correction. Fit A and B from two or more viscosity anchors. An Arrhenius form (mu = a* exp(b / T_K)) is selectable in config.
- Water-in-oil emulsion viscosity uplift with water cut (Richardson or Brinkman), selectable, with an inversion-point guard.
- Asphaltene effect: a configurable multiplier, off by default and labelled an assumption.
- Non-Newtonian option (power-law with temperature-dependent consistency) for drag calculations, off by default.
- Guards: reject non-positive viscosity or temperature below absolute zero with a clear error.

### 8.2 Reservoir thermal model (`twin/reservoir.py`)

- **Injection.** Heated area growth from Marx and Langenheim: A_s = (Q_i *M_R* h) / (4 *K_ob* M_ob *dT)* G(t_D), with t_D = 4 *K_ob* M_ob *t / (M_R^2* h^2) and G(t_D) = exp(t_D) *erfc(sqrt(t_D)) + 2* sqrt(t_D / pi) - 1. Q_i is the net heat rate at the sandface after wellbore loss. Heated radius r_h = sqrt(A_s / pi). Include steam quality and latent heat.
- **Soak.** Conduction-driven temperature decline of the heated zone, with a lumped heat balance.
- **Production cooling.** Boberg and Lantz style heat balance for the average heated-zone temperature, including enthalpy removed by produced fluids and conduction losses.
- **Inflow.** Steady-state composite radial flow with a heated zone out to r_h and cold oil beyond: q_h / q_c = mu_c *ln(r_e / r_w) / (mu_h* ln(r_h / r_w) + mu_c * ln(r_e / r_h)), with mu_h evaluated at the average heated-zone temperature and a pressure decline term for the low reservoir pressure.
- **Cycle decline.** Cycle-to-cycle degradation (loss of energy, rising water cut) as fitted parameters.
- **Optional switches (off by default):** fracture-enhanced heat distribution, dilation and recompaction effects.
- Outputs per day: near-wellbore temperature, heated radius, sandface viscosity, expected inflow, cumulative oil and steam, SOR.

### 8.3 Wellbore (`twin/wellbore.py`)

- **Injection.** Ramey-type transient heat transmission for steam down the tubing, with an overall heat transfer coefficient that reflects vacuum insulated tubing (much smaller than bare tubing). Include quality loss along the string.
- **Production.** Fluid temperature up the string, so the temperature and viscosity at pump depth and along the rods are computed, not assumed.
- **Pressure.** Hydrostatic plus friction; injection pressure checked against a fracture pressure limit from config.
- Limiting-case tests: no heat loss when the coefficient is zero; fluid temperature approaches the formation temperature as time goes to infinity for a bare well.

### 8.4 Rod string and pump (`twin/srp/`)

- **Unit kinematics (plug-in by unit type).** Conventional crank unit (position and velocity from crank angle and geometry, including the sinusoidal speed variation); Mark II or similar; hydraulic unit with an arbitrary velocity-versus-position profile, separate up and down speeds, end-of-stroke dwell, pressure and flow limits.
- **Wave equation (Gibbs).** Solve u_tt = a^2 *u_xx - c* u_t on the tapered string (finite differences, CFL enforced). Damping c = pi *v* a / (2 * L), with v a dimensionless damping factor made a function of local oil viscosity (larger for heavy oil), bounded, and calibratable. Acoustic speed a is about 4,900 m/s in steel; use config.
- **Cards.** Forward (downhole to surface) and inverse (surface to downhole) solutions with the same solver. Extract features: peak and minimum polished rod load, load range, fluid load, fillage, net stroke.
- **Drag.** Viscous drag on the rods from the annular Couette plus pressure-driven return flow solution using local viscosity along the string, plus optional coupling and sinker-bar terms.
- **Float margin.** Float Margin Index FMI = (W_buoyant - F_drag_max) / W_buoyant on the downstroke, evaluated per rod section and for the whole string. FMI at or below zero means the rods cannot fall as fast as the polished rod. Card-based detector: minimum polished rod load below a small threshold (published control methods use a few hundred pounds, about 1 kN) or negative loads on the downstroke; report a float index between 0 and 1, the fraction of the downstroke affected, and an estimated impact load in kN when the polished rod and rods re-engage.
- **Rod stress.** Modified Goodman utilization per rod section; cumulative fatigue damage counter (Miner style) that feeds the failure-risk model.
- **Drive and energy.** Polished rod power, gearbox torque check, motor electrical power with an efficiency map, VFD frequency to SPM mapping (conventional) or hydraulic pump power (hydraulic), and kWh per barrel.
- **Pump.** Volumetric efficiency including slippage (temperature and viscosity dependent), gas interference, fillage.

### 8.5 Surface (`twin/surface.py`)

- Tank heating energy demand for stored heavy crude from the mobile steam generator (heat needed to bring tank contents to a pumpable temperature), and bowser loading schedule as a light model. Purpose: complete the "well-to-surface" story and include surface steam use in the energy and cost KPIs.

### 8.6 Coupled engine (`twin/coupled.py`)

Advance one well through a full CSS cycle in configurable steps: reservoir sets sandface temperature and viscosity, wellbore heat loss sets viscosity at pump and along the rods, viscosity sets drag and damping, drag sets loads, float margin, fillage, and power, and the pump rate limits production, which feeds back to cooling and to the production cut-off decision. Two modes: `simulate` (free-running) and `assimilate` (state and parameter correction from measurements). Every run reproducible from config and seed.

### 8.7 Assimilation (`twin/assimilation.py`)

Ensemble Kalman Filter over a small parameter set (effective permeability or skin, thermal loss factor, rod damping factor, VIT heat coefficient) with covariance inflation and bounds. Fallback: rolling-window least squares re-fit. Report parameter uncertainty in the API.

---

## 9. Data plan

Real Baghewala data is not public. The system runs in two modes with identical code after ingestion:

- **SYNTHETIC mode (default).** Generated by the truth simulator (section 7). A visible SYNTHETIC badge appears on every chart, table, and report.
- **REAL mode.** The same tables loaded from CSV files that follow `docs/DATA_SCHEMA.md`. The validator reports missing columns, unit problems, outliers, and gaps instead of crashing. A calibration wizard fits fluid, thermal, and rod parameters from the uploaded data and shows before-and-after fit quality.

Tables (minimum): `wells`, `reservoir`, `fluid` (viscosity versus temperature points, API, water cut), `css_cycles` (steam volume, injection rate and pressure, quality, temperature, soak days, production days, cut-off reason), `production_daily`, `srp_telemetry` (SPM, stroke, VFD frequency or hydraulic pressure, motor current, power, load and position series or card arrays), `failures` (type, depth, date, cycle). Add `steam_generator_log` (unit id, start, end, rig move time) for the scheduler.

Synthetic generator requirements: at least 10 wells, 6 to 10 cycles each, 2 years of telemetry at 1 minute resolution for a subset and 15 minutes for the rest, fixed seed, identical output for the same seed (tested). Realistic behavior: production peaks after soak then decays as the zone cools, water cut rises, later cycles are weaker, viscosity at the pump rises with cooling, and that raises rod loads and float events. Injected labelled faults: rod float, fluid pound, gas interference, tagging, worn pump, sticking, pump unseating, rod parting, VIT degradation (rising heat loss), and sensor faults.

---

## 10. Machine learning specifications

For each model define target, features, split (by well and by time, never random across time), metrics, baseline, and save metrics under `reports/`.

1. **Production forecast.** LightGBM with quantile outputs for daily oil rate and cycle cumulative oil. Baselines: exponential decline fit and the physics twin alone. Metrics: MAPE, RMSE, interval coverage. Add conformal calibration of the intervals.
2. **Temperature and viscosity residual model.** Physics twin gives the base trajectory; a gradient-boosted model learns the residual. Report improvement over physics alone.
3. **Downhole card classifier.** Small 1D CNN on normalized load versus position. Classes: normal, fluid pound, gas interference, tagging, worn pump or valve leak, sticking, rod float, unseating, parted rod, plus an unknown option when confidence is low. Baseline: rule-based classifier from card features. Report per-class precision and recall and a confusion matrix.
4. **Failure risk.** Probability of rod parting and pump unseating in the next N days from stress cycles, float history, viscosity trend, cycle age, load range. Calibrated gradient boosting with feature contribution per prediction; report AUC, precision at fixed recall, calibration curve.
5. **Anomaly detector** for sensor faults (a light isolation forest or residual test on twin-versus-measured) so bad data does not drive recommendations.
6. `scripts/train_models.py` trains everything from the synthetic data, saves versioned artifacts, and prints a compact metrics table.

Because data is synthetic, every report states that the numbers show the pipeline works and are not field accuracy. Do not report ML accuracy from a random split on time series.

---

## 11. Optimization and control specifications

### 11.1 CSS cycle optimizer

- Variables per next cycle per well: steam volume, injection rate and pressure, steam quality, soak days, production cut-off rule.
- Objectives: maximize cycle oil, minimize SOR, minimize steam and energy cost. NSGA-II (pymoo) returns a Pareto front; a weighted score with weights in config picks a recommendation.
- Constraints: injection pressure below fracture limit, steam generator capacity, soak window, minimum heated radius, and the pump-compatibility constraint (predicted viscosity trajectory must keep the float margin and rod load inside limits at a chosen pump setting). This constraint is the integrated part.
- Evaluation by the coupled twin or a surrogate verified against it periodically; report surrogate error.
- Robustness: evaluate candidates across an ensemble of plausible parameter sets and rank by a conservative percentile, so recommendations survive uncertainty.
- Output: Pareto front, one recommended set, predicted oil, SOR, energy, confidence, comparison with the historical rule (mean of the last three cycles), and a recommended cut-off day from marginal oil per unit energy.

### 11.2 Pump control optimizer

- Variables: SPM, stroke length (allowed discrete set), and an intra-stroke velocity profile (separate upstroke and downstroke multipliers, deceleration into the top of the downstroke, dwell); for hydraulic units, the velocity-versus-position profile directly.
- Objective: maximize net production per unit energy.
- Constraints: float margin above a safe value, peak load below rod and unit rating, minimum load above a small positive margin, gearbox torque below rating, fillage above a minimum, VFD or hydraulic limits, and a limit on the rate of change between successive setpoints.
- Method: model-predictive control over a short horizon on the fast vectorized rod model (Bayesian or constrained local search is acceptable if it responds within seconds). The reservoir cooling forecast enters as the predicted viscosity trend.
- Output: recommended setpoints with predicted change in production, float index, peak load, kWh per barrel, and a plain-language reason such as "Downstroke slowed 15 percent because the viscosity at pump depth rose and the float margin fell to 0.12."

### 11.3 Fleet steam scheduler (OR-Tools CP-SAT)

Sequence CSS injections across wells given a limited number of mobile steam generators, rig move times, injection durations, well due dates from decline forecasts, and cost of deferred oil. Output a Gantt chart, waiting days per well, and deferred oil. Compare against a simple earliest-due-first rule.

### 11.4 Supervisory controller and safety

Loop: read data, run QA, update twin, run diagnostics, run optimizers, apply the guard, publish the recommendation. The guard clamps every setpoint to the safe envelope and limits change per step. Modes: `advisory` (default, human approves) and `auto` (simulation only). Never connect auto mode to hardware. Every recommendation is logged with inputs, outputs, reason, and constraint status.

---

## 12. API (FastAPI, prefix `/api/v1`)

`GET /wells`; `GET /wells/{id}/state`; `GET /wells/{id}/history`; `GET /wells/{id}/cards/latest`; `POST /wells/{id}/forecast`; `POST /wells/{id}/optimize/css`; `POST /wells/{id}/optimize/pump`; `GET /wells/{id}/risk`; `POST /fleet/schedule`; `POST /simulate/scenario`; `POST /ingest/csv`; `POST /calibrate`; `WS /stream/{id}`; `GET /health`. Use pydantic response models, one consistent error format, OpenAPI docs, request validation with readable messages, upload size limits, and no secrets in code (provide `.env.example`).

---

## 13. Dashboard and UX

Pages: (1) Field overview with well status, SOR, oil rate, kWh per barrel, open recommendations; (2) Well twin view with linked reservoir, wellbore, and pump panels and a time slider that replays a cycle, showing viscosity at sandface and at pump side by side; (3) Pump and rod diagnostics with live cards, classifier label and confidence, float gauge, load envelope, rod stress by section; (4) CSS planner with Pareto front, click-through to the predicted cycle, comparison against historical practice; (5) Pump optimizer with current versus recommended setpoints, predicted effect, reason text, Approve and Reject in advisory mode; (6) Fleet schedule Gantt; (7) Scenario and KPIs, baseline versus optimized; (8) Data and model health with the ingestion report, calibration fit, model versions, drift, and the validation summary.

Design rules: calm engineering look (grayscale-first with color reserved for alarms, in the spirit of ISA-101), consistent units with a toggle, light and dark themes, responsive layout, keyboard-accessible controls, loading and error states everywhere, no placeholder text, a permanent SYNTHETIC badge when applicable, and an "Explain this" drawer on every recommendation that lists inputs, binding constraints, and confidence. Before writing UI code, use the frontend-design skill if it is available.

---

## 14. Repository layout

```
sih26120-well-twin/
  BRIEF.md  CLAUDE.md  PLAN.md  README.md  CHANGELOG.md
  docs/  ARCHITECTURE.md PHYSICS.md ASSUMPTIONS.md DATA_SCHEMA.md VALIDATION.md SOURCES.md DEMO_SCRIPT.md SIH_SUBMISSION.md
  config/  field.yaml  wells/*.yaml  optimizer.yaml  truth_priors.yaml
  backend/app/
    api/ core/ ingestion/
    twin/  fluid.py reservoir.py wellbore.py surface.py coupled.py assimilation.py
           srp/ kinematics_conventional.py kinematics_hydraulic.py wave.py cards.py floating.py stress.py power.py
    simulate/  truth/ generator.py scenarios.py
    ml/  features.py forecast_production.py residual_thermal.py card_classifier.py failure_risk.py anomaly.py registry.py
    optimize/  css_optimizer.py pump_optimizer.py fleet_scheduler.py controller.py guard.py
    tests/
  frontend/src/
  scripts/  generate_data.py train_models.py run_scenario.py calibrate.py
  reports/   data/synthetic/   docker-compose.yml   .env.example
```

---

## 15. Milestones and acceptance criteria

Commit at the end of each. Tests must pass before moving on.

- **M1 Foundations.** Skeleton, config loading with validation and readable errors, units helpers, logging, pre-commit, CI script. Accept: linters and pytest run clean on the wired project.
- **M2 Fluid, reservoir, wellbore.** Sections 8.1 to 8.3 with tests and docs. Accept: one CSS cycle simulates end to end from a script with plausible curves (temperature decays, viscosity rises, rate decays, SOR rises), and VIT versus bare-tubing comparison shows the expected direction.
- **M3 Rod and pump.** Section 8.4 including both unit types. Accept: analytic wave test passes; heavy-oil case shows lower float margin than light-oil case; slowing only the downstroke raises the margin more than slowing both strokes equally at equal production; energy per barrel computed.
- **M4 Truth simulator, generator, ingestion.** Sections 7 and 9. Accept: reproducible dataset; ingestion validator accepts generated CSVs and rejects deliberately broken ones with clear messages; twin calibrates on cycles 1 to 3 and predictions on cycles 4 and 5 are reported in `VALIDATION.md`.
- **M5 Coupled engine and assimilation.** Sections 8.5 to 8.7. Accept: the cooling-to-viscosity-to-load chain is visible in one run; assimilation reduces parameter error versus no assimilation on held-out wells.
- **M6 ML layer.** Section 10. Accept: each model beats its baseline on held-out wells or the report says honestly that it does not.
- **M7 Optimizers.** Section 11. Accept: on a held-out well, optimized operation beats the historical rule on SOR and kWh per barrel without any constraint violation; robust mode never violates constraints across the parameter ensemble; guard tests prove clamps and rate limits.
- **M8 API.** Section 12 with client tests. Accept: valid schemas; pump optimize returns within a few seconds and CSS optimize within a minute at demo settings.
- **M9 Dashboard.** Section 13. Accept: all pages work against the API with live replay, no console errors, runs under Docker Compose.
- **M10 Polish and submission pack.** README with screenshots, architecture diagram, limitations, deployment path, `DEMO_SCRIPT.md` (5 minutes), `SIH_SUBMISSION.md`. Accept: fresh clone to working demo in under 5 minutes.

Priority if time runs short: M1 to M5 and M7 (pump optimizer plus CSS optimizer) matter most, then a slim dashboard (well twin view, diagnostics, KPIs). Cut M6 items in this order: anomaly detector, failure risk, then card classifier. Never cut the truth-versus-twin validation.

---

## 16. Verification protocol

### 16.1 Physics test catalog (write these first)

- Viscosity strictly decreases with temperature; Walther fit reproduces a synthetic table to within 1 percent; the 50 degree C anchor returns the configured value.
- Marx-Langenheim: with zero conduction losses the heated area equals injected heat divided by (M_R *h* dT); heated area grows monotonically with time.
- Composite radial inflow: ratio equals 1 when r_h equals r_w; approaches mu_c / mu_h times a geometry term as r_h approaches r_e.
- Ramey: no heat loss when the coefficient is zero; fluid temperature tends to formation temperature at large time for bare tubing; VIT delivers hotter fluid than bare tubing under the same conditions.
- Wave solver: undamped uniform rod matches the analytic solution within tolerance; CFL violation raises a clear error; energy does not grow without an input.
- Float margin: rises when viscosity falls, falls when downstroke speed rises, and equals a hand-calculated value in a unit test.
- Energy balance: injected heat equals heat stored plus heat lost within a stated tolerance in the reservoir model.
- Property tests with hypothesis for unit conversions and monotonicity.

### 16.2 Sanity table (run in CI as warnings, fail on gross violation)

Viscosity at 50 degrees C inside the published 10,000 to 13,000 cP range; viscosity at 150 degrees C at least two orders of magnitude lower; heated radius in the tens of metres, not kilometres; SOR in a plausible range for CSS (a few tonnes of steam per tonne of oil, and rising with cycle number); pump rate never exceeds displacement times speed; no negative flows; no NaN or inf anywhere (guard and fail loudly).

### 16.3 Reproducibility and performance

Fixed seeds; pinned dependencies; deterministic scenario outputs (tested by hashing a result); pump optimizer under a few seconds (profile and record in README); a `make verify` target that runs linters, tests, a mini scenario, and a smoke test of the API.

### 16.4 Independent review step

At the end of M5 and M7, spawn a separate review pass (a fresh agent or a fresh prompt with no memory of the build) to audit `PHYSICS.md` against the code and to attack the optimizer with extreme inputs. Fix everything it finds and log it in `docs/VALIDATION.md`.

---

## 17. Failure modes to avoid

1. Fitting the twin to data made by the twin. (Fixed by section 7.)
2. Reporting ML accuracy from a random split on time series.
3. Hard-coded field numbers; silent unit mix-ups (kg/cm2 versus kPa, bbl versus m3, cP versus cSt).
4. An optimizer that "wins" by violating a constraint the baseline respected. Always check constraint status of both cases.
5. Recommendations that jump between extremes from one step to the next.
6. Claiming gains as field results. Say "simulated".
7. Dashboard screens that look complete but call stub endpoints.
8. Cosmetic complexity: prefer a smaller model that is correct and tested over a large one that is not.
9. Untested edge cases: zero flow, zero water cut, missing telemetry, very low viscosity (hot well), very high viscosity (cold well).
10. Forgetting hydraulic units: setpoints, limits, and cards differ from crank units.

---

## 18. KPI and evaluation protocol

Run each held-out well through the same hidden-truth scenario twice (historical-practice controls versus optimizer controls), with the same random seeds and the same disturbances. Report: cycle oil, SOR, kWh per barrel including surface steam, float events per month and average float index, peak load and load range, expected rod failure and unseating counts from the risk model, average fillage, steam and energy cost. Give mean and spread across wells and seeds, not a single best case. Include ablations: pump-only optimization, CSS-only optimization, and joint optimization, so the value of coupling is measured rather than claimed.

---

## 19. SIH submission pack (write `docs/SIH_SUBMISSION.md`)

Check the official portal for the current template and deadline before submitting. Typical fields: team details, problem statement (SIH26120, Oil India Limited, Smart Automation, Software), proposed solution, technical approach and architecture diagram, innovation and differentiators, feasibility and viability (data needs, calibration path, shadow-mode pilot before advisory use), impact and benefits (with clearly labelled simulated numbers), technology stack, research and references.

Also produce: a 5-minute demo script, a 10-slide outline, and a Q&A sheet for judges covering at least: where the data comes from and why synthetic; how we avoid fooling ourselves (truth versus twin); what happens with wrong parameters (uncertainty and robust mode); why a hydraulic unit needs different control; how the system stays safe (guard, advisory mode); how it deploys at Oil India (edge gateway, on-premise, OPC-UA or MQTT adapter); what we would calibrate first with real data; how it compares with existing entries and commercial tools; cost and effort to pilot.

---

## 20. Sources (put in `docs/SOURCES.md` with access dates)

- Oil India, Rajasthan Fields page: <https://www.oil-india.com/rajasthan-fields>
- BusinessToday, Oil India ramps up Rajasthan output (Apr 2026): <https://www.businesstoday.in/india/story/hormuz-blocked-india-turns-to-thar-desert-oil-india-ramps-up-crude-output-from-rajasthan-field-524088-2026-04-05>
- Oil & Gas Journal, OIL starts CSS of well in Rajasthan (Dec 2018): <https://www.ogj.com/drilling-production/production-operations/unconventional-resources/article/17296851/oil-starts-css-of-well-in-rajasthan>
- Rod float mitigation with VFD control: US patent 7547196 (Google Patents) and US 10094371 on load and speed control
- Gibbs wave equation diagnostics: US patents 3343409 and 10947833 (background), Gibbs, S. G., Rod Pumping: Modern Methods of Design, Diagnosis and Surveillance
- Marx and Langenheim (1959), Reservoir heating by hot fluid injection, Trans. AIME 216
- Ramey (1962), Wellbore heat transmission, J. Petroleum Technology 14(4)
- Boberg and Lantz (1966), Calculation of the production rate of a thermally stimulated well, J. Petroleum Technology
- Butler (1991), Thermal Recovery of Oil and Bitumen
- ASTM D341, viscosity-temperature charts for liquid petroleum products
- Reservoir heating in CSS, Physics of Fluids 37 (2025): <https://pubs.aip.org/aip/pof/article/37/4/047110/3342003>
- SIH 2026 problem statement list: <https://github.com/NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements>
- Existing entries reviewed for landscape only: gauravchahal20/120-SIH, Zaraar21-cloud/USHNA, viveky1621/sih26120-baghewala-digital-twin, SIHWinners/SIH2

---

## Appendix A: contents for `CLAUDE.md`

```
# Project rules (SIH26120 well twin)
- Read BRIEF.md before any work. PLAN.md tracks progress. Work one milestone at a time.
- Run `make verify` (ruff, mypy, pytest, mini scenario, API smoke test) before every commit. Never commit failing tests.
- SI units inside the code. Units in variable names. Convert at the edges only.
- No field constants in logic; use config/*.yaml. Mark every assumed value assumed: true with a reason.
- Every physics function: docstring with equation, units, assumptions, source; tests against analytic or limiting cases.
- The truth simulator must never be imported by twin/ code. The twin only sees noisy observations.
- Label all synthetic outputs SYNTHETIC. Never present simulated results as field results.
- ML: split by well and time. Always compare with the stated baseline. Report honestly.
- Optimizer comparisons must check constraint status for both baseline and optimized runs.
- Never use the tilde character in code, comments, docs, or UI text.
- Never loosen a test tolerance to pass without documenting why in docs/VALIDATION.md.
- Ask the user only when a decision is expensive to reverse; otherwise document the assumption and continue.
```

## Appendix B: first message to paste into Claude Code

```
Read BRIEF.md and CLAUDE.md completely. Then:
1. Write PLAN.md: a 15-line summary of the brief in your own words, the folder layout you will create, the milestone order, the top 5 technical risks, and any assumption you plan to change (with reasons).
2. Set up Milestone 1 (skeleton, config with validation, units, logging, Makefile with `make verify`, pre-commit, CI script) and commit.
3. Continue milestone by milestone. After each milestone, print a short status: what passed, what is assumed, what is next.
Do not skip the truth-versus-twin separation in section 7, and do not report any result you did not compute in this repository.
```

