"""Build the synthetic dataset from the truth simulator.

Output is a set of tables that match ``docs/DATA_SCHEMA.md`` exactly, so the
same ingestion code path reads generated data and real Oil India CSV exports.
Every file carries a SYNTHETIC marker, the hidden parameters are written to a
sealed file, and the whole thing is reproducible from the seed in
``config/truth_priors.yaml``.

Wells are generated in parallel because each one is independent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from app.core.config import FieldConfig, config_dir, data_dir, load_field_config
from app.core.logging import get_logger
from app.simulate.truth.priors import (
    HiddenParameters,
    draw_hidden_parameters,
    load_truth_priors,
    seal_parameters,
)
from app.simulate.truth.sensors import SensorSettings
from app.simulate.truth.well import TruthSetpoint, TruthWell, TruthWellResult
from app.twin.reservoir import InjectionPlan

LOGGER = get_logger(__name__)

DATA_MODE_MARKER = "SYNTHETIC"
"""Written into every table so no chart can lose track of where the data came from."""

STEAM_PER_METRE_OF_PAY_M3 = 150.0
"""Historical practice rule the generator uses to size a steam slug."""


@dataclass(frozen=True)
class GeneratorSettings:
    """What to generate."""

    well_count: int
    cycles_min: int
    cycles_max: int
    production_days_max: int
    high_rate_well_fraction: float
    telemetry_minutes_high_rate: int
    telemetry_minutes_low_rate: int
    seed: int

    @classmethod
    def from_priors(
        cls, priors: dict[str, Any], production_days_max: int = 150
    ) -> GeneratorSettings:
        """Build from the ``wells`` block of the priors file."""
        wells = priors["wells"]
        return cls(
            well_count=int(wells["count"]),
            cycles_min=int(wells["cycles_min"]),
            cycles_max=int(wells["cycles_max"]),
            production_days_max=production_days_max,
            high_rate_well_fraction=float(wells["high_rate_well_fraction"]),
            telemetry_minutes_high_rate=int(wells["telemetry_minutes_high_rate"]),
            telemetry_minutes_low_rate=int(wells["telemetry_minutes_low_rate"]),
            seed=int(priors["seed"]),
        )


def historical_practice_plan(config: FieldConfig, hidden: HiddenParameters) -> InjectionPlan:
    """The steam design an operator working from history would use.

    Steam volume is scaled to the net pay by the usual rule of thumb, the rate
    is whatever one mobile generator sustains, and the pressure and soak come
    straight from the field defaults. This is the baseline the CSS optimizer has
    to beat, and it is deliberately reasonable rather than a straw man.
    """
    css = config.css
    volume_m3 = STEAM_PER_METRE_OF_PAY_M3 * hidden.net_pay_thickness_m
    return InjectionPlan(
        steam_volume_m3_cwe=float(np.clip(volume_m3, 600.0, 3200.0)),
        injection_rate_m3_per_day_cwe=css.injection_rate_m3_per_day_cwe,
        injection_pressure_kpa=css.injection_pressure_kpa,
        steam_quality_frac=css.steam_quality_frac,
        soak_days=css.soak_days,
        cutoff_marginal_energy_ratio=css.cutoff_marginal_energy_ratio,
    )


def historical_practice_setpoint(
    config: FieldConfig, hidden: HiddenParameters, target_fillage_frac: float = 0.75
) -> TruthSetpoint:
    """The pump setting an operator working from history would use.

    Speed is set so the displacement is a little above the rate the well is
    expected to deliver early in a cycle, which is what operators actually do:
    they would rather over-pump and let the pump-off controller cut in than
    leave oil behind. It is a defensible rule and it is also exactly the habit
    that causes fluid pound and rod float later in a cycle, which is what the
    pump optimizer is there to fix.
    """
    expected_rate_m3_per_day = 0.006 * hidden.permeability_md * hidden.net_pay_thickness_m / 12.0
    expected_rate_m3_per_day = float(np.clip(expected_rate_m3_per_day, 1.5, 18.0))
    stroke_m = config.srp.stroke_length_m
    plunger_stroke_m = 0.9 * stroke_m
    displacement_per_spm = config.srp.plunger_area_m2 * plunger_stroke_m * 1440.0
    spm = expected_rate_m3_per_day / max(target_fillage_frac * displacement_per_spm, 1e-9)
    spm = float(np.clip(spm, config.srp.spm_min, config.srp.spm_max))
    return TruthSetpoint(spm=spm, stroke_length_m=stroke_m)


def simulate_one_well(
    well_index: int,
    config: FieldConfig,
    priors: dict[str, Any],
    settings: GeneratorSettings,
) -> tuple[HiddenParameters, TruthWellResult, dict[str, Any]]:
    """Generate one well's complete history."""
    well_id = f"BGW-{well_index + 1:02d}"
    hidden = draw_hidden_parameters(well_id, well_index, priors)
    rng = np.random.default_rng(hidden.seed + 3)
    cycles = int(rng.integers(settings.cycles_min, settings.cycles_max + 1))
    high_rate = rng.random() < settings.high_rate_well_fraction
    interval = (
        settings.telemetry_minutes_high_rate if high_rate else settings.telemetry_minutes_low_rate
    )
    well = TruthWell(
        config=config,
        hidden=hidden,
        sensor_settings=SensorSettings.from_dict(priors["sensors"]),
        fault_rates=priors["faults"],
        telemetry_interval_minutes=interval,
    )
    plan = historical_practice_plan(config, hidden)
    setpoint = historical_practice_setpoint(config, hidden)
    result = well.run_history(
        cycles=cycles,
        plan=plan,
        setpoint=setpoint,
        max_production_days=settings.production_days_max,
    )
    metadata = {
        "well_id": well_id,
        "unit_type": hidden.unit_type,
        "insulation_type": config.wellbore.insulation_type,
        "pump_depth_m": config.srp.pump_depth_m,
        "plunger_diameter_m": config.srp.plunger_diameter_m,
        "stroke_length_m": setpoint.stroke_length_m,
        "baseline_spm": setpoint.spm,
        "telemetry_interval_minutes": interval,
        "cycles": cycles,
        "data_mode": DATA_MODE_MARKER,
        "api_gravity_deg": config.fluid.api_gravity_deg,
        "reservoir_depth_m": config.reservoir.depth_m,
    }
    return hidden, result, metadata


def _steam_generator_log(cycle_frame: pd.DataFrame, config: FieldConfig, seed: int) -> pd.DataFrame:
    """Reconstruct which mobile steam generator served which injection.

    The mobile units are shared, so the log is what the fleet scheduler is
    calibrated and compared against.
    """
    rng = np.random.default_rng(seed)
    units = config.surface.steam_generator_count
    rows: list[dict[str, Any]] = []
    ordered = cycle_frame.sort_values(["cycle_number", "well_id"]).reset_index(drop=True)
    available_day = np.zeros(units)
    for _, row in ordered.iterrows():
        unit = int(np.argmin(available_day))
        duration = float(row["steam_volume_m3_cwe"]) / float(row["injection_rate_m3_per_day_cwe"])
        start = float(available_day[unit])
        end = start + duration
        available_day[unit] = end + config.surface.rig_move_days
        rows.append(
            {
                "unit_id": f"MSG-{unit + 1}",
                "well_id": row["well_id"],
                "cycle_number": int(row["cycle_number"]),
                "start_day": round(start, 2),
                "end_day": round(end, 2),
                "rig_move_days": config.surface.rig_move_days,
                "steam_volume_m3_cwe": float(row["steam_volume_m3_cwe"]),
                "data_mode": DATA_MODE_MARKER,
            }
        )
        _ = rng.random()
    return pd.DataFrame(rows)


def generate_dataset(
    output_dir: Path | None = None,
    well_count: int | None = None,
    production_days_max: int = 150,
    n_jobs: int = -1,
    write_well_configs: bool = True,
) -> dict[str, Path]:
    """Generate the whole synthetic dataset and write it to disk.

    Args:
        output_dir: Where to write. Defaults to ``data/synthetic``.
        well_count: Override the well count from the priors file.
        production_days_max: Cap on the production phase of each cycle.
        n_jobs: Parallel workers. Minus one uses every core.
        write_well_configs: Also write a per-well overlay under ``config/wells``
            so the twin has a configuration for each generated well.

    Returns:
        A mapping from table name to the CSV path written.
    """
    config = load_field_config()
    priors = load_truth_priors()
    settings = GeneratorSettings.from_priors(priors, production_days_max)
    count = well_count or settings.well_count
    target = output_dir or (data_dir() / "synthetic")
    target.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating synthetic dataset", extra={"wells": count, "output": str(target)})
    outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(simulate_one_well)(index, config, priors, settings) for index in range(count)
    )

    hidden_list: list[HiddenParameters] = []
    well_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    card_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    fluid_rows: list[dict[str, Any]] = []
    sensor_faults: dict[str, dict[str, dict[str, list[int]]]] = {}

    for hidden, result, metadata in outputs:
        hidden_list.append(hidden)
        well_rows.append(metadata)
        daily_rows.extend(result.daily_rows)
        cycle_rows.extend(result.cycle_rows)
        telemetry_rows.extend(result.telemetry_rows)
        card_rows.extend(result.card_rows)
        failure_rows.extend(result.failure_rows)
        sensor_faults[hidden.well_id] = result.sensor_fault_log
        for temp_c in (30.0, 50.0, 80.0, 120.0, 200.0):
            fluid_rows.append(
                {
                    "well_id": hidden.well_id,
                    "temp_c": temp_c,
                    "viscosity_cp": _laboratory_viscosity_cp(hidden, temp_c),
                    "api_gravity_deg": config.fluid.api_gravity_deg,
                    "data_mode": DATA_MODE_MARKER,
                }
            )

    for frame_rows in (daily_rows, cycle_rows, telemetry_rows, failure_rows):
        for row in frame_rows:
            row["data_mode"] = DATA_MODE_MARKER

    tables = {
        "wells": pd.DataFrame(well_rows),
        "css_cycles": pd.DataFrame(cycle_rows),
        "production_daily": pd.DataFrame(daily_rows),
        "srp_telemetry": pd.DataFrame(telemetry_rows),
        "failures": pd.DataFrame(failure_rows),
        "fluid": pd.DataFrame(fluid_rows),
    }
    tables["steam_generator_log"] = _steam_generator_log(
        tables["css_cycles"], config, settings.seed
    )

    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = target / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
        if name in {"production_daily", "srp_telemetry"} and not frame.empty:
            frame.to_parquet(target / f"{name}.parquet", index=False)

    cards_path = target / "dynamometer_cards.json"
    cards_path.write_text(json.dumps(card_rows), encoding="utf-8")
    written["dynamometer_cards"] = cards_path

    faults_path = target / "sensor_fault_log.json"
    faults_path.write_text(json.dumps(sensor_faults), encoding="utf-8")
    written["sensor_fault_log"] = faults_path

    sealed = seal_parameters(hidden_list, target)
    written["sealed_truth_parameters"] = sealed

    if write_well_configs:
        _write_well_configs(hidden_list)

    manifest = {
        "data_mode": DATA_MODE_MARKER,
        "seed": settings.seed,
        "well_count": count,
        "production_days_max": production_days_max,
        "tables": {name: str(path.name) for name, path in written.items()},
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()},
        "note": (
            "Every number in these files is simulated. They show that the pipeline "
            "works. They are not Oil India field data and must never be presented as such."
        ),
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written["manifest"] = manifest_path
    LOGGER.info("Synthetic dataset written", extra={"path": str(target)})
    return written


def _laboratory_viscosity_cp(hidden: HiddenParameters, temp_c: float) -> float:
    """A laboratory viscosity measurement for this well's crude.

    Built from the hidden 50 degree C value on the same Walther line the truth
    simulator uses. This is the table the calibration workflow fits the twin's
    fluid model to, so the twin can learn a well-specific viscosity without ever
    seeing the hidden parameters.
    """
    anchor_hot_cp = 12.0 * (hidden.viscosity_50c_cp / 11500.0) ** 0.35
    temps_k = np.asarray([50.0, 200.0]) + 273.15
    viscosities = np.asarray([hidden.viscosity_50c_cp, anchor_hot_cp])
    z_values = np.log10(np.log10(viscosities + 0.7))
    slope, intercept = np.polyfit(np.log10(temps_k), z_values, 1)
    inner = float(np.clip(intercept + slope * math.log10(temp_c + 273.15), -1.2, 1.4))
    return float(np.clip(10.0 ** (10.0**inner) - 0.7, 0.3, 5.0e7))


def _write_well_configs(hidden_list: list[HiddenParameters]) -> None:
    """Write a per-well overlay so the twin has a configuration for every well.

    Only things an engineer could know without a well test go in: the unit type
    from the completion record and the laboratory viscosity anchors. The hidden
    reservoir properties stay hidden. Calibration is what fills the rest in.
    """
    import yaml

    wells_dir = config_dir() / "wells"
    wells_dir.mkdir(parents=True, exist_ok=True)
    for hidden in hidden_list:
        overlay = {
            "srp": {"unit_type": hidden.unit_type},
            "fluid": {
                "viscosity_anchors": [
                    {
                        "temp_c": 50.0,
                        "viscosity_cp": round(_laboratory_viscosity_cp(hidden, 50.0), 1),
                    },
                    {
                        "temp_c": 200.0,
                        "viscosity_cp": round(_laboratory_viscosity_cp(hidden, 200.0), 3),
                    },
                ]
            },
        }
        path = wells_dir / f"{hidden.well_id}.yaml"
        header = (
            "# Well overlay generated from the synthetic dataset.\n"
            "# Only information an engineer could have without a well test is here:\n"
            "# the completion record and a laboratory viscosity table. Everything\n"
            "# else is left to calibration.\n"
        )
        path.write_text(header + yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
