"""Ingestion, validation and calibration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.config import FieldConfig
from app.core.errors import DataValidationError
from app.ingestion import (
    TABLE_SCHEMAS,
    calibrate_well,
    fit_fluid_anchors,
    load_table,
    validate_dataset,
    validate_table,
)
from app.ingestion.calibration import FitQuality, estimate_decline_from_cycles
from app.ingestion.validators import (
    Severity,
    cross_check_cycle_and_daily,
    cross_check_failures_against_wells,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
HAS_DATA = (DATA_DIR / "production_daily.csv").exists()
needs_data = pytest.mark.skipif(
    not HAS_DATA, reason="run `make data` to generate the synthetic dataset first"
)


def _minimal_daily() -> pd.DataFrame:
    days = np.arange(30.0)
    return pd.DataFrame(
        {
            "well_id": ["BGW-01"] * 30,
            "day": days,
            "cycle_number": [1] * 30,
            "oil_rate_m3_per_day": np.linspace(6.0, 2.0, 30),
            "water_rate_m3_per_day": np.linspace(0.8, 0.4, 30),
            "near_well_temp_c": np.linspace(280.0, 220.0, 30),
        }
    )


# ------------------------------------------------------------------------ schema
def test_every_schema_has_keys_and_descriptions() -> None:
    for schema in TABLE_SCHEMAS.values():
        assert schema.key_columns
        assert schema.description
        for column in schema.columns:
            assert column.description, f"{schema.name}.{column.name} has no description"


def test_numeric_columns_carry_units_in_their_names() -> None:
    """The naming rule the brief sets: units belong in the variable name."""
    allowed_bare = {
        "cycle_number",
        "production_day",
        "day",
        "spm",
        "baseline_spm",
        "timestamp_day",
        "start_day",
        "end_day",
        "soak_days",
        "production_days",
        "rig_move_days",
        "cycles",
        "telemetry_interval_minutes",
    }
    for schema in TABLE_SCHEMAS.values():
        for column in schema.numeric_columns():
            if column.name in allowed_bare:
                continue
            tail = column.name.rsplit("_", 1)[-1]
            assert tail in {
                "m",
                "c",
                "kpa",
                "cp",
                "n",
                "w",
                "frac",
                "deg",
                "m3",
                "hz",
                "day",
            } or column.name.endswith(
                ("_m3_per_day", "_m3_per_day_cwe", "_m3_cwe")
            ), f"{schema.name}.{column.name} does not state its unit"


# --------------------------------------------------------------------- validation
def test_clean_table_is_accepted() -> None:
    report = validate_table("production_daily", _minimal_daily())
    assert report.is_accepted
    assert report.frame is not None


def test_missing_required_column_is_an_error() -> None:
    frame = _minimal_daily().drop(columns=["oil_rate_m3_per_day"])
    report = validate_table("production_daily", frame)
    assert not report.is_accepted
    assert report.errors[0].code == "missing_column"
    assert "oil_rate_m3_per_day" in report.errors[0].message
    assert "DATA_SCHEMA" in (report.errors[0].suggestion or "")


def test_non_numeric_value_is_an_error() -> None:
    frame = _minimal_daily()
    frame["oil_rate_m3_per_day"] = frame["oil_rate_m3_per_day"].astype(object)
    frame.loc[3:5, "oil_rate_m3_per_day"] = "n/a"
    report = validate_table("production_daily", frame)
    assert any(issue.code == "non_numeric" for issue in report.errors)


def test_out_of_range_value_is_reported() -> None:
    frame = _minimal_daily()
    frame.loc[2, "near_well_temp_c"] = 5000.0
    report = validate_table("production_daily", frame)
    assert any(issue.code in {"out_of_range", "suspected_unit_error"} for issue in report.issues)


def test_pressure_in_the_wrong_unit_is_named() -> None:
    """kg/cm2 reported as kPa is a real mistake in Indian field practice."""
    frame = pd.DataFrame(
        {
            "well_id": ["BGW-01"] * 12,
            "cycle_number": range(1, 13),
            "steam_volume_m3_cwe": [1800.0] * 12,
            "injection_rate_m3_per_day_cwe": [160.0] * 12,
            "injection_pressure_kpa": [112.0] * 12,
            "steam_quality_frac": [0.78] * 12,
            "soak_days": [6.0] * 12,
            "production_days": [120.0] * 12,
            "oil_m3": [500.0] * 12,
        }
    )
    report = validate_table("css_cycles", frame)
    named = [issue for issue in report.issues if issue.code == "suspected_unit_error"]
    assert named, "a pressure of 112 kPa should be recognised as a unit error"
    # 112 is plausible as psi and as kg/cm2, so both must be offered rather than
    # one of them guessed at.
    suggestion = named[0].suggestion or ""
    assert "kgf_per_cm2" in suggestion
    assert "psi" in suggestion


def test_unit_swap_that_range_checks_cannot_see_is_caught_across_tables() -> None:
    """An oil rate of 19 is plausible in either unit. The cycle total is not."""
    daily = _minimal_daily()
    cycles = pd.DataFrame(
        {
            "well_id": ["BGW-01"],
            "cycle_number": [1],
            "oil_m3": [float(daily["oil_rate_m3_per_day"].sum())],
        }
    )
    assert cross_check_cycle_and_daily(cycles, daily) == []

    wrong = daily.copy()
    wrong["oil_rate_m3_per_day"] = wrong["oil_rate_m3_per_day"] / 0.158987294928
    issues = cross_check_cycle_and_daily(cycles, wrong)
    assert issues and issues[0].severity == Severity.ERROR
    assert "barrels" in issues[0].message


def test_unknown_well_in_failures_is_reported() -> None:
    wells = pd.DataFrame({"well_id": ["BGW-01"]})
    failures = pd.DataFrame({"well_id": ["BGW-01", "BGW-99"]})
    issues = cross_check_failures_against_wells(failures, wells)
    assert issues and "BGW-99" in issues[0].message


def test_stuck_sensor_is_only_checked_on_measured_channels() -> None:
    """A constant setpoint is not a stuck transmitter."""
    frame = _minimal_daily()
    frame["spm"] = 3.0
    frame["near_well_temp_c"] = 250.0
    report = validate_table("production_daily", frame)
    stuck = [issue.column for issue in report.issues if issue.code == "stuck_sensor"]
    assert "near_well_temp_c" in stuck
    assert "spm" not in stuck


def test_time_gaps_are_reported() -> None:
    frame = _minimal_daily()
    frame.loc[15:, "day"] = frame.loc[15:, "day"] + 40.0
    report = validate_table("production_daily", frame)
    assert any(issue.code == "time_gaps" for issue in report.issues)


def test_duplicate_keys_are_reported() -> None:
    frame = pd.concat([_minimal_daily(), _minimal_daily().head(3)], ignore_index=True)
    report = validate_table("production_daily", frame)
    assert any(issue.code == "duplicate_keys" for issue in report.issues)


def test_unknown_table_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="Unknown table"):
        validate_table("not_a_table", _minimal_daily())


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="File not found"):
        load_table(tmp_path / "absent.csv", "production_daily")


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "production_daily.csv"
    path.write_text("well_id,day\n")
    report = load_table(path)
    assert not report.is_accepted


def test_missing_required_table_is_reported(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path)
    assert not report.is_accepted
    assert "production_daily" in report.missing_tables
    assert "Missing required tables" in report.summary()


def test_rejected_table_cannot_be_read() -> None:
    frame = _minimal_daily().drop(columns=["oil_rate_m3_per_day"])
    report = validate_table("production_daily", frame)
    assert report.frame is None


# -------------------------------------------------------------------- calibration
def test_fit_quality_on_a_perfect_match() -> None:
    values = np.linspace(1.0, 10.0, 20)
    quality = FitQuality.compare(values, values)
    assert quality.root_mean_square_error == pytest.approx(0.0)
    assert quality.sample_count == 20


def test_fit_quality_ignores_missing_samples() -> None:
    measured = np.array([1.0, np.nan, 3.0])
    modelled = np.array([1.0, 5.0, 3.0])
    assert FitQuality.compare(measured, modelled).sample_count == 2


def test_fluid_anchors_recover_a_known_line() -> None:
    temps = np.array([30.0, 50.0, 80.0, 120.0, 200.0])
    truth_a, truth_b = 8.9, 3.3
    kinematic = 10.0 ** (10.0 ** (truth_a - truth_b * np.log10(temps + 273.15))) - 0.7
    viscosity_cp = kinematic * 950.0 / 1000.0
    frame = pd.DataFrame({"well_id": ["BGW-01"] * 5, "temp_c": temps, "viscosity_cp": viscosity_cp})
    anchors = fit_fluid_anchors(frame, "BGW-01")
    assert len(anchors) == 2
    predicted_50 = float(
        np.interp(
            50.0,
            [anchor["temp_c"] for anchor in anchors],
            [anchor["viscosity_cp"] for anchor in anchors],
        )
    )
    assert predicted_50 == pytest.approx(float(viscosity_cp[1]), rel=0.02)


def test_fluid_fit_needs_two_measurements() -> None:
    frame = pd.DataFrame({"well_id": ["BGW-01"], "temp_c": [50.0], "viscosity_cp": [11000.0]})
    with pytest.raises(DataValidationError, match="At least two"):
        fit_fluid_anchors(frame, "BGW-01")


def test_decline_is_read_from_the_cycle_history() -> None:
    cycles = pd.DataFrame(
        {
            "well_id": ["BGW-01"] * 5,
            "cycle_number": [1, 2, 3, 4, 5],
            "oil_m3": [600.0, 540.0, 486.0, 437.4, 393.7],
            "water_m3": [60.0, 70.0, 82.0, 95.0, 110.0],
        }
    )
    estimates = estimate_decline_from_cycles(cycles, "BGW-01")
    assert estimates["productivity_decline_per_cycle_frac"] == pytest.approx(0.10, abs=0.01)
    assert estimates["water_cut_growth_per_cycle_frac"] > 0.0


# ------------------------------------------------------------ end to end on data
@needs_data
def test_generated_dataset_passes_validation() -> None:
    report = validate_dataset(DATA_DIR)
    assert report.is_accepted, report.summary()
    assert report.data_mode == "SYNTHETIC"
    assert report.tables["production_daily"].row_count > 100


@needs_data
def test_every_generated_table_is_marked_synthetic() -> None:
    report = validate_dataset(DATA_DIR)
    for name, table in report.tables.items():
        frame = table.frame
        if frame is None or "data_mode" not in frame.columns:
            continue
        assert set(frame["data_mode"].astype(str)) == {"SYNTHETIC"}, name


@needs_data
@pytest.mark.slow
def test_calibration_improves_the_fit_and_finds_the_hidden_viscosity(
    field_config: FieldConfig,
) -> None:
    """Acceptance criterion for milestone 4.

    Calibration must reduce the error against the measured history, and the
    fluid model it fits must land close to the hidden viscosity the truth
    simulator used, which the twin never sees.
    """
    from app.core.config import load_well_config
    from app.simulate.truth.priors import read_sealed_parameters

    report = validate_dataset(DATA_DIR)
    daily = report.frame("production_daily")
    cycles = report.frame("css_cycles")
    fluid = report.frame("fluid")
    sealed = {item.well_id: item for item in read_sealed_parameters(DATA_DIR)}
    well_id = sorted(sealed)[0]

    result = calibrate_well(
        load_well_config(well_id),
        well_id,
        daily,
        cycles,
        fluid,
        training_cycles=[1, 2, 3],
        n_jobs=1,
        ensemble_size=12,
        filter_passes=2,
    )
    assert result.after is not None and result.before is not None
    assert result.after.root_mean_square_error <= result.before.root_mean_square_error
    fitted_50 = result.fluid_anchors[0]["viscosity_cp"]
    assert fitted_50 == pytest.approx(sealed[well_id].viscosity_50c_cp, rel=0.05)
