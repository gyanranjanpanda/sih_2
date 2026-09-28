"""Data quality validation for ingested tables.

The rule the brief sets is that the validator reports problems instead of
crashing. A real Oil India export will have missing columns, gaps, spikes,
stuck transmitters and at least one unit mix-up. All of those should produce a
clear, actionable message naming the table, the column, the rows and, where it
can be worked out, the unit the data appears to be in.

Severity has three levels. ``error`` blocks ingestion. ``warning`` lets the
data through but marks it. ``info`` is a note for the data health page.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import DataValidationError
from app.core.logging import get_logger
from app.core.units import convert
from app.ingestion.schema import (
    OPTIONAL_TABLES,
    REQUIRED_TABLES,
    TABLE_SCHEMAS,
    ColumnSpec,
    TableSchema,
)

LOGGER = get_logger(__name__)

SPIKE_SIGMA = 8.0
"""Robust z-score above which a sample is reported as a spike."""

STUCK_MINIMUM_RUN = 12
"""Consecutive identical samples that count as a stuck transmitter."""

UNIT_MATCH_TOLERANCE = 0.35
"""How close the median must be to the expected range for a unit to be blamed."""


class Severity(StrEnum):
    """How serious a validation finding is."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    """One finding, with enough detail for an engineer to act on it."""

    table: str
    severity: Severity
    code: str
    message: str
    column: str | None = None
    row_count: int = 0
    sample_rows: tuple[int, ...] = ()
    suggestion: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the API and the data health page."""
        return {
            "table": self.table,
            "severity": str(self.severity),
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "row_count": self.row_count,
            "sample_rows": list(self.sample_rows),
            "suggestion": self.suggestion,
        }


@dataclass
class TableReport:
    """Validation outcome for one table."""

    table: str
    row_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    frame: pd.DataFrame | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues that block ingestion."""
        return [issue for issue in self.issues if issue.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Issues that do not block ingestion."""
        return [issue for issue in self.issues if issue.severity == Severity.WARNING]

    @property
    def is_accepted(self) -> bool:
        """Whether the table can be used."""
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "table": self.table,
            "row_count": self.row_count,
            "accepted": self.is_accepted,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass
class IngestionReport:
    """Validation outcome for a whole dataset."""

    tables: dict[str, TableReport] = field(default_factory=dict)
    missing_tables: list[str] = field(default_factory=list)
    data_mode: str = "UNKNOWN"

    @property
    def is_accepted(self) -> bool:
        """Whether every required table passed."""
        if self.missing_tables:
            return False
        return all(report.is_accepted for report in self.tables.values())

    def all_issues(self) -> list[ValidationIssue]:
        """Every issue across every table, errors first."""
        issues = [issue for report in self.tables.values() for issue in report.issues]
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        return sorted(issues, key=lambda issue: order[issue.severity])

    def summary(self) -> str:
        """A short human readable summary."""
        errors = sum(len(report.errors) for report in self.tables.values())
        warnings = sum(len(report.warnings) for report in self.tables.values())
        rows = sum(report.row_count for report in self.tables.values())
        verdict = "accepted" if self.is_accepted else "rejected"
        missing = (
            f" Missing required tables: {', '.join(self.missing_tables)}."
            if self.missing_tables
            else ""
        )
        return (
            f"Ingestion {verdict}: {len(self.tables)} tables, {rows} rows, "
            f"{errors} errors, {warnings} warnings.{missing}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the API."""
        return {
            "accepted": self.is_accepted,
            "data_mode": self.data_mode,
            "summary": self.summary(),
            "missing_tables": self.missing_tables,
            "tables": {name: report.as_dict() for name, report in self.tables.items()},
        }

    def frame(self, table: str) -> pd.DataFrame:
        """The validated frame for one table.

        Raises:
            DataValidationError: if the table was not ingested or was rejected.
        """
        report = self.tables.get(table)
        if report is None or report.frame is None:
            raise DataValidationError(f"Table '{table}' was not ingested.", table=table)
        if not report.is_accepted:
            raise DataValidationError(
                f"Table '{table}' was rejected: "
                + "; ".join(issue.message for issue in report.errors),
                table=table,
            )
        return report.frame


def _robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Median-absolute-deviation z-scores, which outliers cannot inflate."""
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return np.zeros_like(values)
    median = float(np.median(finite))
    deviation = float(np.median(np.abs(finite - median)))
    if deviation <= 0.0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / deviation


def _suspected_units(column: ColumnSpec, values: np.ndarray) -> list[str]:
    """Units a column might actually be in, when its range looks wrong.

    Every unit the column is commonly confused with is tried, and the ones that
    bring the median inside the expected range are returned, closest to the
    middle of that range first. Several candidates are reported rather than one
    guessed, because a pressure of 112 is equally plausible as psi or as kg/cm2
    and picking one silently would send an engineer the wrong way.
    """
    if not column.common_wrong_units or column.unit is None:
        return []
    if column.minimum is None or column.maximum is None:
        return []
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    median = float(np.median(finite))
    expected_mid = 0.5 * (column.minimum + column.maximum)
    scored: list[tuple[float, str]] = []
    for candidate in column.common_wrong_units:
        try:
            converted = float(convert(median, candidate, column.unit))
        except Exception:
            continue
        if column.minimum <= converted <= column.maximum:
            scored.append((abs(converted - expected_mid) / max(abs(expected_mid), 1.0), candidate))
    scored.sort()
    return [candidate for _, candidate in scored]


def _check_columns(schema: TableSchema, frame: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in schema.required_columns:
        if name not in frame.columns:
            issues.append(
                ValidationIssue(
                    table=schema.name,
                    severity=Severity.ERROR,
                    code="missing_column",
                    message=f"Required column '{name}' is missing from table '{schema.name}'.",
                    column=name,
                    suggestion=(
                        f"Add a '{name}' column. See docs/DATA_SCHEMA.md for its "
                        "meaning and unit."
                    ),
                )
            )
    known = {column.name for column in schema.columns}
    extra = [name for name in frame.columns if name not in known]
    if extra:
        issues.append(
            ValidationIssue(
                table=schema.name,
                severity=Severity.INFO,
                code="extra_columns",
                message=(
                    f"Table '{schema.name}' has {len(extra)} column(s) the schema does not "
                    f"describe: {', '.join(sorted(extra)[:8])}. They are carried through "
                    "unchanged."
                ),
                row_count=len(extra),
            )
        )
    return issues


def _check_values(schema: TableSchema, frame: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for column in schema.numeric_columns():
        if column.name not in frame.columns:
            continue
        series = pd.to_numeric(frame[column.name], errors="coerce")
        values = series.to_numpy(dtype=float)

        non_numeric = series.isna() & frame[column.name].notna()
        if bool(non_numeric.any()):
            rows = np.nonzero(non_numeric.to_numpy())[0]
            issues.append(
                ValidationIssue(
                    table=schema.name,
                    severity=Severity.ERROR,
                    code="non_numeric",
                    message=(
                        f"Column '{column.name}' has {rows.size} value(s) that are not " "numbers."
                    ),
                    column=column.name,
                    row_count=int(rows.size),
                    sample_rows=tuple(int(row) for row in rows[:5]),
                    suggestion="Remove the text, or leave the cell empty for a missing value.",
                )
            )

        missing = int(np.count_nonzero(np.isnan(values)))
        if missing:
            fraction = missing / max(values.size, 1)
            issues.append(
                ValidationIssue(
                    table=schema.name,
                    severity=Severity.WARNING if fraction < 0.3 else Severity.ERROR,
                    code="missing_values",
                    message=(
                        f"Column '{column.name}' is missing {missing} of {values.size} "
                        f"values ({fraction * 100:.1f} percent)."
                    ),
                    column=column.name,
                    row_count=missing,
                    sample_rows=tuple(int(row) for row in np.nonzero(np.isnan(values))[0][:5]),
                    suggestion=(
                        "Gaps are interpolated where short. A column missing more than "
                        "30 percent of its values is not usable."
                    ),
                )
            )

        if column.minimum is None and column.maximum is None:
            continue
        low = column.minimum if column.minimum is not None else -np.inf
        high = column.maximum if column.maximum is not None else np.inf
        out_of_range = np.isfinite(values) & ((values < low) | (values > high))
        count = int(np.count_nonzero(out_of_range))
        if count:
            suspected = _suspected_units(column, values[out_of_range])
            fraction = count / max(values.size, 1)
            severity = Severity.ERROR if fraction > 0.5 else Severity.WARNING
            candidates = ", ".join(f"'{unit}'" for unit in suspected)
            suggestion = (
                f"The values are plausible in {candidates}. Convert them to "
                f"'{column.unit}' before loading, or set the unit on import."
                if suspected
                else f"Expected values between {low} and {high}."
            )
            issues.append(
                ValidationIssue(
                    table=schema.name,
                    severity=severity,
                    code="out_of_range" if not suspected else "suspected_unit_error",
                    message=(
                        f"Column '{column.name}' has {count} value(s) outside the plausible "
                        f"range [{low}, {high}]."
                        + (f" They match the range for {candidates}." if suspected else "")
                    ),
                    column=column.name,
                    row_count=count,
                    sample_rows=tuple(int(row) for row in np.nonzero(out_of_range)[0][:5]),
                    suggestion=suggestion,
                )
            )

        z_scores = _robust_z_scores(values)
        spikes = np.isfinite(z_scores) & (np.abs(z_scores) > SPIKE_SIGMA)
        spike_count = int(np.count_nonzero(spikes))
        if spike_count:
            issues.append(
                ValidationIssue(
                    table=schema.name,
                    severity=Severity.WARNING,
                    code="spikes",
                    message=(
                        f"Column '{column.name}' has {spike_count} sample(s) more than "
                        f"{SPIKE_SIGMA} robust standard deviations from the median."
                    ),
                    column=column.name,
                    row_count=spike_count,
                    sample_rows=tuple(int(row) for row in np.nonzero(spikes)[0][:5]),
                    suggestion="Check the transmitter. Spikes are excluded from calibration.",
                )
            )

        stuck_runs = _stuck_runs(values, STUCK_MINIMUM_RUN) if column.check_stuck else []
        if stuck_runs:
            total = sum(length for _, length in stuck_runs)
            issues.append(
                ValidationIssue(
                    table=schema.name,
                    severity=Severity.WARNING,
                    code="stuck_sensor",
                    message=(
                        f"Column '{column.name}' holds the same value for "
                        f"{len(stuck_runs)} run(s), {total} samples in total. The "
                        "transmitter may be stuck."
                    ),
                    column=column.name,
                    row_count=total,
                    sample_rows=tuple(int(start) for start, _ in stuck_runs[:5]),
                    suggestion="Stuck runs are excluded from calibration and from training.",
                )
            )
    return issues


def _stuck_runs(values: np.ndarray, minimum_run: int) -> list[tuple[int, int]]:
    """Start index and length of every run of identical finite values."""
    runs: list[tuple[int, int]] = []
    if values.size < minimum_run:
        return runs
    start = 0
    for index in range(1, values.size + 1):
        if (
            index < values.size
            and np.isfinite(values[index])
            and np.isfinite(values[start])
            and values[index] == values[start]
        ):
            continue
        length = index - start
        if length >= minimum_run and np.isfinite(values[start]):
            runs.append((start, length))
        start = index
    return runs


def _check_keys_and_gaps(schema: TableSchema, frame: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    present_keys = [name for name in schema.key_columns if name in frame.columns]
    if len(present_keys) == len(schema.key_columns) and present_keys:
        duplicated = frame.duplicated(subset=present_keys, keep=False)
        count = int(duplicated.sum())
        if count:
            issues.append(
                ValidationIssue(
                    table=schema.name,
                    severity=Severity.WARNING,
                    code="duplicate_keys",
                    message=(
                        f"Table '{schema.name}' has {count} row(s) that repeat the key "
                        f"({', '.join(present_keys)})."
                    ),
                    row_count=count,
                    sample_rows=tuple(int(row) for row in np.nonzero(duplicated.to_numpy())[0][:5]),
                    suggestion="Duplicates are kept as they are. De-duplicate before loading.",
                )
            )

    if schema.time_column and schema.time_column in frame.columns and schema.max_gap_days:
        groups: Iterable[tuple[Any, pd.DataFrame]]
        if schema.group_column and schema.group_column in frame.columns:
            groups = frame.groupby(schema.group_column)
        else:
            groups = [("all", frame)]
        for key, group in groups:
            times = pd.to_numeric(group[schema.time_column], errors="coerce").dropna()
            if times.size < 3:
                continue
            gaps = np.diff(np.sort(times.to_numpy()))
            large = gaps[gaps > schema.max_gap_days]
            if large.size:
                issues.append(
                    ValidationIssue(
                        table=schema.name,
                        severity=Severity.WARNING,
                        code="time_gaps",
                        message=(
                            f"{key}: {large.size} gap(s) in '{schema.time_column}' longer "
                            f"than {schema.max_gap_days} days, the largest "
                            f"{float(large.max()):.1f} days."
                        ),
                        column=schema.time_column,
                        row_count=int(large.size),
                        suggestion=(
                            "Short gaps are interpolated. Long gaps split the series so the "
                            "twin is not asked to bridge a shut-in it knows nothing about."
                        ),
                    )
                )
    return issues


def validate_table(name: str, frame: pd.DataFrame) -> TableReport:
    """Validate one table against its schema.

    Args:
        name: Table name, which must be one of :data:`TABLE_SCHEMAS`.
        frame: The loaded data.

    Returns:
        The report, with the frame attached when the table is usable.
    """
    schema = TABLE_SCHEMAS.get(name)
    if schema is None:
        raise DataValidationError(
            f"Unknown table '{name}'. Known tables: {', '.join(sorted(TABLE_SCHEMAS))}.",
            table=name,
        )
    issues = _check_columns(schema, frame)
    if any(issue.severity == Severity.ERROR for issue in issues):
        return TableReport(table=name, row_count=len(frame), issues=issues)
    issues.extend(_check_values(schema, frame))
    issues.extend(_check_keys_and_gaps(schema, frame))
    report = TableReport(table=name, row_count=len(frame), issues=issues)
    if report.is_accepted:
        report.frame = frame
    return report


def load_table(path: Path, name: str | None = None) -> TableReport:
    """Read a CSV and validate it.

    Args:
        path: Path to the CSV file.
        name: Table name. Defaults to the file stem.
    """
    table_name = name or path.stem
    if not path.exists():
        raise DataValidationError(f"File not found: {path}", path=str(path))
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(
            f"Could not read {path.name} as CSV: {exc}", path=str(path)
        ) from exc
    if frame.empty:
        return TableReport(
            table=table_name,
            row_count=0,
            issues=[
                ValidationIssue(
                    table=table_name,
                    severity=Severity.ERROR,
                    code="empty_table",
                    message=f"File {path.name} has no rows.",
                    suggestion="Export the table again.",
                )
            ],
        )
    return validate_table(table_name, frame)


KNOWN_VOLUME_UNIT_RATIOS: dict[str, float] = {
    "barrels reported as cubic metres": 1.0 / 0.158987294928,
    "cubic metres reported as barrels": 0.158987294928,
    "litres reported as cubic metres": 1000.0,
    "tonnes of oil reported as cubic metres": 1.0 / 0.9,
}
"""Ratios that name a specific unit mistake when two tables disagree."""


def cross_check_cycle_and_daily(cycles: pd.DataFrame, daily: pd.DataFrame) -> list[ValidationIssue]:
    """Check that daily rates integrate to the reported cycle volumes.

    A range check cannot catch every unit error. An oil rate of 19 is plausible
    in cubic metres per day and equally plausible in barrels per day, so nothing
    in the column itself gives it away. What does give it away is that the same
    quantity is reported twice, once as a daily rate and once as a cycle total,
    and the two no longer agree. When the ratio lands on a known unit mistake
    the message names it.
    """
    issues: list[ValidationIssue] = []
    required = {"well_id", "cycle_number", "oil_m3"}
    if not required.issubset(cycles.columns):
        return issues
    if not {"well_id", "cycle_number", "oil_rate_m3_per_day"}.issubset(daily.columns):
        return issues

    integrated = (
        daily.groupby(["well_id", "cycle_number"])["oil_rate_m3_per_day"].sum().rename("summed")
    )
    reported = cycles.set_index(["well_id", "cycle_number"])["oil_m3"].rename("reported")
    joined = pd.concat([integrated, reported], axis=1).dropna()
    joined = joined[(joined["reported"] > 1.0) & (joined["summed"] > 1.0)]
    if joined.empty:
        return issues

    ratios = (joined["summed"] / joined["reported"]).to_numpy(dtype=float)
    median_ratio = float(np.median(ratios))
    if 0.85 <= median_ratio <= 1.18:
        return issues

    named = None
    for label, expected in KNOWN_VOLUME_UNIT_RATIOS.items():
        if abs(median_ratio - expected) / expected < 0.08:
            named = label
            break
        if abs(median_ratio - 1.0 / expected) / (1.0 / expected) < 0.08:
            named = label + ", the other way round"
            break

    issues.append(
        ValidationIssue(
            table="production_daily",
            severity=Severity.ERROR if named else Severity.WARNING,
            code="cross_table_mismatch",
            column="oil_rate_m3_per_day",
            message=(
                "Daily oil rates integrate to "
                f"{median_ratio:.2f} times the cycle oil reported in css_cycles."
                + (f" That ratio is {named}." if named else "")
            ),
            row_count=int(joined.shape[0]),
            suggestion=(
                "Convert the daily rates to cubic metres per day before loading."
                if named
                else (
                    "Check whether the daily record covers the whole cycle and whether "
                    "both tables use the same units."
                )
            ),
        )
    )
    return issues


def cross_check_failures_against_wells(
    failures: pd.DataFrame, wells: pd.DataFrame
) -> list[ValidationIssue]:
    """Check that every failure refers to a well that exists."""
    if "well_id" not in failures.columns or "well_id" not in wells.columns:
        return []
    known = set(wells["well_id"].astype(str))
    unknown = sorted(set(failures["well_id"].astype(str)) - known)
    if not unknown:
        return []
    return [
        ValidationIssue(
            table="failures",
            severity=Severity.WARNING,
            code="unknown_well",
            column="well_id",
            message=(
                f"{len(unknown)} failure record(s) refer to wells that are not in the wells "
                f"table: {', '.join(unknown[:6])}."
            ),
            row_count=len(unknown),
            suggestion="Add the wells, or correct the identifiers.",
        )
    ]


def validate_dataset(directory: Path) -> IngestionReport:
    """Validate every table found in a directory.

    Required tables that are absent are reported as missing rather than raising,
    so the data health page can show the operator exactly what to supply next.
    """
    report = IngestionReport()
    for name in REQUIRED_TABLES:
        path = directory / f"{name}.csv"
        if not path.exists():
            report.missing_tables.append(name)
            continue
        report.tables[name] = load_table(path, name)
    for name in OPTIONAL_TABLES:
        path = directory / f"{name}.csv"
        if path.exists():
            report.tables[name] = load_table(path, name)
    cycles_report = report.tables.get("css_cycles")
    daily_report = report.tables.get("production_daily")
    if (
        cycles_report is not None
        and daily_report is not None
        and cycles_report.frame is not None
        and daily_report.frame is not None
    ):
        extra = cross_check_cycle_and_daily(cycles_report.frame, daily_report.frame)
        daily_report.issues.extend(extra)
        if daily_report.errors:
            daily_report.frame = None

    wells_report = report.tables.get("wells")
    failures_report = report.tables.get("failures")
    if (
        wells_report is not None
        and failures_report is not None
        and wells_report.frame is not None
        and failures_report.frame is not None
    ):
        failures_report.issues.extend(
            cross_check_failures_against_wells(failures_report.frame, wells_report.frame)
        )

    wells = report.tables.get("wells")
    if wells is not None and wells.frame is not None and "data_mode" in wells.frame.columns:
        modes = sorted(set(wells.frame["data_mode"].astype(str)))
        report.data_mode = modes[0] if len(modes) == 1 else "MIXED"
    LOGGER.info("Ingestion complete", extra={"summary": report.summary()})
    return report
