"""Table schemas for ingested data.

These definitions are the single source of truth behind ``docs/DATA_SCHEMA.md``
and behind the validator. The same schema accepts the generated synthetic
tables and a real Oil India CSV export, which is what makes the switch from
SYNTHETIC to REAL mode a configuration change rather than a code change.

Every numeric column carries its unit in its name and a physically plausible
range. The range is a quality gate, not a physical law: a value outside it is
reported as a problem for a human to look at, and in the case of a suspected
unit mix-up the validator says which unit it thinks was used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    """One column of an ingested table.

    Attributes:
        name: Column name as it must appear in the CSV.
        dtype: One of ``float``, ``int``, ``str``.
        required: Whether ingestion fails without it.
        minimum: Lowest plausible value.
        maximum: Highest plausible value.
        unit: Unit key from :mod:`app.core.units`, or None for dimensionless
            and text columns.
        description: What the column means, reproduced in the data schema doc.
        common_wrong_units: Units this column is often reported in by mistake,
            used to produce a helpful message rather than a bare range error.
        check_stuck: Whether a run of identical values means a stuck
            transmitter. Only measured channels qualify. A design value such as
            the injection pressure of a cycle, or a setpoint such as strokes per
            minute, is supposed to stay constant, and flagging those would bury
            the real findings.
    """

    name: str
    dtype: str
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    description: str = ""
    common_wrong_units: tuple[str, ...] = ()
    check_stuck: bool = False


@dataclass(frozen=True)
class TableSchema:
    """One ingested table."""

    name: str
    columns: tuple[ColumnSpec, ...]
    key_columns: tuple[str, ...]
    description: str = ""
    time_column: str | None = None
    max_gap_days: float | None = None
    group_column: str | None = "well_id"

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Names of the columns that must be present."""
        return tuple(column.name for column in self.columns if column.required)

    def column(self, name: str) -> ColumnSpec | None:
        """Look up one column specification."""
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def numeric_columns(self) -> tuple[ColumnSpec, ...]:
        """Columns that carry numbers."""
        return tuple(column for column in self.columns if column.dtype in {"float", "int"})


def _well_id() -> ColumnSpec:
    return ColumnSpec(
        name="well_id", dtype="str", description="Well identifier, for example BGW-07."
    )


WELLS_SCHEMA = TableSchema(
    name="wells",
    description="One row per well: completion, equipment and provenance.",
    key_columns=("well_id",),
    group_column=None,
    columns=(
        _well_id(),
        ColumnSpec(
            "unit_type", "str", description="conventional or hydraulic sucker rod pump unit."
        ),
        ColumnSpec("insulation_type", "str", required=False, description="vit or bare tubing."),
        ColumnSpec(
            "pump_depth_m",
            "float",
            minimum=100.0,
            maximum=4000.0,
            unit="m",
            description="Depth of the pump intake.",
            common_wrong_units=("ft",),
        ),
        ColumnSpec(
            "plunger_diameter_m",
            "float",
            minimum=0.01,
            maximum=0.2,
            unit="m",
            description="Plunger diameter.",
            common_wrong_units=("in",),
        ),
        ColumnSpec(
            "stroke_length_m",
            "float",
            minimum=0.3,
            maximum=12.0,
            unit="m",
            description="Surface stroke length.",
            common_wrong_units=("in",),
        ),
        ColumnSpec(
            "baseline_spm",
            "float",
            required=False,
            minimum=0.2,
            maximum=25.0,
            description="Historical pumping speed in strokes per minute.",
        ),
        ColumnSpec(
            "api_gravity_deg",
            "float",
            required=False,
            minimum=5.0,
            maximum=50.0,
            description="Crude API gravity.",
        ),
        ColumnSpec(
            "reservoir_depth_m",
            "float",
            required=False,
            minimum=100.0,
            maximum=5000.0,
            unit="m",
            description="Average reservoir depth.",
        ),
        ColumnSpec(
            "data_mode",
            "str",
            required=False,
            description="SYNTHETIC or REAL. Drives the badge on every chart.",
        ),
    ),
)

FLUID_SCHEMA = TableSchema(
    name="fluid",
    description="Laboratory viscosity against temperature, per well.",
    key_columns=("well_id", "temp_c"),
    columns=(
        _well_id(),
        ColumnSpec(
            "temp_c",
            "float",
            minimum=-20.0,
            maximum=400.0,
            unit="c",
            description="Measurement temperature.",
            common_wrong_units=("f",),
        ),
        ColumnSpec(
            "viscosity_cp",
            "float",
            minimum=0.1,
            maximum=5.0e7,
            unit="cp",
            description="Dynamic viscosity of the dead crude.",
            common_wrong_units=("pa_s", "cst"),
        ),
        ColumnSpec(
            "api_gravity_deg",
            "float",
            required=False,
            minimum=5.0,
            maximum=50.0,
            description="Crude API gravity.",
        ),
        ColumnSpec("data_mode", "str", required=False, description="SYNTHETIC or REAL."),
    ),
)

CSS_CYCLES_SCHEMA = TableSchema(
    name="css_cycles",
    description="One row per cyclic steam stimulation cycle.",
    key_columns=("well_id", "cycle_number"),
    columns=(
        _well_id(),
        ColumnSpec("cycle_number", "int", minimum=1, maximum=60, description="Cycle number."),
        ColumnSpec(
            "steam_volume_m3_cwe",
            "float",
            minimum=50.0,
            maximum=20000.0,
            unit="m3",
            description="Steam injected, cold water equivalent.",
            common_wrong_units=("bbl",),
        ),
        ColumnSpec(
            "injection_rate_m3_per_day_cwe",
            "float",
            minimum=5.0,
            maximum=2000.0,
            unit="m3_per_day",
            description="Average injection rate, cold water equivalent.",
            common_wrong_units=("bbl_per_day",),
        ),
        ColumnSpec(
            "injection_pressure_kpa",
            "float",
            minimum=500.0,
            maximum=40000.0,
            unit="kpa",
            description="Wellhead injection pressure.",
            common_wrong_units=("psi", "kgf_per_cm2", "bar"),
        ),
        ColumnSpec(
            "steam_quality_frac",
            "float",
            minimum=0.0,
            maximum=1.0,
            description="Steam quality at the wellhead.",
            common_wrong_units=("percent",),
        ),
        ColumnSpec(
            "soak_days",
            "float",
            minimum=0.0,
            maximum=120.0,
            description="Shut-in soak duration.",
        ),
        ColumnSpec(
            "production_days",
            "float",
            minimum=0.0,
            maximum=1000.0,
            description="Length of the production phase.",
        ),
        ColumnSpec(
            "oil_m3",
            "float",
            minimum=0.0,
            maximum=100000.0,
            unit="m3",
            description="Cycle oil production.",
            common_wrong_units=("bbl",),
        ),
        ColumnSpec(
            "water_m3",
            "float",
            required=False,
            minimum=0.0,
            maximum=200000.0,
            unit="m3",
            description="Cycle water production.",
            common_wrong_units=("bbl",),
        ),
        ColumnSpec(
            "cutoff_reason",
            "str",
            required=False,
            description="Why the cycle was ended.",
        ),
        ColumnSpec("data_mode", "str", required=False, description="SYNTHETIC or REAL."),
    ),
)

PRODUCTION_DAILY_SCHEMA = TableSchema(
    name="production_daily",
    description="Daily production and well state.",
    key_columns=("well_id", "day"),
    time_column="day",
    max_gap_days=3.0,
    columns=(
        _well_id(),
        ColumnSpec(
            "day",
            "float",
            minimum=0.0,
            maximum=20000.0,
            description="Days since the start of the record.",
        ),
        ColumnSpec("cycle_number", "int", minimum=1, maximum=60, description="Cycle number."),
        ColumnSpec(
            "production_day",
            "float",
            required=False,
            minimum=0.0,
            maximum=2000.0,
            description="Days since this cycle was put on production.",
        ),
        ColumnSpec(
            "oil_rate_m3_per_day",
            "float",
            minimum=0.0,
            maximum=500.0,
            unit="m3_per_day",
            description="Daily oil rate.",
            common_wrong_units=("bbl_per_day",),
            check_stuck=True,
        ),
        ColumnSpec(
            "water_rate_m3_per_day",
            "float",
            required=False,
            minimum=0.0,
            maximum=1000.0,
            unit="m3_per_day",
            description="Daily water rate.",
            common_wrong_units=("bbl_per_day",),
            check_stuck=True,
        ),
        ColumnSpec(
            "water_cut_frac",
            "float",
            required=False,
            minimum=0.0,
            maximum=1.0,
            description="Water cut as a fraction.",
            common_wrong_units=("percent",),
        ),
        ColumnSpec(
            "near_well_temp_c",
            "float",
            required=False,
            minimum=-20.0,
            maximum=400.0,
            unit="c",
            description="Near-wellbore reservoir temperature.",
            common_wrong_units=("f", "k"),
            check_stuck=True,
        ),
        ColumnSpec(
            "reservoir_pressure_kpa",
            "float",
            required=False,
            minimum=100.0,
            maximum=40000.0,
            unit="kpa",
            description="Average reservoir pressure.",
            common_wrong_units=("psi", "kgf_per_cm2"),
            check_stuck=True,
        ),
        ColumnSpec(
            "bottomhole_pressure_kpa",
            "float",
            required=False,
            minimum=0.0,
            maximum=40000.0,
            unit="kpa",
            description="Flowing bottomhole pressure.",
            common_wrong_units=("psi", "kgf_per_cm2"),
            check_stuck=True,
        ),
        ColumnSpec(
            "wellhead_temp_c",
            "float",
            required=False,
            minimum=-20.0,
            maximum=400.0,
            unit="c",
            description="Flowline temperature at the wellhead.",
            common_wrong_units=("f",),
            check_stuck=True,
        ),
        ColumnSpec(
            "spm",
            "float",
            required=False,
            minimum=0.0,
            maximum=25.0,
            description="Pumping speed in strokes per minute.",
        ),
        ColumnSpec(
            "stroke_length_m",
            "float",
            required=False,
            minimum=0.3,
            maximum=12.0,
            unit="m",
            description="Surface stroke length.",
            common_wrong_units=("in",),
        ),
        ColumnSpec(
            "peak_polished_rod_load_n",
            "float",
            required=False,
            minimum=0.0,
            maximum=1.0e6,
            unit="n",
            description="Peak polished rod load.",
            common_wrong_units=("lbf", "kn"),
            check_stuck=True,
        ),
        ColumnSpec(
            "minimum_polished_rod_load_n",
            "float",
            required=False,
            minimum=-1.0e6,
            maximum=1.0e6,
            unit="n",
            description="Minimum polished rod load.",
            common_wrong_units=("lbf", "kn"),
            check_stuck=True,
        ),
        ColumnSpec(
            "fillage_frac",
            "float",
            required=False,
            minimum=0.0,
            maximum=1.0,
            description="Pump fillage.",
            common_wrong_units=("percent",),
        ),
        ColumnSpec(
            "motor_power_w",
            "float",
            required=False,
            minimum=0.0,
            maximum=5.0e5,
            unit="w",
            description="Motor electrical power.",
            common_wrong_units=("kw", "hp"),
            check_stuck=True,
        ),
        ColumnSpec(
            "card_class",
            "str",
            required=False,
            description="Diagnosed or labelled card class.",
        ),
        ColumnSpec("data_mode", "str", required=False, description="SYNTHETIC or REAL."),
    ),
)

SRP_TELEMETRY_SCHEMA = TableSchema(
    name="srp_telemetry",
    description="High-rate pumping unit telemetry.",
    key_columns=("well_id", "timestamp_day"),
    time_column="timestamp_day",
    max_gap_days=1.0,
    columns=(
        _well_id(),
        ColumnSpec(
            "timestamp_day",
            "float",
            minimum=0.0,
            maximum=20000.0,
            description="Days since the start of the record.",
        ),
        ColumnSpec(
            "cycle_number",
            "int",
            required=False,
            minimum=1,
            maximum=60,
            description="Cycle number.",
        ),
        ColumnSpec(
            "spm",
            "float",
            minimum=0.0,
            maximum=25.0,
            description="Pumping speed in strokes per minute.",
        ),
        ColumnSpec(
            "stroke_length_m",
            "float",
            required=False,
            minimum=0.3,
            maximum=12.0,
            unit="m",
            description="Surface stroke length.",
            common_wrong_units=("in",),
        ),
        ColumnSpec(
            "vfd_frequency_hz",
            "float",
            required=False,
            minimum=0.0,
            maximum=120.0,
            description="Variable speed drive output frequency.",
        ),
        ColumnSpec(
            "motor_power_w",
            "float",
            required=False,
            minimum=0.0,
            maximum=5.0e5,
            unit="w",
            description="Motor electrical power.",
            common_wrong_units=("kw", "hp"),
            check_stuck=True,
        ),
        ColumnSpec(
            "peak_polished_rod_load_n",
            "float",
            required=False,
            minimum=0.0,
            maximum=1.0e6,
            unit="n",
            description="Peak polished rod load.",
            common_wrong_units=("lbf", "kn"),
            check_stuck=True,
        ),
        ColumnSpec(
            "minimum_polished_rod_load_n",
            "float",
            required=False,
            minimum=-1.0e6,
            maximum=1.0e6,
            unit="n",
            description="Minimum polished rod load.",
            common_wrong_units=("lbf", "kn"),
            check_stuck=True,
        ),
        ColumnSpec(
            "fillage_frac",
            "float",
            required=False,
            minimum=0.0,
            maximum=1.0,
            description="Pump fillage.",
            common_wrong_units=("percent",),
        ),
        ColumnSpec(
            "pump_intake_temp_c",
            "float",
            required=False,
            minimum=-20.0,
            maximum=400.0,
            unit="c",
            description="Temperature at the pump intake.",
            common_wrong_units=("f",),
            check_stuck=True,
        ),
        ColumnSpec(
            "wellhead_temp_c",
            "float",
            required=False,
            minimum=-20.0,
            maximum=400.0,
            unit="c",
            description="Flowline temperature.",
            common_wrong_units=("f",),
            check_stuck=True,
        ),
        ColumnSpec(
            "casing_pressure_kpa",
            "float",
            required=False,
            minimum=0.0,
            maximum=40000.0,
            unit="kpa",
            description="Casing head pressure.",
            common_wrong_units=("psi", "kgf_per_cm2"),
            check_stuck=True,
        ),
        ColumnSpec("data_mode", "str", required=False, description="SYNTHETIC or REAL."),
    ),
)

FAILURES_SCHEMA = TableSchema(
    name="failures",
    description="Rod partings, pump unseatings and other interventions.",
    key_columns=("well_id", "day", "failure_type"),
    columns=(
        _well_id(),
        ColumnSpec("day", "float", minimum=0.0, maximum=20000.0, description="Day of the event."),
        ColumnSpec(
            "cycle_number",
            "int",
            required=False,
            minimum=1,
            maximum=60,
            description="Cycle the event happened in.",
        ),
        ColumnSpec(
            "failure_type",
            "str",
            description="rod_parting, pump_unseating, tubing_leak and so on.",
        ),
        ColumnSpec(
            "depth_m",
            "float",
            required=False,
            minimum=0.0,
            maximum=5000.0,
            unit="m",
            description="Depth of the failure.",
            common_wrong_units=("ft",),
        ),
        ColumnSpec("data_mode", "str", required=False, description="SYNTHETIC or REAL."),
    ),
)

STEAM_GENERATOR_LOG_SCHEMA = TableSchema(
    name="steam_generator_log",
    description="Which mobile steam generator served which injection.",
    key_columns=("unit_id", "well_id", "cycle_number"),
    group_column="unit_id",
    columns=(
        ColumnSpec("unit_id", "str", description="Mobile steam generator identifier."),
        _well_id(),
        ColumnSpec("cycle_number", "int", minimum=1, maximum=60, description="Cycle number."),
        ColumnSpec(
            "start_day",
            "float",
            minimum=0.0,
            maximum=20000.0,
            description="Day the injection started.",
        ),
        ColumnSpec(
            "end_day",
            "float",
            minimum=0.0,
            maximum=20000.0,
            description="Day the injection ended.",
        ),
        ColumnSpec(
            "rig_move_days",
            "float",
            required=False,
            minimum=0.0,
            maximum=30.0,
            description="Days to move the unit to the next well.",
        ),
        ColumnSpec(
            "steam_volume_m3_cwe",
            "float",
            required=False,
            minimum=0.0,
            maximum=20000.0,
            unit="m3",
            description="Steam delivered, cold water equivalent.",
        ),
        ColumnSpec("data_mode", "str", required=False, description="SYNTHETIC or REAL."),
    ),
)

TABLE_SCHEMAS: dict[str, TableSchema] = {
    schema.name: schema
    for schema in (
        WELLS_SCHEMA,
        FLUID_SCHEMA,
        CSS_CYCLES_SCHEMA,
        PRODUCTION_DAILY_SCHEMA,
        SRP_TELEMETRY_SCHEMA,
        FAILURES_SCHEMA,
        STEAM_GENERATOR_LOG_SCHEMA,
    )
}
"""Every table the system knows how to ingest, keyed by name."""

REQUIRED_TABLES: tuple[str, ...] = ("wells", "css_cycles", "production_daily")
"""Tables without which nothing useful can be done."""

OPTIONAL_TABLES: tuple[str, ...] = (
    "fluid",
    "srp_telemetry",
    "failures",
    "steam_generator_log",
)
"""Tables that unlock extra capability but are not needed to run."""
