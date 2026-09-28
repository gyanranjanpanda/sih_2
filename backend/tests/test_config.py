"""Configuration loading, validation and provenance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from app.core.config import (
    FieldConfig,
    config_dir,
    load_field_config,
    load_well_config,
    read_yaml,
)
from app.core.errors import ConfigError


def _numeric_leaves(payload: Any, prefix: str = "") -> list[str]:
    """Dotted keys of every leaf in a configuration section."""
    keys: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                keys.extend(_numeric_leaves(value, path))
            else:
                keys.append(path)
    return keys


def test_field_config_loads_and_validates(field_config: FieldConfig) -> None:
    assert field_config.field.name == "Baghewala"
    assert field_config.field.operator == "Oil India Limited"
    assert field_config.reservoir.depth_m == pytest.approx(1150.0)
    assert 17.0 <= field_config.fluid.api_gravity_deg <= 19.0
    assert 46.0 <= field_config.reservoir.initial_temp_c <= 48.0


def test_every_configured_value_has_provenance() -> None:
    """No field constant may sit in config without a recorded source."""
    payload = read_yaml(config_dir() / "field.yaml")
    provenance = payload.pop("provenance")
    documented = set(provenance)
    for section, values in payload.items():
        for key in _numeric_leaves(values, section):
            assert key in documented, f"{key} has no provenance entry in config/field.yaml"


def test_assumed_values_state_a_reason(field_config: FieldConfig) -> None:
    for key in field_config.assumed_keys():
        entry = field_config.provenance[key]
        assert len(entry.source) > 20, f"{key} is marked assumed with no usable reason"


def test_published_facts_are_not_marked_assumed(field_config: FieldConfig) -> None:
    """Values taken from Oil India or the problem statement must not be labelled assumed."""
    for key in (
        "reservoir.depth_m",
        "reservoir.initial_temp_c",
        "fluid.api_gravity_deg",
        "wellbore.insulation_type",
        "srp.unit_type",
    ):
        assert not field_config.provenance[key].assumed


def test_viscosity_anchor_at_fifty_degrees_matches_published_range(
    field_config: FieldConfig,
) -> None:
    anchor = next(a for a in field_config.fluid.viscosity_anchors if a.temp_c == 50.0)
    assert 10000.0 <= anchor.viscosity_cp <= 13000.0


def test_optimizer_config_loads(optimizer_config: Any) -> None:
    assert optimizer_config.css.population_size >= 8
    assert optimizer_config.guard.mode in {"advisory", "auto"}
    assert optimizer_config.pump.variables["spm"].low > 0.0


def test_missing_file_gives_a_readable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_field_config(tmp_path / "absent.yaml")


def test_malformed_yaml_gives_a_readable_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("fluid: [unclosed\n")
    with pytest.raises(ConfigError, match="Could not parse YAML"):
        load_field_config(bad)


def test_out_of_range_value_names_the_key(tmp_path: Path, field_config: FieldConfig) -> None:
    payload = field_config.model_dump()
    payload["reservoir"]["porosity_frac"] = 1.8
    path = tmp_path / "field.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError) as excinfo:
        load_field_config(path)
    assert "reservoir.porosity_frac" in str(excinfo.value)


def test_unknown_key_is_rejected(tmp_path: Path, field_config: FieldConfig) -> None:
    payload = field_config.model_dump()
    payload["reservoir"]["mystery_constant"] = 1.0
    path = tmp_path / "field.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError, match="mystery_constant"):
        load_field_config(path)


def test_cross_field_validation_catches_inconsistency(
    tmp_path: Path, field_config: FieldConfig
) -> None:
    payload = field_config.model_dump()
    payload["reservoir"]["residual_oil_saturation_frac"] = 0.95
    path = tmp_path / "field.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError, match="residual_oil_saturation_frac"):
        load_field_config(path)


def test_rod_string_length_must_match_pump_depth(tmp_path: Path, field_config: FieldConfig) -> None:
    payload = field_config.model_dump()
    payload["srp"]["rod_sections"][0]["length_m"] = 100.0
    path = tmp_path / "field.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError, match="rod_sections total length"):
        load_field_config(path)


def test_viscosity_anchors_must_decrease_with_temperature(
    tmp_path: Path, field_config: FieldConfig
) -> None:
    payload = field_config.model_dump()
    payload["fluid"]["viscosity_anchors"][1]["viscosity_cp"] = 20000.0
    path = tmp_path / "field.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError, match="viscosity must decrease"):
        load_field_config(path)


def test_well_overlay_merges_over_the_template(
    tmp_path: Path, field_config: FieldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "config"
    (config_root / "wells").mkdir(parents=True)
    (config_root / "field.yaml").write_text(yaml.safe_dump(field_config.model_dump()))
    (config_root / "wells" / "BGW-99.yaml").write_text(
        yaml.safe_dump({"reservoir": {"permeability_md": 2500.0}})
    )
    monkeypatch.setenv("WELL_TWIN_CONFIG_DIR", str(config_root))
    well = load_well_config("BGW-99")
    assert well.well_id == "BGW-99"
    assert well.reservoir.permeability_md == pytest.approx(2500.0)
    assert well.reservoir.depth_m == pytest.approx(field_config.reservoir.depth_m)


def test_fracture_pressure_derives_from_gradient_and_depth(field_config: FieldConfig) -> None:
    expected = (
        field_config.reservoir.fracture_pressure_gradient_kpa_per_m * field_config.reservoir.depth_m
    )
    assert field_config.reservoir.fracture_pressure_kpa == pytest.approx(expected)


def test_default_injection_pressure_is_below_fracture_pressure(
    field_config: FieldConfig,
) -> None:
    assert field_config.css.injection_pressure_kpa < field_config.reservoir.fracture_pressure_kpa


def test_config_objects_are_immutable(field_config: FieldConfig) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        field_config.reservoir.depth_m = 900.0  # type: ignore[misc]
