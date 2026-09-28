"""Hidden per-well parameters drawn from the priors in ``config/truth_priors.yaml``.

These are the numbers the twin is not allowed to know. They are written to a
sealed file under ``data/synthetic/`` so a run can be audited afterwards, and a
test asserts that no module under ``app.twin`` imports anything from this
package.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.config import config_dir, read_yaml
from app.core.errors import ConfigError

SEALED_FILENAME = "sealed_truth_parameters.json"
"""Name of the sealed parameter file. Reading it from twin code is a test failure."""


def _draw(rng: np.random.Generator, spec: dict[str, Any]) -> float:
    """Draw one value from a prior specification."""
    distribution = spec.get("distribution")
    if distribution == "normal":
        value = rng.normal(spec["mean"], spec["sigma"])
    elif distribution == "lognormal":
        value = float(spec["median"]) * float(np.exp(rng.normal(0.0, spec["log_sigma"])))
    elif distribution == "uniform":
        value = rng.uniform(spec["low"], spec["high"])
    elif distribution == "bernoulli":
        value = float(rng.random() < float(spec["probability"]))
    else:
        raise ConfigError(
            f"Unknown prior distribution '{distribution}'.", distribution=distribution
        )
    low = spec.get("low") if distribution != "uniform" else None
    high = spec.get("high") if distribution != "uniform" else None
    if low is not None:
        value = max(value, float(low))
    if high is not None:
        value = min(value, float(high))
    return float(value)


@dataclass(frozen=True)
class LayerProperties:
    """One reservoir layer in the axisymmetric truth grid."""

    thickness_m: float
    permeability_md: float
    porosity_frac: float
    is_fracture_streak: bool


@dataclass
class HiddenParameters:
    """Everything about one well that the twin must infer rather than be told."""

    well_id: str
    seed: int
    permeability_md: float
    net_pay_thickness_m: float
    porosity_frac: float
    initial_pressure_kpa: float
    skin_dimensionless: float
    viscosity_50c_cp: float
    overburden_thermal_conductivity_w_per_m_k: float
    vit_overall_heat_transfer_w_per_m2_k: float
    vit_degradation_per_year_frac: float
    rod_damping_factor: float
    water_cut_initial_frac: float
    water_cut_growth_per_cycle_frac: float
    cycle_energy_retention_frac: float
    productivity_decline_per_cycle_frac: float
    fracture_streak_present: bool
    fracture_permeability_multiplier: float
    gravity_override_strength_frac: float
    unit_type: str
    layers: list[LayerProperties] = field(default_factory=list)
    injected_faults: dict[str, list[int]] = field(default_factory=dict)

    def layer_thicknesses_m(self) -> NDArray[np.float64]:
        """Thickness of each layer."""
        return np.asarray([layer.thickness_m for layer in self.layers], dtype=float)

    def layer_permeabilities_md(self) -> NDArray[np.float64]:
        """Permeability of each layer."""
        return np.asarray([layer.permeability_md for layer in self.layers], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the sealed file."""
        payload = asdict(self)
        payload["layers"] = [asdict(layer) for layer in self.layers]
        return payload


def load_truth_priors(path: Path | None = None) -> dict[str, Any]:
    """Load and lightly validate ``config/truth_priors.yaml``."""
    target = path or (config_dir() / "truth_priors.yaml")
    payload = read_yaml(target)
    for section in ("seed", "wells", "priors", "layers", "sensors", "faults"):
        if section not in payload:
            raise ConfigError(f"Section '{section}' is missing from {target}.", path=str(target))
    return payload


def draw_hidden_parameters(
    well_id: str, well_index: int, priors: dict[str, Any] | None = None
) -> HiddenParameters:
    """Draw one well's hidden parameters reproducibly.

    The seed is the master seed combined with the well index, so the same well
    always gets the same properties and adding a well never changes the ones
    that came before it.
    """
    payload = priors or load_truth_priors()
    seed = int(payload["seed"]) + 1000 * well_index
    rng = np.random.default_rng(seed)
    specs = payload["priors"]

    values = {name: _draw(rng, spec) for name, spec in specs.items()}
    fracture_present = bool(values["fracture_streak_present"] > 0.5)

    layer_count = int(payload["layers"]["count"])
    log_sigma = float(payload["layers"]["permeability_contrast_log_sigma"])
    total_thickness_m = values["net_pay_thickness_m"]
    thickness_weights = rng.uniform(0.6, 1.4, layer_count)
    thickness_weights /= thickness_weights.sum()
    base_permeability = values["permeability_md"]
    contrasts = np.exp(rng.normal(0.0, log_sigma, layer_count))
    contrasts *= layer_count / contrasts.sum()

    fracture_layer = int(rng.integers(0, layer_count)) if fracture_present else -1
    layers: list[LayerProperties] = []
    for index in range(layer_count):
        permeability = base_permeability * float(contrasts[index])
        is_streak = index == fracture_layer
        if is_streak:
            permeability *= values["fracture_permeability_multiplier"]
        layers.append(
            LayerProperties(
                thickness_m=float(total_thickness_m * thickness_weights[index]),
                permeability_md=float(permeability),
                porosity_frac=float(
                    np.clip(values["porosity_frac"] + rng.normal(0.0, 0.015), 0.08, 0.36)
                ),
                is_fracture_streak=is_streak,
            )
        )

    return HiddenParameters(
        well_id=well_id,
        seed=seed,
        permeability_md=values["permeability_md"],
        net_pay_thickness_m=values["net_pay_thickness_m"],
        porosity_frac=values["porosity_frac"],
        initial_pressure_kpa=values["initial_pressure_kpa"],
        skin_dimensionless=values["skin_dimensionless"],
        viscosity_50c_cp=values["viscosity_50c_cp"],
        overburden_thermal_conductivity_w_per_m_k=values[
            "overburden_thermal_conductivity_w_per_m_k"
        ],
        vit_overall_heat_transfer_w_per_m2_k=values["vit_overall_heat_transfer_w_per_m2_k"],
        vit_degradation_per_year_frac=values["vit_degradation_per_year_frac"],
        rod_damping_factor=values["rod_damping_factor"],
        water_cut_initial_frac=values["water_cut_initial_frac"],
        water_cut_growth_per_cycle_frac=values["water_cut_growth_per_cycle_frac"],
        cycle_energy_retention_frac=values["cycle_energy_retention_frac"],
        productivity_decline_per_cycle_frac=values["productivity_decline_per_cycle_frac"],
        fracture_streak_present=fracture_present,
        fracture_permeability_multiplier=values["fracture_permeability_multiplier"],
        gravity_override_strength_frac=values["gravity_override_strength_frac"],
        unit_type="hydraulic" if rng.random() < 0.3 else "conventional",
        layers=layers,
    )


def seal_parameters(parameters: list[HiddenParameters], directory: Path) -> Path:
    """Write the hidden parameters to the sealed file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SEALED_FILENAME
    payload = {
        "warning": (
            "These are the hidden parameters of the truth simulator. Code under "
            "app/twin must never read this file. It exists so a validation run "
            "can be audited after the fact."
        ),
        "wells": [item.to_dict() for item in parameters],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_sealed_parameters(directory: Path) -> list[HiddenParameters]:
    """Read back a sealed parameter file, for validation reporting only."""
    path = directory / SEALED_FILENAME
    if not path.exists():
        raise ConfigError(f"Sealed parameter file not found: {path}", path=str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: list[HiddenParameters] = []
    for item in payload["wells"]:
        layers = [LayerProperties(**layer) for layer in item.pop("layers")]
        result.append(HiddenParameters(layers=layers, **item))
    return result
