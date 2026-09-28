"""Shared fixtures. Every test runs against the repository configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import (  # noqa: E402
    FieldConfig,
    OptimizerConfig,
    load_field_config,
    load_optimizer_config,
)
from app.twin.fluid import FluidModel  # noqa: E402
from app.twin.reservoir import ReservoirModel  # noqa: E402
from app.twin.srp.kinematics import ConventionalKinematics, SpeedProfile  # noqa: E402
from app.twin.srp.wave import RodTaper, build_taper  # noqa: E402
from app.twin.wellbore import WellboreModel  # noqa: E402


@pytest.fixture(scope="session")
def field_config() -> FieldConfig:
    """Validated field configuration."""
    return load_field_config()


@pytest.fixture(scope="session")
def optimizer_config() -> OptimizerConfig:
    """Validated optimizer configuration."""
    return load_optimizer_config()


@pytest.fixture
def fluid(field_config: FieldConfig) -> FluidModel:
    """Fluid model built from the field configuration."""
    return FluidModel(field_config.fluid)


@pytest.fixture
def reservoir(field_config: FieldConfig, fluid: FluidModel) -> ReservoirModel:
    """Fresh reservoir model at virgin conditions."""
    return ReservoirModel(field_config, fluid)


@pytest.fixture
def wellbore(field_config: FieldConfig, fluid: FluidModel) -> WellboreModel:
    """Wellbore model with the configured completion."""
    return WellboreModel(field_config, fluid)


@pytest.fixture
def taper(field_config: FieldConfig) -> RodTaper:
    """Discretised rod string."""
    return build_taper(field_config.srp)


@pytest.fixture
def kinematics(field_config: FieldConfig) -> ConventionalKinematics:
    """Conventional pumping unit kinematics."""
    return ConventionalKinematics(field_config.srp)


@pytest.fixture
def neutral_profile() -> SpeedProfile:
    """Speed profile that leaves the base motion unchanged."""
    return SpeedProfile()


@pytest.fixture
def viscosity_profile(wellbore: WellboreModel, taper: RodTaper) -> np.ndarray:
    """Viscosity along the rod string for a warm producing well."""
    result = wellbore.produce(
        sandface_temp_c=250.0, liquid_rate_m3_per_day=5.0, water_cut_frac=0.15, elapsed_days=30.0
    )
    return np.interp(taper.depth_m, result.depth_m, result.viscosity_pa_s)
