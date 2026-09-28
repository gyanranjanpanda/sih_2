"""The import boundary that keeps the validation honest.

If the twin could see the truth simulator, every validation number in
``docs/VALIDATION.md`` would be meaningless: the model would be marking its own
homework. This test walks the source of ``app.twin`` and ``app.optimize`` and
fails if anything there reaches into ``app.simulate`` or opens the sealed
parameter file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = ("app.simulate",)
SEALED_FILENAME = "sealed_truth_parameters"


def _python_files(package: str) -> list[Path]:
    return sorted((BACKEND / "app" / package).rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("package", ["twin", "optimize", "ml", "ingestion", "api"])
def test_package_does_not_import_the_truth_simulator(package: str) -> None:
    directory = BACKEND / "app" / package
    if not directory.exists():
        pytest.skip(f"package app/{package} does not exist yet")
    offenders: list[str] = []
    for path in _python_files(package):
        for module in _imported_modules(path):
            if any(module.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(BACKEND)} imports {module}")
    assert not offenders, (
        "The twin and the layers built on it must never see the truth simulator:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("package", ["twin", "optimize", "ml"])
def test_package_does_not_read_the_sealed_parameter_file(package: str) -> None:
    directory = BACKEND / "app" / package
    if not directory.exists():
        pytest.skip(f"package app/{package} does not exist yet")
    offenders = [
        str(path.relative_to(BACKEND))
        for path in _python_files(package)
        if SEALED_FILENAME in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "The sealed truth parameters must never be read by the twin: " + ", ".join(offenders)
    )


def test_truth_simulator_may_reuse_twin_components() -> None:
    """The boundary is one way.

    The truth simulator is allowed to import shared pieces from the twin, such
    as the card feature extraction or the unit kinematics, because reusing them
    does not leak any hidden information in the direction that matters. What it
    must not do is share the physics that is supposed to differ, and the
    physics tests check that the two models actually disagree.
    """
    modules: set[str] = set()
    for path in _python_files("simulate"):
        modules.update(_imported_modules(path))
    assert any(module.startswith("app.twin") for module in modules)
