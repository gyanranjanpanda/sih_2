#!/usr/bin/env python3
"""Generate the synthetic dataset.

Usage:
    python scripts/generate_data.py [--wells N] [--days N] [--jobs N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.simulate.generator import generate_dataset  # noqa: E402

LOGGER = get_logger("generate_data")


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wells", type=int, default=None, help="number of wells")
    parser.add_argument("--days", type=int, default=150, help="maximum production days per cycle")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers, -1 for all cores")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    arguments = parser.parse_args()

    configure_logging()
    started = time.time()
    written = generate_dataset(
        output_dir=arguments.out,
        well_count=arguments.wells,
        production_days_max=arguments.days,
        n_jobs=arguments.jobs,
    )
    elapsed = time.time() - started

    print(f"\nSYNTHETIC dataset written in {elapsed:.1f} s")
    for name, path in sorted(written.items()):
        size_kb = path.stat().st_size / 1024.0
        print(f"  {name:28s} {path.name:34s} {size_kb:9.1f} kB")
    print(
        "\nEvery number in these files is simulated. They show that the pipeline works.\n"
        "They are not Oil India field data and must never be presented as such."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
