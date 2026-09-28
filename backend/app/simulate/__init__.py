"""Synthetic data generation.

Nothing in ``app.twin`` may import from this package. The twin is only ever
allowed to see the noisy observations that come out of it, never the hidden
parameters or the higher-fidelity equations inside. A test in
``tests/test_boundaries.py`` enforces this.
"""
