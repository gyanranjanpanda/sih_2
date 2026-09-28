"""Reduced-order digital twin.

Nothing in this package may import from ``app.simulate``. The twin sees only
configuration and noisy observations. A test enforces the boundary so the
validation in ``docs/VALIDATION.md`` stays honest.
"""
