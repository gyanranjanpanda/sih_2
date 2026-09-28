"""Typed errors with a single consistent shape for API and CLI reporting."""

from __future__ import annotations

from typing import Any


class WellTwinError(Exception):
    """Base error. Carries a machine code, a human message and structured detail."""

    code = "well_twin_error"
    http_status = 500

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the one error shape used everywhere."""
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


class ConfigError(WellTwinError):
    """Configuration missing, malformed, or out of the allowed range."""

    code = "config_error"
    http_status = 500


class UnitError(WellTwinError):
    """A unit conversion was asked for that does not exist or is not dimensionally valid."""

    code = "unit_error"
    http_status = 400


class PhysicsDomainError(WellTwinError):
    """A physics function received an argument outside its valid domain."""

    code = "physics_domain_error"
    http_status = 400


class NumericalError(WellTwinError):
    """A solver produced a non-finite value or violated its stability condition."""

    code = "numerical_error"
    http_status = 500


class DataValidationError(WellTwinError):
    """Ingested data failed schema, unit, or quality checks."""

    code = "data_validation_error"
    http_status = 422


class ConstraintViolation(WellTwinError):
    """An optimizer or guard rejected a candidate because a hard constraint failed."""

    code = "constraint_violation"
    http_status = 409


class NotFoundError(WellTwinError):
    """A requested entity does not exist."""

    code = "not_found"
    http_status = 404


class ModelNotTrainedError(WellTwinError):
    """A machine learning artifact was requested before it was trained."""

    code = "model_not_trained"
    http_status = 503
