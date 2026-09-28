"""Structured logging setup shared by the API, the scripts and the tests."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_CONFIGURED = False
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class PlainFormatter(logging.Formatter):
    """Human readable single-line format for terminal use."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )


def configure_logging(level: str | None = None, json_output: bool | None = None) -> None:
    """Configure the root logger once per process.

    Args:
        level: Logging level name. Defaults to the LOG_LEVEL environment
            variable, then to INFO.
        json_output: Emit JSON lines instead of plain text. Defaults to the
            LOG_JSON environment variable being set to a truthy value.
    """
    global _CONFIGURED
    resolved_level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    use_json = (
        json_output
        if json_output is not None
        else os.environ.get("LOG_JSON", "").lower() in {"1", "true", "yes"}
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if use_json else PlainFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root logger on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
