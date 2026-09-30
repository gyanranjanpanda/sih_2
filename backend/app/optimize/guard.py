"""Supervisory guard: the last thing between an optimizer and a setpoint.

Every recommendation passes through here. The guard does three things and
nothing else, so that it is small enough to be read and trusted:

1. Rejects anything that is not a finite number.
2. Clamps every setpoint into the configured safe envelope.
3. Limits how far a setpoint may move in one step, both in absolute terms and
   as a fraction of the current value.

Modes are ``advisory``, where a human approves before anything is applied, and
``auto``, which exists for simulation only. Nothing in this repository connects
either mode to hardware, and the API refuses to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import FieldConfig, GuardConfig, OptimizerConfig
from app.core.errors import ConstraintViolation
from app.core.logging import get_logger
from app.core.numerics import clamp
from app.twin.coupled import PumpSetpoint
from app.twin.srp.kinematics import SpeedProfile

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SafetyEnvelope:
    """Absolute bounds a setpoint may never leave."""

    spm_min: float
    spm_max: float
    stroke_options_m: tuple[float, ...]
    downstroke_speed_min: float
    downstroke_speed_max: float
    upstroke_speed_min: float
    upstroke_speed_max: float
    decel_max: float
    spm_change_max_per_step: float
    stroke_change_max_per_step_m: float

    @classmethod
    def from_config(
        cls, config: FieldConfig, optimizer: OptimizerConfig
    ) -> SafetyEnvelope:
        """Build the envelope from the field and optimizer configuration."""
        variables = optimizer.pump.variables
        constraints = optimizer.pump.constraints
        return cls(
            spm_min=max(config.srp.spm_min, variables["spm"].low),
            spm_max=min(config.srp.spm_max, variables["spm"].high),
            stroke_options_m=tuple(sorted(config.srp.stroke_length_options_m)),
            downstroke_speed_min=variables["downstroke_speed_frac"].low,
            downstroke_speed_max=variables["downstroke_speed_frac"].high,
            upstroke_speed_min=variables["upstroke_speed_frac"].low,
            upstroke_speed_max=variables["upstroke_speed_frac"].high,
            decel_max=variables["top_of_downstroke_decel_frac"].high,
            spm_change_max_per_step=min(
                config.srp.spm_rate_limit_per_step, constraints.max_spm_change_per_step
            ),
            stroke_change_max_per_step_m=constraints.max_stroke_change_per_step_m,
        )

    def nearest_stroke_m(self, value: float) -> float:
        """Closest stroke the unit can actually be set to."""
        options = np.asarray(self.stroke_options_m, dtype=float)
        return float(options[int(np.argmin(np.abs(options - value)))])


@dataclass
class GuardDecision:
    """What the guard did to a proposed setpoint, and why."""

    accepted: bool
    setpoint: PumpSetpoint
    adjustments: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    mode: str = "advisory"
    requires_approval: bool = True

    @property
    def was_modified(self) -> bool:
        """Whether the guard changed anything."""
        return bool(self.adjustments)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the API and the audit log."""
        return {
            "accepted": self.accepted,
            "setpoint": self.setpoint.as_dict(),
            "adjustments": self.adjustments,
            "rejections": self.rejections,
            "mode": self.mode,
            "requires_approval": self.requires_approval,
            "modified": self.was_modified,
        }

    def explanation(self) -> str:
        """One paragraph an operator can read."""
        if not self.accepted:
            return "Rejected: " + "; ".join(self.rejections)
        if not self.adjustments:
            return "Accepted without change."
        return "Accepted with changes: " + "; ".join(self.adjustments)


class Guard:
    """Clamps and rate-limits every setpoint before it is published."""

    def __init__(
        self,
        config: FieldConfig,
        optimizer: OptimizerConfig,
        guard_config: GuardConfig | None = None,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.settings = guard_config or optimizer.guard
        self.envelope = SafetyEnvelope.from_config(config, optimizer)

    @property
    def mode(self) -> str:
        """Operating mode: advisory or auto."""
        return self.settings.mode

    def review(
        self, proposed: PumpSetpoint, current: PumpSetpoint | None = None
    ) -> GuardDecision:
        """Check and if necessary correct a proposed setpoint.

        Args:
            proposed: What the optimizer asked for.
            current: What the well is running now. Rate limits are measured
                against it. ``None`` skips the rate limits, which is only
                correct when the well is being started.
        """
        adjustments: list[str] = []
        rejections: list[str] = []

        values = [
            proposed.spm,
            proposed.stroke_length_m,
            proposed.speed_profile.upstroke_speed_frac,
            proposed.speed_profile.downstroke_speed_frac,
            proposed.speed_profile.top_of_downstroke_decel_frac,
        ]
        if self.settings.reject_on_nan and not np.all(np.isfinite(values)):
            rejections.append(
                "The proposed setpoint contains a value that is not a finite number."
            )
            return GuardDecision(
                accepted=False,
                setpoint=current or proposed,
                rejections=rejections,
                mode=self.mode,
                requires_approval=self.mode == "advisory",
            )

        envelope = self.envelope
        spm = proposed.spm
        if self.settings.clamp_to_config_bounds:
            clamped = clamp(spm, envelope.spm_min, envelope.spm_max)
            if not np.isclose(clamped, spm):
                adjustments.append(
                    f"speed clamped from {spm:.2f} to {clamped:.2f} strokes per minute, "
                    f"the unit envelope is {envelope.spm_min:.2f} to {envelope.spm_max:.2f}"
                )
                spm = clamped

        stroke_m = envelope.nearest_stroke_m(proposed.stroke_length_m)
        if not np.isclose(stroke_m, proposed.stroke_length_m):
            adjustments.append(
                f"stroke moved from {proposed.stroke_length_m:.2f} m to the nearest "
                f"available crank position at {stroke_m:.2f} m"
            )

        upstroke = clamp(
            proposed.speed_profile.upstroke_speed_frac,
            envelope.upstroke_speed_min,
            envelope.upstroke_speed_max,
        )
        downstroke = clamp(
            proposed.speed_profile.downstroke_speed_frac,
            envelope.downstroke_speed_min,
            envelope.downstroke_speed_max,
        )
        decel = clamp(
            proposed.speed_profile.top_of_downstroke_decel_frac, 0.0, envelope.decel_max
        )
        for name, before, after in (
            ("upstroke speed", proposed.speed_profile.upstroke_speed_frac, upstroke),
            ("downstroke speed", proposed.speed_profile.downstroke_speed_frac, downstroke),
            ("top of downstroke deceleration",
             proposed.speed_profile.top_of_downstroke_decel_frac, decel),
        ):
            if not np.isclose(before, after):
                adjustments.append(f"{name} clamped from {before:.2f} to {after:.2f}")

        if current is not None:
            spm_step = spm - current.spm
            limit = envelope.spm_change_max_per_step
            relative_limit = max(
                self.settings.max_relative_change_per_step_frac * current.spm, 1.0e-6
            )
            allowed = min(limit, relative_limit)
            if abs(spm_step) > allowed:
                limited = current.spm + float(np.sign(spm_step)) * allowed
                adjustments.append(
                    f"speed change limited to {allowed:.2f} strokes per minute per step, so "
                    f"{spm:.2f} becomes {limited:.2f}. Large jumps shock the rod string "
                    "and are what the rate limit exists to prevent."
                )
                spm = limited
                spm = clamp(spm, envelope.spm_min, envelope.spm_max)

            stroke_step = stroke_m - current.stroke_length_m
            if abs(stroke_step) > envelope.stroke_change_max_per_step_m:
                adjustments.append(
                    f"stroke change of {abs(stroke_step):.2f} m exceeds the "
                    f"{envelope.stroke_change_max_per_step_m:.2f} m per step limit, so the "
                    "stroke is held. Changing a crank hole is a workover, not a setpoint."
                )
                stroke_m = current.stroke_length_m

        final = PumpSetpoint(
            spm=spm,
            stroke_length_m=stroke_m,
            speed_profile=SpeedProfile(
                upstroke_speed_frac=upstroke,
                downstroke_speed_frac=downstroke,
                top_of_downstroke_decel_frac=decel,
                decel_window_frac=proposed.speed_profile.decel_window_frac,
            ),
        )
        decision = GuardDecision(
            accepted=True,
            setpoint=final,
            adjustments=adjustments,
            mode=self.mode,
            requires_approval=self.mode == "advisory",
        )
        if adjustments:
            LOGGER.info(
                "Guard adjusted a proposed setpoint",
                extra={"adjustments": adjustments, "mode": self.mode},
            )
        return decision

    def require_accepted(
        self, proposed: PumpSetpoint, current: PumpSetpoint | None = None
    ) -> PumpSetpoint:
        """Return the guarded setpoint, raising if the guard rejected it."""
        decision = self.review(proposed, current)
        if not decision.accepted:
            raise ConstraintViolation(
                "The guard rejected the proposed setpoint.", rejections=decision.rejections
            )
        return decision.setpoint
