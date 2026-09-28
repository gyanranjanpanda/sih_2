"""Pumping unit kinematics for conventional crank units and hydraulic units.

Both unit types produce the same :class:`StrokeMotion` record, so the wave
solver, the card builder and the optimizers are written once and work for
either. This matters for Baghewala, where Oil India runs both conventional and
hydraulic sucker rod pump units and the hydraulic units allow an arbitrary
intra-stroke velocity profile that a crank unit cannot reach.

Sign convention: polished rod position is measured upward from the bottom of
the stroke, so it runs from 0 at the bottom to the stroke length at the top.
Positive velocity is upward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.core.config import SrpConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import require_finite, require_positive


@dataclass(frozen=True)
class SpeedProfile:
    """Intra-stroke speed shaping applied on top of the base pumping speed.

    The published rod-float control art notes that reducing motor speed after
    float is detected does not prevent float, because by then the rods may
    already be in the fast part of the stroke. The fix is to shape the speed
    inside the stroke, in particular to decelerate into the top of the
    downstroke. These four numbers are what the pump optimizer searches over.

    Attributes:
        upstroke_speed_frac: Multiplier on the base speed during the upstroke.
        downstroke_speed_frac: Multiplier on the base speed during the downstroke.
        top_of_downstroke_decel_frac: Extra slowdown applied at the start of the
            downstroke, tapering linearly to zero over ``decel_window_frac`` of
            the downstroke. A value of 0.4 means the unit runs at 60 percent of
            the downstroke speed right after the top of stroke.
        decel_window_frac: Fraction of the downstroke over which the extra
            deceleration tapers out.
    """

    upstroke_speed_frac: float = 1.0
    downstroke_speed_frac: float = 1.0
    top_of_downstroke_decel_frac: float = 0.0
    decel_window_frac: float = 0.35

    def __post_init__(self) -> None:
        for name in ("upstroke_speed_frac", "downstroke_speed_frac"):
            value = getattr(self, name)
            if not 0.05 <= value <= 3.0:
                raise PhysicsDomainError(
                    f"{name} must lie in [0.05, 3.0].", quantity=name, value=value
                )
        if not 0.0 <= self.top_of_downstroke_decel_frac < 1.0:
            raise PhysicsDomainError(
                "top_of_downstroke_decel_frac must lie in [0, 1).",
                value=self.top_of_downstroke_decel_frac,
            )
        if not 0.0 < self.decel_window_frac <= 1.0:
            raise PhysicsDomainError(
                "decel_window_frac must lie in (0, 1].", value=self.decel_window_frac
            )

    @property
    def is_neutral(self) -> bool:
        """Whether the profile leaves the base motion unchanged."""
        return (
            math.isclose(self.upstroke_speed_frac, 1.0)
            and math.isclose(self.downstroke_speed_frac, 1.0)
            and self.top_of_downstroke_decel_frac == 0.0
        )

    def multiplier(self, phase_frac: NDArray[np.float64]) -> NDArray[np.float64]:
        """Speed multiplier against cycle phase, 0 at bottom of stroke to 1 at end.

        The upstroke occupies phase 0 to 0.5 and the downstroke 0.5 to 1.0 in
        the unshaped motion.
        """
        phase = np.asarray(phase_frac, dtype=float)
        multiplier = np.where(phase < 0.5, self.upstroke_speed_frac, self.downstroke_speed_frac)
        downstroke_progress = np.clip((phase - 0.5) / 0.5, 0.0, 1.0)
        taper = np.clip(1.0 - downstroke_progress / self.decel_window_frac, 0.0, 1.0)
        decel = np.where(phase >= 0.5, 1.0 - self.top_of_downstroke_decel_frac * taper, 1.0)
        return np.asarray(multiplier * decel, dtype=float)


@dataclass(frozen=True)
class StrokeMotion:
    """One complete pumping cycle sampled at uniform time steps.

    Attributes:
        time_s: Sample times over one cycle, starting at the bottom of stroke.
        position_m: Polished rod position measured up from the bottom of stroke.
        velocity_m_per_s: Polished rod velocity, positive upward.
        acceleration_m_per_s2: Polished rod acceleration.
        crank_angle_rad: Crank angle for a conventional unit, or the equivalent
            cycle phase scaled to 2 pi for a hydraulic unit.
        torque_factor_m: Ratio of polished rod displacement to crank rotation,
            in metres per radian. Used for the gearbox torque check. It is zero
            for a hydraulic unit, which has no gearbox.
        cycle_time_s: Duration of one complete cycle.
        stroke_length_m: Full stroke length.
        unit_type: ``conventional`` or ``hydraulic``.
    """

    time_s: NDArray[np.float64]
    position_m: NDArray[np.float64]
    velocity_m_per_s: NDArray[np.float64]
    acceleration_m_per_s2: NDArray[np.float64]
    crank_angle_rad: NDArray[np.float64]
    torque_factor_m: NDArray[np.float64]
    cycle_time_s: float
    stroke_length_m: float
    unit_type: str

    @property
    def spm(self) -> float:
        """Effective strokes per minute of this cycle."""
        return 60.0 / self.cycle_time_s

    @property
    def time_step_s(self) -> float:
        """Uniform sample interval."""
        return float(self.time_s[1] - self.time_s[0])

    @property
    def downstroke_mask(self) -> NDArray[np.bool_]:
        """True where the polished rod is moving down."""
        return self.velocity_m_per_s < 0.0

    @property
    def peak_downstroke_speed_m_per_s(self) -> float:
        """Largest downward speed reached in the cycle."""
        return float(np.max(np.abs(np.minimum(self.velocity_m_per_s, 0.0))))

    @property
    def peak_upstroke_speed_m_per_s(self) -> float:
        """Largest upward speed reached in the cycle."""
        return float(np.max(np.maximum(self.velocity_m_per_s, 0.0)))


class Kinematics(Protocol):
    """Interface every pumping unit kinematic model implements."""

    unit_type: str

    def motion(
        self, spm: float, stroke_length_m: float, profile: SpeedProfile, samples: int
    ) -> StrokeMotion:
        """Return one complete cycle of polished rod motion."""
        ...


def _resample_uniform_time(
    raw_time_s: NDArray[np.float64],
    raw_position_m: NDArray[np.float64],
    raw_phase: NDArray[np.float64],
    samples: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Resample a cycle given on a non-uniform time base onto a uniform one.

    Returns time, position, velocity, acceleration, phase and the cycle time.
    Velocity and acceleration are obtained by central differences on the
    resampled position with periodic wrap-around, which keeps the cycle closed.
    """
    cycle_time_s = float(raw_time_s[-1])
    require_positive(cycle_time_s, "cycle_time_s")
    time_s = np.linspace(0.0, cycle_time_s, samples, endpoint=False)
    position_m = np.interp(time_s, raw_time_s, raw_position_m)
    phase = np.interp(time_s, raw_time_s, raw_phase)
    step_s = cycle_time_s / samples
    velocity = (np.roll(position_m, -1) - np.roll(position_m, 1)) / (2.0 * step_s)
    acceleration = (np.roll(position_m, -1) - 2.0 * position_m + np.roll(position_m, 1)) / step_s**2
    return time_s, position_m, velocity, acceleration, phase, cycle_time_s


class ConventionalKinematics:
    """Crank-driven beam unit using the API Spec 11E four-bar linkage.

    The linkage is crankshaft, crank arm R, pitman P, beam rear arm C, with the
    crankshaft centre a distance K from the saddle bearing. With crank angle
    theta measured from the line joining the saddle bearing to the crankshaft:

        J(theta)  = sqrt(K^2 + R^2 - 2 K R cos(theta))
        beta      = arccos((J^2 + C^2 - P^2) / (2 J C))
        rho       = arccos((J^2 + K^2 - R^2) / (2 J K))
        psi       = beta + rho for theta up to pi, beta - rho beyond it

    and the polished rod position is the horsehead arm A times the beam angle
    swing. Unlike an in-line slider crank, this linkage is genuinely
    asymmetric: the upstroke and the downstroke take different numbers of
    degrees of crank rotation and reach different peak velocities. That
    asymmetry is not a detail, it is part of why rod float happens on the
    downstroke of a heavy-oil well.

    The crank radius is not configured. It is solved for from the selected
    stroke length, which is what moving the crank pin to a different hole
    actually does on the unit.

    Source: API Spec 11E, Specification for Pumping Units, geometry annex;
    Svinos, J. G. (1983), Exact kinematic analysis of pumping units, SPE 12201.
    """

    unit_type = "conventional"

    def __init__(self, config: SrpConfig) -> None:
        self.config = config
        self.rear_arm_m = config.walking_beam_rear_arm_m
        self.front_arm_m = config.walking_beam_front_arm_m
        self.pitman_m = config.pitman_length_m
        self.saddle_distance_m = config.crank_to_saddle_distance_m
        self._crank_radius_cache: dict[float, float] = {}

    def _beam_angle(self, crank_radius_m: float, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Walking beam angle against crank angle, in radians."""
        k = self.saddle_distance_m
        c = self.rear_arm_m
        p = self.pitman_m
        j = np.sqrt(k**2 + crank_radius_m**2 - 2.0 * k * crank_radius_m * np.cos(theta))
        j = np.maximum(j, 1.0e-6)
        beta = np.arccos(np.clip((j**2 + c**2 - p**2) / (2.0 * j * c), -1.0, 1.0))
        rho = np.arccos(np.clip((j**2 + k**2 - crank_radius_m**2) / (2.0 * j * k), -1.0, 1.0))
        return np.where(theta <= math.pi, beta + rho, beta - rho)

    def _swing_rad(self, crank_radius_m: float, samples: int = 2000) -> float:
        theta = np.linspace(0.0, 2.0 * math.pi, samples)
        psi = self._beam_angle(crank_radius_m, theta)
        return float(psi.max() - psi.min())

    def crank_radius_m(self, stroke_length_m: float) -> float:
        """Crank radius that produces the requested stroke on this linkage.

        Solved by bisection on the monotone relation between crank radius and
        beam angle swing. A stroke the linkage cannot reach raises a clear
        error rather than silently returning the nearest achievable value.
        """
        cached = self._crank_radius_cache.get(stroke_length_m)
        if cached is not None:
            return cached
        require_positive(stroke_length_m, "stroke_length_m")
        low, high = 0.05, 0.9 * min(self.saddle_distance_m, self.pitman_m)

        def stroke_error(radius_m: float) -> float:
            return self._swing_rad(radius_m) * self.front_arm_m - stroke_length_m

        if stroke_error(low) > 0.0 or stroke_error(high) < 0.0:
            raise PhysicsDomainError(
                "The configured linkage cannot reach the requested stroke length.",
                stroke_length_m=stroke_length_m,
                reachable_min_m=self._swing_rad(low) * self.front_arm_m,
                reachable_max_m=self._swing_rad(high) * self.front_arm_m,
            )
        for _ in range(80):
            mid = 0.5 * (low + high)
            if stroke_error(mid) < 0.0:
                low = mid
            else:
                high = mid
        radius_m = 0.5 * (low + high)
        self._crank_radius_cache[stroke_length_m] = radius_m
        return radius_m

    def motion(
        self,
        spm: float,
        stroke_length_m: float,
        profile: SpeedProfile | None = None,
        samples: int = 360,
    ) -> StrokeMotion:
        """Polished rod motion over one crank revolution, starting at the bottom.

        Args:
            spm: Base strokes per minute commanded by the drive. With a
                non-neutral speed profile the achieved strokes per minute
                differ, and the returned motion reports the achieved value.
            stroke_length_m: Selected stroke length.
            profile: Intra-stroke speed shaping applied as a multiplier on the
                crank angular velocity. A variable speed drive is what makes
                this possible on a crank unit.
            samples: Number of uniform time samples in the returned cycle.
        """
        require_positive(spm, "spm")
        profile = profile or SpeedProfile()
        crank_radius_m = self.crank_radius_m(stroke_length_m)

        angle_samples = max(samples * 4, 1440)
        theta_raw = np.linspace(0.0, 2.0 * math.pi, angle_samples, endpoint=False)
        psi = self._beam_angle(crank_radius_m, theta_raw)
        position_raw = self.front_arm_m * (psi - psi.min())

        # Re-index so the cycle starts at the bottom of the stroke.
        bottom_index = int(np.argmin(position_raw))
        position_m = np.roll(position_raw, -bottom_index)
        theta = np.mod(np.roll(theta_raw, -bottom_index) - theta_raw[bottom_index], 2.0 * math.pi)
        theta[0] = 0.0
        theta = np.unwrap(theta)

        d_theta = np.gradient(theta)
        torque_factor_full = np.gradient(position_m) / np.where(
            np.abs(d_theta) < 1.0e-12, 1.0e-12, d_theta
        )

        base_angular_speed = 2.0 * math.pi * spm / 60.0
        angular_speed = base_angular_speed * profile.multiplier(theta / (2.0 * math.pi))
        angular_speed = np.clip(angular_speed, 1.0e-4, None)
        increments = d_theta / angular_speed
        raw_time_s = np.concatenate([[0.0], np.cumsum(increments)[:-1]])
        raw_time_s = np.append(raw_time_s, raw_time_s[-1] + float(increments[-1]))
        position_closed = np.append(position_m, position_m[0])
        phase_closed = np.append(theta, 2.0 * math.pi)

        time_s, position, velocity, acceleration, phase, cycle_time_s = _resample_uniform_time(
            raw_time_s, position_closed, phase_closed, samples
        )
        torque_factor = np.interp(
            phase,
            np.append(theta, 2.0 * math.pi),
            np.append(torque_factor_full, torque_factor_full[0]),
        )
        require_finite(position, "polished rod position")
        return StrokeMotion(
            time_s=time_s,
            position_m=position,
            velocity_m_per_s=velocity,
            acceleration_m_per_s2=acceleration,
            crank_angle_rad=phase,
            torque_factor_m=torque_factor,
            cycle_time_s=cycle_time_s,
            stroke_length_m=stroke_length_m,
            unit_type=self.unit_type,
        )


class HydraulicKinematics:
    """Hydraulic long-stroke unit with an arbitrary velocity versus position profile.

    A hydraulic unit is not tied to crank geometry. The cylinder can hold a
    near-constant velocity through most of the stroke, run the upstroke and the
    downstroke at genuinely different speeds, and dwell at the ends. That is a
    larger control space than a crank unit with a variable speed drive, and it
    is the reason the pump optimizer treats the two unit types differently.

    The profile implemented here is a trapezoid in velocity: a linear ramp at
    each end of each stroke over a fixed fraction of the stroke, a flat section
    between, and an optional dwell at each end of the stroke. Flow and pressure
    limits from the power unit are checked separately in
    :mod:`app.twin.srp.power`.
    """

    unit_type = "hydraulic"

    def __init__(self, config: SrpConfig, ramp_fraction: float = 0.12) -> None:
        self.config = config
        if not 0.0 < ramp_fraction < 0.5:
            raise PhysicsDomainError(
                "ramp_fraction must lie in (0, 0.5).", ramp_fraction=ramp_fraction
            )
        self.ramp_fraction = ramp_fraction

    def motion(
        self,
        spm: float,
        stroke_length_m: float,
        profile: SpeedProfile | None = None,
        samples: int = 360,
    ) -> StrokeMotion:
        """Polished rod motion over one hydraulic cycle.

        Args:
            spm: Reference strokes per minute. It sets the nominal cycle time;
                the speed profile and the dwell then change the achieved value,
                which the returned motion reports.
            stroke_length_m: Full stroke length.
            profile: Up and down speed fractions and the deceleration into the
                top of the downstroke.
            samples: Number of uniform time samples in the returned cycle.
        """
        require_positive(spm, "spm")
        require_positive(stroke_length_m, "stroke_length_m")
        profile = profile or SpeedProfile()
        dwell_s = self.config.hydraulic_dwell_s
        nominal_cycle_s = 60.0 / spm
        moving_time_s = max(nominal_cycle_s - 2.0 * dwell_s, 0.2 * nominal_cycle_s)
        # Split the moving time so that the two strokes take time inversely
        # proportional to their commanded speeds.
        up_weight = 1.0 / profile.upstroke_speed_frac
        down_weight = 1.0 / profile.downstroke_speed_frac
        total_weight = up_weight + down_weight
        up_time_s = moving_time_s * up_weight / total_weight
        down_time_s = moving_time_s * down_weight / total_weight

        resolution = max(samples * 4, 720)
        up_points = max(int(resolution * 0.4), 40)
        down_points = up_points
        dwell_points = max(int(resolution * 0.05), 8)

        def trapezoid_positions(
            length_m: float, duration_s: float, points: int, decel_frac: float
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            """Positions travelled and the times at which they are reached."""
            local_phase = np.linspace(0.0, 1.0, points)
            shape = np.clip(local_phase / self.ramp_fraction, 0.0, 1.0) * np.clip(
                (1.0 - local_phase) / self.ramp_fraction, 0.0, 1.0
            )
            shape = np.clip(shape, 0.02, 1.0)
            if decel_frac > 0.0:
                taper = np.clip(1.0 - local_phase / profile.decel_window_frac, 0.0, 1.0)
                shape = shape * (1.0 - decel_frac * taper)
            shape = np.clip(shape, 0.02, None)
            # Normalise so the integral of the speed shape covers the stroke.
            cumulative = np.concatenate(
                [[0.0], np.cumsum(0.5 * (shape[:-1] + shape[1:]) * np.diff(local_phase))]
            )
            travelled_m = length_m * cumulative / cumulative[-1]
            times_s = duration_s * local_phase
            return times_s, travelled_m

        up_times, up_travel = trapezoid_positions(stroke_length_m, up_time_s, up_points, 0.0)
        down_times, down_travel = trapezoid_positions(
            stroke_length_m, down_time_s, down_points, profile.top_of_downstroke_decel_frac
        )

        times: list[float] = []
        positions: list[float] = []
        cursor_s = 0.0
        times.extend((cursor_s + up_times).tolist())
        positions.extend(up_travel.tolist())
        cursor_s += up_time_s
        if dwell_s > 0.0:
            dwell_times = np.linspace(0.0, dwell_s, dwell_points, endpoint=False)[1:]
            times.extend((cursor_s + dwell_times).tolist())
            positions.extend([stroke_length_m] * len(dwell_times))
            cursor_s += dwell_s
        times.extend((cursor_s + down_times[1:]).tolist())
        positions.extend((stroke_length_m - down_travel[1:]).tolist())
        cursor_s += down_time_s
        if dwell_s > 0.0:
            dwell_times = np.linspace(0.0, dwell_s, dwell_points, endpoint=False)[1:]
            times.extend((cursor_s + dwell_times).tolist())
            positions.extend([0.0] * len(dwell_times))
            cursor_s += dwell_s
        times.append(cursor_s)
        positions.append(0.0)

        raw_time_s = np.asarray(times, dtype=float)
        raw_position_m = np.asarray(positions, dtype=float)
        raw_phase = raw_time_s / raw_time_s[-1] * 2.0 * math.pi
        time_s, position, velocity, acceleration, phase, cycle_time_s = _resample_uniform_time(
            raw_time_s, raw_position_m, raw_phase, samples
        )
        require_finite(position, "polished rod position")
        return StrokeMotion(
            time_s=time_s,
            position_m=position,
            velocity_m_per_s=velocity,
            acceleration_m_per_s2=acceleration,
            crank_angle_rad=phase,
            torque_factor_m=np.zeros_like(position),
            cycle_time_s=cycle_time_s,
            stroke_length_m=stroke_length_m,
            unit_type=self.unit_type,
        )


def build_kinematics(config: SrpConfig) -> Kinematics:
    """Return the kinematic model for the configured unit type."""
    if config.unit_type == "hydraulic":
        return HydraulicKinematics(config)
    return ConventionalKinematics(config)
