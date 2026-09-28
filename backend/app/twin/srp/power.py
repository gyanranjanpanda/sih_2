"""Drive power, gearbox torque and energy per barrel.

Energy per barrel is one of the KPIs the problem statement asks to reduce, so
the path from the card to the electricity meter is modelled explicitly:
polished rod power from the card area, gearbox and belt losses, and a motor
efficiency that falls away from the rated load. Hydraulic units bypass the
gearbox and are charged on hydraulic power instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import SrpConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import clamp, require_finite
from app.core.units import J_PER_KWH, M3_PER_BBL
from app.twin.srp.kinematics import StrokeMotion


def polished_rod_power_w(
    load_n: NDArray[np.float64], velocity_m_per_s: NDArray[np.float64], cycle_time_s: float
) -> float:
    """Average mechanical power delivered at the polished rod.

    Equation: P = (1 / T) times the integral of F v dt over one cycle.
    Units: W. Load in N, velocity in m/s, cycle time in s.
    Assumption: only positive work counts towards the motor load; on the
    downstroke the rods drive the unit backwards and that energy is absorbed by
    the counterbalance, not returned to the grid. The negative part is therefore
    excluded, which is the standard conservative treatment.
    Source: Gibbs, Rod Pumping, chapter 7.
    """
    if load_n.size != velocity_m_per_s.size:
        raise PhysicsDomainError("Load and velocity series differ in length.")
    if cycle_time_s <= 0.0:
        raise PhysicsDomainError("Cycle time must be positive.", cycle_time_s=cycle_time_s)
    instantaneous = load_n * velocity_m_per_s
    positive_work_j = float(
        np.trapezoid(np.maximum(instantaneous, 0.0), dx=cycle_time_s / load_n.size)
    )
    return positive_work_j / cycle_time_s


def net_gearbox_torque_n_m(
    torque_factor_m: NDArray[np.float64],
    load_n: NDArray[np.float64],
    crank_angle_rad: NDArray[np.float64],
    counterbalance_moment_n_m: float,
    structural_unbalance_n: float = 0.0,
) -> NDArray[np.float64]:
    """Net torque on the speed reducer over one crank revolution.

    Equation: T(theta) = TF(theta) (F(theta) - SU) - M sin(theta).
    Units: N.m. Torque factor in m per radian, load in N, moment in N.m.
    Assumptions: the counterbalance phase angle is zero, and the structural
    unbalance of the beam is folded into the supplied value. These are the
    standard simplifications of the API torque calculation and are adequate for
    a rating check.
    Source: API Spec 11E, torque analysis; Gibbs, Rod Pumping, chapter 7.
    """
    if not (torque_factor_m.size == load_n.size == crank_angle_rad.size):
        raise PhysicsDomainError("Torque factor, load and crank angle series differ in length.")
    return torque_factor_m * (load_n - structural_unbalance_n) - (
        counterbalance_moment_n_m * np.sin(crank_angle_rad)
    )


def motor_efficiency_frac(load_fraction: float, peak_efficiency_frac: float) -> float:
    """Induction motor efficiency against load fraction.

    Equation: eta = eta_peak * (1 - 0.5 (1 - x)^2) for x in (0, 1.2], with x the
    shaft load as a fraction of the rated power.
    Units: dimensionless.
    Assumptions: this is a smooth stand-in for a measured efficiency map. It
    reproduces the two features that matter here: efficiency is near its peak
    between about 60 and 100 percent load, and falls away steeply at light load,
    which is the regime a rod pump on a slow setting runs in. A measured map can
    replace it without changing any caller.
    Source: shape follows NEMA premium efficiency motor test curves.
    """
    if peak_efficiency_frac <= 0.0 or peak_efficiency_frac > 1.0:
        raise PhysicsDomainError(
            "Peak motor efficiency must lie in (0, 1].",
            peak_efficiency_frac=peak_efficiency_frac,
        )
    x = clamp(load_fraction, 0.02, 1.2)
    efficiency = peak_efficiency_frac * (1.0 - 0.5 * (1.0 - x) ** 2)
    return float(clamp(efficiency, 0.15, peak_efficiency_frac))


def spm_from_vfd_frequency(config: SrpConfig, frequency_hz: float) -> float:
    """Strokes per minute produced by a variable speed drive frequency.

    Equation: N = N_base f / f_base, since an induction motor speed is
    proportional to supply frequency and the gearbox ratio is fixed.
    Units: strokes per minute.
    """
    if not config.vfd_frequency_min_hz <= frequency_hz <= config.vfd_frequency_max_hz:
        raise PhysicsDomainError(
            "Drive frequency is outside the configured limits.",
            frequency_hz=frequency_hz,
            low=config.vfd_frequency_min_hz,
            high=config.vfd_frequency_max_hz,
        )
    return config.spm_setpoint * frequency_hz / config.vfd_base_frequency_hz


def vfd_frequency_from_spm(config: SrpConfig, spm: float) -> float:
    """Drive frequency needed for a target strokes per minute."""
    if config.spm_setpoint <= 0.0:
        raise PhysicsDomainError("Configured base speed must be positive.")
    return spm * config.vfd_base_frequency_hz / config.spm_setpoint


@dataclass(frozen=True)
class PowerReport:
    """Energy and torque state of the drive at one operating point."""

    polished_rod_power_w: float
    peak_gearbox_torque_n_m: float
    gearbox_torque_utilisation_frac: float
    motor_shaft_power_w: float
    motor_electrical_power_w: float
    motor_load_fraction: float
    motor_efficiency_frac: float
    hydraulic_power_w: float
    hydraulic_pressure_kpa: float
    hydraulic_flow_m3_per_s: float
    energy_kwh_per_m3: float
    energy_kwh_per_bbl: float
    daily_energy_kwh: float
    within_limits: bool
    binding: str

    def as_dict(self) -> dict[str, float | bool | str]:
        """Serialisable form for the API."""
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def evaluate_power(
    config: SrpConfig,
    motion: StrokeMotion,
    surface_load_n: NDArray[np.float64],
    liquid_rate_m3_per_day: float,
    structural_unbalance_n: float = 0.0,
) -> PowerReport:
    """Full drive assessment for one operating point.

    Conventional units are charged through the gearbox, the belts and the motor.
    Hydraulic units are charged on hydraulic power, computed from the cylinder
    force and the plunger velocity, and are checked against the power unit
    pressure and flow ratings instead of a gearbox rating.
    """
    require_finite(surface_load_n, "surface load")
    rod_power_w = polished_rod_power_w(surface_load_n, motion.velocity_m_per_s, motion.cycle_time_s)

    binding_notes: list[str] = []
    within_limits = True

    if motion.unit_type == "hydraulic":
        peak_torque_n_m = 0.0
        torque_utilisation = 0.0
        force_pa = np.abs(surface_load_n) / config.hydraulic_cylinder_area_m2
        peak_pressure_kpa = float(np.max(force_pa)) / 1000.0
        flow_m3_per_s = float(
            np.max(np.abs(motion.velocity_m_per_s)) * config.hydraulic_cylinder_area_m2
        )
        hydraulic_power_w = rod_power_w / max(config.gearbox_efficiency_frac, 1.0e-3)
        shaft_power_w = hydraulic_power_w
        if peak_pressure_kpa > config.hydraulic_max_pressure_kpa:
            within_limits = False
            binding_notes.append(
                f"Hydraulic pressure {peak_pressure_kpa:.0f} kPa exceeds the "
                f"{config.hydraulic_max_pressure_kpa:.0f} kPa power unit rating."
            )
        if flow_m3_per_s > config.hydraulic_max_flow_m3_per_s:
            within_limits = False
            binding_notes.append(
                f"Hydraulic flow {flow_m3_per_s * 1000.0:.2f} L/s exceeds the "
                f"{config.hydraulic_max_flow_m3_per_s * 1000.0:.2f} L/s rating."
            )
    else:
        torque_n_m = net_gearbox_torque_n_m(
            motion.torque_factor_m,
            surface_load_n,
            motion.crank_angle_rad,
            config.counterbalance_moment_n_m,
            structural_unbalance_n,
        )
        peak_torque_n_m = float(np.max(np.abs(torque_n_m)))
        torque_utilisation = peak_torque_n_m / config.gearbox_torque_rating_n_m
        hydraulic_power_w = 0.0
        peak_pressure_kpa = 0.0
        flow_m3_per_s = 0.0
        shaft_power_w = rod_power_w / max(
            config.gearbox_efficiency_frac * config.belt_efficiency_frac, 1.0e-3
        )
        if torque_utilisation > 1.0:
            within_limits = False
            binding_notes.append(
                f"Gearbox torque {peak_torque_n_m / 1000.0:.1f} kN.m exceeds the "
                f"{config.gearbox_torque_rating_n_m / 1000.0:.1f} kN.m rating."
            )

    peak_load_n = float(np.max(surface_load_n))
    if peak_load_n > config.structural_load_rating_n:
        within_limits = False
        binding_notes.append(
            f"Peak polished rod load {peak_load_n / 1000.0:.1f} kN exceeds the "
            f"{config.structural_load_rating_n / 1000.0:.1f} kN structural rating."
        )

    load_fraction = shaft_power_w / max(config.motor_rated_power_w, 1.0)
    efficiency = motor_efficiency_frac(load_fraction, config.motor_efficiency_peak_frac)
    electrical_power_w = shaft_power_w / efficiency
    if load_fraction > 1.0:
        within_limits = False
        binding_notes.append(f"Motor shaft load is {load_fraction * 100.0:.0f} percent of rating.")

    daily_energy_kwh = electrical_power_w * 24.0 * 3600.0 / J_PER_KWH
    energy_per_m3 = (
        daily_energy_kwh / liquid_rate_m3_per_day if liquid_rate_m3_per_day > 0.0 else math.inf
    )
    return PowerReport(
        polished_rod_power_w=rod_power_w,
        peak_gearbox_torque_n_m=peak_torque_n_m,
        gearbox_torque_utilisation_frac=torque_utilisation,
        motor_shaft_power_w=shaft_power_w,
        motor_electrical_power_w=electrical_power_w,
        motor_load_fraction=load_fraction,
        motor_efficiency_frac=efficiency,
        hydraulic_power_w=hydraulic_power_w,
        hydraulic_pressure_kpa=peak_pressure_kpa,
        hydraulic_flow_m3_per_s=flow_m3_per_s,
        energy_kwh_per_m3=energy_per_m3,
        energy_kwh_per_bbl=energy_per_m3 * M3_PER_BBL if math.isfinite(energy_per_m3) else math.inf,
        daily_energy_kwh=daily_energy_kwh,
        within_limits=within_limits,
        binding="; ".join(binding_notes) if binding_notes else "All drive limits satisfied.",
    )
