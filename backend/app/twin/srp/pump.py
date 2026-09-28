"""Downhole pump performance: displacement, slippage, fillage and gas.

The pump sets the upper bound on what the well can produce, so this module is
what couples the surface equipment back to the reservoir in the twin. The
slippage term matters at Baghewala: a hot, low viscosity fluid at the plunger
slips more past the same clearance than a cold one, so the volumetric
efficiency changes through the cycle as the reservoir cools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.config import SrpConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import clamp, require_positive
from app.core.units import SECONDS_PER_DAY


def pump_displacement_m3_per_day(
    plunger_area_m2: float, plunger_stroke_m: float, spm: float
) -> float:
    """Theoretical displacement of the pump.

    Equation: q = A_p S_p N, converted from strokes per minute to per day.
    Units: m3/day. Area in m2, stroke in m, speed in strokes per minute.
    Assumption: the plunger stroke is the downhole stroke from the wave solver,
    not the surface stroke. Rod stretch makes the two differ, typically by 5 to
    15 percent in a 1100 m heavy-oil well.
    """
    require_positive(plunger_area_m2, "plunger_area_m2")
    if plunger_stroke_m < 0.0 or spm < 0.0:
        raise PhysicsDomainError("Stroke and speed must be non-negative.")
    return plunger_area_m2 * plunger_stroke_m * spm * 60.0 * 24.0


def slippage_rate_m3_per_day(
    plunger_diameter_m: float,
    clearance_m: float,
    plunger_length_m: float,
    pressure_difference_pa: float,
    viscosity_pa_s: float,
    plunger_velocity_m_per_s: float = 0.0,
    eccentricity_frac: float = 0.0,
) -> float:
    """Leakage past the plunger, as flow in a thin concentric annulus.

    Equation: q = pi d c^3 dp / (12 mu L) * (1 + 1.5 e^2) + pi d c v / 2,
    the Poiseuille leakage through the plunger-barrel gap plus the Couette
    term dragged by the plunger itself.
    Units: m3/day. Diameter, clearance and length in m, pressure difference in
    Pa, viscosity in Pa.s, velocity in m/s.
    Assumptions: laminar flow in a thin gap, the eccentricity factor bounded at
    a fully eccentric plunger where leakage is 2.5 times the concentric value.
    A worn pump is represented by increasing the clearance, which raises
    leakage with the cube of the gap.
    Source: standard thin-annulus leakage relation, see Takacs, Sucker-Rod
    Pumping Handbook, chapter 4; eccentricity factor from Bird et al.
    """
    require_positive(viscosity_pa_s, "viscosity_pa_s")
    require_positive(plunger_length_m, "plunger_length_m")
    if clearance_m <= 0.0:
        return 0.0
    eccentricity = clamp(eccentricity_frac, 0.0, 1.0)
    poiseuille_m3_per_s = (
        math.pi
        * plunger_diameter_m
        * clearance_m**3
        * max(pressure_difference_pa, 0.0)
        / (12.0 * viscosity_pa_s * plunger_length_m)
    ) * (1.0 + 1.5 * eccentricity**2)
    couette_m3_per_s = 0.5 * math.pi * plunger_diameter_m * clearance_m * plunger_velocity_m_per_s
    return max(poiseuille_m3_per_s + couette_m3_per_s, 0.0) * SECONDS_PER_DAY


@dataclass(frozen=True)
class PumpPerformance:
    """Volumetric performance of the pump at one operating point."""

    displacement_m3_per_day: float
    slippage_m3_per_day: float
    fillage_frac: float
    gas_interference_frac: float
    volumetric_efficiency_frac: float
    liquid_rate_m3_per_day: float
    plunger_stroke_m: float
    spm: float

    def as_dict(self) -> dict[str, float]:
        """Serialisable form for the API."""
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def evaluate_pump(
    config: SrpConfig,
    plunger_stroke_m: float,
    spm: float,
    viscosity_at_pump_pa_s: float,
    pressure_difference_pa: float,
    fillage_frac: float = 1.0,
    gas_interference_frac: float = 0.0,
    plunger_length_m: float = 1.2,
    clearance_multiplier: float = 1.0,
    eccentricity_frac: float = 0.0,
) -> PumpPerformance:
    """Liquid the pump actually delivers at one operating point.

    Equation: q_liquid = (A_p S_p N) * fillage * (1 - gas) - q_slip.
    Units: m3/day.
    Assumptions: the fillage fraction is the liquid share of the barrel at the
    top of the upstroke and the gas fraction the share of the remainder taken
    by free gas. Wear is represented by ``clearance_multiplier`` acting on the
    configured diametral clearance, which raises slippage with the cube.
    """
    if not 0.0 <= fillage_frac <= 1.0:
        raise PhysicsDomainError("Fillage must lie in [0, 1].", fillage_frac=fillage_frac)
    if not 0.0 <= gas_interference_frac < 1.0:
        raise PhysicsDomainError(
            "Gas interference fraction must lie in [0, 1).",
            gas_interference_frac=gas_interference_frac,
        )
    displacement = pump_displacement_m3_per_day(config.plunger_area_m2, plunger_stroke_m, spm)
    average_velocity_m_per_s = 2.0 * plunger_stroke_m * spm / 60.0
    slippage = slippage_rate_m3_per_day(
        plunger_diameter_m=config.plunger_diameter_m,
        clearance_m=config.pump_clearance_m * max(clearance_multiplier, 1.0e-3),
        plunger_length_m=plunger_length_m,
        pressure_difference_pa=pressure_difference_pa,
        viscosity_pa_s=viscosity_at_pump_pa_s,
        plunger_velocity_m_per_s=average_velocity_m_per_s,
        eccentricity_frac=eccentricity_frac,
    )
    gross = displacement * fillage_frac * (1.0 - gas_interference_frac)
    liquid = max(gross - slippage, 0.0)
    efficiency = liquid / displacement if displacement > 0.0 else 0.0
    return PumpPerformance(
        displacement_m3_per_day=displacement,
        slippage_m3_per_day=slippage,
        fillage_frac=fillage_frac,
        gas_interference_frac=gas_interference_frac,
        volumetric_efficiency_frac=efficiency,
        liquid_rate_m3_per_day=liquid,
        plunger_stroke_m=plunger_stroke_m,
        spm=spm,
    )


def fillage_from_inflow(deliverability_m3_per_day: float, displacement_m3_per_day: float) -> float:
    """Fillage implied by the reservoir not keeping up with the pump.

    Equation: fillage = min(1, q_reservoir / q_displacement).
    Units: dimensionless.
    Assumption: the pump fills whatever the reservoir delivers over the stroke,
    with no gas. This is the mechanism that links reservoir cooling to fluid
    pound: as the near-well oil cools and inflow falls, fillage drops and the
    card develops the pound signature.
    """
    if displacement_m3_per_day <= 0.0:
        return 0.0
    return float(clamp(deliverability_m3_per_day / displacement_m3_per_day, 0.0, 1.0))
