"""Rod stress: modified Goodman utilisation and cumulative fatigue damage.

Rod failures at Baghewala are driven by two things: the high loads that heavy
oil puts on the string, and the impact loading that follows a float event. The
Goodman check covers the first. The Miner counter covers the second, by
accumulating damage cycle by cycle so the failure risk model has a physical
feature rather than only a calendar age.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import SrpConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import require_positive
from app.twin.srp.wave import RodTaper, WaveSolution

ENDURANCE_CYCLES = 1.0e7
"""Cycle count at which the modified Goodman allowable is taken as the endurance limit."""

BASQUIN_EXPONENT = 8.0
"""Slope of the assumed S-N line. See the note in the module docstring of the tests."""


def modified_goodman_allowable_pa(
    minimum_stress_pa: float,
    minimum_tensile_strength_pa: float,
    service_factor: float = 1.0,
) -> float:
    """Maximum allowable stress from the API modified Goodman diagram.

    Equation: S_a = (T / 4 + 0.5625 S_min) * SF.
    Units: Pa. T is the minimum tensile strength of the rod grade, S_min the
    minimum stress seen in the cycle, SF the service factor for the environment.
    Assumptions: the standard API rod fatigue diagram applies, the service
    factor is 1.0 for non-corrosive service and lower where hydrogen sulphide or
    carbon dioxide is present.
    Source: API RP 11BR, Recommended Practice for the Care and Handling of
    Sucker Rods, modified Goodman diagram.
    """
    require_positive(minimum_tensile_strength_pa, "minimum_tensile_strength_pa")
    require_positive(service_factor, "service_factor")
    return (0.25 * minimum_tensile_strength_pa + 0.5625 * max(minimum_stress_pa, 0.0)) * (
        service_factor
    )


def cycles_to_failure(
    stress_amplitude_pa: float, allowable_amplitude_pa: float, exponent: float = BASQUIN_EXPONENT
) -> float:
    """Fatigue life at a given stress amplitude, from a Basquin power law.

    Equation: N = N_e (S_allow / S_a)^m.
    Units: cycles. Stresses in Pa.
    Assumptions: the modified Goodman allowable is anchored at the endurance
    limit of 1e7 cycles, and the S-N line has slope m. The exponent is an
    assumption, not a measured rod property, and is recorded as such. It is used
    only to rank relative damage between operating points, never to predict an
    absolute rod life.
    Source: Basquin (1910); anchoring convention follows API RP 11BR practice.
    """
    if stress_amplitude_pa <= 0.0:
        return float("inf")
    require_positive(allowable_amplitude_pa, "allowable_amplitude_pa")
    ratio = allowable_amplitude_pa / stress_amplitude_pa
    if ratio >= 1.0:
        return ENDURANCE_CYCLES * ratio**exponent
    return ENDURANCE_CYCLES * ratio**exponent


@dataclass(frozen=True)
class SectionStress:
    """Stress state of one taper section over a pumping cycle."""

    section_index: int
    diameter_m: float
    grade: str
    maximum_stress_pa: float
    minimum_stress_pa: float
    stress_range_pa: float
    allowable_stress_pa: float
    utilisation_frac: float
    cycles_to_failure: float

    def as_dict(self) -> dict[str, float | str]:
        """Serialisable form for the API and the reports."""
        return {
            "section_index": self.section_index,
            "diameter_m": self.diameter_m,
            "grade": self.grade,
            "maximum_stress_mpa": self.maximum_stress_pa / 1.0e6,
            "minimum_stress_mpa": self.minimum_stress_pa / 1.0e6,
            "stress_range_mpa": self.stress_range_pa / 1.0e6,
            "allowable_stress_mpa": self.allowable_stress_pa / 1.0e6,
            "utilisation_frac": self.utilisation_frac,
            "cycles_to_failure": self.cycles_to_failure,
        }


@dataclass(frozen=True)
class StressReport:
    """Stress state of the whole rod string."""

    sections: tuple[SectionStress, ...]
    maximum_utilisation_frac: float
    limiting_section_index: int

    @property
    def is_within_limit(self) -> bool:
        """Whether every section is inside its modified Goodman allowable."""
        return self.maximum_utilisation_frac <= 1.0

    def damage_per_cycle(self) -> float:
        """Miner damage accumulated in one pumping cycle by the limiting section."""
        worst = self.sections[self.limiting_section_index]
        if not np.isfinite(worst.cycles_to_failure) or worst.cycles_to_failure <= 0.0:
            return 0.0
        return 1.0 / worst.cycles_to_failure


def analyse_stress(solution: WaveSolution, taper: RodTaper, config: SrpConfig) -> StressReport:
    """Modified Goodman check for every taper section.

    The peak and minimum load in each section come from the wave solver, which
    tracks them at every node over the last solved cycle. Dividing by the steel
    area of the section gives the stress state the Goodman diagram needs.
    """
    if solution.node_peak_load_n.size != taper.node_count:
        raise PhysicsDomainError("Wave solution and taper node counts differ.")
    reports: list[SectionStress] = []
    for index, section in enumerate(taper.sections):
        nodes = taper.section_index == index
        if not np.any(nodes):
            continue
        peak_load_n = float(np.max(solution.node_peak_load_n[nodes]))
        min_load_n = float(np.min(solution.node_min_load_n[nodes]))
        area = section.area_m2
        maximum_stress_pa = peak_load_n / area
        minimum_stress_pa = min_load_n / area
        allowable_pa = modified_goodman_allowable_pa(
            minimum_stress_pa,
            section.minimum_tensile_strength_pa,
            config.service_factor_dimensionless,
        )
        utilisation = maximum_stress_pa / allowable_pa if allowable_pa > 0.0 else float("inf")
        amplitude_pa = 0.5 * (maximum_stress_pa - minimum_stress_pa)
        allowable_amplitude_pa = 0.5 * (allowable_pa - minimum_stress_pa)
        life = cycles_to_failure(amplitude_pa, max(allowable_amplitude_pa, 1.0))
        reports.append(
            SectionStress(
                section_index=index,
                diameter_m=section.diameter_m,
                grade=section.grade,
                maximum_stress_pa=maximum_stress_pa,
                minimum_stress_pa=minimum_stress_pa,
                stress_range_pa=maximum_stress_pa - minimum_stress_pa,
                allowable_stress_pa=allowable_pa,
                utilisation_frac=utilisation,
                cycles_to_failure=life,
            )
        )
    if not reports:
        raise PhysicsDomainError("No rod sections were resolved on the wave grid.")
    utilisations = [report.utilisation_frac for report in reports]
    limiting = int(np.argmax(utilisations))
    return StressReport(
        sections=tuple(reports),
        maximum_utilisation_frac=float(max(utilisations)),
        limiting_section_index=limiting,
    )


class FatigueCounter:
    """Miner style cumulative damage counter for one rod string.

    Damage accumulates as the sum of n_i / N_i over operating periods. A value
    of 1.0 means the string has used its nominal fatigue life. Impact events
    from rod float are added separately because a float impact applies a much
    larger stress amplitude than the running cycle does.
    """

    def __init__(self, initial_damage: float = 0.0) -> None:
        if initial_damage < 0.0:
            raise PhysicsDomainError("Initial fatigue damage cannot be negative.")
        self.damage = initial_damage
        self.impact_events = 0

    def accumulate(self, report: StressReport, cycles: float) -> float:
        """Add the damage from ``cycles`` pumping cycles at this stress state."""
        if cycles < 0.0:
            raise PhysicsDomainError("Cycle count cannot be negative.", cycles=cycles)
        increment = report.damage_per_cycle() * cycles
        self.damage += increment
        return increment

    def accumulate_impact(
        self,
        impact_load_n: float,
        report: StressReport,
        taper: RodTaper,
        events: float,
    ) -> float:
        """Add the damage from rod float impacts.

        The impact load is added to the running peak stress of the limiting
        section, which raises the amplitude and therefore the damage per event
        steeply through the Basquin exponent.
        """
        if events <= 0.0 or impact_load_n <= 0.0:
            return 0.0
        worst = report.sections[report.limiting_section_index]
        area = taper.sections[worst.section_index].area_m2
        impact_stress_pa = impact_load_n / area
        amplitude_pa = 0.5 * (worst.stress_range_pa) + impact_stress_pa
        allowable_amplitude_pa = max(
            0.5 * (worst.allowable_stress_pa - worst.minimum_stress_pa), 1.0
        )
        life = cycles_to_failure(amplitude_pa, allowable_amplitude_pa)
        increment = events / life if np.isfinite(life) and life > 0.0 else 0.0
        self.damage += increment
        self.impact_events += int(events)
        return increment

    @property
    def remaining_life_fraction(self) -> float:
        """Fraction of the nominal fatigue life still available."""
        return float(max(1.0 - self.damage, 0.0))


def stress_profile_mpa(
    solution: WaveSolution, taper: RodTaper
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Depth, peak stress and minimum stress along the string, in MPa.

    Used by the dashboard to show which part of the taper is working hardest.
    """
    areas = np.maximum(taper.area_m2, 1.0e-9)
    return (
        taper.depth_m,
        solution.node_peak_load_n / areas / 1.0e6,
        solution.node_min_load_n / areas / 1.0e6,
    )
