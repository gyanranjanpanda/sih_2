"""Dynamometer cards: construction, feature extraction and a rule-based classifier.

A card is the closed loop of load against position over one pumping cycle.
The surface card is what the polished rod transducer measures. The pump card is
what the plunger sees, obtained from the surface card by the Gibbs inverse
solution in :mod:`app.twin.srp.wave`. Diagnosis is done on the pump card
because its shape maps directly onto pump behaviour.

The rule-based classifier here is the published baseline that the learned
classifier in :mod:`app.ml.card_classifier` has to beat. Keeping it in the twin
means a diagnosis is always available even when no model has been trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from app.core.errors import PhysicsDomainError
from app.core.numerics import require_finite

NORMALISED_CARD_POINTS = 64
"""Length of the resampled card fed to the learned classifier."""


class CardClass(StrEnum):
    """Diagnostic classes used by both the rule baseline and the learned model."""

    NORMAL = "normal"
    FLUID_POUND = "fluid_pound"
    GAS_INTERFERENCE = "gas_interference"
    TAGGING = "tagging"
    WORN_PUMP = "worn_pump"
    STICKING = "sticking"
    ROD_FLOAT = "rod_float"
    UNSEATING = "unseating"
    PARTED_ROD = "parted_rod"
    UNKNOWN = "unknown"


CARD_CLASS_ORDER: tuple[CardClass, ...] = (
    CardClass.NORMAL,
    CardClass.FLUID_POUND,
    CardClass.GAS_INTERFERENCE,
    CardClass.TAGGING,
    CardClass.WORN_PUMP,
    CardClass.STICKING,
    CardClass.ROD_FLOAT,
    CardClass.UNSEATING,
    CardClass.PARTED_ROD,
)
"""Fixed class order, so model outputs and confusion matrices stay comparable."""


@dataclass(frozen=True)
class DynamometerCard:
    """One closed load-position loop.

    Attributes:
        position_m: Position over one cycle, measured up from the bottom of the
            stroke. Polished rod position for a surface card, plunger position
            for a pump card.
        load_n: Load at each sample.
        is_surface: True for a surface card, False for a pump card.
        cycle_time_s: Duration of the cycle.
    """

    position_m: NDArray[np.float64]
    load_n: NDArray[np.float64]
    is_surface: bool
    cycle_time_s: float

    def __post_init__(self) -> None:
        if self.position_m.size != self.load_n.size:
            raise PhysicsDomainError("Card position and load arrays differ in length.")
        if self.position_m.size < 8:
            raise PhysicsDomainError("A card needs at least eight samples.")
        require_finite(self.position_m, "card position")
        require_finite(self.load_n, "card load")

    @property
    def stroke_m(self) -> float:
        """Total travel covered by the card."""
        return float(np.max(self.position_m) - np.min(self.position_m))

    def work_per_cycle_j(self) -> float:
        """Net work done per cycle, the signed area enclosed by the loop.

        Equation: W = the contour integral of F dx, evaluated with the shoelace
        formula on the closed polygon.
        Units: J. A positive value means net work is put into the fluid.
        """
        position = np.append(self.position_m, self.position_m[0])
        load = np.append(self.load_n, self.load_n[0])
        area = 0.5 * np.sum((position[:-1] - position[1:]) * (load[:-1] + load[1:]))
        return float(abs(area))

    def normalised(self, points: int = NORMALISED_CARD_POINTS) -> NDArray[np.float64]:
        """Resample the card to a fixed length and scale both axes to [0, 1].

        This is the representation the learned classifier consumes. Scaling
        removes the well-specific load and stroke magnitudes so the classifier
        learns shape rather than size.
        """
        index = np.linspace(0.0, 1.0, self.position_m.size)
        target = np.linspace(0.0, 1.0, points)
        position = np.interp(target, index, self.position_m)
        load = np.interp(target, index, self.load_n)
        position_span = max(float(np.ptp(position)), 1.0e-9)
        load_span = max(float(np.ptp(load)), 1.0e-9)
        normalised_position = (position - float(np.min(position))) / position_span
        normalised_load = (load - float(np.min(load))) / load_span
        return np.stack([normalised_position, normalised_load]).astype(float)


@dataclass(frozen=True)
class CardFeatures:
    """Scalar features extracted from a card, used for diagnosis and for models."""

    peak_load_n: float
    minimum_load_n: float
    load_range_n: float
    mean_load_n: float
    fluid_load_n: float
    stroke_m: float
    net_stroke_m: float
    fillage_frac: float
    work_per_cycle_j: float
    upstroke_load_slope_n_per_m: float
    downstroke_load_slope_n_per_m: float
    downstroke_load_drop_position_frac: float
    load_drop_sharpness: float
    negative_load_fraction: float
    low_load_fraction: float
    bottom_of_stroke_undershoot_n: float

    def as_dict(self) -> dict[str, float]:
        """Flat dictionary, for the feature table and the API."""
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def extract_features(card: DynamometerCard, low_load_threshold_n: float = 1000.0) -> CardFeatures:
    """Compute the diagnostic features of a card.

    The fluid load is taken as the difference between the median load over the
    upper half of the upstroke and the median load over the lower half of the
    downstroke, which is robust to the dynamic overshoot at the stroke ends.
    Fillage is the fraction of the downstroke completed before the load drops
    through the midpoint between those two levels.
    """
    position = card.position_m
    load = card.load_n
    stroke_m = card.stroke_m
    if stroke_m <= 0.0:
        raise PhysicsDomainError("Card has zero stroke, cannot extract features.")

    top_index = int(np.argmax(position))
    order = np.arange(position.size)
    on_upstroke = np.zeros(position.size, dtype=bool)
    on_upstroke[: top_index + 1] = True
    # A card may start part way through a stroke; use the position derivative
    # as the authoritative direction indicator.
    direction = np.gradient(np.concatenate([position, position[:1]]))[: position.size]
    on_upstroke = direction >= 0.0
    on_downstroke = ~on_upstroke

    relative_position = (position - float(np.min(position))) / stroke_m
    upper_upstroke = on_upstroke & (relative_position > 0.5)
    # The lower window stops short of the very bottom so that a tagging
    # excursion does not contaminate the estimate of the released load level.
    lower_downstroke = on_downstroke & (relative_position < 0.5) & (relative_position > 0.15)
    if not np.any(lower_downstroke):
        lower_downstroke = on_downstroke & (relative_position < 0.5)
    upper_load = (
        float(np.median(load[upper_upstroke])) if np.any(upper_upstroke) else float(np.max(load))
    )
    lower_load = (
        float(np.median(load[lower_downstroke]))
        if np.any(lower_downstroke)
        else float(np.min(load))
    )
    fluid_load_n = max(upper_load - lower_load, 0.0)
    midpoint_load = 0.5 * (upper_load + lower_load)

    # Fillage: how far down the downstroke the load crosses the midpoint.
    downstroke_indices = order[on_downstroke]
    fillage_frac = 1.0
    drop_position_frac = 0.0
    sharpness = 0.0
    if downstroke_indices.size > 4 and fluid_load_n > 0.0:
        downstroke_positions = relative_position[downstroke_indices]
        downstroke_loads = load[downstroke_indices]
        sort_order = np.argsort(-downstroke_positions)
        travelled = 1.0 - downstroke_positions[sort_order]
        loads_along = downstroke_loads[sort_order]
        below = np.nonzero(loads_along < midpoint_load)[0]
        if below.size > 0:
            drop_position_frac = float(travelled[below[0]])
            fillage_frac = float(min(max(1.0 - drop_position_frac, 0.0), 1.0))
            window = slice(max(below[0] - 3, 0), min(below[0] + 4, loads_along.size))
            travel_window = float(np.ptp(travelled[window]))
            if travel_window > 0.0:
                sharpness = float(
                    abs(float(np.ptp(loads_along[window]))) / (travel_window * fluid_load_n)
                )

    def slope(mask: NDArray[np.bool_]) -> float:
        if np.count_nonzero(mask) < 3:
            return 0.0
        return float(np.polyfit(position[mask], load[mask], 1)[0])

    # Tagging is the plunger striking the bottom of the barrel. A healthy pump
    # card sits at about zero load once the travelling valve is open, so a
    # substantial compressive excursion is the tagging signature. Measuring it
    # against zero rather than against a position window keeps the feature
    # robust when the contact region is long enough to shift the baseline.
    undershoot_n = max(-float(np.min(load)), 0.0)

    return CardFeatures(
        peak_load_n=float(np.max(load)),
        minimum_load_n=float(np.min(load)),
        load_range_n=float(np.ptp(load)),
        mean_load_n=float(np.mean(load)),
        fluid_load_n=fluid_load_n,
        stroke_m=stroke_m,
        net_stroke_m=stroke_m * fillage_frac,
        fillage_frac=fillage_frac,
        work_per_cycle_j=card.work_per_cycle_j(),
        upstroke_load_slope_n_per_m=slope(on_upstroke),
        downstroke_load_slope_n_per_m=slope(on_downstroke),
        downstroke_load_drop_position_frac=drop_position_frac,
        load_drop_sharpness=sharpness,
        negative_load_fraction=float(np.mean(load < 0.0)),
        low_load_fraction=float(np.mean(load < low_load_threshold_n)),
        bottom_of_stroke_undershoot_n=undershoot_n,
    )


@dataclass(frozen=True)
class Diagnosis:
    """A card diagnosis with the reason stated in plain words."""

    card_class: CardClass
    confidence: float
    reason: str
    features: CardFeatures

    def as_dict(self) -> dict[str, object]:
        """Serialisable form for the API."""
        return {
            "card_class": str(self.card_class),
            "confidence": self.confidence,
            "reason": self.reason,
            "features": self.features.as_dict(),
        }


def classify_by_rules(
    features: CardFeatures,
    expected_fluid_load_n: float,
    minimum_fillage_frac: float = 0.70,
    minimum_load_threshold_n: float = 1000.0,
    surface_float_index: float = 0.0,
) -> Diagnosis:
    """Rule-based pump card diagnosis.

    This is the published baseline. The rules follow standard rod pump
    surveillance practice: an underfilled pump drops its load partway down the
    downstroke, gas drops it gradually, a worn pump or a leaking valve loses
    fluid load altogether, tagging shows a load spike at the bottom, sticking
    shows an elevated upstroke load, and a parted rod leaves a nearly flat card
    at rod weight.

    Args:
        features: Features extracted from a pump card.
        expected_fluid_load_n: Fluid load the twin predicts for a healthy pump.
        minimum_fillage_frac: Fillage below which the pump is called underfilled.
        minimum_load_threshold_n: Load below which the rods are considered to be
            carrying nothing, taken from published rod float control practice.
        surface_float_index: Fraction of the cycle on which the polished rod
            load is below that threshold, from :func:`app.twin.srp.floating.analyse_float`.
            Rod float has to be judged on the surface card, because a healthy
            pump card is near zero load for the whole downstroke by design; only
            the surface card can show that the rods themselves are unloaded.

    Returns:
        The diagnosis, with a confidence between 0 and 1 and a written reason.
    """
    load_ratio = (
        features.fluid_load_n / expected_fluid_load_n if expected_fluid_load_n > 0.0 else 0.0
    )

    if features.load_range_n < 0.06 * max(expected_fluid_load_n, 1.0):
        return Diagnosis(
            CardClass.PARTED_ROD,
            0.75,
            "The card is flat: the load range is under 6 percent of the expected fluid "
            "load, which is what a parted rod or a completely unloaded string looks like.",
            features,
        )
    if load_ratio < 0.25:
        return Diagnosis(
            CardClass.UNSEATING,
            0.70,
            f"Fluid load has collapsed to {load_ratio * 100:.0f} percent of the expected "
            "value while the card still moves, which points to the pump unseating or the "
            "standing valve failing.",
            features,
        )
    if surface_float_index > 0.02:
        return Diagnosis(
            CardClass.ROD_FLOAT,
            min(0.55 + 2.0 * surface_float_index, 0.95),
            f"Polished rod load falls below {minimum_load_threshold_n:.0f} N over "
            f"{surface_float_index * 100:.0f} percent of the cycle. The rods cannot fall "
            "as fast as the polished rod, so the string is floating.",
            features,
        )
    if features.bottom_of_stroke_undershoot_n > 0.4 * expected_fluid_load_n:
        return Diagnosis(
            CardClass.TAGGING,
            0.75,
            "The load drives well below the downstroke baseline at the bottom of the "
            "plunger stroke, which is the plunger tagging the bottom of the barrel.",
            features,
        )
    if load_ratio > 1.25:
        return Diagnosis(
            CardClass.STICKING,
            0.65,
            f"Fluid load is {load_ratio * 100:.0f} percent of the expected value with full "
            "fillage, which suggests the plunger is sticking in the barrel.",
            features,
        )
    if features.fillage_frac < minimum_fillage_frac:
        if features.load_drop_sharpness > 6.0:
            return Diagnosis(
                CardClass.FLUID_POUND,
                0.85,
                f"Fillage is {features.fillage_frac * 100:.0f} percent and the load drops "
                "abruptly partway down the downstroke. That is fluid pound: the plunger "
                "falls through a void and then strikes liquid.",
                features,
            )
        return Diagnosis(
            CardClass.GAS_INTERFERENCE,
            0.75,
            f"Fillage is {features.fillage_frac * 100:.0f} percent and the load falls "
            "gradually rather than abruptly, which is gas compressing in the barrel "
            "rather than a liquid void.",
            features,
        )
    if load_ratio < 0.70:
        return Diagnosis(
            CardClass.WORN_PUMP,
            0.70,
            f"Fluid load is {load_ratio * 100:.0f} percent of expected with full fillage, "
            "which indicates slippage past a worn plunger or a leaking valve.",
            features,
        )
    if 0.80 <= load_ratio <= 1.25:
        return Diagnosis(
            CardClass.NORMAL,
            0.85,
            f"Fillage is {features.fillage_frac * 100:.0f} percent and fluid load is within "
            "25 percent of the expected value. The pump is working normally.",
            features,
        )
    return Diagnosis(
        CardClass.UNKNOWN,
        0.30,
        "The card does not match any rule cleanly. Treat the diagnosis as unknown "
        "and check the raw data before acting.",
        features,
    )
