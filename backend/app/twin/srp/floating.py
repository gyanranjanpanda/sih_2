"""Viscous drag on the rod string and the rod float criterion.

Rod float is the central failure mode at Baghewala. On the downstroke the rods
fall under their own buoyant weight. If the viscous drag of the produced heavy
crude exceeds that weight, the rods cannot keep up with the polished rod, the
string goes into compression, and when the polished rod catches up again the
re-engagement is an impact. Repeated impacts part rods and unseat pumps.

The drag model is the exact solution for axial flow in a concentric annulus
with a moving inner wall and a prescribed net throughput, which is the Couette
plus Poiseuille combination the brief asks for. The viscosity used at each
depth comes from the wellbore temperature profile, so a well with vacuum
insulated tubing has materially lower drag than a bare-tubing well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import SrpConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import require_finite, require_positive
from app.core.units import STANDARD_GRAVITY_M_PER_S2
from app.twin.srp.kinematics import StrokeMotion
from app.twin.srp.wave import DragProfile, RodTaper


@dataclass(frozen=True)
class AnnulusFlowSolution:
    """Coefficients of the annular flow solution for one geometry and viscosity."""

    pressure_gradient_pa_per_m: float
    shear_stress_on_rod_pa: float
    drag_per_length_n_per_m: float


def annular_drag_per_length_n_per_m(
    rod_radius_m: float,
    tubing_radius_m: float,
    rod_velocity_m_per_s: float,
    net_flow_m3_per_s: float,
    viscosity_pa_s: float,
) -> AnnulusFlowSolution:
    """Axial drag on a rod moving inside a tubing that also carries net flow.

    Equations: for steady laminar axial flow in a concentric annulus,

        mu (1/r) d/dr (r du/dr) = P,     u(a) = V,  u(b) = 0
        u(r) = (P / 4mu) r^2 + C1 ln(r) + C2

    with a the rod radius, b the tubing inner radius, V the rod velocity and
    P the dynamic pressure gradient. P is determined by requiring that the
    integral of u over the annulus equals the net production throughput, which
    is the pressure-driven return flow the brief asks for. The drag per unit
    length on the rod is then 2 pi a mu du/dr evaluated at r = a.

    Units: radii in m, velocity in m/s (positive up), flow in m3/s (positive
    up), viscosity in Pa.s. Returns drag in N/m, positive upward.

    Assumptions: concentric rods, fully developed laminar flow, Newtonian
    fluid, no rod couplings. Couplings and sinker bars are added separately by
    :func:`coupling_drag_n`. The concentric assumption understates drag in a
    deviated well, which is stated in ``docs/ASSUMPTIONS.md``.

    Source: Bird, Stewart and Lightfoot, Transport Phenomena, section 2.4
    (annular flow); the moving-wall extension is the standard superposition.
    """
    require_positive(viscosity_pa_s, "viscosity_pa_s")
    if not 0.0 < rod_radius_m < tubing_radius_m:
        raise PhysicsDomainError(
            "Rod radius must be positive and smaller than the tubing radius.",
            rod_radius_m=rod_radius_m,
            tubing_radius_m=tubing_radius_m,
        )
    a, b = rod_radius_m, tubing_radius_m
    log_ratio = math.log(a / b)

    def flow_for(pressure_gradient: float, velocity: float) -> float:
        """Volumetric throughput of the annulus for a given gradient and wall speed."""
        quarter = pressure_gradient / (4.0 * viscosity_pa_s)
        c1 = (velocity - quarter * (a**2 - b**2)) / log_ratio
        c2 = -quarter * b**2 - c1 * math.log(b)
        term_quadratic = quarter * 0.5 * math.pi * (b**4 - a**4)
        term_log = (
            math.pi * c1 * (b**2 * math.log(b) - 0.5 * b**2 - a**2 * math.log(a) + 0.5 * a**2)
        )
        term_constant = math.pi * c2 * (b**2 - a**2)
        return term_quadratic + term_log + term_constant

    flow_from_velocity = flow_for(0.0, rod_velocity_m_per_s)
    flow_from_unit_gradient = flow_for(1.0, 0.0)
    if abs(flow_from_unit_gradient) < 1.0e-30:
        raise PhysicsDomainError("Degenerate annulus geometry in the drag solution.")
    pressure_gradient = (net_flow_m3_per_s - flow_from_velocity) / flow_from_unit_gradient

    quarter = pressure_gradient / (4.0 * viscosity_pa_s)
    c1 = (rod_velocity_m_per_s - quarter * (a**2 - b**2)) / log_ratio
    velocity_gradient_at_rod = 2.0 * quarter * a + c1 / a
    shear_stress = viscosity_pa_s * velocity_gradient_at_rod
    drag_per_length = 2.0 * math.pi * a * shear_stress
    return AnnulusFlowSolution(
        pressure_gradient_pa_per_m=float(pressure_gradient),
        shear_stress_on_rod_pa=float(shear_stress),
        drag_per_length_n_per_m=float(drag_per_length),
    )


def coupling_drag_n(
    coupling_count: int,
    coupling_diameter_m: float,
    coupling_length_m: float,
    tubing_radius_m: float,
    rod_velocity_m_per_s: float,
    viscosity_pa_s: float,
) -> float:
    """Extra drag from rod couplings, treated as short sleeves in the same annulus.

    Couplings are larger than the rod body, so the gap is smaller and the local
    shear rate is higher. The same Couette solution is applied over the coupling
    length with the coupling radius, with zero net throughput assigned to the
    coupling segment because it is short.
    Units: N, positive upward.
    Assumption: couplings are concentric and do not choke the annulus. This term
    is a correction of order 5 to 15 percent, not a dominant effect.
    """
    if coupling_count <= 0 or coupling_length_m <= 0.0:
        return 0.0
    radius = 0.5 * coupling_diameter_m
    if radius >= tubing_radius_m:
        raise PhysicsDomainError(
            "Coupling diameter must be smaller than the tubing inner diameter.",
            coupling_diameter_m=coupling_diameter_m,
        )
    gap = tubing_radius_m - radius
    shear_stress = viscosity_pa_s * rod_velocity_m_per_s / gap
    area = 2.0 * math.pi * radius * coupling_length_m * coupling_count
    return float(-shear_stress * area)


@dataclass(frozen=True)
class FloatAnalysis:
    """Rod float assessment for one operating point.

    Attributes:
        float_margin_index: (W_buoyant - F_drag) / W_buoyant for the whole
            string at the peak downstroke velocity. Zero or below means the rods
            cannot fall as fast as the polished rod.
        section_margin_index: The same index evaluated for each taper section
            from the tension carried at the top of that section.
        minimum_section_margin: The worst section margin.
        float_index: Fraction of the downstroke on which the card-based
            criterion says the string is unloaded, between 0 and 1.
        downstroke_fraction_affected: Same quantity expressed against the
            downstroke only, for operator reporting.
        estimated_impact_load_n: Load spike expected when the polished rod and
            the floating rods re-engage.
        buoyant_weight_n: Buoyant weight of the string.
        peak_drag_n: Total viscous drag on the string at the peak downstroke
            velocity.
        binding: Short plain-words statement of what is limiting.
    """

    float_margin_index: float
    section_margin_index: tuple[float, ...]
    minimum_section_margin: float
    float_index: float
    downstroke_fraction_affected: float
    estimated_impact_load_n: float
    buoyant_weight_n: float
    peak_drag_n: float
    binding: str

    @property
    def is_floating(self) -> bool:
        """Whether either the mechanical or the card criterion says the rods float."""
        return self.minimum_section_margin <= 0.0 or self.float_index > 0.02


def string_drag_n(
    taper: RodTaper,
    tubing_inner_diameter_m: float,
    rod_velocity_m_per_s: float,
    net_flow_m3_per_s: float,
    viscosity_profile_pa_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Viscous drag carried by the part of the string strictly below each node.

    This is exactly what the tension balance at that node needs, and it is built
    the same way as the buoyant weight below each node so the two line up term
    by term. The last entry is therefore zero: nothing hangs below the pump.
    Units: N, positive upward, so it resists a downward moving string.
    """
    if viscosity_profile_pa_s.size != taper.node_count:
        raise PhysicsDomainError(
            "Viscosity profile must have one value per rod node.",
            expected=taper.node_count,
            received=int(viscosity_profile_pa_s.size),
        )
    tubing_radius_m = 0.5 * tubing_inner_diameter_m
    drag_density = np.zeros(taper.node_count)
    for index in range(taper.node_count):
        rod_radius_m = math.sqrt(taper.area_m2[index] / math.pi)
        solution = annular_drag_per_length_n_per_m(
            rod_radius_m=rod_radius_m,
            tubing_radius_m=tubing_radius_m,
            rod_velocity_m_per_s=rod_velocity_m_per_s,
            net_flow_m3_per_s=net_flow_m3_per_s,
            viscosity_pa_s=float(viscosity_profile_pa_s[index]),
        )
        drag_density[index] = solution.drag_per_length_n_per_m
    segment_drag_n = drag_density * taper.step_m
    drag_below_n = np.concatenate([np.cumsum(segment_drag_n[::-1])[::-1][1:], [0.0]])
    return require_finite(drag_below_n, "string drag")


def analyse_float(
    taper: RodTaper,
    config: SrpConfig,
    motion: StrokeMotion,
    viscosity_profile_pa_s: NDArray[np.float64],
    fluid_density_kg_per_m3: float,
    tubing_inner_diameter_m: float,
    liquid_rate_m3_per_day: float,
    surface_load_n: NDArray[np.float64] | None = None,
    plunger_downstroke_resistance_n: float = 0.0,
) -> FloatAnalysis:
    """Assess rod float from both the mechanical balance and the surface card.

    Mechanical criterion. At the peak downstroke velocity, the tension at each
    node must stay positive:

        T(x) = W_buoyant(below x) - F_drag(below x) - F_plunger_resistance

    and the float margin index at that node is T(x) divided by the buoyant
    weight below it. A margin of zero means the rods are exactly on the point of
    floating; the configured operating minimum is above zero to leave room for
    parameter error.

    Card criterion. Published rod float control methods declare float when the
    polished rod load falls below a small threshold, a few hundred pounds or
    about 1 kN. The float index is the fraction of the cycle spent below that
    threshold.

    Impact load. When the polished rod catches the floating rods, the relative
    velocity is converted into load through the characteristic impedance of the
    string, F = rho a A dv. This is the standard one-dimensional impact
    relation for a bar and gives the order of magnitude of the shock the rods
    take.

    Source: US patent 7547196 and US 10094371 for the card-based float
    criterion and the speed control response; Gibbs, Rod Pumping, chapter 6 for
    the impedance relation.
    """
    peak_down_speed = motion.peak_downstroke_speed_m_per_s
    net_flow_m3_per_s = liquid_rate_m3_per_day / 86400.0

    drag_below_n = string_drag_n(
        taper=taper,
        tubing_inner_diameter_m=tubing_inner_diameter_m,
        rod_velocity_m_per_s=-peak_down_speed,
        net_flow_m3_per_s=net_flow_m3_per_s,
        viscosity_profile_pa_s=viscosity_profile_pa_s,
    )
    effective_density = max(config.steel_density_kg_per_m3 - fluid_density_kg_per_m3, 0.0)
    segment_weight_n = effective_density * STANDARD_GRAVITY_M_PER_S2 * taper.area_m2 * taper.step_m
    weight_below_n = np.concatenate([np.cumsum(segment_weight_n[::-1])[::-1][1:], [0.0]])

    tension_n = weight_below_n - drag_below_n - plunger_downstroke_resistance_n
    with np.errstate(divide="ignore", invalid="ignore"):
        node_margin = np.where(
            weight_below_n > 1.0, tension_n / np.maximum(weight_below_n, 1.0), 1.0
        )
    total_weight_n = float(weight_below_n[0])
    total_drag_n = float(drag_below_n[0])
    whole_string_margin = (
        (total_weight_n - total_drag_n - plunger_downstroke_resistance_n) / total_weight_n
        if total_weight_n > 0.0
        else 0.0
    )

    section_margins: list[float] = []
    cursor_m = 0.0
    for section in taper.sections:
        node = int(np.argmin(np.abs(taper.depth_m - cursor_m)))
        section_margins.append(float(node_margin[node]))
        cursor_m += section.length_m
    minimum_section_margin = float(np.min(node_margin[:-1])) if node_margin.size > 1 else 0.0

    float_index = 0.0
    downstroke_fraction = 0.0
    impact_load_n = 0.0
    if surface_load_n is not None and surface_load_n.size == motion.position_m.size:
        threshold = config.minimum_polished_rod_load_n
        below = surface_load_n < threshold
        float_index = float(np.mean(below))
        downstroke = motion.downstroke_mask
        if np.any(downstroke):
            downstroke_fraction = float(np.mean(below[downstroke]))
        if float_index > 0.0:
            # Relative velocity at re-engagement, taken as the polished rod
            # speed during the floating part of the stroke.
            relative_speed = float(np.max(np.abs(motion.velocity_m_per_s[below])))
            impedance = (
                config.steel_density_kg_per_m3
                * config.acoustic_velocity_m_per_s
                * float(taper.area_m2[0])
            )
            impact_load_n = impedance * relative_speed

    if minimum_section_margin <= 0.0:
        binding = (
            "Viscous drag on the downstroke exceeds the buoyant rod weight, so the "
            "string cannot fall with the polished rod."
        )
    elif minimum_section_margin < config.float_margin_minimum_frac:
        binding = (
            f"Float margin is {minimum_section_margin:.2f}, below the operating minimum of "
            f"{config.float_margin_minimum_frac:.2f}. Slow the downstroke or raise the "
            "temperature at the pump."
        )
    else:
        binding = f"Float margin is {minimum_section_margin:.2f}, inside the safe envelope."

    return FloatAnalysis(
        float_margin_index=float(whole_string_margin),
        section_margin_index=tuple(section_margins),
        minimum_section_margin=minimum_section_margin,
        float_index=float_index,
        downstroke_fraction_affected=downstroke_fraction,
        estimated_impact_load_n=float(impact_load_n),
        buoyant_weight_n=total_weight_n,
        peak_drag_n=total_drag_n,
        binding=binding,
    )


def build_drag_profile(
    taper: RodTaper,
    tubing_inner_diameter_m: float,
    viscosity_profile_pa_s: NDArray[np.float64],
    liquid_rate_m3_per_day: float,
    reference_speed_m_per_s: float = 1.0,
) -> DragProfile:
    """Linearise the annular drag at every rod node for the wave solver.

    The annular solution is exactly linear in the rod velocity at a fixed net
    throughput, so two evaluations per node recover the slope and the intercept
    without any approximation.

    Returns:
        A :class:`DragProfile` whose linear coefficient is positive, so that it
        acts as damping in the rod momentum equation.
    """
    if viscosity_profile_pa_s.size != taper.node_count:
        raise PhysicsDomainError(
            "Viscosity profile must have one value per rod node.",
            expected=taper.node_count,
            received=int(viscosity_profile_pa_s.size),
        )
    tubing_radius_m = 0.5 * tubing_inner_diameter_m
    net_flow_m3_per_s = liquid_rate_m3_per_day / 86400.0
    linear = np.zeros(taper.node_count)
    static = np.zeros(taper.node_count)
    for index in range(taper.node_count):
        rod_radius_m = math.sqrt(taper.area_m2[index] / math.pi)
        viscosity = float(viscosity_profile_pa_s[index])
        at_rest = annular_drag_per_length_n_per_m(
            rod_radius_m, tubing_radius_m, 0.0, net_flow_m3_per_s, viscosity
        ).drag_per_length_n_per_m
        moving = annular_drag_per_length_n_per_m(
            rod_radius_m,
            tubing_radius_m,
            reference_speed_m_per_s,
            net_flow_m3_per_s,
            viscosity,
        ).drag_per_length_n_per_m
        slope = (moving - at_rest) / reference_speed_m_per_s
        linear[index] = max(-slope, 0.0)
        static[index] = at_rest
    require_finite(linear, "drag linear coefficient")
    require_finite(static, "drag static force")
    return DragProfile(linear_coefficient_n_s_per_m2=linear, static_force_n_per_m=static)
