"""Surface facilities: tank heating and bowser logistics.

At Baghewala the produced crude is stored in a tank at each well site, heated
with steam and hot water from a mobile steam generator, and then moved by road
tanker to the central tank farm. That heating is real energy use, so leaving it
out would understate the energy per barrel and would let the optimizer take
credit for savings that only moved the cost somewhere else. The model here is
deliberately light: it answers how much heat the tank needs to keep the crude
pumpable, and how often a tanker has to come.

Source for the field practice: Oil India, Rajasthan Fields page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.config import FieldConfig
from app.core.errors import PhysicsDomainError
from app.core.numerics import require_positive
from app.core.units import J_PER_KWH, SECONDS_PER_DAY
from app.twin.fluid import FluidModel


@dataclass(frozen=True)
class TankHeatingResult:
    """Daily heat demand and logistics for one well-site tank."""

    sensible_heat_j_per_day: float
    standing_loss_j_per_day: float
    total_heat_j_per_day: float
    generator_hours_per_day: float
    fuel_energy_j_per_day: float
    fuel_cost_usd_per_day: float
    bowser_loads_per_day: float
    target_temp_c: float
    required_temp_for_pumpable_c: float

    @property
    def total_heat_kwh_per_day(self) -> float:
        """Tank heating demand expressed in kWh."""
        return self.total_heat_j_per_day / J_PER_KWH

    def as_dict(self) -> dict[str, float]:
        """Serialisable form for the API and the reports."""
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


class SurfaceModel:
    """Tank heating energy and tanker logistics for one well site."""

    def __init__(self, config: FieldConfig, fluid: FluidModel | None = None) -> None:
        self.config = config
        self.fluid = fluid or FluidModel(config.fluid)

    def temperature_for_pumpable_crude_c(
        self, pumpable_viscosity_pa_s: float = 0.5, water_cut_frac: float = 0.0
    ) -> float:
        """Temperature at which the stored crude reaches a pumpable viscosity.

        Equation: inverts the ASTM D341 viscosity model by bisection.
        Units: degrees Celsius. Viscosity in Pa.s.
        Assumption: 0.5 Pa.s, that is 500 cP, is taken as the practical limit for
        a road tanker loading pump. This is an operating convention, not a
        measurement, and is recorded in ``docs/ASSUMPTIONS.md``.
        """
        return self.fluid.temperature_for_viscosity_c(pumpable_viscosity_pa_s, water_cut_frac)

    def tank_heating_demand(
        self,
        oil_rate_m3_per_day: float,
        water_rate_m3_per_day: float,
        wellhead_temp_c: float,
        target_temp_c: float | None = None,
    ) -> TankHeatingResult:
        """Heat needed per day to keep the tank contents pumpable.

        Equations:
            sensible heat  Q_s = m_dot c_p (T_target - T_wellhead)
            standing loss  Q_l = U A (T_target - T_ambient) * 86400
        Units: J/day. Rates in m3/day, temperatures in degrees Celsius.
        Assumptions: the tank is held at the target temperature, incoming fluid
        arrives at the wellhead temperature, the tank is well mixed, and the
        heat loss coefficient covers the whole wetted and dry surface. Fluid
        that arrives hotter than the target needs no sensible heat, which is
        exactly the case a well with vacuum insulated tubing can reach; that is
        part of why VIT shows up in the energy KPI and not only in the physics.
        """
        surface = self.config.surface
        target_c = target_temp_c if target_temp_c is not None else surface.tank_target_temp_c
        total_rate = max(oil_rate_m3_per_day + water_rate_m3_per_day, 0.0)
        water_cut = water_rate_m3_per_day / total_rate if total_rate > 0.0 else 0.0
        density = float(self.fluid.mixture_density_kg_per_m3(target_c, water_cut))
        specific_heat = float(self.fluid.mixture_specific_heat_j_per_kg_k(water_cut))

        temperature_rise_k = max(target_c - wellhead_temp_c, 0.0)
        sensible_j_per_day = total_rate * density * specific_heat * temperature_rise_k
        standing_loss_j_per_day = (
            surface.tank_heat_loss_coefficient_w_per_m2_k
            * surface.tank_surface_area_m2
            * max(target_c - surface.tank_ambient_temp_c, 0.0)
            * SECONDS_PER_DAY
        )
        total_j_per_day = sensible_j_per_day + standing_loss_j_per_day

        generator_output_w = (
            surface.steam_generator_thermal_power_w * surface.steam_generator_efficiency_frac
        )
        generator_hours = total_j_per_day / max(generator_output_w, 1.0) / 3600.0
        fuel_j_per_day = total_j_per_day / max(surface.steam_generator_efficiency_frac, 1.0e-3)
        fuel_cost = fuel_j_per_day / 1.0e9 * self.config.economics.fuel_cost_usd_per_gj

        return TankHeatingResult(
            sensible_heat_j_per_day=sensible_j_per_day,
            standing_loss_j_per_day=standing_loss_j_per_day,
            total_heat_j_per_day=total_j_per_day,
            generator_hours_per_day=generator_hours,
            fuel_energy_j_per_day=fuel_j_per_day,
            fuel_cost_usd_per_day=fuel_cost,
            bowser_loads_per_day=total_rate / max(surface.bowser_capacity_m3, 1.0e-6),
            target_temp_c=target_c,
            required_temp_for_pumpable_c=self.temperature_for_pumpable_crude_c(
                water_cut_frac=water_cut
            ),
        )

    def steam_generation_fuel_energy_j(
        self, steam_volume_m3_cwe: float, injection_pressure_kpa: float, quality_frac: float
    ) -> float:
        """Fuel energy needed to raise a CSS steam slug at the wellhead.

        Equation: E_fuel = m (h_f + x h_fg - c_w T_feed) / eta_generator.
        Units: J. Volume in m3 cold water equivalent.
        Assumption: the feed water arrives at the ambient tank temperature.
        Source: standard once-through steam generator energy balance.
        """
        from app.core.steam import steam_enthalpy_j_per_kg

        require_positive(steam_volume_m3_cwe, "steam_volume_m3_cwe")
        surface = self.config.surface
        mass_kg = steam_volume_m3_cwe * self.config.fluid.water_density_kg_per_m3
        feed_enthalpy_j_per_kg = (
            self.config.fluid.water_specific_heat_j_per_kg_k * surface.tank_ambient_temp_c
        )
        steam_enthalpy = steam_enthalpy_j_per_kg(
            min(max(injection_pressure_kpa, 120.0), 19500.0), quality_frac
        )
        useful_j = mass_kg * max(steam_enthalpy - feed_enthalpy_j_per_kg, 0.0)
        return useful_j / max(surface.steam_generator_efficiency_frac, 1.0e-3)

    def steam_generator_days_required(
        self, steam_volume_m3_cwe: float, injection_rate_m3_per_day_cwe: float
    ) -> float:
        """Days one mobile steam generator is occupied by an injection, plus the move.

        Used by the fleet scheduler, which has to share a small number of mobile
        units across many wells.
        """
        if injection_rate_m3_per_day_cwe <= 0.0:
            raise PhysicsDomainError(
                "Injection rate must be positive.",
                injection_rate_m3_per_day_cwe=injection_rate_m3_per_day_cwe,
            )
        injection_days = steam_volume_m3_cwe / injection_rate_m3_per_day_cwe
        return injection_days + self.config.surface.rig_move_days

    def generator_rate_limit_m3_per_day_cwe(
        self, injection_pressure_kpa: float, quality_frac: float
    ) -> float:
        """Largest injection rate one mobile steam generator can sustain.

        Equation: rate = eta P / (rho (h_f + x h_fg - c_w T_feed)).
        Units: m3/day cold water equivalent.
        This is a hard constraint on the CSS optimizer: asking for a rate the
        generator cannot raise is not a plan, it is a wish.
        """
        fuel_j_per_m3 = self.steam_generation_fuel_energy_j(
            1.0, injection_pressure_kpa, quality_frac
        )
        available_j_per_day = self.config.surface.steam_generator_thermal_power_w * SECONDS_PER_DAY
        return available_j_per_day / max(fuel_j_per_m3, 1.0)

    def tank_fill_days(self, liquid_rate_m3_per_day: float) -> float:
        """Days to fill the site tank at the current rate."""
        if liquid_rate_m3_per_day <= 0.0:
            return math.inf
        return self.config.surface.tank_volume_m3 / liquid_rate_m3_per_day
