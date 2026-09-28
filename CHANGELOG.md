# Changelog

## M3 Rod string and pump

- API Spec 11E four-bar kinematics for the conventional unit, with the crank radius solved from
  the selected stroke length. Hydraulic unit kinematics with an arbitrary velocity profile,
  separate up and down speeds and end-of-stroke dwell.
- Intra-stroke speed shaping, including deceleration into the top of the downstroke.
- Finite-volume damped wave solver on the tapered string, with Courant enforcement, a hysteresis
  valve state machine and a finite valve transfer distance derived from fluid compressibility.
  Validated against the exact damped analytic solution to 3 percent in magnitude.
- Gibbs frequency-domain inverse solution, cross-validated against the forward solver.
- Annular Couette plus pressure-driven drag, node-wise drag linearisation feeding the wave
  solver, float margin index per section and from the surface card, and an impact load estimate.
- Modified Goodman utilisation, Basquin fatigue life and a Miner damage counter.
- Pump displacement, slippage, fillage and gas interference. Drive power, gearbox torque, motor
  efficiency map, VFD mapping and energy per barrel.
- Dynamometer card features and a rule-based classifier that recovers all nine injected faults.

## M2 Fluid, reservoir and wellbore

- ASTM D341 Walther fluid model with a selectable Arrhenius form, emulsion uplift with an
  inversion guard, density and specific heat against temperature.
- Marx and Langenheim heated area, evaluated with erfcx so it stays accurate at large times.
- Boberg and Lantz style lumped heat balance using exact vertical and radial conduction unit
  solutions, with produced-fluid enthalpy removed explicitly.
- Composite radial inflow with a transient radius of investigation, Corey relative permeability,
  material balance pressure decline and cycle-to-cycle decline.
- Ramey wellbore heat transmission in both directions, saturated steam quality loss down the
  tubing, vacuum insulated versus bare tubing, hydrostatic and friction pressure.

## M1 Foundations

- Repository skeleton, pinned dependencies, Makefile with `make verify`.
- SI unit layer with an explicit conversion registry, including kg/cm2 for Indian field practice.
- Pydantic v2 configuration with cross-field validation, readable errors and a provenance block
  that a test enforces over every configured value.
- Structured logging, numerics guards that fail loudly on non-finite values, IAPWS-IF97 steam
  properties with a built-in fallback table.
