# drto.dynamic_simulation

**Status:** draft

## Description

As a user of DRTO, I want a transformation that fixes my controls and prepares
the dynamic model to be solved forward over the horizon, so that I can integrate
the model as declared without writing a separate simulation.

## Benefit hypothesis

Reusing the one declared model to simulate keeps simulation and optimization
consistent, and it is the building block the cold-start initializer and
validation runs rely on.

## Acceptance criteria

- `TransformationFactory('drto.dynamic_simulation')` requires `declare_time`,
  `declare_state`, a dynamics declaration, and `declare_control`, and errors
  clearly if any is missing.
- The declared controls are fixed; the mode frees nothing and solves the model
  as declared over the horizon.
- The transform keeps the time horizon.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
