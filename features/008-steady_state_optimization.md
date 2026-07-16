# drto.steady_state_optimization

**Status:** draft

## Description

As a user of DRTO, I want a transformation that reduces my model to steady
state and optimizes the economic objective over the free controls, so that I
get the optimal steady operating point (economic RTO) from the one model.

## Benefit hypothesis

The economic RTO point derived this way is a true equilibrium of the dynamics,
so the setpoint the NMPC tracks is model-consistent rather than a hand-typed
pair. This is what makes the D-RTO name literal.

## Acceptance criteria

- `TransformationFactory('drto.steady_state_optimization')` requires
  `declare_time`, `declare_state`, `declare_continuous_dynamics`,
  `declare_control`, and an economic stage cost, and errors clearly if any is
  missing.
- It composes `drto.dynamic_to_steady_state` (feature 004); the declared
  controls are free.
- The objective is the single-point economic cost, assembled via
  `drto.build_objective` (feature 003).
- Solving the transformed model gives the optimal steady operating point.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
