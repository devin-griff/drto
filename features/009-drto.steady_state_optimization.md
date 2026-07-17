# drto.steady_state_optimization

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want a transformation that reduces my model to steady
state and optimizes the economic objective over the free controls, so that I
get the optimal steady operating point (economic RTO) from the one model.

## Benefit hypothesis

The economic RTO point derived this way is a true equilibrium of the dynamics,
so the setpoint the NMPC tracks is model-consistent rather than a hand-typed
pair. This is what makes the D-RTO name literal. Because the reduction is
optional, the mode runs on a model reduced from dynamic or one the user wrote
directly as steady-state, so the same declaration surface serves both.

## Acceptance criteria

- `TransformationFactory('drto.steady_state_optimization')` requires
  `declare_state`, `declare_control`, and an economic stage cost, and errors
  clearly if any is missing. `declare_time` and `declare_continuous_dynamics`
  are optional, since the user may define either a steady-state or dynamic model
  initially.
- If the model is dynamic (time and continuous dynamics declared), it reduces to
  a single point by composing `drto.dynamic_to_steady_state` (feature 005). If
  the model is already steady-state, that step is skipped. The declared controls
  are free.
- The objective is the single-point economic cost, assembled via
  `drto.build_objective` (feature 003).
- Solving the transformed model gives the optimal steady operating point.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
