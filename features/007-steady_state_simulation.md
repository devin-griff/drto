# drto.steady_state_simulation

**Status:** draft

## Description

As a user of DRTO, I want a transformation that reduces my model to steady
state with the controls fixed and solves for the equilibrium, so that I can
find the resting operating point from the one model.

## Benefit hypothesis

Deriving the equilibrium from the same declarations makes the resting state
model-consistent by construction, and it composes the steady-state reduction
rather than duplicating it.

## Acceptance criteria

- `TransformationFactory('drto.steady_state_simulation')` requires
  `declare_time`, `declare_state`, and `declare_continuous_dynamics`, and errors
  clearly if any is missing.
- It composes `drto.dynamic_to_steady_state` (feature 004) to collapse the model
  to a single equilibrium point, with the declared controls fixed.
- Solving the transformed model gives an equilibrium that satisfies the dynamics
  at rest and the model's algebraic relations.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
