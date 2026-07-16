# drto.steady_state_simulation

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

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
- The objective is zero: the transform calls `drto.build_objective` (feature
  003) with the option for a simulation, which installs a constant-zero
  `Objective` and gives an NLP solver a well-posed square problem for the
  fixed-control equilibrium.
- Solving the transformed model gives an equilibrium that satisfies the dynamics
  at rest and the model's algebraic relations.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
