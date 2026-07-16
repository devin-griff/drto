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
  `declare_state`, and errors clearly if it is missing. `declare_time` and
  `declare_continuous_dynamics` are optional, since drto cannot tell whether the
  user built a dynamic model or a steady-state one.
- If the model is dynamic (time and continuous dynamics declared), it reduces to
  a single equilibrium point by composing `drto.dynamic_to_steady_state`
  (feature 004). If the model is already steady-state, that step is skipped.
  Either way the declared controls are fixed.
- A control option sets the values the fixed controls take: supplied control
  values, or with nothing supplied, the values the control variables are already
  initialized to on the model. The steady state is a single point, so the
  supplied form is values, not a profile.
- The objective is zero: the transform calls `drto.build_objective` (feature
  003) with the option for a simulation, which installs a constant-zero
  `Objective` and gives an NLP solver a well-posed square problem for the
  fixed-control equilibrium.
- Solving the transformed model gives an equilibrium that satisfies the dynamics
  at rest and the model's algebraic relations.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
