# drto.dynamic_optimization

**Status:** draft

## Description

As a user of DRTO, I want a transformation that assembles the dynamic
optimization problem from my declarations, so that I can solve for the optimal
control trajectory over the horizon without hand-wiring the objective and the
free and fixed variables.

## Benefit hypothesis

Assembling the horizon optimization from the same declarations that describe
the model keeps the problem model-consistent and frees the user from rebuilding
the objective and the decision-variable set by hand. This is the headline
dynamic-optimization mode (NMPC and D-RTO) that the closed-loop frameworks run.

## Acceptance criteria

- `TransformationFactory('drto.dynamic_optimization')` requires `declare_time`,
  `declare_state`, `declare_continuous_dynamics`, `declare_control`, and at
  least one stage cost, and errors clearly if any is missing.
- It targets continuous dynamics. Discrete-time (difference-equation)
  optimization is a separate topic, out of scope for this transform.
- The declared controls are the free decision variables, parameterized over the
  time set by their declared profile (pyomo-cvp).
- The objective is assembled by `drto.build_objective` (feature 003) from the
  live cost terms over the horizon.
- The transform keeps the time horizon; it does not reduce the model to steady
  state.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
