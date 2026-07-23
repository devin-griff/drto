# drto.dynamic_optimization

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want a transformation that assembles the dynamic
optimization problem from my declarations, so that I can solve for the optimal
control trajectory over the horizon without hand-wiring the objective and the
free and fixed variables.

```python
import pyomo.environ as pyo
import drto

# ... declared model m (feature 002) ...

pyo.TransformationFactory("dae.collocation").apply_to(
    m, wrt=m.t, nfe=20, ncp=3, scheme="LAGRANGE-RADAU")

pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
# controls free and parameterized by their declared profiles, estimation
# declarations dropped, objective assembled from the live cost terms.
# With both cost kinds declared, the tracking weight is an argument:
# apply_to(m, tracking_weight=10.0)
pyo.SolverFactory("ipopt").solve(m)
```

## Benefit hypothesis

Assembling the horizon optimization from the same declarations that describe
the model keeps the problem model-consistent and frees the user from rebuilding
the objective and the decision-variable set by hand. This is the headline
dynamic-optimization mode (NMPC and D-RTO) that the closed-loop frameworks run.

## Acceptance criteria

- `TransformationFactory('drto.dynamic_optimization')` requires `horizon`,
  `state`, `dynamics`, `control`, `initial_condition`, and at
  least one of `tracking_stage_cost` or `economic_stage_cost`,
  and errors clearly if any is missing.
- It targets continuous dynamics. Discrete-time (difference-equation)
  optimization is a separate topic, out of scope for this transform.
- The declared controls are the free decision variables, parameterized over the
  time set by their declared profile (pyomo-cvp).
- The objective is assembled by `drto.build_objective` (feature 003) from the
  live cost terms over the horizon. When both a tracking and an economic stage
  cost are declared, both are summed into the objective, with a weight applied
  to the tracking stage cost. The transform accepts that weight as an argument,
  used only when both are present, and it defaults to 1. The economic stage
  cost is in currency units and is never scaled, so there is no economic-side
  weight.
- Because the transform assembles the objective as its final step, any
  transform that registers additional cost terms must be applied before it. In
  particular `drto.infinite_horizon` (feature 004), which appends the tail
  cost, must run before `drto.dynamic_optimization`, otherwise the tail never
  enters the objective.
- `tracking_terminal_cost`, `terminal_constraint`, and the
  steady-state targets (`steady_state`, `steady_state_control`)
  are optional. The transform uses them if declared.
- Any estimation-category declarations on the model (measurements,
  disturbances, arrival cost, estimation stage and terminal costs, estimated
  parameters) are dropped.
- The transform keeps the time horizon and does not reduce the model to steady
  state.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
