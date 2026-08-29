# drto.dynamic_optimization

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a transformation that assembles the dynamic
optimization problem from my declarations, so that I can solve for the
optimal control trajectory over the horizon without hand-writing the
objective or choosing the free and fixed variables myself.

```python
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory

import drto

# ... declared model m (feature 002) ...

pyo.TransformationFactory("dae.collocation").apply_to(
    m, wrt=m.t, nfe=20, ncp=3, scheme="LAGRANGE-RADAU")

pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
# controls free and parameterized by their declared profiles, estimation
# declarations dropped, objective assembled from the live cost terms.
# With both cost kinds declared, the tracking weight is an argument:
# apply_to(m, tracking_weight=10.0)
SolverFactory("ipopt").solve(m)
```

From a model statement rather than a model, the function form calls the
builder, discretizes what it returns, and assembles:

```python
m = drto.dynamic_optimization(
    build,                      # the model statement, a function
    N=50,                       # intervals, passed to the builder.
                                # Omitted, the builder's default
    h=1.0,                      # sampling time, likewise
    ncp=3,                      # collocation points per element
    scheme="LAGRANGE-RADAU",    # the default
    infinite_horizon={"nfe": 3, "ncp": 5},   # False (default) adds no
                                # terminal segment. True applies
                                # drto.infinite_horizon with defaults,
                                # a mapping passes its options
    tracking_weight=10.0,       # as above, both cost kinds declared
)
SolverFactory("ipopt").solve(m)
```

## Benefit hypothesis

The user states the model once and gets the horizon optimization
assembled from it, with no objective or decision-variable set rebuilt by
hand and the problem model-consistent by construction. This is the
headline dynamic-optimization mode (NMPC and D-RTO) that the closed-loop
frameworks run.

## Acceptance criteria

### The builder contract

A builder is a function returning a declared, undiscretized model. Its
first two parameters are the interval count and the sampling time, named
`N` and `h`, and a mode function passes them by keyword. Every parameter
has a default, so the bare `build()` is legal. The mode function owns
discretization, so a builder never discretizes what it returns. The
other mode functions reference this contract rather than restating it.

A builder holds a constant input by fixing it at the declared sample
points, which is all an undiscretized model has, and the mode function
completes it. A fixed status covers only the members present when it is
set, so the members `dae.collocation` adds would otherwise be free.

### The function form

- `drto.dynamic_optimization(build, ...)` is the function form under the
  transformation's own name, the dual form feature 003 sets with
  `build_objective`. It calls `build` with whichever of `N` and `h` were
  given, by keyword, so an omitted one takes the builder's default.
- It discretizes the declared sample grid with `dae.collocation` at one
  finite element per declared interval, taking `ncp` and `scheme` from
  its own arguments. The interval count is read from the horizon record,
  which holds the grid as declared.
- A builder returning a model whose declared set is already discretized
  is a descriptive error naming the contract, since the function owns
  the mesh.
- Before `dae.collocation` runs, it records the members fixed at every
  declared sample point together with their values, grouped by the index
  with the time entry removed. After the call returns, it fixes the
  members added to those groups, each at the value of the sample before
  it. The declared controls and disturbances are left out, since
  `drto.parameterize` and the modes set those, and so is a group fixed
  at only some samples, an initial-condition pin.
- `infinite_horizon=False`, the default, adds no terminal segment.
  `True` applies `drto.infinite_horizon` with its own defaults, and a
  mapping passes its contents as that transformation's options.
- It then applies the registered `drto.dynamic_optimization`
  transformation, passing `tracking_weight` when given, and returns the
  model. The model is assembled and uninitialized, so
  `drto.cold_start_dynamic` and the other initializers run on it
  afterward.

### The registered transformation

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
- The estimation-category declarations (feature 018) are neutralized so the
  control problem keeps only what it uses, and the registry mirrors the
  model: a component removed from the model has its record purged, one left on
  the model keeps its record. The estimation costs (`estimation_stage_cost`,
  `estimation_terminal_cost`, `arrival_cost`) and the `measurement` Params are
  deleted and their records purged, since nothing in a control problem reads
  them and a measurement is reachable only from those costs (`h(z)` is written
  inline in the cost). A `disturbance` is fixed at zero and keeps its record,
  the controller predicting on its own model with the process noise off.
  It is fixed, not eliminated, so the model keeps the structure showing
  where the noise enters, the fixing works however the noise enters the
  equations, additive or not, and the solver treats a fixed Var as a
  constant, so the NLP is identical to substituting zero. An
  `estimated_parameter` is fixed at its current value and keeps its
  record, since it stays a live coefficient in the equations the
  controller solves and that value is the estimate the controller should
  use.
- After the objective is assembled, the transform declares the
  initial-condition Params as pounce sensitivity parameters when
  pyomo-pounce is importable, and the transformation log records how many
  were declared. The declaration is inert metadata that every other
  solver ignores, while a pounce solve keeps the converged factorization
  for the advanced-step correction (feature 012). Without pyomo-pounce
  the transform runs unchanged and the log shows no sensitivity entry.
- The transform keeps the time horizon and does not reduce the model to steady
  state.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
