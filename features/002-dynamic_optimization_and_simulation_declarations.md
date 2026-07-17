# Dynamic optimization and simulation declarations

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want to declare the pieces of my optimization or
simulation problem, either by tagging the components I already built on my
Pyomo model or by wrapping the components as I build them, so that DRTO can
find and assemble them into the horizon problem without my restructuring the
model or writing a separate DRTO model.

Tagging: build the model plainly, declare at the end.

```python
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import drto

m = pyo.ConcreteModel()
m.t = ContinuousSet(initialize=range(11))  # the sample grid, dt = 1
m.z = pyo.Var(m.t)
m.dzdt = DerivativeVar(m.z, wrt=m.t)
m.u = pyo.Var(m.t, bounds=(0, 1))

m.z_ss = pyo.Param(initialize=0.5, mutable=True)   # tracking targets
m.u_ss = pyo.Param(initialize=0.3, mutable=True)
m.z_hat = pyo.Param(initialize=0.4, mutable=True)  # state feedback hook

m.cost = pyo.Var(m.t)

@m.Constraint(m.t)
def ode(m, t):
    return m.dzdt[t] == -m.z[t] + m.u[t]

@m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
def stage(m, t):
    return m.cost[t] == 10*(m.z[t] - m.z_ss)**2 + (m.u[t] - m.u_ss)**2

@m.Constraint()
def init(m):
    return m.z[0] == m.z_hat

drto.horizon(m.t)
drto.state(m.z)
drto.continuous_dynamics(m.ode)
drto.control(m.u, profile="piecewise_constant")
drto.tracking_stage_cost(m.stage)
drto.initial_condition(m.init)
drto.steady_state(m.z, m.z_ss)
drto.steady_state_control(m.u, m.u_ss)
```

Wrapping: the same functions around the construction, declaring as the model
is written, with the constraint-role declarations as decorators.

```python
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from drto import (horizon, state, control, continuous_dynamics,
                  tracking_stage_cost, initial_condition,
                  steady_state, steady_state_control)

m = pyo.ConcreteModel()
m.t = horizon(ContinuousSet(initialize=range(11)))  # the sample grid, dt = 1
m.z = state(pyo.Var(m.t))
m.dzdt = DerivativeVar(m.z, wrt=m.t)
m.u = control(pyo.Var(m.t, bounds=(0, 1)), profile="piecewise_constant")

m.z_ss = steady_state(m.z, pyo.Param(initialize=0.5, mutable=True))
m.u_ss = steady_state_control(m.u, pyo.Param(initialize=0.3, mutable=True))
m.z_hat = pyo.Param(initialize=0.4, mutable=True)  # state feedback hook

m.cost = pyo.Var(m.t)

@continuous_dynamics(m, m.t)
def ode(m, t):
    return m.dzdt[t] == -m.z[t] + m.u[t]

@tracking_stage_cost(m, sorted(m.t)[:-1])  # the terminal cost owns the final time
def stage(m, t):
    return m.cost[t] == 10*(m.z[t] - m.z_ss)**2 + (m.u[t] - m.u_ss)**2

@initial_condition(m)
def init(m):
    return m.z[0] == m.z_hat
```

The two styles mix freely: the same functions serve both, so a partly wrapped
model can be finished by tagging and the reverse.

Later features elide this as `# ... declared model m (feature 002) ...`.

## Benefit hypothesis

Declaring by tagging existing components lets DRTO bolt onto an ordinary Pyomo
model rather than replacing how the user builds one, which keeps the model
reusable across problems and modes. Recording every declaration in the
`drto.info` registry gives the transformations one place to find the declared
components, so `build_objective` and `dynamic_to_steady_state` consume the
declarations rather than re-deriving them.

## Acceptance criteria

- Each declaration function tags an existing Pyomo component on the user's model
  (a Var, Constraint, Param, or Set), validates that the component is of the
  expected type and meets the declaration's convention, and records it in
  `drto.info(m)` (feature 001). An invalid target errors clearly.
- Handed an unconstructed component instead, a declaration function wraps it:
  it returns the component so it can sit in the `m.x = ...` assignment, and
  validation and registration fire when Pyomo attaches it to the model. The
  ordering rules are the same in both styles; a declaration's prerequisites
  must be declared by the time it registers, which writing the model top-down
  satisfies.
- The constraint-role declarations (`continuous_dynamics`, the costs,
  `initial_condition`, `terminal_constraint`) double as decorators taking the
  model plus whatever `@m.Constraint` would take, building, attaching, and
  declaring the constraint in one step.
- Arity: `state`, `control`, `continuous_dynamics`, and
  `initial_condition` accept varargs or an
  indexed container (one declaration per container), since they scale with the
  states and controls. `horizon`, `tracking_stage_cost`,
  `economic_stage_cost`, `tracking_terminal_cost`, and
  `terminal_constraint` each take exactly one object and error on more
  than one. `steady_state` and `steady_state_control` take exactly one
  pair per call (see below) and accumulate across calls.
- Re-declaration: a single-object declaration errors on a second call with a
  different object (for example a second `horizon` on a new Set), since the
  model has one of each. A varargs declaration accumulates across calls, but
  declaring the same component twice is rejected as a duplicate. Both checks run
  against the registry (feature 001).
- `horizon(m.t)` tags the horizon Set, a `pyomo.dae` ContinuousSet,
  initialized with the sample grid (the sampling instants). Declaring it
  captures that grid in the registry: the samples define the stage-cost sum
  (feature 003) and the sampling time `dt`, so `horizon` errors if the
  set holds fewer than two points or is already discretized.
- `state(m.z, ...)` tags one or more state Vars. A state carries a
  `DerivativeVar` for its dynamics only in a dynamic model, so a steady-state
  model's states need not have one and `state` does not require it.
- `control(m.u, ..., profile=...)` tags one or more manipulated-input
  Vars and sets their parameterization (piecewise-constant, ...) over the
  declared time set via pyomo-cvp. The `profile` applies to the controls named in
  that call. A control that needs a different parameterization is declared in a
  separate call.
- `continuous_dynamics(m.ode, ...)` tags one or more equality
  Constraints whose left-hand sides are the DerivativeVars of declared states.
- `tracking_stage_cost(m.con)` and `economic_stage_cost(m.con)`
  each tag a per-time-point equality Constraint whose left-hand side is the
  scalar running-cost variable; the right-hand side defines the cost. The
  stage cost does not apply at the final time point, where only the terminal
  cost applies, so it is indexed over the sample grid minus the final point
  (for example `sorted(m.t)[:-1]`), one member per sample: a member at the
  final time, or a missing sample, is rejected. Indexing by a plain list also
  keeps the discretization from expanding it beyond the samples.
- `tracking_terminal_cost(m.con)` tags an equality Constraint whose
  left-hand side is the scalar terminal-cost variable.
- `initial_condition(m.con, ...)` tags one or more equality Constraints
  whose left-hand sides are declared states at the first time point and whose
  right-hand sides are mutable Params, the feedback hook.
- `terminal_constraint(m.con)` tags a single Constraint that references
  only states at the final time point.
- `steady_state(m.z, m.z_ss)` pairs a declared state with the mutable Param
  holding its setpoint; `steady_state_control(m.u, m.u_ss)` pairs a declared
  control with its setpoint Param. One pair per call, accumulating across
  calls; a state or control that already has a target is rejected as a
  duplicate, and the first argument must already be declared. The pairing is
  recorded in the registry so the steady-state/RTO solve (feature 009) knows
  which target Param each solved state or control value populates. The
  function returns the target Param.
- The scalar-left-hand-side conventions (the cost and initial-condition
  constraints) are read from the constraint body's canonical form, so they hold
  regardless of how the user wrote the equality.
- Path constraints are not declared; they are the state variables' own bounds.
- The estimation declarations (measurements, disturbances, arrival cost, and the
  estimation costs) are out of scope here and are specced with the estimation
  follow-on.
