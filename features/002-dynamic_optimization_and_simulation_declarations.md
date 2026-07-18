# Dynamic optimization and simulation declarations

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want to declare the pieces of my optimization or
simulation problem, either by tagging the components I already built on my
Pyomo model or by wrapping the components as I build them, so that DRTO can
find and assemble them into the horizon problem without my restructuring the
model or writing a separate DRTO model.

Tagging: declare each component right after it is built. The tags work
anywhere after the component exists, so a finished model can equally be
declared in one block at the end.

```python
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import drto

m = pyo.ConcreteModel()
N, h = 10, 1  # samples and sampling time
m.t = ContinuousSet(initialize=pyo.RangeSet(0, N*h, h))  # the sample grid
drto.horizon(m.t)

m.z = pyo.Var(m.t)
drto.state(m.z)
m.dzdt = DerivativeVar(m.z, wrt=m.t)

m.u = pyo.Var(m.t, bounds=(0, 1))
drto.control(m.u, profile="piecewise_constant")

m.z_ss = pyo.Param(initialize=0.5, mutable=True)   # tracking targets
drto.steady_state(m.z, m.z_ss)
m.u_ss = pyo.Param(initialize=0.5, mutable=True)  # = z_ss: dz/dt = 0 needs z = u
drto.steady_state_control(m.u, m.u_ss)
m.z_hat = pyo.Param(initialize=0.4, mutable=True)  # state feedback hook

m.cost = pyo.Var(m.t)

@m.Constraint(m.t)
def ode(m, t):
    return m.dzdt[t] == -m.z[t] + m.u[t]
drto.dynamics(m.ode)

@m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
def stage(m, t):
    return m.cost[t] == 10*(m.z[t] - m.z_ss)**2 + (m.u[t] - m.u_ss)**2
drto.tracking_stage_cost(m.stage)

@m.Constraint()
def init(m):
    return m.z[0] == m.z_hat
drto.initial_condition(m.init)
```

Wrapping: the same functions around the construction, declaring as the model
is written, with the constraint-role declarations as decorators.

```python
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import drto

m = pyo.ConcreteModel()
N, h = 10, 1  # samples and sampling time
m.t = drto.horizon(ContinuousSet(initialize=pyo.RangeSet(0, N*h, h)))
m.z = drto.state(pyo.Var(m.t))
m.dzdt = DerivativeVar(m.z, wrt=m.t)
m.u = drto.control(pyo.Var(m.t, bounds=(0, 1)), profile="piecewise_constant")

m.z_ss = drto.steady_state(m.z, pyo.Param(initialize=0.5, mutable=True))
m.u_ss = drto.steady_state_control(m.u, pyo.Param(initialize=0.5, mutable=True))
m.z_hat = pyo.Param(initialize=0.4, mutable=True)  # state feedback hook

m.cost = pyo.Var(m.t)

@drto.dynamics(m, m.t)
def ode(m, t):
    return m.dzdt[t] == -m.z[t] + m.u[t]

@drto.tracking_stage_cost(m, sorted(m.t)[:-1])  # the terminal cost owns the final time
def stage(m, t):
    return m.cost[t] == 10*(m.z[t] - m.z_ss)**2 + (m.u[t] - m.u_ss)**2

@drto.initial_condition(m)
def init(m):
    return m.z[0] == m.z_hat
```

The constraint-role declarations also wrap a fresh Constraint directly. A
detached Constraint can only be built with `rule=`, so the decorator form
above is the usual construction style; the wrap form composes the same way.

```python
def ode_rule(m, t):
    return m.dzdt[t] == -m.z[t] + m.u[t]

m.ode = drto.dynamics(pyo.Constraint(m.t, rule=ode_rule))
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
  wrap form takes exactly one component per call, since it returns it for a
  single assignment; varargs are a tagging-only convenience. In both styles
  the argument is always the component being declared, attached or fresh:
  drto never constructs a component, so an index set where a component
  belongs (for example `state(m.t)`) is a type error, not an implicit
  construction. The
  ordering rules are the same in both styles; a declaration's prerequisites
  must be declared by the time it registers, which writing the model top-down
  satisfies.
- The constraint-role declarations (`dynamics`, the costs,
  `initial_condition`, `terminal_constraint`) double as decorators taking the
  model plus whatever `@m.Constraint` would take, building, attaching, and
  declaring the constraint in one step.
- The styles mix per component: the same functions serve tagging, wrapping,
  and the decorators, so one model may declare some components one way and
  some another (for example decorators for the constraints and tags for the
  Vars).
- Arity: `state`, `control`, `dynamics`, and
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
  set is already discretized (Pyomo itself enforces the two-point minimum
  at construction).
- States may be indexed by time alone or by time plus other sets (a tray
  composition x(tray, comp, t)); the validations that reference a specific
  time point (the initial condition's t0, the terminal constraint's tN) find
  the time coordinate inside the member's index.
- `state(m.z, ...)` tags one or more state Vars. A state carries a
  `DerivativeVar` for its dynamics only in a dynamic model, so a steady-state
  model's states need not have one and `state` does not require it.
- `control(m.u, ..., profile=...)` tags one or more manipulated-input
  Vars and sets their parameterization (piecewise-constant, ...) over the
  declared time set via pyomo-cvp. The `profile` applies to the controls named in
  that call. A control that needs a different parameterization is declared in a
  separate call.
- `dynamics(m.ode, ...)` tags one or more equality
  Constraints whose left-hand sides are the DerivativeVars of declared states.
- `tracking_stage_cost(m.con)` and `economic_stage_cost(m.con)`
  each tag a per-time-point equality Constraint whose left-hand side is the
  scalar running-cost variable; the right-hand side defines the cost. The
  stage cost does not apply at the final time point, where only the terminal
  cost applies, so it is indexed over the sample grid minus the final point
  (for example `sorted(m.t)[:-1]`), one member per sample: a member at the
  final time, or a missing sample, is rejected. The family may not be indexed
  by the time set itself, even with Skip members: discretization expands such
  a family to every collocation point, dragging the cost off the sample grid,
  so declaration rejects the index outright.
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
  calls; re-declaring the same pair is idempotent, a different target for
  a state or control that already has one is rejected, a target Param
  cannot serve two owners (in either target kind), and the first argument
  must already be declared. The pairing is
  recorded in the registry so the steady-state/RTO solve (feature 009) knows
  which target Param each solved state or control value populates. The
  function returns the target Param.
- The scalar-side conventions (the cost and initial-condition constraints)
  are read from the written equality's sides, either orientation, so
  `lhs == rhs` and `rhs == lhs` are equivalent; the constraint must be
  written as an explicit equality.
- Path constraints are not declared; they are the state variables' own bounds.
- The estimation declarations (measurements, disturbances, arrival cost, and the
  estimation costs) are out of scope here and are specced with the estimation
  follow-on.
