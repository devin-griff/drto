# Dynamic optimization and simulation declarations

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want to declare the pieces of my optimization or
simulation problem, either by tagging the components I already built on my
Pyomo model or by wrapping the components as I build them, so that DRTO can
find and assemble them into the horizon problem without my restructuring the
model or writing a separate DRTO model.

Tagging declares each component right after it is built. The tags work
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
m.z_hat = pyo.Param(initialize=0.4, mutable=True)  # overwritten by a loop's measurements

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

Wrapping applies the same functions around the construction, declaring
as the model is written, with the constraint-role declarations as
decorators.

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
m.z_hat = pyo.Param(initialize=0.4, mutable=True)  # overwritten by a loop's measurements

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

The constraint-role declarations also wrap a fresh Constraint directly.
A detached Constraint can only be built with `rule=`, so the decorator
form above is the usual construction style, and the wrap form composes
the same way.

```python
def ode_rule(m, t):
    return m.dzdt[t] == -m.z[t] + m.u[t]

m.ode = drto.dynamics(pyo.Constraint(m.t, rule=ode_rule))
```

The two styles mix freely. The same functions serve both, so a partly
wrapped model can be finished by tagging and the reverse.

Later features elide this as `# ... declared model m (feature 002) ...`.

## Benefit hypothesis

Enforcing each declaration's conventions at the moment it is made is
what allows the model to be consumed automatically by everything
downstream, the transformations and the simulation loops alike, with no
per-model wiring. Declaring by tagging existing components lets DRTO
attach to an ordinary Pyomo model rather than replacing how the user
builds one, which keeps the model reusable across problems and modes.
Recording every declaration in the `drto.info` registry gives the
transformations one place to find the declared components, so
`build_objective` and `dynamic_to_steady_state` consume the declarations
rather than re-deriving them.

## Acceptance criteria

- Each declaration function tags an existing Pyomo component on the user's model
  (a Var, Constraint, Param, or Set), validates that the component is of the
  expected type and meets the declaration's convention, and records it in
  `drto.info(m)` (feature 001). An invalid target errors clearly.
- Handed an unconstructed component instead, a declaration function wraps it:
  it returns the component so it can sit in the `m.x = ...` assignment, and
  validation and registration fire when Pyomo attaches it to the model. The
  wrap form takes exactly one component per call, since it returns it
  for a single assignment, and varargs are a tagging-only convenience. In
  both styles the argument is always the component being declared,
  attached or fresh. drto never constructs a component, so an index set
  where a component belongs (for example `state(m.t)`) is a type error,
  not an implicit construction. The ordering rules are the same in both
  styles. A declaration's prerequisites must be declared by the time it
  registers, which writing the model top-down satisfies.
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
  captures that grid in the registry. The samples define the stage-cost
  sum (feature 003) and the sampling time `dt`, so `horizon` errors if
  the set is already discretized (Pyomo itself enforces the two-point
  minimum at construction).
- States may be indexed by time alone or by time plus other sets (a tray
  composition x(tray, comp, t)). The validations that reference a
  specific time point (the initial condition's t0, the terminal
  constraint's tN) find the time coordinate inside the member's index.
- `state(m.z, ...)` tags one or more state Vars. A state has a
  `DerivativeVar` only in a dynamic model, so a steady-state model's
  states need not have one and `state` does not require it. A
  member-subset slice of an indexed Var (`holdup[:, "Liq", "NaOH"]`)
  declares the true state members when the container also holds
  algebraic entries, an IDAES holdup with water left out. The slice is
  wrapped as a Reference attached beside the sliced Var, the Reference
  is the declared state, and the container's other members stay
  undeclared (gh #20).
- `control(m.u, ..., profile=...)` tags one or more manipulated-input
  Vars and sets their parameterization (piecewise-constant, ...) over the
  declared time set via pyomo-cvp. The `profile` applies to the controls named in
  that call. A control that needs a different parameterization is declared in a
  separate call. With a horizon declared, each control must be indexed by
  the declared time set, checked at the declaration. On a model with no
  declared horizon (a steady-state model), the control registers without a
  profile: there is no time to parameterize over, and the steady modes fix
  the control instead.
- `dynamics(m.ode, ...)` tags one or more equality Constraints with the
  DerivativeVar of a declared state on one side, bare or multiplied by
  derivative-free factors: an IDAES `ControlVolume1D` writes
  `length * accumulation`, a variable-volume balance writes `V * dc/dt`,
  and both are that state's differential equation. A side differentiated
  along the declared time set is preferred over one differentiated along
  a spatial axis, so a 1D balance with the space derivative on its other
  side is read correctly in either orientation. The transforms consume
  the same shape. The steady reduction pins the derivative through the
  coefficient, and the terminal segment copies the coefficient into the
  dilated tail dynamics.
- `tracking_stage_cost(m.con)` and `economic_stage_cost(m.con)`
  each tag a per-time-point equality Constraint whose left-hand side is
  the scalar running-cost variable and whose right-hand side defines the
  cost. The
  stage cost does not apply at the final time point, where only the
  terminal cost applies, so it is indexed over the sample grid minus the
  final point (for example `sorted(m.t)[:-1]`), one member per sample. A
  member at the final time, or a missing sample, is rejected. The family
  may not be indexed by the time set itself, even with Skip members,
  since discretization expands such a family to every collocation point,
  placing cost members off the sample grid, so declaration rejects the
  index outright. On a model with no declared horizon (a steady-state
  model) there is no grid to index over, so a scalar Constraint registers
  instead and an indexed one is rejected. That is the same accommodation
  `control` makes, and the single-point shape the steady-state reduction
  produces from a per-sample cost.
- `move_suppression(m.con)` tags one per-sample equality Constraint
  penalizing the control moves. Each member's defining side references
  only declared control members at that sample and the one before, plus
  Params (the weights, and the previous action the first member's move
  is measured from). The family is indexed like the stage costs and
  requires a declared horizon, since a steady-state model has no moves. `build_objective`
  sums it with the stage costs, the steady-state reduction drops it (the
  moves are zero at any steady point), and the terminal segment keeps it
  on the finite horizon and out of the tail integrand.
- `tracking_terminal_cost(m.con)` tags an equality Constraint whose
  left-hand side is the scalar terminal-cost variable. Its defining side
  covers every declared state and contains only declared states (and
  Params). The stage cost covers and contains the states and controls
  the same way. A missing or foreign component is a descriptive error naming
  it.
- `initial_condition(m.con, ...)` tags one or more equality Constraints
  whose left-hand sides are declared states at the first time point and whose
  right-hand sides are mutable Params, which a loop overwrites with
  each measurement.
- `terminal_constraint(m.con)` tags a single Constraint that references
  only states at the final time point.
- `steady_state(m.z, m.z_ss)` pairs a declared state with the mutable
  Param holding its setpoint, and `steady_state_control(m.u, m.u_ss)`
  pairs a declared control with its setpoint Param. The owner may be
  the same member-subset slice that declared the state, which resolves
  to the declared Reference by member identity. Each call takes one
  pair and calls accumulate. Re-declaring the same pair is idempotent,
  a different target for a state or control that already has one is
  rejected, a target Param cannot be paired with two owners (in either
  target kind), and the first argument must already be declared. The
  pairing is recorded in the registry as the only record of which target
  Param goes with which state or control, which is what lets a
  steady-state/RTO solution be written back into the targets later. That
  write-back is an algorithmic step, not part of a mode transform. The
  function returns the target Param.
- The scalar-side conventions (the cost and initial-condition
  constraints) are read from the written equality's sides, either
  orientation, so `lhs == rhs` and `rhs == lhs` are equivalent. The
  constraint must be written as an explicit equality.
- The tracking costs cover the declared states and controls exactly
  (gh #60). The stage cost's defining side references at least one member
  of every declared state and every declared control, and nothing that is
  not a declared state or control member (targets and scales are Params).
  The terminal cost does the same over the states alone. Violations are
  descriptive errors at declaration, which is why the states and controls
  precede the costs in the declaration order.
- Leaving cost variables unbounded is a documented convention, not a
  rule the declarations enforce. Nothing reads a cost variable's bounds.
  The defining equality fixes the value, so a `NonNegativeReals` bound
  adds no information, and it places the optimum exactly on the bound
  wherever the cost vanishes, at settled samples on a long horizon or
  through a tail at equilibrium. Interior-point solvers take many more
  iterations there (Hicks at N = 50: 43 iterations with the bound
  against 6 without, identical solutions).
- Path constraints are not declared. They are the state variables' own
  bounds.
- The estimation declarations (measurements, disturbances, arrival cost, and the
  estimation costs) are out of scope here and are specced with the estimation
  follow-on.
