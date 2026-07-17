# Dynamic optimization and simulation declarations

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

## Description

As a user of DRTO, I want to declare the pieces of my optimization or
simulation problem by
tagging the components I already built on my Pyomo model, so that DRTO can find
and assemble them into the horizon problem without my restructuring the model
or writing a separate DRTO model.

```python
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import drto

m = pyo.ConcreteModel()
m.t = ContinuousSet(bounds=(0, 10))
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

@m.Constraint(m.t)
def stage(m, t):
    return m.cost[t] == 10*(m.z[t] - m.z_ss)**2 + (m.u[t] - m.u_ss)**2

@m.Constraint()
def init(m):
    return m.z[0] == m.z_hat

drto.declare_time(m.t)
drto.declare_state(m.z)
drto.declare_continuous_dynamics(m.ode)
drto.declare_control(m.u, profile="piecewise_constant")
drto.declare_tracking_stage_cost(m.stage)
drto.declare_initial_condition(m.init)
drto.declare_steady_state(m.z_ss)
drto.declare_steady_state_control(m.u_ss)
```

Later features elide this as `# ... declared model m (feature 002) ...`.

## Benefit hypothesis

Declaring by tagging existing components lets DRTO bolt onto an ordinary Pyomo
model rather than replacing how the user builds one, which keeps the model
reusable across problems and modes. Recording every declaration in the
`drto.info` registry gives the transformations one place to find the declared
components, so `build_objective` and `dynamic_to_steady_state` consume the
declarations rather than re-deriving them.

## Acceptance criteria

- Each `declare_*` function tags an existing Pyomo component on the user's model
  (a Var, Constraint, Param, or Set), validates that the component is of the
  expected type and meets the declaration's convention, and records it in
  `drto.info(m)` (feature 001). An invalid target errors clearly.
- Arity: `declare_state`, `declare_control`, `declare_continuous_dynamics`,
  `declare_initial_condition`, `declare_steady_state`, and
  `declare_steady_state_control` accept varargs or an
  indexed container (one declaration per container), since they scale with the
  states and controls. `declare_time`, `declare_tracking_stage_cost`,
  `declare_economic_stage_cost`, `declare_tracking_terminal_cost`, and
  `declare_terminal_constraint` each take exactly one object and error on more
  than one.
- Re-declaration: a single-object declaration errors on a second call with a
  different object (for example a second `declare_time` on a new Set), since the
  model has one of each. A varargs declaration accumulates across calls, but
  declaring the same component twice is rejected as a duplicate. Both checks run
  against the registry (feature 001).
- `declare_time(m.t)` tags the horizon Set, a `pyomo.dae` ContinuousSet.
- `declare_state(m.z, ...)` tags one or more state Vars. A state carries a
  `DerivativeVar` for its dynamics only in a dynamic model, so a steady-state
  model's states need not have one and `declare_state` does not require it.
- `declare_control(m.u, ..., profile=...)` tags one or more manipulated-input
  Vars and sets their parameterization (piecewise-constant, ...) over the
  declared time set via pyomo-cvp. The `profile` applies to the controls named in
  that call. A control that needs a different parameterization is declared in a
  separate call.
- `declare_continuous_dynamics(m.ode, ...)` tags one or more equality
  Constraints whose left-hand sides are the DerivativeVars of declared states.
- `declare_tracking_stage_cost(m.con)` and `declare_economic_stage_cost(m.con)`
  each tag a per-time-point equality Constraint whose left-hand side is the
  scalar running-cost variable; the right-hand side defines the cost.
- `declare_tracking_terminal_cost(m.con)` tags an equality Constraint whose
  left-hand side is the scalar terminal-cost variable.
- `declare_initial_condition(m.con, ...)` tags one or more equality Constraints
  whose left-hand sides are declared states at the first time point and whose
  right-hand sides are mutable Params, the feedback hook.
- `declare_terminal_constraint(m.con)` tags a single Constraint that references
  only states at the final time point.
- `declare_steady_state(m.z_ss, ...)` and
  `declare_steady_state_control(m.u_ss, ...)` each tag one or more Params holding
  the state or control setpoints the tracking costs drive toward.
- The scalar-left-hand-side conventions (the cost and initial-condition
  constraints) are read from the constraint body's canonical form, so they hold
  regardless of how the user wrote the equality.
- Path constraints are not declared; they are the state variables' own bounds.
- The estimation declarations (measurements, disturbances, arrival cost, and the
  estimation costs) are out of scope here and are specced with the estimation
  follow-on.
