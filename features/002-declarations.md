# Simulation and optimization declarations

**Status:** draft

## Description

As a user of DRTO, I want to declare the pieces of my optimization or
simulation problem by
tagging the components I already built on my Pyomo model, so that DRTO can find
and assemble them into the horizon problem without my restructuring the model
or writing a separate DRTO model.

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
- `declare_time(m.t)` tags the horizon Set. It accepts a `pyomo.dae`
  ContinuousSet or a discrete Set; DRTO does not assume continuity.
- `declare_state(m.z, ...)` tags one or more differential-state Vars. It accepts
  varargs and indexed containers, one declaration per container.
- `declare_control(m.u, ..., profile=...)` tags a manipulated-input Var and sets
  its parameterization (piecewise-constant, ...) over the declared time set via
  pyomo-cvp.
- `declare_continuous_dynamics(m.ode)` tags an equality Constraint whose
  left-hand side is the DerivativeVar of a declared state.
- `declare_discrete_dynamics(m.diff)` tags an equality Constraint whose
  left-hand side is a declared state at the next time point. Continuous versus
  discrete dynamics are told apart by the left-hand-side component type
  (DerivativeVar versus plain Var).
- `declare_tracking_stage_cost` and `declare_economic_stage_cost` each tag a
  per-time-point equality Constraint whose left-hand side is the scalar
  running-cost variable; the right-hand side defines the cost.
- `declare_tracking_terminal_cost(m.con)` tags an equality Constraint whose
  left-hand side is the scalar terminal-cost variable.
- `declare_initial_condition(m.con)` tags an equality Constraint whose left-hand
  side is a declared state at the first time point and whose right-hand side is
  a mutable Param, the feedback hook.
- `declare_terminal_constraint(m.con)` tags a Constraint that references only
  states at the final time point.
- `declare_steady_state(m.z_ss)` and `declare_steady_state_control(m.u_ss)` each
  tag a Param holding the state or control setpoint the tracking costs drive
  toward.
- The scalar-left-hand-side conventions (the cost and initial-condition
  constraints) are read from the constraint body's canonical form, so they hold
  regardless of how the user wrote the equality.
- Path constraints are not declared; they are the state variables' own bounds.
- The estimation declarations (measurements, disturbances, arrival cost, and the
  estimation costs) are out of scope here and are specced with the estimation
  follow-on.
