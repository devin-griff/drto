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
- Arity: `declare_state`, `declare_control`, `declare_continuous_dynamics`,
  `declare_discrete_dynamics`, `declare_initial_condition`,
  `declare_steady_state`, and `declare_steady_state_control` accept varargs or an
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
- `declare_time(m.t)` tags the horizon Set. It accepts a `pyomo.dae`
  ContinuousSet or a discrete Set; DRTO does not assume continuity.
- `declare_state(m.z, ...)` tags one or more differential-state Vars.
- `declare_control(m.u, ..., profile=...)` tags one or more manipulated-input
  Vars and sets their parameterization (piecewise-constant, ...) over the
  declared time set via pyomo-cvp. The `profile` applies to the controls named in
  that call. A control that needs a different parameterization is declared in a
  separate call.
- `declare_continuous_dynamics(m.ode, ...)` tags one or more equality
  Constraints whose left-hand sides are the DerivativeVars of declared states.
- `declare_discrete_dynamics(m.diff, ...)` tags one or more equality Constraints
  whose left-hand sides are declared states at the next time point. Continuous
  versus discrete dynamics are told apart by the left-hand-side component type
  (DerivativeVar versus plain Var).
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
