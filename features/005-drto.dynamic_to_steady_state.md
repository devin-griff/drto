# drto.dynamic_to_steady_state

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a transformation that reduces my dynamic model to
its steady-state form, so that from the one model I can solve for an
equilibrium or the economic operating point without hand-writing a separate
steady-state model.

```python
import pyomo.environ as pyo
import drto

# ... declared dynamic model m (feature 002) ...

ss = pyo.TransformationFactory("drto.dynamic_to_steady_state").create_using(m)
# ss is the steady-state system: time collapsed to a single point, each
# dz/dt collapsed with it and fixed at zero, initial and terminal pieces
# removed; m is unchanged
drto.build_objective(ss)              # e.g. the single-point cost
pyo.SolverFactory("ipopt").solve(ss)
```

### Models that keep per-time structure in Blocks

A model may keep its per-time structure in `Block(time)` members rather
than time-indexed Vars, which is the IDAES property-block idiom. The
reduction treats that structure as the time-varying structure it is:

- A `Block(time)` family collapses to its single steady member: the `t=0`
  member stays, its variables and internal equations as written, and the
  other members leave the model with their contents. Nothing is rebuilt, so
  values, bounds, units, and fixed status carry through trivially.
- Fixed stays fixed: a fixed variable in the surviving member is a
  specification and remains fixed, so an IDAES feed holds at its value
  through the reduction with no declaration involved.
- A time-indexed Reference whose referents live in the members (an IDAES
  Port entry such as `inlet.flow_vol[t]`) collapses to a Reference onto the
  surviving member. It must not become a fresh independent Var: the Port
  keeps pointing at its referent.
- Time-invariant Blocks, including Params and parameter blocks, are shared
  as-is, which is current behavior.
- A Block carrying further indexes — a 1D control volume's `Block(t, x)`,
  an MSContactor's `Block(t, element)` — collapses the same way per
  spatial point: the `t=0` member of every index combination survives,
  so the spatial structure is kept and only time goes (gh #54).
- Spatial discretization equations are algebra, not grid machinery: a
  discretization equation is discarded only when its derivative is taken
  with respect to the declared time set, so a settler's
  `material_flow_dx_disc_eq` survives the reduction.
- A sparse dynamics container — a balance written past an inlet boundary,
  a member skipped by its rule — reduces to its members: index
  combinations with no member anywhere are skipped, not errors, and an
  empty Var is left alone.
- The guard that every dynamics row differentiates a declared state reads
  the row's variables, so a derivative side written as a sum (a balance
  carrying a noise term) checks the same as a bare one.

Out of scope, rejected with a descriptive error: a time-indexed Block
nested inside another time-indexed Block, mirroring the terminal
segment's rule.

## Benefit hypothesis

Deriving the steady-state model from the same dynamic declarations makes the
equilibrium and the economic-RTO operating point model-consistent by
construction, removing the failure mode where a hand-typed steady-state target
is not a true fixed point of the dynamics. Users maintain one model instead of
two, and it is the first structural transform that proves out the "one model,
many modes" promise.

## Acceptance criteria

- `TransformationFactory('drto.dynamic_to_steady_state')` requires `horizon`,
  `state`, and `dynamics` on the model, and errors
  clearly if any is missing.
- It applies to the declared or discretized model, before any drto
  transformation: an applied `drto.infinite_horizon` or applied control
  profiles error clearly. The steady reduction and the dynamic transforms
  are sibling branches of the same declarations, not a pipeline. On a
  discretized model the discretization
  artifacts (the collocation equations and continuity equations pyomo.dae adds)
  are discarded, grid machinery rather than model content, and the
  reduction gives the same steady system as reducing before
  discretization.
- It validates that one side of each dynamics constraint is the
  DerivativeVar of a declared state (either orientation of the equality),
  and errors clearly otherwise. Derivative references outside the dynamics
  (an index-reduced energy balance) are permitted; they get the zero
  substitution like every other reference.
- It removes, if present, the declared initial condition, terminal constraint,
  and both terminal costs (the tracking terminal cost and the estimation
  terminal cost).
- Each declared state's derivative collapses to a single point like every
  other time-indexed Var and is fixed at zero, not eliminated: `dz/dt = 0` is
  what steady state means, so the declared dynamics and any algebraic equation
  carrying a derivative keep their form as the user wrote them, with the
  derivative pinned. There are still no `dz/dt == 0` constraints; the Var is fixed,
  not constrained, and the solver folds it in as a constant. Pyomo cannot hold
  a DerivativeVar that is not indexed by a ContinuousSet, and the time set
  leaves the model, so the collapsed derivative is a plain scalar Var of the
  same name.
- It removes the time index from every variable and constraint, collapsing the
  model to a single point (so a per-time-point stage cost becomes the
  single-point cost).
- It does not construct the objective. Its only interaction with an objective,
  if one is present, is removing the time index from the variables in it while
  collapsing to a single point. Choosing and assembling the mode's objective is
  left to the mode transforms and `drto.build_objective`.
- The transformed model is the steady-state system: solving it gives an
  equilibrium satisfying the dynamics at rest (f(z,u)=0) and the model's
  algebraic relations; for a test model with a known analytic steady state, the
  solution matches it.
- The transform works through both `apply_to` (in place) and `create_using`
  (leaving the source dynamic model unchanged).
- It errors clearly on a time-indexed constraint that spans more than one time
  point, since that cannot be reduced to a single point.

- The dynamic IDAES CSTR (saponification packages, feature-002
  declarations, nothing beyond them) reduces to the steady system:
  `drto.steady_state_simulation` leaves it square, pounce solves it, and
  the solution matches a hand-built `FlowsheetBlock(dynamic=False)`
  model at the reactor's feed-alone equilibrium.
- Exactly one member of each `Block(time)` family survives, and no active
  constraint or Reference reaches a removed member. A model with no
  time-indexed Blocks reduces to the same steady system as before.
- A time-indexed Block nested inside another raises the transform's usual
  descriptive ValueError naming the offending component.
- The registry's transformation log counts the collapsed Block families
  alongside the Var and Constraint counts.
- A `Block(t, x)` family keeps one member per spatial point with its
  internal equations intact, and the spatial discretization equations
  survive with their derivative variables live; the declared time set's
  collocation and continuity equations are discarded.
- A sparse dynamics container reduces to its members, and a
  noise-carrying balance passes the declared-state guard while an
  undeclared differentiation still errors.
- The declared mixer-settler stage reduces to a square steady system at
  held controls, the dynamic model's inert first-instant data freed.
- The IDAES CSTR example notebook declares the flowsheet once and runs
  two solves total: the reduced model for the setpoint, then the
  infinite-horizon controller.
