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
  artifacts (the collocation equations and continuity rows pyomo.dae adds)
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
  derivative pinned. There are still no `dz/dt == 0` rows; the Var is fixed,
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
