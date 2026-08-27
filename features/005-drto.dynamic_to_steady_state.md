# drto.dynamic_to_steady_state

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a transformation that reduces my dynamic model to
its steady-state form, so that from the one model I can solve for an
equilibrium or the economic operating point without hand-writing a separate
steady-state model.

```python
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory

import drto

# ... declared dynamic model m (feature 002) ...

ss = pyo.TransformationFactory("drto.dynamic_to_steady_state").create_using(m)
# ss is the steady-state system: time collapsed to a single point, each
# dz/dt collapsed with it and fixed at zero, initial and terminal pieces
# removed. m is unchanged
drto.build_objective(ss)              # e.g. the single-point cost
SolverFactory("ipopt").solve(ss)
```

From a model statement rather than a model, the function form calls the
builder and reduces what it returns:

```python
ss = drto.dynamic_to_steady_state(build)   # build() is the model statement
```

A `Block(time)` family collapses to its `t0` member, one per non-time
index combination, discarding only the declared time set's discretization
equations. A nested time-indexed Block errors.

## Benefit hypothesis

The user maintains one model instead of two. The steady-state form
derived from the same declarations is model-consistent by construction,
removing the failure mode where a hand-typed steady-state target is not
a true fixed point of the dynamics, and this is the first structural
transform demonstrating that one declared model serves every mode.

## Acceptance criteria

- `TransformationFactory('drto.dynamic_to_steady_state')` requires `horizon`,
  `state`, and `dynamics` on the model, and errors
  clearly if any is missing.
- It applies to the declared or discretized model, before any drto
  transformation. An applied `drto.infinite_horizon` or applied control
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
  (an index-reduced energy balance) are permitted. They get the zero
  substitution like every other reference.
- It removes, if present, the declared initial condition, terminal
  constraint, both terminal costs (the tracking terminal cost and the
  estimation terminal cost), and the move-suppression cost, whose moves
  are zero at any steady point.
- Each declared state's derivative collapses to a single point like
  every other time-indexed Var and is fixed at zero, not eliminated.
  `dz/dt = 0` is what steady state means, so the declared dynamics and
  any algebraic equation carrying a derivative keep their form as the
  user wrote them, with the derivative pinned. There are still no
  `dz/dt == 0` constraints. The Var is fixed, not constrained, and the
  solver treats it as a constant. Pyomo cannot hold
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
  algebraic relations. For a test model with a known analytic steady
  state, the solution matches it.
- The transform works through both `apply_to` (in place) and `create_using`
  (leaving the source dynamic model unchanged).
- `drto.dynamic_to_steady_state(build)` is the function form under the
  transformation's own name, the dual form feature 003 sets with
  `build_objective`. It calls `build()` with no arguments. A result
  declaring a horizon is reduced in place through the registered
  transformation, and a result with no declared horizon is returned
  unchanged, so a statement constructing its steady form natively (an
  IDAES flowsheet built with `dynamic=False`) skips the dynamic build
  entirely. The function owns the model it just built, so it reduces in
  place rather than cloning, and a user holding a model keeps
  `create_using` for the preserving form. The builder contract is
  feature 006's: a builder returns a declared model, its first two
  parameters are the interval count and the sampling time, every
  parameter has a default so `build()`, `build(N)`, and `build(N, h)`
  are all legal, and the returned model may be discretized or not. The
  steady modes use only the bare call.
- It errors clearly on a time-indexed constraint that spans more than one
  time point, since that cannot be reduced to a single point.
- A model keeping per-time structure in `Block(time)` members reduces to a
  square steady system.
- A time-indexed Reference collapses to a view of the surviving member
  or of the collapsed Var, never to a fresh independent Var, and Ports
  keep pointing at their referents.
- A fixed input stays fixed through the collapse.
- A dynamics family with Skip members reduces to its members.
- The registry's records are re-pointed at the collapsed components, and
  a control's or disturbance's profile annotation is dropped along with
  any pending profile over the deleted time set, since a single point
  has no profile.
