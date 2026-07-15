# drto.dyn_2_ss

**Status:** draft

## Description

As a user of DRTO, I want a transformation that reduces my dynamic model to
its steady-state form, so that from the one model I can solve for an
equilibrium or the economic operating point without hand-writing a separate
steady-state model.

## Benefit hypothesis

Deriving the steady-state model from the same dynamic declarations makes the
equilibrium and the economic-RTO operating point model-consistent by
construction, removing the failure mode where a hand-typed steady-state target
is not a true fixed point of the dynamics. Users maintain one model instead of
two, and it is the first structural transform that proves out the "one model,
many modes" promise.

## Acceptance criteria

- `TransformationFactory('drto.dyn_2_ss')` requires `declare_time`,
  `declare_state`, and `declare_continuous_dynamics` on the model, and errors
  clearly if any is missing.
- It validates that each continuous-dynamics constraint's left-hand side is the
  DerivativeVar of a declared state, and errors clearly otherwise.
- It removes the declared initial condition, terminal constraint, and terminal
  cost if present.
- For each continuous-dynamics constraint, it adds a constraint fixing that
  state's DerivativeVar to zero (`dz/dt == 0`).
- It removes the time index from every variable and constraint, collapsing the
  model to a single point (so a per-time-point stage cost becomes the
  single-point cost).
- The transformed model carries a `min` objective equal to the single reduced
  stage cost, assembled via `drto.build_objective` (feature 001).
- The transformed model is the steady-state system: solving it gives an
  equilibrium satisfying the dynamics at rest (f(z,u)=0) and the model's
  algebraic relations; for a test model with a known analytic steady state, the
  solution matches it.
- The transform works through both `apply_to` (in place) and `create_using`
  (leaving the source dynamic model unchanged).
- It errors clearly on a time-indexed constraint that spans more than one time
  point, since that cannot be reduced to a single point.
