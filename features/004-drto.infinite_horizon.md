# drto.infinite_horizon

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a transformation that appends an infinite-horizon
terminal segment to my declared dynamic model, so that a short-horizon dynamic
optimization inherits infinite-horizon stability without my constructing a
terminal cost or terminal region by hand.

```python
import pyomo.environ as pyo
import drto

# ... build a pyomo.dae model: states m.z, controls m.u over ContinuousSet
# m.t, dynamics m.ode, and a tracking stage cost m.stage_con ...

drto.horizon(m.t)
drto.state(m.z)
drto.dynamics(m.ode)
drto.control(m.u, profile="piecewise_constant")
drto.tracking_stage_cost(m.stage_con)

pyo.TransformationFactory("dae.collocation").apply_to(
    m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU")

pyo.TransformationFactory("drto.infinite_horizon").apply_to(
    m, nfe=3, ncp=5, beta=1.2)  # the defaults, shown explicitly;
                                # gamma defaults to the mesh rule

pyo.TransformationFactory("drto.parameterize").apply_to(m)  # feature 017
drto.build_objective(m)
pyo.SolverFactory("ipopt").solve(m)
```

The tail terms it registers are live cost terms, so `drto.build_objective`
picks them up wherever it runs: called directly as above, or as the final
step of `drto.dynamic_optimization`. Applying this transform before the mode
transform is the whole composition. There is no coupling option.

## Benefit hypothesis

Terminal costs and terminal regions are the expert-only part of stabilizing
NMPC, and computing them offline for nonlinear processes is formidable. The
terminal segment of Dinh et al. (2025,
[doi:10.1016/j.jprocont.2025.103565](https://doi.org/10.1016/j.jprocont.2025.103565))
replaces that construction with discretization: the tail to infinity is
compressed onto [0, 1] by the time transformation `tau = tanh(gamma*(t - tN))`
and solved inside the same NLP, so the terminal cost is the actual tail cost
and the terminal condition is an equilibrium the cost selects. Their case
studies match long-horizon baselines with a fraction of the horizon and solve
time. For drto this gives `dynamic_optimization` a stability story with no
hand-built terminal ingredients. The mechanism is settled and verified in
[`examples/hicks_inf.ipynb`](../examples/hicks_inf.ipynb): a 5-step horizon
plus segment reproduces the 50-step policy to about 2 percent on the first
move.

## Acceptance criteria

- `TransformationFactory('drto.infinite_horizon')` requires `horizon`,
  `state`, `dynamics`, `control`, and
  `tracking_stage_cost`, and errors clearly if any is missing.
- A cost declared with `economic_stage_cost` may be present alongside.
  The segment replicates only the tracking stage cost, and the economic terms
  stay on the finite horizon: an economic stage cost is a nonzero constant at
  the equilibrium, so its tail integral diverges and its quadrature would be
  mesh-dependent rather than an approximation. For the same reason
  `economic_stage_cost` alone is rejected.
- It applies to a model whose declared time set is already discretized. It
  builds a segment ContinuousSet on [0, 1] carrying the transformed time and
  discretizes it itself with Gauss-Legendre collocation (`nfe` and `ncp`
  options, defaults 3 and 5). Legendre is required: no collocation equation
  may sit at the singular endpoint `tau = 1`.
- Segment copies of the declared states and controls carry the same bounds.
  The dilated dynamics, the declared dynamics multiplied by
  `gamma*(1 - tau^2)`, are written at interior collocation points only, and
  endpoint values at `tau = 1` come from the discretization's continuity
  extrapolation.
- Linking constraints stitch the segment's initial state to the declared
  states at the end of the horizon.
- The endpoint is a hard equilibrium constraint, `0 = f` at `tau = 1`. There
  are no setpoint pins: the stage cost selects the equilibrium. The
  soft-constrained endpoint (paper eq. 36) is a follow-on, out of scope here.
- The tail cost uses no quadrature state and adds no variables or
  constraints: the declared tracking stage cost is replicated at the segment
  collocation points as named Expressions (a replicated cost Var would sit on
  an active bound as the tail cost vanishes at the equilibrium, wrecking
  interior-point performance) and enters the objective as explicit weighted
  terms, `beta * h_i * omega_k * psi_ik /
  (gamma * dt * (1 - tau_ik^2))`, the paper's `(beta/dt) * phi_f`, so the tail
  is commensurate with the per-sample stage sum. They are assembled by
  `drto.build_objective` (feature 003)
  as an option-dependent outcome. The Gauss weights are derived from the
  discretization's stored collocation nodes, since `pyomo.dae` stores nodes
  but no quadrature weights, and the result equals the paper's
  quadrature-state formulation exactly.
- The segment controls are new variables with their own pyomo-cvp profile,
  declared by the transform and independent of the profile declared on the
  finite-horizon controls: default `'collocation'` (the element's
  collocation polynomial through all its collocation points), the
  accuracy-first class with `beta` carrying the safety margin, and
  `piecewise_constant` as the conservative option. Raw unparameterized copies
  are never left on the segment.
- `gamma` is a mutable Param set by an option whose default, `'rule'`,
  derives it from the mesh rule `tanh(gamma*dt) = tau_11`: the segment's
  first collocation point lands one sampling time past the junction, with
  `dt` read from the sample grid captured by `horizon`. A number
  overrides the rule.
- `beta` is a mutable Param set by an option, default 1.2, and must satisfy
  `beta > 1` (paper section 4.1.2): the terminal cost must overestimate the
  tail, and the margin `beta - 1` is what covers the quadrature error, so
  `beta = 1` leaves no room for the quadrature to err low.
- Both Params are referenced symbolically everywhere they appear, `gamma` in
  the dilated dynamics and both in the tail weights, never baked in as
  numbers, so `set_value` retunes either between solves with the dynamics and
  the objective staying consistent, no re-apply needed.
- The transform records what it added in `drto.info` (feature 001). There is
  no coupling option on the mode transforms: the tail terms it registers are
  live cost terms, so `drto.build_objective` includes them wherever it runs,
  directly or as the final step of `drto.dynamic_optimization` (feature 006).
  Applying this transform before the mode transform is the composition.
- It works through both `apply_to` (in place) and `create_using` (a
  transformed clone).
- Acceptance tests mirror the reference notebook: the short-horizon-plus-
  segment solution reproduces a long-horizon baseline, the explicit-weight
  tail equals the quadrature-state tail to machine precision, and the
  endpoint reaches the setpoint equilibrium with no pins.
