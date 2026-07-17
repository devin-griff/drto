# drto.infinite_horizon

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a transformation that appends an infinite-horizon
terminal segment to my declared dynamic model, so that a short-horizon dynamic
optimization inherits infinite-horizon stability without my constructing a
terminal cost or terminal region by hand.

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

## Example

The usual path is through the mode transform, with the segment as its
terminal strategy:

```python
import pyomo.environ as pyo
import drto

# ... build a pyomo.dae model: states m.z, controls m.u over ContinuousSet
# m.t, dynamics m.ode, and a tracking stage cost m.stage_con ...

drto.declare_time(m.t)
drto.declare_state(m.z)
drto.declare_continuous_dynamics(m.ode)
drto.declare_control(m.u, profile="piecewise_constant")
drto.declare_tracking_stage_cost(m.stage_con)

pyo.TransformationFactory("dae.collocation").apply_to(
    m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU")

pyo.TransformationFactory("drto.dynamic_optimization").apply_to(
    m, terminal="infinite_horizon")

pyo.SolverFactory("ipopt").solve(m)
```

Applied on its own, the segment options are explicit and the objective is
assembled afterward:

```python
pyo.TransformationFactory("drto.infinite_horizon").apply_to(
    m, nfe=3, ncp=5, beta=1.2)  # gamma defaults to the mesh rule
drto.build_objective(m)
```

## Acceptance criteria

- `TransformationFactory('drto.infinite_horizon')` requires `declare_time`,
  `declare_state`, `declare_continuous_dynamics`, `declare_control`, and
  `declare_tracking_stage_cost`, and errors clearly if any is missing.
- A cost declared with `declare_economic_stage_cost` may be present alongside.
  The segment replicates only the tracking stage cost, and the economic terms
  stay on the finite horizon: an economic stage cost is a nonzero constant at
  the equilibrium, so its tail integral diverges and its quadrature would be
  mesh-dependent rather than an approximation. For the same reason
  `declare_economic_stage_cost` alone is rejected.
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
- The tail cost uses no quadrature state. The declared tracking stage cost is
  replicated at the segment collocation points and enters the objective as
  explicit weighted terms, `beta * h_i * omega_k * psi_ik /
  (gamma*(1 - tau_ik^2))`, assembled by `drto.build_objective` (feature 003)
  as an option-dependent outcome. The Gauss weights are derived from the
  discretization's stored collocation nodes, since `pyomo.dae` stores nodes
  but no quadrature weights, and the result equals the paper's
  quadrature-state formulation exactly.
- The segment controls are parameterized through pyomo-cvp: default
  `('reduced_collocation', ncp)`, the accuracy-first class with `beta`
  carrying the safety margin, and `piecewise_constant` as the conservative
  option. Raw unparameterized copies are never left on the segment.
- `gamma` is a mutable Param, derived by default from the rule
  `tanh(gamma*dt) = tau_11`: the segment's first collocation point lands one
  sampling time past the junction, with `dt` read from the discretized
  horizon's element spacing. An explicit `gamma` option overrides the rule.
- `beta` is an option with default 1.2 and must satisfy `beta >= 1`, which
  the stability argument requires.
- The transform records what it added in `drto.info` (feature 001), and
  `drto.dynamic_optimization` (feature 006) composes it through an option as
  its terminal strategy. It also works applied on its own, through both
  `apply_to` (in place) and `create_using` (a transformed clone).
- Acceptance tests mirror the reference notebook: the short-horizon-plus-
  segment solution reproduces a long-horizon baseline, the explicit-weight
  tail equals the quadrature-state tail to machine precision, and the
  endpoint reaches the setpoint equilibrium with no pins.
