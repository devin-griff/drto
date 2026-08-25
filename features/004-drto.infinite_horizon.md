# drto.infinite_horizon

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a transformation that appends an infinite-horizon
terminal segment to my declared dynamic model, so that a short-horizon dynamic
optimization inherits infinite-horizon stability without my constructing a
terminal cost or terminal region by hand.

```python
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory

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
    m, nfe=3, ncp=5, beta=1.2)  # the defaults, shown explicitly.
                                # gamma defaults to the mesh rule

pyo.TransformationFactory("drto.parameterize").apply_to(m)  # feature 017
drto.build_objective(m)
SolverFactory("ipopt").solve(m)
```

The tail terms it registers are live cost terms, so `drto.build_objective`
picks them up wherever it runs: called directly as above, or as the final
step of `drto.dynamic_optimization`. Applying this transform before the mode
transform is the whole composition. There is no coupling option.

Per-time structure kept in `Block(time)` members replicates onto the
segment like any other algebra. A nested time-indexed Block errors.

## Benefit hypothesis

The user gets infinite-horizon stability with no hand-built terminal
cost or terminal region. Those are the expert-only part of stabilizing
NMPC, and computing them offline for nonlinear processes is formidable.
The terminal segment of Dinh et al. (2025,
[doi:10.1016/j.jprocont.2025.103565](https://doi.org/10.1016/j.jprocont.2025.103565))
replaces that construction with discretization. The tail to infinity is
compressed onto [0, 1] by the time transformation
`tau = tanh(gamma*(t - tN))` and solved inside the same NLP, so the
terminal cost is the actual tail cost and the terminal condition is an
equilibrium the cost selects. The paper's case studies match
long-horizon baselines with a fraction of the horizon and solve time,
and the mechanism is verified in
[`examples/hicks_inf.ipynb`](../examples/hicks_inf.ipynb): a 5-step
horizon plus segment reproduces the 50-step policy to about 2 percent
on the first move.

## Acceptance criteria

- A model keeping per-time structure in `Block(time)` members holds its
  degrees of freedom across the transform.
- `TransformationFactory('drto.infinite_horizon')` requires `horizon`,
  `state`, `dynamics`, `control`, and
  `tracking_stage_cost`, and errors clearly if any is missing.
- A cost declared with `economic_stage_cost` may be present alongside.
  The segment replicates only the tracking stage cost, and the economic
  terms stay on the finite horizon. An economic stage cost is a nonzero
  constant at the equilibrium, so its tail integral diverges and its
  quadrature would be mesh-dependent rather than an approximation. For
  the same reason `economic_stage_cost` alone is rejected.
- It applies to a model whose declared time set is already discretized. It
  builds a segment ContinuousSet on [0, 1] carrying the transformed time and
  discretizes it itself with Gauss-Legendre collocation (`nfe` and `ncp`
  options, defaults 3 and 5). Legendre is required: no collocation equation
  may sit at the singular endpoint `tau = 1`.
- Segment copies of the declared states and controls keep the same
  bounds.
  The dilated dynamics, the declared dynamics multiplied by
  `gamma*(1 - tau^2)`, are written at interior collocation points only, and
  endpoint values at `tau = 1` come from the discretization's continuity
  extrapolation.
- Linking constraints stitch the segment's initial state to the declared
  states at the end of the horizon.
- The terminal segment endpoint is pinned to the steady state by default.
  `terminal='soft'` (the
  default) adds the paper's eq. 36 relaxed endpoint constraint, per state
  `z(tau=1) + eps_up - eps_lo == z_s` with the penalty `mu*(eps_up +
  eps_lo + eps_up**2 + eps_lo**2)` registered as a cost group. The linear
  part is the exact L1 pin. The quadratic's gradient vanishes at zero
  slack, so zero slack stays optimal whenever the pin can hold and the
  violation threshold is the L1 one. The quadratic part makes the pin
  strictly convex at its kink, so the pin's multipliers are unique and
  continuous instead of an interval the solver picks an arbitrary point
  of. Without it, consecutive solves of near-identical problems return
  wildly different multipliers, anything passing multipliers between
  solves becomes unreliable, and the conditioning degrades even on cold
  solves (gh #37). Both parts are scaled by the one `mu`.
  `terminal='none'` imposes no endpoint condition: no pin, the singular
  tail cost as the only terminal enforcement, and the endpoint (the
  discretization's Legendre extrapolation, the paper's evaluated
  endpoint z_e) settling as close to the setpoint as the horizon's
  freedoms allow. A pin reads the declared `drto.steady_state` targets
  and requires one per state. With `terminal='none'` the transform needs
  none. The paper's operative problem (eq. 36) imposes the endpoint
  constraint, and it is what pins the unstable modes on
  open-loop-unstable plants, so the pin is the default. There is no
  plain-equality option. One slack-free equality per state
  over-determines the tail NLP when state members outnumber the
  horizon's control freedoms, and the exact L1 penalty means `'soft'` at
  a large enough `mu` reproduces it anyway.
- The tail cost uses no quadrature state and adds no variables or
  constraints. The declared tracking stage cost is replicated at the segment
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
  applied by the transform through cvp's explicit form and independent of the profile declared on the
  finite-horizon controls: default `'collocation'` (the element's
  collocation polynomial through all its collocation points), the
  accuracy-first class with `beta` as the safety margin, and
  `piecewise_constant` as the conservative option. Raw unparameterized
  copies are never left on the segment. A control declared on a
  component that points at existing variables (an inlet Port's
  `flow_vol`) gets one segment copy, and the replicated equations are
  routed to it rather than to separate copies of the variables it
  points at (gh #18).
- `gamma` is a mutable Param set by an option whose default, `'rule'`,
  derives it from the mesh rule `tanh(gamma*dt) = tau_11`, which puts the
  segment's first collocation point one sampling time past the junction,
  with `dt` read from the sample grid captured by `horizon`. A number
  overrides the rule.
- `beta` is a mutable Param set by an option, default 1.2, and must
  satisfy `beta > 1` (paper section 4.1.2). The terminal cost must
  overestimate the tail, and the margin `beta - 1` covers the quadrature
  error, so `beta = 1` leaves no room for the quadrature to err low.
- Both Params are referenced symbolically everywhere they appear,
  `gamma` in the dilated dynamics and both in the tail weights, never
  written in as numbers, so `set_value` retunes either between solves
  with the dynamics and the objective staying consistent and no
  re-application needed.
- States may be indexed by other sets besides time. Segment copies,
  linking, and replication run per member. Controls stay indexed by time
  alone.
- Algebraic variables and equations are copied without being declared.
  Any
  time-indexed variable referenced by the replicated equations that is not a
  declared state or control gets a segment copy, and every active
  time-indexed constraint that is not declared as something else and is not
  a discretization artifact (the collocation and continuity equations) is
  replicated on the segment at the interior collocation points, where the
  dilated dynamics reference its variables. Algebraic copies get no
  element-boundary values and no linking.
- A variable copied to the segment with no replicated equation involving
  it errors, naming the variable. A silently free variable there is a
  wrong tail the solver exploits.
- A declared disturbance referenced by the replicated equations gets a
  segment copy fixed at zero, so the tail continues under nominal
  disturbance unless told otherwise. A `disturbances` option maps a
  declared disturbance to the constant its copy is fixed at, a scalar
  held everywhere or one value per non-time index (a multi-component
  feed). A value for an undeclared disturbance is a descriptive error.
- The finite grid's final instant becomes the linking time. Model
  equations there reference the last move, which pyomo-cvp
  resolves by the constraint's own structure. No convention is declared
  or flipped.
- A declared tracking terminal cost is deactivated on application: the tail
  integral is the cost-to-go, so V_f would double-count. The deactivation is
  noted in the transformation outcome.
- The transform records what it added in `drto.info` (feature 001). There is
  no coupling option on the mode transforms: the tail terms it registers are
  live cost terms, so `drto.build_objective` includes them wherever it runs,
  directly or as the final step of `drto.dynamic_optimization` (feature 006).
  Applying this transform before the mode transform is the composition.
- It works through both `apply_to` (in place) and `create_using` (a
  transformed clone).
- The segment's algebraic copies (the flat algebra, the Block members,
  the indexed Vars' algebraic entries) are indexed over the interior
  collocation points only. Every point that exists is a point some
  replicated equation determines. The state copies keep the full tau set for their
  continuity and discretization equations, as the tail cost's quadrature
  already does. No dead members exist for a reader to find stale
  (gh #32).
- The transformation records, internally on the registry object, which
  segment component belongs to which declaration: each declared state's
  copy, tau derivative, discretization and continuity equations, link,
  and endpoint pin (equation and slacks), each declared control's copy,
  and each declared dynamics family's copy with its algebraic balances.
  The registry
  view renders nothing new for it, and the pairing follows a clone with
  its references remapped. drto's own consumers (cold start, plotting)
  read the recorded pairing instead of reconstructing component names
  (gh #27).
- Acceptance tests mirror the reference notebook: the short-horizon-plus-
  segment solution reproduces a long-horizon baseline, the explicit-weight
  tail equals the quadrature-state tail to machine precision, and the
  endpoint settles at the setpoint equilibrium driven by the cost alone.
  The pin penalty's form is pinned by retuning `mu`. At unit slacks the
  objective moves by the linear and the quadratic part alike, two per
  slack per unit of `mu`.
