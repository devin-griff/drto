# User guide

The narrative guide grows as the core lands: the six modes, the declaration
surface, and the initialization routines. Until then, the
[README](https://github.com/devin-griff/drto#readme) is the overview and
`DESIGN.md` is the authoritative design record.

## The registry: `drto.info`

Every drto model carries one registry: the record of what has been declared
and which transformations have been applied. `drto.info(m)` returns it,
creating it on first access. It lives in Pyomo's namespaced private data, so
it never appears in the model's component tree, and it survives `clone()`
(and every transformation's `create_using` form) with its stored component
references remapped to the clone.

Displaying it renders a drto-aware view of the model: declarations grouped by
role, indexed constraints as one symbolic equation per family (for example
`dzdt[k] == - z[k] + u[k]  for k in t`, not the per-index expansion `pprint`
produces), and the ordered log of applied transformations with their
outcomes.

The declarations (feature 002) write to the registry and the transformations
read it, so it is the one place drto looks for the model's declared pieces.

## The declarations

The control-side declarations (feature 002) declare the pieces of an
optimization or simulation problem: `horizon`,
`state`, `dynamics`, `control` (with its
pyomo-cvp `profile`), the stage and terminal costs, `initial_condition`
(a state at the first time point equal to a mutable Param, the feedback hook),
`terminal_constraint`, and the steady-state targets:
`steady_state(m.z, m.z_ss)` pairs a declared state with its setpoint Param,
and `steady_state_control(m.u, m.u_ss)` pairs a declared control the same
way. Each declaration
validates its convention and records the component in the registry, where the
transformations find it.

Every function serves two calling styles. Tagging: on a model you already
built, an attached component registers immediately, interleaved or in one
block. Wrapping: a fresh component, `m.z = state(pyo.Var(m.t))`, is returned
for the assignment and registers when Pyomo attaches it. The constraint-role
declarations also work as decorators, `@drto.dynamics(m, m.t)` taking what
`@m.Constraint` would. The styles mix per component; in every style a
declaration's prerequisites must be declared by the time it registers, which
writing the model top-down satisfies.

Declarations that scale with the states and controls
take varargs when tagging and accumulate; the wrap form takes exactly one
component; the one-of-each declarations error on a second,
different object. Conventions are read from either side of the written
equality, so `lhs == rhs` and `rhs == lhs` are equivalent.

The estimation side (feature 018) declares the moving-horizon-estimation
pieces through the same surface. `estimated_parameter` tags Vars for unknown
model parameters, constant over the window (so it needs no horizon, and also
serves steady-state data reconciliation). `disturbance` tags the process-noise
Vars the estimator adjusts, and `measurement` tags the mutable Params holding
the window's measured values, the estimation feedback hook. The estimation
costs are scalar-side equalities like the tracking costs: `estimation_stage_cost`
over the samples for the measurement residual plus the process-noise penalty,
`estimation_terminal_cost` for the present-time residual, and `arrival_cost`
for the soft prior on the initial state, the soft dual of `initial_condition`.
Same tagging, wrapping, and decorator forms, same registry. These are the
declaration surface only; the estimation mode transforms that consume them come
with the moving-horizon-estimation follow-on.

## Objective assembly: `drto.build_objective`

One routine owns every mode's objective. The bare call assembles the live
registered cost terms, each group by its weights: declared stage costs sum
their per-point cost var over the active members (the stage cost does not
exist at the final time, where the terminal cost applies), a terminal cost
adds its scalar var, and transforms may register additional weighted cost
groups. Liveness is component presence, so a mode drops a term by dropping or
deactivating its constraint. The marked case, `zero=True`, installs a
constant-zero objective and is what the simulation transforms pass. Any
existing active objective is deactivated first, and the routine is also
registered as `TransformationFactory('drto.build_objective')`.

## The infinite horizon: `drto.infinite_horizon`

Appends the terminal segment of Dinh et al. (2025): the tail of the horizon
to infinity, compressed onto [0, 1] by `tau = tanh(gamma*(t - tN))`. The
segment carries copies of the declared states and controls (states may carry index sets besides time; undeclared algebraic variables and equations ride along automatically), the dilated
dynamics at interior Gauss-Legendre points, and the tracking stage cost
replicated as the tail integrand. The tail enters the objective as explicit Gauss-weighted terms,
the paper's `(beta/dt)*phi_f`, registered as a cost group that
`drto.build_objective` picks up wherever it runs: applying this transform
before the mode transform is the whole composition. `beta` and `gamma` are
mutable Params, symbolic in the dynamics and the weights, so both retune
between solves; `gamma` defaults to the mesh rule and `beta` must exceed 1.

The segment endpoint is pinned to the steady state by default (the paper's
eq. 36). The endpoint `z(tau=1)` is the discretization's Legendre
extrapolation of the last element, the paper's evaluated endpoint. The
`terminal` option selects `'soft'` (the default: `z(tau=1) + eps_up - eps_lo
== z_s` with an L1 penalty `mu*(eps_up + eps_lo)`, `mu` a mutable Param
defaulting to 1000, in the objective) or `'none'` (no pin, the singular tail
weights driving the trajectory to settle on their own). A pin reads the
declared `drto.steady_state` targets, so `terminal='soft'` needs one per
state; `terminal='none'` needs none. The pin is on the state
value, not a derivative: Gauss-Legendre puts no node at `tau=1`, so the
derivative there is undefined while the extrapolated state value is well
defined. Because the soft pin adds its penalty to the objective, run
`drto.build_objective` after `drto.infinite_horizon` (drto enforces that
order).

## Applying the profiles: `drto.parameterize`

`control(profile=...)` records a profile; `drto.parameterize` applies
every pending one by delegating to pyomo-cvp, then repairs the registry so
the control records point at the live replacement components. The mode
transforms run it as one of their steps; standalone workflows call it after
`drto.infinite_horizon` and before `drto.build_objective`.

## Dynamic optimization: `drto.dynamic_optimization`

The headline mode, NMPC and D-RTO. It assembles the horizon optimization from
the declarations, so applying the profiles and building the objective collapse
into one call on the discretized model. It requires `horizon`, `state`,
`dynamics`, `control`, `initial_condition`, and at least one stage cost, and
errors naming what is missing. The declared controls stay free and are
parameterized by their declared profiles, the horizon is kept (this is the
dynamic mode, not a reduction), and `build_objective` runs as the final step.
With both a tracking and an economic stage cost declared, `tracking_weight`
scales the tracking side. It defaults to 1, and the economic cost is in
currency units and is never scaled.

Because the objective is assembled here, anything that registers cost terms
runs first: `drto.infinite_horizon` applies before this transform, or the tail
never reaches the objective.

A model may carry the estimation declarations too, since one model serves
every mode. The transform neutralizes them so the control problem carries only
what it uses, and the registry mirrors the model: a component that leaves has
its record purged, one that stays keeps its record. The estimation costs and
the measurement Params are deleted. A disturbance is eliminated by
substituting zero, which removes it only where it enters additively, so one
inside a nonlinear term errors rather than being silently zeroed. An estimated
parameter is fixed at the value it holds and keeps its record, since it stays
a live coefficient in the equations the controller solves. `drto.dynamic_simulation`
runs the same routine, so the two modes cannot drift apart.

## Dynamic simulation: `drto.dynamic_simulation`

The dual of the mode above: it frees nothing and integrates the declared model
forward over the horizon. It requires the same declarations, including
`initial_condition`, because a forward integration is not square without the
initial state pinned.

The declared profiles are applied first, so the simulated input takes the shape
the model declared and the user picks that shape at declaration time through
`control(profile=...)`. The parameterized controls are then fixed. The
`controls` option sets what they are fixed at, either a constant held across
the horizon or one value per free point the profile leaves, and a control not
named there is fixed at the value it already holds. A control holding no value
errors rather than being fixed at nothing.

A simulation carries no cost, so the declared stage and terminal cost equations
leave the model as they do in the steady-state simulation, and
`build_objective` installs the constant-zero objective. The estimation
declarations are neutralized by the same routine the optimization mode uses,
which also protects squareness here, since a free disturbance would leave the
system underdetermined.
