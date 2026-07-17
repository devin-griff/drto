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

The declaration surface (feature 002) declares the pieces of an optimization
or simulation problem: `horizon`,
`state`, `dynamics`, `control` (with its
pyomo-cvp `profile`), the stage and terminal costs, `initial_condition`
(a state at the first time point equal to a mutable Param, the feedback hook),
`terminal_constraint`, and the steady-state targets, each a
`steady_state(m.z, m.z_ss)` pair of a declared state (or control) with its
setpoint Param. Each declaration
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
dynamics at interior Gauss-Legendre points, a hard equilibrium endpoint the
stage cost selects, and the tracking stage cost replicated as the tail
integrand. The tail enters the objective as explicit Gauss-weighted terms,
the paper's `(beta/dt)*phi_f`, registered as a cost group that
`drto.build_objective` picks up wherever it runs: applying this transform
before the mode transform is the whole composition. `beta` and `gamma` are
mutable Params, symbolic in the dynamics and the weights, so both retune
between solves; `gamma` defaults to the mesh rule and `beta` must exceed 1.

## Applying the profiles: `drto.parameterize`

`control(profile=...)` records a profile; `drto.parameterize` applies
every pending one by delegating to pyomo-cvp, then repairs the registry so
the control records point at the live replacement components. The mode
transforms run it as one of their steps; standalone workflows call it after
`drto.infinite_horizon` and before `drto.build_objective`.
