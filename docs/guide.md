# User guide

drto has one idea: your Pyomo model already contains everything a dynamic
optimization needs, so instead of rebuilding it in a modeling layer, you
declare which components play which role and let transformations rewrite
the model. This page walks the three workflows those declarations drive —
an infinite-horizon controller, a forward simulation, and the steady-state
branch — then the two how-tos that come up on real models: scaling and
IDAES flowsheets.

## The registry

Every declaration records itself in the model's registry, and every
transformation reads the registry rather than re-scanning the model.
`drto.info(m)` returns it; displaying it renders the model in its own
terms — components grouped by role, indexed constraints in compact
symbolic form (one equation per family, not the per-index expansion
`pprint` produces), units where the model carries them, and the ordered
log of applied transformations:

```
<drto registry>
declarations:
  horizon: t (ContinuousSet, 11 points)
  states: z (free)
  dynamics: dzdt[t]  ==  - z[t] + u[t]  for t in t
  controls: u (piecewise_constant, free)
  ...
transformations: (none)
```

The registry is stored outside the component tree and survives `clone()`
(and therefore every `create_using`) with its references remapped, so a
transformed clone knows its own declarations.

## The declarations

The control-side surface, in the order a model usually declares them:

- `drto.horizon(t)` — the time `ContinuousSet`, declared before
  discretization so the registry keeps the sample grid.
- `drto.state(z, ...)` — the differential states. A state may be a whole
  time-indexed Var or a member subset of a packed one (see the IDAES
  how-to below).
- `drto.dynamics(ode, ...)` — the constraint families that are the
  differential equations. Everything not declared here is algebra.
- `drto.control(u, ..., profile=...)` — the manipulated inputs;
  `profile` names the pyomo-cvp parameterization
  (e.g. `"piecewise_constant"`) applied when a transform parameterizes
  the controls.
- `drto.disturbance(w, ...)` — declared zero-mean inputs a simulation or
  the terminal segment can hold at given values.
- `drto.tracking_stage_cost(stage)` / `drto.economic_stage_cost(econ)` /
  `drto.tracking_terminal_cost(term)` — cost rows, written as equality
  constraints defining cost variables in the model's own units.
- `drto.initial_condition(ic, ...)` — rows pinning states at the first
  point to mutable Params, the feedback hooks a receding-horizon loop
  updates.
- `drto.steady_state(z, z_ss)` / `drto.steady_state_control(u, u_ss)` —
  pair each state and control with the mutable Param holding its target.
  The transforms and initializers read these as the setpoint.
- `drto.terminal_constraint(con)` — an explicit endpoint condition, when
  one is wanted.

The estimation-side surface (`drto.measurement`,
`drto.estimated_parameter`, `drto.estimation_stage_cost`,
`drto.estimation_terminal_cost`, `drto.arrival_cost`) mirrors this for
the planned moving-horizon estimation features.

Declarations are checked as they are made, and a bad one raises a
descriptive error at declaration time, not at solve time.

## Workflow: an infinite-horizon controller

The controller workflow runs declare → discretize → terminal segment →
initialize → assemble → solve:

```python
# 1. declare (as above), then discretize the horizon
pyo.TransformationFactory("dae.collocation").apply_to(
    m, wrt=m.t, nfe=10, ncp=3, scheme="LAGRANGE-RADAU")

# 2. append the infinite-horizon terminal segment
pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)

# 3. initialize: states on a profile to the targets, algebra solved
drto.cold_start_dynamic(m)

# 4. parameterize the controls and assemble the objective
pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)

# 5. solve and read back
pyo.SolverFactory("ipopt").solve(m)
drto.plot_states(m)
drto.plot_controls(m)
```

**The terminal segment** (`drto.infinite_horizon`) maps the tail
`t in [tN, inf)` onto `tau in (0, 1]` through `tau = tanh(gamma*(t - tN))`
and appends it as a block: copies of the states, controls, and algebra,
the dilated dynamics, the stage cost integrated with explicit quadrature
weights, and a soft pin holding the endpoint at the declared targets.
Options: `nfe`/`ncp` size the segment mesh, `gamma` overrides the mesh
rule `tanh(gamma*dt) = tau_11`, `beta > 1` sets the tail's overestimation
weight, `terminal="soft"|"none"` selects the endpoint pin, `mu` its
penalty weight, and `disturbances` holds declared disturbances at given
constants across the tail. `beta` and `gamma` are mutable Params on the
segment, so they retune with `set_value` and no re-apply.

**Cold start** (`drto.cold_start_dynamic`) initializes a model whose
initial condition sits away from the steady state, with no prior solution
and no equilibrium solve: each state runs from its declared initial
condition to its declared target, derivatives hold the profile's slope,
controls hold their targets, the tail rests on the targets, and, with
pyomo-pounce installed, the algebraic variables are solved pointwise from
every equation except the declared dynamics. The default profile is a
straight line; `profile="exponential"` runs a normalized decay that lands
exactly on the target at the horizon's end, with `time_constant` in the
horizon's own units (default a third of the horizon). The guess then
satisfies every equation except the dynamics themselves, which carry the
mismatch the optimizer resolves. On the IDAES CSTR example the
exponential start roughly halves the first solve's iteration count.

`drto.initialize_steady_state(m, controls={...})` is the solve-based
alternative: reduce a throwaway clone to its steady state, solve the
equilibrium with the pyomo-pounce pipeline, and broadcast it flat across
the horizon. Use it when the initial condition is at (or near) a steady
state; use cold start when it is not.

**Assembly**: `drto.dynamic_optimization` applies the declared control
profiles (through `drto.parameterize` and pyomo-cvp) and builds the
objective from every live cost term — the finite-horizon stage cost, the
tail's quadrature group, and the pin penalties — via
`drto.build_objective`. Deactivating a cost row removes its term; there
are no coupling options.

**The advanced-step correction** (`drto.advanced_step_controller`)
turns the solved horizon into a between-samples controller: solve at a
predicted state with pounce (the assembly declared the feedback hooks as
sensitivity parameters, inert metadata for every other solver), write
the measured state into the hooks when it arrives, and ask for the
corrected solution — a backsolve on the kept factorization, not a
re-solve. The returned map holds every variable's corrected value; the
model itself is untouched, so the prediction stays in place as the next
solve's warm start. `gradient=True` returns the controls' sensitivities
to the hooks instead.

**Warm start** (`drto.warm_start_dynamic`) is the closed loop's
initializer: every variable takes the value the previous solution had
one sampling time later, copied where the grids line up (which on a
uniform grid with the mesh rule is almost everywhere, the tail's first
point landing exactly one sample past the horizon's end), interpolated
in the small uncovered sliver, moves shifting as the step functions
they are. Past the end of a tailless horizon, states and controls take
their targets and derivatives zero. Values only, nothing solved, and
the loop sets the initial condition from the measurement.

## Workflow: a forward simulation

`drto.dynamic_simulation` turns the same declared model into an
integration: the declared control profiles are applied, the controls are
fixed at given profiles (or the values they hold), declared disturbances
are fixed at given constants or sequences, the objective is zero, and the
square system integrates forward with any NLP solver. The registry
records what was fixed and the report names it. This is the plant in the
planned closed-loop features: the same declarations, simulated one step
at a time.

## Workflow: the steady-state branch

The steady reduction and the dynamic transforms are sibling branches of
the same declarations — the reduction applies to the *declared* model,
before any dynamic transform:

- `drto.dynamic_to_steady_state` collapses the horizon to a single
  point: accumulations pinned at zero, time-indexed Blocks collapsed to
  their steady member, References re-pointed, Ports intact.
- `drto.steady_state_simulation` is the reduction plus held controls: a
  square equilibrium at given inputs (its `create_using` form leaves the
  dynamic model untouched). The CSTR example computes its setpoint this
  way.
- `drto.steady_state_optimization` is the reduction plus the declared
  economic cost: the RTO problem.

## How-to: scaling

Badly scaled models (an energy holdup at 1e8 J next to mole fractions)
are the rule in process systems. The drto contract is one-sided: **tag
the model once, and every internal solve honors it**. Attach the standard
Pyomo suffix with a factor per badly scaled variable and row:

```python
m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
m.scaling_factor[cv.energy_holdup] = 1e-7
m.scaling_factor[cv.enthalpy_balances] = 1e-7
```

`cold_start_dynamic` and `initialize_steady_state` then run their
internal solves on a scaled clone and write every value back in the
model's own units — the model, its declarations, and its Params never
leave physical units, and nothing changes in the call. Your own solver
calls compose the same way with Pyomo's `core.scale_model`:
`create_using` a scaled clone, solve it, `propagate_solution` back. The
CSTR example tags units-driven factors (every J-valued variable gets
1e-7, every W-valued 1e-6) in a ten-line helper and runs every solve
through it.

## How-to: IDAES flowsheets

drto's transforms are built to leave a flowsheet as IDAES wrote it. Three
idioms cover most models:

**States as member subsets.** An IDAES material holdup is one packed Var
over components, but only some members are true states — the water
holdup of the saponification CSTR is fixed by the property package's
closure. Declare exactly the true states as slices:

```python
drto.state(cv.material_holdup[:, "Liq", "NaOH"],
           cv.material_holdup[:, "Liq", "EthylAcetate"],
           cv.energy_holdup)
```

Each slice wraps as an attached Reference; the undeclared members stay
algebraic, their balances close them pointwise, and the declared surface
matches the true state dimension.

**Controls on Ports.** A feed flow lives on the inlet Port as a
time-indexed Reference; declare it directly:

```python
fin = m.fs.cstr.inlet.flow_vol
drto.control(cv.heat, fin, profile="piecewise_constant")
```

**Time-indexed Blocks.** Property and reaction blocks (`Block(t)`
families) ride through every transform: the steady reduction collapses
them to the surviving member, and the terminal segment replicates their
variables and equations onto the tail. No flattening, no rewriting of
the flowsheet.

The [IDAES CSTR examples](examples.md) run all three on an unmodified
`idaes.models.unit_models.CSTR`.

## Reports

Every initializer returns a printable report of what it did — what was
set, what was solved, what was skipped and why — and every drto error
names the component and the declaration to fix. When something behaves
unexpectedly, look at `drto.info(m)` first, then the report.
