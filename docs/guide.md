# User guide

drto has one idea: your Pyomo model already contains everything a dynamic
optimization needs, so instead of rebuilding it in a modeling layer, you
declare which components play which role and let transformations rewrite
the model. This page walks the four workflows those declarations drive —
an infinite-horizon controller, a forward simulation, the closed loop
that runs the two against each other, and the steady-state branch — then
the two how-tos that come up on real models: scaling and IDAES
flowsheets.

## The registry

Every declaration records itself in the model's registry, and every
transformation reads the registry rather than re-scanning the model.
`drto.info(m)` returns it; displaying it renders the model in its own
terms — the problem's size at the top, components grouped by role,
indexed constraints in compact symbolic form (one equation per family,
not the per-index expansion `pprint` produces), units where the model
carries them, and the ordered log of applied transformations:

```
<drto registry>
states: 1, controls: 1
declarations:
  horizon: t (ContinuousSet, 11 points)
  states: z (free)
  dynamics: dzdt[t]  ==  - z[t] + u[t]  for t in t
  controls: u (piecewise_constant, free)
  ...
transformations: (none)
```

The size line counts declared states and controls as members at one time
point, so it reads the model's dimension rather than its grid's: a
129-state flowsheet says so on the first line whether the horizon holds
two samples or two hundred.

The registry is stored outside the component tree and survives `clone()`
(and therefore every `create_using`) with its references remapped, so a
transformed clone knows its own declarations.

## The declarations

The control-side surface, in the order a model usually declares them:

- `drto.horizon(t)` — the time `ContinuousSet`, declared before
  discretization so the registry keeps the sample grid.
- `drto.state(z, ...)` — the differential states. A state may be a whole
  time-indexed Var or a member subset of an indexed one (see the IDAES
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
  `drto.tracking_terminal_cost(term)` — cost constraints: equalities
  defining cost variables in the model's own units. A tracking cost
  covers every declared state (the stage cost the controls too) and
  contains nothing else besides Params; a gap or a foreign variable is
  a declaration-time error.
- `drto.move_suppression(move)` — a cost constraint pricing the control
  moves, each member referencing controls at its own sample and the one
  before, the first member against previous-action Params. The steady
  reduction drops it, and the terminal segment keeps it off the tail.
- `drto.initial_condition(ic, ...)` — constraints pinning states at
  the first point to mutable Params, which a receding-horizon loop overwrites
  with each measurement.
- `drto.steady_state(z, z_ss)` / `drto.steady_state_control(u, u_ss)` —
  pair each state and control with the mutable Param holding its target.
  The transforms and initializers read these as the setpoint.
- `drto.terminal_constraint(con)` — an explicit endpoint condition, when
  one is wanted.

The estimation-side surface (`drto.measurement`,
`drto.estimated_parameter`, `drto.estimation_stage_cost`,
`drto.estimation_terminal_cost`, `drto.arrival_cost`) mirrors this for
moving-horizon estimation.

Declarations are checked as they are made, and a bad one raises a
descriptive error at declaration time, not at solve time.

## Checking the DAE index

Declare first, then check: `drto.check_index(m)` tests whether the
declared model is an index-one DAE — whether every algebraic variable
is determined, pointwise, by the algebraic constraints. A model that
fails is higher-index, and its optimization symptoms are indirect;
the report names the variables no algebraic constraint determines and
the constraints left unmatched, states the structural index, and
estimates the condition of the algebra's Jacobian at the current
values. The [gallery notebook](notebooks/check_index.ipynb) walks the
pendulum's index ladder and a flowsheet posed both ways.

```python
report = drto.check_index(m)
print(report)
```

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
from pyomo.contrib.solver.common.factory import SolverFactory

SolverFactory("ipopt").solve(m)
drto.plot_states(m)
drto.plot_controls(m)
```

From a function that builds the model rather than a model, steps 1, 2,
and 4 are one call:

```python
m = drto.dynamic_optimization(build, N=10, infinite_horizon=True)
drto.cold_start_dynamic(m)          # step 3 is still yours
SolverFactory("ipopt").solve(m)
```

`build` returns a declared, undiscretized model, its first two parameters
are `N` and `h`, and every parameter has a default (feature 006). A
constant input is fixed at the declared sample points, which is all an
undiscretized model has, and the call fixes the members `dae.collocation`
adds to it. Reach for the four calls above when the workflow has to touch
the model between them, and for the one call when it does not.

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
`drto.build_objective`. Deactivating a cost constraint removes its term; there
are no coupling options.

**The advanced-step correction** (`drto.advanced_step_controller`)
turns the solved horizon into a between-samples controller: solve at a
predicted state with pounce (the assembly declared the initial-condition
Params as sensitivity parameters, inert metadata for every other
solver), write the measured state into those Params when it arrives,
and ask for the
corrected solution — a backsolve on the kept factorization, not a
re-solve. The returned map holds every variable's corrected value; the
model itself is untouched, so the prediction stays in place as the next
solve's warm start. `gradient=True` returns the controls' sensitivities
to those Params instead.

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
closed loop below: the same declarations, simulated one step at a time.

## Workflow: the closed loop

`drto.ideal_nmpc` runs the receding-horizon loop the declarations
describe, in one call on the declared, discretized model — terminal
segment applied or not, before `drto.dynamic_optimization`:

```python
history = drto.ideal_nmpc(
    m,
    steps=50,
    initial_condition={"z": 0.2},
    disturbances={"w": 0.05},
    seed=0,
)
drto.plot_states(history)
drto.plot_controls(history)
```

The loop builds both sides of the control problem from the one model.
The input itself becomes the controller through
`drto.dynamic_optimization` (a `dynamic_optimization` mapping passes
options through). A clone becomes the plant: its controls are fixed at
the declared targets, `drto.dynamic_simulation` makes it square, and
everything past the first sampling time is removed, so the plant is the
one-sample integration the loop actually solves regardless of the
declared horizon. Ideal means the solve is treated as instantaneous —
measurement, solve, and implementation at the same instant. Each step
solves the controller at the current state, writes the first control
moves into the plant, fixes the plant's disturbances at the step's
realization, simulates one sample, and feeds the end state back as both
models' initial condition.

`initial_condition` maps declared state names to the values written into
the initial-condition Params before the first step; omitted, the Params'
current values are the first state.

Each declared disturbance's `disturbances` entry is either a sequence,
the per-step realization as given, or a number, the standard deviation of
independent zero-mean normal draws, reproducible through `seed`; a
disturbance with no entry is zero. The disturbance enters the plant only
— the controller solves at zero disturbance, the optimization mode's
convention.

`initialize` picks the first solve's initialization: `"cold"` (the
default) runs `drto.cold_start_dynamic` on controller and plant alike, a
mapping passing through as its options; `"steady"` runs
`drto.initialize_steady_state` on the input before the sides are built,
so both inherit the broadcast; `False` skips initialization. Every later
solve is warm-started with `drto.warm_start_dynamic`.

`solver` names the solver for every solve, `"pounce"` the default. Under
`"pounce"` or `"ipopt"`, warm-started controller solves run with a warm
start recipe — `warm_start_init_point=yes`, `mu_init=1e-6`, and `1e-9`
bound and multiplier pushes — and a `warm_start` mapping lays user
options over it, so `warm_start={"mu_init": 1e-4}` retunes one knob
without restating the rest. Under another solver the loop warm starts on
the shifted values alone and a given `warm_start` mapping passes to the
solves as is. The first solve runs at solver defaults. A solve that
fails stops the loop with an error naming the step.

`tee=True` streams every solve's output as the loop runs and keeps it:
`history.logs` holds one `(step, side, text)` entry per solve in loop
order. The default is quiet, nothing streamed, nothing kept.

`scale` takes a `drto.scale` source, `"point"`, `"bounds"`, or a
mapping of unit names to magnitudes, and writes each side's factors
with it once, after that side's assembly and cold start; every solve
receives them, and the history lands in the model's own units. The
default writes nothing and honors a `scaling_factor` suffix the caller
wrote.

The returned `NmpcHistory` holds the actual closed-loop trajectory —
times, each declared state's values, the implemented moves, and the
disturbance realizations, under the declared names and targets — and
`drto.plot_states` and `drto.plot_controls` accept it in place of a
model, drawing the moves as the staircase they physically are.

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
are the rule in process systems. `drto.scale` assigns a factor per
variable group from a source you choose, and a factor per large
constraint from the Jacobian at the model's current point, so it runs
after an initializer has filled the values. `source="point"`, the
default, reads the values themselves, right when the model sits near
its operating magnitudes, the initialized steady broadcast. `"bounds"`
reads declared operating limits, the controls in practice. A mapping
states the process's magnitudes once per physical dimension, which
covers a quantity whose value says nothing, a duty sitting at its zero
target. `drto.scaled_solve` assigns the factors and solves:

```python
drto.initialize_steady_state(m)
res = drto.scaled_solve(m)                             # values as the source
res = drto.scaled_solve(m, source={"J": 1e7, "W": 1e6})  # stated magnitudes
```

The factors reach the solver through the standard `scaling_factor`
suffix, and no second model is built: pounce and legacy ipopt read it
under `nlp_scaling_method=user-scaling`, and ipopt_v2's NL writer
scales the problem as it writes the file. The model, its declarations,
and its Params never leave physical units. The loops take the same
assignment as an option: `scale` on `drto.ideal_nmpc`,
`drto.approximate_nmpc_data`, and `drto.approximate_nmpc_closed_loop`
forwards a source to `drto.scale` once per solved model, after its
cold start, and holds the factors for every solve. A hand-written suffix composes the same way:
write it and the solves receive it.

## How-to: IDAES flowsheets

drto's transforms are built to leave a flowsheet as IDAES wrote it. Three
idioms cover most models:

**States as member subsets.** An IDAES material holdup is one indexed Var
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
