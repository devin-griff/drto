# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The ideal NMPC loop (feature 014). `drto.ideal_nmpc(m, steps, ...)`
  runs the closed loop the declarations describe from the declared,
  discretized model: a clone becomes the process through
  `drto.dynamic_simulation`, cut to the one-sample simulation each
  plant solve integrates, the input becomes the controller through
  `drto.dynamic_optimization`, and each step solves at the measured
  state, implements the first move, and simulates one sample under a
  per-step disturbance realization (sequences as given, or seeded
  zero-mean draws). The first solve is cold-started, the controller and
  the process alike, and every later one warm-started from the shifted
  previous solution; under `pounce` or `ipopt` the warm-started solves
  run with the warm start recipe, a `warm_start` mapping laying user
  options over the documented default. An active
  `scaling_factor` suffix runs the whole loop on persistent scaled
  clones with the history read back in the model's own units.
  `drto.plot_states` and `drto.plot_controls` accept the returned
  history and draw the actual trajectories, moves as the staircase they
  physically are. On the infinite-horizon hicks model the closed loop
  approaches both declared targets monotonically; on the linear test
  model it lands on the target in four samples. The IDAES CSTR notebook
  (`examples/cstr_ideal_nmpc.ipynb`) runs 10 samples of closed loop in
  11 s, the hot start driven onto the setpoint in about three, and a
  second run under additive process noise, one drawn term per state
  equation from the model's `disturbance=True` build (IDAES's
  custom-term hooks writing dM/dt = f + w into the generated balances),
  holds every state in a band around its setpoint.

- Warm start (feature 013). `drto.warm_start_dynamic(m)` reuses the
  previous solution moved one sampling time forward: every variable
  takes the value the previous solution had one step later, copied
  where the grids line up (on a uniform grid with the mesh rule, almost
  everywhere, the tail's first point landing exactly one sample past
  the horizon's end), interpolated in the uncovered sliver, the moves
  shifting as the step functions they are, and the tail re-gridded
  through `t = tN + atanh(tau)/gamma` with the chain rule on its
  derivatives. Past the end of a tailless horizon, states and controls
  take their declared targets and derivatives zero; with a tail the
  previous solution covers the whole problem and the report's fill
  count is zero. Values only; the loop sets the initial condition from
  the measurement. The shift carries the primal solution only
  (gh #36): the solver rebuilds multipliers from a good primal in one
  linear solve, while a carried certificate must match the next
  problem, its active set, and the restarted barrier level at once, so
  it costs more than it saves. The IDAES
  CSTR notebook (`examples/cstr_warm_start.ipynb`) runs one loop
  iteration on a persistent scaled model: the cold solve takes 17
  iterations (the endpoint pin's quadratic term conditioning it), and
  the warm-started one, shifted values
  with the solver's warm-start options at the call site, takes 6.

- The advanced-step correction (feature 012).
  `drto.advanced_step_controller(m)` corrects a pounce-solved horizon to
  the measured state without re-solving: `drto.dynamic_optimization`
  declares the feedback hooks (the initial-condition Params) as pounce
  sensitivity parameters whenever pyomo-pounce is importable, a pounce
  solve keeps the converged factorization, and the correction is a
  backsolve at the hooks' current values, returned as a map from each
  variable to its corrected value with the model untouched. The
  declaration is inert for every other solver. `gradient=True` returns
  the controls' sensitivities to the hooks instead, and unrecognized
  keyword arguments pass through to pounce. Requires a pounce solve;
  without one the call raises pounce's instruction to solve with pounce
  first. Needs pyomo-pounce with the solve-point estimate baseline (the
  upstream fix this feature contributed). The IDAES CSTR notebook
  (`examples/cstr_advanced_step.ipynb`) runs the flowsheet controller
  through a direct pounce solve (the factorization lives with the model;
  the scaling suffix serves the cold start and then deactivates, since
  the solver-facing keys are presolved away) and reads the correction
  against a warm re-solve: an order of magnitude faster with the
  implemented moves within half a percent.

- Cold start (feature 011). `drto.cold_start_dynamic(m)` initializes a
  dynamic model from its declared initial condition to its declared
  steady-state targets: each state on a straight line with its
  DerivativeVar members at the line's slope, controls and a parameterized
  control's moves at their targets, a terminal segment at rest on the
  targets with the pin slacks at zero, and, with pyomo-pounce installed,
  the algebraic variables solved pointwise from every equation except the
  declared dynamics, one block solve with the states and controls held.
  With an active `scaling_factor` suffix the per-point solves run on a
  scaled clone and the values propagate back in the model's own units.
  `profile="exponential"` runs the states on a normalized exponential
  decay instead of the line, landing exactly on the targets at the
  horizon's end, with `time_constant` in the horizon's own units
  defaulting to a third of the horizon.
  Values only, at any stage, no equilibrium solve; a declared state or
  control without its pairing is a descriptive error, and without
  pyomo-pounce the per-point solves are skipped and the report says so.
  The IDAES CSTR notebook (`examples/cstr_cold_start.ipynb`) shows
  the model before and after, the per-point solves working the outlet
  stream, the reaction cascade, and the packed holdup's water member.

### Changed

- The terminal segment's endpoint pin penalty gains a quadratic term
  (gh #37): `mu*(eps_up + eps_lo + eps_up**2 + eps_lo**2)`, both parts on
  the same `mu`. The linear part is the exact L1 pin as before, zero
  slack staying optimal whenever the pin can hold; the quadratic part
  makes the pin's multipliers unique and continuous, where the bare L1
  kink left them an interval the solver picked arbitrary points of, so
  consecutive solves of near-identical problems disagreed wildly on
  them and warm starts that carry multipliers were a lottery. On the
  IDAES CSTR the same warm-started hand-off went from anywhere between
  6 and 333 iterations to 6, and the cold solve improved from 84 to 59
  (degraded-environment measurements, to be re-verified).

- The terminal segment's algebraic copies (the flat algebra, the Block
  members, the packed residue members) are indexed over the interior
  collocation points only (gh #32): every point that exists is one a
  replicated equation determines, so no dead boundary member exists to
  hold a stale value for a reader to find. State copies keep the full
  tau set for their continuity and discretization rows, as the tail
  cost's quadrature always has. The transform also records these copies
  and the segment itself (with gamma) in the registry pairing (gh #27),
  which the warm start reads. On the IDAES CSTR, the warm-started
  second solve drops from 21 iterations to 10, beating a fresh cold
  start at the same state, once the shift stops reading stale boundary
  values.

- The IDAES saponification CSTR moved into the canonical models
  (`examples/models/idaes_cstr.py`), scaling included: the flowsheet
  builders, the declaration surface, the units-driven `tag_scaling`, and
  `scaled_solve`. The `idaes_cstr` and `cstr_cold_start` notebooks import
  the model instead of defining it inline, with identical results.

- The documentation is rebuilt around the workflows. The README and the
  docs front page open with a runnable quickstart (declare, transform,
  cold start, solve, plot, with real output); the user guide is
  restructured as the three workflows (the infinite-horizon controller,
  the forward simulation, the steady-state branch) plus scaling and
  IDAES how-tos; the thirteen example notebooks render into the docs
  from their committed outputs behind a curated gallery; the API
  reference is grouped by use; and a design page links the feature
  specs, stating plainly what is shipped versus specified.
- The infinite-horizon transformation records, internally on the
  registry, which terminal-segment component belongs to which
  declaration, and `cold_start_dynamic` and the plotting read that
  pairing instead of reconstructing component names (gh #27). The
  registry view renders nothing new, component names are unchanged, and
  a clone carries the pairing with its references remapped.
- `initialize_steady_state` honors an active `scaling_factor` suffix: the
  pipeline runs scaled and the solved values land in the model's own
  units, with no change to the call (gh #24). Previously the suffix was
  ignored, its entries riding the reduced clone into the subsystem solves
  with NL-writer warnings; the scaled path removes both.

## [0.4.0] - 2026-07-26

### Added

- The IDAES CSTR example notebook (`examples/idaes_cstr.ipynb`): a
  saponification CSTR taken straight from `idaes.models.unit_models`,
  declared through the drto surface with no changes to the flowsheet and
  driven in two solves. The true states declare as member-subset slices of
  the packed holdup, the water member staying algebraic (#20); the
  manipulated variables are the jacket duty and the feed flow, the flow
  declared on the inlet Port's time-indexed Reference; the setpoint comes
  from the declarations, `drto.steady_state_simulation` collapsing the
  flowsheet to its feed-alone equilibrium (feature 021); the initial
  condition takes no solve, the setpoint composition knocked back along
  the reaction stoichiometry and hot; `drto.initialize_steady_state`
  broadcasts the equilibrium across the horizon before the transforms; and
  the controller is the infinite-horizon problem (feature 020). Every
  solve runs through a units-driven scaled clone, and the results read
  back with the package plotting (feature 022).
- Registry-aware plotting in the package (feature 022): `drto.plot_states`,
  `drto.plot_controls`, and `drto.plot_stage_cost`, moved from
  `examples/plotting.py`. Everything draws from the registry: the sample
  grid, the declared components, the steady-state pairings as dotted
  setpoint lines, and the `drto_ih` terminal segment mapped back to real
  time through `t = tN + atanh(tau)/gamma` as open points. With no
  selection every declared component draws, a multi-index component
  expanding to one panel per member up to a cap; controls draw as a
  staircase on the finite horizon, each move held over its sampling
  interval. matplotlib rides behind `attempt_import` and a `plot` extra,
  so the base import stays clean and a plot call without it raises with
  the install instruction. The example notebooks import from the package
  and `examples/plotting.py` is gone.
- Time-indexed Blocks in the steady-state reduction (feature 021).
  `drto.dynamic_to_steady_state` now collapses a `Block(time)` family to its
  single steady member: the `t=0` member stays as written, values, bounds,
  units, and fixed status untouched, and the other members leave the model
  with their contents. A time-indexed Reference is a view, not a variable:
  it collapses to a view of the surviving member (a Port entry) or of the
  collapsed Var (an IDAES `heat_duty`), never to a fresh independent Var,
  and Ports keep pointing at their referents. Previously the member Blocks
  survived at every time point and the reduced IDAES CSTR came out broken,
  284 free variables against 95 active constraints; it now reduces to the
  steady system, `drto.steady_state_simulation` leaves it square, and pounce
  solves it to the same equilibrium as a hand-built `dynamic=False`
  flowsheet. Nested time-indexed Blocks and Blocks indexed beyond time are
  rejected with descriptive errors.
- Time-indexed Blocks in the terminal segment (feature 020).
  `drto.infinite_horizon` now treats a variable inside a `Block(time)` member
  as time-varying: discovery climbs from the variable to its parent Blocks,
  the member components get segment copies exactly as flat ones do, and the
  equations inside the members replicate as algebraic families, collected to
  a fixpoint. On the dynamic IDAES CSTR the transform previously wired ~330
  segment references into the single `properties_out[0.0]` member and
  destroyed 69 degrees of freedom; it now replicates the property blocks,
  leaves zero references into the main model, and ipopt solves the
  transformed flowsheet to optimality. Nested time-indexed Blocks, members
  indexed beyond time, and indirect children are rejected with descriptive
  errors.
- The tail handles declared disturbances (feature 020). `infinite_horizon`
  takes a `disturbances` option mirroring `drto.dynamic_simulation`: each
  declared disturbance's segment copy is fixed at the given constant, default
  zero, so the tail continues under nominal disturbance unless told
  otherwise. A dict gives one constant per non-time index (a multi-component
  feed), and a disturbance declared as a time-indexed Reference into Block
  members routes identically to a flat Var. Previously a declared disturbance
  referenced by a replicated equation errored as an orphan copy. Disturbances
  are zero-mean noise; an absolute given input needs no declaration at all: a
  fixed variable's segment copy is fixed at the horizon-end value, so an
  IDAES feed or a fixed volume carries onto the tail as itself.

### Fixed

- A state may be declared as a member subset of a packed Var (#20). A
  slice like `holdup[:, "Liq", "NaOH"]` passed to `drto.state` wraps as an
  attached time-indexed Reference, so a packed Var's algebraic member (an
  IDAES water holdup, constant by the property package's closure) stays
  undeclared and the declared surface matches the true state dimension.
  Classification resolves by data identity everywhere: the dynamics and
  initial-condition declarations accept rows on covered containers, the
  steady-state pairing takes the same slice, the reduction pins every
  accumulation of a covered container at zero (steady state is steady for
  the residue, which closes its row at the point), and the terminal
  segment builds one family per declared state, copies the residue
  members per member over just the referenced combos, and replicates the
  residue balance rows as written. Family-level declarations are
  unchanged.
- A Reference-declared control gets one segment family (#18). The terminal
  segment classified Block members against declared controls by component
  identity, which only ever matches flat declarations, so a control
  declared as a time-indexed Reference into Block members (the IDAES inlet
  idiom) split in two on the segment: an algebraic member copy carrying
  the replicated equations and the cost, and a vestigial control copy
  connected to nothing but its own profile rows, parked wherever the
  solver left it. Members now route to their declared control by data
  identity, the map feature 020 built for disturbances, so the control's
  own copy carries the tail and the shadow family is gone: 15 activated
  variables fewer on the two-control IDAES CSTR with the constraint count
  unchanged, and the solution identical.
- `initialize_steady_state` broadcasts a Reference-declared control. The
  collapsed copy of a control declared as a time-indexed Reference is a
  container even over its single member, and the dynamic path's broadcast
  read `.value` on the container and raised
  (`AttributeError: 'IndexedVar' object has no attribute 'value'`). The
  broadcast reads the member.
- The simulation modes resolve component keys before their rebuilds.
  `steady_state_simulation`'s reduction and `dynamic_simulation`'s profile
  application both replace the very components the `controls` and
  `disturbances` mapping keys point at, detaching them; a detached
  component's name degrades to its local name, so a control below the top
  level errored as undeclared. Both modes now resolve the names at entry
  while the keys are still attached. The docstrings' `controls={m.u: 0.3}`
  example only ever worked because a top-level local name equals its full
  name; the IDAES CSTR's `control_volume.heat` exposed the defect.
  `dynamic_optimization` and `steady_state_optimization` take no such
  mappings and are unaffected.
- The terminal segment carries the declared model's units (#10). The segment
  copies of states, controls and algebraics, the segment derivatives, and the
  soft-pin slacks were all built unitless, so every replicated equation on a
  unit-carrying model stopped being dimensionally consistent, which feature
  019 now makes visible as `(inc)`. Each is built with its source component's
  units; tau is dimensionless, so a derivative over it takes the state's own
  units, and a unitless model is unchanged. The segment cost is an Expression
  and always carried its units. Same defect class as pyomo-cvp#1.


- The registry renders quietly and symbolically on IDAES models (#11). Pyomo
  logs an ERROR on its way to raising when a rule cannot be re-run for the
  compact symbolic form; the failure is expected and handled, so the logger is
  quieted for just that call, and a correctly rendered registry no longer sits
  under blocks of red. The fallback rendering also swaps the member's index
  coordinates everywhere they appear bracket-delimited, reaching indexes on
  other components (`properties_in[0.0].flow_vol` renders as
  `properties_in[t].flow_vol`), and picks the member whose coordinates are
  most distinctive rather than the first, so a `t=0` coordinate no longer
  collides with unrelated `[0]` indexes.

### Added

- Registry units (feature 019): every registry row shows its units, read from
  `pyo.units.get_units`. Variables and parameters annotate their existing
  parenthetical notes (`free`, `piecewise_constant`, ...); constraint rows
  carry a parenthetical after the `for` clause with what the equation balances
  in (`mol/s` for a material balance, `W` for an energy balance). Dimensionless
  renders as nothing, so unitless models display exactly as before. A
  constraint body whose units are inconsistent renders `(inc)`, display only:
  it flags a dimensionally inconsistent equation, which solves without
  complaint, in the view already being checked. Units render compact where an
  exact conversion exists (`J`, `W`, `Pa`, `N`, `s`), keeping declared scales
  (`kJ` stays `kJ`) and declared forms (`mol/s` stays `mol/s`).

## [0.3.0] - 2026-07-25

### Added

- `drto.steady_state_optimization` (feature 009): economic RTO. Reduces the
  model to a single point with the declared controls free and optimizes the
  economic objective over them, giving the optimal steady operating point. It
  requires `state`, `control`, and an economic stage cost; `horizon` and
  `dynamics` are optional, so it runs on a dynamic model (composing the
  feature 005 reduction) or on one authored directly as steady-state. Unlike
  the simulation modes it keeps its cost equations, and a declared tracking
  stage cost is kept rather than dropped, since it regularizes the economic
  optimum toward a known operating point; with both cost kinds declared,
  `tracking_weight` scales the tracking side (default 1, the economic cost
  never scaled). The estimation declarations are neutralized before the
  reduction and the disturbances fixed at zero, which matters more here than in
  a simulation: a disturbance left free would be a decision variable the
  optimizer exploits, optimizing the operating point against fictitious noise.
  The steady-state pairings are left
  intact, since they are the record that makes a later write-back possible;
  that write-back is an algorithmic step outside the transform.

- `drto.dynamic_simulation` (feature 007): the dynamic simulation mode. It
  frees nothing, so the solver is handed the square forward integration of the
  declared model over the horizon. The declared control profiles are applied
  first, so the simulated input takes the shape the model declared, then the
  parameterized controls are fixed. The `controls` option sets what they are
  fixed at, a constant held across the horizon or one value per free point the
  profile leaves; a control not named there holds the value it already has,
  and one holding no value errors rather than being fixed at nothing. A
  simulation carries no cost and no terminal set, so the declared stage costs,
  terminal cost, and terminal constraint leave the model as in
  `drto.steady_state_simulation`, and `build_objective` installs the
  constant-zero objective. `initial_condition`
  is required: a forward integration is not square without the initial state
  pinned. The estimation costs and measurements are neutralized through the
  shared routine, and each disturbance is fixed at its realization, the same
  way the controls are fixed, through a `disturbances` option (default zero),
  so the plant can be driven by a supplied noise sequence.

- `drto.dynamic_optimization` (feature 006): the dynamic optimization mode,
  NMPC and D-RTO. It assembles the horizon optimization from the
  declarations, so on a discretized model it replaces the
  `drto.parameterize` and `drto.build_objective` pair with one call. The
  declared controls stay free and are parameterized by their declared
  profiles, the horizon is kept, and the objective is assembled from the live
  cost terms as the final step. With both a tracking and an economic stage
  cost declared, `tracking_weight` scales the tracking side (default 1; the
  economic cost is in currency units and is never scaled).
  `drto.infinite_horizon` applies before it, since the objective is built
  here and the tail's cost group must be registered by then. A model that
  also carries the estimation declarations still yields a clean control
  problem: the estimation costs and the measurement Params are deleted, a
  disturbance is fixed at zero and kept in the model (so it works however the
  noise enters the equations, and the solver folds a fixed Var in as a
  constant), and an estimated parameter is fixed at the value it holds and
  keeps its registry record, since it stays a live coefficient.

- The estimation declarations (feature 018): `estimated_parameter` and
  `disturbance` (varargs over Vars), `measurement` (varargs over mutable
  Params), and the estimation cost constraints `estimation_stage_cost`,
  `estimation_terminal_cost`, and `arrival_cost`. They ride the feature 002
  surface: tag, wrap, or (for the cost constraints) decorate, with the
  registry recording each. `estimated_parameter` is constant over the window
  and needs no horizon (it also serves steady-state data reconciliation);
  `disturbance` and `measurement` are indexed by the declared time set, and
  `disturbance` takes a piecewise-constant `profile` like `control`, its
  control-side dual, so a simulation drives it at one realization per sample;
  the cost constraints reuse the stage-cost and scalar-cost machinery, so
  `tracking_terminal_cost` now routes through the shared scalar-cost helper.
  Declaration surface only: the estimation mode transforms are a follow-on.

- `drto.initialize_steady_state` (feature 010): initialize a model from
  its steady state. A steady-state model (authored, or a feature 005
  reduction) initializes in place through pyomo-pounce's fill, project,
  block-solve pipeline with the declared controls as the decisions; a
  discretized dynamic model reduces a throwaway clone, solves there, and
  broadcasts the equilibrium flat across the grid with the derivatives at
  zero, returning a printable report with the broadcast counts. Runs
  before the dynamic transforms, which carry the values forward. A
  non-square steady system raises, naming the unmatched variables and
  constraints. pyomo-pounce is optional: the `pounce` extra
  (`pip install drto[pounce]`), imported at call time.

- `drto.dynamic_to_steady_state` (feature 005): reduces a declared dynamic
  model to its steady-state form. Time collapses to a single point, each
  declared state's derivative collapses with it and is fixed at zero, the
  initial condition, terminal constraint, and terminal cost leave the model,
  and a per-sample stage cost becomes the single-point cost
  `build_objective` assembles. The derivative is fixed rather than
  eliminated, so the declared dynamics and any derivative-carrying algebraic
  equation keep their form as written, with `dz/dt` pinned; there are still
  no `dz/dt == 0` rows, and the solver folds a fixed Var in as a constant.
  Pyomo cannot hold a DerivativeVar outside a ContinuousSet, so the
  collapsed derivative is a plain scalar Var of the same name. Applies to
  the declared or discretized model, before any drto transformation (the
  steady reduction and the dynamic transforms are sibling branches of the
  same declarations); on a discretized model the discretization artifacts
  are discarded, and the reduction gives the same steady system either
  way. The refreshed control
  records drop their profile annotation: a single-point control has no
  profile.
- `drto.steady_state_simulation` (feature 008): reduce to steady state,
  fix the declared controls (at supplied values or the values they hold,
  components resolving by name so `create_using` accepts source-model
  keys), shed the optimization-only constructs (the stage costs, the terminal
  cost, and the terminal constraint) and neutralize the estimation
  declarations through the routines the dynamic simulation shares, fix each
  disturbance at a standing value (a `disturbances` option, default zero) for
  the equilibrium under a persistent disturbance, keep the steady-state pairing
  records (the target Params stay and may appear in a deviation-form model's
  equations), and install the simulation's zero objective: the square
  fixed-input equilibrium solve. A dynamic model composes the feature 005
  reduction; a model authored directly as steady-state skips it. With
  that, `drto.control` on a model with no declared horizon registers
  without a profile, so a steady-state model declares through the same
  surface; and with a horizon declared, a control not indexed by the time
  set errors at the declaration instead of later inside pyomo-cvp.

- `drto.infinite_horizon` now pins the terminal segment endpoint to the
  declared steady state, the paper's endpoint constraint (Dinh et al. 2025).
  The `terminal` option selects `'soft'` (the default: the eq. 36 endpoint
  relaxed by an L1 penalty of weight `mu`, a new option, default 1000) or
  `'none'` (no pin). A pin requires a `drto.steady_state` target for every
  state.

### Changed

- **Breaking.** `drto.infinite_horizon` names its block `drto_ih` rather than
  `drto_infinite_horizon`. The block is user-facing, since the segment's
  states, controls and pins are read off it, so a model that referenced
  `m.drto_infinite_horizon` must now say `m.drto_ih`. The short name also cuts
  the prefix repeated on every component in `pprint()` output and in the
  `cost_group` rows of the registry display.


- The stage-cost declarations (`tracking_stage_cost`, `economic_stage_cost`)
  accept a scalar Constraint on a model with no declared horizon. A
  steady-state model has no sample grid to index over, so its cost is a single
  point, the same accommodation `control` already makes and the shape the
  steady-state reduction produces. An indexed cost without a horizon is
  rejected. Without this a model authored directly as steady-state could not
  declare the cost `drto.steady_state_optimization` requires.

- `drto.infinite_horizon` defaults to `terminal='soft'`, so it now imposes
  the endpoint steady-state constraint by default. Pass `terminal='none'`
  for the previous behavior (no terminal condition; the singular tail cost
  is the only terminal enforcement).

## [0.2.1] - 2026-07-18

### Added

- Two canonical example models with their two-case notebooks: the
  cart-pole (`examples/models/cart_pole.py`), the unstable-equilibrium
  example, four states and one force input stabilizing the upright point;
  and the binary distillation column
  (`examples/models/binary_column.py`), the mid-size DAE, a faithful
  translation of the Dinh et al. (2025) 42-tray methanol/n-propanol
  model keeping the index-reduced energy balance that references dx/dt
  inside the algebraic equations. Reference data solved from the
  original model; `initialize.py` gains the binary column helper.

### Changed

- `drto.infinite_horizon` replicates algebraic equations that reference a
  declared state's derivative (the index-reduced energy-balance case): the
  reference maps to the segment derivative with the dilation factor, the
  same rewrite the dynamics get. Previously such equations were rejected.
  Models without them produce byte-identical solver input.

## [0.2.0] - 2026-07-18

### Added

- `plot_stage_cost` in the examples' `plotting.py`: the tracking stage
  cost panel, finite values at the samples, tail values from the
  replicated cost Expressions, and a dotted line at zero, the tracking
  cost's settling value. Every example notebook includes it.

### Changed

- `drto.info` templatizes scalar constraints too, folding their internal
  set sums into symbolic `SUM(...)` form: the double column's terminal
  cost row renders in one line instead of the 246-term expansion. Free
  indices take the rule's own argument names (`dM1[i,t] ... for i in
  tray, t in t`, matching the model as written), the internal sum indices
  get names too, a family whose rule cannot templatize renders its
  representative member symbolically instead of at a concrete index, and
  a stage cost's sample-list index renders as its defining expression,
  `sorted(t)[:-1]`.

## [0.1.2] - 2026-07-17

### Changed

- `drto.infinite_horizon` builds the segment without repeated work: the
  time-substitution map the replication rules hand to
  `replace_expressions` is cached by its two time points instead of being
  rebuilt per constraint member, and the segment control copies go to
  pyomo-cvp as one list call (one substitution pass) instead of one call
  per control. The transformed model is unchanged (byte-identical solver
  input on the double column); the transform drops from 29 to 3 seconds
  there. The list call requires pyomo-cvp >= 0.7.0.

## [0.1.1] - 2026-07-17

### Added

- The double column DAE example: the declared two-column model
  (`examples/models/double_column.py` and its reference data), the two-case
  example notebook, and an initialization helper (`examples/initialize.py`)
  that ramps the states from the initial condition to the steady state and
  computes the algebraic and cost variables from the model's own equations.

### Changed

- `drto.infinite_horizon` imposes no terminal condition: the equilibrium
  constraints at the segment endpoint are removed. The quadrature weights
  are singular there, so the cost itself enforces settling, and the
  removal restores the correct degree-of-freedom count for models with
  many states. Algebraic equations replicate at the interior collocation
  points only.
- `drto.infinite_horizon` does not re-declare control profiles or pass
  `final_node`: pyomo-cvp 0.6.3.1 resolves control references by what
  contains them, so equations at the linking time take the last move with
  no convention to flip. Requires pyomo-cvp >= 0.6.3.1.
- `drto.infinite_horizon` handles states with extra index sets and DAE
  models: algebraic variables and equations are discovered structurally
  (no declaration) and replicated on the segment.
- The initial-condition and terminal-constraint validations handle states
  indexed by time plus other sets.
- `drto.infinite_horizon` deactivates a declared tracking terminal cost:
  the tail integral is the cost-to-go, so V_f would double-count. Recorded
  in the transformation outcome.
- The example models (`examples/models/`) include a tracking terminal cost:
  the stage cost with the controls removed, at the final time.

### Fixed

- `drto.infinite_horizon` no longer fails on a LAGRANGE-LEGENDRE-discretized
  horizon: pyomo.dae's continuity equations are discretization artifacts,
  not algebraic equations to replicate.
- A variable copied to the segment with no replicated equation involving it
  now errors, naming the variable, instead of solving with a silently free
  variable in the tail.
- An invalid `profile` errors before the model is touched, not midway
  through the segment construction.
- A stage cost indexed by the time set itself passed declaration when its
  members skipped the final time, then expanded to every collocation point
  at discretization, dragging the cost off the sample grid.
  `tracking_stage_cost` and `economic_stage_cost` now reject a family
  indexed by the time set outright.

## [0.1.0] - 2026-07-17

### Added

- `drto.parameterize` (feature 017): applies the declared control profiles
  by delegating to pyomo-cvp's declaration-mode transform, refreshes the
  registry to the replacement components, and records itself in the
  transformation log.

- `drto.infinite_horizon` (feature 004): the terminal segment of Dinh et al.
  (2025). Segment copies of the declared states and controls, dilated
  dynamics at interior Gauss-Legendre points, hard equilibrium endpoint, the
  tracking stage cost replicated as the tail integrand, and the tail cost as
  explicit Gauss-weighted terms, `(beta/dt)*phi_f`, registered as a cost
  group for `build_objective`. `beta` and `gamma` are mutable Params,
  symbolic in dynamics and weights; `gamma` defaults to the mesh rule.

### Changed

- `drto.infinite_horizon` replicates the stage cost as named Expressions
  rather than a cost Var with defining constraints: the tail adds no
  variables or constraints, and no bounded intermediates sit at their bound
  as the tail cost vanishes (192/154/66-iteration solves drop to 177/139/8
  on the Hicks study).

- `horizon` captures the sample grid (the ContinuousSet's initialized
  points) and requires an undiscretized set with at least two points; the
  stage-cost sum in `build_objective` runs at the samples, keeping the finite
  horizon commensurate with the infinite-horizon tail.

- `drto.build_objective` (feature 003): one routine owns objective
  installation. Default assembles the live registered cost groups by their
  weights (stage costs per active member, terminal cost, and generic
  registered `cost_group` records); `zero=True` is the marked simulation
  outcome. Also registered as `TransformationFactory('drto.build_objective')`.

- The declaration surface (feature 002), bare nouns: `horizon`, `state`,
  `dynamics`, `control` (profile via pyomo-cvp),
  `tracking_stage_cost`, `economic_stage_cost`,
  `tracking_terminal_cost`, `initial_condition`,
  `terminal_constraint`, and the paired targets
  `steady_state(m.z, m.z_ss)` and `steady_state_control(m.u, m.u_ss)`.
  Each function serves tagging (an attached component registers
  immediately) and wrapping (a fresh component is returned for the
  `m.x = ...` assignment and registers at attachment), and the
  constraint-role declarations double as decorators
  (`@drto.dynamics(m, m.t)`). Each validates its convention (either
  orientation of the equality), enforces the arity and re-declaration rules,
  and records in the registry.

- `drto.info` (feature 001): the per-model registry. Records declarations by
  kind and an ordered transformation log, backed by `Block.private_data` so it
  survives `clone()`/`create_using` with remapped component references, and
  renders a drto-aware view (console and notebook) with indexed constraints in
  compact symbolic form.

## [0.0.0] - 2026-07-14

### Added

- Repository scaffolding and the PyPI name reservation. Design phase: the
  declaration framework and the six modes are recorded in DESIGN.md and the
  README. No functionality yet.

[Unreleased]: https://github.com/devin-griff/drto/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/devin-griff/drto/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/devin-griff/drto/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/devin-griff/drto/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/devin-griff/drto/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/devin-griff/drto/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/devin-griff/drto/releases/tag/v0.0.0
