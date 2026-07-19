# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
  model to its steady-state form. Time collapses to a single point, every
  reference to a declared state's derivative is replaced by zero
  (elimination by substitution, no `dz/dt == 0` rows) and the
  DerivativeVars are deleted, the initial condition, terminal constraint,
  and terminal cost leave the model, and a per-sample stage cost becomes
  the single-point cost `build_objective` assembles. Derivative-carrying
  algebraic equations collapse to their quasi-static forms. Applies to
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
  keys), drop the declared stage costs and the steady-state pairings (a
  simulation carries no cost equations and nothing reads the pairings;
  the target Params stay), and install the simulation's zero objective:
  the square fixed-input equilibrium solve. A dynamic model composes the feature 005
  reduction; a model authored directly as steady-state skips it. With
  that, `drto.control` on a model with no declared horizon registers
  without a profile, so a steady-state model declares through the same
  surface; and with a horizon declared, a control not indexed by the time
  set errors at the declaration instead of later inside pyomo-cvp.

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
