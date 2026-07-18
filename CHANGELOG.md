# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/devin-griff/drto/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/devin-griff/drto/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/devin-griff/drto/releases/tag/v0.0.0
