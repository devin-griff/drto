## What it is

DRTO is a framework that provides steady-state and dynamic estimation, 
simulation, and optimization from a core Pyomo model, as well as the 
moving-horizon machinery for implementations of dynamic real-time optimization.
The intent is to apply to a broad range of model and problem types, with functionality 
being built incrementally. 

## The mode framework (six modes)

drto is one framework over one declared model, run in any of six modes:
the 2x3 grid of {steady-state, dynamic} by {simulation, optimization,
estimation}. The mode fixes what is free and what the objective is; the
model and (mostly) the declarations are shared.

|  | Simulation | Optimization | Estimation |
| --- | --- | --- | --- |
| Steady-state | solve f(z,u)=0 for z at fixed u | economic RTO: optimize phi(z_ss, u_ss) | data reconciliation: fit z to steady data |
| Dynamic | integrate the ODE forward (IVP) | NMPC / D-RTO over the horizon | moving horizon estimation (MHE) |

- Columns are what the mode does with the degrees of freedom. Simulation
  frees nothing and solves the square model. Optimization frees the
  controls and adds a cost. Estimation frees the states (and parameters)
  and fits them to measurements.
- Rows are the time treatment. Steady-state collapses the model to one
  point with every dz/dt = 0. Dynamic keeps the horizon and the
  discretized dynamics.
- The optimization and estimation columns are duals: one frees inputs to
  minimize a cost, the other frees states to fit data. MHE is the dual of
  NMPC, and steady-state reconciliation the dual of RTO. This is why one
  declaration surface can cover both halves; the seams are the initial
  anchor (a hard condition for control, a soft arrival cost for
  estimation) and the measurement.

## Steady-state reduction (setpoint consistency and economic RTO)

A first-class feature: drto reduces the dynamic model to its
steady-state problem, the same equations with every dz/dt = 0 at a
single time point, solving f(z,u)=0 and g(z,u)=0 for an equilibrium
(z_ss, u_ss). Two uses:

1. **Setpoint consistency.** The tracking target is DERIVED from the
   model, not hand-specified, so the state target and the control
   target are a true fixed point of the dynamics by construction.
   Motivating case (2026-07-13): a hand-typed Hicks CSTR tracking pair
   (xss, uss) that was not an exact equilibrium: the control that
   actually holds xss was v1 = 0.578, not the declared 0.583. The
   controller could not zero both the state-tracking and the
   control-tracking terms at once, so the controls never settled: they
   drifted ~5e-4 at the tail hunting a compromise, at any horizon
   (with a model-consistent uss the tail spread dropped ~100x to
   4e-6). A model-derived setpoint removes the whole failure mode.

2. **Economic RTO.** The same reduced problem plus an economic
   objective phi(z_ss, u_ss) over the steady-state variables, subject
   to constraints: the RTO layer that computes the economically-
   optimal operating point, which the NMPC then tracks (or uses as an
   economic-NMPC terminal reference). This is what makes the D-RTO
   name literal: steady-state RTO is the dz/dt -> 0 limit of the
   dynamic problem, so the dynamic controller and its setpoint
   optimizer are one package.

## Dependency stack

- pyomo + pyomo.dae for modeling and discretization.
- pyomo-cvp for piecewise-constant control parameterization
  (`declare_profile`); the controls' declaration already lives there.
  pyomo-cvp STAYS a standalone package (independently useful for
  offline dynamic optimization, and a Pyomo upstream candidate); drto
  depends on it and folds `declare_profile` into `control(profile=...)`
  so users get one import surface. Revisit only if the warm shift needs to reach inside
  cvp's substitution machinery.
- pyomo-pounce (pounce >= 0.8.0) for the in-process solve session,
  sensitivity gradients, and the `estimate()` fast update (merged as
  jkitchin/pounce#199). The estimation-side machinery (covariance with
  hessian="lagrangian"|"gauss-newton" and active-bound projection)
  merged as #203 on 2026-07-12; it ships in the release after 0.8.0.
- pounce coupling is a hard dependency for v1: simpler and honest about
  what works today. A sensitivity-backend interface (with k_aug as a
  legacy alternative) was considered and deferred.

## Declarations

Explicit declarations throughout, matching the pyomo-cvp / pyomo-pounce
family. USER DECISION: no DerivativeVar introspection for states. The
divergence cases that killed auto-detection: quadrature/cost
accumulators carry DerivativeVars but are not plant states; spatial
derivatives are not temporal states; quasi-steady treatments; discrete
time or hand-discretized models have states but no DerivativeVars; and
no structure implies the measured vs unmeasured distinction MHE will
need.

Each declaration tags a Pyomo component the user already wrote, a Var, a
Constraint, a Param, or a Set (USER DECISION 2026-07-14): the point is to
bolt onto an existing Pyomo model, not to introduce a new modeling
framework. The time set tags a Set; state and control tag Vars; the dynamics, cost,
and boundary declarations tag Constraints; the measurement and steady-state
targets tag Params. The control-scope declaration surface (the
estimation-side surface follows below):

- `horizon(m.t)`: tags the time set, the moving-horizon dimension
  shared by every dynamic mode. It may be a `pyomo.dae` ContinuousSet
  (continuous-time) or a discrete Set (discrete-time, difference equations),
  so drto does not assume continuity (USER DECISION 2026-07-14). Continuous
  time is the current build scope, and the discrete-time path (a discrete Set
  plus the discrete dynamics declaration below) is deferred, kept in this
  design for later (USER DECISION 2026-07-16). Declaring
  it is the root handle for the moving-horizon machinery: t0 and tN come off
  its bounds, the warm shift advances along it, and the discretization lives
  on it. The dynamics are declared (below), not scanned for, so drto does
  not hunt for DerivativeVars; it validates that a dynamics constraint's LHS
  DerivativeVar is taken with respect to this set, which is what separates
  time from a spatial axis in a PDE model.
- `state(m.z1, m.z2, ...)`: tags the state Vars.
  Varargs, indexed-container-aware (one call declares all members). The
  state role is declared; its dynamics are declared separately, by
  `dynamics` below (USER
  DECISION 2026-07-14, revising the earlier auto-pickup-from-DerivativeVar
  plan: declaring the dynamics explicitly is uniform across continuous and
  discrete time and matches the scalar-side convention). A state carries a
  DerivativeVar only in a dynamic model. A model the user built as
  steady-state has states with no derivative, so `state` does not
  require one (USER DECISION 2026-07-16).
- `dynamics(m.ode_con)`: tags the equality Constraint of the state
  dynamics, currently a continuous-time ODE. One side is the DerivativeVar of
  a state (dz/dt),
  read from `con.expr` (either orientation); drto gets the state from the DerivativeVar
  (`get_state_var`) and checks it is a declared state and is taken with
  respect to the declared time set (verified against Pyomo 6.10). That check
  is why `state` earns its place even though the dynamics carry the
  DerivativeVar. USER DECISION 2026-07-14. Renamed from
  `continuous_dynamics` (USER DECISION 2026-07-17).
- The discrete-time dynamics declaration (deferred, see `horizon`) tags the
  equality Constraint of a difference equation. Its defining side is a state
  at the
  next time point (z[k+1]), read the same way; drto gets the state (a plain
  Var, which distinguishes it from the continuous case) and advances it
  along the declared discrete time set. Same object as the continuous case,
  a Constraint; only that side differs (USER DECISION 2026-07-14). Its name is
  undecided and out of scope for now: it may share `dynamics`, since the
  defining side tells the cases apart, or take its own name such as `difference`.
- `control(m.u, ..., profile=...)`: tags the manipulated-input
  Vars, the free decision variables. No `wrt` argument: drto uses the
  declared time set, so the control's parameterization is over that set
  (USER DECISION 2026-07-14). The `profile` flag folds in the
  parameterization: `control` calls pyomo-cvp's `declare_profile`
  automatically. One call declares the control and its parameterization; cvp
  stays the dependency that implements it underneath.
- `tracking_stage_cost(m.tracking_stage_con)`: tags the equality
  Constraint defining the setpoint-tracking running cost per time point (the
  ||z - z_ss|| + ||u - u_ss|| regulation penalty). It is indexed over the
  sample points minus the final time, one member per sample (for example
  `@m.Constraint(sorted(m.t)[:-1])`; the terminal cost owns the final time,
  and a family indexed by the time set itself is rejected, since
  discretization would expand it off the sample grid): one side is the
  scalar cost Var per sample (L[t]), the other the per-sample cost
  definition, and drto sums L[t] over the samples in the objective
  it assembles (USER DECISION 2026-07-14, revising the earlier accumulated
  scalar form; the per-point form is what lets the steady-state reduction
  drop the time index to leave the single-point cost). The tracking targets
  are the declared steady-state Params (`steady_state` /
  `steady_state_control` below), populated by the steady-state/RTO
  solve.
- `economic_stage_cost(m.economic_stage_con)`: tags the equality
  Constraint defining the economic running cost phi(z, u), same per-sample
  form as the tracking stage cost. Its single-point
  steady-state form is the economic-RTO objective; the tracking stage cost
  is not (that is a deviation penalty, zero at the equilibrium). So one
  declaration serves economic NMPC and RTO both (economic NMPC is post-v1;
  RTO uses it in v1). USER DECISION 2026-07-14: split the running cost into
  tracking and economic terms so a mode selects which is live.
- `tracking_terminal_cost(m.tracking_terminal_con)`: tags the
  equality Constraint defining the terminal (Mayer) tracking cost
  V_f(z(tN)), the terminal regulation penalty. Same LHS-scalar convention;
  drto adds the terminal-cost Var to the objective. Dropped in
  steady-state (see the objective note below). Carries the `tracking_`
  qualifier for clarity against the economic term (USER DECISION
  2026-07-14).
- `initial_condition(m.init_con)`: tags the equality Constraint
  anchoring the initial state, LHS the anchored state at t0. If the RHS is
  a mutable Param, that Param is the feedback-injection point drto updates
  each step, so this convention doubles as the state-feedback hook (z_hat)
  for the online loop. Named "condition" deliberately: it pins the state
  (an equality), a different job from a boundary set. This is the exact
  seam where MHE will diverge, since its initial anchor is the soft
  arrival cost, not a hard equality, so a soft mode or a separate
  `arrival_cost` is the estimation follow-on, not this.
- `terminal_constraint(m.terminal_con)`: tags the Constraint
  (equality or inequality) defining the terminal set/region z(tN) in X_f.
  Requirement (USER DECISION 2026-07-14): every Var it references is a
  declared state at tN, the final time; a control at tN is excluded, the
  standard OCP convention (X_f lives in state space). No LHS convention,
  which is what separates it from a path constraint (present at every t).
  "Constraint" not "condition" because it restricts to a set rather than
  pinning a value.
- `steady_state(m.z, m.z_ss)`: pairs a declared state with the mutable
  Param holding its steady-state target z_ss. The tracking costs drive
  toward the targets (z - z_ss); the steady-state/RTO mode populates them
  from its solve (or they are set directly, since they are Params), so the
  target is model-derived rather than hand-typed (the Hicks CSTR lesson
  above). The pairing is what makes the populate step possible: drto knows
  which target Param each solved state value writes into (USER DECISION
  2026-07-17, revising the earlier unpaired bag-of-Params form). One pair
  per call, accumulating; the call returns the target.
- `steady_state_control(m.u, m.u_ss)`: pairs a declared control with the
  mutable Param holding its target u_ss, driven toward the same way
  (u - u_ss) and populated the same way (USER DECISION 2026-07-14; paired
  2026-07-17).

Convention on the declared constraints (verified
against Pyomo 6.10):

- The cost and initial-condition constraints must be equalities; an
  inequality is rejected with a clear error (a cost term or anchor written
  as an inequality is a user mistake).
- The appropriate scalar is read from `con.expr`, which preserves the
  as-written relational expression (Pyomo canonicalizes the body to
  `LHS - RHS`, lower=upper=0). Either orientation of the equality works:
  the sides are checked in turn, and a constraint where neither side is
  the scalar is rejected. For a cost the scalar is the cost-term Var drto
  puts in the objective; for the initial condition it is the anchored
  state at t0.
- Cost variables are left UNBOUNDED (USER DECISION 2026-07-17, best
  practice recorded after measurement). The defining equality fixes the
  value, so a `NonNegativeReals` bound adds no information, and it places
  the optimum exactly on the bound wherever the cost vanishes: settled
  samples on a long horizon, a quadrature state through a tail at
  equilibrium. Interior-point solvers drag badly there (Hicks, N = 50:
  43 iterations bounded vs 6 unbounded, identical solutions; the
  hand-built phi variant hit 82). Same mechanism as the earlier
  segment-copy finding that moved the tail integrand to Expressions.

The objective is drto's, not the user's: it assembles `min` over the
declared cost-term Vars that are live in the current mode, summing a stage
cost's per-point Vars over time and adding a terminal cost's single Var.
Modes add or drop terms by including or excluding a cost's (Var, defining
constraint) pair; in steady-state the reduced model has one time point to
sum and never instantiates the terminal cost, so that term is simply absent. This is the
practical payoff of representing each cost as a Var defined by a
constraint rather than a bare expression: the pair is one handle drto can
find and drop.

USER DECISION 2026-07-18 (amends 2026-07-17): `drto.infinite_horizon` pins
the terminal segment endpoint to the steady state by default. The earlier
decision imposed no terminal condition, treating the singular tail cost as
its own enforcement and the paper's endpoint constraint as proof-only theory.
The paper's operative problem (Dinh et al. 2025, eq. 36) does impose the
endpoint constraint, and it is what pins the unstable modes on open-loop
unstable plants, so the default is now the L1-relaxed soft pin
(`terminal='soft'`), with `terminal='hard'` (eq. 21c) and `terminal='none'`
(the prior behavior) available. A pin reads the declared `steady_state`
targets, so the transform now requires one per state unless `terminal='none'`.

USER DECISION 2026-07-20 (amends 2026-07-18): the `terminal='hard'` option is
dropped before release; `drto.infinite_horizon` offers only `terminal='soft'`
(the default) and `terminal='none'`. A plain-equality pin adds one slack-free
row per state member, which over-determines the tail NLP when state members
outnumber the horizon's control freedoms (the two-column example, 246 state
members against 160 moves, exits with too few degrees of freedom). The eq. 36
L1 penalty is exact, so `terminal='soft'` at a large enough `mu` reproduces the
plain-equality solution without ever removing a degree of freedom.

Naming: fully-written-out, not abbreviated
(`tracking_terminal_cost` not `term_cost`,
`initial_condition` not `init_con`). These are setup-time
calls, never hot-path, so brevity buys nothing; `con` is ambiguous
between condition and constraint, the exact distinction that matters; the
full forms are the OCP literature's own vocabulary, so a reader who knows
the theory maps straight on; and spelling them out forces the precise
concept (condition vs constraint vs set) rather than hiding behind an
abbreviation. The declarations are bare nouns, no `declare_` prefix (USER
DECISION 2026-07-17: the prefix read long and repetitive at every call
site, and the same functions now serve construction-time wrapping, where
a verb reads wrong in `m.z = state(...)`). `declare_time` became
`horizon` in the same pass: `time` would shadow the stdlib module on a
bare import, and horizon is the better word for the role.

Not declared, by design:

- Path constraints: the states' upper and lower bounds, i.e. the Var
  bounds, which drto reads off the model. Not
  a separate declaration.

Moving-horizon data hooks now have homes: the state anchor z_hat is the
mutable Param on the RHS of the initial-condition constraint; the tracking
setpoint is the declared steady-state Params z_ss/u_ss; the measurements
are the mutable Param stream in `measurement` below. Each is
updated each step. Dynamics are declared, not auto-detected: `dynamics`
(LHS a DerivativeVar) for continuous time, and the deferred discrete-time
declaration (LHS the next-step state, name undecided) above.

Shared conventions:

- Each declaration has an explicit call-time form as well (the
  declared/explicit duality established in pyomo-cvp and pyomo-pounce).
- Family conventions locked in pounce#203: varargs on the declarations
  that scale with states and controls, keyword options (e.g. `group=`)
  apply to every component in the call. The steady-state targets are the
  exception: one (owner, target) pair per call (USER DECISION 2026-07-17).
- Every declaration serves two moments (USER DECISION 2026-07-17): tagging
  an attached component registers immediately; a fresh component is
  wrapped, returned for the `m.x = ...` assignment, and registered at
  attachment. The argument is always the component being declared: drto
  never constructs one (`state(m.t)` is a type error, the implicit-
  construction form was considered and rejected). The constraint-role
  declarations also work as decorators, `@drto.dynamics(m, m.t)` taking
  what `@m.Constraint` would. Styles mix per component; prerequisites
  must be declared by the time a declaration registers.

Estimation-side surface (surface designed now,
built with the MHE follow-on). MHE is the dual of the control problem, so
the same conventions hold (each tags a Var or a Constraint; cost
constraints are equalities with the scalar on the LHS; drto assembles the
estimation objective from the live cost-term Vars):

- `estimated_parameter(m.theta, ...)`: tags the Vars for unknown
  model parameters to estimate, constant over the window. Shared with the
  steady-state data-reconciliation mode.
- `disturbance(m.w, ...)`: tags the process-noise Vars w in
  dz/dt = f + w, the free variables the estimator adjusts to reconcile the
  model with the data, penalized by their inverse covariance in the
  estimation stage cost. It is noise, not a manipulated input: unrelated to
  `control`, with no profile parameterization (USER DECISION
  2026-07-14).
- `measurement(m.y_meas, ...)`: tags the measurement Param(s), the
  measured values y_meas that appear in the estimation cost residuals
  (||y_meas - h(z)||). Like the z_hat feedback hook, it is a mutable Param
  drto refreshes each step, here the incoming measurements over the window.
  Nothing else to tag: h(z) is written inline in the cost, so there is no
  output Var or defining constraint (USER DECISION 2026-07-14).
- `estimation_stage_cost(m.est_stage_con)`: tags the equality
  Constraint for the running estimation cost over the window, the
  measurement residual ||y_meas - h(z)|| plus the process-noise penalty
  ||w||, weighted by inverse covariances. LHS-scalar convention.
- `estimation_terminal_cost(m.est_terminal_con)`: tags the equality
  Constraint for the current-time (window-present) term, the current-state
  measurement residual ||y_meas(tN) - h(z(tN))|| with no process noise
  (nothing leads out of the last point), which is why it is a distinct
  terminal term rather than part of the stage sum. This IS a standard MHE
  term (USER correction 2026-07-14).
- `arrival_cost(m.arrival_con)`: tags the equality Constraint for
  the soft prior on the window's initial state, ||z(t0) - z_prior||
  weighted by the arrival-cost inverse covariance. The dual of the
  control-side initial condition, but SOFT (a cost, not a hard equality).
  Its weight is the piece the covariance propagation updates each step
  (Gauss-Newton, the pounce#203 machinery in Follow-on).
