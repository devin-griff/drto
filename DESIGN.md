# drto: design record

Status: design only, no code. This document records the decisions made
during the design discussions of 2026-07-11/12, before implementation.

## What it is

drto (dynamic real-time optimization): a receding-horizon control loop
for pyomo.dae models, with advanced-step NMPC as its headline
capability. The name is the field's own term (the D-RTO literature):
dynamic real-time optimization is the umbrella over both halves of the
loop, control (NMPC) now and estimation (MHE) as the follow-on. The
advanced-step update is an acceleration mode of the loop, not a separate
controller, so the package serves plain-NMPC users who never touch
sensitivities and differentiates on the mode nobody else has.

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

The six declarations below are the dynamic-optimization cell's surface;
the other cells reuse the same model with pieces dropped (no cost for
simulation) or added (arrival cost and measurement for estimation).
Steady-state reuse is the reusable-object mechanism under "Steady-state
reduction."

On the roadmap this splits cleanly: the optimization column is the
near-term half (dynamic NMPC/D-RTO plus steady-state RTO via the
reduction), the estimation column is the follow-on half (MHE and its
steady-state dual, data reconciliation), and simulation is largely free in
either row once the model is in place (drop the cost, then solve or
integrate).

Orthogonal to this grid is the online-execution axis (ideal, real-time,
advanced-step), documented below under "Execution variants": those are how
the dynamic-optimization loop runs against a live plant, not additional modes.
The "mode 1/2/3" numbering used elsewhere in this document refers to that
execution axis.

## Why the niche is open

A verified landscape review (2026-07-11) found:

- IDAES Caprese ships but is frozen (no substantive commit since 2024,
  not a regular importable package on current main), does re-solve-per-step
  NMPC only, and its advertised MHE never existed.
- pyomo.contrib.mpc (Parker, in Pyomo core, maintained) is deliberately
  loop-less: data structures and model interfacing only. Its author's own
  example repo hand-writes the NMPC loop.
- No maintained sensitivity-based / advanced-step NMPC exists anywhere in
  the Pyomo ecosystem. The Biegler lineage prototyped it three times
  (Thierry's cappresse, an unused enum in Caprese, Lin's example scripts)
  and packaged it zero times. The only maintained open-source
  advanced-step implementation anywhere is acados (CasADi/C, embedded).
- The prior attempts all depended on k_aug/sIPOPT binaries that are
  dormant upstream. This package's sensitivity backend (pyomo-pounce,
  pounce) is actively maintained, which addresses the one structural risk
  the review identified.

## Execution variants: ideal, real-time, advanced-step

These three are the online-execution axis of dynamic optimization (how the
one loop runs against a live plant when the solve takes real time),
orthogonal to the six-mode grid above, not additional modes. The "mode
1/2/3" numbering elsewhere in this document refers to these.

1. Ideal NMPC: solve at the measurement, apply as if the solve were
   instantaneous. The simulation-study mode.
2. Real-time NMPC: honest about solve delay; the previous input holds
   while the solver runs.
3. Advanced-step NMPC: solve ahead at the predicted state; when the
   measurement arrives, correct the solution instantly with a sensitivity
   update (pyomo-pounce `estimate()` on the held KKT factorization) and
   apply; the full solve for the next step proceeds in the background.

The three-mode comparison on one problem (disturbance rejection with
visible solve delay) is the package's own killer demo.

## Problem-class scope (decided 2026-07-12)

- Scope 1: NLP. Solved in-process with pounce; all three modes.
- Scope 2: MILP. Same overall loop architecture (declare, shift,
  inject measurement, solve, apply) with different system functions
  underneath: any Pyomo MILP solver (HiGHS, Gurobi, CP-SAT via
  pyomo-cp) instead of the pounce session. The sensitivity machinery
  does not apply to MILP, so scope 2 gets modes 1 and 2 only; mode 3
  is NLP-only. This is understood and accepted.
- Architectural commitments this imposes NOW: the loop is
  solver-pluggable, and the held-KKT session is an optional attachment
  to the loop, not its spine.
- The MILP scope is the economic-dispatch and scheduling use case
  (rolling re-solves with commitment decisions, the rolling-intrinsic
  pattern), which is also where the package meets power-market
  dispatch problems.
- Beyond scope 2, noted as future patterns only: MINLP mode 3 via
  fix-the-integers-and-differentiate on the continuous subproblem
  (integer-solution change = fall back to the background solve, the
  same fallback philosophy as the active-set warning), and Sager-style
  relax-and-round for integer controls, which would live beside
  pyomo-cvp's profiles as a transformation, not inside the loop.

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

MECHANISM (USER DECISION 2026-07-13): reuse the user's modeling
object for multiple purposes rather than doing expression surgery on
the discretized constraints. The user writes the dynamics once, in a
reusable form (a rule/function defining the RHS and algebraic
relations); drto instantiates that same object for both the dynamic
(discretized) problem and the steady-state (single-point, derivatives
fixed to zero) problem, and for the economic RTO on top of it.
Rejected: introspect-and-substitute (clone the discretized model and
pin each declared state's DerivativeVar to zero). The reusable-object
route is explicit, matches the family's explicit-over-introspection
philosophy (the same reason declare_state rejected DerivativeVar
introspection), and avoids fragile surgery on transformed constraint
bodies. Consequence to design deliberately: the drto model-building
API must let the user express the dynamics so both the dynamic ODE
constraints and the steady-state constraints are built from the one
source of truth.

## Dependency stack

- pyomo + pyomo.dae for modeling and discretization.
- pyomo-cvp for piecewise-constant control parameterization
  (`declare_profile`); the controls' declaration already lives there.
  pyomo-cvp STAYS a standalone package (independently useful for
  offline dynamic optimization, and a Pyomo upstream candidate); drto
  depends on it and re-exports the declarations so users get one
  import surface. Revisit only if the warm shift needs to reach inside
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
Constraint, or a Param (USER DECISION 2026-07-14): the point is to bolt
onto an existing pyomo.dae model, not to introduce a new modeling
framework. State and control tag Vars; the cost and boundary declarations
tag Constraints; the measurement and steady-state targets tag Params. The
control-scope declaration surface (the estimation-side surface follows
below):

- `declare_state(m.z1, m.z2, ...)`: tags the differential-state Vars. Varargs,
  indexed-container-aware (one call declares all members). drto then picks
  up each state's dynamics automatically from its DerivativeVar (USER
  DECISION 2026-07-14, good-enough starting point; no `declare_dynamics`).
  This is NOT the rejected state auto-detection: the state role is still
  declared, and the DerivativeVar only locates the ODE of an
  already-declared state.
- `declare_control(m.u, ..., wrt=m.t, profile=...)`: tags the
  manipulated-input Vars, the free decision variables. The `profile` flag folds in the
  control's parameterization: `declare_control` calls pyomo-cvp's
  `declare_profile` automatically (USER DECISION 2026-07-14). One call
  declares the control and its parameterization; cvp stays the dependency
  that implements the parameterization underneath.
- `declare_tracking_stage_cost(m.tracking_stage_con)`: tags the equality
  Constraint defining the setpoint-tracking running cost (the
  ||z - z_ss|| + ||u - u_ss|| regulation penalty). LHS a lone scalar Var;
  drto adds it to the objective it assembles. The tracking targets are the
  declared steady-state Params (`declare_steady_state` /
  `declare_steady_state_control` below), populated by the steady-state/RTO
  solve. The RHS is however the user expressed the accumulated cost (a
  finite-element sum, a pyomo.dae Integral, an accumulator's terminal
  value): drto does not care as long as it resolves to the LHS scalar.
- `declare_economic_stage_cost(m.economic_stage_con)`: tags the equality
  Constraint defining the economic running cost phi(z, u). Same LHS-scalar
  convention. This is the same economic objective the steady-state RTO
  mode optimizes at its single point, so one declaration serves economic
  NMPC and RTO both (economic NMPC itself is post-v1; RTO uses it in v1).
  USER DECISION 2026-07-14: split the running cost into tracking and
  economic terms so a mode selects which is live.
- `declare_tracking_terminal_cost(m.tracking_terminal_con)`: tags the
  equality Constraint defining the terminal (Mayer) tracking cost
  V_f(z(tN)), the terminal regulation penalty. Same LHS-scalar convention;
  drto adds the terminal-cost Var to the objective. Dropped in
  steady-state (see the objective note below). Renamed from
  `declare_terminal_cost` for clarity against the economic term (USER
  DECISION 2026-07-14).
- `declare_initial_condition(m.init_con)`: tags the equality Constraint
  anchoring the initial state, LHS the anchored state at t0. If the RHS is
  a mutable Param, that Param is the feedback-injection point drto updates
  each step, so this convention doubles as the state-feedback hook (z_hat)
  for the online loop. Named "condition" deliberately: it pins the state
  (an equality), a different job from a boundary set. This is the exact
  seam where MHE will diverge, since its initial anchor is the soft
  arrival cost, not a hard equality, so a soft mode or a separate
  `declare_arrival_cost` is the estimation follow-on, not this.
- `declare_terminal_constraint(m.terminal_con)`: tags the Constraint
  (equality or inequality) defining the terminal set/region z(tN) in X_f.
  Requirement (USER DECISION 2026-07-14): every Var it references is a
  declared state at tN, the final time; a control at tN is excluded, the
  standard OCP convention (X_f lives in state space). No LHS convention,
  which is what separates it from a path constraint (present at every t).
  "Constraint" not "condition" because it restricts to a set rather than
  pinning a value.
- `declare_steady_state(m.z_ss, ...)`: tags the Params holding the
  steady-state state target z_ss. The tracking costs drive toward these
  (z - z_ss); the steady-state/RTO mode populates them from its solve (or
  they are set directly, since they are Params), so the target is
  model-derived rather than hand-typed (the Hicks CSTR lesson above).
- `declare_steady_state_control(m.u_ss, ...)`: tags the Params holding the
  steady-state control target u_ss, driven toward the same way (u - u_ss)
  and populated the same way (USER DECISION 2026-07-14).

Convention on the declared constraints (USER DECISION 2026-07-14, verified
against Pyomo 6.10):

- The cost and initial-condition constraints must be equalities; an
  inequality is rejected with a clear error (a cost term or anchor written
  as an inequality is a user mistake).
- Their LHS is the appropriate scalar, read as `con.expr.args[0]`. Pyomo
  canonicalizes the constraint body to `LHS - RHS` (lower=upper=0), but
  `con.expr` preserves the as-written relational expression, so the LHS is
  a stable read and a misplaced scalar (a non-Var LHS) is detectable and
  rejected. For a cost the scalar is the cost-term Var drto puts in the
  objective; for the initial condition it is the anchored state at t0.

The objective is drto's, not the user's: it assembles `min` over the
declared cost-term Vars that are live in the current mode. Modes add or
drop terms by including or excluding a cost's (Var, defining constraint)
pair; in steady-state the reduced model never instantiates the terminal
cost, so that term is simply absent, nothing to zero-weight. This is the
practical payoff of representing each cost as a Var defined by a
constraint rather than a bare expression: the pair is one handle drto can
find and drop.

Naming (USER DECISION 2026-07-14): fully-written-out, not abbreviated
(`declare_terminal_cost` not `declare_term_cost`,
`declare_initial_condition` not `declare_init_con`). These are setup-time
calls, never hot-path, so brevity buys nothing; `con` is ambiguous
between condition and constraint, the exact distinction that matters; the
full forms are the OCP literature's own vocabulary, so a reader who knows
the theory maps straight on; and spelling them out forces the precise
concept (condition vs constraint vs set) rather than hiding behind an
abbreviation.

Not declared, by design:

- Path constraints: the states' upper and lower bounds, i.e. the Var
  bounds, which drto reads off the model (USER DECISION 2026-07-14). Not
  a separate declaration.

Moving-horizon data hooks now have homes: the state anchor z_hat is the
mutable Param on the RHS of the initial-condition constraint; the tracking
setpoint is the declared steady-state Params z_ss/u_ss; the measurements
are the mutable Param stream in `declare_measurement` below. Each is
updated each step. Dynamics source is decided too: picked up from each
state's DerivativeVar, above.

Shared conventions:

- Each declaration has an explicit call-time form as well (the
  declared/explicit duality established in pyomo-cvp and pyomo-pounce).
- Family conventions locked in pounce#203: varargs on every declaration,
  keyword options (e.g. `group=`) apply to every component in the call.

Estimation-side surface (USER DECISION 2026-07-14; surface designed now,
built with the MHE follow-on). MHE is the dual of the control problem, so
the same conventions hold (each tags a Var or a Constraint; cost
constraints are equalities with the scalar on the LHS; drto assembles the
estimation objective from the live cost-term Vars):

- `declare_estimated_parameter(m.theta, ...)`: tags the Vars for unknown
  model parameters to estimate, constant over the window. Shared with the
  steady-state data-reconciliation mode.
- `declare_disturbance(m.w, ...)`: tags the process-noise Vars w in
  dz/dt = f + w, the free variables the estimator adjusts to reconcile the
  model with the data, penalized by their inverse covariance in the
  estimation stage cost. It is noise, not a manipulated input: unrelated to
  `declare_control`, with no profile parameterization (USER DECISION
  2026-07-14).
- `declare_measurement(m.y_meas, ...)`: tags the measurement Param(s), the
  measured values y_meas that appear in the estimation cost residuals
  (||y_meas - h(z)||). Like the z_hat feedback hook, it is a mutable Param
  drto refreshes each step, here the incoming measurements over the window.
  Nothing else to tag: h(z) is written inline in the cost, so there is no
  output Var or defining constraint (USER DECISION 2026-07-14).
- `declare_estimation_stage_cost(m.est_stage_con)`: tags the equality
  Constraint for the running estimation cost over the window, the
  measurement residual ||y_meas - h(z)|| plus the process-noise penalty
  ||w||, weighted by inverse covariances. LHS-scalar convention.
- `declare_estimation_terminal_cost(m.est_terminal_con)`: tags the equality
  Constraint for the current-time (window-present) term, the current-state
  measurement residual ||y_meas(tN) - h(z(tN))|| with no process noise
  (nothing leads out of the last point), which is why it is a distinct
  terminal term rather than part of the stage sum. This IS a standard MHE
  term (USER correction 2026-07-14).
- `declare_arrival_cost(m.arrival_con)`: tags the equality Constraint for
  the soft prior on the window's initial state, ||z(t0) - z_prior||
  weighted by the arrival-cost inverse covariance. The dual of the
  control-side initial condition, but SOFT (a cost, not a hard equality).
  Its weight is the piece the covariance propagation updates each step
  (Gauss-Newton, the pounce#203 machinery in Follow-on).

## Ground-up core; what to reuse

- No dependency on pyomo.contrib.mpc. Verified: its interface layer is a
  thin wrapper over `pyomo.dae.flatten`; its remaining content (data
  containers, cross-model linker, shift helpers) is thin and
  rebuildable. Worth one reading pass over its shift/extract code for
  the off-by-one edge-case inventory, as a checklist only.
- `pyomo.dae.flatten.flatten_dae_components` is the traversal primitive
  available if needed: it partitions any model (multi-dimensional
  indexes, time in any position, variables inside time-indexed blocks)
  into non-time scalars plus time-trajectory References where `ref[t]`
  works uniformly. Use case in the loop: the whole-solution warm shift
  (`ref[t] <- ref[t+h]` per trajectory). v1 may restrict to declared
  components on flat models instead; decision deferred until the spike.

## Known hard problem

Active-set changes at the fast update: the sensitivity correction is a
linear step and is untrustworthy when the active set changes between the
predicted and measured state. The pyomo-pounce `estimate()` machinery
already clamps to bounds and warns; the controller should treat a
triggered warning as "fall back to waiting for the background solve."
Every prior asNMPC attempt ignored this; handling it visibly is part of
the contribution. pounce's Schur-refinement issue (jkitchin/pounce#7) is
the eventual upgrade path.

## Delivery plan

1. Spike (one day): quad-tank, mode 1 end-to-end (build on the
   Quad_tank_cvp model and notebook 25's declarations), then flip on
   mode 3 with the existing `estimate()`. The mode comparison plot falls
   out.
2. v1 package: the loop, three modes, declarations, tests, docs,
   executed notebooks, per the pounce-repo definition-of-done style.
3. Stretch example (credibility tier, not a v1 gate): one dynamic IDAES
   flowsheet. Candidates to verify against IDAES 2.12 when budget
   allows: Caprese's moving-bed chemical-looping reactor (the
   succession story), the subcritical boiler dynamic flowsheet
   (industrial load-following; connects to power-price-aware operation),
   Parker's gas-pipeline NMPC network.

## Follow-on

Moving horizon estimation (asMHE): the twin Caprese promised and never
shipped. The arrival cost is a covariance-propagation question, and the
pounce#203 covariance machinery computes the needed pieces from the same
factorizations. asNMPC + asMHE is the complete output-feedback stack;
it exists nowhere in open source. One known consumer obligation: the
covariance's active-bound projection returns a SINGULAR matrix when an
estimated state sits on a bound, so the arrival-cost update must
pseudo-invert for the weight and carry the active bound into the next
horizon as a constraint. Gauss-Newton (`hessian="gauss-newton"`) is the
PSD-guaranteed choice for arrival costs.

## Name and home (resolved 2026-07-12)

- Name: drto. D-RTO is the literature's own abbreviation for dynamic
  real-time optimization, it covers the control and estimation halves
  under one term, and it passes the say-it-aloud test. PyPI name free
  (claim with a stub at first release); GitHub handle `drto` is taken,
  so the repo lives at devin-griff/drto; drto.io free. Rejected in the
  naming search: a pyomo- prefix (the package should outgrow "Pyomo
  toolbox" into a standalone product), dynoptic (collides with
  DynOptic Systems, an Acoem instrumentation brand selling into the
  same industrial market), dynarto (fails the say-it-aloud test).
- Standalone package in this repo; not a pyomo-pounce module.

## Open questions

- v1 scope boundaries confirmed so far: no MHE, no economic NMPC, no
  amsNMPC multistep variant.
- Dynamics source: RESOLVED 2026-07-14 (good-enough starting point). drto
  picks up each declared state's dynamics from its DerivativeVar; no
  `declare_dynamics`. Remaining thread: this is the dynamic-problem
  source, while the steady-state reduction's reusable-object mechanism was
  chosen to avoid DerivativeVar surgery, so how the two square (one
  reusable rule feeding both, versus picking up DerivativeVars for the
  loop and reducing to steady state separately) still needs settling.
- Moving-horizon data: RESOLVED 2026-07-14. Each hook is a mutable Param,
  updated each step: z_hat on the initial-condition RHS, the setpoint as
  the declared steady-state Params z_ss/u_ss, the measurements in
  `declare_measurement`. Remaining detail: the measurement Param's window
  bookkeeping (which measurement maps to which time point) and whether a
  thin layer wraps the updates.
- MHE surface settled 2026-07-14 (estimation declarations above);
  implementation deferred to the follow-on. `declare_measurement` tags a
  mutable Param (the measured values in the estimation costs); the only
  remaining bit is the mundane window-shift bookkeeping of that Param as
  the horizon advances, shared with the rest of the moving-horizon
  plumbing.
- Economic terminal cost: RESOLVED 2026-07-14, no separate
  `declare_economic_terminal_cost`. Economic NMPC can carry one in the
  literature, but drto will not add the declaration.
- `declare_control` vs `declare_profile`: RESOLVED 2026-07-14. The
  `profile` flag on `declare_control` calls cvp's `declare_profile`.
