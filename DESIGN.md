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
Constraint, a Param, or a Set (USER DECISION 2026-07-14): the point is to
bolt onto an existing Pyomo model, not to introduce a new modeling
framework. The time set tags a Set; state and control tag Vars; the cost
and boundary declarations tag Constraints; the measurement and steady-state
targets tag Params. The control-scope declaration surface (the
estimation-side surface follows below):

- `declare_time(m.t)`: tags the time set, the moving-horizon dimension
  shared by every dynamic mode. It may be a `pyomo.dae` ContinuousSet
  (continuous-time) or a discrete Set (discrete-time, difference equations),
  so drto does not assume continuity (USER DECISION 2026-07-14). Declaring
  it separates time from any other set (a spatial axis in a PDE model): the
  DerivativeVars taken with respect to this set are the temporal dynamics
  drto picks up and shifts, while DerivativeVars over other sets are spatial
  and left alone. For a continuous time set the dynamics are those
  DerivativeVars (see `declare_state`); for a discrete time set there are no
  DerivativeVars and the dynamics are the user's difference-equation
  constraints (that pickup is still open).
- `declare_state(m.z1, m.z2, ...)`: tags the differential-state Vars. Varargs,
  indexed-container-aware (one call declares all members). drto then picks
  up each state's dynamics automatically from its DerivativeVar (USER
  DECISION 2026-07-14, good-enough starting point; no `declare_dynamics`).
  This is NOT the rejected state auto-detection: the state role is still
  declared, and the DerivativeVar only locates the ODE of an
  already-declared state.
- `declare_control(m.u, ..., profile=...)`: tags the manipulated-input
  Vars, the free decision variables. No `wrt` argument: drto uses the
  declared time set, so the control's parameterization is over that set
  (USER DECISION 2026-07-14). The `profile` flag folds in the
  parameterization: `declare_control` calls pyomo-cvp's `declare_profile`
  automatically. One call declares the control and its parameterization; cvp
  stays the dependency that implements it underneath.
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

Convention on the declared constraints (verified
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

Naming: fully-written-out, not abbreviated
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
  bounds, which drto reads off the model. Not
  a separate declaration.

Moving-horizon data hooks now have homes: the state anchor z_hat is the
mutable Param on the RHS of the initial-condition constraint; the tracking
setpoint is the declared steady-state Params z_ss/u_ss; the measurements
are the mutable Param stream in `declare_measurement` below. Each is
updated each step. Dynamics source: each state's DerivativeVar taken with
respect to the declared time set (continuous-time); the discrete-time case
(difference-equation constraints) is still open, see `declare_time`.

Shared conventions:

- Each declaration has an explicit call-time form as well (the
  declared/explicit duality established in pyomo-cvp and pyomo-pounce).
- Family conventions locked in pounce#203: varargs on every declaration,
  keyword options (e.g. `group=`) apply to every component in the call.

Estimation-side surface (surface designed now,
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
