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

## The three modes (one loop)

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

The control-scope declaration surface (USER DECISION 2026-07-14;
estimation-side declarations deferred with the MHE half):

- `declare_state(m.z1, m.z2, ...)`: the differential states. Varargs,
  indexed-container-aware (one call declares all members). drto then picks
  up each state's dynamics automatically from its DerivativeVar (USER
  DECISION 2026-07-14, good-enough starting point; no `declare_dynamics`).
  This is NOT the rejected state auto-detection: the state role is still
  declared, and the DerivativeVar only locates the ODE of an
  already-declared state.
- `declare_control(m.u, ..., wrt=m.t, profile=...)`: the manipulated
  inputs, the free decision variables. The `profile` flag folds in the
  control's parameterization: `declare_control` calls pyomo-cvp's
  `declare_profile` automatically (USER DECISION 2026-07-14). One call
  declares the control and its parameterization; cvp stays the dependency
  that implements the parameterization underneath.
- `declare_stage_cost(expr)`: the running (Lagrange) cost integrand
  L(z, u); drto integrates/sums it over the horizon.
- `declare_terminal_cost(expr)`: the terminal (Mayer) cost V_f(z(T)).
- `declare_initial_condition(...)`: the initial-state anchor z(0) = z_hat,
  a hard equality (the measurement feedback in NMPC; the prediction/RTO
  anchor in DRTO). Named "condition" deliberately: it pins the state (an
  equality), a different job from a boundary set. This is the exact seam
  where MHE will diverge, since its initial anchor is the soft arrival
  cost, not a hard equality, so a soft mode or a separate
  `declare_arrival_cost` is the estimation follow-on, not this.
- `declare_terminal_constraint(...)`: the terminal set/region z(T) in
  X_f. "Constraint" not "condition" because it restricts to a set rather
  than pinning a value.

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

Still open (see Open questions): how the moving-horizon data (the mutable
z_hat, tracking setpoint, and later the measurements) is supplied and
updated each solve. Dynamics source is now decided: picked up from each
state's DerivativeVar, above.

Shared conventions:

- Each declaration has an explicit call-time form as well (the
  declared/explicit duality established in pyomo-cvp and pyomo-pounce).
- Family conventions locked in pounce#203: varargs on every declaration,
  keyword options (e.g. `group=`) apply to every component in the call.

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
- Moving-horizon data: how the mutable feedback (z_hat), tracking
  setpoint, and later the measurements are supplied and updated each
  solve. A thin layer over the declarations is the leaning, not settled.
- (MHE, deferred) the measurement notion and the soft arrival cost noted
  under `declare_initial_condition`.
- `declare_control` vs `declare_profile`: RESOLVED 2026-07-14. The
  `profile` flag on `declare_control` calls cvp's `declare_profile`.
