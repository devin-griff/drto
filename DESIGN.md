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

- `declare_state(m.z1, m.z2, ...)`: varargs, indexed-container-aware
  (one call declares all members), like the rest of the family.
- Controls: already declared via pyomo-cvp `declare_profile`.
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
