# drto.build_objective

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want every mode to install its objective the same way
through one routine, so that the objective is correct and consistent across
modes and I never hand-write or maintain it.

## Benefit hypothesis

Owning objective installation in one shared routine keeps the objective
consistent across every mode, the optimization and simulation transforms now
and the estimation transforms later, and removes drift-prone duplication. Each
mode selects the objective it needs through an option, and this routine installs
it the same way every time.

## Acceptance criteria

- `drto.build_objective(m, ...)` installs exactly one active minimize
  `Objective` on the model. Every mode transform calls it as its final step, so
  objective installation lives in one place. Any existing active objective is
  deactivated first.
- The outcome is option dependent. For the optimization modes it assembles the
  objective from the live cost terms. For the simulation modes it installs a
  constant-zero `Objective`, since a simulation has no cost to assemble.
- The cost objective is the plain sum of each live stage cost's per-point cost
  var over its time index, excluding the last time point, plus each live
  terminal-cost var. For a time set over 0..N, the stage-cost sum runs over
  0..N-1, and a live terminal cost applies at N.
- It is also registered as `TransformationFactory('drto.build_objective')` so a
  user can apply it on its own. The transform calls the same function, and both
  `apply_to` (in place) and `create_using` (a clone) work.
- An empty cost sum is not a case build_objective has to guard: the optimization
  transforms each require a stage cost (features 005 and 008), so it is never
  asked to assemble a cost objective with no live cost term. A dynamic
  optimization with no stage cost is impossible by construction, caught by the
  transform's requirements. The simulation modes take the zero option rather
  than relying on an empty sum.
