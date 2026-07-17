# drto.build_objective

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want every mode to install its objective the same way
through one routine, so that the objective is correct and consistent across
modes and I never hand-write or maintain it.

```python
import pyomo.environ as pyo
import drto

# ... declared model m (feature 002), discretized ...

drto.build_objective(m)   # default: assemble the live cost terms

# the transform form is equivalent
pyo.TransformationFactory("drto.build_objective").apply_to(m)

# the marked case, what the simulation transforms pass
drto.build_objective(m, zero=True)
```

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
- The outcome is option dependent, and the default is the cost assembly: the
  bare call `drto.build_objective(m)` assembles the objective from the live
  cost terms. The constant-zero `Objective` is the marked case, selected by an
  explicit option, which the simulation transforms pass since a simulation has
  no cost to assemble. The function never infers a mode: callers request
  outcomes.
- The cost objective is the sum of the live registered cost groups, each with
  its group's weights. The declared stage costs are the uniform-weight group:
  each live stage cost's per-point cost var summed over its time index,
  excluding the last time point, plus each live terminal-cost var. For a time
  set over 0..N, the stage-cost sum runs over 0..N-1, and a live terminal cost
  applies at N.
- Transforms may register additional cost groups carrying their own per-point
  weights (`drto.infinite_horizon`, feature 004, registers its tail terms this
  way), and the assembly includes every registered group that is live: its
  backing components present and active on the model at assembly time.
- Group weights are how the modes weight costs, not options here: the tracking
  weight `drto.dynamic_optimization` accepts (feature 006) is recorded as the
  tracking group's weight in the registry by that transform, and this routine
  just sums the live groups by their weights. The zero option is the only flag
  on the call, since an empty objective is the one outcome that cannot be
  expressed as registry state.
- It is also registered as `TransformationFactory('drto.build_objective')` so a
  user can apply it on its own. The transform calls the same function, and both
  `apply_to` (in place) and `create_using` (a clone) work.
- An empty cost sum is not a case build_objective has to guard: the optimization
  transforms each require a stage cost (features 006 and 009), so it is never
  asked to assemble a cost objective with no live cost term. A dynamic
  optimization with no stage cost is impossible by construction, caught by the
  transform's requirements. The simulation modes take the zero option rather
  than relying on an empty sum.
