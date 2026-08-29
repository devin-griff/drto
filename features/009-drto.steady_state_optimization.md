# drto.steady_state_optimization

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a transformation that reduces my model to steady
state and optimizes the economic objective over the free controls, so that I
get the optimal steady operating point (economic RTO) from the one model.

```python
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory

import drto

# ... declared model m (feature 002) with an economic stage cost ...

rto = pyo.TransformationFactory(
    "drto.steady_state_optimization").create_using(m)
SolverFactory("ipopt").solve(rto)   # the optimal steady operating point
```

## Benefit hypothesis

The user gets the optimal steady operating point from the one declared
model, and the setpoint the NMPC then tracks is a true equilibrium of
the dynamics rather than a hand-typed pair. This is what makes the
D-RTO name literal. Because the reduction is optional, the mode runs on
a model reduced from dynamic or one the user wrote directly as
steady-state, so the same declaration surface serves both.

## Acceptance criteria

- `TransformationFactory('drto.steady_state_optimization')` requires
  `state`, `control`, and a stage cost of either kind, and errors
  clearly if any is missing. `horizon` and `dynamics`
  are optional, since the user may define either a steady-state or dynamic model
  initially.
- If the model is dynamic (time and continuous dynamics declared), it reduces to
  a single point by composing `drto.dynamic_to_steady_state` (feature 005). If
  the model is already steady-state, that step is skipped. The declared controls
  are free.
- The objective is assembled via `drto.build_objective` (feature 003) from the
  live single-point cost terms: the economic cost, plus a tracking stage cost
  if one is declared. A tracking term is kept rather than dropped, since it
  regularizes the economic optimum toward a known operating point, the
  RTO-layer equivalent of move suppression. With both declared,
  `tracking_weight` scales the tracking side, mirroring
  `drto.dynamic_optimization` (feature 006). It defaults to 1, and the
  economic cost is in currency units and is never scaled. With only a
  tracking stage cost declared, the objective is that cost alone, the
  steady point nearest the declared targets.
- The estimation-category declarations (feature 018) are neutralized
  through the routine shared with the other control-side modes. The
  estimation costs and the measurement Params are deleted, and an
  estimated parameter is fixed at the value it holds and keeps its
  record. Each disturbance is fixed at zero, after the reduction
  collapses it to a point. This matters more here than in a simulation,
  because a disturbance left free would be a decision variable the
  optimizer exploits to lower the economic cost, optimizing against
  fictitious noise. Fixing it keeps it in the model without making it a
  decision.
- Unlike the simulation modes, the cost equations stay, since this mode
  needs them. The terminal constraint and terminal cost need no handling
  either, since the reduction removes both and a steady-authored model
  cannot have them.
- Solving the transformed model gives the optimal steady operating point.
  Writing that solution back into the declared steady-state targets is an
  algorithmic step outside this transform, which does nothing after a solve.
  The `steady_state` pairings are left intact, since they are the record that
  makes such a write-back possible.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
