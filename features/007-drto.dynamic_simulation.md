# drto.dynamic_simulation

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a transformation that fixes my controls and prepares
the dynamic model to be solved forward over the horizon, so that I can integrate
the model as declared without writing a separate simulation.

```python
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory

import drto

# ... declared model m (feature 002), discretized ...

pyo.TransformationFactory("drto.dynamic_simulation").apply_to(m)
# controls fixed at the values they already hold, the objective zero.
# A supplied constant or profile is the option: apply_to(m, controls={m.u: 0.3})
SolverFactory("ipopt").solve(m)   # square forward integration
```

## Benefit hypothesis

The user simulates the model exactly as declared, with no separate
simulation model to write and none to keep consistent with the
optimization, since both modes read the same declarations. The
cold-start initializer and validation runs use this mode.

## Acceptance criteria

- `TransformationFactory('drto.dynamic_simulation')` requires `horizon`,
  `state`, `dynamics`, `control`, and `initial_condition`, and errors clearly
  if any is missing. A forward integration needs the initial state pinned, or
  the horizon problem is not square.
- The declared control profiles are applied first (`drto.parameterize`,
  feature 017), so the simulated input takes the shape the model declared. The
  user chooses that shape at declaration time through `control(profile=...)`.
- The parameterized controls are then fixed, so the mode frees nothing and
  solves the model as declared over the horizon. A `controls` option sets the
  values they are fixed at: a constant, held across the horizon, or one value
  per free point the applied profile leaves. The mapping's key is the
  declared component or its name, resolved by name, since parameterizing
  replaces the control components, so keys from the source model work
  through `create_using` and a control on a sub-Block passes as a
  component. With nothing supplied a control is fixed at the value it
  already holds, and a control holding no value errors rather than fixing
  at nothing.
- A simulation has no cost and no terminal set. The declared stage
  costs, move suppression, terminal cost, and terminal constraint leave
  the model and their records are purged, as in
  `drto.steady_state_simulation` (feature 008), through the routine both
  simulation modes share. A terminal constraint would over-constrain the
  square forward integration. The cost variables are left unused.
- The steady-state target Params (`steady_state`, `steady_state_control`)
  stay on the model and keep their records, since a deviation-form
  model's equations may reference them.
- The estimation-category declarations (feature 018) are neutralized exactly
  as in `drto.dynamic_optimization` (feature 006), through the same shared
  routine so the two modes cannot drift apart. The estimation costs and
  the measurement Params are deleted and purged, and an estimated
  parameter is fixed at the value it holds and keeps its record.
- Each disturbance is fixed at its realization, the same way the controls are
  fixed, after the profiles are applied. A `disturbances` option maps a
  declared disturbance to its realized noise, a constant held across the
  horizon or one value per free point, with the same key forms as
  `controls`, so the plant can be driven by a supplied noise sequence. A
  disturbance not in the mapping is fixed at zero.
  Fixing keeps the disturbance in the model and keeps the forward integration
  square, and it works however the noise enters the equations.
- The objective is zero. The transform calls `drto.build_objective`
  (feature 003) with the option for a simulation, which installs a
  constant-zero `Objective` and gives an NLP solver a well-posed square
  problem for the fixed-control model.
- The transform keeps the time horizon.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
