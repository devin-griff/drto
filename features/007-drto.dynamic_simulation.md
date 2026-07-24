# drto.dynamic_simulation

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

## Description

As a user of DRTO, I want a transformation that fixes my controls and prepares
the dynamic model to be solved forward over the horizon, so that I can integrate
the model as declared without writing a separate simulation.

```python
import pyomo.environ as pyo
import drto

# ... declared model m (feature 002), discretized ...

pyo.TransformationFactory("drto.dynamic_simulation").apply_to(m)
# controls fixed at the values they already hold, objective zero; a
# supplied constant or profile is the option: apply_to(m, controls={m.u: 0.3})
pyo.SolverFactory("ipopt").solve(m)   # square forward integration
```

## Benefit hypothesis

Reusing the one declared model to simulate keeps simulation and optimization
consistent, and it is the building block the cold-start initializer and
validation runs rely on.

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
  per free point the applied profile leaves. With nothing supplied a control
  is fixed at the value it already holds, and a control holding no value
  errors rather than fixing at nothing.
- A simulation carries no cost: the declared stage and terminal cost equations
  leave the model and their records are purged, as in
  `drto.steady_state_simulation` (feature 008). Their cost variables are left
  unused.
- The estimation-category declarations (feature 018) are neutralized exactly
  as in `drto.dynamic_optimization` (feature 006), through the same shared
  routine so the two modes cannot drift apart: the estimation costs and the
  measurement Params are deleted and purged, a disturbance is eliminated by
  substituting zero behind the additivity guard, and an estimated parameter is
  fixed at the value it holds and keeps its record. This also protects the
  squareness the mode promises, since a free disturbance would leave the
  system underdetermined.
- The objective is zero: the transform calls `drto.build_objective` (feature
  003) with the option for a simulation, which installs a constant-zero
  `Objective` and gives an NLP solver a well-posed square problem for the
  fixed-control model.
- The transform keeps the time horizon.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
