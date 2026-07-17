# drto.dynamic_simulation

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

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
  `state`, `dynamics`, and `control`, and
  errors clearly if any is missing.
- The declared controls are fixed, so the mode frees nothing and solves the
  model as declared over the horizon. A control-profile option sets the values
  they are fixed to: a supplied constant, held across the horizon, or a supplied
  profile over the time set. With nothing supplied, the controls stay at the
  values the control variables are already initialized to on the model.
- The objective is zero: the transform calls `drto.build_objective` (feature
  003) with the option for a simulation, which installs a constant-zero
  `Objective` and gives an NLP solver a well-posed square problem for the
  fixed-control model.
- The transform keeps the time horizon.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
