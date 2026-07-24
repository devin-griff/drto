# drto.steady_state_simulation

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

## Description

As a user of DRTO, I want a transformation that reduces my model to steady
state with the controls fixed and solves for the equilibrium, so that I can
find the resting operating point from the one model.

```python
import pyomo.environ as pyo
import drto

# ... declared model m (feature 002), dynamic or already steady-state ...

sim = pyo.TransformationFactory("drto.steady_state_simulation").create_using(
    m, controls={m.u: 0.3})
pyo.SolverFactory("ipopt").solve(sim)   # the equilibrium under u = 0.3
```

## Benefit hypothesis

Deriving the equilibrium from the same declarations makes the resting state
model-consistent by construction, and it composes the steady-state reduction
rather than duplicating it. Because that reduction is optional, this mode also
runs on a model the user wrote directly as steady-state, not only a dynamic
model reduced to rest, so one declaration surface lets a steady-state model be
used across the modes.

## Acceptance criteria

- `TransformationFactory('drto.steady_state_simulation')` requires
  `state`, and errors clearly if it is missing. `horizon` and
  `dynamics` are optional, since the user may define either a
  steady-state or dynamic model initially.
- If the model is dynamic (time and continuous dynamics declared), it reduces to
  a single equilibrium point by composing `drto.dynamic_to_steady_state`
  (feature 005). If the model is already steady-state, that step is skipped.
  Either way the declared controls are fixed.
- A control option sets the values the fixed controls take: supplied control
  values, or with nothing supplied, the values the control variables are already
  initialized to on the model. The steady state is a single point, so the
  supplied form is values, not a profile.
- The objective is zero: the transform calls `drto.build_objective` (feature
  003) with the option for a simulation, which installs a constant-zero
  `Objective` and gives an NLP solver a well-posed square problem for the
  fixed-control equilibrium.
- The optimization-only constructs leave the model and the registry, through
  the routine both simulation modes share: the declared stage costs (tracking
  and economic), the terminal cost, and the terminal constraint. A simulation
  carries no cost and no terminal set, and the cost variables they defined are
  left unused.
- The steady-state pairings are kept. The target Params stay on the model, the
  user's components, and they may appear in the equations of a model written
  in deviations from the steady state, so their records stay with them and the
  registry mirrors the model (USER DECISION 2026-07-24, amends 2026-07-18,
  which had dropped the pairing records).
- The estimation-category declarations (feature 018) are neutralized, before
  the reduction, through the routine shared with the dynamic modes: the
  estimation costs and the measurement Params are deleted, a disturbance is
  eliminated by substituting zero, and an estimated parameter is fixed at the
  value it holds and keeps its record. It runs before the reduction, which
  collapses the control-side costs and not the window-based estimation costs,
  and it keeps the equilibrium solve square by removing the free disturbance.
- Solving the transformed model gives an equilibrium that satisfies the dynamics
  at rest and the model's algebraic relations.
- It works through both `apply_to` (in place) and `create_using` (a transformed
  clone).
