# drto

Dynamic real-time optimization: receding-horizon NMPC for Pyomo models,
with advanced-step NMPC as the headline capability and moving horizon
estimation as the planned follow-on.

## The framework

drto is declaration-first. You write your dynamic model once as an
ordinary `pyomo.dae` model, then declare the pieces that turn it into a
receding-horizon control problem; drto assembles the horizon problem and
runs the loop. Those pieces are the six object types of an optimal
control problem:

| Object type | Declaration | What it is |
| --- | --- | --- |
| State | `declare_state(m.z, ...)` | A differential state. drto reads its dynamics from the state's `DerivativeVar`. |
| Control | `declare_control(m.u, ..., wrt=m.t, profile=...)` | A manipulated input, the decision variable. The `profile` flag sets its parameterization (piecewise-constant, ...) via pyomo-cvp. |
| Stage cost | `declare_stage_cost(expr)` | The running cost, summed over the horizon. |
| Terminal cost | `declare_terminal_cost(expr)` | The cost on the state at the end of the horizon. |
| Initial condition | `declare_initial_condition(...)` | The initial-state anchor, the measurement feedback in NMPC. |
| Terminal constraint | `declare_terminal_constraint(...)` | The terminal set or region the final state must lie in. |

Two things you never declare, because they already live in the model: the
**dynamics** are read from the `pyomo.dae` `DerivativeVar`s of the
declared states, and the **path constraints** are the state variables'
own upper and lower bounds.

The vocabulary is the optimal-control literature's own (stage cost,
terminal cost, terminal constraint), so a model reads the way the theory
does, and the same six declarations describe every control-side mode:
ideal, real-time, and advanced-step NMPC, plus economic D-RTO. Moving
horizon estimation, the estimation half, adds its own pieces (a
measurement and a soft arrival cost) and is the planned follow-on.

## Status

Design phase: see [DESIGN.md](DESIGN.md). No code yet.
