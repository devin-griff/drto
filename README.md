# drto

Dynamic real-time optimization: receding-horizon optimization and
estimation for Pyomo models, with advanced-step NMPC as the headline
capability and moving horizon estimation as the planned follow-on.

## The six modes

drto runs one declared model in any of six modes, the 2x3 grid of
{steady-state, dynamic} by {simulation, optimization, estimation}. You
write the model once; the mode fixes what is free and what the objective
is.

|  | Simulation | Optimization | Estimation |
| --- | --- | --- | --- |
| **Steady-state** | solve the model at equilibrium | economic RTO | data reconciliation |
| **Dynamic** | integrate the model forward | NMPC / D-RTO | moving horizon estimation |

Down the columns: simulation frees nothing and solves the model as given;
optimization frees the controls and adds a cost; estimation frees the
states and fits them to measurements. Across the rows: steady-state
collapses the model to a single equilibrium point, dynamic keeps the time
horizon. The optimization and estimation columns are duals (NMPC with MHE,
RTO with reconciliation), so one declaration surface serves both.

The near-term focus is the optimization column: dynamic NMPC/D-RTO, whose
ideal, real-time, and advanced-step execution variants are the headline,
plus steady-state RTO. Estimation is the planned follow-on.

## Declaring a control problem

drto is declaration-first, and each declaration tags a Pyomo component you
already wrote: a Variable or a Constraint. You build your dynamic model as
an ordinary `pyomo.dae` model, then point the declarations at its pieces;
drto assembles the horizon problem and runs the loop. It bolts onto an
existing model rather than replacing how you build one. The pieces are the
six object types of an optimal control problem (the dynamic-optimization
mode):

| DRTO object type | Pyomo object type | Declaration | What it is |
| --- | --- | --- | --- |
| State | Variable | `declare_state(m.z, ...)` | A differential state. drto reads its dynamics from the state's `DerivativeVar`. |
| Control | Variable | `declare_control(m.u, ..., wrt=m.t, profile=...)` | A manipulated input, the decision variable. The `profile` flag sets its parameterization (piecewise-constant, ...) via pyomo-cvp. |
| Stage cost | Constraint | `declare_stage_cost(m.stage_cost_con)` | Equality defining the running cost; its left-hand-side scalar is the term drto adds to the objective. |
| Terminal cost | Constraint | `declare_terminal_cost(m.terminal_cost_con)` | Equality defining the terminal cost; its left-hand-side scalar is added to the objective. |
| Initial condition | Constraint | `declare_initial_condition(m.init_con)` | Equality anchoring the initial state; left-hand side is the state at t0, right-hand side the feedback. |
| Terminal constraint | Constraint | `declare_terminal_constraint(m.terminal_con)` | Constraint on the states at the final time; the terminal set the final state must lie in. |

Conventions drto enforces on those constraints: the cost and
initial-condition constraints are equalities whose left-hand side is the
scalar the declaration is about (the cost term, or the anchored state); a
terminal constraint may reference only states at the final time, which is
what separates it from a path constraint. The objective is drto's own: it
sums the declared cost terms that are live in the current mode, so a mode
drops a term just by leaving out its constraint.

Two things you never declare, because they already live in the model: the
**dynamics** are read from the `pyomo.dae` `DerivativeVar`s of the
declared states, and the **path constraints** are the state variables'
own upper and lower bounds.

The vocabulary is the optimal-control literature's own (stage cost,
terminal cost, terminal constraint), so a model reads the way the theory
does. The other modes reuse the same model: simulation drops the cost, and
estimation swaps the initial condition for a soft arrival cost and adds a
measurement.

## Status

Design phase: see [DESIGN.md](DESIGN.md). No code yet.
