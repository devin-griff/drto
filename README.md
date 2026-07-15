# drto

[![PyPI](https://img.shields.io/pypi/v/drto.svg)](https://pypi.org/project/drto/)
[![Python versions](https://img.shields.io/pypi/pyversions/drto.svg)](https://pypi.org/project/drto/)
[![CI](https://github.com/devin-griff/drto/actions/workflows/ci.yml/badge.svg)](https://github.com/devin-griff/drto/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)

DRTO is a unified framework for dynamic real-time optimization built on
Pyomo. The pounce solver is required for full functionality, DRTO relies on
capabilities in [pyomo-pounce](https://github.com/jkitchin/pounce) including
sensitivity analysis, covariance matrix calculations, and initialization
routines.

## Status

Design phase: see [DESIGN.md](DESIGN.md). No code yet.

## Spec-driven development

drto is built spec-first: every feature is specified under
[`features/`](features/) before it is implemented, as a short doc with a
status, a user-story description, a benefit hypothesis, and acceptance
criteria that drive the tests. See [`features/README.md`](features/README.md).

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

## Transformations

The six modes above are each a single solve exposed
as a Pyomo transformation under the `drto.` namespace, alongside the two
lower-level transformations they compose. You build one declared model and
apply the transformation you want, the same way you apply any Pyomo
transformation.

| Transformation | Registered as |
| --- | --- |
| Steady-state simulation | `drto.steady_state_simulation` |
| Steady-state optimization | `drto.steady_state_optimization` |
| Steady-state estimation | `drto.steady_state_estimation` |
| Dynamic simulation | `drto.dynamic_simulation` |
| Dynamic optimization | `drto.dynamic_optimization` |
| Dynamic estimation | `drto.dynamic_estimation` |
| Objective assembly | `drto.build_objective` |
| Steady-state reduction | `drto.dynamic_to_steady_state` |

## Closed-loop frameworks

Separately from the modes come four **closed-loop frameworks**: `NMPC`,
`asNMPC`, `MHE`, and `asMHE`. These are specced later.

## Initialization

A good initial guess is often the difference between an IPOPT solve that
converges and one that stalls, so drto provides named initializers rather than
leaving each user to hand-roll one. These are plain functions, not
transformations: they leave the model structure alone and only populate the
variable values (the initial guess), so there is no `apply_to` or
`create_using` form.

| Initializer | What it does |
| --- | --- |
| `initialize_steady_state` | Solve the steady-state form and broadcast that equilibrium across every time point: a flat starting trajectory. |
| `cold_start_dynamic` | Simulate the model forward from the initial condition under nominal controls, a feasible starting trajectory for the first solve with no history. |
| `warm_start_dynamic` | Shift the previous horizon's solution forward one step and fill the tail: the standard receding-horizon warm start. |

`warm_start_dynamic` is really a loop operation that belongs with the
closed-loop frameworks, listed here alongside its siblings for completeness.

## Declaring a control problem

drto is declaration-first, and each declaration tags a Pyomo component you
already wrote: a Variable, a Constraint, a Parameter, or a Set. You build your dynamic model as
an ordinary Pyomo model, then point the declarations at its pieces;
drto assembles the horizon problem and runs the loop. It bolts onto an
existing model rather than replacing how you build one. The pieces are the
object types of an optimal control problem (the dynamic-optimization
mode):

| DRTO object type | Pyomo object type | Declaration | What it is |
| --- | --- | --- | --- |
| Time set | Set | `declare_time(m.t)` | The moving-horizon dimension. A `pyomo.dae` ContinuousSet or a discrete Set; the root handle for the horizon. Dynamics are declared separately, below. |
| State | Variable | `declare_state(m.z, ...)` | A differential state; its dynamics are declared separately, below. |
| Continuous dynamics | Constraint | `declare_continuous_dynamics(m.ode_con)` | Equality ODE; its left-hand side is the state's DerivativeVar (dz/dt). |
| Discrete dynamics | Constraint | `declare_discrete_dynamics(m.diff_con)` | Equality difference equation; its left-hand side is the state at the next time point (z[k+1]). |
| Control | Variable | `declare_control(m.u, ..., profile=...)` | A manipulated input, the decision variable. The `profile` flag sets its parameterization (piecewise-constant, ...) via pyomo-cvp, over the declared time set. |
| Tracking stage cost | Constraint | `declare_tracking_stage_cost(m.tracking_stage_con)` | Per-time-point equality for the setpoint-tracking running cost; drto sums its left-hand-side cost var over time in the objective. |
| Economic stage cost | Constraint | `declare_economic_stage_cost(m.economic_stage_con)` | Per-time-point equality for the economic running cost; its single-point steady-state form is the RTO objective. |
| Tracking terminal cost | Constraint | `declare_tracking_terminal_cost(m.tracking_terminal_con)` | Equality defining the terminal tracking cost; its left-hand-side scalar goes in the objective. |
| Initial condition | Constraint | `declare_initial_condition(m.init_con)` | Equality anchoring the initial state; left-hand side is the state at t0, right-hand side the feedback. |
| Terminal constraint | Constraint | `declare_terminal_constraint(m.terminal_con)` | Constraint on the states at the final time; the terminal set the final state must lie in. |
| Steady-state target | Parameter | `declare_steady_state(m.z_ss)` | The state setpoint the tracking costs drive toward; populated by the steady-state/RTO solve. |
| Steady-state control target | Parameter | `declare_steady_state_control(m.u_ss)` | The control setpoint the tracking costs drive toward. |

Conventions drto enforces on those constraints: the cost and
initial-condition constraints are equalities whose left-hand side is the
scalar the declaration is about (the cost term, or the anchored state); a
terminal constraint may reference only states at the final time, which is
what separates it from a path constraint. The objective is drto's own: it
sums the declared cost terms that are live in the current mode, so a mode
drops a term just by leaving out its constraint.

One thing you never declare, because it already lives in the model: the
**path constraints** are the state variables' own upper and lower bounds.

The vocabulary is the optimal-control literature's own (stage cost,
terminal cost, terminal constraint), so a model reads the way the theory
does. The other modes reuse the same model: simulation drops the cost, and
estimation swaps the initial condition for a soft arrival cost and adds the
estimation pieces below.

## Declaring an estimation problem

Estimation is the dual half (moving horizon estimation, the planned
follow-on), and it declares its own pieces the same way. MHE fits the model
to a moving window of measurements, so the free variables and the objective
terms differ, but the conventions carry over: each declaration tags a Var,
a Constraint, or a Param, and drto assembles the estimation objective from
the live cost terms.

| DRTO object type | Pyomo object type | Declaration | What it is |
| --- | --- | --- | --- |
| Estimated parameter | Variable | `declare_estimated_parameter(m.theta, ...)` | Unknown model parameters to estimate, constant over the window. Shared with steady-state data reconciliation. |
| Disturbance | Variable | `declare_disturbance(m.w, ...)` | Process-noise variables (`dz/dt = f + w`) the estimator adjusts to fit the data, penalized by their covariance. |
| Measurement | Parameter | `declare_measurement(m.y_meas, ...)` | The measured values in the estimation cost residuals; a mutable Param drto refreshes each step. |
| Estimation stage cost | Constraint | `declare_estimation_stage_cost(m.est_stage_con)` | Equality defining the running estimation cost: measurement residual plus process-noise penalty over the window. |
| Estimation terminal cost | Constraint | `declare_estimation_terminal_cost(m.est_terminal_con)` | Equality for the current-time measurement residual (no process noise leads out of the last point). |
| Arrival cost | Constraint | `declare_arrival_cost(m.arrival_con)` | Equality for the soft prior on the window's initial state; its weight is updated by covariance propagation. |

The arrival cost is the soft dual of the control side's initial condition,
and the estimation stage and terminal costs are the measurement-fitting
counterparts of the tracking costs.
