# drto.cold_start_dynamic

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want a function that initializes a dynamic model whose
initial condition sits away from the steady state, so that the first solve
of a receding-horizon run starts from a consistent guess with no prior
solution in hand.

```python
import drto

# ... declared and discretized model m, initial condition set,
# steady-state targets filled ...

drto.cold_start_dynamic(m)
```

Values only, at any stage: the declared discretized model, or after
`drto.infinite_horizon`, `drto.dynamic_optimization`, or
`drto.dynamic_simulation`. The registry's records follow the transforms, so
the same function reads the live components at every stage.

The steady-state values come from the declarations: every declared state
needs a `steady_state` pairing and every declared control a
`steady_state_control` pairing, filled with model-consistent values, and a
missing pairing is a descriptive error naming the component. No
equilibrium solve is run; the targets give only the states and controls,
and everything else comes out of the per-point solves. The initialization:

- **States**: a straight line from the declared initial condition to the
  declared steady-state target, one value per grid point; a state starting
  on its target gets a flat line.
- **Derivative variables**: a declared state's DerivativeVar members hold
  the line's slope, `(z_ss - z0) / T`, one constant per member, zero for a
  state on its target. Any other DerivativeVar member (a packed Var's
  undeclared members) comes out of the per-point solves through its own
  balance row.
- **Controls**: held constant at their declared steady-state targets. A
  parameterized control's move variables hold the same target; a fixed
  control (a simulation's) keeps the value it holds.
- **Algebraic variables**: solved, not guessed, when pyomo-pounce is
  installed. At each grid point the states and controls hold the values
  above and the model's remaining equations, everything except the
  declared dynamics, solve for the rest: one small square solve per
  point, the block pipeline feature 010 uses. Without pyomo-pounce the
  per-point solves are skipped, the algebraic variables keep their
  values, and the report says so; everything else needs no solver.
- **The terminal segment**, when one is attached: the tail rests at the
  targets. State copies and segment controls hold the targets, the tau
  derivatives and the pin slacks are zero, and the segment's algebraic
  copies come from the same per-point solves at the segment's points.

With the per-point solves run, the guess satisfies every equation at
every grid point except the declared dynamics: the discretization rows
hold exactly, since a line is differentiated exactly, the initial
condition holds at the first point, and the algebra is consistent
pointwise. The declared dynamics carry the
mismatch between the line and the true transient, which is the optimizer's
job to resolve.

`drto.initialize_steady_state` remains the solve-based seed: it computes
the equilibrium and broadcasts it flat. `cold_start_dynamic` reads the
declared targets instead and interpolates; an initial condition already at
the targets gives the same flat trajectory.

Forward simulation, this feature's earlier draft, is rejected: simulating
forward is itself a full dynamic solve needing its own initial guess, and
under nominal controls the free response of an unstable model runs away
from the setpoint instead of toward it.

## Benefit hypothesis

The first solve of a receding-horizon run has no previous solution to
warm-start from, and a raw or flat guess leaves the solver to find the
whole transient on its own. A guess that is consistent everywhere except
the declared dynamics starts the solver at the problem's actual content,
and it composes with the terminal segment: the tail starts at rest on its
fixed point, the soft pin already satisfied.

## Acceptance criteria

- `drto.cold_start_dynamic(m)` populates variable values only, adds and
  removes nothing, and restores the fixed flags it touches, on all four
  shapes: the declared discretized model, an infinite-horizon model, a
  dynamic optimization, and a dynamic simulation.
- A declared state without a `steady_state` pairing, or a declared control
  without a `steady_state_control` pairing, raises a descriptive error
  naming the component. No equilibrium solve is run.
- States run linearly from the declared initial condition to the declared
  targets; their DerivativeVar members hold the line's slope; controls,
  and a parameterized control's moves, hold their declared targets; a
  fixed control keeps its value.
- With pyomo-pounce installed, every equation except the declared
  dynamics is satisfied at every grid point to the pipeline's tolerance;
  on an infinite-horizon model the same holds at the segment's points,
  with the tail at the targets and the pin slacks at zero. Without it,
  the states, derivatives, and controls initialize the same way, the
  algebraic variables keep their values, and the report records the
  skipped solves.
- An initial condition at the targets reproduces the
  `drto.initialize_steady_state` flat trajectory.
- The cart-pole initializes from rest and the first dynamic optimization
  solves; a model with `Block(time)` structure initializes the same way.
- Returns a readable report in the feature 010 shape, adding the
  interpolation and the per-point solves.
