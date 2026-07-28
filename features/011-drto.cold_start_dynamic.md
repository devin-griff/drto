# drto.cold_start_dynamic

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that initializes a declared, discretized
dynamic model whose initial condition sits away from the steady state, so
that the first solve of a receding-horizon run starts from a consistent
guess with no prior solution in hand.

```python
import drto

# ... declared and discretized model m, initial condition set,
# steady-state targets filled ...

drto.cold_start_dynamic(m)
```

Values only, before any drto transformation. The steady-state values come
from the declarations: every declared state needs a `steady_state` pairing
and every declared control a `steady_state_control` pairing, filled with
model-consistent values, and a missing pairing is a descriptive error
naming the component. No equilibrium solve is run; the targets give only
the states and controls, and the endpoint's algebraic variables come out
of the same per-point solve as every other grid point's. The
initialization:

- **States**: a straight line from the declared initial condition to the
  declared steady-state target, one value per grid point. Each state's
  DerivativeVar gets the line's slope, `(z_ss - z0) / T`, one constant per
  member; a state starting on its target gets zero.
- **Controls**: held constant at their declared steady-state targets.
- **Algebraic variables**: solved, not guessed. At each grid point the
  states and controls hold the values above and the model's remaining
  equations, everything except the declared dynamics, solve for the rest:
  one small square solve per point, the block pipeline feature 010 uses.

The guess then satisfies every equation at every grid point except the
declared dynamics: the discretization rows hold exactly, since a line is
differentiated exactly, the initial condition holds at the first point, and
the algebra is consistent pointwise. The declared dynamics carry the
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
and it composes with the terminal segment: the segment initializes from the
horizon end, so a trajectory ending at the equilibrium starts the tail at
its fixed point.

## Acceptance criteria

- `drto.cold_start_dynamic(m)` takes a declared, discretized dynamic model
  before any drto transformation, populates variable values only, adds and
  removes nothing, and restores the fixed flags it touches.
- A declared state without a `steady_state` pairing, or a declared control
  without a `steady_state_control` pairing, raises a descriptive error
  naming the component. No equilibrium solve is run; the endpoint's
  algebraic variables come from its per-point solve like every other
  grid point's.
- States run linearly from the declared initial condition to the declared
  targets; their DerivativeVars hold the slope; controls hold their
  declared targets.
- At every grid point, every equation except the declared dynamics is
  satisfied to the pipeline's tolerance.
- An initial condition at the targets reproduces the
  `drto.initialize_steady_state` flat trajectory.
- The cart-pole initializes from rest and the first dynamic optimization
  solves; a model with `Block(time)` structure initializes the same way.
- Returns a readable report in the feature 010 shape, adding the
  interpolation and the per-point solves.
