# drto.cold_start_dynamic

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that initializes a declared, discretized
dynamic model whose initial condition sits away from the steady state, so
that the first solve of a receding-horizon run starts from a consistent
guess with no prior solution in hand.

```python
import drto

# ... declared and discretized model m, initial condition set ...

drto.cold_start_dynamic(m, controls={m.u.name: 0.3})
```

Values only, before any drto transformation, the feature 010 conventions.
The initialization:

- The steady state solves on a reduced clone, the pipeline
  `drto.initialize_steady_state` already runs, with the controls held at
  the mapping's nominal values.
- **States**: a straight line from the declared initial condition to the
  steady state, one value per grid point. Each state's DerivativeVar gets
  the line's slope, `(z_ss - z0) / T`, one constant per member; a state
  starting on its steady value gets zero.
- **Controls**: held constant at the nominal values the steady solve used.
- **Algebraic variables**: solved, not guessed. At each grid point the
  states and controls hold the values above and the model's remaining
  equations, everything except the declared dynamics, solve for the rest:
  one small square solve per point, the same block pipeline.

The guess then satisfies every equation at every grid point except the
declared dynamics: the discretization rows hold exactly, since a line is
differentiated exactly, the initial condition holds at the first point, and
the algebra is consistent pointwise. The declared dynamics carry the
mismatch between the line and the true transient, which is the optimizer's
job to resolve.

An initial condition already at the steady state reduces to the flat
broadcast of `drto.initialize_steady_state`: zero slopes and the
equilibrium everywhere. The two functions share their internals.

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

- `drto.cold_start_dynamic(m, controls=...)` takes a declared, discretized
  dynamic model before any drto transformation, populates variable values
  only, adds and removes nothing, and restores the fixed flags it touches.
  The controls mapping follows the feature 008 convention.
- States run linearly from the declared initial condition to the solved
  steady state; their DerivativeVars hold the slope; controls hold the
  nominal values.
- At every grid point, every equation except the declared dynamics is
  satisfied to the pipeline's tolerance.
- An initial condition at the steady state reproduces the
  `drto.initialize_steady_state` flat result.
- The cart-pole initializes from rest and the first dynamic optimization
  solves; a model with `Block(time)` structure initializes the same way.
- Returns a readable report in the feature 010 shape, adding the
  interpolation and the per-point solves.
