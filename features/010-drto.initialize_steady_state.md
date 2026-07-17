# drto.initialize_steady_state

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that initializes my dynamic model from its
steady state, so that the horizon problem starts from a model-consistent flat
trajectory that helps the solver converge.

```python
import drto

# ... declared model m (feature 002), discretized ...

drto.initialize_steady_state(m)   # solve the steady-state form, broadcast
                                  # the equilibrium across every time point
```

## Benefit hypothesis

A steady-state-based initial guess is often the difference between an IPOPT
solve that converges and one that stalls, and deriving it from the model keeps
it consistent with the dynamics.

## Acceptance criteria

- `drto.initialize_steady_state(m)` solves the steady-state form of the model
  and broadcasts that equilibrium across every time point, a flat starting
  trajectory.
- It populates variable values (the initial guess) and does not change the
  model structure. It is a plain function, with no `apply_to` or `create_using`
  form.
- The steady-state solve it uses is the same reduction as
  `drto.dynamic_to_steady_state` (feature 005).
