# drto.advanced_step_controller

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that turns a horizon problem solved at a
predicted state into the corrected control for the actual state, so that I get
the advanced-step NMPC correction without re-solving online.

```python
import drto

# ... m solved by drto.dynamic_optimization at the PREDICTED next state ...

m.z_hat.set_value(z_measured)          # the actual measurement arrives
u0 = drto.advanced_step_controller(m)  # pounce estimate() of the controls,
                                       # corrected to z_hat, no re-solve
dudz = drto.advanced_step_controller(m, gradient=True)  # sensitivities instead
```

## Benefit hypothesis

Advanced-step NMPC's whole value is replacing the online solve with a fast
sensitivity update: solve at a predicted state between samples, then correct
instantly when the real measurement arrives. Exposing that correction as one
function built on pounce's sensitivity is what makes the advanced-step framework
practical, and it is DRTO's headline differentiator.

## Acceptance criteria

- `drto.advanced_step_controller(m, ...)` returns the advanced-step control for a
  model already solved by `drto.dynamic_optimization` at a predicted state, using
  the initial-condition parameter (the state feedback hook) as the sensitivity
  parameter.
- By default it returns pyomo-pounce's `estimate()` of the declared controls: the
  fast sensitivity-based update of the optimal controls at the actual (measured)
  state, without re-solving.
- A flag returns pyomo-pounce's `gradient()` instead: the sensitivity of the
  declared controls with respect to the state parameter.
- It reads the declared controls and the initial-condition parameter from
  `drto.info`, and errors clearly if the model has not been solved or has no
  declared initial condition.
- It does not re-solve the model. The correction is a sensitivity update of the
  existing solution.
