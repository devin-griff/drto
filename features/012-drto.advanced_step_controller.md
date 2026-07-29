# drto.advanced_step_controller

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want the advanced-step correction: solve the horizon
at a predicted state between samples, then correct the solution to the
measured state the moment it arrives, without re-solving.

```python
import drto

# ... m solved by drto.dynamic_optimization at the predicted state,
# with pounce ...

m.z_hat.set_value(z_measured)              # the measurement arrives
est = drto.advanced_step_controller(m)     # pounce estimate(), corrected
u0 = est[m.u[0]]                           # the move to implement

dudz = drto.advanced_step_controller(m, gradient=True)  # sensitivities
```

`drto.dynamic_optimization` declares the initial-condition Params, the
state feedback hooks, as pounce sensitivity parameters whenever
pyomo-pounce is importable. The declaration is inert metadata: every
other solver ignores it and solves the model unchanged; a pounce solve
keeps the converged factorization, so the correction afterwards is a
backsolve, not a solve.

`drto.advanced_step_controller(m)` reads the hooks' current values as the
perturbation, the difference between the measured state the loop wrote
and the predicted state the model was solved at, and returns
`pyomo_pounce.estimate()`: the corrected solution as a map from each
variable to its estimated value, clamped to bounds. The model is not
touched; the solution at the predicted state stays in place as the next
solve's warm start. With `gradient=True` it returns
`pyomo_pounce.gradient()` for the declared controls with respect to the
hooks instead. Keyword arguments it does not recognize pass through to
the pounce call, so options pounce grows need no change here.

The feature requires pounce: without a pounce solve there is no
factorization, and the call fails with pounce's own instruction to solve
with pounce first.

## Benefit hypothesis

Advanced-step NMPC replaces the online solve with a sensitivity update:
the expensive solve runs between samples at a prediction, and the
correction at the measurement is instant. One function turns the solved
model and the updated hooks into the corrected controls, and it is the
piece the asnmpc loop (feature 015) is built from.

## Acceptance criteria

- `drto.dynamic_optimization` declares the initial-condition Params as
  pounce sensitivity parameters when pyomo-pounce is importable. A model
  solved with any other solver is unchanged by the declaration.
- `drto.advanced_step_controller(m)` returns the `estimate()` map at the
  hooks' current values, without modifying the model. With
  `gradient=True` it returns the `gradient()` of the declared controls
  with respect to the hooks.
- Unrecognized keyword arguments pass through to the pounce call.
- Without a pounce solve, the call raises the no-session error
  instructing to solve with pounce.
- On a solved model, re-solving at the perturbed hooks agrees with the
  estimate to first order.
