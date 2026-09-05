# drto.linearize_steady_state

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a steady-state linearization of my model about
a solved equilibrium, so that I get the local gains of the states with
respect to the inputs without perturbation solves or hand-derived
derivatives.

```python
import pyomo.environ as pyo

import drto

# ... declared model m (feature 002), dynamic or already steady-state ...

lin = drto.linearize_steady_state(m, controls={"u": 0.3})
# lin.x0["z"], lin.u0["u"], lin.b["z"]["u"]: x = b*(u - u0) + x0
```

## Benefit hypothesis

The user linearizes the steady state of the one declared model in one
call, `x = b (u - u0) + x0` with `b` read from analytical derivatives at
the solved equilibrium, no perturbation solves and no finite
differencing. The same declarations that drive every other mode name the
states and inputs, so the linearization covers exactly the declared
surface.

## Acceptance criteria

- `drto.linearize_steady_state(m, controls=None, disturbances=None,
  solver_options=None)` takes a declared model. When the registry's
  transformation log lacks `drto.steady_state_simulation`, the function
  applies it (feature 008), passing `controls` and `disturbances`
  through, so a declared untransformed model works in one call. When the
  log has it, the model is used as it stands, and supplying `controls`
  or `disturbances` is a descriptive error, since that model's inputs
  are already fixed.
- The feature requires pyomo-pounce. Without it the call raises the
  instruction to install `drto[pounce]`, as the advanced-step controller
  (feature 012) does. The solve inside the call is a pounce solve, since
  the Jacobian answers from that solve's factorization, and
  `solver_options` passes through to it.
- Before the solve, every declared control member is declared as a
  pounce sensitivity parameter through `pyomo_pounce.declare_sens_param`,
  which accepts a fixed Var, unfixes it, and pins it where it stands
  with a defining equality. The pinned problem solves to the same
  equilibrium as the fixed one.
- A termination other than optimal is a descriptive error naming the
  termination, and nothing is returned.
- The return is a `SteadyStateLinearization` with three fields keyed by
  member name: `x0` maps each declared state member to its value at the
  solution, `u0` maps each declared control member to the value it was
  held at, and `b` maps each declared state member to a map from control
  member to the derivative, read from
  `pyomo_pounce.sens_jacobian(x, wrt=u)` per pair. It renders readably,
  like the package's other reports.
- Disturbances stay fixed at their standing values and are not
  differentiated.
- The model the caller passed is left solved at the linearization point,
  its controls pinned by the defining equalities rather than fixed,
  which is the declaration's documented effect.
- On the linear model `dz/dt = u - z`, `b` is 1 and `x0` equals `u0` to
  solver tolerance. On a nonlinear model, `b` agrees with a finite
  difference of two equilibrium solves at nearby inputs.
- The implementation calls the sensitivity surface by its current names,
  `declare_sens_param` and `sens_jacobian`, so it lands with or after
  the pyomo-pounce floor bump of gh #136.
