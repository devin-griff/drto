# Canonical models

One declared model per module, as a builder function, so the example
notebooks import a model and get straight to transformations and results.
Every model carries the full feature 002 declaration surface, including the
paired steady-state targets and a tracking terminal cost (the stage cost
with the controls removed, at the final time).

## Preferences for writing Pyomo models

Every model here follows these, and new models should too:

- **Decorator-style constraints.** Always `@m.Constraint(...)` (or the drto
  constraint decorators), never `rule=lambda` or a `def` passed as `rule=`.
- **Named mutable Params, no folded literals.** Physical constants, scale
  factors, and setpoints are named mutable Params referenced by name in the
  equations, so expressions render symbolically in `drto.info` and values
  retune by `set_value`. Literals are for pure structure only (exponents,
  cost weights).
- **The sample grid is N and h.** `N` samples of sampling time `h`, built as
  `ContinuousSet(initialize=pyo.RangeSet(0, N*h, h))`.
- **A builder returns a declared, undiscretized model.** Its first two
  parameters are the interval count and the sampling time, named `N` and
  `h`, since the mode functions pass them by keyword, and every parameter
  has a default, so the bare `build()` is legal. The caller owns
  discretization, which is what `drto.dynamic_optimization` does when it
  is handed the builder (feature 006). A value that has to be fixed at
  every time point is fixed by a function the caller runs after the mesh
  exists, not inside the builder, since a fixed status covers only the
  members present when it is set.
- **Model-consistent setpoints.** The steady-state target pair must be a
  true equilibrium of the dynamics. A hand-typed pair that is not a fixed
  point leaves the controller settling at a weighted compromise instead of
  the target.
- **Unbounded cost variables.** The defining equality already fixes each
  cost value, and a `NonNegativeReals` bound puts the optimum exactly on the
  bound wherever the cost vanishes, which drags interior-point solvers.
- **Declarations tagged in one block at the end.** The drto declarations sit
  together after the model is built, so the model reads as plain Pyomo first
  and the declared surface reads in one place. (Tagging works anywhere after
  a component exists; this is the style these models standardize on.)

| Module | Builder | Model |
| --- | --- | --- |
| `hicks.py` | `hicks(N=5, h=1)` | Hicks-Ray CSTR (Hicks & Ray 1971), the canonical nonlinear example: two states, two controls, exothermic reaction. |
| `first_order.py` | `first_order(N=10, h=1)` | First-order linear system, the minimal example from the feature 002 spec. |
| `quad_tank.py` | `quad_tank(N=15, h=10)` | Johansson's quadruple tank (IEEE TCST 2000), nonminimum-phase configuration: four levels, two pumps with crossed splits. |
| `double_column.py` | `double_column(N=25, h=1)` | Two 41-tray distillation columns in series separating a ternary mixture (the Skogestad Column A lineage), the DAE example: tray compositions and holdups are states, eight flow controls, and the weir flows, vapor compositions, temperatures, and purity aliases ride along as undeclared algebraics. |
| `cart_pole.py` | `cart_pole(N=10, h=1)` | Inverted pendulum on a cart (the Dinh et al. 2025 pendulum example): four states, one force input, underactuated, and the upright setpoint is an unstable equilibrium, the regime where terminal machinery matters. |
| `idaes_cstr.py` | `build(N=10, h=1)` | The IDAES saponification CSTR, the unmodified-flowsheet example: member-subset holdup states with the water member algebraic, jacket duty and Port-declared feed flow as controls, targets and hooks as Params in the owners' units, and the constant feed streams held by `hold_feed`, which the caller runs once the mesh exists. Needs `idaes-pse`. |

From a notebook in `examples/`:

```python
from models.hicks import hicks

m = hicks(N=5)
```
