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
- **Model-consistent setpoints.** The steady-state target pair must be a
  true equilibrium of the dynamics. A hand-typed pair that is not a fixed
  point leaves the controller settling at a weighted compromise instead of
  the target.
- **Unbounded cost variables.** The defining equality already fixes each
  cost value, and a `NonNegativeReals` bound puts the optimum exactly on the
  bound wherever the cost vanishes, which drags interior-point solvers.

| Module | Builder | Model |
| --- | --- | --- |
| `hicks.py` | `hicks(N=5, h=1)` | Hicks-Ray CSTR (Hicks & Ray 1971), the canonical nonlinear example: two states, two controls, exothermic reaction. |
| `first_order.py` | `first_order(N=10, h=1)` | First-order linear system, the minimal example from the feature 002 spec. |

From a notebook in `examples/`:

```python
from models.hicks import hicks

m = hicks(N=5)
```
