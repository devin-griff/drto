# Canonical models

One declared model per module, as a builder function, so the example
notebooks import a model and get straight to transformations and results.
Every model follows the same conventions: the time set initialized with the
sample grid, physical constants and setpoints as named mutable Params,
decorator-style constraints, unbounded cost variables, and the full feature
002 declaration surface including the paired steady-state targets.

| Module | Builder | Model |
| --- | --- | --- |
| `hicks.py` | `hicks(N=5, h=1)` | Hicks-Ray CSTR (Hicks & Ray 1971), the canonical nonlinear example: two states, two controls, exothermic reaction. |
| `first_order.py` | `first_order(N=10, h=1)` | First-order linear system, the minimal example from the feature 002 spec. |

From a notebook in `examples/`:

```python
from models.hicks import hicks

m = hicks(N=5)
```
