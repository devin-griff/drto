# Examples

Every example is a committed, executed notebook — the outputs on these
pages are real runs. Read them in order: the first-order system is the
tutorial, the Hicks-Ray family walks one model through each piece of the
package, and the flowsheet examples scale the same workflow up to
unmodified IDAES models.

**Start here**

- [First-order system](notebooks/first_order.ipynb) — the smallest
  complete workflow, and why the infinite horizon beats a long finite
  one.
- [Hicks-Ray CSTR: the infinite horizon](notebooks/hicks.ipynb) — the
  canonical nonlinear CSTR, short horizon plus terminal segment against
  a long-horizon baseline.

**One model, every piece** (the Hicks-Ray family)

- [The terminal segment](notebooks/hicks_inf.ipynb) — what
  `drto.infinite_horizon` builds and how to tune it.
- [Dynamic optimization](notebooks/hicks_dynamic_optimization.ipynb) —
  the assembly: profiles, objective, solve.
- [Forward integration](notebooks/hicks_dynamic_simulation.ipynb) —
  `drto.dynamic_simulation` as the plant.
- [The steady-state branch](notebooks/hicks_steady_state.ipynb) — the
  reduction and the equilibrium at held controls.
- [Initializing from the steady state](notebooks/hicks_initialize.ipynb)
  — `drto.initialize_steady_state`, solve-based and broadcast flat.

**More systems**

- [Quadruple tank](notebooks/quad_tank.ipynb) — a multivariable
  benchmark.
- [Cart-pole](notebooks/cart_pole.ipynb) — stabilizing an unstable
  equilibrium.
- [Binary distillation column](notebooks/binary_column.ipynb) — the
  infinite horizon on a full column model.
- [Double column](notebooks/double_column.ipynb) — a larger flowsheet,
  infinite versus finite horizon.

**IDAES flowsheets**

- [An IDAES flowsheet under drto](notebooks/idaes_cstr.ipynb) — the
  saponification CSTR straight from `idaes.models.unit_models`:
  member-subset states, a Port-declared control, the setpoint from the
  steady reduction, and the infinite-horizon controller, every solve
  through a units-driven scaled clone.
- [Cold start on the IDAES CSTR](notebooks/cstr_cold_start.ipynb) —
  `drto.cold_start_dynamic` working the full algebraic cascade, linear
  and exponential profiles through the same solve, with the solver logs.
- [Warm start on the IDAES CSTR](notebooks/cstr_warm_start.ipynb) —
  one loop iteration: the solution shifts one sampling time forward,
  tail included with zero fills, and the next solve beats a fresh cold
  start.
- [Advanced step on the IDAES CSTR](notebooks/cstr_advanced_step.ipynb)
  — solve at a prediction, correct at the measurement without
  re-solving: the implemented moves within half a percent of the warm
  re-solve, twenty times faster.

```{toctree}
:maxdepth: 1
:hidden:

notebooks/first_order
notebooks/hicks
notebooks/hicks_inf
notebooks/hicks_dynamic_optimization
notebooks/hicks_dynamic_simulation
notebooks/hicks_steady_state
notebooks/hicks_initialize
notebooks/quad_tank
notebooks/cart_pole
notebooks/binary_column
notebooks/double_column
notebooks/idaes_cstr
notebooks/cstr_cold_start
notebooks/cstr_warm_start
notebooks/cstr_advanced_step
```
