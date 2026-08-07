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
  one loop iteration on a persistent scaled model: the solution shifts
  one sampling time forward, tail included with zero fills, and the
  warm-started solve lands in single digits against the cold
  seventeen.
- [Advanced step on the IDAES CSTR](notebooks/cstr_advanced_step.ipynb)
  — solve at a prediction, correct at the measurement without
  re-solving: the implemented moves within a percent of the warm
  re-solve, an order of magnitude faster.
- [Checking the DAE index](notebooks/check_index.ipynb) — the
  pendulum's index ladder and the solvent extraction stage posed with
  extents and on the reaction invariants: the higher-index forms fail
  with the offending variables named, the index-one forms pass.
- [Ideal NMPC on the IDAES CSTR](notebooks/cstr_ideal_nmpc.ipynb) —
  the closed loop in one call: measure, solve, implement, simulate,
  the cold start then warm-started re-solves on persistent scaled
  clones, the hot start driven onto the setpoint in three samples.
- [Ideal NMPC on the PrOMMiS mixer-settler](notebooks/sx_ideal_nmpc.ipynb)
  — rare earth extraction under closed-loop control: the PrOMMiS
  flowsheet declared as PrOMMiS wrote it, the states identified
  physically as the inventories with memory, the setpoint from the
  steady flowsheet, and the solvent flow holding the extraction
  against feed noise.

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
notebooks/double_column
notebooks/idaes_cstr
notebooks/cstr_cold_start
notebooks/cstr_warm_start
notebooks/cstr_advanced_step
notebooks/cstr_ideal_nmpc
notebooks/check_index
notebooks/sx_ideal_nmpc
```
