# API reference

Generated from the docstrings, grouped the way a session uses them. The
transformations (`drto.infinite_horizon`, `drto.dynamic_optimization`,
`drto.dynamic_simulation`, `drto.dynamic_to_steady_state`,
`drto.steady_state_simulation`, `drto.steady_state_optimization`,
`drto.parameterize`) are Pyomo `TransformationFactory` entries; their
options are documented in the [user guide](guide.md).

## The registry

```{eval-rst}
.. autofunction:: drto.info
.. autoclass:: drto.Info
   :members:
```

## Declarations

The control-side surface:

```{eval-rst}
.. autofunction:: drto.horizon
.. autofunction:: drto.state
.. autofunction:: drto.dynamics
.. autofunction:: drto.control
.. autofunction:: drto.disturbance
.. autofunction:: drto.tracking_stage_cost
.. autofunction:: drto.economic_stage_cost
.. autofunction:: drto.tracking_terminal_cost
.. autofunction:: drto.initial_condition
.. autofunction:: drto.terminal_constraint
.. autofunction:: drto.steady_state
.. autofunction:: drto.steady_state_control
```

The estimation-side surface:

```{eval-rst}
.. autofunction:: drto.measurement
.. autofunction:: drto.estimated_parameter
.. autofunction:: drto.estimation_stage_cost
.. autofunction:: drto.estimation_terminal_cost
.. autofunction:: drto.arrival_cost
```

## Objective assembly

```{eval-rst}
.. autofunction:: drto.build_objective
```

## Initialization

```{eval-rst}
.. autofunction:: drto.initialize_steady_state
.. autoclass:: drto.SteadyStateInitReport
   :members:
.. autofunction:: drto.cold_start_dynamic
.. autoclass:: drto.ColdStartReport
   :members:
```

## Plotting

```{eval-rst}
.. autofunction:: drto.plot_states
.. autofunction:: drto.plot_controls
.. autofunction:: drto.plot_stage_cost
```
