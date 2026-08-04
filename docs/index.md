# drto

drto is dynamic real-time optimization on the Pyomo model you already have.
You declare which components play which role — the time horizon, the states,
the dynamics, the controls, the costs — and drto's transformations rewrite
the model from there: collapse it to its steady state, append an
infinite-horizon terminal segment, assemble the optimization or fix the
inputs for a simulation. The declarations live in a registry the model
carries, so one set of declarations drives every mode, and displaying the
registry shows the model in its own physical terms.

Nothing about the model changes to make this work: the examples run
unmodified IDAES flowsheets, with states declared as slices of indexed
holdups and controls declared on inlet ports.

## The shape of a session

```python
import drto

# declare roles on your existing model
drto.horizon(m.t)
drto.state(m.z)
drto.dynamics(m.ode)
drto.control(m.u, profile="piecewise_constant")
drto.tracking_stage_cost(m.stage)
drto.initial_condition(m.ic)
drto.steady_state(m.z, m.z_ss)
drto.steady_state_control(m.u, m.u_ss)
drto.info(m)          # the model, rendered in drto's terms

# transform, initialize, solve
pyo.TransformationFactory("dae.collocation").apply_to(m, wrt=m.t, nfe=10, ncp=3)
pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
drto.cold_start_dynamic(m)
pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
pyo.SolverFactory("ipopt").solve(m)
drto.plot_states(m)
```

The [user guide](guide.md) walks through this workflow and its siblings; the
[examples](examples.md) run them on real systems, up to a full IDAES
flowsheet; the [API reference](api.md) documents every public name; and
[spec-first development](spec_first.md) explains how the package is
developed.

```{toctree}
:maxdepth: 2
:hidden:

guide
examples
api
spec_first
```
