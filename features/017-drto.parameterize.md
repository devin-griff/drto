# drto.parameterize

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want to apply the control profiles I declared without
calling pyomo-cvp directly, so that a drto workflow stays in the `drto.`
namespace and the registry stays consistent when the control components are
replaced.

```python
import pyomo.environ as pyo
import drto

# ... declared model m (feature 002), discretized ...

pyo.TransformationFactory("drto.parameterize").apply_to(m)
# every control now has its declared profile's free values only
```

## Benefit hypothesis

`control(profile=...)` records the profile; something must apply it.
The mode transforms will do it as a step, but a standalone workflow (the
feature 004 example, the Hicks notebook) otherwise has to call
`cvp.parameterize` itself, leaking the dependency into user code. Wrapping it
also fixes a real bookkeeping gap: pyomo-cvp parameterizes by replacing the
control component, which would leave the registry's control records pointing
at detached components.

## Acceptance criteria

- `TransformationFactory('drto.parameterize')` applies every pending declared
  control profile by delegating to pyomo-cvp's declaration-mode
  `cvp.parameterize`.
- It requires declared controls and errors clearly when there is nothing to
  apply: no `control` yet, or the profiles were already applied.
- After application it refreshes the registry: each control record points at
  the live replacement component, found by name, so `drto.info` and later
  transforms see the real model.
- It records itself in the transformation log with each control's profile as
  the outcome annotation.
- Ordering: it runs after `drto.infinite_horizon` (feature 004), which reads
  the declared controls in their original time indexing; the mode transforms
  call it as one of their steps.
- It works through both `apply_to` (in place) and `create_using` (a
  transformed clone).
