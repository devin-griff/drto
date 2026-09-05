# drto.initialize_steady_state

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want a function that initializes my model from its
steady state, so that the solve starts from a model-consistent point: a flat
trajectory for a dynamic model, a consistent equilibrium point for a
steady-state one.

```python
import drto

# a declared dynamic model (feature 002), discretized, before any drto
# transformation is applied:
drto.initialize_steady_state(m)   # clone, reduce, solve, broadcast flat
drto.initialize_steady_state(m, controls={m.u: 0.3})  # held at 0.3
drto.initialize_steady_state(m, scale={"J": 1e7, "W": 1e6})  # factors first

# a steady-state model (authored steady, or a feature 005 reduction):
drto.initialize_steady_state(ss)  # in place: fill, project, block-solve
```

## Benefit hypothesis

The user initializes a model from its steady state in one call, a start
that is often the difference between a solve that converges and one that
stalls, and deriving it from the model keeps it consistent with the
dynamics. One function serves both declared shapes, since the core
calculation is the same, so the examples need no hand-written per-model
initialization helpers.

## Acceptance criteria

- `drto.initialize_steady_state(m, controls=None, scale=None)`
  dispatches on the
  declared shape: `horizon` and `dynamics` declared takes the dynamic path;
  a model without them, authored steady-state or already reduced by feature
  005, takes the steady path. Both paths require declared states and error
  clearly without them.
- The steady solve is `pyomo_pounce.initialize` (the fill, project,
  block-solve pipeline: bounds-aware fill of valueless variables, min-norm
  nonlinear projection, then the Dulmage-Mendelsohn block-triangular solve
  of the equality system), with the declared controls as the decisions.
  drto contributes what that suite cannot know: which variables are
  decisions, the steady reduction, and the horizon broadcast.
- An active `scaling_factor` suffix does not change the pipeline, which
  runs in the model's own units, the block solves working the square
  equality system in calculation order (gh #92). The suffix stays on
  the model, unread, for the solves that follow.
- `scale` takes a feature 023 source and forwards it to `drto.scale`
  before the pipeline runs, so the model leaves initialization with its
  factors in place for the solves that follow.
  The sources that read no values, the bounds and a units mapping, are
  the ones that make sense here. The default, `scale=None`, writes
  nothing and leaves a suffix the caller wrote untouched.
- The steady path runs the pipeline on the model in place, so the solved
  values land in `Var.value`.
- The dynamic path requires a discretized horizon and no drto
  transformation applied yet, the same ordering guard as feature 005,
  and errors clearly otherwise. It reduces a throwaway clone with the
  feature 005 reduction, runs the pipeline there, and broadcasts: every
  variable indexed by the time set gets its collapsed counterpart's value
  at every grid point, and the state derivatives get zero. The source
  model's structure is untouched.
- Initializing before the dynamic transforms is sufficient. The later
  transforms seed their new components from the values in place
  (`drto.parameterize` seeds each move variable from the control member
  values it replaces, and `drto.infinite_horizon` copies the horizon-end
  values onto the segment copies), so the flat steady start propagates
  through the whole pipeline (verified against both transforms,
  2026-07-19).
- `controls=` follows the feature 008 convention: a mapping of declared
  control (the component, or its name) to the value the steady solve
  holds it at. Controls not in the mapping hold the values they already
  have, and an unknown name errors.
- A non-square system raises in both paths, consistently: the error names
  the unmatched variables and constraints from
  the pipeline's report and says what to fix (declare the missing decision,
  or remove the redundant specification). Deliberately partial
  initialization is `pyomo_pounce.initialize` called directly.
- It populates variable values only: no components are added or removed,
  and variable fixed flags are restored after the pipeline. It is a plain
  function, with no `apply_to` or `create_using` form.
- The return value tells the user what happened, printable in a
  notebook. The steady path returns the pipeline's `InitializeReport`
  as-is (fills, projection outcome, block counts), and the dynamic path
  returns a thin drto wrapper around it that adds the broadcast line,
  the variables seeded across the grid points and the derivatives
  zeroed.
- pyomo-pounce is an optional dependency, not a requirement of drto: it
  lives in the `pounce` extras group (`pip install drto[pounce]`, shared
  with the other pounce-backed features, the advanced-step controller of
  feature 012 among them), the import happens inside the function so
  `import drto` never touches it, and a missing install raises the
  house-style error naming the extra. drto's core stays solver-agnostic.
- A declared disturbance is process noise, zero in the nominal
  equilibrium. The pipeline holds it at zero for the solve, the same
  convention as every control-side mode, and restores the fixed flags it
  touched, and the broadcast lands the zeros across the grid.
