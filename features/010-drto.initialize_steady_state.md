# drto.initialize_steady_state

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

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

# a steady-state model (authored steady, or a feature 005 reduction):
drto.initialize_steady_state(ss)  # in place: fill, project, block-solve
```

## Benefit hypothesis

A steady-state-based initial guess is often the difference between a solve
that converges and one that stalls, and deriving it from the model keeps it
consistent with the dynamics. One function serves both declared shapes,
since the core calculation is the same (USER DECISION 2026-07-19), and it
replaces the hand-written per-model initialization helpers the examples
carry today.

## Acceptance criteria

- `drto.initialize_steady_state(m, controls=None)` dispatches on the
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
- The steady path runs the pipeline on the model in place: the solved
  values land in `Var.value`.
- The dynamic path requires a discretized horizon and no drto
  transformation applied yet, the same sibling-branch guard as feature 005,
  and errors clearly otherwise. It reduces a throwaway clone with the
  feature 005 reduction, runs the pipeline there, and broadcasts: every
  variable indexed by the time set gets its collapsed counterpart's value
  at every grid point, and the state derivatives get zero. The source
  model's structure is untouched.
- Initializing before the dynamic transforms is sufficient: the later
  transforms carry values forward on their own (`drto.parameterize` seeds
  each move variable from the control member values it replaces, and
  `drto.infinite_horizon` copies the horizon-end values onto the segment
  copies), so the flat steady start propagates through the whole pipeline
  (verified against both transforms, 2026-07-19).
- `controls=` follows the feature 008 convention: a mapping of declared
  control (the component, or its name) to the value the steady solve holds
  it at; controls not in the mapping hold the values they already have; an
  unknown name errors.
- A non-square system raises in both paths, consistently (USER DECISION
  2026-07-19): the error names the unmatched variables and constraints from
  the pipeline's report and says what to fix (declare the missing decision,
  or remove the redundant specification). Deliberately partial
  initialization is `pyomo_pounce.initialize` called directly.
- It populates variable values only: no components are added or removed,
  and variable fixed flags are restored after the pipeline. It is a plain
  function, with no `apply_to` or `create_using` form.
- The return value tells the user what happened, printable in a notebook:
  the steady path returns the pipeline's `InitializeReport` as-is (fills,
  projection outcome, block counts); the dynamic path returns a thin drto
  wrapper around it that adds the broadcast line, the variables seeded
  across the grid points and the derivatives zeroed (USER DECISION
  2026-07-19).
- pyomo-pounce is an optional dependency, not a requirement of drto: it
  lives in the `pounce` extras group (`pip install drto[pounce]`, shared
  with the future pounce-backed features, e.g. the advanced-step
  controller and sensitivities), the import happens inside the function so
  `import drto` never touches it, and a missing install raises the
  house-style error naming the extra (USER DECISION 2026-07-19). drto's
  core stays solver-agnostic.
