# drto.cold_start_dynamic

**Status:** draft

## Description

As a user of DRTO, I want a function that simulates my model forward under
nominal controls to seed the first dynamic solve, so that a problem with no
prior solution starts from a feasible trajectory.

## Benefit hypothesis

A forward simulation gives a feasible starting trajectory for the first solve of
a receding-horizon run, removing the need to hand-roll an initial guess and
reducing first-solve failures.

## Acceptance criteria

- `drto.cold_start_dynamic(m)` simulates the model forward from the initial
  condition under nominal controls, producing a feasible starting trajectory.
- It populates variable values (the initial guess) and does not change the
  model structure. It is a plain function, with no `apply_to` or `create_using`
  form.
- The forward simulation it uses is the `drto.dynamic_simulation` mode
  (feature 006).
