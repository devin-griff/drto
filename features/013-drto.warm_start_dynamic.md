# drto.warm_start_dynamic

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

## Description

As a user of DRTO, I want to reuse the previous solution, moved one step
forward, as the initialization of the current problem, so that each solve
in a closed loop starts from the last one.

```python
import drto

# ... m solved; the loop implements the first move and sets the
# initial condition from the measurement ...

drto.warm_start_dynamic(m)
```

It sets values only and changes nothing else. It works on the model the
loop holds, transformed or not, with or without the infinite-horizon tail.

Every variable takes the value the previous solution had one sampling time
later. Where the grids line up, that is a copy. Where they do not, between
grid points and on the tail, whose points sit in time through
`t = tN + atanh(tau)/gamma`, the previous solution is interpolated. Past
the end of the previous solution, which only exists when there is no tail,
states and controls take their declared steady-state targets, derivatives
zero, and algebraic variables keep their values. Nothing is solved.

The initial condition is not touched: the loop sets it from the
measurement.

## Benefit hypothesis

Under closed-loop control the next problem is the last one moved one step,
so the last solution moved one step is nearly its answer. Starting there
is what makes each solve fast, and it costs only copies and interpolation.

## Acceptance criteria

- `drto.warm_start_dynamic(m)` sets values only: nothing added, nothing
  removed, and fixed variables left alone, value and flag.
- Every variable takes the previous solution's value one sampling time
  later: copied where the grids line up, interpolated where they do not,
  tail included.
- Past the end of the previous solution, states and controls take their
  steady-state targets, derivatives zero, and algebraic variables keep
  their values; a missing target is an error naming the component.
- With a tail there is no past the end: the previous solution covers the
  whole problem and no targets are needed.
- A solution resting at the targets shifts to itself.
- The multipliers are part of the previous solution: when the model
  carries the suffixes (`dual`, `ipopt_zL/zU_out` with `ipopt_zL/zU_in`
  declared), the shift moves them too, equality duals within each
  constraint family through the recorded tail rows, bound multipliers
  over the same trajectories as the primals. Every solve-1 bound
  multiplier seeds the `_in` suffixes densely first, since an absent
  entry reads as zero to the solver, then the shifted trajectories
  overwrite theirs. Absent suffixes are skipped; the report names what
  was shifted. Solver warm-start options belong to the solve call, not
  to the shift.
- Returns a readable report of what was copied, interpolated, and filled.
