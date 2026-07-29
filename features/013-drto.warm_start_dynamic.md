# drto.warm_start_dynamic

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that shifts the previous solution one
sample forward to seed the next solve, so that every receding-horizon
iteration after the first starts warm.

```python
import drto

# ... m solved at time k; the loop implements the first move and
# updates the initial-condition Params with the measurement ...

drto.warm_start_dynamic(m)
```

Values only, in place, on the model the loop holds: a dynamic optimization
or dynamic simulation, with or without a terminal segment, or the declared
discretized model. The registry reads the live components at every stage.

One rule: everything moves one sampling interval forward in real time.
Every time-indexed variable, states, controls and their move variables,
algebraic variables, derivatives, and `Block(time)` members alike, takes
the previous solution's value at `t + h` wherever the previous solution
covers that time, and the declared steady-state targets wherever it does
not, the state derivatives zero and the algebraic variables keeping the
values they hold on those points. A target needed but not declared is a
descriptive error naming the component. No solve is run.

Reading the previous solution: on the horizon the grid repeats interval to
interval, so the shift is an exact copy, `v[t] <- v[t + h]`. Off the grid,
the value comes from the element's collocation interpolant, the Lagrange
evaluation `control_value` already does for controls, applied to every
variable. A terminal segment is part of the previous solution like the
rest of it, its points sitting in real time through
`t = tN + atanh(tau)/gamma`: the new value at `tau` is the old solution at
`tanh(atanh(tau) + gamma h)`, and the mesh rule `tanh(gamma h) = tau_11`
places the old tail's first collocation point exactly one sample past the
horizon end. A consequence, not a case: with a segment attached the
previous solution covers the entire shifted problem, so the target
fallback never fires and no pairing is needed.

The initial-condition Params are the loop's job, not the shift's: the
shifted state at the first point is the model's one-sample prediction, and
the loop overwrites the Params with the measurement.

The counterpart to `drto.cold_start_dynamic` (feature 011): 011 seeds the
first solve from the declarations, this seeds every solve after it from
the previous solution.

## Benefit hypothesis

The receding-horizon warm start is what makes each iteration converge in a
few steps instead of from scratch: the shifted solution is already nearly
optimal for the new problem. Shifting by copy and interpolant evaluation
is cheap and solve-free, and the infinite tail means the previous solution
already contains the horizon end that a finite formulation has to invent.

## Acceptance criteria

- `drto.warm_start_dynamic(m)` populates variable values only, in place,
  adds and removes nothing, and restores the fixed flags it touches, on
  the declared discretized model, a dynamic optimization, and a dynamic
  simulation, each with or without a terminal segment.
- Every time-indexed variable, `Block(time)` members included, takes the
  previous solution's value at `t + h` where the previous solution covers
  it, and the declared targets where it does not, state derivatives zero
  and algebraic variables keeping their values on the filled points; a
  missing needed pairing raises a descriptive error naming the component.
  No solve is run.
- Values read as exact copies on the repeating grid and through the
  element's collocation interpolant off it; a terminal segment reads
  through its time map, the old tail's first collocation point sitting
  one sample past the horizon end by the mesh rule.
- With a segment attached, the previous solution covers the entire
  shifted problem and no steady-state pairing is required.
- A solution resting at the targets shifts to itself.
- Returns a readable report in the feature 010 shape: what was copied,
  what was evaluated, what was filled.
