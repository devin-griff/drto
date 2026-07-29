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
not. No solve is run anywhere.

- **On the horizon**: the grid repeats interval to interval, so the shift
  is an exact copy, `v[t] <- v[t + h]`.
- **Across the junction and on the tail**, when a terminal segment is
  attached: the previous solution extends past the horizon end, so the
  shift stays inside it. The exposed last interval reads the old tail's
  first element, whose first collocation point the mesh rule
  `tanh(gamma h) = tau_11` places exactly one sample past the horizon
  end: this shift is what the rule exists for. The segment itself shifts
  one sample deeper by the same map, the new value at `tau` being the old
  solution at `tanh(atanh(tau) + gamma h)`, which lands strictly inside
  the old grid, so the tail always covers itself. Off-grid values come
  from the element's collocation interpolant, the Lagrange evaluation
  `control_value` already does for controls, applied to every shifted
  variable. Nothing here needs the targets.
- **On uncovered points**, which exist only without a segment, on the
  exposed last interval: states, controls, and moves take their declared
  `steady_state` and `steady_state_control` targets, the state
  derivatives zero; algebraic variables keep the values they hold. A
  target needed but not declared is a descriptive error naming the
  component.

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
is cheap and solve-free, and with a terminal segment attached it needs no
data beyond the previous solution, the tail supplying the horizon end that
a finite formulation has to invent.

## Acceptance criteria

- `drto.warm_start_dynamic(m)` populates variable values only, in place,
  adds and removes nothing, and restores the fixed flags it touches, on
  the declared discretized model, a dynamic optimization, and a dynamic
  simulation, each with or without a terminal segment.
- Every time-indexed variable, `Block(time)` members included, takes the
  previous solution's value at `t + h`: exact copies on the repeating
  grid, the element's collocation interpolant off it. No solve is run.
- With a terminal segment, the exposed last interval reads the old tail's
  first element and the segment shifts by `tanh(atanh(tau) + gamma h)`;
  the whole shifted problem is covered by the previous solution and no
  steady-state pairing is required.
- Without one, the exposed interval's states, controls, and moves take
  their declared targets and the state derivatives zero, algebraic
  variables keeping their values; a missing needed pairing raises a
  descriptive error naming the component.
- A solution resting at the targets shifts to itself.
- Returns a readable report in the feature 010 shape: what was copied,
  what was evaluated, what was filled.
