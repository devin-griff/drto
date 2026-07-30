# drto.asnmpc

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want the advanced-step NMPC loop: the loop of
`drto.ideal_nmpc`, with the implemented control coming from the
advanced-step correction of the previous solution at the newly simulated
state instead of from a fresh solve, so that the loop I study is the one
where the solve happens between samples and the measurement is handled by
the fast correction.

```python
import drto

# m: the declared, discretized model, before drto.dynamic_optimization

history = drto.asnmpc(
    m,
    steps=50,                          # loop length, in samples
    initial_condition={"z": 0.2},      # written into the hooks; omitted,
                                       # the hooks' current values
    dynamic_optimization={},           # options through to the transform
    advanced_step={},                  # options through to
                                       # drto.advanced_step_controller
    disturbances={"w": 0.05},          # per declared disturbance: a
                                       # number is a std dev of zero-mean
                                       # draws, a sequence the per-step
                                       # values; omitted is zero
    seed=0,                            # makes the draws reproducible
    cold_start=True,                   # cold start the first solve
    solve=None,                        # callable for the solves; None is
                                       # a plain pounce solve
)

drto.plot_states(history)
drto.plot_controls(history)
```

The setup, the options, the disturbance handling, the history, and the
plotting are those of `drto.ideal_nmpc` (feature 014), with the
`advanced_step` options passing through to
`drto.advanced_step_controller` as given. The controller solves must be
pounce solves: the correction is the feature 012 backsolve, and there is
no session without one.

The loop differs from the ideal one in where the implemented control
comes from. The first step implements the first move of the solution
itself, there being no previous solution to correct. Every later step
implements the correction of the previous solution at the state the
simulation just produced:

1. Simulate the process one sample under the implemented control and the
   step's realization; the end state is the new actual state, written
   into the initial-condition Params.
2. Correct: `drto.advanced_step_controller` on the previous solution at
   the hooks, before anything else touches the model; its first moves are
   the next implemented control.
3. Solve: warm start, then solve the dynamic optimization at the new
   actual state. This solution is the one the next step's correction
   works from.
4. Record the time, the actual state, the implemented moves, and the
   realization.

The correction runs before the solve because the solve replaces the
stored factorization: the estimate must be taken from the previous
solution while it is still the one in the session.

## Benefit hypothesis

The advanced-step loop is the framework's reason to exist: the horizon
solve moves off the measurement instant and the online step becomes a
backsolve. Whether that trade holds on a given model, how far the
correction drifts from the true re-solve, and what the closed loop loses
for it are exactly the studies this function makes a one-liner, on the
same history and plots as the ideal loop it is compared against.

## Acceptance criteria

- `drto.asnmpc(m, steps, ...)` takes what `drto.ideal_nmpc` takes, plus
  `advanced_step` options passed through to
  `drto.advanced_step_controller` as given, and builds the controller and
  the process the same way.
- The first step implements the solution's own first moves. Every later
  step implements the first moves of the advanced-step correction of the
  previous solution at the newly simulated state, taken before the next
  solve runs.
- Each step then warm starts and solves at the new actual state, and the
  process simulates the next sample under the implemented control and the
  step's realization.
- The history and the plotting are those of `drto.ideal_nmpc`.
- On hicks with zero disturbances, the actual states settle to the
  declared targets, and the implemented controls track the ideal loop's
  to first order.
