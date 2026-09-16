# drto.asnmpc

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want the advanced-step NMPC loop: the loop of
`drto.ideal_nmpc`, with the horizon solve running between samples at the
model's prediction of the next state and the measurement answered by the
fast correction of that solution, so that the loop I study is the one
where the expensive solve sits off the feedback path.

```python
import drto

# build: the model statement (feature 006)

history = drto.asnmpc(
    build,
    steps=50,                          # loop length, in samples
    h=1.0,                             # sampling time, to every builder
                                       # call. Omitted, the builder's
                                       # default
    initial_condition={"z": 0.2},      # written into the initial-condition
                                       # Params. Omitted, their current values
    dynamic_optimization={},           # the controller-only options: N,
                                       # infinite_horizon, tracking_weight
    advanced_step={},                  # options through to
                                       # drto.advanced_step_controller
    disturbances={"w": 0.05},          # per declared disturbance: a
                                       # number is a std dev of zero-mean
                                       # draws, a sequence the per-step
                                       # values. Omitted is zero
    seed=0,                            # makes the draws reproducible
    initialize="cold",                 # the first solve's
                                       # initialization: "cold" the cold
                                       # start (a mapping its options),
                                       # "steady" the steady broadcast,
                                       # False skips it
    solver="pounce",                   # the solver; the correction is a
                                       # pounce backsolve, so pounce only
)

drto.plot_states(history)
drto.plot_controls(history)
```

The loop is the asNMPC controller of Huang, Zavala, and Biegler,
J. Process Control 19 (2009) 678-685. The setup, the options, the
disturbance handling, the history, and the plotting are those of
`drto.ideal_nmpc` (feature 014), with the `advanced_step` options
passing through to `drto.advanced_step_controller` as given. `solver`
stays at `"pounce"`, since the correction is the feature 012 backsolve
and there is no session without a pounce solve.

The input is the model statement, under feature 006's builder contract,
and the loop builds its sides from it the way `drto.ideal_nmpc` does:
the controller over the declared horizon, the process and the predictor
over one sampling interval, `h`, `ncp`, and `scheme` stated once on the
loop and used for every side, and the controller-only options, `N`,
`infinite_horizon`, and `tracking_weight`, in the
`dynamic_optimization` mapping. A first argument that is not callable
and a mesh option repeated in that mapping are descriptive errors.

The loop builds one side more than the ideal one: alongside the
process, a predictor, built from the same statement the same way but
with its disturbances held at zero. The predictor is the controller's
own model run forward, the plant as the model expects it to move.

The first step solves at the initial state and implements the solution's
own first moves, there being no background solution to correct. Every
step then:

1. Predict: simulate the predictor one sample from the actual state
   under the implemented moves. Its end state is the predicted next
   state, written into the initial-condition Params.
2. Solve in background: warm start, then solve the dynamic optimization
   at the predicted state.
3. Simulate the process one sample from the actual state under the same
   implemented moves and the step's realization. Its end state is the
   new actual state.
4. Correct: write the new actual state into the initial-condition Params
   and call `drto.advanced_step_controller` on the background solution.
   Its first moves are the next implemented control.
5. Record the time, the actual state, the implemented moves, and the
   realization.

The last step records and stops, with no prediction, background solve,
or correction, since no step implements the moves they would produce.
The loop then makes `steps` controller solves, as `drto.ideal_nmpc`
does. Under `tee=True`, the history's logs hold the predictor's solves
beside the controller's and the process's, under the side
`"predictor"`.

The correction's perturbation is the gap between the prediction and the
measurement, one sample of disturbance and model error, not the state's
motion over the sample. With no disturbance and a perfect model it is
zero and the corrected moves are the background solution's own. The
correction runs before the next background solve because the solve
replaces the stored factorization, so the estimate must be taken from
the background solution while it is still the one in the session.

## Benefit hypothesis

The advanced-step loop is the framework's reason to exist: the horizon
solve moves off the measurement instant and the online step becomes a
backsolve whose perturbation is one sample of disturbance and model
error. Whether that trade holds on a given model, how far the correction
drifts from the true re-solve, and what the closed loop loses for it are
exactly the studies this function makes a one-liner, on the same history
and plots as the ideal loop it is compared against.

## Acceptance criteria

- `drto.asnmpc(build, steps, ...)` takes what `drto.ideal_nmpc` takes,
  plus `advanced_step` options passed through to
  `drto.advanced_step_controller` as given, and builds the controller
  and the process the same way from the statement, plus the predictor,
  a third side built the same way with its disturbances held at zero.
- The first step implements the solution's own first moves. Every later
  step implements the first moves of the advanced-step correction of the
  background solution at the newly simulated actual state, the
  background solution solved at the predictor's one-sample prediction
  from the previous actual state under the implemented moves.
- Each background solve warm starts at the predicted state, and the
  correction runs before the next solve replaces the stored
  factorization.
- The last step implements its moves, simulates the process, and
  records, with no prediction, background solve, or correction.
- The history and the plotting are those of `drto.ideal_nmpc`, and
  under `tee=True` the logs name the predictor's solves `"predictor"`.
- On hicks with zero disturbances, the prediction equals the simulated
  state, the correction is by zero, the implemented controls match the
  ideal loop's, and the actual states settle to the declared targets.
