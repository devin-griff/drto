# drto.asnmpc

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

## Description

As a user of DRTO, I want the advanced-step NMPC loop, so that I can
study a closed loop whose expensive solve runs between samples rather
than at the measurement. It is the loop of `drto.ideal_nmpc` with the
horizon solved during each sample at the model's prediction of the next
state, and that solution corrected to the measurement when it arrives.

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
                                       # "steady" initialize_steady_state,
                                       # False skips it
    solver="pounce",                   # "pounce" or "pounce_v2". The
                                       # correction is a pounce backsolve
)

drto.plot_states(history)
drto.plot_controls(history)
```

The loop is the asNMPC controller of Huang, Zavala, and Biegler,
J. Process Control 19 (2009) 678-685. The setup, the options, the
disturbance handling, the history, and the plotting are those of
`drto.ideal_nmpc` (feature 014), with the `advanced_step` options
passing through to `drto.advanced_step_controller` as given. `solver`
takes `"pounce"` or `"pounce_v2"`, since the correction is the feature
012 backsolve, which needs the factorization a pounce solve keeps.

The input is the model statement, under feature 006's builder contract,
and the loop builds its sides from it the way `drto.ideal_nmpc` does:
the controller over the declared horizon, the process and the predictor
over one sampling interval, `h`, `ncp`, and `scheme` stated once on the
loop and used for every side, and the controller-only options, `N`,
`infinite_horizon`, and `tracking_weight`, in the
`dynamic_optimization` mapping. A first argument that is not callable,
a mesh option repeated in that mapping, any other solver name, and an
`advanced_step` that is not a mapping are descriptive errors.

The loop builds one side more than the ideal one. Alongside the process
it builds a predictor from the same statement the same way, with its
disturbances held at zero. The predictor is the controller's own model
simulated forward, so its end state is the state the controller
predicts.

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
the background solution while it is still the one in the session. When
the loop returns, including on a failed solve, it frees the controller's
pounce factorization.

## Benefit hypothesis

The user runs the advanced-step loop in one call and compares it with
the ideal loop on the same history and plots, which lets them study three
questions on a given model: whether solving between samples and
correcting with a backsolve works there, how far the corrected moves are
from a full re-solve's, and what the closed loop gives up for it. The
online step is a backsolve whose perturbation is one sample of
disturbance and model error, rather than the state's motion over the
sample.

## Acceptance criteria

- `drto.asnmpc(build, steps, ...)` takes what `drto.ideal_nmpc` takes,
  plus `advanced_step` options passed through to
  `drto.advanced_step_controller` as given, and builds the controller
  and the process the same way from the statement, plus the predictor,
  a third side built the same way with its disturbances held at zero.
- A solver other than `"pounce"` or `"pounce_v2"`, and an
  `advanced_step` that is not a mapping, are descriptive errors raised
  before anything is built.
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
  state, the correction is zero, the implemented controls match the
  ideal loop's, and the actual states settle to the declared targets.
- When the loop returns, including on a failed solve, the controller
  holds no pounce factorization.
