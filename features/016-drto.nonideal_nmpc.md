# drto.nonideal_nmpc

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want the NMPC loop with the solve time in it. It is
the loop of `drto.ideal_nmpc`, with each new move taking effect one delay
after its measurement and the previous move holding in the meantime, so
that the closed loop is delayed by the computation the way a real
controller is.

```python
import drto

# m: the declared, discretized model, before drto.dynamic_optimization

history = drto.nonideal_nmpc(
    m,
    steps=50,                          # loop length, in samples
    initial_condition={"z": 0.2},      # written into the initial-condition
                                       # Params. Omitted, they keep their
                                       # current values
    dynamic_optimization={},           # options through to the transform
    delay="solver",                    # "solver" takes each solve's time
                                       # as the solver reports it. A number
                                       # or per-step sequence prescribes
                                       # the delay instead
    disturbances={"w": 0.05},          # per declared disturbance: a
                                       # number is a std dev of zero-mean
                                       # draws, a sequence the per-step
                                       # values. Omitted is zero
    seed=0,                            # makes the draws reproducible
    initialize="cold",                 # the first solve's
                                       # initialization: "cold" runs
                                       # drto.cold_start_dynamic (a
                                       # mapping passes its options),
                                       # "steady" runs
                                       # drto.initialize_steady_state,
                                       # False skips it
    solver="pounce",                   # the solver, by name
)

drto.plot_states(history)
drto.plot_controls(history)
```

The setup, the options, the disturbance handling, and the plotting are
those of `drto.ideal_nmpc` (feature 014), with `delay` added.

The loop differs from the ideal one in when a move takes effect. Each
step's delay is the solve time the solver reports, in seconds, converted
into the declared time set's units, or the value `delay` prescribes,
already in those units. With no units on the declared time set the
reported seconds cannot be converted, so `delay="solver"` is a
descriptive error saying to prescribe a delay instead. The new move takes
effect one delay into the sample. Two moves therefore apply in each
sample, which runs as two simulations of the one-element plant. The
process advances for the delay under the previous move, then for the
rest of the sample under the new one. The process simulation's duration
is a parameter, so both lengths are exact and the plant is discretized
once. A piece of zero length is not simulated. A zero delay runs the
sample in one piece under the new move, and a delay at the sample length
runs it in one piece under the previous move. On the first step, the
move is the declared control targets, the inputs the process held
before the loop.

When a solve takes longer than the sample, a delay at or past the
sampling time, the previous move holds for the whole sample and the new
move takes effect at the next sample boundary, recorded as clamped.

The history is `drto.ideal_nmpc`'s, adding each step's delay and the time
each move took effect. The moves plot as a staircase stepping at those
times.

## Benefit hypothesis

A closed loop that accounts for the controller's computation time is
cumbersome to program by hand. Each sample has to be split at the moment
the solve returns, the process advanced under the previous move and
then under the new one, and the split point recomputed every step from
that step's solve time. Writing that once per model is enough work that
the comparison usually goes unmade.

`drto.nonideal_nmpc` runs the loop from the same call `drto.ideal_nmpc`
takes, with one added argument, on any declared model. It is the
baseline the faster implementations are measured against.
`drto.advanced_step_controller` and the fitted policy of feature 026 both
exist to take the delay out of the loop, and what they remove is only
measurable against a loop that has it.

## Acceptance criteria

- `drto.nonideal_nmpc(m, steps, ...)` takes what `drto.ideal_nmpc` takes
  plus `delay`, and builds the controller and the process the same way.
- Each step solves at the measurement. The delay is the solver-reported
  solve time converted into the declared time set's units when `delay` is
  `"solver"`, and a number or per-step sequence in those units otherwise.
  On a model whose time set has no units, `delay="solver"` is a
  descriptive error. The sample runs as two simulations of exact
  lengths, the delay under the previous move and the rest of the sample
  under the new one, the duration a parameter of the process simulation
  and a zero-length piece not simulated.
- The first step's previous move is the declared control targets. A delay
  at or past the sample end keeps the previous move for the whole sample
  and the move takes effect at the next boundary, recorded as clamped.
- `delay=0` reproduces `drto.ideal_nmpc` exactly. A delay at the
  sampling time holds every move for a full sample before it takes
  effect.
- The history adds each step's delay and each move's effect time, and the
  moves plot as a staircase stepping at those times.
- On hicks with zero disturbances and a delay shorter than the sampling
  time, the actual states settle to the declared targets.
