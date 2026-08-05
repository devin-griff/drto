# drto.nonideal_nmpc

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want the NMPC loop with the solve time in it: the
loop of `drto.ideal_nmpc`, with each new move taking effect one solve
time after its measurement, the previous move holding in the meantime, so
that the closed loop pays for the computation the way a real controller
does.

```python
import drto

# m: the declared, discretized model, before drto.dynamic_optimization

history = drto.nonideal_nmpc(
    m,
    steps=50,                          # loop length, in samples
    initial_condition={"z": 0.2},      # written into the initial-condition
                                       # Params; omitted, their current values
    dynamic_optimization={},           # options through to the transform
    delay=None,                        # None takes each solve's time as
                                       # the solver reports it; a number
                                       # or per-step sequence prescribes
                                       # the delay instead
    disturbances={"w": 0.05},          # per declared disturbance: a
                                       # number is a std dev of zero-mean
                                       # draws, a sequence the per-step
                                       # values; omitted is zero
    seed=0,                            # makes the draws reproducible
    initialize="cold",                 # the first solve's
                                       # initialization: "cold" the cold
                                       # start (a mapping its options),
                                       # "steady" the steady broadcast,
                                       # False skips it
    solver="pounce",                   # the solver, by name
)

drto.plot_states(history)
drto.plot_controls(history)
```

The setup, the options, the disturbance handling, and the plotting are
those of `drto.ideal_nmpc` (feature 014), with `delay` added.

The loop differs from the ideal one in when a move takes effect. The
solve at the step's measurement takes `delta`: the solve time the solver
reports, in seconds, converted into the declared time set's units, or the
value `delay` prescribes, already in those units. A model whose time
carries no units cannot place reported seconds on its time axis, so
`delay=None` there is a descriptive error asking for a prescribed delay.
The new move takes effect `delta` into the sample. Every
simulation step therefore carries two control actions, and runs as two
finite elements of different lengths: the process advances `delta` under
the previous move, then `h - delta` under the new one. The process
simulation's duration is a parameter, so both lengths are exact on its
fixed grid. A piece of zero length is not simulated: a zero delay runs
the sample in one piece under the new move, and a delay at the sample
length one piece under the previous move. On the first step, the previous
move is the declared control targets, the inputs the process sat at
before the loop.

A solve that outruns the sample, `delta >= h`, keeps the previous move
for the whole sample, and its move takes effect at the next sample
boundary, recorded as clamped.

The history is `drto.ideal_nmpc`'s, adding each step's delay and the time
each move took effect; the moves plot as a staircase stepping at those
times.

## Benefit hypothesis

Nonideal is the honest middle of the three execution variants: ideal
pretends the solve is free, advanced-step hides it between samples, and
this one pays it in the open. It is the baseline that shows what the
advanced step buys: against ideal alone, the correction looks like a
cheap approximation of a free solve; against nonideal, it recovers the
performance the delay costs. The same history and plots make the
three-way comparison direct.

## Acceptance criteria

- `drto.nonideal_nmpc(m, steps, ...)` takes what `drto.ideal_nmpc` takes
  plus `delay`, and builds the controller and the process the same way.
- Each step solves at the measurement, the delay the solver-reported
  solve time converted into the declared time set's units when `delay` is
  None and a number or per-step sequence in those units otherwise; on a
  model whose time carries no units, `delay=None` is a descriptive error.
  The sample
  simulates as two elements of exact lengths, `delta` under the previous
  move and `h - delta` under the new one, the duration a parameter of
  the process simulation and a zero-length piece not simulated.
- The first step's previous move is the declared control targets. A delay
  at or past the sample end keeps the previous move for the whole sample
  and the move takes effect at the next boundary, recorded as clamped.
- `delay=0` reproduces `drto.ideal_nmpc` exactly; `delay=h` is the
  one-sample-delay baseline.
- The history adds each step's delay and each move's effect time, and the
  moves plot as a staircase stepping at those times.
- On hicks with zero disturbances and a delay under a sample, the actual
  states settle to the declared targets.
