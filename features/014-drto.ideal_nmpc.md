# drto.ideal_nmpc

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

## Description

As a user of DRTO, I want the ideal NMPC loop: at each step, solve the
dynamic optimization at the current state, implement the first control
move on the process, and simulate the process one sample forward under
that move and a disturbance realization, so that I can run and study the
closed loop the declarations describe.

```python
import drto

# build: the model statement, a function (feature 006)

history = drto.ideal_nmpc(
    build,
    steps=50,                          # loop length, in samples
    h=1.0,                             # sampling time, for both sides
    initial_condition={"z": 0.2},      # written into the initial-condition
                                       # Params; omitted, their current values
    dynamic_optimization={},           # options through to the transform
    disturbances={"w": 0.05},          # per declared disturbance: a
                                       # number is a std dev of zero-mean
                                       # draws, a sequence the per-step
                                       # values; omitted is zero
    seed=0,                            # makes the draws reproducible
    initialize="cold",                 # the first solve's
                                       # initialization: "cold" runs
                                       # drto.cold_start_dynamic (a
                                       # mapping passes its options),
                                       # "steady" runs
                                       # drto.initialize_steady_state,
                                       # False skips it
    solver="pounce",                   # the solver, by name
    warm_start={"mu_init": 1e-6},      # options for the warm-started
                                       # solves, laid over the default
                                       # recipe
    tee=False,                         # True streams every solve's
                                       # output and returns it on the
                                       # history
)

drto.plot_states(history)
drto.plot_controls(history)
```

The input is the model statement, and the builder contract is feature
006's. The loop calls it twice, which is what makes the two sides the
same physics: the controller over the declared horizon and the plant
over one sampling interval, at `N=1`. `h`, `ncp`, and `scheme` belong to
the loop and reach both calls, so the two grids agree by construction,
and repeating one of them in `dynamic_optimization` is an error. That
mapping carries the controller-only options, `N`, `infinite_horizon`,
and `tracking_weight`.

Each side is discretized, initialized when asked, and then transformed,
the plant by `drto.dynamic_simulation` with its controls first fixed at
the declared control targets and the controller by
`drto.infinite_horizon` when asked and then
`drto.dynamic_optimization`. The plant is built over one sampling
interval rather than cut back to one, so nothing is deleted and it never
carries a terminal segment.

The first actual state is the initial condition: `initial_condition`, a
mapping of declared state names to values, is written into the
initial-condition Params before the first step. Omitted, the Params'
current values are used.

Ideal means the solve is treated as instantaneous: the measurement
arrives, the problem is solved, and the move is implemented at the same
instant. The loop, each step:

1. Initialize, per the `initialize` option. The cold start is the
   default, the controller and the process alike so the plant's first
   simulation starts initialized too, a mapping passing through as
   `drto.cold_start_dynamic`'s options. `"steady"` is the steady
   broadcast, `drto.initialize_steady_state` running on the input
   before the sides are built so both inherit it, and `False` is
   nothing. `drto.warm_start_dynamic` runs on every later step.
2. Solve the dynamic optimization at the current initial condition.
3. Implement: read each control's first move and write it into the
   process clone's fixed controls.
4. Realize: fix the process clone's disturbances at this step's values.
5. Simulate the one-sample process from the current actual state. Its
   end state is the new actual state, written into both models' initial
   conditions.
6. Record the time, the actual state, the implemented moves, and the
   realization.

The disturbance enters the process only, the controller solving at
zero disturbance, the optimization mode's convention.

Each declared disturbance's entry is either a sequence, the realization
per step as given, or a number, the standard deviation of independent
zero-mean normal draws each step, reproducible through `seed`. A disturbance
with no entry is zero.

`solver` names the solver that runs every controller and process solve.
Under `"ipopt"` every warm-started solve runs with the warm start
recipe, and `warm_start` lays user options over it: the default is
`warm_start_init_point=yes`, `mu_init=1e-6`, and the `1e-9` bound and
multiplier pushes, so `warm_start={"mu_init": 1e-4}` retunes one option
without restating the rest. Under the pounce names the default is
`mu_init=1e-6` alone, the shifted start with the barrier already
small, the one recipe option measured to help pounce. On the CSTR warm
start it takes the shifted solve from ten iterations to seven, while
the full recipe takes it to 867, a regression that requires its three
ingredients together, the warm-start switch, the small barrier, and
the 1e-9 pushes. Under any other solver the loop warm starts on the
shifted values alone, a given `warm_start` mapping passing to the
solves as is. A solve that fails stops the loop with an error naming
the step.

`tee=True` streams every solve's output as the loop runs and returns
it. The history's `logs` holds one entry per solve in loop order, the
step, the side (controller or process), and the solver's text. The
default is quiet, nothing streamed, nothing kept.

An active `scaling_factor` suffix reaches every solver solve on that
side, and the history reads back in the model's own units. The
initial-condition Params stay physical, and the cold starts' block
solves run in the model's own units either way (gh #92). `scale` takes
a feature 023 source, `"point"`, `"bounds"`, or a mapping of units to
magnitudes, and forwards it to `drto.scale` at entry, before the sides
are built, so both sides hold the factors and every solver solve
receives them. They are written once and held for the whole loop. A caller choosing `"point"` passes the model at the
point to measure. The default, `scale=None`, writes nothing and honors
what the caller wrote.

The returned history holds the actual trajectory: the times, each
declared state member's actual values, the implemented moves, and the
realizations, with the declared names, targets, and bounds (each
recorded label's declared (lo, hi), which the plots draw). `drto.plot_states` and
`drto.plot_controls` accept it and draw the actual closed-loop
trajectories the same way they draw a model's, the implemented moves as
the staircase they physically are.

## Benefit hypothesis

A closed-loop study is a single call on the declared model, where a
hand-written loop is a page of code whose every line touches an
internal detail: which container holds a control's first move after
parameterization, which Params take each measurement, how to shift the
previous solution, how to keep the process model consistent with the
controller's, how to seed reproducible noise, how to collect the
results in a plottable form. Each user re-derives those details, and a
loop that gets one wrong runs and quietly studies the wrong thing. The
packaged loop reads all of them from the registry, is tested against
the acceptance criteria below, stays correct as the surface evolves,
and returns a history that plots in one line.

## Acceptance criteria

- `drto.ideal_nmpc(build, steps, ...)` takes the model statement and
  errors descriptively on anything not callable. The builder contract is
  feature 006's.
- It builds both sides from that one statement, the controller over the
  declared horizon and the plant at `N=1`, passing `h`, `ncp`, and
  `scheme` to both calls, so the two grids agree by construction. A
  `dynamic_optimization` mapping repeating one of those three is a
  descriptive error, and the mapping otherwise carries `N`,
  `infinite_horizon`, and `tracking_weight` for the controller alone.
- Each side is discretized, then initialized when asked, then
  transformed: the plant by `drto.dynamic_simulation` with its controls
  fixed at the declared control targets, and the controller by
  `drto.infinite_horizon` when asked and then
  `drto.dynamic_optimization`. No active plant member or constraint lies
  past one sampling time, the plant carries no terminal segment, and its
  cold start and each plant solve are one element's worth.
- `initialize="steady"` runs `drto.initialize_steady_state` on each side
  after it is discretized and before its transforms, the only point that
  function accepts, so it runs with a terminal segment asked for.
- `initial_condition` writes the given state values into the
  initial-condition Params before the first step. Omitted, the Params'
  current values are the first actual state.
- Each step solves the controller at the current initial condition,
  implements the first moves on the process clone, fixes its
  disturbances at the step's realization, simulates, and feeds the state
  one sample in back as both models' initial condition.
- `initialize` picks the first solve's initialization. `"cold"` (the
  default) runs `drto.cold_start_dynamic` on the controller and the
  process alike, a mapping passing through as its options. `"steady"`
  runs `drto.initialize_steady_state` on the input before the sides are
  built, so both inherit the broadcast, under that function's own
  contract (the input precedes `drto.infinite_horizon`, whose violation
  raises its descriptive error). `False` skips initialization, and
  anything else is a descriptive error. Every later solve is
  warm-started.
- A disturbance entry that is a sequence is used as given, a number
  draws independent zero-mean normal realizations with that standard
  deviation, reproducibly under `seed`, and a missing entry is zero.
- `solver` names the solver for every controller and process solve.
  Under `"ipopt"` every warm-started solve runs with the warm start
  recipe (`warm_start_init_point=yes`, `mu_init=1e-6`, the `1e-9`
  pushes), a `warm_start` mapping laid over it. Under the pounce names
  the warm solves carry `mu_init=1e-6` alone, the mapping laid over
  it. Under any other solver the loop warm starts on the shifted
  values alone, a given mapping passing through as is. A failed solve
  raises an error naming the step.
- `tee=True` streams each solve's output and returns it. The history's
  `logs` holds (step, side, text) for every controller and process
  solve in loop order, and the default keeps and prints nothing.
- With an active `scaling_factor` suffix every solve on that side
  receives the factors, the history lands in the model's own units, and
  a hicks loop with factors written reproduces the unscaled loop's
  history.
- With `scale` given a feature 023 source, the loop writes the factors
  through `drto.scale` at entry, before the sides are built, so both
  sides hold them and every solver solve receives them, with the cold
  starts' block solves running in the model's own units (gh #92). The
  default `scale=None` writes no factors.
- The history holds times, actual states, implemented moves, and
  realizations under their declared names. `drto.plot_states` and
  `drto.plot_controls` gain a second input kind: handed a history instead
  of a model, they draw its actual trajectories, the moves as a
  staircase, with the setpoint lines from the recorded targets.
- On hicks with zero disturbances, the actual states settle to the
  declared targets.
