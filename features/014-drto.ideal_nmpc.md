# drto.ideal_nmpc

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want the ideal NMPC loop: at each step, solve the
dynamic optimization at the current state, implement the first control
move on the process, and simulate the process one sample forward under
that move and a disturbance realization, so that I can run and study the
closed loop the declarations describe.

```python
import drto

# m: the declared, discretized model, untransformed

history = drto.ideal_nmpc(m, steps=50, infinite_horizon=True,
                          disturbances={"w": 0.05}, seed=0)

drto.plot_states(history)
drto.plot_controls(history)
```

The input is the declared, discretized model, before any transform. The
loop builds both sides from it: a clone becomes the process, transformed
by `drto.dynamic_simulation` with its controls first fixed at the
declared control targets, and the model itself becomes the controller,
`drto.infinite_horizon` applied first when the `infinite_horizon` option
is given (`True` for the defaults, a dict for its options), then
`drto.dynamic_optimization`. The `infinite_horizon` and
`dynamic_optimization` options pass through to the transforms as given,
so their options stay reachable without changes here.

The first actual state is the initial condition: `initial_condition`, a
mapping of declared state names to values, is written into the
initial-condition Params before the first step; omitted, the Params'
current values are used.

Ideal means the solve is treated as instantaneous: the measurement
arrives, the problem is solved, and the move is implemented at the same
instant. The loop, each step:

1. Initialize: `drto.cold_start_dynamic` on the first step (on by
   default, `cold_start=False` skips it), `drto.warm_start_dynamic` on
   every later one.
2. Solve the dynamic optimization at the current initial condition.
3. Implement: read each control's first move and write it into the
   process clone's fixed controls.
4. Realize: fix the process clone's disturbances at this step's values.
5. Simulate the process clone from the current actual state and read the
   state one sample in; that is the new actual state, written into both
   models' initial conditions.
6. Record the time, the actual state, the implemented moves, and the
   realization.

The disturbance enters the process only: the controller solves at zero
disturbance, the optimization mode's convention.

Each declared disturbance's entry is either a sequence, the realization
per step as given, or a number, the standard deviation of independent
zero-mean draws each step, reproducible through `seed`. A disturbance
with no entry is zero.

`solve` is a callable applied to the controller and the process,
defaulting to a plain pounce solve, so a model that needs its own solve
wrapper (the scaled solve of the IDAES example) passes it in. A solve
that fails stops the loop with an error naming the step.

The returned history holds the actual trajectory: the times, each
declared state member's actual values, the implemented moves, and the
realizations, with the declared names and targets. `drto.plot_states` and
`drto.plot_controls` accept it and draw the actual closed-loop
trajectories the same way they draw a model's, the implemented moves as
the staircase they physically are.

## Benefit hypothesis

A hand-written closed loop is a page of code whose every line touches an
internal detail: which container holds a control's first move after
parameterization, which Params are the feedback hooks, how to shift a
solution that carries a tail, how to keep the process model consistent
with the controller's, how to seed reproducible noise, how to collect the
results in a plottable form. Each user re-derives those details, and a
loop that gets one wrong runs and quietly studies the wrong thing. The
packaged loop reads all of them from the registry, is tested against the
acceptance criteria below, stays correct as the surface evolves, and
returns a history that plots in one line, so a closed-loop study is a
single call on the declared model.

## Acceptance criteria

- `drto.ideal_nmpc(m, steps, ...)` requires a declared, discretized,
  untransformed model and errors descriptively otherwise. It clones the
  process before transforming, puts the clone in simulation mode with the
  controls first fixed at the declared control targets, and transforms
  the input into the controller, `drto.infinite_horizon` first when the
  option is given, then `drto.dynamic_optimization`; both options pass
  through to their transforms as given.
- `initial_condition` writes the given state values into the
  initial-condition Params before the first step; omitted, the Params'
  current values are the first actual state.
- Each step solves the controller at the current initial condition,
  implements the first moves on the process clone, fixes its
  disturbances at the step's realization, simulates, and feeds the state
  one sample in back as both models' initial condition.
- The first solve is cold-started by default and `cold_start=False` skips
  it; every later solve is warm-started.
- A disturbance entry that is a sequence is used as given; a number draws
  independent zero-mean realizations with that standard deviation,
  reproducibly under `seed`; a missing entry is zero.
- `solve` is applied to every controller and process solve, defaulting to
  pounce; a failed solve raises an error naming the step.
- The history holds times, actual states, implemented moves, and
  realizations under their declared names, and `drto.plot_states` and
  `drto.plot_controls` accept it, the moves drawn as a staircase.
- On hicks with zero disturbances, the actual states settle to the
  declared targets.
