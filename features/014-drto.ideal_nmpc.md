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

# m: the dynamic optimization model
# plant: the same model, one sample long, in dynamic simulation mode

history = drto.ideal_nmpc(m, plant, steps=50,
                          disturbances={"w": 0.05}, seed=0)

drto.plot_states(history)
drto.plot_controls(history)
```

Ideal means the solve is treated as instantaneous: the measurement
arrives, the problem is solved, and the move is implemented at the same
instant. The loop, each step:

1. Initialize: `drto.cold_start_dynamic` on the first step (on by
   default, `cold_start=False` skips it), `drto.warm_start_dynamic` on
   every later one.
2. Solve the dynamic optimization at the current initial condition.
3. Implement: read each control's first move and write it into the
   plant's fixed controls.
4. Realize: fix the plant's disturbances at this step's values.
5. Simulate the plant one sample from the current actual state; its end
   state is the new actual state, written into both models' initial
   conditions.
6. Record the time, the actual state, the implemented moves, and the
   realization.

The plant is the same declared model with a one-sample horizon,
transformed by `drto.dynamic_simulation`; the loop checks the mode and
the horizon length and errors descriptively otherwise. The disturbance
enters the plant only: the controller solves at zero disturbance, the
optimization mode's convention.

Each declared disturbance's entry is either a sequence, the realization
per step as given, or a number, the standard deviation of independent
zero-mean draws each step, reproducible through `seed`. A disturbance
with no entry is zero.

`solve` is a callable applied to the controller and the plant, defaulting
to a plain pounce solve, so a model that needs its own solve wrapper (the
scaled solve of the IDAES example) passes it in. A solve that fails stops
the loop with an error naming the step.

The returned history holds the actual trajectory: the times, each
declared state member's actual values, the implemented moves, and the
realizations, with the declared names and targets. `drto.plot_states` and
`drto.plot_controls` accept it and draw the actual closed-loop
trajectories the same way they draw a model's, the implemented moves as
the staircase they physically are.

## Benefit hypothesis

The closed loop is what the framework exists to run, and it is where the
pieces compose: the cold start seeds the first solve, the warm start
carries each solution to the next, the tail supplies the horizon end, and
the simulation mode is the plant. One function runs the loop and returns
the actual behavior, which is the object of study, on any declared model.

## Acceptance criteria

- `drto.ideal_nmpc(m, plant, steps, ...)` requires `m` transformed by
  `drto.dynamic_optimization` and `plant` by `drto.dynamic_simulation`
  over one sample, and errors descriptively otherwise.
- Each step solves the controller at the current initial condition,
  implements the first moves on the plant, fixes the plant's disturbances
  at the step's realization, simulates one sample, and feeds the plant's
  end state back as both models' initial condition.
- The first solve is cold-started by default and `cold_start=False` skips
  it; every later solve is warm-started.
- A disturbance entry that is a sequence is used as given; a number draws
  independent zero-mean realizations with that standard deviation,
  reproducibly under `seed`; a missing entry is zero.
- `solve` is applied to every controller and plant solve, defaulting to
  pounce; a failed solve raises an error naming the step.
- The history holds times, actual states, implemented moves, and
  realizations under their declared names, and `drto.plot_states` and
  `drto.plot_controls` accept it, the moves drawn as a staircase.
- On hicks with zero disturbances, the actual states settle to the
  declared targets.
