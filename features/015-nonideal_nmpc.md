# drto.nonideal_nmpc

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that runs the NMPC loop with the
computational solve delay modeled honestly, so that I can see the closed-loop
cost of the delay that ideal NMPC ignores and advanced-step removes.

## Benefit hypothesis

Nonideal NMPC is the honest middle of the three execution variants. It applies
each control one solve-time late, under the previous control held while the NLP
runs, so the closed loop pays for the computation the way a real controller
does. It is the baseline that shows what advanced-step buys: without it the
ideal-versus-advanced-step comparison shows the sensitivity correction is
accurate but not the performance gap it recovers.

## Acceptance criteria

- `drto.nonideal_nmpc(m, ...)` runs the NMPC loop on a model carrying the
  dynamic-optimization declarations, the same cycle as `drto.ideal_nmpc` except
  the control computed from the current measurement is applied only after the
  solve finishes, not at the measurement instant.
- Each cycle it measures the state and solves via `drto.dynamic_optimization`.
  Before the new control is applied, the plant advances over the solve delay
  under the previously applied control. The new control is then applied and the
  plant runs the rest of the sample under it. The applied control is therefore
  computed for a state the plant has already left.
- The solve delay is either the measured wall-clock solve time (a live demo) or
  a supplied model of it (a reproducible figure).
- The plant advance is the same plant simulation the loop uses for measurement,
  a `drto.dynamic_simulation` on its own instance that advances by an arbitrary
  interval through a time-scaling parameter over a fixed mesh, with the controls
  fixed to the held input and the initial condition at the current plant state.
  It does not overwrite the controller's horizon solution.
- It reads the states, controls, and initial-condition parameter from
  `drto.info`, and runs the same shared receding-horizon loop skeleton as
  `drto.ideal_nmpc` and `drto.asnmpc`, differing only in when the computed
  control is applied.
- It takes a source of the measured state each cycle (a plant model to simulate
  forward, or externally supplied measurements) and the solve-delay source.
