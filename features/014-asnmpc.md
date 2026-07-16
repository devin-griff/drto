# drto.asnmpc

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that runs the advanced-step NMPC loop, so
that the expensive horizon solve happens between samples at a predicted state and
each measurement is handled by a fast correction instead of an online solve.

## Benefit hypothesis

Advanced-step NMPC removes the online computational delay: solve at a predicted
state ahead of time, then correct to the real measurement instantly with a pounce
sensitivity update. This is DRTO's differentiator and the reason the pounce
dependency exists.

## Acceptance criteria

- `drto.asnmpc(m, ...)` runs the advanced-step NMPC loop on a model carrying the
  dynamic-optimization declarations. Each cycle it solves
  `drto.dynamic_optimization` at a predicted next state, and when the measurement
  arrives it corrects the control to the actual state with
  `drto.advanced_step_controller`, applies it, and warm-starts the next cycle with
  `drto.warm_start_dynamic`.
- The expensive solve is off the critical path, at the predicted state between
  samples. The online step is the fast sensitivity correction, not a re-solve.
- It reads the states, controls, and initial-condition parameter from `drto.info`,
  and runs the same shared receding-horizon loop skeleton as `drto.nmpc`.
- It takes a source of the measured state each cycle and a way to predict the
  next state (for example a forward simulation of the model).
