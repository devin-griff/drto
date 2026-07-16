# drto.nmpc

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that runs the NMPC moving-horizon loop on my
declared model, so that I get closed-loop control without wiring the measure,
solve, apply, shift cycle by hand.

## Benefit hypothesis

NMPC is the same receding-horizon cycle every time: feed the measured state,
solve the dynamic optimization, apply the first control, shift and repeat.
Providing it as one call on the declared model is the ideal-timing baseline the
advanced-step variant is measured against.

## Acceptance criteria

- `drto.nmpc(m, ...)` runs the ideal NMPC loop on a model carrying the
  dynamic-optimization declarations. Each cycle it writes the measured state into
  the initial-condition parameter, solves via `drto.dynamic_optimization`,
  applies the first control move, and warm-starts the next cycle with
  `drto.warm_start_dynamic`.
- The solve happens after the measurement arrives (ideal timing, no solve delay
  modeled).
- It reads the states, controls, and initial-condition parameter from
  `drto.info`.
- It runs a shared receding-horizon loop skeleton (the same one `drto.asnmpc`
  uses), differing only in the per-cycle control computation.
- It takes a source of the measured state each cycle: a plant model to simulate
  forward under the applied control, or externally supplied measurements.
