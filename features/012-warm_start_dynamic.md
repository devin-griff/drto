# drto.warm_start_dynamic

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that shifts the previous horizon solution
forward one step to seed the next solve, so that each receding-horizon iteration
starts warm instead of cold.

## Benefit hypothesis

The receding-horizon warm start, shift the last solution forward and fill the
tail, is what makes each NMPC iteration converge in a few steps instead of from
scratch. It is the loop's per-step initializer, the counterpart to
`cold_start_dynamic` for the first solve.

## Acceptance criteria

- `drto.warm_start_dynamic(m)` shifts the model's current horizon solution
  forward one time step and fills the final point (repeat the last point or
  re-simulate the tail), populating the variable values as the next iteration's
  guess.
- It shifts every declared trajectory (`ref[t] <- ref[t+1]`) using the
  `pyomo.dae` flatten traversal, over the declared time set read from `drto.info`.
- It populates variable values (the initial guess) and does not change the model
  structure. It is a plain function, with no `apply_to` or `create_using` form.
