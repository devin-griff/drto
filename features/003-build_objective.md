# drto.build_objective

**Status:** ready

## Description

As a user of DRTO, I want DRTO to build my model's optimization objective
from the cost terms I declare, so that I declare the pieces, a per-time-point
stage cost and an optional terminal cost, and DRTO assembles the `min`
objective instead of my writing and maintaining it by hand.

## Benefit hypothesis

If DRTO owns objective assembly from the declared cost terms, the objective
stays correct and consistent as the same model is reused across problems and
modes, and users are freed from hand-writing summed objectives and rewriting
them per problem. This establishes the "objective is DRTO's" principle that
later features (the six modes, `dynamic_to_steady_state`) build on, so getting it reusable
and correct now de-risks all of them.

## Acceptance criteria

- Applying `TransformationFactory('drto.build_objective')` to a model with one
  or more declared cost terms installs exactly one minimize `Objective`
  assembled from them.
- The objective is the plain sum of each declared stage cost's per-point cost
  var over its time index, excluding the last time point, plus each declared
  terminal-cost var.
- For a time set initialized over 0..N, the stage-cost sum runs over 0..N-1;
  a declared terminal cost applies at N.
- With no terminal cost declared, the objective is the stage-cost sum alone.
- An existing active objective on the model is deactivated before the
  assembled objective is installed.
- The transform works through both `apply_to` (in place) and `create_using`
  (a transformed clone); `create_using` leaves the source model's objective
  unchanged.
- If no stage cost is declared, the transform raises a clear error, since
  there is nothing to assemble.
