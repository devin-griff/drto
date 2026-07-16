# drto.build_objective

**Status:** ready

## Description

As a user of DRTO, I want every mode to assemble my objective the same way from
the cost terms I declare, so that the objective is correct and identical across
modes and I never hand-write or maintain it.

## Benefit hypothesis

Owning objective assembly in one shared routine keeps the objective consistent
across every cost-bearing mode, the optimization transforms now and the
estimation transforms later, and removes drift-prone duplication. Each mode
decides which cost terms are live, and this routine sums them the same way every
time.

## Acceptance criteria

- `drto.build_objective(m)` is a function the mode transforms call as their
  final step to assemble the objective from the live cost terms on the model
  (the cost declarations the calling mode has kept).
- It installs exactly one minimize `Objective` summed from those live terms.
- The objective is the plain sum of each live stage cost's per-point cost var
  over its time index, excluding the last time point, plus each live
  terminal-cost var.
- For a time set over 0..N, the stage-cost sum runs over 0..N-1, and a live
  terminal cost applies at N.
- An existing active objective on the model is deactivated before the assembled
  objective is installed.
- If no cost term is live, it raises a clear error, since there is nothing to
  assemble.
