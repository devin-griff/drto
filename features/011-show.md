# drto.show

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a readable, DRTO-aware view of my model, so that I can
see the outcome of a transformation, the states, controls, dynamics, costs, and
objective organized by meaning, instead of reading Pyomo's raw `pprint` dump.

## Benefit hypothesis

`pprint` is role-blind and expands every index, so it is unreadable for a
horizon model. DRTO already knows the role of every component from the
declarations in `drto.info`, so it can render the model grouped by meaning and
compact over the horizon, and annotate what each applied transform did. This is
the inspection tool that makes the transforms legible, and a differentiator over
generic Pyomo output, since DRTO is the only thing that knows the roles.

## Acceptance criteria

- `drto.show(m)` renders a readable, DRTO-aware summary of the model, reading the
  roles and the transformation log from `drto.info` (feature 001).
- It groups components by role: the horizon (or single point), states, controls
  (marked free or fixed), continuous or discrete dynamics, stage and terminal
  costs, boundary conditions, and steady-state targets, one labeled line each,
  using the index set rather than expanding every time point.
- It renders the objective in compact symbolic form (the summed stage cost over
  its range plus any terminal term), not the fully-written-out sum.
- It annotates the applied transformations from the info log, showing each one's
  outcome: what it freed or fixed, the terms it dropped, the objective it
  assembled, and whether it kept or collapsed the horizon.
- It prints readable text to the console and provides a `_repr_html_` so it
  renders as a panel in a Jupyter notebook.
- It does not modify the model; it only reads it.
