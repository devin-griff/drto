# drto.plotting

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want the registry-aware plotting that lives in
`examples/plotting.py` available from the package itself, so that any
declared model can be inspected visually, with the setpoint lines and the
terminal segment drawn correctly, without copying a helper out of the
examples tree.

The helper is registry-driven already: everything it draws comes from
`drto.info`. The feature is the move, as `drto.plot_states`,
`drto.plot_controls`, and `drto.plot_stage_cost`:

- Everything reads from the registry: the declared horizon's sample grid,
  the declared states and controls, the steady-state pairings for the
  dotted setpoint lines, and the stage-cost equality for the cost variable.
  If the model carries a terminal segment (`drto_ih`), its points draw
  open, mapped back to real time through `t = tN + atanh(tau)/gamma`, with
  squares marking the element boundaries on state panels.
- Selection takes component names, components, or member strings like
  `"x1[41,1]"`. With no selection every declared component is drawn, a
  multi-index component expanding to one panel per member, up to a cap;
  past the cap, a descriptive error names the multi-index components and
  the member syntax.
- Controls draw as a staircase on the finite horizon: each move holds
  over its sampling interval, the last one to the end of the horizon. The
  tail draws as open points at the free values in every profile.
- matplotlib is optional to drto: it goes through
  `pyomo.common.dependencies.attempt_import` and a `plot` extra, so the
  package imports cleanly without it and a plot call without it raises
  with the install instruction.
- `examples/plotting.py` leaves the tree: the notebooks import from the
  package, and one copy exists.

## Benefit hypothesis

The plots are how every example notebook reads its results, and they
encode drto-specific knowledge that a user should not have to rediscover:
where the tail lives, how tau maps to time, which pairings carry the
setpoints, and which side of the stage-cost equality is the cost. Keeping
that in an examples-only helper makes it invisible to installed users and
copies it into every downstream project by hand. In the package, a user
plots a transformed model in one line, and the examples stop carrying
infrastructure.

## Acceptance criteria

- `drto.plot_states`, `drto.plot_controls`, and `drto.plot_stage_cost`
  import from `drto` and return the panel axes on a declared model,
  transformed or not, reading only the registry and the `drto_ih` block.
- With no selection every declared component draws, a multi-index
  component expanding to one panel per member; past the panel cap the
  error names the multi-index components and the member-string syntax.
- Controls draw as a staircase on the finite horizon, the last move held
  to the end of the horizon; states and the stage cost draw as points.
- With a terminal segment present, the tail points draw at
  `t = tN + atanh(tau)/gamma`, open markers, element-boundary squares on
  state panels, clipped to `t_max`.
- Setpoint lines come from the `steady_state` and `steady_state_control`
  pairings, per member.
- `import drto` succeeds without matplotlib; calling a plot function
  without it raises naming the `plot` extra.
- Tests pin the panel counts and labels on a synthetic model under a
  non-interactive backend, and the cap rejection message.
- The example notebooks import from the package, `examples/plotting.py` is
  gone, and the rendered results are unchanged.
