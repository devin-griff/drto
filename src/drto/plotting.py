# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Registry-aware plotting: ``drto.plot_states``, ``drto.plot_controls``,
``drto.plot_stage_cost`` (feature 022).

The functions read everything from the model's registry (``drto.info``): the
declared horizon and its sample grid, the declared states or controls, and
the paired steady-state targets for the dotted setpoint lines. If the model
carries an infinite-horizon terminal segment (``drto_ih``), the
tail is found automatically, mapped back to real time through
``t = tN + atanh(tau)/gamma``, and drawn with open markers.
``plot_states(..., element_boundaries=True)`` adds squares at the segment's
element boundaries, where the value is the element polynomial extended to
its edge rather than a point the solver placed on the trajectory.

Each selected quantity gets its own fixed-size panel in a two-column grid.
Selection takes component names, components, or member strings like
``"x1[41,1]"``. With no selection every declared component is drawn, a
multi-index component expanding to one panel per member up to a panel cap;
past the cap, members must be selected by name. States and the stage cost
draw as points, finite horizon filled and tail open; controls draw as a
staircase on the finite horizon, each move held over its sampling interval.
Everything clips to ``t_max``, and the functions return the list of panel
axes for further tweaking.

matplotlib is optional to drto: it is imported here through
``attempt_import``, so the package imports cleanly without it and a plot
call raises with the ``pip install drto[plot]`` instruction.
"""
import itertools
import math
import re

import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import

import drto

_MPL_MESSAGE = (
    "drto plotting draws with matplotlib, which is not installed; "
    "install it with: pip install drto[plot]"
)
plt, _plt_available = attempt_import("matplotlib.pyplot", error_message=_MPL_MESSAGE)
_mlines, _ = attempt_import("matplotlib.lines", error_message=_MPL_MESSAGE)

#: Fixed panel size (inches): every plot the same size regardless of count.
_PANEL = (5.0, 3.2)

#: With no selection, a multi-index component expands to one panel per
#: member up to this many total panels; past it, select members by name.
_MAX_PANELS = 12

#: Inches reserved above the panels for the figure legend, so the gap
#: under it is the same whatever the row count.
_LEGEND_BAND = 0.55

_MEMBER = re.compile(r"^\s*(\w+)\s*\[([^\]]+)\]\s*$")


def _tail(m):
    """Return (segment block, tN, gamma, copies) or None if no segment.

    ``copies`` maps ``id(declared component)`` to its segment copy, from
    the pairing the transform records on the registry (gh #27). A control
    whose profile has been applied is keyed under both the component as
    declared and the replacement cvp put in its place (gh #70).
    """
    b = m.component("drto_ih")
    if b is None:
        return None
    reg = drto.info(m)
    time = reg.components("horizon")[0]
    copies = {}
    for r in reg._segment_records():
        if r.get("copy") is None:
            continue
        copies[id(r["of"])] = r["copy"]
        if r.get("live") is not None:
            copies[id(r["live"])] = r["copy"]
    return b, time.last(), pyo.value(b.gamma), copies


def _time_pos(comp, time):
    """Position of the declared time set in ``comp``'s index, or None."""
    subs = list(comp.index_set().subsets())
    for n, s in enumerate(subs):
        if s is time:
            return n, len(subs)
    return None, len(subs)


def _at(comp, other, t, pos):
    """The member of ``comp`` at other-coordinates ``other`` and time ``t``."""
    if not other:
        return comp[t]
    other = tuple(other)
    return comp[other[:pos] + (t,) + other[pos:]]


def _coerce(token):
    """An index token from a member string: int if possible, else float/str."""
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


def _select(declared, selection, what, time):
    """Resolve a selection into (component, other-index, label) panels.

    With no selection every declared component is drawn, a multi-index
    component expanding to one panel per member, up to ``_MAX_PANELS``
    panels in total; past the cap, members must be selected by name."""
    by_name = {c.local_name: c for c in declared}
    if selection is None:
        panels = []
        for c in declared:
            pos, nsub = _time_pos(c, time)
            if nsub == 1:
                panels.append((c, (), c.local_name))
                continue
            others = [s for n, s in enumerate(c.index_set().subsets()) if n != pos]
            for raw in itertools.product(*others):
                o = tuple(x for i in raw for x in (i if isinstance(i, tuple) else (i,)))
                label = f"{c.local_name}[{','.join(str(i) for i in o)}]"
                panels.append((c, o, label))
        if len(panels) > _MAX_PANELS:
            multi = [c.local_name for c in declared if _time_pos(c, time)[1] > 1]
            raise ValueError(
                f"drawing every {what} needs {len(panels)} panels; select "
                f"members of the multi-index {what}s ({', '.join(multi)}) "
                f"by name, like '{multi[0]}[1,1]'."
            )
        return panels
    panels = []
    for item in selection:
        if isinstance(item, str):
            match = _MEMBER.match(item)
            if match:
                name, idx = match.group(1), match.group(2)
                comp = by_name.get(name)
                if comp is None:
                    raise ValueError(
                        f"'{name}' is not a declared {what}; declared: "
                        f"{', '.join(by_name)}."
                    )
                panels.append(
                    (comp, tuple(_coerce(x) for x in idx.split(",")), item.strip())
                )
                continue
            comp = by_name.get(item.strip())
            if comp is None:
                raise ValueError(
                    f"'{item}' is not a declared {what}; declared: "
                    f"{', '.join(by_name)}."
                )
            item = comp
        if item not in declared and not any(c is item for c in declared):
            raise ValueError(f"'{item}' is not a declared {what}.")
        pos, nsub = _time_pos(item, time)
        if nsub > 1:
            raise ValueError(
                f"'{item.local_name}' carries index sets besides time; "
                f"select members like '{item.local_name}[1,1]'."
            )
        panels.append((item, (), item.local_name))
    return panels


def _targets(reg, kind):
    """Map each declared owner component to its paired target Param."""
    return {id(rec["of"]): rec["component"] for rec in reg.declarations(kind)}


def _history_keys(recorded, selection, what):
    """Resolve a history selection into recorded labels."""
    if selection is None:
        return list(recorded)
    keys = []
    for item in selection:
        name = item if isinstance(item, str) else getattr(item, "local_name", item)
        if name not in recorded:
            raise ValueError(
                f"'{name}' is not a recorded {what} of this history; "
                f"recorded: {', '.join(recorded)}."
            )
        keys.append(name)
    return keys


def _bound_lines(ax, lo, hi):
    """Draw the bound lines, the window pinned to the data.

    The limits are captured before the lines land and restored after,
    so a distant bound sits outside the window instead of stretching
    it. Call after the panel's data is drawn.
    """
    levels = [b for b in (lo, hi) if b is not None]
    if not levels:
        return False
    lim = ax.get_ylim()
    for b in levels:
        ax.axhline(b, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylim(lim)
    return True


def _series_names(history):
    """The legend's names for the drawn series.

    A report from ``drto.approximate_nmpc_closed_loop`` carries the
    solver's controls beside the policy's, so its two series are the
    fitted policy and the horizon solves it is compared against. Every
    other history is the solver's own loop.
    """
    if hasattr(history, "solver_moves"):
        return "Approximate NMPC", "NMPC comparison", "no NMPC solution"
    return "actual", "solver", "no solver solution"


def _draw_history(
    history,
    keys,
    series,
    targets,
    t_max,
    staircase,
    second=None,
    bounds=None,
    failures=None,
):
    """Draw a history's recorded trajectories, one fixed-size panel each.

    The actual values draw the way a model's draw: states as filled
    points at the sample instants, moves as the staircase they
    physically are, each held over its sample. Setpoint lines come from
    the recorded targets. ``second`` lays a second recorded series on
    the same panels, dashed (the solver's controls at the visited
    states, when a closed-loop report carries them). ``failures`` are
    the times that second series has no value at, marked with a red x
    on the actual trajectory.
    """
    rows = max(1, math.ceil(len(keys) / 2))
    fig, axes = plt.subplots(
        rows, 2, figsize=(2 * _PANEL[0], rows * _PANEL[1]), sharex=True, squeeze=False
    )
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        # full tick values, never matplotlib's offset notation
        ax.ticklabel_format(useOffset=False)
    for ax in flat[len(keys) :]:
        ax.axis("off")
    drew_target = drew_second = drew_bound = drew_failure = False
    times = history.times
    for ax, key in zip(flat, keys):
        vals = series[key]
        if staircase:
            # a move holds over its sample: the last one extends to the
            # final recorded instant
            pts = [(t, v) for t, v in zip(times, vals + [vals[-1]]) if t <= t_max]
            ax.step(*zip(*pts), where="post", color="C0")
        else:
            pts = [(t, v) for t, v in zip(times, vals) if t <= t_max]
            ax.plot(*zip(*pts), "o", color="C0")
        if second and second.get(key):
            v2 = second[key]
            pts = [(t, v) for t, v in zip(times, v2 + [v2[-1]]) if t <= t_max]
            ax.step(*zip(*pts), where="post", color="C1", linestyle="--")
            drew_second = True
            at = dict(zip(times, vals))
            marks = [(t, at[t]) for t in (failures or ()) if t in at and t <= t_max]
            if marks:
                ax.plot(*zip(*marks), "x", color="red", markersize=5, linestyle="")
                drew_failure = True
        target = targets.get(key)
        if target is not None:
            ax.axhline(target, color="C0", linestyle=":")
            drew_target = True
        pair = (bounds or {}).get(key) or (None, None)
        drew_bound = _bound_lines(ax, *pair) or drew_bound
        ax.set_title(key)
    for ax in flat[max(0, len(keys) - 2) : len(keys)]:
        ax.set_xlabel("time")
    handles = [
        (
            _mlines.Line2D([], [], color="C0", drawstyle="steps-post")
            if staircase
            else _mlines.Line2D([], [], marker="o", color="C0", linestyle="")
        )
    ]
    drawn, compared, unsolved = _series_names(history)
    labels = [drawn]
    if drew_second:
        handles.append(
            _mlines.Line2D([], [], color="C1", linestyle="--", drawstyle="steps-post")
        )
        labels.append(compared)
    if drew_failure:
        handles.append(_mlines.Line2D([], [], marker="x", color="red", linestyle=""))
        labels.append(unsolved)
    if drew_target:
        handles.append(_mlines.Line2D([], [], color="C0", linestyle=":"))
        labels.append("setpoint")
    if drew_bound:
        handles.append(
            _mlines.Line2D([], [], color="grey", linewidth=0.8, linestyle="--")
        )
        labels.append("bound")
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - _LEGEND_BAND / fig.get_figheight()))
    return flat[: len(keys)]


def _tail_points(b, tN, gamma, comp, other, taus, t_max):
    """Map a segment member's points to real time, split at element boundaries.

    Returns (interior, boundary) lists of (t, value) pairs, tau = 1 excluded
    (it maps to t = infinity; its value is the equilibrium endpoint).
    """
    fe = set(b.tau.get_finite_elements())
    interior, boundary = [], []
    for s in taus:
        if not s < 1:
            continue
        t = tN + math.atanh(s) / gamma
        if t > t_max:
            continue
        member = comp[tuple(other) + (s,)] if other else comp[s]
        (boundary if s in fe else interior).append((t, pyo.value(member)))
    return interior, boundary


def _draw(m, panels, targets, sample_slice, t_max, boundary_squares, staircase=False):
    reg = drto.info(m)
    time = reg.components("horizon")[0]
    all_samples = reg.declarations("horizon")[0]["samples"]
    samples = all_samples[sample_slice]
    tail = _tail(m)
    rows = max(1, math.ceil(len(panels) / 2))
    fig, axes = plt.subplots(
        rows, 2, figsize=(2 * _PANEL[0], rows * _PANEL[1]), sharex=True, squeeze=False
    )
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        # full tick values, never matplotlib's offset notation
        ax.ticklabel_format(useOffset=False)
    for ax in flat[len(panels) :]:
        ax.axis("off")  # keep the empty slot so every panel stays the same size
    drew_tail = drew_boundary = drew_target = drew_bound = False
    for ax, (comp, other, label) in zip(flat, panels):
        pos, _ = _time_pos(comp, time)
        values = [pyo.value(_at(comp, other, t, pos)) for t in samples]
        if staircase:
            # a move holds over its sampling interval: the last one extends
            # to the end of the horizon
            ax.step(
                list(samples) + [all_samples[-1]],
                values + [values[-1]],
                where="post",
                color="C0",
            )
        else:
            ax.plot(samples, values, "o", color="C0")
        target = targets.get(id(comp))
        if target is not None:
            tval = target[tuple(other)] if other else target
            ax.axhline(pyo.value(tval), color="C0", linestyle=":")
            drew_target = True
        if tail is not None:
            b, tN, gamma, copies = tail
            seg = copies.get(id(comp))
            if seg is not None:
                # a member panel iterates the tau grid; a time-only panel
                # iterates the copy's own index set (a parameterized segment
                # control keeps free values at a subset of points)
                taus = sorted(b.tau) if other else sorted(seg.index_set())
                interior, boundary = _tail_points(b, tN, gamma, seg, other, taus, t_max)
                if interior:
                    ax.plot(*zip(*interior), "o", mfc="none", color="C0")
                    drew_tail = True
                if boundary and boundary_squares:
                    ax.plot(*zip(*boundary), "s", mfc="none", color="C0")
                    drew_boundary = True
            ax.axvline(tN, color="grey", linewidth=0.8)
        member = _at(comp, other, samples[0], pos)
        drew_bound = (
            _bound_lines(ax, getattr(member, "lb", None), getattr(member, "ub", None))
            or drew_bound
        )
        ax.set_title(label)
    for ax in flat[max(0, len(panels) - 2) : len(panels)]:
        ax.set_xlabel("time")
    handles = [
        (
            _mlines.Line2D([], [], color="C0", drawstyle="steps-post")
            if staircase
            else _mlines.Line2D([], [], marker="o", color="C0", linestyle="")
        )
    ]
    labels = ["finite horizon"]
    if drew_tail:
        handles.append(
            _mlines.Line2D([], [], marker="o", mfc="none", color="C0", linestyle="")
        )
        labels.append("tail")
    if drew_boundary:
        handles.append(
            _mlines.Line2D([], [], marker="s", mfc="none", color="C0", linestyle="")
        )
        labels.append("element boundary")
    if drew_target:
        handles.append(_mlines.Line2D([], [], color="C0", linestyle=":"))
        labels.append("setpoint")
    if drew_bound:
        handles.append(
            _mlines.Line2D([], [], color="grey", linewidth=0.8, linestyle="--")
        )
        labels.append("bound")
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - _LEGEND_BAND / fig.get_figheight()))
    return flat[: len(panels)]


def plot_states(m, states=None, t_max=50, element_boundaries=False):
    """Plot declared states, one fixed-size panel each, two columns.

    ``states`` selects by name, component, or member string like
    ``"x1[41,1]"``. With no selection every declared state is drawn, a
    multi-index state expanding to one panel per member, up to a cap of
    ``_MAX_PANELS`` panels; past it, select members by name.
    Setpoint lines come from the ``steady_state`` pairings.

    ``element_boundaries=True`` adds squares at the terminal segment's
    element boundaries. The segment collocates on Gauss-Legendre points,
    which lie strictly inside each element, so a boundary value is the
    element polynomial extended to its edge and tied to the next element
    by a continuity equation, not a point the solver placed on the
    trajectory. A boundary far from its neighbors says the mesh is too
    coarse there, which is what the squares are for. They stay off by
    default: on a converged trajectory they are the only points away
    from the curve, so they set the axis limits and a settled state
    reads as if it were oscillating. Returns the panel axes.

    Handed an :class:`drto.NmpcHistory` instead of a model, draws its
    actual state trajectories at the recorded sample instants, the
    setpoints from the recorded targets; ``states`` then selects the
    recorded labels.
    """
    from drto.ideal_nmpc import NmpcHistory

    if isinstance(m, NmpcHistory):
        keys = _history_keys(m.states, states, "state")
        return _draw_history(
            m,
            keys,
            m.states,
            m.state_targets,
            t_max,
            False,
            bounds=getattr(m, "state_bounds", None),
        )
    reg = drto.info(m)
    time = reg.components("horizon")[0]
    panels = _select(reg.components("state"), states, "state", time)
    return _draw(
        m,
        panels,
        _targets(reg, "steady_state"),
        slice(None),
        t_max,
        boundary_squares=element_boundaries,
    )


def plot_stage_cost(m, t_max=50):
    """Plot the tracking stage cost, one fixed-size panel.

    The cost variable is read off the declared stage-cost equality
    (whichever side of it is the scalar). Finite values sit at the samples
    minus the final time, where only the terminal cost applies. On the
    tail the replicated stage-cost Expressions carry the values, drawn
    open at the interior collocation points. A dotted line marks zero,
    the tracking cost's settling value. Returns the panel axes.
    """
    reg = drto.info(m)
    cons = reg.components("tracking_stage_cost")
    if not cons:
        raise ValueError("no tracking stage cost is declared on this model.")
    member = next(iter(cons[0].values()))
    cost = None
    for side in (member.expr.args[0], member.expr.args[1]):
        if getattr(side, "is_variable_type", lambda: False)():
            cost = side.parent_component()
            break
    if cost is None:
        raise ValueError("no scalar cost variable side on the stage cost.")
    panels = [(cost, (), cost.local_name)]
    return _draw(
        m, panels, {id(cost): 0}, slice(None, -1), t_max, boundary_squares=False
    )


def plot_controls(m, controls=None, t_max=50):
    """Plot declared controls, one fixed-size panel each, two columns.

    ``controls`` selects by name or component (all by default; controls are
    time-only). The finite horizon draws as a staircase: each move holds
    over its sampling interval, the last one to the end of the horizon (the
    final sample belongs to the terminal cost, so no move starts there).
    Setpoint lines come from the ``steady_state_control`` pairings.
    Segment controls have no boundary values, so no squares. Returns the
    panel axes.

    Handed an :class:`drto.NmpcHistory` instead of a model, draws the
    implemented moves as the staircase they physically are, each held
    over its sample; ``controls`` then selects the recorded labels. A
    closed-loop report carrying the solver's controls draws them on the
    same panels, dashed.
    """
    from drto.ideal_nmpc import NmpcHistory

    if isinstance(m, NmpcHistory):
        keys = _history_keys(m.moves, controls, "control")
        return _draw_history(
            m,
            keys,
            m.moves,
            m.control_targets,
            t_max,
            True,
            second=getattr(m, "solver_moves", None),
            failures=getattr(m, "solver_failures", None),
            bounds=getattr(m, "control_bounds", None),
        )
    reg = drto.info(m)
    time = reg.components("horizon")[0]
    panels = _select(reg.components("control"), controls, "control", time)
    return _draw(
        m,
        panels,
        _targets(reg, "steady_state_control"),
        slice(None, -1),
        t_max,
        boundary_squares=False,
        staircase=True,
    )


def plot_history(policy):
    """Plot a fitted policy's training and validation losses, one panel.

    Both curves come from ``policy.history``, on a log scale against the
    epoch each checkpoint was taken at. Returns the panel axes.
    """
    fig, ax = plt.subplots(figsize=_PANEL)
    h = policy.history
    ax.semilogy(h["epoch"], h["train_loss"], color="C0", label="training loss")
    ax.semilogy(h["epoch"], h["val_loss"], color="C1", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    return [ax]


def _r_squared(y, y_hat):
    """One minus the residual sum of squares over the total."""
    mean = sum(y) / len(y)
    total = sum((v - mean) ** 2 for v in y)
    residual = sum((v - w) ** 2 for v, w in zip(y, y_hat))
    return 1.0 - residual / total if total else float("nan")


def plot_parity(policy, data, validation=None):
    """Plot a fitted policy's actions against the solver's, one panel each.

    Each panel draws one control, the label on the horizontal axis and
    the policy's action on the vertical, with the line where the two
    agree. Training points draw as circles and validation points as
    triangles, and each series carries its coefficient of determination
    in the legend. The split comes from ``policy.validation_index``, the
    points the training held out, unless a ``validation`` dataset is
    given, in which case its points are the second series. Without
    either, one series draws. Returns the panel axes.
    """
    points = list(getattr(data, "points", data))
    held = policy.meta.get("validation_index")
    if validation is not None:
        series = [
            (points, "o", "training"),
            (list(getattr(validation, "points", validation)), "^", "validation"),
        ]
    elif held:
        held = set(held)
        series = [
            ([p for i, p in enumerate(points) if i not in held], "o", "training"),
            ([p for i, p in enumerate(points) if i in held], "^", "validation"),
        ]
    else:
        series = [(points, "o", "sampled")]

    names = list(policy.meta["inputs"])
    controls = list(policy.meta["u_bounds"])
    drawn = [
        (
            marker,
            which,
            [[p["u0"][c] for c in controls] for p in group],
            [
                [policy({k: p["x"][k] for k in names})[c] for c in controls]
                for p in group
            ],
        )
        for group, marker, which in series
        if group
    ]
    fig, axes = plt.subplots(
        1,
        len(controls),
        figsize=(len(controls) * _PANEL[0], _PANEL[1] + 0.6),
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]
    for j, (ax, name) in enumerate(zip(flat, controls)):
        ax.ticklabel_format(useOffset=False)
        low = high = None
        for marker, which, label, pred in drawn:
            y = [row[j] for row in label]
            y_hat = [row[j] for row in pred]
            ax.scatter(
                y,
                y_hat,
                s=16,
                marker=marker,
                alpha=0.6,
                label=f"{which}, $R^2$ = {_r_squared(y, y_hat):.5f}",
            )
            low = min(y) if low is None else min(low, min(y))
            high = max(y) if high is None else max(high, max(y))
        if low is not None:
            ax.plot([low, high], [low, high], color="0.4", linewidth=0.8, zorder=0)
        ax.set_title(name)
        ax.set_xlabel("solver")
        ax.set_ylabel("policy")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return flat
