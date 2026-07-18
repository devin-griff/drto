# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""drto-aware plotting for the example notebooks.

The functions read everything from the model's registry (``drto.info``): the
declared horizon and its sample grid, the declared states or controls, and
the paired steady-state targets for the dotted setpoint lines. If the model
carries an infinite-horizon terminal segment (``drto_infinite_horizon``), the
tail is found automatically, mapped back to real time through
``t = tN + atanh(tau)/gamma``, and drawn with open markers, squares marking
the segment's finite element boundaries (continuity extrapolations).

Each selected quantity gets its own fixed-size panel in a two-column grid.
Selection takes component names or components for time-only Vars, and member
strings like ``"x1[41,1]"`` for states carrying other index sets (a
multi-index Var must be selected by member: one panel per tray-by-component
combination is not a plot). With no selection, all time-only declared
components are drawn and multi-index ones must be named. Points only, finite
horizon filled, tail open, clipped to ``t_max``. The functions return the
list of panel axes for further tweaking.
"""
import math
import re

import matplotlib.pyplot as plt
import pyomo.environ as pyo
from matplotlib.lines import Line2D

import drto

#: Fixed panel size (inches): every plot the same size regardless of count.
_PANEL = (5.0, 3.2)

_MEMBER = re.compile(r"^\s*(\w+)\s*\[([^\]]+)\]\s*$")


def _tail(m):
    """Return (segment block, tN, gamma) or None if no terminal segment."""
    b = m.component("drto_infinite_horizon")
    if b is None:
        return None
    time = drto.info(m).components("horizon")[0]
    return b, time.last(), pyo.value(b.gamma)


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
    """Resolve a selection into (component, other-index, label) panels."""
    by_name = {c.local_name: c for c in declared}
    if selection is None:
        multi = [c.local_name for c in declared if _time_pos(c, time)[1] > 1]
        if multi:
            raise ValueError(
                f"select members of the multi-index {what}s "
                f"({', '.join(multi)}) by name, like "
                f"'{multi[0]}[1,1]'; a panel per member of every index "
                f"combination is not a plot."
            )
        return [(c, (), c.local_name) for c in declared]
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


def _draw(m, panels, targets, sample_slice, t_max, boundary_squares):
    reg = drto.info(m)
    time = reg.components("horizon")[0]
    samples = reg.declarations("horizon")[0]["samples"][sample_slice]
    tail = _tail(m)
    rows = max(1, math.ceil(len(panels) / 2))
    fig, axes = plt.subplots(
        rows, 2, figsize=(2 * _PANEL[0], rows * _PANEL[1]), sharex=True, squeeze=False
    )
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(panels) :]:
        ax.axis("off")  # keep the empty slot so every panel stays the same size
    drew_tail = drew_boundary = drew_target = False
    for ax, (comp, other, label) in zip(flat, panels):
        pos, _ = _time_pos(comp, time)
        ax.plot(samples, [pyo.value(_at(comp, other, t, pos)) for t in samples], "o", color="C0")
        target = targets.get(id(comp))
        if target is not None:
            tval = target[tuple(other)] if other else target
            ax.axhline(pyo.value(tval), color="C0", linestyle=":")
            drew_target = True
        if tail is not None:
            b, tN, gamma = tail
            seg = b.component(comp.local_name)
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
        ax.set_title(label)
    for ax in flat[max(0, len(panels) - 2) : len(panels)]:
        ax.set_xlabel("time")
    handles = [Line2D([], [], marker="o", color="C0", linestyle="")]
    labels = ["finite horizon"]
    if drew_tail:
        handles.append(Line2D([], [], marker="o", mfc="none", color="C0", linestyle=""))
        labels.append("tail")
    if drew_boundary:
        handles.append(Line2D([], [], marker="s", mfc="none", color="C0", linestyle=""))
        labels.append("element boundary")
    if drew_target:
        handles.append(Line2D([], [], color="C0", linestyle=":"))
        labels.append("setpoint")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return flat[: len(panels)]


def plot_states(m, states=None, t_max=50):
    """Plot declared states, one fixed-size panel each, two columns.

    ``states`` selects by name, component, or member string like
    ``"x1[41,1]"`` (required for states with index sets besides time).
    Setpoint lines come from the ``steady_state`` pairings; squares mark the
    segment's element boundaries, where the state value is the continuity
    extrapolation rather than a collocation point. Returns the panel axes.
    """
    reg = drto.info(m)
    time = reg.components("horizon")[0]
    panels = _select(reg.components("state"), states, "state", time)
    return _draw(m, panels, _targets(reg, "steady_state"), slice(None), t_max, boundary_squares=True)


def plot_stage_cost(m, t_max=50):
    """Plot the tracking stage cost, one fixed-size panel.

    The cost variable is read off the declared stage-cost equality
    (whichever side of it is the scalar). Finite values sit at the samples
    minus the final time, where only the terminal cost applies. On the
    tail the replicated stage-cost Expressions carry the values, drawn
    open at the interior collocation points. Returns the panel axes.
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
    return _draw(m, panels, {}, slice(None, -1), t_max, boundary_squares=False)


def plot_controls(m, controls=None, t_max=50):
    """Plot declared controls, one fixed-size panel each, two columns.

    ``controls`` selects by name or component (all by default; controls are
    time-only). The finite points sit at the start of each sampling interval
    (the final sample belongs to the terminal cost, so no move starts
    there). Setpoint lines come from the ``steady_state_control`` pairings.
    Segment controls have no boundary values, so no squares. Returns the
    panel axes.
    """
    reg = drto.info(m)
    time = reg.components("horizon")[0]
    panels = _select(reg.components("control"), controls, "control", time)
    return _draw(m, panels, _targets(reg, "steady_state_control"), slice(None, -1), t_max, boundary_squares=False)
