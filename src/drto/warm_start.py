# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Warm start: ``drto.warm_start_dynamic`` (feature 013).

Reuses the previous solution, moved one sampling time forward, as the
initialization of the current problem: every variable takes the value the
previous solution had one sampling time later, copied where the grids
line up and interpolated where they do not, the infinite-horizon tail
included through ``t = tN + atanh(tau)/gamma``. Past the end of the
previous solution, which only exists without a tail, states and controls
take their declared steady-state targets, derivatives zero, and
algebraic variables keep their values. Values only; nothing is solved,
and the initial condition is left to the loop, which sets it from the
measurement.
"""
import math
from bisect import bisect_left
from dataclasses import dataclass

import pyomo.environ as pyo
from pyomo.core import Var
from pyomo.dae import DerivativeVar

from drto.cold_start import _target
from drto.infinite_horizon import _join_index, _split_index, _time_index
from drto.info import info

_EPS = 1e-9


@dataclass
class WarmStartReport:
    """What the shift set: copies, interpolations, and end fills."""

    dt: float = 0.0
    n_copied: int = 0
    n_interpolated: int = 0
    n_filled: int = 0
    tail: str = "(none attached)"
    multipliers: str = "(no suffixes declared)"
    filled_names: list = None

    def __str__(self):
        return (
            "drto warm_start_dynamic (the previous solution, one step on)\n"
            f"  shift         : {self.dt:g} time units\n"
            f"  copied        : {self.n_copied} values on aligned points\n"
            f"  interpolated  : {self.n_interpolated} values between points\n"
            f"  filled        : {self.n_filled} values past the end\n"
            f"  tail          : {self.tail}\n"
            f"  multipliers   : {self.multipliers}"
        )


def _interp(xs, ys, x):
    """Linear interpolation with flat extrapolation at the ends."""
    pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
    if not pts:
        return None
    for a, b in pts:
        if abs(x - a) < 1e-7:
            return b
    if x <= pts[0][0] + _EPS:
        return pts[0][1]
    if x >= pts[-1][0] - _EPS:
        return pts[-1][1]
    hi = bisect_left([a for a, _ in pts], x)
    (x0, y0), (x1, y1) = pts[hi - 1], pts[hi]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _axis(comp, time):
    """(time position, index combos) of a component's own numeric axis."""
    pos, subs = _time_index(comp, time)
    if pos is not None:
        combos = set()
        for idx in comp:
            o, _t = _split_index(idx, pos, len(subs))
            combos.add(o)
        return pos, sorted(combos)
    # a numeric axis of its own (a cvp move set, a stage-cost set): the
    # last index element is the point
    combos = {(idx[:-1] if isinstance(idx, tuple) else ()) for idx in comp}
    return None, sorted(combos)


def _points(comp, pos, o):
    if pos is not None:
        return None  # caller iterates the declared grid
    return sorted(
        (idx if not isinstance(idx, tuple) else idx[-1])
        for idx in comp
        if (idx[:-1] if isinstance(idx, tuple) else ()) == o
    )


def _floor_val(xs, ys, x):
    """The value active at ``x`` of a piecewise-constant profile."""
    pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
    if not pts:
        return None
    val = pts[0][1]
    for a, b in pts:
        if a <= x + 1e-7:
            val = b
        else:
            break
    return val


def _clamped(vd, val):
    lo = vd.lb
    hi = vd.ub
    if lo is not None and val < lo:
        return lo
    if hi is not None and val > hi:
        return hi
    return val


def warm_start_dynamic(m):
    """Shift ``m``'s values one sampling time forward; see the module
    docstring. Returns a :class:`WarmStartReport`; values only, nothing
    added or removed, fixed variables left alone."""
    fn = "warm_start_dynamic"
    reg = info(m)
    for kind in ("horizon", "state"):
        if not reg.has_declaration(kind):
            raise ValueError(f"drto: {fn} requires the {kind} declaration.")
    time = reg.components("horizon")[0]
    if not time.get_discretization_info():
        raise ValueError(f"drto: {fn} shifts across the grid; discretize first.")
    samples = reg.declarations("horizon")[0]["samples"]
    dt = samples[1] - samples[0]
    grid = sorted(time)
    t0, tN = grid[0], grid[-1]
    report = WarmStartReport(dt=dt)
    report.filled_names = []

    # the tail, through the records the transform wrote (gh #27)
    recs = reg._segment_records()
    gamma, taus = None, None
    if recs:
        (seg,) = (r for r in recs if r["kind"] == "segment")
        gamma = pyo.value(seg["gamma"])
        taus = sorted(seg["of"].tau)
        report.tail = "shifted through t = tN + atanh(tau)/gamma"

    def tail_curve(copy, o):
        pts = sorted(
            (ci if not isinstance(ci, tuple) else ci[-1])
            for ci in copy
            if (ci[:-1] if isinstance(ci, tuple) else ()) == o
        )
        ys = []
        for p in pts:
            member = copy[tuple(o) + (p,)] if o else copy[p]
            try:
                ys.append(pyo.value(member))
            except (ValueError, TypeError):
                ys.append(None)
        return pts, ys

    def tail_eval(copy, o, t_abs, step=False, anchor=None):
        tau = 1.0 if t_abs >= tN + 1e12 else math.tanh(gamma * (t_abs - tN))
        pts, ys = tail_curve(copy, o)
        if anchor is not None and (not pts or pts[0] > _EPS):
            pts, ys = [0.0] + pts, [anchor] + ys
        return _floor_val(pts, ys, tau) if step else _interp(pts, ys, tau)

    # tail sources per finite trajectory: id of any member -> (copy, o)
    tail_of, dtail_of = {}, {}
    states = list(reg.components("state"))
    controls = list(reg.components("control"))
    ctrl_ids = {id(u) for u in controls}
    if recs:
        ctrl_recs = [r for r in recs if r["kind"] == "control"]
        for r in recs:
            if r["kind"] not in ("state", "control", "algebraic"):
                continue
            sources = [r["of"]]
            if r["kind"] == "control":
                live = controls[ctrl_recs.index(r)]
                if live is not r["of"]:
                    # cvp replaced the component; the original reference's
                    # underlying members still shift through the same copy
                    sources.append(live)
            copy = r["copy"]
            for comp in sources:
                pos, combos = _axis(comp, time)
                for o in combos:
                    pts = _points(comp, pos, o) or grid
                    for t in pts:
                        vd = (
                            comp[_join_index(o, t, pos)]
                            if pos is not None
                            else (comp[tuple(o) + (t,)] if o else comp[t])
                        )
                        tail_of[id(vd)] = (copy, o)
                        if r["kind"] == "state" and r.get("derivative") is not None:
                            dtail_of[id(vd)] = (r["derivative"], o)
        for r in recs:
            if r["kind"] == "packed_member":
                pcomp, copy = r["of"], r["copy"]
                pos, _ = _time_index(pcomp, time)
                for cidx in copy:
                    o = cidx[:-1] if isinstance(cidx, tuple) else ()
                    for t in grid:
                        tail_of[id(pcomp[_join_index(o, t, pos)])] = (copy, o)
            elif r["kind"] == "block_member":
                B, lname, copy = r["of"], r["member"], r["copy"]
                for cidx in copy:
                    o = cidx[:-1] if isinstance(cidx, tuple) else ()
                    for t in B:
                        c = getattr(B[t], lname)
                        vd = c[o if len(o) > 1 else o[0]] if o else c
                        tail_of[id(vd)] = (copy, o)

    z_cover = set()
    z_target = {}
    for z in states:
        pos, combos = _axis(z, time)
        for o in combos:
            for t in grid:
                vd = z[_join_index(o, t, pos)]
                z_cover.add(id(vd))
                z_target[id(vd)] = (z, o)

    def shift_axis(comp, step=False):
        """Shift one component over its own numeric axis, in place."""
        pos, combos = _axis(comp, time)
        for o in combos:
            pts = _points(comp, pos, o) or grid
            members = [
                (
                    comp[_join_index(o, t, pos)]
                    if pos is not None
                    else (comp[tuple(o) + (t,)] if o else comp[t])
                )
                for t in pts
            ]
            old = [vd.value for vd in members]
            for t, vd in zip(pts, members):
                if vd.fixed:
                    continue
                tp = t + dt
                if tp <= pts[-1] + _EPS:
                    val = _floor_val(pts, old, tp) if step else _interp(pts, old, tp)
                    exact = step or any(abs(tp - x) < 1e-7 for x in pts)
                    report.n_copied += exact
                    report.n_interpolated += not exact
                elif id(vd) in tail_of:
                    copy, oo = tail_of[id(vd)]
                    val = tail_eval(copy, oo, tp, step=step, anchor=old[-1])
                    report.n_copied += step
                    report.n_interpolated += not step
                elif id(vd) in z_cover:
                    z, oo = z_target[id(vd)]
                    p = _target(reg.declarations("steady_state"), z, "steady_state", fn)
                    val = pyo.value(p[oo] if oo else p)
                    report.n_filled += 1
                elif id(vd) in ctrl_cover:
                    u, oo = ctrl_cover[id(vd)]
                    p = _target(
                        reg.declarations("steady_state_control"),
                        u,
                        "steady_state_control",
                        fn,
                    )
                    val = pyo.value(p[oo] if oo else p)
                    report.n_filled += 1
                else:
                    report.n_filled += 1
                    report.filled_names.append(vd.name)
                    continue  # algebra keeps its value past the end
                if val is not None:
                    vd.set_value(_clamped(vd, val))

    ctrl_cover = {}
    for u in controls:
        pos, combos = _axis(u, time)
        for o in combos:
            for t in _points(u, pos, o) or grid:
                vd = (
                    u[_join_index(o, t, pos)]
                    if pos is not None
                    else (u[tuple(o) + (t,)] if o else u[t])
                )
                ctrl_cover[id(vd)] = (u, o)

    # cost vars from the declared cost rows join the shifted set even on
    # a plain numeric index (a stage set)
    from drto.declarations import _is_var_member, _side_matching

    cost_vars = []
    for kind in ("tracking_stage_cost", "economic_stage_cost"):
        for r in reg.declarations(kind):
            con = r["component"]
            cd = next(iter(con.values())) if con.is_indexed() else con
            side, _ = _side_matching(cd, _is_var_member, fn, "the cost variable")
            if id(side.parent_component()) not in {id(c) for c in cost_vars}:
                cost_vars.append(side.parent_component())

    done = set()
    for comp in list(m.component_objects(Var, active=True)):
        if isinstance(comp, DerivativeVar) and comp.get_continuousset_list() == [time]:
            continue
        if id(comp) in ctrl_ids or id(comp) in done:
            continue
        pos, _subs = _time_index(comp, time)
        if pos is None:
            continue
        if recs and comp.parent_block() is next(
            r["of"] for r in recs if r["kind"] == "segment"
        ):
            continue  # the tail shifts in its own pass below
        shift_axis(comp)
        done.add(id(comp))
    for u in controls:
        shift_axis(u, step=True)
    for cv in cost_vars:
        if id(cv) not in done and id(cv) not in ctrl_ids:
            shift_axis(cv)

    # time-indexed Block families (the IDAES property idiom): each
    # member shifts over the family's own time axis, the tail through
    # the recorded copies, algebra keeping its value past an end
    from pyomo.core import Block

    for B in m.component_objects(Block, active=True):
        subs = list(B.index_set().subsets())
        if len(subs) != 1 or subs[0] is not time:
            continue
        bt = sorted(B)
        for v in B[bt[0]].component_objects(Var):
            lname = v.local_name
            combos = (
                [()]
                if not v.is_indexed()
                else [(i if isinstance(i, tuple) else (i,)) for i in v]
            )
            for o in combos:
                members = []
                for t in bt:
                    c = getattr(B[t], lname)
                    members.append(c[o if len(o) > 1 else o[0]] if o else c)
                old_vals = [vd.value for vd in members]
                for t, vd in zip(bt, members):
                    if vd.fixed:
                        continue
                    tp = t + dt
                    if tp <= bt[-1] + _EPS:
                        val = _interp(bt, old_vals, tp)
                        exact = any(abs(tp - x) < _EPS for x in bt)
                        report.n_copied += exact
                        report.n_interpolated += not exact
                    elif id(vd) in tail_of:
                        copy, oo = tail_of[id(vd)]
                        val = tail_eval(copy, oo, tp, anchor=old_vals[-1])
                        report.n_interpolated += 1
                    else:
                        report.n_filled += 1
                        report.filled_names.append(vd.name)
                        continue
                    if val is not None:
                        vd.set_value(_clamped(vd, val))

    # derivatives: the previous slope one step later; the tail supplies
    # dz/dt = gamma*(1 - tau^2)*dz/dtau past the horizon's end
    for dv in m.component_objects(Var, active=True):
        if not (
            isinstance(dv, DerivativeVar) and dv.get_continuousset_list() == [time]
        ):
            continue
        zvar = dv.get_state_var()
        pos, combos = _axis(dv, time)
        for o in combos:
            members = [dv[_join_index(o, t, pos)] for t in grid]
            old = [vd.value for vd in members]
            for t, vd in zip(grid, members):
                if vd.fixed:
                    continue
                tp = t + dt
                if tp <= tN + _EPS:
                    val = _interp(grid, old, tp)
                elif recs:
                    key = id(zvar[_join_index(o, t, pos)])
                    if key not in dtail_of:
                        continue
                    dcopy, oo = dtail_of[key]
                    tau = math.tanh(gamma * (tp - tN))
                    dpts, dys = tail_curve(dcopy, oo)
                    if old[-1] is not None and (not dpts or dpts[0] > _EPS):
                        dpts, dys = [0.0] + dpts, [old[-1] / gamma] + dys
                    dval = _interp(dpts, dys, tau)
                    val = None if dval is None else gamma * (1 - tau**2) * dval
                else:
                    val = 0.0
                    report.n_filled += 1
                if val is not None:
                    vd.set_value(_clamped(vd, val))

    # the tail itself: every Var copy takes the trajectory at its own
    # time plus dt, still on the tail; derivatives rescale by the chain
    # rule through the map, and the pin slacks keep their values
    if recs:
        for r in recs:
            copy = r.get("copy")
            if copy is None or not hasattr(copy, "ctype") or copy.ctype is not Var:
                continue
            for cidx, vd in copy.items():
                if vd.fixed:
                    continue
                tau = cidx[-1] if isinstance(cidx, tuple) else cidx
                o = cidx[:-1] if isinstance(cidx, tuple) else ()
                if tau >= 1 - _EPS:
                    continue  # the endpoint maps to itself
                t_abs = tN + math.atanh(tau) / gamma
                val = tail_eval(copy, o, t_abs + dt, step=r["kind"] == "control")
                if val is not None:
                    vd.set_value(_clamped(vd, val))
                    report.n_interpolated += 1
            dcopy = r.get("derivative")
            if r["kind"] == "state" and dcopy is not None:
                for cidx, vd in dcopy.items():
                    if vd.fixed:
                        continue
                    tau = cidx[-1] if isinstance(cidx, tuple) else cidx
                    o = cidx[:-1] if isinstance(cidx, tuple) else ()
                    if tau >= 1 - _EPS:
                        continue
                    t_abs = tN + math.atanh(tau) / gamma
                    tau2 = math.tanh(gamma * (t_abs + dt - tN))
                    dpts, dys = tail_curve(dcopy, o)
                    dval = _interp(dpts, dys, tau2)
                    if dval is not None:
                        vd.set_value(_clamped(vd, dval * (1 - tau2**2) / (1 - tau**2)))
    # --- multipliers: part of the previous solution, shifted the same
    # way when the model carries the suffixes. Equality duals (m.dual)
    # shift within each constraint family, the seam reading the recorded
    # tail row copies; bound multipliers (ipopt_zL/zU_out) shift over the
    # same trajectories as the primals into ipopt_zL/zU_in. Declare the
    # suffixes before the first solve; absent suffixes are skipped
    from pyomo.core import Constraint, Suffix

    def _is_suffix(c):
        return c is not None and c.ctype is Suffix

    seg_block = None
    if recs:
        seg_block = next(r["of"] for r in recs if r["kind"] == "segment")

    def _own(container, o):
        return sorted(
            (ci if not isinstance(ci, tuple) else ci[-1])
            for ci in container
            if (ci[:-1] if isinstance(ci, tuple) else ()) == o
        )

    def shift_var_suffix(get, put):
        def one(comp):
            pos, combos = _axis(comp, time)
            for o in combos:
                pts = _points(comp, pos, o) or grid
                members = [
                    (
                        comp[_join_index(o, t, pos)]
                        if pos is not None
                        else (comp[tuple(o) + (t,)] if o else comp[t])
                    )
                    for t in pts
                ]
                old_v = [get(vd) for vd in members]
                if all(v is None for v in old_v):
                    return
                for t, vd in zip(pts, members):
                    if vd.fixed:
                        continue
                    tp = t + dt
                    if tp <= pts[-1] + _EPS:
                        val = _interp(pts, old_v, tp)
                    elif (
                        id(vd) in tail_of
                        and getattr(tail_of[id(vd)][0], "ctype", None) is Var
                    ):
                        copy, oo = tail_of[id(vd)]
                        cpts = _own(copy, oo)
                        cys = [
                            get(copy[tuple(oo) + (q,)] if oo else copy[q]) for q in cpts
                        ]
                        tau = math.tanh(gamma * (tp - tN))
                        val = _interp([0.0] + cpts, [old_v[-1]] + cys, tau)
                    else:
                        val = old_v[-1]
                    if val is not None:
                        put(vd, val)

        for comp in list(m.component_objects(Var, active=True)):
            if id(comp) in ctrl_ids:
                continue
            pos, _su = _time_index(comp, time)
            if pos is None:
                continue
            if recs and comp.parent_block() is seg_block:
                continue
            one(comp)
        for u in controls:
            one(u)
        if recs:
            for r in recs:
                copy = r.get("copy")
                if copy is None or getattr(copy, "ctype", None) is not Var:
                    continue
                for cidx, vd in list(copy.items()):
                    if vd.fixed:
                        continue
                    tau = cidx[-1] if isinstance(cidx, tuple) else cidx
                    o = cidx[:-1] if isinstance(cidx, tuple) else ()
                    if tau >= 1 - _EPS:
                        continue
                    cpts = _own(copy, o)
                    cys = [get(copy[tuple(o) + (q,)] if o else copy[q]) for q in cpts]
                    if all(v is None for v in cys):
                        continue
                    t_abs = tN + math.atanh(tau) / gamma
                    tau2 = math.tanh(gamma * (t_abs + dt - tN))
                    val = _interp(cpts, cys, tau2)
                    if val is not None:
                        put(vd, val)

    shifted = []
    dual = m.component("dual")
    if _is_suffix(dual):
        row_tail = {}
        tail_row_fams = set()
        if recs:
            for r in recs:
                if r["kind"] in ("dynamics", "algebraic_row"):
                    row_tail[id(r["of"])] = r["copy"]
                if r["kind"] in ("dynamics", "algebraic_row", "block_row") and (
                    r.get("copy") is not None
                ):
                    tail_row_fams.add(id(r["copy"]))
                if r["kind"] == "dynamics" and r.get("residue") is not None:
                    tail_row_fams.add(id(r["residue"]))
        for con in list(m.component_objects(Constraint, active=True)):
            axis_pos, _su = _time_index(con, time)
            if axis_pos is None and id(con) not in tail_row_fams:
                continue
            axis_pos, combos = _axis(con, time)
            tailfam = row_tail.get(id(con))
            in_tail = id(con) in tail_row_fams
            for o in combos:
                pts = _points(con, axis_pos, o) or grid
                rows = []
                for t in pts:
                    try:
                        rows.append(
                            con[_join_index(o, t, axis_pos)]
                            if axis_pos is not None
                            else (con[tuple(o) + (t,)] if o else con[t])
                        )
                    except KeyError:
                        rows.append(None)
                old_v = [None if cd is None else dual.get(cd) for cd in rows]
                if all(v is None for v in old_v):
                    continue
                for t, cd in zip(pts, rows):
                    if cd is None:
                        continue
                    if in_tail:
                        t_abs = tN + math.atanh(min(t, 1 - 1e-12)) / gamma
                        tau2 = math.tanh(gamma * (t_abs + dt - tN))
                        val = _interp(pts, old_v, tau2)
                    else:
                        tp = t + dt
                        if tp <= pts[-1] + _EPS:
                            val = _interp(pts, old_v, tp)
                        elif tailfam is not None:
                            fpts = _own(tailfam, o)
                            fys = [
                                dual.get(tailfam[tuple(o) + (q,)] if o else tailfam[q])
                                for q in fpts
                            ]
                            tau = math.tanh(gamma * (tp - tN))
                            val = _interp(fpts, fys, tau)
                        else:
                            val = old_v[-1]
                    if val is not None:
                        dual[cd] = val
        shifted.append("dual")

    for zname in ("ipopt_zL", "ipopt_zU"):
        z_out = m.component(zname + "_out")
        z_in = m.component(zname + "_in")
        if _is_suffix(z_out) and _is_suffix(z_in):
            shift_var_suffix(
                lambda vd, _s=z_out: _s.get(vd),
                lambda vd, val, _s=z_in: _s.__setitem__(vd, val),
            )
            shifted.append(zname)
    if shifted:
        report.multipliers = "shifted: " + ", ".join(shifted)
    return report
