# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Cold start: ``drto.cold_start_dynamic`` (feature 011).

Initializes a dynamic model whose initial condition sits away from the
steady state: each declared state runs on a straight line from its
declared initial condition to its declared steady-state target, its
DerivativeVar members hold the line's slope, the controls (and a
parameterized control's moves) hold their declared targets, and, with
pyomo-pounce installed, the algebraic variables solve pointwise from
every equation except the declared dynamics; with an active
``scaling_factor`` suffix those solves run on a scaled clone and the
values propagate back. A terminal segment rests at the targets: copies
and segment controls at the targets, tau derivatives and pin slacks at
zero.

Values only, at any stage: the declared discretized model, or after
``drto.infinite_horizon``, ``drto.dynamic_optimization``, or
``drto.dynamic_simulation``. The steady-state values come from the
declared pairings; a missing pairing is an error naming the component,
and no equilibrium solve is run. Without pyomo-pounce the per-point
solves are skipped, the algebraic variables keep their values, and the
report says so.
"""
from dataclasses import dataclass, field

import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
from pyomo.core import Var
from pyomo.dae import DerivativeVar

from drto.declarations import _side_matching
from drto.infinite_horizon import _join_index, _split_index, _time_index
from drto.info import info

pyomo_pounce, pounce_available = attempt_import("pyomo_pounce")

#: The declarations the function requires.
_REQUIRED = ("horizon", "state", "dynamics", "initial_condition")


@dataclass
class ColdStartReport:
    """What the cold start set, and what the per-point solves did."""

    n_states: int = 0
    n_grid_points: int = 0
    n_derivatives: int = 0
    n_controls: int = 0
    segment: str = "(none attached)"
    point_solves: str = "skipped (pyomo-pounce not installed)"
    pipeline: object = None
    notes: list = field(default_factory=list)

    def __str__(self):
        lines = [
            "drto cold_start_dynamic (interpolate to the targets)",
            f"  states        : {self.n_states} on a line across "
            f"{self.n_grid_points} grid points",
            f"  derivatives   : {self.n_derivatives} at the line's slope",
            f"  controls      : {self.n_controls} at their targets",
            f"  segment       : {self.segment}",
            f"  point solves  : {self.point_solves}",
        ]
        lines.extend("  " + n for n in self.notes)
        if self.pipeline is not None:
            lines.extend("  " + ln for ln in str(self.pipeline).splitlines())
        return "\n".join(lines)


def _target(pairings, comp, kind, fn):
    """The paired target Param of ``comp``, or a descriptive error."""
    for rec in pairings:
        if rec["of"] is comp:
            return rec["component"]
    raise ValueError(
        f"drto: {fn}: '{comp.name}' has no declared {kind} pairing; the "
        f"targets are the interpolation's endpoint, so every declared "
        f"component needs one (drto.{kind} first)."
    )


def cold_start_dynamic(m):
    """Initialize ``m`` from its declared initial condition to its declared
    steady-state targets; see the module docstring.

    Returns a :class:`ColdStartReport`; values only, nothing added or
    removed, and the fixed flags are untouched.
    """
    fn = "cold_start_dynamic"
    reg = info(m)
    missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
    if missing:
        raise ValueError(
            f"drto: {fn} requires the declarations "
            f"{', '.join(_REQUIRED)}; missing: {', '.join(missing)}."
        )
    time = reg.components("horizon")[0]
    if not time.get_discretization_info():
        raise ValueError(
            f"drto: {fn} initializes across the grid, so the model must be "
            f"discretized first (apply a dae.* transformation)."
        )
    states = list(reg.components("state"))
    controls = list(reg.components("control"))
    ss = list(reg.declarations("steady_state"))
    uss = list(reg.declarations("steady_state_control"))
    z_target = {id(z): _target(ss, z, "steady_state", fn) for z in states}
    u_target = {id(u): _target(uss, u, "steady_state_control", fn) for u in controls}

    grid = sorted(time)
    t0, tN = grid[0], grid[-1]
    horizon = tN - t0

    # the declared initial condition: each row pins a state member at t0
    # to a mutable Param, the feedback hook; the values are the line's
    # start, keyed by the pinned member's data id
    z0 = {}
    for con in reg.components("initial_condition"):
        for cd in con.values() if con.is_indexed() else (con,):
            side, other = _side_matching(
                cd,
                lambda s: getattr(s, "is_variable_type", lambda: False)(),
                fn,
                "a state member",
            )
            z0[id(side)] = pyo.value(other)

    report = ColdStartReport(n_grid_points=len(grid))

    # states on the line, their declared members' derivatives at its slope
    slope_of = {}  # id(state member data at any t) -> slope, for derivatives
    for z in states:
        pos, subs = _time_index(z, time)
        combos = set()
        for idx in z:
            o, _t = _split_index(idx, pos, len(subs))
            combos.add(o)
        tgt_param = z_target[id(z)]
        for o in combos:
            member0 = z[_join_index(o, t0, pos)]
            tgt = pyo.value(tgt_param[o] if o else tgt_param)
            start = z0.get(id(member0), tgt)
            slope = (tgt - start) / horizon
            for t in grid:
                vd = z[_join_index(o, t, pos)]
                if not vd.fixed:
                    vd.set_value(start + slope * (t - t0))
                slope_of[id(vd)] = slope
        report.n_states += 1

    for dv in m.component_objects(Var, active=True):
        if not (
            isinstance(dv, DerivativeVar) and dv.get_continuousset_list() == [time]
        ):
            continue
        zvar = dv.get_state_var()
        for idx, dvd in dv.items():
            slope = slope_of.get(id(zvar[idx]))
            if slope is not None and not dvd.fixed:
                dvd.set_value(slope)
                report.n_derivatives += 1

    # controls, and a parameterized control's moves, at their targets; a
    # fixed control (a simulation's) keeps the value it holds
    for u in controls:
        tgt_param = u_target[id(u)]
        pos, subs = _time_index(u, time)
        for idx, vd in u.items():
            if vd.fixed:
                continue
            o, _t = _split_index(idx, pos, len(subs)) if pos is not None else ((), None)
            vd.set_value(pyo.value(tgt_param[o] if o else tgt_param))
        report.n_controls += 1

    # a terminal segment rests at the targets
    b = m.component("drto_ih")
    if b is not None:
        n_seg = 0
        for z in states:
            tgt_param = z_target[id(z)]
            copy = b.component(z.local_name)
            deriv = b.component(z.local_name + "_dtau")
            for comp, val in ((copy, None), (deriv, 0.0)):
                if comp is None:
                    continue
                for idx, vd in comp.items():
                    if vd.fixed:
                        continue
                    o = idx[:-1] if isinstance(idx, tuple) else ()
                    vd.set_value(
                        val
                        if val is not None
                        else pyo.value(tgt_param[o] if o else tgt_param)
                    )
                n_seg += 1
            for suffix in ("_pin_up", "_pin_lo"):
                slack = b.component(z.local_name + suffix)
                if slack is not None:
                    for vd in slack.values():
                        vd.set_value(0.0)
        for u in controls:
            copy = b.component(u.local_name)
            if copy is None:
                continue
            tgt = pyo.value(u_target[id(u)])
            for vd in copy.values():
                if not vd.fixed:
                    vd.set_value(tgt)
            n_seg += 1
        report.segment = f"{n_seg} copies at the targets, slacks at zero"

    # the per-point solves: everything except the declared dynamics and
    # the initial condition determines the rest, each grid point its own
    # block once the states and controls are held. An undeclared member
    # of a packed Var comes from its closures, and its derivative from
    # the discretization rows; a variable only a set-aside balance would
    # close keeps its value and is reported underconstrained. With an
    # active scaling_factor suffix the solves run on a scaled clone and
    # the values propagate back; the model stays in its own units.
    if not pounce_available:
        return report
    scaled = any(
        s.local_name == "scaling_factor"
        for s in m.component_objects(pyo.Suffix, active=True)
    )
    if scaled:
        xfrm = pyo.TransformationFactory("core.scale_model")
        solve_m = xfrm.create_using(m, rename=False)
        solve_reg = info(solve_m)
    else:
        solve_m, solve_reg = m, reg
    held = []
    for kind in ("state", "control"):
        for comp in solve_reg.components(kind):
            held.extend(vd for vd in comp.values() if not vd.fixed)
    rows = []
    for kind in ("dynamics", "initial_condition"):
        for con in solve_reg.components(kind):
            if con.active:
                rows.append(con)
                con.deactivate()
    try:
        report.pipeline = pyomo_pounce.initialize(
            solve_m, decisions=held, fill=None, project=False
        )
        report.point_solves = (
            "run (pyomo-pounce block solve, scaled clone)"
            if scaled
            else "run (pyomo-pounce block solve)"
        )
    finally:
        for con in rows:
            con.activate()
    if scaled:
        xfrm.propagate_solution(solve_m, m)
    return report
