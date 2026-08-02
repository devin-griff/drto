# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Cold start: ``drto.cold_start_dynamic`` (feature 011).

Initializes a dynamic model whose initial condition sits away from the
steady state: each declared state runs from its declared initial
condition to its declared steady-state target, on a straight line or,
with ``profile="exponential"``, on a normalized exponential decay that
lands exactly on the target at the horizon's end. Its
DerivativeVar members hold the profile's slope, the controls (and a
parameterized control's moves) hold their declared targets, and, with
pyomo-pounce installed, the algebraic variables solve pointwise from
every equation except the declared dynamics (``point_solves=False``
skips them deliberately); with an active
``scaling_factor`` suffix those solves run on a scaled clone, the
values propagate back, and the report carries the initialized clone as
``scaled_model`` for a consumer that wants a persistent scaled model.
A terminal segment rests at the targets: copies and segment controls
at the targets, tau derivatives and pin slacks at zero.

Values only, at any stage: the declared discretized model, or after
``drto.infinite_horizon``, ``drto.dynamic_optimization``, or
``drto.dynamic_simulation``. The steady-state values come from the
declared pairings; a missing pairing is an error naming the component,
and no equilibrium solve is run. Without pyomo-pounce the per-point
solves are skipped, the algebraic variables keep their values, and the
report says so.
"""
import math
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
    shape: str = "a line"
    segment: str = "(none attached)"
    point_solves: str = "skipped (pyomo-pounce not installed)"
    pipeline: object = None
    #: The initialized scaled clone, when the solves ran scaled (its
    #: factor map rides on it as ``component_scaling_factor_map``); the
    #: closed loop adopts it as its persistent solve model (gh #42).
    #: Lives as long as the report does; drop the report to release it.
    scaled_model: object = None
    notes: list = field(default_factory=list)

    def __str__(self):
        lines = [
            "drto cold_start_dynamic (interpolate to the targets)",
            f"  states        : {self.n_states} on {self.shape} across "
            f"{self.n_grid_points} grid points",
            f"  derivatives   : {self.n_derivatives} at the profile's slope",
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


def cold_start_dynamic(m, profile="linear", time_constant=None, point_solves=True):
    """Initialize ``m`` from its declared initial condition to its declared
    steady-state targets; see the module docstring.

    ``profile`` is ``"linear"`` (the default, a straight line) or
    ``"exponential"`` (a normalized decay landing exactly on the target
    at the horizon's end). ``time_constant`` is the decay's time
    constant in the horizon's own units, a third of the horizon when not
    given; it belongs to the exponential profile only. ``point_solves``
    is the algebra choice: ``True`` (the default) runs the per-point
    solves, ``False`` skips them deliberately, the profiles and targets
    landing without a solve, the scaled clone never built, and the
    report saying so.

    Returns a :class:`ColdStartReport`; values only, nothing added or
    removed, and the fixed flags are untouched.
    """
    fn = "cold_start_dynamic"
    if profile not in ("linear", "exponential"):
        raise ValueError(
            f"drto: {fn}: unknown profile '{profile}'; the profiles are "
            f"'linear' and 'exponential'."
        )
    if point_solves not in (True, False):
        raise ValueError(
            f"drto: {fn}: point_solves is True (run the per-point algebra "
            f"solves) or False (skip them); got {point_solves!r}."
        )
    if time_constant is not None:
        if profile != "exponential":
            raise ValueError(
                f"drto: {fn}: time_constant is the exponential profile's "
                f"knob; pass profile='exponential' with it."
            )
        if time_constant <= 0:
            raise ValueError(
                f"drto: {fn}: time_constant must be positive, got " f"{time_constant}."
            )
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
    if profile == "exponential":
        tau = horizon / 3.0 if time_constant is None else time_constant
        d = horizon / tau
        denom = 1.0 - math.exp(-d)
        report.shape = f"an exponential decay (time constant {tau:g})"

    # states on the profile, their declared members' derivatives at its
    # pointwise slope
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
            span = tgt - start
            for t in grid:
                # a member that does not exist is skipped: a model cut
                # to a window of the horizon (the loop's one-sample
                # plant) initializes over the members it kept
                idx = _join_index(o, t, pos)
                if idx not in z:
                    continue
                s = (t - t0) / horizon
                if profile == "exponential":
                    val = start + span * (1.0 - math.exp(-d * s)) / denom
                    slope = span * d * math.exp(-d * s) / (denom * horizon)
                else:
                    val = start + span * s
                    slope = span / horizon
                vd = z[idx]
                if not vd.fixed:
                    vd.set_value(val)
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

    # a terminal segment rests at the targets; the transform recorded
    # which tail component belongs to which declaration (gh #27)
    seg_state = {id(r["of"]): r for r in reg._segment_records("state")}
    seg_control = {id(r["of"]): r for r in reg._segment_records("control")}
    if seg_state or seg_control:
        n_seg = 0
        for z in states:
            rec = seg_state.get(id(z))
            if rec is None:
                continue
            tgt_param = z_target[id(z)]
            for comp, val in ((rec["copy"], None), (rec["derivative"], 0.0)):
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
            for slack in (rec["pin_up"], rec["pin_lo"]):
                if slack is not None:
                    for vd in slack.values():
                        vd.set_value(0.0)
        for u in controls:
            rec = seg_control.get(id(u))
            if rec is None or rec["copy"] is None:
                continue
            tgt = pyo.value(u_target[id(u)])
            for vd in rec["copy"].values():
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
    if not point_solves:
        # the deliberate skip: the profiles and targets are the whole
        # initialization, no clone, no solve
        report.point_solves = "skipped (by option)"
        return report
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
    # the terminal segment mirrors the finite horizon: its state and
    # control copies are held at the values set above, and set aside are
    # its copies of the declared dynamics (the residue rows included)
    # plus the rows those values satisfy by construction (the link, the
    # continuity, the pin), which would otherwise re-solve held copies.
    # The pin slacks then sit in no active row and stay at zero, and a
    # variable only the dynamics would close keeps its value, as on the
    # finite side. The pairing comes from the transform's records, which
    # a clone carries with its references remapped (gh #27)
    set_aside = []
    for rec in solve_reg._segment_records():
        if rec["kind"] in ("state", "control") and rec["copy"] is not None:
            held.extend(vd for vd in rec["copy"].values() if not vd.fixed)
        if rec["kind"] == "dynamics":
            set_aside += [rec["copy"], rec["residue"]]
        elif rec["kind"] == "state":
            set_aside += [rec["link"], rec["continuity"], rec["pin"]]
    for con in set_aside:
        if con is not None and con.active:
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
        # values only, copied by position: the clone is this model's
        # deepcopy, so the var data objects align one to one. Pyomo's
        # propagate_solution is avoided deliberately: it iterates Var
        # containers including References, and indexing a Reference
        # rebuilds members on demand, which resurrects what a model cut
        # to a window of the horizon removed (the closed loop's
        # one-sample plant); it also wants an objective to rescale the
        # clone's solver suffixes, which values-only never needs
        fmap = solve_m.component_scaling_factor_map
        for vo, vs in zip(
            m.component_data_objects(Var, descend_into=True),
            solve_m.component_data_objects(Var, descend_into=True),
        ):
            if vs.value is not None:
                vo.set_value(
                    pyo.value(vs) / (fmap[vs] if vs in fmap else 1.0),
                    skip_validation=True,
                )
        report.scaled_model = solve_m
    return report
