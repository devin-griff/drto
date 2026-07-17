# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The infinite-horizon terminal segment: ``drto.infinite_horizon``
(feature 004).

Appends a terminal segment to a declared, discretized dynamic model: the tail
of the horizon to t = infinity, compressed onto [0, 1] by the transformation
``tau = tanh(gamma*(t - tN))`` of Dinh et al. (2025,
doi:10.1016/j.jprocont.2025.103565). The segment carries copies of the
declared states and controls, the declared dynamics dilated by the
transformation Jacobian ``gamma*(1 - tau^2)`` at interior Gauss-Legendre
collocation points, a hard equilibrium endpoint ``0 = f`` at ``tau = 1``, and
the declared tracking stage cost replicated at the segment's collocation
points. The tail cost enters the objective as explicit Gauss-weighted terms,
the paper's ``(beta/dt)*phi_f`` with the quadrature state eliminated,
registered as a ``cost_group`` that ``drto.build_objective`` (feature 003)
picks up wherever it runs. There is no coupling option: applying this
transform before the mode transform is the whole composition.

States may carry index sets besides time; copies, linking, equilibrium, and
replication run per member. Algebraic variables and equations ride along
without being declared: any time-indexed variable the replicated equations
reference that is not a declared state or control gets a segment copy, and
every active time-indexed constraint not declared as something else (and not
a discretization artifact) is replicated at the interior points plus the
endpoint, which pins the algebraic values the equilibrium references at
``tau = 1``. A variable copied to the segment with no replicated equation
involving it is an error, not a silent free variable.

With the tail attached, the finite grid's final instant is the linking
time, not an end, so the transform re-declares each declared control's cvp
profile with ``final_node='keep'``: the control there is the held last
move, and the model's equations at that instant may reference it. The
segment's own control copies are parameterized the same way.
"""
import math
from itertools import product

from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue, PositiveInt
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.core import (
    Block,
    Constraint,
    Expression,
    Param,
    Transformation,
    TransformationFactory,
    Var,
)
from pyomo.core.expr import identify_variables, replace_expressions
from pyomo.dae import ContinuousSet, DerivativeVar

from drto.declarations import (
    _is_var_member,
    _side_matching,
    pyomo_cvp,
    pyomo_cvp_available,
)
from drto.info import info

#: The block the transform adds to the model.
_BLOCK_NAME = "drto_infinite_horizon"

#: The declarations the transform requires.
_REQUIRED = ("horizon", "state", "dynamics", "control", "tracking_stage_cost")


def _gauss_weights(nodes):
    """Quadrature weights for interior ``nodes`` on [0, 1], by moment solve.

    ``pyomo.dae`` stores the collocation nodes but no quadrature weights, so
    they are derived from the nodes the discretization actually used: the
    K-point rule integrating 1, x, ..., x^(K-1) exactly.
    """
    k = len(nodes)
    a = numpy.array([[x**p for x in nodes] for p in range(k)], dtype=float)
    b = numpy.array([1.0 / (p + 1) for p in range(k)])
    return numpy.linalg.solve(a, b)


def _is_derivative(node):
    """Return whether ``node`` is a DerivativeVar member."""
    parent = getattr(node, "parent_component", lambda: None)()
    return isinstance(parent, DerivativeVar)


def _time_index(comp, time):
    """Return ``(position, subsets)`` of the time set in ``comp``'s index.

    ``position`` is None when ``comp`` is not indexed by ``time``.
    """
    subs = list(comp.index_set().subsets())
    for n, s in enumerate(subs):
        if s is time:
            return n, subs
    return None, subs


def _split_index(idx, pos, nsub):
    """Split a member index into (other-coordinates, time-coordinate)."""
    if nsub == 1:
        return (), idx
    idx = tuple(idx)
    return idx[:pos] + idx[pos + 1 :], idx[pos]


def _join_index(other, t, pos):
    """Rebuild a member index from other-coordinates and a time coordinate."""
    if not other:
        return t
    other = tuple(other)
    return other[:pos] + (t,) + other[pos:]


@TransformationFactory.register(
    "drto.infinite_horizon",
    doc="Append the infinite-horizon terminal segment of Dinh et al. (2025) "
    "to a declared, discretized dynamic model (drto).",
)
class InfiniteHorizonTransformation(Transformation):
    """Append the terminal segment; see the module docstring.

    Options: ``nfe`` and ``ncp`` set the segment mesh (defaults 3 and 5),
    ``beta`` the tail overestimation factor (mutable Param, default 1.2,
    strictly greater than 1), ``gamma`` overrides the mesh rule
    ``tanh(gamma*dt) = tau_11`` (mutable Param, derived by default), and
    ``profile`` sets the segment controls' pyomo-cvp profile (default
    ``'collocation'``, with ``'piecewise_constant'`` the conservative
    alternative).
    """

    CONFIG = ConfigDict("drto.infinite_horizon")
    CONFIG.declare(
        "nfe",
        ConfigValue(
            default=3,
            domain=PositiveInt,
            description="Finite elements on the terminal segment.",
        ),
    )
    CONFIG.declare(
        "ncp",
        ConfigValue(
            default=5,
            domain=PositiveInt,
            description="Gauss-Legendre collocation points per element.",
        ),
    )
    CONFIG.declare(
        "beta",
        ConfigValue(
            default=1.2,
            domain=float,
            description="Tail overestimation safety factor, strictly "
            "greater than 1.",
        ),
    )
    CONFIG.declare(
        "gamma",
        ConfigValue(
            default="rule",
            description="Time-compression rate: 'rule' (the default) derives "
            "it from the mesh rule, the segment's first collocation point "
            "one sampling time past the junction; a number overrides.",
        ),
    )
    CONFIG.declare(
        "profile",
        ConfigValue(
            default="collocation",
            description="pyomo-cvp profile for the segment controls: "
            "'collocation' (default) or 'piecewise_constant'.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        if config.beta <= 1:
            raise ValueError(
                f"drto: infinite_horizon requires beta > 1 (the terminal "
                f"cost must overestimate the tail; the margin beta - 1 "
                f"covers the quadrature error). Got beta={config.beta}."
            )
        if not pyomo_cvp_available:
            raise RuntimeError(
                "drto: infinite_horizon requires pyomo-cvp for the segment "
                "control profiles (pip install pyomo-cvp)."
            )
        if not numpy_available:
            raise RuntimeError(
                "drto: infinite_horizon requires numpy for the quadrature " "weights."
            )
        # validate before anything is added to the model: a bad profile must
        # not error midway through the segment construction
        pyomo_cvp.parameterize._validate_profile(config.profile)

        reg = info(model)
        missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
        if "tracking_stage_cost" in missing and reg.has_declaration(
            "economic_stage_cost"
        ):
            raise ValueError(
                "drto: infinite_horizon requires a tracking stage cost. An "
                "economic stage cost alone is rejected: it is nonzero at the "
                "equilibrium, so its tail integral diverges and its "
                "quadrature would be mesh-dependent."
            )
        if missing:
            raise ValueError(
                "drto: infinite_horizon requires "
                + ", ".join(f"drto.{k}" for k in missing)
                + " first."
            )
        if reg.has_transformation("drto.infinite_horizon"):
            raise ValueError(
                "drto: infinite_horizon was already applied to this model."
            )
        drto_obj = model.component("drto_objective")
        if drto_obj is not None and drto_obj.active:
            raise ValueError(
                "drto: the objective is already assembled; apply "
                "drto.infinite_horizon first, then rebuild with "
                "drto.build_objective."
            )

        time = reg.components("horizon")[0]
        if not time.get_discretization_info():
            raise ValueError(
                f"drto: discretize the declared time set '{time.name}' "
                f"(dae.collocation) before applying infinite_horizon."
            )
        samples = reg.declarations("horizon")[0]["samples"]
        dt = samples[1] - samples[0]
        t_end = time.last()

        states = reg.components("state")
        controls = reg.components("control")
        dynamics = reg.components("dynamics")
        (stage_record,) = reg.declarations("tracking_stage_cost")
        stage_con = stage_record["component"]

        if reg.has_transformation("drto.parameterize"):
            raise ValueError(
                "drto: the control profiles are already applied; apply "
                "drto.infinite_horizon before drto.parameterize (it "
                "replicates the controls in their original time indexing)."
            )
        for comp in controls:
            if comp.index_set() is not time:
                raise ValueError(
                    f"drto: infinite_horizon supports controls indexed by "
                    f"the declared time set only; '{comp.name}' is not."
                )
        for comp in states:
            pos, _ = _time_index(comp, time)
            if pos is None:
                raise ValueError(
                    f"drto: infinite_horizon requires states indexed by the "
                    f"declared time set; '{comp.name}' is not."
                )

        states_set = ComponentSet(states)
        controls_set = ComponentSet(controls)

        # --- index layout helpers -------------------------------------
        layout = {}

        def _layout(comp):
            if comp not in layout:
                pos, subs = _time_index(comp, time)
                layout[comp] = (pos, [s for n, s in enumerate(subs) if n != pos])
            return layout[comp]

        def _combos(comp):
            _, others = _layout(comp)
            return list(product(*others)) if others else [()]

        def _member(comp, o, t):
            pos, _ = _layout(comp)
            return comp[_join_index(o, t, pos)]

        def _representatives(con):
            """One member per other-combo: ``{other: (t_rep, condata)}``."""
            pos, subs = _time_index(con, time)
            reps = {}
            for idx, cd in con.items():
                o, t = _split_index(idx, pos, len(subs))
                if o not in reps:
                    reps[o] = (t, cd)
            return reps

        # --- discovery: algebraic constraints are every active
        # time-indexed constraint not declared as something else and not a
        # discretization artifact; algebraic variables are every
        # time-indexed variable the replicated equations reference that is
        # not a declared state or control ------------------------------
        declared_cons = ComponentSet()
        for kind in (
            "dynamics",
            "tracking_stage_cost",
            "economic_stage_cost",
            "tracking_terminal_cost",
            "initial_condition",
            "terminal_constraint",
        ):
            declared_cons.update(reg.components(kind))
        alg_cons = []
        for con in model.component_objects(Constraint, active=True):
            # pyomo.dae artifacts: collocation equations ('_disc_') and the
            # Legendre continuity equations ('_cont_eq'); the segment builds
            # its own discretization with its own continuity
            if (
                con in declared_cons
                or "_disc_" in con.local_name
                or con.local_name.endswith("_cont_eq")
            ):
                continue
            pos, _ = _time_index(con, time)
            if pos is None:
                continue
            alg_cons.append(con)

        algebraic = ComponentSet()

        def _scan(expr, t_rep, where):
            """Validate a template; collect the algebraic components."""
            for v in identify_variables(expr, include_fixed=True):
                comp = v.parent_component()
                if isinstance(comp, DerivativeVar):
                    raise ValueError(
                        f"drto: infinite_horizon cannot replicate "
                        f"'{where}': it references the derivative "
                        f"'{v.name}' outside its own dynamics equation."
                    )
                pos, subs = _time_index(comp, time)
                if pos is None:
                    continue  # time-invariant: shared with the segment as-is
                _, t = _split_index(v.index(), pos, len(subs))
                if t != t_rep:
                    raise ValueError(
                        f"drto: infinite_horizon cannot replicate "
                        f"'{where}': it references '{v.name}' away from "
                        f"the constraint's own time point."
                    )
                if comp not in states_set and comp not in controls_set:
                    algebraic.add(comp)

        dyn_reps = {}
        for con in dynamics:
            entries = {}
            for o, (t_rep, cd) in _representatives(con).items():
                deriv_side, rhs = _side_matching(
                    cd, _is_derivative, "infinite_horizon", "a DerivativeVar"
                )
                z = deriv_side.parent_component().get_state_var()
                zpos, zsubs = _time_index(z, time)
                zo, _ = _split_index(deriv_side.index(), zpos, len(zsubs))
                _scan(rhs, t_rep, cd.name)
                entries[o] = (z, zo, rhs, t_rep)
            dyn_reps[con] = entries

        alg_reps = {}
        for con in alg_cons:
            entries = {}
            for o, (t_rep, cd) in _representatives(con).items():
                _scan(cd.expr, t_rep, cd.name)
                entries[o] = (cd.expr, t_rep)
            alg_reps[con] = entries

        cd = next(iter(stage_con.values())) if stage_con.is_indexed() else stage_con
        t_rep_cost = cd.index()
        cost_side, psi = _side_matching(
            cd, _is_var_member, "infinite_horizon", "the cost variable"
        )
        _scan(psi, t_rep_cost, cd.name)
        cost_var = cost_side.parent_component()

        # every variable copied to the segment must have at least one
        # replicated equation involving it; a variable with none would be
        # free there, and the solver would exploit it silently
        defined = ComponentSet()
        for entries in alg_reps.values():
            for expr, _ in entries.values():
                for v in identify_variables(expr, include_fixed=True):
                    defined.add(v.parent_component())
        for comp in algebraic:
            if comp not in defined:
                raise ValueError(
                    f"drto: infinite_horizon copies '{comp.name}' to the "
                    f"segment, but no replicated equation involves it; its "
                    f"defining equation must be indexed by the declared "
                    f"time set '{time.name}'."
                )

        # --- the segment block ----------------------------------------
        b = Block(concrete=True)
        model.add_component(_BLOCK_NAME, b)
        b.tau = ContinuousSet(bounds=(0, 1))
        b.gamma = Param(initialize=1.0, mutable=True)
        b.beta = Param(initialize=config.beta, mutable=True)

        seg = {}
        for comp in list(states) + list(controls) + list(algebraic):
            _, others = _layout(comp)
            v = Var(*others, b.tau) if others else Var(b.tau)
            b.add_component(comp.local_name, v)
            seg[comp] = v
        derivs = {}
        for z in states:
            dv = DerivativeVar(seg[z], wrt=b.tau)
            b.add_component(z.local_name + "_dtau", dv)
            derivs[z] = dv

        def _seg_at(comp, o, s):
            v = seg[comp]
            return v[tuple(o) + (s,)] if o else v[s]

        def _emap(t_rep, s):
            """Model members at ``t_rep`` mapped to segment members at
            ``s``."""
            mmap = {}
            for comp in seg:
                for o in _combos(comp):
                    mmap[id(_member(comp, o, t_rep))] = _seg_at(comp, o, s)
            return mmap

        # --- dilated dynamics at interior collocation points (eq. 25) ---
        for con in dynamics:
            pos, subs = _time_index(con, time)
            others = [s_ for n, s_ in enumerate(subs) if n != pos]

            def dyn_rule(blk, *idx, _entries=dyn_reps[con]):
                s = idx[-1]
                o = tuple(idx[:-1])
                if s in blk.tau.get_finite_elements() or o not in _entries:
                    return Constraint.Skip
                z, zo, rhs, t_rep = _entries[o]
                dv = derivs[z]
                deriv = dv[tuple(zo) + (s,)] if zo else dv[s]
                return blk.gamma * (1 - s**2) * deriv == replace_expressions(
                    rhs, _emap(t_rep, s)
                )

            b.add_component(con.local_name, Constraint(*others, b.tau, rule=dyn_rule))

        # --- algebraic equations, replicated as written at the interior
        # points plus the endpoint (they pin the algebraic values the
        # equilibrium references at tau = 1); no boundary values ---------
        for con in alg_cons:
            pos, subs = _time_index(con, time)
            others = [s_ for n, s_ in enumerate(subs) if n != pos]

            def alg_rule(blk, *idx, _entries=alg_reps[con]):
                s = idx[-1]
                o = tuple(idx[:-1])
                fe_pts = blk.tau.get_finite_elements()
                if (s in fe_pts and s != fe_pts[-1]) or o not in _entries:
                    return Constraint.Skip
                expr, t_rep = _entries[o]
                return replace_expressions(expr, _emap(t_rep, s))

            b.add_component(con.local_name, Constraint(*others, b.tau, rule=alg_rule))

        # --- link the segment to the end of the horizon ------------------
        for z in states:
            pos, others = _layout(z)

            def link_rule(blk, *o, _z=z):
                o = tuple(v for v in o if v is not None)  # scalar rules get None
                return _seg_at(_z, o, 0) == _member(_z, o, t_end)

            b.add_component(
                z.local_name + "_link",
                (
                    Constraint(*others, rule=link_rule)
                    if others
                    else Constraint(rule=link_rule)
                ),
            )

        # --- discretize the segment: Gauss-Legendre only, no collocation
        # equation may sit at the singular endpoint tau = 1 ---
        TransformationFactory("dae.collocation").apply_to(
            b, wrt=b.tau, nfe=config.nfe, ncp=config.ncp, scheme="LAGRANGE-LEGENDRE"
        )

        # --- gamma: the mesh rule, or the explicit override ---
        tau11 = sorted(b.tau)[1]
        if config.gamma in (None, "rule"):
            gamma_val = math.atanh(tau11) / dt
        else:
            try:
                gamma_val = float(config.gamma)
            except (TypeError, ValueError):
                raise ValueError(
                    f"drto: gamma must be 'rule' (derive from the mesh "
                    f"rule) or a number; got {config.gamma!r}."
                ) from None
        b.gamma.set_value(gamma_val)

        # --- per-member bounds and initial values from the horizon end ---
        for comp in seg:
            for o in _combos(comp):
                src = _member(comp, o, t_end)
                for s in sorted(b.tau):
                    v = _seg_at(comp, o, s)
                    v.setlb(src.lb)
                    v.setub(src.ub)
                    v.set_value(src.value)

        # --- the tracking stage cost, replicated as named Expressions at the
        # interior collocation points: the tail integrand. Expressions add no
        # variables and no constraints (a replicated cost Var would sit on an
        # active bound as the tail cost vanishes at the equilibrium), and
        # cvp's substitution sweep rewrites them like any constraint ---
        pts = sorted(b.tau)
        fe = b.tau.get_finite_elements()
        interior_pts = [p for p in pts if p not in fe]
        seg_cost = Expression(
            interior_pts,
            rule=lambda blk, s: replace_expressions(psi, _emap(t_rep_cost, s)),
        )
        b.add_component(cost_var.local_name, seg_cost)

        # --- hard equilibrium endpoint, 0 = f at tau = 1 (eq. 21c). The
        # state values at tau = 1 come from the Legendre extrapolation, the
        # algebraic values from the replicated equations at the endpoint,
        # and the control values from the segment profile at the endpoint
        # (the segment is parameterized with final_node='keep': the tail
        # continues to t = infinity, where the last move is held) ---
        for con in dynamics:
            pos, subs = _time_index(con, time)
            others = [s_ for n, s_ in enumerate(subs) if n != pos]

            def eq_rule(blk, *o, _entries=dyn_reps[con]):
                o = tuple(v for v in o if v is not None)  # scalar rules get None
                if o not in _entries:
                    return Constraint.Skip
                _, _, rhs, t_rep = _entries[o]
                return 0 == replace_expressions(rhs, _emap(t_rep, 1))

            b.add_component(
                con.local_name + "_equilibrium",
                (
                    Constraint(*others, rule=eq_rule)
                    if others
                    else Constraint(rule=eq_rule)
                ),
            )

        # --- segment control profiles: applied now, so raw unparameterized
        # copies are never left on the segment ---
        for u in controls:
            TransformationFactory("cvp.parameterize").apply_to(
                b, var=seg[u], contset=b.tau, profile=config.profile, final_node="keep"
            )

        # --- the tail cost: explicit Gauss weights, (beta/dt) * phi_f with
        # the quadrature state eliminated. beta and gamma stay symbolic in
        # the weights, so set_value retunes them without a re-apply ---
        interior = [[p for p in pts if lo < p < hi] for lo, hi in zip(fe, fe[1:])]
        h0 = fe[1] - fe[0]
        omega = _gauss_weights([(p - fe[0]) / h0 for p in interior[0]])
        terms = []
        for lo, hi, points in zip(fe, fe[1:], interior):
            h = hi - lo
            for p, w in zip(points, omega):
                weight = b.beta * (h * float(w)) / (b.gamma * dt * (1 - p**2))
                terms.append((seg_cost[p], weight))
        reg.record_declaration("cost_group", b, terms=tuple(terms))

        # the tail continues past the grid, so the finite horizon's final
        # instant is the linking time, not an end: re-declare each control's
        # profile with final_node='keep' (the held last move exists there),
        # which drto.parameterize applies later
        for record in reg.declarations("control"):
            pyomo_cvp.declare_profile(
                record["component"],
                wrt=time,
                profile=record["profile"],
                final_node="keep",
            )

        # the tail integral IS the cost-to-go, so a declared terminal cost
        # would double-count: deactivate it (build_objective's liveness rule
        # then drops its term) and record the outcome
        terminal = None
        for comp in reg.components("tracking_terminal_cost"):
            if comp.active:
                comp.deactivate()
                terminal = comp.name

        reg.record_transformation(
            "drto.infinite_horizon",
            segment=f"{config.nfe} elements x {config.ncp} Legendre points",
            beta=config.beta,
            gamma=round(gamma_val, 8),
            profile=config.profile,
            horizon="kept, infinite tail appended",
            **(
                {
                    "algebraic": f"{len(algebraic)} component"
                    + ("s" if len(algebraic) != 1 else "")
                    + " replicated"
                }
                if algebraic
                else {}
            ),
            **(
                {"terminal_cost": f"{terminal} deactivated (the tail owns it)"}
                if terminal
                else {}
            ),
        )
