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
collocation points, and the declared tracking stage cost replicated at those
points. The tail cost enters the objective as explicit Gauss-weighted terms,
the paper's ``(beta/dt)*phi_f`` with the quadrature state eliminated,
registered as a ``cost_group`` that ``drto.build_objective`` (feature 003)
picks up wherever it runs. There is no coupling option: applying this
transform before the mode transform is the whole composition.

The segment endpoint is pinned to the steady state by default (the paper's
eq 36). The endpoint ``z(tau=1)`` is the Legendre extrapolation of the last
element (Pyomo's continuity equation), the paper's evaluated endpoint z_e.
``terminal='soft'`` (the default) adds, per state, the relaxed endpoint
equality ``z(tau=1) + eps_up - eps_lo == z_s`` with the penalty
``mu*(eps_up + eps_lo + eps_up**2 + eps_lo**2)`` in the objective: the
linear part is the exact L1 pin, and the quadratic part makes the pin's
multipliers unique (gh #37); ``terminal='none'`` imposes no
endpoint condition, leaving the singular tail cost as the only terminal
enforcement (the endpoint settles as close to the setpoint as the horizon's
freedoms allow). A pin requires a declared ``drto.steady_state`` target for
every state.

States may carry index sets besides time; copies, linking, and replication
run per member. Algebraic variables and equations need no declarations:
any time-indexed variable the replicated equations reference that
is not a declared state or control gets a segment copy, and every active
time-indexed constraint not declared as something else (and not a
discretization artifact of the declared time set) is replicated at the
interior collocation points. A time-indexed Block family may carry further
indices (an IDAES stage element, a spatial node): each non-time combination
replicates as its own family. A derivative over another ContinuousSet (a
spatial axis) is ordinary algebra, its discretization rows replicating with
it, and same-named components from different units take distinct segment
names.
A replicated equation may reference a declared state's derivative (the
index-reduced energy-balance case): the reference maps to the segment
derivative with the dilation factor, the same rewrite the dynamics get.
A variable copied to the segment with no replicated equation involving it
is an error, not a silent free variable.
"""
import itertools
import math
from itertools import product

from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue, In, PositiveInt
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.core import (
    Block,
    Constraint,
    Expression,
    NonNegativeReals,
    Param,
    Set,
    Transformation,
    TransformationFactory,
    Var,
)
from pyomo.core.expr import identify_variables, replace_expressions
from pyomo.dae import ContinuousSet, DerivativeVar

from drto.declarations import (
    _dynamics_sides,
    _is_var_member,
    _side_matching,
    pyomo_cvp,
    pyomo_cvp_available,
)
from drto.dynamic_optimization import _spread
from drto.info import info

#: The block the transform adds to the model.
_BLOCK_NAME = "drto_ih"

#: The declarations the transform requires.
_REQUIRED = ("horizon", "state", "dynamics", "control", "tracking_stage_cost")


def _units_of(comp):
    """The component's declared units, read off one member; None if unitless.

    The segment copies, derivatives and pin slacks are built with these, so a
    unit-carrying model keeps its units across the transform (gh #10). tau is
    dimensionless, so a derivative over it takes the state's own units.
    """
    vd = next(iter(comp.values())) if comp.is_indexed() else comp
    return vd.get_units()


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
    alternative). ``terminal`` pins the segment endpoint to the steady state:
    ``'soft'`` (the default, eq 36, L1-penalized with weight ``mu``, default
    1000) or ``'none'`` (no pin). A pin requires a ``drto.steady_state``
    target for every state.
    """

    CONFIG = ConfigDict("drto.infinite_horizon")
    CONFIG.declare(
        "disturbances",
        ConfigValue(
            default=None,
            description="Mapping of declared disturbance (component or name) "
            "to the constant its segment copy is fixed at. Disturbances not "
            "in the mapping fix at zero: the tail continues under nominal "
            "disturbance unless told otherwise.",
        ),
    )
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
    CONFIG.declare(
        "terminal",
        ConfigValue(
            default="soft",
            domain=In(("none", "soft")),
            description="Endpoint pin on the extrapolated segment endpoint "
            "z(tau=1). 'soft' (the default): eq 36, z(tau=1) + eps_up - eps_lo "
            "== z_s with the penalty mu*(eps_up + eps_lo + eps_up**2 + "
            "eps_lo**2) in the objective. "
            "'none': no pin, the singular tail cost is the only terminal "
            "enforcement. A pin requires a drto.steady_state target for every "
            "state.",
        ),
    )
    CONFIG.declare(
        "mu",
        ConfigValue(
            default=1000.0,
            domain=float,
            description="Penalty weight for the soft endpoint pin "
            "(terminal='soft'); ignored otherwise. Weights the linear and the "
            "quadratic slack terms alike. A mutable Param on the "
            "segment, so it retunes with set_value and no re-apply. The paper "
            "requires mu above the endpoint multiplier norm for the penalty to "
            "be exact, driving the endpoint onto the setpoint.",
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

        # the endpoint pin needs a steady-state target per state; validate now,
        # before the segment block is built, so a missing target does not leave
        # a half-built block on the model
        ss_target = None
        if config.terminal != "none":
            ss_target = {
                id(r["of"]): r["component"] for r in reg.declarations("steady_state")
            }
            missing = [z.name for z in states if id(z) not in ss_target]
            if missing:
                raise ValueError(
                    f"drto: infinite_horizon terminal={config.terminal!r} pins "
                    f"the segment endpoint z(tau=1) to the steady state, so "
                    f"every declared state needs a drto.steady_state target; "
                    f"missing: {', '.join(missing)}. Declare "
                    f"drto.steady_state(state, target) for each, or pass "
                    f"terminal='none'."
                )

        states_set = ComponentSet(states)
        controls_set = ComponentSet(controls)

        # declared disturbances, by member data-id: a disturbance may be
        # declared as a flat Var or as a time-indexed Reference into Block
        # members (the IDAES inlet idiom), and data identity covers both
        disturbances = list(reg.components("disturbance"))
        disturbance_data = {}
        for d in disturbances:
            for vd in d.values() if d.is_indexed() else (d,):
                disturbance_data[id(vd)] = d
        disturbed = ComponentSet()

        # declared controls likewise, by member data-id: a control declared
        # as a Reference into Block members routes its members to the
        # declared control, so the control's own segment copy serves and no
        # separate member family is built (gh #18)
        control_data = {}
        for u in controls:
            for vd in u.values() if u.is_indexed() else (u,):
                control_data[id(vd)] = u

        # declared state members by data-id, with their declared component
        # and other-index: a state may be a Reference over a member subset
        # of an indexed Var (gh #20). A container with any declared member is
        # covered for the derivative checks, and the undeclared members of
        # a covered container (with their derivatives) copy per member as
        # algebraic equations like any other.
        state_member = {}
        for z in states:
            zpos, zsubs = _time_index(z, time)
            for idx, vd in z.items():
                zo, _zt = _split_index(idx, zpos, len(zsubs))
                state_member[id(vd)] = (z, zo)
        covered = {id(z) for z in states}
        for z in states:
            for vd in z.values():
                covered.add(id(vd.parent_component()))
        flat_partial = {}  # indexed Var -> its referenced algebraic entries

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

        def _time_artifact(con):
            """Whether a pyomo.dae artifact belongs to the declared time
            set. A discretization equation is the time set's when its
            derivative is taken with respect to it; one over another
            ContinuousSet (a spatial axis) is real algebra the segment
            must replicate (a settler's ``material_flow_dx_disc_eq``).
            Continuity rows carry no derivative and arise from the
            declared-time collocation in the meshes drto supports."""
            cd = next(iter(con.values())) if con.is_indexed() else con
            for v in identify_variables(cd.body, include_fixed=True):
                dv = v.parent_component()
                if isinstance(dv, DerivativeVar):
                    return time in dv.get_continuousset_list()
            return True

        alg_cons = []
        for con in model.component_objects(Constraint, active=True):
            # pyomo.dae artifacts: collocation equations ('_disc_') and the
            # Legendre continuity equations ('_cont_eq'); the segment builds
            # its own discretization with its own continuity
            if con in declared_cons:
                continue
            if (
                "_disc_" in con.local_name or con.local_name.endswith("_cont_eq")
            ) and _time_artifact(con):
                continue
            pos, _ = _time_index(con, time)
            if pos is None:
                continue
            alg_cons.append(con)

        algebraic = ComponentSet()
        block_alg = {}  # (Block component, member-local name) -> own subsets

        def _block_key(v, where=None):
            """(B, other-coords, t, local_name) for data reached through a
            time-indexed Block member, or None. A Block family may carry
            indices besides time (an IDAES stage or a spatial node); each
            non-time combination is its own family, replicated separately.
            With ``where`` set, unsupported shapes raise; without it they
            return None (the guard's silent mode)."""
            comp = v.parent_component()
            first = comp.parent_block()
            bd, found = first, None
            while bd is not None and bd is not model:
                pc = bd.parent_component()
                bpos, bsubs = _time_index(pc, time)
                if bpos is not None:
                    if found is not None:
                        if where:
                            raise ValueError(
                                f"drto: infinite_horizon cannot replicate "
                                f"'{where}': '{v.name}' sits inside a "
                                f"time-indexed Block nested in another "
                                f"time-indexed Block ('{pc.name}'), which "
                                f"is not supported."
                            )
                        return None
                    if bd is not first:
                        if where:
                            raise ValueError(
                                f"drto: infinite_horizon cannot replicate "
                                f"'{where}': '{v.name}' is not a direct "
                                f"child of the time-indexed Block member "
                                f"'{bd.name}', which is not supported."
                            )
                        return None
                    bo, bt = _split_index(bd.index(), bpos, len(bsubs))
                    found = (pc, bo, bt, comp.local_name)
                bd = bd.parent_block()
            return found

        def _bpos(B):
            return _time_index(B, time)[0]

        def _bname(bo):
            """A component-name fragment for a Block's non-time coords."""
            return "".join(f"_{str(x).replace('.', 'p')}" for x in bo)

        def _scan(expr, t_rep, where):
            """Validate a template; collect the algebraic components."""
            for v in identify_variables(expr, include_fixed=True):
                comp = v.parent_component()
                if isinstance(
                    comp, DerivativeVar
                ) and comp.get_continuousset_list() == [time]:
                    # a declared state's time derivative is allowed: the
                    # replication maps it to the segment derivative with
                    # the dilation factor, the same rewrite the dynamics
                    # get (an index-reduced energy balance is the real case).
                    # A derivative over another ContinuousSet (a spatial
                    # axis) is ordinary algebra: copied per member, closed
                    # by its replicated discretization equation
                    if id(comp.get_state_var()) not in covered:
                        raise ValueError(
                            f"drto: infinite_horizon cannot replicate "
                            f"'{where}': it references the derivative "
                            f"'{v.name}', which is not a declared state's "
                            f"derivative with respect to the declared time "
                            f"set."
                        )
                pos, subs = _time_index(comp, time)
                if pos is None:
                    bb = _block_key(v, where)
                    if bb is None:
                        continue  # time-invariant: shared as-is
                    B, bo, tb, lname = bb
                    if tb != t_rep:
                        raise ValueError(
                            f"drto: infinite_horizon cannot replicate "
                            f"'{where}': it references '{v.name}' away from "
                            f"the constraint's own time point."
                        )
                    if id(v) in disturbance_data:
                        disturbed.add(disturbance_data[id(v)])
                    elif id(v) in control_data:
                        # the declared control's segment copy serves; a
                        # member family here would shadow it in the
                        # expression map and orphan the control copy
                        pass
                    elif (B, bo, lname) not in block_alg:
                        block_alg[(B, bo, lname)] = (
                            list(comp.index_set().subsets())
                            if comp.is_indexed()
                            else []
                        )
                    continue
                _, t = _split_index(v.index(), pos, len(subs))
                if t != t_rep:
                    raise ValueError(
                        f"drto: infinite_horizon cannot replicate "
                        f"'{where}': it references '{v.name}' away from "
                        f"the constraint's own time point."
                    )
                if id(v) in disturbance_data:
                    disturbed.add(disturbance_data[id(v)])
                elif id(v) in state_member or id(v) in control_data:
                    pass  # routed to the declared component's copy
                elif isinstance(comp, DerivativeVar):
                    # an algebraic entry's derivative copies entry by
                    # entry; a declared state's derivative maps to the
                    # dilated segment derivative
                    xvd = comp.get_state_var()[v.index()]
                    if id(xvd) not in state_member:
                        o2, _t2 = _split_index(v.index(), pos, len(subs))
                        flat_partial.setdefault(comp, set()).add(o2)
                elif id(comp) in covered:
                    # only some entries of this Var are states: the
                    # algebraic entry copies alone, not the whole Var
                    o2, _t2 = _split_index(v.index(), pos, len(subs))
                    flat_partial.setdefault(comp, set()).add(o2)
                elif comp not in states_set and comp not in controls_set:
                    algebraic.add(comp)

        dyn_reps = {}
        dyn_residue = {}
        for con in dynamics:
            entries = {}
            residue = {}
            for o, (t_rep, cd) in _representatives(con).items():
                deriv_side, coeff, rhs = _dynamics_sides(cd, time, "infinite_horizon")
                zvd = deriv_side.parent_component().get_state_var()[deriv_side.index()]
                hit = state_member.get(id(zvd))
                if hit is None:
                    # a balance differentiating an entry that is not a
                    # declared state is replicated as written, an algebraic
                    # equation determining that entry's derivative copy
                    _scan(cd.expr, t_rep, cd.name)
                    residue[o] = (cd.expr, t_rep)
                    continue
                z, zo = hit
                _scan(rhs, t_rep, cd.name)
                if coeff is not None:
                    _scan(coeff, t_rep, cd.name)
                entries[o] = (z, zo, rhs, t_rep, coeff)
            dyn_reps[con] = entries
            if residue:
                dyn_residue[con] = residue

        alg_reps = {}
        for con in alg_cons:
            entries = {}
            for o, (t_rep, cd) in _representatives(con).items():
                _scan(cd.expr, t_rep, cd.name)
                entries[o] = (cd.expr, t_rep)
            alg_reps[con] = entries

        # --- member-internal constraint families: equations living inside
        # the members of every discovered time-indexed Block replicate like
        # algebraic equations. Scanning them can discover further Blocks, so
        # collection runs to a fixpoint. -------------------------------
        bcons = {}
        bexamined = set()

        def _collect_block_cons():
            grew = False
            for B, bo in {(B, bo) for (B, bo, _) in block_alg}:
                if (B, bo) in bexamined:
                    continue
                bexamined.add((B, bo))
                grew = True
                rep_t = t_end
                member = B[_join_index(bo, rep_t, _bpos(B))]
                for c in member.component_objects(Constraint, active=True):
                    entries = {}
                    if c.is_indexed():
                        for o, ccd in c.items():
                            o = o if isinstance(o, tuple) else (o,)
                            entries[o] = (ccd.expr, rep_t)
                    else:
                        entries[()] = (c.expr, rep_t)
                    for o, (e, tr) in entries.items():
                        _scan(e, tr, f"{member.name}.{c.local_name}")
                    bcons[(B, bo, c.local_name)] = (
                        list(c.index_set().subsets()) if c.is_indexed() else [],
                        entries,
                    )
            return grew

        cd = next(iter(stage_con.values())) if stage_con.is_indexed() else stage_con
        t_rep_cost = cd.index()
        cost_side, psi = _side_matching(
            cd, _is_var_member, "infinite_horizon", "the cost variable"
        )
        _scan(psi, t_rep_cost, cd.name)
        cost_var = cost_side.parent_component()

        while _collect_block_cons():
            pass

        # every variable copied to the segment must have at least one
        # replicated equation involving it; a variable with none would be
        # free there, and the solver would exploit it silently
        defined = ComponentSet()
        bdefined = set()

        def _note_defined(expr, flat_too=True):
            for v in identify_variables(expr, include_fixed=True):
                if flat_too:
                    defined.add(v.parent_component())
                bb = _block_key(v)
                if bb is not None:
                    bdefined.add((bb[0], bb[1], bb[3]))

        for entries in alg_reps.values():
            for expr, _ in entries.values():
                _note_defined(expr)
        for _, entries in bcons.values():
            for expr, _ in entries.values():
                _note_defined(expr)
        # block-borne variables are additionally credited by the dilated
        # dynamics and the tail integrand: a member arrives with its whole
        # point-wise subsystem, where closure through the balances is the
        # norm (an outlet flow determined jointly by the component
        # balances). The flat guard stays strict, since a flat variable
        # appearing ONLY in the dynamics is a free input the solver would
        # exploit, which is the smell it exists to catch.
        for entries in dyn_reps.values():
            for _, _, rhs, _, coeff in entries.values():
                _note_defined(rhs, flat_too=False)
                if coeff is not None:
                    _note_defined(coeff, flat_too=False)
        # an algebraic entry's balance, replicated as written, defines
        # that entry's derivative copy
        for residue in dyn_residue.values():
            for expr, _t in residue.values():
                _note_defined(expr)
        _note_defined(psi, flat_too=False)
        for key in block_alg:
            if key not in bdefined:
                B, bo, lname = key
                at = "[t]" if not bo else f"[t, {', '.join(map(str, bo))}]"
                raise ValueError(
                    f"drto: infinite_horizon copies '{B.name}{at}.{lname}' to "
                    f"the segment, but no replicated equation involves it; "
                    f"its defining equation must live in the Block members "
                    f"or be indexed by the declared time set '{time.name}'."
                )
        for comp in algebraic:
            if comp not in defined:
                raise ValueError(
                    f"drto: infinite_horizon copies '{comp.name}' to the "
                    f"segment, but no replicated equation involves it; its "
                    f"defining equation must be indexed by the declared "
                    f"time set '{time.name}'."
                )
        # the same strictness for partially copied containers: their
        # members appearing only in the dilated dynamics would be free
        # inputs on the tail (a spatial flow derivative whose
        # discretization row was mistaken for a time artifact)
        for pcomp in flat_partial:
            if pcomp not in defined:
                raise ValueError(
                    f"drto: infinite_horizon copies members of "
                    f"'{pcomp.name}' to the segment, but no replicated "
                    f"equation involves them; their defining equations "
                    f"must be indexed by the declared time set "
                    f"'{time.name}'."
                )

        # --- the segment block ----------------------------------------
        b = Block(concrete=True)
        model.add_component(_BLOCK_NAME, b)
        b.tau = ContinuousSet(bounds=(0, 1))
        b.gamma = Param(initialize=1.0, mutable=True)
        b.beta = Param(initialize=config.beta, mutable=True)

        def _fresh(base, full):
            """A segment-unique component name: the local name, or the
            sanitized full path when two model components share it (the
            two settlers of a mixer-settler both carry ``_flow_terms``)."""
            if b.component(base) is None:
                return base
            alt = "".join(
                ch if ch.isalnum() or ch == "_" else "_" for ch in full
            ).strip("_")
            while b.component(alt) is not None:
                alt += "_"
            return alt

        seg = {}
        for comp in list(states) + list(controls):
            _, others = _layout(comp)
            u = _units_of(comp)
            v = Var(*others, b.tau, units=u) if others else Var(b.tau, units=u)
            b.add_component(comp.local_name, v)
            seg[comp] = v
        derivs = {}
        for z in states:
            dv = DerivativeVar(seg[z], wrt=b.tau, units=_units_of(z))
            b.add_component(z.local_name + "_dtau", dv)
            derivs[z] = dv

        bseg = {}

        def _bcombos(bsubs):
            if not bsubs:
                return [()]
            return [
                (i if isinstance(i, tuple) else (i,)) for i in itertools.product(*bsubs)
            ]

        def _bmember(B, bo, lname, o, t):
            c = getattr(B[_join_index(bo, t, _bpos(B))], lname)
            if not o:
                return c
            return c[o if len(o) > 1 else o[0]]

        def _bseg_at(key, o, s):
            v = bseg[key]
            return v[tuple(o) + (s,)] if o else v[s]

        # algebraic entries of partly declared Vars copy entry by entry,
        # over a set of the referenced combos, not the container wholesale
        pseg = {}

        def _pseg_at(comp, o, s):
            v = pseg[comp]
            return v[tuple(o) + (s,)] if o else v[s]

        def _seg_at(comp, o, s):
            v = seg[comp]
            return v[tuple(o) + (s,)] if o else v[s]

        # --- disturbance copies fixed at their constants: the tail
        # continues under nominal disturbance unless told otherwise ------
        if config.disturbances:
            declared_names = {d.name: d for d in disturbances}
            for key, val in config.disturbances.items():
                name = key if isinstance(key, str) else key.name
                if name not in declared_names:
                    raise ValueError(
                        f"drto: infinite_horizon got a disturbance value for "
                        f"'{name}', which is not a declared disturbance; "
                        f"declared: {', '.join(declared_names) or '(none)'}."
                    )
        dist_values = {}
        for d in disturbed:
            key = next(
                (
                    k
                    for k in (config.disturbances or {})
                    if (k if isinstance(k, str) else k.name) == d.name
                ),
                None,
            )
            val = (config.disturbances or {}).get(key, 0.0) if key else 0.0
            if not isinstance(val, dict):
                val = float(val)
            dist_values[d] = val

        # the model DerivativeVars over covered states, for mapping
        # derivative references inside replicated equations per member
        deriv_infos = []
        for dv in model.component_objects(Var, active=True):
            if (
                isinstance(dv, DerivativeVar)
                and id(dv.get_state_var()) in covered
                and dv.get_continuousset_list() == [time]
            ):
                dpos, dsubs = _time_index(dv, time)
                deriv_infos.append((dv, dpos, len(dsubs)))

        _emaps = {}

        def _emap(t_rep, s):
            """Model members at ``t_rep`` mapped to segment members at
            ``s``, cached: the map depends only on the two time points,
            never on the member the replication rule is building. A state
            derivative maps to the segment derivative with the dilation
            factor, the same rewrite the dilated dynamics apply."""
            key = (t_rep, s)
            if key not in _emaps:
                mmap = {}
                for comp in seg:
                    for o in _combos(comp):
                        mmap[id(_member(comp, o, t_rep))] = _seg_at(comp, o, s)
                for (B, bo, lname), bsubs in block_alg.items():
                    for o in _bcombos(bsubs):
                        mmap[id(_bmember(B, bo, lname, o, t_rep))] = _bseg_at(
                            (B, bo, lname), o, s
                        )
                for dv, dpos, dn in deriv_infos:
                    for idx, dvd in dv.items():
                        do, dt = _split_index(idx, dpos, dn)
                        if dt != t_rep:
                            continue
                        hit = state_member.get(id(dv.get_state_var()[idx]))
                        if hit is None:
                            continue  # not a state: its entry loop maps it
                        z, zo = hit
                        dseg = derivs[z]
                        seg_deriv = dseg[tuple(zo) + (s,)] if zo else dseg[s]
                        mmap[id(dvd)] = b.gamma * (1 - s**2) * seg_deriv
                for pcomp, combos in flat_partial.items():
                    ppos, _psubs = _time_index(pcomp, time)
                    for o in combos:
                        mmap[id(pcomp[_join_index(o, t_rep, ppos)])] = _pseg_at(
                            pcomp, o, s
                        )
                _emaps[key] = mmap
            return _emaps[key]

        # --- link the segment to the end of the horizon ------------------
        links = {}
        for z in states:
            pos, others = _layout(z)

            def link_rule(blk, *o, _z=z):
                o = tuple(v for v in o if v is not None)  # scalar rules get None
                return _seg_at(_z, o, 0) == _member(_z, o, t_end)

            links[z] = (
                Constraint(*others, rule=link_rule)
                if others
                else Constraint(rule=link_rule)
            )
            b.add_component(z.local_name + "_link", links[z])

        # --- discretize the segment: Gauss-Legendre only, no collocation
        # equation may sit at the singular endpoint tau = 1 ---
        TransformationFactory("dae.collocation").apply_to(
            b, wrt=b.tau, nfe=config.nfe, ncp=config.ncp, scheme="LAGRANGE-LEGENDRE"
        )

        # --- the algebraic copies live on the interior collocation
        # points only: every point that exists is one a replicated
        # equation determines (gh #32). Created here, after the
        # discretization, when the interior points exist ---
        _fe = set(b.tau.get_finite_elements())
        b.tau_i = Set(initialize=[p for p in b.tau if p not in _fe], ordered=True)
        for comp in list(algebraic) + list(disturbed):
            _, others = _layout(comp)
            v = (
                Var(*others, b.tau_i, units=_units_of(comp))
                if others
                else Var(b.tau_i, units=_units_of(comp))
            )
            b.add_component(_fresh(comp.local_name, comp.name), v)
            seg[comp] = v
        for (B, bo, lname), bsubs in block_alg.items():
            u_b = _units_of(getattr(B[_join_index(bo, t_end, _bpos(B))], lname))
            v = Var(*bsubs, b.tau_i, units=u_b) if bsubs else Var(b.tau_i, units=u_b)
            b.add_component(
                _fresh(
                    f"{B.local_name}{_bname(bo)}_{lname}",
                    f"{B.name}{_bname(bo)}_{lname}",
                ),
                v,
            )
            bseg[(B, bo, lname)] = v
        for pcomp, combos in flat_partial.items():
            u_p = _units_of(pcomp)
            pset = sorted(combos)
            if pset == [()]:
                v = Var(b.tau_i, units=u_p)
            else:
                cset = Set(initialize=pset, dimen=len(pset[0]))
                v = Var(cset, b.tau_i, units=u_p)
            b.add_component(
                _fresh(f"{pcomp.local_name}_members", f"{pcomp.name}_members"), v
            )
            pseg[pcomp] = v

        # --- dilated dynamics at interior collocation points (eq. 25) ---
        dyn_copies = {}
        for con in dynamics:
            pos, subs = _time_index(con, time)
            others = [s_ for n, s_ in enumerate(subs) if n != pos]

            def dyn_rule(blk, *idx, _entries=dyn_reps[con]):
                s = idx[-1]
                o = tuple(idx[:-1])
                if s in blk.tau.get_finite_elements() or o not in _entries:
                    return Constraint.Skip
                z, zo, rhs, t_rep, coeff = _entries[o]
                dv = derivs[z]
                deriv = dv[tuple(zo) + (s,)] if zo else dv[s]
                lhs = blk.gamma * (1 - s**2) * deriv
                if coeff is not None:
                    # the written side's fixed coefficient (an IDAES CV1D's
                    # length) multiplies the dilated derivative unchanged
                    lhs = replace_expressions(coeff, _emap(t_rep, s)) * lhs
                return lhs == replace_expressions(rhs, _emap(t_rep, s))

            dyn_copies[con] = Constraint(*others, b.tau_i, rule=dyn_rule)
            b.add_component(_fresh(con.local_name, con.name), dyn_copies[con])

        block_rows = {}
        # --- member-internal equations, replicated at the interior
        # collocation points exactly like flat algebraic equations ------
        for (B, bo, cname), (csubs, entries) in bcons.items():

            def bcon_rule(blk, *idx, _entries=entries):
                sp = idx[-1]
                o = tuple(idx[:-1])
                if sp in blk.tau.get_finite_elements() or o not in _entries:
                    return Constraint.Skip
                expr, tr = _entries[o]
                return replace_expressions(expr, _emap(tr, sp))

            block_rows[(B, bo, cname)] = Constraint(*csubs, b.tau_i, rule=bcon_rule)
            b.add_component(
                _fresh(
                    f"{B.local_name}{_bname(bo)}_{cname}",
                    f"{B.name}{_bname(bo)}_{cname}",
                ),
                block_rows[(B, bo, cname)],
            )

        alg_rows = {}
        # --- algebraic equations, replicated as written at the interior
        # collocation points, where the dilated dynamics reference their
        # variables; no boundary or endpoint values ----------------------
        for con in alg_cons:
            pos, subs = _time_index(con, time)
            others = [s_ for n, s_ in enumerate(subs) if n != pos]

            def alg_rule(blk, *idx, _entries=alg_reps[con]):
                s = idx[-1]
                o = tuple(idx[:-1])
                if s in blk.tau.get_finite_elements() or o not in _entries:
                    return Constraint.Skip
                expr, t_rep = _entries[o]
                return replace_expressions(expr, _emap(t_rep, s))

            alg_rows[con] = Constraint(*others, b.tau_i, rule=alg_rule)
            b.add_component(_fresh(con.local_name, con.name), alg_rows[con])

        # --- the algebraic entries' balances, replicated as written
        # at the interior collocation points like algebraic equations -----
        residues = {}
        for con, residue in dyn_residue.items():
            pos, subs = _time_index(con, time)
            others = [s_ for n, s_ in enumerate(subs) if n != pos]

            def res_rule(blk, *idx, _entries=residue):
                sp = idx[-1]
                o = tuple(idx[:-1])
                if sp in blk.tau.get_finite_elements() or o not in _entries:
                    return Constraint.Skip
                expr, tr = _entries[o]
                return replace_expressions(expr, _emap(tr, sp))

            residues[con] = Constraint(*others, b.tau_i, rule=res_rule)
            b.add_component(
                _fresh(con.local_name + "_residue", con.name + "_residue"),
                residues[con],
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
        def _own_pts(container):
            return sorted({ci[-1] if isinstance(ci, tuple) else ci for ci in container})

        for comp in seg:
            for o in _combos(comp):
                src = _member(comp, o, t_end)
                for s in _own_pts(seg[comp]):
                    v = _seg_at(comp, o, s)
                    v.setlb(src.lb)
                    v.setub(src.ub)
                    v.set_value(src.value)
                    # fixed means fixed: a given input's copy holds the
                    # horizon-end value on the tail
                    if comp in algebraic and src.fixed:
                        v.fix()

        for (B, bo, lname), bsubs in block_alg.items():
            for o in _bcombos(bsubs):
                src = _bmember(B, bo, lname, o, t_end)
                for sp in _own_pts(bseg[(B, bo, lname)]):
                    v = _bseg_at((B, bo, lname), o, sp)
                    v.setlb(src.lb)
                    v.setub(src.ub)
                    v.set_value(src.value)
                    if src.fixed:
                        v.fix()

        for pcomp, combos in flat_partial.items():
            ppos, _psubs = _time_index(pcomp, time)
            for o in combos:
                src = pcomp[_join_index(o, t_end, ppos)]
                for sp in _own_pts(pseg[pcomp]):
                    v = _pseg_at(pcomp, o, sp)
                    v.setlb(src.lb)
                    v.setub(src.ub)
                    v.set_value(src.value)
                    if src.fixed:
                        v.fix()

        # --- disturbance copies fixed at their realization, after the mesh
        # exists (fixing at construction touches only the endpoints) and
        # after the horizon-end init, which would overwrite the values. A
        # scalar holds one constant everywhere; a dict gives one constant
        # per non-time index (a multi-component feed) ---
        for d in disturbed:
            val = dist_values[d]
            for o in _combos(d):
                if isinstance(val, dict):
                    ko = o if len(o) > 1 else o[0]
                    if ko not in val:
                        raise ValueError(
                            f"drto: infinite_horizon disturbances for "
                            f"'{d.name}': no value for index {ko!r}."
                        )
                    vo = float(val[ko])
                else:
                    vo = val
                for sp in _own_pts(seg[d]):
                    _seg_at(d, o, sp).fix(vo)

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

        # --- segment control profiles: applied now, so raw unparameterized
        # copies are never left on the segment; one call, one pass over
        # the block ---
        if controls:
            TransformationFactory("cvp.parameterize").apply_to(
                b, var=[seg[u] for u in controls], contset=b.tau, profile=config.profile
            )
            for u in controls:
                # cvp swapped each copy under its own name; follow the swap
                seg[u] = b.component(seg[u].local_name)

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

        # --- the terminal endpoint pin (Dinh et al. 2025): constrain the
        # extrapolated endpoint z(tau=1) to the steady state. The endpoint is
        # the Legendre continuity extrapolation, the paper's evaluated z_e; the
        # derivative there is undefined, so the pin is on the state value. It
        # references only states (cvp never replaces those), so it is order-free
        # relative to the control parameterization above ---
        pins = {}
        if config.terminal != "none":
            # 'soft': the L1-relaxed endpoint of eq 36, split-nonneg slacks
            tau_end = b.tau.last()

            def _tgt(z, o):
                p = ss_target[id(z)]
                return p[tuple(o)] if o else p

            b.mu = Param(initialize=config.mu, mutable=True)
            pin_terms = []
            for z in states:
                _, others = _layout(z)
                zu = _units_of(z)
                up = (
                    Var(*others, domain=NonNegativeReals, units=zu)
                    if others
                    else Var(domain=NonNegativeReals, units=zu)
                )
                lo = (
                    Var(*others, domain=NonNegativeReals, units=zu)
                    if others
                    else Var(domain=NonNegativeReals, units=zu)
                )
                b.add_component(z.local_name + "_pin_up", up)
                b.add_component(z.local_name + "_pin_lo", lo)

                def soft_rule(blk, *o, _z=z, _up=up, _lo=lo):
                    o = tuple(v for v in o if v is not None)
                    eu = _up[tuple(o)] if o else _up
                    el = _lo[tuple(o)] if o else _lo
                    return _seg_at(_z, o, tau_end) + eu - el == _tgt(_z, o)

                pin_eq = (
                    Constraint(*others, rule=soft_rule)
                    if others
                    else Constraint(rule=soft_rule)
                )
                b.add_component(z.local_name + "_pin_eq", pin_eq)
                pins[z] = (pin_eq, up, lo)
                for o in _combos(z):
                    eu = up[tuple(o)] if o else up
                    el = lo[tuple(o)] if o else lo
                    # the linear part is the exact L1 pin; the quadratic
                    # part gives the pin a strictly convex penalty at the
                    # kink, so its multipliers are unique and consecutive
                    # solves agree on them (gh #37). Zero slack stays
                    # optimal whenever the pin can hold: the quadratic's
                    # gradient vanishes there
                    pin_terms.append((eu, b.mu))
                    pin_terms.append((el, b.mu))
                    pin_terms.append((eu**2, b.mu))
                    pin_terms.append((el**2, b.mu))
            # a separate cost_group keeps liveness independent of the tail
            reg.record_declaration("cost_group", b, terms=tuple(pin_terms))

        # the tail integral IS the cost-to-go, so a declared terminal cost
        # would double-count: deactivate it (build_objective's liveness rule
        # then drops its term) and record the outcome
        terminal = None
        for comp in reg.components("tracking_terminal_cost"):
            if comp.active:
                comp.deactivate()
                terminal = comp.name

        # --- record the segment pairing on the registry: which tail
        # component belongs to which declaration. Internal bookkeeping,
        # never rendered; the consumers read this instead of rebuilding
        # component names (gh #27) ---
        for z in states:
            pin_eq, up, lo = pins.get(z) or (None, None, None)
            reg._record_segment(
                "state",
                z,
                copy=seg[z],
                derivative=derivs[z],
                disc=b.component(derivs[z].local_name + "_disc_eq"),
                continuity=b.component(seg[z].local_name + "_tau_cont_eq"),
                link=links[z],
                pin=pin_eq,
                pin_up=up,
                pin_lo=lo,
            )
        for u in controls:
            reg._record_segment("control", u, copy=seg[u])
        for con in dynamics:
            reg._record_segment(
                "dynamics", con, copy=dyn_copies[con], residue=residues.get(con)
            )
        for comp in list(algebraic) + list(disturbed):
            reg._record_segment("algebraic", comp, copy=seg[comp])
        reg._record_segment("algebraic", cost_var, copy=seg_cost)
        for (B, _bo, lname), v in bseg.items():
            reg._record_segment("block_member", B, member=lname, copy=v)
        for pcomp, v in pseg.items():
            reg._record_segment("packed_member", pcomp, copy=v)
        for con, row in alg_rows.items():
            reg._record_segment("algebraic_row", con, copy=row)
        for (B, _bo, cname), row in block_rows.items():
            reg._record_segment("block_row", B, member=cname, copy=row)
        reg._record_segment("segment", b, gamma=b.gamma)

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
                {
                    "blocks": f"{len({B for (B, _, _) in block_alg})} "
                    f"time-indexed Block(s): {len(block_alg)} components,"
                    f" {len(bcons)} equation families replicated"
                }
                if block_alg
                else {}
            ),
            **(
                {
                    "disturbances": ", ".join(
                        f"{d.name} fixed at "
                        + (
                            "per-index values"
                            if isinstance(dist_values[d], dict)
                            else str(dist_values[d])
                        )
                        for d in disturbed
                    )
                }
                if disturbed
                else {}
            ),
            **(
                {
                    "partial": f"{len(flat_partial)} partially declared "
                    f"container(s) copied per member"
                }
                if flat_partial
                else {}
            ),
            **(
                {"terminal_cost": f"{terminal} deactivated (the tail owns it)"}
                if terminal
                else {}
            ),
            **(
                {
                    "terminal": (
                        f"{config.terminal} pin z(tau=1)=z_s on {len(states)} "
                        f"state{'' if len(states) == 1 else 's'}, mu={config.mu}"
                    )
                }
                if config.terminal != "none"
                else {}
            ),
        )
