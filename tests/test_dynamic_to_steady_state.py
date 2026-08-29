# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 005: the steady-state reduction."""
import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port

import drto
from test_declarations import base_model, declared_model
from test_estimation_declarations import est_declared
from test_infinite_horizon import (
    _dof,
    block_model,
    indexed_model,
    packed_model,
    ready_model,
)

ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")

SS = "drto.dynamic_to_steady_state"


def test_a_fixed_input_stays_fixed_through_the_reduction():
    # the collapse rebuilds each time-indexed Var; the fixed flag is carried
    # through, not dropped
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    for vd in m.u.values():
        vd.set_value(0.5)
        vd.fix()
    pyo.TransformationFactory(SS).apply_to(m)
    assert not m.u.is_indexed()
    assert m.u.fixed and pyo.value(m.u) == 0.5


def test_requires_the_declarations():
    m = base_model()
    drto.horizon(m.t)
    with pytest.raises(ValueError, match="Missing: state, dynamics"):
        pyo.TransformationFactory(SS).apply_to(m)


def snapshot(m):
    return sorted(
        f"{c.name}: {c.expr}"
        for c in m.component_data_objects(pyo.Constraint, active=True)
    )


def test_a_discretized_model_reduces_to_the_same_steady_system():
    plain = pyo.TransformationFactory(SS).create_using(declared_model())
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    ss = pyo.TransformationFactory(SS).create_using(m)
    assert snapshot(ss) == snapshot(plain)
    # the discretization artifacts are discarded, not collapsed
    assert not any("_disc_" in c.name for c in ss.component_objects(pyo.Constraint))
    rec = [r for r in drto.info(ss).transformations if r["name"] == SS][0]
    assert "discretization artifacts" in rec["outcome"]["discarded"]


def test_applied_drto_transforms_still_error():
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    with pytest.raises(ValueError, match="before any drto"):
        pyo.TransformationFactory(SS).apply_to(m)


def test_collapse_structure():
    m = declared_model()
    ss = pyo.TransformationFactory(SS).create_using(m)
    # time collapsed: every Var and Constraint single-point
    assert not ss.z.is_indexed() and not ss.u.is_indexed()
    assert not ss.cost.is_indexed()
    assert ss.component("t") is None
    assert not ss.ode.is_indexed()
    assert not ss.stage.is_indexed()
    # initial condition removed; the source dynamic model is untouched
    assert ss.component("init") is None
    assert m.z.is_indexed() and m.component("init") is not None
    # the derivative stays, collapsed to a point and fixed at zero: the
    # declared dynamics still read as written
    assert "dzdt" in str(ss.ode.expr)
    assert not ss.dzdt.is_indexed()
    assert ss.dzdt.fixed and pyo.value(ss.dzdt) == 0


def test_registry_reflects_the_reduction():
    m = declared_model()
    ss = pyo.TransformationFactory(SS).create_using(m)
    reg = drto.info(ss)
    assert reg.components("state") == (ss.z,)
    assert reg.components("control") == (ss.u,)
    assert not reg.has_declaration("horizon")
    assert not reg.has_declaration("initial_condition")
    assert reg.has_transformation(SS)
    # the steady-state pairing follows the collapsed state
    (pair,) = reg.declarations("steady_state")
    assert pair["of"] is ss.z
    # a single-point control has no profile
    (control,) = reg.declarations("control")
    assert "profile" not in control


@needs_ipopt
def test_steady_solve_matches_the_fixed_point():
    m = declared_model()
    ss = pyo.TransformationFactory(SS).create_using(m)
    drto.build_objective(ss)
    r = drto.scaling.solver_by_name("ipopt").solve(ss)
    assert drto.scaling.solved_to_optimality(r)
    # dz/dt = -z + u at rest with the tracking cost: z = u = the setpoint
    assert pyo.value(ss.z) == pytest.approx(0.5, abs=1e-6)
    assert pyo.value(ss.u) == pytest.approx(0.5, abs=1e-6)


def test_multi_time_reference_errors():
    m = declared_model()

    @m.Constraint()
    def span(mm):
        return mm.z[0] == mm.z[mm.t.last()]

    with pytest.raises(ValueError, match="more than one time point"):
        pyo.TransformationFactory(SS).apply_to(m)


def test_derivative_in_an_algebraic_equation_is_fixed_at_zero():
    m = base_model()
    m.w = pyo.Var(m.t, initialize=0.5)

    @m.Constraint(m.t)
    def w_def(mm, t):
        return mm.w[t] == 0.5 * (mm.z[t] + mm.u[t]) + 0.1 * mm.dzdt[t]

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    pyo.TransformationFactory(SS).apply_to(m)
    # the equation keeps its form; its derivative is fixed at zero, so it is
    # the quasi-static relation without the expression being rewritten
    assert "dzdt" in str(m.w_def.expr)
    assert m.dzdt.fixed and pyo.value(m.dzdt) == 0
    assert not m.w_def.is_indexed()


def test_indexed_state_collapses_per_member():
    m = indexed_model()
    ss = pyo.TransformationFactory(SS).create_using(m)
    assert len(ss.x) == 2  # one member per non-time index
    assert len(ss.ode) == 2


def test_apply_to_reduces_in_place():
    m = declared_model()
    pyo.TransformationFactory(SS).apply_to(m)
    assert not m.z.is_indexed()
    assert m.component("t") is None


# ── time-indexed Blocks in the reduction (feature 005) ───────────────────────


def test_block_family_collapses_to_the_steady_member():
    m = block_model()
    pyo.TransformationFactory(SS).apply_to(m)
    assert list(m.props.keys()) == [0]
    # the surviving member is untouched: its variable and equation as written
    assert m.props[0].y.value == 0.4
    assert m.props[0].gain.active
    rec = [r for r in drto.info(m).transformations if r["name"] == SS][0]
    assert "1 time-indexed Block" in rec["outcome"]["blocks"]


def test_no_constraint_reaches_a_removed_member():
    from pyomo.core.expr.visitor import identify_variables

    m = block_model()
    pyo.TransformationFactory(SS).apply_to(m)
    live = set(id(vd) for vd in m.props[0].component_data_objects(pyo.Var))
    for c in m.component_data_objects(pyo.Constraint, active=True):
        for v in identify_variables(c.body, include_fixed=True):
            blk = v.parent_block()
            while blk is not None and blk is not m:
                if blk.parent_component() is m.props:
                    assert id(v) in live, c.name
                blk = blk.parent_block()


def test_block_reduction_dof_matches_flat():
    # the same physics through a Block(t) member and through a flat Var
    # reduce to the same freedom
    dofs = {}
    for flat in (True, False):
        m = block_model(flat=flat)
        pyo.TransformationFactory(SS).apply_to(m)
        dofs[flat] = _dof(m)
    assert dofs[False] == dofs[True]


def test_references_collapse_to_views():
    # the Port idiom: a Reference into members follows the surviving
    # member, a Reference onto a flat Var follows the collapsed Var,
    # and the Port keeps pointing at its referents
    m = block_model()
    m.yref = pyo.Reference(m.props[:].y)
    m.zref = pyo.Reference(m.z[:])
    m.big = pyo.Var(m.t, [1, 2], initialize=0.0)
    m.bigref = pyo.Reference(m.big[:, :])
    m.w = pyo.Var(initialize=1.5)
    m.wref = pyo.Reference(m.w)
    m.port = Port()
    m.port.add(m.yref, "y")
    pyo.TransformationFactory(SS).apply_to(m)
    assert m.yref.is_reference()
    assert next(iter(m.yref.values())) is m.props[0].y
    assert m.zref.is_reference()
    assert next(iter(m.zref.values())) is m.z
    assert m.port.vars["y"] is m.yref
    # an extra index survives the collapse; a time-invariant Reference
    # is untouched
    assert m.bigref[1] is m.big[1] and m.bigref[2] is m.big[2]
    assert next(iter(m.wref.values())) is m.w


def test_block_under_a_plain_container_collapses():
    # the IDAES shape: the time-indexed Block sits levels below the model,
    # under plain containers; a time-invariant Block is shared as-is
    m = block_model()
    m.side = pyo.Block()
    m.side.sub = pyo.Block(
        m.t, rule=lambda b, t: setattr(b, "v", pyo.Var(initialize=t))
    )
    pyo.TransformationFactory(SS).apply_to(m)
    assert list(m.side.sub.keys()) == [0]
    assert m.side.active


def test_member_subset_state_derivatives_all_rest():
    # gh #20: every accumulation of a covered container is fixed at zero,
    # the algebraic entry's too. Steady state is steady for it,
    # which is what closes its row at the point (the water balance
    # determining the outlet flow on the IDAES CSTR).
    m = packed_model()
    pyo.TransformationFactory(SS).apply_to(m)
    assert m.dx["A"].fixed and pyo.value(m.dx["A"]) == 0
    assert m.dx["W"].fixed and pyo.value(m.dx["W"]) == 0


def test_nested_time_indexed_block_is_rejected():
    m = block_model(nested=True)
    with pytest.raises(ValueError, match="nested in another"):
        pyo.TransformationFactory(SS).apply_to(m)


@needs_ipopt
def test_block_reduction_reaches_the_fixed_point():
    m = block_model()
    pyo.TransformationFactory(SS).apply_to(m)
    drto.build_objective(m)
    r = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(r)
    # dz/dt = u - y at rest with y = 2z + 0.1q and u in [0, 1]: the
    # setpoint needs u = 1.3, so the optimum rides the bound, u = y = 1
    # and z = (1 - 0.3) / 2, the member equation carrying the physics
    assert pyo.value(m.u) == pytest.approx(1.0, abs=1e-6)
    assert pyo.value(m.props[0].y) == pytest.approx(1.0, abs=1e-6)
    assert pyo.value(m.z) == pytest.approx(0.35, abs=1e-6)


def test_a_scaled_derivative_side_reduces_like_the_bare_form():
    # ``2*dz/dt == u - z``: the derivative is recognized through the fixed
    # coefficient (gh #51), the reduction pins it at zero as usual
    from test_infinite_horizon import _first_order

    m = _first_order(scaled=True)
    pyo.TransformationFactory(SS).apply_to(m)
    assert not m.z.is_indexed()
    assert m.dzdt.fixed and pyo.value(m.dzdt) == 0


# ----------------------------------------------------------------------
# spatially distributed models (gh #54)
# ----------------------------------------------------------------------
def spatial_model(noise=False):
    """A 1D transport model with ``Block(t, x)`` members and a spatial
    derivative: the minimal shape of a 1D control volume. The transport
    balance is written past the inlet only, so the dynamics container is
    sparse, and the spatial discretization equations are real algebra
    the reduction must keep."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 4, 1))
    m.x = ContinuousSet(bounds=(0, 1))
    m.z = pyo.Var(m.t, m.x, initialize=0.2)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.dzdx = DerivativeVar(m.z, wrt=m.x)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    if noise:
        m.w = pyo.Var(m.t, initialize=0.0)

    def props_rule(blk, t, x):
        blk.y = pyo.Var(initialize=0.4)
        blk.gain = pyo.Constraint(expr=blk.y == 2.0 * blk.model().z[t, x])

    m.props = pyo.Block(m.t, m.x, rule=props_rule)

    @m.Constraint(m.t, m.x)
    def transport(mm, t, x):
        if x == mm.x.first():
            return pyo.Constraint.Skip
        rhs = -mm.dzdx[t, x] - mm.props[t, x].y
        if noise:
            rhs = rhs + mm.w[t]
        return mm.dzdt[t, x] == rhs

    @m.Constraint(m.t)
    def inlet(mm, t):
        return mm.z[t, mm.x.first()] == mm.u[t]

    interior = [x for x in m.x if x != m.x.first()]

    @m.Constraint(interior)
    def z_init(mm, x):
        return mm.z[0, x] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.transport)
    drto.control(m.u, profile="piecewise_constant")
    drto.initial_condition(m.z_init)
    if noise:
        drto.disturbance(m.w)
    pyo.TransformationFactory("dae.finite_difference").apply_to(
        m, wrt=m.x, nfe=4, scheme="BACKWARD"
    )
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def test_a_spatial_block_family_collapses_per_point():
    m = spatial_model()
    xs = sorted(m.x)
    pyo.TransformationFactory(SS).apply_to(m)
    assert sorted(m.props.keys()) == [(0, x) for x in xs]
    for x in xs:
        bd = m.props[0, x]
        assert bd.gain.active
    assert not m.z.is_indexed() or set(m.z.index_set()) == set(xs)


def test_the_spatial_discretization_equations_survive():
    m = spatial_model()
    pyo.TransformationFactory(SS).apply_to(m)
    names = [
        c.parent_component().local_name
        for c in m.component_data_objects(pyo.Constraint, active=True)
    ]
    assert names.count("dzdx_disc_eq") == 4
    assert all("dzdt_disc_eq" != n for n in names)
    assert all(not n.endswith("_cont_eq") for n in names)
    # every spatial derivative at an interior point stays a live unknown
    for x in [x for x in m.dzdx.index_set() if x != 0]:
        assert not m.dzdx[x].fixed


def test_a_sparse_dynamics_container_reduces_to_its_members():
    m = spatial_model()
    interior = [x for x in m.x if x != m.x.first()]
    pyo.TransformationFactory(SS).apply_to(m)
    assert sorted(m.transport.keys()) == interior
    # the time derivative rests at zero in every surviving member
    for x in interior:
        assert m.dzdt[x].fixed and pyo.value(m.dzdt[x]) == 0


def test_a_noise_carrying_balance_passes_the_guard():
    # the derivative side is a sum once the noise term rides the row;
    # the guard reads the row's variables, not its written shape
    m = spatial_model(noise=True)
    pyo.TransformationFactory(SS).apply_to(m)
    assert sorted(m.transport.keys()) == [x for x in m.x if x != 0]


def test_an_undeclared_differentiation_still_errors():
    m = spatial_model()
    m.v = pyo.Var(m.t, initialize=0.0)
    m.dv = DerivativeVar(m.v, wrt=m.t)

    @m.Constraint(m.t)
    def rogue(mm, t):
        return mm.dv[t] == -mm.v[t]

    reg = drto.info(m)
    reg._declarations["dynamics"].append(
        dict(reg.declarations("dynamics")[0], component=m.rogue)
    )
    with pytest.raises(ValueError, match="undeclared state"):
        pyo.TransformationFactory(SS).apply_to(m)


def test_an_empty_var_is_skipped():
    m = spatial_model()
    m.none = pyo.Var(m.t, [], initialize=0.0)
    pyo.TransformationFactory(SS).apply_to(m)
    assert m.component("none") is not None


@needs_ipopt
def test_the_steady_spatial_profile_solves():
    m = spatial_model()
    pyo.TransformationFactory(SS).apply_to(m)
    for vd in m.u.values() if m.u.is_indexed() else (m.u,):
        vd.fix(1.0)
    # the inlet spatial derivative sits in no equation of the reduced
    # model (backward differences start past the inlet): held for the
    # square solve
    m.dzdx[0].fix(0.0)
    r = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(r)
    # steady transport with first-order loss: the profile decays along
    # the vessel from the held inlet
    xs = sorted(m.x)
    vals = [pyo.value(m.z[x]) for x in xs]
    assert vals[0] == pytest.approx(1.0, abs=1e-6)
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_the_mixer_settler_stage_reduces_square():
    pytest.importorskip("prommis")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    try:
        from models import prommis_sx
    finally:
        sys.path.pop(0)
    from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    from pyomo.core.expr import identify_variables

    m = prommis_sx.build(N=1, h=0.25)
    sm = pyo.TransformationFactory(SS).create_using(m)
    reg = drto.info(sm)
    ms = sm.fs.ms
    msc = ms.mixer[1].unit.mscontactor
    aq, og = ms.aqueous_settler[1].unit, ms.organic_settler[1].unit
    # the dynamic model's inert first-instant data is free at steady state
    msc.aqueous_inherent_reaction_extent.unfix()
    msc.heterogeneous_reaction_extent.unfix()
    aq.inherent_reaction_extent.unfix()
    for blk in (msc.aqueous, msc.organic):
        for bd in blk.values():
            bd.flow_vol.unfix()
    for st in (aq, og):
        for bd in st.properties.values():
            bd.flow_vol.unfix()
    for kind in ("control", "disturbance"):
        for comp in reg.components(kind):
            for vd in comp.values() if comp.is_indexed() else (comp,):
                vd.fix()

    rows = [c for c in sm.component_data_objects(pyo.Constraint, active=True)]
    incident = set()
    for c in rows:
        for v in identify_variables(c.expr, include_fixed=False):
            incident.add(id(v))
    free = [
        v
        for v in sm.component_data_objects(pyo.Var)
        if not v.fixed and id(v) in incident
    ]
    assert len(free) == len(rows)
    igi = IncidenceGraphInterface()
    matching = igi.maximum_matching(free, rows)
    assert len(matching) == len(rows)
    spatial = sum(1 for c in rows if "_disc_eq" in c.name)
    assert spatial > 0


def test_the_estimation_terminal_cost_leaves_the_model():
    # the spec removes both terminal costs, the tracking one and the
    # estimation one (gh #111)
    m = est_declared()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=3, ncp=3, scheme="LAGRANGE-RADAU"
    )
    ss = pyo.TransformationFactory(SS).create_using(m)
    assert ss.component("est_term_con") is None
    assert not drto.info(ss).has_declaration("estimation_terminal_cost")
    # the source model keeps its own
    assert m.component("est_term_con") is not None


# ----------------------------------------------------------------------
# the builder-consuming function form (gh #115)
# ----------------------------------------------------------------------
def test_the_function_reduces_a_dynamic_builder_in_place():
    calls = []

    def build():
        calls.append(1)
        return declared_model()

    ss = drto.dynamic_to_steady_state(build)
    assert calls == [1]  # called once, with no arguments
    reg = drto.info(ss)
    assert reg.has_transformation(SS)
    assert not reg.has_declaration("horizon")
    assert not ss.z.is_indexed()


def test_the_function_returns_the_model_the_builder_made():
    # it owns the model it just built, so it reduces in place rather than
    # cloning, and the caller can hold on to what came back
    built = {}

    def build():
        built["m"] = declared_model()
        return built["m"]

    ss = drto.dynamic_to_steady_state(build)
    assert ss is built["m"]


def test_the_function_returns_a_steady_build_untouched():
    # no declared horizon means the statement built its steady form
    # natively, so there is nothing to reduce
    def build():
        m = pyo.ConcreteModel()
        m.z = pyo.Var(initialize=0.5)
        m.u = pyo.Var(initialize=0.2)

        @m.Constraint()
        def balance(mm):
            return mm.u == mm.z

        drto.info(m).record_declaration("state", m.z)
        return m

    ss = drto.dynamic_to_steady_state(build)
    assert not drto.info(ss).has_transformation(SS)
    assert ss.component("balance") is not None
    assert drto.info(ss).components("state") == (ss.z,)


def test_the_factory_form_still_preserves_the_source():
    # the export shadows the module attribute, and the registered
    # transformation is reached through the factory either way
    m = declared_model()
    ss = pyo.TransformationFactory(SS).create_using(m)
    assert drto.info(m).has_declaration("horizon")  # the source is untouched
    assert not drto.info(ss).has_declaration("horizon")
