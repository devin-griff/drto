# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 005: the steady-state reduction."""
import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port

import drto
from test_declarations import base_model, declared_model
from test_infinite_horizon import _dof, block_model, indexed_model, ready_model

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
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
    with pytest.raises(ValueError, match="missing: state, dynamics"):
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
    r = pyo.SolverFactory("ipopt").solve(ss)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
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


# ── time-indexed Blocks in the reduction (feature 021) ───────────────────────


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


def test_nested_time_indexed_block_is_rejected():
    m = block_model(nested=True)
    with pytest.raises(ValueError, match="nested in another"):
        pyo.TransformationFactory(SS).apply_to(m)


def test_block_indexed_beyond_time_is_rejected():
    m = block_model()
    m.extra = pyo.Block(m.t, [1, 2])
    with pytest.raises(ValueError, match="more than the declared time set"):
        pyo.TransformationFactory(SS).apply_to(m)


@needs_ipopt
def test_block_reduction_reaches_the_fixed_point():
    m = block_model()
    pyo.TransformationFactory(SS).apply_to(m)
    drto.build_objective(m)
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
    # dz/dt = u - y at rest with y = 2z + 0.1q and u in [0, 1]: the
    # setpoint needs u = 1.3, so the optimum rides the bound, u = y = 1
    # and z = (1 - 0.3) / 2, the member equation carrying the physics
    assert pyo.value(m.u) == pytest.approx(1.0, abs=1e-6)
    assert pyo.value(m.props[0].y) == pytest.approx(1.0, abs=1e-6)
    assert pyo.value(m.z) == pytest.approx(0.35, abs=1e-6)
