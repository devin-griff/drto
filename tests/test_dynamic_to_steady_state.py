# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 005: the steady-state reduction."""
import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from test_declarations import base_model, declared_model
from test_infinite_horizon import indexed_model, ready_model

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")

SS = "drto.dynamic_to_steady_state"


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
    # time collapsed: every Var and Constraint single-point, derivatives gone
    assert not ss.z.is_indexed() and not ss.u.is_indexed()
    assert not ss.cost.is_indexed()
    assert ss.component("dzdt") is None
    assert ss.component("t") is None
    assert not ss.ode.is_indexed()
    assert not ss.stage.is_indexed()
    # initial condition removed; the source dynamic model is untouched
    assert ss.component("init") is None
    assert m.z.is_indexed() and m.component("init") is not None
    # the dynamics are the equilibrium: 0 = f has no derivative left
    assert "dzdt" not in str(ss.ode.expr)


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


def test_derivative_in_an_algebraic_equation_is_zeroed():
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
    # the quasi-static form: the derivative term is gone
    assert "dzdt" not in str(m.w_def.expr)
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
