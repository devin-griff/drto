# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 002: the move_suppression declaration."""
import pyomo.environ as pyo
import pytest
from pyomo.core.expr import identify_variables
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

IH = "drto.infinite_horizon"
SSO = "drto.steady_state_optimization"


def moved_model(declare_moves=True):
    """dz/dt = -z + u with a tracking cost and a priced move.

    The move members price u against the previous sample, the first one
    against the ``u_prev`` Param.
    """
    m = pyo.ConcreteModel()
    N, h = 4, 2.5
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z = pyo.Var(m.t, initialize=0.4)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.z_ss = pyo.Param(initialize=0.3, mutable=True)
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)
    m.u_prev = pyo.Param(initialize=0.5, mutable=True)
    m.cost = pyo.Var(m.t)
    m.mcost = pyo.Var(m.t)
    ts = sorted(m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t]

    @m.Constraint(ts[:-1])
    def stage(m, t):
        return m.cost[t] == (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

    @m.Constraint(ts[:-1])
    def move(m, t):
        k = ts.index(t)
        prev = m.u_prev if k == 0 else m.u[ts[k - 1]]
        return m.mcost[t] == 0.1 * (m.u[t] - prev) ** 2

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    drto.steady_state(m.z, m.z_ss)
    drto.steady_state_control(m.u, m.u_ss)
    if declare_moves:
        drto.move_suppression(m.move)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def objective_vars(m):
    return {v.name for v in identify_variables(m.drto_objective.expr)}


# ----------------------------------------------------------------------
# the declaration
# ----------------------------------------------------------------------
def test_declares_and_renders():
    m = moved_model()
    assert drto.info(m).declarations("move_suppression")
    assert "move suppression" in str(drto.info(m))


def test_one_per_model():
    m = moved_model()
    # after discretization sorted(m.t) is the collocation grid; the cost
    # families index over the recorded samples
    ts = list(drto.info(m).declarations("horizon")[0]["samples"])

    @m.Constraint(ts[:-1])
    def move2(m, t):
        return m.mcost[t] == 0.0

    with pytest.raises(ValueError, match="already called"):
        drto.move_suppression(m.move2)


def test_requires_a_horizon():
    # a steady-authored model: control declared, no horizon
    m2 = pyo.ConcreteModel()
    m2.u = pyo.Var(initialize=0.5)
    m2.mcost = pyo.Var()
    m2.z = pyo.Var(initialize=0.4)
    drto.state(m2.z)
    drto.control(m2.u)

    @m2.Constraint()
    def move(m2):
        return m2.mcost == 0.1 * m2.u**2

    with pytest.raises(ValueError, match="requires a declared horizon"):
        drto.move_suppression(m2.move)


def test_a_state_reference_is_rejected():
    m = moved_model(declare_moves=False)
    ts = list(drto.info(m).declarations("horizon")[0]["samples"])

    @m.Constraint(ts[:-1])
    def bad(m, t):
        return m.mcost[t] == 0.1 * (m.u[t] - m.z[t]) ** 2

    with pytest.raises(ValueError, match="not a declared control member"):
        drto.move_suppression(m.bad)


def test_a_reference_outside_the_window_is_rejected():
    m = moved_model(declare_moves=False)
    ts = list(drto.info(m).declarations("horizon")[0]["samples"])

    @m.Constraint(ts[:-1])
    def bad(m, t):
        k = ts.index(t)
        other = m.u[ts[k - 2]] if k >= 2 else m.u_prev
        return m.mcost[t] == 0.1 * (m.u[t] - other) ** 2

    with pytest.raises(ValueError, match="outside its own sample"):
        drto.move_suppression(m.bad)


# ----------------------------------------------------------------------
# the transforms
# ----------------------------------------------------------------------
def test_the_objective_sums_it():
    m = moved_model()
    drto.build_objective(m)
    names = objective_vars(m)
    assert any(n.startswith("mcost") for n in names)
    assert any(n.startswith("cost") for n in names)


def test_the_steady_reduction_drops_it():
    m = moved_model()
    pyo.TransformationFactory("drto.dynamic_to_steady_state").apply_to(m)
    assert m.component("move") is None
    assert not drto.info(m).has_declaration("move_suppression")
    assert drto.info(m).has_declaration("tracking_stage_cost")


def test_steady_state_optimization_excludes_it():
    m = moved_model()
    sm = pyo.TransformationFactory(SSO).create_using(m)
    assert sm.component("move") is None
    assert not any(n.startswith("mcost") for n in objective_vars(sm))


def test_the_simulation_sheds_it():
    m = moved_model()
    pyo.TransformationFactory("drto.dynamic_simulation").apply_to(m)
    assert m.component("move") is None
    assert not drto.info(m).has_declaration("move_suppression")


def test_the_terminal_segment_keeps_it_off_the_tail():
    m = moved_model()
    pyo.TransformationFactory(IH).apply_to(m)
    # kept on the finite horizon
    assert m.component("move") is not None
    assert m.move.active
    # not replicated on the segment
    b = m.component("drto_ih")
    assert b.component("mcost") is None
    # the tail and pin cost groups reference no move-cost member and not
    # the previous action
    reg = drto.info(m)
    for record in reg.declarations("cost_group"):
        for expr, _ in record["terms"]:
            for v in identify_variables(expr, include_fixed=True):
                assert not v.name.startswith("mcost")
    # the assembled objective still sums the horizon move cost
    drto.build_objective(m)
    assert any(n.startswith("mcost") for n in objective_vars(m))
