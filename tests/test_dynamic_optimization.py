# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 006: the dynamic optimization assembly."""
import pyomo.environ as pyo
import pytest
from pyomo.core.expr import identify_variables
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from test_declarations import base_model, declared_model
from test_infinite_horizon import ready_model

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")

DO = "drto.dynamic_optimization"


def estimation_model(nonlinear_noise=False):
    """A declared control model that also carries the estimation surface."""
    m = pyo.ConcreteModel()
    N, h = 4, 2.5
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z = pyo.Var(m.t, initialize=0.4)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.u_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)
    m.cost = pyo.Var(m.t)

    m.k = pyo.Var(initialize=1.0)  # estimated parameter
    m.w = pyo.Var(m.t, initialize=0.0)  # process noise
    m.y_meas = pyo.Param(m.t, mutable=True, initialize=0.0)
    m.z_prior = pyo.Param(initialize=0.4, mutable=True)
    m.est_stage = pyo.Var(m.t)
    m.arrival = pyo.Var()

    @m.Constraint(m.t)
    def ode(m, t):
        noise = m.w[t] * m.z[t] if nonlinear_noise else m.w[t]
        return m.dzdt[t] == -m.k * m.z[t] + m.u[t] + noise

    @m.Constraint(sorted(m.t)[:-1])
    def stage(m, t):
        return m.cost[t] == 10 * (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    @m.Constraint(sorted(m.t)[:-1])
    def est_stage_con(m, t):
        return m.est_stage[t] == (m.y_meas[t] - m.z[t]) ** 2 + m.w[t] ** 2

    @m.Constraint()
    def arrival_con(m):
        return m.arrival == (m.z[0] - m.z_prior) ** 2

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    drto.estimated_parameter(m.k)
    drto.disturbance(m.w)
    drto.measurement(m.y_meas)
    drto.estimation_stage_cost(m.est_stage_con)
    drto.arrival_cost(m.arrival_con)

    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def both_costs_model():
    """A ready model with a tracking and an economic stage cost declared."""
    m = declared_model()
    m.ecost = pyo.Var(m.t)

    @m.Constraint(sorted(m.t)[:-1])
    def econ(m, t):
        return m.ecost[t] == -m.u[t]

    drto.economic_stage_cost(m.econ)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_requires_the_declarations():
    m = base_model()
    drto.horizon(m.t)
    with pytest.raises(ValueError, match="missing: state, dynamics"):
        pyo.TransformationFactory(DO).apply_to(m)


def test_requires_a_stage_cost():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.initial_condition(m.init)
    with pytest.raises(ValueError, match="requires a stage cost"):
        pyo.TransformationFactory(DO).apply_to(m)


def test_bad_weight_errors_before_the_model_is_touched():
    m = ready_model()
    with pytest.raises(ValueError):
        pyo.TransformationFactory(DO).apply_to(m, tracking_weight="heavy")
    assert m.component("drto_objective") is None


def test_nonlinear_disturbance_is_refused():
    m = estimation_model(nonlinear_noise=True)
    with pytest.raises(ValueError, match="appears nonlinearly"):
        pyo.TransformationFactory(DO).apply_to(m)


# ----------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------
def test_controls_are_parameterized_and_the_horizon_is_kept():
    m = ready_model()
    pyo.TransformationFactory(DO).apply_to(m)
    # piecewise constant: one free control value per finite element
    assert len(m.u) == 4
    assert len(m.t) > 1  # the horizon is not reduced to a point


def test_objective_is_assembled():
    m = ready_model()
    pyo.TransformationFactory(DO).apply_to(m)
    assert m.component("drto_objective") is not None


def test_tracking_weight_applies_only_when_both_costs_are_declared():
    m = both_costs_model()
    pyo.TransformationFactory(DO).apply_to(m, tracking_weight=3.0)
    (record,) = drto.info(m).declarations("tracking_stage_cost")
    assert record["weight"] == 3.0


def test_no_weight_recorded_with_one_stage_cost():
    m = ready_model()
    pyo.TransformationFactory(DO).apply_to(m)
    (record,) = drto.info(m).declarations("tracking_stage_cost")
    assert "weight" not in record


def test_application_is_recorded():
    m = ready_model()
    pyo.TransformationFactory(DO).apply_to(m)
    reg = drto.info(m)
    assert reg.has_transformation(DO)
    assert reg.transformations[-1]["outcome"]["horizon"] == "kept"


def test_create_using_leaves_the_source_alone():
    m = ready_model()
    m2 = pyo.TransformationFactory(DO).create_using(m)
    assert m2.component("drto_objective") is not None
    assert m.component("drto_objective") is None
    assert drto.info(m2).has_transformation(DO)
    assert not drto.info(m).has_transformation(DO)


# ----------------------------------------------------------------------
# the estimation drop
# ----------------------------------------------------------------------
def test_estimation_costs_and_measurements_leave_the_model():
    m = estimation_model()
    pyo.TransformationFactory(DO).apply_to(m)
    reg = drto.info(m)
    for kind in ("estimation_stage_cost", "arrival_cost", "measurement", "disturbance"):
        assert not reg.has_declaration(kind), kind
    assert m.component("est_stage_con") is None
    assert m.component("arrival_con") is None
    assert m.component("y_meas") is None


def test_disturbance_is_eliminated_by_substitution():
    m = estimation_model()
    pyo.TransformationFactory(DO).apply_to(m)
    assert m.component("w") is None
    # no vestigial fixed-at-zero variable: the reference is gone from the ode
    cd = m.ode[sorted(m.t)[1]]
    assert all(
        v.parent_component().local_name != "w"
        for v in identify_variables(cd.body, include_fixed=True)
    )


def test_estimated_parameter_is_fixed_and_keeps_its_record():
    m = estimation_model()
    pyo.TransformationFactory(DO).apply_to(m)
    # it stays a live coefficient in the equations, so the record stays
    assert drto.info(m).components("estimated_parameter") == (m.k,)
    assert m.k.fixed
    assert pyo.value(m.k) == pytest.approx(1.0)


def test_estimated_parameter_without_a_value_errors():
    m = estimation_model()
    m.k.set_value(None)
    with pytest.raises(ValueError, match="has none"):
        pyo.TransformationFactory(DO).apply_to(m)


# ----------------------------------------------------------------------
# the numbers
# ----------------------------------------------------------------------
@needs_ipopt
def test_solve_drives_the_state_toward_the_setpoint():
    m = ready_model()
    pyo.TransformationFactory(DO).apply_to(m)
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
    # z starts at z_hat = 0.4 and is driven toward z_ss = 0.5
    assert pyo.value(m.z[m.t.last()]) == pytest.approx(0.5, abs=5e-2)


@needs_ipopt
def test_estimation_model_solves_as_a_control_problem():
    m = estimation_model()
    pyo.TransformationFactory(DO).apply_to(m)
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal


@needs_ipopt
def test_infinite_horizon_tail_reaches_the_objective():
    # infinite_horizon applies first: the objective is assembled here, so the
    # tail's cost group must be registered by then
    m = ready_model()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    pyo.TransformationFactory(DO).apply_to(m)
    reg = drto.info(m)
    assert reg.has_transformation("drto.infinite_horizon")
    assert reg.has_declaration("cost_group")
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
