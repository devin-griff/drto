# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 009: drto.steady_state_optimization."""
import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")

SSO = "drto.steady_state_optimization"


def econ_model(tracking=False, estimation=False):
    """A dynamic model with an economic cost: dz/dt = -z + u, min z^2 - u.

    At rest z = u, so the economic optimum over u in [0, 1] is u = 0.5.
    """
    m = pyo.ConcreteModel()
    N, h = 4, 2.5
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z = pyo.Var(m.t, initialize=0.4)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.z_ss = pyo.Param(initialize=0.2, mutable=True)  # a known operating point
    m.u_ss = pyo.Param(initialize=0.2, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)
    m.ecost = pyo.Var(m.t)

    if estimation:
        m.k = pyo.Var(initialize=1.0)
        m.w = pyo.Var(m.t, initialize=0.0)
        m.y_meas = pyo.Param(m.t, mutable=True, initialize=0.0)
        m.est_stage = pyo.Var(m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        noise = m.w[t] if estimation else 0
        gain = m.k if estimation else 1
        return m.dzdt[t] == -gain * m.z[t] + m.u[t] + noise

    @m.Constraint(sorted(m.t)[:-1])
    def econ(m, t):
        return m.ecost[t] == m.z[t] ** 2 - m.u[t]

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.economic_stage_cost(m.econ)
    drto.initial_condition(m.init)
    drto.steady_state(m.z, m.z_ss)
    drto.steady_state_control(m.u, m.u_ss)

    if tracking:
        m.tcost = pyo.Var(m.t)

        @m.Constraint(sorted(m.t)[:-1])
        def track(m, t):
            return m.tcost[t] == (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

        drto.tracking_stage_cost(m.track)

    if estimation:
        drto.estimated_parameter(m.k)
        drto.disturbance(m.w)
        drto.measurement(m.y_meas)

        @m.Constraint(sorted(m.t)[:-1])
        def est_stage_con(m, t):
            return m.est_stage[t] == (m.y_meas[t] - m.z[t]) ** 2 + m.w[t] ** 2

        drto.estimation_stage_cost(m.est_stage_con)

    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def steady_authored_model():
    """The same economics written directly as steady-state: no horizon."""
    m = pyo.ConcreteModel()
    m.z = pyo.Var(initialize=0.4)
    m.u = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.ecost = pyo.Var()

    @m.Constraint()
    def balance(m):
        return m.z == m.u

    @m.Constraint()
    def econ(m):
        return m.ecost == m.z**2 - m.u

    drto.state(m.z)
    drto.control(m.u)
    drto.economic_stage_cost(m.econ)
    return m


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_requires_the_declarations():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    drto.state(m.z)
    with pytest.raises(ValueError, match="missing: control, economic_stage_cost"):
        pyo.TransformationFactory(SSO).apply_to(m)


def test_bad_weight_errors_before_the_model_is_touched():
    m = econ_model()
    with pytest.raises(ValueError):
        pyo.TransformationFactory(SSO).apply_to(m, tracking_weight="heavy")
    assert m.component("drto_objective") is None


# ----------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------
def test_dynamic_model_composes_the_reduction():
    m = econ_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    applied = [r["name"] for r in drto.info(m).transformations]
    assert "drto.dynamic_to_steady_state" in applied
    assert not m.z.is_indexed()  # collapsed to a single point


def test_steady_authored_model_skips_the_reduction():
    m = steady_authored_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    applied = [r["name"] for r in drto.info(m).transformations]
    assert "drto.dynamic_to_steady_state" not in applied
    assert m.component("drto_objective") is not None


def test_controls_stay_free():
    # the optimization mode frees the controls, unlike the simulation modes
    m = econ_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    assert not m.u.fixed


def test_cost_equations_stay():
    # unlike the simulation modes, this one needs its costs
    m = econ_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    assert drto.info(m).has_declaration("economic_stage_cost")
    assert m.component("drto_objective") is not None


def test_steady_state_pairings_are_kept():
    # they are the record that makes a later write-back possible
    m = econ_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    reg = drto.info(m)
    assert reg.has_declaration("steady_state")
    assert reg.has_declaration("steady_state_control")


def test_application_is_recorded():
    m = econ_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    reg = drto.info(m)
    assert reg.has_transformation(SSO)
    assert reg.transformations[-1]["outcome"]["controls"] == "free"


def test_create_using_leaves_the_source_alone():
    m = econ_model()
    m2 = pyo.TransformationFactory(SSO).create_using(m)
    assert m2.component("drto_objective") is not None
    assert m.component("drto_objective") is None
    assert m.z.is_indexed()  # the source stays dynamic


# ----------------------------------------------------------------------
# the estimation neutralization
# ----------------------------------------------------------------------
def test_estimation_declarations_are_neutralized():
    # a free disturbance would be a decision variable the optimizer exploits
    m = econ_model(estimation=True)
    pyo.TransformationFactory(SSO).apply_to(m)
    reg = drto.info(m)
    for kind in ("estimation_stage_cost", "measurement", "disturbance"):
        assert not reg.has_declaration(kind), kind
    assert m.component("w") is None
    assert m.component("y_meas") is None
    assert reg.has_declaration("estimated_parameter")
    assert m.k.fixed


# ----------------------------------------------------------------------
# the numbers
# ----------------------------------------------------------------------
@needs_ipopt
def test_economic_optimum():
    # at rest z = u, so minimizing z^2 - u over u in [0, 1] gives u = 0.5
    m = econ_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.u) == pytest.approx(0.5, abs=1e-6)
    assert pyo.value(m.z) == pytest.approx(0.5, abs=1e-6)


@needs_ipopt
def test_steady_authored_reaches_the_same_optimum():
    m = steady_authored_model()
    pyo.TransformationFactory(SSO).apply_to(m)
    pyo.SolverFactory("ipopt").solve(m)
    assert pyo.value(m.u) == pytest.approx(0.5, abs=1e-6)


@needs_ipopt
def test_tracking_weight_regularizes_toward_the_known_point():
    # the tracking term pulls the economic optimum (0.5) toward u_ss = 0.2
    results = {}
    for w in (0.0, 1.0, 20.0):
        m = econ_model(tracking=True)
        pyo.TransformationFactory(SSO).apply_to(m, tracking_weight=w)
        pyo.SolverFactory("ipopt").solve(m)
        results[w] = pyo.value(m.u)
    assert results[0.0] == pytest.approx(0.5, abs=1e-6)  # pure economics
    assert results[0.0] > results[1.0] > results[20.0]  # pulled toward 0.2
    assert results[20.0] == pytest.approx(0.2, abs=0.02)
