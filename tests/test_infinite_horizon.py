# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 004: the infinite-horizon terminal segment."""
import math

import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from test_declarations import base_model, declared_model

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")

IH = "drto.infinite_horizon"


def ready_model():
    """The declared linear model, discretized: ready for the transform."""
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def hicks(N, h=1):
    """The Hicks-Ray CSTR, declared, with an N-step horizon."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.u1sf = pyo.Param(initialize=600, mutable=True)  # coolant-flow scale factor
    m.u2sf = pyo.Param(initialize=40, mutable=True)  # residence-time scale factor
    m.k0 = pyo.Param(initialize=300, mutable=True)  # Arrhenius pre-exponential
    m.ea = pyo.Param(initialize=5, mutable=True)  # dimensionless activation energy
    m.a0 = pyo.Param(initialize=1.95e-4, mutable=True)  # heat-transfer coefficient
    m.ztcw = pyo.Param(initialize=0.38, mutable=True)  # coolant temperature
    m.ztf = pyo.Param(initialize=0.395, mutable=True)  # feed temperature

    m.zc_ss = pyo.Param(initialize=0.6416, mutable=True)  # steady-state targets
    m.zt_ss = pyo.Param(initialize=0.5387, mutable=True)
    m.v1_ss = pyo.Param(initialize=0.57828, mutable=True)
    m.v2_ss = pyo.Param(initialize=0.49989, mutable=True)
    m.zc_hat = pyo.Param(initialize=0.625, mutable=True)  # state feedback hooks
    m.zt_hat = pyo.Param(initialize=0.525, mutable=True)

    m.zc = pyo.Var(m.t, bounds=(0, 1), initialize=0.6416)
    m.zt = pyo.Var(m.t, bounds=(0, None), initialize=0.5387)
    m.dzc = DerivativeVar(m.zc, wrt=m.t)
    m.dzt = DerivativeVar(m.zt, wrt=m.t)
    m.v1 = pyo.Var(m.t, bounds=(0.166666666666667, 1), initialize=0.57828)
    m.v2 = pyo.Var(m.t, bounds=(0.025, 1), initialize=0.49989)
    m.cost = pyo.Var(m.t)  # unbounded: a cost var pinned at a bound drags ipopt

    @m.Constraint(m.t)
    def zc_ode(m, t):
        return m.dzc[t] == (1 - m.zc[t]) / (m.u2sf * m.v2[t]) - m.k0 * m.zc[
            t
        ] * pyo.exp(-m.ea / m.zt[t])

    @m.Constraint(m.t)
    def zt_ode(m, t):
        return m.dzt[t] == (
            (m.ztf - m.zt[t]) / (m.u2sf * m.v2[t])
            + m.k0 * m.zc[t] * pyo.exp(-m.ea / m.zt[t])
            - m.a0 * m.u1sf * m.v1[t] * (m.zt[t] - m.ztcw)
        )

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == (
            10 * (m.zc[t] - m.zc_ss) ** 2
            + 2 * (m.zt[t] - m.zt_ss) ** 2
            + (m.v1[t] - m.v1_ss) ** 2
            + 0.5 * (m.v2[t] - m.v2_ss) ** 2
        )

    @m.Constraint()
    def zc_init(m):
        return m.zc[0] == m.zc_hat

    @m.Constraint()
    def zt_init(m):
        return m.zt[0] == m.zt_hat

    drto.horizon(m.t)
    drto.state(m.zc, m.zt)
    drto.dynamics(m.zc_ode, m.zt_ode)
    drto.control(m.v1, m.v2, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.zc_init, m.zt_init)
    drto.steady_state(m.zc, m.zc_ss)
    drto.steady_state(m.zt, m.zt_ss)
    drto.steady_state_control(m.v1, m.v1_ss)
    drto.steady_state_control(m.v2, m.v2_ss)
    return m


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_requires_the_declarations():
    m = base_model()
    with pytest.raises(ValueError, match="horizon"):
        pyo.TransformationFactory(IH).apply_to(m)


def test_economic_alone_is_rejected():
    m = base_model()
    m.ecost = pyo.Var(m.t)

    @m.Constraint(sorted(m.t)[:-1])
    def econ(m, t):
        return m.ecost[t] == -m.u[t]

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u)
    drto.economic_stage_cost(m.econ)
    with pytest.raises(ValueError, match="tail integral diverges"):
        pyo.TransformationFactory(IH).apply_to(m)


def test_requires_a_discretized_time_set():
    m = declared_model()
    with pytest.raises(ValueError, match="discretize"):
        pyo.TransformationFactory(IH).apply_to(m)


def test_beta_must_exceed_one():
    m = ready_model()
    with pytest.raises(ValueError, match="beta > 1"):
        pyo.TransformationFactory(IH).apply_to(m, beta=1.0)


def test_double_application_errors():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    with pytest.raises(ValueError, match="already applied"):
        pyo.TransformationFactory(IH).apply_to(m)


def test_parameterized_controls_block_application():
    m = ready_model()
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    with pytest.raises(ValueError, match="before drto.parameterize"):
        pyo.TransformationFactory(IH).apply_to(m)


def test_assembled_objective_blocks_application():
    m = ready_model()
    drto.build_objective(m)
    with pytest.raises(ValueError, match="already assembled"):
        pyo.TransformationFactory(IH).apply_to(m)


# ----------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------
def test_segment_structure():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_infinite_horizon
    fe = b.tau.get_finite_elements()
    assert len(fe) == 4  # nfe=3 default
    # dilated dynamics at interior collocation points only
    assert all(s not in b.ode for s in fe)
    assert len(b.ode) == 15  # 3 elements x 5 points
    # linking present; no terminal condition is imposed
    assert b.component("ode_equilibrium") is None
    assert b.component("z_link") is not None
    # segment control parameterized: free values at collocation points only
    assert len(b.u) == 15
    assert 0 not in b.u and 1 not in b.u


def test_gamma_follows_the_mesh_rule_and_option_overrides():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_infinite_horizon
    dt = 2.5  # the declared sample spacing
    tau11 = sorted(b.tau)[1]
    assert pyo.value(b.gamma) == pytest.approx(math.atanh(tau11) / dt)

    m2 = ready_model()
    pyo.TransformationFactory(IH).apply_to(m2, gamma=0.05)
    assert pyo.value(m2.drto_infinite_horizon.gamma) == 0.05

    m3 = ready_model()
    pyo.TransformationFactory(IH).apply_to(m3, gamma="rule")
    assert pyo.value(m3.drto_infinite_horizon.gamma) == pytest.approx(
        pyo.value(m.drto_infinite_horizon.gamma)
    )

    m4 = ready_model()
    with pytest.raises(ValueError, match="'rule' .* or a number"):
        pyo.TransformationFactory(IH).apply_to(m4, gamma="fast")


def test_declared_terminal_cost_is_deactivated():
    m = declared_model()
    m.term = pyo.Var()
    tN = m.t.last()

    @m.Constraint()
    def terminal(m):
        return m.term == 10 * (m.z[tN] - m.z_ss) ** 2

    drto.tracking_terminal_cost(m.terminal)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    # the tail owns the cost-to-go: V_f would double-count
    assert not m.terminal.active
    obj = drto.build_objective(m)
    from pyomo.core.expr import identify_variables

    assert not any(v is m.term for v in identify_variables(obj.expr))
    (ih_rec,) = [r for r in drto.info(m).transformations if r["name"] == IH]
    assert "deactivated" in ih_rec["outcome"]["terminal_cost"]


def test_tail_terms_reach_the_objective():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_infinite_horizon
    obj = drto.build_objective(m)
    from pyomo.common.collections import ComponentSet
    from pyomo.core.expr import identify_variables

    in_obj = ComponentSet(identify_variables(obj.expr))
    (group,) = drto.info(m).declarations("cost_group")
    assert len(group["terms"]) == 15  # 3 elements x 5 points
    # the terms are named Expressions (no tail variables or constraints);
    # every variable under them reaches the objective
    for term, _ in group["terms"]:
        for v in identify_variables(term.expr):
            assert v in in_obj


def test_beta_and_gamma_retune_without_reapply():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_infinite_horizon
    drto.build_objective(m)
    for t in m.t:
        m.cost[t].set_value(0.0)
    for v in b.component_data_objects(pyo.Var):
        v.set_value(0.5)
    obj = m.component("drto_objective")
    before = pyo.value(obj.expr)
    b.beta.set_value(2.4)
    assert pyo.value(obj.expr) == pytest.approx(2 * before)


def test_create_using_leaves_the_source_alone():
    m = ready_model()
    m2 = pyo.TransformationFactory(IH).create_using(m)
    assert m2.component("drto_infinite_horizon") is not None
    assert m.component("drto_infinite_horizon") is None
    assert drto.info(m2).has_transformation(IH)
    assert not drto.info(m).has_transformation(IH)


def test_application_is_recorded():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m, nfe=2, ncp=4)
    reg = drto.info(m)
    assert reg.has_transformation(IH)
    outcome = reg.transformations[-1]["outcome"]
    assert outcome["segment"] == "2 elements x 4 Legendre points"


# ----------------------------------------------------------------------
# states with extra indexes, and algebraic variables and equations
# ----------------------------------------------------------------------
def indexed_model():
    """Two coupled first-order states as one Var over (i, t)."""
    m = pyo.ConcreteModel()
    N, h = 4, 2.5  # samples and sampling time
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.i = pyo.Set(initialize=[1, 2])
    m.tau_p = pyo.Param(initialize=1.0, mutable=True)  # time constant
    m.x_ss = pyo.Param(m.i, initialize={1: 0.5, 2: 0.5}, mutable=True)
    m.u_ss = pyo.Param(initialize=0.5, mutable=True)  # = x_ss: the fixed point
    m.x_hat = pyo.Param(m.i, initialize={1: 0.2, 2: 0.8}, mutable=True)

    m.x = pyo.Var(m.i, m.t, initialize=0.5)
    m.dx = DerivativeVar(m.x, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.cost = pyo.Var(m.t)

    @m.Constraint(m.i, m.t)
    def ode(m, i, t):
        if i == 1:
            return m.dx[1, t] == (-m.x[1, t] + m.u[t]) / m.tau_p
        return m.dx[2, t] == (m.x[1, t] - m.x[2, t]) / m.tau_p

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return (
            m.cost[t]
            == sum((m.x[i, t] - m.x_ss[i]) ** 2 for i in m.i) + (m.u[t] - m.u_ss) ** 2
        )

    @m.Constraint(m.i)
    def init(m, i):
        return m.x[i, 0] == m.x_hat[i]

    drto.horizon(m.t)
    drto.state(m.x)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    return m


def dae_model():
    """One state, one undeclared algebraic variable with its equation."""
    m = pyo.ConcreteModel()
    N, h = 4, 2.5  # samples and sampling time
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.u_ss = pyo.Param(initialize=0.5, mutable=True)  # = z_ss: the fixed point
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)

    m.z = pyo.Var(m.t, initialize=0.5)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.w = pyo.Var(m.t, initialize=0.5)  # algebraic: not declared
    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t)
    def w_def(m, t):
        return m.w[t] == 0.5 * (m.z[t] + m.u[t])

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dz[t] == m.w[t] - m.z[t]

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return (
            m.cost[t]
            == (m.z[t] - m.z_ss) ** 2 + (m.w[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2
        )

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    return m


def test_indexed_state_segment_structure():
    m = indexed_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_infinite_horizon
    ntau = len(sorted(b.tau))
    assert len(b.x) == 2 * ntau  # a copy member per (i, tau)
    assert len(b.ode) == 2 * 15  # dilated dynamics per member
    assert len(b.x_link) == 2  # linked per member


@needs_ipopt
def test_indexed_state_reaches_the_fixed_point():
    m = indexed_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    drto.build_objective(m)
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
    b = m.drto_infinite_horizon
    for i in m.i:
        assert pyo.value(b.x[i, 1]) == pytest.approx(0.5, abs=1e-4)


def test_algebraic_variables_are_discovered_and_replicated():
    m = dae_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_infinite_horizon
    # the algebraic copy exists without a declaration
    assert b.component("w") is not None
    # its equation holds at the interior collocation points only
    fe = b.tau.get_finite_elements()
    assert len(b.w_def) == 15
    assert not any(s in b.w_def for s in fe)
    (ih_rec,) = [r for r in drto.info(m).transformations if r["name"] == IH]
    assert "1 component " in ih_rec["outcome"]["algebraic"] + " "


@needs_ipopt
def test_algebraic_model_reaches_the_fixed_point():
    m = dae_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    drto.build_objective(m)
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
    b = m.drto_infinite_horizon
    assert pyo.value(b.z[1]) == pytest.approx(0.5, abs=1e-4)
    assert pyo.value(b.w[1]) == pytest.approx(0.5, abs=1e-4)


def test_legendre_discretized_horizon_applies():
    # pyomo.dae's Legendre continuity equations are discretization
    # artifacts, not algebraic equations: they must not be replicated
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-LEGENDRE"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    assert m.drto_infinite_horizon.component("z_link") is not None


def test_unpinned_algebraic_copy_errors():
    # a variable copied to the segment with no replicated equation would be
    # free there; the transform stops instead of letting the solver exploit it
    m = dae_model()
    m.del_component(m.w_def)

    @m.Constraint(sorted(m.t))  # a list of numbers, not the time set
    def w_def(m, t):
        return m.w[t] == 0.5 * (m.z[t] + m.u[t])

    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    with pytest.raises(ValueError, match="no replicated equation involves"):
        pyo.TransformationFactory(IH).apply_to(m)


def test_bad_profile_errors_before_the_model_is_touched():
    m = ready_model()
    with pytest.raises(ValueError, match="profile"):
        pyo.TransformationFactory(IH).apply_to(m, profile="colocation")
    assert m.component("drto_infinite_horizon") is None


# ----------------------------------------------------------------------
# the numbers: the Hicks study, compressed
# ----------------------------------------------------------------------
@needs_ipopt
def test_hicks_short_horizon_reproduces_the_long_one():
    ipopt = pyo.SolverFactory("ipopt")

    m50 = hicks(50)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m50, wrt=m50.t, nfe=50, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("cvp.parameterize").apply_to(m50)
    drto.build_objective(m50)
    r = ipopt.solve(m50)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal

    m5 = hicks(5)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m5, wrt=m5.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m5)
    pyo.TransformationFactory("cvp.parameterize").apply_to(m5)
    drto.build_objective(m5)
    r = ipopt.solve(m5)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal

    # the first control move matches the long horizon
    assert pyo.value(m5.v1[0]) == pytest.approx(pyo.value(m50.v1[0]), rel=0.05)
    assert pyo.value(m5.v2[0]) == pytest.approx(pyo.value(m50.v2[0]), rel=0.05)

    # the endpoint found the setpoint equilibrium with no pins
    b = m5.drto_infinite_horizon
    assert pyo.value(b.zc[1]) == pytest.approx(0.6416, abs=2e-3)
    assert pyo.value(b.zt[1]) == pytest.approx(0.5387, abs=2e-3)
