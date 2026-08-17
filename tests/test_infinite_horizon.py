# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 004: the infinite-horizon terminal segment."""
import math

import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from test_declarations import base_model, declared_model

ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
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
    pyo.TransformationFactory(IH).apply_to(m, terminal="none")
    b = m.drto_ih
    fe = b.tau.get_finite_elements()
    assert len(fe) == 4  # nfe=3 default
    # dilated dynamics at interior collocation points only
    assert all(s not in b.ode for s in fe)
    assert len(b.ode) == 15  # 3 elements x 5 points
    # linking present; terminal='none' imposes no endpoint pin
    assert b.component("z_link") is not None
    assert b.component("z_pin") is None and b.component("z_pin_eq") is None
    # segment control parameterized: free values at collocation points only
    assert len(b.u) == 15
    assert 0 not in b.u and 1 not in b.u


def test_segment_pairing_recorded_on_the_registry():
    # the transform writes down which tail component belongs to which
    # declaration: actual objects, resolved once at build (gh #27)
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    reg = drto.info(m)
    (zrec,) = reg._segment_records("state")
    assert zrec["of"] is m.z
    assert zrec["copy"] is b.z
    assert zrec["derivative"] is b.z_dtau
    assert zrec["disc"] is b.z_dtau_disc_eq
    assert zrec["continuity"] is b.z_tau_cont_eq
    assert zrec["link"] is b.z_link
    assert zrec["pin"] is b.z_pin_eq
    assert zrec["pin_up"] is b.z_pin_up
    assert zrec["pin_lo"] is b.z_pin_lo
    (urec,) = reg._segment_records("control")
    assert urec["of"] is m.u and urec["copy"] is b.u
    (drec,) = reg._segment_records("dynamics")
    assert drec["of"] is m.ode and drec["copy"] is b.ode
    assert drec["algebraic"] is None  # a flat state has no algebraic members

    # a clone carries the pairing with its references remapped
    m2 = m.clone()
    (zrec2,) = drto.info(m2)._segment_records("state")
    assert zrec2["copy"] is m2.drto_ih.z and zrec2["copy"] is not b.z

    # terminal='none' records no pin pieces
    m3 = ready_model()
    pyo.TransformationFactory(IH).apply_to(m3, terminal="none")
    (zrec3,) = drto.info(m3)._segment_records("state")
    assert zrec3["pin"] is None and zrec3["pin_up"] is None


def test_segment_pairing_never_renders():
    # the registry view shows the declarations and the transformation
    # log; the pairing is invisible bookkeeping
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    reg = drto.info(m)
    with_records = repr(reg)
    reg._segment.clear()
    assert repr(reg) == with_records


def test_gamma_follows_the_mesh_rule_and_option_overrides():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    dt = 2.5  # the declared sample spacing
    tau11 = sorted(b.tau)[1]
    assert pyo.value(b.gamma) == pytest.approx(math.atanh(tau11) / dt)

    m2 = ready_model()
    pyo.TransformationFactory(IH).apply_to(m2, gamma=0.05)
    assert pyo.value(m2.drto_ih.gamma) == 0.05

    m3 = ready_model()
    pyo.TransformationFactory(IH).apply_to(m3, gamma="rule")
    assert pyo.value(m3.drto_ih.gamma) == pytest.approx(pyo.value(m.drto_ih.gamma))

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
    pyo.TransformationFactory(IH).apply_to(m, terminal="none")  # isolate the tail group
    b = m.drto_ih
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


def test_tail_weights_integrate_through_the_jacobian():
    # each tail weight is beta*h*w / (gamma*dt*(1 - tau^2)). The Gauss
    # weights sum to one over the tau span, so weight*(1 - tau^2) summed
    # over every point recovers beta/(gamma*dt) exactly; dropping the
    # Jacobian factor breaks the sum
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m, terminal="none")
    b = m.drto_ih
    fe = m.t.get_finite_elements()
    dt = fe[1] - fe[0]
    (group,) = drto.info(m).declarations("cost_group")
    assert all(pyo.value(w) > 0 for _, w in group["terms"])
    total = sum(pyo.value(w) * (1 - term.index() ** 2) for term, w in group["terms"])
    assert total == pytest.approx(pyo.value(b.beta) / (pyo.value(b.gamma) * dt))


def test_beta_and_gamma_retune_without_reapply():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m, terminal="none")  # tail is the only term
    b = m.drto_ih
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
    assert m2.component("drto_ih") is not None
    assert m.component("drto_ih") is None
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
    drto.steady_state(m.x, m.x_ss)  # endpoint pin target (default terminal='soft')
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
        # the algebraic w stays out of the cost (a tracking cost holds
        # states and controls only); its discovery is exercised by the ode
        return m.cost[t] == (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    drto.steady_state(m.z, m.z_ss)  # endpoint pin target (default terminal='soft')
    return m


def test_indexed_state_segment_structure():
    m = indexed_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
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
    r = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(r)
    b = m.drto_ih
    for i in m.i:
        assert pyo.value(b.x[i, 1]) == pytest.approx(0.5, abs=1e-4)


def test_algebraic_variables_are_discovered_and_replicated():
    m = dae_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    # the algebraic copy exists without a declaration
    assert b.component("w") is not None
    # its equation holds at the interior collocation points only
    fe = b.tau.get_finite_elements()
    assert len(b.w_def) == 15
    assert not any(s in b.w_def for s in fe)
    # and the copy itself holds no boundary members: every point that
    # exists is one an equation determines (gh #32)
    assert len(b.w) == 15
    assert not any(s in b.w for s in fe)
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
    r = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(r)
    b = m.drto_ih
    assert pyo.value(b.z[1]) == pytest.approx(0.5, abs=1e-4)
    # the algebra holds no endpoint copy (gh #32: interior points only);
    # the last interior point carries the settled value
    assert pyo.value(b.w[sorted(b.tau_i)[-1]]) == pytest.approx(0.5, abs=1e-4)


def test_legendre_discretized_horizon_applies():
    # pyomo.dae's Legendre continuity equations are discretization
    # artifacts, not algebraic equations: they must not be replicated
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-LEGENDRE"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    assert m.drto_ih.component("z_link") is not None


def test_derivative_reference_in_an_algebraic_equation_dilates():
    # the index-reduced energy-balance case: a replicated equation that
    # references a declared state's derivative maps it to the segment
    # derivative with the dilation factor, the same rewrite the dynamics get
    m = dae_model()
    m.del_component(m.w_def)

    @m.Constraint(m.t)
    def w_def(m, t):
        return m.w[t] == 0.5 * (m.z[t] + m.u[t]) + 0.1 * m.dz[t]

    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    s = sorted(b.tau)[1]  # an interior collocation point
    text = str(b.w_def[s].expr)
    assert "z_dtau" in text and "gamma" in text
    assert "dz[" not in text  # no reference back to the finite grid


def test_an_undeclared_states_derivative_still_errors():
    m = dae_model()
    m.w2 = pyo.Var(m.t, initialize=0.0)
    m.dw2 = DerivativeVar(m.w2, wrt=m.t)
    m.del_component(m.w_def)

    @m.Constraint(m.t)
    def w_def(m, t):
        return m.w[t] == m.dw2[t] + m.z[t]

    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    with pytest.raises(ValueError, match="not a declared state's derivative"):
        pyo.TransformationFactory(IH).apply_to(m)


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
    assert m.component("drto_ih") is None


# ----------------------------------------------------------------------
# the numbers: the Hicks study, compressed
# ----------------------------------------------------------------------
@needs_ipopt
def test_hicks_short_horizon_reproduces_the_long_one():
    ipopt = drto.scaling.solver_by_name("ipopt")

    m50 = hicks(50)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m50, wrt=m50.t, nfe=50, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("cvp.parameterize").apply_to(m50)
    drto.build_objective(m50)
    r = ipopt.solve(m50)
    assert drto.scaling.solved_to_optimality(r)

    m5 = hicks(5)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m5, wrt=m5.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    # terminal='none': the tail cost alone must reach the setpoint, the paper's
    # unpinned result being reproduced by the short horizon
    pyo.TransformationFactory(IH).apply_to(m5, terminal="none")
    pyo.TransformationFactory("cvp.parameterize").apply_to(m5)
    drto.build_objective(m5)
    r = ipopt.solve(m5)
    assert drto.scaling.solved_to_optimality(r)

    # the first control move matches the long horizon
    assert pyo.value(m5.v1[0]) == pytest.approx(pyo.value(m50.v1[0]), rel=0.05)
    assert pyo.value(m5.v2[0]) == pytest.approx(pyo.value(m50.v2[0]), rel=0.05)

    # the endpoint found the setpoint equilibrium with no pins
    b = m5.drto_ih
    assert pyo.value(b.zc[1]) == pytest.approx(0.6416, abs=2e-3)
    assert pyo.value(b.zt[1]) == pytest.approx(0.5387, abs=2e-3)


# ----------------------------------------------------------------------
# the terminal endpoint pin (Dinh et al. 2025, eq 36 soft)
# ----------------------------------------------------------------------
def test_default_is_soft_pin():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)  # default terminal='soft'
    b = m.drto_ih
    # per-state slacks, the endpoint equality, and the penalty weight
    assert b.component("z_pin_eq") is not None
    assert b.component("z_pin_up") is not None and b.component("z_pin_lo") is not None
    assert b.component("mu") is not None
    # two cost groups: the tail and the endpoint L1 penalty
    assert len(drto.info(m).declarations("cost_group")) == 2
    (rec,) = [r for r in drto.info(m).transformations if r["name"] == IH]
    assert rec["outcome"]["terminal"].startswith("soft")


def test_soft_pin_is_per_member_for_indexed_states():
    m = indexed_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)  # default terminal='soft'
    b = m.drto_ih
    # one relaxed endpoint equality and one slack pair per member i, like x_link
    assert len(b.x_pin_eq) == 2
    assert len(b.x_pin_up) == 2 and len(b.x_pin_lo) == 2


def test_soft_pin_slacks_are_nonnegative_and_reach_the_objective():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m, terminal="soft")
    b = m.drto_ih
    slacks = list(b.z_pin_up.values()) + list(b.z_pin_lo.values())
    for v in slacks:
        assert v.lb == 0 and v.ub is None
    obj = drto.build_objective(m)
    from pyomo.common.collections import ComponentSet
    from pyomo.core.expr import identify_variables

    in_obj = ComponentSet(identify_variables(obj.expr))
    assert all(v in in_obj for v in slacks)


def test_soft_pin_mu_retunes_without_reapply():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m, terminal="soft")
    b = m.drto_ih
    obj = drto.build_objective(m)
    slacks = list(b.z_pin_up.values()) + list(b.z_pin_lo.values())
    for v in b.component_data_objects(pyo.Var):
        v.set_value(0.5)
    for t in m.t:
        m.cost[t].set_value(0.0)
    for v in slacks:
        v.set_value(1.0)
    before = pyo.value(obj.expr)
    # +100 per unit of slack and of slack squared: the penalty is
    # mu*(eps + eps**2) per slack (gh #37), so at eps = 1 each slack
    # contributes 2 per unit of mu
    b.mu.set_value(pyo.value(b.mu) + 100.0)
    assert pyo.value(obj.expr) - before == pytest.approx(200.0 * len(slacks))


@needs_ipopt
def test_soft_pin_lands_the_endpoint_on_the_setpoint():
    m = hicks(5)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)  # default terminal='soft'
    pyo.TransformationFactory("cvp.parameterize").apply_to(m)
    drto.build_objective(m)
    r = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(r)
    b = m.drto_ih
    # the L1 penalty is exact: at the default mu the slacks vanish and the
    # extrapolated endpoint lands on the setpoint (soft reproduces the pin)
    assert pyo.value(b.zc[b.tau.last()]) == pytest.approx(0.6416, abs=1e-6)
    assert pyo.value(b.zt[b.tau.last()]) == pytest.approx(0.5387, abs=1e-6)
    assert pyo.value(b.zc_pin_up) == pytest.approx(0.0, abs=1e-6)
    assert pyo.value(b.zc_pin_lo) == pytest.approx(0.0, abs=1e-6)


def test_pin_requires_steady_state_targets():
    # a fully declared model WITHOUT steady_state targets; the default soft pin
    # needs one per state, and must error before the segment block is built
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    with pytest.raises(ValueError, match="steady_state target"):
        pyo.TransformationFactory(IH).apply_to(m)
    assert m.component("drto_ih") is None


@pytest.mark.parametrize("bad", ["always", "hard"])
def test_bad_terminal_value_errors_before_the_model_is_touched(bad):
    # 'hard' was removed; only 'soft' and 'none' remain, so it now errors too
    m = ready_model()
    with pytest.raises(ValueError, match="terminal"):
        pyo.TransformationFactory(IH).apply_to(m, terminal=bad)
    assert m.component("drto_ih") is None


# ── segment units (gh #10) ───────────────────────────────────────────────────


def unit_model_ih():
    pytest.importorskip("pint")
    U = pyo.units
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True, units=U.mol)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True, units=U.mol)
    m.z = pyo.Var(m.t, initialize=0.2, units=U.mol)
    m.dz = DerivativeVar(m.z, wrt=m.t, units=U.mol)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3, units=U.mol)
    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == mm.u[t] - mm.z[t]

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return (
            mm.cost[t]
            == (mm.z[t] - mm.z_ss) ** 2 / U.mol**2
            + 0.01 * (mm.u[t] - 0.5 * U.mol) ** 2 / U.mol**2
        )

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def test_segment_carries_units():
    U = pyo.units
    m = unit_model_ih()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    for comp in (b.z, b.u, b.z_dtau, b.z_pin_up, b.z_pin_lo):
        member = next(iter(comp.values())) if comp.is_indexed() else comp
        assert str(U.get_units(member)) == "mol", comp.name
    # the replicated dynamics stay dimensionally consistent on the segment
    con = next(iter(b.ode.values()))
    assert str(U.get_units(con.body)) == "mol"


def test_segment_unitless_stays_unitless():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    for comp in (b.z, b.z_dtau):
        member = next(iter(comp.values()))
        assert member.get_units() is None, comp.name


# ── the tail's disturbance handling (feature 004) ────────────────────────────


def disturbed_model():
    """The linear model with an additive disturbance in the ode."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.w = pyo.Var(m.t, initialize=0.0)
    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == mm.u[t] - mm.z[t] + mm.w[t]

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.z[t] - mm.z_ss) ** 2 + 0.01 * (mm.u[t] - 0.5) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.disturbance(m.w)
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def test_tail_fixes_disturbance_at_zero_by_default():
    m = disturbed_model()
    pyo.TransformationFactory(IH).apply_to(m)
    copy = m.drto_ih.w
    assert all(vd.fixed for vd in copy.values())
    assert all(pyo.value(vd) == 0.0 for vd in copy.values())
    log = drto.info(m)._transformations[-1]["outcome"]
    assert "w fixed at 0.0" in log["disturbances"]


def test_tail_fixes_disturbance_at_the_given_constant():
    m = disturbed_model()
    pyo.TransformationFactory(IH).apply_to(m, disturbances={"w": 0.25})
    copy = m.drto_ih.w
    assert all(vd.fixed and pyo.value(vd) == 0.25 for vd in copy.values())


def test_tail_rejects_an_undeclared_disturbance_value():
    m = disturbed_model()
    with pytest.raises(ValueError, match="not a declared disturbance"):
        pyo.TransformationFactory(IH).apply_to(m, disturbances={"nope": 1.0})


# ── time-indexed Blocks on the segment (feature 004) ─────────────────────────


def block_model(nested=False, indirect=False, flat=False):
    """The linear model with its gain routed through a Block(t) member,
    the minimal shape of the IDAES property-block idiom. ``flat=True``
    builds the same physics with a flat algebraic Var, the twin for the
    dof-parity assertion."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.cost = pyo.Var(m.t)

    m.q = pyo.Var(m.t, initialize=3.0)  # a given input, fixed everywhere
    for vd in m.q.values():
        vd.fix()

    if flat:
        m.y = pyo.Var(m.t, initialize=0.4)

        @m.Constraint(m.t)
        def gain(mm, t):
            return mm.y[t] == 2.0 * mm.z[t] + 0.1 * mm.q[t]

    def props_rule(blk, t):
        mm = blk.model()
        if indirect:
            blk.inner = pyo.Block()
            blk.inner.y = pyo.Var(initialize=0.4)
            blk.gain = pyo.Constraint(expr=blk.inner.y == 2.0 * mm.z[t])
        elif nested:
            blk.sub = pyo.Block(mm.t)
            blk.sub[t].y = pyo.Var(initialize=0.4)
            blk.gain = pyo.Constraint(expr=blk.sub[t].y == 2.0 * mm.z[t])
        else:
            blk.y = pyo.Var(initialize=0.4)
            blk.gain = pyo.Constraint(expr=blk.y == 2.0 * mm.z[t] + 0.1 * mm.q[t])

    if not flat:
        m.props = pyo.Block(m.t, rule=props_rule)

    @m.Constraint(m.t)
    def ode(mm, t):
        if flat:
            y = mm.y[t]
        elif indirect:
            y = mm.props[t].inner.y
        elif nested:
            y = mm.props[t].sub[t].y
        else:
            y = mm.props[t].y
        return mm.dz[t] == mm.u[t] - y

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.z[t] - mm.z_ss) ** 2 + 0.01 * (mm.u[t] - 0.5) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def _dof(m):
    from pyomo.util.model_size import build_model_size_report

    r = build_model_size_report(m)
    return r.activated.variables - r.activated.constraints


def test_block_members_replicate_onto_the_segment():
    m = block_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    assert hasattr(b, "props_y") and hasattr(b, "props_gain")
    log = drto.info(m)._transformations[-1]["outcome"]
    assert "1 time-indexed Block" in log["blocks"]


def test_no_segment_reference_into_main_model_members():
    from pyomo.core.expr.visitor import identify_variables

    m = block_model()
    pyo.TransformationFactory(IH).apply_to(m)
    for c in m.drto_ih.component_data_objects(pyo.Constraint, active=True):
        for v in identify_variables(c.body, include_fixed=True):
            blk = v.parent_block()
            while blk is not None and blk is not m:
                assert blk.parent_component() is not m.props, c.name
                blk = blk.parent_block()


def test_block_support_is_dof_neutral_relative_to_flat():
    """The Block route adds exactly the dof delta the flat route adds to
    the same physics: the original defect was a relative swing (21 to -48
    on the IDAES CSTR)."""
    deltas = {}
    for flat in (True, False):
        m = block_model(flat=flat)
        before = _dof(m)
        pyo.TransformationFactory(IH).apply_to(m)
        deltas[flat] = _dof(m) - before
    assert deltas[False] == deltas[True]


def test_fixed_inputs_stay_fixed_on_the_tail():
    """A fixed variable is a specification, not a decision: its segment
    copy is fixed at the horizon-end value, with no declaration involved,
    in both the flat and the Block-member form."""
    for flat in (True, False):
        m = block_model(flat=flat)
        pyo.TransformationFactory(IH).apply_to(m)
        copy = m.drto_ih.q
        assert all(vd.fixed for vd in copy.values()), f"flat={flat}"
        assert all(pyo.value(vd) == 3.0 for vd in copy.values()), f"flat={flat}"


def ref_control_model(flat=False):
    """The inlet idiom: the declared control is a Reference into Block
    members. ``flat=True`` builds the same physics with a flat control,
    the twin for the parity assertion (gh #18)."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.cost = pyo.Var(m.t)

    if flat:
        m.f = pyo.Var(m.t, bounds=(0, 2), initialize=0.3)
        fin = m.f
    else:
        m.props = pyo.Block(
            m.t,
            rule=lambda blk, t: setattr(
                blk, "f", pyo.Var(bounds=(0, 2), initialize=0.3)
            ),
        )
        m.fin = pyo.Reference(m.props[:].f)
        fin = m.fin

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == fin[t] - mm.z[t]

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.z[t] - mm.z_ss) ** 2 + 0.1 * (fin[t] - mm.z_ss) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(fin, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def test_reference_control_has_one_segment_family():
    from pyomo.core.expr.visitor import identify_variables

    m = ref_control_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    # the control's own copy serves; no shadow member family is built
    assert b.component("fin") is not None
    assert b.component("props_f") is None
    # and the replicated equations are wired to it, not orphaning it
    seg_ids = {id(vd) for vd in b.fin.values()}
    assert any(
        id(v) in seg_ids
        for c in b.component_data_objects(pyo.Constraint, active=True)
        for v in identify_variables(c.body, include_fixed=True)
    )


def test_reference_control_matches_the_flat_route():
    from pyomo.util.model_size import build_model_size_report

    sizes = {}
    for flat in (True, False):
        m = ref_control_model(flat=flat)
        pyo.TransformationFactory(IH).apply_to(m)
        r = build_model_size_report(m)
        sizes[flat] = (r.activated.variables, r.activated.constraints)
    assert sizes[False] == sizes[True]


@needs_ipopt
def test_reference_control_solves_through_the_tail():
    m = ref_control_model()
    pyo.TransformationFactory(IH).apply_to(m)
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    res = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(res)
    # at rest dz = f - z = 0 and the cost pins z at the setpoint, so the
    # tail control settles at the steady input through its own copy
    tail_end = max(m.drto_ih.fin.keys())
    assert pyo.value(m.drto_ih.fin[tail_end]) == pytest.approx(0.5, abs=1e-2)


def packed_model():
    """An indexed Var with an algebraic member: the true state is declared
    as a member-subset slice, and the W member (constant by closure) stays
    undeclared, its balance an algebraic constraint (gh #20)."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.j = pyo.Set(initialize=["A", "W"])
    m.x = pyo.Var(m.t, m.j, initialize=0.2)
    m.dx = DerivativeVar(m.x, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 2), initialize=0.3)
    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t, m.j)
    def bal(mm, t, j):
        if j == "A":
            return mm.dx[t, j] == mm.u[t] - mm.x[t, j]
        return mm.dx[t, j] == 0 * mm.u[t]

    @m.Constraint(m.t)
    def closure(mm, t):
        return mm.x[t, "W"] == 55.0

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.x[t, "A"] - mm.z_ss) ** 2 + 0.01 * (mm.u[t] - 0.5) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.x[0, "A"] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.x[:, "A"])
    drto.dynamics(m.bal)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    # the pairing takes the same slice, resolving to the wrapped Reference
    drto.steady_state(m.x[:, "A"], m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def test_member_subset_state_declares_by_slice():
    m = packed_model()
    (za,) = [c for c in drto.info(m).components("state") if c.local_name == "x_A"]
    # the slice wrapped as an attached Reference, indexed by the time set
    assert za.is_reference() and za.index_set() is m.t
    assert next(iter(za.values())) is m.x[0, "A"]


def test_member_subset_state_segment_structure():
    m = packed_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    # one family per declared state; the algebraic entries copy per entry
    assert b.component("x_A") is not None
    assert b.component("bal_algebraic") is not None
    xm = b.component("x_members")
    assert {k[0] for k in xm.keys()} == {"W"}
    assert {k[0] for k in b.component("dx_members").keys()} == {"W"}
    log = drto.info(m)._transformations[-1]["outcome"]
    assert "partially declared" in log["partial"]


@needs_ipopt
def test_member_subset_state_solves_through_the_tail():
    m = packed_model()
    pyo.TransformationFactory(IH).apply_to(m)
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    res = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(res)
    b = m.drto_ih
    assert pyo.value(b.x_A[1]) == pytest.approx(0.5, abs=1e-6)
    sp = sorted(b.tau)[3]
    assert pyo.value(b.component("x_members")["W", sp]) == pytest.approx(55.0, abs=1e-6)


def test_nested_time_indexed_block_is_rejected():
    m = block_model(nested=True)
    with pytest.raises(ValueError, match="nested in another"):
        pyo.TransformationFactory(IH).apply_to(m)


def test_indirect_member_child_is_rejected():
    m = block_model(indirect=True)
    with pytest.raises(ValueError, match="not a direct child"):
        pyo.TransformationFactory(IH).apply_to(m)


@needs_ipopt
def test_block_model_solves_through_the_tail():
    m = block_model()
    pyo.TransformationFactory(IH).apply_to(m)
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    res = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(res)


def _first_order(scaled):
    """``2*dz/dt == u - z`` or the algebraically identical bare form.

    Two samples of horizon against a time constant of two: the tail
    carries a live transient, so its dilated dynamics are load-bearing
    and a dropped coefficient changes the solution.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 2, 1))
    m.z = pyo.Var(m.t, initialize=0.4)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.u_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)
    m.L = pyo.Param(initialize=2.0, mutable=True)
    m.cost = pyo.Var(m.t)

    if scaled:

        @m.Constraint(m.t)
        def ode(mm, t):
            return mm.L * mm.dzdt[t] == mm.u[t] - mm.z[t]

    else:

        @m.Constraint(m.t)
        def ode(mm, t):
            return mm.dzdt[t] == (mm.u[t] - mm.z[t]) / mm.L

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == 10 * (mm.z[t] - mm.z_ss) ** 2 + (mm.u[t] - mm.u_ss) ** 2

    @m.Constraint()
    def init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    drto.steady_state(m.z, m.z_ss)
    drto.steady_state_control(m.u, m.u_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=2, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


@needs_ipopt
def test_a_scaled_derivative_side_dilates_like_the_bare_form():
    # ``2*dz/dt == u - z`` and ``dz/dt == (u - z)/2`` are the same
    # dynamics, so the coefficient must ride the dilated tail derivative
    # too (gh #51): both forms solve to the same trajectory, tail included
    results = []
    for scaled in (True, False):
        m = _first_order(scaled)
        pyo.TransformationFactory(IH).apply_to(m)
        pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
        res = drto.scaling.solver_by_name("ipopt").solve(m)
        assert drto.scaling.solved_to_optimality(res)
        results.append(
            [pyo.value(m.z[t]) for t in m.t]
            + [pyo.value(m.drto_ih.z[s]) for s in sorted(m.drto_ih.tau)]
        )
    for a, b in zip(*results):
        assert a == pytest.approx(b, abs=1e-8)


def spatial_block_model(flat=False):
    """The linear model with its gain routed through a Block(t, s) family,
    two members per time point, the minimal shape of an IDAES
    multi-element unit (a stage-indexed state block, a spatial node).
    ``flat=True`` builds the same physics with flat algebra, the twin for
    the equivalence assertion."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.s = pyo.Set(initialize=[1, 2])
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.cost = pyo.Var(m.t)

    if flat:
        m.y = pyo.Var(m.t, m.s, initialize=0.4)

        @m.Constraint(m.t, m.s)
        def gain(mm, t, s):
            return mm.y[t, s] == (1.0 + s) * mm.z[t]

    else:

        def props_rule(blk, t, s):
            mm = blk.model()
            blk.y = pyo.Var(initialize=0.4)
            blk.gain = pyo.Constraint(expr=blk.y == (1.0 + s) * mm.z[t])

        m.props = pyo.Block(m.t, m.s, rule=props_rule)

    @m.Constraint(m.t)
    def ode(mm, t):
        if flat:
            y1, y2 = mm.y[t, 1], mm.y[t, 2]
        else:
            y1, y2 = mm.props[t, 1].y, mm.props[t, 2].y
        return mm.dz[t] == mm.u[t] - 0.5 * (y1 + y2)

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.z[t] - mm.z_ss) ** 2 + 0.01 * (mm.u[t] - 0.5) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def test_multi_index_block_members_replicate_per_coordinate():
    # a Block(t, s) family is one Block(t)-like family per non-time
    # coordinate: each gets its own segment copies and replicated rows
    m = spatial_block_model()
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    assert hasattr(b, "props_1_y") and hasattr(b, "props_2_y")
    assert hasattr(b, "props_1_gain") and hasattr(b, "props_2_gain")
    log = drto.info(m)._transformations[-1]["outcome"]
    assert "1 time-indexed Block(s): 2 components" in log["blocks"]


@needs_ipopt
def test_multi_index_blocks_solve_like_the_flat_twin():
    sols = []
    for flat in (False, True):
        m = spatial_block_model(flat=flat)
        pyo.TransformationFactory(IH).apply_to(m)
        pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
        res = drto.scaling.solver_by_name("ipopt").solve(m)
        assert drto.scaling.solved_to_optimality(res)
        sols.append([pyo.value(m.z[t]) for t in sorted(m.t)])
    for za, zb in zip(*sols):
        assert za == pytest.approx(zb, abs=1e-8)


def test_a_spatial_discretization_row_replicates_as_algebra():
    # a DerivativeVar over a spatial ContinuousSet is ordinary algebra:
    # its members copy to the segment and its discretization equation,
    # despite the '_disc_eq' name, replicates with them. Dropping either
    # half leaves free variables on the tail, which the guard rejects
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.x = ContinuousSet(initialize=[0, 1])
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.cost = pyo.Var(m.t)
    m.w = pyo.Var(m.t, m.x, initialize=0.2)
    m.dwdx = DerivativeVar(m.w, wrt=m.x)

    @m.Constraint(m.t, m.x)
    def w_def(mm, t, x):
        return mm.w[t, x] == (1.0 + x) * mm.z[t]

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == mm.u[t] - mm.z[t] - 0.1 * mm.dwdx[t, 1]

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.z[t] - mm.z_ss) ** 2 + 0.01 * (mm.u[t] - 0.5) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("dae.finite_difference").apply_to(
        m, wrt=m.x, nfe=1, scheme="BACKWARD"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    assert b.component("dwdx_members") is not None
    assert b.component("dwdx_disc_eq") is not None


def test_same_named_components_get_fresh_segment_names():
    # two sub-units both carrying a Var and a row named 'y'/'close': the
    # segment keeps one copy per component under distinct names instead
    # of colliding (the two settlers of a mixer-settler)
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 5, 1))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.cost = pyo.Var(m.t)
    m.left = pyo.Block()
    m.right = pyo.Block()
    for unit, gain in ((m.left, 2.0), (m.right, 3.0)):
        unit.y = pyo.Var(m.t, initialize=0.4)

        def close_rule(_, t, _g=gain, _u=unit):
            return _u.y[t] == _g * m.z[t]

        unit.close = pyo.Constraint(m.t, rule=close_rule)

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == mm.u[t] - 0.5 * (mm.left.y[t] + mm.right.y[t])

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.z[t] - mm.z_ss) ** 2 + 0.01 * (mm.u[t] - 0.5) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(IH).apply_to(m)
    b = m.drto_ih
    copies = [v.name for v in b.component_objects(pyo.Var) if v.local_name != "u"]
    ys = [n for n in copies if "y" in n.split(".")[-1]]
    assert len(ys) >= 2, ys
