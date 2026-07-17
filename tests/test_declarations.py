# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 002: the dynamic optimization and simulation declarations."""
import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def base_model():
    """The feature 002 example model, before any declarations."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10), initialize=[0, 2.5, 5, 7.5, 10])
    m.z = pyo.Var(m.t)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1))

    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)

    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t]

    @m.Constraint(m.t)
    def stage(m, t):
        return m.cost[t] == 10 * (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    return m


def declared_model():
    """The feature 002 example, fully declared."""
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_state(m.z)
    drto.declare_continuous_dynamics(m.ode)
    drto.declare_control(m.u, profile="piecewise_constant")
    drto.declare_tracking_stage_cost(m.stage)
    drto.declare_initial_condition(m.init)
    drto.declare_steady_state(m.z_ss)
    drto.declare_steady_state_control(m.u_ss)
    return m


# ----------------------------------------------------------------------
# the happy path
# ----------------------------------------------------------------------
def test_full_surface_records_in_the_registry():
    m = declared_model()
    reg = drto.info(m)
    assert reg.components("time") == (m.t,)
    assert reg.components("state") == (m.z,)
    assert reg.components("continuous_dynamics") == (m.ode,)
    assert reg.components("control") == (m.u,)
    assert reg.components("tracking_stage_cost") == (m.stage,)
    assert reg.components("initial_condition") == (m.init,)
    assert reg.components("steady_state") == (m.z_ss,)
    assert reg.components("steady_state_control") == (m.u_ss,)
    (control,) = reg.declarations("control")
    assert control["profile"] == "piecewise_constant"


def test_control_profile_reaches_pyomo_cvp():
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    # The final stage-cost member references the control at the final time,
    # which a piecewise-constant profile has no value for; its cost term is
    # outside build_objective's 0..N-1 sum, so the mode transforms deactivate
    # it before parameterizing (features 006/007). Mimic that here.
    m.stage[m.t.last()].deactivate()
    pyo.TransformationFactory("cvp.parameterize").apply_to(m)
    # piecewise constant: one free control value per finite element
    assert len(m.u) == 4


# ----------------------------------------------------------------------
# arity and re-declaration
# ----------------------------------------------------------------------
def test_single_object_declarations_take_exactly_one():
    m = base_model()
    m.t2 = ContinuousSet(bounds=(0, 1))
    with pytest.raises(TypeError):
        drto.declare_time(m.t, m.t2)


def test_single_object_redeclaration_errors_on_a_different_object():
    m = base_model()
    m.t2 = ContinuousSet(bounds=(0, 1))
    drto.declare_time(m.t)
    with pytest.raises(ValueError, match="already"):
        drto.declare_time(m.t2)


def test_single_object_redeclaration_of_same_object_is_idempotent():
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_time(m.t)
    assert drto.info(m).components("time") == (m.t,)


def test_varargs_accumulate_across_calls():
    m = base_model()
    m.z2 = pyo.Var(m.t)
    drto.declare_state(m.z)
    drto.declare_state(m.z2)
    assert drto.info(m).components("state") == (m.z, m.z2)


def test_varargs_duplicate_is_rejected():
    m = base_model()
    drto.declare_state(m.z)
    with pytest.raises(ValueError, match="already declared"):
        drto.declare_state(m.z)


def test_member_of_a_container_is_rejected():
    m = base_model()
    with pytest.raises(TypeError, match="whole components"):
        drto.declare_state(m.z[0])


# ----------------------------------------------------------------------
# per-kind validation
# ----------------------------------------------------------------------
def test_time_must_be_a_continuousset():
    m = base_model()
    m.s = pyo.Set(initialize=[1, 2, 3])
    with pytest.raises(TypeError, match="ContinuousSet"):
        drto.declare_time(m.s)


def test_state_must_be_a_var():
    m = base_model()
    with pytest.raises(TypeError, match="expects a Var"):
        drto.declare_state(m.z_ss)


def test_state_without_derivative_is_accepted():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()  # a steady-state model's state: no DerivativeVar
    drto.declare_state(m.z)
    assert drto.info(m).components("state") == (m.z,)


def test_dynamics_requires_time_and_state_first():
    m = base_model()
    with pytest.raises(ValueError, match="declare_time first"):
        drto.declare_continuous_dynamics(m.ode)
    drto.declare_time(m.t)
    with pytest.raises(ValueError, match="declare_state first"):
        drto.declare_continuous_dynamics(m.ode)


def test_dynamics_lhs_must_be_a_derivativevar():
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_state(m.z)
    with pytest.raises(ValueError, match="DerivativeVar"):
        drto.declare_continuous_dynamics(m.stage)


def test_dynamics_accepts_either_orientation():
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_state(m.z)

    @m.Constraint(m.t)
    def ode_flipped(m, t):
        return -m.z[t] + m.u[t] == m.dzdt[t]

    drto.declare_continuous_dynamics(m.ode_flipped)
    assert drto.info(m).components("continuous_dynamics") == (m.ode_flipped,)


def test_dynamics_state_must_be_declared():
    m = base_model()
    m.w = pyo.Var(m.t)
    m.dwdt = DerivativeVar(m.w, wrt=m.t)

    @m.Constraint(m.t)
    def w_ode(m, t):
        return m.dwdt[t] == -m.w[t]

    drto.declare_time(m.t)
    drto.declare_state(m.z)
    with pytest.raises(ValueError, match="not a declared state"):
        drto.declare_continuous_dynamics(m.w_ode)


def test_dynamics_wrt_must_be_the_declared_time():
    m = base_model()
    m.x = ContinuousSet(bounds=(0, 1))  # a spatial axis
    m.T = pyo.Var(m.x)
    m.dTdx = DerivativeVar(m.T, wrt=m.x)

    @m.Constraint(m.x)
    def pde(m, x):
        return m.dTdx[x] == -m.T[x]

    drto.declare_time(m.t)
    drto.declare_state(m.z, m.T)
    with pytest.raises(ValueError, match="declared time set"):
        drto.declare_continuous_dynamics(m.pde)


def test_control_requires_time_first():
    m = base_model()
    with pytest.raises(ValueError, match="declare_time first"):
        drto.declare_control(m.u)


def test_stage_cost_must_be_indexed_by_the_time_set():
    m = base_model()
    drto.declare_time(m.t)

    @m.Constraint()
    def scalar_cost(m):
        return m.cost[0] == m.z[0] ** 2

    with pytest.raises(ValueError, match="indexed by the declared time set"):
        drto.declare_tracking_stage_cost(m.scalar_cost)


def test_stage_cost_must_be_an_equality():
    m = base_model()
    drto.declare_time(m.t)

    @m.Constraint(m.t)
    def bound(m, t):
        return m.cost[t] >= 0

    with pytest.raises(ValueError, match="equality"):
        drto.declare_economic_stage_cost(m.bound)


def test_stage_cost_needs_a_cost_variable_side():
    m = base_model()
    drto.declare_time(m.t)

    @m.Constraint(m.t)
    def no_var(m, t):
        return m.z[t] ** 2 == m.u[t] ** 2

    with pytest.raises(ValueError, match="cost variable"):
        drto.declare_tracking_stage_cost(m.no_var)


def test_terminal_cost_must_be_scalar():
    m = base_model()
    with pytest.raises(ValueError, match="scalar Constraint"):
        drto.declare_tracking_terminal_cost(m.stage)


def test_terminal_cost_happy_path():
    m = base_model()
    m.term = pyo.Var()

    @m.Constraint()
    def term_def(m):
        return m.term == 10 * (m.z[10] - m.z_ss) ** 2

    drto.declare_tracking_terminal_cost(m.term_def)
    assert drto.info(m).components("tracking_terminal_cost") == (m.term_def,)


def test_initial_condition_rhs_must_be_a_mutable_param():
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_state(m.z)

    @m.Constraint()
    def bad_init(m):
        return m.z[0] == 0.4  # a constant, not a mutable Param

    with pytest.raises(ValueError, match="mutable Param"):
        drto.declare_initial_condition(m.bad_init)


def test_initial_condition_state_must_be_at_the_first_point():
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_state(m.z)

    @m.Constraint()
    def mid_init(m):
        return m.z[5] == m.z_hat

    with pytest.raises(ValueError, match="first time point"):
        drto.declare_initial_condition(m.mid_init)


def test_initial_condition_accepts_either_orientation():
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_state(m.z)

    @m.Constraint()
    def init_flipped(m):
        return m.z_hat == m.z[0]

    drto.declare_initial_condition(m.init_flipped)
    assert drto.info(m).components("initial_condition") == (m.init_flipped,)


def test_terminal_constraint_only_final_time_states():
    m = base_model()
    drto.declare_time(m.t)
    drto.declare_state(m.z)

    @m.Constraint()
    def good(m):
        return m.z[10] <= 1

    @m.Constraint()
    def bad(m):
        return m.z[5] + m.u[10] <= 1

    drto.declare_terminal_constraint(m.good)
    assert drto.info(m).components("terminal_constraint") == (m.good,)
    m2 = base_model()
    drto.declare_time(m2.t)
    drto.declare_state(m2.z)

    @m2.Constraint()
    def bad2(m2):
        return m2.z[5] <= 1

    with pytest.raises(ValueError, match="final time point"):
        drto.declare_terminal_constraint(m2.bad2)


def test_steady_state_targets_must_be_mutable_params():
    m = base_model()
    m.frozen = pyo.Param(initialize=0.5)
    with pytest.raises(TypeError, match="expects a Param"):
        drto.declare_steady_state(m.z)
    with pytest.raises(ValueError, match="mutable"):
        drto.declare_steady_state(m.frozen)


def test_declarations_render_in_the_registry_view():
    m = declared_model()
    text = repr(drto.info(m))
    assert "controls: u (piecewise_constant, free)" in text
    assert "dynamics: dzdt[k]" in text
    assert "steady-state targets: z_ss" in text
