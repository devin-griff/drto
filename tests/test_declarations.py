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
    N, h = 4, 2.5  # samples and sampling time
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z = pyo.Var(m.t)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1))

    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.u_ss = pyo.Param(initialize=0.5, mutable=True)  # = z_ss: a true equilibrium pair
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)

    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t]

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == 10 * (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    return m


def declared_model():
    """The feature 002 example, fully declared."""
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    drto.steady_state(m.z, m.z_ss)
    drto.steady_state_control(m.u, m.u_ss)
    return m


def wrapped_model():
    """The feature 002 wrapping example: declared as the model is written."""
    m = pyo.ConcreteModel()
    N, h = 4, 2.5  # samples and sampling time
    m.t = drto.horizon(ContinuousSet(initialize=pyo.RangeSet(0, N * h, h)))
    m.z = drto.state(pyo.Var(m.t))
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = drto.control(pyo.Var(m.t, bounds=(0, 1)), profile="piecewise_constant")

    m.z_ss = drto.steady_state(m.z, pyo.Param(initialize=0.5, mutable=True))
    m.u_ss = drto.steady_state_control(m.u, pyo.Param(initialize=0.5, mutable=True))
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)

    m.cost = pyo.Var(m.t)

    @drto.dynamics(m, m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t]

    @drto.tracking_stage_cost(m, sorted(m.t)[:-1])
    def stage(m, t):
        return m.cost[t] == 10 * (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

    @drto.initial_condition(m)
    def init(m):
        return m.z[0] == m.z_hat

    return m


# ----------------------------------------------------------------------
# the happy path
# ----------------------------------------------------------------------
def test_full_surface_records_in_the_registry():
    m = declared_model()
    reg = drto.info(m)
    assert reg.components("horizon") == (m.t,)
    assert reg.components("state") == (m.z,)
    assert reg.components("dynamics") == (m.ode,)
    assert reg.components("control") == (m.u,)
    assert reg.components("tracking_stage_cost") == (m.stage,)
    assert reg.components("initial_condition") == (m.init,)
    assert reg.components("steady_state") == (m.z_ss,)
    assert reg.components("steady_state_control") == (m.u_ss,)
    (control,) = reg.declarations("control")
    assert control["profile"] == "piecewise_constant"


def test_control_profile_reaches_pyomo_cvp():
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU")
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
        drto.horizon(m.t, m.t2)


def test_single_object_redeclaration_errors_on_a_different_object():
    m = base_model()
    m.t2 = ContinuousSet(bounds=(0, 1))
    drto.horizon(m.t)
    with pytest.raises(ValueError, match="already"):
        drto.horizon(m.t2)


def test_single_object_redeclaration_of_same_object_is_idempotent():
    m = base_model()
    drto.horizon(m.t)
    drto.horizon(m.t)
    assert drto.info(m).components("horizon") == (m.t,)


def test_varargs_accumulate_across_calls():
    m = base_model()
    m.z2 = pyo.Var(m.t)
    drto.state(m.z)
    drto.state(m.z2)
    assert drto.info(m).components("state") == (m.z, m.z2)


def test_varargs_duplicate_is_rejected():
    m = base_model()
    drto.state(m.z)
    with pytest.raises(ValueError, match="already declared"):
        drto.state(m.z)


def test_member_of_a_container_is_rejected():
    m = base_model()
    with pytest.raises(TypeError, match="whole components"):
        drto.state(m.z[0])


# ----------------------------------------------------------------------
# per-kind validation
# ----------------------------------------------------------------------
def test_time_must_be_a_continuousset():
    m = base_model()
    m.s = pyo.Set(initialize=[1, 2, 3])
    with pytest.raises(TypeError, match="ContinuousSet"):
        drto.horizon(m.s)


def test_state_must_be_a_var():
    m = base_model()
    with pytest.raises(TypeError, match="expects a Var"):
        drto.state(m.z_ss)


def test_state_without_derivative_is_accepted():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()  # a steady-state model's state: no DerivativeVar
    drto.state(m.z)
    assert drto.info(m).components("state") == (m.z,)


def test_dynamics_requires_time_and_state_first():
    m = base_model()
    with pytest.raises(ValueError, match="horizon.*first"):
        drto.dynamics(m.ode)
    drto.horizon(m.t)
    with pytest.raises(ValueError, match="state first"):
        drto.dynamics(m.ode)


def test_dynamics_lhs_must_be_a_derivativevar():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    with pytest.raises(ValueError, match="DerivativeVar"):
        drto.dynamics(m.stage)


def test_dynamics_accepts_either_orientation():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    @m.Constraint(m.t)
    def ode_flipped(m, t):
        return -m.z[t] + m.u[t] == m.dzdt[t]

    drto.dynamics(m.ode_flipped)
    assert drto.info(m).components("dynamics") == (m.ode_flipped,)


def test_dynamics_state_must_be_declared():
    m = base_model()
    m.w = pyo.Var(m.t)
    m.dwdt = DerivativeVar(m.w, wrt=m.t)

    @m.Constraint(m.t)
    def w_ode(m, t):
        return m.dwdt[t] == -m.w[t]

    drto.horizon(m.t)
    drto.state(m.z)
    with pytest.raises(ValueError, match="not a declared state"):
        drto.dynamics(m.w_ode)


def test_dynamics_wrt_must_be_the_declared_time():
    m = base_model()
    m.x = ContinuousSet(bounds=(0, 1))  # a spatial axis
    m.T = pyo.Var(m.x)
    m.dTdx = DerivativeVar(m.T, wrt=m.x)

    @m.Constraint(m.x)
    def pde(m, x):
        return m.dTdx[x] == -m.T[x]

    drto.horizon(m.t)
    drto.state(m.z, m.T)
    with pytest.raises(ValueError, match="declared time set"):
        drto.dynamics(m.pde)


def test_control_requires_time_first():
    m = base_model()
    with pytest.raises(ValueError, match="horizon.*first"):
        drto.control(m.u)


def test_stage_cost_must_be_indexed_by_the_time_set():
    m = base_model()
    drto.horizon(m.t)

    @m.Constraint()
    def scalar_cost(m):
        return m.cost[0] == m.z[0] ** 2

    with pytest.raises(ValueError, match="one member per sample"):
        drto.tracking_stage_cost(m.scalar_cost)


def test_stage_cost_must_be_an_equality():
    m = base_model()
    drto.horizon(m.t)

    @m.Constraint(sorted(m.t)[:-1])
    def bound(m, t):
        return m.cost[t] >= 0

    with pytest.raises(ValueError, match="equality"):
        drto.economic_stage_cost(m.bound)


def test_stage_cost_needs_a_cost_variable_side():
    m = base_model()
    drto.horizon(m.t)

    @m.Constraint(sorted(m.t)[:-1])
    def no_var(m, t):
        return m.z[t] ** 2 == m.u[t] ** 2

    with pytest.raises(ValueError, match="cost variable"):
        drto.tracking_stage_cost(m.no_var)


def test_stage_cost_must_skip_the_final_time():
    m = base_model()
    drto.horizon(m.t)

    @m.Constraint(m.t)
    def full_span(m, t):
        return m.cost[t] == m.z[t] ** 2

    with pytest.raises(ValueError, match="one member per sample"):
        drto.tracking_stage_cost(m.full_span)


def test_terminal_cost_must_be_scalar():
    m = base_model()
    with pytest.raises(ValueError, match="scalar Constraint"):
        drto.tracking_terminal_cost(m.stage)


def test_terminal_cost_happy_path():
    m = base_model()
    m.term = pyo.Var()

    @m.Constraint()
    def term_def(m):
        return m.term == 10 * (m.z[10] - m.z_ss) ** 2

    drto.tracking_terminal_cost(m.term_def)
    assert drto.info(m).components("tracking_terminal_cost") == (m.term_def,)


def test_initial_condition_rhs_must_be_a_mutable_param():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    @m.Constraint()
    def bad_init(m):
        return m.z[0] == 0.4  # a constant, not a mutable Param

    with pytest.raises(ValueError, match="mutable Param"):
        drto.initial_condition(m.bad_init)


def test_initial_condition_state_must_be_at_the_first_point():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    @m.Constraint()
    def mid_init(m):
        return m.z[5] == m.z_hat

    with pytest.raises(ValueError, match="first time point"):
        drto.initial_condition(m.mid_init)


def test_initial_condition_accepts_either_orientation():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    @m.Constraint()
    def init_flipped(m):
        return m.z_hat == m.z[0]

    drto.initial_condition(m.init_flipped)
    assert drto.info(m).components("initial_condition") == (m.init_flipped,)


def test_terminal_constraint_only_final_time_states():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    @m.Constraint()
    def good(m):
        return m.z[10] <= 1

    @m.Constraint()
    def bad(m):
        return m.z[5] + m.u[10] <= 1

    drto.terminal_constraint(m.good)
    assert drto.info(m).components("terminal_constraint") == (m.good,)
    m2 = base_model()
    drto.horizon(m2.t)
    drto.state(m2.z)

    @m2.Constraint()
    def bad2(m2):
        return m2.z[5] <= 1

    with pytest.raises(ValueError, match="final time point"):
        drto.terminal_constraint(m2.bad2)


def test_steady_state_targets_must_be_mutable_params():
    m = base_model()
    m.frozen = pyo.Param(initialize=0.5)
    drto.state(m.z)
    with pytest.raises(TypeError, match="expects a Param"):
        drto.steady_state(m.z, m.u)
    with pytest.raises(ValueError, match="mutable"):
        drto.steady_state(m.z, m.frozen)


def test_steady_state_requires_a_declared_state():
    m = base_model()
    with pytest.raises(ValueError, match="not a declared state"):
        drto.steady_state(m.z, m.z_ss)


def test_steady_state_control_requires_a_declared_control():
    m = base_model()
    drto.horizon(m.t)
    with pytest.raises(ValueError, match="not a declared control"):
        drto.steady_state_control(m.u, m.u_ss)


def test_steady_state_pairing_is_recorded():
    m = declared_model()
    (rec,) = drto.info(m).declarations("steady_state")
    assert rec["component"] is m.z_ss
    assert rec["of"] is m.z
    (rec,) = drto.info(m).declarations("steady_state_control")
    assert rec["component"] is m.u_ss
    assert rec["of"] is m.u


def test_steady_state_pair_is_idempotent_and_returns_the_target():
    m = declared_model()
    assert drto.steady_state(m.z, m.z_ss) is m.z_ss
    assert drto.info(m).components("steady_state") == (m.z_ss,)


def test_steady_state_second_target_for_a_state_is_rejected():
    m = declared_model()
    m.z_ss2 = pyo.Param(initialize=0.6, mutable=True)
    with pytest.raises(ValueError, match="already has the target"):
        drto.steady_state(m.z, m.z_ss2)


def test_steady_state_target_cannot_serve_two_states():
    m = declared_model()
    m.z2 = pyo.Var(m.t)
    drto.state(m.z2)
    with pytest.raises(ValueError, match="already declared"):
        drto.steady_state(m.z2, m.z_ss)


# ----------------------------------------------------------------------
# wrapping and the decorators
# ----------------------------------------------------------------------
def test_wrapped_model_matches_the_tagged_one():
    m = wrapped_model()
    reg = drto.info(m)
    assert reg.components("horizon") == (m.t,)
    assert reg.components("state") == (m.z,)
    assert reg.components("dynamics") == (m.ode,)
    assert reg.components("control") == (m.u,)
    assert reg.components("tracking_stage_cost") == (m.stage,)
    assert reg.components("initial_condition") == (m.init,)
    assert reg.components("steady_state") == (m.z_ss,)
    assert reg.components("steady_state_control") == (m.u_ss,)
    (control,) = reg.declarations("control")
    assert control["profile"] == "piecewise_constant"
    (target,) = reg.declarations("steady_state")
    assert target["of"] is m.z
    # the sample grid was captured at attachment
    (hor,) = reg.declarations("horizon")
    assert hor["samples"] == (0, 2.5, 5, 7.5, 10)


def test_wrapping_registers_at_attachment_not_before():
    m = pyo.ConcreteModel()
    fresh = pyo.Var()
    wrapped = drto.state(fresh)
    assert wrapped is fresh
    assert not drto.info(m).has_declaration("state")
    m.z = wrapped
    assert drto.info(m).components("state") == (m.z,)


def test_wrapping_takes_exactly_one_component():
    with pytest.raises(TypeError, match="exactly one component"):
        drto.state(pyo.Var(), pyo.Var())


def test_wrapping_checks_prerequisites_at_attachment():
    m = pyo.ConcreteModel()
    with pytest.raises(ValueError, match="horizon"):
        m.u = drto.control(pyo.Var())


def test_a_set_where_a_component_belongs_is_a_type_error():
    m = base_model()
    with pytest.raises(TypeError, match="expects a Var"):
        drto.state(m.t)


def test_decorator_returns_the_attached_constraint():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    @drto.dynamics(m, m.t)
    def ode2(m, t):
        return m.dzdt[t] == -m.z[t]

    assert ode2 is m.ode2
    assert drto.info(m).components("dynamics") == (m.ode2,)


def test_decorator_validation_still_applies():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    with pytest.raises(ValueError, match="one member per sample"):

        @drto.tracking_stage_cost(m, m.t)  # spans the final time
        def bad_stage(m, t):
            return m.cost[t] == m.z[t] ** 2


def test_styles_mix_per_component():
    # decorator constraints over tagged variables (the spec's example mix)
    m = pyo.ConcreteModel()
    N, h = 4, 2.5  # samples and sampling time
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    drto.horizon(m.t)
    m.z = pyo.Var(m.t)
    drto.state(m.z)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1))
    drto.control(m.u)

    @drto.dynamics(m, m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t]

    reg = drto.info(m)
    assert reg.components("dynamics") == (m.ode,)
    assert reg.components("control") == (m.u,)


def test_declarations_render_in_the_registry_view():
    m = declared_model()
    text = repr(drto.info(m))
    assert "controls: u (piecewise_constant, free)" in text
    assert "dynamics: dzdt[k]" in text
    assert "steady-state targets: z_ss (of z)" in text


# ----------------------------------------------------------------------
# review hardening: scalar components, guards, and the style corners
# ----------------------------------------------------------------------
def scalar_pair_model():
    """A steady-state model: two scalar states with targets."""
    m = pyo.ConcreteModel()
    m.a = pyo.Var()
    m.b = pyo.Var()
    m.a_ss = pyo.Param(initialize=0.5, mutable=True)
    m.b_ss = pyo.Param(initialize=0.3, mutable=True)
    return m


def test_scalar_states_accumulate():
    # scalar Vars overload ==, so membership must be by identity
    m = scalar_pair_model()
    drto.state(m.a)
    drto.state(m.b)
    assert drto.info(m).components("state") == (m.a, m.b)


def test_scalar_states_pair_with_targets():
    m = scalar_pair_model()
    drto.state(m.a, m.b)
    drto.steady_state(m.a, m.a_ss)
    drto.steady_state(m.b, m.b_ss)
    assert drto.info(m).components("steady_state") == (m.a_ss, m.b_ss)


def test_undeclared_scalar_owner_gets_the_clear_error():
    m = scalar_pair_model()
    drto.state(m.a)
    with pytest.raises(ValueError, match="not a declared state"):
        drto.steady_state(m.b, m.b_ss)


def test_horizon_rejects_a_discretized_set():
    m = base_model()
    pyo.TransformationFactory("dae.finite_difference").apply_to(m, wrt=m.t, nfe=4)
    with pytest.raises(ValueError, match="discretized"):
        drto.horizon(m.t)


def test_varargs_reject_components_from_another_model():
    m1, m2 = base_model(), base_model()
    with pytest.raises(ValueError, match="different model"):
        drto.state(m1.z, m2.z)


def test_target_cannot_serve_both_kinds():
    m = declared_model()
    m.z2 = pyo.Var(m.t)
    drto.state(m.z2)
    with pytest.raises(ValueError, match="already declared"):
        drto.steady_state(m.z2, m.u_ss)  # u_ss is a control target already


def test_double_wrapping_the_same_component_errors():
    v = pyo.Var()
    drto.state(v)
    with pytest.raises(ValueError, match="already wrapped"):
        drto.state(v)


def test_wrapping_on_an_abstract_model_errors_clearly():
    am = pyo.AbstractModel()
    am.t = drto.horizon(ContinuousSet(initialize=range(3)))
    with pytest.raises(ValueError, match="AbstractModel"):
        am.create_instance()


def test_dynamics_wrt_rejects_a_same_grid_spatial_axis():
    # a spatial axis holding the same points must not pass by set equality
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 0.5, 1])
    m.x = ContinuousSet(initialize=[0, 0.5, 1])
    m.T = pyo.Var(m.x)
    m.dTdx = DerivativeVar(m.T, wrt=m.x)

    @m.Constraint(m.x)
    def pde(m, x):
        return m.dTdx[x] == -m.T[x]

    drto.horizon(m.t)
    drto.state(m.T)
    with pytest.raises(ValueError, match="declared time set"):
        drto.dynamics(m.pde)


def test_terminal_cost_and_terminal_constraint_decorators():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    m.term = pyo.Var()

    @drto.tracking_terminal_cost(m)
    def term_def(m):
        return m.term == 10 * (m.z[10] - m.z_ss) ** 2

    @drto.terminal_constraint(m)
    def term_set(m):
        return m.z[10] <= 1

    reg = drto.info(m)
    assert reg.components("tracking_terminal_cost") == (m.term_def,)
    assert reg.components("terminal_constraint") == (m.term_set,)


def test_economic_stage_cost_decorator():
    m = base_model()
    drto.horizon(m.t)
    m.ecost = pyo.Var(m.t)

    @drto.economic_stage_cost(m, sorted(m.t)[:-1])
    def econ(m, t):
        return m.ecost[t] == -m.u[t]

    assert drto.info(m).components("economic_stage_cost") == (m.econ,)


def test_decorator_passes_keywords_through_to_constraint():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    @drto.dynamics(m, m.t, doc="the ODE")
    def ode3(m, t):
        return m.dzdt[t] == -m.z[t]

    assert m.ode3.doc == "the ODE"


def test_keywords_outside_the_decorator_form_error():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    with pytest.raises(TypeError, match="decorator form"):
        drto.dynamics(m.ode, doc="nope")


def test_wrapping_a_fresh_constraint():
    # the wrap form needs a detached Constraint, which only rule= can build;
    # written-out models use @m.Constraint or the drto decorators instead
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)

    def ode4_rule(m_, t):
        return m.dzdt[t] == -m.z[t]

    m.ode4 = drto.dynamics(pyo.Constraint(m.t, rule=ode4_rule))
    assert drto.info(m).components("dynamics") == (m.ode4,)
