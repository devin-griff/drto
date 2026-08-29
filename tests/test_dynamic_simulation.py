# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 007: the dynamic simulation mode."""
import pyomo.environ as pyo
import pytest

import drto
from test_declarations import base_model, declared_model
from test_dynamic_optimization import estimation_model
from test_infinite_horizon import ready_model

ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")

DS = "drto.dynamic_simulation"


def sim_model():
    """A ready model whose controls hold values to be fixed at.

    ``declared_model`` builds its control without ``initialize``, and the
    transformation raises on a control holding no value.
    """
    m = ready_model()
    for vd in m.u.values():
        vd.set_value(0.5)
    return m


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_requires_the_declarations():
    m = base_model()
    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    # a forward integration is not square without the initial state pinned
    with pytest.raises(ValueError, match="Missing: initial_condition"):
        pyo.TransformationFactory(DS).apply_to(m)


def test_unknown_control_errors():
    m = sim_model()
    with pytest.raises(ValueError, match="not a declared control"):
        pyo.TransformationFactory(DS).apply_to(m, controls={"nope": 0.5})


def test_wrong_length_sequence_errors():
    m = sim_model()
    with pytest.raises(ValueError, match="free points"):
        pyo.TransformationFactory(DS).apply_to(m, controls={"u": [0.1, 0.2]})


def test_control_with_no_value_errors():
    m = sim_model()
    for vd in m.u.values():
        vd.set_value(None)
    with pytest.raises(ValueError, match="has none"):
        pyo.TransformationFactory(DS).apply_to(m)


# ----------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------
def test_controls_are_fixed_after_the_profile_is_applied():
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m)
    # piecewise constant. The profile is applied, then its free points fix
    assert len(m.u) == 4
    assert all(vd.fixed for vd in m.u.values())


def test_zero_objective_is_installed():
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m)
    obj = m.component("drto_objective")
    assert obj is not None
    assert pyo.value(obj) == 0.0


def test_costs_leave_the_model():
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m)
    assert m.component("stage") is None
    assert not drto.info(m).has_declaration("tracking_stage_cost")


def test_terminal_constraint_is_shed():
    # a terminal constraint is an optimization construct. A simulation would
    # be over-constrained by it
    m = declared_model()

    @m.Constraint()
    def term_set(m):
        return m.z[m.t.last()] <= 1

    drto.terminal_constraint(m.term_set)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    for vd in m.u.values():
        vd.set_value(0.5)
    pyo.TransformationFactory(DS).apply_to(m)
    assert m.component("term_set") is None
    assert not drto.info(m).has_declaration("terminal_constraint")


def test_steady_state_records_are_kept():
    # the target Params stay on the model (they may appear in a deviation-form
    # model's equations), so their records stay with them
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m)
    reg = drto.info(m)
    assert reg.has_declaration("steady_state")
    assert reg.has_declaration("steady_state_control")
    assert m.component("z_ss") is not None


def test_horizon_is_kept():
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m)
    assert len(m.t) > 1
    assert drto.info(m).has_declaration("horizon")


def test_application_is_recorded():
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m)
    reg = drto.info(m)
    assert reg.has_transformation(DS)
    assert reg.transformations[-1]["outcome"]["horizon"] == "kept"


def test_create_using_leaves_the_source_alone():
    m = sim_model()
    m2 = pyo.TransformationFactory(DS).create_using(m)
    assert m2.component("drto_objective") is not None
    assert m.component("drto_objective") is None
    assert not any(vd.fixed for vd in m.u.values())
    assert drto.info(m2).has_transformation(DS)
    assert not drto.info(m).has_transformation(DS)


# ----------------------------------------------------------------------
# the controls option
# ----------------------------------------------------------------------
def test_a_constant_is_held_across_the_horizon():
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m, controls={"u": 0.9})
    assert [pyo.value(m.u[i]) for i in sorted(m.u)] == [0.9] * 4


def test_a_nested_control_component_resolves():
    # parameterizing replaces the control component, detaching the mapping's
    # key, whose name then degrades to its local name. Resolution happens
    # while the key is still attached, so a control below the top level
    # passes as a component
    from pyomo.dae import ContinuousSet, DerivativeVar

    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1, 2])
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.sub = pyo.Block()
    m.sub.u = pyo.Var(m.t, initialize=0.3)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == mm.sub.u[t] - mm.z[t]

    @m.Constraint()
    def init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.sub.u, profile="piecewise_constant")
    drto.initial_condition(m.init)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=2, ncp=2, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory(DS).apply_to(m, controls={m.sub.u: 0.7})
    assert all(vd.fixed and pyo.value(vd) == 0.7 for vd in m.sub.u.values())


def test_one_value_per_free_point():
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m, controls={"u": [0.2, 0.4, 0.6, 0.8]})
    assert [pyo.value(m.u[i]) for i in sorted(m.u)] == [0.2, 0.4, 0.6, 0.8]


def test_controls_hold_their_values_when_nothing_is_supplied():
    m = sim_model()
    for vd in m.u.values():
        vd.set_value(0.25)
    pyo.TransformationFactory(DS).apply_to(m)
    assert all(pyo.value(vd) == 0.25 for vd in m.u.values())


# ----------------------------------------------------------------------
# the estimation neutralization, shared with feature 006
# ----------------------------------------------------------------------
def test_estimation_declarations_are_neutralized():
    m = estimation_model()
    pyo.TransformationFactory(DS).apply_to(m)
    reg = drto.info(m)
    for kind in ("estimation_stage_cost", "arrival_cost", "measurement"):
        assert not reg.has_declaration(kind), kind
    assert m.component("y_meas") is None
    # the disturbance stays, fixed at its realization (zero by default)
    assert reg.has_declaration("disturbance")
    assert all(vd.fixed and pyo.value(vd) == 0 for vd in m.w.values())
    # the estimated parameter stays a live coefficient, so it keeps its record
    assert reg.components("estimated_parameter") == (m.k,)
    assert m.k.fixed


def test_disturbance_realization_drives_the_plant():
    # the plant is stepped with a supplied noise realization, held across the
    # horizon. The disturbance is fixed at it
    m = estimation_model()
    pyo.TransformationFactory(DS).apply_to(m, disturbances={"w": 0.1})
    assert all(vd.fixed and pyo.value(vd) == 0.1 for vd in m.w.values())


def test_disturbance_realization_per_free_point():
    m = estimation_model()
    pyo.TransformationFactory(DS).apply_to(m, disturbances={"w": [0.1, 0.2, 0.3, 0.4]})
    assert [pyo.value(m.w[i]) for i in sorted(m.w)] == [0.1, 0.2, 0.3, 0.4]


def test_unfixed_disturbance_defaults_to_zero():
    m = estimation_model()
    pyo.TransformationFactory(DS).apply_to(m)  # no realization supplied
    assert all(vd.fixed and pyo.value(vd) == 0 for vd in m.w.values())


def test_unknown_disturbance_errors():
    m = estimation_model()
    with pytest.raises(ValueError, match="not a declared disturbance"):
        pyo.TransformationFactory(DS).apply_to(m, disturbances={"nope": 0.1})


# ----------------------------------------------------------------------
# the numbers
# ----------------------------------------------------------------------
@needs_ipopt
def test_integrates_to_the_equilibrium():
    # dz/dt = -z + u with u held: the state settles at u
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m, controls={"u": 0.9})
    r = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(r)
    assert pyo.value(m.z[0]) == pytest.approx(0.4, abs=1e-6)  # the pinned start
    assert pyo.value(m.z[m.t.last()]) == pytest.approx(0.9, abs=1e-3)


@needs_ipopt
def test_the_estimation_model_simulates():
    m = estimation_model()
    pyo.TransformationFactory(DS).apply_to(m)
    r = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(r)
