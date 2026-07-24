# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 007: the dynamic simulation mode."""
import pyomo.environ as pyo
import pytest

import drto
from test_declarations import base_model
from test_dynamic_optimization import estimation_model
from test_infinite_horizon import ready_model

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")

DS = "drto.dynamic_simulation"


def sim_model():
    """A ready model whose controls carry values to be fixed at.

    ``declared_model`` builds its control without ``initialize``, and a
    simulation refuses to fix a control holding no value.
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
    with pytest.raises(ValueError, match="missing: initial_condition"):
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
    # piecewise constant: the profile is applied, then its free points fix
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
    for kind in ("estimation_stage_cost", "arrival_cost", "measurement", "disturbance"):
        assert not reg.has_declaration(kind), kind
    assert m.component("w") is None
    assert m.component("y_meas") is None
    # the estimated parameter stays a live coefficient, so it keeps its record
    assert reg.components("estimated_parameter") == (m.k,)
    assert m.k.fixed


# ----------------------------------------------------------------------
# the numbers
# ----------------------------------------------------------------------
@needs_ipopt
def test_integrates_to_the_equilibrium():
    # dz/dt = -z + u with u held: the state settles at u
    m = sim_model()
    pyo.TransformationFactory(DS).apply_to(m, controls={"u": 0.9})
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.z[0]) == pytest.approx(0.4, abs=1e-6)  # the pinned start
    assert pyo.value(m.z[m.t.last()]) == pytest.approx(0.9, abs=1e-3)


@needs_ipopt
def test_the_estimation_model_simulates():
    m = estimation_model()
    pyo.TransformationFactory(DS).apply_to(m)
    r = pyo.SolverFactory("ipopt").solve(m)
    assert r.solver.termination_condition == pyo.TerminationCondition.optimal
