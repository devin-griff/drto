# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 003: objective assembly (drto.build_objective)."""
import pyomo.environ as pyo
import pytest
from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables

import drto
from test_declarations import declared_model


def cost_vars(obj):
    return ComponentSet(identify_variables(obj.expr))


def test_bare_call_assembles_the_live_stage_cost():
    m = declared_model()
    obj = drto.build_objective(m)
    assert obj.active and obj.sense == pyo.minimize
    # one cost var per time point except the final one
    expected = ComponentSet(m.cost[t] for t in m.t if t != m.t.last())
    assert cost_vars(obj) == expected


def test_stage_sum_stays_at_the_samples_after_discretization():
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    obj = drto.build_objective(m)
    # cost members now exist at every collocation point, but only the
    # sample grid captured by horizon enters the sum
    expected = ComponentSet(m.cost[t] for t in (0, 2.5, 5, 7.5))
    assert cost_vars(obj) == expected


def test_terminal_cost_var_joins_the_sum():
    m = declared_model()
    m.term = pyo.Var()

    @m.Constraint()
    def term_def(m):
        return m.term == 10 * (m.z[10] - m.z_ss) ** 2

    drto.tracking_terminal_cost(m.term_def)
    obj = drto.build_objective(m)
    assert m.term in cost_vars(obj)


def test_group_weight_scales_the_group():
    m = declared_model()
    reg = drto.info(m)
    (record,) = reg.declarations("tracking_stage_cost")
    record["weight"] = 10.0
    obj = drto.build_objective(m)
    for t in m.t:
        if t != m.t.last():
            m.cost[t].set_value(1.0)
    assert pyo.value(obj.expr) == pytest.approx(10.0 * 4)


def test_registered_cost_group_is_included():
    m = declared_model()
    m.phi = pyo.Var(initialize=1.0)
    drto.info(m).record_declaration("cost_group", m.phi, terms=((m.phi, 2.5),))
    obj = drto.build_objective(m)
    assert m.phi in cost_vars(obj)
    for t in m.t:
        m.cost[t].set_value(0.0)
    assert pyo.value(obj.expr) == pytest.approx(2.5)


def test_zero_installs_a_constant_zero_objective():
    m = declared_model()
    obj = drto.build_objective(m, zero=True)
    assert obj.active
    assert pyo.value(obj.expr) == 0.0
    assert len(cost_vars(obj)) == 0


def test_existing_active_objective_is_deactivated():
    m = declared_model()
    m.user_obj = pyo.Objective(expr=m.z[0] ** 2)
    drto.build_objective(m)
    assert not m.user_obj.active
    active = list(m.component_data_objects(pyo.Objective, active=True))
    assert len(active) == 1


def test_repeated_calls_rebuild_one_objective():
    m = declared_model()
    drto.build_objective(m)
    obj = drto.build_objective(m, zero=True)
    active = list(m.component_data_objects(pyo.Objective, active=True))
    assert active == [obj]
    assert pyo.value(obj.expr) == 0.0


def test_deactivated_members_drop_out():
    m = declared_model()
    for t in m.t:
        if t != m.t.last() and t > 4:
            m.stage[t].deactivate()
    obj = drto.build_objective(m)
    assert cost_vars(obj) == ComponentSet([m.cost[0], m.cost[2.5]])


def test_no_live_cost_errors_clearly():
    m = declared_model()
    m.stage.deactivate()
    with pytest.raises(ValueError, match="no live cost terms"):
        drto.build_objective(m)


def test_transform_form_and_zero_option():
    m = declared_model()
    pyo.TransformationFactory("drto.build_objective").apply_to(m)
    assert m.component("drto_objective").active
    pyo.TransformationFactory("drto.build_objective").apply_to(m, zero=True)
    obj = m.component("drto_objective")
    assert pyo.value(obj.expr) == 0.0


def test_create_using_leaves_the_source_unchanged():
    m = declared_model()
    m2 = pyo.TransformationFactory("drto.build_objective").create_using(m)
    assert m2.component("drto_objective") is not None
    assert m.component("drto_objective") is None
    # the clone's objective references the clone's cost vars
    assert m2.cost[0] in cost_vars(m2.component("drto_objective"))


def test_application_is_recorded_in_the_registry():
    m = declared_model()
    drto.build_objective(m)
    reg = drto.info(m)
    assert reg.has_transformation("drto.build_objective")
    assert "cost terms" in reg.transformations[-1]["outcome"]["objective"]


def test_both_cost_kinds_sum_together():
    m = declared_model()
    m.ecost = pyo.Var(m.t)

    @m.Constraint(sorted(m.t)[:-1])
    def econ(m, t):
        return m.ecost[t] == -m.u[t]

    drto.economic_stage_cost(m.econ)
    obj = drto.build_objective(m)
    expected = ComponentSet(
        [m.cost[t] for t in m.t if t != m.t.last()]
        + [m.ecost[t] for t in m.t if t != m.t.last()]
    )
    assert cost_vars(obj) == expected
