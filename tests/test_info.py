# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 001: the drto registry (drto.info)."""
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def declared_model():
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10), initialize=[0, 2.5, 5, 7.5, 10])
    m.z = pyo.Var(m.t)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1))

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t]

    reg = drto.info(m)
    reg.record_declaration("horizon", m.t)
    reg.record_declaration("state", m.z)
    reg.record_declaration("dynamics", m.ode)
    reg.record_declaration("control", m.u, profile="piecewise_constant")
    return m


def test_created_once_and_returned_again():
    m = pyo.ConcreteModel()
    reg = drto.info(m)
    assert isinstance(reg, drto.Info)
    assert drto.info(m) is reg


def test_backed_by_private_data_not_a_component():
    m = pyo.ConcreteModel()
    n_before = len(list(m.component_objects()))
    reg = drto.info(m)
    assert len(list(m.component_objects())) == n_before
    # stored under the 'drto' private_data scope; only drto's own modules can
    # call m.private_data('drto') (Pyomo enforces the caller's module name),
    # so assert through the underlying store
    assert m._private_data["drto"]["info"] is reg


def test_declarations_recorded_and_read_back():
    m = declared_model()
    reg = drto.info(m)
    assert reg.components("state") == (m.z,)
    assert reg.components("control") == (m.u,)
    assert reg.has_declaration("horizon")
    assert not reg.has_declaration("terminal_constraint")
    assert reg.components("terminal_constraint") == ()
    (control,) = reg.declarations("control")
    assert control["profile"] == "piecewise_constant"
    assert set(reg.declarations()) == {"horizon", "state", "dynamics", "control"}


def test_transformation_log_is_ordered_and_queryable():
    m = pyo.ConcreteModel()
    reg = drto.info(m)
    assert reg.transformations == ()
    reg.record_transformation("drto.first", horizon="kept")
    reg.record_transformation("drto.second")
    assert [r["name"] for r in reg.transformations] == ["drto.first", "drto.second"]
    assert reg.has_transformation("drto.first")
    assert not reg.has_transformation("drto.third")
    assert reg.transformations[0]["outcome"] == {"horizon": "kept"}


def test_registry_survives_clone_with_remapped_references():
    m = declared_model()
    drto.info(m).record_transformation("drto.marker")
    m2 = m.clone()
    reg2 = drto.info(m2)
    assert reg2 is not drto.info(m)
    # component references point at the clone's components, not the source's
    assert reg2.components("state") == (m2.z,)
    assert reg2.components("state")[0] is not m.z
    assert reg2.has_transformation("drto.marker")
    # the registries are independent after the clone
    reg2.record_transformation("drto.only_on_clone")
    assert not drto.info(m).has_transformation("drto.only_on_clone")


def test_repr_groups_by_role():
    m = declared_model()
    text = repr(drto.info(m))
    assert "horizon: t (ContinuousSet, 5 points)" in text
    assert "states: z (free)" in text
    assert "controls: u (piecewise_constant, free)" in text
    assert "transformations: (none)" in text


def test_repr_marks_fixed_variables():
    m = declared_model()
    m.u.fix(0.5)
    assert "controls: u (piecewise_constant, fixed)" in repr(drto.info(m))


def test_repr_renders_indexed_constraints_compactly():
    m = declared_model()
    text = repr(drto.info(m))
    assert "dynamics: dzdt[k]  ==  - z[k] + u[k]  for k in t" in text
    # the symbolic form, not the per-index expansion
    assert "[2.5]" not in text


def test_repr_falls_back_for_skip_guarded_rules():
    m = declared_model()

    @m.Constraint(m.t)
    def guarded(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.z[t] <= 1

    reg = drto.info(m)
    reg.record_declaration("terminal_constraint", m.guarded)
    text = repr(reg)
    assert "z[k]" in text and "for k in t" in text


def test_repr_annotates_transformation_outcomes():
    m = declared_model()
    reg = drto.info(m)
    reg.record_transformation(
        "drto.dynamic_simulation", fixed="u", objective="zero", horizon="kept"
    )
    text = repr(reg)
    assert "drto.dynamic_simulation: fixed=u, objective=zero, horizon=kept" in text


def test_repr_html_contains_the_same_view():
    m = declared_model()
    reg = drto.info(m)
    reg.record_transformation("drto.marker")
    htm = reg._repr_html_()
    assert "<table>" in htm
    assert "drto.marker" in htm
    assert "controls" in htm


def test_scalar_constraint_renders_directly():
    m = declared_model()
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    reg = drto.info(m)
    reg.record_declaration("initial_condition", m.init)
    assert "initial conditions: z[0]  ==  z_hat" in repr(reg)
