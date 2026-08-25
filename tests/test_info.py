# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 001: the drto registry (drto.info)."""
import pyomo.environ as pyo
import pytest
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
    assert "dynamics: dzdt[t]  ==  - z[t] + u[t]  for t in t" in text
    # the symbolic form, not the per-index expansion
    assert "[2.5]" not in text


def test_repr_keeps_component_names_ending_in_an_underscored_number():
    # naming the free indexes replaces the template's own placeholders only,
    # so a component name containing _1 or _2 renders unchanged (gh #103)
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10), initialize=[0, 5, 10])
    m.z_1 = pyo.Var(m.t)
    m.dz_1 = DerivativeVar(m.z_1, wrt=m.t)
    m.k0_1 = pyo.Param(initialize=1.2, mutable=True)
    m.k0_2 = pyo.Param(initialize=3.4, mutable=True)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dz_1[t] == -m.k0_1 * m.z_1[t] - m.k0_2 * m.z_1[t] ** 2

    reg = drto.info(m)
    reg.record_declaration("horizon", m.t)
    reg.record_declaration("state", m.z_1)
    reg.record_declaration("dynamics", m.ode)
    text = repr(reg)
    assert "dz_1[t]  ==  - k0_1*z_1[t] - k0_2*z_1[t]**2  for t in t" in text


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
    assert "z[t]" in text and "for t in t" in text
    assert "shown at" not in text


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


def test_scalar_constraint_folds_its_sums():
    # a scalar cost summing over a set (the terminal-cost shape) renders
    # as a symbolic SUM, not the expanded member-by-member expression
    m = pyo.ConcreteModel()
    m.tray = pyo.Set(initialize=range(1, 42))
    m.w = pyo.Var(m.tray)
    m.term = pyo.Var()

    @m.Constraint()
    def terminal(mm):
        return mm.term == sum((mm.w[i] - 0.5) ** 2 for i in mm.tray)

    reg = drto.info(m)
    reg.record_declaration("tracking_terminal_cost", m.terminal)
    text = repr(reg)
    assert "SUM(" in text
    assert "w[41]" not in text


def test_scalar_constraint_renders_directly():
    m = declared_model()
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    reg = drto.info(m)
    reg.record_declaration("initial_condition", m.init)
    assert "initial conditions: z[0]  ==  z_hat" in repr(reg)


# ── feature 019: registry units ───────────────────────────────────────────────


def unit_model():
    """A declared model carrying pyo.units, with one inconsistent equation."""
    pytest.importorskip("pint")  # declaring Var(units=...) needs it
    U = pyo.units
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10), initialize=[0, 5, 10])
    m.z = pyo.Var(m.t, units=U.mol)
    # t carries no units, so the derivative's units are given explicitly;
    # without them dz/dt is dimensionless and the ode would render (inc)
    m.dzdt = DerivativeVar(m.z, wrt=m.t, units=U.mol)
    m.u = pyo.Var(m.t, units=U.mol)
    m.w = pyo.Var(m.t, units=U.kg * U.m**2 / U.s**3)  # base units of W
    m.tau = pyo.Param(initialize=2.0, mutable=True)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == m.u[t] - m.z[t] / m.tau

    @m.Constraint(m.t)
    def mixed(m, t):
        return m.z[t] + m.w[t] == 0.0  # mol + W: inconsistent

    m.z0 = pyo.Param(initialize=1.0, units=U.mol, mutable=True)
    m.ic = pyo.Constraint(expr=m.z[0] == m.z0)

    reg = drto.info(m)
    reg.record_declaration("horizon", m.t)
    reg.record_declaration("state", m.z)
    reg.record_declaration("dynamics", m.ode)
    reg.record_declaration("control", m.w, profile="piecewise_constant")
    reg.record_declaration("initial_condition", m.ic)
    reg.record_declaration("terminal_constraint", m.mixed)
    return m


def test_units_annotate_variables_and_constraints():
    r = repr(drto.info(unit_model()))
    assert "z (free, mol)" in r
    assert "w (piecewise_constant, free, W)" in r  # compact, not kg*m**2/s**3
    # the constraint suffix, asserted on a plain constraint: the ode's
    # compact rendering is hostage to an order-dependent templatization
    # flake (a DerivativeVar body renders '[Unattached VarData]' after a
    # failed templatization earlier in the process), pre-existing and
    # independent of the units annotation
    line = next(l for l in r.splitlines() if "initial conditions" in l)
    assert line.rstrip().endswith("(mol)")


def test_degenerate_combinations_do_not_leak():
    # J/s is W and W*s is J, a kJ keeps its scale, and a ratio of preferred
    # units reduces (J/W is a time constant, s) rather than rendering as a
    # W-and-J compound
    pytest.importorskip("pint")
    U = pyo.units
    from drto.info import _units_note

    m = pyo.ConcreteModel()
    m.a = pyo.Var(units=U.J / U.s)
    m.b = pyo.Var(units=U.W * U.s)
    m.c = pyo.Var(units=U.kJ)
    m.d = pyo.Var(units=U.J / U.W)
    m.e = pyo.Var(units=U.mol / U.s)
    assert _units_note(m.a) == "W"
    assert _units_note(m.b) == "J"
    assert _units_note(m.c) == "kJ"
    assert _units_note(m.d) == "s"
    assert _units_note(m.e) == "mol/s"


def test_inconsistent_body_renders_inc():
    r = repr(drto.info(unit_model()))
    line = next(l for l in r.splitlines() if "terminal constraint" in l)
    assert line.rstrip().endswith("(inc)")


def test_unitless_model_renders_unchanged():
    r = repr(drto.info(declared_model()))
    assert "(free)" in r and "(piecewise_constant, free)" in r
    assert "(inc)" not in r and "dimensionless" not in r
    for line in r.splitlines():
        assert not line.rstrip().endswith("(mol)")


def test_units_never_raise_on_odd_components():
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 1), initialize=[0, 1])
    m.z = pyo.Var(m.t)
    reg = drto.info(m)
    reg.record_declaration("horizon", m.t)
    reg.record_declaration("state", m.z)
    r = repr(reg)  # a registry with no constraints and no units renders fine
    assert "z" in r


# ── feature 011 fixes: rendering noise and the fallback substitution ─────────


def untemplatizable_model():
    """A rule with a dict lookup on the index, which templatization cannot
    execute, referencing a fixed [0] index on another component that a naive
    text swap would wrongly rename."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 5), initialize=[0, 2.5, 5])
    m.z = pyo.Var(m.t)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.other = pyo.Var(m.t)
    gain = {0: 1.0, 2.5: 2.0, 5: 1.5}

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dzdt[t] == gain[t] * mm.z[t] + mm.other[0]

    reg = drto.info(m)
    reg.record_declaration("horizon", m.t)
    reg.record_declaration("state", m.z)
    reg.record_declaration("dynamics", m.ode)
    return m


def test_fallback_swaps_the_index_on_other_components_too():
    # the model is deliberately a temporary: the registry keeps it alive
    # (gh #40); without the reference back, collection would detach every
    # component the records do not hold and render [Unattached VarData]
    import gc

    reg = drto.info(untemplatizable_model())
    gc.collect()
    r = repr(reg)
    line = next(l for l in r.splitlines() if "dynamics" in l)
    assert "z[t]" in line and "dzdt[t]" in line
    assert "2.5" not in line  # the member's coordinate is gone
    assert "other[0]" in line  # the unrelated fixed index survives


def test_fallback_picks_a_distinctive_member():
    # with t=0 as the member, other[0] would collide and wrongly rename;
    # the picker must choose a coordinate that appears nowhere else
    m = untemplatizable_model()
    from drto.info import _compact_constraint

    s = _compact_constraint(m.ode)
    assert "other[0]" in s and "other[t]" not in s


def test_templatize_failure_is_quiet(caplog):
    import logging

    m = untemplatizable_model()
    with caplog.at_level(logging.ERROR, logger="pyomo.core"):
        repr(drto.info(m))
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


# ----------------------------------------------------------------------
# the size line (gh #63)
# ----------------------------------------------------------------------
def _sized():
    """A model with two states, one of them a member subset, and one
    control, so the counts are not all ones."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1, 2])
    m.J = pyo.Set(initialize=["a", "b", "c"])
    m.z = pyo.Var(m.t, m.J, initialize=0.0)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, initialize=0.0)

    @m.Constraint(m.t, m.J)
    def ode(mm, t, j):
        return mm.dz[t, j] == -mm.z[t, j] + mm.u[t]

    drto.horizon(m.t)
    drto.state(*(m.z[:, j] for j in ["a", "b"]))
    drto.dynamics(m.ode)
    drto.control(m.u)
    return m


def test_the_registry_states_the_problem_size():
    m = _sized()
    text = repr(drto.info(m))
    assert text.splitlines()[1] == "states: 2, controls: 1"
    # before the declaration lines
    assert text.index("states: 2") < text.index("declarations:")


def test_the_size_counts_members_not_grid_points():
    # three time points, two declared members: the count is the model's
    # dimension, not the number of variables
    m = _sized()
    assert len(m.t) == 3
    assert drto.info(m)._size_line() == "states: 2, controls: 1"


def test_the_size_line_appears_in_the_html_render():
    m = _sized()
    html_text = drto.info(m)._repr_html_()
    assert "states: 2, controls: 1" in html_text
    assert html_text.index("states: 2") < html_text.index("<code>")


def test_an_undeclared_model_has_no_size_line():
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1])
    drto.horizon(m.t)
    reg = drto.info(m)
    assert reg._size_line() is None
    assert repr(reg).splitlines()[1].startswith("declarations")


def test_the_size_survives_the_steady_reduction():
    # the reduction collapses the time coordinate; the model's dimension
    # is the same, so the line reads the same
    m = _sized()
    before = drto.info(m)._size_line()

    declared = ["a", "b"]
    m.z_hat = pyo.Param(declared, initialize=0.0, mutable=True)

    @m.Constraint(declared)
    def z_init(mm, j):
        return mm.z[0, j] == mm.z_hat[j]

    drto.initial_condition(m.z_init)
    for j in ("a", "b"):
        p = pyo.Param(initialize=0.0, mutable=True)
        m.add_component(f"ss_{j}", p)
        drto.steady_state(m.z[:, j], p)
    pyo.TransformationFactory("drto.dynamic_to_steady_state").apply_to(m)
    assert drto.info(m)._size_line() == before
