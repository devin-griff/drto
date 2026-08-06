# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 024: the index-one check."""
import sys
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

G, MASS, CORD = 9.81, 1.0, 1.0


def pendulum(index):
    """The APMonitor pendulum, one model per index version.

    States x, y (positions), v, w (velocities); the cord tension lam is
    algebraic in the DAE versions and a fifth state in the ODE version,
    where it carries its own differential equation.
    """
    m = pyo.ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 1))
    m.x = pyo.Var(m.time, initialize=0.5)
    m.y = pyo.Var(m.time, initialize=-0.866)
    m.v = pyo.Var(m.time, initialize=0.1)
    m.w = pyo.Var(m.time, initialize=0.1)
    m.lam = pyo.Var(m.time, initialize=0.1)
    m.dx = DerivativeVar(m.x, wrt=m.time, initialize=0)
    m.dy = DerivativeVar(m.y, wrt=m.time, initialize=0)
    m.dv = DerivativeVar(m.v, wrt=m.time, initialize=0)
    m.dw = DerivativeVar(m.w, wrt=m.time, initialize=0)
    if index == 0:
        m.dlam = DerivativeVar(m.lam, wrt=m.time, initialize=0)

    drto.horizon(m.time)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.time, nfe=2, ncp=2, scheme="LAGRANGE-RADAU"
    )

    m.ode_x = pyo.Constraint(m.time, rule=lambda b, t: b.dx[t] == b.v[t])
    m.ode_y = pyo.Constraint(m.time, rule=lambda b, t: b.dy[t] == b.w[t])
    m.ode_v = pyo.Constraint(
        m.time, rule=lambda b, t: MASS * b.dv[t] == -2 * b.x[t] * b.lam[t]
    )
    m.ode_w = pyo.Constraint(
        m.time, rule=lambda b, t: MASS * b.dw[t] == -MASS * G - 2 * b.y[t] * b.lam[t]
    )
    dyn = [m.ode_x, m.ode_y, m.ode_v, m.ode_w]
    if index == 3:
        m.alg = pyo.Constraint(
            m.time, rule=lambda b, t: b.x[t] ** 2 + b.y[t] ** 2 == CORD**2
        )
    elif index == 2:
        m.alg = pyo.Constraint(
            m.time, rule=lambda b, t: b.x[t] * b.v[t] + b.y[t] * b.w[t] == 0
        )
    elif index == 1:
        m.alg = pyo.Constraint(
            m.time,
            rule=lambda b, t: MASS * (b.v[t] ** 2 + b.w[t] ** 2 - G * b.y[t])
            - 2 * b.lam[t] * (b.x[t] ** 2 + b.y[t] ** 2)
            == 0,
        )
    else:
        m.ode_lam = pyo.Constraint(
            m.time,
            rule=lambda b, t: b.dlam[t] * (b.x[t] ** 2 + b.y[t] ** 2)
            == -4 * b.lam[t] * (b.x[t] * b.v[t] + b.y[t] * b.w[t])
            - 1.5 * G * MASS * b.w[t],
        )
        dyn.append(m.ode_lam)

    states = [m.x, m.y, m.v, m.w] + ([m.lam] if index == 0 else [])
    drto.state(*states)
    drto.dynamics(*dyn)
    return m


def test_the_index_one_pendulum_passes():
    report = drto.check_index(pendulum(1))
    assert report.verdict == "index one"
    assert not report.unmatched_variables
    assert report.condition_estimate is not None
    assert report.condition_estimate < 1e10


def test_the_ode_pendulum_is_index_zero():
    report = drto.check_index(pendulum(0))
    assert "index zero" in report.verdict
    assert "ODE" in report.verdict
    assert report.n_algebraic_constraints == 0


def test_the_index_two_pendulum_fails_with_the_tension_named():
    report = drto.check_index(pendulum(2))
    assert "not index one" in report.verdict
    assert any("lam" in n for n in report.unmatched_variables)
    assert report.structural_index == 2


def test_the_index_three_pendulum_fails_at_depth_three():
    report = drto.check_index(pendulum(3))
    assert "not index one" in report.verdict
    assert any("lam" in n for n in report.unmatched_variables)
    assert report.structural_index == 3


def test_the_check_writes_nothing():
    m = pendulum(3)
    before = {id(c) for c in m.component_data_objects(pyo.Constraint)}
    values = {id(v): v.value for v in m.component_data_objects(pyo.Var)}
    drto.check_index(m)
    after = {id(c) for c in m.component_data_objects(pyo.Constraint)}
    assert before == after
    assert all(values[id(v)] == v.value for v in m.component_data_objects(pyo.Var))


def test_missing_values_skip_the_numerical_layer():
    m = pendulum(1)
    for vd in m.lam.values():
        vd.set_value(None)
    report = drto.check_index(m)
    assert "skipped" in report.numerical
    assert "initialize" in report.numerical
    assert report.verdict == "index one structurally; numerical layer skipped"


def test_the_condition_limit_is_honored():
    m = pendulum(1)
    report = drto.check_index(m, condition_limit=1e-6)
    assert "not index one at this point" in report.verdict
    with pytest.raises(ValueError, match="condition_limit"):
        drto.check_index(m, condition_limit=0)


def test_requires_the_declarations():
    m = pyo.ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 1))
    with pytest.raises(ValueError, match="horizon"):
        drto.check_index(m)


def test_the_report_prints_readably():
    text = str(drto.check_index(pendulum(3)))
    assert "drto check_index" in text
    assert "structural index: 3" in text
    assert "verdict" in text


def test_a_numerically_singular_algebra_names_its_block():
    # two algebraic variables tied by a dependent pair of rows: full
    # structural matching, singular values
    m = pyo.ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 1))
    m.z = pyo.Var(m.time, initialize=1.0)
    m.a = pyo.Var(m.time, initialize=1.0)
    m.b = pyo.Var(m.time, initialize=1.0)
    m.dz = DerivativeVar(m.z, wrt=m.time, initialize=0)
    drto.horizon(m.time)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.time, nfe=2, ncp=2, scheme="LAGRANGE-RADAU"
    )
    m.bal = pyo.Constraint(m.time, rule=lambda b, t: b.dz[t] == -b.z[t])
    m.g1 = pyo.Constraint(m.time, rule=lambda b, t: b.a[t] + b.b[t] == b.z[t])
    m.g2 = pyo.Constraint(
        m.time, rule=lambda b, t: 2 * b.a[t] + 2 * b.b[t] == 2 * b.z[t]
    )
    drto.state(m.z)
    drto.dynamics(m.bal)
    report = drto.check_index(m)
    assert "singular" in report.verdict


# ----------------------------------------------------------------------
# the two solvent extraction models, feature 024's acceptance anchors
# ----------------------------------------------------------------------
_EX = Path(__file__).parent.parent / "examples"


def _example_module(name):
    pytest.importorskip("prommis")
    sys.path.insert(0, str(_EX))
    try:
        module = __import__(f"models.{name}", fromlist=[name])
    finally:
        sys.path.pop(0)
    return module


def test_the_mscontactor_form_fails_with_the_extents_named():
    mod = _example_module("prommis_sx")
    report = drto.check_index(mod.build(N=1, h=0.25))
    assert "not index one" in report.verdict
    assert any("heterogeneous_reaction_extent" in n for n in report.unmatched_variables)
    assert any("_reaction_extent" in n for n in report.unmatched_variables)


def test_the_reaction_invariant_form_passes():
    mod = _example_module("prommis_sx2")
    report = drto.check_index(mod.build(N=1, h=0.25))
    assert report.verdict == "index one"
