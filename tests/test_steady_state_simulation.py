# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 008: drto.steady_state_simulation."""
import pyomo.environ as pyo
import pytest

import drto
from test_declarations import declared_model

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def steady_authored_model():
    """A model written directly as steady-state: no horizon, no dynamics."""
    m = pyo.ConcreteModel()
    m.z = pyo.Var(initialize=1.0)
    m.u = pyo.Var(initialize=0.25, bounds=(0, 1))

    @m.Constraint()
    def balance(m):
        return m.z == 2 * m.u

    drto.state(m.z)
    drto.control(m.u)
    return m


def test_requires_declared_states():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    with pytest.raises(ValueError, match="requires declared states"):
        pyo.TransformationFactory("drto.steady_state_simulation").apply_to(m)


def test_dynamic_model_composes_the_reduction():
    m = declared_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={m.u: 0.3}
    )
    reg = drto.info(m)
    applied = [r["name"] for r in reg.transformations]
    assert "drto.dynamic_to_steady_state" in applied
    assert "drto.steady_state_simulation" in applied
    assert not m.u.is_indexed() and m.u.fixed and pyo.value(m.u) == 0.3
    assert m.component("drto_objective") is not None


def test_steady_authored_model_skips_the_reduction():
    m = steady_authored_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(m)
    applied = [r["name"] for r in drto.info(m).transformations]
    assert "drto.dynamic_to_steady_state" not in applied
    assert m.u.fixed and pyo.value(m.u) == 0.25  # held at its own value


def test_create_using_resolves_source_model_controls_by_name():
    m = declared_model()
    sim = pyo.TransformationFactory("drto.steady_state_simulation").create_using(
        m, controls={m.u: 0.4}
    )
    assert sim is not m
    assert sim.u.fixed and pyo.value(sim.u) == 0.4
    assert not m.u[0].fixed  # the source dynamic model is untouched


def test_stage_cost_is_dropped():
    # a simulation carries no cost equations
    m = declared_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={m.u: 0.3}
    )
    assert m.component("stage") is None
    assert not drto.info(m).has_declaration("tracking_stage_cost")


def test_steady_state_pairings_are_dropped():
    # nothing in a simulation reads the pairings; the Params stay
    m = declared_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={m.u: 0.3}
    )
    assert not drto.info(m).has_declaration("steady_state")
    assert not drto.info(m).has_declaration("steady_state_control")
    assert m.component("z_ss") is not None  # the user's Param remains


def test_unknown_control_errors():
    m = declared_model()
    m.w = pyo.Var()
    with pytest.raises(ValueError, match="not a declared control"):
        pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
            m, controls={"w": 1.0}
        )


def test_valueless_control_without_a_supplied_value_errors():
    m = steady_authored_model()
    m.u.set_value(None)
    with pytest.raises(ValueError, match="has none"):
        pyo.TransformationFactory("drto.steady_state_simulation").apply_to(m)


@needs_ipopt
def test_simulation_solves_the_fixed_control_equilibrium():
    # dzdt = -z + u at rest gives z = u
    m = declared_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={"u": 0.3}
    )
    pyo.SolverFactory("ipopt").solve(m)
    assert pyo.value(m.z) == pytest.approx(0.3, abs=1e-8)


@needs_ipopt
def test_steady_authored_simulation_solves():
    m = steady_authored_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={"u": 0.4}
    )
    pyo.SolverFactory("ipopt").solve(m)
    assert pyo.value(m.z) == pytest.approx(0.8, abs=1e-8)
