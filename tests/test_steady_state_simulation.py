# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 008: drto.steady_state_simulation."""
import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from test_declarations import declared_model
from test_dynamic_optimization import estimation_model, linear_builder

ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
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


def test_create_using_resolves_a_nested_control_component():
    # create_using remaps the mapping's keys onto the clone and the
    # reduction replaces the control component, detaching the key, whose
    # name then degrades to its local name. Resolution happens while the
    # key is still attached, so a control below the top level passes too
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1, 2])
    m.sub = pyo.Block()
    m.sub.u = pyo.Var(m.t, initialize=0.3)
    m.z = pyo.Var(m.t, initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == mm.sub.u[t] - mm.z[t]

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.sub.u, profile="piecewise_constant")
    sim = pyo.TransformationFactory("drto.steady_state_simulation").create_using(
        m, controls={m.sub.u: 0.7}
    )
    assert sim.sub.u.fixed and pyo.value(sim.sub.u) == 0.7


def test_stage_cost_is_dropped():
    # the mode installs no cost equations
    m = declared_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={m.u: 0.3}
    )
    assert m.component("stage") is None
    assert not drto.info(m).has_declaration("tracking_stage_cost")


def test_steady_state_pairings_are_kept():
    # the target Params stay on the model and may appear in a deviation-form
    # model's equations, so their records stay with them and the registry
    # mirrors the model
    m = declared_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={m.u: 0.3}
    )
    assert drto.info(m).has_declaration("steady_state")
    assert drto.info(m).has_declaration("steady_state_control")
    assert m.component("z_ss") is not None


def test_terminal_constraint_leaves_the_model():
    m = declared_model()

    @m.Constraint()
    def term_set(m):
        return m.z[m.t.last()] <= 1

    drto.terminal_constraint(m.term_set)
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={m.u: 0.3}
    )
    assert m.component("term_set") is None
    assert not drto.info(m).has_declaration("terminal_constraint")


def test_estimation_declarations_are_neutralized():
    # a steady-state simulation of a model that also declares the estimation
    # surface sheds it, so the equilibrium solve stays square
    m = estimation_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(m)
    reg = drto.info(m)
    for kind in ("estimation_stage_cost", "arrival_cost", "measurement"):
        assert not reg.has_declaration(kind), kind
    assert m.component("y_meas") is None
    # the disturbance is kept, collapsed to a single point and fixed at zero
    assert reg.has_declaration("disturbance")
    assert m.w.fixed and pyo.value(m.w) == 0
    # the estimated parameter stays a live coefficient, so it keeps its record
    assert reg.has_declaration("estimated_parameter")
    assert m.k.fixed


def test_disturbance_realization_is_a_standing_value():
    # the equilibrium under a persistent disturbance, a single constant offset
    m = estimation_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, disturbances={"w": 0.1}
    )
    assert m.w.fixed and pyo.value(m.w) == 0.1


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
    drto.scaling.solver_by_name("ipopt").solve(m)
    assert pyo.value(m.z) == pytest.approx(0.3, abs=1e-8)


@needs_ipopt
def test_steady_authored_simulation_solves():
    m = steady_authored_model()
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(
        m, controls={"u": 0.4}
    )
    drto.scaling.solver_by_name("ipopt").solve(m)
    assert pyo.value(m.z) == pytest.approx(0.8, abs=1e-8)


def test_a_model_with_no_declared_control_reports_none():
    # steady_state_simulation requires states alone, so a model declaring no
    # control reaches the empty listing in the shared routine's error and in
    # the transformation log
    m = pyo.ConcreteModel()
    m.z = pyo.Var(initialize=1.0)

    @m.Constraint()
    def balance(mm):
        return mm.z == 2.0

    drto.state(m.z)
    with pytest.raises(ValueError, match=r"declared controls are \(none\)"):
        pyo.TransformationFactory("drto.steady_state_simulation").create_using(
            m, controls={"x": 1.0}
        )
    pyo.TransformationFactory("drto.steady_state_simulation").apply_to(m)
    assert "controls=(none declared)" in repr(drto.info(m))


# ----------------------------------------------------------------------
# the builder-consuming function form (gh #116)
# ----------------------------------------------------------------------
def test_the_function_reduces_and_prepares_the_equilibrium():
    sim = drto.steady_state_simulation(linear_builder)
    applied = [r["name"] for r in drto.info(sim).transformations]
    assert "drto.dynamic_to_steady_state" in applied
    assert "drto.steady_state_simulation" in applied
    assert sim.component("drto_objective") is not None
    # the reduction collapsed time rather than discretizing it
    assert sim.component("t") is None
    assert not drto.info(sim).has_declaration("horizon")


def test_the_function_passes_the_controls_through():
    sim = drto.steady_state_simulation(linear_builder, controls={"u": 0.3})
    assert sim.u.fixed and pyo.value(sim.u) == pytest.approx(0.3)


def test_the_function_passes_the_disturbances_through():
    sim = drto.steady_state_simulation(estimation_model, disturbances={"w": 0.1})
    assert sim.w.fixed and pyo.value(sim.w) == pytest.approx(0.1)


def test_the_function_takes_the_reductions_skip_on_a_steady_statement():
    sim = drto.steady_state_simulation(steady_authored_model)
    applied = [r["name"] for r in drto.info(sim).transformations]
    assert "drto.dynamic_to_steady_state" not in applied
    assert "drto.steady_state_simulation" in applied


def test_the_function_builds_what_create_using_gives():
    import hashlib

    def rows(model):
        r = [
            f"V|{v.name}|{v.lb}|{v.ub}|{v.fixed}|{v.value}"
            for v in model.component_data_objects(pyo.Var, active=True)
        ]
        r += [
            f"C|{c.name}|{c.lower}|{c.upper}|{c.body}"
            for c in model.component_data_objects(pyo.Constraint, active=True)
        ]
        r.sort()
        return len(r), hashlib.sha256(chr(10).join(r).encode()).hexdigest()

    held = linear_builder()
    cloned = pyo.TransformationFactory("drto.steady_state_simulation").create_using(
        held, controls={held.u: 0.3}
    )
    built = drto.steady_state_simulation(linear_builder, controls={"u": 0.3})
    assert rows(built) == rows(cloned)
