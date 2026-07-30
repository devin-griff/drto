# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 011: drto.cold_start_dynamic."""
import pyomo.environ as pyo
import pytest

import drto
from test_infinite_horizon import block_model, ready_model

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def seeded():
    """ready_model with the hook away from the target: a real ramp."""
    m = ready_model()
    m.z_hat.set_value(0.1)  # target z_ss = 0.5
    for vd in m.u.values():
        vd.set_value(0.9)
    return m


def test_states_run_on_the_line_and_derivatives_hold_the_slope():
    m = seeded()
    drto.cold_start_dynamic(m)
    t0, tN = sorted(m.t)[0], sorted(m.t)[-1]
    slope = (0.5 - 0.1) / (tN - t0)
    for t in sorted(m.t):
        assert pyo.value(m.z[t]) == pytest.approx(0.1 + slope * (t - t0))
        assert pyo.value(m.dzdt[t]) == pytest.approx(slope, abs=1e-6)


def test_controls_hold_their_targets():
    m = seeded()
    drto.cold_start_dynamic(m)
    assert all(pyo.value(vd) == pytest.approx(0.5) for vd in m.u.values())


def test_missing_pairing_errors():
    m = seeded()
    reg = drto.info(m)
    reg._declarations.pop("steady_state_control")
    with pytest.raises(ValueError, match="no declared steady_state_control"):
        drto.cold_start_dynamic(m)


def test_on_target_reproduces_the_flat_broadcast():
    m = seeded()
    m.z_hat.set_value(0.5)  # start on the target
    drto.cold_start_dynamic(m)
    assert all(pyo.value(m.z[t]) == pytest.approx(0.5) for t in m.t)
    assert all(pyo.value(vd) == pytest.approx(0, abs=1e-6) for vd in m.dzdt.values())


def test_algebra_holds_everywhere_but_the_dynamics():
    m = block_model()
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    drto.steady_state_control(m.u, m.u_ss)
    for vd in m.q.values():
        vd.fix(3.0)  # the fixture fixes q before discretization only
    m.z_hat.set_value(0.1)
    drto.cold_start_dynamic(m)
    # everything except the declared dynamics is satisfied: the props
    # member equations and (structure) the discretization rows
    for con in m.component_data_objects(pyo.Constraint, active=True):
        if con.parent_component() is m.ode:
            continue
        assert abs(pyo.value(con.body) - pyo.value(con.lower)) < 1e-7, con.name


def test_transformed_shapes_initialize():
    for transform in ("drto.dynamic_optimization", "drto.infinite_horizon"):
        m = seeded()
        pyo.TransformationFactory(transform).apply_to(m)
        report = drto.cold_start_dynamic(m)
        assert report.n_states == 1 and report.n_controls == 1
        if transform == "drto.infinite_horizon":
            assert "targets" in report.segment
            b = m.drto_ih
            assert all(pyo.value(vd) == pytest.approx(0.5) for vd in b.z.values())
            assert all(pyo.value(vd) == 0 for vd in b.z_dtau.values())


@needs_ipopt
def test_cold_started_optimization_solves():
    m = seeded()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    drto.cold_start_dynamic(m)
    res = pyo.SolverFactory("ipopt").solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal


def test_report_reads():
    m = seeded()
    text = str(drto.cold_start_dynamic(m))
    assert "on a line" in text and "point solves" in text
