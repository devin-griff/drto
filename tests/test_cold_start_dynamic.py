# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 011: drto.cold_start_dynamic."""
import math

import pyomo.environ as pyo
import pytest

import drto
import drto.cold_start
from test_infinite_horizon import block_model, packed_model, ready_model

ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def seeded():
    """ready_model with the initial condition away from the target: a real ramp."""
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


def test_a_scale_source_writes_the_factors_first(monkeypatch):
    import drto.scaling as sc

    m = seeded()
    order = []
    real = sc.scale
    monkeypatch.setattr(
        sc, "scale", lambda mm, source: order.append("scale") or real(mm, source=source)
    )
    real_info = drto.cold_start.info
    monkeypatch.setattr(
        drto.cold_start, "info", lambda mm: order.append("run") or real_info(mm)
    )
    drto.cold_start_dynamic(m, scale="bounds")
    assert order[0] == "scale"
    assert m.component("scaling_factor") is not None


def test_the_default_writes_no_factors():
    m = seeded()
    drto.cold_start_dynamic(m)
    assert m.component("scaling_factor") is None


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
            # the tau discretization rows re-solve the derivatives to
            # the pipeline's tolerance, like the finite side's
            assert all(
                pyo.value(vd) == pytest.approx(0, abs=1e-6) for vd in b.z_dtau.values()
            )


@needs_ipopt
def test_cold_started_optimization_solves():
    m = seeded()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    drto.cold_start_dynamic(m)
    res = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(res)


def test_every_set_aside_row_comes_back_active():
    # the per-point solves set the dynamics, the initial condition, and
    # the segment's structural rows aside; every one comes back active
    m = seeded()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    before = {cd.name for cd in m.component_data_objects(pyo.Constraint, active=True)}
    drto.cold_start_dynamic(m)
    after = {cd.name for cd in m.component_data_objects(pyo.Constraint, active=True)}
    assert before == after


def test_without_pounce_the_values_set_and_the_algebra_keeps(monkeypatch):
    # the degradation path: states, derivatives, and controls initialize
    # the same way, the algebraic variables keep their values, nothing
    # is deactivated, and the report records the skip
    monkeypatch.setattr(drto.cold_start, "pounce_available", False)
    m = block_model()
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    drto.steady_state_control(m.u, m.u_ss)
    for vd in m.q.values():
        vd.fix(3.0)
    m.z_hat.set_value(0.1)
    for blk in m.props.values():
        blk.y.set_value(-7.0)  # sentinel: the skip must not touch it
    report = drto.cold_start_dynamic(m)
    assert "skipped" in report.point_solves
    t0, tN = sorted(m.t)[0], sorted(m.t)[-1]
    slope = (0.5 - 0.1) / (tN - t0)
    for t in sorted(m.t):
        assert pyo.value(m.z[t]) == pytest.approx(0.1 + slope * (t - t0))
        assert pyo.value(m.dz[t]) == pytest.approx(slope)
    assert all(pyo.value(vd) == pytest.approx(0.3) for vd in m.u.values())
    assert all(blk.y.value == -7.0 for blk in m.props.values())
    assert all(cd.active for cd in m.component_data_objects(pyo.Constraint))


def test_report_reads():
    m = seeded()
    text = str(drto.cold_start_dynamic(m))
    assert "on a line" in text and "point solves" in text


def test_exponential_profile_runs_on_the_decay():
    m = seeded()
    tau = 2.0
    report = drto.cold_start_dynamic(m, profile="exponential", time_constant=tau)
    assert "exponential decay" in str(report)
    t0, tN = sorted(m.t)[0], sorted(m.t)[-1]
    T = tN - t0
    d = T / tau
    denom = 1 - math.exp(-d)
    for t in sorted(m.t):
        want = 0.1 + 0.4 * (1 - math.exp(-d * (t - t0) / T)) / denom
        assert pyo.value(m.z[t]) == pytest.approx(want)
    assert pyo.value(m.z[tN]) == pytest.approx(0.5)  # lands on the target


def test_exponential_derivatives_hold_the_pointwise_slope(monkeypatch):
    # without pounce the analytic seeds stay: the decay's slope at each t
    monkeypatch.setattr(drto.cold_start, "pounce_available", False)
    m = seeded()
    tau = 2.0
    drto.cold_start_dynamic(m, profile="exponential", time_constant=tau)
    t0, tN = sorted(m.t)[0], sorted(m.t)[-1]
    T = tN - t0
    d = T / tau
    denom = 1 - math.exp(-d)
    for t in sorted(m.t):
        want = 0.4 * d * math.exp(-d * (t - t0) / T) / (denom * T)
        assert pyo.value(m.dzdt[t]) == pytest.approx(want)


def test_default_time_constant_is_a_third_of_the_horizon():
    m1, m2 = seeded(), seeded()
    T = sorted(m1.t)[-1] - sorted(m1.t)[0]
    drto.cold_start_dynamic(m1, profile="exponential")
    drto.cold_start_dynamic(m2, profile="exponential", time_constant=T / 3)
    for t in sorted(m1.t):
        assert pyo.value(m1.z[t]) == pytest.approx(pyo.value(m2.z[t]))


def test_profile_errors():
    with pytest.raises(ValueError, match="unknown profile"):
        drto.cold_start_dynamic(seeded(), profile="cubic")
    with pytest.raises(ValueError, match="exponential profile's"):
        drto.cold_start_dynamic(seeded(), time_constant=2.0)
    with pytest.raises(ValueError, match="positive"):
        drto.cold_start_dynamic(seeded(), profile="exponential", time_constant=0.0)


def test_a_scaling_suffix_does_not_change_the_point_solves():
    # the solves run in the model's own units; an active suffix stays
    # on the model, unread, for the NLP solves that follow (gh #92)
    def built(suffix):
        m = block_model()
        m.u_ss = pyo.Param(initialize=0.3, mutable=True)
        drto.steady_state_control(m.u, m.u_ss)
        for vd in m.q.values():
            vd.fix(3.0)
        m.z_hat.set_value(0.1)
        if suffix:
            m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            for blk in m.props.values():
                m.scaling_factor[blk.y] = 1e-6
                m.scaling_factor[blk.gain] = 1e-6
        return m

    plain, carrying = built(False), built(True)
    drto.cold_start_dynamic(plain)
    report = drto.cold_start_dynamic(carrying)
    assert report.point_solves == "run (pyomo-pounce block solve)"
    # the suffix survives the call, entries intact
    assert len(carrying.scaling_factor) == 2 * len(carrying.props)
    pairs = zip(
        plain.component_data_objects(pyo.Var),
        carrying.component_data_objects(pyo.Var),
        strict=True,
    )
    for vd, svd in pairs:
        assert (vd.value is None) == (svd.value is None), svd.name
        if vd.value is not None:
            assert svd.value == pytest.approx(vd.value, abs=1e-7), svd.name


def test_segment_algebra_solves_at_the_tail():
    # the tail's algebraic copies come from the same per-point solves:
    # with the tail at the targets, everything but the dynamics copies
    # holds at the segment's points
    m = block_model()
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    drto.steady_state_control(m.u, m.u_ss)
    for vd in m.q.values():
        vd.fix(3.0)
    m.z_hat.set_value(0.1)
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    drto.cold_start_dynamic(m)
    dynamics_copies = {m.drto_ih.component("ode")}
    checked = 0
    for con in m.drto_ih.component_objects(pyo.Constraint):
        if con in dynamics_copies:
            continue
        for cd in con.values():
            if cd.active and cd.equality:
                assert abs(pyo.value(cd.body) - pyo.value(cd.lower)) < 1e-6, cd.name
                checked += 1
    assert checked > 0


def test_undeclared_entries_algebra_solves_in_the_point_solves():
    # an indexed Var's undeclared member: its closure determines it and
    # the discretization rows its derivative
    m = packed_model()
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    drto.steady_state_control(m.u, m.u_ss)
    m.z_hat.set_value(0.1)
    drto.cold_start_dynamic(m)
    for t in m.t:
        assert pyo.value(m.x[t, "W"]) == pytest.approx(55.0)
    # the discretization equations determine its derivative at the
    # collocation points; the first point has no such row and stays put
    for t in sorted(m.t)[1:]:
        assert pyo.value(m.dx[t, "W"]) == pytest.approx(0, abs=1e-6)
    for con in m.component_data_objects(pyo.Constraint, active=True):
        if con.parent_component() is m.bal:
            continue
        assert abs(pyo.value(con.body) - pyo.value(con.lower)) < 1e-6, con.name


def test_point_solves_false_skips_the_algebra_deliberately():
    # the profiles-only mode as a choice (gh #43): sentinel algebraic
    # values survive, and the report names the option, not a missing
    # install
    pytest.importorskip("pyomo_pounce")
    m = seeded()
    for t in m.t:
        m.cost[t].set_value(123.25)
    report = drto.cold_start_dynamic(m, point_solves=False)
    assert all(pyo.value(m.cost[t]) == 123.25 for t in m.t)
    assert report.point_solves == "skipped (by option)"
    # the states still ran their profile: the ramp landed
    assert pyo.value(m.z[sorted(m.t)[-1]]) == pytest.approx(0.5)


def test_point_solves_rejects_non_booleans():
    with pytest.raises(ValueError, match="point_solves"):
        drto.cold_start_dynamic(seeded(), point_solves="maybe")


def test_the_report_carries_no_clone():
    # the report holds values and counts, not a second model: the loop
    # solves the model itself and no clone exists to adopt (gh #92)
    pytest.importorskip("pyomo_pounce")
    m = seeded()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for vd in m.z.values():
        m.scaling_factor[vd] = 2.0
    report = drto.cold_start_dynamic(m)
    assert not hasattr(report, "scaled_model")


def test_the_segment_controls_reach_their_targets_after_parameterize():
    # drto.parameterize replaces the declared control, and the segment
    # record keeps the component as declared, so a lookup keyed on that
    # alone finds nothing and the segment controls are skipped (gh #125)
    m = seeded()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    drto.cold_start_dynamic(m)

    target = pyo.value(
        drto.info(m).declarations("steady_state_control")[0]["component"]
    )
    copy = drto.info(m)._segment_records("control")[0]["copy"]
    assert len(copy) > 0
    for vd in copy.values():
        assert pyo.value(vd) == pytest.approx(target)
