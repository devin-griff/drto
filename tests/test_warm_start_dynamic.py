# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 013: drto.warm_start_dynamic."""
import contextlib
import io
import math
import sys
from pathlib import Path

import pyomo.environ as pyo
import pytest

import drto
from test_infinite_horizon import block_model, ready_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.hicks import hicks  # noqa: E402

DT = 2.5  # the fixture's sample step

ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def test_values_only_and_fixed_left_alone():
    m = ready_model()
    for t in m.t:
        m.z[t].set_value(0.3)
        m.u[t].set_value(0.4)
    m.u[sorted(m.t)[0]].fix(0.9)
    before = {c.name for c in m.component_data_objects(pyo.Constraint, active=True)}
    drto.warm_start_dynamic(m)
    after = {c.name for c in m.component_data_objects(pyo.Constraint, active=True)}
    assert before == after
    assert m.u[sorted(m.t)[0]].fixed and pyo.value(m.u[sorted(m.t)[0]]) == 0.9


def test_shifts_copy_interpolate_and_fill():
    m = ready_model()
    grid = sorted(m.t)
    for t in grid:
        m.z[t].set_value(0.1 * t)
        m.u[t].set_value(0.5)
        m.dzdt[t].set_value(0.1)
    r = drto.warm_start_dynamic(m)
    for t in grid:
        if t + DT <= grid[-1] + 1e-9:
            assert pyo.value(m.z[t]) == pytest.approx(0.1 * (t + DT))
            assert pyo.value(m.dzdt[t]) == pytest.approx(0.1)
        else:
            # past the end: the declared targets, derivatives zero
            assert pyo.value(m.z[t]) == pytest.approx(pyo.value(m.z_ss))
            assert pyo.value(m.dzdt[t]) == 0.0
            assert pyo.value(m.u[t]) == pytest.approx(pyo.value(m.u_ss))
    assert r.n_copied > 0 and r.n_filled > 0
    assert "one step on" in str(r)


def test_missing_target_is_a_named_error():
    m = ready_model()
    for t in m.t:
        m.z[t].set_value(0.3)
        m.u[t].set_value(0.4)
    drto.info(m)._declarations.pop("steady_state")
    with pytest.raises(ValueError, match="steady_state"):
        drto.warm_start_dynamic(m)


def test_at_targets_shifts_to_itself():
    m = ready_model()
    for t in m.t:
        m.z[t].set_value(0.5)
        m.u[t].set_value(0.5)
        m.dzdt[t].set_value(0.0)
    drto.warm_start_dynamic(m)
    assert all(pyo.value(m.z[t]) == pytest.approx(0.5) for t in m.t)
    assert all(pyo.value(m.u[t]) == pytest.approx(0.5) for t in m.t)


def test_tail_covers_the_whole_problem_no_targets_needed():
    m = ready_model()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    reg = drto.info(m)
    reg._declarations.pop("steady_state")  # a tail needs no targets
    b = m.drto_ih
    gamma = pyo.value(b.gamma)
    tN = sorted(m.t)[-1]
    f = lambda t: 0.5 + 0.3 * math.exp(-0.4 * t)
    for t in m.t:
        m.z[t].set_value(f(t))
        m.u[t].set_value(0.5)
    for p in b.tau:
        b.z[p].set_value(f(tN + math.atanh(p) / gamma) if p < 1 else 0.5)
    r = drto.warm_start_dynamic(m)
    assert r.n_filled == 0, r.filled_names  # the tail covers everything
    worst = max(abs(pyo.value(m.z[t]) - f(t + DT)) for t in sorted(m.t))
    assert worst < 5e-3  # interpolated, tail included
    worst_tail = max(
        abs(pyo.value(b.z[p]) - f(tN + math.atanh(p) / gamma + DT))
        for p in sorted(b.tau)
        if p < 1
    )
    assert worst_tail < 5e-3


def test_moves_shift_as_step_functions_from_the_tail():
    # the last move takes the tail's first stored control value exactly,
    # never a blend of neighboring values
    m = ready_model()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    b = m.drto_ih
    for t in m.t:
        m.z[t].set_value(0.5)
        m.u[t].set_value(0.5)
    upts = sorted(i if not isinstance(i, tuple) else i[-1] for i in b.u)
    for n, pt in enumerate(upts):
        b.u[pt].set_value(0.1 * (n + 1))  # a staircase the blend would smear
    drto.warm_start_dynamic(m)
    grid = sorted(m.t)
    last = [t for t in grid if t + DT > grid[-1] + 1e-9]
    import math

    gamma = pyo.value(b.gamma)
    for t in last:
        tau = math.tanh(gamma * (t + DT - grid[-1]))
        # the shift anchors tau = 0 at the trajectory's own value (0.5)
        cands = [(0.0, 0.5)] + [(pt, 0.1 * (n + 1)) for n, pt in enumerate(upts)]
        want = max((c for c in cands if c[0] <= tau + 1e-7))[1]
        assert pyo.value(m.u[t]) == pytest.approx(want), t


def test_tail_derivatives_rescale_through_the_map():
    # dz/dtau shifts with the chain-rule factor (1 - tau2^2)/(1 - tau^2)
    # gamma large enough that the shift moves far in tau: the chain
    # factor is near identity on the default mesh and untestable there
    m = ready_model()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m, gamma=0.8)
    b = m.drto_ih
    gamma = pyo.value(b.gamma)
    tN = sorted(m.t)[-1]
    f = lambda t: 0.5 + 0.3 * math.exp(-0.4 * t)
    fp = lambda t: -0.12 * math.exp(-0.4 * t)
    for t in m.t:
        m.z[t].set_value(f(t))
        m.u[t].set_value(0.5)
        m.dzdt[t].set_value(fp(t))
    for p_ in b.tau:
        tt = tN + math.atanh(min(p_, 1 - 1e-12)) / gamma
        b.z[p_].set_value(f(tt))
        b.z_dtau[p_].set_value(fp(tt) / (gamma * (1 - p_**2)) if p_ < 1 else 0.0)
    for i in b.u:
        b.u[i].set_value(0.5)
    drto.warm_start_dynamic(m)
    worst = max(
        abs(
            pyo.value(b.z_dtau[p_])
            - fp(tN + math.atanh(p_) / gamma + DT) / (gamma * (1 - p_**2))
        )
        for p_ in sorted(b.tau)
        if 0.2 < p_ < 0.9
    )
    assert worst < 5e-3


def test_a_reference_controls_underlying_members_shift(monkeypatch):
    # cvp replaces the declared control; the original reference's
    # underlying members still shift through the same tail copy
    from test_infinite_horizon import ref_control_model

    m = ref_control_model()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    for t in m.t:
        m.z[t].set_value(0.5)
        m.props[t].f.set_value(0.5)
    b = m.drto_ih
    for rec in drto.info(m)._segment_records("control"):
        for ci in rec["copy"]:
            rec["copy"][ci].set_value(0.5)
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    r = drto.warm_start_dynamic(m)
    assert r.n_filled == 0, r.filled_names


def test_declared_suffixes_are_left_untouched():
    # the shift carries the primal solution only (gh #36): a carried
    # certificate must match the next problem, its active set, and the
    # restarted barrier level at once, and one sampling time of
    # staleness costs more than it saves; declared suffixes keep
    # exactly what the previous solve put there
    m = ready_model()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    b = m.drto_ih
    grid = sorted(m.t)
    for t in grid:
        m.z[t].set_value(0.5)
        m.u[t].set_value(0.5)
        m.ipopt_zL_out[m.u[t]] = 1.0 + 0.1 * t
    m.ipopt_zL_out[b.z_pin_lo] = 42.0
    con = next(iter(m.component_data_objects(pyo.Constraint, active=True)))
    m.dual[con] = 7.0
    before = {id(k): v for k, v in m.ipopt_zL_out.items()}
    r = drto.warm_start_dynamic(m)
    assert not hasattr(r, "multipliers")
    assert len(m.ipopt_zL_in) == 0  # nothing seeded
    assert {id(k): v for k, v in m.ipopt_zL_out.items()} == before
    assert m.dual[con] == 7.0


def test_algebra_shifts_through_the_recorded_tail():
    m = block_model()
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    b = m.drto_ih
    gamma = pyo.value(b.gamma)
    tN = sorted(m.t)[-1]
    f = lambda t: 0.2 + 0.3 * math.exp(-0.5 * t)
    for t in m.t:
        m.z[t].set_value(f(t))
        m.u[t].set_value(0.3)
        m.props[t].y.set_value(2 * f(t) + 0.3)
    seg_y = b.component("props_y")
    for p in seg_y:
        tp = p if not isinstance(p, tuple) else p[-1]
        seg_y[p].set_value(2 * f(tN + math.atanh(min(tp, 1 - 1e-12)) / gamma) + 0.3)
    drto.warm_start_dynamic(m)
    worst = max(
        abs(pyo.value(m.props[t].y) - (2 * f(t + 1.0) + 0.3)) for t in sorted(m.t)
    )
    assert worst < 5e-3


@needs_ipopt
def test_shifted_primals_warm_start_in_single_digits():
    # the loop hand-off (gh #36): solve, measure one sample in, shift,
    # re-solve with the warm-start options; the shifted primals alone
    # land the warm solve in single digits
    m = hicks(5)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    drto.cold_start_dynamic(m)
    drto.scaling.solver_by_name("ipopt").solve(m)
    m.zc_hat.set_value(pyo.value(m.zc[1]))
    m.zt_hat.set_value(pyo.value(m.zt[1]))
    drto.warm_start_dynamic(m)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = drto.scaling.solver_by_name("ipopt").solve(
            m,
            solver_options={
                "warm_start_init_point": "yes",
                "mu_init": 1e-6,
                "warm_start_bound_push": 1e-9,
                "warm_start_mult_bound_push": 1e-9,
            },
            tee=True,
        )
    assert drto.scaling.solved_to_optimality(res)
    iters = next(
        int(line.split(":")[-1])
        for line in buf.getvalue().splitlines()
        if "Number of Iterations" in line
    )
    assert iters <= 10
