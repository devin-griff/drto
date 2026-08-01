# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 014: drto.ideal_nmpc."""
import sys
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

# drto.ideal_nmpc the attribute is the function; the module for
# monkeypatching comes from the import system
loop_module = sys.modules["drto.ideal_nmpc"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.hicks import hicks

try:
    import pyomo_pounce  # noqa: F401  registers the pounce solver
except ImportError:
    pass
pounce_ok = pyo.SolverFactory("pounce").available(exception_flag=False)
needs_pounce = pytest.mark.skipif(not pounce_ok, reason="pounce not available")
ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def loop_model(N=5, discretize=True):
    """dz = u - z + w, with the state and control targets meeting at 0.5."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N, 1))
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)
    m.u_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.2, mutable=True)
    m.z = pyo.Var(m.t, bounds=(-1, 2), initialize=0.2)
    m.dz = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.w = pyo.Var(m.t, initialize=0.0)
    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dz[t] == mm.u[t] - mm.z[t] + mm.w[t]

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (mm.z[t] - mm.z_ss) ** 2 + 0.1 * (mm.u[t] - mm.u_ss) ** 2

    @m.Constraint()
    def z_init(mm):
        return mm.z[0] == mm.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.disturbance(m.w)
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.z_init)
    drto.steady_state(m.z, m.z_ss)
    drto.steady_state_control(m.u, m.u_ss)
    if discretize:
        pyo.TransformationFactory("dae.collocation").apply_to(
            m, wrt=m.t, nfe=N, ncp=3, scheme="LAGRANGE-RADAU"
        )
    return m


class _Recorder:
    """Wraps a real solver, recording each solve call's options."""

    def __init__(self, real):
        self.real, self.calls = real, []

    def available(self, exception_flag=False):
        return True

    def solve(self, model, **kwds):
        self.calls.append(dict(kwds.get("options") or {}))
        return self.real.solve(model, **kwds)


# ── validation ───────────────────────────────────────────────────────────────


def test_requires_the_untransformed_model():
    m = loop_model()
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    with pytest.raises(ValueError, match="already applied"):
        drto.ideal_nmpc(m, steps=2)


def test_requires_a_discretized_horizon():
    with pytest.raises(ValueError, match="discretized"):
        drto.ideal_nmpc(loop_model(discretize=False), steps=2)


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        drto.ideal_nmpc(loop_model(), steps=0)


def test_unknown_state_name_errors():
    with pytest.raises(ValueError, match="not a pinned state"):
        drto.ideal_nmpc(loop_model(), steps=2, initial_condition={"nope": 1.0})


def test_unknown_disturbance_name_errors():
    with pytest.raises(ValueError, match="not a declared disturbance"):
        drto.ideal_nmpc(loop_model(), steps=2, disturbances={"nope": 0.1})


def test_short_disturbance_sequence_errors():
    with pytest.raises(ValueError, match="one per step"):
        drto.ideal_nmpc(loop_model(), steps=3, disturbances={"w": [0.1]})


def test_unavailable_solver_errors():
    with pytest.raises(RuntimeError, match="not available"):
        drto.ideal_nmpc(loop_model(), steps=2, solver="no_such_solver")


# ── the loop ─────────────────────────────────────────────────────────────────


@needs_pounce
def test_loop_settles_and_records():
    h = drto.ideal_nmpc(loop_model(), steps=8, seed=0)
    assert h.times == list(range(9))
    assert len(h.states["z"]) == 9 and len(h.moves["u"]) == 8
    assert h.states["z"][0] == pytest.approx(0.2)
    assert h.states["z"][-1] == pytest.approx(0.5, abs=1e-4)
    assert h.moves["u"][-1] == pytest.approx(0.5, abs=1e-3)
    assert h.state_targets["z"] == pytest.approx(0.5)
    assert h.control_targets["u"] == pytest.approx(0.5)
    assert h.realizations["w"] == [0.0] * 8


@needs_pounce
def test_initial_condition_reaches_the_first_solve():
    h = drto.ideal_nmpc(loop_model(), steps=2, initial_condition={"z": 0.4})
    assert h.states["z"][0] == pytest.approx(0.4)
    # one step from 0.4 lands closer to the target than one from 0.2
    assert abs(h.states["z"][1] - 0.5) < 0.01


@needs_pounce
def test_constant_disturbance_offsets_the_plant():
    h = drto.ideal_nmpc(loop_model(), steps=8, disturbances={"w": [0.2] * 8})
    assert h.realizations["w"] == [0.2] * 8
    # the controller plans at zero noise, so the plant holds an offset
    assert h.states["z"][-1] == pytest.approx(0.63624, abs=1e-3)


@needs_pounce
def test_draws_are_reproducible_under_seed():
    kw = dict(steps=3, disturbances={"w": 0.05})
    a = drto.ideal_nmpc(loop_model(), seed=3, **kw)
    b = drto.ideal_nmpc(loop_model(), seed=3, **kw)
    c = drto.ideal_nmpc(loop_model(), seed=4, **kw)
    assert a.realizations["w"] == b.realizations["w"]
    assert a.realizations["w"] != c.realizations["w"]
    assert a.states["z"] == pytest.approx(b.states["z"])


@needs_pounce
def test_hicks_settles_to_the_declared_targets():
    m = hicks(N=5)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=5, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("drto.infinite_horizon").apply_to(m)
    h = drto.ideal_nmpc(m, steps=10)
    for name in ("zc", "zt"):
        errs = [abs(v - h.state_targets[name]) for v in h.states[name]]
        assert errs == sorted(errs, reverse=True), f"{name} does not approach"
        assert errs[-1] < 0.3 * errs[0]


@needs_pounce
def test_member_subset_states_label_by_their_reference():
    """A state declared as a slice of a packed Var (gh #20)."""

    def packed_model(N=5):
        m = pyo.ConcreteModel()
        m.t = ContinuousSet(initialize=pyo.RangeSet(0, N, 1))
        m.xA_ss = pyo.Param(initialize=0.5, mutable=True)
        m.u_ss = pyo.Param(initialize=0.5, mutable=True)
        m.xA_hat = pyo.Param(initialize=0.2, mutable=True)
        m.x = pyo.Var(m.t, ["A", "B"], initialize=0.2)
        m.dx = DerivativeVar(m.x, wrt=m.t)
        m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
        m.cost = pyo.Var(m.t)

        @m.Constraint(m.t)
        def ode(mm, t):
            return mm.dx[t, "A"] == mm.u[t] - mm.x[t, "A"]

        @m.Constraint(m.t)
        def alg(mm, t):
            return mm.x[t, "B"] == 2 * mm.x[t, "A"]

        @m.Constraint(sorted(m.t)[:-1])
        def stage(mm, t):
            return (
                mm.cost[t]
                == (mm.x[t, "A"] - mm.xA_ss) ** 2 + 0.1 * (mm.u[t] - mm.u_ss) ** 2
            )

        @m.Constraint()
        def x_init(mm):
            return mm.x[0, "A"] == mm.xA_hat

        drto.horizon(m.t)
        drto.state(m.x[:, "A"])
        drto.dynamics(m.ode)
        drto.control(m.u, profile="piecewise_constant")
        drto.tracking_stage_cost(m.stage)
        drto.initial_condition(m.x_init)
        drto.steady_state(m.x[:, "A"], m.xA_ss)
        drto.steady_state_control(m.u, m.u_ss)
        pyo.TransformationFactory("dae.collocation").apply_to(
            m, wrt=m.t, nfe=N, ncp=3, scheme="LAGRANGE-RADAU"
        )
        return m

    h = drto.ideal_nmpc(packed_model(), steps=4, initial_condition={"x_A": 0.3})
    assert list(h.states) == ["x_A"]
    assert h.states["x_A"][0] == pytest.approx(0.3)
    assert h.states["x_A"][-1] == pytest.approx(0.5, abs=1e-3)
    assert h.state_targets["x_A"] == pytest.approx(0.5)


# ── the solver plumbing ──────────────────────────────────────────────────────


@needs_ipopt
def test_warm_started_solves_get_the_recipe(monkeypatch):
    rec = _Recorder(pyo.SolverFactory("ipopt"))
    monkeypatch.setattr(loop_module, "SolverFactory", lambda name: rec)
    m = loop_model()
    drto.ideal_nmpc(m, steps=2, solver="ipopt")
    # call order: controller, process, controller (warm), process
    assert len(rec.calls) == 4
    assert "warm_start_init_point" not in rec.calls[0]
    assert rec.calls[1] == {}
    assert rec.calls[2]["warm_start_init_point"] == "yes"
    assert rec.calls[2]["mu_init"] == pytest.approx(1e-6)
    assert rec.calls[3] == {}
    # the loop declares no suffixes: the warm start is the shifted
    # values plus the recipe, nothing else
    assert m.component("dual") is None
    assert m.component("ipopt_zL_in") is None


@needs_ipopt
def test_warm_start_options_lay_over_the_recipe(monkeypatch):
    rec = _Recorder(pyo.SolverFactory("ipopt"))
    monkeypatch.setattr(loop_module, "SolverFactory", lambda name: rec)
    drto.ideal_nmpc(loop_model(), steps=2, solver="ipopt", warm_start={"mu_init": 1e-4})
    assert rec.calls[2]["mu_init"] == pytest.approx(1e-4)  # the override
    assert rec.calls[2]["warm_start_init_point"] == "yes"  # the rest stays


@needs_ipopt
def test_another_solver_warm_starts_on_the_shifted_values_alone(monkeypatch):
    rec = _Recorder(pyo.SolverFactory("ipopt"))
    monkeypatch.setattr(loop_module, "SolverFactory", lambda name: rec)
    m = loop_model()
    drto.ideal_nmpc(m, steps=2, solver="other")
    assert all(c == {} for c in rec.calls)
    assert m.component("dual") is None
    # a given mapping still reaches the warm solves as is
    rec = _Recorder(pyo.SolverFactory("ipopt"))
    monkeypatch.setattr(loop_module, "SolverFactory", lambda name: rec)
    drto.ideal_nmpc(loop_model(), steps=2, solver="other", warm_start={"max_iter": 400})
    assert rec.calls[2] == {"max_iter": 400}


@needs_ipopt
def test_a_failed_solve_names_the_step(monkeypatch):
    class Failing(_Recorder):
        def solve(self, model, **kwds):
            res = self.real.solve(model, **kwds)
            if len(self.calls) == len(self.fail_after):
                res.solver.termination_condition = (
                    pyo.TerminationCondition.maxIterations
                )
            self.calls.append({})
            return res

    rec = Failing(pyo.SolverFactory("ipopt"))
    rec.fail_after = [None, None]  # the third call, step 1's controller
    monkeypatch.setattr(loop_module, "SolverFactory", lambda name: rec)
    with pytest.raises(RuntimeError, match="controller solve failed at step 1"):
        drto.ideal_nmpc(loop_model(), steps=3, solver="ipopt")


@needs_ipopt
def test_cold_start_options_pass_through(monkeypatch):
    seen = []
    monkeypatch.setattr(
        loop_module, "cold_start_dynamic", lambda m, **kw: seen.append(kw)
    )
    drto.ideal_nmpc(
        loop_model(), steps=1, solver="ipopt", cold_start={"profile": "exponential"}
    )
    # the controller and the process cold-start alike
    assert seen == [{"profile": "exponential"}, {"profile": "exponential"}]
    seen.clear()
    drto.ideal_nmpc(loop_model(), steps=1, solver="ipopt", cold_start=False)
    assert seen == []


# ── scaling ──────────────────────────────────────────────────────────────────


@needs_ipopt
def test_scaled_loop_reproduces_the_unscaled_history():
    def tagged():
        m = loop_model()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        for vd in m.z.values():
            m.scaling_factor[vd] = 2.0
        for vd in m.u.values():
            m.scaling_factor[vd] = 4.0
        return m

    hs = drto.ideal_nmpc(tagged(), steps=5, solver="ipopt")
    hu = drto.ideal_nmpc(loop_model(), steps=5, solver="ipopt")
    assert hs.states["z"] == pytest.approx(hu.states["z"], abs=1e-6)
    assert hs.moves["u"] == pytest.approx(hu.moves["u"], abs=1e-6)
