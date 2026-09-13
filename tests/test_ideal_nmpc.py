# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 014: drto.ideal_nmpc."""
import contextlib
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
pounce_ok = bool(drto.scaling.solver_by_name("pounce").available())
needs_pounce = pytest.mark.skipif(not pounce_ok, reason="pounce not available")
ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def loop_model(N=5, h=1):
    """dz = u - z + w, with the state and control targets meeting at 0.5."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[i * h for i in range(N + 1)])
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
    return m


class _Stop(Exception):
    """Raised by a spy to end the loop once it has seen what it needs."""


class _Available:
    """A solver that resolves and reports itself available, solving nothing."""

    def available(self):
        return True


class _Recorder:
    """Wraps a real native solver, recording each solve's options and model."""

    def __init__(self, real):
        self.real, self.calls, self.models = real, [], []

    def available(self):
        return True

    def solve(self, model, **kwds):
        self.calls.append(dict(kwds.get("solver_options") or {}))
        self.models.append(model)
        return self.real.solve(model, **kwds)


# ── validation ───────────────────────────────────────────────────────────────


def test_requires_the_model_statement():
    # the loop builds both sides itself, so it takes the builder
    with pytest.raises(ValueError, match="model statement"):
        drto.ideal_nmpc(loop_model(), steps=2)


def test_a_mesh_option_belongs_to_the_loop():
    with pytest.raises(ValueError, match="states the mesh once"):
        drto.ideal_nmpc(loop_model, steps=2, dynamic_optimization={"ncp": 2})


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        drto.ideal_nmpc(loop_model, steps=0)


def test_unknown_state_name_errors():
    with pytest.raises(ValueError, match="not a pinned state"):
        drto.ideal_nmpc(loop_model, steps=2, initial_condition={"nope": 1.0})


def test_unknown_disturbance_name_errors():
    with pytest.raises(ValueError, match="not a declared disturbance"):
        drto.ideal_nmpc(loop_model, steps=2, disturbances={"nope": 0.1})


def test_short_disturbance_sequence_errors():
    with pytest.raises(ValueError, match="one per step"):
        drto.ideal_nmpc(loop_model, steps=3, disturbances={"w": [0.1]})


def test_an_unknown_solver_names_the_registry():
    with pytest.raises(ValueError, match="native factory"):
        drto.ideal_nmpc(loop_model, steps=2, solver="no_such_solver")


# ── the loop ─────────────────────────────────────────────────────────────────


@needs_pounce
def test_loop_settles_and_records():
    h = drto.ideal_nmpc(loop_model, steps=8, seed=0)
    assert h.times == list(range(9))
    assert len(h.states["z"]) == 9 and len(h.moves["u"]) == 8
    assert h.states["z"][0] == pytest.approx(0.2)
    assert h.states["z"][-1] == pytest.approx(0.5, abs=1e-4)
    assert h.moves["u"][-1] == pytest.approx(0.5, abs=1e-3)
    assert h.state_bounds["z"] == (-1, 2)
    assert h.control_bounds["u"] == (0, 1)
    assert h.state_targets["z"] == pytest.approx(0.5)
    assert h.control_targets["u"] == pytest.approx(0.5)
    assert h.realizations["w"] == [0.0] * 8


@needs_pounce
def test_initial_condition_reaches_the_first_solve():
    h = drto.ideal_nmpc(loop_model, steps=2, initial_condition={"z": 0.4})
    assert h.states["z"][0] == pytest.approx(0.4)
    # one step from 0.4 lands closer to the target than one from 0.2
    assert abs(h.states["z"][1] - 0.5) < 0.01


@needs_pounce
def test_constant_disturbance_offsets_the_plant():
    h = drto.ideal_nmpc(loop_model, steps=8, disturbances={"w": [0.2] * 8})
    assert h.realizations["w"] == [0.2] * 8
    # the controller plans at zero noise, so the plant holds an offset
    assert h.states["z"][-1] == pytest.approx(0.63624, abs=1e-3)


@needs_pounce
def test_draws_are_reproducible_under_seed():
    kw = dict(steps=3, disturbances={"w": 0.05})
    a = drto.ideal_nmpc(loop_model, seed=3, **kw)
    b = drto.ideal_nmpc(loop_model, seed=3, **kw)
    c = drto.ideal_nmpc(loop_model, seed=4, **kw)
    assert a.realizations["w"] == b.realizations["w"]
    assert a.realizations["w"] != c.realizations["w"]
    assert a.states["z"] == pytest.approx(b.states["z"])


@needs_pounce
def test_hicks_settles_to_the_declared_targets():
    h = drto.ideal_nmpc(
        hicks, steps=10, dynamic_optimization={"infinite_horizon": True}
    )
    for name in ("zc", "zt"):
        errs = [abs(v - h.state_targets[name]) for v in h.states[name]]
        assert errs == sorted(errs, reverse=True), f"{name} does not approach"
        assert errs[-1] < 0.3 * errs[0]


@needs_pounce
def test_member_subset_states_label_by_their_reference():
    """A state declared as a slice of an indexed Var (gh #20)."""

    def packed_model(N=5, h=1):
        m = pyo.ConcreteModel()
        m.t = ContinuousSet(initialize=[i * h for i in range(N + 1)])
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
        return m

    h = drto.ideal_nmpc(packed_model, steps=4, initial_condition={"x_A": 0.3})
    assert list(h.states) == ["x_A"]
    assert h.states["x_A"][0] == pytest.approx(0.3)
    assert h.states["x_A"][-1] == pytest.approx(0.5, abs=1e-3)
    assert h.state_targets["x_A"] == pytest.approx(0.5)


# ── the solver plumbing ──────────────────────────────────────────────────────


@needs_ipopt
def test_warm_started_solves_get_the_recipe(monkeypatch):
    rec = _Recorder(drto.scaling.solver_by_name("ipopt"))
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)
    drto.ideal_nmpc(loop_model, steps=2, solver="ipopt")
    # call order: controller, process, controller (warm), process
    assert len(rec.calls) == 4
    assert "warm_start_init_point" not in rec.calls[0]
    assert rec.calls[1] == {}
    assert rec.calls[2]["warm_start_init_point"] == "yes"
    assert rec.calls[2]["mu_init"] == pytest.approx(1e-6)
    assert rec.calls[3] == {}
    # the loop declares no suffixes: the warm start is the shifted
    # values plus the recipe, nothing else
    ctrl = rec.models[0]
    assert ctrl.component("dual") is None
    assert ctrl.component("ipopt_zL_in") is None


@needs_ipopt
def test_warm_start_options_lay_over_the_recipe(monkeypatch):
    rec = _Recorder(drto.scaling.solver_by_name("ipopt"))
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)
    drto.ideal_nmpc(loop_model, steps=2, solver="ipopt", warm_start={"mu_init": 1e-4})
    assert rec.calls[2]["mu_init"] == pytest.approx(1e-4)  # the override
    assert rec.calls[2]["warm_start_init_point"] == "yes"  # the rest stays


@needs_ipopt
def test_another_solver_warm_starts_on_the_shifted_values_alone(monkeypatch):
    rec = _Recorder(drto.scaling.solver_by_name("ipopt"))
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)
    drto.ideal_nmpc(loop_model, steps=2, solver="other")
    assert all(c == {} for c in rec.calls)
    # a given mapping still reaches the warm solves as is
    rec = _Recorder(drto.scaling.solver_by_name("ipopt"))
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)
    drto.ideal_nmpc(loop_model, steps=2, solver="other", warm_start={"max_iter": 400})
    assert rec.calls[2] == {"max_iter": 400}


@needs_ipopt
def test_pounce_warm_solves_carry_mu_init_alone(monkeypatch):
    # measured on the CSTR warm start: mu_init takes the shifted solve
    # from ten iterations to seven, and the full recipe regresses
    # pounce, so its warm solves carry the small barrier and nothing
    # else
    rec = _Recorder(drto.scaling.solver_by_name("ipopt"))
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)
    drto.ideal_nmpc(loop_model, steps=2, solver="pounce_v2")
    assert rec.calls[2]["mu_init"] == pytest.approx(1e-6)
    assert "warm_start_init_point" not in rec.calls[2]
    assert "warm_start_bound_push" not in rec.calls[2]


@needs_ipopt
def test_a_failed_solve_names_the_step(monkeypatch):
    class Failing(_Recorder):
        def solve(self, model, **kwds):
            res = self.real.solve(model, **kwds)
            if len(self.calls) == len(self.fail_after):
                from pyomo.contrib.solver.common.results import TerminationCondition

                res.termination_condition = TerminationCondition.iterationLimit
            self.calls.append({})
            return res

    rec = Failing(drto.scaling.solver_by_name("ipopt"))
    rec.fail_after = [None, None]  # the third call, step 1's controller
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)
    with pytest.raises(RuntimeError, match="controller solve failed at step 1"):
        drto.ideal_nmpc(loop_model, steps=3, solver="ipopt")


@needs_ipopt
def test_initialize_mapping_passes_to_the_cold_start(monkeypatch):
    seen = []
    monkeypatch.setattr(
        loop_module, "cold_start_dynamic", lambda m, **kw: seen.append(kw)
    )
    drto.ideal_nmpc(
        loop_model, steps=1, solver="ipopt", initialize={"profile": "exponential"}
    )
    # the controller and the process cold-start alike
    assert seen == [{"profile": "exponential"}, {"profile": "exponential"}]
    seen.clear()
    drto.ideal_nmpc(loop_model, steps=1, solver="ipopt", initialize=False)
    assert seen == []


def test_initialize_rejects_unknown_values():
    with pytest.raises(ValueError, match="'cold'"):
        drto.ideal_nmpc(loop_model, steps=1, initialize="warm")


@needs_ipopt
def test_initialize_steady_runs_on_the_input_before_the_sides(monkeypatch):
    calls = []

    def spy(mm):
        # the input is untransformed at this point: the broadcast must
        # precede the mode transforms so both sides inherit it
        assert not drto.info(mm).transformations
        calls.append(mm)

    monkeypatch.setattr(loop_module, "initialize_steady_state", spy)
    drto.ideal_nmpc(loop_model, steps=1, solver="ipopt", initialize="steady")
    # once per side, each before that side's transforms
    assert len(calls) == 2


@needs_pounce
def test_initialize_steady_initializes_the_loop():
    # hicks: no declared disturbance, so its steady reduction is square
    # (initialize_steady_state leaves a declared disturbance free, its
    # own descriptive error; the loop adds nothing to that contract)
    h = drto.ideal_nmpc(hicks, steps=2, initialize="steady")
    assert h.states["zc"][0] == pytest.approx(0.625)  # the hooks still rule
    # from the flat steady start the first step still moves to target
    assert abs(h.states["zc"][1] - 0.6416) < abs(0.625 - 0.6416)


@needs_pounce
def test_initialize_steady_runs_with_a_terminal_segment():
    # the loop initializes each side after discretization and before its
    # transforms, so the segment no longer precedes the broadcast
    h = drto.ideal_nmpc(
        hicks,
        steps=1,
        initialize="steady",
        dynamic_optimization={"infinite_horizon": True},
    )
    assert h.states["zc"][0] == pytest.approx(0.625)


@needs_ipopt
def test_the_plant_is_the_one_sample_simulation(monkeypatch):
    seen = []

    class Rec(_Recorder):
        def solve(self, model, **kwds):
            seen.append(model)
            return super().solve(model, **kwds)

    rec = Rec(drto.scaling.solver_by_name("ipopt"))
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)
    drto.ideal_nmpc(
        hicks, steps=1, solver="ipopt", dynamic_optimization={"infinite_horizon": True}
    )
    plant = seen[1]
    # the terminal segment serves the horizon problem, not the plant
    assert plant.component("drto_ih") is None
    # nothing active lies past one sampling time
    from drto.infinite_horizon import _split_index, _time_index

    reg = drto.info(plant)
    time = reg.components("horizon")[0]
    t1 = reg.declarations("horizon")[0]["samples"][1]
    for ctype in (pyo.Constraint, pyo.Var):
        for cd in plant.component_data_objects(ctype, active=True):
            if ctype is pyo.Var and cd.fixed:
                continue
            pos, subs = _time_index(cd.parent_component(), time)
            if pos is None:
                continue
            _o, t = _split_index(cd.index(), pos, len(subs))
            assert t is None or t <= t1 + 1e-9, cd.name


@needs_ipopt
def test_tee_streams_and_returns_every_solves_output(capsys):
    h = drto.ideal_nmpc(loop_model, steps=2, solver="ipopt", tee=True)
    streamed = capsys.readouterr().out
    assert "Number of Iterations" in streamed
    assert [(s, w) for s, w, _t in h.logs] == [
        (0, "controller"),
        (0, "process"),
        (1, "controller"),
        (1, "process"),
    ]
    assert all("Number of Iterations" in text for _s, _w, text in h.logs)
    quiet = drto.ideal_nmpc(loop_model, steps=1, solver="ipopt")
    assert quiet.logs == []
    assert "Number of Iterations" not in capsys.readouterr().out


def test_both_sides_share_one_mesh(monkeypatch):
    # stating h, ncp, and scheme once is what makes the two grids agree,
    # captured where the sides are built since the history carries no model
    built = []
    real = loop_module._build_and_discretize

    def spy(*a, **k):
        m = real(*a, **k)
        built.append(sorted(drto.info(m).components("horizon")[0]))
        if len(built) == 2:
            raise _Stop
        return m

    monkeypatch.setattr(loop_module, "_build_and_discretize", spy)
    monkeypatch.setattr(
        loop_module.drto_scaling, "solver_by_name", lambda name: _Available()
    )
    # both grids are settled before any solve, so stopping at the second
    # build keeps this solver-free
    with contextlib.suppress(_Stop):
        drto.ideal_nmpc(loop_model, steps=1, h=2.0, ncp=2, solver="ipopt")
    assert len(built) == 2
    ctrl_grid, plant_grid = built
    # the plant is one sampling interval of the controller's own grid
    assert plant_grid == ctrl_grid[: len(plant_grid)]
    assert plant_grid[-1] == pytest.approx(2.0)


@needs_ipopt
def test_a_suffix_loop_solves_the_models_themselves(monkeypatch):
    # no clone anywhere: the controller solve is the input model and
    # the factors travel as the user-scaling option (gh #92)
    seen = []

    class Rec(_Recorder):
        def solve(self, model, **kwds):
            seen.append(model)
            return super().solve(model, **kwds)

    rec = Rec(drto.scaling.solver_by_name("ipopt"))
    monkeypatch.setattr(loop_module.drto_scaling, "solver_by_name", lambda name: rec)

    def tagged(N=5, h=1):
        m = loop_model(N=N, h=h)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        for vd in m.z.values():
            m.scaling_factor[vd] = 2.0
        return m

    drto.ideal_nmpc(tagged, steps=1, solver="ipopt")
    assert seen[0].component("scaling_factor") is not None
    # ipopt needs no option: the NL writer consumes the Suffix and
    # scales the problem as it writes
    assert all("nlp_scaling_method" not in opts for opts in rec.calls)


# ── scaling ──────────────────────────────────────────────────────────────────


@needs_ipopt
def test_scaled_loop_reproduces_the_unscaled_history():
    def tagged(N=5, h=1):
        m = loop_model(N=N, h=h)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        for vd in m.z.values():
            m.scaling_factor[vd] = 2.0
        for vd in m.u.values():
            m.scaling_factor[vd] = 4.0
        return m

    hs = drto.ideal_nmpc(tagged, steps=5, solver="ipopt")
    hu = drto.ideal_nmpc(loop_model, steps=5, solver="ipopt")
    assert hs.states["z"] == pytest.approx(hu.states["z"], abs=1e-6)
    assert hs.moves["u"] == pytest.approx(hu.moves["u"], abs=1e-6)


@needs_ipopt
def test_scale_writes_the_factors_and_the_default_does_not(monkeypatch):
    built = []
    real = loop_module._build_and_discretize

    def spy(*a, **k):
        m = real(*a, **k)
        built.append(m)
        return m

    monkeypatch.setattr(loop_module, "_build_and_discretize", spy)
    drto.ideal_nmpc(loop_model, steps=1, solver="ipopt", scale="point")
    assert all(m.component("scaling_factor") is not None for m in built)
    built.clear()
    drto.ideal_nmpc(loop_model, steps=1, solver="ipopt")
    assert all(m.component("scaling_factor") is None for m in built)
