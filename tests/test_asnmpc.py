# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 015: drto.asnmpc."""
import contextlib
import sys

import pyomo.environ as pyo
import pytest

import drto
from drto.dynamic_optimization import _initial_condition_params
from drto.ideal_nmpc import _first_move
from test_ideal_nmpc import _Stop, hicks, loop_model

# drto.asnmpc the attribute is the function. The modules for
# monkeypatching come from the import system
asnmpc_module = sys.modules["drto.asnmpc"]
loop_module = sys.modules["drto.ideal_nmpc"]

try:
    import pyomo_pounce  # noqa: F401  registers the pounce solver
except ImportError:
    pass
pounce_ok = bool(drto.scaling.solver_by_name("pounce").available())
needs_pounce = pytest.mark.skipif(not pounce_ok, reason="pounce not available")


def _state(m):
    """The controller's initial-condition Param values, in declaration order."""
    return [
        pyo.value(p) for p in _initial_condition_params(drto.info(m), "test_asnmpc")
    ]


class _Spies:
    """Record the loop's solves, predictions, and corrections as they happen.

    The warm start runs straight after the prediction is written, so the
    Params it sees are the predicted state. The correction runs straight
    after the measurement is written, so the Params it sees are the
    measured state.
    """

    def __init__(self, monkeypatch):
        self.events, self.predictions, self.measurements = [], [], []
        self.corrected, self.background, self.kwargs = [], [], []
        real_solve = loop_module._Loop.solve
        real_warm = asnmpc_module.warm_start_dynamic
        real_correct = asnmpc_module.advanced_step_controller
        spies = self

        def solve(loop, model, what, step, options=None):
            spies.events.append((what, step))
            return real_solve(loop, model, what, step, options=options)

        def warm(m, **kwargs):
            spies.predictions.append(_state(m))
            return real_warm(m, **kwargs)

        def correct(m, **kwargs):
            spies.events.append(("correct",))
            spies.measurements.append(_state(m))
            spies.kwargs.append(kwargs)
            est = real_correct(m, **kwargs)
            firsts = [_first_move(u) for u in drto.info(m).components("control")]
            spies.corrected.append([est[f] for f in firsts])
            spies.background.append([pyo.value(f) for f in firsts])
            return est

        monkeypatch.setattr(loop_module._Loop, "solve", solve)
        monkeypatch.setattr(asnmpc_module, "warm_start_dynamic", warm)
        monkeypatch.setattr(asnmpc_module, "advanced_step_controller", correct)


# ── validation ───────────────────────────────────────────────────────────────


def test_the_solver_is_pounce():
    # the correction is a backsolve on the pounce factorization
    with pytest.raises(ValueError, match="pounce factorization"):
        drto.asnmpc(loop_model, steps=2, solver="ipopt")


def test_advanced_step_is_a_mapping():
    with pytest.raises(ValueError, match="advanced_step is a mapping"):
        drto.asnmpc(loop_model, steps=2, advanced_step=["clamp"])


def test_requires_the_model_statement():
    with pytest.raises(ValueError, match="model statement"):
        drto.asnmpc(loop_model(), steps=2)


def test_a_mesh_option_belongs_to_the_loop():
    with pytest.raises(ValueError, match="states the mesh once for every side"):
        drto.asnmpc(loop_model, steps=2, dynamic_optimization={"h": 2.0})


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        drto.asnmpc(loop_model, steps=0)


def test_three_sides_share_one_mesh(monkeypatch):
    # the controller, the process, and the predictor come from one
    # statement on one mesh, and every grid is settled before any solve
    built = []
    real = loop_module._build_and_discretize

    def spy(*a, **k):
        m = real(*a, **k)
        built.append(sorted(drto.info(m).components("horizon")[0]))
        if len(built) == 3:
            raise _Stop
        return m

    monkeypatch.setattr(loop_module, "_build_and_discretize", spy)
    with contextlib.suppress(_Stop):
        drto.asnmpc(loop_model, steps=1, h=2.0, ncp=2)
    ctrl_grid, process_grid, predictor_grid = built
    assert process_grid == predictor_grid == ctrl_grid[: len(process_grid)]
    assert process_grid[-1] == pytest.approx(2.0)


# ── the loop ─────────────────────────────────────────────────────────────────


@needs_pounce
def test_with_no_disturbance_the_loop_is_the_ideal_one(monkeypatch):
    # the feature's hicks criterion. With no disturbance the prediction is
    # the measurement, the correction is by zero, the moves are the ideal
    # loop's, and the states settle to the declared targets
    opts = dict(steps=10, dynamic_optimization={"infinite_horizon": True})
    ideal = drto.ideal_nmpc(hicks, **opts)
    spies = _Spies(monkeypatch)
    h = drto.asnmpc(hicks, **opts)

    assert len(spies.predictions) == len(spies.measurements) == 9
    for predicted, measured in zip(spies.predictions, spies.measurements):
        assert predicted == pytest.approx(measured, abs=1e-8)
    for corrected, background in zip(spies.corrected, spies.background):
        assert corrected == pytest.approx(background, abs=1e-8)
    for name in ideal.moves:
        assert h.moves[name] == pytest.approx(ideal.moves[name], abs=1e-7)
    for name in ("zc", "zt"):
        errs = [abs(v - h.state_targets[name]) for v in h.states[name]]
        assert errs == sorted(errs, reverse=True), f"{name} does not approach"
        assert errs[-1] < 0.3 * errs[0]


@needs_pounce
def test_the_correction_runs_before_the_next_solve(monkeypatch):
    spies = _Spies(monkeypatch)
    drto.asnmpc(loop_model, steps=3)
    expected = [("controller", 0)]
    for k in (0, 1):
        expected += [("predictor", k), ("controller", k), ("process", k)]
        expected.append(("correct",))
    # the last step implements its move and measures, nothing more
    expected.append(("process", 2))
    assert spies.events == expected


@needs_pounce
def test_the_predictor_holds_its_disturbances_at_zero(monkeypatch):
    built = []
    real = loop_module._build_and_discretize

    def keep(*a, **k):
        m = real(*a, **k)
        built.append(m)
        return m

    monkeypatch.setattr(loop_module, "_build_and_discretize", keep)
    spies = _Spies(monkeypatch)
    h = drto.asnmpc(loop_model, steps=3, disturbances={"w": [0.1, 0.1, 0.1]})
    _ctrl, process, predictor = built

    assert h.realizations["w"] == [0.1, 0.1, 0.1]
    assert all(pyo.value(vd) == 0.0 for vd in predictor.w.values())
    assert all(pyo.value(vd) == 0.1 for vd in process.w.values())
    # the realization moves the measurement off the prediction, and the
    # correction moves the moves off the background solution
    for predicted, measured in zip(spies.predictions, spies.measurements):
        assert abs(measured[0] - predicted[0]) > 1e-2
    for corrected, background in zip(spies.corrected, spies.background):
        assert corrected != pytest.approx(background, abs=1e-6)


@needs_pounce
def test_advanced_step_options_reach_the_correction(monkeypatch):
    spies = _Spies(monkeypatch)
    drto.asnmpc(loop_model, steps=3, advanced_step={"clamp": False})
    assert spies.kwargs == [{"clamp": False}, {"clamp": False}]


@needs_pounce
def test_the_history_and_the_logs(capsys):
    h = drto.asnmpc(loop_model, steps=3, tee=True, disturbances={"w": 0.05}, seed=1)
    capsys.readouterr()
    assert h.times == [0, 1, 2, 3]
    assert len(h.states["z"]) == 4
    assert len(h.moves["u"]) == 3
    assert len(h.realizations["w"]) == 3
    assert h.state_targets == {"z": 0.5}
    assert h.control_targets == {"u": 0.5}
    sides = [what for _step, what, _text in h.logs]
    assert sides.count("controller") == 3
    assert sides.count("predictor") == 2
    assert sides.count("process") == 3
    assert all(text for _step, _what, text in h.logs)


@needs_pounce
def test_scaled_loop_reproduces_the_unscaled_history():
    def tagged(N=5, h=1):
        m = loop_model(N=N, h=h)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        for vd in m.z.values():
            m.scaling_factor[vd] = 2.0
        for vd in m.u.values():
            m.scaling_factor[vd] = 4.0
        return m

    noise = {"w": [0.1, -0.05, 0.08, 0.0, -0.1]}
    hs = drto.asnmpc(tagged, steps=5, disturbances=noise)
    hu = drto.asnmpc(loop_model, steps=5, disturbances=noise)
    assert hs.states["z"] == pytest.approx(hu.states["z"], abs=1e-6)
    assert hs.moves["u"] == pytest.approx(hu.moves["u"], abs=1e-6)


@needs_pounce
def test_the_loop_releases_the_controller_factorization(monkeypatch):
    # pounce raises when its solver object is freed on another thread, so
    # the loop frees the factorization before it discards the controller
    import pyomo_pounce

    built = []
    real = loop_module._build_and_discretize

    def keep(*a, **k):
        m = real(*a, **k)
        built.append(m)
        return m

    monkeypatch.setattr(loop_module, "_build_and_discretize", keep)
    drto.asnmpc(loop_model, steps=2)
    assert pyomo_pounce.sens_release_kkt(built[0]) is False
