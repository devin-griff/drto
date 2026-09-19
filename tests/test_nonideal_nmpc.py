# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 016: drto.nonideal_nmpc."""
import sys
import types
from functools import partial

import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from test_ideal_nmpc import _Available, hicks, loop_model

loop_module = sys.modules["drto.ideal_nmpc"]
nonideal_module = sys.modules["drto.nonideal_nmpc"]

try:
    import pyomo_pounce  # noqa: F401  registers the pounce solver
except ImportError:
    pass
pounce_ok = bool(drto.scaling.solver_by_name("pounce").available())
needs_pounce = pytest.mark.skipif(not pounce_ok, reason="pounce not available")

U = pyo.units


def moles_model(N=5, h=1, rate_units=False):
    """dn/dt = q - n in moles, the DerivativeVar's units optional.

    Pyomo gives a DerivativeVar no units unless the model writer passes
    them, so the declared time units read as mol here, which the solver's
    seconds do not convert to. With ``rate_units`` they read as s.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[i * h for i in range(N + 1)])
    m.n_ss = pyo.Param(initialize=1.0, mutable=True, units=U.mol)
    m.q_ss = pyo.Param(initialize=1.0, mutable=True, units=U.mol)
    m.n_hat = pyo.Param(initialize=0.5, mutable=True, units=U.mol)
    m.n = pyo.Var(m.t, initialize=0.5, units=U.mol)
    m.dn = DerivativeVar(m.n, wrt=m.t, units=U.mol / U.s if rate_units else None)
    m.q = pyo.Var(m.t, initialize=1.0, units=U.mol)
    m.cost = pyo.Var(m.t)

    @m.Constraint(m.t)
    def ode(mm, t):
        return mm.dn[t] == mm.q[t] - mm.n[t]

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (
            ((mm.n[t] - mm.n_ss) / U.mol) ** 2 + ((mm.q[t] - mm.q_ss) / U.mol) ** 2
        )

    @m.Constraint()
    def n_init(mm):
        return mm.n[0] == mm.n_hat

    drto.horizon(m.t)
    drto.state(m.n)
    drto.dynamics(m.ode)
    drto.control(m.q, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.n_init)
    drto.steady_state(m.n, m.n_ss)
    drto.steady_state_control(m.q, m.q_ss)
    return m


class _Result:
    """A solve's results, carrying only the reported wall time."""

    def __init__(self, seconds):
        self.timing_info = types.SimpleNamespace(wall_time=seconds)


def _piece_lengths(monkeypatch, seconds=None):
    """Record the plant's piece length at every process solve."""
    pieces = []
    real = loop_module._Loop.solve

    def solve(loop, model, what, step, options=None):
        if what == "process":
            pieces.append(pyo.value(model.component("_drto_duration")))
        if seconds is not None:
            return _Result(seconds)
        return real(loop, model, what, step, options=options)

    monkeypatch.setattr(loop_module._Loop, "solve", solve)
    return pieces


def _plant_moves(monkeypatch):
    """Record the control the plant holds at every process solve."""
    held = []
    real = loop_module._Loop.solve

    def solve(loop, model, what, step, options=None):
        if what == "process":
            u = drto.info(model).components("control")[0]
            held.append(pyo.value(next(iter(u.values()))))
        return real(loop, model, what, step, options=options)

    monkeypatch.setattr(loop_module._Loop, "solve", solve)
    return held


# ── validation ───────────────────────────────────────────────────────────────


def test_requires_the_model_statement():
    with pytest.raises(ValueError, match="model statement"):
        drto.nonideal_nmpc(loop_model(), steps=2)


def test_a_mesh_option_belongs_to_the_loop():
    with pytest.raises(ValueError, match="states the mesh once for every side"):
        drto.nonideal_nmpc(loop_model, steps=2, dynamic_optimization={"ncp": 2})


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        drto.nonideal_nmpc(loop_model, steps=0)


def test_an_unknown_delay_errors():
    with pytest.raises(ValueError, match="delay is 'solver'"):
        drto.nonideal_nmpc(loop_model, steps=2, delay="measured")


def test_a_short_delay_sequence_errors():
    with pytest.raises(ValueError, match="one per step"):
        drto.nonideal_nmpc(loop_model, steps=3, delay=[0.1, 0.1])


@pytest.mark.parametrize("build", [hicks, moles_model])
def test_solver_delay_needs_convertible_time_units(build):
    # hicks carries no units at all, and moles_model carries them on the
    # state alone, so the ratio reads mol. Seconds convert to neither
    with pytest.raises(ValueError, match="Seconds do not convert"):
        drto.nonideal_nmpc(build, steps=2, delay="solver")


# ── the pieces ───────────────────────────────────────────────────────────────


@needs_pounce
def test_the_interval_runs_in_two_exact_pieces(monkeypatch):
    pieces = _piece_lengths(monkeypatch)
    h = drto.nonideal_nmpc(loop_model, steps=3, delay=0.3)
    assert pieces == pytest.approx([0.3, 0.7] * 3)
    assert h.effect_times == pytest.approx([0.3, 1.3, 2.3])
    assert h.clamped == []


@needs_pounce
@pytest.mark.parametrize("delay", [0.0, 1.0])
def test_a_zero_length_piece_is_not_simulated(monkeypatch, delay):
    pieces = _piece_lengths(monkeypatch)
    drto.nonideal_nmpc(loop_model, steps=3, delay=delay)
    assert pieces == pytest.approx([1.0] * 3)


@needs_pounce
def test_delay_zero_reproduces_the_ideal_loop():
    opts = dict(steps=5, initial_condition={"z": 0.2}, disturbances={"w": 0.05}, seed=3)
    ideal = drto.ideal_nmpc(loop_model, **opts)
    here = drto.nonideal_nmpc(loop_model, delay=0.0, **opts)
    assert here.realizations == ideal.realizations
    assert here.states["z"] == ideal.states["z"]
    assert here.moves["u"] == ideal.moves["u"]


@needs_pounce
def test_the_first_previous_move_is_the_declared_target(monkeypatch):
    held = _plant_moves(monkeypatch)
    h = drto.nonideal_nmpc(loop_model, steps=3, delay=0.4)
    # the first piece of every interval holds the previous move, the
    # declared control target before the first move takes effect
    assert held[0] == pytest.approx(h.control_targets["u"])
    assert held[2] == pytest.approx(h.moves["u"][0])
    assert held[4] == pytest.approx(h.moves["u"][1])


@needs_pounce
def test_a_delay_of_one_interval_holds_every_move(monkeypatch):
    held = _plant_moves(monkeypatch)
    h = drto.nonideal_nmpc(loop_model, steps=3, delay=1.0)
    assert held == pytest.approx([h.control_targets["u"]] + h.moves["u"][:-1])
    assert h.clamped == [0, 1, 2]
    assert h.effect_times == pytest.approx([1.0, 2.0, 3.0])


@needs_pounce
def test_a_delay_past_the_interval_is_clamped(monkeypatch):
    held = _plant_moves(monkeypatch)
    h = drto.nonideal_nmpc(loop_model, steps=3, delay=1.5)
    assert h.delays == pytest.approx([1.5, 1.5, 1.5])
    assert h.clamped == [0, 1, 2]
    # each move takes effect at the next boundary, so the interval runs
    # in one piece under the move before it
    assert h.effect_times == pytest.approx([1.0, 2.0, 3.0])
    assert held == pytest.approx([h.control_targets["u"]] + h.moves["u"][:-1])


@needs_pounce
def test_hicks_settles_with_a_delay(monkeypatch):
    h = drto.nonideal_nmpc(
        hicks, steps=10, delay=0.4, dynamic_optimization={"infinite_horizon": True}
    )
    for name in ("zc", "zt"):
        errs = [abs(v - h.state_targets[name]) for v in h.states[name]]
        assert errs[-1] < 0.3 * errs[0]


@needs_pounce
def test_the_duration_param_scales_the_piece():
    # a piece of 0.4 on the plant built at h=1 reaches the state a plant
    # built at h=0.4 reaches, which is what the scaled dynamics mean
    scaled = drto.dynamic_simulation(loop_model, N=1, h=1, controls={"u": 0.7})
    short = drto.dynamic_simulation(loop_model, N=1, h=0.4, controls={"u": 0.7})
    nonideal_module._duration_param(
        scaled, 1.0, pyo.units.dimensionless, "test_nonideal_nmpc"
    )
    scaled.component("_drto_duration").set_value(0.4)
    for m in (scaled, short):
        m.z_hat.set_value(0.2)
        drto.scaled_solve(m, solver="pounce")
    assert pyo.value(scaled.z[1]) == pytest.approx(pyo.value(short.z[0.4]), abs=1e-8)


# ── the delay the solver reports ─────────────────────────────────────────────


def test_the_solver_seconds_convert_into_the_declared_units(monkeypatch):
    # the declared units are seconds here, so the reported wall time is
    # the delay as it stands. No solve runs: the stub returns the time
    units_model = partial(moles_model, rate_units=True)

    pieces = _piece_lengths(monkeypatch, seconds=0.25)
    monkeypatch.setattr(
        loop_module.drto_scaling, "solver_by_name", lambda name: _Available()
    )
    h = drto.nonideal_nmpc(units_model, steps=3, delay="solver", initialize=False)
    assert h.delays == pytest.approx([0.25, 0.25, 0.25])
    assert h.effect_times == pytest.approx([0.25, 1.25, 2.25])
    assert pieces == pytest.approx([0.25, 0.75] * 3)


# ── the history and the plot ─────────────────────────────────────────────────


@needs_pounce
def test_the_history_records_one_entry_per_step():
    h = drto.nonideal_nmpc(loop_model, steps=3, delay=0.2, disturbances={"w": 0.05})
    assert h.times == [0, 1, 2, 3]
    assert len(h.states["z"]) == 4
    assert len(h.moves["u"]) == len(h.delays) == len(h.effect_times) == 3
    assert len(h.realizations["w"]) == 3


def test_the_moves_step_at_the_effect_times():
    h = drto.NonidealNmpcHistory()
    h.times = [0, 1, 2]
    h.moves = {"u": [0.7, 0.6]}
    h.control_targets = {"u": 0.5}
    h.delays = [0.3, 0.3]
    h.effect_times = [0.3, 1.3]
    (ax,) = drto.plot_controls(h)
    (line,) = ax.get_lines()[:1]
    assert line.get_drawstyle() == "steps-post"
    # the declared target holds until the first move takes effect, and
    # the last move holds to the final recorded instant
    assert list(line.get_xdata()) == [0, 0.3, 1.3, 2]
    assert list(line.get_ydata()) == [0.5, 0.7, 0.6, 0.6]
