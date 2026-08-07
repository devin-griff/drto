# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 022: drto.plotting."""
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet

import drto
from test_infinite_horizon import indexed_model, ready_model

IH = "drto.infinite_horizon"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _valued(m):
    """The plots read solved values; give every Var one."""
    for v in m.component_data_objects(pyo.Var, descend_into=True):
        if v.value is None:
            v.set_value(0.0)
    return m


def test_default_selection_expands_members():
    m = indexed_model()
    axes = drto.plot_states(m)
    assert [ax.get_title() for ax in axes] == ["x[1]", "x[2]"]


def test_controls_draw_as_a_staircase():
    m = _valued(ready_model())
    (ax,) = drto.plot_controls(m)
    (line,) = ax.get_lines()[:1]
    assert line.get_drawstyle() == "steps-post"
    # the last move holds to the end of the horizon
    assert line.get_xdata()[-1] == max(m.t)


def test_panel_cap_asks_for_members():
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1])
    m.x = pyo.Var(m.t, range(13))
    drto.horizon(m.t)
    drto.state(m.x)
    with pytest.raises(ValueError, match="13 panels"):
        drto.plot_states(m)


def test_tail_draws_with_a_segment():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    _valued(m)
    (ax,) = drto.plot_states(m)
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert "tail" in labels and "setpoint" in labels


def test_stage_cost_panel():
    m = _valued(ready_model())
    (ax,) = drto.plot_stage_cost(m)
    assert ax.get_title() == "cost"


# ── an NmpcHistory instead of a model (feature 014) ──────────────────────────


def _history():
    h = drto.NmpcHistory()
    h.times = [0, 1, 2]
    h.states = {"z": [0.2, 0.4, 0.5]}
    h.state_targets = {"z": 0.5}
    h.moves = {"u": [0.7, 0.6]}
    h.control_targets = {"u": 0.5}
    return h


def test_history_states_draw_points_at_the_samples():
    (ax,) = drto.plot_states(_history())
    assert ax.get_title() == "z"
    (line,) = ax.get_lines()[:1]
    assert list(line.get_xdata()) == [0, 1, 2]
    assert list(line.get_ydata()) == [0.2, 0.4, 0.5]
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert labels == ["actual", "setpoint"]


def test_history_moves_draw_as_a_staircase():
    (ax,) = drto.plot_controls(_history())
    (line,) = ax.get_lines()[:1]
    assert line.get_drawstyle() == "steps-post"
    # the last move holds to the final recorded instant
    assert list(line.get_xdata()) == [0, 1, 2]
    assert list(line.get_ydata()) == [0.7, 0.6, 0.6]


def test_history_selection_errors_on_unknown_labels():
    with pytest.raises(ValueError, match="not a recorded state"):
        drto.plot_states(_history(), states=["nope"])


def test_a_parameterized_control_keeps_its_segment_copy():
    # cvp replaces the profiled control, and the segment records pair a
    # declared component with its copy: without remapping, the copy stops
    # being reachable and plot_controls loses the tail (gh #70)
    from drto.plotting import _tail

    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    _valued(m)
    control = drto.info(m).components("control")[0]
    assert id(control) in _tail(m)[3]


def test_a_parameterized_control_draws_its_tail():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    _valued(m)
    (ax,) = drto.plot_controls(m)
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert "tail" in labels
