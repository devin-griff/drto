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


def test_element_boundaries_stay_off_by_default():
    # the boundary value is the element polynomial extended to its edge,
    # so on a converged trajectory it is the only point off the curve and
    # would set the axis limits
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    _valued(m)
    (ax,) = drto.plot_states(m)
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert "tail" in labels
    assert "element boundary" not in labels


def test_element_boundaries_draw_on_request():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    _valued(m)
    (ax,) = drto.plot_states(m, element_boundaries=True)
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert "element boundary" in labels


def test_ticks_never_use_offset_notation():
    h = _history()
    h.states = {"z": [-714.55, -714.6, -714.62]}
    h.state_targets = {"z": -714.6}
    (ax,) = drto.plot_states(h)
    assert not ax.yaxis.get_major_formatter().get_useOffset()
    (ax,) = drto.plot_states(_valued(ready_model()))
    assert not ax.yaxis.get_major_formatter().get_useOffset()


def test_bounds_draw_within_the_data_window():
    h = _history()
    h.state_bounds = {"z": (0.0, 1.0)}
    (ax,) = drto.plot_states(h)
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert "bound" in labels
    levels = sorted(
        line.get_ydata()[0]
        for line in ax.get_lines()
        if line.get_linestyle() == "--" and len(set(line.get_ydata())) == 1
    )
    assert levels == [0.0, 1.0]
    # the axis window comes from the data, not the bounds
    assert ax.get_ylim()[1] < 1.0


def test_model_plots_draw_the_declared_bounds():
    m = _valued(ready_model())
    (ax,) = drto.plot_controls(m)
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert "bound" in labels
    levels = sorted(
        line.get_ydata()[0]
        for line in ax.get_lines()
        if line.get_linestyle() == "--" and len(set(line.get_ydata())) == 1
    )
    assert levels == [0.0, 1.0]
    # an unbounded state draws no bound line
    (ax,) = drto.plot_states(m)
    labels = [t.get_text() for t in ax.figure.legends[0].texts]
    assert "bound" not in labels


# ── a fitted policy instead of a model (feature 026) ─────────────────────────


def _policy_and_data():
    """A quickly fitted policy on the toy dataset, and that dataset."""
    from test_approximate_nmpc_train import toy_dataset

    data = toy_dataset(n=40)
    policy = drto.approximate_nmpc_train(
        data, epochs=40, hidden=(8,), schedule="flat", seeds=1
    )
    return policy, data


def test_the_history_draws_both_curves():
    policy, _ = _policy_and_data()
    (ax,) = drto.plot_history(policy)
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["training loss", "validation loss"]
    assert ax.get_yscale() == "log"
    assert len(ax.get_lines()[0].get_xdata()) == len(policy.history["epoch"])


def test_parity_draws_a_panel_per_control_and_splits_by_the_recorded_index():
    policy, data = _policy_and_data()
    held = policy.meta["validation_index"]
    assert held and len(held) == 8  # a fifth of 40, split off by fraction
    axes = drto.plot_parity(policy, data)
    assert [ax.get_title() for ax in axes] == list(data.config["u_bounds"])
    for ax in axes:
        which = [t.get_text() for t in ax.get_legend().get_texts()]
        assert [w.split(",")[0] for w in which] == ["training", "validation"]
        assert all("R^2" in w for w in which)
        counts = [len(c.get_offsets()) for c in ax.collections]
        assert counts == [len(data.points) - len(held), len(held)]


def test_parity_takes_a_validation_dataset_and_falls_back_to_one_series():
    from test_approximate_nmpc_train import toy_dataset

    policy, data = _policy_and_data()
    axes = drto.plot_parity(policy, data, validation=toy_dataset(n=12, seed=3))
    counts = [len(c.get_offsets()) for c in axes[0].collections]
    assert counts == [len(data.points), 12]
    policy.meta["validation_index"] = None
    axes = drto.plot_parity(policy, data)
    which = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert len(which) == 1 and which[0].startswith("sampled")


def test_the_legend_band_is_the_same_at_every_row_count():
    def above(n):
        h = drto.NmpcHistory()
        h.times = [0, 1, 2]
        h.states = {f"z{i}": [0.2, 0.4, 0.5] for i in range(n)}
        h.state_targets = {f"z{i}": 0.5 for i in range(n)}
        axes = drto.plot_states(h)
        fig = axes[0].figure
        top = max(ax.get_position().y1 for ax in axes)
        return (1 - top) * fig.get_figheight()

    one, three = above(2), above(6)
    assert one == pytest.approx(three, abs=0.02)
