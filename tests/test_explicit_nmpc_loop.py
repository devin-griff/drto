# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 026: drto.explicit_nmpc_closed_loop."""
import matplotlib

matplotlib.use("Agg")

import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

pounce_ok = pyo.SolverFactory("pounce").available(exception_flag=False)
needs_pounce = pytest.mark.skipif(not pounce_ok, reason="pounce not available")


def assembled_model():
    """dz/dt = -z + u + w, tracking cost, assembled for optimization."""
    m = pyo.ConcreteModel()
    N, h = 4, 2.5
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z = pyo.Var(m.t, bounds=(0, 1), initialize=0.4)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.w = pyo.Var(m.t, initialize=0.0)
    m.z_ss = pyo.Param(initialize=0.3, mutable=True)
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)
    m.cost = pyo.Var(m.t)
    ts = sorted(m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t] + m.w[t]

    @m.Constraint(ts[:-1])
    def stage(m, t):
        return m.cost[t] == (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

    @m.Constraint()
    def init(m):
        return m.z[0] == m.z_hat

    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.control(m.u, profile="piecewise_constant")
    drto.disturbance(m.w)
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    drto.steady_state(m.z, m.z_ss)
    drto.steady_state_control(m.u, m.u_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    return m


def fitted(m):
    """A quickly trained policy on the model's own labels."""
    data = drto.explicit_nmpc_data(m, n=8, seed=0)
    return drto.explicit_nmpc_train(
        data, epochs=300, hidden=(16,), lr=1e-2, schedule="flat"
    )


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------
@needs_pounce
def test_the_loop_records_the_trajectory():
    m = assembled_model()
    policy = fitted(m)
    report = drto.explicit_nmpc_closed_loop(policy, m, samples=4)
    assert len(report.times) == 5
    assert len(report.states["z"]) == 5
    assert len(report.moves["u"]) == 4
    assert len(report.stage_costs) == 4
    assert not report.solver_moves
    assert "closed-loop cost" in str(report)


@needs_pounce
def test_x0_is_honored_and_the_loop_is_deterministic():
    m = assembled_model()
    policy = fitted(m)
    a = drto.explicit_nmpc_closed_loop(policy, m, samples=3, x0={"z": 0.7})
    assert a.states["z"][0] == pytest.approx(0.7)
    b = drto.explicit_nmpc_closed_loop(policy, m, samples=3, x0={"z": 0.7})
    assert a.states["z"] == pytest.approx(b.states["z"])
    assert a.moves["u"] == pytest.approx(b.moves["u"])


@needs_pounce
def test_a_disturbance_realization_is_honored():
    m = assembled_model()
    policy = fitted(m)
    seq = [0.05, -0.05, 0.0]
    report = drto.explicit_nmpc_closed_loop(
        policy, m, samples=3, x0={"z": 0.5}, disturbances={"w": seq}
    )
    assert report.realizations["w"] == pytest.approx(seq)
    quiet = drto.explicit_nmpc_closed_loop(policy, m, samples=3, x0={"z": 0.5})
    assert report.states["z"][1] != pytest.approx(quiet.states["z"][1])


def test_guards():
    m = assembled_model()
    with pytest.raises(ValueError, match="not a pinned state"):
        drto.explicit_nmpc_closed_loop(None, m, samples=1, x0={"y": 0.5})
    with pytest.raises(ValueError, match="one per sample"):
        drto.explicit_nmpc_closed_loop(None, m, samples=3, disturbances={"w": [0.1]})


@needs_pounce
def test_compare_records_the_solver_controls():
    m = assembled_model()
    policy = fitted(m)
    report = drto.explicit_nmpc_closed_loop(
        policy, m, samples=3, x0={"z": 0.8}, compare=True
    )
    assert len(report.solver_moves["u"]) == 3
    assert "solver" in str(report)
    # the solver's controls are recorded, not applied: the visited states
    # follow the policy's moves
    assert report.moves["u"] != pytest.approx(report.solver_moves["u"])


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
@needs_pounce
def test_the_plots_draw_the_report():
    m = assembled_model()
    policy = fitted(m)
    report = drto.explicit_nmpc_closed_loop(
        policy, m, samples=3, x0={"z": 0.8}, compare=True
    )
    axes = drto.plot_states(report)
    assert len(axes) == 1
    axes = drto.plot_controls(report)
    assert len(axes) == 1
    # the overlay legend names the solver series
    fig = axes[0].figure
    labels = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert "solver" in labels
