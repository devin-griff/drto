# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 026: drto.explicit_nmpc_data."""
import sys

import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

pounce_ok = pyo.SolverFactory("pounce").available(exception_flag=False)
needs_pounce = pytest.mark.skipif(not pounce_ok, reason="pounce not available")


def assembled_model():
    """dz/dt = -z + u with a tracking cost, assembled for optimization."""
    m = pyo.ConcreteModel()
    N, h = 4, 2.5
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z = pyo.Var(m.t, bounds=(0, 1), initialize=0.4)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.3)
    m.z_ss = pyo.Param(initialize=0.3, mutable=True)
    m.u_ss = pyo.Param(initialize=0.3, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)
    m.cost = pyo.Var(m.t)
    ts = sorted(m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.z[t] + m.u[t]

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
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.init)
    drto.steady_state(m.z, m.z_ss)
    drto.steady_state_control(m.u, m.u_ss)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    return m


def xs(dataset, name="z_hat"):
    return [p["x"][name] for p in dataset.points]


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_gradients_need_pounce():
    m = assembled_model()
    with pytest.raises(ValueError, match="pounce factorization"):
        drto.explicit_nmpc_data(m, n=1, solver="ipopt")


def test_an_unknown_method_errors():
    m = assembled_model()
    with pytest.raises(ValueError, match="sobol, lhs, uniform"):
        drto.explicit_nmpc_data(m, n=1, method="grid", gradients=False)


def test_missing_scipy_names_the_install(monkeypatch):
    m = assembled_model()
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    with pytest.raises(RuntimeError, match="pip install scipy"):
        drto.explicit_nmpc_data(m, n=1, method="sobol", gradients=False)


def test_an_extra_input_needs_a_box():
    m = assembled_model()
    with pytest.raises(ValueError, match="no box for 'z_ss'"):
        drto.explicit_nmpc_data(m, n=1, inputs=[m.z_ss], gradients=False)


# ----------------------------------------------------------------------
# the designs
# ----------------------------------------------------------------------
@needs_pounce
def test_a_sobol_pool_prefixes_a_larger_one():
    big = drto.explicit_nmpc_data(assembled_model(), n=8, method="sobol", seed=3)
    small = drto.explicit_nmpc_data(assembled_model(), n=4, method="sobol", seed=3)
    assert xs(big)[:4] == pytest.approx(xs(small))


@needs_pounce
def test_an_lhs_stratifies_every_coordinate():
    n = 8
    d = drto.explicit_nmpc_data(assembled_model(), n=n, method="lhs", seed=1)
    lo, hi = d.config["ranges"]["z_hat"]
    bins = {int((v - lo) / (hi - lo) * n) for v in xs(d)}
    assert len(bins) == n


@needs_pounce
def test_uniform_is_seeded_and_reproducible():
    a = drto.explicit_nmpc_data(assembled_model(), n=4, method="uniform", seed=7)
    b = drto.explicit_nmpc_data(assembled_model(), n=4, method="uniform", seed=7)
    assert xs(a) == pytest.approx(xs(b))


@needs_pounce
def test_ranges_override_the_default_box():
    d = drto.explicit_nmpc_data(
        assembled_model(), n=4, ranges={"z_hat": (0.2, 0.4)}, seed=0
    )
    assert d.config["ranges"]["z_hat"] == [0.2, 0.4]
    assert all(0.2 <= v <= 0.4 for v in xs(d))


@needs_pounce
def test_inputs_extend_the_sampled_set():
    m = assembled_model()
    d = drto.explicit_nmpc_data(
        m, n=4, inputs=[m.z_ss], ranges={"z_ss": (0.1, 0.5)}, seed=0
    )
    assert d.config["inputs"] == ["z_hat", "z_ss"]
    assert all(0.1 <= p["x"]["z_ss"] <= 0.5 for p in d.points)


# ----------------------------------------------------------------------
# the labels
# ----------------------------------------------------------------------
@needs_pounce
def test_each_point_carries_the_labels():
    d = drto.explicit_nmpc_data(assembled_model(), n=4, seed=0)
    assert len(d) == 4 and not d.failures
    for p in d.points:
        assert set(p) == {"x", "u0", "V", "du0_dx"}
        assert set(p["u0"]) == {"u"}
        assert set(p["du0_dx"]["u"]) == {"z_hat"}
        assert p["V"] >= 0


@needs_pounce
def test_gradients_false_omits_them():
    d = drto.explicit_nmpc_data(assembled_model(), n=2, gradients=False, seed=0)
    assert all("du0_dx" not in p for p in d.points)


@needs_pounce
def test_a_failed_solve_is_recorded_and_excluded():
    # z is bounded at 1, so an initial condition above it is infeasible
    d = drto.explicit_nmpc_data(
        assembled_model(), n=2, ranges={"z_hat": (5.0, 6.0)}, seed=0
    )
    assert len(d) == 0 and len(d.failures) == 2
    for f in d.failures:
        assert set(f) == {"x", "termination"}


@needs_pounce
def test_the_json_round_trips(tmp_path):
    out = tmp_path / "data.json"
    d = drto.explicit_nmpc_data(assembled_model(), n=4, seed=0, path=str(out))
    back = drto.ExplicitNmpcDataset.load(str(out))
    assert back.config == d.config
    assert back.points == d.points
    assert back.failures == d.failures


# ----------------------------------------------------------------------
# the stored information matrices
# ----------------------------------------------------------------------
def test_information_needs_pounce():
    m = assembled_model()
    with pytest.raises(ValueError, match="information=False"):
        drto.explicit_nmpc_data(
            m, n=1, gradients=False, information=True, solver="ipopt"
        )


@needs_pounce
def test_information_is_stored_and_round_trips(tmp_path):
    out = tmp_path / "d.json"
    d = drto.explicit_nmpc_data(
        assembled_model(), n=2, information=True, seed=0, path=str(out)
    )
    for p in d.points:
        assert set(p["information"]) == {"u"} and set(p["information"]["u"]) == {"u"}
        assert p["information"]["u"]["u"] > 0
    back = drto.ExplicitNmpcDataset.load(str(out))
    assert back.points[0]["information"] == d.points[0]["information"]


def test_information_warnings_are_suppressed_and_counted(monkeypatch):
    import warnings as w

    import pyomo_pounce

    real = pyomo_pounce.information

    def noisy(m, wrt=None, **kw):
        w.warn("information: member has curvature below the noise scale")
        w.warn("information: member is held by its bound at the optimum")
        w.warn("information: the direction is NOT projected")
        w.warn("information: something else entirely")
        return real(m, wrt=wrt, **kw)

    monkeypatch.setattr(pyomo_pounce, "information", noisy)
    with w.catch_warnings(record=True) as leaked:
        w.simplefilter("always")
        d = drto.explicit_nmpc_data(assembled_model(), n=2, information=True, seed=0)
    assert not [x for x in leaked if "information:" in str(x.message)]
    counts = d.config["information_warnings"]
    assert counts["unidentified curvature"] == 2
    assert counts["pinned member"] == 2
    assert counts["unprojected constraint"] == 2
    assert counts["other"] == 2
    assert "information warnings" in repr(d)
