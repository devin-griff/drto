# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 018: the estimation declarations."""
import pyomo.environ as pyo
import pytest
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def est_base():
    """A window model with the estimation pieces built but not declared."""
    m = pyo.ConcreteModel()
    N, h = 4, 2.5  # window samples and sampling time
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.z = pyo.Var(m.t)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.k = pyo.Var(initialize=1.0)  # unknown parameter, constant over the window
    m.w = pyo.Var(m.t)  # process noise
    m.y_meas = pyo.Param(m.t, mutable=True, initialize=0.0)  # measurements

    m.z_prior = pyo.Param(initialize=0.4, mutable=True)
    m.inv_r = pyo.Param(initialize=100, mutable=True)
    m.inv_q = pyo.Param(initialize=10, mutable=True)
    m.inv_p0 = pyo.Param(initialize=1, mutable=True)

    m.est_stage = pyo.Var(m.t)
    m.est_term = pyo.Var()
    m.arrival = pyo.Var()

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dzdt[t] == -m.k * m.z[t] + m.w[t]

    @m.Constraint(sorted(m.t)[:-1])
    def est_stage_con(m, t):
        return (
            m.est_stage[t]
            == m.inv_r * (m.y_meas[t] - m.z[t]) ** 2 + m.inv_q * m.w[t] ** 2
        )

    @m.Constraint()
    def est_term_con(m):
        tN = m.t.last()
        return m.est_term == m.inv_r * (m.y_meas[tN] - m.z[tN]) ** 2

    @m.Constraint()
    def arrival_con(m):
        return m.arrival == m.inv_p0 * (m.z[0] - m.z_prior) ** 2

    return m


def est_declared():
    """The window model with the horizon, state, and all six declarations."""
    m = est_base()
    drto.horizon(m.t)
    drto.state(m.z)
    drto.dynamics(m.ode)
    drto.estimated_parameter(m.k)
    drto.disturbance(m.w)
    drto.measurement(m.y_meas)
    drto.estimation_stage_cost(m.est_stage_con)
    drto.estimation_terminal_cost(m.est_term_con)
    drto.arrival_cost(m.arrival_con)
    return m


# ----------------------------------------------------------------------
# the happy path
# ----------------------------------------------------------------------
def test_full_estimation_surface_records():
    m = est_declared()
    reg = drto.info(m)
    assert reg.components("estimated_parameter") == (m.k,)
    assert reg.components("disturbance") == (m.w,)
    assert reg.components("measurement") == (m.y_meas,)
    assert reg.components("estimation_stage_cost") == (m.est_stage_con,)
    assert reg.components("estimation_terminal_cost") == (m.est_term_con,)
    assert reg.components("arrival_cost") == (m.arrival_con,)


def test_estimation_declarations_render_in_the_registry_view():
    text = repr(drto.info(est_declared()))
    assert "estimated parameters" in text
    assert "disturbances" in text
    assert "measurements" in text
    assert "arrival cost" in text


# ----------------------------------------------------------------------
# estimated_parameter
# ----------------------------------------------------------------------
def test_estimated_parameter_must_be_a_var():
    m = est_base()
    with pytest.raises(TypeError, match="expects a Var"):
        drto.estimated_parameter(m.y_meas)  # a Param


def test_estimated_parameter_rejects_time_indexing():
    m = est_base()
    drto.horizon(m.t)
    m.theta_t = pyo.Var(m.t)
    with pytest.raises(ValueError, match="constant over the window"):
        drto.estimated_parameter(m.theta_t)


def test_estimated_parameter_needs_no_horizon():
    # shared with steady-state data reconciliation: valid with no horizon
    m = pyo.ConcreteModel()
    m.theta = pyo.Var()
    drto.estimated_parameter(m.theta)
    assert drto.info(m).components("estimated_parameter") == (m.theta,)


def test_estimated_parameter_accumulates_and_rejects_duplicates():
    m = est_base()
    m.k2 = pyo.Var()
    drto.estimated_parameter(m.k)
    drto.estimated_parameter(m.k2)
    assert drto.info(m).components("estimated_parameter") == (m.k, m.k2)
    with pytest.raises(ValueError, match="already declared"):
        drto.estimated_parameter(m.k)


# ----------------------------------------------------------------------
# disturbance
# ----------------------------------------------------------------------
def test_disturbance_must_be_a_var():
    m = est_base()
    with pytest.raises(TypeError, match="expects a Var"):
        drto.disturbance(m.y_meas)  # a Param


def test_disturbance_must_be_time_indexed_with_a_horizon():
    m = est_base()
    drto.horizon(m.t)
    m.w0 = pyo.Var()  # a scalar, not over the window
    with pytest.raises(ValueError, match="not indexed by the declared time set"):
        drto.disturbance(m.w0)


def test_disturbance_takes_no_profile():
    m = est_base()
    drto.horizon(m.t)
    with pytest.raises(TypeError):
        drto.disturbance(m.w, profile="piecewise_constant")


# ----------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------
def test_measurement_must_be_a_param():
    m = est_base()
    drto.horizon(m.t)
    with pytest.raises(TypeError, match="expects a Param"):
        drto.measurement(m.z)  # a Var


def test_measurement_must_be_mutable():
    m = est_base()
    drto.horizon(m.t)
    m.frozen = pyo.Param(m.t, initialize=0.0)  # not mutable
    with pytest.raises(ValueError, match="mutable"):
        drto.measurement(m.frozen)


def test_measurement_must_be_time_indexed_with_a_horizon():
    m = est_base()
    drto.horizon(m.t)
    m.y0 = pyo.Param(mutable=True, initialize=0.0)  # a scalar
    with pytest.raises(ValueError, match="not indexed by the declared time set"):
        drto.measurement(m.y0)


# ----------------------------------------------------------------------
# the cost constraints
# ----------------------------------------------------------------------
def test_estimation_stage_cost_over_the_samples():
    m = est_base()
    drto.horizon(m.t)
    drto.estimation_stage_cost(m.est_stage_con)
    assert drto.info(m).components("estimation_stage_cost") == (m.est_stage_con,)


def test_estimation_stage_cost_rejects_a_time_set_index():
    m = est_base()
    drto.horizon(m.t)

    @m.Constraint(m.t)
    def full_span(m, t):
        return m.est_stage[t] == m.z[t] ** 2

    with pytest.raises(ValueError, match="indexed by the time set"):
        drto.estimation_stage_cost(m.full_span)


def test_estimation_terminal_cost_must_be_scalar():
    m = est_base()
    drto.horizon(m.t)
    with pytest.raises(ValueError, match="scalar Constraint"):
        drto.estimation_terminal_cost(m.est_stage_con)  # an indexed family


def test_estimation_terminal_and_arrival_record():
    m = est_base()
    drto.estimation_terminal_cost(m.est_term_con)
    drto.arrival_cost(m.arrival_con)
    reg = drto.info(m)
    assert reg.components("estimation_terminal_cost") == (m.est_term_con,)
    assert reg.components("arrival_cost") == (m.arrival_con,)


def test_arrival_cost_needs_a_cost_variable_side():
    m = est_base()

    @m.Constraint()
    def no_var(m):
        return m.z[0] ** 2 == m.z_prior

    with pytest.raises(ValueError, match="cost variable"):
        drto.arrival_cost(m.no_var)


def test_arrival_cost_must_be_an_equality():
    m = est_base()

    @m.Constraint()
    def bound(m):
        return m.arrival >= 0

    with pytest.raises(ValueError, match="equality"):
        drto.arrival_cost(m.bound)


# ----------------------------------------------------------------------
# wrapping and the decorators
# ----------------------------------------------------------------------
def test_estimated_parameter_wraps():
    m = pyo.ConcreteModel()
    fresh = pyo.Var()
    wrapped = drto.estimated_parameter(fresh)
    assert wrapped is fresh
    assert not drto.info(m).has_declaration("estimated_parameter")
    m.k = wrapped
    assert drto.info(m).components("estimated_parameter") == (m.k,)


def test_measurement_wraps():
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, 4, 1))
    drto.horizon(m.t)
    m.y = drto.measurement(pyo.Param(m.t, mutable=True, initialize=0.0))
    assert drto.info(m).components("measurement") == (m.y,)


def test_arrival_cost_decorator():
    m = est_base()

    @drto.arrival_cost(m)
    def arr(m):
        return m.arrival == m.inv_p0 * (m.z[0] - m.z_prior) ** 2

    assert arr is m.arr
    assert drto.info(m).components("arrival_cost") == (m.arr,)


def test_estimation_stage_cost_decorator_validation_applies():
    m = est_base()
    drto.horizon(m.t)
    with pytest.raises(ValueError, match="indexed by the time set"):

        @drto.estimation_stage_cost(m, m.t)  # the time set itself
        def bad(m, t):
            return m.est_stage[t] == m.z[t] ** 2
