# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The Hicks-Ray CSTR, declared for drto: the canonical nonlinear example.

A first-order exothermic reaction A -> B in a cooled continuous stirred-tank
reactor (Hicks & Ray, 1971), in the dimensionless form used by
Dinh et al. (2025), doi:10.1016/j.jprocont.2025.103565. Two states
(concentration zc, temperature zt), two manipulated inputs (coolant flow v1,
residence time v2), a tracking stage cost toward the model's steady state,
and mutable-Param feedback hooks (zc_hat, zt_hat) anchoring the initial
state.

Usage from a notebook in ``examples/``::

    from models.hicks import hicks
    m = hicks(n_samples=5)
"""
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def hicks(n_samples=5, dt=1.0):
    """Return the declared Hicks-Ray CSTR with an ``n_samples``-step horizon.

    The time set is initialized with the sample grid (``n_samples`` steps of
    ``dt``), the convention ``drto.horizon`` captures. All physical constants
    and setpoints are mutable Params, so they retune by ``set_value``; the
    initial state is set through ``m.zc_hat`` / ``m.zt_hat``.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[i * dt for i in range(n_samples + 1)])

    m.u1sf = pyo.Param(initialize=600, mutable=True)  # coolant-flow scale factor
    m.u2sf = pyo.Param(initialize=40, mutable=True)  # residence-time scale factor
    m.k0 = pyo.Param(initialize=300, mutable=True)  # Arrhenius pre-exponential
    m.ea = pyo.Param(initialize=5, mutable=True)  # dimensionless activation energy
    m.a0 = pyo.Param(initialize=1.95e-4, mutable=True)  # heat-transfer coefficient
    m.ztcw = pyo.Param(initialize=0.38, mutable=True)  # coolant temperature
    m.ztf = pyo.Param(initialize=0.395, mutable=True)  # feed temperature

    m.zc_ss = pyo.Param(initialize=0.6416, mutable=True)  # steady-state targets
    m.zt_ss = pyo.Param(initialize=0.5387, mutable=True)
    m.v1_ss = pyo.Param(initialize=0.57828, mutable=True)
    m.v2_ss = pyo.Param(initialize=0.49989, mutable=True)
    m.zc_hat = pyo.Param(initialize=0.625, mutable=True)  # state feedback hooks
    m.zt_hat = pyo.Param(initialize=0.525, mutable=True)

    m.zc = pyo.Var(m.t, bounds=(0, 1), initialize=0.6416)
    m.zt = pyo.Var(m.t, bounds=(0, None), initialize=0.5387)
    m.dzc = DerivativeVar(m.zc, wrt=m.t)
    m.dzt = DerivativeVar(m.zt, wrt=m.t)
    m.v1 = pyo.Var(m.t, bounds=(0.166666666666667, 1), initialize=0.57828)
    m.v2 = pyo.Var(m.t, bounds=(0.025, 1), initialize=0.49989)
    m.cost = pyo.Var(m.t)  # unbounded: a cost var pinned at a bound drags ipopt

    @m.Constraint(m.t)
    def zc_ode(m, t):
        return m.dzc[t] == (1 - m.zc[t]) / (m.u2sf * m.v2[t]) - m.k0 * m.zc[
            t
        ] * pyo.exp(-m.ea / m.zt[t])

    @m.Constraint(m.t)
    def zt_ode(m, t):
        return m.dzt[t] == (
            (m.ztf - m.zt[t]) / (m.u2sf * m.v2[t])
            + m.k0 * m.zc[t] * pyo.exp(-m.ea / m.zt[t])
            - m.a0 * m.u1sf * m.v1[t] * (m.zt[t] - m.ztcw)
        )

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == (
            10 * (m.zc[t] - m.zc_ss) ** 2
            + 2 * (m.zt[t] - m.zt_ss) ** 2
            + (m.v1[t] - m.v1_ss) ** 2
            + 0.5 * (m.v2[t] - m.v2_ss) ** 2
        )

    @m.Constraint()
    def zc_init(m):
        return m.zc[0] == m.zc_hat

    @m.Constraint()
    def zt_init(m):
        return m.zt[0] == m.zt_hat

    drto.horizon(m.t)
    drto.state(m.zc, m.zt)
    drto.dynamics(m.zc_ode, m.zt_ode)
    drto.control(m.v1, m.v2, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.initial_condition(m.zc_init, m.zt_init)
    drto.steady_state(m.zc, m.zc_ss)
    drto.steady_state(m.zt, m.zt_ss)
    drto.steady_state_control(m.v1, m.v1_ss)
    drto.steady_state_control(m.v2, m.v2_ss)
    return m
