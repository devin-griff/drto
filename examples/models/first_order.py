# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""A first-order linear system, declared for drto: the minimal example.

The feature 002 example model: one state z with time constant tau_p, one
control u, a tracking stage cost toward mutable-Param targets, and a
mutable-Param feedback hook (z_hat) anchoring the initial state. Small enough
that a notebook's structure, not the model, is the story.

Usage from a notebook in ``examples/``::

    from models.first_order import first_order
    m = first_order(n_samples=10)
"""
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def first_order(n_samples=10, dt=1.0):
    """Return the declared first-order model with an ``n_samples``-step horizon.

    The time set is initialized with the sample grid (``n_samples`` steps of
    ``dt``). The gain and time constant, the setpoints, and the initial state
    are mutable Params.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[i * dt for i in range(n_samples + 1)])

    m.tau_p = pyo.Param(initialize=2.0, mutable=True)  # time constant
    # a consistent equilibrium pair: dz/dt = 0 requires z = u for this model
    m.z_ss = pyo.Param(initialize=0.5, mutable=True)  # tracking targets
    m.u_ss = pyo.Param(initialize=0.5, mutable=True)
    m.z_hat = pyo.Param(initialize=0.4, mutable=True)  # state feedback hook

    m.z = pyo.Var(m.t, initialize=0.4)
    m.dzdt = DerivativeVar(m.z, wrt=m.t)
    m.u = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.cost = pyo.Var(m.t)  # unbounded: a cost var pinned at a bound drags ipopt

    @m.Constraint(m.t)
    def ode(m, t):
        # one side must be the bare DerivativeVar (the dynamics convention)
        return m.dzdt[t] == (-m.z[t] + m.u[t]) / m.tau_p

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == 10 * (m.z[t] - m.z_ss) ** 2 + (m.u[t] - m.u_ss) ** 2

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
    return m
