# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The cart-pole, declared for drto: the unstable-equilibrium example.

An inverted pendulum hinged on a cart, driven only by a horizontal force
on the cart: four states (cart position and velocity, pole angle and
rate), one control, underactuated, and the upright target is an unstable
equilibrium, the regime where terminal machinery earns its keep. The
parameters and dynamics follow the Dinh et al. (2025) pendulum example
(pole 0.2 kg and 0.3 m, cart 0.5 kg, uniform-rod inertia factor 1/3,
cart friction), written in explicit ODE form: the mass factors divide
the right-hand sides, and both denominators are structurally nonzero.

The angle enters the cost per degree (the ``theta_scale`` Param), their
scaling, so a 40-degree tilt prices comparably to the force bound. The
default initial state is that 40-degree tilt at rest: stabilization, not
swing-up. The setpoint is upright at the track origin with zero force, a
true equilibrium of the dynamics.

Usage from a notebook in ``examples/``::

    from models.cart_pole import cart_pole
    m = cart_pole(N=10, h=1)
"""
import math

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def cart_pole(N=10, h=1):
    """Return the declared cart-pole with an ``N``-step horizon.

    The time set is initialized with the sample grid (``N`` steps of the
    sampling time ``h``, seconds; the default horizon is 10). Physical
    constants are mutable Params; the initial state is set through
    ``m.x_hat`` .. ``m.theta_dot_hat``.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))

    m.m_p = pyo.Param(initialize=0.2, mutable=True)  # pole mass (kg)
    m.m_c = pyo.Param(initialize=0.5, mutable=True)  # cart mass (kg)
    m.l = pyo.Param(initialize=0.3, mutable=True)  # pole half-length (m)
    m.kJ = pyo.Param(initialize=1 / 3, mutable=True)  # uniform-rod inertia factor
    m.g = pyo.Param(initialize=9.8, mutable=True)  # gravity (m/s^2)
    m.b = pyo.Param(initialize=0.1, mutable=True)  # cart friction
    m.theta_scale = pyo.Param(initialize=math.radians(1), mutable=True)  # cost per degree

    # the setpoint: upright at the origin, zero force (a true equilibrium)
    m.x_ss = pyo.Param(initialize=0.0, mutable=True)
    m.x_dot_ss = pyo.Param(initialize=0.0, mutable=True)
    m.theta_ss = pyo.Param(initialize=0.0, mutable=True)
    m.theta_dot_ss = pyo.Param(initialize=0.0, mutable=True)
    m.F_ss = pyo.Param(initialize=0.0, mutable=True)

    # the initial state, set as feedback: a 40-degree tilt at rest
    m.x_hat = pyo.Param(initialize=0.0, mutable=True)
    m.x_dot_hat = pyo.Param(initialize=0.0, mutable=True)
    m.theta_hat = pyo.Param(initialize=math.radians(40), mutable=True)
    m.theta_dot_hat = pyo.Param(initialize=0.0, mutable=True)

    m.x = pyo.Var(m.t, initialize=0.0)
    m.x_dot = pyo.Var(m.t, bounds=(-10, 10), initialize=0.0)
    m.theta = pyo.Var(m.t, bounds=(-math.pi, math.pi), initialize=math.radians(40))
    m.theta_dot = pyo.Var(m.t, bounds=(-math.pi / 2, math.pi / 2), initialize=0.0)
    m.dx = DerivativeVar(m.x, wrt=m.t)
    m.dx_dot = DerivativeVar(m.x_dot, wrt=m.t)
    m.dtheta = DerivativeVar(m.theta, wrt=m.t)
    m.dtheta_dot = DerivativeVar(m.theta_dot, wrt=m.t)

    m.F = pyo.Var(m.t, bounds=(-5, 5), initialize=0.0)

    # unbounded cost vars: a cost var pinned at a bound drags ipopt
    m.cost = pyo.Var(m.t)
    m.term = pyo.Var()

    @m.Constraint(m.t)
    def x_ode(m, t):
        return m.dx[t] == m.x_dot[t]

    @m.Constraint(m.t)
    def x_dot_ode(m, t):
        return m.dx_dot[t] == (m.m_p * m.g * pyo.sin(m.theta[t]) * pyo.cos(m.theta[t]) - (1 + m.kJ) * (m.F[t] + m.m_p * m.l * m.theta_dot[t] ** 2 * pyo.sin(m.theta[t]) - m.b * m.x_dot[t])) / (m.m_p * pyo.cos(m.theta[t]) ** 2 - (1 + m.kJ) * m.m_c)

    @m.Constraint(m.t)
    def theta_ode(m, t):
        return m.dtheta[t] == m.theta_dot[t]

    @m.Constraint(m.t)
    def theta_dot_ode(m, t):
        return m.dtheta_dot[t] == (m.m_c * m.g * pyo.sin(m.theta[t]) - pyo.cos(m.theta[t]) * (m.F[t] + m.m_p * m.l * m.theta_dot[t] ** 2 * pyo.sin(m.theta[t]))) / ((1 + m.kJ) * m.m_c * m.l - m.m_p * m.l * pyo.cos(m.theta[t]) ** 2)

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == (m.theta[t] - m.theta_ss) ** 2 / m.theta_scale ** 2 + 0.01 * (m.x[t] - m.x_ss) ** 2 + 0.01 * (m.x_dot[t] - m.x_dot_ss) ** 2 + 0.01 * (m.theta_dot[t] - m.theta_dot_ss) ** 2 / m.theta_scale ** 2 + 0.1 * (m.F[t] - m.F_ss) ** 2

    tN = m.t.last()

    @m.Constraint()  # the stage cost with the control removed, at tN
    def terminal(m):
        return m.term == (m.theta[tN] - m.theta_ss) ** 2 / m.theta_scale ** 2 + 0.01 * (m.x[tN] - m.x_ss) ** 2 + 0.01 * (m.x_dot[tN] - m.x_dot_ss) ** 2 + 0.01 * (m.theta_dot[tN] - m.theta_dot_ss) ** 2 / m.theta_scale ** 2

    @m.Constraint()
    def x_ic(m):
        return m.x[0] == m.x_hat

    @m.Constraint()
    def x_dot_ic(m):
        return m.x_dot[0] == m.x_dot_hat

    @m.Constraint()
    def theta_ic(m):
        return m.theta[0] == m.theta_hat

    @m.Constraint()
    def theta_dot_ic(m):
        return m.theta_dot[0] == m.theta_dot_hat

    drto.horizon(m.t)
    drto.state(m.x, m.x_dot, m.theta, m.theta_dot)
    drto.dynamics(m.x_ode, m.x_dot_ode, m.theta_ode, m.theta_dot_ode)
    drto.control(m.F, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    drto.initial_condition(m.x_ic, m.x_dot_ic, m.theta_ic, m.theta_dot_ic)
    drto.steady_state(m.x, m.x_ss)
    drto.steady_state(m.x_dot, m.x_dot_ss)
    drto.steady_state(m.theta, m.theta_ss)
    drto.steady_state(m.theta_dot, m.theta_dot_ss)
    drto.steady_state_control(m.F, m.F_ss)
    return m
