# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The Klatt-Engell CSTR, declared for drto: the approximate-MPC benchmark.

The van de Vusse reaction A -> B -> C, 2A -> D in a cooled continuous
stirred-tank reactor (Klatt & Engell, 1998, doi:10.1016/S0098-1354(97)00261-5).
Four states (concentrations c_a and c_b, reactor temperature T_R, coolant
temperature T_K), two manipulated inputs (dilution rate F, cooling power
Q_dot), and the setpoint-tracking cost of Lueken, Brandner & Lucia (2023,
doi:10.1016/j.ifacol.2023.10.545) eq 21: the states tracked under
Q = diag(1, 1, 0, 0), so the concentrations carry unit weight and the
temperatures zero, and each control move priced under S against the
previous sample, the first one against the ``F_prev`` / ``Q_dot_prev``
Params. The weights are Params. The k = 0 tracking term is constant
through the initial condition, so the minimizer matches eq 21's
tracking sum starting at k = 1.

The dynamics are eq 19 of that paper with the parameters of its Table 1,
which is the CSTR example shipped with do-mpc, the toolbox the study runs.
Two values differ between the printed table and do-mpc's model file:
E_A,3/R (5560.0 K printed, 8560.0 in do-mpc and in the original benchmark)
and C_p,K (2.03 printed, 2.0 in do-mpc). This module carries do-mpc's
values. The horizon and sampling time defaults (N=20 steps of 0.005 h) are
do-mpc's, which the paper does not state.

Every state and control carries a steady-state pairing. The paper states
targets only for the concentrations; the temperature and control targets
are the equilibrium consistent with them, which a steady solve computes
and writes into the Params.

Each state balance carries an additive zero-mean noise term
(``w_ca`` .. ``w_tk``), declared as disturbances: the optimizations fix
them at zero, and a simulation fixes them at a supplied realization.

Usage from a notebook or script in ``examples/``::

    from models.klatt_engell import klatt_engell
    m = klatt_engell(N=20, h=0.005)
"""
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def klatt_engell(N=20, h=0.005, ncp=2, move_penalty=True):
    """Return the declared Klatt-Engell CSTR with an ``N``-step horizon.

    The time set is initialized with the sample grid (``N`` steps of the
    sampling time ``h``, hours), the convention ``drto.horizon`` captures,
    and discretized by Radau collocation of degree ``ncp`` with one
    finite element per step, do-mpc's settings. Physical constants and
    setpoints are mutable Params; the initial state is set through
    ``m.ca_hat`` .. ``m.tk_hat``, and the previous control action through
    ``m.F_prev`` / ``m.Q_dot_prev``.

    ``move_penalty=True``, the default, declares eq 21's move penalty
    through ``drto.move_suppression``: a separate cost constraint
    pricing each control move against the previous sample, the first
    against ``F_prev`` / ``Q_dot_prev``. S = diag(0.1, 1e-3) applies to
    the scaled differences (dF/100, dQ/2000), the way do-mpc's rterm
    prices them, which is the code the paper ran; eq 22 prints S in
    problem units.
    The steady reduction drops it and the terminal segment keeps it off
    the tail, so the infinite-horizon problem prices moves on the finite
    horizon only. The stage cost itself is pointwise: the tracking terms
    plus control deviation from the steady targets at the ``r_f`` /
    ``r_q`` weights, zero by default.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[round(i * h, 10) for i in range(N + 1)])

    m.rho = pyo.Param(initialize=0.9342, mutable=True)  # density, kg/L
    m.cp_r = pyo.Param(initialize=3.01, mutable=True)  # reactor heat capacity, kJ/kg/K
    m.cp_k = pyo.Param(initialize=2.0, mutable=True)  # coolant heat capacity, kJ/kg/K
    m.t_in = pyo.Param(initialize=130.0, mutable=True)  # feed temperature, C
    m.c_in = pyo.Param(initialize=5.1, mutable=True)  # feed concentration of A, mol/L
    m.m_k = pyo.Param(initialize=5.0, mutable=True)  # coolant mass, kg
    m.v_r = pyo.Param(initialize=10.01, mutable=True)  # reactor volume, L
    m.a_r = pyo.Param(initialize=0.215, mutable=True)  # heat-exchange area, m2
    m.k_w = pyo.Param(initialize=4032.0, mutable=True)  # heat-transfer coefficient, kJ/h/m2/K
    m.k0_1 = pyo.Param(initialize=1.287e12, mutable=True)  # pre-exponentials, 1/h
    m.k0_2 = pyo.Param(initialize=1.287e12, mutable=True)  # 1/h
    m.k0_3 = pyo.Param(initialize=9.043e9, mutable=True)  # L/mol/h
    m.e1 = pyo.Param(initialize=9758.3, mutable=True)  # activation energies over R, K
    m.e2 = pyo.Param(initialize=9758.3, mutable=True)
    m.e3 = pyo.Param(initialize=8560.0, mutable=True)
    m.dh1 = pyo.Param(initialize=4.2, mutable=True)  # heats of reaction, kJ/mol
    m.dh2 = pyo.Param(initialize=-11.0, mutable=True)
    m.dh3 = pyo.Param(initialize=-41.85, mutable=True)

    m.ca_sp = pyo.Param(initialize=0.7, mutable=True)  # setpoints, mol/L
    m.cb_sp = pyo.Param(initialize=0.6, mutable=True)
    # the temperatures' tracking weight is zero (eq 22's Q carries
    # diag(1, 1, 0, 0)), so these enter the cost inert; as steady-state
    # targets they take the equilibrium a steady solve writes into them
    m.tr_sp = pyo.Param(initialize=134.14, mutable=True)
    m.tk_sp = pyo.Param(initialize=130.0, mutable=True)
    # steady control targets, likewise written from a steady solve
    m.f_sp = pyo.Param(initialize=52.5, mutable=True)
    m.q_sp = pyo.Param(initialize=-1000.0, mutable=True)
    m.q_ca = pyo.Param(initialize=1.0, mutable=True)  # eq 22: Q = R = diag(...)
    m.q_cb = pyo.Param(initialize=1.0, mutable=True)
    m.q_tr = pyo.Param(initialize=0.0, mutable=True)
    m.q_tk = pyo.Param(initialize=0.0, mutable=True)
    m.s_f = pyo.Param(initialize=0.1, mutable=True)  # eq 22: S = diag(0.1, 1e-3)
    m.s_q = pyo.Param(initialize=1e-3, mutable=True)
    # do-mpc's input scaling: its rterm prices the scaled differences,
    # which is the code the paper ran (eq 22 prints S in problem units)
    m.f_scale = pyo.Param(initialize=100.0, mutable=True)
    m.q_scale = pyo.Param(initialize=2000.0, mutable=True)
    # control-deviation weights for the move_penalty=False form
    m.r_f = pyo.Param(initialize=0.0, mutable=True)
    m.r_q = pyo.Param(initialize=0.0, mutable=True)
    m.ca_hat = pyo.Param(initialize=0.8, mutable=True)  # the initial state,
    m.cb_hat = pyo.Param(initialize=0.5, mutable=True)  # set as feedback
    m.tr_hat = pyo.Param(initialize=134.14, mutable=True)
    m.tk_hat = pyo.Param(initialize=130.0, mutable=True)
    m.F_prev = pyo.Param(initialize=52.5, mutable=True)  # previous control action
    m.Q_dot_prev = pyo.Param(initialize=-1000.0, mutable=True)

    m.c_a = pyo.Var(m.t, bounds=(0.1, 2.0), initialize=0.8)
    m.c_b = pyo.Var(m.t, bounds=(0.1, 2.0), initialize=0.5)
    m.T_R = pyo.Var(m.t, bounds=(110.0, 150.0), initialize=134.14)
    m.T_K = pyo.Var(m.t, bounds=(110.0, 140.0), initialize=130.0)
    m.dc_a = DerivativeVar(m.c_a, wrt=m.t)
    m.dc_b = DerivativeVar(m.c_b, wrt=m.t)
    m.dT_R = DerivativeVar(m.T_R, wrt=m.t)
    m.dT_K = DerivativeVar(m.T_K, wrt=m.t)
    m.F = pyo.Var(m.t, bounds=(5.0, 100.0), initialize=52.5)  # 1/h
    m.Q_dot = pyo.Var(m.t, bounds=(-2000.0, 0.0), initialize=-1000.0)  # kJ/h
    # additive zero-mean process noise, one channel per state balance:
    # mol/L/h on the concentrations, C/h on the temperatures
    m.w_ca = pyo.Var(m.t, initialize=0.0)
    m.w_cb = pyo.Var(m.t, initialize=0.0)
    m.w_tr = pyo.Var(m.t, initialize=0.0)
    m.w_tk = pyo.Var(m.t, initialize=0.0)
    # unbounded cost vars: a cost var pinned at a bound drags the solver
    m.cost = pyo.Var(m.t)
    m.term = pyo.Var()

    def k1(m, t):
        return m.k0_1 * pyo.exp(-m.e1 / (m.T_R[t] + 273.15))

    def k2(m, t):
        return m.k0_2 * pyo.exp(-m.e2 / (m.T_R[t] + 273.15))

    def k3(m, t):
        return m.k0_3 * pyo.exp(-m.e3 / (m.T_R[t] + 273.15))

    @m.Constraint(m.t)
    def ca_ode(m, t):
        return m.dc_a[t] == (
            m.F[t] * (m.c_in - m.c_a[t])
            - k1(m, t) * m.c_a[t]
            - k3(m, t) * m.c_a[t] ** 2
            + m.w_ca[t]
        )

    @m.Constraint(m.t)
    def cb_ode(m, t):
        return m.dc_b[t] == (
            -m.F[t] * m.c_b[t]
            + k1(m, t) * m.c_a[t]
            - k2(m, t) * m.c_b[t]
            + m.w_cb[t]
        )

    @m.Constraint(m.t)
    def tr_ode(m, t):
        return m.dT_R[t] == (
            m.F[t] * (m.t_in - m.T_R[t])
            + m.k_w * m.a_r / (m.rho * m.cp_r * m.v_r) * (m.T_K[t] - m.T_R[t])
            - (
                k1(m, t) * m.c_a[t] * m.dh1
                + k2(m, t) * m.c_b[t] * m.dh2
                + k3(m, t) * m.c_a[t] ** 2 * m.dh3
            )
            / (m.rho * m.cp_r)
            + m.w_tr[t]
        )

    @m.Constraint(m.t)
    def tk_ode(m, t):
        return m.dT_K[t] == (
            m.Q_dot[t] + m.k_w * m.a_r * (m.T_R[t] - m.T_K[t])
        ) / (m.m_k * m.cp_k) + m.w_tk[t]

    # eq 21's tracking terms per sample. The k = 0 tracking term is a
    # constant through the initial condition, so the minimizer matches
    # eq 21's tracking sum starting at k = 1
    ts = sorted(m.t)

    def tracking(m, t):
        return (
            m.q_ca * (m.c_a[t] - m.ca_sp) ** 2
            + m.q_cb * (m.c_b[t] - m.cb_sp) ** 2
            + m.q_tr * (m.T_R[t] - m.tr_sp) ** 2
            + m.q_tk * (m.T_K[t] - m.tk_sp) ** 2
        )

    @m.Constraint(ts[:-1])
    def stage(m, t):
        return m.cost[t] == (
            tracking(m, t)
            + m.r_f * (m.F[t] - m.f_sp) ** 2
            + m.r_q * (m.Q_dot[t] - m.q_sp) ** 2
        )

    if move_penalty:
        m.mcost = pyo.Var(m.t)

        @m.Constraint(ts[:-1])
        def move(m, t):
            k = ts.index(t)
            f_prev = m.F_prev if k == 0 else m.F[ts[k - 1]]
            q_prev = m.Q_dot_prev if k == 0 else m.Q_dot[ts[k - 1]]
            return m.mcost[t] == (
                m.s_f * ((m.F[t] - f_prev) / m.f_scale) ** 2
                + m.s_q * ((m.Q_dot[t] - q_prev) / m.q_scale) ** 2
            )

    tN = m.t.last()

    @m.Constraint()  # eq 21's terminal term, R = Q, no move penalty
    def terminal(m):
        return m.term == tracking(m, tN)

    @m.Constraint()
    def ca_init(m):
        return m.c_a[0] == m.ca_hat

    @m.Constraint()
    def cb_init(m):
        return m.c_b[0] == m.cb_hat

    @m.Constraint()
    def tr_init(m):
        return m.T_R[0] == m.tr_hat

    @m.Constraint()
    def tk_init(m):
        return m.T_K[0] == m.tk_hat

    drto.horizon(m.t)
    drto.state(m.c_a, m.c_b, m.T_R, m.T_K)
    drto.dynamics(m.ca_ode, m.cb_ode, m.tr_ode, m.tk_ode)
    drto.control(m.F, m.Q_dot, profile="piecewise_constant")
    drto.disturbance(m.w_ca, m.w_cb, m.w_tr, m.w_tk)
    drto.tracking_stage_cost(m.stage)
    if move_penalty:
        drto.move_suppression(m.move)
    drto.tracking_terminal_cost(m.terminal)
    drto.initial_condition(m.ca_init, m.cb_init, m.tr_init, m.tk_init)
    drto.steady_state(m.c_a, m.ca_sp)
    drto.steady_state(m.c_b, m.cb_sp)
    drto.steady_state(m.T_R, m.tr_sp)
    drto.steady_state(m.T_K, m.tk_sp)
    drto.steady_state_control(m.F, m.f_sp)
    drto.steady_state_control(m.Q_dot, m.q_sp)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=N, ncp=ncp, scheme="LAGRANGE-RADAU"
    )
    return m
