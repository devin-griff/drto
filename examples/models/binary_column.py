# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The binary distillation column, declared for drto: the mid-size DAE.

A 42-tray methanol/n-propanol column (Diehl 2001, via Lopez-Negrete,
Thierry, Lin, and Dinh), translated faithfully from the Dinh et al. (2025)
code. States are tray holdup M and liquid composition x. The vapor flow V
comes from the tray energy balance, which carries the liquid enthalpy's
time derivative expanded by the chain rule: the balance references dx/dt
directly, and the temperature rate Tdot is a plain algebraic variable
defined by the differentiated bubble-point identity. That index-reduced
structure is kept exactly as written.

Algebraic Vars carry the physics: T (bubble point), Tdot, y (Antoine VLE
with tray efficiency), V (energy balance), L (Francis weir with their
smoothed max), Mv (volume holdup, the alias carrying the weir-minimum
bounds), and Qc (condenser duty). The property correlations underneath
(vapor pressures, molar volume, enthalpies) are inline expressions inside
those equations. Controls: reflux ratio Rec and reboiler duty Qr.

The reference steady state (Qr = 1.65, Rec = 1) and the initial state
(Qr = 1.5, Rec = 10) load from ``data/binary_column.json``, both solved
from the original model. Tracking stage and terminal costs toward the
reference (weight 10 on states, 1 on controls; holdup deviations are
relative, since M spans thousands to two hundred thousand mol).

Usage from a notebook in ``examples/``::

    from models.binary_column import binary_column
    m = binary_column(N=20, h=60)
"""
import json
from pathlib import Path

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

_DATA = json.load(open(Path(__file__).parent / "data" / "binary_column.json"))


def binary_column(N=20, h=60):
    """Return the declared binary column with an ``N``-step horizon.

    The time set is initialized with the sample grid (``N`` steps of the
    sampling time ``h``, seconds; the default horizon is 20 minutes).
    Physical constants, the reference profiles, and the initial state are
    mutable Params; the initial state is set through ``m.M_hat`` and
    ``m.x_hat``.
    """
    d = _DATA
    NT, NF = d["NT"], d["feedTray"]
    ref, ini = d["ref"], d["init"]

    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.tray = pyo.Set(initialize=range(1, NT + 1))
    m.tray_v = pyo.Set(initialize=range(1, NT))  # vapor leaves trays 1..NT-1

    # feed and per-tray operating profiles
    m.feed = pyo.Param(m.tray, initialize=lambda m, k: d["feed"] if k == NF else 0.0, mutable=True)
    m.xf = pyo.Param(initialize=d["xf"], mutable=True)  # feed mole fraction
    m.hf = pyo.Param(initialize=d["hf"], mutable=True)  # feed enthalpy
    m.p = pyo.Param(m.tray, initialize=lambda m, k: d["p"][k - 1], mutable=True)  # tray pressure
    m.alpha = pyo.Param(m.tray, initialize=lambda m, k: d["alpha"][k - 1], mutable=True)  # tray efficiency

    # liquid-enthalpy polynomial coefficients (methanol m, n-propanol n)
    m.hlm0 = pyo.Param(initialize=2.6786e-04, mutable=True)
    m.hlma = pyo.Param(initialize=-0.14779, mutable=True)
    m.hlmb = pyo.Param(initialize=97.4289, mutable=True)
    m.hlmc = pyo.Param(initialize=-2.1045e04, mutable=True)
    m.hln0 = pyo.Param(initialize=4.0449e-04, mutable=True)
    m.hlna = pyo.Param(initialize=-0.1435, mutable=True)
    m.hlnb = pyo.Param(initialize=121.7981, mutable=True)
    m.hlnc = pyo.Param(initialize=-3.0718e04, mutable=True)

    # vaporization-enthalpy correlation constants
    m.r = pyo.Param(initialize=8.3147, mutable=True)
    m.a = pyo.Param(initialize=6.09648, mutable=True)
    m.b = pyo.Param(initialize=1.28862, mutable=True)
    m.c1 = pyo.Param(initialize=1.016, mutable=True)
    m.d = pyo.Param(initialize=15.6875, mutable=True)
    m.l = pyo.Param(initialize=13.4721, mutable=True)
    m.f = pyo.Param(initialize=2.615, mutable=True)
    m.gm = pyo.Param(initialize=0.557, mutable=True)
    m.Tkm = pyo.Param(initialize=512.6, mutable=True)
    m.Pkm = pyo.Param(initialize=8.096e06, mutable=True)
    m.gn = pyo.Param(initialize=0.612, mutable=True)
    m.Tkn = pyo.Param(initialize=536.7, mutable=True)
    m.Pkn = pyo.Param(initialize=5.166e06, mutable=True)

    # Antoine coefficients
    m.CapAm = pyo.Param(initialize=23.48, mutable=True)
    m.CapBm = pyo.Param(initialize=3626.6, mutable=True)
    m.CapCm = pyo.Param(initialize=-34.29, mutable=True)
    m.CapAn = pyo.Param(initialize=22.437, mutable=True)
    m.CapBn = pyo.Param(initialize=3166.64, mutable=True)
    m.CapCn = pyo.Param(initialize=-80.15, mutable=True)

    # molar-volume correlation constants
    m.vmc1 = pyo.Param(initialize=1 / 2288, mutable=True)
    m.vmc2 = pyo.Param(initialize=0.2685, mutable=True)
    m.vmc3 = pyo.Param(initialize=512.4, mutable=True)
    m.vmc4 = pyo.Param(initialize=0.2453, mutable=True)
    m.vnc1 = pyo.Param(initialize=1 / 1235, mutable=True)
    m.vnc2 = pyo.Param(initialize=0.27136, mutable=True)
    m.vnc3 = pyo.Param(initialize=536.4, mutable=True)
    m.vnc4 = pyo.Param(initialize=0.24, mutable=True)

    # weir constants: flow coefficient and the tray-class volume minimums
    m.Kw = pyo.Param(initialize=0.166, mutable=True)
    m.Mv0r = pyo.Param(initialize=8.5, mutable=True)  # reboiler
    m.Mv0c = pyo.Param(initialize=0.17, mutable=True)  # condenser
    m.Mv0t = pyo.Param(initialize=0.155, mutable=True)  # trays
    m.q_scale = pyo.Param(initialize=1.0e6, mutable=True)  # duty scale (Qr, Qc in MW-ish units)

    # the reference steady state (targets) and the initial state (hooks)
    m.M_ss = pyo.Param(m.tray, initialize=lambda m, k: ref["M"][k - 1], mutable=True)
    m.x_ss = pyo.Param(m.tray, initialize=lambda m, k: ref["x"][k - 1], mutable=True)
    m.Rec_ss = pyo.Param(initialize=ref["Rec"], mutable=True)
    m.Qr_ss = pyo.Param(initialize=ref["Qr"], mutable=True)
    m.M_hat = pyo.Param(m.tray, initialize=lambda m, k: ini["M"][k - 1], mutable=True)
    m.x_hat = pyo.Param(m.tray, initialize=lambda m, k: ini["x"][k - 1], mutable=True)

    # states, initialized at the reference
    m.M = pyo.Var(m.tray, m.t, initialize=lambda m, k, t: ref["M"][k - 1])
    m.x = pyo.Var(m.tray, m.t, bounds=(0, 1), initialize=lambda m, k, t: ref["x"][k - 1])
    m.dM = DerivativeVar(m.M, wrt=m.t)
    m.dx = DerivativeVar(m.x, wrt=m.t)

    # controls: reflux ratio and reboiler duty
    m.Rec = pyo.Var(m.t, bounds=(0.1, 99.999), initialize=ref["Rec"])
    m.Qr = pyo.Var(m.t, bounds=(0, None), initialize=ref["Qr"])

    # algebraic variables
    m.T = pyo.Var(m.tray, m.t, bounds=(200, 450), initialize=lambda m, k, t: ref["T"][k - 1])
    m.Tdot = pyo.Var(m.tray, m.t, initialize=0.0)  # dT/dt, the chain-rule companion
    m.y = pyo.Var(m.tray_v, m.t, bounds=(0, 1), initialize=lambda m, k, t: ref["y"][k - 1])
    m.V = pyo.Var(m.tray_v, m.t, bounds=(0, 1.0e3), initialize=lambda m, k, t: ref["V"][k - 1])
    # reference values for the volume holdup and weir flow, so every Var
    # starts inside its bounds (the transform copies values onto the tail)
    def _vm_num(x, T):
        return x * ((1 / 2288) * 0.2685 ** (1 + (1 - T / 512.4) ** 0.2453)) + (1 - x) * ((1 / 1235) * 0.27136 ** (1 + (1 - T / 536.4) ** 0.24))

    _weir = {k: (8.5 if k == 1 else (0.17 if k == NT else 0.155)) for k in range(1, NT + 1)}
    _mv_ref = {k: _vm_num(ref["x"][k - 1], ref["T"][k - 1]) * ref["M"][k - 1] for k in range(1, NT + 1)}
    _l_ref = {k: 0.166 * max(_mv_ref[k] - _weir[k], 1e-6) ** 1.5 / _vm_num(ref["x"][k - 1], ref["T"][k - 1]) for k in range(1, NT + 1)}

    m.L = pyo.Var(m.tray, m.t, bounds=(0, 1.0e3), initialize=lambda m, k, t: _l_ref[k])
    m.Mv = pyo.Var(m.tray, m.t, bounds=lambda m, k, t: (_weir[k] + 1e-6, 1.0e4), initialize=lambda m, k, t: _mv_ref[k])  # volume holdup, carries the weir minimums
    m.Qc = pyo.Var(m.t, bounds=(0, 1.0e2), initialize=ref["Qc"])
    # unbounded cost vars: a cost var pinned at a bound drags ipopt
    m.cost = pyo.Var(m.t)
    m.term = pyo.Var()

    # property correlations, inline
    def clip(z):
        return 0.5 * (z + pyo.sqrt(z**2 + 0.001))  # smoothed max(z, 0)

    def pm(m, k, t):
        return pyo.exp(m.CapAm - m.CapBm / (m.T[k, t] + m.CapCm))

    def pn(m, k, t):
        return pyo.exp(m.CapAn - m.CapBn / (m.T[k, t] + m.CapCn))

    def Vm(m, k, t):
        return m.x[k, t] * (m.vmc1 * m.vmc2 ** (1 + (1 - m.T[k, t] / m.vmc3) ** m.vmc4)) + (1 - m.x[k, t]) * (m.vnc1 * m.vnc2 ** (1 + (1 - m.T[k, t] / m.vnc3) ** m.vnc4))

    def hl(m, k, t):
        return m.x[k, t] * (m.hlm0 * m.T[k, t] ** 3 + m.hlma * m.T[k, t] ** 2 + m.hlmb * m.T[k, t] + m.hlmc) + (1 - m.x[k, t]) * (m.hln0 * m.T[k, t] ** 3 + m.hlna * m.T[k, t] ** 2 + m.hlnb * m.T[k, t] + m.hlnc)

    def hv(m, k, t):
        return m.y[k, t] * (m.hlm0 * m.T[k, t] ** 3 + m.hlma * m.T[k, t] ** 2 + m.hlmb * m.T[k, t] + m.hlmc + m.r * m.Tkm * pyo.sqrt(1 - (m.p[k] / m.Pkm) * (m.Tkm / m.T[k, t]) ** 3) * (m.a - m.b * m.T[k, t] / m.Tkm + m.c1 * (m.T[k, t] / m.Tkm) ** 7 + m.gm * (m.d - m.l * m.T[k, t] / m.Tkm + m.f * (m.T[k, t] / m.Tkm) ** 7))) + (1 - m.y[k, t]) * (m.hln0 * m.T[k, t] ** 3 + m.hlna * m.T[k, t] ** 2 + m.hlnb * m.T[k, t] + m.hlnc + m.r * m.Tkn * pyo.sqrt(1 - (m.p[k] / m.Pkn) * (m.Tkn / m.T[k, t]) ** 3) * (m.a - m.b * m.T[k, t] / m.Tkn + m.c1 * (m.T[k, t] / m.Tkn) ** 7 + m.gn * (m.d - m.l * m.T[k, t] / m.Tkn + m.f * (m.T[k, t] / m.Tkn) ** 7)))

    def dhl_dx(m, k, t):
        return (m.hlm0 - m.hln0) * m.T[k, t] ** 3 + (m.hlma - m.hlna) * m.T[k, t] ** 2 + (m.hlmb - m.hlnb) * m.T[k, t] + (m.hlmc - m.hlnc)

    def dhl_dT(m, k, t):
        return 3 * m.hln0 * m.T[k, t] ** 2 + 2 * m.hlna * m.T[k, t] + m.hlnb + m.x[k, t] * (3 * (m.hlm0 - m.hln0) * m.T[k, t] ** 2 + 2 * (m.hlma - m.hlna) * m.T[k, t] + (m.hlmb - m.hlnb))

    def D(m, t):
        return m.L[NT, t] / m.Rec[t]  # distillate, set by the reflux ratio

    # --- algebraic equations -------------------------------------------
    @m.Constraint(m.tray, m.t)  # volume holdup (the weir-minimum alias)
    def Mv_def(m, k, t):
        return m.Mv[k, t] == Vm(m, k, t) * m.M[k, t]

    @m.Constraint(m.tray, m.t)  # Francis weir with the smoothed max
    def L_def(m, k, t):
        Mv0 = m.Mv0r if k == 1 else (m.Mv0c if k == NT else m.Mv0t)
        return m.L[k, t] * Vm(m, k, t) == m.Kw * clip(m.Mv[k, t] - Mv0) ** 1.5

    @m.Constraint(m.tray, m.t)  # bubble point (Raoult) defines T
    def T_def(m, k, t):
        return m.p[k] == m.x[k, t] * pm(m, k, t) + (1 - m.x[k, t]) * pn(m, k, t)

    @m.Constraint(m.tray, m.t)  # the differentiated bubble point defines Tdot
    def Tdot_def(m, k, t):
        return m.Tdot[k, t] * (m.x[k, t] * pm(m, k, t) * m.CapBm / (m.T[k, t] + m.CapCm) ** 2 + (1 - m.x[k, t]) * pn(m, k, t) * m.CapBn / (m.T[k, t] + m.CapCn) ** 2) == -(pm(m, k, t) - pn(m, k, t)) * m.dx[k, t]

    @m.Constraint(m.tray_v, m.t)  # Antoine VLE with tray efficiency
    def y_def(m, k, t):
        if k == 1:
            return m.p[1] * m.y[1, t] == m.x[1, t] * pm(m, 1, t)
        return m.y[k, t] == m.alpha[k] * m.x[k, t] * pm(m, k, t) / m.p[k] + (1 - m.alpha[k]) * m.y[k - 1, t]

    @m.Constraint(m.tray_v, m.t)  # tray energy balance defines V; carries dhl/dt
    def V_def(m, k, t):
        lhs = m.M[k, t] * (m.dx[k, t] * dhl_dx(m, k, t) + m.Tdot[k, t] * dhl_dT(m, k, t))
        if k == 1:
            return lhs == m.L[2, t] * (hl(m, 2, t) - hl(m, 1, t)) - m.V[1, t] * (hv(m, 1, t) - hl(m, 1, t)) + m.Qr[t] * m.q_scale
        return lhs == m.V[k - 1, t] * (hv(m, k - 1, t) - hl(m, k, t)) + m.L[k + 1, t] * (hl(m, k + 1, t) - hl(m, k, t)) - m.V[k, t] * (hv(m, k, t) - hl(m, k, t)) + m.feed[k] * (m.hf - hl(m, k, t))

    @m.Constraint(m.t)  # condenser energy balance defines Qc
    def Qc_def(m, t):
        return m.M[NT, t] * (m.dx[NT, t] * dhl_dx(m, NT, t) + m.Tdot[NT, t] * dhl_dT(m, NT, t)) == m.V[NT - 1, t] * (hv(m, NT - 1, t) - hl(m, NT, t)) - m.Qc[t] * m.q_scale

    # --- dynamics: holdup and composition balances ---------------------
    @m.Constraint(m.tray, m.t)
    def M_bal(m, k, t):
        if k == 1:  # reboiler
            return m.dM[1, t] == m.L[2, t] - m.L[1, t] - m.V[1, t]
        if k == NT:  # total condenser
            return m.dM[NT, t] == m.V[NT - 1, t] - m.L[NT, t] - D(m, t)
        return m.dM[k, t] == m.V[k - 1, t] - m.V[k, t] + m.L[k + 1, t] - m.L[k, t] + m.feed[k]

    @m.Constraint(m.tray, m.t)
    def x_bal(m, k, t):
        if k == 1:
            rhs = m.L[2, t] * (m.x[2, t] - m.x[1, t]) - m.V[1, t] * (m.y[1, t] - m.x[1, t])
        elif k == NT:
            rhs = m.V[NT - 1, t] * (m.y[NT - 1, t] - m.x[NT, t])
        else:
            rhs = m.V[k - 1, t] * (m.y[k - 1, t] - m.x[k, t]) + m.L[k + 1, t] * (m.x[k + 1, t] - m.x[k, t]) - m.V[k, t] * (m.y[k, t] - m.x[k, t]) + m.feed[k] * (m.xf - m.x[k, t])
        return m.dx[k, t] == rhs / m.M[k, t]

    # --- costs ---------------------------------------------------------
    def state_dev(m, t):
        return sum((m.x[k, t] - m.x_ss[k]) ** 2 + ((m.M[k, t] - m.M_ss[k]) / m.M_ss[k]) ** 2 for k in m.tray)

    def control_dev(m, t):
        return (m.Rec[t] - m.Rec_ss) ** 2 + (m.Qr[t] - m.Qr_ss) ** 2

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == 10 * state_dev(m, t) + control_dev(m, t)

    tN = m.t.last()

    @m.Constraint()  # the stage cost with the controls removed, at tN
    def terminal(m):
        return m.term == 10 * state_dev(m, tN)

    # --- initial conditions --------------------------------------------
    @m.Constraint(m.tray)
    def M_ic(m, k):
        return m.M[k, 0] == m.M_hat[k]

    @m.Constraint(m.tray)
    def x_ic(m, k):
        return m.x[k, 0] == m.x_hat[k]

    drto.horizon(m.t)
    drto.state(m.M, m.x)
    drto.dynamics(m.M_bal, m.x_bal)
    drto.control(m.Rec, m.Qr, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    drto.initial_condition(m.M_ic, m.x_ic)
    drto.steady_state(m.M, m.M_ss)
    drto.steady_state(m.x, m.x_ss)
    drto.steady_state_control(m.Rec, m.Rec_ss)
    drto.steady_state_control(m.Qr, m.Qr_ss)
    return m
