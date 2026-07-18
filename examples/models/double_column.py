# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The double distillation column, declared for drto: the DAE example.

Two 41-stage columns in series separating a ternary mixture (A light, B
middle, C heavy): column 1 takes A overhead, its bottoms feed column 2,
which takes B overhead and C out the bottom. Constant relative volatility
VLE (division-free form), constant molar vapor flows, Francis weir liquid
flows with a smoothed max, tray temperatures as a linear composition map.
The Skogestad "Column A" lineage (Leer's steady model, Yang's dynamic
form, Griffith's NMPC studies), translated from the original AMPL.

States: tray compositions x1, x2 (tray, component, t) and holdups M1, M2
(tray, t). Controls: reflux LT, boilup VB, distillate D, and bottoms B for
each column, eight in all. The vapor flows and condenser liquid flows are
the controls acting there and appear directly in the balances. Everything
else (weir flows L, vapor compositions y, temperatures TC, the purity
aliases) is algebraic, undeclared, and rides along.

The reference profiles, the nominal feed, and the initial state load from
``data/double_column.json``, converted from the original research data.
Tracking stage and terminal costs toward the reference (weights 10 on
states, 1 on controls); purity specs and physical limits are hard Var
bounds.

Usage from a notebook in ``examples/``::

    from models.double_column import double_column
    m = double_column(N=25, h=1)
"""
import json
from pathlib import Path

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto

_DATA = json.load(open(Path(__file__).parent / "data" / "double_column.json"))


def double_column(N=25, h=1):
    """Return the declared double column with an ``N``-step horizon.

    The time set is initialized with the sample grid (``N`` steps of the
    sampling time ``h``, minutes; the default horizon is 25). Physical
    constants, the reference profiles, and the initial state are mutable
    Params; the initial state is set through ``m.x1_hat`` .. ``m.M2_hat``.
    """
    d = _DATA
    NT, NF = d["NT"], d["NF"]

    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))
    m.tray = pyo.Set(initialize=range(1, NT + 1))
    m.comp = pyo.Set(initialize=[1, 2])  # A and B; C is 1 - x_A - x_B
    m.tray_v = pyo.Set(initialize=range(1, NT))  # vapor leaves trays 1..NT-1
    m.tray_weir = pyo.Set(initialize=range(2, NT))  # Francis weir trays

    m.alpha1 = pyo.Param(initialize=2.0, mutable=True)  # volatility A/B
    m.alpha2 = pyo.Param(initialize=1.5, mutable=True)  # volatility B/C
    m.Kbf = pyo.Param(initialize=29.65032, mutable=True)  # weir constant below feed
    m.Kuf = pyo.Param(initialize=21.65032, mutable=True)  # weir constant above feed
    m.Muw = pyo.Param(initialize=0.25, mutable=True)  # holdup under the weir (kmol)
    m.TbA = pyo.Param(initialize=353.3, mutable=True)  # boiling temperatures (K)
    m.TbB = pyo.Param(initialize=383.8, mutable=True)
    m.TbC = pyo.Param(initialize=411.5, mutable=True)
    m.F = pyo.Param(initialize=d["F"], mutable=True)  # feed rate
    m.qF = pyo.Param(initialize=d["qF"], mutable=True)  # feed liquid fraction
    m.zF1 = pyo.Param(initialize=d["zF"][0], mutable=True)  # feed composition A
    m.zF2 = pyo.Param(initialize=d["zF"][1], mutable=True)  # feed composition B

    # the reference steady state (targets) and the initial state (hooks)
    x1r, x2r = d["x1_ref"], d["x2_ref"]
    m.x1_ss = pyo.Param(m.tray, m.comp, initialize=lambda m, i, j: x1r[i - 1][j - 1], mutable=True)
    m.x2_ss = pyo.Param(m.tray, m.comp, initialize=lambda m, i, j: x2r[i - 1][j - 1], mutable=True)
    m.M1_ss = pyo.Param(m.tray, initialize=lambda m, i: d["M1_ref"][i - 1], mutable=True)
    m.M2_ss = pyo.Param(m.tray, initialize=lambda m, i: d["M2_ref"][i - 1], mutable=True)
    cr = d["controls_ref"]
    m.LT1_ss = pyo.Param(initialize=cr["LT1"], mutable=True)
    m.VB1_ss = pyo.Param(initialize=cr["VB1"], mutable=True)
    m.D1_ss = pyo.Param(initialize=cr["D1"], mutable=True)
    m.B1_ss = pyo.Param(initialize=cr["B1"], mutable=True)
    m.LT2_ss = pyo.Param(initialize=cr["LT2"], mutable=True)
    m.VB2_ss = pyo.Param(initialize=cr["VB2"], mutable=True)
    m.D2_ss = pyo.Param(initialize=cr["D2"], mutable=True)
    m.B2_ss = pyo.Param(initialize=cr["B2"], mutable=True)
    x1i, x2i = d["x1_init"], d["x2_init"]
    m.x1_hat = pyo.Param(m.tray, m.comp, initialize=lambda m, i, j: x1i[i - 1][j - 1], mutable=True)
    m.x2_hat = pyo.Param(m.tray, m.comp, initialize=lambda m, i, j: x2i[i - 1][j - 1], mutable=True)
    m.M1_hat = pyo.Param(m.tray, initialize=lambda m, i: d["M1_init"][i - 1], mutable=True)
    m.M2_hat = pyo.Param(m.tray, initialize=lambda m, i: d["M2_init"][i - 1], mutable=True)

    # states, initialized at the reference
    m.x1 = pyo.Var(m.tray, m.comp, m.t, bounds=(0, 1), initialize=lambda m, i, j, t: x1r[i - 1][j - 1])
    m.x2 = pyo.Var(m.tray, m.comp, m.t, bounds=(0, 1), initialize=lambda m, i, j, t: x2r[i - 1][j - 1])
    m.M1 = pyo.Var(m.tray, m.t, bounds=(0.25, 0.75), initialize=lambda m, i, t: d["M1_ref"][i - 1])
    m.M2 = pyo.Var(m.tray, m.t, bounds=(0.25, 0.75), initialize=lambda m, i, t: d["M2_ref"][i - 1])
    m.dx1 = DerivativeVar(m.x1, wrt=m.t)
    m.dx2 = DerivativeVar(m.x2, wrt=m.t)
    m.dM1 = DerivativeVar(m.M1, wrt=m.t)
    m.dM2 = DerivativeVar(m.M2, wrt=m.t)

    # controls: reflux, boilup, distillate, bottoms per column
    m.LT1 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["LT1"])
    m.VB1 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["VB1"])
    m.D1 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["D1"])
    m.B1 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["B1"])
    m.LT2 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["LT2"])
    m.VB2 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["VB2"])
    m.D2 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["D2"])
    m.B2 = pyo.Var(m.t, bounds=(0, 10), initialize=cr["B2"])

    # algebraic variables: weir liquid flows, vapor compositions, tray
    # temperatures, and the purity aliases carrying the spec bounds
    m.L1 = pyo.Var(m.tray_weir, m.t, bounds=(0, None), initialize=1.0)
    m.L2 = pyo.Var(m.tray_weir, m.t, bounds=(0, None), initialize=1.0)
    m.y1 = pyo.Var(m.tray_v, m.comp, m.t, bounds=(0, 1), initialize=0.4)
    m.y2 = pyo.Var(m.tray_v, m.comp, m.t, bounds=(0, 1), initialize=0.4)
    m.TC1 = pyo.Var(m.tray, m.t, bounds=(350, 412), initialize=380)
    m.TC2 = pyo.Var(m.tray, m.t, bounds=(350, 412), initialize=380)
    m.purA1 = pyo.Var(m.t, bounds=(0.95, None), initialize=x1r[NT - 1][0])  # A overhead, column 1
    m.purB2 = pyo.Var(m.t, bounds=(0.95, None), initialize=x2r[NT - 1][1])  # B overhead, column 2
    m.purC2 = pyo.Var(m.t, bounds=(0.95, None), initialize=1 - x2r[0][0] - x2r[0][1])  # C bottoms, column 2
    # unbounded cost vars: a cost var pinned at a bound drags ipopt
    m.cost = pyo.Var(m.t)
    m.term = pyo.Var()

    # vapor flows and condenser liquid flows are the controls acting there
    def V1(m, i, t):
        return m.VB1[t] if i <= NF - 1 else m.VB1[t] + (1 - m.qF) * m.F

    def V2(m, i, t):
        return m.VB2[t]

    def L1_(m, i, t):
        return m.LT1[t] if i == NT else m.L1[i, t]

    def L2_(m, i, t):
        return m.LT2[t] if i == NT else m.L2[i, t]

    def zF(m, j):
        return m.zF1 if j == 1 else m.zF2

    # --- algebraic equations -------------------------------------------
    @m.Constraint(m.tray_weir, m.t)  # Francis weir with a smoothed max
    def L1_def(m, i, t):
        K = m.Kbf if i <= NF else m.Kuf
        return m.L1[i, t] == K * ((m.M1[i, t] - m.Muw + pyo.sqrt((m.M1[i, t] - m.Muw) ** 2 + 1e-8)) / 2) ** 1.5

    @m.Constraint(m.tray_weir, m.t)
    def L2_def(m, i, t):
        K = m.Kbf if i <= NF else m.Kuf
        return m.L2[i, t] == K * ((m.M2[i, t] - m.Muw + pyo.sqrt((m.M2[i, t] - m.Muw) ** 2 + 1e-8)) / 2) ** 1.5

    @m.Constraint(m.tray_v, m.comp, m.t)  # constant-volatility VLE, division-free
    def y1_def(m, i, j, t):
        alpha = m.alpha1 if j == 1 else m.alpha2
        denom = 1 + (m.alpha1 - 1) * m.x1[i, 1, t] + (m.alpha2 - 1) * m.x1[i, 2, t]
        return m.y1[i, j, t] * denom == alpha * m.x1[i, j, t]

    @m.Constraint(m.tray_v, m.comp, m.t)
    def y2_def(m, i, j, t):
        alpha = m.alpha1 if j == 1 else m.alpha2
        denom = 1 + (m.alpha1 - 1) * m.x2[i, 1, t] + (m.alpha2 - 1) * m.x2[i, 2, t]
        return m.y2[i, j, t] * denom == alpha * m.x2[i, j, t]

    @m.Constraint(m.tray, m.t)  # tray temperature: linear composition map
    def TC1_def(m, i, t):
        return m.TC1[i, t] == m.TbA * m.x1[i, 1, t] + m.TbB * m.x1[i, 2, t] + m.TbC * (1 - m.x1[i, 1, t] - m.x1[i, 2, t])

    @m.Constraint(m.tray, m.t)
    def TC2_def(m, i, t):
        return m.TC2[i, t] == m.TbA * m.x2[i, 1, t] + m.TbB * m.x2[i, 2, t] + m.TbC * (1 - m.x2[i, 1, t] - m.x2[i, 2, t])

    @m.Constraint(m.t)  # the purity aliases carry the hard spec bounds
    def purA1_def(m, t):
        return m.purA1[t] == m.x1[NT, 1, t]

    @m.Constraint(m.t)
    def purB2_def(m, t):
        return m.purB2[t] == m.x2[NT, 2, t]

    @m.Constraint(m.t)
    def purC2_def(m, t):
        return m.purC2[t] == 1 - m.x2[1, 1, t] - m.x2[1, 2, t]

    # --- dynamics: holdup and composition balances ---------------------
    @m.Constraint(m.tray, m.t)
    def M1_bal(m, i, t):
        if i == 1:  # reboiler
            return m.dM1[1, t] == L1_(m, 2, t) - V1(m, 1, t) - m.B1[t]
        if i == NT:  # total condenser
            return m.dM1[NT, t] == V1(m, NT - 1, t) - m.LT1[t] - m.D1[t]
        feed = m.F if i == NF else 0
        return m.dM1[i, t] == L1_(m, i + 1, t) - L1_(m, i, t) + V1(m, i - 1, t) - V1(m, i, t) + feed

    @m.Constraint(m.tray, m.comp, m.t)
    def x1_bal(m, i, j, t):
        if i == 1:
            rhs = L1_(m, 2, t) * (m.x1[2, j, t] - m.x1[1, j, t]) - V1(m, 1, t) * (m.y1[1, j, t] - m.x1[1, j, t])
        elif i == NT:
            rhs = V1(m, NT - 1, t) * (m.y1[NT - 1, j, t] - m.x1[NT, j, t])
        else:
            rhs = (
                L1_(m, i + 1, t) * (m.x1[i + 1, j, t] - m.x1[i, j, t])
                + V1(m, i - 1, t) * (m.y1[i - 1, j, t] - m.x1[i, j, t])
                - V1(m, i, t) * (m.y1[i, j, t] - m.x1[i, j, t])
            )
            if i == NF:
                rhs = rhs + m.F * (zF(m, j) - m.x1[i, j, t])
        return m.dx1[i, j, t] == rhs / m.M1[i, t]

    @m.Constraint(m.tray, m.t)
    def M2_bal(m, i, t):
        if i == 1:
            return m.dM2[1, t] == L2_(m, 2, t) - V2(m, 1, t) - m.B2[t]
        if i == NT:
            return m.dM2[NT, t] == V2(m, NT - 1, t) - m.LT2[t] - m.D2[t]
        feed = m.B1[t] if i == NF else 0
        return m.dM2[i, t] == L2_(m, i + 1, t) - L2_(m, i, t) + V2(m, i - 1, t) - V2(m, i, t) + feed

    @m.Constraint(m.tray, m.comp, m.t)
    def x2_bal(m, i, j, t):
        if i == 1:
            rhs = L2_(m, 2, t) * (m.x2[2, j, t] - m.x2[1, j, t]) - V2(m, 1, t) * (m.y2[1, j, t] - m.x2[1, j, t])
        elif i == NT:
            rhs = V2(m, NT - 1, t) * (m.y2[NT - 1, j, t] - m.x2[NT, j, t])
        else:
            rhs = (
                L2_(m, i + 1, t) * (m.x2[i + 1, j, t] - m.x2[i, j, t])
                + V2(m, i - 1, t) * (m.y2[i - 1, j, t] - m.x2[i, j, t])
                - V2(m, i, t) * (m.y2[i, j, t] - m.x2[i, j, t])
            )
            if i == NF:  # column 1 bottoms is column 2 feed
                rhs = rhs + m.B1[t] * (m.x1[1, j, t] - m.x2[i, j, t])
        return m.dx2[i, j, t] == rhs / m.M2[i, t]

    # --- costs ---------------------------------------------------------
    def state_dev(m, t):
        return sum(
            (m.x1[i, j, t] - m.x1_ss[i, j]) ** 2 + (m.x2[i, j, t] - m.x2_ss[i, j]) ** 2
            for i in m.tray
            for j in m.comp
        ) + sum((m.M1[i, t] - m.M1_ss[i]) ** 2 + (m.M2[i, t] - m.M2_ss[i]) ** 2 for i in m.tray)

    def control_dev(m, t):
        return (
            (m.LT1[t] - m.LT1_ss) ** 2
            + (m.VB1[t] - m.VB1_ss) ** 2
            + (m.D1[t] - m.D1_ss) ** 2
            + (m.B1[t] - m.B1_ss) ** 2
            + (m.LT2[t] - m.LT2_ss) ** 2
            + (m.VB2[t] - m.VB2_ss) ** 2
            + (m.D2[t] - m.D2_ss) ** 2
            + (m.B2[t] - m.B2_ss) ** 2
        )

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == 10 * state_dev(m, t) + control_dev(m, t)

    tN = m.t.last()

    @m.Constraint()  # the stage cost with the controls removed, at tN
    def terminal(m):
        return m.term == 10 * state_dev(m, tN)

    # --- initial conditions --------------------------------------------
    @m.Constraint(m.tray, m.comp)
    def x1_ic(m, i, j):
        return m.x1[i, j, 0] == m.x1_hat[i, j]

    @m.Constraint(m.tray, m.comp)
    def x2_ic(m, i, j):
        return m.x2[i, j, 0] == m.x2_hat[i, j]

    @m.Constraint(m.tray)
    def M1_ic(m, i):
        return m.M1[i, 0] == m.M1_hat[i]

    @m.Constraint(m.tray)
    def M2_ic(m, i):
        return m.M2[i, 0] == m.M2_hat[i]

    drto.horizon(m.t)
    drto.state(m.x1, m.x2, m.M1, m.M2)
    drto.dynamics(m.M1_bal, m.x1_bal, m.M2_bal, m.x2_bal)
    drto.control(m.LT1, m.VB1, m.D1, m.B1, m.LT2, m.VB2, m.D2, m.B2, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    drto.initial_condition(m.x1_ic, m.x2_ic, m.M1_ic, m.M2_ic)
    drto.steady_state(m.x1, m.x1_ss)
    drto.steady_state(m.x2, m.x2_ss)
    drto.steady_state(m.M1, m.M1_ss)
    drto.steady_state(m.M2, m.M2_ss)
    drto.steady_state_control(m.LT1, m.LT1_ss)
    drto.steady_state_control(m.VB1, m.VB1_ss)
    drto.steady_state_control(m.D1, m.D1_ss)
    drto.steady_state_control(m.B1, m.B1_ss)
    drto.steady_state_control(m.LT2, m.LT2_ss)
    drto.steady_state_control(m.VB2, m.VB2_ss)
    drto.steady_state_control(m.D2, m.D2_ss)
    drto.steady_state_control(m.B2, m.B2_ss)
    return m
