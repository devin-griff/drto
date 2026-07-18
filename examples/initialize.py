# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Initialization helpers for the column examples.

Each helper interpolates the states linearly in time from the initial
condition to the steady-state target, then fills each algebraic variable
from its own defining equation at those states, member by member, so the
solve starts from a consistent point instead of the flat construction
values. No physics lives here: the algebraic values are computed from the
model's equations with ``calculate_variable_from_constraint``.

A stand-in: initialization proper is a future drto feature. Call after all
transforms, right before the solve; the ``drto.infinite_horizon`` segment
block is handled too when present (its copies hold the horizon-end values,
the steady state, which is where the tail should sit).

Usage from a notebook in ``examples/``::

    from initialize import initialize_column, initialize_binary_column
    initialize_column(m)         # the double (ternary) column
    initialize_binary_column(m)  # the binary column
"""
import logging

import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# state family -> its initial-condition and steady-state Params
_STATES = [
    ("x1", "x1_hat", "x1_ss"),
    ("x2", "x2_hat", "x2_ss"),
    ("M1", "M1_hat", "M1_ss"),
    ("M2", "M2_hat", "M2_ss"),
]

# variable family -> the equation family that defines it, member for member
_PAIRS = [
    ("L1", "L1_def"),
    ("L2", "L2_def"),
    ("y1", "y1_def"),
    ("y2", "y2_def"),
    ("TC1", "TC1_def"),
    ("TC2", "TC2_def"),
    ("purA1", "purA1_def"),
    ("purB2", "purB2_def"),
    ("purC2", "purC2_def"),
    ("cost", "stage"),
    ("term", "terminal"),
]

# the binary column: states, their derivatives, and the pair order (each
# equation's inputs are computed by an earlier pair; y and V solve tray by
# tray in index order, which the family iteration provides)
_BC_STATES = [
    ("M", "M_hat", "M_ss", "dM"),
    ("x", "x_hat", "x_ss", "dx"),
]
_BC_PAIRS = [
    ("T", "T_def"),
    ("Mv", "Mv_def"),
    ("L", "L_def"),
    ("Tdot", "Tdot_def"),
    ("y", "y_def"),
    ("V", "V_def"),
    ("Qc", "Qc_def"),
    ("cost", "stage"),
    ("term", "terminal"),
]


def initialize_column(m):
    """The double column: interpolate the states, compute the algebraics."""
    _interpolate_states(m, _STATES)
    _solve_pairs(m, _PAIRS)
    seg = m.component("drto_infinite_horizon")
    if seg is not None:
        _solve_pairs(seg, _PAIRS)


def initialize_binary_column(m):
    """The binary column: interpolate, slope the derivatives, compute.

    The energy balance and the Tdot identity reference dx/dt, so the
    derivatives get the interpolation slope before the pair solves. On the
    segment the tail sits at the steady state and its derivative copies
    are zero.
    """
    _interpolate_states(m, [(v, h, s) for v, h, s, _ in _BC_STATES])
    tN = m.t.last()
    for var_name, hat_name, ss_name, dv_name in _BC_STATES:
        hat, ss, dv = m.component(hat_name), m.component(ss_name), m.component(dv_name)
        for idx in dv:
            other = idx[:-1]
            dv[idx].set_value((pyo.value(ss[other]) - pyo.value(hat[other])) / tN)
    _solve_pairs(m, _BC_PAIRS)
    seg = m.component("drto_infinite_horizon")
    if seg is not None:
        for var_name, _, _, _ in _BC_STATES:
            dtau = seg.component(var_name + "_dtau")
            for v in dtau.values():
                v.set_value(0.0)
        _solve_pairs(seg, _BC_PAIRS)


def initialize_asu(m):
    """The ASU: the nominal point is steady, so no interpolation.

    The model builds initialized at the nominal data; only the derivatives
    (zero at steady state) and the cost variables need values. On the
    segment the derivative copies are zero.
    """
    for name in ("dMh", "dMl", "dxh", "dxl"):
        for v in m.component(name).values():
            v.set_value(0.0)
    _solve_pairs(m, [("cost", "stage"), ("term", "terminal")])
    seg = m.component("drto_infinite_horizon")
    if seg is not None:
        for name in ("Mh", "Ml", "xh", "xl"):
            for v in seg.component(name + "_dtau").values():
                v.set_value(0.0)


def _interpolate_states(m, states):
    tN = m.t.last()
    for var_name, hat_name, ss_name in states:
        var = m.component(var_name)
        hat, ss = m.component(hat_name), m.component(ss_name)
        for idx in var:
            other, t = idx[:-1], idx[-1]
            a, b = pyo.value(hat[other]), pyo.value(ss[other])
            var[idx].set_value(a + (t / tN) * (b - a))


def _solve_pairs(blk, pairs):
    # a Newton solve mid-ramp can legitimately pass outside a Var's bounds;
    # the value is clamped back in, and Pyomo's outside-bounds warnings are
    # quieted for the duration instead of flooding the notebook
    logger = logging.getLogger("pyomo.core")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        for var_name, con_name in pairs:
            var, con = blk.component(var_name), blk.component(con_name)
            if var is None or con is None:  # the segment carries no cost vars
                continue
            for cd in con.values():
                v = var[cd.index()]
                calculate_variable_from_constraint(v, cd)
                if v.lb is not None and v.value < v.lb:
                    v.set_value(v.lb)
                elif v.ub is not None and v.value > v.ub:
                    v.set_value(v.ub)
    finally:
        logger.setLevel(level)
