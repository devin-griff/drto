# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The advanced-step correction: ``drto.advanced_step_controller``
(feature 012).

Solve the horizon at a predicted state between samples with pounce, write
the measured state into the initial-condition Params the moment it arrives, and ask
for the corrected solution without re-solving: the converged
factorization is kept by the solve, and the correction is a backsolve.
``drto.dynamic_optimization`` declares those Params as pounce sensitivity
parameters whenever pyomo-pounce is importable, so the only requirements
here are that transform and a pounce solve.

The model is not touched: the solution at the predicted state stays in
place as the next solve's warm start, and the Params keep the measured
values the loop wrote.
"""
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap

from drto.declarations import _is_var_member, _side_matching
from drto.info import info


def advanced_step_controller(m, gradient=False, **kwargs):
    """Return the correction of ``m``'s solution to the measured state.

    The perturbation is read from the initial-condition Params' current values, the
    difference between the measured state the loop wrote and the
    predicted state the model was solved at (the solve point is the
    baseline, so writing the measurement first is the expected pattern).
    Returns ``pyomo_pounce.estimate()``: a map from each variable to its
    corrected value, clamped to bounds. The model itself is not modified.

    With ``gradient=True`` it returns ``pyomo_pounce.gradient()`` of the
    declared controls with respect to those Params instead, as nested
    ComponentMaps: ``result[control][param]``. Unrecognized keyword
    arguments pass through to the pounce call, so options pounce grows
    need no change here.

    Requires a pounce solve of ``m``: without one there is no
    factorization, and the call raises pounce's own no-session error
    instructing to solve with pounce first.
    """
    fn = "advanced_step_controller"
    try:
        import pyomo_pounce
    except ImportError as err:
        raise RuntimeError(
            "drto: advanced_step_controller requires pyomo-pounce "
            "(pip install drto[pounce], or pip install pyomo-pounce)."
        ) from err

    reg = info(m)
    if not reg.has_declaration("initial_condition"):
        raise ValueError(
            f"drto: {fn} reads the perturbation from the initial-condition Params, "
            f"the initial-condition Params; declare the initial condition "
            f"first (drto.initial_condition)."
        )
    hooks, seen = [], set()
    for con in reg.components("initial_condition"):
        for cd in con.values() if con.is_indexed() else (con,):
            _, other = _side_matching(cd, _is_var_member, fn, "a state member")
            if id(other) not in seen:
                seen.add(id(other))
                hooks.append(other)

    if gradient:
        out = ComponentMap()
        for u in reg.components("control"):
            out[u] = ComponentMap(
                (p, pyomo_pounce.gradient(u, wrt=p, **kwargs)) for p in hooks
            )
        return out
    perturb = [(p, pyo.value(p)) for p in hooks]
    return pyomo_pounce.estimate(m, perturb, **kwargs)
