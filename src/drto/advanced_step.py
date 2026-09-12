# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The advanced-step correction: ``drto.advanced_step_controller``
(feature 012).

Solve the horizon at a predicted state between samples with pounce, write
the measured state into the initial-condition Params the moment it arrives, and ask
for the corrected solution without re-solving. The solve keeps the
converged factorization, and the correction is a backsolve.
``drto.dynamic_optimization`` declares those Params as pounce sensitivity
parameters whenever pyomo-pounce is importable, so the only requirements
here are that transform and a pounce solve.

The model is not touched. The solution at the predicted state stays in
place as the next solve's warm start, and the Params keep the measured
values the loop wrote.
"""
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap

from drto.dynamic_optimization import _initial_condition_params
from drto.info import info


def advanced_step_controller(m, gradient=False, **kwargs):
    """Return the correction of ``m``'s solution to the measured state.

    The perturbation is read from the initial-condition Params' current values, the
    difference between the measured state the loop wrote and the
    predicted state the model was solved at (the solve point is the
    baseline, so writing the measurement first is the expected pattern).
    Returns ``pyomo_pounce.sens_solution()``, a map from each variable to
    its corrected value, clamped to bounds. The model itself is not
    modified.

    With ``gradient=True`` it returns ``pyomo_pounce.sens_jacobian()`` of the
    declared controls with respect to those Params instead, as nested
    ComponentMaps: ``result[control][param]``. Unrecognized keyword
    arguments pass through to the pounce call, so options pounce grows
    need no change here.

    Requires a pounce solve of ``m``. Without one there is no
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
            f"drto: {fn} reads the perturbation from the initial-condition "
            f"Params. Declare the initial condition first "
            f"(drto.initial_condition)."
        )
    params = _initial_condition_params(reg, fn)

    if gradient:
        out = ComponentMap()
        for u in reg.components("control"):
            out[u] = ComponentMap(
                (p, pyomo_pounce.sens_jacobian(u, wrt=p, **kwargs)) for p in params
            )
        return out
    perturb = [(p, pyo.value(p)) for p in params]
    return pyomo_pounce.sens_solution(m, perturb, **kwargs)
