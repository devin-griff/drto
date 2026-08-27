# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 012: drto.advanced_step_controller."""
import sys

import pyomo.environ as pyo
import pytest

import drto
from test_infinite_horizon import ready_model

pyomo_pounce = pytest.importorskip("pyomo_pounce")

pounce_ok = bool(drto.scaling.solver_by_name("pounce").available())
needs_pounce = pytest.mark.skipif(not pounce_ok, reason="pounce not available")
ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def controller(z_hat=0.1):
    """The declared linear model as a dynamic optimization."""
    m = ready_model()
    m.z_hat.set_value(z_hat)
    pyo.TransformationFactory("drto.dynamic_optimization").apply_to(m)
    return m


def test_dynamic_optimization_declares_the_initial_condition_params():
    m = controller()
    assert pyomo_pounce.sens.has_declarations(m)
    assert "1 initial-condition Params declared" in repr(drto.info(m))


@needs_ipopt
def test_declaration_is_inert_for_other_solvers():
    m = controller()
    res = drto.scaling.solver_by_name("ipopt").solve(m)
    assert drto.scaling.solved_to_optimality(res)


def test_without_a_pounce_solve_raises_the_no_session_error():
    m = controller()
    with pytest.raises(RuntimeError, match="SolverFactory\\('pounce'\\)"):
        drto.advanced_step_controller(m)


def test_without_pyomo_pounce_the_error_names_the_extra(monkeypatch):
    m = controller()
    monkeypatch.setitem(sys.modules, "pyomo_pounce", None)
    with pytest.raises(RuntimeError, match="drto\\[pounce\\]"):
        drto.advanced_step_controller(m)


@needs_pounce
def test_estimate_agrees_with_a_resolve_and_leaves_the_model_alone():
    m = controller(z_hat=0.1)
    drto.scaling.solver_by_name("pounce").solve(m)
    before = {v.name: v.value for v in m.component_data_objects(pyo.Var)}

    m.z_hat.set_value(0.12)  # the measurement arrives
    est = drto.advanced_step_controller(m)

    # the model is untouched: the solution at the prediction stays as the
    # next solve's warm start, and the Param keeps the measured value
    after = {v.name: v.value for v in m.component_data_objects(pyo.Var)}
    assert after == before
    assert pyo.value(m.z_hat) == 0.12

    # the corrected solution agrees with a re-solve at the measurement
    mt = controller(z_hat=0.12)
    drto.scaling.solver_by_name("pounce").solve(mt)
    for v, vt in ((m.z, mt.z), (m.u, mt.u)):
        for idx in v:
            assert est[v[idx]] == pytest.approx(pyo.value(vt[idx]), abs=5e-3)


@needs_pounce
def test_gradient_maps_controls_to_initial_condition_params():
    m = controller(z_hat=0.1)
    drto.scaling.solver_by_name("pounce").solve(m)
    grads = drto.advanced_step_controller(m, gradient=True)
    G = grads[m.u][m.z_hat]

    # finite difference on the first free control value (the control's
    # own index set: cvp rebuilds it over the profile's free points)
    h = 1e-4
    hi = controller(z_hat=0.1 + h)
    lo = controller(z_hat=0.1 - h)
    drto.scaling.solver_by_name("pounce").solve(hi)
    drto.scaling.solver_by_name("pounce").solve(lo)
    i1 = sorted(m.u)[0]
    fd = (pyo.value(hi.u[i1]) - pyo.value(lo.u[i1])) / (2 * h)
    assert G[m.u[i1], m.z_hat] == pytest.approx(fd, abs=1e-3)


@needs_pounce
def test_unrecognized_kwargs_pass_through_to_pounce():
    m = controller()
    drto.scaling.solver_by_name("pounce").solve(m)
    with pytest.raises(TypeError, match="definitely_not_an_option"):
        drto.advanced_step_controller(m, definitely_not_an_option=1)
