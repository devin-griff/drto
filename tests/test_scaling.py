# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 023: drto.scale and drto.scaled_solve."""
import pytest

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from test_infinite_horizon import ready_model

IH = "drto.infinite_horizon"

ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
needs_ipopt = pytest.mark.skipif(not ipopt_ok, reason="ipopt not available")


def spanning_model(valued=True):
    """A model whose species span many orders of magnitude.

    ``big`` sits at 1e6 and ``trace`` at 1e-7, both members of one Var, so
    the two must take different factors while the time points of either
    take the same one.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1, 2])
    m.J = pyo.Set(initialize=["big", "trace"])
    m.c = pyo.Var(m.t, m.J)
    m.n = pyo.Var(m.t, m.J)
    m.dn = DerivativeVar(m.n, wrt=m.t)
    m.u = pyo.Var(m.t)
    m.V = pyo.Param(initialize=2.0)
    m.hat = pyo.Param(m.J, initialize={"big": 2e6, "trace": 2e-7}, mutable=True)
    m.ss_big = pyo.Param(initialize=2e6, mutable=True)
    m.ss_trace = pyo.Param(initialize=2e-7, mutable=True)
    m.u_ss = pyo.Param(initialize=1.0, mutable=True)
    m.cost = pyo.Var(sorted(m.t)[:-1])

    if valued:
        for t in m.t:
            for j, v in (("big", 1e6), ("trace", 1e-7)):
                m.c[t, j].set_value(v)
                m.n[t, j].set_value(m.V.value * v)
            m.u[t].set_value(1.0)

    @m.Constraint(m.t, m.J)
    def holdup(mm, t, j):
        return mm.n[t, j] == mm.V * mm.c[t, j]

    @m.Constraint(m.t, m.J)
    def balance(mm, t, j):
        return mm.dn[t, j] == -mm.n[t, j] + mm.u[t] * mm.V * mm.c[t, j]

    @m.Constraint(sorted(m.t)[:-1])
    def stage(mm, t):
        return mm.cost[t] == (
            (mm.n[t, 'big'] - mm.ss_big) ** 2
            + (mm.n[t, 'trace'] - mm.ss_trace) ** 2
            + (mm.u[t] - mm.u_ss) ** 2
        )

    @m.Constraint()
    def terminal(mm):
        tN = mm.t.last()
        return (
            (mm.n[tN, 'big'] - mm.ss_big) ** 2 + (mm.n[tN, 'trace'] - mm.ss_trace) ** 2
        ) >= 0

    @m.Constraint(m.J)
    def ic(mm, j):
        return mm.n[0, j] == mm.hat[j]

    drto.horizon(m.t)
    drto.state(*(m.n[:, j] for j in ("big", "trace")))
    drto.dynamics(m.balance)
    drto.control(m.u)
    drto.initial_condition(m.ic)
    drto.steady_state(m.n[:, "big"], m.ss_big)
    drto.steady_state(m.n[:, "trace"], m.ss_trace)
    drto.steady_state_control(m.u, m.u_ss)
    drto.tracking_stage_cost(m.stage)
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=2, ncp=2, scheme="LAGRANGE-RADAU"
    )
    return m


def factors(m):
    """The Suffix as a name-to-factor mapping."""
    return {k.name: v for k, v in m.scaling_factor.items()}


# ----------------------------------------------------------------------
# variable factors
# ----------------------------------------------------------------------
def test_the_suffix_is_filled_with_powers_of_ten():
    m = spanning_model()
    drto.scale(m)
    assert m.scaling_factor.ctype is pyo.Suffix
    for value in m.scaling_factor.values():
        assert value == pytest.approx(10.0 ** round(pyo.log10(value)))


def test_repeat_calls_replace_the_suffix():
    m = spanning_model()
    drto.scale(m)
    first = len(m.scaling_factor)
    drto.scale(m)
    assert len(m.scaling_factor) == first


def test_time_points_share_a_factor_and_species_do_not():
    m = spanning_model()
    drto.scale(m)
    f = factors(m)
    times = {f[f"c[{t},big]"] for t in m.t}
    assert len(times) == 1
    assert f["c[0,big]"] != f["c[0,trace]"]


def test_the_band_and_the_floor_get_no_entry():
    m = spanning_model()
    m.u[0].set_value(1.0)  # inside [1e-2, 1e2]
    m.c[0, "trace"].set_value(0.0)  # a numerical zero
    for t in m.t:
        m.c[t, "trace"].set_value(0.0)
    drto.scale(m)
    f = factors(m)
    assert "u[0]" not in f
    assert "c[0,trace]" not in f


def test_fixed_variables_get_no_entry():
    m = spanning_model()
    m.c[:, "big"].fix(1e6)
    drto.scale(m)
    assert "c[0,big]" not in factors(m)


def test_a_model_without_values_errors():
    m = spanning_model(valued=False)
    with pytest.raises(ValueError, match="no unfixed variable holds a value"):
        drto.scale(m)


# ----------------------------------------------------------------------
# constraint factors
# ----------------------------------------------------------------------
def test_large_rows_come_to_order_one_and_small_rows_are_left():
    import numpy as np
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

    m = spanning_model()
    drto.scale(m)
    m.probe = pyo.Objective(expr=0.0)  # PyomoNLP reads exactly one
    nlp = PyomoNLP(m)
    jac = nlp.evaluate_jacobian_eq().tocsr()
    variables = nlp.get_pyomo_variables()
    cons = nlp.get_pyomo_equality_constraints()
    vf = np.array([m.scaling_factor.get(v, 1.0) for v in variables])
    cf = np.array([m.scaling_factor.get(c, 1.0) for c in cons])
    for i, c in enumerate(cons):
        row = jac.getrow(i)
        if row.nnz == 0:
            continue
        scaled = max(abs(a) * cf[i] / vf[j] for j, a in zip(row.indices, row.data))
        assert scaled <= 1e2 + 1e-9
    # nothing was scaled up
    assert all(
        v <= 1.0 + 1e-12
        for k, v in m.scaling_factor.items()
        if k.ctype is pyo.Constraint
    )


def test_a_model_with_no_objective_scales():
    m = spanning_model()
    assert next(m.component_data_objects(pyo.Objective, active=True), None) is None
    drto.scale(m)
    assert len(m.scaling_factor) > 0
    assert m.component("_drto_scale_objective") is None


def test_a_constant_zero_objective_scales_like_any_other():
    m = spanning_model()
    drto.scale(m)
    without = dict(factors(m))
    m.obj = pyo.Objective(expr=0.0)
    drto.scale(m)
    assert factors(m) == without


# ----------------------------------------------------------------------
# the terminal segment's pins
# ----------------------------------------------------------------------
def test_the_endpoint_pins_get_no_entries():
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    for v in m.component_data_objects(pyo.Var, descend_into=True):
        if v.value is None:
            v.set_value(1e5)
    drto.scale(m)
    reg = drto.info(m)
    for record in reg._segment_records("state"):
        for key in ("pin", "pin_up", "pin_lo"):
            comp = record.get(key)
            if comp is None:
                continue
            for cd in comp.values() if comp.is_indexed() else (comp,):
                assert cd not in m.scaling_factor


# ----------------------------------------------------------------------
# scaled_solve
# ----------------------------------------------------------------------
@needs_ipopt
def test_the_direct_path_builds_no_clone_and_returns_own_units():
    m = spanning_model()
    m.obj = pyo.Objective(expr=sum(m.cost[t] for t in m.cost))
    res = drto.scaled_solve(m, solver="ipopt")
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    # the model itself was solved: its values are the solution, unscaled
    assert pyo.value(m.n[0, "big"]) == pytest.approx(2e6, rel=1e-6)


@needs_ipopt
def test_both_paths_reach_the_same_solution(monkeypatch):
    # the fallback is chosen by name, so a solver drto does not list as
    # reading the Suffix takes the clone: patch the list to send ipopt
    # down it, and the two paths must agree
    import drto.scaling as scaling

    direct = spanning_model()
    direct.obj = pyo.Objective(expr=sum(direct.cost[t] for t in direct.cost))
    drto.scaled_solve(direct, solver="ipopt")

    fallback = spanning_model()
    fallback.obj = pyo.Objective(expr=sum(fallback.cost[t] for t in fallback.cost))
    monkeypatch.setattr(scaling, "_READS_SUFFIX", ())
    drto.scaled_solve(fallback, solver="ipopt")

    for j in ("big", "trace"):
        assert pyo.value(fallback.n[0, j]) == pytest.approx(
            pyo.value(direct.n[0, j]), rel=1e-6
        )


def test_a_missing_asl_library_names_both_installers(monkeypatch):
    import drto.scaling as scaling
    import pyomo.common.fileutils as fileutils

    monkeypatch.setattr(fileutils, "find_library", lambda *a, **k: None)
    m = spanning_model()
    with pytest.raises(RuntimeError, match="pyomo download-extensions"):
        drto.scale(m)


def test_the_pounce_names_take_the_direct_path():
    from drto.scaling import _POUNCE_SOLVERS, _READS_SUFFIX

    # both names reach the same solver, and both forward the Suffix, so
    # neither builds a scaled clone
    assert _POUNCE_SOLVERS == ("pounce_v2", "pounce")
    assert all(name in _READS_SUFFIX for name in _POUNCE_SOLVERS)
