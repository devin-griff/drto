# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 023: drto.scale and drto.scaled_solve."""
import pytest

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from drto.scaling import _CLAMP
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
# the sources
# ----------------------------------------------------------------------
def test_bounds_source_reads_no_value():
    m = spanning_model()
    for t in m.t:
        m.u[t].setlb(-1e6)
        m.u[t].setub(1e6)
        m.u[t].set_value(0.0)  # a control at its target
    drto.scale(m, source="bounds")
    f = factors(m)
    # the control at zero takes its factor from the bounds
    assert f["u[0]"] == pytest.approx(1e-6)
    # a group with no doubly bounded member keeps factor one
    assert "c[0,big]" not in f
    assert "c[0,trace]" not in f


def test_units_mapping_scales_its_dimensions():
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1])
    m.e = pyo.Var(m.t, initialize=1e7, units=pyo.units.J)
    m.q = pyo.Var(m.t, initialize=0.0, units=pyo.units.W)  # a duty at zero
    m.x = pyo.Var(m.t, initialize=5e4)  # dimensionless, unmapped
    m.c = pyo.Constraint(m.t, rule=lambda mm, t: mm.e[t] == 3600.0 * mm.q[t])
    drto.horizon(m.t)
    drto.scale(m, source={"J": 1e7, "W": 1e6})
    f = factors(m)
    assert f["e[0]"] == pytest.approx(1e-7)
    assert f["q[0]"] == pytest.approx(1e-6)
    assert "x[0]" not in f


def test_an_unknown_source_errors():
    m = spanning_model()
    with pytest.raises(ValueError, match="'point', source='bounds', or a"):
        drto.scale(m, source="units")


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


def test_a_segment_derivative_takes_its_state_factor():
    """The tail's derivatives vanish at the equilibrium it approaches, so
    their own magnitudes there are zeros rather than scales."""
    m = ready_model()
    pyo.TransformationFactory(IH).apply_to(m)
    for v in m.component_data_objects(pyo.Var, descend_into=True):
        if v.value is None:
            v.set_value(1e5)
    # the tail at rest: every segment derivative sits at a numerical zero
    # while the state it differentiates stays at its own magnitude
    segment = drto.info(m)._segment_records()
    copies = {id(r["copy"]) for r in segment if r.get("copy") is not None}
    derivatives = [
        c
        for c in m.component_objects(pyo.Var, active=True, descend_into=True)
        if isinstance(c, DerivativeVar) and id(c.get_state_var()) in copies
    ]
    assert derivatives, "the transform left no segment derivatives to check"
    for comp in derivatives:
        for v in comp.values() if comp.is_indexed() else (comp,):
            v.set_value(1e-16)

    drto.scale(m)
    for comp in derivatives:
        state = comp.get_state_var()
        for v in comp.values() if comp.is_indexed() else (comp,):
            partner = state[v.index()]
            assert m.scaling_factor[v] == m.scaling_factor[partner]
            # measured from its own value it would have taken the clamp
            assert m.scaling_factor[v] != 10.0**_CLAMP


# ----------------------------------------------------------------------
# scaled_solve
# ----------------------------------------------------------------------
class _RecordingFactory:
    """Stands in for SolverFactory, recording the options it is handed."""

    def __init__(self, record):
        self.record = record

    def __call__(self, name):
        return self

    def solve(self, model, tee=False, options=None):
        self.record.update(options or {})
        return "solved"


@needs_ipopt
def test_the_solve_builds_no_clone_and_returns_own_units():
    m = spanning_model()
    m.obj = pyo.Objective(expr=sum(m.cost[t] for t in m.cost))
    res = drto.scaled_solve(m, solver="ipopt")
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    # the model itself was solved: its values are the solution, unscaled
    assert pyo.value(m.n[0, "big"]) == pytest.approx(2e6, rel=1e-6)


def test_ipopt_v2_gets_no_scaling_option(monkeypatch):
    # the NL-v2 writer consumes the Suffix and scales the problem as it
    # writes, so the option would name a job already done
    import pyomo.environ

    record = {}
    monkeypatch.setattr(pyomo.environ, "SolverFactory", _RecordingFactory(record))
    m = spanning_model()
    drto.scaled_solve(m, solver="ipopt_v2")
    assert "nlp_scaling_method" not in record


def test_scaled_solve_forwards_the_source(monkeypatch):
    import pyomo.environ

    record = {}
    monkeypatch.setattr(pyomo.environ, "SolverFactory", _RecordingFactory(record))
    m = spanning_model()
    for t in m.t:
        m.u[t].setlb(-1e6)
        m.u[t].setub(1e6)
    drto.scaled_solve(m, source="bounds", solver="ipopt")
    f = factors(m)
    assert f["u[0]"] == pytest.approx(1e-6)
    assert "c[0,big]" not in f


def test_an_unlisted_solver_warns_and_solves_unscaled(monkeypatch):
    import pyomo.environ

    record = {}
    monkeypatch.setattr(pyomo.environ, "SolverFactory", _RecordingFactory(record))
    m = spanning_model()
    with pytest.warns(UserWarning, match="factors were not applied"):
        drto.scaled_solve(m, solver="cbc")
    assert "nlp_scaling_method" not in record
    # the factors were still measured onto the model
    assert len(m.scaling_factor) > 0


def test_a_missing_asl_library_names_both_installers(monkeypatch):
    import drto.scaling as scaling
    import pyomo.common.fileutils as fileutils

    monkeypatch.setattr(fileutils, "find_library", lambda *a, **k: None)
    m = spanning_model()
    with pytest.raises(RuntimeError, match="pyomo download-extensions"):
        drto.scale(m)


def test_the_solver_lists_agree():
    from drto.scaling import _POUNCE_SOLVERS, _READS_SUFFIX, _TAKES_OPTION

    # both pounce names reach the same solver and take the option;
    # ipopt_v2 applies the factors without one, in the NL-v2 writer
    assert _POUNCE_SOLVERS == ("pounce_v2", "pounce")
    assert all(name in _TAKES_OPTION for name in _POUNCE_SOLVERS)
    assert "ipopt_v2" in _READS_SUFFIX and "ipopt_v2" not in _TAKES_OPTION
