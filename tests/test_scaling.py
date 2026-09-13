# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 023: drto.scale and drto.scaled_solve."""
import pytest

import pyomo.environ as pyo
from pyomo.core.expr import identify_variables
from pyomo.dae import ContinuousSet, DerivativeVar

import drto
from drto.scaling import _CLAMP, _prune_suffixes
from test_infinite_horizon import ready_model

IH = "drto.infinite_horizon"

ipopt_ok = bool(drto.scaling.solver_by_name("ipopt").available())
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

    # the members a discretization equation reaches. Gauss-Legendre puts
    # its points inside each element, so the derivative at an element
    # boundary sits in no equation and the NL file has no column for it
    live = set()
    for con in m.component_data_objects(pyo.Constraint, active=True):
        for v in identify_variables(con.expr, include_fixed=False):
            live.add(id(v))
    for obj in m.component_data_objects(pyo.Objective, active=True):
        for v in identify_variables(obj.expr, include_fixed=False):
            live.add(id(v))

    drto.scale(m)
    boundary = 0
    for comp in derivatives:
        state = comp.get_state_var()
        for v in comp.values() if comp.is_indexed() else (comp,):
            if id(v) not in live:
                assert v not in m.scaling_factor
                boundary += 1
                continue
            partner = state[v.index()]
            assert m.scaling_factor[v] == m.scaling_factor[partner]
            # measured from its own value it would have taken the clamp
            assert m.scaling_factor[v] != 10.0**_CLAMP
    assert boundary, "the segment left no boundary derivative to check"


# ----------------------------------------------------------------------
# scaled_solve
# ----------------------------------------------------------------------
class _RecordingSolver:
    """Stands in for solver_by_name, recording the solver options."""

    def __init__(self, record):
        self.record = record

    def __call__(self, name):
        self.record["name"] = name
        return self

    def solve(self, model, tee=False, solver_options=None, **kwargs):
        from pyomo.contrib.solver.common.results import Results

        self.record.update(solver_options or {})
        res = Results()
        return res


@needs_ipopt
def test_the_solve_builds_no_clone_and_returns_own_units():
    m = spanning_model()
    m.obj = pyo.Objective(expr=sum(m.cost[t] for t in m.cost))
    res = drto.scaled_solve(m, solver="ipopt")
    assert drto.scaling.solved_to_optimality(res)
    # the model itself was solved: its values are the solution, unscaled
    assert pyo.value(m.n[0, "big"]) == pytest.approx(2e6, rel=1e-6)


def test_ipopt_gets_no_scaling_option(monkeypatch):
    # the NL writer consumes the Suffix and scales the problem as it
    # writes, so the option would name a job already done
    record = {}
    monkeypatch.setattr(drto.scaling, "solver_by_name", _RecordingSolver(record))
    m = spanning_model()
    drto.scaled_solve(m, solver="ipopt")
    assert "nlp_scaling_method" not in record


def test_scaled_solve_forwards_the_source(monkeypatch):
    record = {}
    monkeypatch.setattr(drto.scaling, "solver_by_name", _RecordingSolver(record))
    m = spanning_model()
    for t in m.t:
        m.u[t].setlb(-1e6)
        m.u[t].setub(1e6)
    drto.scaled_solve(m, source="bounds", solver="ipopt")
    f = factors(m)
    assert f["u[0]"] == pytest.approx(1e-6)
    assert "c[0,big]" not in f


def test_an_unlisted_solver_warns_and_solves_unscaled(monkeypatch):
    record = {}
    monkeypatch.setattr(drto.scaling, "solver_by_name", _RecordingSolver(record))
    m = spanning_model()
    with pytest.warns(UserWarning, match="factors were not applied"):
        drto.scaled_solve(m, solver="highs")
    assert "nlp_scaling_method" not in record
    # the factors were still measured onto the model
    assert len(m.scaling_factor) > 0


def test_an_unknown_name_raises_with_the_registry():
    m = spanning_model()
    with pytest.warns(UserWarning, match="factors were not applied"):
        with pytest.raises(ValueError, match="native factory"):
            drto.scaled_solve(m, solver="no_such_solver")


def test_a_missing_asl_library_names_both_installers(monkeypatch):
    import drto.scaling as scaling
    import pyomo.common.fileutils as fileutils

    monkeypatch.setattr(fileutils, "find_library", lambda *a, **k: None)
    m = spanning_model()
    with pytest.raises(RuntimeError, match="pyomo download-extensions"):
        drto.scale(m)


def test_the_solver_lists_agree():
    from drto.scaling import _POUNCE_SOLVERS, _READS_SUFFIX, _TAKES_OPTION

    # both pounce names resolve to the one native solver and take the
    # option; ipopt applies the factors without one, in the NL writer
    assert _POUNCE_SOLVERS == ("pounce", "pounce_v2")
    assert all(name in _TAKES_OPTION for name in _POUNCE_SOLVERS)
    assert "ipopt" in _READS_SUFFIX and "ipopt" not in _TAKES_OPTION


def test_both_pounce_names_resolve_natively():
    from pyomo.contrib.solver.common.factory import SolverFactory

    for name in ("pounce", "pounce_v2"):
        solver = drto.scaling.solver_by_name(name)
        assert type(solver) is type(SolverFactory("pounce"))


# ── the entries the NL writer has no place for ───────────────────────────────


def test_pruning_drops_detached_and_deactivated_keys():
    # the transforms delete some tagged components and deactivate
    # others, and the NL writer warns about an entry keyed on either
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.kept = pyo.Constraint(expr=m.x >= 0)
    m.gone = pyo.Constraint(expr=m.y >= 0)
    m.off = pyo.Constraint(expr=m.x + m.y >= 0)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for con in (m.kept, m.gone, m.off):
        m.scaling_factor[con] = 2.0
    m.scaling_factor[m.x] = 3.0

    detached = m.gone
    m.del_component(m.gone)
    m.off.deactivate()
    _prune_suffixes(m)

    assert m.kept in m.scaling_factor
    assert m.x in m.scaling_factor
    assert detached not in m.scaling_factor
    assert m.off not in m.scaling_factor


def test_pruning_keeps_a_members_siblings():
    # deactivating one member leaves the rest of the container active,
    # so the read is on the data object rather than on its parent
    m = pyo.ConcreteModel()
    m.i = pyo.Set(initialize=[1, 2])
    m.x = pyo.Var(m.i, initialize=1.0)
    m.c = pyo.Constraint(m.i, rule=lambda b, i: b.x[i] >= 0)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for i in m.i:
        m.scaling_factor[m.c[i]] = 2.0

    m.c[1].deactivate()
    _prune_suffixes(m)

    assert m.c[1] not in m.scaling_factor
    assert m.c[2] in m.scaling_factor


def test_scale_drops_a_var_the_writer_will_not_write():
    # drto.infinite_horizon leaves the segment's boundary derivative Vars
    # in no equation, and scale measures every unfixed Var, so the writer
    # warned about a factor keyed on a column it never wrote
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1e6)
    m.orphan = pyo.Var(initialize=1e6)
    m.c = pyo.Constraint(expr=m.x == 1e6)
    drto.scale(m, source="point")

    assert m.x in m.scaling_factor
    assert m.orphan not in m.scaling_factor


def test_pruning_keeps_a_var_the_objective_alone_reaches():
    # a variable the objective reaches and no constraint does is written,
    # so its factor stays
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.c = pyo.Constraint(expr=m.x >= 0)
    m.obj = pyo.Objective(expr=m.y)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.x] = 2.0
    m.scaling_factor[m.y] = 3.0

    _prune_suffixes(m)

    assert m.x in m.scaling_factor
    assert m.y in m.scaling_factor
