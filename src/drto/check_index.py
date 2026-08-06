# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""An index-one check on the declared DAE.

The declarations define a semi-explicit DAE: the declared states and
their dynamics are the differential part, every other equality is the
algebra, and every variable that is not a state member, a derivative,
or held data is algebraic. The model is index one exactly when the
algebra's Jacobian with respect to the algebraic variables is
nonsingular. :func:`check_index` tests that condition at one time
point, in two layers:

- structurally, by a maximum matching of the algebraic constraints to
  the algebraic variables — an unmatched variable is one no algebraic
  constraint determines, and the model is structurally higher-index;
- numerically, by a condition estimate of the Jacobian at the model's
  current values, since the structure can pass while the values fail.

When the structural layer fails, the structural index is computed by
Pantelides' algorithm and reported. The structural index can disagree
with the differentiation index when cancellation hides a dependency;
the report says so whenever it prints one.

The check writes nothing: no components added or removed, no values
changed.
"""
from dataclasses import dataclass, field

import pyomo.environ as pyo
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import identify_variables
from pyomo.dae import ContinuousSet, DerivativeVar

from .info import info

_NAME_CAP = 10  # names listed in reports and errors before "and N more"


@dataclass
class CheckIndexReport:
    """The check's findings; prints readably."""

    verdict: str = ""
    time_point: float = None
    n_algebraic_constraints: int = 0
    n_algebraic_variables: int = 0
    unmatched_variables: list = field(default_factory=list)
    unmatched_constraints: list = field(default_factory=list)
    structural_index: int = None
    condition_estimate: float = None
    condition_limit: float = None
    numerical: str = ""
    notes: list = field(default_factory=list)

    def __str__(self):
        lines = [
            f"drto check_index at t = {self.time_point}",
            f"  algebra: {self.n_algebraic_constraints} constraints, "
            f"{self.n_algebraic_variables} algebraic variables",
        ]
        if self.unmatched_variables or self.unmatched_constraints:
            lines.append(
                "  structural: FAILED - variables no algebraic "
                "constraint determines:"
            )
            lines.extend(f"    {n}" for n in _capped(self.unmatched_variables))
            lines.append("  unmatched constraints:")
            lines.extend(f"    {n}" for n in _capped(self.unmatched_constraints))
            if self.structural_index is not None:
                lines.append(
                    f"  structural index: {self.structural_index}, "
                    "computed from which variables appear in which "
                    "equations (Pantelides' algorithm); a coefficient "
                    "that cancels numerically can hide a dependency the "
                    "pattern shows, so the true index can be higher"
                )
            else:
                lines.append(
                    "  structural index: at least 2 (the pointwise "
                    "algebra cannot determine its variables, and "
                    "differentiation cannot balance an overdetermined "
                    "pointwise system)"
                )
        else:
            lines.append("  structural: full matching")
        if self.numerical:
            lines.append(f"  numerical: {self.numerical}")
        lines.extend(f"  note: {n}" for n in self.notes)
        lines.append(f"  verdict: {self.verdict}")
        return "\n".join(lines)


def _capped(names):
    if len(names) <= _NAME_CAP:
        return names
    return names[:_NAME_CAP] + [f"... and {len(names) - _NAME_CAP} more"]


def _time_position(comp, time):
    """The time set's position among ``comp``'s index factor sets.

    Returns ``(pos, nsets)``: the factor position carrying ``time`` (or
    the position of a ContinuousSet sharing its name, for a clone), and
    the number of factor sets; ``(None, n)`` when no factor is the time
    set. A scalar component reports ``(None, 0)``.
    """
    if not comp.is_indexed():
        return None, 0
    subsets = list(comp.index_set().subsets())
    for i, s in enumerate(subsets):
        if s is time or (isinstance(s, ContinuousSet) and s.name == time.name):
            return i, len(subsets)
    return None, len(subsets)


def _coord(idx, pos, nsets):
    """The time coordinate inside a member index, by factor position."""
    if nsets == 1:
        return idx
    return idx[pos]


def check_index(m, condition_limit=1e10):
    """Check whether the declared model is an index-one DAE.

    Parameters
    ----------
    m : Block
        A declared, discretized model.
    condition_limit : float, optional
        The condition estimate above which the numerical layer fails,
        1e10 by default.

    Returns
    -------
    CheckIndexReport
        The verdict, the counts, and the named members on failure;
        prints readably.

    Raises
    ------
    ValueError
        On missing declarations, a non-positive ``condition_limit``, or
        unequal counts of algebraic constraints and variables (the
        error names the surplus side).
    """
    if condition_limit <= 0:
        raise ValueError(
            "drto: check_index: condition_limit must be positive, got "
            f"{condition_limit}."
        )
    reg = info(m)
    for kind, fn in (
        ("horizon", "drto.horizon"),
        ("state", "drto.state"),
        ("dynamics", "drto.dynamics"),
    ):
        if not reg.has_declaration(kind):
            raise ValueError(
                f"drto: check_index requires a declared {kind} ({fn} first)."
            )
    time = reg.components("horizon")[0]
    points = sorted(time)
    if len(points) < 2:
        raise ValueError(
            "drto: check_index requires a discretized time set with at "
            "least two points."
        )
    t = points[1]  # past the first point, clear of the initial conditions

    # membership sets: what the pointwise algebra excludes
    state_members = set()
    for comp in reg.components("state"):
        state_members.update(id(vd) for vd in comp.values())
    held = set()
    for kind in ("control", "disturbance"):
        if reg.has_declaration(kind):
            for comp in reg.components(kind):
                held.update(id(vd) for vd in comp.values())
    dyn_rows = set()
    for comp in reg.components("dynamics"):
        dyn_rows.update(id(cd) for cd in comp.values())
    ic_rows = set()
    if reg.has_declaration("initial_condition"):
        for comp in reg.components("initial_condition"):
            ic_rows.update(id(cd) for cd in comp.values())

    # the algebraic constraints at t: active equalities that are not
    # declared dynamics, initial conditions, or discretization equations
    rows = []
    n_timefree_rows = 0
    for con in m.component_objects(pyo.Constraint, active=True):
        if con.local_name.endswith(("_disc_eq", "_cont_eq")):
            continue
        pos, nsets = _time_position(con, time)
        if pos is None:
            n_timefree_rows += sum(1 for _ in con.values())
            continue
        for idx, cd in con.items():
            if not cd.active or id(cd) in dyn_rows or id(cd) in ic_rows:
                continue
            if _coord(idx, pos, nsets) != t:
                continue
            if not cd.equality:
                continue
            rows.append(cd)

    # the algebraic variables at t: unfixed members that are not states,
    # derivatives, or held data
    variables = []
    n_timefree_vars = 0
    for var in m.component_objects(pyo.Var, active=True):
        if isinstance(var, DerivativeVar):
            continue
        pos, nsets = _time_position(var, time)
        if pos is None:
            n_timefree_vars += sum(1 for vd in var.values() if not vd.fixed)
            continue
        for idx, vd in var.items():
            if vd.fixed or id(vd) in state_members or id(vd) in held:
                continue
            if _coord(idx, pos, nsets) != t:
                continue
            variables.append(vd)

    report = CheckIndexReport(
        time_point=t,
        n_algebraic_constraints=len(rows),
        n_algebraic_variables=len(variables),
        condition_limit=condition_limit,
    )
    if n_timefree_rows:
        report.notes.append(
            f"{n_timefree_rows} constraints without the time coordinate "
            "are outside the pointwise algebra"
        )
    if n_timefree_vars:
        report.notes.append(
            f"{n_timefree_vars} variables without the time coordinate "
            "are outside the pointwise algebra"
        )

    # ------------------------------------------------------------------
    # structural layer: maximum matching and, on failure, the
    # Dulmage-Mendelsohn partition. A non-square pointwise algebra is
    # itself a structural failure: a semi-explicit index-one DAE
    # determines its algebraic variables pointwise, one constraint each.
    # ------------------------------------------------------------------
    from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

    igi = IncidenceGraphInterface()
    matching = igi.maximum_matching(variables, rows)
    matched = {id(k) for k in matching} | {id(v) for v in matching.values()}
    report.unmatched_constraints = [c.name for c in rows if id(c) not in matched]
    report.unmatched_variables = [v.name for v in variables if id(v) not in matched]
    if len(rows) != len(variables):
        report.notes.append(
            f"the pointwise algebra is not square: {len(rows)} "
            f"constraints for {len(variables)} algebraic variables"
        )
    if report.unmatched_variables or report.unmatched_constraints:
        try:
            var_part, con_part = igi.dulmage_mendelsohn(variables, rows)
            report.notes.append(
                "Dulmage-Mendelsohn: "
                f"{len(con_part.overconstrained) + len(con_part.unmatched)} "
                "constraints in the overconstrained subsystem, "
                f"{len(var_part.underconstrained) + len(var_part.unmatched)} "
                "variables in the underconstrained subsystem"
            )

        except Exception:
            pass

    if report.unmatched_variables or report.unmatched_constraints:
        report.structural_index = _structural_index(m, reg, time, t, rows, variables)
        report.verdict = (
            "not index one: structurally higher index, see the unmatched " "variables"
        )
        return report

    # ------------------------------------------------------------------
    # numerical layer: condition estimate at the current point
    # ------------------------------------------------------------------
    values_missing = any(
        v.value is None for cd in rows for v in identify_variables(cd.body)
    )
    if values_missing:
        report.numerical = "skipped (the model holds no values; initialize first)"
        report.verdict = "index one structurally; numerical layer skipped"
        return report

    if not rows:
        report.numerical = "nothing to evaluate"
        report.verdict = (
            "index zero: the pointwise algebra is empty, the model is an ODE"
        )
        return report

    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    col = {id(v): j for j, v in enumerate(variables)}
    data, ri, ci = [], [], []
    for i, cd in enumerate(rows):
        incident = [v for v in identify_variables(cd.body) if id(v) in col]
        grads = differentiate(cd.body, wrt_list=incident, mode=Modes.reverse_numeric)
        for v, g in zip(incident, grads):
            if g != 0.0:
                data.append(float(g))
                ri.append(i)
                ci.append(col[id(v)])
    n = len(rows)
    jac = sp.coo_matrix((data, (ri, ci)), shape=(n, n)).tocsc()
    # equilibrate before estimating: the index property is independent
    # of units, so each row and then each column is scaled to unit
    # largest entry, and the condition estimate measures the algebra's
    # coupling rather than the spread of the model's units
    absj = abs(jac)
    rmax = absj.max(axis=1).toarray().ravel()
    rmax[rmax == 0.0] = 1.0
    jac = sp.diags(1.0 / rmax) @ jac
    absj = abs(jac.tocsc())
    cmax = absj.max(axis=0).toarray().ravel()
    cmax[cmax == 0.0] = 1.0
    jac = (jac @ sp.diags(1.0 / cmax)).tocsc()
    try:
        lu = spla.splu(jac)
        norm_a = spla.onenormest(jac) if n > 1 else abs(jac.toarray()[0, 0])
        inv_op = spla.LinearOperator(
            (n, n), matvec=lu.solve, rmatvec=lambda x: lu.solve(x, trans="T")
        )
        norm_inv = spla.onenormest(inv_op) if n > 1 else 1.0 / norm_a
        cond = float(norm_a * norm_inv)
        report.condition_estimate = cond
        if cond > condition_limit:
            report.numerical = (
                f"condition estimate {cond:.3e} above the limit "
                f"{condition_limit:.1e}; nearly singular members:"
                + _sick_blocks(igi, rows, variables, jac)
            )
            report.verdict = (
                "not index one at this point: the algebra's Jacobian is "
                "numerically near singular"
            )
            return report
        report.numerical = f"condition estimate {cond:.3e}"
    except RuntimeError:
        report.condition_estimate = float("inf")
        report.numerical = (
            "the factorization found the Jacobian singular; nearly "
            "singular members:" + _sick_blocks(igi, rows, variables, jac)
        )
        report.verdict = (
            "not index one at this point: the algebra's Jacobian is " "singular"
        )
        return report

    report.verdict = "index one"
    return report


def _sick_blocks(igi, rows, variables, jac):
    """Name the members of the near-singular diagonal blocks.

    The block triangularization splits the algebra into square diagonal
    blocks; each is factored densely (they are small in practice) and
    the ones that are singular or worst conditioned are named.
    """
    import numpy as np

    try:
        vblocks, cblocks = igi.block_triangularize(variables, rows)
    except Exception:
        return " (block detail unavailable)"
    worst = []
    col = {id(v): j for j, v in enumerate(variables)}
    row = {id(c): i for i, c in enumerate(rows)}
    dense = jac.toarray()
    for vb, cb in zip(vblocks, cblocks):
        if len(vb) > 200:
            continue
        sub = dense[[row[id(c)] for c in cb]][:, [col[id(v)] for v in vb]]
        s = np.linalg.svd(sub, compute_uv=False)
        blockcond = float("inf") if s[-1] == 0 else float(s[0] / s[-1])
        worst.append((blockcond, [c.name for c in cb], [v.name for v in vb]))
    worst.sort(key=lambda w: -w[0])
    out = []
    for cond, cnames, vnames in worst[:3]:
        out.append(
            f"\n    block condition {cond:.2e}: " + ", ".join(_capped(cnames + vnames))
        )
    return "".join(out) if out else " (block detail unavailable)"


def _structural_index(m, reg, time, t, alg_rows, alg_vars):
    """The structural index by Pantelides' algorithm, on the pointwise
    system.

    The system holds the differential rows (each declared dynamics
    member at ``t``, whose structure is the state derivative against
    the row's other variables) and the algebraic rows. Equations are
    differentiated when they belong to a minimally structurally
    singular subset with respect to the highest-order derivatives; the
    structural index is one more than the deepest differentiation.
    """
    # node numbering: variables are (member id, order); equations are
    # (row id, order). Structure only — supports are sets of ids.
    state_members = set()
    for comp in reg.components("state"):
        state_members.update(id(vd) for vd in comp.values())
    held = set()
    for kind in ("control", "disturbance"):
        if reg.has_declaration(kind):
            for comp in reg.components(kind):
                held.update(id(vd) for vd in comp.values())

    # supports of the pointwise rows, over state and algebraic ids
    alg_ids = {id(v) for v in alg_vars}
    var_order = {}  # id -> current highest derivative order
    eq_support = []  # list of sets of (id, order)
    eq_order = []  # differentiation depth per equation

    def support(cd):
        out = set()
        for v in identify_variables(cd.body):
            if v.fixed or id(v) in held:
                continue
            if isinstance(v.parent_component(), DerivativeVar):
                sv = v.parent_component().get_state_var()
                pos, nsets = _time_position(sv, time)
                idx = v.index()
                base = sv[idx] if pos is not None or nsets == 0 else None
                if base is not None:
                    out.add((id(base), 1))
                continue
            if id(v) in alg_ids or id(v) in state_members:
                out.add((id(v), 0))
        return out

    for comp in reg.components("dynamics"):
        pos, nsets = _time_position(comp, time)
        if pos is None:
            continue
        for idx, cd in comp.items():
            if cd.active and _coord(idx, pos, nsets) == t and cd.equality:
                eq_support.append(support(cd))
                eq_order.append(0)
    for cd in alg_rows:
        eq_support.append(support(cd))
        eq_order.append(0)

    for vid_order in {v for s in eq_support for v in s}:
        vid, order = vid_order
        var_order[vid] = max(var_order.get(vid, 0), order)

    def differentiate_eq(i):
        new = set()
        for vid, order in eq_support[i]:
            new.add((vid, order))
            new.add((vid, order + 1))
            if var_order.get(vid, 0) < order + 1:
                var_order[vid] = order + 1
        eq_support[i] = new
        eq_order[i] += 1

    # Pantelides presumes the differentiated system can reach a perfect
    # matching; with more equations than variables no depth achieves
    # one, so the caller reports the lower bound instead
    n_vars = len({vid for s_ in eq_support for (vid, _o) in s_})
    if len(eq_support) > n_vars:
        return None

    # Pantelides: repeatedly match equations to highest-order
    # derivatives; an augmenting-path failure names a minimally
    # structurally singular subset, which is differentiated.
    max_rounds = 200
    for _ in range(max_rounds):
        highest = {
            (vid, order)
            for s in eq_support
            for (vid, order) in s
            if order == var_order[vid]
        }
        assign = {}  # highest-order var node -> equation index

        def augment(i, seen):
            for node in eq_support[i]:
                vid, order = node
                if order != var_order[vid] or node in seen:
                    continue
                seen.add(node)
                if node not in assign or augment(assign[node], seen):
                    assign[node] = i
                    return True
            return False

        stuck = []
        for i in range(len(eq_support)):
            seen = set()
            if not augment(i, seen):
                stuck.append((i, seen))
        if not stuck:
            return max(eq_order) + 1 if any(eq_order) else 1
        for i, seen in stuck:
            differentiate_eq(i)
            for node in seen:
                differentiate_eq(assign[node]) if node in assign else None
    return None
