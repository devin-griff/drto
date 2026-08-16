# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Scaling factors from the model's own values: ``drto.scale`` (feature 023).

``drto.scale`` fills Pyomo's standard ``scaling_factor`` Suffix from the
values the model currently holds, and ``drto.scaled_solve`` applies them
and solves.

The factors are measured, not declared. A variable's factor comes from
the magnitude its own members hold, and a constraint's from its largest
Jacobian entry once the variables are scaled, so a model carrying its
physics' units needs no hand-written table.

Three properties of the assignment matter more than the arithmetic. The
members of one Var share a factor unless they differ in a string index
element, so time points and spatial nodes of one quantity are scaled
alike while species are scaled apart. A derivative on the terminal
segment takes the factor of the state it differentiates, since it goes
to zero at the equilibrium the tail approaches and the magnitude
measured there is a zero rather than a scale. And the floor sits at machine
zero: an unscaled variable keeps its bound in its own units, and an
interior-point method moves a variable near a bound away from it before
the first step, which on a trace quantity is a displacement of orders of
magnitude.
"""
import math
import warnings

from pyomo.core import Constraint, Objective, Suffix, Var
from pyomo.dae import DerivativeVar

from drto.info import info

#: Magnitudes inside this band are already order one.
_BAND = (1e-2, 1e2)

#: Below this a group holds numerical zeros, not a magnitude.
_FLOOR = 1e-16

#: The largest exponent a factor may carry.
_CLAMP = 12

#: The names pounce registers, both reaching the same solver: the
#: legacy interface and the v2 engine through that same API.
_POUNCE_SOLVERS = ("pounce_v2", "pounce")

#: Solvers that apply the Suffix's factors. ipopt_v2 is here because
#: Pyomo's NL-v2 writer consumes the Suffix and scales the problem as
#: it writes the file, so that solver receives an already-scaled
#: problem and needs no option.
_READS_SUFFIX = _POUNCE_SOLVERS + ("ipopt", "ipopt_v2")

#: The subset that reads the factors inside the solver, under
#: ``nlp_scaling_method=user-scaling``.
_TAKES_OPTION = _POUNCE_SOLVERS + ("ipopt",)


def _suffix_active(m):
    """Whether ``m`` carries an active ``scaling_factor`` Suffix."""
    return any(
        s.local_name == "scaling_factor"
        for s in m.component_objects(Suffix, active=True)
    )


def _scaling_options(solver, fn):
    """The options that make ``solver`` apply the Suffix.

    A solver that does not receive the factors gets an empty mapping
    and a warning naming it, so an unscaled solve is never silent.
    """
    if solver in _TAKES_OPTION:
        return {"nlp_scaling_method": "user-scaling"}
    if solver not in _READS_SUFFIX:
        warnings.warn(
            f"drto: {fn}: solver '{solver}' does not receive the "
            f"scaling_factor Suffix, so the factors were not applied "
            f"and the solve runs unscaled.",
            stacklevel=3,
        )
    return {}


def _group_key(vardata):
    """The Var and the string parts of a member's index.

    Numeric index elements (time points, spatial nodes, ordinal counters)
    carry no scale information, so members differing only in those share a
    factor. String elements name a species or a phase, whose magnitudes
    differ, so they are scaled apart.
    """
    idx = vardata.index()
    if not isinstance(idx, tuple):
        idx = (idx,)
    return (id(vardata.parent_component()), tuple(e for e in idx if isinstance(e, str)))


def _tail_derivative_keys(m, reg):
    """The group each terminal-segment derivative member belongs to.

    A derivative on the terminal segment goes to zero at the equilibrium
    the tail approaches, so the magnitude measured there is a zero rather
    than a scale. Each member takes the group of the state it
    differentiates, keyed the way ``_group_key`` keys that state.

    Discretization reclassifies a ``DerivativeVar`` to ctype ``Var``, so
    the walk is over ``Var`` with an isinstance check.
    """
    copies = {
        id(record["copy"])
        for record in reg._segment_records()
        if record.get("copy") is not None
    }
    keys = {}
    for comp in m.component_objects(Var, active=True, descend_into=True):
        if not isinstance(comp, DerivativeVar):
            continue
        state = comp.get_state_var()
        if id(state) not in copies:
            continue
        for v in comp.values() if comp.is_indexed() else (comp,):
            idx = v.index()
            if not isinstance(idx, tuple):
                idx = (idx,)
            keys[id(v)] = (id(state), tuple(e for e in idx if isinstance(e, str)))
    return keys


def _pin_components(reg):
    """The terminal segment's pin constraints and slacks, by id.

    The pin's penalty weight is stated in each state's own units, so a
    factor on a slack or its constraint would change the pin's weight
    against the objective.
    """
    out = set()
    for record in reg._segment_records("state"):
        for key in ("pin", "pin_up", "pin_lo"):
            comp = record.get(key)
            if comp is None:
                continue
            out.add(id(comp))
            for cd in comp.values() if comp.is_indexed() else (comp,):
                out.add(id(cd))
    return out


def scale(m):
    """Fill ``m``'s ``scaling_factor`` Suffix from its current values.

    Parameters
    ----------
    m : Block
        A model holding values, which an initializer leaves behind.

    Raises
    ------
    ValueError
        If no unfixed variable holds a value, since the factors are
        measured at the point the model is sitting at.
    """
    reg = info(m)
    skip = _pin_components(reg)
    tail = _tail_derivative_keys(m, reg)

    groups = {}
    for v in m.component_data_objects(Var, descend_into=True):
        if v.fixed or id(v) in skip:
            continue
        key = tail[id(v)] if id(v) in tail else _group_key(v)
        groups.setdefault(key, []).append(v)
    if not any(v.value is not None for g in groups.values() for v in g):
        raise ValueError(
            "drto: scale measures the factors at the model's current "
            "point, and no unfixed variable holds a value. Initialize "
            "first (drto.initialize_steady_state or "
            "drto.cold_start_dynamic)."
        )

    if m.component("scaling_factor") is not None:
        m.del_component("scaling_factor")
    m.scaling_factor = Suffix(direction=Suffix.EXPORT)

    for members in groups.values():
        mag = max((abs(v.value) for v in members if v.value is not None), default=0.0)
        if mag < _FLOOR or _BAND[0] <= mag <= _BAND[1]:
            continue
        exponent = max(-_CLAMP, min(_CLAMP, round(math.log10(mag))))
        factor = 10.0**-exponent
        for v in members:
            m.scaling_factor[v] = factor

    _constraint_factors(m, skip)


def _constraint_factors(m, skip):
    """Bring each large constraint's biggest scaled entry to order one.

    A row is only ever scaled down. Scaling one up multiplies its
    residual along with its entries, so a row whose terms cancel to their
    rounding floor would become a violation.
    """
    from pyomo.common.fileutils import find_library

    if find_library("pynumero_ASL") is None:
        raise RuntimeError(
            "drto: scale measures the constraint factors through "
            "PyNumero, whose pynumero_ASL library is not on this "
            "machine's library path. Install it with 'pyomo "
            "download-extensions', or with 'idaes get-extensions' if "
            "IDAES is present (importing idaes then registers its "
            "directory with pyomo)."
        )
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

    # PyomoNLP reads exactly one objective; a square model carries none
    temporary = next(m.component_data_objects(Objective, active=True), None) is None
    if temporary:
        m.add_component("_drto_scale_objective", Objective(expr=0.0))
    try:
        nlp = PyomoNLP(m)
    finally:
        if temporary:
            m.del_component("_drto_scale_objective")

    jac = nlp.evaluate_jacobian_eq().tocsr()
    variables = nlp.get_pyomo_variables()
    constraints = nlp.get_pyomo_equality_constraints()
    factors = [m.scaling_factor.get(v, 1.0) for v in variables]
    for i, con in enumerate(constraints):
        if id(con) in skip or id(con.parent_component()) in skip:
            continue
        row = jac.getrow(i)
        if row.nnz == 0:
            continue
        largest = max(
            abs(a) / factors[j] for j, a in zip(row.indices, row.data) if a != 0.0
        )
        if largest <= _BAND[1]:
            continue
        m.scaling_factor[con] = 10.0 ** -round(math.log10(largest))


def scaled_solve(m, solver="pounce_v2", tee=False, options=None):
    """Assign the factors and solve, returning the model's own units.

    The factors reach the solver through the Suffix, and no second
    model is built. pounce and legacy ipopt read it under
    ``nlp_scaling_method=user-scaling``: objective and constraint
    factors travel through the NL file's suffix segments, and variable
    factors are applied as a change of variables inside the solver.
    ipopt_v2 gets no option, since Pyomo's NL-v2 writer consumes the
    Suffix and scales the problem as it writes the file. Any other
    solver does not receive the factors: the solve runs unscaled and a
    warning says so. Every route solves ``m`` itself and the solution
    comes back in the model's own units.

    Parameters
    ----------
    m : Block
        A model holding values.
    solver : str
        The solver's name.
    tee : bool
        Whether to stream the solver's log.
    options : mapping, optional
        Solver options, overriding the defaults set here.

    Returns
    -------
    SolverResults
        The solver's results object.
    """
    from pyomo.environ import SolverFactory

    if solver in _POUNCE_SOLVERS:
        import pyomo_pounce  # noqa: F401  registers the solver

    scale(m)
    opts = _scaling_options(solver, "scaled_solve")
    opts.update(options or {})
    return SolverFactory(solver).solve(m, tee=tee, options=opts)
