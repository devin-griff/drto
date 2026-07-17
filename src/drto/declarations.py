# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The dynamic optimization and simulation declarations (feature 002).

Each ``declare_*`` function tags a Pyomo component the user already wrote,
validates that it is of the expected type and meets the declaration's
convention, and records it in the model's registry (``drto.info``, feature
001), where the transformations find it. Declarations are the public surface:
drto bolts onto an existing model rather than replacing how the user builds
one.

Arity follows the spec: the declarations that scale with the states and
controls (``declare_state``, ``declare_control``,
``declare_continuous_dynamics``, ``declare_initial_condition``,
``declare_steady_state``, ``declare_steady_state_control``) take varargs and
accumulate across calls, rejecting duplicates. The one-of-each declarations
(``declare_time``, the stage and terminal costs, the terminal constraint)
take exactly one object and error on a second call with a different one.
"""
from pyomo.common.dependencies import attempt_import
from pyomo.core.expr import identify_variables
from pyomo.core.expr.relational_expr import EqualityExpression
from pyomo.dae import ContinuousSet, DerivativeVar

from drto.info import info

pyomo_cvp, pyomo_cvp_available = attempt_import("pyomo_cvp")


# ----------------------------------------------------------------------
# shared plumbing
# ----------------------------------------------------------------------
def _container(component, fn):
    """Return ``component`` validated as a component container.

    Declarations tag whole components (one declaration per container), not
    individual members, so a ``ComponentData`` argument errors.
    """
    if getattr(component, "parent_component", None) is None:
        raise TypeError(
            f"drto: {fn} expects a Pyomo component, got " f"{type(component).__name__}."
        )
    if component.parent_component() is not component:
        raise TypeError(
            f"drto: {fn} declares whole components (one declaration per "
            f"container); got the member '{component.name}'. Declare "
            f"'{component.parent_component().name}' instead."
        )
    return component


def _check_ctype(component, ctype_name, fn):
    """Validate ``component``'s ctype by name, with a clear error."""
    actual = getattr(component.ctype, "__name__", type(component).__name__)
    if actual != ctype_name:
        raise TypeError(
            f"drto: {fn} expects a {ctype_name}, got {actual} " f"'{component.name}'."
        )


def _declare_single(kind, component, fn):
    """Record a one-of-each declaration, enforcing the re-declaration rule."""
    reg = info(component.model())
    existing = reg.components(kind)
    if existing:
        if existing[0] is component:
            return reg  # idempotent re-declaration of the same object
        raise ValueError(
            f"drto: {fn} was already called with '{existing[0].name}'; the "
            f"model has one {kind.replace('_', ' ')}. Got '{component.name}'."
        )
    reg.record_declaration(kind, component)
    return reg


def _declare_many(kind, components, fn, **metadata):
    """Record an accumulating declaration, rejecting duplicates."""
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    reg = info(_container(components[0], fn).model())
    for comp in components:
        _container(comp, fn)
        if comp in reg.components(kind):
            raise ValueError(
                f"drto: '{comp.name}' is already declared as a "
                f"{kind.replace('_', ' ')}."
            )
    for comp in components:
        reg.record_declaration(kind, comp, **metadata)
    return reg


def _declared_time(reg, fn):
    """Return the declared time set, erroring clearly if there is none."""
    time_sets = reg.components("time")
    if not time_sets:
        raise ValueError(f"drto: {fn} requires declare_time first.")
    return time_sets[0]


def _equality_sides(condata, fn):
    """Return the two sides of an equality constraint member.

    The conventions are read from the written equality's sides, either
    orientation, so ``lhs == rhs`` and ``rhs == lhs`` are equivalent.
    """
    if not condata.equality:
        raise ValueError(
            f"drto: {fn}: '{condata.name}' must be an equality constraint."
        )
    expr = condata.expr
    if not isinstance(expr, EqualityExpression):
        raise ValueError(
            f"drto: {fn}: write '{condata.name}' as an explicit equality "
            f"(lhs == rhs)."
        )
    return expr.args[0], expr.args[1]


def _is_var_member(node):
    """Return whether ``node`` is a single Var member (a VarData)."""
    return getattr(node, "is_variable_type", lambda: False)()


def _side_matching(condata, predicate, fn, expected):
    """Return the side of an equality that satisfies ``predicate``.

    Checks the written left side first, then the right, so the convention
    holds regardless of how the user oriented the equality.
    """
    lhs, rhs = _equality_sides(condata, fn)
    for side, other in ((lhs, rhs), (rhs, lhs)):
        if predicate(side):
            return side, other
    raise ValueError(f"drto: {fn}: neither side of '{condata.name}' is {expected}.")


def _members(con):
    """Yield the ConstraintData members of a scalar or indexed Constraint."""
    if con.is_indexed():
        yield from con.values()
    else:
        yield con


# ----------------------------------------------------------------------
# the declaration surface
# ----------------------------------------------------------------------
def declare_time(component):
    """Declare the horizon time set, a ``pyomo.dae`` ContinuousSet.

    The root handle for the moving-horizon machinery: the horizon's start and
    end come off its bounds, and the dynamics are validated against it.
    Exactly one per model.
    """
    _container(component, "declare_time")
    if not isinstance(component, ContinuousSet):
        raise TypeError(
            f"drto: declare_time expects a pyomo.dae ContinuousSet, got "
            f"{type(component).__name__} '{component.name}'."
        )
    _declare_single("time", component, "declare_time")


def declare_state(*components):
    """Declare one or more state Vars.

    A state carries a DerivativeVar only in a dynamic model, so no derivative
    is required here: a steady-state model's states qualify as written.
    """
    for comp in components:
        _container(comp, "declare_state")
        _check_ctype(comp, "Var", "declare_state")
    _declare_many("state", components, "declare_state")


def declare_continuous_dynamics(*components):
    """Declare one or more continuous-dynamics equality Constraints.

    One side of each member is the DerivativeVar of a declared state, taken
    with respect to the declared time set. Requires ``declare_time`` and
    ``declare_state`` first.
    """
    fn = "declare_continuous_dynamics"
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Constraint", fn)
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    reg = info(components[0].model())
    time = _declared_time(reg, fn)
    states = reg.components("state")
    if not states:
        raise ValueError(f"drto: {fn} requires declare_state first.")
    for comp in components:
        for cd in _members(comp):
            deriv, _ = _side_matching(
                cd,
                lambda s: isinstance(
                    getattr(s, "parent_component", lambda: None)(), DerivativeVar
                ),
                fn,
                "a DerivativeVar (dz/dt)",
            )
            dv = deriv.parent_component()
            state = dv.get_state_var()
            if state not in states:
                raise ValueError(
                    f"drto: {fn}: '{cd.name}' differentiates "
                    f"'{state.name}', which is not a declared state."
                )
            if time not in dv.get_continuousset_list():
                raise ValueError(
                    f"drto: {fn}: '{dv.name}' is not differentiated with "
                    f"respect to the declared time set '{time.name}'."
                )
    _declare_many("continuous_dynamics", components, fn)


def declare_control(*components, profile="piecewise_constant"):
    """Declare one or more manipulated-input Vars and their profile.

    The ``profile`` (a pyomo-cvp profile) parameterizes the named controls
    over the declared time set; it applies to the controls in this call, so a
    control needing a different parameterization is declared separately.
    Requires ``declare_time`` first and pyomo-cvp installed.
    """
    fn = "declare_control"
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Var", fn)
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    if not pyomo_cvp_available:
        raise RuntimeError(
            "drto: declare_control requires pyomo-cvp for the control "
            "profile (pip install pyomo-cvp)."
        )
    reg = info(components[0].model())
    time = _declared_time(reg, fn)
    _declare_many("control", components, fn, profile=profile)
    pyomo_cvp.declare_profile(*components, wrt=time, profile=profile)


def _declare_stage_cost(kind, component, fn):
    """Shared validation for the per-time-point stage-cost declarations."""
    _container(component, fn)
    _check_ctype(component, "Constraint", fn)
    reg = info(component.model())
    time = _declared_time(reg, fn)
    if component.index_set() is not time:
        raise ValueError(
            f"drto: {fn}: '{component.name}' must be indexed by the declared "
            f"time set '{time.name}' (a per-time-point cost)."
        )
    for cd in _members(component):
        _side_matching(
            cd, _is_var_member, fn, "the scalar cost variable (the cost term)"
        )
    _declare_single(kind, component, fn)


def declare_tracking_stage_cost(component):
    """Declare the tracking stage cost, a per-time-point equality.

    One side of each member is the scalar running-cost variable; the other
    defines the cost. One per model.
    """
    _declare_stage_cost("tracking_stage_cost", component, "declare_tracking_stage_cost")


def declare_economic_stage_cost(component):
    """Declare the economic stage cost, a per-time-point equality.

    One side of each member is the scalar running-cost variable; the other
    defines the cost. One per model.
    """
    _declare_stage_cost("economic_stage_cost", component, "declare_economic_stage_cost")


def declare_tracking_terminal_cost(component):
    """Declare the terminal tracking cost, a scalar equality.

    One side is the scalar terminal-cost variable; the other defines the
    cost. One per model.
    """
    fn = "declare_tracking_terminal_cost"
    _container(component, fn)
    _check_ctype(component, "Constraint", fn)
    if component.is_indexed():
        raise ValueError(
            f"drto: {fn}: '{component.name}' must be a scalar Constraint "
            f"(the terminal cost applies at the final time only)."
        )
    _side_matching(component, _is_var_member, fn, "the scalar terminal-cost variable")
    _declare_single("tracking_terminal_cost", component, fn)


def declare_initial_condition(*components):
    """Declare one or more initial-condition equality Constraints.

    One side of each is a declared state at the first time point; the other
    is a mutable Param, the state feedback hook the loop writes measurements
    into.
    """
    fn = "declare_initial_condition"
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Constraint", fn)
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    reg = info(components[0].model())
    time = _declared_time(reg, fn)
    states = reg.components("state")
    t0 = time.first()
    for comp in components:
        for cd in _members(comp):
            state_side, param_side = _side_matching(
                cd,
                lambda s: _is_var_member(s) and s.parent_component() in states,
                fn,
                "a declared state",
            )
            if state_side.index() != t0:
                raise ValueError(
                    f"drto: {fn}: '{cd.name}' anchors "
                    f"'{state_side.name}', which is not at the first time "
                    f"point ({t0})."
                )
            param = getattr(param_side, "parent_component", lambda: None)()
            if param is None or param.ctype.__name__ != "Param":
                raise ValueError(
                    f"drto: {fn}: the other side of '{cd.name}' must be a "
                    f"mutable Param, the state feedback hook."
                )
            if not param.mutable:
                raise ValueError(
                    f"drto: {fn}: Param '{param.name}' must be mutable so "
                    f"the loop can write measurements into it."
                )
    _declare_many("initial_condition", components, fn)


def declare_terminal_constraint(component):
    """Declare the terminal constraint, referencing only final-time states.

    A single Constraint whose variables are all declared states at the final
    time point, which is what separates it from a path constraint.
    """
    fn = "declare_terminal_constraint"
    _container(component, fn)
    _check_ctype(component, "Constraint", fn)
    reg = info(component.model())
    time = _declared_time(reg, fn)
    states = reg.components("state")
    tN = time.last()
    for cd in _members(component):
        for v in identify_variables(cd.body, include_fixed=True):
            if v.parent_component() not in states or v.index() != tN:
                raise ValueError(
                    f"drto: {fn}: '{cd.name}' references '{v.name}'; a "
                    f"terminal constraint may reference only declared states "
                    f"at the final time point ({tN})."
                )
    _declare_single("terminal_constraint", component, fn)


def _declare_targets(kind, components, fn):
    """Shared validation for the steady-state target declarations."""
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Param", fn)
        if not comp.mutable:
            raise ValueError(
                f"drto: {fn}: Param '{comp.name}' must be mutable so the "
                f"steady-state solve can populate it."
            )
    _declare_many(kind, components, fn)


def declare_steady_state(*components):
    """Declare one or more mutable Params holding the state setpoints.

    The targets the tracking costs drive toward, populated by the
    steady-state/RTO solve.
    """
    _declare_targets("steady_state", components, "declare_steady_state")


def declare_steady_state_control(*components):
    """Declare one or more mutable Params holding the control setpoints.

    The control targets the tracking costs drive toward.
    """
    _declare_targets("steady_state_control", components, "declare_steady_state_control")
