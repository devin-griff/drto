# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The dynamic optimization and simulation declarations (feature 002).

Each declaration function tags a Pyomo component, validates that it is of the
expected type and meets the declaration's convention, and records it in the
model's registry (``drto.info``, feature 001), where the transformations find
it. Declarations are the public surface: drto bolts onto an existing model
rather than replacing how the user builds one.

Every function serves two calling styles. Tagging: handed a component already
attached to the model, it registers immediately, so a finished model is
declared after the fact, interleaved or in one block. Wrapping: handed a
fresh (unconstructed) component, it returns it so it can sit in the
``m.x = ...`` assignment, and validation and registration fire when Pyomo
attaches it. The argument is always the component being declared, attached or
fresh: drto never constructs a component, so an index set where a component
belongs (``state(m.t)``) is a type error. The constraint-role declarations
additionally double as decorators, ``@drto.dynamics(m, m.t)`` taking the
model plus whatever ``@m.Constraint`` would take and building, attaching, and
declaring the constraint in one step. The styles mix freely per component;
the one rule, in every style, is that a declaration's prerequisites must be
declared by the time it registers, which writing the model top-down
satisfies.

Arity: the declarations that scale with the states and controls (``state``,
``control``, ``dynamics``, ``initial_condition``) take varargs when tagging
and accumulate across calls, rejecting duplicates; the wrapping form takes
exactly one component, since it is returned for a single assignment. The
one-of-each declarations (``horizon``, the stage and terminal costs, the
terminal constraint) take exactly one object and error on a second call with
a different one. ``steady_state`` and ``steady_state_control`` take one
(state or control, target Param) pair per call and accumulate.
"""
from pyomo.common.dependencies import attempt_import
from pyomo.core import Constraint
from pyomo.core.base.block import BlockData
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


def _is_block(obj):
    """Return whether ``obj`` is a block (the decorator form's first arg)."""
    return isinstance(obj, BlockData)


def _single(args, fn):
    """Return the one component of a one-of-each declaration call."""
    if len(args) != 1:
        raise TypeError(f"drto: {fn} takes exactly one component; got {len(args)}.")
    return args[0]


def _declared_in(component, components):
    """Identity membership over a component tuple.

    Pyomo components overload ``==`` (a scalar Var's builds a relational
    expression, and ``bool()`` on it raises), so ``in`` is never safe here.
    """
    return any(c is component for c in components)


def _no_kwargs(kwargs, fn):
    """Reject keyword arguments outside the decorator form."""
    if kwargs:
        raise TypeError(
            f"drto: {fn} got unexpected keyword arguments {sorted(kwargs)}: "
            f"keywords pass through to Constraint in the decorator form only."
        )


def _wrap_form(components, fn):
    """Return whether the call is the wrapping form: one fresh component."""
    if all(comp.is_constructed() for comp in components):
        return False
    if len(components) != 1:
        raise TypeError(
            f"drto: {fn}: the wrapping form takes exactly one component "
            f"(it is returned for a single assignment); varargs are for "
            f"tagging attached components."
        )
    return True


def _defer(component, register, fn):
    """Wrap a fresh component: run ``register`` when Pyomo attaches it.

    Attachment (``m.x = component``) constructs the component, so the hook
    shadows ``construct``, runs the original, removes itself, and registers,
    at which point the component has its model and name.
    """
    if "construct" in component.__dict__:
        raise ValueError(
            f"drto: {fn}: "
            f"'{component.name or type(component).__name__}' is already "
            f"wrapped by a declaration."
        )
    original = component.construct

    def construct(data=None):
        model = component.model()
        if type(model).__name__ == "AbstractModel":
            # attachment to an AbstractModel does not construct, so the hook
            # survives into create_instance's clone, where it would construct
            # and register the original component instead of the instance's
            raise ValueError(
                f"drto: {fn}: wrapping registers at attachment to a concrete "
                f"model, and "
                f"'{component.name or type(component).__name__}' belongs to "
                f"an AbstractModel. Declare by tagging on the instance after "
                f"create_instance()."
            )
        original(data)
        del component.construct
        register()

    component.construct = construct
    return component


def _constraint_decorator(block, sets, register_one, kwargs=None):
    """The construction form of a constraint-role declaration.

    ``@drto.<fn>(m, *sets, **kwargs)`` builds the Constraint from the
    decorated rule, exactly as ``@m.Constraint(*sets, **kwargs)`` would,
    attaches it under the rule's name, declares it, and returns the
    component.
    """

    def decorate(rule):
        con = Constraint(*sets, rule=rule, **(kwargs or {}))
        setattr(block, rule.__name__, con)
        register_one(con)
        return con

    return decorate


def _declare_single(kind, component, fn, **metadata):
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
    reg.record_declaration(kind, component, **metadata)
    return reg


def _declare_many(kind, components, fn, **metadata):
    """Record an accumulating declaration, rejecting duplicates."""
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    model = _container(components[0], fn).model()
    reg = info(model)
    for comp in components:
        _container(comp, fn)
        if comp.model() is not model:
            raise ValueError(
                f"drto: {fn}: '{comp.name}' is on a different model than "
                f"'{components[0].name}'; declare each model separately."
            )
        if _declared_in(comp, reg.components(kind)):
            raise ValueError(
                f"drto: '{comp.name}' is already declared as a "
                f"{kind.replace('_', ' ')}."
            )
    for comp in components:
        reg.record_declaration(kind, comp, **metadata)
    return reg


def _declared_horizon(reg, fn):
    """Return the declared time set, erroring clearly if there is none."""
    time_sets = reg.components("horizon")
    if not time_sets:
        raise ValueError(f"drto: {fn} requires the horizon (drto.horizon) first.")
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


def _time_coord(vardata, time):
    """Return the time coordinate of a Var member, or None.

    Handles states indexed by time alone (the index is the coordinate) and
    by time plus other sets (the coordinate sits inside the index tuple at
    the time set's position).
    """
    subs = list(vardata.parent_component().index_set().subsets())
    for n, s in enumerate(subs):
        if s is time:
            idx = vardata.index()
            return idx if len(subs) == 1 else tuple(idx)[n]
    return None


def _members(con):
    """Yield the ConstraintData members of a scalar or indexed Constraint."""
    if con.is_indexed():
        yield from con.values()
    else:
        yield con


# ----------------------------------------------------------------------
# the declaration surface
# ----------------------------------------------------------------------
def horizon(component):
    """Declare the horizon time set, a ``pyomo.dae`` ContinuousSet.

    The root handle for the moving-horizon machinery. The set is initialized
    with the sample grid (the sampling instants), and declaring it captures
    that grid in the registry: the samples define the stage-cost sum (feature
    003) and the sampling time. Exactly one per model, declared before the
    set is discretized. Tags an attached set or wraps a fresh one.
    """
    fn = "horizon"
    _container(component, fn)
    if not isinstance(component, ContinuousSet):
        raise TypeError(
            f"drto: horizon expects a pyomo.dae ContinuousSet, got "
            f"{type(component).__name__} '{component.name}'."
        )

    def register():
        if component.get_discretization_info():
            raise ValueError(
                f"drto: horizon must be called before '{component.name}' is "
                f"discretized: the set's points are captured as the sample grid."
            )
        # a constructed ContinuousSet always holds at least two points
        # (Pyomo enforces it), so the grid is the set's points as written
        samples = tuple(sorted(component))
        _declare_single("horizon", component, fn, samples=samples)

    if not component.is_constructed():
        return _defer(component, register, fn)
    register()
    return component


def state(*components):
    """Declare one or more state Vars.

    A state carries a DerivativeVar only in a dynamic model, so no derivative
    is required here: a steady-state model's states qualify as written. Tags
    attached Vars or wraps one fresh Var.
    """
    fn = "state"
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Var", fn)
    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: _declare_many("state", (comp,), fn), fn)
    _declare_many("state", components, fn)
    return components[0] if len(components) == 1 else None


def dynamics(*components, **kwargs):
    """Declare one or more dynamics equality Constraints.

    Currently continuous-time: one side of each member is the DerivativeVar
    of a declared state, taken with respect to the declared time set.
    Requires ``horizon`` and ``state`` first. Tags attached Constraints,
    wraps a fresh one, or builds one as a decorator:
    ``@drto.dynamics(m, m.t)``.
    """
    fn = "dynamics"
    if components and _is_block(components[0]):
        block, sets = components[0], components[1:]
        return _constraint_decorator(
            block, sets, lambda c: _register_dynamics((c,)), kwargs
        )
    _no_kwargs(kwargs, fn)
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Constraint", fn)
    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: _register_dynamics((comp,)), fn)
    _register_dynamics(components)
    return components[0] if len(components) == 1 else None


def _register_dynamics(components):
    """Validate and record dynamics Constraints (attached and constructed)."""
    fn = "dynamics"
    reg = info(components[0].model())
    time = _declared_horizon(reg, fn)
    states = reg.components("state")
    if not states:
        raise ValueError(f"drto: {fn} requires drto.state first.")
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
            if not _declared_in(state, states):
                raise ValueError(
                    f"drto: {fn}: '{cd.name}' differentiates "
                    f"'{state.name}', which is not a declared state."
                )
            if not _declared_in(time, dv.get_continuousset_list()):
                raise ValueError(
                    f"drto: {fn}: '{dv.name}' is not differentiated with "
                    f"respect to the declared time set '{time.name}'."
                )
    _declare_many("dynamics", components, fn)


def control(*components, profile="piecewise_constant"):
    """Declare one or more manipulated-input Vars and their profile.

    The ``profile`` (a pyomo-cvp profile) parameterizes the named controls
    over the declared time set; it applies to the controls in this call, so a
    control needing a different parameterization is declared separately.
    Requires ``horizon`` first and pyomo-cvp installed. Tags attached Vars or
    wraps one fresh Var.
    """
    fn = "control"
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Var", fn)
    if not pyomo_cvp_available:
        raise RuntimeError(
            "drto: control requires pyomo-cvp for the control "
            "profile (pip install pyomo-cvp)."
        )

    def register(comps):
        reg = info(comps[0].model())
        time = _declared_horizon(reg, fn)
        _declare_many("control", comps, fn, profile=profile)
        pyomo_cvp.declare_profile(*comps, wrt=time, profile=profile)

    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: register((comp,)), fn)
    register(components)
    return components[0] if len(components) == 1 else None


def _register_stage_cost(kind, component, fn):
    """Validate and record a stage cost (attached and constructed)."""
    reg = info(component.model())
    time = _declared_horizon(reg, fn)
    if any(s is time for s in component.index_set().subsets()):
        raise ValueError(
            f"drto: {fn}: '{component.name}' is indexed by the time set "
            f"'{time.name}': discretization expands such a family to every "
            f"collocation point, dragging the cost off the sample grid. "
            f"Index it over the samples, for example "
            f"@m.Constraint(sorted(m.t)[:-1])."
        )
    samples = reg.declarations("horizon")[0]["samples"]
    expected = list(samples[:-1])
    members = sorted(component.keys()) if component.is_indexed() else []
    if members != expected:
        raise ValueError(
            f"drto: {fn}: '{component.name}' must have one member per sample "
            f"point except the final one, where only the terminal cost "
            f"applies: index it over the samples, for example "
            f"@m.Constraint(sorted(m.t)[:-1])."
        )
    for cd in _members(component):
        _side_matching(
            cd, _is_var_member, fn, "the scalar cost variable (the cost term)"
        )
    _declare_single(kind, component, fn)


def _declare_stage_cost(kind, args, fn, kwargs):
    """Dispatch a stage-cost declaration across the three calling styles."""
    if args and _is_block(args[0]):
        block, sets = args[0], args[1:]
        return _constraint_decorator(
            block, sets, lambda c: _register_stage_cost(kind, c, fn), kwargs
        )
    _no_kwargs(kwargs, fn)
    component = _single(args, fn)
    _container(component, fn)
    _check_ctype(component, "Constraint", fn)
    if not component.is_constructed():
        return _defer(component, lambda: _register_stage_cost(kind, component, fn), fn)
    _register_stage_cost(kind, component, fn)
    return component


def tracking_stage_cost(*args, **kwargs):
    """Declare the tracking stage cost, a per-time-point equality.

    One side of each member is the scalar running-cost variable; the other
    defines the cost. One per model, indexed over the samples minus the
    final time (the terminal cost owns it). Tags, wraps, or builds as a
    decorator: ``@drto.tracking_stage_cost(m, sorted(m.t)[:-1])``.
    """
    return _declare_stage_cost(
        "tracking_stage_cost", args, "tracking_stage_cost", kwargs
    )


def economic_stage_cost(*args, **kwargs):
    """Declare the economic stage cost, a per-time-point equality.

    One side of each member is the scalar running-cost variable; the other
    defines the cost. One per model. Tags, wraps, or builds as a decorator.
    """
    return _declare_stage_cost(
        "economic_stage_cost", args, "economic_stage_cost", kwargs
    )


def tracking_terminal_cost(*args, **kwargs):
    """Declare the terminal tracking cost, a scalar equality.

    One side is the scalar terminal-cost variable; the other defines the
    cost. One per model. Tags, wraps, or builds as a decorator:
    ``@drto.tracking_terminal_cost(m)``.
    """
    fn = "tracking_terminal_cost"

    def register(component):
        if component.is_indexed():
            raise ValueError(
                f"drto: {fn}: '{component.name}' must be a scalar Constraint "
                f"(the terminal cost applies at the final time only)."
            )
        _side_matching(
            component, _is_var_member, fn, "the scalar terminal-cost variable"
        )
        _declare_single("tracking_terminal_cost", component, fn)

    if args and _is_block(args[0]):
        return _constraint_decorator(args[0], args[1:], register, kwargs)
    _no_kwargs(kwargs, fn)
    component = _single(args, fn)
    _container(component, fn)
    _check_ctype(component, "Constraint", fn)
    if not component.is_constructed():
        return _defer(component, lambda: register(component), fn)
    register(component)
    return component


def initial_condition(*components, **kwargs):
    """Declare one or more initial-condition equality Constraints.

    One side of each is a declared state at the first time point; the other
    is a mutable Param, the state feedback hook the loop writes measurements
    into. Tags attached Constraints, wraps a fresh one, or builds one as a
    decorator: ``@drto.initial_condition(m)``.
    """
    fn = "initial_condition"
    if components and _is_block(components[0]):
        block, sets = components[0], components[1:]
        return _constraint_decorator(
            block, sets, lambda c: _register_initial_condition((c,)), kwargs
        )
    _no_kwargs(kwargs, fn)
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Constraint", fn)
    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: _register_initial_condition((comp,)), fn)
    _register_initial_condition(components)
    return components[0] if len(components) == 1 else None


def _register_initial_condition(components):
    """Validate and record initial conditions (attached and constructed)."""
    fn = "initial_condition"
    reg = info(components[0].model())
    time = _declared_horizon(reg, fn)
    states = reg.components("state")
    t0 = time.first()
    for comp in components:
        for cd in _members(comp):
            state_side, param_side = _side_matching(
                cd,
                lambda s: _is_var_member(s)
                and _declared_in(s.parent_component(), states),
                fn,
                "a declared state",
            )
            if _time_coord(state_side, time) != t0:
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


def terminal_constraint(*args, **kwargs):
    """Declare the terminal constraint, referencing only final-time states.

    A single Constraint whose variables are all declared states at the final
    time point, which is what separates it from a path constraint. Tags,
    wraps, or builds as a decorator: ``@drto.terminal_constraint(m)``.
    """
    fn = "terminal_constraint"

    def register(component):
        reg = info(component.model())
        time = _declared_horizon(reg, fn)
        states = reg.components("state")
        tN = time.last()
        for cd in _members(component):
            for v in identify_variables(cd.body, include_fixed=True):
                if (
                    not _declared_in(v.parent_component(), states)
                    or _time_coord(v, time) != tN
                ):
                    raise ValueError(
                        f"drto: {fn}: '{cd.name}' references '{v.name}'; a "
                        f"terminal constraint may reference only declared states "
                        f"at the final time point ({tN})."
                    )
        _declare_single("terminal_constraint", component, fn)

    if args and _is_block(args[0]):
        return _constraint_decorator(args[0], args[1:], register, kwargs)
    _no_kwargs(kwargs, fn)
    component = _single(args, fn)
    _container(component, fn)
    _check_ctype(component, "Constraint", fn)
    if not component.is_constructed():
        return _defer(component, lambda: register(component), fn)
    register(component)
    return component


def _declare_target(kind, owner, target, fn, owner_kind):
    """Pair a declared state or control with its setpoint target Param."""
    _container(owner, fn)
    _check_ctype(owner, "Var", fn)
    if not owner.is_constructed():
        raise ValueError(
            f"drto: {fn}: declare the {owner_kind} first; "
            f"'{owner.name}' is not attached to a model yet."
        )
    reg = info(owner.model())
    if not _declared_in(owner, reg.components(owner_kind)):
        raise ValueError(
            f"drto: {fn}: '{owner.name}' is not a declared {owner_kind}; "
            f"drto.{owner_kind} first."
        )
    _container(target, fn)
    _check_ctype(target, "Param", fn)
    if not target.mutable:
        raise ValueError(
            f"drto: {fn}: Param '{target.name}' must be mutable so the "
            f"steady-state solve can populate it."
        )

    def register():
        if target.model() is not owner.model():
            raise ValueError(
                f"drto: {fn}: target '{target.name}' is on a different "
                f"model than '{owner.name}'."
            )
        # a target Param serves exactly one owner, in either target kind
        for a_kind in ("steady_state", "steady_state_control"):
            for rec in reg.declarations(a_kind):
                if a_kind == kind and rec["of"] is owner:
                    if rec["component"] is target:
                        return  # idempotent re-declaration of the same pair
                    raise ValueError(
                        f"drto: {fn}: '{owner.name}' already has the target "
                        f"'{rec['component'].name}'."
                    )
                if rec["component"] is target:
                    raise ValueError(
                        f"drto: '{target.name}' is already declared as a "
                        f"{a_kind.replace('_', ' ')} target, of "
                        f"'{rec['of'].name}'."
                    )
        reg.record_declaration(kind, target, of=owner)

    if not target.is_constructed():
        return _defer(target, register, fn)
    register()
    return target


def steady_state(owner, target):
    """Pair a declared state with the mutable Param holding its setpoint.

    The target the tracking costs drive toward, populated by the
    steady-state/RTO solve (feature 009), which is why the pairing is
    recorded: drto writes each solved state value into its target. One pair
    per call; returns the target, so a fresh Param wraps:
    ``m.z_ss = drto.steady_state(m.z, pyo.Param(initialize=0.5, mutable=True))``.
    """
    return _declare_target("steady_state", owner, target, "steady_state", "state")


def steady_state_control(owner, target):
    """Pair a declared control with the mutable Param holding its setpoint.

    The control target the tracking costs drive toward. One pair per call;
    returns the target, so a fresh Param wraps.
    """
    return _declare_target(
        "steady_state_control", owner, target, "steady_state_control", "control"
    )
