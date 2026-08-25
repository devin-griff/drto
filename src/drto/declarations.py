# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The dynamic optimization, simulation, and estimation declarations
(features 002 and 018).

Each declaration function tags a Pyomo component, validates that it is of the
expected type and meets the declaration's convention, and records it in the
model's registry (``drto.info``, feature 001), where the transformations find
it. Declarations are the public surface. Each one tags a component the
user already wrote, so the model is built as an ordinary Pyomo model and
declared afterwards or as it is written.

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
``control``, ``dynamics``, ``initial_condition``, and the estimation-side
``estimated_parameter``, ``disturbance``, ``measurement``) take varargs when
tagging and accumulate across calls, rejecting duplicates; the wrapping form
takes exactly one component, since it is returned for a single assignment. The
one-of-each declarations (``horizon``, the stage, terminal, and arrival costs,
the terminal constraint) take exactly one object and error on a second call
with a different one. ``steady_state`` and ``steady_state_control`` take one
(state or control, target Param) pair per call and accumulate.
"""
import re as _re

from pyomo.common.dependencies import attempt_import
from pyomo.core import Constraint, Reference
from pyomo.core.base.block import BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.expr import identify_variables
from pyomo.core.expr.numeric_expr import MonomialTermExpression, ProductExpression
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

    A component handed to a declaration before attachment has no model
    and no name yet, so validation and recording cannot run at the call.
    The assignment ``m.x = component`` is what constructs it, so this
    wrapper shadows ``construct``: when the assignment runs, it calls the
    original ``construct``, removes itself, and then registers, at which
    point the component has its model and name.
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
            # attachment to an AbstractModel does not construct, so the
            # wrapper would still be waiting when create_instance clones
            # the model, and would then construct and register the
            # original component instead of the instance's
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
    # every caller validates each component with _container first, so this
    # takes the model from the first component and checks the rest against it
    model = components[0].model()
    reg = info(model)
    for comp in components:
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


def _mult_factors(node):
    """Yield the multiplicative factors of a product expression tree."""
    if isinstance(node, (ProductExpression, MonomialTermExpression)):
        for a in node.args:
            yield from _mult_factors(a)
    else:
        yield node


def _dynamics_sides(condata, time, fn):
    """Return ``(derivative, coefficient, other)`` for a dynamics equality.

    One written side must be a DerivativeVar member, bare or multiplied by
    derivative-free factors (an IDAES ``ControlVolume1D`` writes
    ``length * accumulation``; a variable-volume balance writes
    ``V * dc/dt``); ``coefficient`` is the product of those factors, None
    for the bare side. A side differentiated with respect to ``time`` wins
    over one differentiated along another axis, so a 1D balance carrying
    the space derivative on its other side reads correctly in either
    orientation; with no time-side at all the first structural match
    returns, for the caller's wrt check to reject descriptively.
    """
    lhs, rhs = _equality_sides(condata, fn)
    fallback = None
    for side, other in ((lhs, rhs), (rhs, lhs)):
        factors = list(_mult_factors(side))
        derivs = [
            f
            for f in factors
            if _is_var_member(f) and isinstance(f.parent_component(), DerivativeVar)
        ]
        if len(derivs) != 1:
            continue
        rest = [f for f in factors if f is not derivs[0]]
        if any(
            isinstance(v.parent_component(), DerivativeVar)
            for f in rest
            for v in identify_variables(f, include_fixed=True)
        ):
            continue
        coeff = None
        for f in rest:
            coeff = f if coeff is None else coeff * f
        candidate = (derivs[0], coeff, other)
        sets = derivs[0].parent_component().get_continuousset_list()
        if _declared_in(time, sets):
            return candidate
        if fallback is None:
            fallback = candidate
    if fallback is not None:
        return fallback
    raise ValueError(
        f"drto: {fn}: neither side of '{condata.name}' is a DerivativeVar "
        f"(dz/dt), bare or multiplied by derivative-free factors."
    )


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


def _attach_slice_reference(sl, fn):
    """Wrap a member-subset slice as an attached, named Reference.

    An indexed Var (an IDAES holdup) can hold members that are not states;
    a slice like ``holdup[:, "Liq", "NaOH"]`` declares the true state
    member (gh #20). The Reference attaches to the sliced component's
    parent block, named from the component and the constant coordinates.
    """
    ref = Reference(sl)
    vds = list(ref.values())
    root = vds[0].parent_component()
    idx0 = vds[0].index()
    idx0 = idx0 if isinstance(idx0, tuple) else (idx0,)
    if len(vds) > 1:
        idx1 = vds[1].index()
        idx1 = idx1 if isinstance(idx1, tuple) else (idx1,)
        const = [a for a, b in zip(idx0, idx1) if a == b]
    else:
        const = list(idx0)
    parent = root.parent_block()
    name = _re.sub(r"\W+", "_", "_".join([root.local_name] + [str(c) for c in const]))
    if parent.component(name) is not None:
        raise ValueError(
            f"drto: {fn}: cannot wrap the slice as '{name}': the component "
            f"already exists on '{parent.name}'."
        )
    parent.add_component(name, ref)
    return ref


def state(*components):
    """Declare one or more state Vars.

    A state carries a DerivativeVar only in a dynamic model, so no derivative
    is required here: a steady-state model's states qualify as written. Tags
    attached Vars, wraps one fresh Var, or wraps member-subset slices
    (``holdup[:, "Liq", "NaOH"]``) as attached References, so an indexed
    Var's algebraic member stays undeclared.
    """
    fn = "state"
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    components = tuple(
        _attach_slice_reference(c, fn) if isinstance(c, IndexedComponent_slice) else c
        for c in components
    )
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
    of a declared state, taken with respect to the declared time set, bare
    or multiplied by derivative-free factors (an IDAES ``ControlVolume1D``
    writes ``length * accumulation``; a variable-volume balance writes
    ``V * dc/dt``). Requires ``horizon`` and ``state`` first. Tags attached
    Constraints, wraps a fresh one, or builds one as a decorator:
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
    # a container with any declared member is covered: a state may be a
    # Reference over a member subset (gh #20): an indexed Var with only
    # some entries declared as states, the IDAES holdup with water left
    # out. The left-out entries' balance equations are treated as
    # ordinary algebraic equations and replicated as written
    covered = {id(s) for s in states}
    for s in states:
        for vd in s.values() if s.is_indexed() else (s,):
            covered.add(id(vd.parent_component()))
    for comp in components:
        for cd in _members(comp):
            deriv, _coeff, _ = _dynamics_sides(cd, time, fn)
            dv = deriv.parent_component()
            state = dv.get_state_var()
            if id(state) not in covered:
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
    Requires ``horizon`` first when one is declared; on a model with no
    horizon (a steady-state model) the control registers without a profile,
    since there is no time to parameterize over. Requires pyomo-cvp
    installed. Tags attached Vars or wraps one fresh Var.
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
        # a steady-state model declares no horizon: the control registers
        # without a profile, since there is no time to parameterize over
        if reg.has_declaration("horizon"):
            time = _declared_horizon(reg, fn)
            for comp in comps:
                if not any(s is time for s in comp.index_set().subsets()):
                    raise ValueError(
                        f"drto: {fn}: '{comp.name}' is not indexed by the "
                        f"declared time set '{time.name}'. A control is a "
                        f"decision over time; only a model with no horizon "
                        f"declares time-free controls."
                    )
            _declare_many("control", comps, fn, profile=profile)
            pyomo_cvp.declare_profile(*comps, wrt=time, profile=profile)
        else:
            _declare_many("control", comps, fn, profile=profile)

    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: register((comp,)), fn)
    register(components)
    return components[0] if len(components) == 1 else None


def _tracking_cost_coverage(component, fn, controls_too):
    """The tracking cost covers every state (and control) and nothing else.

    Each member's defining side may reference only declared state members
    (and, for the stage cost, declared control members), and must reference
    at least one member of every declared state (and control). Targets and
    scales are Params, not variables, so they pass untouched; the defining
    cost variable sits on the other side of the equality.
    """
    reg = info(component.model())
    states = list(reg.components("state"))
    controls = list(reg.components("control")) if controls_too else []
    what = "state or control" if controls_too else "state"
    if not states or (controls_too and not controls):
        raise ValueError(
            f"drto: {fn}: declare the states"
            f"{' and controls' if controls_too else ''} before the tracking "
            f"cost; it must cover every one of them."
        )
    owner_of = {}
    for comp in states + controls:
        for vd in comp.values() if comp.is_indexed() else (comp,):
            owner_of[id(vd)] = comp
    for cd in _members(component):
        _, expr = _side_matching(
            cd, _is_var_member, fn, "the scalar cost variable (the cost term)"
        )
        seen = set()
        for v in identify_variables(expr, include_fixed=True):
            owner = owner_of.get(id(v))
            if owner is None:
                raise ValueError(
                    f"drto: {fn}: '{cd.name}' references '{v.name}', which "
                    f"is not a declared {what}. A tracking cost contains "
                    f"declared {what} members and Params (targets, scales) "
                    f"only."
                )
            seen.add(id(owner))
        missing = [c.name for c in states + controls if id(c) not in seen]
        if missing:
            raise ValueError(
                f"drto: {fn}: '{cd.name}' does not reference "
                f"'{missing[0]}'. The tracking cost includes every declared "
                f"state{' and control' if controls_too else ''}; missing: "
                f"{', '.join(missing)}."
            )


def _register_stage_cost(kind, component, fn):
    """Validate and record a stage cost (attached and constructed).

    With a horizon declared the cost is per sample, indexed over the sample
    grid minus the final point. On a model with no declared horizon (a
    steady-state model) there is no grid to index over, so the cost is a
    scalar Constraint: the single-point shape the steady modes assemble and
    the one the steady-state reduction produces from a per-sample cost.
    """
    reg = info(component.model())
    if kind == "move_suppression" and not reg.has_declaration("horizon"):
        raise ValueError(
            f"drto: {fn} requires a declared horizon; a steady-state model "
            f"has no moves."
        )
    if reg.has_declaration("horizon"):
        time = _declared_horizon(reg, fn)
        if any(s is time for s in component.index_set().subsets()):
            raise ValueError(
                f"drto: {fn}: '{component.name}' is indexed by the time set "
                f"'{time.name}': discretization expands such a family to every "
                f"collocation point, and the objective sums the cost at the "
                f"samples only, so it never includes the members "
                f"discretization added. "
                f"Index it over the samples, for example "
                f"@m.Constraint(sorted(m.t)[:-1])."
            )
        samples = reg.declarations("horizon")[0]["samples"]
        expected = list(samples[:-1])
        members = sorted(component.keys()) if component.is_indexed() else []
        if members != expected:
            raise ValueError(
                f"drto: {fn}: '{component.name}' must have one member per "
                f"sample point except the final one, where only the terminal "
                f"cost applies: index it over the samples, for example "
                f"@m.Constraint(sorted(m.t)[:-1])."
            )
    elif component.is_indexed():
        raise ValueError(
            f"drto: {fn}: '{component.name}' is indexed, but no horizon is "
            f"declared. A steady-state model's cost is a single point: "
            f"declare a scalar Constraint."
        )
    for cd in _members(component):
        _side_matching(
            cd, _is_var_member, fn, "the scalar cost variable (the cost term)"
        )
    if kind == "tracking_stage_cost":
        _tracking_cost_coverage(component, fn, controls_too=True)
    if kind == "move_suppression":
        _move_cost_containment(component, fn)
    _declare_single(kind, component, fn)


def _declare_stage_cost(kind, args, kwargs):
    """Dispatch a stage-cost declaration across the three calling styles.

    ``kind`` is both the registry key and the declaring function's name,
    and the error messages use it.
    """
    fn = kind
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
    defines the cost, covering every declared state and control and
    containing nothing else (targets and scales are Params). One per
    model, indexed over the samples minus the final time (the terminal
    cost owns it). Declare the states and controls first. Tags, wraps, or
    builds as a decorator: ``@drto.tracking_stage_cost(m, sorted(m.t)[:-1])``.
    """
    return _declare_stage_cost("tracking_stage_cost", args, kwargs)


def economic_stage_cost(*args, **kwargs):
    """Declare the economic stage cost, a per-time-point equality.

    One side of each member is the scalar running-cost variable; the other
    defines the cost. One per model. Tags, wraps, or builds as a decorator.
    """
    return _declare_stage_cost("economic_stage_cost", args, kwargs)


def _move_cost_containment(component, fn):
    """The move cost penalizes control members inside each member's window.

    Each member's defining side may reference only declared control
    members, at the member's own sample or the one before it. Params (the
    weights, and the previous action the first member is measured against)
    pass untouched; the defining cost variable sits on the other side of
    the equality.
    """
    reg = info(component.model())
    controls = list(reg.components("control"))
    if not controls:
        raise ValueError(
            f"drto: {fn}: declare the controls before the move cost; it "
            f"penalizes their moves."
        )
    owner_of = {}
    for comp in controls:
        for vd in comp.values() if comp.is_indexed() else (comp,):
            owner_of[id(vd)] = comp
    samples = list(reg.declarations("horizon")[0]["samples"])
    for cd in _members(component):
        _, expr = _side_matching(
            cd, _is_var_member, fn, "the scalar cost variable (the cost term)"
        )
        pos = samples.index(cd.index())
        allowed = set(samples[max(pos - 1, 0) : pos + 1])
        for v in identify_variables(expr, include_fixed=True):
            if id(v) not in owner_of:
                raise ValueError(
                    f"drto: {fn}: '{cd.name}' references '{v.name}', which "
                    f"is not a declared control member. A move cost contains "
                    f"declared control members and Params (weights, the "
                    f"previous action) only."
                )
            idx = v.index()
            elems = idx if isinstance(idx, tuple) else (idx,)
            if not any(e in allowed for e in elems):
                raise ValueError(
                    f"drto: {fn}: '{cd.name}' references '{v.name}', which "
                    f"is outside its own sample and the one before."
                )


def move_suppression(*args, **kwargs):
    """Declare the move-suppression cost, a per-sample equality.

    One side of each member is the scalar move-cost variable; the other
    penalizes the control moves, referencing declared control members at
    that sample and the one before, plus Params (the weights, and the
    previous action the first member is measured against). One per model,
    indexed over the samples minus the final time; requires a declared
    horizon, since a steady-state model has no moves. The objective sums
    it with the stage costs, the steady-state reduction drops it, and
    the terminal segment keeps it on the finite horizon and out of the
    tail integrand. Tags, wraps, or builds as a decorator:
    ``@drto.move_suppression(m, sorted(m.t)[:-1])``.
    """
    return _declare_stage_cost("move_suppression", args, kwargs)


def _declare_scalar_cost(kind, args, kwargs, scalar_reason, var_desc):
    """Declare a scalar-LHS equality cost across the three calling styles.

    Shared by the terminal-form costs, a single scalar equality Constraint
    whose one side is the scalar cost variable: the tracking terminal cost,
    and the estimation terminal and arrival costs (feature 018).

    ``kind`` is both the registry key and the declaring function's name,
    and the error messages use it.
    """
    fn = kind

    def register(component):
        if component.is_indexed():
            raise ValueError(
                f"drto: {fn}: '{component.name}' must be a scalar Constraint "
                f"({scalar_reason})."
            )
        _side_matching(component, _is_var_member, fn, var_desc)
        if kind == "tracking_terminal_cost":
            _tracking_cost_coverage(component, fn, controls_too=False)
        _declare_single(kind, component, fn)

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


def tracking_terminal_cost(*args, **kwargs):
    """Declare the terminal tracking cost, a scalar equality.

    One side is the scalar terminal-cost variable; the other defines the
    cost, covering every declared state and containing nothing else
    (targets and scales are Params). One per model; declare the states
    first. Tags, wraps, or builds as a decorator:
    ``@drto.tracking_terminal_cost(m)``.
    """
    return _declare_scalar_cost(
        "tracking_terminal_cost",
        args,
        kwargs,
        "the terminal cost applies at the final time only",
        "the scalar terminal-cost variable",
    )


def initial_condition(*components, **kwargs):
    """Declare one or more initial-condition equality Constraints.

    One side of each is a declared state at the first time point; the other
    is a mutable Param, which a loop overwrites with each measurement
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
    # by data identity: a declared state may be a Reference over a member
    # subset (gh #20), and the pinned member is the referent, not a member
    # of the Reference container
    state_data = set()
    for s in states:
        for vd in s.values() if s.is_indexed() else (s,):
            state_data.add(id(vd))
    for comp in components:
        for cd in _members(comp):
            state_side, param_side = _side_matching(
                cd,
                lambda s: _is_var_member(s) and id(s) in state_data,
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
                    f"mutable Param a loop can overwrite."
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
    if isinstance(owner, IndexedComponent_slice):
        # the same slice that declared the member-subset state resolves to
        # the wrapped Reference by data identity (gh #20)
        referents = list(owner)
        ids = {id(v) for v in referents}
        reg_s = info(referents[0].model())
        owner = next(
            (
                c
                for c in reg_s.components(owner_kind)
                if {id(vd) for vd in (c.values() if c.is_indexed() else (c,))} == ids
            ),
            None,
        )
        if owner is None:
            raise ValueError(
                f"drto: {fn}: the slice matches no declared {owner_kind}; "
                f"drto.{owner_kind} first."
            )
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
    steady-state/RTO solve, which is why the pairing is
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


# ----------------------------------------------------------------------
# the estimation surface (feature 018)
# ----------------------------------------------------------------------
def estimated_parameter(*components):
    """Declare one or more Vars for unknown model parameters to estimate.

    The parameters are constant over the window, so they are not indexed by
    the declared time set. Shared with the steady-state data-reconciliation
    mode, so no horizon is required. Tags attached Vars or wraps one fresh Var.
    """
    fn = "estimated_parameter"
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Var", fn)

    def register(comps):
        reg = info(comps[0].model())
        if reg.has_declaration("horizon"):
            time = _declared_horizon(reg, fn)
            for comp in comps:
                if comp.is_indexed() and any(
                    s is time for s in comp.index_set().subsets()
                ):
                    raise ValueError(
                        f"drto: {fn}: '{comp.name}' is indexed by the declared "
                        f"time set '{time.name}'; an estimated parameter is "
                        f"constant over the window."
                    )
        _declare_many("estimated_parameter", comps, fn)

    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: register((comp,)), fn)
    register(components)
    return components[0] if len(components) == 1 else None


def disturbance(*components, profile="piecewise_constant"):
    """Declare one or more process-noise Vars and their profile.

    A disturbance is the estimation-side dual of a control: a Var the
    estimation frees and a simulation fixes at a realization. The ``profile``
    (a pyomo-cvp profile) parameterizes it over the declared time set, one
    value per sample, the same piecewise-constant representation a control
    takes; process noise is a zero-order hold over a sampling interval, not a
    value per collocation point. How the noise enters the model equations is
    the user's, not fixed here. Requires pyomo-cvp installed; on a model with
    no horizon the disturbance registers without a profile, since there is no
    time to parameterize over. Tags attached Vars or wraps one fresh Var.
    """
    fn = "disturbance"
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Var", fn)
    if not pyomo_cvp_available:
        raise RuntimeError(
            "drto: disturbance requires pyomo-cvp for the profile "
            "(pip install pyomo-cvp)."
        )

    def register(comps):
        reg = info(comps[0].model())
        if reg.has_declaration("horizon"):
            time = _declared_horizon(reg, fn)
            for comp in comps:
                if not any(s is time for s in comp.index_set().subsets()):
                    raise ValueError(
                        f"drto: {fn}: '{comp.name}' is not indexed by the "
                        f"declared time set '{time.name}'; process noise is a "
                        f"free variable over the window."
                    )
            _declare_many("disturbance", comps, fn, profile=profile)
            pyomo_cvp.declare_profile(*comps, wrt=time, profile=profile)
        else:
            _declare_many("disturbance", comps, fn, profile=profile)

    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: register((comp,)), fn)
    register(components)
    return components[0] if len(components) == 1 else None


def measurement(*components):
    """Declare one or more measurement Params.

    The measured values over the window, mutable Params drto refreshes each
    step like the initial-condition Params. They appear in the estimation cost
    residuals; the measurement map ``h(z)`` is written inline in the cost, so
    there is nothing else to tag. Indexed by the declared time set when a
    horizon is declared. Tags attached Params or wraps one fresh Param.
    """
    fn = "measurement"
    if not components:
        raise TypeError(f"drto: {fn} needs at least one component.")
    for comp in components:
        _container(comp, fn)
        _check_ctype(comp, "Param", fn)
        if not comp.mutable:
            raise ValueError(
                f"drto: {fn}: Param '{comp.name}' must be mutable so the loop "
                f"can write the incoming measurements into it."
            )

    def register(comps):
        reg = info(comps[0].model())
        if reg.has_declaration("horizon"):
            time = _declared_horizon(reg, fn)
            for comp in comps:
                if not any(s is time for s in comp.index_set().subsets()):
                    raise ValueError(
                        f"drto: {fn}: '{comp.name}' is not indexed by the "
                        f"declared time set '{time.name}'; measurements arrive "
                        f"over the window."
                    )
        _declare_many("measurement", comps, fn)

    if _wrap_form(components, fn):
        (comp,) = components
        return _defer(comp, lambda: register((comp,)), fn)
    register(components)
    return components[0] if len(components) == 1 else None


def estimation_stage_cost(*args, **kwargs):
    """Declare the estimation stage cost, a per-time-point equality.

    One side of each member is the scalar running estimation-cost variable (the
    measurement residual plus the process-noise penalty); the other defines it.
    One per model, indexed over the samples minus the final time (the terminal
    term owns it). Tags, wraps, or builds as a decorator.
    """
    return _declare_stage_cost("estimation_stage_cost", args, kwargs)


def estimation_terminal_cost(*args, **kwargs):
    """Declare the terminal estimation cost, a scalar equality.

    One side is the scalar terminal estimation-cost variable (the current-state
    measurement residual at the present time, no process noise); the other
    defines it. One per model. Tags, wraps, or builds as a decorator.
    """
    return _declare_scalar_cost(
        "estimation_terminal_cost",
        args,
        kwargs,
        "the terminal estimation cost applies at the present time only",
        "the scalar terminal estimation-cost variable",
    )


def arrival_cost(*args, **kwargs):
    """Declare the arrival cost, a scalar equality.

    One side is the scalar arrival-cost variable (the soft prior on the
    window's initial state); the other defines it. The soft dual of the
    control-side initial condition. One per model. Tags, wraps, or builds as a
    decorator.
    """
    return _declare_scalar_cost(
        "arrival_cost",
        args,
        kwargs,
        "the arrival cost applies at the initial time only",
        "the scalar arrival-cost variable",
    )
