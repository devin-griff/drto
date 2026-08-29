# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic optimization: ``drto.dynamic_optimization`` (feature 006).

Assembles the horizon optimization from the declarations. The declared
controls are the free decisions, parameterized over the time set by their
declared profiles, and the objective is the live cost terms. The horizon is
kept, since this is the dynamic mode rather than a reduction.

A model that also carries the estimation declarations (feature 018) still
yields a clean control problem. The estimation costs and measurements are
dropped and any estimated parameter fixed (``_neutralize_estimation``, shared
by the four control-side modes), and the disturbances are fixed at zero
(``_fix_disturbances``), the process noise off in the controller's own model.

``drto.infinite_horizon`` (feature 004) applies before this transform,
because the objective is assembled here as the final step, so the tail's
cost group must be registered by then.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory, Var

from drto.declarations import _is_var_member, _side_matching
from drto.info import info
from drto.objective import build_objective

#: The declarations the transform requires.
_REQUIRED = ("horizon", "state", "dynamics", "control", "initial_condition")

#: The stage-cost kinds. At least one must be declared.
_STAGE_KINDS = ("tracking_stage_cost", "economic_stage_cost")

#: Estimation kinds whose components leave the model outright. A measurement
#: is reachable only from these costs, since h(z) is written inline in the
#: cost, so it is orphaned once they go.
_REMOVED_ESTIMATION_KINDS = (
    "estimation_stage_cost",
    "estimation_terminal_cost",
    "arrival_cost",
    "measurement",
)


def _members(comp):
    """Yield the members of a scalar or indexed component."""
    return comp.values() if comp.is_indexed() else (comp,)


def _spread(val, n_free, name, fn):
    """Return ``n_free`` values from a constant or a per-point sequence."""
    if isinstance(val, (list, tuple)):
        if len(val) != n_free:
            raise ValueError(
                f"drto: {fn} got {len(val)} values for '{name}', which has "
                f"{n_free} free points after its profile is applied. Pass a "
                f"constant or one value each."
            )
        return list(val)
    return [val] * n_free


def _time_position(comp, time):
    """The position of ``time`` in this component's index, or None.

    None when the component is not indexed by the horizon, or when the
    horizon appears more than once, or when another index set is not
    one-dimensional, in which case a position in the index tuple does not
    follow from a position among the subsets.
    """
    index = comp.index_set()
    subsets = list(index.subsets()) if hasattr(index, "subsets") else [index]
    at = [i for i, sub in enumerate(subsets) if sub is time]
    if len(at) != 1 or any(sub.dimen != 1 for sub in subsets):
        return None
    return at[0]


def _held_inputs(m, time, samples, reg):
    """Record the values of the members held at every declared sample point.

    Walks the Vars indexed by the horizon and groups their members by the
    index with the time entry removed. A group whose members are fixed at
    every declared sample point is one the builder holds, and this returns
    it as ``(component, at, rest, {sample: value})``, where ``at`` is the
    position of the time entry in the index and ``rest`` is the index
    without it. ``dae.collocation`` adds members afterward, and
    ``_hold_new_members`` fixes them from the values recorded here.

    The declared controls and disturbances are left out, since
    ``drto.parameterize`` and the modes set those. A group fixed at only
    some samples, an initial-condition pin, is left out too.
    """
    owned = {
        id(vd)
        for kind in ("control", "disturbance")
        for comp in reg.components(kind)
        for vd in _members(comp)
    }
    held = []
    for comp in m.component_objects(Var, active=True, descend_into=True):
        at = _time_position(comp, time)
        if at is None:
            continue
        groups = {}
        for key, vd in comp.items():
            key = key if isinstance(key, tuple) else (key,)
            groups.setdefault(key[:at] + key[at + 1 :], {})[key[at]] = vd
        for rest, group in groups.items():
            members = [group.get(t) for t in samples]
            if any(vd is None or not vd.fixed for vd in members) or any(
                id(vd) in owned for vd in group.values()
            ):
                continue
            held.append((comp, at, rest, {t: group[t].value for t in samples}))
    return held


def _hold_new_members(held, samples):
    """Fix the members ``dae.collocation`` added, at the previous sample.

    Each entry names a component and one index of it but for the time
    entry, recorded by ``_held_inputs`` before the mesh was refined. Every
    member added sits inside an element, so the sample before it is the
    element's left end. Returns the number fixed.
    """
    n = 0
    for comp, at, rest, values in held:
        for key, vd in comp.items():
            key = key if isinstance(key, tuple) else (key,)
            if key[:at] + key[at + 1 :] != rest or key[at] in values:
                continue
            vd.fix(values[max(s for s in samples if s < key[at])])
            n += 1
    return n


def _initial_condition_params(reg, fn):
    """The initial-condition Params, one ParamData per pinned member.

    Each declared initial-condition constraint pins a state member to a bare
    mutable Param. ``drto.initial_condition`` enforces that shape at
    declaration time, so the non-variable side here is always that Param
    itself.
    """
    params, seen = [], set()
    for con in reg.components("initial_condition"):
        for cd in _members(con):
            _, other = _side_matching(cd, _is_var_member, fn, "a state member")
            if id(other) not in seen:
                seen.add(id(other))
                params.append(other)
    return params


def _declare_sens_params(reg, fn):
    """Declare the initial-condition Params as pounce sensitivity parameters.

    Runs only when pyomo-pounce is importable. The declaration is inert
    metadata that every other solver ignores, while a pounce solve keeps
    the converged factorization for the advanced-step correction
    (feature 012). Returns the number declared, or None without pounce.
    """
    try:
        import pyomo_pounce
    except ImportError:
        return None
    params = _initial_condition_params(reg, fn)
    pyomo_pounce.declare_sens_param(*params)
    return len(params)


def _fix_disturbances(reg, requested, fn):
    """Fix the declared disturbances at their realization, defaulting to zero.

    A disturbance is a declared Var like a control. The simulations fix it
    at a supplied realization, the optimizations fix it at zero, and the
    estimation modes leave it free. It is never eliminated, so the model
    keeps the structure showing where the noise enters, and fixing (unlike
    substituting zero) works however the noise enters the equations. With a
    piecewise-constant profile the free per-element values are fixed and the
    profile's dependent copies follow, and on a reduced model it is a single
    point. Resolution is by name,
    since parameterizing and reducing both replace the component. Returns the
    fixed display names.
    """
    declared = {c.name: c for c in reg.components("disturbance")}
    wanted = {}
    for key, val in (requested or {}).items():
        name = key if isinstance(key, str) else key.name
        if name not in declared:
            raise ValueError(
                f"drto: {fn} got a disturbance realization for '{name}', "
                f"which is not a declared disturbance. The declared "
                f"disturbances are {', '.join(declared) or '(none)'}."
            )
        wanted[name] = val

    fixed = []
    for name, comp in declared.items():
        members = list(_members(comp))
        values = _spread(wanted.get(name, 0.0), len(members), name, fn)
        for vd, v in zip(members, values):
            vd.set_value(v)
            vd.fix()
        fixed.append(f"{name}={wanted[name]}" if name in wanted else f"{name}=0")
    return fixed


def _fix_estimated_parameters(reg, fn):
    """Fix the declared estimated parameters at the values they hold.

    The parameter is known to a control-side mode, so its current value is
    the estimate. The Var stays in the equations, so its record stays too.
    """
    pinned = []
    for comp in reg.components("estimated_parameter"):
        for vd in _members(comp):
            if vd.value is None:
                raise ValueError(
                    f"drto: {fn} fixes the estimated parameter "
                    f"'{comp.name}' at the value it holds, but it has none. "
                    f"Initialize it first."
                )
            vd.fix()
        pinned.append(comp.name)
    return pinned


def _neutralize_estimation(reg, fn):
    """Neutralize the estimation costs for a control-side mode.

    Shared by the four control-side modes so they cannot drift apart. The
    registry mirrors the model, so a component that leaves has its record
    purged and one that stays keeps its record. The estimation costs and the measurement
    Params are deleted, and an estimated parameter is fixed and keeps its
    record, since it stays a live coefficient in the equations. The disturbance
    is not touched here, and each mode fixes it at its own value afterward (see
    ``_fix_disturbances``). Returns the outcome fields for the transformation
    log.
    """
    removed = []
    for kind in _REMOVED_ESTIMATION_KINDS:
        for record in reg.declarations(kind):
            comp = record["component"]
            if comp.parent_block() is not None:
                comp.parent_block().del_component(comp)
        if reg.has_declaration(kind):
            removed.append(kind.replace("_", " "))
        # the records describe components that no longer exist on the
        # control-side model, so this reaches into the registry's own store
        reg._declarations.pop(kind, None)

    pinned = _fix_estimated_parameters(reg, fn)

    outcome = {}
    if removed:
        outcome["removed"] = ", ".join(removed)
    if pinned:
        outcome["fixed"] = ", ".join(pinned)
    return outcome


def dynamic_optimization(
    build,
    N=None,
    h=None,
    ncp=3,
    scheme="LAGRANGE-RADAU",
    infinite_horizon=False,
    tracking_weight=None,
):
    """Build a model, discretize it, and assemble the horizon optimization.

    Takes the model statement rather than a model, so one call reaches a
    problem ready to solve. The builder contract is feature 006's: ``build``
    returns a declared, undiscretized model, its first two parameters are
    the interval count and the sampling time named ``N`` and ``h``, and
    every parameter has a default, so the bare ``build()`` is legal.

    A builder holds its constant inputs by fixing them at the declared
    sample points, which is all an undiscretized model has, and
    discretization here completes them. Before ``dae.collocation`` runs,
    the members fixed at every sample point are recorded with their values,
    and the members it adds to those are fixed afterward at the value of
    the sample before them. The declared controls and disturbances are left
    out, since ``drto.parameterize`` and the modes set those, and so is
    anything fixed at only some samples, an initial-condition pin.

    Parameters
    ----------
    build : callable
        The model statement. Called with whichever of ``N`` and ``h`` were
        given, by keyword, so an omitted one takes the builder's default.
    N : int, optional
        Intervals, passed to the builder.
    h : float, optional
        Sampling time, passed to the builder.
    ncp : int, optional
        Collocation points per finite element (default 3).
    scheme : str, optional
        The collocation scheme (default ``"LAGRANGE-RADAU"``).
    infinite_horizon : bool or mapping, optional
        ``False`` (the default) adds no terminal segment. ``True`` applies
        ``drto.infinite_horizon`` with its own defaults, and a mapping
        passes its contents as that transformation's options.
    tracking_weight : float, optional
        Passed to the registered transformation when given, weighting the
        tracking stage cost against the economic one.

    Returns
    -------
    Block
        The model the builder returned, discretized and assembled in place.
        It is uninitialized, so ``drto.cold_start_dynamic`` and the other
        initializers run on it afterward.

    Raises
    ------
    ValueError
        If the builder returns a model whose declared time set is already
        discretized, since this function owns the mesh.

    Examples
    --------
    ::

        m = drto.dynamic_optimization(build, N=50, infinite_horizon=True)
    """
    kwargs = {}
    if N is not None:
        kwargs["N"] = N
    if h is not None:
        kwargs["h"] = h
    m = build(**kwargs)

    reg = info(m)
    time = reg.components("horizon")[0]
    if time.get_discretization_info():
        raise ValueError(
            f"drto: dynamic_optimization discretizes the model the builder "
            f"returns, but '{time.name}' is already discretized. A builder "
            f"returns a declared, undiscretized model (feature 006)."
        )
    # one finite element per declared interval. The horizon record holds the
    # grid as declared, since horizon errors once the set is discretized
    samples = reg.declarations("horizon")[0]["samples"]
    held = _held_inputs(m, time, samples, reg)
    TransformationFactory("dae.collocation").apply_to(
        m, wrt=time, nfe=len(samples) - 1, ncp=ncp, scheme=scheme
    )
    _hold_new_members(held, samples)

    if infinite_horizon:
        opts = infinite_horizon if infinite_horizon is not True else {}
        TransformationFactory("drto.infinite_horizon").apply_to(m, **opts)

    opts = {} if tracking_weight is None else {"tracking_weight": tracking_weight}
    TransformationFactory("drto.dynamic_optimization").apply_to(m, **opts)
    return m


@TransformationFactory.register(
    "drto.dynamic_optimization",
    doc="Assemble the dynamic optimization problem from the declarations (drto).",
)
class DynamicOptimizationTransformation(Transformation):
    """The dynamic optimization mode. See the module docstring.

    Options: ``tracking_weight`` weights the tracking stage cost, and applies
    only when both a tracking and an economic stage cost are declared.

    ``apply_to`` assembles in place. ``create_using`` assembles a clone and
    leaves the source model alone.
    """

    CONFIG = ConfigDict("drto.dynamic_optimization")
    CONFIG.declare(
        "tracking_weight",
        ConfigValue(
            default=1.0,
            domain=float,
            description="Weight on the tracking stage cost, used only when "
            "both a tracking and an economic stage cost are declared. The "
            "economic cost is in currency units and is never scaled.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        reg = info(model)
        missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
        if missing:
            raise ValueError(
                f"drto: dynamic_optimization requires the declarations "
                f"{', '.join(_REQUIRED)}. Missing: {', '.join(missing)}."
            )
        if not any(reg.has_declaration(k) for k in _STAGE_KINDS):
            raise ValueError(
                "drto: dynamic_optimization requires a stage cost. Missing: "
                "tracking_stage_cost or economic_stage_cost."
            )

        outcome = _neutralize_estimation(reg, "dynamic_optimization")

        # --- the tracking weight, when both cost kinds are declared -------
        # build_objective reads it off the group's record
        weighted = None
        if all(reg.has_declaration(k) for k in _STAGE_KINDS):
            for record in reg.declarations("tracking_stage_cost"):
                record["weight"] = config.tracking_weight
            weighted = config.tracking_weight

        TransformationFactory("drto.parameterize").apply_to(model)
        # the controller predicts on its own model with the process noise off,
        # fixed at zero after parameterization exposes the free per-element
        # values
        noise = _fix_disturbances(reg, {}, "dynamic_optimization")
        build_objective(model)
        n_params = _declare_sens_params(reg, "dynamic_optimization")

        reg.record_transformation(
            "drto.dynamic_optimization",
            horizon="kept",
            tracking_weight=(
                weighted if weighted is not None else "(one stage cost declared)"
            ),
            **({"disturbances": ", ".join(noise)} if noise else {}),
            **(
                {"sensitivity": f"{n_params} initial-condition Params declared"}
                if n_params is not None
                else {}
            ),
            **outcome,
        )
        return model
