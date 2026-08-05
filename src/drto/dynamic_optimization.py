# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic optimization: ``drto.dynamic_optimization`` (feature 006).

Assembles the horizon optimization from the declarations: the declared
controls are the free decisions, parameterized over the time set by their
declared profiles, and the objective is the live cost terms. The horizon is
kept; this is the dynamic mode, not a reduction.

A model that also carries the estimation declarations (feature 018) still
yields a clean control problem: the estimation costs and measurements are
dropped and any estimated parameter fixed (``_neutralize_estimation``, shared
by the four control-side modes), and the disturbances are fixed at zero
(``_fix_disturbances``), the process noise off in the controller's own model.

``drto.infinite_horizon`` (feature 004) applies before this transform: the
objective is assembled here as the final step, so the tail's cost group must
be registered by then.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory

from drto.declarations import _is_var_member, _side_matching
from drto.info import info
from drto.objective import build_objective

#: The declarations the transform requires.
_REQUIRED = ("horizon", "state", "dynamics", "control", "initial_condition")

#: The stage-cost kinds; at least one must be declared.
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
                f"{n_free} free points after its profile is applied; pass a "
                f"constant or one value each."
            )
        return list(val)
    return [val] * n_free


def _feedback_hooks(reg, fn):
    """The initial-condition Params, one ParamData per pinned member.

    Each declared initial-condition constraint pins a state member to a bare
    mutable Param; ``drto.initial_condition`` enforces
    that shape at declaration time, so the non-variable side here is
    always that Param itself.
    """
    hooks, seen = [], set()
    for con in reg.components("initial_condition"):
        for cd in _members(con):
            _, other = _side_matching(cd, _is_var_member, fn, "a state member")
            if id(other) not in seen:
                seen.add(id(other))
                hooks.append(other)
    return hooks


def _declare_sens_hooks(reg, fn):
    """Declare the initial-condition Params as pounce sensitivity parameters.

    Runs only when pyomo-pounce is importable; the declaration is inert
    metadata that every other solver ignores, while a pounce solve keeps
    the converged factorization for the advanced-step correction
    (feature 012). Returns the number declared, or None without pounce.
    """
    try:
        import pyomo_pounce
    except ImportError:
        return None
    hooks = _feedback_hooks(reg, fn)
    pyomo_pounce.declare_sens_param(*hooks)
    return len(hooks)


def _fix_disturbances(reg, requested, fn):
    """Fix the declared disturbances at their realization, defaulting to zero.

    A disturbance is a declared Var like a control: the simulations fix it at a
    supplied realization, the optimizations fix it at zero, and the estimation
    modes leave it free. It is never eliminated, so the "noise enters here"
    structure stays in the model, and fixing (unlike substituting zero) works
    however the noise enters the equations. With a piecewise-constant profile
    the free per-element values are fixed and the profile's dependent copies
    follow; on a reduced model it is a single point. Resolution is by name,
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
                f"which is not a declared disturbance; declared: "
                f"{', '.join(declared) or '(none)'}."
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

    The parameter is known to a control-side mode: its current value is the
    estimate. The Var stays in the equations, so its record stays too.
    """
    pinned = []
    for comp in reg.components("estimated_parameter"):
        for vd in _members(comp):
            if vd.value is None:
                raise ValueError(
                    f"drto: {fn} fixes the estimated parameter "
                    f"'{comp.name}' at the value it holds, but it has none; "
                    f"initialize it first."
                )
            vd.fix()
        pinned.append(comp.name)
    return pinned


def _neutralize_estimation(reg, fn):
    """Neutralize the estimation costs for a control-side mode.

    Shared by the four control-side modes so they cannot drift apart. The
    registry mirrors the model: a component that leaves has its record purged,
    one that stays keeps its record. The estimation costs and the measurement
    Params are deleted, and an estimated parameter is fixed and keeps its
    record, since it stays a live coefficient in the equations. The disturbance
    is not touched here; each mode fixes it at its own value afterward (see
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
        # same-package registry surgery: the records describe components that
        # no longer exist on the control-side model
        reg._declarations.pop(kind, None)

    pinned = _fix_estimated_parameters(reg, fn)

    outcome = {}
    if removed:
        outcome["removed"] = ", ".join(removed)
    if pinned:
        outcome["fixed"] = ", ".join(pinned)
    return outcome


@TransformationFactory.register(
    "drto.dynamic_optimization",
    doc="Assemble the dynamic optimization problem from the declarations (drto).",
)
class DynamicOptimizationTransformation(Transformation):
    """The dynamic optimization mode; see the module docstring.

    Options: ``tracking_weight`` weights the tracking stage cost, and applies
    only when both a tracking and an economic stage cost are declared.

    ``apply_to`` assembles in place; ``create_using`` assembles a clone and
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
                f"{', '.join(_REQUIRED)}; missing: {', '.join(missing)}."
            )
        if not any(reg.has_declaration(k) for k in _STAGE_KINDS):
            raise ValueError(
                "drto: dynamic_optimization requires a stage cost; missing: "
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
        n_hooks = _declare_sens_hooks(reg, "dynamic_optimization")

        reg.record_transformation(
            "drto.dynamic_optimization",
            horizon="kept",
            tracking_weight=(
                weighted if weighted is not None else "(one stage cost declared)"
            ),
            **({"disturbances": ", ".join(noise)} if noise else {}),
            **(
                {"sensitivity": f"{n_hooks} initial-condition Params declared"}
                if n_hooks is not None
                else {}
            ),
            **outcome,
        )
        return model
