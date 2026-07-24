# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic optimization: ``drto.dynamic_optimization`` (feature 006).

Assembles the horizon optimization from the declarations: the declared
controls are the free decisions, parameterized over the time set by their
declared profiles, and the objective is the live cost terms. The horizon is
kept; this is the dynamic mode, not a reduction.

The estimation-category declarations (feature 018) are neutralized first, so a
model that also carries the estimation problem still yields a clean control
problem. The registry mirrors the model throughout: a component that leaves
the model has its record purged, one that stays keeps its record. The
estimation costs and the measurement Params are deleted, a disturbance is
eliminated by substituting zero, and an estimated parameter is fixed at its
current value and keeps its record, since it stays a live coefficient in the
equations the controller solves.

``drto.infinite_horizon`` (feature 004) applies before this transform: the
objective is assembled here as the final step, so the tail's cost group must
be registered by then.
"""
from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import (
    Constraint,
    Expression,
    Objective,
    Transformation,
    TransformationFactory,
)
from pyomo.core.expr import identify_variables, replace_expressions
from pyomo.repn import generate_standard_repn

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


@TransformationFactory.register(
    "drto.dynamic_optimization",
    doc="Assemble the dynamic optimization problem from the declarations " "(drto).",
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

        # --- the estimation declarations are neutralized -----------------
        # the cost equations and the measurements leave the model; their cost
        # variables are left unused, as in the steady-state simulation
        removed = []
        for kind in _REMOVED_ESTIMATION_KINDS:
            for record in reg.declarations(kind):
                comp = record["component"]
                if comp.parent_block() is not None:
                    comp.parent_block().del_component(comp)
            if reg.has_declaration(kind):
                removed.append(kind.replace("_", " "))
            # same-package registry surgery: the records describe components
            # that no longer exist on the control model
            reg._declarations.pop(kind, None)

        n_noise = self._eliminate_disturbances(model, reg)
        pinned = self._fix_estimated_parameters(reg)

        # --- the tracking weight, when both cost kinds are declared -------
        # build_objective reads it off the group's record
        weighted = None
        if all(reg.has_declaration(k) for k in _STAGE_KINDS):
            for record in reg.declarations("tracking_stage_cost"):
                record["weight"] = config.tracking_weight
            weighted = config.tracking_weight

        TransformationFactory("drto.parameterize").apply_to(model)
        build_objective(model)

        reg.record_transformation(
            "drto.dynamic_optimization",
            horizon="kept",
            tracking_weight=(
                weighted if weighted is not None else "(one stage cost declared)"
            ),
            **({"removed": ", ".join(removed)} if removed else {}),
            **(
                {"disturbances": f"{n_noise} references replaced by zero"}
                if n_noise
                else {}
            ),
            **({"fixed": ", ".join(pinned)} if pinned else {}),
        )
        return model

    def _eliminate_disturbances(self, model, reg):
        """Substitute zero for the declared disturbances and delete them.

        Elimination by substitution, as the steady-state reduction does for
        derivatives: no vestigial fixed-at-zero variables. Substituting zero
        removes the noise only where it enters additively, so a disturbance
        inside a nonlinear term errors rather than silently zeroing that term.
        """
        comps = reg.components("disturbance")
        if not comps:
            return 0
        noise, submap = ComponentSet(), {}
        for comp in comps:
            for vd in _members(comp):
                noise.add(vd)
                submap[id(vd)] = 0

        for con in model.component_objects(Constraint, active=True):
            for cd in _members(con):
                present = [
                    v
                    for v in identify_variables(cd.body, include_fixed=True)
                    if v in noise
                ]
                if not present:
                    continue
                repn = generate_standard_repn(cd.body)
                nonlinear = ComponentSet(repn.nonlinear_vars or ())
                for pair in repn.quadratic_vars or ():
                    nonlinear.update(pair)
                offenders = [v for v in present if v in nonlinear]
                if offenders:
                    raise ValueError(
                        f"drto: dynamic_optimization eliminates a disturbance "
                        f"by substituting zero, which removes it only where it "
                        f"enters additively, but '{offenders[0].name}' appears "
                        f"nonlinearly in '{cd.name}'."
                    )
                cd.set_value(replace_expressions(cd.expr, submap))
        for obj in model.component_data_objects(Objective, active=True):
            obj.set_value(replace_expressions(obj.expr, submap))
        for e in model.component_data_objects(Expression, active=True):
            e.set_value(replace_expressions(e.expr, submap))

        for comp in comps:
            comp.parent_block().del_component(comp)
        reg._declarations.pop("disturbance", None)
        return len(noise)

    def _fix_estimated_parameters(self, reg):
        """Fix the declared estimated parameters at the values they hold.

        The parameter is known to the controller: its current value is the
        estimate. The Var stays in the equations, so its record stays too.
        """
        pinned = []
        for comp in reg.components("estimated_parameter"):
            for vd in _members(comp):
                if vd.value is None:
                    raise ValueError(
                        f"drto: dynamic_optimization fixes the estimated "
                        f"parameter '{comp.name}' at the value it holds, but "
                        f"it has none; initialize it first."
                    )
                vd.fix()
            pinned.append(comp.name)
        return pinned
