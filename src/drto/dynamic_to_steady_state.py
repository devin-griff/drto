# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The steady-state reduction: ``drto.dynamic_to_steady_state`` (feature 005).

Reduces a declared dynamic model to its steady-state form: time collapses
to a single point, every reference to a declared state's time derivative
is replaced by zero and the DerivativeVars are deleted, and the initial
condition, terminal constraint, and terminal cost leave the model. The
result is the equilibrium system, dynamics as ``0 = f(z, u)``, algebraic
relations intact (a derivative-carrying energy balance collapses to its
quasi-static form), and a per-sample stage cost becomes the single-point
cost that ``drto.build_objective`` assembles for the steady modes.

Elimination is by substitution: no ``dz/dt == 0`` rows and no vestigial
variables. The transform applies to the declared or discretized model,
before any drto transformation: applied control profiles or an attached
terminal segment error, the sibling-branch rule. On a discretized model
the discretization artifacts (the collocation equations and continuity
rows pyomo.dae adds) are discarded, grid machinery rather than model
content, and the reduction gives the same steady system either way.
Objective assembly is not performed here; an existing objective only has
its references collapsed.
"""
from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict
from pyomo.core import (
    Constraint,
    Expression,
    Objective,
    Transformation,
    TransformationFactory,
    Var,
)
from pyomo.core.expr import identify_variables, replace_expressions
from pyomo.dae import DerivativeVar

from drto.declarations import _side_matching, pyomo_cvp_available
from drto.infinite_horizon import _is_derivative, _split_index, _time_index
from drto.info import info

#: The declarations the transform requires.
_REQUIRED = ("horizon", "state", "dynamics")

#: The declaration kinds whose components leave the model outright.
_REMOVED_KINDS = ("initial_condition", "terminal_constraint", "tracking_terminal_cost")

#: The stage-cost kinds, indexed by the sample list: they collapse to scalars.
_STAGE_KINDS = ("tracking_stage_cost", "economic_stage_cost")


@TransformationFactory.register(
    "drto.dynamic_to_steady_state",
    doc="Reduce a declared dynamic model to its steady-state form (drto).",
)
class DynamicToSteadyStateTransformation(Transformation):
    """Collapse a declared dynamic model to its equilibrium; see the module
    docstring.

    ``apply_to`` reduces in place; ``create_using`` reduces a clone and
    leaves the dynamic source unchanged.
    """

    CONFIG = ConfigDict("drto.dynamic_to_steady_state")

    def _apply_to(self, model, **kwds):
        self.CONFIG(kwds)  # no options; unknown keywords error
        reg = info(model)
        missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
        if missing:
            raise ValueError(
                f"drto: dynamic_to_steady_state requires the declarations "
                f"{', '.join(_REQUIRED)}; missing: {', '.join(missing)}."
            )
        time = reg.components("horizon")[0]
        for name in ("drto.infinite_horizon", "drto.parameterize"):
            if reg.has_transformation(name):
                raise ValueError(
                    f"drto: dynamic_to_steady_state applies before any drto "
                    f"transformation; '{name}' is already applied. The steady "
                    f"reduction and the dynamic transforms are sibling "
                    f"branches of the same declarations."
                )

        states_set = ComponentSet(reg.components("state"))
        for con in reg.components("dynamics"):
            for cd in con.values() if con.is_indexed() else (con,):
                side, _ = _side_matching(
                    cd,
                    _is_derivative,
                    "dynamic_to_steady_state",
                    "a DerivativeVar (dz/dt)",
                )
                if side.parent_component().get_state_var() not in states_set:
                    raise ValueError(
                        f"drto: dynamic_to_steady_state: '{cd.name}' "
                        f"differentiates an undeclared state."
                    )

        # --- the components leaving the model outright ------------------
        removed = []
        for kind in _REMOVED_KINDS:
            for record in reg.declarations(kind):
                comp = record["component"]
                if comp.parent_block() is not None:
                    comp.parent_block().del_component(comp)
            if reg.has_declaration(kind):
                removed.append(kind.replace("_", " "))
            # same-package registry surgery: the records describe components
            # that no longer exist on the reduced model
            reg._declarations.pop(kind, None)

        # --- discretization artifacts are grid machinery, not model
        # content: discarded, so a discretized model reduces to the same
        # steady system as the declared one ------------------------------
        n_artifacts = 0
        for con in list(model.component_objects(Constraint, active=True)):
            if "_disc_" in con.local_name or con.local_name.endswith("_cont_eq"):
                con.parent_block().del_component(con)
                n_artifacts += 1

        # --- no member may span more than one time point ----------------
        for con in model.component_objects(Constraint, active=True):
            for cd in con.values() if con.is_indexed() else (con,):
                per_comp = {}
                for v in identify_variables(cd.expr, include_fixed=True):
                    comp = v.parent_component()
                    if isinstance(comp, DerivativeVar):
                        continue  # every derivative reference becomes zero
                    pos, subs = _time_index(comp, time)
                    if pos is None:
                        continue
                    _, t = _split_index(v.index(), pos, len(subs))
                    per_comp.setdefault(id(comp), set()).add(t)
                if any(len(ts) > 1 for ts in per_comp.values()):
                    raise ValueError(
                        f"drto: dynamic_to_steady_state cannot reduce "
                        f"'{cd.name}': it references a variable at more than "
                        f"one time point, which has no single-point form."
                    )

        # --- collapse the time-indexed Vars -----------------------------
        t0 = time.first()
        submap = {}
        replaced = {}
        tvars = [
            comp
            for comp in model.component_objects(Var, active=True)
            if not isinstance(comp, DerivativeVar)
            and _time_index(comp, time)[0] is not None
        ]
        for comp in tvars:
            pos, subs = _time_index(comp, time)
            others = [s for n, s in enumerate(subs) if n != pos]
            name, parent = comp.local_name, comp.parent_block()
            attrs, members = {}, {}
            for idx, vd in comp.items():
                o, t = _split_index(idx, pos, len(subs))
                members[(o, t)] = vd
                if t == t0:
                    attrs[o] = (vd.domain, vd.lb, vd.ub, vd.value)
            parent.del_component(comp)
            any_dom = next(iter(attrs.values()))[0]
            if others:
                new = Var(
                    *others,
                    domain=any_dom,
                    bounds=lambda m, *o, _a=attrs: (_a[o][1], _a[o][2]),
                    initialize=lambda m, *o, _a=attrs: _a[o][3],
                )
            else:
                dom, lb, ub, val = attrs[()]
                new = Var(domain=dom, bounds=(lb, ub), initialize=val)
            parent.add_component(name, new)
            replaced[id(comp)] = new
            for (o, t), vd in members.items():
                submap[id(vd)] = new[o] if o else new

        # --- every derivative reference becomes zero ---------------------
        # a DerivativeVar is its own ctype before discretization and is
        # reclassified to Var by pyomo.dae afterward: scan both
        seen, derivs = set(), []
        for query in (DerivativeVar, Var):
            for dv in model.component_objects(query):
                if (
                    isinstance(dv, DerivativeVar)
                    and id(dv) not in seen
                    and dv.get_continuousset_list() == [time]
                    and dv.get_state_var() in states_set
                ):
                    seen.add(id(dv))
                    derivs.append(dv)
        for dv in derivs:
            for vd in dv.values():
                submap[id(vd)] = 0

        # --- collapse the constraints ------------------------------------
        stage_cons = ComponentSet()
        for kind in _STAGE_KINDS:
            stage_cons.update(reg.components(kind))
        n_cons = 0
        for con in list(model.component_objects(Constraint, active=True)):
            pos, subs = _time_index(con, time)
            name, parent = con.local_name, con.parent_block()
            if pos is not None:
                # one member per other-combo: the representative's expression
                others = [s for n, s in enumerate(subs) if n != pos]
                reps = {}
                for idx, cd in con.items():
                    o, _ = _split_index(idx, pos, len(subs))
                    if o not in reps:
                        reps[o] = replace_expressions(cd.expr, submap)
                parent.del_component(con)
                if others:
                    new = Constraint(*others, rule=lambda m, *o, _r=reps: _r[o])
                else:
                    new = Constraint(expr=reps[()])
                parent.add_component(name, new)
                replaced[id(con)] = new
                n_cons += 1
            elif con in stage_cons and con.is_indexed():
                # indexed by the sample list: the single-point cost
                expr = replace_expressions(next(iter(con.values())).expr, submap)
                parent.del_component(con)
                new = Constraint(expr=expr)
                parent.add_component(name, new)
                replaced[id(con)] = new
                n_cons += 1
            else:
                for cd in con.values() if con.is_indexed() else (con,):
                    cd.set_value(replace_expressions(cd.expr, submap))
        for obj in model.component_data_objects(Objective, active=True):
            obj.set_value(replace_expressions(obj.expr, submap))
        for e in model.component_data_objects(Expression, active=True):
            e.set_value(replace_expressions(e.expr, submap))

        # --- the time dimension leaves the model --------------------------
        for dv in derivs:
            dv.parent_block().del_component(dv)
        time.parent_block().del_component(time)
        reg._declarations.pop("horizon", None)
        if pyomo_cvp_available:
            from pyomo_cvp.parameterize import _cvp_data

            store = _cvp_data(model)
            decls = store.get("profiles") if store else None
            if decls:
                # pending profiles parameterize over the deleted time set
                decls[:] = [d for d in decls if d["wrt"] is not time]

        # --- point the registry at the collapsed components ---------------
        for records in reg._declarations.values():
            for record in records:
                for key in ("component", "of"):
                    new = replaced.get(id(record.get(key)))
                    if new is not None:
                        record[key] = new
        for record in reg.declarations("control"):
            # a single-point control has no profile: the annotation came
            # from the dynamic declaration and describes nothing here
            record.pop("profile", None)
        reg.record_transformation(
            "drto.dynamic_to_steady_state",
            removed=", ".join(removed) if removed else "(nothing to remove)",
            collapsed=f"{len(tvars)} Vars and {n_cons} Constraints to a "
            f"single point",
            derivatives=f"{sum(len(dv) for dv in derivs)} derivative "
            f"references replaced by zero",
            **(
                {"discarded": f"{n_artifacts} discretization artifacts"}
                if n_artifacts
                else {}
            ),
        )
