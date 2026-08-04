# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The steady-state reduction: ``drto.dynamic_to_steady_state`` (feature 005).

Reduces a declared dynamic model to its steady-state form: time collapses to
a single point, each declared state's time derivative collapses with it and is
fixed at zero, and the initial condition, terminal constraint, and terminal
cost leave the model. The result is the equilibrium system, the declared
dynamics reading as written with ``dz/dt`` pinned at zero, algebraic relations
intact (a derivative-carrying energy balance keeps its form, its derivative
fixed), and a per-sample stage cost becomes the single-point cost that
``drto.build_objective`` assembles for the steady modes.

A ``Block(time)`` family (the IDAES property-block idiom) collapses to its
single steady member: the ``t0`` member stays as written, its variables and
internal equations untouched, and the other members leave the model with
their contents (feature 021). A time-indexed Reference is a view, not a
variable: it collapses to a view of the surviving member (a Port entry) or
of the collapsed Var (an IDAES ``heat_duty``), never to a fresh independent
Var, and Ports keep pointing at their referents.

A derivative is fixed, not eliminated: ``dz/dt = 0`` is what steady state
means, so pinning it keeps the user's equations readable instead of leaving a
side missing, and the solver folds a fixed Var in as a constant. Pyomo cannot
hold a DerivativeVar that is not indexed by a ContinuousSet, and the time set
leaves the model, so the collapsed derivative is a plain scalar Var of the same
name. There are still no ``dz/dt == 0`` rows: the Var is fixed, not constrained.
The transform applies to the declared or discretized model,
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
    Block,
    Constraint,
    Expression,
    Objective,
    Reference,
    Transformation,
    TransformationFactory,
    Var,
)
from pyomo.core.expr import identify_variables, replace_expressions
from pyomo.dae import DerivativeVar
from pyomo.network import Port

from drto.declarations import _dynamics_sides, pyomo_cvp_available
from drto.infinite_horizon import _split_index, _time_index
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
        # a state may be a Reference over a member subset of an indexed Var
        # (gh #20): a container with any declared member is covered, its
        # dynamics rows accepted and every one of its accumulations pinned
        # at zero, including the entries never declared as states (the
        # water holdup), since steady state holds for them as well
        covered = {id(s) for s in states_set}
        for s in states_set:
            for vd in s.values() if s.is_indexed() else (s,):
                covered.add(id(vd.parent_component()))
        for con in reg.components("dynamics"):
            for cd in con.values() if con.is_indexed() else (con,):
                side, _coeff, _ = _dynamics_sides(cd, time, "dynamic_to_steady_state")
                if id(side.parent_component().get_state_var()) not in covered:
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

        # --- time-indexed Blocks collapse to their single steady member --
        # a Block(time) member is per-time structure (the IDAES
        # property-block idiom): the t0 member is the steady point and
        # stays as written, values, bounds, units, and fixed status
        # untouched; the other members leave the model with their contents
        t0 = time.first()
        tblocks = []
        for B in model.component_objects(Block, active=True):
            pos, _bsubs = _time_index(B, time)
            if pos is None:
                continue
            bd = B.parent_block()
            while bd is not None and bd is not model:
                pc = bd.parent_component()
                if _time_index(pc, time)[0] is not None:
                    raise ValueError(
                        f"drto: dynamic_to_steady_state cannot reduce "
                        f"'{B.name}': it is a time-indexed Block nested in "
                        f"another time-indexed Block ('{pc.name}'), which "
                        f"is not supported."
                    )
                bd = bd.parent_block()
            if len(_bsubs) != 1:
                raise ValueError(
                    f"drto: dynamic_to_steady_state cannot reduce "
                    f"'{B.name}': it is indexed by more than the declared "
                    f"time set, which is not supported."
                )
            tblocks.append(B)
        n_members = 0
        for B in tblocks:
            for t in [t for t in B.keys() if t != t0]:
                del B[t]
                n_members += 1

        # --- time-indexed References leave the Var collapse ---------------
        # a Reference is a view, not a variable: it collapses to a view of
        # the surviving member (a Port entry) or of the collapsed Var (an
        # IDAES heat_duty), never to a fresh independent Var. The t0 slice
        # is recorded here and rebuilt after the collapse maps its referents.
        refs = []
        for comp in list(model.component_objects(Var, active=True)):
            if not comp.is_reference():
                continue
            pos, subs = _time_index(comp, time)
            if pos is None:
                continue
            entries = {}
            for idx, vd in comp.items():
                o, t = _split_index(idx, pos, len(subs))
                if t == t0:
                    entries[o] = vd
            refs.append((comp.local_name, comp.parent_block(), id(comp), entries))
            comp.parent_block().del_component(comp)

        # --- no member may span more than one time point ----------------
        for con in model.component_objects(Constraint, active=True):
            for cd in con.values() if con.is_indexed() else (con,):
                per_comp = {}
                for v in identify_variables(cd.expr, include_fixed=True):
                    comp = v.parent_component()
                    if isinstance(comp, DerivativeVar):
                        continue  # collapses to one point like the states
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

        # --- the declared states' time derivatives -----------------------
        # a DerivativeVar is its own ctype before discretization and is
        # reclassified to Var by pyomo.dae afterward: scan both
        seen, derivs = set(), []
        for query in (DerivativeVar, Var):
            for dv in model.component_objects(query):
                if (
                    isinstance(dv, DerivativeVar)
                    and id(dv) not in seen
                    and dv.get_continuousset_list() == [time]
                    and id(dv.get_state_var()) in covered
                ):
                    seen.add(id(dv))
                    derivs.append(dv)
        deriv_ids = {id(dv) for dv in derivs}
        n_derivs = len(derivs)

        # --- collapse the time-indexed Vars, the derivatives included ----
        # a derivative collapses like any other time-indexed Var and is then
        # fixed at zero: dz/dt = 0 is what steady state means, so the declared
        # dynamics stay readable as written instead of losing a side
        submap = {}
        replaced = {}
        tvars, tvar_seen = [], set()
        for query in (DerivativeVar, Var):
            for comp in model.component_objects(query, active=True):
                if id(comp) not in tvar_seen and _time_index(comp, time)[0] is not None:
                    tvar_seen.add(id(comp))
                    tvars.append(comp)
        for comp in tvars:
            pos, subs = _time_index(comp, time)
            others = [s for n, s in enumerate(subs) if n != pos]
            name, parent = comp.local_name, comp.parent_block()
            attrs, members = {}, {}
            for idx, vd in comp.items():
                o, t = _split_index(idx, pos, len(subs))
                members[(o, t)] = vd
                if t == t0:
                    attrs[o] = (vd.domain, vd.lb, vd.ub, vd.value, vd.fixed)
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
                dom, lb, ub, val, _ = attrs[()]
                new = Var(domain=dom, bounds=(lb, ub), initialize=val)
            parent.add_component(name, new)
            if id(comp) in deriv_ids:
                # zero at steady state, by definition: every accumulation
                # rests, including the entries never declared as states,
                # which is what closes their balance equations at the
                # point (the water balance determining the outlet flow)
                for vd in new.values() if new.is_indexed() else (new,):
                    vd.set_value(0.0)
                    vd.fix()
            else:
                # a fixed input stays fixed through the collapse
                for o, a in attrs.items():
                    if a[4]:
                        (new[o] if o else new).fix()
            replaced[id(comp)] = new
            for (o, t), vd in members.items():
                submap[id(vd)] = new[o] if o else new

        # --- rebuild the References onto the collapsed model --------------
        for name, parent, old_id, entries in refs:
            entries = {o: submap.get(id(vd), vd) for o, vd in entries.items()}
            if list(entries) == [()]:
                new = Reference(entries[()])
            else:
                new = Reference(
                    {(o[0] if len(o) == 1 else o): vd for o, vd in entries.items()}
                )
            parent.add_component(name, new)
            replaced[old_id] = new
        # a Port holds its entries by object: swap in the rebuilt views
        for port in model.component_objects(Port, active=True):
            for pname, item in list(port.vars.items()):
                new = replaced.get(id(item))
                if new is not None:
                    port.vars[pname] = new

        # --- collapse the constraints ------------------------------------
        stage_cons = ComponentSet()
        for kind in _STAGE_KINDS:
            stage_cons.update(reg.components(kind))
        n_cons = 0
        for con in list(model.component_objects(Constraint, active=True)):
            pos, subs = _time_index(con, time)
            name, parent = con.local_name, con.parent_block()
            if pos is not None:
                # one member per other-combo: the t0 representative's
                # expression, since only the t0 Block members survive; a
                # family with no t0 member falls back to its first
                others = [s for n, s in enumerate(subs) if n != pos]
                chosen = {}
                for idx, cd in con.items():
                    o, t = _split_index(idx, pos, len(subs))
                    if t == t0 or o not in chosen:
                        chosen[o] = cd
                reps = {
                    o: replace_expressions(cd.expr, submap) for o, cd in chosen.items()
                }
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
        for kind in ("control", "disturbance"):
            for record in reg.declarations(kind):
                # a single-point control or disturbance has no profile: the
                # annotation came from the dynamic declaration and describes
                # nothing here
                record.pop("profile", None)
        reg.record_transformation(
            "drto.dynamic_to_steady_state",
            removed=", ".join(removed) if removed else "(nothing to remove)",
            collapsed=f"{len(tvars)} Vars and {n_cons} Constraints to a "
            f"single point",
            derivatives=f"{n_derivs} fixed at zero",
            **(
                {
                    "blocks": f"{len(tblocks)} time-indexed Block(s) collapsed "
                    f"to the steady member, {n_members} member(s) removed"
                }
                if tblocks
                else {}
            ),
            **(
                {"discarded": f"{n_artifacts} discretization artifacts"}
                if n_artifacts
                else {}
            ),
        )
