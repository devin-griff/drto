# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Objective assembly: ``drto.build_objective`` (feature 003).

One routine owns objective installation for every mode. The default outcome
assembles the objective from the live registered cost groups, each with its
group's weights; the marked case, selected by the explicit ``zero`` option
that the simulation transforms pass, installs a constant-zero objective. The
function never infers a mode: callers request outcomes.

Liveness is component presence: a cost term is included when its backing
component is on the model and active at assembly time, so a mode drops a term
just by dropping or deactivating its constraint. Mode weighting travels as
group weights in the registry (a ``weight`` entry on the declaration record),
not as options here, and transforms may register additional ``cost_group``
records carrying their own per-point weights (the infinite-horizon tail,
feature 004).
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Objective, Transformation, TransformationFactory, minimize

from drto.declarations import _is_var_member, _side_matching
from drto.info import info

#: The registry kinds whose cost vars are summed per time point.
_STAGE_KINDS = ("tracking_stage_cost", "economic_stage_cost")

#: The registry kinds whose cost var is a single scalar.
_TERMINAL_KINDS = ("tracking_terminal_cost",)

#: The component name of the installed objective.
_OBJECTIVE_NAME = "drto_objective"


def build_objective(m, zero=False):
    """Install the model's objective, exactly one active minimize Objective.

    Parameters
    ----------
    m : Block
        The declared model.
    zero : bool, optional
        The marked case: install a constant-zero objective instead of
        assembling the live cost terms. The simulation transforms pass this,
        since a simulation has no cost to assemble.

    Returns
    -------
    Objective
        The installed objective.

    Raises
    ------
    ValueError
        If no cost term is live and ``zero`` was not requested.
    """
    reg = info(m)
    if zero:
        expr, n_terms = 0.0, 0
    else:
        terms = list(_live_cost_terms(reg))
        if not terms:
            raise ValueError(
                "drto: build_objective found no live cost terms; declare a "
                "stage cost (feature 002), or pass zero=True for the "
                "simulation objective."
            )
        expr = sum(w * v for v, w in terms)
        n_terms = len(terms)

    for obj in m.component_data_objects(Objective, active=True):
        obj.deactivate()
    if m.component(_OBJECTIVE_NAME) is not None:
        m.del_component(_OBJECTIVE_NAME)
    m.add_component(_OBJECTIVE_NAME, Objective(expr=expr, sense=minimize))
    reg.record_transformation(
        "drto.build_objective",
        objective="zero" if zero else f"sum of {n_terms} weighted cost terms",
    )
    return m.component(_OBJECTIVE_NAME)


def _live_cost_terms(reg):
    """Yield ``(cost_var, weight)`` for every live registered cost term.

    The declared stage costs contribute their per-point cost var at each
    active member; a terminal cost contributes its scalar cost var; and
    generic ``cost_group`` records contribute their own ``terms`` pairs. A
    record's ``weight`` entry (default 1) scales its group.
    """
    samples = None
    time_records = reg.declarations("horizon")
    if time_records:
        samples = set(time_records[0]["samples"])
    for kind in _STAGE_KINDS + _TERMINAL_KINDS:
        stage = kind in _STAGE_KINDS
        for record in reg.declarations(kind):
            con = record["component"]
            if con.parent_block() is None:  # removed from the model
                continue
            weight = record.get("weight", 1)
            members = con.values() if con.is_indexed() else (con,)
            for cd in members:
                if not cd.active:
                    continue
                # the stage-cost sum runs at the sample points: cost members
                # at interior collocation points exist after discretization
                # but are not summed, keeping the finite horizon commensurate
                # with the infinite-horizon tail
                if stage and samples is not None and cd.index() not in samples:
                    continue
                var, _ = _side_matching(
                    cd, _is_var_member, "build_objective", "the cost variable"
                )
                yield var, weight
    for record in reg.declarations("cost_group"):
        comp = record["component"]
        if comp is not None and comp.parent_block() is None:
            continue
        for var, weight in record["terms"]:
            yield var, weight


@TransformationFactory.register(
    "drto.build_objective",
    doc="Assemble the objective from the live registered cost terms (drto).",
)
class BuildObjectiveTransformation(Transformation):
    """The transformation form of :func:`build_objective`.

    Calls the same function, so ``apply_to`` (in place) and ``create_using``
    (a clone) both work::

        TransformationFactory('drto.build_objective').apply_to(m)
        TransformationFactory('drto.build_objective').apply_to(m, zero=True)
    """

    CONFIG = ConfigDict("drto.build_objective")
    CONFIG.declare(
        "zero",
        ConfigValue(
            default=False,
            domain=bool,
            description="Install a constant-zero objective (the simulation "
            "outcome) instead of assembling the live cost terms.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        build_objective(model, zero=config.zero)
