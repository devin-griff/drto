# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Steady-state optimization: ``drto.steady_state_optimization`` (feature 009).

Economic RTO: reduces the model to steady state with the declared controls
free and optimizes the economic objective over them, giving the optimal steady
operating point. A dynamic model (horizon and dynamics declared) first
composes ``drto.dynamic_to_steady_state`` (feature 005), and a model authored
directly as steady-state skips the reduction.

The cost equations stay, unlike the simulation modes, since this mode needs
them. A declared tracking stage cost is kept rather than dropped, since it
regularizes the economic optimum toward a known operating point, the RTO-layer
equivalent of move suppression. With both cost kinds declared,
``tracking_weight`` scales the tracking side, as in
``drto.dynamic_optimization`` (feature 006). With only a tracking stage cost
declared, the objective is that cost alone, the steady point nearest the
declared targets.

The estimation-category declarations (feature 018) are neutralized before the
reduction, through the routine shared with the other control-side modes. That
matters more here than in a simulation. A free disturbance would become a
decision variable the optimizer exploits to lower the economic cost, so the
operating point would be optimized against fictitious noise.

Writing the solution back into the declared steady-state targets is an
algorithmic step outside this transform, which shapes the model and does
nothing after a solve. The pairings are left intact, since they are the record
that makes such a write-back possible.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory

from drto.dynamic_optimization import _fix_disturbances, _neutralize_estimation
from drto.info import info
from drto.objective import build_objective

#: The declarations the transform requires, alongside a stage cost of either
#: kind. No horizon or dynamics, since the user may author the model as
#: steady-state.
_REQUIRED = ("state", "control")

#: Both stage-cost kinds. The tracking weight applies only with both present.
_STAGE_KINDS = ("tracking_stage_cost", "economic_stage_cost")


def steady_state_optimization(build, tracking_weight=None):
    """Build a model and assemble the steady-state optimization.

    Takes the model statement rather than a model, so a script that already
    has a builder reaches its RTO problem in one call. The builder contract
    is feature 006's: ``build`` returns a declared, undiscretized model, its
    first two parameters are the interval count and the sampling time, and
    every parameter has a default, so the bare ``build()`` this makes is
    legal. The steady modes pass no ``N`` and no ``h``, since the reduction
    collapses the grid either way.

    Nothing is discretized on this path. The registered transformation
    composes ``drto.dynamic_to_steady_state`` (feature 005) for a model
    declaring a horizon and dynamics, and a statement that constructs its
    steady form natively takes that reduction's skip.

    Parameters
    ----------
    build : callable
        The model statement, called with no arguments.
    tracking_weight : float, optional
        Passed to the registered transformation when given, weighting a
        declared tracking stage cost against the economic one.

    Returns
    -------
    Block
        The RTO problem. This is the object ``build`` returned, since the
        function owns the model it just built and transforms it in place.
        A caller holding a model of its own keeps
        ``TransformationFactory('drto.steady_state_optimization').create_using``
        for the form that leaves the source unchanged.

    Examples
    --------
    ::

        rto = drto.steady_state_optimization(build)
    """
    m = build()
    opts = {} if tracking_weight is None else {"tracking_weight": tracking_weight}
    TransformationFactory("drto.steady_state_optimization").apply_to(m, **opts)
    return m


@TransformationFactory.register(
    "drto.steady_state_optimization",
    doc="Reduce to steady state and optimize the economic objective over the "
    "free controls, the economic RTO point (drto).",
)
class SteadyStateOptimizationTransformation(Transformation):
    """The steady-state optimization mode. See the module docstring.

    Options: ``tracking_weight`` weights a declared tracking stage cost, and
    applies only when both a tracking and an economic stage cost are declared.

    ``apply_to`` assembles in place. ``create_using`` assembles a clone and
    leaves the source model alone.
    """

    CONFIG = ConfigDict("drto.steady_state_optimization")
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
        if not any(reg.has_declaration(k) for k in _STAGE_KINDS):
            missing.append("a stage cost of either kind")
        if missing:
            raise ValueError(
                f"drto: steady_state_optimization requires state, control, "
                f"and a stage cost of either kind. Missing: "
                f"{', '.join(missing)}."
            )

        # before the reduction, which collapses the control-side costs and not
        # the window-based estimation costs
        outcome = _neutralize_estimation(reg, "steady_state_optimization")

        if reg.has_declaration("horizon") and reg.has_declaration("dynamics"):
            TransformationFactory("drto.dynamic_to_steady_state").apply_to(model)

        # the process noise is off in the RTO point, since a free disturbance
        # would be a decision the optimizer exploits. Fixed at zero after the
        # reduction collapses it to a single point
        noise = _fix_disturbances(reg, {}, "steady_state_optimization")

        # build_objective reads the weight off the group's record
        weighted = None
        if all(reg.has_declaration(k) for k in _STAGE_KINDS):
            for record in reg.declarations("tracking_stage_cost"):
                record["weight"] = config.tracking_weight
            weighted = config.tracking_weight

        build_objective(model)
        reg.record_transformation(
            "drto.steady_state_optimization",
            controls="free",
            tracking_weight=(
                weighted if weighted is not None else "(one stage cost declared)"
            ),
            **({"disturbances": ", ".join(noise)} if noise else {}),
            **outcome,
        )
        return model
