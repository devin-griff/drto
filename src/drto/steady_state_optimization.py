# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Steady-state optimization: ``drto.steady_state_optimization`` (feature 009).

Economic RTO: reduces the model to steady state with the declared controls
free and optimizes the economic objective over them, giving the optimal steady
operating point. A dynamic model (horizon and dynamics declared) first
composes ``drto.dynamic_to_steady_state`` (feature 005); a model authored
directly as steady-state skips the reduction.

The cost equations stay, unlike the simulation modes: this mode needs them. A
declared tracking stage cost is kept rather than dropped, since it regularizes
the economic optimum toward a known operating point, the RTO-layer equivalent
of move suppression. With both cost kinds declared, ``tracking_weight`` scales
the tracking side, as in ``drto.dynamic_optimization`` (feature 006).

The estimation-category declarations (feature 018) are neutralized before the
reduction, through the routine shared with the other control-side modes. That
matters more here than in a simulation: a free disturbance would become a
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

#: The declarations the transform requires. No horizon or dynamics: the user
#: may author the model as steady-state.
_REQUIRED = ("state", "control", "economic_stage_cost")

#: Both stage-cost kinds; the tracking weight applies only with both present.
_STAGE_KINDS = ("tracking_stage_cost", "economic_stage_cost")


@TransformationFactory.register(
    "drto.steady_state_optimization",
    doc="Reduce to steady state and optimize the economic objective over the "
    "free controls: the economic RTO point (drto).",
)
class SteadyStateOptimizationTransformation(Transformation):
    """The steady-state optimization mode; see the module docstring.

    Options: ``tracking_weight`` weights a declared tracking stage cost, and
    applies only when both a tracking and an economic stage cost are declared.

    ``apply_to`` assembles in place; ``create_using`` assembles a clone and
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
        if missing:
            raise ValueError(
                f"drto: steady_state_optimization requires the declarations "
                f"{', '.join(_REQUIRED)}; missing: {', '.join(missing)}."
            )

        # before the reduction, which collapses the control-side costs and not
        # the window-based estimation costs
        outcome = _neutralize_estimation(reg, "steady_state_optimization")

        if reg.has_declaration("horizon") and reg.has_declaration("dynamics"):
            TransformationFactory("drto.dynamic_to_steady_state").apply_to(model)

        # the process noise is off in the RTO point: a free disturbance would
        # be a decision the optimizer exploits. Fixed at zero after the
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
                weighted if weighted is not None else "(economic cost only)"
            ),
            **({"disturbances": ", ".join(noise)} if noise else {}),
            **outcome,
        )
        return model
