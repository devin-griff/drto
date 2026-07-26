# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic simulation: ``drto.dynamic_simulation`` (feature 007).

Prepares the declared dynamic model to be solved forward over the horizon: the
declared control profiles are applied, the controls are fixed at the values
they hold or at supplied values, and ``drto.build_objective`` installs the
constant-zero objective. The mode frees nothing, so the result is the square
forward integration of the model as declared, and the horizon is kept.

The profiles are applied before the controls are fixed, so the simulated input
takes the shape the model declared; the user chooses that shape at declaration
time through ``control(profile=...)``.

A simulation carries no cost, so the declared stage and terminal cost
equations leave the model, as in ``drto.steady_state_simulation`` (feature
008). The estimation costs and measurements are neutralized through the routine
shared with the other control-side modes, and each disturbance is fixed at its
realization (default zero), the same way the controls are fixed, so the plant
can be driven by a supplied noise sequence and the system stays square.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory

from drto.dynamic_optimization import (
    _fix_disturbances,
    _members,
    _neutralize_estimation,
    _spread,
)
from drto.info import info
from drto.objective import build_objective

#: The declarations the transform requires. A forward integration needs the
#: initial state pinned, or the horizon problem is not square.
_REQUIRED = ("horizon", "state", "dynamics", "control", "initial_condition")

#: The optimization-only constructs a simulation sheds: it carries no cost, and
#: a terminal constraint would over-constrain the square forward integration.
#: Shared with ``drto.steady_state_simulation`` (feature 008) through
#: ``_shed_optimization_constructs`` so the two modes cannot diverge.
_SIMULATION_SHED = (
    "tracking_stage_cost",
    "economic_stage_cost",
    "tracking_terminal_cost",
    "terminal_constraint",
)


def _shed_optimization_constructs(reg):
    """Delete the optimization-only cost and constraint declarations.

    The stage costs, the terminal cost, and the terminal constraint leave the
    model and their records are purged; the cost variables are left unused.
    Returns the dropped display names for the transformation log.
    """
    dropped = []
    for kind in _SIMULATION_SHED:
        for record in reg.declarations(kind):
            comp = record["component"]
            if comp.parent_block() is not None:
                comp.parent_block().del_component(comp)
        if reg.has_declaration(kind):
            dropped.append(kind.replace("_", " "))
        reg._declarations.pop(kind, None)
    return dropped


@TransformationFactory.register(
    "drto.dynamic_simulation",
    doc="Fix the controls and prepare the declared model for forward "
    "integration over the horizon (drto).",
)
class DynamicSimulationTransformation(Transformation):
    """The dynamic simulation mode; see the module docstring.

    Options: ``controls`` maps a declared control (the component, or its name)
    to what it is fixed at, either a constant held across the horizon or one
    value per free point the applied profile leaves. Controls not in the
    mapping fix at the values they already hold. ``disturbances`` maps a
    declared disturbance the same way, the plant's realized noise; a
    disturbance not in the mapping is fixed at zero. Components from the source
    model resolve by name, so ``create_using(m, controls={m.u: 0.3})`` works on
    the clone.

    ``apply_to`` prepares in place; ``create_using`` prepares a clone and
    leaves the source model alone.
    """

    CONFIG = ConfigDict("drto.dynamic_simulation")
    CONFIG.declare(
        "controls",
        ConfigValue(
            default=None,
            description="Mapping of declared control (component or name) to "
            "the value it is fixed at: a constant held across the horizon, or "
            "one value per free point of the applied profile. Controls not in "
            "the mapping fix at the values they already hold.",
        ),
    )
    CONFIG.declare(
        "disturbances",
        ConfigValue(
            default=None,
            description="Mapping of declared disturbance (component or name) to "
            "its realization: a constant held across the horizon, or one value "
            "per free point of the applied profile. A disturbance not in the "
            "mapping is fixed at zero.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        # resolve component keys to names before the profiles are applied:
        # parameterizing replaces the very components the keys point at,
        # detaching them, and a detached component's name degrades to its
        # local name
        controls = {
            (k if isinstance(k, str) else k.name): v
            for k, v in (config.controls or {}).items()
        }
        disturbances = {
            (k if isinstance(k, str) else k.name): v
            for k, v in (config.disturbances or {}).items()
        }
        reg = info(model)
        missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
        if missing:
            raise ValueError(
                f"drto: dynamic_simulation requires the declarations "
                f"{', '.join(_REQUIRED)}; missing: {', '.join(missing)}."
            )

        # a simulation carries no cost and no terminal set: the cost equations
        # and the terminal constraint leave the model
        dropped = _shed_optimization_constructs(reg)

        outcome = _neutralize_estimation(reg, "dynamic_simulation")

        # the declared profiles shape the simulated input, so they are applied
        # before the controls and disturbances are fixed
        TransformationFactory("drto.parameterize").apply_to(model)
        fixed = self._fix_controls(reg, controls)
        noise = _fix_disturbances(reg, disturbances, "dynamic_simulation")

        build_objective(model, zero=True)
        reg.record_transformation(
            "drto.dynamic_simulation",
            horizon="kept",
            controls=", ".join(fixed) if fixed else "(none declared)",
            **({"disturbances": ", ".join(noise)} if noise else {}),
            **({"dropped": ", ".join(dropped)} if dropped else {}),
            **outcome,
        )
        return model

    def _fix_controls(self, reg, requested):
        """Fix each declared control, at a supplied value or the one it holds.

        Resolution is by name: ``create_using`` hands keys from the source
        model, and parameterizing replaces the control components, so the name
        is the stable handle.
        """
        declared = {c.name: c for c in reg.components("control")}
        wanted = {}
        for key, val in requested.items():
            name = key if isinstance(key, str) else key.name
            if name not in declared:
                raise ValueError(
                    f"drto: dynamic_simulation got a value for '{name}', "
                    f"which is not a declared control; declared: "
                    f"{', '.join(declared) or '(none)'}."
                )
            wanted[name] = val

        fixed = []
        for name, comp in declared.items():
            members = list(_members(comp))
            if name in wanted:
                val = wanted[name]
                values = _spread(val, len(members), name, "dynamic_simulation")
                for vd, v in zip(members, values):
                    vd.set_value(v)
            for vd in members:
                if vd.value is None:
                    raise ValueError(
                        f"drto: dynamic_simulation fixes '{name}' at the "
                        f"value it already holds, but it has none; pass "
                        f"controls={{{name}: value}} or initialize it."
                    )
                vd.fix()
            fixed.append(
                f"{name}={wanted[name]}" if name in wanted else f"{name} (held)"
            )
        return fixed
