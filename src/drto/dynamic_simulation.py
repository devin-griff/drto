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
008). The estimation-category declarations are neutralized through the routine
shared with ``drto.dynamic_optimization`` (feature 006), which also protects
squareness: a free disturbance would leave the system underdetermined.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory

from drto.dynamic_optimization import _members, _neutralize_estimation
from drto.info import info
from drto.objective import build_objective

#: The declarations the transform requires. A forward integration needs the
#: initial state pinned, or the horizon problem is not square.
_REQUIRED = ("horizon", "state", "dynamics", "control", "initial_condition")

#: The cost kinds a simulation carries no use for.
_COST_KINDS = ("tracking_stage_cost", "economic_stage_cost", "tracking_terminal_cost")


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
    mapping fix at the values they already hold. Components from the source
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

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        reg = info(model)
        missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
        if missing:
            raise ValueError(
                f"drto: dynamic_simulation requires the declarations "
                f"{', '.join(_REQUIRED)}; missing: {', '.join(missing)}."
            )

        # a simulation carries no cost: the cost equations leave the model and
        # their cost variables are left unused, as in the steady-state
        # simulation
        dropped = []
        for kind in _COST_KINDS:
            for record in reg.declarations(kind):
                comp = record["component"]
                if comp.parent_block() is not None:
                    comp.parent_block().del_component(comp)
            if reg.has_declaration(kind):
                dropped.append(kind.replace("_", " "))
            reg._declarations.pop(kind, None)

        outcome = _neutralize_estimation(model, reg, "dynamic_simulation")

        # the declared profiles shape the simulated input, so they are applied
        # before the controls are fixed
        TransformationFactory("drto.parameterize").apply_to(model)
        fixed = self._fix_controls(reg, config.controls or {})

        build_objective(model, zero=True)
        reg.record_transformation(
            "drto.dynamic_simulation",
            horizon="kept",
            controls=", ".join(fixed) if fixed else "(none declared)",
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
                values = self._spread(val, len(members), name)
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

    def _spread(self, val, n_free, name):
        """Return ``n_free`` values from a constant or a per-point sequence."""
        if isinstance(val, (list, tuple)):
            if len(val) != n_free:
                raise ValueError(
                    f"drto: dynamic_simulation got {len(val)} values for "
                    f"'{name}', which has {n_free} free points after its "
                    f"profile is applied; pass a constant or one value each."
                )
            return list(val)
        return [val] * n_free
