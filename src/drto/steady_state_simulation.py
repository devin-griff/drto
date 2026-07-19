# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Steady-state simulation: ``drto.steady_state_simulation`` (feature 008).

Reduces the model to steady state with the controls fixed and a zero
objective: the square problem whose solution is the equilibrium under the
given inputs. A dynamic model (horizon and dynamics declared) first composes
``drto.dynamic_to_steady_state`` (feature 005); a model authored directly as
steady-state skips the reduction. Either way the declared controls are
fixed, at supplied values or at the values they already hold, the declared
stage costs leave the model (a simulation carries no cost), and
``drto.build_objective`` installs the simulation's constant-zero objective.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory

from drto.info import info
from drto.objective import build_objective


@TransformationFactory.register(
    "drto.steady_state_simulation",
    doc="Reduce to steady state, fix the controls, and install the zero "
    "objective: the fixed-input equilibrium solve (drto).",
)
class SteadyStateSimulationTransformation(Transformation):
    """The steady-state simulation mode; see the module docstring.

    Options: ``controls`` maps a declared control (the component, or its
    name) to the value it is fixed at; controls not in the mapping fix at
    the value they already hold. Components from the source model resolve
    by name, so ``create_using(m, controls={m.u: 0.3})`` works on the
    clone. A scalar Var is unhashable as a plain dict key, so a control on
    an already-steady model goes in by name (or a ``ComponentMap``).
    """

    CONFIG = ConfigDict("drto.steady_state_simulation")
    CONFIG.declare(
        "controls",
        ConfigValue(
            default=None,
            description="Mapping of declared control (component or name) to "
            "the value it is fixed at. Controls not in the mapping fix at "
            "the value they already hold.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        reg = info(model)
        if not reg.has_declaration("state"):
            raise ValueError(
                "drto: steady_state_simulation requires declared states "
                "(drto.state first)."
            )
        if reg.has_declaration("horizon") and reg.has_declaration("dynamics"):
            TransformationFactory("drto.dynamic_to_steady_state").apply_to(model)

        # a simulation carries no cost: the stage-cost equations leave the
        # model (the reduction already removed the terminal pieces on the
        # dynamic path), and their cost variables are left unused
        dropped = []
        for kind in ("tracking_stage_cost", "economic_stage_cost"):
            for record in reg.declarations(kind):
                comp = record["component"]
                if comp.parent_block() is not None:
                    comp.parent_block().del_component(comp)
                dropped.append(kind.split("_")[0])
            # same-package registry surgery, matching the reduction's removals
            reg._declarations.pop(kind, None)
        # the steady-state pairings serve the costs, the endpoint pin, and
        # the optimization mode's write-back, none of which a simulation
        # has: the records go, the target Params stay (the user's components)
        for kind in ("steady_state", "steady_state_control"):
            reg._declarations.pop(kind, None)

        # resolve the requested values against THIS model: create_using
        # hands keys from the source model, and the reduction above replaces
        # the control components, so names are the stable handle
        declared = {c.name: c for c in reg.components("control")}
        requested = {}
        for key, val in (config.controls or {}).items():
            name = key if isinstance(key, str) else key.name
            if name not in declared:
                raise ValueError(
                    f"drto: steady_state_simulation got a value for "
                    f"'{name}', which is not a declared control; declared: "
                    f"{', '.join(declared) or '(none)'}."
                )
            requested[name] = val

        fixed = []
        for name, comp in declared.items():
            for vd in comp.values() if comp.is_indexed() else (comp,):
                if name in requested:
                    vd.set_value(requested[name])
                elif vd.value is None:
                    raise ValueError(
                        f"drto: steady_state_simulation fixes '{name}' at "
                        f"the value it already holds, but it has none; pass "
                        f"controls={{{name}: value}} or initialize it."
                    )
                vd.fix()
            shown = requested.get(name)
            fixed.append(f"{name}={shown}" if shown is not None else f"{name} (held)")

        build_objective(model, zero=True)
        reg.record_transformation(
            "drto.steady_state_simulation",
            controls=", ".join(fixed) if fixed else "(none declared)",
            **(
                {"dropped": f"{' and '.join(sorted(set(dropped)))} stage cost"}
                if dropped
                else {}
            ),
        )
        return model
