# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Apply the declared profiles: ``drto.parameterize`` (feature 017).

A thin drto-native wrapper over pyomo-cvp's declaration-mode
``cvp.parameterize``: applies every profile recorded by ``drto.control`` and
``drto.disturbance``, then repairs the registry, since pyomo-cvp parameterizes
by *replacing* the component and the registry's records would otherwise point
at detached components. The mode transforms call this as one of their steps; a
standalone workflow calls it directly and never touches the cvp namespace.
"""
from pyomo.common.config import ConfigDict
from pyomo.core import Transformation, TransformationFactory

from drto.info import info


@TransformationFactory.register(
    "drto.parameterize",
    doc="Apply the declared control profiles (delegates to pyomo-cvp).",
)
class ParameterizeTransformation(Transformation):
    """Apply every pending declared control profile; see the module docstring."""

    CONFIG = ConfigDict("drto.parameterize")

    def _apply_to(self, model, **kwds):
        self.CONFIG(kwds)  # no options; unknown keywords error
        reg = info(model)
        # controls and disturbances both carry a pyomo-cvp profile
        records = list(reg.declarations("control")) + list(
            reg.declarations("disturbance")
        )
        if not records:
            raise ValueError("drto: nothing to parameterize; declare a control first.")
        names = [r["component"].name for r in records]
        try:
            TransformationFactory("cvp.parameterize").apply_to(model)
        except RuntimeError as err:
            raise ValueError(
                "drto: no profiles to apply: the declared profiles were "
                "already applied."
            ) from err
        # cvp replaced the profiled components; point the registry at the live
        # replacements so drto.info and later transforms see the model,
        # including the steady-state pairings that own a replaced control
        replaced = {}
        for record, name in zip(records, names):
            replacement = model.find_component(name)
            if replacement is not None:
                replaced[id(record["component"])] = replacement
                record["component"] = replacement
        for target in reg.declarations("steady_state_control"):
            replacement = replaced.get(id(target.get("of")))
            if replacement is not None:
                target["of"] = replacement
        # the terminal segment's records pair each declared component with
        # its copy. "of" stays the component as declared, which is what
        # drto.warm_start_dynamic shifts a replaced Reference's underlying
        # members through; the replacement is recorded beside it so a
        # consumer holding the live component can still reach the copy
        # (gh #70, the missing tail points in plot_controls)
        for record in reg._segment_records():
            replacement = replaced.get(id(record.get("of")))
            if replacement is not None:
                record["live"] = replacement
        reg.record_transformation(
            "drto.parameterize",
            profiles=", ".join(
                f"{name} ({record.get('profile')})"
                for record, name in zip(records, names)
            ),
        )
