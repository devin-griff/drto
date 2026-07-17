# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Apply the declared control profiles: ``drto.parameterize`` (feature 017).

A thin drto-native wrapper over pyomo-cvp's declaration-mode
``cvp.parameterize``: applies every profile recorded by ``drto.control``,
then repairs the registry, since pyomo-cvp parameterizes by *replacing* the
control component and the registry's records would otherwise point at
detached components. The mode transforms call this as one of their steps; a
standalone workflow calls it directly and never touches the cvp namespace.
"""
from pyomo.common.config import ConfigDict
from pyomo.core import Transformation, TransformationFactory

from drto.info import info


@TransformationFactory.register("drto.parameterize", doc="Apply the declared control profiles (delegates to pyomo-cvp).")
class ParameterizeTransformation(Transformation):
    """Apply every pending declared control profile; see the module docstring."""

    CONFIG = ConfigDict("drto.parameterize")

    def _apply_to(self, model, **kwds):
        self.CONFIG(kwds)  # no options; unknown keywords error
        reg = info(model)
        records = reg.declarations("control")
        if not records:
            raise ValueError("drto: no controls declared; drto.control first.")
        names = [r["component"].name for r in records]
        try:
            TransformationFactory("cvp.parameterize").apply_to(model)
        except RuntimeError as err:
            raise ValueError("drto: no control profiles to apply: the declared profiles were already applied.") from err
        # cvp replaced the control components; point the registry at the
        # live replacements so drto.info and later transforms see the model,
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
        reg.record_transformation("drto.parameterize", controls=", ".join(f"{name} ({record.get('profile')})" for record, name in zip(records, names)))
