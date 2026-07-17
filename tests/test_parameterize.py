# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 017: applying the declared control profiles (drto.parameterize)."""
import pyomo.environ as pyo
import pytest

import drto
from test_declarations import declared_model


def discretized():
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU")
    return m


def test_applies_the_declared_profiles():
    m = discretized()
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    # piecewise constant: one free value per finite element
    assert len(m.u) == 4


def test_refreshes_the_registry_to_the_replacement_component():
    m = discretized()
    reg = drto.info(m)
    stale = reg.components("control")[0]
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    fresh = reg.components("control")[0]
    assert fresh is m.u
    assert fresh is not stale
    assert stale.parent_block() is None  # the old component is detached


def test_records_in_the_transformation_log():
    m = discretized()
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    reg = drto.info(m)
    assert reg.has_transformation("drto.parameterize")
    assert reg.transformations[-1]["outcome"]["controls"] == "u (piecewise_constant)"


def test_errors_without_declared_controls():
    m = pyo.ConcreteModel()
    with pytest.raises(ValueError, match="control first"):
        pyo.TransformationFactory("drto.parameterize").apply_to(m)


def test_second_application_errors():
    m = discretized()
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    with pytest.raises(ValueError, match="already applied"):
        pyo.TransformationFactory("drto.parameterize").apply_to(m)


def test_create_using_leaves_the_source_alone():
    m = discretized()
    m2 = pyo.TransformationFactory("drto.parameterize").create_using(m)
    assert len(m2.u) == 4
    assert len(m.u) > 4  # the source keeps every collocation copy
    assert drto.info(m2).has_transformation("drto.parameterize")
    assert not drto.info(m).has_transformation("drto.parameterize")


def test_refreshes_the_steady_state_pairing_too():
    m = discretized()
    reg = drto.info(m)
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    (target,) = reg.declarations("steady_state_control")
    assert target["of"] is m.u  # the live replacement, not the detached Var
    (state_target,) = reg.declarations("steady_state")
    assert state_target["of"] is m.z
