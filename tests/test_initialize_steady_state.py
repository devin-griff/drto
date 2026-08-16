# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 010: drto.initialize_steady_state."""
import pyomo.environ as pyo
import pytest

import drto
from test_declarations import declared_model
from test_infinite_horizon import ref_control_model

pyomo_pounce = pytest.importorskip("pyomo_pounce")


def discretized_model():
    m = declared_model()
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=4, ncp=3, scheme="LAGRANGE-RADAU"
    )
    return m


def steady_authored_model():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    m.u = pyo.Var(initialize=0.25, bounds=(0, 1))

    @m.Constraint()
    def balance(m):
        return m.z == 2 * m.u

    drto.state(m.z)
    drto.control(m.u)
    return m


def test_requires_declared_states():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    with pytest.raises(ValueError, match="requires declared states"):
        drto.initialize_steady_state(m)


def test_steady_path_initializes_in_place():
    m = steady_authored_model()
    report = drto.initialize_steady_state(m)
    assert pyo.value(m.z) == pytest.approx(0.5, abs=1e-8)  # z = 2u at u = 0.25
    assert not m.u.fixed  # the pipeline restores the fixed flags
    assert report.ok
    assert "block" in str(report) or "initialize" in str(report)


def test_a_scale_source_writes_the_factors_first():
    import pyomo_pounce

    m = steady_authored_model()
    seen = []
    real = pyomo_pounce.initialize

    def recording(model, *a, **kw):
        seen.append(model.component("scaling_factor") is not None)
        return real(model, *a, **kw)

    pyomo_pounce.initialize = recording
    try:
        drto.initialize_steady_state(m, scale="bounds")
    finally:
        pyomo_pounce.initialize = real
    assert seen == [True]


def test_dynamic_path_broadcasts_a_reference_control():
    # the inlet idiom: the collapsed copy of a Reference-declared control
    # is a container even over its single member, and the broadcast reads
    # the member, not the container
    m = ref_control_model()
    report = drto.initialize_steady_state(m, controls={"fin": 0.5})
    assert all(pyo.value(m.props[t].f) == pytest.approx(0.5, abs=1e-8) for t in m.t)
    assert all(pyo.value(m.z[t]) == pytest.approx(0.5, abs=1e-8) for t in m.t)
    assert report.n_grid_points == len(m.t)


def test_dynamic_path_broadcasts_flat():
    m = discretized_model()
    report = drto.initialize_steady_state(m, controls={m.u: 0.3})
    # dz/dt = -z + u at rest: z = u = 0.3 at every grid point
    assert all(pyo.value(m.z[t]) == pytest.approx(0.3, abs=1e-8) for t in m.t)
    assert all(pyo.value(m.dzdt[t]) == 0 for t in m.t)
    assert all(pyo.value(m.u[t]) == pytest.approx(0.3) for t in m.t)
    assert report.n_grid_points == len(m.t)
    assert report.n_broadcast_vars >= 3  # z, u, cost
    assert "broadcast" in str(report)
    # structure untouched: the model is still dynamic and unreduced
    assert m.z.is_indexed() and m.component("dzdt") is not None
    assert not drto.info(m).has_transformation("drto.dynamic_to_steady_state")


def test_dynamic_path_requires_discretization():
    m = declared_model()
    with pytest.raises(ValueError, match="must be discretized"):
        drto.initialize_steady_state(m)


def test_dynamic_path_runs_before_the_transforms():
    m = discretized_model()
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    with pytest.raises(ValueError, match="before the dynamic transforms"):
        drto.initialize_steady_state(m)


def test_unknown_control_errors():
    m = steady_authored_model()
    with pytest.raises(ValueError, match="not a declared control"):
        drto.initialize_steady_state(m, controls={"w": 1.0})


def test_valueless_unheld_control_errors():
    m = steady_authored_model()
    m.u.set_value(None)
    with pytest.raises(ValueError, match="has none"):
        drto.initialize_steady_state(m)


def test_non_square_system_raises_with_names():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    m.u = pyo.Var(initialize=0.25)
    m.w = pyo.Var(initialize=0.0)  # a free variable nobody determines

    @m.Constraint()
    def balance(m):
        return m.z == 2 * m.u + m.w

    drto.state(m.z)
    drto.control(m.u)
    with pytest.raises(ValueError, match="non-square.*w"):
        drto.initialize_steady_state(m)


def test_values_survive_the_dynamic_transforms():
    # initialize first, transform after: the flat start propagates
    m = discretized_model()
    drto.initialize_steady_state(m, controls={m.u: 0.3})
    pyo.TransformationFactory("drto.parameterize").apply_to(m)
    assert all(pyo.value(m.u[t]) == pytest.approx(0.3) for t in m.u)


def test_a_scaling_suffix_does_not_change_the_steady_pipeline():
    # the pipeline runs in the model's own units; the suffix stays on
    # the model, unread, for the solves that follow (gh #92)
    m = steady_authored_model()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.z] = 1e-6
    m.scaling_factor[m.balance] = 1e-6
    report = drto.initialize_steady_state(m)
    assert report.ok
    assert pyo.value(m.z) == pytest.approx(0.5, abs=1e-8)
    assert not m.u.fixed
    assert len(m.scaling_factor) == 2


def test_the_pipeline_receives_the_model_itself(monkeypatch):
    # no clone: the pipeline solves the model in place, in its own
    # units, suffix or no suffix (gh #92)
    seen = {}
    real = pyomo_pounce.initialize

    def spy(model, **kw):
        seen["model"] = model
        seen["u"] = model.find_component("u").value
        return real(model, **kw)

    monkeypatch.setattr(pyomo_pounce, "initialize", spy)
    m = steady_authored_model()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.u] = 10.0
    drto.initialize_steady_state(m)
    assert seen["model"] is m
    assert seen["u"] == pytest.approx(0.25)  # its own units throughout
    assert pyo.value(m.u) == pytest.approx(0.25)
    assert pyo.value(m.z) == pytest.approx(0.5, abs=1e-8)


def test_scaling_suffix_scales_the_dynamic_pipeline():
    def built(suffix):
        m = discretized_model()
        if suffix:
            m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            for vd in m.z.values():
                m.scaling_factor[vd] = 1e-6
        return m

    plain, scaled = built(False), built(True)
    drto.initialize_steady_state(plain, controls={plain.u: 0.3})
    drto.initialize_steady_state(scaled, controls={scaled.u: 0.3})
    for vd, svd in zip(
        plain.component_data_objects(pyo.Var),
        scaled.component_data_objects(pyo.Var),
        strict=True,
    ):
        assert (vd.value is None) == (svd.value is None), svd.name
        if vd.value is not None:
            assert svd.value == pytest.approx(vd.value, abs=1e-7), svd.name


def test_a_declared_disturbance_holds_at_zero():
    # process noise is zero in the nominal equilibrium (gh #44): the
    # pipeline holds a declared disturbance at zero for the solve, the
    # control-side convention, and the broadcast lands it at zero with
    # the fixed flags untouched on the model
    from test_infinite_horizon import disturbed_model

    m = disturbed_model()
    for t in m.t:
        m.w[t].set_value(0.7)  # away from zero, to prove the hold
    drto.initialize_steady_state(m)
    for t in m.t:
        assert pyo.value(m.w[t]) == pytest.approx(0.0)
        assert not m.w[t].fixed
    # the equilibrium of dz = u - z + w at w = 0 is z = u
    t0 = m.t.first()
    assert pyo.value(m.z[t0]) == pytest.approx(pyo.value(m.u[t0]), abs=1e-6)
