# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Feature 026: drto.explicit_nmpc_train and the ExplicitNMPC policy."""
import sys

import numpy as np
import pytest

import drto

#: A linear map in scaled coordinates: two inputs, two controls.
A = np.array([[0.6, -0.2], [0.3, 0.8]])
X_BOX = {"a_hat": (0.0, 1.0), "b_hat": (0.0, 2.0)}
U_BOX = {"u1": (0.0, 10.0), "u2": (-5.0, 0.0)}


def toy_dataset(n=64, gradients=True, u_bounds=True, seed=0):
    """Labels from the linear map, with its exact derivatives."""
    rng = np.random.default_rng(seed)
    x_lo, x_hi = np.array(list(X_BOX.values())).T
    u_lo, u_hi = np.array(list(U_BOX.values())).T
    rx, ru = x_hi - x_lo, u_hi - u_lo
    points = []
    for _ in range(n):
        xs = rng.uniform(size=2)
        us = A @ xs
        point = {
            "x": {k: float(v) for k, v in zip(X_BOX, x_lo + xs * rx)},
            "u0": {k: float(v) for k, v in zip(U_BOX, u_lo + us * ru)},
            "V": float(us @ us),
        }
        if gradients:
            point["du0_dx"] = {
                uk: {xk: float(A[i, j] * ru[i] / rx[j]) for j, xk in enumerate(X_BOX)}
                for i, uk in enumerate(U_BOX)
            }
        points.append(point)
    us_mid = A @ np.array([0.5, 0.5])
    config = {
        "n": n,
        "method": "uniform",
        "seed": seed,
        "gradients": gradients,
        "inputs": list(X_BOX),
        "ranges": {k: list(v) for k, v in X_BOX.items()},
        # the steady targets: the map's value at the box midpoint
        "x_ss": {k: (lo + hi) / 2 for k, (lo, hi) in X_BOX.items()},
        "u_ss": {k: float(u_lo[i] + us_mid[i] * ru[i]) for i, k in enumerate(U_BOX)},
    }
    if u_bounds:
        config["u_bounds"] = {k: list(v) for k, v in U_BOX.items()}
    return drto.ExplicitNmpcDataset(config, points, [])


def quick(**kw):
    args = dict(epochs=300, hidden=(16,), lr=1e-2, schedule="flat", seeds=1)
    args.update(kw)
    return drto.explicit_nmpc_train(toy_dataset(), **args)


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_missing_torch_names_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="pip install torch"):
        drto.explicit_nmpc_train(toy_dataset())


def test_sobolev_needs_gradient_labels():
    with pytest.raises(ValueError, match="carries none"):
        drto.explicit_nmpc_train(toy_dataset(gradients=False), epochs=10)


def test_a_dataset_without_control_bounds_errors():
    with pytest.raises(ValueError, match="no control bounds"):
        drto.explicit_nmpc_train(toy_dataset(u_bounds=False), epochs=10)


def test_bad_options_error():
    with pytest.raises(ValueError, match="cosine, flat"):
        quick(schedule="steps")
    with pytest.raises(ValueError, match="tanh, relu, silu, sigmoid"):
        quick(activation="softmax")


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def test_the_policy_learns_the_map():
    policy = quick(epochs=2000)
    x = {"a_hat": 0.5, "b_hat": 1.0}
    xs = np.array([0.5, 0.5])
    expect = A @ xs
    u_lo, u_hi = np.array(list(U_BOX.values())).T
    got = policy(x)
    for i, k in enumerate(U_BOX):
        scaled = (got[k] - u_lo[i]) / (u_hi[i] - u_lo[i])
        assert scaled == pytest.approx(expect[i], abs=0.05)


def test_training_is_reproducible():
    a = quick()
    b = quick()
    assert a.validation_error == b.validation_error
    probe = {"a_hat": 0.3, "b_hat": 0.6}
    assert a(probe) == b(probe)


def test_seeds_keep_the_best_by_validation():
    one = quick(seeds=1)
    three = quick(seeds=3)
    assert three.validation_error <= one.validation_error


def test_the_history_is_recorded():
    policy = quick(epochs=100)
    assert policy.history["epoch"][-1] == 100
    assert len(policy.history["val_loss"]) == len(policy.history["epoch"])
    assert policy.validation_error == min(policy.history["val_loss"])


def test_a_validation_dataset_is_used_as_given():
    vset = toy_dataset(n=16, seed=5)
    policy = drto.explicit_nmpc_train(
        toy_dataset(), validation=vset, epochs=100, hidden=(8,), schedule="flat"
    )
    assert policy.validation_error < 1.0


def test_every_protocol_option_runs():
    quick(schedule="cosine", clip=None, fine_tune=0.0, activation="silu")
    quick(training_loss="value", validation_loss="value", gamma=0.0)
    quick(steady_state_enforced=False)


# ----------------------------------------------------------------------
# the policy object
# ----------------------------------------------------------------------
def test_the_call_checks_its_inputs():
    policy = quick(epochs=50)
    with pytest.raises(ValueError, match="missing: b_hat"):
        policy({"a_hat": 0.5})


def test_save_load_round_trips(tmp_path):
    policy = quick(epochs=100)
    out = tmp_path / "policy.pt"
    policy.save(str(out))
    back = drto.ExplicitNMPC.load(str(out))
    probe = {"a_hat": 0.7, "b_hat": 1.3}
    assert back(probe) == policy(probe)
    assert back.history == policy.history
    assert back.validation_error == policy.validation_error


# ----------------------------------------------------------------------
# the enforced steady state
# ----------------------------------------------------------------------
def test_the_steady_state_is_enforced_exactly(tmp_path):
    policy = quick(epochs=50)
    d = toy_dataset()
    x_ss, u_ss = d.config["x_ss"], d.config["u_ss"]
    got = policy(x_ss)
    for k in u_ss:
        assert got[k] == pytest.approx(u_ss[k], abs=1e-12)
    # the offset survives the round trip through a file
    policy.save(str(tmp_path / "p.pt"))
    back = drto.ExplicitNMPC.load(str(tmp_path / "p.pt"))
    got = back(x_ss)
    for k in u_ss:
        assert got[k] == pytest.approx(u_ss[k], abs=1e-12)


def test_enforcement_needs_the_recorded_targets():
    d = toy_dataset()
    del d.config["x_ss"]
    with pytest.raises(ValueError, match="steady targets"):
        drto.explicit_nmpc_train(d, epochs=10)


def test_a_bad_training_loss_errors():
    with pytest.raises(ValueError, match="sobolev, value"):
        quick(training_loss="both")


# ----------------------------------------------------------------------
# weight decay
# ----------------------------------------------------------------------
def test_weight_decay_shrinks_the_weights():
    plain = quick(weight_decay=0.0)
    decayed = quick(weight_decay=1.0)

    def norm(policy):
        return sum(float((w**2).sum()) for w in policy._model.parameters())

    assert norm(decayed) < norm(plain)


# ----------------------------------------------------------------------
# the validation loss
# ----------------------------------------------------------------------
def test_the_validation_loss_option():
    a = quick(validation_loss="value")
    b = quick(validation_loss="sobolev")
    # the training trajectories are identical, so at every checkpoint the
    # sobolev metric is the value metric plus the gradient term
    assert b.history["val_loss"][0] > a.history["val_loss"][0]
    with pytest.raises(ValueError, match="sobolev, value"):
        quick(validation_loss="both")


def test_the_sobolev_validation_loss_needs_gradient_labels():
    train = toy_dataset(seed=0)
    val = toy_dataset(gradients=False, seed=1)
    with pytest.raises(ValueError, match="carries none"):
        drto.explicit_nmpc_train(train, validation=val, epochs=10)
    policy = drto.explicit_nmpc_train(
        train,
        validation=val,
        validation_loss="value",
        epochs=50,
        hidden=(8,),
        schedule="flat",
    )
    assert policy.validation_error > 0
