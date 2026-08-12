# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit NMPC: data generation and training (feature 026).

``drto.explicit_nmpc_data`` samples the assembled optimization into a
labeled dataset. Each draw writes values into the sampled Params,
cold-starts the model, solves once, and records the first control
action per declared control, the objective, and, when asked, the
derivative of each first action with respect to each sampled Param,
read from the converged factorization the way
``drto.advanced_step_controller`` reads it. The sampled Params default
to the initial-condition Params, and each one's box defaults to the
bounds of the state it pins. Three designs fill the box: ``sobol``
stays evenly spread in any leading subset, so one pool serves nested
training-set sizes; ``lhs`` stratifies every coordinate exactly at its
own ``n``; ``uniform`` is independent draws. The Sobol and Latin
hypercube generators come from scipy, an optional dependency; the
uniform design runs without it.

``drto.explicit_nmpc_train`` fits the control policy to a dataset and
returns it as :class:`ExplicitNMPC`, callable in the model's own
units. The inputs are min-max scaled by the sampled boxes, and each
control by the span its labels occupy; the Sobolev loss adds ``gamma``
times the squared error of the network's Jacobian against the stored
derivatives. Every run trains the full budget and keeps the weights
with the best validation loss. The defaults are the
configuration the package's own study measured best. Training and the
policy run on torch, an optional dependency.
"""
import json
from dataclasses import dataclass, field

from pyomo.core import Objective, value
from pyomo.opt import SolverFactory, TerminationCondition

from drto.cold_start import _target, cold_start_dynamic
from drto.declarations import _is_var_member, _side_matching
from drto.dynamic_optimization import _members, _spread
from drto.ideal_nmpc import (
    _WARM_SOLVERS,
    _WARM_START_OPTIONS,
    NmpcHistory,
    _first_move,
    _one_sample,
    _pinned,
    _prune_suffixes,
)
from drto.infinite_horizon import _join_index, _split_index, _time_index
from drto.info import info
from drto.warm_start import warm_start_dynamic

#: The designs explicit_nmpc_data draws by.
_DESIGNS = ("sobol", "lhs", "uniform")


class ExplicitNmpcDataset:
    """A labeled dataset: the sampled values, the labels, the failures.

    ``points`` is a list of dicts, one per labeled draw, with keys
    ``x`` (the sampled values by Param name), ``u0`` (the first control
    action by control name), ``V`` (the objective), and, when gradients
    were recorded, ``du0_dx`` (nested by control then Param name).
    ``failures`` records the draws whose solve did not return optimal,
    each with its ``x`` and the termination condition. ``config`` is
    the generation record: the draw count, the design, the seed, the
    sampled names, and the boxes.
    """

    def __init__(self, config, points, failures):
        self.config = config
        self.points = points
        self.failures = failures

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        c = self.config
        summary = (
            f"drto explicit-NMPC dataset: points {len(self.points)}, "
            f"failures {len(self.failures)}, method {c.get('method')}, "
            f"inputs {', '.join(c.get('inputs', ())) or '(none)'}"
        )
        return summary

    def save(self, path):
        """Write the dataset as JSON; ``load`` reads it back."""
        with open(path, "w") as f:
            json.dump(
                {
                    "config": self.config,
                    "points": self.points,
                    "failures": self.failures,
                },
                f,
                indent=1,
            )

    @classmethod
    def load(cls, path):
        """Read a dataset ``save`` wrote."""
        with open(path) as f:
            d = json.load(f)
        return cls(d["config"], d["points"], d["failures"])


def _design(method, n, dim, seed):
    """``n`` points in the ``dim``-dimensional unit cube."""
    if method == "uniform":
        import numpy as np

        return np.random.default_rng(seed).uniform(size=(n, dim))
    if method not in _DESIGNS:
        raise ValueError(
            f"drto: explicit_nmpc_data got method='{method}'; the designs "
            f"are {', '.join(_DESIGNS)}."
        )
    try:
        from scipy.stats import qmc
    except ImportError as err:
        raise RuntimeError(
            "drto: the sobol and lhs designs draw through scipy (pip "
            "install scipy); the uniform design runs without it."
        ) from err
    if method == "sobol":
        return qmc.Sobol(d=dim, scramble=True, seed=seed).random(n)
    return qmc.LatinHypercube(d=dim, seed=seed).random(n)


def _sampled_params(reg, inputs, ranges, fn):
    """The sampled Params and their boxes, as ``(param, (lo, hi))`` pairs.

    The initial-condition Params are always sampled, each defaulting to
    the bounds of the state its constraint pins; ``inputs`` extends the
    set, and every extra Param needs an entry in ``ranges``, since
    nothing pairs it with a bounded state.
    """
    by_name = {
        (k if isinstance(k, str) else k.name): tuple(v)
        for k, v in (ranges or {}).items()
    }
    pairs, seen = [], set()
    for con in reg.components("initial_condition"):
        for cd in con.values() if con.is_indexed() else (con,):
            state, param = _side_matching(cd, _is_var_member, fn, "a state member")
            if id(param) in seen:
                continue
            seen.add(id(param))
            box = by_name.get(param.name)
            if box is None:
                box = (state.lb, state.ub)
                if box[0] is None or box[1] is None:
                    raise ValueError(
                        f"drto: {fn} has no box for '{param.name}': the "
                        f"state it pins carries no finite bounds. Pass "
                        f"ranges={{{param.name}: (lo, hi)}}."
                    )
            pairs.append((param, box))
    for comp in inputs or ():
        for p in comp.values() if comp.is_indexed() else (comp,):
            if id(p) in seen:
                continue
            seen.add(id(p))
            box = by_name.get(p.name)
            if box is None:
                raise ValueError(
                    f"drto: {fn} has no box for '{p.name}': an input beyond "
                    f"the initial-condition Params pairs with no bounded "
                    f"state. Pass ranges={{{p.name}: (lo, hi)}}."
                )
            pairs.append((p, box))
    return pairs


def _first_moves(reg):
    """The member holding each declared control's first action."""
    out = []
    for u in reg.components("control"):
        if u.is_indexed():
            out.append(u[sorted(u.keys())[0]])
        else:
            out.append(u)
    return out


def _steady_targets(reg, fn):
    """The steady target value per sampled Param and per control.

    Each initial-condition Param takes the value of the steady_state
    target paired with the state it pins, and each control the value
    of its steady_state_control target. A Param or control with no
    pairing is skipped, so the dicts hold what the declarations cover.
    """
    x_ss, u_ss = {}, {}
    if not reg.has_declaration("horizon"):
        return x_ss, u_ss
    time = reg.components("horizon")[0]
    t0 = reg.declarations("horizon")[0]["samples"][0]
    owner = {}
    for z in reg.components("state"):
        pos, subs = _time_index(z, time)
        for idx in z:
            o, tt = _split_index(idx, pos, len(subs))
            if tt == t0:
                owner[id(z[idx])] = (z, o)
    ss = list(reg.declarations("steady_state"))
    for con in reg.components("initial_condition"):
        for cd in con.values() if con.is_indexed() else (con,):
            sd, param = _side_matching(cd, _is_var_member, fn, "a state member")
            if id(sd) not in owner:
                continue
            z, o = owner[id(sd)]
            try:
                tgt = _target(ss, z, "steady_state", fn)
            except ValueError:
                continue
            x_ss[param.name] = float(value(tgt[o] if o else tgt))
    uss = list(reg.declarations("steady_state_control"))
    for u in reg.components("control"):
        try:
            tgt = _target(uss, u, "steady_state_control", fn)
        except ValueError:
            continue
        u_ss[u.name] = float(value(tgt))
    return x_ss, u_ss


def explicit_nmpc_data(
    m,
    n=1000,
    method="sobol",
    inputs=None,
    ranges=None,
    gradients=True,
    solver="pounce",
    seed=0,
    path=None,
):
    """Sample the assembled optimization into a labeled dataset.

    Parameters
    ----------
    m : Block
        The assembled optimization: the declared model with its
        transforms applied.
    n : int
        The number of draws.
    method : str
        The design filling the box: ``"sobol"``, ``"lhs"``, or
        ``"uniform"``.
    inputs : iterable of Params, optional
        Sampled Params beyond the initial-condition Params, each
        needing a ``ranges`` entry.
    ranges : mapping, optional
        Param (or name) to ``(lo, hi)``, overriding a default box or
        supplying one where no default exists.
    gradients : bool
        Record the derivative of each first action with respect to each
        sampled Param, read from the pounce factorization.
    solver : str
        The labeling solver's name.
    seed : int
        The design's seed.
    path : str, optional
        When given, the dataset is also written there as JSON.

    Returns
    -------
    ExplicitNmpcDataset
        The labeled points, the failures, and the generation record.
        When the model declares ``steady_state`` and
        ``steady_state_control``, the record includes the targets'
        values as ``x_ss`` and ``u_ss``, which
        ``steady_state_enforced`` training reads.
    """
    fn = "explicit_nmpc_data"
    if gradients and solver != "pounce":
        raise ValueError(
            f"drto: {fn} reads the gradients from the pounce factorization; "
            f"got solver='{solver}'. Pass gradients=False, or solve with "
            f"pounce."
        )
    reg = info(m)
    pairs = _sampled_params(reg, inputs, ranges, fn)
    moves = _first_moves(reg)
    if not moves:
        raise ValueError(f"drto: {fn} needs declared controls to label.")

    if solver == "pounce":
        import pyomo_pounce

        pyomo_pounce.declare_sens_param(*(p for p, _ in pairs))
    factory = SolverFactory(solver)

    cube = _design(method, n, len(pairs), seed)
    x_ss, u_ss = _steady_targets(reg, fn)
    points, failures = [], []
    for row in cube:
        draw = {}
        for (param, (lo, hi)), r in zip(pairs, row):
            v = lo + float(r) * (hi - lo)
            param.set_value(v)
            draw[param.name] = v
        cold_start_dynamic(m)
        res = factory.solve(m)
        if res.solver.termination_condition != TerminationCondition.optimal:
            failures.append(
                {"x": draw, "termination": str(res.solver.termination_condition)}
            )
            continue
        objective = next(m.component_data_objects(Objective, active=True))
        point = {
            "x": draw,
            "u0": {u.parent_component().name: value(u) for u in moves},
            "V": value(objective),
        }
        if gradients:
            import pyomo_pounce

            point["du0_dx"] = {
                u.parent_component().name: {
                    p.name: pyomo_pounce.gradient(u, wrt=p) for p, _ in pairs
                }
                for u in moves
            }
        points.append(point)

    dataset = ExplicitNmpcDataset(
        {
            "n": n,
            "method": method,
            "seed": seed,
            "gradients": bool(gradients),
            "solver": solver,
            "inputs": [p.name for p, _ in pairs],
            "ranges": {p.name: list(box) for p, box in pairs},
            # the control bounds, which the training scales outputs by
            "u_bounds": {u.parent_component().name: [u.lb, u.ub] for u in moves},
            # the declared steady targets, which steady_state_enforced
            # training reads
            **({"x_ss": x_ss} if x_ss else {}),
            **({"u_ss": u_ss} if u_ss else {}),
        },
        points,
        failures,
    )
    if path is not None:
        dataset.save(path)
    return dataset


#: Epochs between validation checkpoints.
_VAL_EVERY = 10

#: The learning rate of the value-only fine-tune phase, the rate the
#: package's study ran it at.
_FINE_TUNE_LR = 1e-4

#: The activations the trainer builds.
_ACTIVATIONS = ("tanh", "relu", "silu", "sigmoid")


def _torch():
    try:
        import torch
    except ImportError as err:
        raise RuntimeError(
            "drto: the explicit-NMPC training and policy run on torch "
            "(pip install torch)."
        ) from err
    return torch


def _resolve_dataset(data, what):
    if isinstance(data, str):
        return ExplicitNmpcDataset.load(data)
    if isinstance(data, ExplicitNmpcDataset):
        return data
    raise TypeError(
        f"drto: explicit_nmpc_train takes {what} as an ExplicitNmpcDataset "
        f"or a path; got {type(data).__name__}."
    )


def _arrays(dataset, gradients, fn, u_scale=None):
    """The dataset as scaled arrays: x, u, the Jacobian, and the
    control scale.

    The inputs scale by their sampled boxes. Each control scales by
    ``u_scale``, or, when it is None, by the span this dataset's own
    labels occupy, floored at one thousandth of the control's bound
    range so a near-constant control keeps a finite scale. Pass the
    training set's scale for a validation set, so both sets measure
    in the same units.
    """
    import numpy as np

    cfg = dataset.config
    names = list(cfg["inputs"])
    u_bounds = cfg.get("u_bounds")
    if not u_bounds:
        raise ValueError(
            f"drto: {fn}: the dataset records no control bounds, which "
            f"the output scaling needs; regenerate it with "
            f"drto.explicit_nmpc_data."
        )
    u_names = list(u_bounds)
    x = np.array([[p["x"][k] for k in names] for p in dataset.points])
    u = np.array([[p["u0"][k] for k in u_names] for p in dataset.points])
    x_lo, x_hi = np.array([cfg["ranges"][k] for k in names]).T
    rx = x_hi - x_lo
    if u_scale is None:
        b_lo, b_hi = np.array([u_bounds[k] for k in u_names]).T
        span = np.maximum(u.max(axis=0) - u.min(axis=0), 1e-3 * (b_hi - b_lo))
        u_scale = {
            k: [float(lo), float(sp)] for k, lo, sp in zip(u_names, u.min(axis=0), span)
        }
    u_lo = np.array([u_scale[k][0] for k in u_names])
    ru = np.array([u_scale[k][1] for k in u_names])
    J = None
    if gradients:
        if any("du0_dx" not in p for p in dataset.points):
            raise ValueError(
                f"drto: {fn}: the 'sobolev' loss needs the gradient labels, "
                f"and this dataset carries none. Regenerate it with "
                f"gradients=True, or pass training_loss='value' and "
                f"validation_loss='value'."
            )
        J = np.array(
            [
                [[p["du0_dx"][uk][xk] for xk in names] for uk in u_names]
                for p in dataset.points
            ]
        )
        J = J * rx[None, None, :] / ru[None, :, None]
    return (x - x_lo) / rx, (u - u_lo) / ru, J, u_scale


def _network(torch, hidden, activation, n_in, n_out):
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"drto: explicit_nmpc_train got activation='{activation}'; "
            f"the choices are {', '.join(_ACTIVATIONS)}."
        )
    act = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "silu": torch.nn.SiLU,
        "sigmoid": torch.nn.Sigmoid,
    }[activation]
    layers, width = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(width, h), act()]
        width = h
    layers.append(torch.nn.Linear(width, n_out))
    return torch.nn.Sequential(*layers).double()


def _anchor(torch, net, x_ss, u_ss):
    """The policy as the net plus the offset pinning the equilibrium.

    The two net terms cancel at ``x_ss``, so the output equals
    ``u_ss`` there exactly, whatever the weights. Both points are
    buffers in scaled units, so ``state_dict`` round-trips them.
    """

    class Anchored(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("x_ss", x_ss)
            self.register_buffer("u_ss", u_ss)

        def forward(self, x):
            return self.u_ss + self.net(x) - self.net(self.x_ss)

    return Anchored()


def _jacobian(torch, model, x, create_graph=True):
    """The network's per-point Jacobian, differentiable in the weights."""
    x = x.requires_grad_(True)
    y = model(x)
    last = y.shape[1] - 1
    rows = [
        torch.autograd.grad(
            y[:, k].sum(),
            x,
            create_graph=create_graph,
            retain_graph=create_graph or k < last,
        )[0]
        for k in range(y.shape[1])
    ]
    return torch.stack(rows, dim=1)


def explicit_nmpc_train(
    data,
    validation=0.2,
    validation_loss="sobolev",
    training_loss="sobolev",
    gamma=1.0,
    hidden=(100, 100, 100),
    activation="tanh",
    steady_state_enforced=True,
    lr=1e-3,
    schedule="cosine",
    lr_min=1e-5,
    clip=1.0,
    weight_decay=0.0,
    epochs=50000,
    fine_tune=0.2,
    seeds=1,
    device="auto",
):
    """Fit the control policy to a labeled dataset.

    Parameters
    ----------
    data : ExplicitNmpcDataset or str
        The training dataset, or a path to one.
    validation : ExplicitNmpcDataset, str, or float
        The validation set, a path to one, or a fraction split off
        ``data``, shuffled under the dataset's own seed.
    validation_loss : str
        The metric the kept checkpoint and the ``seeds`` winner are
        chosen by, evaluated on the validation set every 10 epochs.
        ``"sobolev"``, the default, is the value error plus ``gamma``
        times the gradient term, one definition across both phases.
        ``"value"`` is the value error alone. The choice is independent
        of ``training_loss``.
    training_loss : str
        The loss the run minimizes. ``"sobolev"``, the default, adds
        ``gamma`` times the squared error of the policy's Jacobian
        against the stored derivatives to the value error. ``"value"``
        fits the values alone.
    gamma : float
        The gradient term's weight.
    hidden : tuple of int
        The hidden layer widths.
    activation : str
        ``"tanh"``, ``"relu"``, ``"silu"``, or ``"sigmoid"``.
    steady_state_enforced : bool
        Build the policy as the net plus the constant offset that
        makes it return the recorded steady control at the recorded
        steady state exactly, whatever the weights. Needs the
        ``x_ss`` and ``u_ss`` the dataset records when the model
        declares ``steady_state`` and ``steady_state_control``.
    lr : float
        The learning rate, the cosine schedule's starting rate.
    schedule : str
        ``"cosine"`` decays the rate to ``lr_min`` over the Sobolev
        phase; ``"flat"`` holds ``lr``.
    lr_min : float
        The cosine schedule's final rate.
    clip : float, optional
        The gradient-norm cap per step; ``None`` disables clipping.
    weight_decay : float
        AdamW's decoupled L2 penalty on the weights, applied in the
        Sobolev and fine-tune phases alike; ``0`` disables it.
    epochs : int
        The training budget; every run trains all of it.
    fine_tune : float
        The final fraction of the budget trained on the value error
        alone, at a flat 1e-4; ``0`` disables the phase.
    seeds : int
        Networks trained from distinct initializations. The best by
        the validation loss is kept.
    device : str
        ``"auto"`` picks cuda when torch sees it; any torch device
        string passes through.

    Returns
    -------
    ExplicitNMPC
        The fitted policy, callable in model units, carrying its
        training history and validation error.
    """
    fn = "explicit_nmpc_train"
    torch = _torch()
    import numpy as np

    if schedule not in ("cosine", "flat"):
        raise ValueError(
            f"drto: {fn} got schedule='{schedule}'; the choices are " f"cosine, flat."
        )
    if validation_loss not in ("sobolev", "value"):
        raise ValueError(
            f"drto: {fn} got validation_loss='{validation_loss}'. The "
            f"choices are sobolev, value."
        )
    if training_loss not in ("sobolev", "value"):
        raise ValueError(
            f"drto: {fn} got training_loss='{training_loss}'. The "
            f"choices are sobolev, value."
        )
    dataset = _resolve_dataset(data, "data")
    val_sobolev = validation_loss == "sobolev"
    need_j = training_loss == "sobolev" or val_sobolev
    x, u, J, u_scale = _arrays(dataset, need_j, fn)
    if isinstance(validation, float):
        rng = np.random.default_rng(dataset.config.get("seed", 0))
        order = rng.permutation(len(x))
        n_val = max(1, round(validation * len(x)))
        val_idx, train_idx = order[:n_val], order[n_val:]
        x_val, u_val = x[val_idx], u[val_idx]
        J_val = J[val_idx] if J is not None else None
        x, u = x[train_idx], u[train_idx]
        if J is not None:
            J = J[train_idx]
    else:
        vset = _resolve_dataset(validation, "validation")
        x_val, u_val, J_val, _ = _arrays(vset, val_sobolev, fn, u_scale=u_scale)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(a):
        return torch.tensor(a, dtype=torch.float64, device=device)

    x, u, x_val, u_val = t(x), t(u), t(x_val), t(u_val)
    J = t(J) if J is not None else None
    J_val = t(J_val) if J_val is not None else None

    anchor = None
    if steady_state_enforced:
        cfg = dataset.config
        stored_x, stored_u = cfg.get("x_ss") or {}, cfg.get("u_ss") or {}
        names, u_names = list(cfg["inputs"]), list(cfg["u_bounds"])
        missing = [k for k in names if k not in stored_x]
        missing += [k for k in u_names if k not in stored_u]
        if missing:
            raise ValueError(
                f"drto: {fn}: steady_state_enforced=True needs the steady "
                f"targets the dataset records at generation, and this one "
                f"has none for {', '.join(missing)}. Regenerate it with "
                f"drto.explicit_nmpc_data on a model declaring "
                f"steady_state and steady_state_control, or pass "
                f"steady_state_enforced=False."
            )
        xs, us = [], []
        for k in names:
            lo, hi = cfg["ranges"][k]
            xs.append((float(stored_x[k]) - lo) / (hi - lo))
        for k in u_names:
            lo, span = u_scale[k]
            us.append((float(stored_u[k]) - lo) / span)
        anchor = (t([xs]), t([us]))

    def value_error(model_out, target):
        return torch.mean((model_out - target) ** 2)

    phase2 = epochs - int(epochs * fine_tune)
    best_overall = None
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = _network(torch, hidden, activation, x.shape[1], u.shape[1])
        if anchor is not None:
            model = _anchor(torch, model, *anchor)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(phase2, 1), eta_min=lr_min
            )
            if schedule == "cosine"
            else None
        )
        clipping = clip
        history = {"epoch": [], "train_loss": [], "val_loss": []}
        best, best_state = float("inf"), None
        for epoch in range(epochs):
            if epoch == phase2 and fine_tune:
                opt = torch.optim.AdamW(
                    model.parameters(), lr=_FINE_TUNE_LR, weight_decay=weight_decay
                )
                sched, clipping = None, None
            opt.zero_grad()
            loss = value_error(model(x), u)
            if training_loss == "sobolev" and epoch < phase2:
                jac = _jacobian(torch, model, x)
                loss = loss + gamma * torch.mean((jac - J) ** 2)
            loss.backward()
            if clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            opt.step()
            if sched is not None:
                sched.step()
            if (epoch + 1) % _VAL_EVERY:
                del loss
                continue
            with torch.no_grad():
                val = value_error(model(x_val), u_val).item()
            if val_sobolev:
                jac = _jacobian(torch, model, x_val, create_graph=False)
                with torch.no_grad():
                    val += gamma * float(torch.mean((jac - J_val) ** 2))
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(val)
            del loss
            if val < best:
                best = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if best_overall is None or best < best_overall[0]:
            best_overall = (best, best_state, history)

    val_loss, state, history = best_overall
    meta = {
        "inputs": list(dataset.config["inputs"]),
        "ranges": {k: list(v) for k, v in dataset.config["ranges"].items()},
        "u_bounds": {k: list(v) for k, v in dataset.config["u_bounds"].items()},
        "u_scale": {k: list(v) for k, v in u_scale.items()},
        "hidden": list(hidden),
        "activation": activation,
        "steady_state_enforced": bool(steady_state_enforced),
        "history": history,
        "validation_error": val_loss,
    }
    return ExplicitNMPC(meta, state)


class ExplicitNMPC:
    """The fitted control policy, callable in the model's own units.

    Call it with a mapping of input names to values and it returns the
    first control action per control, unscaled. ``history`` is the kept
    run's per-epoch record, and ``validation_error`` its best
    validation loss. ``save`` and ``load`` round-trip the
    policy, history included, through one torch file.
    """

    def __init__(self, meta, state):
        torch = _torch()
        self.meta = meta
        n_in, n_out = len(meta["inputs"]), len(meta["u_bounds"])
        self._model = _network(
            torch, tuple(meta["hidden"]), meta["activation"], n_in, n_out
        )
        if meta.get("steady_state_enforced"):
            self._model = _anchor(
                torch,
                self._model,
                torch.zeros((1, n_in), dtype=torch.float64),
                torch.zeros((1, n_out), dtype=torch.float64),
            )
        self._model.load_state_dict(state)
        self._model.eval()

    @property
    def history(self):
        return self.meta["history"]

    @property
    def validation_error(self):
        return self.meta["validation_error"]

    def __call__(self, values):
        torch = _torch()
        missing = [k for k in self.meta["inputs"] if k not in values]
        if missing:
            raise ValueError(
                f"drto: the policy takes {', '.join(self.meta['inputs'])}; "
                f"missing: {', '.join(missing)}."
            )
        xs = []
        for k in self.meta["inputs"]:
            lo, hi = self.meta["ranges"][k]
            xs.append((float(values[k]) - lo) / (hi - lo))
        with torch.no_grad():
            y = self._model(torch.tensor([xs], dtype=torch.float64))[0]
        # policies saved before the label-span scaling trained on the
        # bound range, which the fallback reproduces exactly
        scale = self.meta.get("u_scale") or {
            k: [lo, hi - lo] for k, (lo, hi) in self.meta["u_bounds"].items()
        }
        out = {}
        for j, name in enumerate(self.meta["u_bounds"]):
            lo, span = scale[name]
            out[name] = lo + float(y[j]) * span
        return out

    def __repr__(self):
        return (
            f"drto explicit-NMPC policy: inputs "
            f"{', '.join(self.meta['inputs'])} -> controls "
            f"{', '.join(self.meta['u_bounds'])}, validation error "
            f"{self.meta['validation_error']:.3e}"
        )

    def save(self, path):
        """Write the policy, its scaling, and its history to one file."""
        torch = _torch()
        torch.save({"meta": self.meta, "state": self._model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        """Read a policy ``save`` wrote."""
        torch = _torch()
        d = torch.load(path, weights_only=False)
        return cls(d["meta"], d["state"])


#: The stage-cost kinds whose first member evaluates a visited sample's
#: cost on the plant.
_STAGE_KINDS = ("tracking_stage_cost", "economic_stage_cost", "move_suppression")

#: The horizon-only cost constructs the plant sheds outright.
_PLANT_SHED = ("tracking_terminal_cost", "terminal_constraint")

#: The plant's constant-zero objective.
_PLANT_OBJECTIVE = "_drto_policy_plant_objective"


@dataclass
class ExplicitNmpcReport(NmpcHistory):
    """The policy's closed loop, in the loop-history shape.

    Everything :class:`drto.NmpcHistory` records, plus ``stage_costs``,
    the stage cost at each visited sample, and, when the comparison ran,
    ``solver_moves``: the control the horizon solve takes at the same
    visited states, beside the policy's applied ``moves``. The summary
    states the closed-loop cost, the stage costs summed; it is computed
    here and stored nowhere else.
    """

    solver_moves: dict = field(default_factory=dict)
    stage_costs: list = field(default_factory=list)

    def __str__(self):
        text = (
            f"drto explicit-NMPC closed loop: "
            f"{max(0, len(self.times) - 1)} samples, closed-loop cost "
            f"{sum(self.stage_costs):.6g} (the stage cost summed over the "
            f"visited samples). States {', '.join(self.states) or '(none)'}. "
            f"Moves {', '.join(self.moves) or '(none)'}"
        )
        if self.solver_moves:
            text += ". The solver's controls are recorded at the visited states"
        return text


def _policy_plant(m, fn):
    """The one-sample plant, cloned from the assembled optimization.

    The controls are fixed (the loop writes each step's move), the
    objective is the constant zero, the terminal constructs leave, and
    each stage cost keeps only its first member, which evaluates the
    visited sample's cost once the step is simulated. Every unfixed
    variable loses its domain and bounds. A square simulation cannot
    steer away from a bound, so a kept bound could only turn a state
    excursion into an infeasible step, and the loop's job is to record
    where the policy actually drives the plant.
    """
    from pyomo.core import Objective, Reals, Var

    plant = m.clone()
    regp = info(plant)
    t0 = regp.declarations("horizon")[0]["samples"][0]
    for kind in _PLANT_SHED:
        for record in regp.declarations(kind):
            comp = record["component"]
            if comp.parent_block() is not None:
                comp.parent_block().del_component(comp)
        regp._declarations.pop(kind, None)
    for kind in _STAGE_KINDS:
        for record in regp.declarations(kind):
            con = record["component"]
            if con.parent_block() is None or not con.is_indexed():
                continue
            for idx in list(con.keys()):
                if idx != t0:
                    del con[idx]
    for u in regp.components("control"):
        for vd in _members(u):
            vd.fix()
    for vd in plant.component_data_objects(Var, descend_into=True):
        if vd.fixed:
            continue
        vd.domain = Reals
        vd.setlb(None)
        vd.setub(None)
    for obj in plant.component_data_objects(Objective, active=True):
        obj.deactivate()
    plant.add_component(_PLANT_OBJECTIVE, Objective(expr=0.0))
    _one_sample(plant)
    _prune_suffixes(plant)
    return plant


def _stage_cost_vars(reg, t0, fn):
    """The scalar cost variables of the stage costs' first members."""
    out = []
    for kind in _STAGE_KINDS:
        for record in reg.declarations(kind):
            con = record["component"]
            if con.parent_block() is None:
                continue
            cd = con[t0] if con.is_indexed() else con
            side, _ = _side_matching(cd, _is_var_member, fn, "the cost variable")
            out.append(side)
    return out


def explicit_nmpc_closed_loop(
    policy, m, samples=50, x0=None, disturbances=None, solver="pounce", compare=False
):
    """Run the fitted policy closed loop against the declared model.

    Each action is clamped to the declared control's bounds before it
    is applied, the way an actuator holds an out-of-range command at
    its limit, and ``moves`` records the applied values. The plant
    carries no state bounds, so a step converges wherever the dynamics
    land and the report records an excursion outside the controller's
    box.

    Parameters
    ----------
    policy : ExplicitNMPC
        The fitted policy.
    m : Block
        The assembled optimization the policy was sampled from. With
        ``compare`` it is solved at each visited state; the plant is a
        one-sample simulation cloned from it either way.
    samples : int
        The loop length.
    x0 : mapping, optional
        Declared state (the component, or its name) to the value
        written into its initial-condition Params before the first
        step. Omitted, the Params' current values are the first state.
    disturbances : mapping, optional
        Declared disturbance (the component, or its name) to its
        per-step realization, one value per sample. A disturbance with
        no entry is zero, and the loop is deterministic.
    solver : str
        The solver for the plant steps and the compare solves.
    compare : bool
        Also solve the horizon problem at each visited state and record
        the control it takes there, beside the policy's.

    Returns
    -------
    ExplicitNmpcReport
        The visited trajectory, the applied controls, the per-sample
        stage costs, and, with ``compare``, the solver's controls at
        the same states. ``drto.plot_states`` and ``drto.plot_controls``
        draw it.
    """
    fn = "explicit_nmpc_closed_loop"
    reg = info(m)
    if not reg.has_declaration("horizon"):
        raise ValueError(f"drto: {fn} requires the horizon declaration.")
    grid = reg.declarations("horizon")[0]["samples"]
    t0, t1 = grid[0], grid[1]
    dt = t1 - t0
    time = reg.components("horizon")[0]

    owner = {}
    for z in reg.components("state"):
        pos, subs = _time_index(z, time)
        for idx in z:
            o, t = _split_index(idx, pos, len(subs))
            if t == t0:
                owner[id(z[idx])] = (z, o)
    pins = _pinned(reg, fn)
    hooks_of = {}
    for vd, hook in pins:
        hooks_of.setdefault(owner[id(vd)][0].local_name, []).append(hook)
    for key, val in (x0 or {}).items():
        name = key if isinstance(key, str) else key.local_name
        hooks = hooks_of.get(name)
        if hooks is None:
            raise ValueError(
                f"drto: {fn} got an initial state for '{name}', which is "
                f"not a pinned state; pinned: {', '.join(hooks_of) or '(none)'}."
            )
        for hook, v in zip(hooks, _spread(val, len(hooks), name, fn)):
            hook.set_value(v)

    declared_dist = [w.local_name for w in reg.components("disturbance")]
    plan = {}
    for key, val in (disturbances or {}).items():
        name = key if isinstance(key, str) else key.local_name
        if name not in declared_dist:
            raise ValueError(
                f"drto: {fn} got a realization for '{name}', which is not "
                f"a declared disturbance; declared: "
                f"{', '.join(declared_dist) or '(none)'}."
            )
        if len(val) < samples:
            raise ValueError(
                f"drto: {fn} runs {samples} samples but the sequence for "
                f"'{name}' has {len(val)} values; give one per sample."
            )
        plan[name] = list(val)

    if solver == "pounce":
        import pyomo_pounce  # noqa: F401
    opt = SolverFactory(solver)

    plant = _policy_plant(m, fn)
    regp = info(plant)
    p_pins = _pinned(regp, fn)
    time_p = regp.components("horizon")[0]
    ss = list(reg.declarations("steady_state"))
    labels, targets, p_read = [], [], []
    for (c_vd, _h), (p_vd, _hp) in zip(pins, p_pins):
        z, o = owner[id(c_vd)]
        labels.append(
            z.local_name if not o else f"{z.local_name}[{','.join(map(str, o))}]"
        )
        tgt = _target(ss, z, "steady_state", fn)
        targets.append(value(tgt[o] if o else tgt))
        zp = p_vd.parent_component()
        pos, subs = _time_index(zp, time_p)
        po, _t = _split_index(p_vd.index(), pos, len(subs))
        p_read.append(zp[_join_index(po, t1, pos)])
    c_hooks = [h for _vd, h in pins]
    p_hooks = [h for _vd, h in p_pins]

    report = ExplicitNmpcReport()
    report.times.append(t0)
    for label, hook, tgt in zip(labels, c_hooks, targets):
        report.states[label] = [value(hook)]
        report.state_targets[label] = tgt
    m_controls = list(reg.components("control"))
    p_controls = list(regp.components("control"))
    ucss = list(reg.declarations("steady_state_control"))
    for u in m_controls:
        report.moves[u.local_name] = []
        report.control_targets[u.local_name] = value(
            _target(ucss, u, "steady_state_control", fn)
        )
        if compare:
            report.solver_moves[u.local_name] = []
    p_dist = list(regp.components("disturbance"))
    for w in p_dist:
        report.realizations[w.local_name] = []
    cost_vars = _stage_cost_vars(regp, t0, fn)
    u_bounds = [(_first_move(u).lb, _first_move(u).ub) for u in m_controls]

    for k in range(samples):
        action = policy({h.name: value(h) for h in c_hooks})
        if compare:
            if k > 0:
                warm_start_dynamic(m)
            options = (
                dict(_WARM_START_OPTIONS) if k > 0 and solver in _WARM_SOLVERS else {}
            )
            res = opt.solve(m, options=options)
            if res.solver.termination_condition != TerminationCondition.optimal:
                raise RuntimeError(
                    f"drto: {fn}: the compare solve failed at sample {k} "
                    f"({res.solver.termination_condition})."
                )
            for u in m_controls:
                report.solver_moves[u.local_name].append(value(_first_move(u)))
        for u, pu, (lo, hi) in zip(m_controls, p_controls, u_bounds):
            move = float(action[u.name])
            if lo is not None and move < lo:
                move = lo
            if hi is not None and move > hi:
                move = hi
            report.moves[u.local_name].append(move)
            for vd in _members(pu):
                vd.set_value(move)
        for w in p_dist:
            seq = plan.get(w.local_name)
            realized = float(seq[k]) if seq is not None else 0.0
            report.realizations[w.local_name].append(realized)
            for vd in _members(w):
                vd.set_value(realized)
        res = opt.solve(plant)
        if res.solver.termination_condition != TerminationCondition.optimal:
            state = ", ".join(
                f"{label} {value(hook):.6g}" for label, hook in zip(labels, c_hooks)
            )
            applied = ", ".join(
                f"{u.local_name} {report.moves[u.local_name][-1]:.6g}"
                for u in m_controls
            )
            raise RuntimeError(
                f"drto: {fn}: the plant solve failed at sample {k} "
                f"({res.solver.termination_condition}). Visited state "
                f"{state}. Applied action {applied}."
            )
        report.stage_costs.append(sum(value(cv) for cv in cost_vars))
        for c_hook, p_hook, src, label in zip(c_hooks, p_hooks, p_read, labels):
            v = value(src)
            c_hook.set_value(v)
            p_hook.set_value(v)
            report.states[label].append(v)
        report.times.append(t0 + (k + 1) * dt)
    return report
