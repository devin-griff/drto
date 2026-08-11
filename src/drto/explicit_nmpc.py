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
units. Inputs, outputs, and gradient labels are min-max scaled by the
sampled boxes and the control bounds; the Sobolev loss adds ``gamma``
times the squared error of the network's Jacobian against the stored
derivatives. Every run trains the full budget and keeps the weights
with the best validation value error; the defaults are the
configuration the package's own study measured best. Training and the
policy run on torch, an optional dependency.
"""
import json

from pyomo.core import Objective, value
from pyomo.opt import SolverFactory, TerminationCondition

from drto.cold_start import cold_start_dynamic
from drto.declarations import _is_var_member, _side_matching
from drto.info import info

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
        return (
            f"drto explicit-NMPC dataset: points {len(self.points)}, "
            f"failures {len(self.failures)}, method {c.get('method')}, "
            f"inputs {', '.join(c.get('inputs', ())) or '(none)'}"
        )

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


def explicit_nmpc_data(
    m,
    n=1000,
    method="sobol",
    inputs=None,
    ranges=None,
    gradients=True,
    hessians=False,
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
    hessians : bool
        Record each labeled point's reduced Hessian over the first
        moves, read from the same factorization through pounce's
        ``information``: one query per solve, symmetric, in control
        units.
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
    """
    fn = "explicit_nmpc_data"
    if gradients and solver != "pounce":
        raise ValueError(
            f"drto: {fn} reads the gradients from the pounce factorization; "
            f"got solver='{solver}'. Pass gradients=False, or solve with "
            f"pounce."
        )
    if hessians and solver != "pounce":
        raise ValueError(
            f"drto: {fn} reads the reduced Hessians from the pounce "
            f"factorization; got solver='{solver}'. Pass hessians=False, "
            f"or solve with pounce."
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
        if hessians:
            import pyomo_pounce

            block = pyomo_pounce.information(m, wrt=moves)
            point["H"] = {
                ui.parent_component().name: {
                    uj.parent_component().name: float(block.matrix[i][j])
                    for j, uj in enumerate(moves)
                }
                for i, ui in enumerate(moves)
            }
        points.append(point)

    dataset = ExplicitNmpcDataset(
        {
            "n": n,
            "method": method,
            "seed": seed,
            "gradients": bool(gradients),
            "hessians": bool(hessians),
            "solver": solver,
            "inputs": [p.name for p, _ in pairs],
            "ranges": {p.name: list(box) for p, box in pairs},
            # the control bounds, which the training scales outputs by
            "u_bounds": {u.parent_component().name: [u.lb, u.ub] for u in moves},
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


def _arrays(dataset, gradients, fn, hessians=False):
    """The dataset as scaled arrays: x, u, the Jacobian, the Hessians."""
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
    u_lo, u_hi = np.array([u_bounds[k] for k in u_names]).T
    rx, ru = x_hi - x_lo, u_hi - u_lo
    J = None
    if gradients:
        if any("du0_dx" not in p for p in dataset.points):
            raise ValueError(
                f"drto: {fn}: sobolev=True needs the gradient labels, and "
                f"this dataset carries none; regenerate it with "
                f"gradients=True, or train with sobolev=False."
            )
        J = np.array(
            [
                [[p["du0_dx"][uk][xk] for xk in names] for uk in u_names]
                for p in dataset.points
            ]
        )
        J = J * rx[None, None, :] / ru[None, :, None]
    H = None
    if hessians:
        if any("H" not in p for p in dataset.points):
            raise ValueError(
                f"drto: {fn}: weighting='hessian' needs the stored reduced "
                f"Hessians, and this dataset carries none; regenerate it "
                f"with hessians=True."
            )
        H = np.array(
            [
                [[p["H"][ui][uj] for uj in u_names] for ui in u_names]
                for p in dataset.points
            ]
        )
        # scaled control units: H-tilde = Du H Du
        H = H * ru[None, :, None] * ru[None, None, :]
    return (x - x_lo) / rx, (u - u_lo) / ru, J, H


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
    sobolev=True,
    gamma=1.0,
    hidden=(100, 100, 100),
    activation="tanh",
    lr=1e-3,
    schedule="cosine",
    lr_min=1e-5,
    clip=1.0,
    epochs=50000,
    fine_tune=0.2,
    weighting=None,
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
    sobolev : bool
        Add ``gamma`` times the squared error of the network's Jacobian
        against the stored derivatives to the loss.
    gamma : float
        The gradient term's weight.
    hidden : tuple of int
        The hidden layer widths.
    activation : str
        ``"tanh"``, ``"relu"``, ``"silu"``, or ``"sigmoid"``.
    lr : float
        The learning rate, the cosine schedule's starting rate.
    schedule : str
        ``"cosine"`` decays the rate to ``lr_min`` over the Sobolev
        phase; ``"flat"`` holds ``lr``.
    lr_min : float
        The cosine schedule's final rate.
    clip : float, optional
        The gradient-norm cap per step; ``None`` disables clipping.
    epochs : int
        The training budget; every run trains all of it.
    fine_tune : float
        The final fraction of the budget trained on the value error
        alone, at a flat 1e-4; ``0`` disables the phase.
    weighting : str, optional
        ``None`` is the plain loss. ``"hessian"`` weights both terms by
        each point's stored reduced Hessian in scaled control units,
        every matrix divided by the dataset's mean trace so ``gamma``
        keeps its scale; the validation value error is weighted the
        same way. The fit is then accurate where the cost surface is
        steep and tolerant where it is flat.
    seeds : int
        Networks trained from distinct initializations; the best by
        validation value error is kept.
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
    if weighting not in (None, "hessian"):
        raise ValueError(
            f"drto: {fn} got weighting='{weighting}'; the choices are "
            f"None, hessian."
        )
    weighted = weighting == "hessian"
    dataset = _resolve_dataset(data, "data")
    x, u, J, H = _arrays(dataset, sobolev, fn, hessians=weighted)
    if isinstance(validation, float):
        rng = np.random.default_rng(dataset.config.get("seed", 0))
        order = rng.permutation(len(x))
        n_val = max(1, round(validation * len(x)))
        val_idx, train_idx = order[:n_val], order[n_val:]
        x_val, u_val = x[val_idx], u[val_idx]
        H_val = H[val_idx] if H is not None else None
        x, u = x[train_idx], u[train_idx]
        if J is not None:
            J = J[train_idx]
        if H is not None:
            H = H[train_idx]
    else:
        vset = _resolve_dataset(validation, "validation")
        x_val, u_val, _, H_val = _arrays(vset, False, fn, hessians=weighted)
    if weighted:
        # one normalization for both sets, so gamma keeps its scale
        norm = float(np.mean(np.trace(H, axis1=1, axis2=2))) / H.shape[1]
        H = H / norm
        H_val = H_val / norm

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(a):
        return torch.tensor(a, dtype=torch.float64, device=device)

    x, u, x_val, u_val = t(x), t(u), t(x_val), t(u_val)
    J = t(J) if J is not None else None
    H = t(H) if H is not None else None
    H_val = t(H_val) if H is not None else None

    def value_error(model_out, target, W):
        e = model_out - target
        if W is None:
            return torch.mean(e**2)
        return torch.mean(torch.einsum("bi,bij,bj->b", e, W, e)) / e.shape[1]

    phase2 = epochs - int(epochs * fine_tune)
    best_overall = None
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = _network(torch, hidden, activation, x.shape[1], u.shape[1])
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(phase2, 1), eta_min=lr_min
            )
            if schedule == "cosine"
            else None
        )
        clipping = clip
        history = {"epoch": [], "train_loss": [], "val_mse": []}
        best, best_state = float("inf"), None
        for epoch in range(epochs):
            if epoch == phase2 and fine_tune:
                opt = torch.optim.Adam(model.parameters(), lr=_FINE_TUNE_LR)
                sched, clipping = None, None
            opt.zero_grad()
            loss = value_error(model(x), u, H)
            if sobolev and epoch < phase2:
                jac = _jacobian(torch, model, x)
                E = jac - J
                if H is None:
                    gterm = torch.mean(E**2)
                else:
                    gterm = torch.mean(torch.einsum("boi,bop,bpi->b", E, H, E)) / (
                        E.shape[1] * E.shape[2]
                    )
                loss = loss + gamma * gterm
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
                val = value_error(model(x_val), u_val, H_val).item()
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(float(loss.item()))
            history["val_mse"].append(val)
            del loss
            if val < best:
                best = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if best_overall is None or best < best_overall[0]:
            best_overall = (best, best_state, history)

    val_mse, state, history = best_overall
    meta = {
        "inputs": list(dataset.config["inputs"]),
        "ranges": {k: list(v) for k, v in dataset.config["ranges"].items()},
        "u_bounds": {k: list(v) for k, v in dataset.config["u_bounds"].items()},
        "hidden": list(hidden),
        "activation": activation,
        "weighting": weighting,
        "history": history,
        "validation_error": val_mse,
    }
    return ExplicitNMPC(meta, state)


class ExplicitNMPC:
    """The fitted control policy, callable in the model's own units.

    Call it with a mapping of input names to values and it returns the
    first control action per control, unscaled. ``history`` is the kept
    run's per-epoch record, and ``validation_error`` its best
    validation value error. ``save`` and ``load`` round-trip the
    policy, history included, through one torch file.
    """

    def __init__(self, meta, state):
        torch = _torch()
        self.meta = meta
        self._model = _network(
            torch,
            tuple(meta["hidden"]),
            meta["activation"],
            len(meta["inputs"]),
            len(meta["u_bounds"]),
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
        out = {}
        for j, (name, (lo, hi)) in enumerate(self.meta["u_bounds"].items()):
            out[name] = lo + float(y[j]) * (hi - lo)
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
