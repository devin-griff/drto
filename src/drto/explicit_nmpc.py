# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit NMPC data generation: ``drto.explicit_nmpc_data`` (feature 026).

Samples the assembled optimization into a labeled dataset. Each draw
writes values into the sampled Params, cold-starts the model, solves
once, and records the first control action per declared control, the
objective, and, when asked, the derivative of each first action with
respect to each sampled Param, read from the converged factorization
the way ``drto.advanced_step_controller`` reads it.

The sampled Params default to the initial-condition Params, and each
one's box defaults to the bounds of the state it pins. Three designs
fill the box: ``sobol`` stays evenly spread in any leading subset, so
one pool serves nested training-set sizes; ``lhs`` stratifies every
coordinate exactly at its own ``n``; ``uniform`` is independent draws.
The Sobol and Latin hypercube generators come from scipy, an optional
dependency; the uniform design runs without it.
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
            f"drto explicit-NMPC dataset: {len(self.points)} points, "
            f"{len(self.failures)} failures, method {c.get('method')}, "
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
        },
        points,
        failures,
    )
    if path is not None:
        dataset.save(path)
    return dataset
