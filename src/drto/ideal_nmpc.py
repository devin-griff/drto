# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The ideal NMPC loop: ``drto.ideal_nmpc`` (feature 014).

One call runs the closed loop the declarations describe: at each step the
dynamic optimization solves at the current state, the first control move
is implemented on the process, and the process simulates one sample
forward under that move and a disturbance realization. Ideal means the
solve is treated as instantaneous: measurement, solve, and move all land
at the same instant.

The input is the declared, discretized model, with
``drto.infinite_horizon`` applied or not, before the mode transforms.
The loop builds both sides from it: a clone becomes the process through
``drto.dynamic_simulation`` with the controls first fixed at the
declared control targets, and the input becomes the controller through
``drto.dynamic_optimization``. The first solve is cold-started, every
later one warm-started, and under the ``pounce`` or ``ipopt`` solvers
the multiplier suffixes are declared before the first solve so
``drto.warm_start_dynamic`` shifts the multipliers along with the
primals and the warm-started solves run with the feature 013 settings.

An active ``scaling_factor`` suffix is honored the way the initializers
honor it: the loop builds each side's scaled clone once, runs every
solve and warm start on those clones for the whole loop, and reads the
history back in the model's own units. The feedback hooks are Params
and stay physical.

The returned :class:`NmpcHistory` holds the actual trajectory under the
declared names, and ``drto.plot_states`` / ``drto.plot_controls`` draw
it directly.
"""
import random
from dataclasses import dataclass, field

import pyomo.environ as pyo
from pyomo.core import Suffix, TransformationFactory
from pyomo.opt import SolverFactory

from drto.cold_start import _target, cold_start_dynamic
from drto.declarations import _is_var_member, _side_matching
from drto.dynamic_optimization import _members, _spread
from drto.infinite_horizon import _join_index, _split_index, _time_index
from drto.info import info
from drto.initialize_steady_state import _attached
from drto.warm_start import warm_start_dynamic

#: The feature 013 warm start settings, applied to every warm-started
#: solve when the solver understands them.
_WARM_START_OPTIONS = {
    "warm_start_init_point": "yes",
    "mu_init": 1e-6,
    "warm_start_bound_push": 1e-9,
    "warm_start_mult_bound_push": 1e-9,
}

#: The solvers that read the warm start settings and the multiplier
#: suffixes; any other solver runs the loop on the primal shift alone.
_WARM_SOLVERS = ("pounce", "ipopt")

#: The multiplier suffixes the warm start shifts, with their directions:
#: the solver writes dual and the _out pair, the shift fills the _in pair.
_SUFFIXES = (
    ("dual", Suffix.IMPORT),
    ("ipopt_zL_out", Suffix.IMPORT),
    ("ipopt_zU_out", Suffix.IMPORT),
    ("ipopt_zL_in", Suffix.EXPORT),
    ("ipopt_zU_in", Suffix.EXPORT),
)

#: The mode transforms; the loop applies its own, so the input comes first.
_TRANSFORMED = (
    "drto.parameterize",
    "drto.dynamic_optimization",
    "drto.dynamic_simulation",
    "drto.dynamic_to_steady_state",
)


@dataclass
class NmpcHistory:
    """The closed loop's actual trajectory, under the declared names.

    ``times`` holds the sample instants, the initial one included;
    ``states`` maps each pinned state member's label to its actual values
    at those instants; ``moves`` and ``realizations`` map each control
    and disturbance to its per-step values, one shorter than ``times``.
    The targets are the declared steady-state pairings' values, the
    plots' setpoint lines.
    """

    times: list = field(default_factory=list)
    states: dict = field(default_factory=dict)
    moves: dict = field(default_factory=dict)
    realizations: dict = field(default_factory=dict)
    state_targets: dict = field(default_factory=dict)
    control_targets: dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"drto nmpc history: {max(0, len(self.times) - 1)} steps, "
            f"states {', '.join(self.states) or '(none)'}; "
            f"moves {', '.join(self.moves) or '(none)'}"
        )


def _pinned(reg, fn):
    """The initial-condition rows' (pinned state member, hook Param) pairs.

    Row order is declaration order, so the controller's and the process
    clone's lists line up positionally.
    """
    pairs, seen = [], set()
    for con in reg.components("initial_condition"):
        for cd in _members(con):
            side, other = _side_matching(cd, _is_var_member, fn, "a state member")
            if id(other) not in seen:
                seen.add(id(other))
                pairs.append((side, other))
    return pairs


def _first_move(u):
    """The member holding a parameterized control's first move."""
    if not u.is_indexed():
        return u
    return u[sorted(u.keys())[0]]


def _factor(vd, fmap):
    """The member's scaling factor on a solve model, 1 when unscaled."""
    if fmap is not None and vd in fmap:
        return fmap[vd]
    return 1.0


def _prune_suffixes(model):
    """Drop suffix entries whose components the transforms removed.

    The mode transforms delete components (shed costs, replaced
    controls); a stale entry breaks any later clone and makes the NL
    writer warn.
    """
    for sfx in model.component_objects(Suffix, active=True):
        for key in [k for k in sfx if not _attached(k, model)]:
            del sfx[key]


def ideal_nmpc(
    m,
    steps,
    initial_condition=None,
    dynamic_optimization=None,
    disturbances=None,
    seed=None,
    cold_start=True,
    solver="pounce",
):
    """Run the ideal NMPC loop for ``steps`` samples; see the module
    docstring.

    Parameters
    ----------
    m : Block
        The declared, discretized model, ``drto.infinite_horizon``
        applied or not, before the mode transforms. It becomes the
        controller in place; the process is a clone.
    steps : int
        The loop length, in samples.
    initial_condition : mapping, optional
        Declared state (the component, or its name) to the value written
        into its feedback hooks before the first step: a constant for
        every pinned member, or one value each. Omitted, the hooks'
        current values are the first actual state.
    dynamic_optimization : mapping, optional
        Options through to the ``drto.dynamic_optimization`` transform.
    disturbances : mapping, optional
        Declared disturbance (the component, or its name) to its
        realization: a sequence gives the per-step values as given, a
        number the standard deviation of independent zero-mean draws.
        A disturbance with no entry is zero.
    seed : int, optional
        Seeds the disturbance draws, making them reproducible.
    cold_start : bool or mapping
        The first solve's cold start: ``True`` (the default) runs
        ``drto.cold_start_dynamic``, a mapping passes through as its
        options, ``False`` skips it.
    solver : str
        The solver, by name, for every controller and process solve.
        ``"pounce"`` and ``"ipopt"`` warm start between steps.

    Returns
    -------
    NmpcHistory
        The actual closed-loop trajectory; ``drto.plot_states`` and
        ``drto.plot_controls`` draw it.

    Raises
    ------
    ValueError
        On a transformed input, an undiscretized horizon, an unknown
        state or disturbance name, or a disturbance sequence shorter
        than the loop.
    RuntimeError
        If the solver is not available, or a solve fails (the error
        names the step).
    """
    fn = "ideal_nmpc"
    if steps < 1:
        raise ValueError(f"drto: {fn}: steps must be at least 1, got {steps}.")
    reg = info(m)
    for name in _TRANSFORMED:
        if reg.has_transformation(name):
            raise ValueError(
                f"drto: {fn} builds the controller and the process itself, "
                f"so it takes the declared, discretized model before the "
                f"mode transforms; '{name}' is already applied."
            )
    if not reg.has_declaration("horizon"):
        raise ValueError(f"drto: {fn} requires the horizon declaration.")
    time = reg.components("horizon")[0]
    if not time.get_discretization_info():
        raise ValueError(
            f"drto: {fn} runs on the grid, so the model must be discretized "
            f"first (apply a dae.* transformation)."
        )

    # a declared state may be a Reference over a member subset of a
    # packed Var (gh #20), so a pinned member matches its declared owner
    # by data identity, the package convention
    t0 = time.first()
    owner = {}
    for z in reg.components("state"):
        pos, subs = _time_index(z, time)
        for idx in z:
            o, t = _split_index(idx, pos, len(subs))
            if t == t0:
                owner[id(z[idx])] = (z, o)

    # the initial condition lands in the hooks on the input, so the
    # process clone inherits it with everything else
    pins = _pinned(reg, fn)
    hooks_of = {}
    for vd, hook in pins:
        hooks_of.setdefault(owner[id(vd)][0].local_name, []).append(hook)
    for key, val in (initial_condition or {}).items():
        name = key if isinstance(key, str) else key.local_name
        hooks = hooks_of.get(name)
        if hooks is None:
            raise ValueError(
                f"drto: {fn} got an initial condition for '{name}', which "
                f"is not a pinned state; pinned: "
                f"{', '.join(hooks_of) or '(none)'}."
            )
        for hook, v in zip(hooks, _spread(val, len(hooks), name, fn)):
            hook.set_value(v)

    # the per-step disturbance plan, validated before anything is built
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
        if isinstance(val, (list, tuple)) and len(val) < steps:
            raise ValueError(
                f"drto: {fn} runs {steps} steps but the sequence for "
                f"'{name}' has {len(val)} values; give one per step."
            )
        plan[name] = val

    if solver == "pounce":
        # importing registers the in-process plugin; without it the name
        # falls back to a PATH executable behind pyomo's ASL wrapper,
        # a different solver than the one the drto stack is built on
        try:
            import pyomo_pounce  # noqa: F401
        except ImportError as err:
            raise RuntimeError(
                f"drto: {fn}: solver 'pounce' requires pyomo-pounce "
                f"(pip install drto[pounce], or pip install pyomo-pounce)."
            ) from err
    opt = SolverFactory(solver)
    if not opt.available(exception_flag=False):
        raise RuntimeError(f"drto: {fn}: solver '{solver}' is not available.")

    # the process: a clone in simulation mode, its controls first fixed
    # at the declared control targets
    uss = list(reg.declarations("steady_state_control"))
    at_targets = {
        u.name: pyo.value(_target(uss, u, "steady_state_control", fn))
        for u in reg.components("control")
    }
    process = TransformationFactory("drto.dynamic_simulation").create_using(
        m, controls=at_targets
    )
    _prune_suffixes(process)

    # the input becomes the controller
    TransformationFactory("drto.dynamic_optimization").apply_to(
        m, **(dynamic_optimization or {})
    )
    _prune_suffixes(m)
    warm = solver in _WARM_SOLVERS
    if warm:
        for name, direction in _SUFFIXES:
            if m.component(name) is None:
                m.add_component(name, Suffix(direction=direction))
    if cold_start is not False:
        cold_start_dynamic(m, **({} if cold_start is True else dict(cold_start)))

    # an active scaling_factor suffix: the loop runs both sides on scaled
    # clones, built once, and reads back in the model's own units
    scaled = any(
        s.local_name == "scaling_factor"
        for s in m.component_objects(Suffix, active=True)
    )
    if scaled:
        xfrm = TransformationFactory("core.scale_model")
        ctrl = xfrm.create_using(m, rename=False)
        plant = xfrm.create_using(process, rename=False)
        fmap = ctrl.component_scaling_factor_map
        pmap = plant.component_scaling_factor_map
    else:
        ctrl, plant, fmap, pmap = m, process, None, None

    # the pinned rows parse on the physical pair; a scaled clone's
    # rewritten rows do not, so the hooks and the read points are found
    # here and mapped onto the solve models by name (rename=False keeps
    # the names identical)
    reg_m, reg_p = info(m), info(process)
    samples = reg_m.declarations("horizon")[0]["samples"]
    t0, t1 = samples[0], samples[1]
    dt = t1 - t0
    c_pins, p_pins = _pinned(reg_m, fn), _pinned(reg_p, fn)
    time_m = reg_m.components("horizon")[0]
    time_p = reg_p.components("horizon")[0]

    # the pinned members' labels and targets come from the declared
    # owner (the member-id map above); the read points, one sample in,
    # from the process's own underlying containers
    ss = list(reg_m.declarations("steady_state"))
    labels, targets, read_phys = [], [], []
    for (c_vd, _h), (p_vd, _hp) in zip(c_pins, p_pins):
        z, o = owner[id(c_vd)]
        labels.append(
            z.local_name if not o else f"{z.local_name}[{','.join(map(str, o))}]"
        )
        tgt = _target(ss, z, "steady_state", fn)
        targets.append(pyo.value(tgt[o] if o else tgt))
        zp = p_vd.parent_component()
        pos, subs = _time_index(zp, time_p)
        po, _t = _split_index(p_vd.index(), pos, len(subs))
        read_phys.append(zp[_join_index(po, t1, pos)])

    if scaled:
        c_hooks = [ctrl.find_component(h.name) for _vd, h in c_pins]
        p_hooks = [plant.find_component(h.name) for _vd, h in p_pins]
        p_read = [plant.find_component(vd.name) for vd in read_phys]
    else:
        c_hooks = [h for _vd, h in c_pins]
        p_hooks = [h for _vd, h in p_pins]
        p_read = read_phys

    history = NmpcHistory()
    history.times.append(t0)
    for label, hook, tgt in zip(labels, c_hooks, targets):
        history.states[label] = [pyo.value(hook)]
        history.state_targets[label] = tgt

    c_controls = list(info(ctrl).components("control"))
    p_controls = list(info(plant).components("control"))
    ucss = list(info(ctrl).declarations("steady_state_control"))
    for u in c_controls:
        history.moves[u.local_name] = []
        history.control_targets[u.local_name] = pyo.value(
            _target(ucss, u, "steady_state_control", fn)
        )
    p_dist = list(info(plant).components("disturbance"))
    for w in p_dist:
        history.realizations[w.local_name] = []

    rng = random.Random(seed)

    def _solve(model, what, step, options=None):
        res = opt.solve(model, options=options or {})
        if not pyo.check_optimal_termination(res):
            raise RuntimeError(
                f"drto: {fn}: the {what} solve failed at step {step} "
                f"({res.solver.termination_condition})."
            )

    for k in range(steps):
        # solve the controller at the current state, warm-started after
        # the first step
        if k > 0:
            warm_start_dynamic(ctrl)
        _solve(
            ctrl,
            "controller",
            k,
            options=_WARM_START_OPTIONS if k > 0 and warm else None,
        )

        # implement each control's first move on the process
        for u, pu in zip(c_controls, p_controls):
            move = pyo.value(_first_move(u)) / _factor(_first_move(u), fmap)
            history.moves[u.local_name].append(move)
            for vd in _members(pu):
                vd.set_value(move * _factor(vd, pmap))

        # realize this step's disturbances on the process
        for w in p_dist:
            entry = plan.get(w.local_name)
            if entry is None:
                val = 0.0
            elif isinstance(entry, (list, tuple)):
                val = entry[k]
            else:
                val = rng.gauss(0.0, entry)
            history.realizations[w.local_name].append(val)
            for vd in _members(w):
                vd.set_value(val * _factor(vd, pmap))

        # simulate one sample and feed the state back into both hooks
        _solve(plant, "process", k)
        for c_hook, p_hook, src, label in zip(c_hooks, p_hooks, p_read, labels):
            val = pyo.value(src) / _factor(src, pmap)
            c_hook.set_value(val)
            p_hook.set_value(val)
            history.states[label].append(val)
        history.times.append(t0 + (k + 1) * dt)

    return history
