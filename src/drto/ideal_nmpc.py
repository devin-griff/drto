# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The ideal NMPC loop: ``drto.ideal_nmpc`` (feature 014).

One call runs the closed loop the declarations describe: at each step the
dynamic optimization solves at the current state, the first control move
is implemented on the process, and the process simulates one sample
forward under that move and a disturbance realization. Ideal means the
solve is treated as instantaneous: measurement, solve, and move all land
at the same instant.

The input is the model statement (feature 006), and the loop builds its
sides from it: the controller over the declared horizon through
``drto.dynamic_optimization``, and the process over one sampling
interval through ``drto.dynamic_simulation`` with the controls first
fixed at the declared control targets. The first solve is initialized
per the ``initialize`` option, the cold start by default on the
controller and the process alike, and every later one warm-starts from
the shifted previous solution, ipopt adding the warm start recipe and
pounce ``mu_init=1e-6`` alone, the ``warm_start`` mapping laid over the
documented default.

``_loop_setup`` holds everything before the first solve, and
``drto.asnmpc`` (feature 015) calls it too, with a second one-sample
side for its predictor.

An active ``scaling_factor`` suffix reaches every solver solve on that
side, and the history reads back in the model's own units. The
initial-condition Params stay physical, and the cold starts' block solves
run in the model's own units either way (gh #92). ``scale`` given a
source is forwarded to ``drto.scale`` on each side once that side is
built, so every side holds the factors and every solver solve receives
them. They are written once and held for the whole loop.

The returned :class:`NmpcHistory` holds the actual trajectory under the
declared names, and ``drto.plot_states`` / ``drto.plot_controls`` draw
it directly.
"""
import contextlib
import io
import random
from collections.abc import Mapping
from dataclasses import dataclass, field

import pyomo.environ as pyo
from pyomo.core import TransformationFactory

from drto.cold_start import _target, cold_start_dynamic
from drto.declarations import _is_var_member, _side_matching
from drto.dynamic_optimization import _build_and_discretize, _members, _spread
from drto.infinite_horizon import _join_index, _split_index, _time_index
from drto.info import info
from drto.initialize_steady_state import initialize_steady_state
from drto import scaling as drto_scaling
from drto.scaling import _POUNCE_SOLVERS, _prune_suffixes
from drto.warm_start import warm_start_dynamic

#: The default recipe for the warm-started solves under ipopt, the one
#: solver measured to gain from all of it. The ``warm_start`` option
#: lays over this.
_WARM_START_OPTIONS = {
    "warm_start_init_point": "yes",
    "mu_init": 1e-6,
    "warm_start_bound_push": 1e-9,
    "warm_start_mult_bound_push": 1e-9,
}

#: The pounce warm default: the shifted start with the barrier already
#: small, the one recipe option measured to help pounce. The full
#: recipe regresses it to 867 iterations on the CSTR warm start, and
#: the regression needs the warm-start switch, the small barrier, and
#: the 1e-9 pushes together.
_POUNCE_WARM_OPTIONS = {"mu_init": 1e-6}

#: The solvers that read the full warm start recipe.
_WARM_SOLVERS = ("ipopt",)


def _warm_options(solver):
    """The warm-started solves' default options for ``solver``."""
    if solver in _WARM_SOLVERS:
        return dict(_WARM_START_OPTIONS)
    if solver in _POUNCE_SOLVERS:
        return dict(_POUNCE_WARM_OPTIONS)
    return {}


#: The mode transforms. The loop applies its own, so the input comes first.
_TRANSFORMED = (
    "drto.parameterize",
    "drto.dynamic_optimization",
    "drto.dynamic_simulation",
    "drto.dynamic_to_steady_state",
)


@dataclass
class NmpcHistory:
    """The closed loop's actual trajectory, under the declared names.

    ``times`` holds the sample instants, the initial one included.
    ``states`` maps each pinned state member's label to its actual values
    at those instants, and ``moves`` and ``realizations`` map each control
    and disturbance to its per-step values, one shorter than ``times``.
    The targets are the declared steady-state pairings' values, the
    plots' setpoint lines. ``state_bounds`` and ``control_bounds`` map
    each label to the declared (lo, hi), the plots' bound lines. Under
    ``tee=True``, ``logs`` holds every solve's output in loop order as
    ``(step, side, text)``.
    """

    times: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    states: dict = field(default_factory=dict)
    moves: dict = field(default_factory=dict)
    realizations: dict = field(default_factory=dict)
    state_targets: dict = field(default_factory=dict)
    control_targets: dict = field(default_factory=dict)
    state_bounds: dict = field(default_factory=dict)
    control_bounds: dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"drto nmpc history: {max(0, len(self.times) - 1)} steps, "
            f"states {', '.join(self.states) or '(none)'}, "
            f"moves {', '.join(self.moves) or '(none)'}"
        )


def _pinned(reg, fn):
    """The initial-condition constraints' (pinned state member, Param) pairs.

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


def _one_sample(process):
    """Cut the process to its first sample: the plant the loop needs.

    The clone arrives with the declared horizon, but the loop reads the
    state one sample in, so the plant is built as the one-sample
    simulation: the terminal segment leaves whole (it serves the horizon
    problem, not the plant), and every time-indexed member past the
    first sampling time leaves with it, time-indexed sub-Blocks (an
    IDAES model's property and reaction blocks) included. Radau
    collocation is sequential by element, so the first element stands
    alone, square, given the initial condition and the fixed inputs.
    The parameterized inputs keep their later members. They are fixed,
    in no remaining equation, and never reach the solver.
    """
    from pyomo.core import Block, Constraint, Expression, Var

    reg = info(process)
    time = reg.components("horizon")[0]
    samples = reg.declarations("horizon")[0]["samples"]
    t1 = samples[1] + 1e-9

    seg = process.component("drto_ih")
    if seg is not None:
        process.del_component(seg)
        reg._segment.clear()

    for ctype in (Block, Constraint, Expression, Var):
        for comp in list(
            process.component_objects(ctype, active=None, descend_into=True)
        ):
            if comp.is_reference() or not comp.is_indexed():
                continue
            pos, subs = _time_index(comp, time)
            if pos is None:
                continue
            for idx in list(comp.keys()):
                _o, t = _split_index(idx, pos, len(subs))
                if t is not None and t > t1:
                    del comp[idx]


def _owner_and_params(reg, fn):
    """The pinned members' declared owners and the Params pinning them.

    A declared state may be a Reference over a member subset of an
    indexed Var (gh #20), so a pinned member matches its declared owner by
    data identity, the package convention. Returns the owner map, keyed by
    the member's id, and the Params grouped under the owner's local name.
    """
    time = reg.components("horizon")[0]
    t0 = time.first()
    owner = {}
    for z in reg.components("state"):
        pos, subs = _time_index(z, time)
        for idx in z:
            o, t = _split_index(idx, pos, len(subs))
            if t == t0:
                owner[id(z[idx])] = (z, o)
    params_of = {}
    for vd, param in _pinned(reg, fn):
        params_of.setdefault(owner[id(vd)][0].local_name, []).append(param)
    return owner, params_of


@dataclass
class _Plant:
    """A one-sample simulation side and what each step reads and writes."""

    model: object
    params: list
    reads: list
    controls: list
    disturbances: list


@dataclass
class _Loop:
    """A loop's sides, its history, and the writes and solves each step makes.

    ``controls`` and ``params`` are the controller's live controls and
    initial-condition Params, and every plant's lists line up with them
    positionally.
    """

    fn: str
    opt: object
    tee: bool
    ctrl: object
    plants: list
    controls: list
    params: list
    labels: list
    plan: dict
    rng: object
    warm_opts: dict
    suffix_opts: dict
    history: NmpcHistory
    t0: float
    dt: float

    def solve(self, model, what, step, options=None):
        """Solve one side and load the result, naming the step on failure.

        Returns the solver's results, which ``drto.nonideal_nmpc`` reads
        the solve time from.
        """
        opts = {**self.suffix_opts, **(options or {})}
        kwargs = dict(
            solver_options=opts,
            load_solutions=False,
            raise_exception_on_nonoptimal_result=False,
        )
        if self.tee:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = self.opt.solve(model, tee=True, **kwargs)
            text = buf.getvalue()
            print(text, end="")
            self.history.logs.append((step, what, text))
        else:
            res = self.opt.solve(model, **kwargs)
        if not drto_scaling.solved_to_optimality(res):
            raise RuntimeError(
                f"drto: {self.fn}: the {what} solve failed at step {step} "
                f"({res.termination_condition.name})."
            )
        res.solution_loader.load_vars()
        return res

    def implement(self, moves, *plants):
        """Record one move per control and write it into each plant."""
        for u, move in zip(self.controls, moves):
            self.history.moves[u.local_name].append(move)
        self.write_moves(moves, *plants)

    def write_moves(self, moves, *plants):
        """Write one move per control into each plant, recording nothing."""
        for plant in plants:
            for pu, move in zip(plant.controls, moves):
                for vd in _members(pu):
                    vd.set_value(move)

    def realize(self, k, plant):
        """Write step ``k``'s disturbance values into the plant and record them."""
        for w in plant.disturbances:
            entry = self.plan.get(w.local_name)
            if entry is None:
                val = 0.0
            elif isinstance(entry, (list, tuple)):
                val = entry[k]
            else:
                val = self.rng.gauss(0.0, entry)
            self.history.realizations[w.local_name].append(val)
            for vd in _members(w):
                vd.set_value(val)

    def measure(self, plant, k):
        """Read the plant's state one sample in and record it at step ``k``.

        The values are written into every plant's initial-condition
        Params, the state the next step starts from, and returned for the
        caller to write into the controller's.
        """
        values = [pyo.value(src) for src in plant.reads]
        for side in self.plants:
            for param, val in zip(side.params, values):
                param.set_value(val)
        for label, val in zip(self.labels, values):
            self.history.states[label].append(val)
        self.history.times.append(self.t0 + (k + 1) * self.dt)
        return values

    def carry(self, plant):
        """Write the plant's state one sample in into its own Params.

        The next piece of the same interval starts where this one ended,
        and nothing is recorded, since the interval is not over.
        """
        values = [pyo.value(src) for src in plant.reads]
        for param, val in zip(plant.params, values):
            param.set_value(val)
        return values

    def write_state(self, values):
        """Write a state into the controller's initial-condition Params."""
        for param, val in zip(self.params, values):
            param.set_value(val)

    def release(self):
        """Drop the controller's pounce factorization before the loop returns.

        A pounce solve keeps the factorization on the controller whenever
        its initial-condition Params are declared, and the loop discards the
        controller on return. pounce raises when its solver object is freed
        on a thread other than the one that made it, which a later garbage
        collection on another solve's output thread does, so the loop frees
        it here instead.
        """
        try:
            import pyomo_pounce
        except ImportError:
            return
        pyomo_pounce.sens_release_kkt(self.ctrl)


def _loop_setup(
    fn,
    build,
    steps,
    *,
    h,
    ncp,
    scheme,
    initial_condition,
    dynamic_optimization,
    disturbances,
    seed,
    initialize,
    solver,
    scale,
    warm_start,
    tee,
    plants,
    history_type=NmpcHistory,
):
    """Check a loop's arguments and build its sides from the statement.

    The controller is built over the declared horizon and ``plants``
    one-sample simulations beside it, all on the mesh stated once. The
    initial condition lands in every side, ``scale`` and ``initialize``
    apply to every side, and each side gets its mode transform.
    ``history_type`` is the history the loop fills, ``NmpcHistory``
    unless a loop records more. Returns the ``_Loop`` the steps run on.
    """
    if not callable(build):
        raise ValueError(
            f"drto: {fn} takes the model statement, a function returning a "
            f"declared, undiscretized model (feature 006). Got {build!r}."
        )
    if not (
        initialize is False
        or initialize in ("cold", "steady")
        or isinstance(initialize, Mapping)
    ):
        raise ValueError(
            f"drto: {fn}: initialize is 'cold' (a mapping passes the cold "
            f"start's options), 'steady', or False. Got {initialize!r}."
        )
    if steps < 1:
        raise ValueError(f"drto: {fn}: steps must be at least 1, got {steps}.")

    if solver in _POUNCE_SOLVERS:
        # importing registers the in-process plugin. Without it the name
        # falls back to a PATH executable behind pyomo's ASL wrapper,
        # a different solver than the one the drto stack is built on
        try:
            import pyomo_pounce  # noqa: F401
        except ImportError as err:
            raise RuntimeError(
                f"drto: {fn}: solver 'pounce' requires pyomo-pounce "
                f"(pip install drto[pounce], or pip install pyomo-pounce)."
            ) from err
    opt = drto_scaling.solver_by_name(solver)
    if not opt.available():
        raise RuntimeError(f"drto: {fn}: solver '{solver}' is not available.")

    do_opts = dict(dynamic_optimization or {})
    repeated = [k for k in ("h", "ncp", "scheme") if k in do_opts]
    if repeated:
        raise ValueError(
            f"drto: {fn} states the mesh once for every side, so "
            f"{', '.join(repeated)} belongs to {fn} itself rather than to "
            f"dynamic_optimization."
        )
    segment = do_opts.pop("infinite_horizon", False)

    # every side comes from the one statement, which makes them the same
    # physics. The controller spans the declared horizon and each simulation
    # one sampling interval, all on the mesh stated once
    ctrl = _build_and_discretize(build, do_opts.pop("N", None), h, ncp, scheme, fn)
    sims = [_build_and_discretize(build, 1, h, ncp, scheme, fn) for _ in range(plants)]
    sides = [ctrl, *sims]
    reg = info(ctrl)

    # the initial condition lands in every side's Params, the values the
    # first step reads
    owner, c_params_of = _owner_and_params(reg, fn)
    sim_params_of = [_owner_and_params(info(sim), fn)[1] for sim in sims]
    for key, val in (initial_condition or {}).items():
        name = key if isinstance(key, str) else key.local_name
        params = c_params_of.get(name)
        if params is None:
            raise ValueError(
                f"drto: {fn} got an initial condition for '{name}', which "
                f"is not a pinned state. The pinned states are "
                f"{', '.join(c_params_of) or '(none)'}."
            )
        values = _spread(val, len(params), name, fn)
        for side in [params] + [of[name] for of in sim_params_of]:
            for param, v in zip(side, values):
                param.set_value(v)

    # the per-step disturbance plan, validated before anything is solved
    declared_dist = [w.local_name for w in reg.components("disturbance")]
    plan = {}
    for key, val in (disturbances or {}).items():
        name = key if isinstance(key, str) else key.local_name
        if name not in declared_dist:
            raise ValueError(
                f"drto: {fn} got a realization for '{name}', which is not "
                f"a declared disturbance. The declared disturbances are "
                f"{', '.join(declared_dist) or '(none)'}."
            )
        if isinstance(val, (list, tuple)) and len(val) < steps:
            raise ValueError(
                f"drto: {fn} runs {steps} steps but the sequence for "
                f"'{name}' has {len(val)} values. Give one per step."
            )
        plan[name] = val

    # a scale source writes the factors on each side once it is built, so
    # every side holds them and every solver solve receives them. The cold
    # starts' block solves run in the model's own units either way (gh #92)
    if scale is not None:
        for side in sides:
            drto_scaling.scale(side, source=scale)

    # the steady initialization runs on each side after it is discretized
    # and before the transforms, the only point that function accepts
    if initialize == "steady":
        for side in sides:
            initialize_steady_state(side)

    # the simulations run with their controls fixed at the declared control
    # targets. Each is built over one sampling interval, so nothing is cut
    # away and none carries a terminal segment
    uss = list(reg.declarations("steady_state_control"))
    at_targets = {
        u.name: pyo.value(_target(uss, u, "steady_state_control", fn))
        for u in reg.components("control")
    }
    for sim in sims:
        TransformationFactory("drto.dynamic_simulation").apply_to(
            sim, controls=at_targets
        )
        _prune_suffixes(sim)

    # the controller takes the terminal segment when asked, then assembles
    if segment:
        seg_opts = segment if segment is not True else {}
        TransformationFactory("drto.infinite_horizon").apply_to(ctrl, **seg_opts)
    TransformationFactory("drto.dynamic_optimization").apply_to(ctrl, **do_opts)
    _prune_suffixes(ctrl)

    # the warm-started solves take the recipe under the solvers that read
    # it, with the warm_start mapping laid over
    warm_opts = _warm_options(solver)
    warm_opts.update(warm_start or {})

    # every side cold-starts alike, so the simulations' first solves start
    # initialized too. Each simulation spans one sampling interval, so its
    # cold start covers one element
    if initialize == "cold" or isinstance(initialize, Mapping):
        opts = {} if initialize == "cold" else dict(initialize)
        for side in sides:
            cold_start_dynamic(side, **opts)

    # with an active scaling_factor suffix, every solve on that side
    # receives the factors, through the solver option for the solvers
    # that take one, and the history reads back in the model's own
    # units. A solver that does not receive them warns here, once.
    suffix_opts = (
        drto_scaling._scaling_options(solver, fn)
        if drto_scaling._suffix_active(ctrl)
        else {}
    )

    samples = reg.declarations("horizon")[0]["samples"]
    t0, t1 = samples[0], samples[1]
    c_pins = _pinned(reg, fn)
    c_params = [p for _vd, p in c_pins]

    # the pinned members' labels and targets come from the declared
    # owner (the member-id map above)
    ss = list(reg.declarations("steady_state"))
    labels, targets = [], []
    for c_vd, _p in c_pins:
        z, o = owner[id(c_vd)]
        labels.append(
            z.local_name if not o else f"{z.local_name}[{','.join(map(str, o))}]"
        )
        tgt = _target(ss, z, "steady_state", fn)
        targets.append(pyo.value(tgt[o] if o else tgt))

    # each simulation's read points, one sample in, come from its own
    # underlying containers
    plant_sides = []
    for sim in sims:
        reg_s = info(sim)
        time_s = reg_s.components("horizon")[0]
        pins = _pinned(reg_s, fn)
        reads = []
        for vd, _p in pins:
            zp = vd.parent_component()
            pos, subs = _time_index(zp, time_s)
            po, _t = _split_index(vd.index(), pos, len(subs))
            reads.append(zp[_join_index(po, t1, pos)])
        plant_sides.append(
            _Plant(
                model=sim,
                params=[p for _vd, p in pins],
                reads=reads,
                controls=list(reg_s.components("control")),
                disturbances=list(reg_s.components("disturbance")),
            )
        )

    history = history_type()
    history.times.append(t0)
    for label, param, tgt, (vd, _p) in zip(labels, c_params, targets, c_pins):
        history.states[label] = [pyo.value(param)]
        history.state_targets[label] = tgt
        history.state_bounds[label] = (vd.lb, vd.ub)

    c_controls = list(reg.components("control"))
    ucss = list(reg.declarations("steady_state_control"))
    for u in c_controls:
        history.moves[u.local_name] = []
        history.control_targets[u.local_name] = pyo.value(
            _target(ucss, u, "steady_state_control", fn)
        )
        first = _first_move(u)
        history.control_bounds[u.local_name] = (first.lb, first.ub)
    for w in plant_sides[0].disturbances:
        history.realizations[w.local_name] = []

    return _Loop(
        fn=fn,
        opt=opt,
        tee=tee,
        ctrl=ctrl,
        plants=plant_sides,
        controls=c_controls,
        params=c_params,
        labels=labels,
        plan=plan,
        rng=random.Random(seed),
        warm_opts=warm_opts,
        suffix_opts=suffix_opts,
        history=history,
        t0=t0,
        dt=t1 - t0,
    )


def ideal_nmpc(
    build,
    steps,
    h=None,
    ncp=3,
    scheme="LAGRANGE-RADAU",
    initial_condition=None,
    dynamic_optimization=None,
    disturbances=None,
    seed=None,
    initialize="cold",
    solver="pounce",
    scale=None,
    warm_start=None,
    tee=False,
):
    """Run the ideal NMPC loop for ``steps`` samples. See the module
    docstring.

    Parameters
    ----------
    build : callable
        The model statement. The loop builds both sides from it, the
        controller over the declared horizon and the plant over one
        sampling interval, so the two grids agree by construction. The
        builder contract is feature 006's.
    steps : int
        The loop length, in samples.
    h : float, optional
        Sampling time, passed to both builder calls. Omitted, the
        builder's default.
    ncp : int, optional
        Collocation points per finite element, for both sides (default 3).
    scheme : str, optional
        The collocation scheme, for both sides (default
        ``"LAGRANGE-RADAU"``).
    initial_condition : mapping, optional
        Declared state (the component, or its name) to the value written
        into its initial-condition Params before the first step: a constant for
        every pinned member, or one value each. Omitted, the Params'
        current values are the first actual state.
    dynamic_optimization : mapping, optional
        The controller-only options: ``N``, the interval count passed to
        its builder call, ``infinite_horizon``, and ``tracking_weight``.
        ``h``, ``ncp``, and ``scheme`` belong to the loop, which states
        the mesh once for both sides, and repeating one here is an
        error.
    disturbances : mapping, optional
        Declared disturbance (the component, or its name) to its
        realization: a sequence gives the per-step values as given, a
        number the standard deviation of independent zero-mean normal draws.
        A disturbance with no entry is zero.
    seed : int, optional
        Seeds the disturbance draws, making them reproducible.
    initialize : str, mapping, or False
        The first solve's initialization. ``"cold"`` (the default) runs
        ``drto.cold_start_dynamic`` on the controller and the process
        alike, a mapping passing through as its options. ``"steady"``
        runs ``drto.initialize_steady_state`` on each side after it is
        discretized and before the transforms, the only point that
        function accepts. ``False`` skips initialization.
    solver : str
        The solver, by name, for every controller and process solve.
        Every solver warm starts between steps on the shifted
        values, and ipopt also receives the warm start recipe.
    scale : str or mapping, optional
        A ``drto.scale`` source, ``"point"``, ``"bounds"``, or a
        mapping of units to magnitudes. Given, the factors are written
        on each side once it is built, so both hold them and every
        solver solve receives them. The
        default, ``None``, writes nothing and honors a
        ``scaling_factor`` Suffix the caller wrote.
    warm_start : mapping, optional
        Solver options for the warm-started solves. Under ``"ipopt"``
        they lay over the default recipe (``warm_start_init_point=yes``,
        ``mu_init=1e-6``, ``warm_start_bound_push`` and
        ``warm_start_mult_bound_push`` at ``1e-9``). Under the pounce
        names they lay over ``mu_init=1e-6`` alone, and under any other
        solver the mapping is used as given and the default is the
        shifted values alone.
    tee : bool
        ``True`` streams every solve's output as the loop runs and
        returns it on the history's ``logs``, one ``(step, side,
        text)`` entry per solve in loop order. Quiet by default.

    Returns
    -------
    NmpcHistory
        The actual closed-loop trajectory. ``drto.plot_states`` and
        ``drto.plot_controls`` draw it.

    Raises
    ------
    ValueError
        On an input that is not callable, a mesh option repeated in
        ``dynamic_optimization``, an unknown state or disturbance name,
        or a disturbance sequence shorter than the loop.
    RuntimeError
        If the solver is not available, or a solve fails (the error
        names the step).
    """
    loop = _loop_setup(
        "ideal_nmpc",
        build,
        steps,
        h=h,
        ncp=ncp,
        scheme=scheme,
        initial_condition=initial_condition,
        dynamic_optimization=dynamic_optimization,
        disturbances=disturbances,
        seed=seed,
        initialize=initialize,
        solver=solver,
        scale=scale,
        warm_start=warm_start,
        tee=tee,
        plants=1,
    )
    (plant,) = loop.plants

    try:
        for k in range(steps):
            # solve the controller at the current state, warm-started after
            # the first step
            if k > 0:
                warm_start_dynamic(loop.ctrl)
            loop.solve(
                loop.ctrl, "controller", k, options=loop.warm_opts if k > 0 else None
            )

            # implement each control's first move and this step's
            # disturbances on the process, simulate one sample, and start the
            # next solve from the state it reaches
            loop.implement([pyo.value(_first_move(u)) for u in loop.controls], plant)
            loop.realize(k, plant)
            loop.solve(plant.model, "process", k)
            loop.write_state(loop.measure(plant, k))
    finally:
        loop.release()

    return loop.history
