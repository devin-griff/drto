# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The NMPC loop with the solve time in it: ``drto.nonideal_nmpc``
(feature 016).

The loop of ``drto.ideal_nmpc`` with each move taking effect one delay
after the measurement it was solved at, the previous move holding until
then. Each interval therefore runs as two simulations of the one-sample
plant, the delay under the previous move and the rest under the new one.

The plant is discretized once and simulates a piece of any length
through a duration Param: each declared dynamics member is rewritten as
``coefficient * dz/dt == hP / h * other``, with ``hP`` the piece length
and ``h`` the declared spacing, so over an element of the declared
length the state advances as it would over ``hP``.

``delay="solver"`` takes each solve's ``timing_info.wall_time``, the
seconds the solver interface reports, and converts them into the
declared time units, read from a declared state and its DerivativeVar as
``units(z) / units(dz/dt)``. A model those seconds do not convert to
takes a prescribed delay instead.
"""
from dataclasses import dataclass, field

import pyomo.environ as pyo
from pyomo.core import Param
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyo_units

from drto.declarations import _dynamics_sides
from drto.dynamic_optimization import _members
from drto.ideal_nmpc import NmpcHistory, _first_move, _loop_setup
from drto.info import info
from drto.warm_start import warm_start_dynamic

#: The Param the plant's declared dynamics are scaled by.
_DURATION = "_drto_duration"


@dataclass
class NonidealNmpcHistory(NmpcHistory):
    """``drto.ideal_nmpc``'s history, plus what the delay adds.

    ``delays`` holds each step's delay in the declared time units, as
    measured or as prescribed, and ``effect_times`` the instant each
    move took effect, one entry per step. ``clamped`` lists the steps
    whose delay reached the interval, where the previous move held
    throughout and the new move took effect at the next boundary.
    """

    delays: list = field(default_factory=list)
    effect_times: list = field(default_factory=list)
    clamped: list = field(default_factory=list)


def _time_units(model, fn):
    """The declared time units, as ``units(z) / units(dz/dt)``.

    A ContinuousSet carries no units of its own, and a declared state
    and its DerivativeVar carry the ones the model writer gave them.
    """
    reg = info(model)
    time = reg.components("horizon")[0]
    con = reg.components("dynamics")[0]
    cd = next(iter(_members(con)))
    deriv, _coeff, _other = _dynamics_sides(cd, time, fn)
    state = deriv.parent_component().get_state_var()
    member = next(iter(state.values())) if state.is_indexed() else state
    return pyo_units.get_units(pyo_units.get_units(member) / pyo_units.get_units(deriv))


def _duration_param(plant, dt, units, fn):
    """Scale the plant's declared dynamics by a mutable piece length.

    Each member is rewritten once, in place, so the plant simulates a
    piece of any length on the mesh it was discretized on.
    """
    reg = info(plant)
    time = reg.components("horizon")[0]
    plant.add_component(
        _DURATION, Param(initialize=dt, mutable=True, units=units, doc="piece length")
    )
    duration = plant.component(_DURATION)
    for con in reg.components("dynamics"):
        for cd in _members(con):
            deriv, coeff, other = _dynamics_sides(cd, time, fn)
            side = deriv if coeff is None else coeff * deriv
            cd.set_value(side == duration / (dt * units) * other)
    return duration


def _delay_of(delay, steps, units, fn):
    """Check ``delay`` and return the step's delay as a function of the solve."""
    if isinstance(delay, str):
        if delay != "solver":
            raise ValueError(
                f"drto: {fn}: delay is 'solver', a number, or one number per "
                f"step. Got {delay!r}."
            )
        try:
            pyo_units.convert_value(1.0, from_units=pyo_units.s, to_units=units)
        except UnitsError as err:
            raise ValueError(
                f"drto: {fn}: delay='solver' converts the solver's seconds "
                f"into the declared time units, read from a declared state "
                f"and its DerivativeVar as {units}. Seconds do not convert "
                f"to that. Prescribe the delay in the declared units "
                f"instead."
            ) from err
        return lambda k, res: pyo_units.convert_value(
            float(res.timing_info.wall_time), from_units=pyo_units.s, to_units=units
        )
    if isinstance(delay, (list, tuple)):
        if len(delay) < steps:
            raise ValueError(
                f"drto: {fn} runs {steps} steps but the delay sequence has "
                f"{len(delay)} values. Give one per step."
            )
        return lambda k, res: float(delay[k])
    return lambda k, res: float(delay)


def nonideal_nmpc(
    build,
    steps,
    h=None,
    ncp=3,
    scheme="LAGRANGE-RADAU",
    initial_condition=None,
    dynamic_optimization=None,
    delay="solver",
    disturbances=None,
    seed=None,
    initialize="cold",
    solver="pounce",
    scale=None,
    warm_start=None,
    tee=False,
):
    """Run the NMPC loop with the solve time in it, for ``steps`` samples.

    See the module docstring. Each step solves the controller at the
    measurement, holds the previous move for the delay, and implements
    the new move for the rest of the interval. The first step's previous
    move is the declared control targets, which the process already
    holds.

    Parameters
    ----------
    build : callable
        The model statement. The loop builds the controller over the
        declared horizon and the process over one sampling interval, as
        ``drto.ideal_nmpc`` does. The builder contract is feature 006's.
    steps : int
        The loop length, in intervals.
    h, ncp, scheme, initial_condition, dynamic_optimization
        As in ``drto.ideal_nmpc``.
    delay : str, number, or sequence
        ``"solver"`` (the default) takes each solve's reported wall time
        and converts it into the declared time units. A number is that
        delay at every step, and a sequence one per step, both in the
        declared units.
    disturbances, seed, initialize, solver, scale, warm_start, tee
        As in ``drto.ideal_nmpc``. The step's realization holds over both
        pieces of its interval.

    Returns
    -------
    NonidealNmpcHistory
        The actual closed-loop trajectory, with each step's delay and
        the instant each move took effect. ``drto.plot_states`` and
        ``drto.plot_controls`` draw it, the moves stepping at those
        instants.

    Raises
    ------
    ValueError
        On a delay that is neither ``"solver"``, a number, nor one number
        per step, on ``delay="solver"`` where the solver's seconds do not
        convert into the declared time units, or on any input
        ``drto.ideal_nmpc`` rejects.
    RuntimeError
        If the solver is not available, or a solve fails (the error names
        the step).
    """
    fn = "nonideal_nmpc"
    loop = _loop_setup(
        fn,
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
        history_type=NonidealNmpcHistory,
    )
    (plant,) = loop.plants
    units = _time_units(plant.model, fn)
    delay_of = _delay_of(delay, steps, units, fn)
    duration = _duration_param(plant.model, loop.dt, units, fn)
    history = loop.history

    pending = None
    try:
        for k in range(steps):
            # a move the previous step could not fit takes effect here, at
            # the boundary, and holds until the new one does
            if pending is not None:
                loop.write_moves(pending, plant)
                pending = None

            # solve at the measurement, warm-started after the first step
            if k > 0:
                warm_start_dynamic(loop.ctrl)
            res = loop.solve(
                loop.ctrl, "controller", k, options=loop.warm_opts if k > 0 else None
            )
            moves = [pyo.value(_first_move(u)) for u in loop.controls]

            # the delay splits the interval. A delay at or past its end
            # leaves the new move for the next boundary, which is where the
            # next step's hold implements it
            d = delay_of(k, res)
            held = min(max(d, 0.0), loop.dt)
            history.delays.append(d)
            history.effect_times.append(loop.history.times[k] + held)
            if d >= loop.dt:
                history.clamped.append(k)
            loop.implement(moves)
            loop.realize(k, plant)

            # the process runs the interval in pieces, the previous move
            # for the delay and the new move for the rest, each piece's end
            # state starting the next
            if held > 0:
                duration.set_value(held)
                loop.solve(plant.model, "process", k)
                loop.carry(plant)
            if held < loop.dt:
                loop.write_moves(moves, plant)
                duration.set_value(loop.dt - held)
                loop.solve(plant.model, "process", k)
            else:
                pending = moves
            loop.write_state(loop.measure(plant, k))
    finally:
        loop.release()

    return history
