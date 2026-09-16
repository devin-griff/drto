# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The advanced-step NMPC loop: ``drto.asnmpc`` (feature 015).

The loop of ``drto.ideal_nmpc`` with the horizon solve moved between
samples, after Huang, Zavala, and Biegler, J. Process Control 19 (2009)
678-685. During each sample a predictor, the controller's own model run
forward with its disturbances at zero, gives the state the controller
expects next, and the controller solves there with pounce. When the
process reaches the end of the sample, the measured state is written into
the initial-condition Params and ``drto.advanced_step_controller``
corrects the background solution to it with a backsolve. The corrected
first moves are the next implemented control.

The setup, the options, the disturbance handling, and the history are
the ideal loop's, through the same ``_loop_setup`` with one more
one-sample side for the predictor. pounce keeps one factorization per
model and a solve replaces only its own, so the process solve between the
background solve and the correction leaves the controller's in place.
"""
from collections.abc import Mapping

import pyomo.environ as pyo

from drto.advanced_step import advanced_step_controller
from drto.ideal_nmpc import _first_move, _loop_setup
from drto.scaling import _POUNCE_SOLVERS
from drto.warm_start import warm_start_dynamic


def asnmpc(
    build,
    steps,
    h=None,
    ncp=3,
    scheme="LAGRANGE-RADAU",
    initial_condition=None,
    dynamic_optimization=None,
    advanced_step=None,
    disturbances=None,
    seed=None,
    initialize="cold",
    solver="pounce",
    scale=None,
    warm_start=None,
    tee=False,
):
    """Run the advanced-step NMPC loop for ``steps`` samples. See the module
    docstring.

    Before the loop, the controller is solved in full at the initial
    state, and that solution's first moves are implemented as solved,
    since there is no background solution yet to correct. Each step ``k``
    then implements its moves on the process and the predictor, writes
    its disturbance values into the process, simulates the predictor one
    sample and solves the controller at the predicted state, simulates the
    process one sample, records the measured state, and corrects the
    background solution to it for the next step's moves. The last step
    skips the prediction, the background solve, and the correction, since
    nothing implements their moves, so the loop makes ``steps`` controller
    solves, as the ideal loop does.

    Parameters
    ----------
    build : callable
        The model statement. The loop builds the controller over the
        declared horizon, and the process and the predictor over one
        sampling interval, all from it. The builder contract is feature
        006's.
    steps : int
        The loop length, in samples.
    h, ncp, scheme, initial_condition, dynamic_optimization
        As in ``drto.ideal_nmpc``. The initial condition lands in every
        side.
    advanced_step : mapping, optional
        Keyword arguments passed to ``drto.advanced_step_controller`` as
        given, and through it to ``pyomo_pounce.sens_solution``.
    disturbances, seed, initialize, scale, warm_start
        As in ``drto.ideal_nmpc``. The realizations reach the process
        alone, and ``initialize`` and ``scale`` apply to every side.
    solver : str
        ``"pounce"`` or ``"pounce_v2"``, for every solve. The correction
        is a backsolve on the controller's pounce factorization, so no
        other solver can run this loop.
    tee : bool
        As in ``drto.ideal_nmpc``. The logs hold one ``(step, side,
        text)`` entry per solve in loop order, the side being
        ``"controller"``, ``"predictor"``, or ``"process"`` and the step
        the one the solve ran in.

    Returns
    -------
    NmpcHistory
        The actual closed-loop trajectory. ``drto.plot_states`` and
        ``drto.plot_controls`` draw it.

    Raises
    ------
    ValueError
        On a solver other than pounce, an ``advanced_step`` that is not a
        mapping, or any input ``drto.ideal_nmpc`` rejects.
    RuntimeError
        If pyomo-pounce is missing, or a solve fails (the error names the
        step).
    """
    fn = "asnmpc"
    if solver not in _POUNCE_SOLVERS:
        raise ValueError(
            f"drto: {fn} corrects each solution with a backsolve on its "
            f"pounce factorization, so the solver is one of "
            f"{', '.join(repr(n) for n in _POUNCE_SOLVERS)}. Got {solver!r}."
        )
    if advanced_step is not None and not isinstance(advanced_step, Mapping):
        raise ValueError(
            f"drto: {fn}: advanced_step is a mapping of keyword arguments "
            f"for drto.advanced_step_controller. Got {advanced_step!r}."
        )

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
        plants=2,
    )
    process, predictor = loop.plants
    ctrl = loop.ctrl
    correction = dict(advanced_step or {})

    try:
        # the first moves come from a full solve at the initial state
        loop.solve(ctrl, "controller", 0)
        moves = [pyo.value(_first_move(u)) for u in loop.controls]

        for k in range(steps):
            last = k == steps - 1
            loop.implement(moves, process, predictor)
            loop.realize(k, process)

            if not last:
                # the predictor runs the controller's own model one sample, its
                # disturbances at zero, and the controller solves at the state
                # it reaches. pounce keeps this solve's factorization on the
                # controller
                loop.solve(predictor.model, "predictor", k)
                loop.write_state([pyo.value(src) for src in predictor.reads])
                warm_start_dynamic(ctrl)
                loop.solve(ctrl, "controller", k, options=loop.warm_opts)

            # the process runs one sample under the realization, and the state
            # it reaches is the measurement
            loop.solve(process.model, "process", k)
            measured = loop.measure(process, k)

            if not last:
                # correct the background solution to the measurement. The model
                # keeps that solution for the next warm start
                loop.write_state(measured)
                est = advanced_step_controller(ctrl, **correction)
                moves = [est[_first_move(u)] for u in loop.controls]
    finally:
        loop.release()

    return loop.history
