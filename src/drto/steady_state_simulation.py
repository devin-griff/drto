# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Steady-state simulation: ``drto.steady_state_simulation`` (feature 008).

Reduces the model to steady state with the controls fixed and a zero
objective, the square problem whose solution is the equilibrium under the
given inputs. A dynamic model (horizon and dynamics declared) first composes
``drto.dynamic_to_steady_state`` (feature 005), and a model authored directly
as steady-state skips the reduction. Either way the declared controls are fixed,
at supplied values or at the values they already hold, the optimization-only
constructs (the costs and the terminal constraint) leave the model through the
routine the dynamic simulation shares, the estimation declarations are
neutralized the same way the dynamic modes do, and ``drto.build_objective``
installs the simulation's constant-zero objective.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory

from drto.dynamic_optimization import _fix_disturbances, _neutralize_estimation
from drto.dynamic_simulation import _fix_controls, _shed_optimization_constructs
from drto.info import info
from drto.objective import build_objective


def steady_state_simulation(build, controls=None, disturbances=None):
    """Build a model and reduce it to the fixed-input equilibrium.

    Takes the model statement rather than a model, so a script that already
    has a builder reaches its equilibrium in one call. The builder contract
    is feature 006's: ``build`` returns a declared, undiscretized model, its
    first two parameters are the interval count and the sampling time, and
    every parameter has a default, so the bare ``build()`` this makes is
    legal. The steady modes pass no ``N`` and no ``h``, since the reduction
    collapses the grid either way.

    Nothing is discretized on this path. The registered transformation
    composes ``drto.dynamic_to_steady_state`` (feature 005) for a model
    declaring a horizon and dynamics, and a statement that constructs its
    steady form natively takes that reduction's skip.

    Parameters
    ----------
    build : callable
        The model statement, called with no arguments.
    controls : mapping, optional
        Passed to the registered transformation when given, mapping a
        declared control to the value it is fixed at. The keys are names
        here, since the build happens inside the call and the caller never
        holds the components the transformation's other key form takes.
    disturbances : mapping, optional
        Passed to the registered transformation when given, mapping a
        declared disturbance to its standing realization, keyed by name
        for the same reason.

    Returns
    -------
    Block
        The square equilibrium problem. This is the object ``build``
        returned, since the function owns the model it just built and
        transforms it in place. A caller holding a model of its own keeps
        ``TransformationFactory('drto.steady_state_simulation').create_using``
        for the form that leaves the source unchanged.

    Examples
    --------
    ::

        sim = drto.steady_state_simulation(build, controls={"u": 0.3})
    """
    m = build()
    opts = {}
    if controls is not None:
        opts["controls"] = controls
    if disturbances is not None:
        opts["disturbances"] = disturbances
    TransformationFactory("drto.steady_state_simulation").apply_to(m, **opts)
    return m


@TransformationFactory.register(
    "drto.steady_state_simulation",
    doc="Reduce to steady state, fix the controls, and install the zero "
    "objective, the fixed-input equilibrium solve (drto).",
)
class SteadyStateSimulationTransformation(Transformation):
    """The steady-state simulation mode. See the module docstring.

    Options: ``controls`` maps a declared control (the component, or its
    name) to the value it is fixed at. Controls not in the mapping fix at
    the value they already hold. Components from the source model resolve
    by name, so ``create_using(m, controls={m.u: 0.3})`` works on the
    clone. A scalar Var is unhashable as a plain dict key, so a control on
    an already-steady model goes in by name (or a ``ComponentMap``).
    """

    CONFIG = ConfigDict("drto.steady_state_simulation")
    CONFIG.declare(
        "controls",
        ConfigValue(
            default=None,
            description="Mapping of declared control (component or name) to "
            "the value it is fixed at. Controls not in the mapping fix at "
            "the value they already hold.",
        ),
    )
    CONFIG.declare(
        "disturbances",
        ConfigValue(
            default=None,
            description="Mapping of declared disturbance (component or name) to "
            "the constant standing value it is fixed at. A disturbance not in "
            "the mapping is fixed at zero.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        # resolve component keys to names before any rebuild. create_using
        # remaps the mapping onto the clone, and the reduction below
        # replaces the very components the keys point at, detaching them,
        # and a detached component's name degrades to its local name
        controls = {
            (k if isinstance(k, str) else k.name): v
            for k, v in (config.controls or {}).items()
        }
        disturbances = {
            (k if isinstance(k, str) else k.name): v
            for k, v in (config.disturbances or {}).items()
        }
        reg = info(model)
        if not reg.has_declaration("state"):
            raise ValueError(
                "drto: steady_state_simulation requires declared states "
                "(drto.state first)."
            )
        # neutralize the estimation costs before the reduction, which
        # collapses the control-side costs and not the window-based
        # estimation costs, so those must leave the model before it runs
        outcome = _neutralize_estimation(reg, "steady_state_simulation")

        if reg.has_declaration("horizon") and reg.has_declaration("dynamics"):
            TransformationFactory("drto.dynamic_to_steady_state").apply_to(model)

        # the mode installs no cost and no terminal set, through the routine
        # the dynamic simulation shares. The steady-state target Params stay
        # (they may appear in a deviation-form model's equations), so their
        # records stay with them and the registry mirrors the model.
        dropped = _shed_optimization_constructs(reg)

        # the reduction replaced the control components, so names are the
        # stable handle, resolved above while the keys were still attached.
        # The dynamic simulation fixes its controls the same way, through
        # the routine both modes share
        fixed = _fix_controls(reg, controls, "steady_state_simulation")

        # the reduction collapsed a disturbance to a single point, so fix it at
        # the standing realization, defaulting to zero
        noise = _fix_disturbances(reg, disturbances, "steady_state_simulation")

        build_objective(model, zero=True)
        reg.record_transformation(
            "drto.steady_state_simulation",
            controls=", ".join(fixed) if fixed else "(none declared)",
            **({"disturbances": ", ".join(noise)} if noise else {}),
            **({"dropped": ", ".join(dropped)} if dropped else {}),
            **outcome,
        )
        return model
