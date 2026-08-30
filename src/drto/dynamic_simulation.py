# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic simulation: ``drto.dynamic_simulation`` (feature 007).

Prepares the declared dynamic model to be solved forward over the horizon.
The declared control profiles are applied, the controls are fixed at the
values they hold or at supplied values, and ``drto.build_objective`` installs
the constant-zero objective. The mode frees nothing, so the result is the
square forward integration of the model as declared, and the horizon is kept.

The profiles are applied before the controls are fixed, so the simulated input
takes the shape the model declared. The user chooses that shape at declaration
time through ``control(profile=...)``.

The mode installs no cost, so the declared stage and terminal cost equations
leave the model, as in ``drto.steady_state_simulation`` (feature 008). The
estimation costs and measurements are neutralized through the routine shared
with the other control-side modes, and each disturbance is fixed at its
realization (default zero), the same way the controls are fixed, so the plant
can be driven by a supplied noise sequence and the system stays square.
"""
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Transformation, TransformationFactory

from drto.dynamic_optimization import (
    _build_and_discretize,
    _fix_disturbances,
    _members,
    _neutralize_estimation,
    _spread,
)
from drto.info import info
from drto.objective import build_objective

#: The declarations the transform requires. A forward integration needs the
#: initial state pinned, or the horizon problem is not square.
_REQUIRED = ("horizon", "state", "dynamics", "control", "initial_condition")

#: The optimization-only constructs a simulation sheds. The mode installs no
#: cost, and a terminal constraint would over-constrain the square forward
#: integration.
#: Shared with ``drto.steady_state_simulation`` (feature 008) through
#: ``_shed_optimization_constructs`` so the two modes cannot diverge.
_SIMULATION_SHED = (
    "tracking_stage_cost",
    "economic_stage_cost",
    "move_suppression",
    "tracking_terminal_cost",
    "terminal_constraint",
)


def _shed_optimization_constructs(reg):
    """Delete the optimization-only cost and constraint declarations.

    The stage costs, the terminal cost, and the terminal constraint leave the
    model and their records are purged. The cost variables are left unused.
    Returns the dropped display names for the transformation log.
    """
    dropped = []
    for kind in _SIMULATION_SHED:
        for record in reg.declarations(kind):
            comp = record["component"]
            if comp.parent_block() is not None:
                comp.parent_block().del_component(comp)
        if reg.has_declaration(kind):
            dropped.append(kind.replace("_", " "))
        reg._declarations.pop(kind, None)
    return dropped


def _fix_controls(reg, requested, fn):
    """Fix each declared control, at a supplied value or the one it holds.

    Shared by the two simulation modes, which fix their controls the same
    way, with ``fn`` naming the caller in the errors. Resolution is by
    name. ``create_using`` hands keys from the source model, and
    parameterizing and the steady reduction both replace the control
    components, so the name is the stable handle. Returns the display
    names for the transformation log.

    ``drto.steady_state_simulation`` does not require a declared control,
    so a model with none reaches the empty listing in the first error.
    """
    declared = {c.name: c for c in reg.components("control")}
    wanted = {}
    for key, val in requested.items():
        name = key if isinstance(key, str) else key.name
        if name not in declared:
            raise ValueError(
                f"drto: {fn} got a value for '{name}', which is not a "
                f"declared control. The declared controls are "
                f"{', '.join(declared) or '(none)'}."
            )
        wanted[name] = val

    fixed = []
    for name, comp in declared.items():
        members = list(_members(comp))
        if name in wanted:
            values = _spread(wanted[name], len(members), name, fn)
            for vd, v in zip(members, values):
                vd.set_value(v)
        for vd in members:
            if vd.value is None:
                raise ValueError(
                    f"drto: {fn} fixes '{name}' at the value it already "
                    f"holds, but it has none. Pass "
                    f"controls={{{name}: value}} or initialize it."
                )
            vd.fix()
        fixed.append(f"{name}={wanted[name]}" if name in wanted else f"{name} (held)")
    return fixed


def dynamic_simulation(
    build,
    N=None,
    h=None,
    ncp=3,
    scheme="LAGRANGE-RADAU",
    controls=None,
    disturbances=None,
):
    """Build a model, discretize it, and prepare the forward integration.

    Takes the model statement rather than a model, so one call reaches a
    square system ready to solve. The builder contract is feature 006's:
    ``build`` returns a declared, undiscretized model, its first two
    parameters are the interval count and the sampling time named ``N``
    and ``h``, and every parameter has a default, so the bare ``build()``
    is legal.

    A builder holds its constant inputs by fixing them at the declared
    sample points, which is all an undiscretized model has, and
    discretization here completes them, the same rule feature 006 states
    for ``drto.dynamic_optimization`` and the same code.

    Parameters
    ----------
    build : callable
        The model statement. Called with whichever of ``N`` and ``h`` were
        given, by keyword, so an omitted one takes the builder's default.
    N : int, optional
        Intervals, passed to the builder.
    h : float, optional
        Sampling time, passed to the builder.
    ncp : int, optional
        Collocation points per finite element (default 3).
    scheme : str, optional
        The collocation scheme (default ``"LAGRANGE-RADAU"``).
    controls : mapping, optional
        Passed to the registered transformation when given, mapping a
        declared control to the value it is fixed at. The keys are names
        here, since the build happens inside the call and the caller never
        holds the components the transformation's other key form takes.
    disturbances : mapping, optional
        Passed to the registered transformation when given, mapping a
        declared disturbance to its realization, keyed by name for the
        same reason.

    Returns
    -------
    Block
        The model the builder returned, discretized and prepared in place.
        It is square, so an NLP solver integrates it forward.

    Raises
    ------
    ValueError
        If the builder returns a model whose declared time set is already
        discretized, since this function owns the mesh.

    Examples
    --------
    ::

        plant = drto.dynamic_simulation(build, N=1, controls={"u": 0.3})
    """
    m = _build_and_discretize(build, N, h, ncp, scheme, "dynamic_simulation")
    opts = {}
    if controls is not None:
        opts["controls"] = controls
    if disturbances is not None:
        opts["disturbances"] = disturbances
    TransformationFactory("drto.dynamic_simulation").apply_to(m, **opts)
    return m


@TransformationFactory.register(
    "drto.dynamic_simulation",
    doc="Fix the controls and prepare the declared model for forward "
    "integration over the horizon (drto).",
)
class DynamicSimulationTransformation(Transformation):
    """The dynamic simulation mode. See the module docstring.

    Options: ``controls`` maps a declared control (the component, or its name)
    to what it is fixed at, either a constant held across the horizon or one
    value per free point the applied profile leaves. Controls not in the
    mapping fix at the values they already hold. ``disturbances`` maps a
    declared disturbance the same way, the plant's realized noise. A
    disturbance not in the mapping is fixed at zero. Components from the source
    model resolve by name, so ``create_using(m, controls={m.u: 0.3})`` works on
    the clone.

    ``apply_to`` prepares in place. ``create_using`` prepares a clone and
    leaves the source model alone.
    """

    CONFIG = ConfigDict("drto.dynamic_simulation")
    CONFIG.declare(
        "controls",
        ConfigValue(
            default=None,
            description="Mapping of declared control (component or name) to "
            "the value it is fixed at: a constant held across the horizon, or "
            "one value per free point of the applied profile. Controls not in "
            "the mapping fix at the values they already hold.",
        ),
    )
    CONFIG.declare(
        "disturbances",
        ConfigValue(
            default=None,
            description="Mapping of declared disturbance (component or name) to "
            "its realization: a constant held across the horizon, or one value "
            "per free point of the applied profile. A disturbance not in the "
            "mapping is fixed at zero.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        # resolve component keys to names before the profiles are applied.
        # Parameterizing replaces the very components the keys point at,
        # detaching them, and a detached component's name degrades to its
        # local name
        controls = {
            (k if isinstance(k, str) else k.name): v
            for k, v in (config.controls or {}).items()
        }
        disturbances = {
            (k if isinstance(k, str) else k.name): v
            for k, v in (config.disturbances or {}).items()
        }
        reg = info(model)
        missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
        if missing:
            raise ValueError(
                f"drto: dynamic_simulation requires the declarations "
                f"{', '.join(_REQUIRED)}. Missing: {', '.join(missing)}."
            )

        # the mode installs no cost and no terminal set, so the cost
        # equations and the terminal constraint leave the model
        dropped = _shed_optimization_constructs(reg)

        outcome = _neutralize_estimation(reg, "dynamic_simulation")

        # the declared profiles shape the simulated input, so they are applied
        # before the controls and disturbances are fixed
        TransformationFactory("drto.parameterize").apply_to(model)
        fixed = _fix_controls(reg, controls, "dynamic_simulation")
        noise = _fix_disturbances(reg, disturbances, "dynamic_simulation")

        build_objective(model, zero=True)
        reg.record_transformation(
            "drto.dynamic_simulation",
            horizon="kept",
            controls=", ".join(fixed),
            **({"disturbances": ", ".join(noise)} if noise else {}),
            **({"dropped": ", ".join(dropped)} if dropped else {}),
            **outcome,
        )
        return model
