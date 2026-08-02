# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Steady-state initialization: ``drto.initialize_steady_state`` (feature 010).

One function, both declared shapes. A model without a declared horizon and
dynamics (authored steady-state, or a feature 005 reduction) is initialized
in place: ``pyomo_pounce.initialize`` runs its fill, project, block-solve
pipeline with the declared controls as the decisions, and the solved values
land in ``Var.value``. A dynamic model (horizon and dynamics declared,
discretized, before any drto transformation) is initialized from its steady
state: a throwaway clone is reduced with ``drto.dynamic_to_steady_state``,
the same pipeline solves the equilibrium there, and the result broadcasts
flat across the horizon, every time-indexed variable at every grid point,
with the state derivatives at zero. The later transforms carry the values
forward on their own (cvp seeds its move variables from the members it
replaces; the terminal segment copies the horizon-end values), which is why
the dynamic path runs first.

An active ``scaling_factor`` suffix is honored: the pipeline runs scaled
and the solved values land in the model's own units, with no change to
the call.

pyomo-pounce is optional to drto: it is imported here at call time and a
missing install raises with the ``pip install drto[pounce]`` instruction.
Values only: no components are added or removed, and the pipeline restores
the variable fixed flags it touches.
"""
from dataclasses import dataclass

from pyomo.core import Suffix, TransformationFactory, Var
from pyomo.dae import DerivativeVar

from drto.infinite_horizon import _split_index, _time_index
from drto.info import info


@dataclass
class SteadyStateInitReport:
    """The dynamic path's report: the pipeline's, plus the broadcast."""

    #: The ``pyomo_pounce.initialize`` report from the reduced clone.
    pipeline: object = None
    n_broadcast_vars: int = 0
    n_grid_points: int = 0
    n_derivatives_zeroed: int = 0

    def __str__(self):
        lines = [
            "drto initialize_steady_state (reduce a clone -> solve -> broadcast)",
            f"  broadcast: {self.n_broadcast_vars} variables across "
            f"{self.n_grid_points} grid points, "
            f"{self.n_derivatives_zeroed} derivatives zeroed",
        ]
        lines.extend("  " + line for line in str(self.pipeline).splitlines())
        return "\n".join(lines)


def initialize_steady_state(m, controls=None):
    """Initialize ``m`` from its steady state; see the module docstring.

    Parameters
    ----------
    m : Block
        A declared model (feature 002): steady-state (initialized in
        place) or dynamic and discretized (initialized from a reduced
        clone, broadcast flat).
    controls : mapping, optional
        Declared control (the component, or its name) to the value the
        steady solve holds it at; controls not in the mapping hold the
        values they already have. The feature 008 convention.

    Returns
    -------
    InitializeReport or SteadyStateInitReport
        The pipeline's report (steady path), or the wrapper adding the
        broadcast line (dynamic path). Both print readably.

    Raises
    ------
    RuntimeError
        If pyomo-pounce is not installed.
    ValueError
        On a missing declaration, a guard violation, an unknown control,
        a valueless unheld control, or a non-square steady system (the
        error names the unmatched variables and constraints).
    """
    try:
        import pyomo_pounce
    except ImportError as err:
        raise RuntimeError(
            "drto: initialize_steady_state requires pyomo-pounce "
            "(pip install drto[pounce], or pip install pyomo-pounce)."
        ) from err

    reg = info(m)
    if not reg.has_declaration("state"):
        raise ValueError(
            "drto: initialize_steady_state requires declared states "
            "(drto.state first)."
        )
    dynamic = reg.has_declaration("horizon") and reg.has_declaration("dynamics")
    if not dynamic:
        return _run_pipeline(m, info(m), controls, pyomo_pounce)

    time = reg.components("horizon")[0]
    if not time.get_discretization_info():
        raise ValueError(
            "drto: initialize_steady_state broadcasts across the grid, so "
            "the dynamic model must be discretized first (apply a dae.* "
            "transformation)."
        )
    for name in ("drto.parameterize", "drto.infinite_horizon"):
        if reg.has_transformation(name):
            raise ValueError(
                f"drto: initialize_steady_state runs before the dynamic "
                f"transforms; '{name}' is already applied. Initialize "
                f"first: the transforms carry the values forward on their "
                f"own."
            )

    work = m.clone()
    TransformationFactory("drto.dynamic_to_steady_state").apply_to(work)
    # the reduction removed components; their suffix entries go with
    # them, or the scaled clone cannot be built and the NL writer warns
    for sfx in work.component_objects(Suffix, active=True):
        for key in [k for k in sfx if not _attached(k, work)]:
            del sfx[key]
    report = _run_pipeline(work, info(work), controls, pyomo_pounce)

    # broadcast: every time-indexed Var takes its collapsed counterpart's
    # value at every grid point; the derivatives are zero at steady state
    n_vars = 0
    for comp in m.component_objects(Var, active=True):
        if isinstance(comp, DerivativeVar):
            continue
        pos, subs = _time_index(comp, time)
        if pos is None:
            continue
        counterpart = work.find_component(comp.name)
        if counterpart is None:
            continue
        # a collapsed Reference is a container even over a single member
        # (a Reference-declared control, gh #18's shape), so a time-only
        # component reads its one member rather than the container
        point = (
            next(iter(counterpart.values()))
            if counterpart.is_indexed()
            else counterpart
        )
        for idx, vd in comp.items():
            o, _ = _split_index(idx, pos, len(subs))
            src = counterpart[o] if o else point
            vd.set_value(src.value)
        n_vars += 1
    n_deriv = 0
    for query in (DerivativeVar, Var):
        for dv in m.component_objects(query):
            if isinstance(dv, DerivativeVar) and dv.get_continuousset_list() == [time]:
                for vd in dv.values():
                    if vd.value != 0:
                        vd.set_value(0)
                        n_deriv += 1
    return SteadyStateInitReport(
        pipeline=report,
        n_broadcast_vars=n_vars,
        n_grid_points=len(time),
        n_derivatives_zeroed=n_deriv,
    )


def _attached(comp, root):
    """Whether ``comp`` still hangs off ``root``."""
    if comp.parent_component() is None:
        return False
    b = comp.parent_block()
    while b is not None:
        if b is root:
            return True
        b = b.parent_block()
    return False


def _run_pipeline(model, reg, controls, pyomo_pounce):
    """Resolve the control values, run the pipeline, enforce squareness."""
    declared = {c.name: c for c in reg.components("control")}
    requested = {}
    for key, val in (controls or {}).items():
        name = key if isinstance(key, str) else key.name
        if name not in declared:
            raise ValueError(
                f"drto: initialize_steady_state got a value for '{name}', "
                f"which is not a declared control; declared: "
                f"{', '.join(declared) or '(none)'}."
            )
        requested[name] = val
    for name, comp in declared.items():
        for vd in comp.values() if comp.is_indexed() else (comp,):
            if name in requested:
                vd.set_value(requested[name])
            elif vd.value is None:
                raise ValueError(
                    f"drto: initialize_steady_state holds '{name}' at the "
                    f"value it already has, but it has none; pass "
                    f"controls={{{name}: value}} or initialize it."
                )

    # a declared disturbance is process noise, zero in the nominal
    # equilibrium: held at zero for the solve, the same convention as
    # every control-side mode, with the touched fixed flags restored
    # after (gh #44)
    held_zero = []
    for comp in reg.components("disturbance"):
        for vd in comp.values() if comp.is_indexed() else (comp,):
            if not vd.fixed:
                vd.set_value(0.0)
                vd.fix()
                held_zero.append(vd)
    try:
        # an active scaling_factor suffix: the pipeline runs on a scaled
        # clone and the values propagate back; the model stays in its own
        # units, the same contract as cold_start_dynamic's per-point solves
        scaled = any(
            s.local_name == "scaling_factor"
            for s in model.component_objects(Suffix, active=True)
        )
        if scaled:
            xfrm = TransformationFactory("core.scale_model")
            target = xfrm.create_using(model, rename=False)
            decisions = list(info(target).components("control"))
        else:
            target, decisions = model, list(declared.values())
        report = pyomo_pounce.initialize(target, decisions=decisions)
        block = report.block
        if block is not None and not block.square:
            detail = []
            if block.underconstrained_variables:
                detail.append(
                    "underconstrained (specify or declare as controls): "
                    + ", ".join(block.underconstrained_variables)
                )
            if block.overconstrained_constraints:
                detail.append(
                    "overconstrained (redundant or conflicting): "
                    + ", ".join(block.overconstrained_constraints)
                )
            raise ValueError(
                "drto: initialize_steady_state found a non-square steady "
                "system, so the equilibrium is not fully determined; "
                + "; ".join(detail)
                + ". For deliberately partial initialization call "
                "pyomo_pounce.initialize directly."
            )
        if scaled:
            # values only: the throwaway clone's dual and rc suffixes would
            # make propagate_solution demand an objective to rescale them
            for _nm in ("dual", "rc"):
                _c = target.component(_nm)
                if _c is not None and _c.ctype is Suffix:
                    target.del_component(_c)
            xfrm.propagate_solution(target, model)
        return report
    finally:
        for vd in held_zero:
            vd.unfix()
