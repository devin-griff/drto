# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The infinite-horizon terminal segment: ``drto.infinite_horizon``
(feature 004).

Appends a terminal segment to a declared, discretized dynamic model: the tail
of the horizon to t = infinity, compressed onto [0, 1] by the transformation
``tau = tanh(gamma*(t - tN))`` of Dinh et al. (2025,
doi:10.1016/j.jprocont.2025.103565). The segment carries copies of the
declared states and controls, the declared dynamics dilated by the
transformation Jacobian ``gamma*(1 - tau^2)`` at interior Gauss-Legendre
collocation points, a hard equilibrium endpoint ``0 = f`` at ``tau = 1``, and
the declared tracking stage cost replicated at the segment's collocation
points. The tail cost enters the objective as explicit Gauss-weighted terms,
the paper's ``(beta/dt)*phi_f`` with the quadrature state eliminated,
registered as a ``cost_group`` that ``drto.build_objective`` (feature 003)
picks up wherever it runs. There is no coupling option: applying this
transform before the mode transform is the whole composition.
"""
import math

from pyomo.common.config import ConfigDict, ConfigValue, PositiveInt
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.core import (
    Block,
    Constraint,
    Expression,
    Param,
    Transformation,
    TransformationFactory,
    Var,
)
from pyomo.core.expr import identify_variables, replace_expressions
from pyomo.dae import ContinuousSet, DerivativeVar

from drto.declarations import _is_var_member, _side_matching, pyomo_cvp_available
from drto.info import info

#: The block the transform adds to the model.
_BLOCK_NAME = "drto_infinite_horizon"

#: The declarations the transform requires.
_REQUIRED = ("horizon", "state", "dynamics", "control", "tracking_stage_cost")


def _gauss_weights(nodes):
    """Quadrature weights for interior ``nodes`` on [0, 1], by moment solve.

    ``pyomo.dae`` stores the collocation nodes but no quadrature weights, so
    they are derived from the nodes the discretization actually used: the
    K-point rule integrating 1, x, ..., x^(K-1) exactly.
    """
    k = len(nodes)
    a = numpy.array([[x**p for x in nodes] for p in range(k)], dtype=float)
    b = numpy.array([1.0 / (p + 1) for p in range(k)])
    return numpy.linalg.solve(a, b)


def _is_derivative(node):
    """Return whether ``node`` is a DerivativeVar member."""
    parent = getattr(node, "parent_component", lambda: None)()
    return isinstance(parent, DerivativeVar)


@TransformationFactory.register(
    "drto.infinite_horizon",
    doc="Append the infinite-horizon terminal segment of Dinh et al. (2025) "
    "to a declared, discretized dynamic model (drto).",
)
class InfiniteHorizonTransformation(Transformation):
    """Append the terminal segment; see the module docstring.

    Options: ``nfe`` and ``ncp`` set the segment mesh (defaults 3 and 5),
    ``beta`` the tail overestimation factor (mutable Param, default 1.2,
    strictly greater than 1), ``gamma`` overrides the mesh rule
    ``tanh(gamma*dt) = tau_11`` (mutable Param, derived by default), and
    ``profile`` sets the segment controls' pyomo-cvp profile (default
    ``'collocation'``, with ``'piecewise_constant'`` the conservative
    alternative).
    """

    CONFIG = ConfigDict("drto.infinite_horizon")
    CONFIG.declare(
        "nfe",
        ConfigValue(
            default=3,
            domain=PositiveInt,
            description="Finite elements on the terminal segment.",
        ),
    )
    CONFIG.declare(
        "ncp",
        ConfigValue(
            default=5,
            domain=PositiveInt,
            description="Gauss-Legendre collocation points per element.",
        ),
    )
    CONFIG.declare(
        "beta",
        ConfigValue(
            default=1.2,
            domain=float,
            description="Tail overestimation safety factor, strictly "
            "greater than 1.",
        ),
    )
    CONFIG.declare(
        "gamma",
        ConfigValue(
            default="rule",
            description="Time-compression rate: 'rule' (the default) derives "
            "it from the mesh rule, the segment's first collocation point "
            "one sampling time past the junction; a number overrides.",
        ),
    )
    CONFIG.declare(
        "profile",
        ConfigValue(
            default="collocation",
            description="pyomo-cvp profile for the segment controls: "
            "'collocation' (default) or 'piecewise_constant'.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)
        if config.beta <= 1:
            raise ValueError(
                f"drto: infinite_horizon requires beta > 1 (the terminal "
                f"cost must overestimate the tail; the margin beta - 1 "
                f"covers the quadrature error). Got beta={config.beta}."
            )
        if not pyomo_cvp_available:
            raise RuntimeError(
                "drto: infinite_horizon requires pyomo-cvp for the segment "
                "control profiles (pip install pyomo-cvp)."
            )
        if not numpy_available:
            raise RuntimeError(
                "drto: infinite_horizon requires numpy for the quadrature " "weights."
            )

        reg = info(model)
        missing = [k for k in _REQUIRED if not reg.has_declaration(k)]
        if "tracking_stage_cost" in missing and reg.has_declaration(
            "economic_stage_cost"
        ):
            raise ValueError(
                "drto: infinite_horizon requires a tracking stage cost. An "
                "economic stage cost alone is rejected: it is nonzero at the "
                "equilibrium, so its tail integral diverges and its "
                "quadrature would be mesh-dependent."
            )
        if missing:
            raise ValueError(
                "drto: infinite_horizon requires "
                + ", ".join(f"drto.{k}" for k in missing)
                + " first."
            )
        if reg.has_transformation("drto.infinite_horizon"):
            raise ValueError(
                "drto: infinite_horizon was already applied to this model."
            )
        drto_obj = model.component("drto_objective")
        if drto_obj is not None and drto_obj.active:
            raise ValueError(
                "drto: the objective is already assembled; apply "
                "drto.infinite_horizon first, then rebuild with "
                "drto.build_objective."
            )

        time = reg.components("horizon")[0]
        if not time.get_discretization_info():
            raise ValueError(
                f"drto: discretize the declared time set '{time.name}' "
                f"(dae.collocation) before applying infinite_horizon."
            )
        samples = reg.declarations("horizon")[0]["samples"]
        dt = samples[1] - samples[0]
        t_end = time.last()

        states = reg.components("state")
        controls = reg.components("control")
        dynamics = reg.components("dynamics")
        (stage_record,) = reg.declarations("tracking_stage_cost")
        stage_con = stage_record["component"]

        if reg.has_transformation("drto.parameterize"):
            raise ValueError(
                "drto: the control profiles are already applied; apply "
                "drto.infinite_horizon before drto.parameterize (it "
                "replicates the controls in their original time indexing)."
            )
        for comp in states + controls:
            if comp.index_set() is not time:
                raise ValueError(
                    f"drto: infinite_horizon supports states and controls "
                    f"indexed by the declared time set only; "
                    f"'{comp.name}' is not."
                )

        b = Block(concrete=True)
        model.add_component(_BLOCK_NAME, b)
        b.tau = ContinuousSet(bounds=(0, 1))
        b.gamma = Param(initialize=1.0, mutable=True)
        b.beta = Param(initialize=config.beta, mutable=True)

        # --- segment copies of the declared states and controls ---
        seg = {}
        for comp in states + controls:
            first = next(iter(comp.values()))
            v = Var(b.tau, bounds=first.bounds, initialize=comp[t_end].value)
            b.add_component(comp.local_name, v)
            seg[comp] = v
        derivs = {}
        for z in states:
            dv = DerivativeVar(seg[z], wrt=b.tau)
            b.add_component(z.local_name + "_dtau", dv)
            derivs[z] = dv

        def substituted(template, t_rep, s):
            """The template expression with model vars at ``t_rep`` swapped
            for segment vars at ``s``."""
            emap = {id(comp[t_rep]): seg[comp][s] for comp in states + controls}
            return replace_expressions(template, emap)

        def check_template(expr, t_rep, where):
            """v1 scope: the replicated expression may reference only
            declared states and controls, at the representative time."""
            for v in identify_variables(expr, include_fixed=True):
                comp = v.parent_component()
                if comp not in states and comp not in controls:
                    raise ValueError(
                        f"drto: infinite_horizon cannot replicate "
                        f"'{where}': it references '{v.name}', which is "
                        f"not a declared state or control."
                    )
                if v.index() != t_rep:
                    raise ValueError(
                        f"drto: infinite_horizon cannot replicate "
                        f"'{where}': it references '{v.name}' away from "
                        f"the constraint's own time point."
                    )

        # --- dilated dynamics at interior collocation points (eq. 25) ---
        rhs_templates = []
        for con in dynamics:
            cd = next(iter(con.values())) if con.is_indexed() else con
            t_rep = cd.index()
            deriv_side, rhs = _side_matching(
                cd, _is_derivative, "infinite_horizon", "a DerivativeVar"
            )
            z = deriv_side.parent_component().get_state_var()
            check_template(rhs, t_rep, cd.name)
            rhs_templates.append((con, z, rhs, t_rep))

            def dyn_rule(blk, s, _z=z, _rhs=rhs, _t=t_rep):
                if s in blk.tau.get_finite_elements():
                    return Constraint.Skip
                return blk.gamma * (1 - s**2) * derivs[_z][s] == substituted(
                    _rhs, _t, s
                )

            b.add_component(con.local_name, Constraint(b.tau, rule=dyn_rule))

        # --- the tracking stage cost, replicated (the tail integrand) ---
        cd = next(iter(stage_con.values())) if stage_con.is_indexed() else stage_con
        t_rep_cost = cd.index()
        cost_side, psi = _side_matching(
            cd, _is_var_member, "infinite_horizon", "the cost variable"
        )
        check_template(psi, t_rep_cost, cd.name)
        cost_var = cost_side.parent_component()

        # --- link the segment to the end of the horizon ---
        for z in states:
            b.add_component(
                z.local_name + "_link", Constraint(expr=seg[z][0] == z[t_end])
            )

        # --- discretize the segment: Gauss-Legendre only, no collocation
        # equation may sit at the singular endpoint tau = 1 ---
        TransformationFactory("dae.collocation").apply_to(
            b, wrt=b.tau, nfe=config.nfe, ncp=config.ncp, scheme="LAGRANGE-LEGENDRE"
        )

        # --- gamma: the mesh rule, or the explicit override ---
        tau11 = sorted(b.tau)[1]
        if config.gamma in (None, "rule"):
            gamma_val = math.atanh(tau11) / dt
        else:
            try:
                gamma_val = float(config.gamma)
            except (TypeError, ValueError):
                raise ValueError(
                    f"drto: gamma must be 'rule' (derive from the mesh "
                    f"rule) or a number; got {config.gamma!r}."
                ) from None
        b.gamma.set_value(gamma_val)

        # --- the tracking stage cost, replicated as named Expressions at the
        # interior collocation points: the tail integrand. Expressions add no
        # variables and no constraints (a replicated cost Var would sit on an
        # active bound as the tail cost vanishes at the equilibrium), and
        # cvp's substitution sweep rewrites them like any constraint ---
        pts = sorted(b.tau)
        fe = b.tau.get_finite_elements()
        interior_pts = [p for p in pts if p not in fe]
        seg_cost = Expression(
            interior_pts, rule=lambda blk, s: substituted(psi, t_rep_cost, s)
        )
        b.add_component(cost_var.local_name, seg_cost)

        # --- hard equilibrium endpoint, 0 = f at tau = 1 (eq. 21c). The
        # state values at tau = 1 come from the Legendre extrapolation; the
        # control values come from the segment profile: the polynomial's
        # endpoint for 'collocation', the last element's constant for
        # 'piecewise_constant' ---
        u_point = 1 if config.profile == "collocation" else fe[-2]
        for con, z, rhs, t_rep in rhs_templates:
            emap = {}
            for comp in states:
                emap[id(comp[t_rep])] = seg[comp][1]
            for comp in controls:
                emap[id(comp[t_rep])] = seg[comp][u_point]
            b.add_component(
                con.local_name + "_equilibrium",
                Constraint(expr=0 == replace_expressions(rhs, emap)),
            )

        # --- segment control profiles: applied now, so raw unparameterized
        # copies are never left on the segment ---
        for u in controls:
            TransformationFactory("cvp.parameterize").apply_to(
                b, var=seg[u], contset=b.tau, profile=config.profile
            )

        # --- the tail cost: explicit Gauss weights, (beta/dt) * phi_f with
        # the quadrature state eliminated. beta and gamma stay symbolic in
        # the weights, so set_value retunes them without a re-apply ---
        interior = [[p for p in pts if lo < p < hi] for lo, hi in zip(fe, fe[1:])]
        h0 = fe[1] - fe[0]
        omega = _gauss_weights([(p - fe[0]) / h0 for p in interior[0]])
        terms = []
        for lo, hi, points in zip(fe, fe[1:], interior):
            h = hi - lo
            for p, w in zip(points, omega):
                weight = b.beta * (h * float(w)) / (b.gamma * dt * (1 - p**2))
                terms.append((seg_cost[p], weight))
        reg.record_declaration("cost_group", b, terms=tuple(terms))

        reg.record_transformation(
            "drto.infinite_horizon",
            segment=f"{config.nfe} elements x {config.ncp} Legendre points",
            beta=config.beta,
            gamma=round(gamma_val, 8),
            profile=config.profile,
            horizon="kept, infinite tail appended",
        )
