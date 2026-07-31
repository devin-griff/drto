# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The IDAES saponification CSTR, declared for drto, with its scaling.

The flowsheet is ``idaes.models.unit_models.CSTR`` with the saponification
property and reaction packages, taken as IDAES wrote it: the true states
declare as member-subset slices of the packed material holdup (the water
member stays algebraic, closed by the property package), the energy holdup
is the fifth state, and the manipulated inputs are the jacket duty and the
feed flow, the flow declared on the inlet Port's time-indexed Reference.
Targets and feedback hooks are Params in the owners' units, filled by the
caller (the setpoint from ``drto.steady_state_simulation``, the start from
the setpoint composition knocked back along the stoichiometry, hot).

The scaling rides along: ``tag_scaling`` writes the units-driven
``scaling_factor`` suffix (every J-valued variable 1e-7, every W-valued
1e-6; the pin slacks stay unscaled), which the drto initializers honor,
and ``scaled_solve`` retags, solves a ``core.scale_model`` clone, and maps
the solution back.

Usage from a notebook in ``examples/``::

    from models.idaes_cstr import F_IN, build, scaled_solve, tag_scaling
    m = build(N=10)
"""
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.models.properties.examples.saponification_reactions import (
    SaponificationReactionParameterBlock,
)
from idaes.models.properties.examples.saponification_thermo import (
    SaponificationParameterBlock,
)
from idaes.models.unit_models import CSTR

import drto

F_IN, C_NAOH, C_EA = 1.0, 100.0, 100.0     # m3/s, mol/m3, mol/m3
T_IN, P_IN, VOLUME = 303.15, 101325.0, 1.0  # K, Pa, m3
T_START = 318.0                             # K, the perturbed start
DC_START = 20.0                             # mol/m3, conversion knocked back


def feed(blk, t):
    # the flow is a manipulated variable: initialized, not fixed
    blk.inlet.flow_vol[t].set_value(F_IN)
    blk.inlet.conc_mol_comp[t, "H2O"].fix(55388.0)
    blk.inlet.conc_mol_comp[t, "NaOH"].fix(C_NAOH)
    blk.inlet.conc_mol_comp[t, "EthylAcetate"].fix(C_EA)
    blk.inlet.conc_mol_comp[t, "SodiumAcetate"].fix(0.0)
    blk.inlet.conc_mol_comp[t, "Ethanol"].fix(0.0)
    blk.inlet.temperature[t].fix(T_IN)
    blk.inlet.pressure[t].fix(P_IN)


def cstr(m):
    m.fs.props = SaponificationParameterBlock()
    m.fs.rxn = SaponificationReactionParameterBlock(property_package=m.fs.props)
    m.fs.cstr = CSTR(property_package=m.fs.props, reaction_package=m.fs.rxn,
                     has_equilibrium_reactions=False, has_heat_of_reaction=True,
                     has_heat_transfer=True)
    return m

def build(N=10, h=1, ncp=3):
    samples = [i * h for i in range(N + 1)]
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=True, time_set=samples, time_units=pyo.units.s)
    cstr(m)

    drto.horizon(m.fs.time)             # before discretization: it takes the grid
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.fs.time, nfe=N, ncp=ncp, scheme="LAGRANGE-RADAU")
    for t in m.fs.time:
        feed(m.fs.cstr, t)
        m.fs.cstr.volume[t].fix(VOLUME)

    cv, t0 = m.fs.cstr.control_volume, m.fs.time.first()
    fin = m.fs.cstr.inlet.flow_vol
    es = list(cv.energy_holdup.index_set().subsets())[1]
    U = pyo.units

    # targets and feedback Params declared in the owners' units, so the
    # registry reads back with the physics visible and the equations that
    # use them stay dimensionally consistent; filled after the steady solve
    m.eng_ss = pyo.Param(es, initialize=0.0, mutable=True, units=U.J)
    m.duty_ss = pyo.Param(initialize=0.0, mutable=True, units=U.W)
    m.flow_ss = pyo.Param(initialize=F_IN, mutable=True, units=U.m**3 / U.s)

    # the initial condition pins the true states: the four reacting species
    # and the energy. The water holdup is algebraic, its concentration fixed
    # by the property package, so it carries no initial condition. The
    # energy holdup is closed-form in T with this constant-property package.
    comps = ["NaOH", "EthylAcetate", "SodiumAcetate", "Ethanol"]
    props = m.fs.props
    e0 = VOLUME * pyo.value(
        props.dens_mol * props.cp_mol * (T_START * U.K - props.temperature_ref))
    m.mat0 = pyo.Param(comps, initialize=0.0, mutable=True, units=U.mol)
    m.eng0 = pyo.Param(es, initialize=e0, mutable=True, units=U.J)

    @m.Constraint(comps)
    def ic_mat(mm, j):
        return cv.material_holdup[t0, "Liq", j] == mm.mat0[j]

    @m.Constraint(es)
    def ic_eng(mm, p):
        return cv.energy_holdup[t0, p] == mm.eng0[p]

    # unit-carrying scales make the cost dimensionless, so it renders with
    # clean units instead of (inc): each term is (quantity/scale)**2
    m.scale_E = pyo.Param(initialize=1.0e6, units=U.J, mutable=True)
    m.scale_Q = pyo.Param(initialize=1.0e6, units=U.W, mutable=True)
    m.scale_F = pyo.Param(initialize=0.1, units=U.m**3 / U.s, mutable=True)
    stages = samples[:-1]
    m.cost = pyo.Var(stages, initialize=0.0)
    m.term = pyo.Var(initialize=0.0)

    @m.Constraint(stages)
    def stage(mm, t):
        return mm.cost[t] == (
            ((cv.energy_holdup[t, "Liq"] - mm.eng_ss["Liq"]) / mm.scale_E) ** 2
            + ((cv.heat[t] - mm.duty_ss) / mm.scale_Q) ** 2
            + ((fin[t] - mm.flow_ss) / mm.scale_F) ** 2)

    tN = m.fs.time.last()

    @m.Constraint()  # the stage cost with the controls removed, at tN
    def terminal(mm):
        return mm.term == (
            (cv.energy_holdup[tN, "Liq"] - mm.eng_ss["Liq"]) / mm.scale_E) ** 2

    drto.state(cv.material_holdup[:, "Liq", "NaOH"],
               cv.material_holdup[:, "Liq", "EthylAcetate"],
               cv.material_holdup[:, "Liq", "SodiumAcetate"],
               cv.material_holdup[:, "Liq", "Ethanol"],
               cv.energy_holdup)
    drto.dynamics(cv.material_balances, cv.enthalpy_balances)
    drto.control(cv.heat, fin, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    drto.initial_condition(m.ic_mat, m.ic_eng)
    m.ss_naoh = drto.steady_state(cv.material_holdup[:, "Liq", "NaOH"],
        pyo.Param(initialize=0.0, mutable=True, units=U.mol))
    m.ss_ea = drto.steady_state(cv.material_holdup[:, "Liq", "EthylAcetate"],
        pyo.Param(initialize=0.0, mutable=True, units=U.mol))
    m.ss_sa = drto.steady_state(cv.material_holdup[:, "Liq", "SodiumAcetate"],
        pyo.Param(initialize=0.0, mutable=True, units=U.mol))
    m.ss_etoh = drto.steady_state(cv.material_holdup[:, "Liq", "Ethanol"],
        pyo.Param(initialize=0.0, mutable=True, units=U.mol))
    drto.steady_state(cv.energy_holdup, m.eng_ss)
    drto.steady_state_control(cv.heat, m.duty_ss)
    drto.steady_state_control(fin, m.flow_ss)
    return m



U = pyo.units
_cache = {}


def scale_factor(u):
    s = str(u)
    if s not in _cache:
        _cache[s] = 1.0
        for target, sf in ((U.J, 1e-7), (U.W, 1e-6), (U.J / U.s, 1e-6)):
            try:
                U.convert_value(1.0, from_units=u, to_units=target)
                _cache[s] = sf
                break
            except Exception:  # incompatible units: unscaled
                pass
    return _cache[s]


def tag_scaling(model):
    """Tag every Var and active row with its units-driven factor."""
    if model.component("scaling_factor") is not None:
        model.del_component("scaling_factor")
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for v in model.component_data_objects(pyo.Var, descend_into=True):
        # the pin slacks enter the objective linearly with weight mu, so
        # scaling them blows the objective's gradients; near zero anyway,
        # they stay unscaled
        if v.parent_component().local_name.endswith(("_pin_up", "_pin_lo")):
            continue
        f = scale_factor(U.get_units(v))
        if f != 1.0:
            model.scaling_factor[v] = f
    for c in model.component_data_objects(pyo.Constraint, active=True,
                                          descend_into=True):
        try:
            f = scale_factor(U.get_units(c.body))
        except Exception:  # a unitless or inconsistent row: unscaled
            f = 1.0
        if f != 1.0:
            model.scaling_factor[c] = f


def scaled_solve(model, tee=False):
    """Retag by units, solve the scaled clone, map the solution back."""
    tag_scaling(model)
    sm = pyo.TransformationFactory("core.scale_model").create_using(model)
    res = pyo.SolverFactory("pounce").solve(sm, tee=tee)
    pyo.TransformationFactory("core.scale_model").propagate_solution(sm, model)
    return res
