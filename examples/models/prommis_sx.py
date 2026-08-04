# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The PrOMMiS mixer-settler solvent extraction stage, declared for drto.

One extraction stage of ``prommis.solvent_extraction``'s
``MixerSettlerExtraction``, taken as PrOMMiS wrote it: a two-phase mixer
where the DEHPA complexation equilibria move the rare earths into the
organic phase, and two single-phase settlers carrying the effluents out.
The declared states are the inventories with memory: the mixer's aqueous
holdups (all but bisulfate, which rides its dissociation equilibrium;
the water member carries the phase split), the mixer's free extractant
holdup, and the settlers' species holdups at the outlet node (solvents
and bisulfate excluded, closed by density and equilibrium). The organic
metal holdups follow the aqueous side instantaneously through the
equilibria, so they are algebra, not states.

The manipulated input is the organic feed flow; the aqueous feed carries
a declared zero-mean flow disturbance. The transfer extents and the
stage flows at the first time point are inert data of the high-index
equilibrium formulation (no rate law determines them there) and are
fixed, the same choice PrOMMiS's own dynamic driver makes.

Usage from a notebook in ``examples/``::

    from models.prommis_sx import build, F_AQ, F_OG
    m = build(N=8, h=1, ncp=2)
"""
import math

import pyomo.environ as pyo
from pyomo.environ import units as U
from idaes.core import FlowDirection, FlowsheetBlock
from prommis.properties.sulfuric_acid_leaching_properties import (
    SulfuricAcidLeachingParameters,
)
from prommis.solvent_extraction.mixer_settler_extraction import (
    MixerSettlerExtraction,
)
from prommis.solvent_extraction.ree_og_distribution import REESolExOgParameters
from prommis.solvent_extraction.solvent_extraction_reaction_package import (
    SolventExtractionReactions,
)

import drto

#: Feed composition of the aqueous leachate (mg/L), PrOMMiS's coal-refuse
#: leachate, and of the organic solvent (kerosene carrying DEHPA).
AQ_FEED = {
    "H2O": 1e6, "H": 10.75, "SO4": 100, "HSO4": 1e4, "Al": 422.375,
    "Ca": 109.542, "Cl": 1e-7, "Fe": 688.266, "Sc": 0.032, "Y": 0.124,
    "La": 0.986, "Ce": 2.277, "Pr": 0.303, "Nd": 0.946, "Sm": 0.097,
    "Gd": 0.2584, "Dy": 0.047,
}
OG_FEED = {
    "Kerosene": 820e3, "Al_o": 1.267e-5, "Ca_o": 2.684e-5, "Fe_o": 2.873e-6,
    "Sc_o": 1.734, "Y_o": 2.179e-5, "La_o": 0.000105, "Ce_o": 0.00031,
    "Pr_o": 3.711e-5, "Nd_o": 0.000165, "Sm_o": 1.701e-5, "Gd_o": 3.357e-5,
    "Dy_o": 8.008e-6,
}
F_AQ, F_OG, DOSAGE = 62.01, 62.01, 5    # m3/hr, m3/hr, percent DEHPA
VOLUME, AREA, LENGTH = 0.4, 1.0, 0.4    # m3 mixer, m2 and m per settler
TEMPERATURE = 305.15                    # K


def build(N=8, h=1, ncp=2):
    """One declared stage on an ``N``-sample horizon of ``h`` hours."""
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(
        dynamic=True, time_set=[i * h for i in range(N + 1)],
        time_units=U.hour,
    )
    m.fs.prop_o = REESolExOgParameters()
    m.fs.leach_soln = SulfuricAcidLeachingParameters()
    m.fs.reaxn = SolventExtractionReactions()
    m.fs.reaxn.extractant_dosage = DOSAGE
    m.fs.ms = MixerSettlerExtraction(
        number_of_stages=1,
        aqueous_stream={
            "property_package": m.fs.leach_soln,
            "flow_direction": FlowDirection.forward,
            "has_energy_balance": False,
            "has_pressure_balance": False,
        },
        organic_stream={
            "property_package": m.fs.prop_o,
            "flow_direction": FlowDirection.backward,
            "has_energy_balance": False,
            "has_pressure_balance": False,
        },
        heterogeneous_reaction_package=m.fs.reaxn,
        has_holdup=True,
        settler_transformation_method="dae.finite_difference",
        settler_transformation_scheme="BACKWARD",
        settler_finite_elements=1,
    )

    drto.horizon(m.fs.time)             # before discretization: it takes the grid
    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.fs.time, nfe=N, ncp=ncp, scheme="LAGRANGE-RADAU")

    ms = m.fs.ms
    msc = ms.mixer[1].unit.mscontactor
    aq, og = ms.aqueous_settler[1].unit, ms.organic_settler[1].unit
    xn = aq.length_domain.last()

    # the feed streams: compositions fixed; the organic flow is the
    # manipulated input (initialized, not fixed); the aqueous flow is
    # closed by the disturbance equation below
    for j, v in AQ_FEED.items():
        ms.aqueous_inlet.conc_mass_comp[:, j].fix(v)
    for j, v in OG_FEED.items():
        ms.organic_inlet.conc_mass_comp[:, j].fix(v)
    ms.organic_inlet.conc_mass_comp[:, "DEHPA"].fix(975.8e3 * DOSAGE / 100)
    m.fs.og_feed = pyo.Reference(msc.organic_inlet_state[:].flow_vol)
    m.fs.aq_feed = pyo.Reference(msc.aqueous_inlet_state[:].flow_vol)
    for t in m.fs.og_feed:
        m.fs.og_feed[t].set_value(F_OG)
        # MV limits: the plant's phase-split algebra is solved reliably
        # inside this envelope; the controller respects it as bounds
        m.fs.og_feed[t].setlb(35.0)
        m.fs.og_feed[t].setub(75.0)

    # additive zero-mean disturbance in the aqueous feed flow
    m.fs.w_feed = pyo.Var(m.fs.time, initialize=0.0, units=U.m**3 / U.hour)
    m.fs.aq_feed[:].unfix()

    @m.fs.Constraint(m.fs.time)
    def feed_flow(fs, t):
        return fs.aq_feed[t] == F_AQ * U.m**3 / U.hour + fs.w_feed[t]

    # geometry and temperatures, PrOMMiS's dynamic flowsheet values
    msc.volume[:].fix(VOLUME * U.m**3)
    msc.aqueous[:, :].temperature.fix(TEMPERATURE * U.K)
    msc.organic[:, :].temperature.fix(TEMPERATURE * U.K)
    for st in (aq, og):
        st.area.fix(AREA)
        st.length.fix(LENGTH)

    # the states: inventories with memory. Mixer aqueous holdups (water
    # carries the phase split, bisulfate rides its equilibrium), the free
    # extractant holdup, and the settler holdups at the outlet node
    # (solvents closed by density, the settlers running full)
    maq = [j for j in m.fs.leach_soln.component_list if j != "HSO4"]
    saq = [j for j in maq if j != "H2O"]
    sog = [j for j in m.fs.prop_o.component_list if j != "Kerosene"]
    drto.state(*(msc.aqueous_material_holdup[:, 1, "liquid", j] for j in maq))
    drto.state(msc.organic_material_holdup[:, 1, "organic", "DEHPA"])
    drto.state(*(aq.material_holdup[:, xn, "liquid", j] for j in saq))
    drto.state(*(og.material_holdup[:, xn, "organic", j] for j in sog))
    drto.dynamics(
        msc.aqueous_material_balance, msc.organic_material_balance,
        aq.material_balances, og.material_balances)
    drto.control(m.fs.og_feed, profile="piecewise_constant")
    drto.disturbance(m.fs.w_feed)

    # feedback hooks: one Param per state, filled by the caller
    t0 = m.fs.time.first()
    m.ic_maq = pyo.Param(maq, initialize=1.0, mutable=True, units=U.mol)
    m.ic_dehpa = pyo.Param(initialize=1.0, mutable=True, units=U.mol)
    m.ic_saq = pyo.Param(saq, initialize=1.0, mutable=True, units=U.mol / U.m)
    m.ic_sog = pyo.Param(sog, initialize=1.0, mutable=True, units=U.mol / U.m)

    @m.Constraint(maq)
    def ic_mixer_aq(mm, j):
        return msc.aqueous_material_holdup[t0, 1, "liquid", j] == mm.ic_maq[j]

    @m.Constraint()
    def ic_mixer_dehpa(mm):
        return msc.organic_material_holdup[t0, 1, "organic", "DEHPA"] == mm.ic_dehpa

    @m.Constraint(saq)
    def ic_settler_aq(mm, j):
        return aq.material_holdup[t0, xn, "liquid", j] == mm.ic_saq[j]

    @m.Constraint(sog)
    def ic_settler_og(mm, j):
        return og.material_holdup[t0, xn, "organic", j] == mm.ic_sog[j]

    drto.initial_condition(m.ic_mixer_aq, m.ic_mixer_dehpa,
                           m.ic_settler_aq, m.ic_settler_og)

    # inert first-instant data of the high-index equilibrium formulation:
    # no rate law determines the extents or flows at t0
    msc.aqueous_inherent_reaction_extent[t0, :, "Ka2"].fix(0.0)
    msc.heterogeneous_reaction_extent[t0, :, :].fix(0.0)
    aq.inherent_reaction_extent[t0, xn, "Ka2"].fix(0.0)
    msc.aqueous[t0, 1].flow_vol.fix(F_AQ)
    msc.organic[t0, 1].flow_vol.fix(F_OG)
    aq.properties[t0, xn].flow_vol.fix(F_AQ)
    og.properties[t0, xn].flow_vol.fix(F_OG)

    # steady-state targets, one scalar Param per state (the pairing takes
    # one component per call), filled after the steady solve
    def target(name, unit):
        p = pyo.Param(initialize=1.0, mutable=True, units=unit)
        m.add_component(name, p)
        return p

    ss_maq = {j: target(f"ss_maq_{j}", U.mol) for j in maq}
    ss_saq = {j: target(f"ss_saq_{j}", U.mol / U.m) for j in saq}
    ss_sog = {j: target(f"ss_sog_{j}", U.mol / U.m) for j in sog}
    m.ss_dehpa = pyo.Param(initialize=1.0, mutable=True, units=U.mol)
    m.ss_fog = pyo.Param(initialize=F_OG, mutable=True, units=U.m**3 / U.hour)
    for j in maq:
        drto.steady_state(msc.aqueous_material_holdup[:, 1, "liquid", j],
                          ss_maq[j])
    drto.steady_state(msc.organic_material_holdup[:, 1, "organic", "DEHPA"],
                      m.ss_dehpa)
    for j in saq:
        drto.steady_state(aq.material_holdup[:, xn, "liquid", j], ss_saq[j])
    for j in sog:
        drto.steady_state(og.material_holdup[:, xn, "organic", j], ss_sog[j])
    drto.steady_state_control(m.fs.og_feed, m.ss_fog)

    # the tracking cost: hold the rare earth inventories in the mixer's
    # aqueous phase at their targets (what is not extracted), spend the
    # organic flow gently. Unit-carrying scales keep it dimensionless
    ree = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy"]
    m.scale_n = pyo.Param(initialize=1e-4, mutable=True, units=U.mol)
    m.scale_F = pyo.Param(initialize=5.0, mutable=True, units=U.m**3 / U.hour)
    samples = sorted(m.fs.time.get_finite_elements() if False else [i * h for i in range(N + 1)])
    stages = samples[:-1]
    m.cost = pyo.Var(stages, initialize=0.0)
    m.term = pyo.Var(initialize=0.0)

    @m.Constraint(stages)
    def stage(mm, t):
        return mm.cost[t] == (
            sum(((msc.aqueous_material_holdup[t, 1, "liquid", j]
                  - mm.component(f"ss_maq_{j}")) / mm.scale_n) ** 2 for j in ree)
            + ((m.fs.og_feed[t] - mm.ss_fog) / mm.scale_F) ** 2)

    tN = m.fs.time.last()

    @m.Constraint()  # the stage cost with the control removed, at tN
    def terminal(mm):
        return mm.term == sum(
            ((msc.aqueous_material_holdup[tN, 1, "liquid", j]
              - mm.component(f"ss_maq_{j}")) / mm.scale_n) ** 2 for j in ree)

    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    return m


def steady_targets(m, tee=False):
    """Fill the targets and hooks from PrOMMiS's own steady flowsheet.

    The drto steady branch does not yet reduce spatially distributed
    Blocks (gh #54), so the setpoint comes from
    ``mixer_settler_ex_flowsheet_steady``: build it at the same dosage
    and stage count, solve it, and read the solution back. The holdup
    values are evaluated through this model's own holdup rows at the
    steady concentrations and phase split, so every unit conversion is
    the model's. The first-instant extents are refixed at their steady
    values. Returns the solved steady flowsheet.
    """
    from prommis.solvent_extraction.mixer_settler_ex_flowsheet_steady import (
        initialize_steady_model, model_buildup_and_set_inputs, solve_model,
    )

    sm = model_buildup_and_set_inputs(DOSAGE, 1)
    initialize_steady_model(sm)
    solve_model(sm)

    their = sm.fs.mixer_settler_ex
    tmsc = their.mixer[1].unit.mscontactor
    taq = their.aqueous_settler[1].unit
    tog = their.organic_settler[1].unit
    ts = sm.fs.time.first()
    txn = taq.length_domain.last()

    ms = m.fs.ms
    msc = ms.mixer[1].unit.mscontactor
    aq, og = ms.aqueous_settler[1].unit, ms.organic_settler[1].unit
    t0 = m.fs.time.first()
    xn = aq.length_domain.last()
    maq = [j for j in m.fs.leach_soln.component_list if j != "HSO4"]
    saq = [j for j in maq if j != "H2O"]
    sog = [j for j in m.fs.prop_o.component_list if j != "Kerosene"]

    # the steady point, written into this model's t0 algebra so the
    # holdup rows can be evaluated in the model's own units
    for j in m.fs.leach_soln.component_list:
        msc.aqueous[t0, 1].conc_mass_comp[j].set_value(
            pyo.value(tmsc.aqueous[ts, 1].conc_mass_comp[j]))
        aq.properties[t0, xn].conc_mass_comp[j].set_value(
            pyo.value(taq.properties[ts, txn].conc_mass_comp[j]))
    for j in m.fs.prop_o.component_list:
        msc.organic[t0, 1].conc_mass_comp[j].set_value(
            pyo.value(tmsc.organic[ts, 1].conc_mass_comp[j]))
        og.properties[t0, xn].conc_mass_comp[j].set_value(
            pyo.value(tog.properties[ts, txn].conc_mass_comp[j]))
    for p in ("aqueous", "organic"):
        msc.volume_frac_stream[t0, 1, p].set_value(
            pyo.value(tmsc.volume_frac_stream[ts, 1, p]))

    def rhs(cd):
        return pyo.value(cd.expr.args[1])

    for j in maq:
        v = rhs(msc.aqueous_material_holdup_constraint[t0, 1, "liquid", j])
        m.ic_maq[j] = v
        m.component(f"ss_maq_{j}").set_value(v)
    v = rhs(msc.organic_material_holdup_constraint[t0, 1, "organic", "DEHPA"])
    m.ic_dehpa = v
    m.ss_dehpa = v
    for j in saq:
        v = rhs(aq.material_holdup_calculation[t0, xn, "liquid", j])
        m.ic_saq[j] = v
        m.component(f"ss_saq_{j}").set_value(v)
    for j in sog:
        v = rhs(og.material_holdup_calculation[t0, xn, "organic", j])
        m.ic_sog[j] = v
        m.component(f"ss_sog_{j}").set_value(v)

    # the extents: refixed at the steady values at the first instant,
    # and set there everywhere else as the initial guess (they are the
    # one algebra the pointwise cold-start solves cannot reach, since
    # the equilibrium formulation determines them only jointly with the
    # dynamics)
    ka2 = pyo.value(tmsc.aqueous_inherent_reaction_extent[ts, 1, "Ka2"])
    het = {r: pyo.value(tmsc.heterogeneous_reaction_extent[ts, 1, r])
           for r in m.fs.reaxn.reaction_idx}
    ska2 = pyo.value(taq.inherent_reaction_extent[ts, txn, "Ka2"])
    for t in m.fs.time:
        msc.aqueous_inherent_reaction_extent[t, 1, "Ka2"].set_value(ka2)
        for r, v in het.items():
            msc.heterogeneous_reaction_extent[t, 1, r].set_value(v)
        aq.inherent_reaction_extent[t, xn, "Ka2"].set_value(ska2)
    msc.aqueous_inherent_reaction_extent[t0, 1, "Ka2"].fix()
    msc.heterogeneous_reaction_extent[t0, 1, :].fix()
    aq.inherent_reaction_extent[t0, xn, "Ka2"].fix()
    return sm


def tag_scaling(model):
    """Value-based scaling factors from the model's current magnitudes.

    Concentrations in this model span 1e-8 to 1e6, so the factors come
    from the values themselves: after ``steady_targets`` and a cold
    start have filled the model at its operating point, every variable
    whose magnitude sits outside [1e-2, 1e2] is tagged with the nearest
    power of ten bringing it to order one. The drto initializers and the
    closed loop honor the suffix on their internal solves.
    """
    if model.component("scaling_factor") is not None:
        model.del_component("scaling_factor")
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for v in model.component_data_objects(pyo.Var, descend_into=True):
        val = v.value
        if val is None or val == 0.0:
            continue
        mag = abs(val)
        if 1e-2 <= mag <= 1e2:
            continue
        model.scaling_factor[v] = 10.0 ** (-round(math.log10(mag)))
