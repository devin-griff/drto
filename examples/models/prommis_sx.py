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
holdup, and the settlers' species holdups at every node of the backward
finite-difference cascade, four tanks in series per settler, PrOMMiS's
own mesh (solvents and bisulfate excluded, closed by density and
equilibrium). The organic metal holdups follow the aqueous side
instantaneously through the equilibria, so they are algebra, not
states.

The manipulated inputs are the two feed flows. The declared
disturbances are additive zero-mean noise terms, one in every state's
balance, added into the balance equations, since MSContactor takes no
custom terms. They are zero nominally, so the noise-free model is
untouched. The transfer extents and the
stage flows at the first time point are inert data of the high-index
equilibrium formulation (no rate law determines them there) and are
fixed, the same choice PrOMMiS's own dynamic driver makes.

Usage from a notebook in ``examples/``::

    from models.prommis_sx import build, F_AQ, F_OG
    m = build(N=8, h=1)
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


def build(N=8, h=1, noise=True):
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
        settler_finite_elements=4,
    )

    drto.horizon(m.fs.time)             # before discretization: it takes the grid

    ms = m.fs.ms
    msc = ms.mixer[1].unit.mscontactor
    aq, og = ms.aqueous_settler[1].unit, ms.organic_settler[1].unit
    # every node past the inlet boundary is a tank of the backward
    # finite-difference cascade, one inventory per species per tank
    xs = [x for x in aq.length_domain if x != aq.length_domain.first()]

    # the feed streams. The compositions are held by equations over the time
    # set rather than fixed, since a fixed status covers only the members
    # present when it is set and the caller discretizes after this returns.
    # The organic flow is the manipulated input (initialized, not held), and
    # the aqueous flow is closed by the disturbance equation below
    feed_conc = dict(AQ_FEED)
    feed_conc_og = dict(OG_FEED)
    feed_conc_og["DEHPA"] = 975.8e3 * DOSAGE / 100

    @m.Constraint(m.fs.time, list(feed_conc))
    def aqueous_feed_conc(mm, t, j):
        return ms.aqueous_inlet.conc_mass_comp[t, j] == feed_conc[j]

    @m.Constraint(m.fs.time, list(feed_conc_og))
    def organic_feed_conc(mm, t, j):
        return ms.organic_inlet.conc_mass_comp[t, j] == feed_conc_og[j]

    m.fs.og_feed = pyo.Reference(msc.organic_inlet_state[:].flow_vol)
    m.fs.aq_feed = pyo.Reference(msc.aqueous_inlet_state[:].flow_vol)
    for ref, nominal in ((m.fs.og_feed, F_OG), (m.fs.aq_feed, F_AQ)):
        for t in ref:
            ref[t].set_value(nominal)
            # MV limits: the plant's phase-split algebra is solved
            # reliably inside this envelope; the controller respects it
            ref[t].setlb(45.0)
            ref[t].setub(75.0)
    m.fs.aq_feed[:].unfix()

    # geometry and temperatures, PrOMMiS's dynamic flowsheet values. The
    # volume is indexed by element alone, so fixing it covers every member.
    # The temperatures carry time, so they are equations for the same reason
    # as the feed
    msc.volume[:].fix(VOLUME * U.m**3)

    @m.Constraint(m.fs.time, msc.elements)
    def aqueous_temperature(mm, t, e):
        return msc.aqueous[t, e].temperature == TEMPERATURE * U.K

    @m.Constraint(m.fs.time, msc.elements)
    def organic_temperature(mm, t, e):
        return msc.organic[t, e].temperature == TEMPERATURE * U.K

    for st in (aq, og):
        st.area.fix(AREA)
        st.length.fix(LENGTH)

    # the states: inventories with memory. Mixer aqueous holdups (water
    # carries the phase split, bisulfate rides its equilibrium), the free
    # extractant holdup, and the settler holdups at every cascade node
    # (solvents closed by density, the settlers running full)
    maq = [j for j in m.fs.leach_soln.component_list if j != "HSO4"]
    saq = [j for j in maq if j != "H2O"]
    sog = [j for j in m.fs.prop_o.component_list if j != "Kerosene"]
    drto.state(*(msc.aqueous_material_holdup[:, 1, "liquid", j] for j in maq))
    drto.state(msc.organic_material_holdup[:, 1, "organic", "DEHPA"])
    drto.state(*(aq.material_holdup[:, x, "liquid", j]
                 for x in xs for j in saq))
    drto.state(*(og.material_holdup[:, x, "organic", j]
                 for x in xs for j in sog))
    drto.dynamics(
        msc.aqueous_material_balance, msc.organic_material_balance,
        aq.material_balances, og.material_balances)
    drto.control(m.fs.og_feed, m.fs.aq_feed, profile="piecewise_constant")

    # additive process noise, one zero-mean term per state's balance
    # (mol/hr), added into the balance equations, since MSContactor takes
    # no custom terms. Zero nominally, so the noise-free model is untouched
    def _noise(name, con, index):
        w = pyo.Var(m.fs.time, initialize=0.0, units=U.mol / U.hour)
        m.fs.add_component(name, w)
        for t in m.fs.time:
            cd = con[(t,) + index]
            cd.set_value(cd.expr.args[0] == cd.expr.args[1] + w[t])
        return w

    if noise:
        terms = []
        for j in maq:
            terms.append(_noise(f"w_m_{j}", msc.aqueous_material_balance, (1, j)))
        terms.append(_noise("w_m_DEHPA", msc.organic_material_balance, (1, "DEHPA")))
        for i, x in enumerate(xs, 1):
            for j in saq:
                terms.append(_noise(f"w_sa{i}_{j}", aq.material_balances, (x, j)))
            for j in sog:
                terms.append(_noise(f"w_so{i}_{j}", og.material_balances, (x, j)))
        drto.disturbance(*terms)

    # the initial state, one Param per state, filled by the caller
    t0 = m.fs.time.first()
    m.ic_maq = pyo.Param(maq, initialize=1.0, mutable=True, units=U.mol)
    m.ic_dehpa = pyo.Param(initialize=1.0, mutable=True, units=U.mol)
    m.ic_saq = pyo.Param(xs, saq, initialize=1.0, mutable=True,
                         units=U.mol / U.m)
    m.ic_sog = pyo.Param(xs, sog, initialize=1.0, mutable=True,
                         units=U.mol / U.m)

    @m.Constraint(maq)
    def ic_mixer_aq(mm, j):
        return msc.aqueous_material_holdup[t0, 1, "liquid", j] == mm.ic_maq[j]

    @m.Constraint()
    def ic_mixer_dehpa(mm):
        return msc.organic_material_holdup[t0, 1, "organic", "DEHPA"] == mm.ic_dehpa

    @m.Constraint(xs, saq)
    def ic_settler_aq(mm, x, j):
        return aq.material_holdup[t0, x, "liquid", j] == mm.ic_saq[x, j]

    @m.Constraint(xs, sog)
    def ic_settler_og(mm, x, j):
        return og.material_holdup[t0, x, "organic", j] == mm.ic_sog[x, j]

    drto.initial_condition(m.ic_mixer_aq, m.ic_mixer_dehpa,
                           m.ic_settler_aq, m.ic_settler_og)

    # inert first-instant data of the high-index equilibrium formulation:
    # no rate law determines the extents or flows at t0
    msc.aqueous_inherent_reaction_extent[t0, :, "Ka2"].fix(0.0)
    msc.heterogeneous_reaction_extent[t0, :, :].fix(0.0)
    msc.aqueous[t0, 1].flow_vol.fix(F_AQ)
    msc.organic[t0, 1].flow_vol.fix(F_OG)
    for x in xs:
        aq.inherent_reaction_extent[t0, x, "Ka2"].fix(0.0)
        aq.properties[t0, x].flow_vol.fix(F_AQ)
        og.properties[t0, x].flow_vol.fix(F_OG)

    # steady-state targets, one scalar Param per state (the pairing takes
    # one component per call), filled after the steady solve
    def target(name, unit):
        p = pyo.Param(initialize=1.0, mutable=True, units=unit)
        m.add_component(name, p)
        return p

    ss_maq = {j: target(f"ss_maq_{j}", U.mol) for j in maq}
    m.ss_dehpa = pyo.Param(initialize=1.0, mutable=True, units=U.mol)
    m.ss_fog = pyo.Param(initialize=F_OG, mutable=True, units=U.m**3 / U.hour)
    for j in maq:
        drto.steady_state(msc.aqueous_material_holdup[:, 1, "liquid", j],
                          ss_maq[j])
    drto.steady_state(msc.organic_material_holdup[:, 1, "organic", "DEHPA"],
                      m.ss_dehpa)
    for i, x in enumerate(xs, 1):
        for j in saq:
            drto.steady_state(aq.material_holdup[:, x, "liquid", j],
                              target(f"ss_saq{i}_{j}", U.mol / U.m))
        for j in sog:
            drto.steady_state(og.material_holdup[:, x, "organic", j],
                              target(f"ss_sog{i}_{j}", U.mol / U.m))
    drto.steady_state_control(m.fs.og_feed, m.ss_fog)
    m.ss_faq = pyo.Param(initialize=F_AQ, mutable=True, units=U.m**3 / U.hour)
    drto.steady_state_control(m.fs.aq_feed, m.ss_faq)

    # the tracking cost: hold the rare earth inventories in the mixer's
    # aqueous phase at their targets (what is not extracted), spend the
    # organic flow gently. Unit-carrying scales keep it dimensionless
    ree = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy"]
    bulk = [j for j in maq if j not in ree]
    m.scale_n = pyo.Param(initialize=1e-4, mutable=True, units=U.mol)
    m.scale_F = pyo.Param(initialize=2.0, mutable=True, units=U.m**3 / U.hour)
    # loose scales for the remaining inventories: the cost covers every
    # state, with the rare earth terms still carrying the steering
    m.scale_naq = pyo.Param(initialize=1e5, mutable=True, units=U.mol)
    m.scale_sett = pyo.Param(initialize=1e3, mutable=True, units=U.mol / U.m)
    samples = sorted(m.fs.time.get_finite_elements() if False else [i * h for i in range(N + 1)])
    stages = samples[:-1]
    m.cost = pyo.Var(stages, initialize=0.0)
    m.term = pyo.Var(initialize=0.0)

    def _other_inventories(mm, t):
        return (
            sum(((msc.aqueous_material_holdup[t, 1, "liquid", j]
                  - mm.component(f"ss_maq_{j}")) / mm.scale_naq) ** 2
                for j in bulk)
            + ((msc.organic_material_holdup[t, 1, "organic", "DEHPA"]
                - mm.ss_dehpa) / mm.scale_naq) ** 2
            + sum(((aq.material_holdup[t, x, "liquid", j]
                    - mm.component(f"ss_saq{i}_{j}")) / mm.scale_sett) ** 2
                  for i, x in enumerate(xs, 1) for j in saq)
            + sum(((og.material_holdup[t, x, "organic", j]
                    - mm.component(f"ss_sog{i}_{j}")) / mm.scale_sett) ** 2
                  for i, x in enumerate(xs, 1) for j in sog))

    @m.Constraint(stages)
    def stage(mm, t):
        return mm.cost[t] == (
            sum(((msc.aqueous_material_holdup[t, 1, "liquid", j]
                  - mm.component(f"ss_maq_{j}")) / mm.scale_n) ** 2 for j in ree)
            + ((m.fs.og_feed[t] - mm.ss_fog) / mm.scale_F) ** 2
            + ((m.fs.aq_feed[t] - mm.ss_faq) / mm.scale_F) ** 2
            + _other_inventories(mm, t))

    tN = m.fs.time.last()

    @m.Constraint()  # the stage cost with the controls removed, at tN
    def terminal(mm):
        return mm.term == (
            sum(((msc.aqueous_material_holdup[tN, 1, "liquid", j]
                  - mm.component(f"ss_maq_{j}")) / mm.scale_n) ** 2 for j in ree)
            + _other_inventories(mm, tN))

    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    return m


def noise_sigmas(m, frac=0.3):
    """Per-channel standard deviations for the rare earth noise, mol/hr.

    The nine rare earth channels in the mixer, each drawing with a
    standard deviation of ``frac`` times its steady inventory per hour:
    the grade of the incoming leachate wandering. The solvent, acid,
    impurity, and settler channels stay at zero by default; their
    inventories are tied into the closure algebra tightly enough that
    comparable forcing defeats the plant solves rather than perturbing
    the process.
    """
    ree = ("Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy")
    return {
        f"w_m_{j}": frac * abs(pyo.value(m.component(f"ss_maq_{j}")))
        for j in ree
    }
