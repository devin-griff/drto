# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The PrOMMiS mixer-settler solvent extraction stage, posed index one.

The same stage as ``prommis_sx``, written directly in Pyomo instead of
through the MSContactor: PrOMMiS supplies the chemistry data (feed
compositions, molar masses, the distribution correlations, Ka2, the
reaction stoichiometry), and the equations are the notebook's, with the
balances recombined so the transfer extents never appear. The module builds and declares
the stage; the notebook solves the steady state and fills the targets
from it. The states
are the inventories the reactions cannot touch: each metal's total
across both phases, the hydrogen and sulfur combinations with
bisulfate, the extractant total with the bound metal, and the plain
holdups of the inert species. Every balance carries one derivative and
closes with flow terms alone; the equilibria determine the split of
each total algebraically at every instant. The model is an index-one
DAE, and nothing is undetermined at the first time point.

There are 129 states, the same count as before: 17 in the mixer, 15
per aqueous settler tank, and 13 per organic settler tank, four tanks
per settler. The per-phase holdups are algebra through the linking
rows and the equilibria. The balance rows are skipped at the first
time point, where Radau collocation writes no discretization
equations: the derivative members there constrain nothing, and are
fixed at zero along with the stage outlet flows.

Usage from a notebook in ``examples/``::

    from models.prommis_sx_index_one import build
    m = build(N=8, h=1)
"""
import math

import pyomo.environ as pyo
import pyomo.dae as dae

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
NTANKS = 4                              # tanks per settler cascade
VTANK = AREA * LENGTH / NTANKS          # m3 per settler tank
C_DEHPA_FEED = 975.8e3 * DOSAGE / 100   # mg/L in the organic feed
REE = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy"]
ACID = ["H", "SO4", "HSO4"]


def _chemistry():
    """Read the chemistry data out of PrOMMiS's own packages.

    Returns the element list, the per-element proton stoichiometry, the
    distribution correlation constants, Ka2, and the molar masses of
    both phases (g/mol), all as plain numbers.
    """
    from prommis.properties.sulfuric_acid_leaching_properties import (
        SulfuricAcidLeachingParameters,
    )
    from prommis.solvent_extraction.ree_og_distribution import (
        REESolExOgParameters,
    )
    from prommis.solvent_extraction.solvent_extraction_reaction_package import (
        SolventExtractionReactions,
    )

    holder = pyo.ConcreteModel()
    holder.aq = SulfuricAcidLeachingParameters()
    holder.og = REESolExOgParameters()
    holder.rx = SolventExtractionReactions()

    elements = list(holder.rx.element_list)
    stoich = holder.rx.reaction_stoichiometry
    z = {e: stoich[(f"{e}_mass_transfer", "liquid", "H")] for e in elements}
    corr = {
        e: (
            pyo.value(holder.rx.m0[e]), pyo.value(holder.rx.m1[e]),
            pyo.value(holder.rx.B0[e]), pyo.value(holder.rx.B1[e]),
            pyo.value(holder.rx.K_corr[e]), pyo.value(holder.rx.K1[e]),
        )
        for e in elements
    }
    ka2 = pyo.value(holder.aq.Ka2)
    mw_aq = {j: pyo.value(pyo.units.convert(
        holder.aq.mw[j], pyo.units.g / pyo.units.mol))
        for j in holder.aq.component_list}
    mw_og = {j: pyo.value(pyo.units.convert(
        holder.og.mw[j], pyo.units.g / pyo.units.mol))
        for j in holder.og.component_list}
    return elements, z, corr, ka2, mw_aq, mw_og


def build(N=8, h=1, noise=True):
    """One stage on an ``N``-sample horizon of ``h`` hours, index one."""
    elements, z, corr, ka2, mw_aq, mw_og = _chemistry()

    m = pyo.ConcreteModel()
    m.time = dae.ContinuousSet(initialize=[i * h for i in range(N + 1)])

    AQ = list(mw_aq)                       # 17 aqueous species
    OG = list(mw_og)                       # 14 organic species
    inert = [j for j in AQ if j not in ACID and j not in elements
             and j != "H2O"]               # Cl
    stk = elements + ["Cl"]                # settler tank per-species states
    tanks = list(range(1, NTANKS + 1))
    m.AQ, m.OG = pyo.Set(initialize=AQ), pyo.Set(initialize=OG)
    m.E = pyo.Set(initialize=elements)
    m.TK = pyo.Set(initialize=tanks)

    # ------------------------------------------------------------------
    # variables. Concentrations, holdups, and flows are nonnegative:
    # nothing in the balances stops them going negative, so the bounds
    # carry that physics
    # ------------------------------------------------------------------
    m.og_feed = pyo.Var(m.time, initialize=F_OG, bounds=(45.0, 75.0))
    m.aq_feed = pyo.Var(m.time, initialize=F_AQ, bounds=(45.0, 75.0))
    m.c_aq = pyo.Var(m.time, m.AQ, initialize=lambda b, t, j: AQ_FEED[j],
                     within=pyo.NonNegativeReals)
    m.c_og = pyo.Var(m.time, m.OG, initialize=lambda b, t, j:
                     OG_FEED.get(j, C_DEHPA_FEED),
                     within=pyo.NonNegativeReals)
    m.n_aq = pyo.Var(m.time, m.AQ, initialize=1.0)
    m.n_og = pyo.Var(m.time, m.OG, initialize=1.0)
    m.cm = pyo.Var(m.time, ACID, initialize=1e-3,
                   within=pyo.NonNegativeReals)   # mol/L, acid system
    m.pH = pyo.Var(m.time, initialize=2.0)
    # the dosage is the argument of a logarithm in the distribution
    # correlation, so it carries a bound; at 4.75 percent it sits far
    # from that bound and the solver's starting point leaves it alone
    m.dosage = pyo.Var(m.time, initialize=DOSAGE,
                       within=pyo.PositiveReals)
    m.th_aq = pyo.Var(m.time, initialize=0.5, bounds=(0.05, 0.95))
    m.th_og = pyo.Var(m.time, initialize=0.5, bounds=(0.05, 0.95))
    m.F_aq = pyo.Var(m.time, initialize=F_AQ,
                     within=pyo.NonNegativeReals)
    m.F_og = pyo.Var(m.time, initialize=F_OG,
                     within=pyo.NonNegativeReals)

    # the states the reactions cannot touch: metal totals, the hydrogen,
    # sulfur, and extractant combinations (mol)
    m.nt_metal = pyo.Var(m.time, m.E, initialize=1.0)
    m.nt_h = pyo.Var(m.time, initialize=1.0)
    m.nt_s = pyo.Var(m.time, initialize=1.0)
    m.nt_a = pyo.Var(m.time, initialize=1.0)

    m.dn_aq = dae.DerivativeVar(m.n_aq, wrt=m.time, initialize=0.0)
    m.dn_og = dae.DerivativeVar(m.n_og, wrt=m.time, initialize=0.0)
    m.dnt_metal = dae.DerivativeVar(m.nt_metal, wrt=m.time, initialize=0.0)
    m.dnt_h = dae.DerivativeVar(m.nt_h, wrt=m.time, initialize=0.0)
    m.dnt_s = dae.DerivativeVar(m.nt_s, wrt=m.time, initialize=0.0)
    m.dnt_a = dae.DerivativeVar(m.nt_a, wrt=m.time, initialize=0.0)

    m.sc_aq = pyo.Var(m.time, m.TK, m.AQ, initialize=lambda b, t, i, j:
                      AQ_FEED[j], within=pyo.NonNegativeReals)
    m.sn_aq = pyo.Var(m.time, m.TK, m.AQ, initialize=1.0)
    m.scm = pyo.Var(m.time, m.TK, ACID, initialize=1e-3,
                    within=pyo.NonNegativeReals)
    m.snt_h = pyo.Var(m.time, m.TK, initialize=1.0)
    m.snt_s = pyo.Var(m.time, m.TK, initialize=1.0)
    m.sc_og = pyo.Var(m.time, m.TK, m.OG, initialize=lambda b, t, i, j:
                      OG_FEED.get(j, C_DEHPA_FEED),
                      within=pyo.NonNegativeReals)
    m.sn_og = pyo.Var(m.time, m.TK, m.OG, initialize=1.0)
    m.dsn_aq = dae.DerivativeVar(m.sn_aq, wrt=m.time, initialize=0.0)
    m.dsn_og = dae.DerivativeVar(m.sn_og, wrt=m.time, initialize=0.0)
    m.dsnt_h = dae.DerivativeVar(m.snt_h, wrt=m.time, initialize=0.0)
    m.dsnt_s = dae.DerivativeVar(m.snt_s, wrt=m.time, initialize=0.0)

    # additive process noise, one zero-mean term per state's balance
    # (mol/hr), written into the rows; zero nominally, so the noise-free
    # model is untouched
    m.w_metal = pyo.Var(m.time, m.E, initialize=0.0)
    m.w_h = pyo.Var(m.time, initialize=0.0)
    m.w_s = pyo.Var(m.time, initialize=0.0)
    m.w_a = pyo.Var(m.time, initialize=0.0)
    m.w_h2o = pyo.Var(m.time, initialize=0.0)
    m.w_cl = pyo.Var(m.time, initialize=0.0)
    m.w_sa = pyo.Var(m.time, m.TK, stk, initialize=0.0)
    m.w_sh = pyo.Var(m.time, m.TK, initialize=0.0)
    m.w_ss = pyo.Var(m.time, m.TK, initialize=0.0)
    m.w_so = pyo.Var(m.time, m.TK,
                     [j for j in OG if j != "Kerosene"], initialize=0.0)

    drto.horizon(m.time)
    t0 = m.time.first()
    sog = [j for j in OG if j != "Kerosene"]

    # ------------------------------------------------------------------
    # mixer algebra
    # ------------------------------------------------------------------
    # the linking rows: each state is its combination of phase holdups
    m.link_metal = pyo.Constraint(
        m.time, m.E, rule=lambda b, t, e:
        b.nt_metal[t, e] == b.n_aq[t, e] + b.n_og[t, f"{e}_o"])
    m.link_h = pyo.Constraint(
        m.time, rule=lambda b, t:
        b.nt_h[t] == b.n_aq[t, "H"] + b.n_aq[t, "HSO4"]
        - sum(z[e] * b.n_og[t, f"{e}_o"] for e in elements))
    m.link_s = pyo.Constraint(
        m.time, rule=lambda b, t:
        b.nt_s[t] == b.n_aq[t, "SO4"] + b.n_aq[t, "HSO4"])
    m.link_a = pyo.Constraint(
        m.time, rule=lambda b, t:
        b.nt_a[t] == b.n_og[t, "DEHPA"]
        + sum(z[e] * b.n_og[t, f"{e}_o"] for e in elements))

    # holdups: n [mol] = V [m3] * theta * c [mg/L = g/m3] / mw [g/mol]
    m.holdup_aq = pyo.Constraint(
        m.time, m.AQ, rule=lambda b, t, j:
        b.n_aq[t, j] == VOLUME * b.th_aq[t] * b.c_aq[t, j] / mw_aq[j])
    m.holdup_og = pyo.Constraint(
        m.time, m.OG, rule=lambda b, t, j:
        b.n_og[t, j] == VOLUME * b.th_og[t] * b.c_og[t, j] / mw_og[j])

    # the solvents' bulk concentrations are constants by assumption, held by
    # equations over the time set rather than fixed, since a fixed status
    # covers only the members present when it is set and the caller
    # discretizes after this returns
    @m.Constraint(m.time)
    def bulk_water(b, t):
        return b.c_aq[t, "H2O"] == AQ_FEED["H2O"]

    @m.Constraint(m.time)
    def bulk_kerosene(b, t):
        return b.c_og[t, "Kerosene"] == OG_FEED["Kerosene"]

    # the vessel and the withdrawal (the withdrawal is skipped at the
    # first point, where both outlet flows are fixed data)
    m.split = pyo.Constraint(
        m.time, rule=lambda b, t: b.th_aq[t] + b.th_og[t] == 1)
    m.withdrawal = pyo.Constraint(
        m.time, rule=lambda b, t:
        b.th_aq[t] * b.F_og[t] == b.th_og[t] * b.F_aq[t])

    # acid chemistry, molar concentrations: cm [mol/L] = c [mg/L] / (1000 mw)
    m.molar = pyo.Constraint(
        m.time, ACID, rule=lambda b, t, j:
        1000.0 * mw_aq[j] * b.cm[t, j] == b.c_aq[t, j])
    m.dissociation = pyo.Constraint(
        m.time, rule=lambda b, t:
        b.cm[t, "HSO4"] * ka2 == b.cm[t, "H"] * b.cm[t, "SO4"])
    m.ph_relation = pyo.Constraint(
        m.time, rule=lambda b, t: 10.0 ** (-b.pH[t]) == b.cm[t, "H"])

    # the extractant dosage follows the mixer's local free DEHPA, as
    # PrOMMiS defines it: the mass concentration over pure DEHPA's,
    # as a volume percent
    m.dosage_relation = pyo.Constraint(
        m.time, rule=lambda b, t:
        b.dosage[t] == b.c_og[t, "DEHPA"] / 975.8e3 * 100)

    # the metal transfer: the phases sit on the distribution equilibrium
    # in molar concentrations, c_og,e / mw_og = D_e * c_aq,e / mw_aq.
    # D_e blends PrOMMiS's log-linear pH correlation with a constant
    # fitted coefficient, K_corr switching between them: the rare earths
    # except scandium ride the correlation, and Sc, Al, Ca, and Fe carry
    # constants.
    def _distribution(b, t, e):
        m0, m1, b0, b1, kc, k1 = corr[e]
        logd = (m0 + b.dosage[t] * m1) * b.pH[t]             + b0 + b1 * pyo.log10(b.dosage[t])
        d = 10.0 ** logd * (1 - kc) + kc * k1
        return b.c_og[t, f"{e}_o"] / mw_og[f"{e}_o"]             == d * b.c_aq[t, e] / mw_aq[e]
    m.distribution = pyo.Constraint(m.time, m.E, rule=_distribution)

    # ------------------------------------------------------------------
    # mixer balances: one derivative per row, no extents, skipped at the
    # first point where no discretization equations reach
    # ------------------------------------------------------------------
    def net_aq(b, t, j):     # aqueous feed inflow minus outflow, mol/hr
        return (b.aq_feed[t] * AQ_FEED[j] - b.F_aq[t] * b.c_aq[t, j]) \
            / mw_aq[j]

    def net_og(b, t, j):     # organic feed inflow minus outflow, mol/hr
        feed = C_DEHPA_FEED if j == "DEHPA" else OG_FEED[j]
        return (b.og_feed[t] * feed - b.F_og[t] * b.c_og[t, j]) / mw_og[j]

    m.metal_balance = pyo.Constraint(
        m.time, m.E, rule=lambda b, t, e: pyo.Constraint.Skip if t == t0
        else b.dnt_metal[t, e]
        == net_aq(b, t, e) + net_og(b, t, f"{e}_o") + b.w_metal[t, e])
    m.hydrogen_balance = pyo.Constraint(
        m.time, rule=lambda b, t: pyo.Constraint.Skip if t == t0
        else b.dnt_h[t] == net_aq(b, t, "H") + net_aq(b, t, "HSO4")
        - sum(z[e] * net_og(b, t, f"{e}_o") for e in elements) + b.w_h[t])
    m.sulfur_balance = pyo.Constraint(
        m.time, rule=lambda b, t: pyo.Constraint.Skip if t == t0
        else b.dnt_s[t] == net_aq(b, t, "SO4") + net_aq(b, t, "HSO4") + b.w_s[t])
    m.extractant_balance = pyo.Constraint(
        m.time, rule=lambda b, t: pyo.Constraint.Skip if t == t0
        else b.dnt_a[t] == net_og(b, t, "DEHPA")
        + sum(z[e] * net_og(b, t, f"{e}_o") for e in elements) + b.w_a[t])
    m.chloride_balance = pyo.Constraint(
        m.time, rule=lambda b, t: pyo.Constraint.Skip if t == t0
        else b.dn_aq[t, "Cl"] == net_aq(b, t, "Cl") + b.w_cl[t])

    m.water_balance = pyo.Constraint(
        m.time, rule=lambda b, t: pyo.Constraint.Skip if t == t0
        else b.dn_aq[t, "H2O"] == net_aq(b, t, "H2O") + b.w_h2o[t])

    # the kerosene balance with its derivative eliminated: both solvent
    # concentrations are constants, so the kerosene and water holdups
    # are locked to the phase split and dn_K/dt = -k * dn_H2O/dt
    # exactly. Substituting the water balance leaves an algebraic row
    # that determines the organic outlet flow.
    k_sol = (OG_FEED["Kerosene"] / mw_og["Kerosene"])         / (AQ_FEED["H2O"] / mw_aq["H2O"])
    m.kerosene_closure = pyo.Constraint(
        m.time, rule=lambda b, t:
        net_og(b, t, "Kerosene")
        + k_sol * (net_aq(b, t, "H2O") + b.w_h2o[t]) == 0)

    # ------------------------------------------------------------------
    # settlers: four transport tanks per cascade, fed by the mixer; the
    # aqueous tanks carry the dissociation, so hydrogen and sulfur ride
    # their combinations there too
    # ------------------------------------------------------------------
    def sa_in(b, t, i, j):
        c = b.c_aq[t, j] if i == 1 else b.sc_aq[t, i - 1, j]
        return b.F_aq[t] * (c - b.sc_aq[t, i, j]) / mw_aq[j]

    def so_in(b, t, i, j):
        c = b.c_og[t, j] if i == 1 else b.sc_og[t, i - 1, j]
        return b.F_og[t] * (c - b.sc_og[t, i, j]) / mw_og[j]

    m.slink_h = pyo.Constraint(
        m.time, m.TK, rule=lambda b, t, i:
        b.snt_h[t, i] == b.sn_aq[t, i, "H"] + b.sn_aq[t, i, "HSO4"])
    m.slink_s = pyo.Constraint(
        m.time, m.TK, rule=lambda b, t, i:
        b.snt_s[t, i] == b.sn_aq[t, i, "SO4"] + b.sn_aq[t, i, "HSO4"])

    m.sholdup_aq = pyo.Constraint(
        m.time, m.TK, m.AQ, rule=lambda b, t, i, j:
        b.sn_aq[t, i, j] == VTANK * b.sc_aq[t, i, j] / mw_aq[j])
    m.sholdup_og = pyo.Constraint(
        m.time, m.TK, m.OG, rule=lambda b, t, i, j:
        b.sn_og[t, i, j] == VTANK * b.sc_og[t, i, j] / mw_og[j])
    @m.Constraint(m.time, m.TK)
    def settler_water(b, t, i):
        return b.sc_aq[t, i, "H2O"] == AQ_FEED["H2O"]

    @m.Constraint(m.time, m.TK)
    def settler_kerosene(b, t, i):
        return b.sc_og[t, i, "Kerosene"] == OG_FEED["Kerosene"]

    m.smolar = pyo.Constraint(
        m.time, m.TK, ACID, rule=lambda b, t, i, j:
        1000.0 * mw_aq[j] * b.scm[t, i, j] == b.sc_aq[t, i, j])
    m.sdissociation = pyo.Constraint(
        m.time, m.TK, rule=lambda b, t, i:
        b.scm[t, i, "HSO4"] * ka2 == b.scm[t, i, "H"] * b.scm[t, i, "SO4"])

    m.smetal_balance = pyo.Constraint(
        m.time, m.TK, stk, rule=lambda b, t, i, e:
        pyo.Constraint.Skip if t == t0
        else b.dsn_aq[t, i, e] == sa_in(b, t, i, e) + b.w_sa[t, i, e])
    m.shydrogen_balance = pyo.Constraint(
        m.time, m.TK, rule=lambda b, t, i:
        pyo.Constraint.Skip if t == t0
        else b.dsnt_h[t, i] == sa_in(b, t, i, "H") + sa_in(b, t, i, "HSO4")
        + b.w_sh[t, i])
    m.ssulfur_balance = pyo.Constraint(
        m.time, m.TK, rule=lambda b, t, i:
        pyo.Constraint.Skip if t == t0
        else b.dsnt_s[t, i] == sa_in(b, t, i, "SO4")
        + sa_in(b, t, i, "HSO4") + b.w_ss[t, i])
    # the settler solvent rows carry no content: with the solvent
    # concentrations constant and one flow through each cascade, their
    # balances read zero equals zero, so only the other species get rows
    m.sorganic_balance = pyo.Constraint(
        m.time, m.TK, m.OG, rule=lambda b, t, i, j:
        pyo.Constraint.Skip if (t == t0 or j == "Kerosene")
        else b.dsn_og[t, i, j] == so_in(b, t, i, j) + b.w_so[t, i, j])

    # the first time point: no discretization equation reaches the
    # derivative members there, so they appear in no active constraint
    # and are fixed at zero. The stage outlet flows stay free: the
    # withdrawal relation gives their ratio and the kerosene closure
    # their magnitude, and both are algebraic, so they hold at every
    # time point including this one
    for var in (m.dn_aq, m.dn_og, m.dnt_metal, m.dsn_aq, m.dsn_og):
        for k in var:
            if k[0] == t0:
                var[k].fix(0.0)
    for var in (m.dnt_h, m.dnt_s, m.dnt_a):
        var[t0].fix(0.0)
    for var in (m.dsnt_h, m.dsnt_s):
        var[t0, :].fix(0.0)


    # ------------------------------------------------------------------
    # drto declarations
    # ------------------------------------------------------------------
    drto.state(*(m.nt_metal[:, e] for e in elements))
    drto.state(m.nt_h, m.nt_s, m.nt_a)
    drto.state(m.n_aq[:, "H2O"], m.n_aq[:, "Cl"])
    drto.state(*(m.sn_aq[:, i, j] for i in tanks for j in stk))
    drto.state(*(m.snt_h[:, i] for i in tanks))
    drto.state(*(m.snt_s[:, i] for i in tanks))
    drto.state(*(m.sn_og[:, i, j] for i in tanks for j in sog))
    drto.dynamics(
        m.metal_balance, m.hydrogen_balance, m.sulfur_balance,
        m.extractant_balance, m.chloride_balance, m.water_balance,
        m.smetal_balance, m.shydrogen_balance, m.ssulfur_balance,
        m.sorganic_balance)
    drto.control(m.og_feed, m.aq_feed, profile="piecewise_constant")
    if noise:
        drto.disturbance(m.w_metal, m.w_h, m.w_s, m.w_a, m.w_h2o,
                         m.w_cl, m.w_sa, m.w_sh, m.w_ss, m.w_so)
    else:
        # held at zero by equations for the same reason as the solvents,
        # and undeclared, so no drto transform fixes them later
        for w in (m.w_metal, m.w_h, m.w_s, m.w_a, m.w_h2o, m.w_cl,
                  m.w_sa, m.w_sh, m.w_ss, m.w_so):
            m.add_component(
                w.local_name + "_zero",
                pyo.Constraint(w.index_set(), rule=lambda b, *k, _w=w: _w[k] == 0.0),
            )

    # the initial state, one Param per state, filled by the caller
    m.ic_metal = pyo.Param(elements, initialize=1.0, mutable=True)
    m.ic_h = pyo.Param(initialize=1.0, mutable=True)
    m.ic_s = pyo.Param(initialize=1.0, mutable=True)
    m.ic_a = pyo.Param(initialize=1.0, mutable=True)
    m.ic_h2o = pyo.Param(initialize=1.0, mutable=True)
    m.ic_cl = pyo.Param(initialize=1.0, mutable=True)
    m.ic_sa = pyo.Param(tanks, stk, initialize=1.0, mutable=True)
    m.ic_sh = pyo.Param(tanks, initialize=1.0, mutable=True)
    m.ic_ss = pyo.Param(tanks, initialize=1.0, mutable=True)
    m.ic_so = pyo.Param(tanks, sog, initialize=1.0, mutable=True)

    @m.Constraint(elements)
    def ic_mixer_metal(mm, e):
        return mm.nt_metal[t0, e] == mm.ic_metal[e]

    @m.Constraint()
    def ic_mixer_h(mm):
        return mm.nt_h[t0] == mm.ic_h

    @m.Constraint()
    def ic_mixer_s(mm):
        return mm.nt_s[t0] == mm.ic_s

    @m.Constraint()
    def ic_mixer_a(mm):
        return mm.nt_a[t0] == mm.ic_a

    @m.Constraint()
    def ic_mixer_h2o(mm):
        return mm.n_aq[t0, "H2O"] == mm.ic_h2o

    @m.Constraint()
    def ic_mixer_cl(mm):
        return mm.n_aq[t0, "Cl"] == mm.ic_cl

    @m.Constraint(tanks, stk)
    def ic_settler_aq(mm, i, j):
        return mm.sn_aq[t0, i, j] == mm.ic_sa[i, j]

    @m.Constraint(tanks)
    def ic_settler_h(mm, i):
        return mm.snt_h[t0, i] == mm.ic_sh[i]

    @m.Constraint(tanks)
    def ic_settler_s(mm, i):
        return mm.snt_s[t0, i] == mm.ic_ss[i]

    @m.Constraint(tanks, sog)
    def ic_settler_og(mm, i, j):
        return mm.sn_og[t0, i, j] == mm.ic_so[i, j]

    drto.initial_condition(
        m.ic_mixer_metal, m.ic_mixer_h, m.ic_mixer_s, m.ic_mixer_a,
        m.ic_mixer_h2o, m.ic_mixer_cl, m.ic_settler_aq, m.ic_settler_h,
        m.ic_settler_s, m.ic_settler_og)

    # steady-state targets, one scalar Param per state (the pairing takes
    # one component per call), filled after the steady solve
    def target(name):
        p = pyo.Param(initialize=1.0, mutable=True)
        m.add_component(name, p)
        return p

    for e in elements:
        drto.steady_state(m.nt_metal[:, e], target(f"ss_metal_{e}"))
    drto.steady_state(m.nt_h, target("ss_h"))
    drto.steady_state(m.nt_s, target("ss_s"))
    drto.steady_state(m.nt_a, target("ss_a"))
    drto.steady_state(m.n_aq[:, "H2O"], target("ss_h2o"))
    drto.steady_state(m.n_aq[:, "Cl"], target("ss_cl"))
    for i in tanks:
        for j in stk:
            drto.steady_state(m.sn_aq[:, i, j], target(f"ss_sa{i}_{j}"))
        drto.steady_state(m.snt_h[:, i], target(f"ss_sh{i}"))
        drto.steady_state(m.snt_s[:, i], target(f"ss_ss{i}"))
        for j in sog:
            drto.steady_state(m.sn_og[:, i, j], target(f"ss_so{i}_{j}"))
    m.ss_fog = pyo.Param(initialize=F_OG, mutable=True)
    m.ss_faq = pyo.Param(initialize=F_AQ, mutable=True)
    drto.steady_state_control(m.og_feed, m.ss_fog)
    drto.steady_state_control(m.aq_feed, m.ss_faq)

    # the tracking cost: hold the rare earth totals at their targets,
    # spend the feed flows gently, with loose scales covering every
    # other state
    m.scale_n = pyo.Param(initialize=1e-4, mutable=True)
    m.scale_F = pyo.Param(initialize=2.0, mutable=True)
    m.scale_naq = pyo.Param(initialize=1e5, mutable=True)
    m.scale_sett = pyo.Param(initialize=1e3, mutable=True)
    samples = [i * h for i in range(N + 1)]
    stages = samples[:-1]
    m.cost = pyo.Var(stages, initialize=0.0)
    m.term = pyo.Var(initialize=0.0)
    impurities = [e for e in elements if e not in REE]

    def _other_inventories(mm, t):
        return (
            sum(((mm.nt_metal[t, e] - mm.component(f"ss_metal_{e}"))
                 / mm.scale_naq) ** 2 for e in impurities)
            + ((mm.nt_h[t] - mm.ss_h) / mm.scale_naq) ** 2
            + ((mm.nt_s[t] - mm.ss_s) / mm.scale_naq) ** 2
            + ((mm.nt_a[t] - mm.ss_a) / mm.scale_naq) ** 2
            + ((mm.n_aq[t, "H2O"] - mm.ss_h2o) / mm.scale_naq) ** 2
            + ((mm.n_aq[t, "Cl"] - mm.ss_cl) / mm.scale_naq) ** 2
            + sum(((mm.sn_aq[t, i, j] - mm.component(f"ss_sa{i}_{j}"))
                   / mm.scale_sett) ** 2 for i in tanks for j in stk)
            + sum(((mm.snt_h[t, i] - mm.component(f"ss_sh{i}"))
                   / mm.scale_sett) ** 2 for i in tanks)
            + sum(((mm.snt_s[t, i] - mm.component(f"ss_ss{i}"))
                   / mm.scale_sett) ** 2 for i in tanks)
            + sum(((mm.sn_og[t, i, j] - mm.component(f"ss_so{i}_{j}"))
                   / mm.scale_sett) ** 2 for i in tanks for j in sog))

    @m.Constraint(stages)
    def stage(mm, t):
        return mm.cost[t] == (
            sum(((mm.nt_metal[t, e] - mm.component(f"ss_metal_{e}"))
                 / mm.scale_n) ** 2 for e in REE)
            + ((mm.og_feed[t] - mm.ss_fog) / mm.scale_F) ** 2
            + ((mm.aq_feed[t] - mm.ss_faq) / mm.scale_F) ** 2
            + _other_inventories(mm, t))

    tN = m.time.last()

    @m.Constraint()  # the stage cost with the controls removed, at tN
    def terminal(mm):
        return mm.term == (
            sum(((mm.nt_metal[tN, e] - mm.component(f"ss_metal_{e}"))
                 / mm.scale_n) ** 2 for e in REE)
            + _other_inventories(mm, tN))

    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    return m


def noise_sigmas(m, frac=0.3):
    """Per-channel standard deviations for the rare earth noise, mol/hr.

    The nine rare earth channels in the mixer, each drawing with a
    standard deviation of ``frac`` of the element's steady feed rate.
    """
    mw_aq = _chemistry()[4]
    return {m.w_metal[:, e].name: frac * F_AQ * AQ_FEED[e] / mw_aq[e]
            for e in REE}
