"""
e3_water.py -- E3 on water (phase-d-build-plan.md section 3), on the synthetic polarisable truth PW (sseft.water, the
rehearsal), on the published CCSD(T) potential q-AQUA v1 (--truth qaqua: E3-Q, notes/e3q-preregistration.md; the
library is built by external/qaqua/build.sh) or on q-AQUA-pol, its CCSD(T) corrections on TTM3-F's polarisable
long range (--truth qaqua_pol: E3-QP, notes/e3qp-preregistration.md; external/qaqua/build_pol.sh).

    python scripts/e3_water.py --part dimer   --rungs M_inf,M_G,M_A,M_1,M_1mu,M_1p --seeds 4
    python scripts/e3_water.py --part clusters --rungs M_1p,M_S0p,M_S2p,M_1,M_S0 --seeds 2
    python scripts/run_experiment.py --config configs/water_qaqua.yaml --out_dir results/e3q      # E3-Q, both parts
    python scripts/run_experiment.py --config configs/water_qaqua_pol.yaml --out_dir results/e3qp # E3-QP, both parts

(a) The dimer at the fixed orientation (p* = 3): the Phase 1c ladder on the O-O window [1.35 R_e, 1.8 x that],
    the mean's relative error, spread, local exponent and plateau slope on [30, 300] A, and the learned monomer
    dipole (sum q_i r_i + mu_i over an isolated monomer) against the truth's (PW 1.855 D; q-AQUA's LTP2011 dipole).
(b) Clusters n = 3..5 for training, n = 2 and 6 held out: interaction energies and the non-additive energy
    E - sum of the model's own dimer energies, for v1 (sources only) and v1.5 (band-field inputs at order 0 / 2,
    near sources excluded from the band field at r_s = l_1).  On q-AQUA, every trained model is also read on the raw
    CCSD(T) 2-, 3- and 4-body sets of the q-AQUA repository (--heldout; frames with near-rigid monomers), against
    the labels and with the PES's own error on the same frames as the reference.  With --probe_train / --probe_test
    (any truth) the part also trains on and reads probe configurations: a relaxed trimer and one monomer at a
    distance d from it -- the probe's interaction with the trimer and its non-additive part (minus its three dimer
    energies: the many-body induction of a distant molecule), against the truth's.

Rungs (the dagger rungs pin q to the monomer's point charges):
    M_inf, M_G, M_A   as in the dimer ladder
    M_1               Coulomb band, q learned
    M_1mu             Coulomb band, q and atomic dipoles learned
    M_1p              Coulomb band, q pinned                                   (v1 dagger)
    M_S0p, M_S2p      M_1p + band-field inputs at order 0 / 2                  (v1.5 dagger)
    M_S0, M_S2        M_1 + band-field inputs (q learned)
    M_16p, M_16S0p    the daggers with the dispersion band on the truth's alpha and C6 (E3-R, E3-QP)
    M_1pm, M_16pm     q-AQUA-pol: the truth's permanent charges exactly (TTM3-F's H charges and M-site, a virtual site)
    M_16S0pm, M_16S2pm   + the read-out (coefficients from the species), first order in the polarisabilities, order 0 / 2
    M_16C0pm, M_16C2pm   + the read-out with its induced dipoles' mutual fields to all orders (self-consistent)
    M_16C0p           M_16C0pm's response on the dipole-surface charges (the control for the sources)
"""
from experiment_config import (CLUSTER_DEFAULTS, CLUSTER_PROBE_DEFAULTS, DIMER_DEFAULTS, add_training_arguments, parse_selection,
                               training_kwargs, training_seeds)
import argparse, time, json, sys, os, pickle, math, itertools
import numpy as np
import jax
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sseft
from sseft import water as W, model as M, train as T, measure as ME, data as D, kernels as kn, qaqua as QA
from sseft.units import bohr_to_ang, ang_to_bohr

R_E = W.PW_TRUTH.r_e                           # A, the PW truth's dimer minimum at phi = 0 (the far and ion parts)
WINDOW = (1.35 * R_E, 1.35 * R_E * 1.8)        # [3.86, 6.95] A
HELDOUT_SETS = tuple(QA.RAW_SETS)              # 2b, 3b, 4b
HELDOUT_OO_BINS = ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, math.inf))    # largest O-O (A): q-AQUA's 3-b switch is 7-9 A

RUNGS = {
    "M_inf": M.RUNGS["M_inf"], "M_G": M.RUNGS["M_G"], "M_A": M.RUNGS["M_A"],
    "M_1": M.RUNGS["M_1"],
    "M_1mu": M.with_rung(M.RUNGS["M_1"], dipoles=True),
    "M_1p": M.with_rung(M.RUNGS["M_1"], pin_q=True),
    "M_S0p": M.with_rung(M.RUNGS["M_1"], pin_q=True, band_fields=0),
    "M_S2p": M.with_rung(M.RUNGS["M_1"], pin_q=True, band_fields=2),
    "M_S0": M.with_rung(M.RUNGS["M_1"], band_fields=0),
    "M_S2": M.with_rung(M.RUNGS["M_1"], band_fields=2),
    "M_S0mu": M.with_rung(M.RUNGS["M_1"], dipoles=True, band_fields=0),       # learned q and mu (environment-dependent sources) + band fields
    "M_S2mu": M.with_rung(M.RUNGS["M_1"], dipoles=True, band_fields=2),
    "M_S0pmu": M.with_rung(M.RUNGS["M_1"], pin_q=True, dipoles=True, band_fields=0),   # pinned q, learned mu + band fields
    "M_1pemb": M.with_rung(M.RUNGS["M_1"], pin_q=True, dipoles=True, pin_mu=True),     # the embedded dagger: q and mu pinned to the
    # monomer's multipoles *in its own cluster* (the truth's self-consistent induced dipoles; a probe outside the cluster carries
    # the isolated monomer's)
    "M_S0pemb": M.with_rung(M.RUNGS["M_1"], pin_q=True, dipoles=True, pin_mu=True, band_fields=0),
    "M_16pemb": M.with_rung(M.RUNGS["M_1"], pin_q=True, dipoles=True, pin_mu=True, dispersion=True, pin_alpha=True),   # + the
    # dispersion band on a pinned one-oscillator alpha_O(i w) with the truth's static alpha and C6 (zero on H)
    # the band-field response test (E3-R): PERMANENT sources (the isolated monomer's charges, the ion's charge) pinned, the
    # dispersion band on the pinned alpha, and the read-out carrying the induction -- first order in the band field of the
    # permanent sources: the pairwise induction and the first-order many-body cross terms -- against the dagger without it
    "M_16p": M.with_rung(M.RUNGS["M_1"], pin_q=True, dispersion=True, pin_alpha=True),
    "M_16S0p": M.with_rung(M.RUNGS["M_1"], pin_q=True, dispersion=True, pin_alpha=True, band_fields=0),
    "M_16S2p": M.with_rung(M.RUNGS["M_1"], pin_q=True, dispersion=True, pin_alpha=True, band_fields=2),
    # E3-QP2 (q-AQUA-pol): the truth's permanent charges exactly -- TTM3-F's H charges and its M-site as a virtual site
    # (Rung.msites) -- the exact dagger, and the read-out on it at first order (S) or with the induced dipoles' mutual
    # fields to all orders (C: self-consistent, the alpha^2 T terms and beyond), at order 0 or with the orders 1-2
    # (the read-out's coefficients from the species alone, matched on the far field: Rung.bf_species, training.readout.stage)
    "M_1pm": M.with_rung(M.RUNGS["M_1"], pin_q=True, msites=True),
    "M_16pm": M.with_rung(M.RUNGS["M_1"], pin_q=True, msites=True, dispersion=True, pin_alpha=True),
    "M_16S0pm": M.with_rung(M.RUNGS["M_1"], pin_q=True, msites=True, dispersion=True, pin_alpha=True, band_fields=0, bf_species=True),
    "M_16S2pm": M.with_rung(M.RUNGS["M_1"], pin_q=True, msites=True, dispersion=True, pin_alpha=True, band_fields=2, bf_species=True),
    "M_16C0pm": M.with_rung(M.RUNGS["M_1"], pin_q=True, msites=True, dispersion=True, pin_alpha=True, band_fields=0, bf_species=True,
                            bf_mutual=-1),
    "M_16C2pm": M.with_rung(M.RUNGS["M_1"], pin_q=True, msites=True, dispersion=True, pin_alpha=True, band_fields=2, bf_species=True,
                            bf_mutual=-1),
    # the control: the same self-consistent response on the dipole-surface charges q3 (E3-QP's sources, their quadrupole off)
    "M_16C0p": M.with_rung(M.RUNGS["M_1"], pin_q=True, dispersion=True, pin_alpha=True, band_fields=0, bf_species=True, bf_mutual=-1),
}
C6_OO_TRUTH = 45.4


def pinned_alpha_oscillator(numbers, K=8, alpha=None, c6=None):
    """one-oscillator alpha(i w) on the oxygens through the truth's static polarisability and O-O C6 (London), zero on H
    (PW's unless given: q-AQUA-pol's are TTM3-F's, its M-site polarisability and O-O C6)."""
    alpha = W.ALPHA_O if alpha is None else alpha
    c6 = C6_OO_TRUTH if c6 is None else c6
    omega, w = kn.casimir_polder_grid(K)
    w0 = 4 * c6 / (3 * alpha ** 2)
    aO = alpha / (1 + (np.array(omega) / w0) ** 2)
    return np.array([aO if int(z) == 8 else np.zeros(K) for z in numbers])


def truth_dispersion(truth):
    """(alpha, C6) of the truth's polarisable model for the pinned dispersion band: q-AQUA-pol's, else PW's."""
    return (getattr(truth, "alpha", None), getattr(truth, "c6", None)) if truth is not None else (None, None)


def pin_dispersion(structs, truth=None):
    """pin the one-oscillator alpha(i w) on every structure (the dispersion band of the permanent-source daggers)."""
    alpha, c6 = truth_dispersion(truth)
    for s in structs:
        s.pinned = {**s.pinned, "alpha": pinned_alpha_oscillator(s.numbers, alpha=alpha, c6=c6)}
    return structs


def needs(a, key):
    return any(key in name for name in a.rungs.split(","))


def pw_only(name):
    """rungs whose pinned sources come from the PW truth's polarisable model: the embedded induced dipoles (pin_mu) and
    the one-oscillator alpha_O(i w) with PW's static alpha and C6 (pin_alpha)."""
    return RUNGS[name].pin_mu or RUNGS[name].pin_alpha


TRUTH_ROOTS = {"qaqua": "external/qaqua/build", "qaqua_pol": "external/qaqua/build_pol"}
TRUTH_LIBRARIES = {"qaqua": ("libqaqua.so", "external/qaqua/build.sh"),
                   "qaqua_pol": ("libqaqua_pol.so", "external/qaqua/build_pol.sh")}


def truth_root(a):
    """the build directory of the truth: --qaqua_root (YAML truth.root), else the truth's default build directory."""
    return getattr(a, "qaqua_root", None) or TRUTH_ROOTS[a.truth]


def make_truth(a):
    """the truth of the run: PW (default), q-AQUA v1 or q-AQUA-pol from its build (one instance per process)."""
    name = getattr(a, "truth", "pw")
    if name == "qaqua":
        return W.qaqua_truth(truth_root(a))
    if name == "qaqua_pol":
        return W.qaqua_pol_truth(truth_root(a))
    return W.PW_TRUTH


def heldout_selection(a):
    sel = [s for s in getattr(a, "heldout", "none").split(",") if s and s != "none"]
    return sel


def validate_water_args(a):
    """checks that need e3_water's own knowledge (rungs, truths, parts); raised as parser errors before anything runs."""
    names = a.rungs.split(",")
    sets = heldout_selection(a)
    if a.truth in ("qaqua", "qaqua_pol"):
        label = {"qaqua": "q-AQUA", "qaqua_pol": "q-AQUA-pol"}[a.truth]
        if a.part not in ("dimer", "clusters"):
            why = (", which q-AQUA does not have)" if a.truth == "qaqua" else
                   "; on q-AQUA-pol the far field is read in the clusters part, data.probe_train / probe_test)")
            raise ValueError(f"the {label} truth runs the dimer and clusters parts (the far and ion parts read the PW truth's "
                             f"polarisable decomposition{why}")
        if a.truth == "qaqua":
            bad = [n for n in names if pw_only(n)]
            if bad:
                raise ValueError(f"rungs {', '.join(bad)} pin sources of the PW truth's polarisable model (embedded dipoles or its "
                                 "alpha); q-AQUA has no polarisable sites and no dispersion beyond 7.8 A -- use the charge-only rungs")
        else:
            bad = [n for n in names if RUNGS[n].pin_mu]
            if bad:
                raise ValueError(f"rungs {', '.join(bad)} pin embedded induced dipoles, which are not set up for q-AQUA-pol "
                                 "(TTM3-F's sit on the M-site); its permanent-source daggers M_16p, M_16S0p, M_16S2p pin TTM3-F's "
                                 "alpha and C6")
        if a.truth == "qaqua":
            bad = [n for n in names if RUNGS[n].msites]
            if bad:
                raise ValueError(f"rungs {', '.join(bad)} pin TTM3-F's site charges, which only q-AQUA-pol has")
        root = os.path.abspath(os.path.expanduser(truth_root(a)))
        lib, script = TRUTH_LIBRARIES[a.truth]
        if not os.path.isfile(os.path.join(root, lib)):
            raise ValueError(f"no {label} build at {root}: run sh {script} (or set truth.root / --qaqua_root)")
        unknown = set(sets) - set(HELDOUT_SETS)
        if unknown:
            raise ValueError(f"unknown held-out sets {', '.join(sorted(unknown))}; choose from {', '.join(HELDOUT_SETS)}")
        # the held-out sets are read in the clusters part (with the cluster-trained models); a shared truth mapping
        # leaves them inert in the dimer part
        missing = [s for s in sets if not os.path.isfile(os.path.join(root, "q-AQUA", QA.RAW_SETS[s][0]))]
        if missing:
            raise ValueError(f"raw sets {', '.join(missing)} not found under {os.path.join(root, 'q-AQUA')} (re-run the build script)")
        if a.heldout_strict_oh_tol > a.heldout_oh_tol or a.heldout_strict_angle_tol > a.heldout_angle_tol:
            raise ValueError("the strict near-rigid tolerance must not exceed the rigid tolerance (it selects a subset)")
    elif sets:
        raise ValueError("the held-out raw CCSD(T) sets need a q-AQUA truth (--truth qaqua or qaqua_pol)")
    else:
        bad = [n for n in names if RUNGS[n].msites]
        if bad:
            raise ValueError(f"rungs {', '.join(bad)} pin TTM3-F's site charges, which only q-AQUA-pol has (--truth qaqua_pol)")
    if getattr(a, "readout_far_weight", None) is not None and not (a.part == "clusters" and getattr(a, "probe_train", 0) > 0):
        raise ValueError("training.readout.far_weight weights the training probes in the read-out's solve: the clusters part "
                         "with data.probe_train > 0")
    if getattr(a, "far_labels", "total") == "interaction" and a.part != "clusters":
        raise ValueError("training.loss.far_labels: interaction labels the clusters part's training probes")
    if getattr(a, "readout_ridge_selection", "fixed") == "groups" and a.part != "clusters":
        raise ValueError("training.readout.ridge_selection: groups holds out clusters and probe cores: the clusters part")
    if getattr(a, "readout_stage", "after_adam") == "far_first":
        if not (a.part == "clusters" and getattr(a, "probe_train", 0) > 0 and getattr(a, "far_labels", "total") == "interaction"):
            raise ValueError("training.readout.stage: far_first matches the read-out on the training probes' interaction "
                             "energies: the clusters part with data.probe_train > 0 and training.loss.far_labels: interaction")
        bad = [n for n in names if RUNGS[n].band_fields >= 0 and not RUNGS[n].bf_species]
        if bad:
            raise ValueError(f"training.readout.stage: far_first needs a read-out independent of E_0 (the species read-out); "
                             f"rungs {', '.join(bad)} read E_0's features")
        if not getattr(a, "lbfgs_freeze_readout", False):
            raise ValueError("training.readout.stage: far_first keeps the matched read-out: set training.lbfgs.freeze_readout")


def restore_fragment_charges(structs):
    """fragment charges on structures loaded from caches written before v0.0.8, when Structure.frag_charges did not
    exist (an unpickled instance falls back to the class default None, and pad_batch rejects a charged structure
    without them): in an ion structure the ion is the last atom and carries the formal charge, the waters are neutral."""
    for s in structs:
        if getattr(s, "frag", None) is None or getattr(s, "frag_charges", None) is not None or s.total_charge == 0:
            continue
        if int(np.sum(s.numbers == W.ION["Z"])) != 1 or int(s.numbers[-1]) != W.ION["Z"]:
            raise ValueError("cannot restore the fragment charges of a cached charged structure that is not water + one ion")
        s.frag_charges = np.concatenate([np.zeros(int(np.max(s.frag))), [s.total_charge]])
    return structs


def min_intermolecular(st):
    """the smallest distance (A) between atoms of different fragments: a dimer node identifies the tail only when it
    exceeds r_c (every intermolecular pair beyond E_0's cutoff)."""
    pos = bohr_to_ang(st.positions); d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    return float(d[st.frag[:, None] != st.frag[None, :]].min())


def save_params(a, name, seed, params):
    path = os.path.splitext(a.out)[0] + f"_{name}_seed{seed}.params.pkl"
    with open(path, "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)


def pin_sources(a, *sets, n_core=None, truth=None):
    """apply the pins the requested rungs need to the given structure lists: embedded sources for "emb" rungs, the
    dispersion alpha for "16" rungs (the truth's alpha and C6: PW's unless the truth has its own), the truth's site
    model for the site-model rungs (q-AQUA-pol's TTM3-F charges)."""
    site_model = any(RUNGS[name].msites for name in a.rungs.split(","))
    for structs in sets:
        if needs(a, "emb"): pin_embedded(structs, n_core=n_core)
        elif needs(a, "16"): pin_dispersion(structs, truth)
        if site_model: W.pin_site_model(structs, truth)
    return sets


def pin_for(rung, structs, truth):
    """the pins one rung needs on structures built for a reading (sub-clusters, dimers, held-out frames)."""
    if rung.pin_mu:
        pin_embedded(structs)
    elif rung.pin_alpha:
        pin_dispersion(structs, truth)
    if rung.msites:
        W.pin_site_model(structs, truth)
    return structs


def pin_embedded(structs, n_core=None, self_energy=True):
    """pin each structure's atomic dipoles to the truth's induced dipoles of the molecules embedded in their own cluster:
    the whole structure, or, for probe configurations, the core alone (atoms before 3 n_core) with zeros on the probe."""
    for s in structs:
        pos = bohr_to_ang(s.positions); n_ion = int(np.sum(s.numbers == W.ION["Z"])); n = (len(pos) - n_ion) // 3
        if n_ion > 0:
            # the ion is the probe: the molecules carry their induced dipoles of the core alone, the ion nothing
            mu = np.zeros((len(pos), 3)); mu[:3 * n] = W.pw_components(pos[:3 * n], n)["mu_ind"] if n > 1 else 0.0
        elif n_core is not None and n > n_core:
            mu = np.zeros((3 * n, 3)); mu[:3 * n_core] = W.pw_components(pos[:3 * n_core], n_core)["mu_ind"]
        else:
            mu = W.pw_components(pos, n)["mu_ind"] if n > 1 else np.zeros((3, 3))
        # the polarisation work of the pinned induced dipoles (the truth's alpha on O): with it the band reproduces the
        # self-consistent induction energy exactly, without it the pinned dipoles double count it inside the cluster
        eself = 0.5 * np.sum(mu ** 2, axis=1) / W.ALPHA_O if self_energy else np.zeros(len(mu))
        s.pinned = {**s.pinned, "mu": mu, "eself": eself, "alpha": pinned_alpha_oscillator(s.numbers)}
    return structs


def configure(rung, a):
    """the rung with the architecture settings of the run: E_0's cutoff and width, M_A's envelope, the charge source
    and the neutrality shift (per fragment, or over the structure as before v0.0.8), the band edge l_1 and the
    near-source switch of the band-field input (r_s = rs_over_l1 x l_1; 0 switches it off)."""
    kw = dict(lA=ang_to_bohr(a.lA), e0={**rung.e0, "r_cut": ang_to_bohr(a.rcut), "F": a.F, "n_rbf_readout": 8})
    charge_source = getattr(a, "charges", None)
    if rung.charges and charge_source is not None:
        kw["pin_q"] = charge_source == "pinned"
    kw["frag_neutral"] = getattr(a, "charge_neutrality", "fragment") == "fragment"
    if a.l1 > 0:
        kw["l1"] = ang_to_bohr(a.l1)                                                   # the band edge (default 1.5 A)
    if rung.band_fields >= 0:
        kw["bf_rs"] = kw.get("l1", rung.l1) * a.rs_over_l1
        if getattr(a, "response_edge", 0) > 0:
            kw["bf_l"] = ang_to_bohr(a.response_edge)                                  # the response's own band edge
    return M.with_rung(rung, **kw)


def model_record(name, a):
    """the architecture of a rung as run, for the results json: the sources (pinned / learned / off), the neutrality
    shift, the band-field order, the band edge, the near-source switch, E_0's cutoff and width, M_A's envelope."""
    rung = configure(RUNGS[name], a)
    source = lambda on, pinned: ("pinned" if pinned else "learned") if on else "off"
    ang = lambda x: round(float(bohr_to_ang(x)), 12)                     # the unit round trip, without its last bit
    return {"charges": source(rung.charges, rung.pin_q),
            "charge_neutrality": "fragment" if rung.frag_neutral else "structure",
            "dipoles": source(rung.dipoles, rung.pin_mu),
            "polarisabilities": source(rung.dispersion, rung.pin_alpha),
            "band_fields": int(rung.band_fields) if rung.band_fields >= 0 else None,
            "band_edge_angstrom": ang(rung.l1),
            "near_source_switch_angstrom": ang(rung.bf_rs) if rung.band_fields >= 0 else None,
            **({"response_edge_angstrom": ang(rung.bf_l)} if rung.band_fields >= 0 and rung.bf_l > 0 else {}),
            **({"site_model": True} if rung.msites else {}),
            **({"readout_coefficients": "species"} if rung.bf_species else {}),
            **({"mutual_response": int(rung.bf_mutual)} if rung.bf_mutual else {}),
            "r_cut_angstrom": ang(rung.e0["r_cut"]), "features": int(rung.e0["F"]),
            "analytic_envelope_angstrom": ang(rung.lA), "periodic": bool(rung.periodic)}


def monomer_dipole(params, rung, truth=None):
    """|sum_i q_i r_i + sum_i mu_i| of the isolated monomer, in Debye."""
    aux = T.predict_aux(params, rung, [W.monomer_structure(truth)])
    pos = W.monomer_positions()                                  # A, O at the origin
    q = np.array(aux["q"][0, :3]); mu = np.array(aux["mu"][0, :3]) if "mu" in aux else np.zeros((3, 3))
    d = (q[:, None] * pos).sum(0) + bohr_to_ang(mu).sum(0)      # e A
    return float(np.linalg.norm(d) * W.DEBYE_PER_E_ANG), q


def truth_monomer(truth):
    """the truth's isolated rigid monomer: its dipole (Debye) and charges (O, H, H)."""
    pos = W.monomer_positions()
    q = np.asarray(truth.molecule_charges(pos))
    return float(np.linalg.norm((q[:, None] * pos).sum(0)) * W.DEBYE_PER_E_ANG), q


def dimer_data_settings(a, truth):
    """the dimer design (the data mapping of the dimer part), with the window from the truth's own R_e unless given."""
    s = {key: getattr(a, key, default) for key, default in DIMER_DEFAULTS.items()}
    r_e = s["re_angstrom"] if s["re_angstrom"] is not None else truth.r_e
    lo = s["window_lower_over_re"] * r_e
    s.update(truth=truth.name, r_e_angstrom=float(r_e), window_angstrom=[lo, lo * s["window_ratio"]])
    return s


def _save(a, results):
    old = {}
    if os.path.exists(a.out):
        try: old = json.load(open(a.out))
        except Exception: old = {}
    if old and "dataset" in results and old.get("dataset") != results["dataset"]:
        raise ValueError("output contains results for different or unknown dataset settings; use a new --out path")
    if hasattr(a, "rungs"):
        for name in a.rungs.split(","):
            if name in results:
                results[name]["model"] = model_record(name, a)
                results[name]["training"] = {"seeds": training_seeds(a), "far_weight": a.w_far if hasattr(a, "w_far") else 1.0,
                                             **training_kwargs(a)}
                for key in ("far_labels", "readout_far_weight", "readout_stage"):
                    if getattr(a, key, None) not in (None, "total", "after_adam"):
                        results[name]["training"][key] = getattr(a, key)
    old.update(results)
    json.dump(old, open(a.out, "w"), indent=1)


def run_dimer(a, truth=None):
    truth = truth or make_truth(a)
    s = dimer_data_settings(a, truth)
    lo, hi = s["window_angstrom"]; n_win = s["window_nodes"]; phi = s["phi_degrees"]
    R_train = bohr_to_ang(D.window_design(lo, hi, n_win))
    R_eval = bohr_to_ang(D.eval_grid(hi, s["eval_max_angstrom"], s["eval_points"]))
    rng = np.random.default_rng(s["train_seed"])
    train = (W.dimer_scan(R_train, sigma=s["sigma"], rng=rng, truth=truth, phi_deg=phi)
             + [W.monomer_structure(truth)] * s["monomer_references"])
    test = W.dimer_scan(R_eval, truth=truth, phi_deg=phi)
    pin_sources(a, train, test, truth=truth)
    f_true = np.array([st.energy for st in test])
    identifying = [min_intermolecular(st) > a.rcut for st in train[:n_win]]
    mu_truth, q_truth = truth_monomer(truth)
    dd = W.dipole_dipole_coefficient(truth, phi)
    print(f"water dimer ({truth.name} truth, phi = {phi:g}): R_e = {s['r_e_angstrom']:.4f} A, window [{lo:.2f}, {hi:.2f}] A, "
          f"{n_win} nodes, sigma = {s['sigma']:g}; {sum(identifying)} identifying nodes (every intermolecular pair beyond "
          f"r_c = {a.rcut:g} A); |f*| at R+ = {abs(test[0].energy):.2e} E_h; E R^3 at {R_eval[-1]:.0f} A = "
          f"{f_true[-1] * ang_to_bohr(R_eval[-1]) ** 3:+.4f} (point dipoles {dd:+.4f}); truth's monomer dipole {mu_truth:.4f} D", flush=True)
    results = {"dataset": s, "setup": {"truth": truth.describe(), "identifying_nodes": int(sum(identifying)),
                                       "truth_dipole_debye": mu_truth, "truth_charges": q_truth.tolist(),
                                       "dipole_dipole_coefficient": dd}}
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a)
        preds, rmses, dipoles, charges_O = [], [], [], []; t0 = time.time()
        fitted_options = []
        for seed in training_seeds(a):
            params, info = T.fit(rung, train, seed=seed, **training_kwargs(a))
            fitted_options.append(info.get("training_options", {"seed": seed, **training_kwargs(a)}))
            if truth.name != "pw":
                save_params(a, name, seed, params)
            E_tr, _ = T.predict(params, rung, train[:n_win])
            rmse = float(np.sqrt(np.mean((E_tr - np.array([st.energy for st in train[:n_win]])) ** 2)))
            E_te, _ = T.predict(params, rung, test); preds.append(E_te); rmses.append(rmse)
            msg = f"  {name} seed {seed}: window rmse {rmse:.2e}"
            if rung.charges and not rung.pin_q:
                mu, q = monomer_dipole(params, rung, truth); dipoles.append(mu); charges_O.append(float(q[0]))
                msg += f"; monomer dipole {mu:.3f} D (truth {mu_truth:.3f}), q_O = {q[0]:+.3f} (truth {q_truth[0]:+.3f})"
            print(msg + f"; {info['time']:.0f} s", flush=True)
        P = np.array(preds); rm = np.array(rmses); conv = rm <= 2.0 * rm.min()
        if (~conv).any():
            print(f"  {name}: {int((~conv).sum())} unconverged seeds excluded (window rmse {rm[~conv]} vs best {rm.min():.2e})", flush=True)
        if conv.sum() >= 2: P = P[conv]
        mean, Wsp = ME.ensemble_stats(P)
        w_floor = 1e-14 * max(abs(st.energy) for st in train[:n_win])
        Wsp = np.where(Wsp > w_floor, Wsp, np.nan)
        q_exp = ME.local_exponent(R_eval, Wsp)
        rel_err = np.abs(mean - f_true) / np.abs(f_true); relW = Wsp / np.abs(f_true)
        print(f"  {name}: {'R (A)':>7s} {'|f*|':>9s} {'mean':>10s} {'rel err':>9s} {'spread W':>9s} {'W/|f*|':>9s} {'q':>7s}")
        rows = {}
        for Rv in (8, 10, 12, 15, 20, 30, 50, 100, 300):
            i = int(np.argmin(np.abs(R_eval - Rv)))
            print(f"  {'':6s} {R_eval[i]:7.1f} {abs(f_true[i]):9.2e} {mean[i]:+10.2e} {rel_err[i]:9.2e} {Wsp[i]:9.2e} {relW[i]:9.2e} {q_exp[i]:7.2f}")
            rows[Rv] = dict(mean=float(mean[i]), rel_err=float(rel_err[i]), W=float(np.nan_to_num(Wsp[i])), relW=float(np.nan_to_num(relW[i])), q=float(np.nan_to_num(q_exp[i])))
        slope = ME.plateau_slope(R_eval, relW, 30, 300)
        dip_c = [d for d, c in zip(dipoles, conv) if c] if (dipoles and conv.sum() >= 2) else dipoles
        print(f"  {name}: slope of log(W/|f*|) vs log R on [30, 300] A = {slope:+.2f}  (branch 2: 0; branch 1: p* - p_min = 3 - 1 = 2 for a monopole leak; branch 3: -> -inf)"
              + (f";  monomer dipole {np.mean(dip_c):.3f} +- {np.std(dip_c, ddof=1) if len(dip_c) > 1 else 0:.3f} D over the converged seeds" if dipoles else "") + f"  [{time.time() - t0:.0f} s]", flush=True)
        results[name] = dict(rows=rows, slope=slope, window_rmse=rm.tolist(), converged=conv.tolist(), dipoles=dipoles,
                             charges_O=charges_O, grid=dict(R_A=R_eval.tolist(), f_true=f_true.tolist(), mean=mean.tolist(), W=np.nan_to_num(Wsp).tolist(), preds=P.tolist()))
        results[name]["training_runs"] = fitted_options
        _save(a, results)
    return results


def run_far(a):
    """the far-field induction test: training = the dimer window (its nodes beyond r_c teach the response to the band
    field) + the clusters; test = a probe monomer at 6-10 A from a trimer.  The probe's interaction energy minus its
    electrostatics with the pinned charges is the induction; v1 cannot represent it beyond r_c, v1.5 can."""
    t0 = time.time()
    dists = (6.0, 7.0, 8.0, 10.0)
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as fh: base, far_train, E_core_tr, test, E_core = pickle.load(fh)
        print(f"data from {a.cache}", flush=True)
    else:
        R_train = bohr_to_ang(D.window_design(*WINDOW, 40)); rng = np.random.default_rng(0)
        base = (W.dimer_scan(R_train, sigma=a.sigma, rng=rng) + W.cluster_dataset((3, 4, 5), a.n_train, seed=0, sigma=a.sigma, mc_steps=a.mc_steps)
                + [W.monomer_structure()] * 4)
        far_train, E_core_tr = [], np.zeros(0)
        if a.train_far > 0:
            # far-field configurations in the training set (other cores, other seeds): the only data in which anything
            # beyond the isolated-monomer sources is resolvable above the fit's floor
            far_train, E_core_tr = W.probe_configurations(3, dists, a.train_far, seed=400, mc_steps=a.mc_steps, n_cores=a.train_far_cores)
        test, E_core = W.probe_configurations(3, dists, a.n_test, seed=300, mc_steps=a.mc_steps, n_cores=a.test_cores)
        if a.cache:
            with open(a.cache, "wb") as fh: pickle.dump((base, far_train, E_core_tr, test, E_core), fh)
    for s in far_train: s.weight = a.w_far                                             # --w_far: weight of the far-field training structures
    pin_sources(a, base); pin_sources(a, far_train, test, n_core=3)
    train = base + far_train
    clusters = [s for s in base if s.n_atoms > 6]

    def probe_parts(structs, E_c):
        """the probe's interaction with the core and its induction (total minus the point-charge electrostatics, repulsion
        and dispersion of the truth's components)."""
        yy = np.array([s.energy for s in structs]) - E_c
        es = []
        for s in structs:
            pos = bohr_to_ang(s.positions); c_all = W.pw_components(pos, 4); c_core = W.pw_components(pos[:9], 3)
            es.append((c_all["es"] - c_core["es"]) + (c_all["rep"] - c_core["rep"]) + (c_all["disp"] - c_core["disp"]))
        return yy, yy - np.array(es)
    y, ind = probe_parts(test, E_core)
    y_tr, ind_tr = probe_parts(far_train, E_core_tr) if far_train else (np.zeros(0), np.zeros(0))
    print(f"far-field induction test: {len(test)} probe configurations at d = {dists} A from a trimer; probe interaction rms "
          f"{np.sqrt(np.mean(y ** 2)) * 627.5:.4f} kcal/mol, of which induction rms {np.sqrt(np.mean(ind ** 2)) * 627.5:.4f} "
          f"({np.sqrt(np.mean(ind ** 2)) / np.sqrt(np.mean(y ** 2)) * 100:.1f} %)  [{time.time() - t0:.0f} s]", flush=True)
    if far_train:
        print(f"  {len(far_train)} far-field training configurations from {a.train_far_cores} cores (weight {a.w_far:g}); their induction rms "
              f"{np.sqrt(np.mean(ind_tr ** 2)) * 627.5:.4f} kcal/mol ({np.sqrt(np.mean(ind_tr ** 2)) / np.sqrt(np.mean(y_tr ** 2)) * 100:.1f} % of the interaction)", flush=True)
    results = {"d": [s.info["d"] for s in test], "probe_interaction": y.tolist(), "induction": ind.tolist()}

    def probe_rmse(params, rung, structs, yy):
        Et, _ = T.predict(params, rung, structs)
        cores = [W.water_structure(bohr_to_ang(s.positions)[:9]) for s in structs]
        if rung.pin_mu: pin_embedded(cores)
        elif rung.pin_alpha: pin_dispersion(cores)
        Ec, _ = T.predict(params, rung, cores)
        return (Et - Ec) - yy
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a); t1 = time.time(); errs = []; train_errs = []
        fitted_options = []
        for seed in training_seeds(a):
            params, info = T.fit(rung, train, seed=seed, **training_kwargs(a))
            fitted_options.append(info.get("training_options", {"seed": seed, **training_kwargs(a)}))
            res = probe_rmse(params, rung, test, y)
            r = np.sqrt(np.mean(res ** 2)); errs.append(r)
            print(f"  {name} seed {seed}: probe-interaction rmse {r * 627.5:.4f} kcal/mol = {r / np.sqrt(np.mean(ind ** 2)) * 100:.0f} % of the induction "
                  f"(per distance: " + ", ".join(f"{d:.0f} A {np.sqrt(np.mean(res[np.array(results['d']) == d] ** 2)) / np.sqrt(np.mean(ind[np.array(results['d']) == d] ** 2)) * 100:.0f} %" for d in dists) + f"); {info['time']:.0f} s", flush=True)
            # the fit's own floor: residuals on the training clusters and on the far-field training configurations (the
            # latter as a share of *their* induction -- above 100 % the far-field signal was never fitted, and the test
            # measures the floor, not generalisation)
            Ecl, _ = T.predict(params, rung, clusters); r_cl = np.sqrt(np.mean((Ecl - np.array([s.energy for s in clusters])) ** 2))
            msg = f"      training residuals: clusters {r_cl * 627.5:.4f} kcal/mol"
            tr = {"clusters": float(r_cl)}
            if far_train:
                res_tr = probe_rmse(params, rung, far_train, y_tr); r_tr = np.sqrt(np.mean(res_tr ** 2)); tr["far"] = float(r_tr)
                msg += f"; far-field configurations {r_tr * 627.5:.4f} kcal/mol = {r_tr / np.sqrt(np.mean(ind_tr ** 2)) * 100:.0f} % of their induction"
            print(msg, flush=True); train_errs.append(tr)
        results[name] = dict(rmse=errs, train=train_errs, w_far=a.w_far)
        results[name]["training_runs"] = fitted_options
        _save(a, results)
    return results


def cluster_data_settings(a, truth=None):
    settings = {key: getattr(a, key, default) for key, default in CLUSTER_DEFAULTS.items()}
    for key in ("train_sizes", "test_sizes"):
        settings[key] = [int(v) for v in settings[key].split(',')]
    if truth is not None and truth.name != "pw":
        settings["truth"] = truth.name              # PW caches and outputs keep their pre-v0.0.9 metadata
    if probes_requested(a):                         # likewise the settings of runs without probe configurations
        settings.update({key: getattr(a, key, default) for key, default in CLUSTER_PROBE_DEFAULTS.items()})
        settings["probe_distances_angstrom"] = [float(v) for v in str(settings["probe_distances_angstrom"]).split(',')]
        sizes = [int(v) for v in str(settings.pop("probe_train_core_sizes")).split(',')]
        if sizes != [3]:                            # and trimer cores alone keep the settings of before
            settings["probe_train_core_sizes"] = sizes
    return settings


def probes_requested(a):
    return bool(getattr(a, "probe_train", 0) or getattr(a, "probe_test", 0))


def probe_parts(structs):
    """the cores (every molecule but the last) and the probe's dimers (core molecule i + the probe, the last molecule)
    of probe configurations, as positions in A, with the configuration each dimer belongs to."""
    cores, pairs, owner = [], [], []
    for k, st in enumerate(structs):
        pos = bohr_to_ang(st.positions); n = len(pos) // 3
        cores.append(pos[:3 * (n - 1)])
        pairs += [np.concatenate([pos[3 * i:3 * i + 3], pos[3 * (n - 1):]]) for i in range(n - 1)]
        owner += [k] * (n - 1)
    return cores, pairs, np.array(owner)


def probe_truth(structs, truth):
    """the truth's probe interaction E(core + probe) - E(core) and its non-additive part (minus the probe's dimer
    energies) for every probe configuration, and -- for a truth with TTM3-F's pieces -- the probe's induction (the
    change of the induction energy, its pairwise part included) and the non-additive induction through first and
    second order in the polarisability (TTM3-F re-evaluated from its sites: the share of alpha^2 T and beyond)."""
    _, pairs, owner = probe_parts(structs)
    inter = np.array([st.energy - st.info["E_core"] for st in structs])
    E_pairs = np.bincount(owner, weights=[truth.energy(p, 2) for p in pairs], minlength=len(structs))
    out = dict(d=np.array([st.info["d"] for st in structs]), interaction=inter, nonadditive=inter - E_pairs)
    if hasattr(getattr(truth, "qa", None), "ttm"):
        out["induction"] = np.array([truth.qa.ttm(st.positions)["ind"] - truth.qa.ttm(st.positions[:-3])["ind"] for st in structs])
    if hasattr(getattr(truth, "qa", None), "induction"):
        for key, order in (("nonadditive_induction_first", 1), ("nonadditive_induction_second", 2)):
            ind = lambda P: truth.qa.induction(P, order)[0]
            pair_ind = np.bincount(owner, weights=[ind(ang_to_bohr(p)) for p in pairs], minlength=len(structs))
            out[key] = np.array([ind(st.positions) - ind(st.positions[:-3]) for st in structs]) - pair_ind
    return out


def far_field_blind(structs, rcut_ang):
    """the probe configurations whose probe has no atom within r_c of any core atom: E_0 (cutoff r_c, messages along
    pairs within it) sees the probe and the core as disconnected, so the probe's interaction energy is the bands' and the
    read-out's alone."""
    out = []
    for st in structs:
        pos = bohr_to_ang(st.positions)
        gap = np.linalg.norm(pos[:-3, None, :] - pos[None, -3:, :], axis=-1).min()
        out.append(gap > rcut_ang)
    return np.array(out, dtype=bool)


def probe_interaction_labels(structs, truth):
    """training probes relabelled with the probe's interaction energy and its forces: E(core + probe) - E(core) and
    F(all) - F(core), with the core and the probe as the subtraction groups (Structure.sub), so that the model's energy
    is the same difference.  The cluster energy, common to every probe of a core, leaves the label: what remains is the
    pairwise electrostatics and dispersion of the probe with the core (the pinned bands' part beyond r_c) and the
    induction, the signal the read-out is matched on.  Copies; the originals keep their totals for the readings."""
    from dataclasses import replace
    out, cache = [], {}
    for st in structs:
        n = st.n_atoms // 3; core = bohr_to_ang(st.positions[:3 * (n - 1)])
        key = core.tobytes()
        if key not in cache:
            cache[key] = truth.energy_forces(core, n - 1)
        E_core, F_core = cache[key]
        if abs(E_core - st.info["E_core"]) > 1e-9:                # the same core (to the A-bohr round trip's noise)
            raise ValueError("the probe configuration's core energy does not match the truth's")
        F = np.array(st.forces, float); F[:3 * (n - 1)] -= F_core
        out.append(replace(st, energy=st.energy - st.info["E_core"], forces=F, sub=np.repeat([0, 1], [3 * (n - 1), 3]),
                           info={**st.info, "label": "interaction"}))
    return out


def probe_model(params, rung, structs, truth, a):
    """the model's probe interaction and its non-additive part, as probe_truth (each sub-structure with its own pins)."""
    cores, pairs, owner = probe_parts(structs)
    cores = pin_for(rung, [W.water_structure(p, charges=truth.molecule_charges(p)) for p in cores], truth)
    pairs = pin_for(rung, [W.water_structure(p, charges=truth.molecule_charges(p)) for p in pairs], truth)
    inter = T.predict_energies(params, rung, structs) - T.predict_energies(params, rung, cores)
    return inter, inter - np.bincount(owner, weights=T.predict_energies(params, rung, pairs), minlength=len(structs))


def probe_metrics(model, ref):
    """rmse of the model's probe interaction and non-additive energy against the truth's, overall and by distance,
    with the truth's rms (hartree)."""
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    inter, nadd = model
    def block(m):
        return {"count": int(m.sum()), "interaction_rmse": rms(inter[m] - ref["interaction"][m]),
                "interaction_rms": rms(ref["interaction"][m]), "nonadditive_rmse": rms(nadd[m] - ref["nonadditive"][m]),
                "nonadditive_rms": rms(ref["nonadditive"][m])}
    out = block(np.ones(len(inter), dtype=bool))
    out["by_distance_angstrom"] = {f"{d:g}": block(ref["d"] == d) for d in np.unique(ref["d"])}
    return out


def probe_summary(m):
    """one line: the probe readings as shares of the truth's rms, overall and by distance."""
    pct = lambda b, k: 100 * b[f"{k}_rmse"] / b[f"{k}_rms"] if b[f"{k}_rms"] > 0 else float("nan")
    return (f"interaction {m['interaction_rmse'] * 627.5094740631:.4f} kcal/mol ({pct(m, 'interaction'):.0f} %), "
            f"non-additive {m['nonadditive_rmse'] * 627.5094740631:.4f} ({pct(m, 'nonadditive'):.0f} %); by d: "
            + ", ".join(f"{d} A {pct(b, 'interaction'):.0f} / {pct(b, 'nonadditive'):.0f} %" for d, b in m["by_distance_angstrom"].items()))


def prepare_heldout(a, truth):
    """the raw CCSD(T) n-body sets of the q-AQUA repository (sseft.qaqua.RAW_SETS) on the frames whose every monomer is
    within the rigid tolerance of the rigid geometry (the models are trained on rigid monomers): positions (A), labels,
    the PES's own n-body values on the same frames (its fit error is the reference reading), the LTP2011 charges of
    every molecule at its own geometry (pinned for the dagger rungs), the strict subset and the largest O-O distance."""
    sets = {}
    for key in heldout_selection(a):
        fname, n = QA.RAW_SETS[key]
        pos, lab = QA.read_raw(os.path.join(truth.qa.root, "q-AQUA", fname), n)
        r, th = QA.monomer_internal(pos)
        dr = np.abs(r - W.R_OH).max(axis=(1, 2)); dt = np.abs(th - math.degrees(W.THETA)).max(axis=1)
        idx = np.flatnonzero((dr <= a.heldout_oh_tol) & (dt <= a.heldout_angle_tol))
        n_rigid = len(idx)
        if a.heldout_max_pair_kcal:
            # the training domain: no pair more repulsive (on the truth the models are trained on) than the cap
            worst = np.array([max(truth.energy(np.concatenate([pos[f][3 * i:3 * i + 3], pos[f][3 * j:3 * j + 3]]), 2)
                                  for i in range(n) for j in range(i + 1, n)) for f in idx])
            idx = idx[worst * 627.5094740631 <= a.heldout_max_pair_kcal]
        n_domain = len(idx)
        if len(idx) == 0:
            print(f"  held-out {key}: no frame within the rigid tolerance and the training domain; set skipped", flush=True)
            continue
        if a.heldout_max and len(idx) > a.heldout_max:
            idx = np.sort(np.random.default_rng(0).choice(idx, a.heldout_max, replace=False))
        P = pos[idx]
        sets[key] = dict(n=n, pos=P, label=lab[idx], total=len(pos), near_rigid=n_rigid, in_domain=n_domain,
                         pes=np.array([truth.qa.nbody(ang_to_bohr(p), max_order=n)[n] for p in P]),
                         charges=np.array([truth.molecule_charges(p) for p in P]),
                         strict=(dr[idx] <= a.heldout_strict_oh_tol) & (dt[idx] <= a.heldout_strict_angle_tol),
                         roo=QA.largest_oo(P))
    return sets


def model_nbody(params, rung, frames, charges, n, truth=None):
    """the model's n-body energy of each frame (A, (frames, 3n, 3)) by the subset expansion over its sub-clusters, the
    same decomposition as the truth's: E_int of every subset of 2..n molecules, each with its molecules' charges and
    the rung's other pins (the dispersion alpha, the site model).  Before v0.0.11 the held-out frames carried the
    charges only: the M_16 rungs' 2-body readings lacked their dispersion band (their 3- and 4-body readings are
    unaffected -- the band is pairwise)."""
    subsets = [c for k in range(2, n + 1) for c in itertools.combinations(range(n), k)]
    atoms = {c: np.concatenate([np.arange(3 * i, 3 * i + 3) for i in c]) for c in subsets}
    structs = [W.water_structure(p[atoms[c]], charges=q[atoms[c]]) for p, q in zip(frames, charges) for c in subsets]
    if truth is not None:
        pin_for(rung, structs, truth)
    E = T.predict_energies(params, rung, structs).reshape(len(frames), len(subsets))
    out = np.zeros(len(frames))
    for f in range(len(frames)):
        term = {}
        for j, c in enumerate(subsets):
            term[c] = E[f, j] - sum(term[d] for k in range(2, len(c)) for d in itertools.combinations(c, k))
        out[f] = term[tuple(range(n))]
    return out


def heldout_metrics(pred, d):
    """rmse of the model's n-body energies against the CCSD(T) labels, with the PES's on the same frames: all frames,
    the strict near-rigid subset, and by the largest O-O distance (hartree)."""
    err = pred - d["label"]; ref = d["pes"] - d["label"]
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    def block(m):
        if not np.any(m):
            return {"count": 0}
        return {"count": int(np.sum(m)), "rmse_model": rms(err[m]), "rmse_pes": rms(ref[m]), "label_rms": rms(d["label"][m]),
                "mean_error_model": float(np.mean(err[m]))}
    out = block(np.ones(len(err), dtype=bool))
    out["strict"] = block(d["strict"])
    out["by_largest_oo_angstrom"] = {f"{lo:g}-{hi:g}": block((d["roo"] >= lo) & (d["roo"] < hi)) for lo, hi in HELDOUT_OO_BINS}
    return out


def run_clusters(a, truth=None):
    t0 = time.time()
    truth = truth or make_truth(a)
    settings = cluster_data_settings(a, truth)
    geometry = dict(mc_steps=settings["mc_steps"], T_K=settings["temperature_kelvin"],
                    radius_ang=settings["radius_angstrom"], min_OO=settings["min_oo_angstrom"],
                    min_any=settings["min_atom_distance_angstrom"])
    probes = {"train": [], "test": []}
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as fh:
            cached = pickle.load(fh)
        if not isinstance(cached, dict) or cached.get("dataset_config") != settings:
            raise ValueError("cluster cache has missing or different generation settings; use a new --cache path")
        train, tests = cached["train"], cached["tests"]
        probes = cached.get("probes", probes)
        print(f"data from {a.cache}", flush=True)
    else:
        train = W.cluster_dataset(settings["train_sizes"], settings["n_train"],
                                  seed=settings["train_seed"], sigma=settings["sigma"], truth=truth, **geometry)
        train += [W.monomer_structure(truth)] * settings["monomer_references"]
        tests = {n: W.cluster_dataset((n,), settings["n_test"],
                                     seed=settings["test_seed"] + 100 * i, truth=truth, **geometry)
                 for i, n in enumerate(settings["test_sizes"])}
        if probes_requested(a):
            # a relaxed trimer (relaxed as the clusters are) and one monomer at each distance, in random directions and
            # orientations: the training ones identify the read-out's response to a distant molecule; with
            # probe_train_core_sizes the training cores are also tetramers, pentamers ... (one seed per size, the first
            # size's being probe_seed, so that [3] gives the probes of before)
            core_geometry = {k: v for k, v in geometry.items() if k != "mc_steps"}
            sizes = {"train": settings.get("probe_train_core_sizes", [3]), "test": [3]}
            for key, seed, n_cores in (("train", settings["probe_seed"], settings["probe_train_cores"]),
                                       ("test", settings["probe_seed"] + 100, settings["probe_test_cores"])):
                count = settings[f"probe_{key}"]
                for k, n_core in enumerate(sizes[key] if count else []):
                    made = W.probe_configurations(n_core, settings["probe_distances_angstrom"], count, seed=seed + 1000 * k,
                                                  mc_steps=settings["mc_steps"], n_cores=n_cores, truth=truth,
                                                  **core_geometry)[0]
                    for st in made:
                        st.info.update(n_core=n_core, core_id=1000 * k + st.info["core"])
                    probes[key] += made
        if a.cache:
            with open(a.cache, "wb") as fh:
                pickle.dump(dict(dataset_config=settings, train=train, tests=tests,
                                 **({"probes": probes} if probes_requested(a) else {})), fh)
    pin_sources(a, train, *tests.values(), truth=truth)
    pin_sources(a, probes["train"], probes["test"], n_core=3, truth=truth)
    for st in probes["train"]:
        st.weight = a.w_far                                          # training.loss.far_weight
    far_train = probes["train"]
    if probes["train"] and getattr(a, "far_labels", "total") == "interaction":
        far_train = probe_interaction_labels(probes["train"], truth)
    nadd_truth = {n: np.array([s.energy - W.pair_additive_energy(bohr_to_ang(s.positions), n, truth)
                              for s in structs]) for n, structs in tests.items()}
    train_count = len(train) - settings["monomer_references"]
    print(f"clusters ({truth.name} truth): {train_count} training (sizes {settings['train_sizes']}), "
          + ", ".join(f"{len(structs)} test clusters of size {n}" for n, structs in tests.items()), flush=True)
    results = {"dataset": {**settings, "train_count": train_count,
                           "test_counts": {str(n): len(structs) for n, structs in tests.items()}},
               "reference_nonadditive": {str(n): v.tolist() for n, v in nadd_truth.items()}}
    if 6 in tests:
        results["nadd6_truth"] = nadd_truth[6].tolist()
    kcal = 627.5094740631
    probe_ref = {key: probe_truth(structs, truth) for key, structs in probes.items() if structs}
    if probe_ref:
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        results["probe_truth"] = {key: {k: v.tolist() for k, v in ref.items()} for key, ref in probe_ref.items()}
        for key, ref in probe_ref.items():
            print(f"  probe configurations ({key}): {len(ref['d'])} (a trimer and a monomer at d = "
                  + ", ".join(f"{d:g}" for d in np.unique(ref["d"])) + " A); by d: interaction rms / non-additive rms"
                  + (" / induction rms" if "induction" in ref else "") + " (kcal/mol) "
                  + "; ".join(f"{d:g} A {rms(ref['interaction'][ref['d'] == d]) * kcal:.4f} / {rms(ref['nonadditive'][ref['d'] == d]) * kcal:.4f}"
                              + (f" / {rms(ref['induction'][ref['d'] == d]) * kcal:.4f}" if "induction" in ref else "")
                              for d in np.unique(ref["d"])), flush=True)
    if truth.name != "pw":
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        y_tr = [s.energy for s in train if s.n_atoms > 3]                  # the clusters (the probes are reported above)
        nb = {n: [truth.qa.nbody(s.positions) for s in structs] for n, structs in tests.items() if n >= 3}
        results["setup"] = {"truth": truth.describe(), "train_energy_rms": rms(y_tr),
                            "test_nbody": {str(n): {f"{k}b": [d[k] for d in v] for k in (3, 4) if k <= n} for n, v in nb.items()}}
        print(f"  training interaction energies rms {rms(y_tr) * kcal:.2f} kcal/mol; non-additive energy of the test clusters: "
              + "; ".join(f"size {n} rms {rms(nadd_truth[n]) * kcal:.3f} kcal/mol ({rms(nadd_truth[n]) / rms([s.energy for s in tests[n]]) * 100:.1f} % "
                          f"of the interaction)" + (f", 3-body rms {rms([d[3] for d in nb[n]]) * kcal:.3f}" if n in nb else "")
                          + (f", 4-body rms {rms([d[4] for d in nb[n]]) * kcal:.3f}" if n in nb and n >= 4 else "")
                          for n in tests), flush=True)
    heldout = prepare_heldout(a, truth) if heldout_selection(a) else {}
    if heldout:
        summary = {}
        for key, d in heldout.items():
            m = heldout_metrics(d["pes"], d)                        # the PES against itself: the reference columns only
            summary[key] = {"frames": int(len(d["label"])), "of": d["total"], "near_rigid": d["near_rigid"],
                            "in_domain": d["in_domain"], "strict": int(d["strict"].sum()),
                            "label_rms": m["label_rms"] if m["count"] else None, "pes_rmse": m["rmse_pes"] if m["count"] else None,
                            "by_largest_oo_angstrom": {b: {"count": v["count"], **({"label_rms": v["label_rms"], "pes_rmse": v["rmse_pes"]}
                                                                                    if v["count"] else {})}
                                                       for b, v in m["by_largest_oo_angstrom"].items()}}
            print(f"  held-out {key}: {len(d['label'])} raw CCSD(T) frames read ({d['near_rigid']} of {d['total']} with near-rigid "
                  f"monomers, {d['in_domain']} of those in the training domain; {int(d['strict'].sum())} strict); labels rms {summary[key]['label_rms'] * kcal:.4f} kcal/mol, the PES's own rmse "
                  f"{summary[key]['pes_rmse'] * kcal:.4f}; by largest O-O: "
                  + ", ".join(f"{b} A n = {v['count']}" for b, v in summary[key]["by_largest_oo_angstrom"].items()), flush=True)
        results["heldout_sets"] = summary
    if os.path.exists(a.out):
        with open(a.out) as fh:
            previous = json.load(fh)
        if previous and (not isinstance(previous, dict) or previous.get("dataset") != results["dataset"]):
            raise ValueError("output contains results for different or unknown dataset settings; use a new --out path")
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a)
        metrics = {str(n): dict(energy_rmse=[], nonadditive_rmse=[]) for n in tests}
        heldout_runs = {key: [] for key in heldout}
        probe_runs = {key: [] for key in probe_ref}
        fitted_options = []
        fit_kwargs = training_kwargs(a)
        if rung.band_fields >= 0 and getattr(a, "readout_stage", "after_adam") == "far_first":
            # the response matched on the far field before E_0 is trained: solved on the training probes E_0 cannot see
            # (no probe atom within r_c of the core: their interaction energy is the bands' and the read-out's alone),
            # the ridge chosen on held-out cores, then held while E_0 learns the rest
            blind = far_field_blind(far_train, a.rcut)
            if not blind.any():
                raise ValueError("no training probe lies beyond r_c of its core: nothing to match the response on")
            idx = np.flatnonzero(blind)
            # energies only: the truth's atomic forces on a rigid monomer carry the flux of its geometry-dependent charges
            # (TTM3-F's dipole-surface charges), which pinned charges cannot represent; its energies carry no such term
            fit_kwargs.update(readout_first=True, readout_solve_subset=len(train) + idx, readout_solve_w_force=0.0,
                              readout_solve_groups=np.array([far_train[i].info.get("core_id", far_train[i].info["core"]) for i in idx]))
            print(f"  {name}: the read-out matched on {len(idx)} of {len(far_train)} training probes (beyond r_c of their "
                  f"cores), before E_0", flush=True)
        elif rung.band_fields >= 0 and getattr(a, "readout_far_weight", None) is not None and far_train:
            # the read-out matched on the far field: the training probes (their interaction with the core, where E_0 sees
            # nothing beyond r_c) weighted up in the solve, the clusters keeping the response sane in the near field
            fit_kwargs["readout_solve_weights"] = np.array([st.weight for st in train] + [a.readout_far_weight] * len(far_train))
        if rung.band_fields >= 0 and getattr(a, "readout_ridge_selection", "fixed") == "groups" and "readout_solve_groups" not in fit_kwargs:
            # the ridge on held-out groups of the training set: each cluster one group, the probes by core
            fit_kwargs["readout_solve_groups"] = np.array([f"c{i}" for i in range(len(train))]
                                                          + [f"p{st.info.get('core_id', st.info['core'])}" for st in far_train])
        for seed in training_seeds(a):
            params, info = T.fit(rung, train + far_train, seed=seed, **fit_kwargs)
            fitted_options.append(info.get("training_options", {"seed": seed, **training_kwargs(a)}))
            if truth.name != "pw":
                save_params(a, name, seed, params)
            for n, structs in tests.items():
                # energies only, every cluster's dimers in one call (one compilation per shape, reused across seeds)
                energies = T.predict_energies(params, rung, structs)
                dims, owner = [], []
                for k, st in enumerate(structs):
                    pos = bohr_to_ang(st.positions)
                    for i in range(n):
                        for j in range(i + 1, n):
                            pair = np.concatenate([pos[3*i:3*i+3], pos[3*j:3*j+3]])
                            dims.append(W.water_structure(pair, charges=truth.molecule_charges(pair))); owner.append(k)
                pin_for(rung, dims, truth)
                pair_energies = T.predict_energies(params, rung, dims)
                nadd_model = energies - np.bincount(owner, weights=pair_energies, minlength=len(structs))
                energy_rmse = float(np.sqrt(np.mean((energies - np.array([st.energy for st in structs])) ** 2)))
                nonadd_rmse = float(np.sqrt(np.mean((np.array(nadd_model) - nadd_truth[n]) ** 2)))
                metrics[str(n)]["energy_rmse"].append(energy_rmse)
                metrics[str(n)]["nonadditive_rmse"].append(nonadd_rmse)
                print(f"  {name} seed {seed}: size {n} energy rmse {energy_rmse * kcal:.3f} kcal/mol, "
                      f"non-additive rmse {nonadd_rmse * kcal:.3f} kcal/mol; {info['time']:.0f} s", flush=True)
            for key, ref in probe_ref.items():
                m = probe_metrics(probe_model(params, rung, probes[key], truth, a), ref)
                if rung.bf_mutual != 0 and key == "test":
                    # the model's own expansion: its read-out truncated at first and at second order in its polarisabilities
                    m["model_orders"] = {str(k): probe_metrics(probe_model(params, M.with_rung(rung, bf_mutual=k), probes[key],
                                                                           truth, a), ref) for k in (1, 2)}
                probe_runs[key].append(m)
                print(f"  {name} seed {seed}: probes ({key}) {probe_summary(m)}", flush=True)
                for k, mk in m.get("model_orders", {}).items():
                    print(f"  {name} seed {seed}: probes ({key}), its read-out through order {k} in the polarisabilities: "
                          f"{probe_summary(mk)}", flush=True)
            for key, d in heldout.items():
                t1 = time.time()
                m = heldout_metrics(model_nbody(params, rung, d["pos"], d["charges"], d["n"], truth), d)
                heldout_runs[key].append(m)
                far = m["by_largest_oo_angstrom"]
                print(f"  {name} seed {seed}: held-out {key} rmse {m['rmse_model'] * kcal:.4f} kcal/mol (PES {m['rmse_pes'] * kcal:.4f}, "
                      f"labels {m['label_rms'] * kcal:.4f}); strict {m['strict'].get('rmse_model', float('nan')) * kcal:.4f}; by largest O-O "
                      + ", ".join(f"{b} A {v['rmse_model'] * kcal:.4f} / PES {v['rmse_pes'] * kcal:.4f}" for b, v in far.items() if v["count"])
                      + f"  [{time.time() - t1:.0f} s]", flush=True)
        results[name] = {"clusters_by_size": metrics}
        if "6" in metrics:
            results[name].update(hexamer=metrics["6"]["energy_rmse"],
                                 nonadditive=metrics["6"]["nonadditive_rmse"])
        if "2" in metrics:
            results[name]["dimer"] = metrics["2"]["energy_rmse"]
        if heldout:
            results[name]["heldout"] = heldout_runs
        if probe_ref:
            results[name]["probes"] = probe_runs
        results[name]["training_runs"] = fitted_options
        _save(a, results)
    print(f"clusters completed in {time.time() - t0:.0f} s", flush=True)
    return results


def response_coefficient(params, rung):
    """the model's band-field energy for a monomer with the ion at 20 A (every pair beyond r_c, so the band-field
    read-out is the only term beyond electrostatics) against the truth's exact one-site response -1/2 alpha |E|^2:
    returns a_eff / alpha."""
    pos = np.vstack([W.monomer_positions(), [[0.0, 0.0, 20.0]]])
    st = W.ion_structure(pos, 1); pin_dispersion([st])
    aux = T.predict_aux(params, rung, [st])
    truth = W.pw_components(pos, 1, n_ion=1)["ind"]
    return float(aux["E_bf"][0]) / truth if "E_bf" in aux else float("nan")


def run_ion(a):
    """the band-field response test (E3-R): a unit charge at d from a relaxed trimer.  Beyond r_c the ion's interaction
    with the core, minus its point-charge electrostatics with the pinned permanent charges, is the induction -- the
    core's embedded induced dipoles against the ion's field plus the core's response to it -- of which 86-98 % is first
    order in the band field of the permanent sources (scratch/first_order_share.py), i.e. the order-0 operator's."""
    t0 = time.time()
    dists_tr = (4.0, 5.0, 6.0, 7.0, 8.0, 10.0); dists = (6.0, 7.0, 8.0, 10.0, 12.0)
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as fh: base, ion_mono, ion_train, test, E_core = pickle.load(fh)
        restore_fragment_charges(ion_mono + ion_train + test)                       # caches written before v0.0.8
        print(f"data from {a.cache}", flush=True)
    else:
        if a.base_cache and os.path.exists(a.base_cache):
            with open(a.base_cache, "rb") as fh: base = pickle.load(fh)[0]
            print(f"base data (dimer window + clusters) from {a.base_cache}", flush=True)
        else:
            R_train = bohr_to_ang(D.window_design(*WINDOW, 40)); rng = np.random.default_rng(0)
            base = (W.dimer_scan(R_train, sigma=a.sigma, rng=rng) + W.cluster_dataset((3, 4, 5), a.n_train, seed=0, sigma=a.sigma, mc_steps=a.mc_steps)
                    + [W.monomer_structure()] * 4)
        ion_mono = W.ion_monomer_scan((2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0), 6, seed=0)
        ion_train, _ = W.ion_configurations(3, dists_tr, a.train_far, seed=400, mc_steps=a.mc_steps, n_cores=a.train_far_cores)
        test, E_core = W.ion_configurations(3, dists, a.n_test, seed=300, mc_steps=a.mc_steps, n_cores=a.test_cores)
        if a.cache:
            with open(a.cache, "wb") as fh: pickle.dump((base, ion_mono, ion_train, test, E_core), fh)
    extra = []
    if a.extra_far > 0:
        # pure far-field identification data: the ion beyond r_c of every atom (d >= 10 A from a trimer's centre of mass,
        # 8-14 A from a monomer's oxygen), from other cores -- the only configurations in which the band-field coefficient
        # is not shared with E_0 (e3r-bandfield-response-preregistration.md section 10)
        if a.extra_cache and os.path.exists(a.extra_cache):
            with open(a.extra_cache, "rb") as fh: extra = pickle.load(fh)
            restore_fragment_charges(extra)
        else:
            far_cores, _ = W.ion_configurations(3, (10.0, 12.0, 14.0), a.extra_far, seed=500, mc_steps=a.mc_steps, n_cores=a.train_far_cores)
            far_mono = W.ion_monomer_scan((8.0, 10.0, 12.0, 14.0), 3 * a.extra_far, seed=1)
            extra = far_cores + far_mono
            if a.extra_cache:
                with open(a.extra_cache, "wb") as fh: pickle.dump(extra, fh)
        for st in extra: st.weight = a.w_far
    for st in ion_train: st.weight = a.w_far
    pin_sources(a, base, ion_mono); pin_sources(a, ion_train, test, n_core=3); pin_sources(a, extra, n_core=3)
    train = base + ion_mono + ion_train + extra
    clusters = [s for s in base if s.n_atoms > 6]
    dec = [W.ion_response_decomposition(st) for st in test]; dec_tr = [W.ion_response_decomposition(st) for st in ion_train]
    y = np.array([d["y"] for d in dec]); ind = np.array([d["y"] - d["es"] for d in dec]); resp = np.array([d["response"] for d in dec])
    y_tr = np.array([d["y"] for d in dec_tr]); ind_tr = np.array([d["y"] - d["es"] for d in dec_tr])
    dd = np.array([s.info["d"] for s in test]); K = 627.5
    rms = lambda v: np.sqrt(np.mean(v ** 2))
    print(f"band-field response test: {len(test)} ion configurations from {a.test_cores} trimer cores at d = {dists} A; ion interaction rms "
          f"{rms(y) * K:.4f} kcal/mol, induction (embedded dipoles x ion field + response) rms {rms(ind) * K:.4f} ({rms(ind) / rms(y) * 100:.1f} %), "
          f"of which the response {rms(resp) * K:.4f}; per distance induction " + ", ".join(f"{d:.0f} A {rms(ind[dd == d]) * K:.4f}" for d in dists)
          + f"  [{time.time() - t0:.0f} s]", flush=True)
    print(f"  training: {len(base)} base + {len(ion_mono)} ion-monomer + {len(ion_train)} ion-core configurations ({a.train_far_cores} cores, d = {dists_tr}, weight {a.w_far:g})"
          + (f" + {len(extra)} pure far-field configurations (ion beyond r_c of every atom)" if extra else "") + f"; the ion-core configurations' induction rms {rms(ind_tr) * K:.4f} kcal/mol", flush=True)
    results = {"d": dd.tolist(), "ion_interaction": y.tolist(), "induction": ind.tolist(), "response": resp.tolist()}

    def ion_rmse(params, rung, structs, yy):
        Et, _ = T.predict(params, rung, structs)
        cores = [W.water_structure(bohr_to_ang(s.positions)[:-1]) for s in structs]
        if rung.pin_mu: pin_embedded(cores)
        elif rung.pin_alpha: pin_dispersion(cores)
        Ec, _ = T.predict(params, rung, cores)
        return (Et - Ec) - yy
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a); t1 = time.time(); errs = []; train_errs = []; coefs = []
        fitted_options = []
        for seed in training_seeds(a):
            params, info = T.fit(rung, train, seed=seed, **training_kwargs(a))
            fitted_options.append(info.get("training_options", {"seed": seed, **training_kwargs(a)}))
            res = ion_rmse(params, rung, test, y); r = rms(res); errs.append(float(r))
            print(f"  {name} seed {seed}: ion-interaction rmse {r * K:.4f} kcal/mol = {r / rms(ind) * 100:.0f} % of the induction "
                  f"(per distance: " + ", ".join(f"{d:.0f} A {rms(res[dd == d]) / rms(ind[dd == d]) * 100:.0f} %" for d in dists)
                  + f"; of the response alone {r / rms(resp) * 100:.0f} %); {info['time']:.0f} s", flush=True)
            Ecl, _ = T.predict(params, rung, clusters); r_cl = rms(Ecl - np.array([s.energy for s in clusters]))
            res_tr = ion_rmse(params, rung, ion_train, y_tr); r_tr = rms(res_tr)
            d_tr = np.array([s.info["d"] for s in ion_train]); far = d_tr >= 6.0
            r_far = rms(res_tr[far]); ind_far = rms(ind_tr[far])
            per_d = ", ".join(f"{d:.0f} A {rms(res_tr[d_tr == d]) / rms(ind_tr[d_tr == d]) * 100:.0f} %" for d in dists_tr)
            coef = response_coefficient(params, rung); coefs.append(coef)
            msg = (f"      training residuals: clusters {r_cl * K:.4f} kcal/mol; ion-core configurations {r_tr * K:.4f} = {r_tr / rms(ind_tr) * 100:.0f} % of their induction "
                   f"(d >= 6 A: {r_far / ind_far * 100:.0f} %; per distance {per_d})")
            if extra:
                cores_x = [s for s in extra if s.n_atoms > 4]; dec_x = [W.ion_response_decomposition(st) for st in cores_x]
                y_x = np.array([d["y"] for d in dec_x]); ind_x = np.array([d["y"] - d["es"] for d in dec_x])
                r_x = rms(ion_rmse(params, rung, cores_x, y_x)); msg += f"; pure far-field configurations {r_x / rms(ind_x) * 100:.0f} %"
            print(msg + f"; response coefficient a_eff / alpha from a monomer + ion at 20 A: {coef:.3f}", flush=True)
            train_errs.append(dict(clusters=float(r_cl), ion=float(r_tr), ion_far=float(r_far)))
            with open(a.out.replace(".json", f"_{name}_seed{seed}.params.pkl"), "wb") as fh:
                pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)
        results[name] = dict(rmse=errs, train=train_errs, a_eff_over_alpha=coefs, w_far=a.w_far)
        results[name]["training_runs"] = fitted_options
        _save(a, results)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="dimer", choices=["dimer", "clusters", "far", "ion"])
    ap.add_argument("--rungs", default="M_inf,M_G,M_A,M_1,M_1mu,M_1p")
    ap.add_argument("--charges", choices=["pinned", "learned"], default=None,
                    help="override charge sources for all charge-bearing water rungs; omitted keeps preset defaults")
    ap.add_argument("--charge_neutrality", choices=["fragment", "structure"], default="fragment",
                    help="neutrality shift of learned charges: per fragment (each molecule neutral, the ion charged) or one shift over the structure (before v0.0.8)")
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lbfgs", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=1e-9); ap.add_argument("--rcut", type=float, default=6.0); ap.add_argument("--lA", type=float, default=4.0)
    ap.add_argument("--F", type=int, default=16); ap.add_argument("--rs_over_l1", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=12); ap.add_argument("--n_test", type=int, default=8); ap.add_argument("--mc_steps", type=int, default=300)
    ap.add_argument("--train_sizes", default="3,4,5", help="cluster training sizes, comma-separated molecule counts")
    ap.add_argument("--test_sizes", default="6,2", help="cluster test sizes, comma-separated molecule counts")
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--test_seed", type=int, default=100, help="test data seed; subsequent sizes add 100 each")
    ap.add_argument("--monomer_references", type=int, default=4)
    ap.add_argument("--temperature_kelvin", type=float, default=300.0)
    ap.add_argument("--radius_angstrom", type=float, default=None)
    ap.add_argument("--min_oo_angstrom", type=float, default=2.7)
    ap.add_argument("--min_atom_distance_angstrom", type=float, default=1.7)
    ap.add_argument("--batch", type=int, default=12); ap.add_argument("--lbfgs_chunk", type=int, default=6)
    ap.add_argument("--train_far", type=int, default=0, help="far-field probe configurations per distance and core added to the training set")
    ap.add_argument("--train_far_cores", type=int, default=8, help="number of different core trimers in the far-field training set")
    ap.add_argument("--test_cores", type=int, default=1)
    ap.add_argument("--l1", type=float, default=0.0, help="band edge l_1 in A (0: the rung's default, 1.5 A); YAML model.band_edge_angstrom")
    ap.add_argument("--response_edge", type=float, default=0.0,
                    help="the read-out's band edge in A: its band field and dipole tensor (0: the band edge); YAML model.response_edge_angstrom")
    ap.add_argument("--w_far", type=float, default=1.0, help="training weight of the far-field configurations (energies and forces)")
    ap.add_argument("--cache", default="", help="pickle of the generated data sets of the part (written if absent, read if present)")
    ap.add_argument("--base_cache", default="", help="ion part: a far-part cache whose base (dimer window + clusters) is reused")
    ap.add_argument("--extra_far", type=int, default=0, help="ion part: pure far-field configurations per distance and core (d = 10, 12, 14 A) added to the training set")
    ap.add_argument("--extra_cache", default="", help="ion part: cache of the pure far-field configurations")
    ap.add_argument("--out", default="e3_water_results.json")
    # the truth (YAML: truth.name / truth.root) and the held-out raw CCSD(T) sets of q-AQUA (YAML: truth.heldout)
    ap.add_argument("--truth", choices=["pw", "qaqua", "qaqua_pol"], default="pw",
                    help="pw: the synthetic polarisable water of the rehearsal; qaqua: q-AQUA v1 (external/qaqua/build.sh); "
                         "qaqua_pol: q-AQUA-pol (external/qaqua/build_pol.sh)")
    ap.add_argument("--qaqua_root", default=None, help="the q-AQUA (-pol) build directory; omitted: external/qaqua/build "
                                                       "(qaqua) or external/qaqua/build_pol (qaqua_pol)")
    ap.add_argument("--heldout", default="none", help="clusters part on q-AQUA: raw CCSD(T) sets read with every model, "
                                                      "comma-separated from 2b,3b,4b, or none")
    ap.add_argument("--heldout_oh_tol", type=float, default=0.03, help="held-out frames: every O-H within this (A) of the rigid geometry")
    ap.add_argument("--heldout_angle_tol", type=float, default=5.0, help="held-out frames: every H-O-H within this (degrees)")
    ap.add_argument("--heldout_strict_oh_tol", type=float, default=0.02, help="the strict subset reported separately (A)")
    ap.add_argument("--heldout_strict_angle_tol", type=float, default=3.0, help="the strict subset reported separately (degrees)")
    ap.add_argument("--heldout_max_pair_kcal", type=float, default=2.0,
                    help="held-out frames: every pair's truth 2-body energy at most this (kcal/mol; the training domain); 0: no cap")
    ap.add_argument("--heldout_max", type=int, default=0, help="at most this many frames per held-out set (0: all)")
    # probe configurations in the clusters part (YAML: its data mapping): a trimer and one monomer at each distance
    ap.add_argument("--probe_distances_angstrom", default=CLUSTER_PROBE_DEFAULTS["probe_distances_angstrom"],
                    help="probe distances (A, the probe's O to the trimer's O centroid), comma-separated")
    ap.add_argument("--probe_train", type=int, default=0, help="training probe configurations per distance and core (0: none)")
    ap.add_argument("--probe_train_cores", type=int, default=CLUSTER_PROBE_DEFAULTS["probe_train_cores"])
    ap.add_argument("--probe_test", type=int, default=0, help="test probe configurations per distance and core (0: none)")
    ap.add_argument("--probe_test_cores", type=int, default=CLUSTER_PROBE_DEFAULTS["probe_test_cores"])
    ap.add_argument("--probe_seed", type=int, default=CLUSTER_PROBE_DEFAULTS["probe_seed"],
                    help="seed of the training probes (the test probes: +100)")
    ap.add_argument("--probe_train_core_sizes", default=CLUSTER_PROBE_DEFAULTS["probe_train_core_sizes"],
                    help="molecules in the training probes' cores, comma-separated (one seed per size: probe_seed + 1000 k)")
    # the dimer design (YAML: the data mapping of the dimer part)
    ap.add_argument("--re_angstrom", type=float, default=None, help="dimer minimum R_e (A) setting the window; omitted: the truth's own")
    ap.add_argument("--window_lower_over_re", type=float, default=1.35)
    ap.add_argument("--window_ratio", type=float, default=1.8)
    ap.add_argument("--window_nodes", type=int, default=40)
    ap.add_argument("--phi_degrees", type=float, default=0.0, help="acceptor tilt of the fixed-orientation dimer")
    ap.add_argument("--eval_max_angstrom", type=float, default=500.0)
    ap.add_argument("--eval_points", type=int, default=61)
    add_training_arguments(ap, "water")
    a = parse_selection(ap, "water", rungs=RUNGS, validate=validate_water_args)
    for path in (a.out, a.cache, a.extra_cache):                 # outputs and caches may name new directories
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    res = {"dimer": run_dimer, "clusters": run_clusters, "far": run_far, "ion": run_ion}[a.part](a)
    _save(a, res)


if __name__ == "__main__":
    main()
