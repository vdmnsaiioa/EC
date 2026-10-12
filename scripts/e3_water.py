"""
e3_water.py -- E3 on water (phase-d-build-plan.md section 3), runnable today on the synthetic polarisable truth PW
(sseft.water) and, with --truth file, on CCSD(T) data in the same layout.

    python scripts/e3_water.py --part dimer   --rungs M_inf,M_G,M_A,M_1,M_1mu,M_1p --seeds 4
    python scripts/e3_water.py --part clusters --rungs M_1p,M_S0p,M_S2p,M_1,M_S0 --seeds 2

(a) The dimer at the fixed orientation (p* = 3): the Phase 1c ladder on the O-O window [1.35 R_e, 1.8 x that],
    the mean's relative error, spread, local exponent and plateau slope on [30, 300] A, and the learned monomer
    dipole (sum q_i r_i + mu_i over an isolated monomer) against the truth's 1.855 D.
(b) Clusters n = 3..5 for training, n = 2 and 6 held out: interaction energies and the non-additive energy
    E - sum of the model's own dimer energies, for v1 (sources only) and v1.5 (band-field inputs at order 0 / 2,
    near sources excluded from the band field at r_s = l_1).

Rungs (the dagger rungs pin q to the monomer's point charges):
    M_inf, M_G, M_A   as in the dimer ladder
    M_1               Coulomb band, q learned
    M_1mu             Coulomb band, q and atomic dipoles learned
    M_1p              Coulomb band, q pinned                                   (v1 dagger)
    M_S0p, M_S2p      M_1p + band-field inputs at order 0 / 2                  (v1.5 dagger)
    M_S0, M_S2        M_1 + band-field inputs (q learned)
"""
from experiment_config import CLUSTER_DEFAULTS, add_training_arguments, parse_selection, training_kwargs, training_seeds
import argparse, time, json, sys, os, pickle
import numpy as np
import jax
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sseft
from sseft import water as W, model as M, train as T, measure as ME, data as D, kernels as kn
from sseft.units import bohr_to_ang, ang_to_bohr

R_E = 2.86                                     # A, the truth's dimer minimum at phi = 0
WINDOW = (1.35 * R_E, 1.35 * R_E * 1.8)        # [3.86, 6.95] A

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
}
C6_OO_TRUTH = 45.4


def pinned_alpha_oscillator(numbers, K=8):
    """one-oscillator alpha(i w) on the oxygens through the truth's static polarisability and O-O C6 (London), zero on H."""
    omega, w = kn.casimir_polder_grid(K)
    w0 = 4 * C6_OO_TRUTH / (3 * W.ALPHA_O ** 2)
    aO = W.ALPHA_O / (1 + (np.array(omega) / w0) ** 2)
    return np.array([aO if int(z) == 8 else np.zeros(K) for z in numbers])


def pin_dispersion(structs):
    """pin the one-oscillator alpha(i w) on every structure (the dispersion band of the permanent-source daggers)."""
    for s in structs:
        s.pinned = {**s.pinned, "alpha": pinned_alpha_oscillator(s.numbers)}
    return structs


def needs(a, key):
    return any(key in name for name in a.rungs.split(","))


def pin_sources(a, *sets, n_core=None):
    """apply the pins the requested rungs need to the given structure lists: embedded sources for "emb" rungs, the
    dispersion alpha for "16" rungs."""
    for structs in sets:
        if needs(a, "emb"): pin_embedded(structs, n_core=n_core)
        elif needs(a, "16"): pin_dispersion(structs)
    return sets


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
            "r_cut_angstrom": ang(rung.e0["r_cut"]), "features": int(rung.e0["F"]),
            "analytic_envelope_angstrom": ang(rung.lA), "periodic": bool(rung.periodic)}


def monomer_dipole(params, rung):
    """|sum_i q_i r_i + sum_i mu_i| of the isolated monomer, in Debye."""
    aux = T.predict_aux(params, rung, [W.monomer_structure()])
    pos = W.monomer_positions()                                  # A, O at the origin
    q = np.array(aux["q"][0, :3]); mu = np.array(aux["mu"][0, :3]) if "mu" in aux else np.zeros((3, 3))
    d = (q[:, None] * pos).sum(0) + bohr_to_ang(mu).sum(0)      # e A
    return float(np.linalg.norm(d) * W.DEBYE_PER_E_ANG), q


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
    old.update(results)
    json.dump(old, open(a.out, "w"), indent=1)


def run_dimer(a):
    R_train = bohr_to_ang(D.window_design(*WINDOW, 40))
    R_eval = bohr_to_ang(D.eval_grid(WINDOW[1], 500.0, 61))
    rng = np.random.default_rng(0)
    train = W.dimer_scan(R_train, sigma=a.sigma, rng=rng) + [W.monomer_structure()] * 4
    test = W.dimer_scan(R_eval)
    pin_sources(a, train, test)
    f_true = np.array([s.energy for s in test])
    print(f"water dimer (PW truth, phi = 0): window [{WINDOW[0]:.2f}, {WINDOW[1]:.2f}] A, 40 nodes, sigma = {a.sigma:g}; "
          f"|f*| at R+ = {abs(test[0].energy):.2e} E_h; E R^3 at 300 A = {f_true[-1] * ang_to_bohr(R_eval[-1]) ** 3:+.4f} (dipole-dipole -0.652)", flush=True)
    results = {}
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a)
        preds, rmses, dipoles = [], [], []; t0 = time.time()
        fitted_options = []
        for seed in training_seeds(a):
            params, info = T.fit(rung, train, seed=seed, **training_kwargs(a))
            fitted_options.append(info.get("training_options", {"seed": seed, **training_kwargs(a)}))
            E_tr, _ = T.predict(params, rung, train[:40])
            rmse = float(np.sqrt(np.mean((E_tr - np.array([s.energy for s in train[:40]])) ** 2)))
            E_te, _ = T.predict(params, rung, test); preds.append(E_te); rmses.append(rmse)
            msg = f"  {name} seed {seed}: window rmse {rmse:.2e}"
            if rung.charges and not rung.pin_q:
                mu, q = monomer_dipole(params, rung); dipoles.append(mu)
                msg += f"; monomer dipole {mu:.3f} D (truth 1.855), q_O = {q[0]:+.3f} (truth {W.Q_O:+.3f})"
            print(msg + f"; {info['time']:.0f} s", flush=True)
        P = np.array(preds); rm = np.array(rmses); conv = rm <= 2.0 * rm.min()
        if (~conv).any():
            print(f"  {name}: {int((~conv).sum())} unconverged seeds excluded (window rmse {rm[~conv]} vs best {rm.min():.2e})", flush=True)
        if conv.sum() >= 2: P = P[conv]
        mean, Wsp = ME.ensemble_stats(P)
        w_floor = 1e-14 * max(abs(s.energy) for s in train[:40])
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
                             grid=dict(R_A=R_eval.tolist(), f_true=f_true.tolist(), mean=mean.tolist(), W=np.nan_to_num(Wsp).tolist(), preds=P.tolist()))
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


def cluster_data_settings(a):
    settings = {key: getattr(a, key, default) for key, default in CLUSTER_DEFAULTS.items()}
    for key in ("train_sizes", "test_sizes"):
        settings[key] = [int(v) for v in settings[key].split(',')]
    return settings


def run_clusters(a):
    t0 = time.time()
    settings = cluster_data_settings(a)
    geometry = dict(mc_steps=settings["mc_steps"], T_K=settings["temperature_kelvin"],
                    radius_ang=settings["radius_angstrom"], min_OO=settings["min_oo_angstrom"],
                    min_any=settings["min_atom_distance_angstrom"])
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as fh:
            cached = pickle.load(fh)
        if not isinstance(cached, dict) or cached.get("dataset_config") != settings:
            raise ValueError("cluster cache has missing or different generation settings; use a new --cache path")
        train, tests = cached["train"], cached["tests"]
        print(f"data from {a.cache}", flush=True)
    else:
        train = W.cluster_dataset(settings["train_sizes"], settings["n_train"],
                                  seed=settings["train_seed"], sigma=settings["sigma"], **geometry)
        train += [W.monomer_structure()] * settings["monomer_references"]
        tests = {n: W.cluster_dataset((n,), settings["n_test"],
                                     seed=settings["test_seed"] + 100 * i, **geometry)
                 for i, n in enumerate(settings["test_sizes"])}
        if a.cache:
            with open(a.cache, "wb") as fh:
                pickle.dump(dict(dataset_config=settings, train=train, tests=tests), fh)
    pin_sources(a, train, *tests.values())
    nadd_truth = {n: np.array([s.energy - W.pair_additive_energy(bohr_to_ang(s.positions), n)
                              for s in structs]) for n, structs in tests.items()}
    train_count = len(train) - settings["monomer_references"]
    print(f"clusters: {train_count} training (sizes {settings['train_sizes']}), "
          + ", ".join(f"{len(structs)} test clusters of size {n}" for n, structs in tests.items()), flush=True)
    results = {"dataset": {**settings, "train_count": train_count,
                           "test_counts": {str(n): len(structs) for n, structs in tests.items()}},
               "reference_nonadditive": {str(n): v.tolist() for n, v in nadd_truth.items()}}
    if 6 in tests:
        results["nadd6_truth"] = nadd_truth[6].tolist()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            previous = json.load(fh)
        if previous and (not isinstance(previous, dict) or previous.get("dataset") != results["dataset"]):
            raise ValueError("output contains results for different or unknown dataset settings; use a new --out path")
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a)
        metrics = {str(n): dict(energy_rmse=[], nonadditive_rmse=[]) for n in tests}
        fitted_options = []
        for seed in training_seeds(a):
            params, info = T.fit(rung, train, seed=seed, **training_kwargs(a))
            fitted_options.append(info.get("training_options", {"seed": seed, **training_kwargs(a)}))
            for n, structs in tests.items():
                energies, _ = T.predict(params, rung, structs)
                nadd_model = []
                for st, energy in zip(structs, energies):
                    pos = bohr_to_ang(st.positions)
                    dims = [W.water_structure(np.concatenate([pos[3*i:3*i+3], pos[3*j:3*j+3]]))
                            for i in range(n) for j in range(i + 1, n)]
                    if rung.pin_mu:
                        pin_embedded(dims)
                    elif rung.pin_alpha:
                        pin_dispersion(dims)
                    pair_energies, _ = T.predict(params, rung, dims)
                    nadd_model.append(energy - np.sum(pair_energies))
                energy_rmse = float(np.sqrt(np.mean((energies - np.array([st.energy for st in structs])) ** 2)))
                nonadd_rmse = float(np.sqrt(np.mean((np.array(nadd_model) - nadd_truth[n]) ** 2)))
                metrics[str(n)]["energy_rmse"].append(energy_rmse)
                metrics[str(n)]["nonadditive_rmse"].append(nonadd_rmse)
                print(f"  {name} seed {seed}: size {n} energy rmse {energy_rmse * 627.5:.3f} kcal/mol, "
                      f"non-additive rmse {nonadd_rmse * 627.5:.3f} kcal/mol; {info['time']:.0f} s", flush=True)
        results[name] = {"clusters_by_size": metrics}
        if "6" in metrics:
            results[name].update(hexamer=metrics["6"]["energy_rmse"],
                                 nonadditive=metrics["6"]["nonadditive_rmse"])
        if "2" in metrics:
            results[name]["dimer"] = metrics["2"]["energy_rmse"]
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
    ap.add_argument("--w_far", type=float, default=1.0, help="training weight of the far-field configurations (energies and forces)")
    ap.add_argument("--cache", default="", help="pickle of the generated data sets of the part (written if absent, read if present)")
    ap.add_argument("--base_cache", default="", help="ion part: a far-part cache whose base (dimer window + clusters) is reused")
    ap.add_argument("--extra_far", type=int, default=0, help="ion part: pure far-field configurations per distance and core (d = 10, 12, 14 A) added to the training set")
    ap.add_argument("--extra_cache", default="", help="ion part: cache of the pure far-field configurations")
    ap.add_argument("--out", default="e3_water_results.json")
    add_training_arguments(ap, "water")
    a = parse_selection(ap, "water", rungs=RUNGS)
    res = {"dimer": run_dimer, "clusters": run_clusters, "far": run_far, "ion": run_ion}[a.part](a)
    _save(a, res)


if __name__ == "__main__":
    main()
