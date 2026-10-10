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
import argparse, time, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sseft
from sseft import water as W, model as M, train as T, measure as ME, data as D
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
}


def configure(rung, a):
    kw = dict(lA=ang_to_bohr(a.lA), e0={**rung.e0, "r_cut": ang_to_bohr(a.rcut), "F": a.F, "n_rbf_readout": 8})
    if rung.band_fields >= 0:
        kw["bf_rs"] = rung.l1 * a.rs_over_l1
    return M.with_rung(rung, **kw)


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
    old.update(results)
    json.dump(old, open(a.out, "w"), indent=1)


def run_dimer(a):
    R_train = bohr_to_ang(D.window_design(*WINDOW, 40))
    R_eval = bohr_to_ang(D.eval_grid(WINDOW[1], 500.0, 61))
    rng = np.random.default_rng(0)
    train = W.dimer_scan(R_train, sigma=a.sigma, rng=rng) + [W.monomer_structure()] * 4
    test = W.dimer_scan(R_eval)
    f_true = np.array([s.energy for s in test])
    print(f"water dimer (PW truth, phi = 0): window [{WINDOW[0]:.2f}, {WINDOW[1]:.2f}] A, 40 nodes, sigma = {a.sigma:g}; "
          f"|f*| at R+ = {abs(test[0].energy):.2e} E_h; E R^3 at 300 A = {f_true[-1] * ang_to_bohr(R_eval[-1]) ** 3:+.4f} (dipole-dipole -0.652)", flush=True)
    results = {}
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a)
        preds, rmses, dipoles = [], [], []; t0 = time.time()
        for seed in range(a.seeds):
            params, info = T.fit(rung, train, seed=seed, steps=a.steps, lbfgs_steps=a.lbfgs, lr=3e-3, w_force=1.0)
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
        print(f"  {name}: slope of log(W/|f*|) vs log R on [30, 300] A = {slope:+.2f}  (branch 2: 0; branch 1: p* - p_min = 3 - 1 = 2 for a monopole leak; branch 3: -> -inf)"
              + (f";  monomer dipole {np.mean(dipoles):.3f} +- {np.std(dipoles, ddof=1) if len(dipoles) > 1 else 0:.3f} D" if dipoles else "") + f"  [{time.time() - t0:.0f} s]", flush=True)
        results[name] = dict(rows=rows, slope=slope, window_rmse=rm.tolist(), converged=conv.tolist(), dipoles=dipoles,
                             grid=dict(R_A=R_eval.tolist(), f_true=f_true.tolist(), mean=mean.tolist(), W=np.nan_to_num(Wsp).tolist(), preds=P.tolist()))
        _save(a, results)
    return results


def run_far(a):
    """the far-field induction test: training = the dimer window (its nodes beyond r_c teach the response to the band
    field) + the clusters; test = a probe monomer at 6-10 A from a trimer.  The probe's interaction energy minus its
    electrostatics with the pinned charges is the induction; v1 cannot represent it beyond r_c, v1.5 can."""
    t0 = time.time()
    R_train = bohr_to_ang(D.window_design(*WINDOW, 40)); rng = np.random.default_rng(0)
    train = (W.dimer_scan(R_train, sigma=a.sigma, rng=rng) + W.cluster_dataset((3, 4, 5), a.n_train, seed=0, sigma=a.sigma, mc_steps=a.mc_steps)
             + [W.monomer_structure()] * 4)
    dists = (6.0, 7.0, 8.0, 10.0)
    if a.train_far > 0:
        # far-field configurations in the training set (another core, other seeds): the only data in which the
        # response to the band field is resolvable above the fit's floor
        far_train, _ = W.probe_configurations(3, dists, a.train_far, seed=400, mc_steps=a.mc_steps)
        train = train + far_train
    test, E_core = W.probe_configurations(3, dists, a.n_test, seed=300, mc_steps=a.mc_steps)
    y = np.array([s.energy for s in test]) - E_core                                   # the probe's interaction with the core
    # the probe's induction: total minus the pure electrostatics of the point charges (smeared as in the truth) --
    # computed from the truth's components
    es = []
    for s in test:
        pos = bohr_to_ang(s.positions); c_all = W.pw_components(pos, 4); c_core = W.pw_components(pos[:9], 3)
        es.append((c_all["es"] - c_core["es"]) + (c_all["rep"] - c_core["rep"]) + (c_all["disp"] - c_core["disp"]))
    ind = y - np.array(es)
    print(f"far-field induction test: {len(test)} probe configurations at d = {dists} A from a trimer; probe interaction rms "
          f"{np.sqrt(np.mean(y ** 2)) * 627.5:.4f} kcal/mol, of which induction rms {np.sqrt(np.mean(ind ** 2)) * 627.5:.4f} "
          f"({np.sqrt(np.mean(ind ** 2)) / np.sqrt(np.mean(y ** 2)) * 100:.1f} %)  [{time.time() - t0:.0f} s]", flush=True)
    results = {"d": [s.info["d"] for s in test], "probe_interaction": y.tolist(), "induction": ind.tolist()}
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a); t1 = time.time(); errs = []; errs_ind = []
        for seed in range(a.seeds):
            params, info = T.fit(rung, train, seed=seed, steps=a.steps, lbfgs_steps=a.lbfgs, lr=3e-3, w_force=1.0, batch_size=a.batch, lbfgs_chunk=a.lbfgs_chunk)
            Et, _ = T.predict(params, rung, test)
            cores = [W.water_structure(bohr_to_ang(s.positions)[:9]) for s in test]
            Ec, _ = T.predict(params, rung, cores)
            yp = Et - Ec
            r = np.sqrt(np.mean((yp - y) ** 2)); errs.append(r)
            print(f"  {name} seed {seed}: probe-interaction rmse {r * 627.5:.4f} kcal/mol = {r / np.sqrt(np.mean(ind ** 2)) * 100:.0f} % of the induction "
                  f"(per distance: " + ", ".join(f"{d:.0f} A {np.sqrt(np.mean((yp - y)[np.array(results['d']) == d] ** 2)) / np.sqrt(np.mean(ind[np.array(results['d']) == d] ** 2)) * 100:.0f} %" for d in dists) + f"); {info['time']:.0f} s", flush=True)
        results[name] = dict(rmse=errs)
        _save(a, results)
    return results


def run_clusters(a):
    t0 = time.time()
    train = W.cluster_dataset((3, 4, 5), a.n_train, seed=0, sigma=a.sigma, mc_steps=a.mc_steps) + [W.monomer_structure()] * 4
    test6 = W.cluster_dataset((6,), a.n_test, seed=100, mc_steps=a.mc_steps)
    test2 = W.cluster_dataset((2,), a.n_test, seed=200, mc_steps=a.mc_steps)
    y_tr = np.array([s.energy for s in train if s.n_atoms > 3])
    nadd6 = np.array([s.energy - W.pair_additive_energy(bohr_to_ang(s.positions), 6) for s in test6])
    print(f"clusters: {len(train) - 4} training (n = 3, 4, 5), {len(test6)} hexamers and {len(test2)} dimers held out; "
          f"rms E_int train {np.sqrt(np.mean(y_tr ** 2)) * 627.5:.2f} kcal/mol; hexamer non-additive energy rms {np.sqrt(np.mean(nadd6 ** 2)) * 627.5:.3f} kcal/mol "
          f"({np.sqrt(np.mean(nadd6 ** 2)) / np.sqrt(np.mean(np.array([s.energy for s in test6]) ** 2)) * 100:.1f} % of the interaction energy)  [{time.time() - t0:.0f} s]", flush=True)
    results = {"nadd6_truth": nadd6.tolist()}
    for name in a.rungs.split(","):
        rung = configure(RUNGS[name], a); t1 = time.time()
        errs6, errs2, errsN = [], [], []
        for seed in range(a.seeds):
            params, info = T.fit(rung, train, seed=seed, steps=a.steps, lbfgs_steps=a.lbfgs, lr=3e-3, w_force=1.0, batch_size=a.batch, lbfgs_chunk=a.lbfgs_chunk)
            E6, _ = T.predict(params, rung, test6); E2, _ = T.predict(params, rung, test2)
            # the model's own non-additive energy: E(hexamer) - sum over its 15 dimers
            nadd_model = []
            for s, e in zip(test6, E6):
                pos = bohr_to_ang(s.positions); dims = []
                for i in range(6):
                    for j in range(i + 1, 6):
                        dims.append(W.water_structure(np.concatenate([pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]])))
                Ed, _ = T.predict(params, rung, dims)
                nadd_model.append(e - np.sum(Ed))
            nadd_model = np.array(nadd_model)
            r6 = np.sqrt(np.mean((E6 - np.array([s.energy for s in test6])) ** 2)); r2 = np.sqrt(np.mean((E2 - np.array([s.energy for s in test2])) ** 2))
            rN = np.sqrt(np.mean((nadd_model - nadd6) ** 2))
            errs6.append(r6); errs2.append(r2); errsN.append(rN)
            print(f"  {name} seed {seed}: hexamer rmse {r6 * 627.5:.3f} kcal/mol, dimer rmse {r2 * 627.5:.3f}, non-additive rmse {rN * 627.5:.3f} "
                  f"({rN / np.sqrt(np.mean(nadd6 ** 2)) * 100:.0f} % of the non-additive energy); {info['time']:.0f} s", flush=True)
        results[name] = dict(hexamer=errs6, dimer=errs2, nonadditive=errsN)
        _save(a, results)
        print(f"  {name}: hexamer {np.mean(errs6) * 627.5:.3f}, dimer {np.mean(errs2) * 627.5:.3f}, non-additive {np.mean(errsN) * 627.5:.3f} kcal/mol (means over seeds)  [{time.time() - t1:.0f} s]", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="dimer", choices=["dimer", "clusters", "far"])
    ap.add_argument("--rungs", default="M_inf,M_G,M_A,M_1,M_1mu,M_1p")
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lbfgs", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=1e-9); ap.add_argument("--rcut", type=float, default=6.0); ap.add_argument("--lA", type=float, default=4.0)
    ap.add_argument("--F", type=int, default=16); ap.add_argument("--rs_over_l1", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=12); ap.add_argument("--n_test", type=int, default=8); ap.add_argument("--mc_steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=12); ap.add_argument("--lbfgs_chunk", type=int, default=6)
    ap.add_argument("--train_far", type=int, default=0, help="far-field probe configurations per distance added to the training set")
    ap.add_argument("--out", default="e3_water_results.json")
    a = ap.parse_args()
    res = {"dimer": run_dimer, "clusters": run_clusters, "far": run_far}[a.part](a)
    _save(a, res)


if __name__ == "__main__":
    main()
