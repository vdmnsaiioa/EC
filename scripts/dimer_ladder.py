"""
dimer_ladder.py -- the ladder on dimers: train rungs on a window of separations, measure the extrapolation
with the Phase 1c protocol (ensemble spread, local exponent, closure, the mean's relative error), and read
the learned sources against the sealed targets (E2 matching).

    python scripts/dimer_ladder.py --system Ar2 --truth tt --rungs M_inf,M_6,M_A --seeds 4 --steps 1500
    python scripts/dimer_ladder.py --system Ne2,Ar2 --truth published --rungs M_6 --seeds 8      # joint fit, cross C6

Truths: `tt` is the damped Tang-Toennies potential (the synthetic truth of the programme), `published` the
ab initio potentials of sseft.data.PUBLISHED (Ar2 Jaeger 2009, Ne2 Hellmann 2008).  Window 40 log-uniform
nodes on [R-, R+] with R+ = 1.8 R- (phase2-reference-data.md), label noise sigma = 1e-9 E_h by default,
isolated atoms at zero energy included so that the dimer energy is the interaction energy.  With several
systems one model is trained on the union of the windows; the Casimir-Polder C6 of every pair of species
(including the mixed pairs never seen in training) is then compared with DOSD.
"""
import argparse, time, json, sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sseft
from sseft import data as D, model as M, train as T, measure as ME, kernels as kn
from sseft.units import bohr_to_ang, ang_to_bohr

WINDOWS = {"Ne2": (4.22, 7.59), "Ar2": (5.15, 9.26), "Kr2": (5.57, 10.02), "Xe2": (6.14, 11.05), "He2": (3.5, 6.3)}


def truth_of(system, kind):
    if kind == "tt":
        return D.tt_potential(system), D.TT[system]["Z"]
    if kind == "published":
        return D.published_potential(system), D.PUBLISHED[system]["Z"]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="Ar2", help="comma-separated; several systems -> one joint model")
    ap.add_argument("--truth", default="tt", choices=["tt", "published"])
    ap.add_argument("--rungs", default="M_inf,M_6,M_A")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lbfgs", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1e-9)
    ap.add_argument("--lA", type=float, default=4.0, help="M_A envelope in A")
    ap.add_argument("--NA", type=int, default=4)
    ap.add_argument("--rcut", type=float, default=6.0, help="E0 cutoff in A")
    ap.add_argument("--one_oscillator", action="store_true")
    ap.add_argument("--out", default="dimer_ladder_results.json")
    a = ap.parse_args()

    systems = a.system.split(",")
    rng = np.random.default_rng(0)
    train, tests = [], {}
    for sysname in systems:
        truth, Z = truth_of(sysname, a.truth)
        Rm, Rp = WINDOWS[sysname]
        R_train = D.window_design(Rm, Rp, 40); R_eval = D.eval_grid(Rp, 500.0, 61)
        train += D.dimer_structures(Z, R_train, truth, sigma=a.sigma, rng=rng) + [D.isolated_atom(Z)] * 4
        tests[sysname] = dict(Z=Z, truth=truth, R_eval=R_eval, f_true=np.array([truth(R) for R in R_eval]),
                              train=D.dimer_structures(Z, R_train, truth))
        print(f"{sysname} ({a.truth}): window [{Rm}, {Rp}] A, 40 nodes, sigma = {a.sigma:g}; "
              f"|f*| at R+ = {abs(truth(ang_to_bohr(Rp))):.2e} E_h", flush=True)
    species = sorted({tests[s]["Z"] for s in systems})
    omega, w = kn.casimir_polder_grid(8)
    # pinned sources for the dagger rungs (M_16p): q = 0, and alpha(i w) in the one-oscillator form through the
    # static polarisability and the DOSD C6 -- independent data, never the training curves
    pinned_alpha = {}
    for Z in species:
        sym = D.SYMBOL[Z]; a0 = D.ALPHA_STATIC[sym]; w0 = 4 * D.DOSD_C6[(sym, sym)] / (3 * a0 ** 2)
        pinned_alpha[Z] = a0 / (1 + (np.array(omega) / w0) ** 2)
    for st in train + [x for s in systems for x in tests[s]["train"]]:
        st.pinned = {"q": np.zeros(st.n_atoms), "alpha": np.array([pinned_alpha[int(z)] for z in st.numbers])}
    def with_pins(structs):
        for st in structs:
            st.pinned = {"q": np.zeros(st.n_atoms), "alpha": np.array([pinned_alpha[int(z)] for z in st.numbers])}
        return structs

    results = {}
    for name in a.rungs.split(","):
        rung = M.RUNGS[name]
        rung = M.with_rung(rung, lA=ang_to_bohr(a.lA), NA=a.NA, one_oscillator=a.one_oscillator,
                           e0={**rung.e0, "r_cut": ang_to_bohr(a.rcut)})
        preds = {s: [] for s in systems}; rmses = {s: [] for s in systems}
        alphas = {Z: [] for Z in species}; alphas2 = {Z: [] for Z in species}; t0 = time.time()
        for seed in range(a.seeds):
            params, info = T.fit(rung, train, seed=seed, steps=a.steps, lbfgs_steps=a.lbfgs, lr=3e-3, w_force=1.0)
            msg = []
            for s in systems:
                E_tr, _ = T.predict(params, rung, tests[s]["train"])
                rmse = np.sqrt(np.mean((E_tr - np.array([st.energy for st in tests[s]["train"]])) ** 2))
                E_te, _ = T.predict(params, rung, with_pins(D.dimer_structures(tests[s]["Z"], tests[s]["R_eval"], tests[s]["truth"])))
                preds[s].append(E_te); rmses[s].append(rmse); msg.append(f"{s} window rmse {rmse:.2e}")
            if rung.dispersion and not rung.pin_alpha:
                for Z in species:
                    aux_iso = T.predict_aux(params, rung, with_pins([D.isolated_atom(Z)]))
                    alphas[Z].append(np.array(aux_iso["alpha"][0, 0]))
                    if rung.dispersion8: alphas2[Z].append(np.array(aux_iso["alpha2"][0, 0]))
            if rung.charges and not rung.pin_q:
                qmax = max(float(np.max(np.abs(T.predict_aux(params, rung, tests[s]["train"])["q"]))) for s in systems)
                print(f"    max |q_i| over the windows: {qmax:.2e}  (homonuclear dimers: 0 by neutrality + equivalence)", flush=True)
            print(f"  {name} seed {seed}: " + "; ".join(msg) + f"; {info['time']:.0f} s", flush=True)
        results[name] = {}
        for s in systems:
            P = np.array(preds[s]); R_eval = tests[s]["R_eval"]; f_true = tests[s]["f_true"]
            # convergence rule (e1-e2-results.md section 5): seeds whose window rmse exceeds 2x the best seed's are
            # reported but excluded from the ensemble statistics
            rm = np.array(rmses[s]); conv = rm <= 2.0 * rm.min()
            if (~conv).any():
                print(f"  {name} / {s}: {int((~conv).sum())} of {len(rm)} seeds unconverged (window rmse {rm[~conv]} vs best {rm.min():.2e}); excluded", flush=True)
            if conv.sum() >= 2:
                P = P[conv]
            mean, W = ME.ensemble_stats(P)
            # a spread at the float64 floor of the interaction energy (1e-16 of the window energies) is zero:
            # the Gaussian collapse of M_A reaches it, and q / the plateau slope are undefined there
            w_floor = 1e-14 * max(abs(st.energy) for st in tests[s]["train"])
            W = np.where(W > w_floor, W, np.nan)
            q = ME.local_exponent(bohr_to_ang(R_eval), W)
            rel_err = np.abs(mean - f_true) / np.abs(f_true); relW = W / np.abs(f_true)
            print(f"  {name} / {s}: {'R (A)':>7s} {'|f*|':>9s} {'mean':>10s} {'rel err':>9s} {'spread W':>9s} {'W/|f*|':>9s} {'q':>7s}")
            rows = {}
            for Rv in (10, 12, 15, 20, 30, 50, 100, 300):
                i = int(np.argmin(np.abs(bohr_to_ang(R_eval) - Rv)))
                print(f"  {'':12s} {bohr_to_ang(R_eval[i]):7.1f} {abs(f_true[i]):9.2e} {mean[i]:+10.2e} {rel_err[i]:9.2e} "
                      f"{W[i]:9.2e} {relW[i]:9.2e} {q[i]:7.2f}")
                rows[Rv] = dict(mean=float(mean[i]), rel_err=float(rel_err[i]), W=float(W[i]), relW=float(relW[i]), q=float(q[i]))
            slope = ME.plateau_slope(bohr_to_ang(R_eval), relW, 30, 300)
            print(f"  {name} / {s}: slope of log(W/|f*|) vs log R on [30, 300] A = {slope:+.2f}  "
                  f"(branch 2: 0; branch 1: p* - p_min; branch 3: -> -inf)", flush=True)
            results[name][s] = dict(rows=rows, slope=slope, window_rmse=rm.tolist(), converged=conv.tolist(),
                                    grid=dict(R_A=bohr_to_ang(R_eval).tolist(), f_true=f_true.tolist(), mean=mean.tolist(),
                                              W=np.nan_to_num(W, nan=0.0).tolist(), q=np.nan_to_num(q, nan=0.0).tolist(),
                                              preds=P.tolist()))
        if alphas[species[0]]:
            # E2: learned alpha(i omega) of the free atom, Casimir-Polder C6 of every pair vs DOSD
            e2 = {}
            for Z in species:
                A = np.array(alphas[Z]); sym = D.SYMBOL[Z]
                print(f"  {name} E2 {sym}: alpha(i w) mean over seeds {np.array2string(A.mean(0), precision=3)}  "
                      f"spread {np.array2string(A.std(0, ddof=1) if len(A) > 1 else 0 * A[0], precision=3)}  "
                      f"[alpha_0 ref {D.ALPHA_STATIC.get(sym, float('nan')):.3f}]")
                e2[sym] = dict(alpha_mean=A.mean(0).tolist(), alpha_spread=(A.std(0, ddof=1) if len(A) > 1 else 0 * A[0]).tolist())
            for Za, Zb in itertools.combinations_with_replacement(species, 2):
                c6 = np.array([float(kn.c6_from_alpha(aa, bb, w)) for aa, bb in zip(alphas[Za], alphas[Zb])])
                key = (D.SYMBOL[Za], D.SYMBOL[Zb]); ref = D.DOSD_C6.get(key, D.DOSD_C6.get(key[::-1], float("nan")))
                tag = "seen" if Za == Zb else "UNSEEN mixed pair"
                print(f"  {name} E2 C6({key[0]}-{key[1]}): CP from learned alpha {c6.mean():.2f} +- {c6.std(ddof=1) if len(c6) > 1 else 0:.2f}  "
                      f"vs DOSD {ref:.2f}  (ratio {c6.mean() / ref:.3f}; {tag})", flush=True)
                e2[f"C6_{key[0]}{key[1]}"] = dict(cp=c6.tolist(), dosd=ref)
                if rung.dispersion8:
                    c8 = np.array([float(kn.c8_from_alpha(a1, a2, b1, b2, w)) for a1, a2, b1, b2 in
                                   zip(alphas[Za], alphas2[Za], alphas[Zb], alphas2[Zb])])
                    ref8 = D.DOSD_C8.get(key, D.DOSD_C8.get(key[::-1], float("nan")))
                    print(f"  {name} E2 C8({key[0]}-{key[1]}): CP from learned alpha1, alpha2 {c8.mean():.1f} +- {c8.std(ddof=1) if len(c8) > 1 else 0:.1f}  "
                          f"vs DOSD {ref8:.1f}  (ratio {c8.mean() / ref8:.3f}; {tag})", flush=True)
                    e2[f"C8_{key[0]}{key[1]}"] = dict(cp=c8.tolist(), dosd=ref8)
            results[name]["E2"] = e2
        print(f"  {name}: {time.time() - t0:.0f} s total", flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
