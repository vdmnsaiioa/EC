"""
dimer_ladder.py -- the ladder on a dimer: train rungs on a window of separations, measure the extrapolation
with the Phase 1c protocol (ensemble spread, local exponent, closure, the mean's relative error).

    python scripts/dimer_ladder.py --system Ar2 --rungs M_inf,M_6,M_A --seeds 4 --steps 1500

Default truth: the damped Tang-Toennies potential (the synthetic truth of the programme); `--truth` can name a
published potential once those are added to sseft.data.  Window 40 log-uniform nodes on [R-, R+] with
R+ = 1.8 R- (phase2-reference-data.md), label noise sigma = 1e-9 E_h by default, isolated atoms at zero energy
included so that the dimer energy is the interaction energy.
"""
import argparse, time, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sseft
from sseft import data as D, model as M, train as T, measure as ME
from sseft.units import bohr_to_ang, ang_to_bohr

WINDOWS = {"Ne2": (4.22, 7.59), "Ar2": (5.15, 9.26), "Kr2": (5.57, 10.02), "Xe2": (6.14, 11.05), "He2": (3.5, 6.3)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="Ar2")
    ap.add_argument("--rungs", default="M_inf,M_6,M_A")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lbfgs", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1e-9)
    ap.add_argument("--lA", type=float, default=4.0, help="M_A envelope in A")
    ap.add_argument("--NA", type=int, default=4)
    ap.add_argument("--rcut", type=float, default=6.0, help="E0 cutoff in A")
    ap.add_argument("--out", default="dimer_ladder_results.json")
    a = ap.parse_args()

    truth = D.tt_potential(a.system); Z = D.TT[a.system]["Z"]
    Rm, Rp = WINDOWS[a.system]
    R_train = D.window_design(Rm, Rp, 40)
    R_eval = D.eval_grid(Rp, 500.0, 61)
    rng = np.random.default_rng(0)
    train = D.dimer_structures(Z, R_train, truth, sigma=a.sigma, rng=rng) + [D.isolated_atom(Z)] * 4
    test = D.dimer_structures(Z, R_eval, truth)
    f_true = np.array([truth(R) for R in R_eval])
    print(f"{a.system}: window [{Rm}, {Rp}] A, 40 nodes, sigma = {a.sigma:g}; |f*| at R+ = {abs(truth(ang_to_bohr(Rp))):.2e} E_h", flush=True)
    results = {}
    for name in a.rungs.split(","):
        rung = M.RUNGS[name]
        rung = M.with_rung(rung, lA=ang_to_bohr(a.lA), NA=a.NA, e0={**rung.e0, "r_cut": ang_to_bohr(a.rcut)})
        preds = []; t0 = time.time()
        for seed in range(a.seeds):
            params, info = T.fit(rung, train, seed=seed, steps=a.steps, lbfgs_steps=a.lbfgs, lr=3e-3, w_force=1.0)
            E_tr, _ = T.predict(params, rung, train[:40])
            rmse = np.sqrt(np.mean((E_tr - np.array([s.energy for s in train[:40]])) ** 2))
            E_te, _ = T.predict(params, rung, test)
            preds.append(E_te)
            print(f"  {name} seed {seed}: window rmse {rmse:.2e} E_h (rmse/sigma {rmse / max(a.sigma, 1e-30):.1f}); "
                  f"{info['time']:.0f} s", flush=True)
        preds = np.array(preds)
        mean, W = ME.ensemble_stats(preds)
        q = ME.local_exponent(bohr_to_ang(R_eval), W)
        rel_err = np.abs(mean - f_true) / np.abs(f_true)
        relW = W / np.abs(f_true)
        print(f"  {name}: {'R (A)':>7s} {'|f*|':>9s} {'mean':>10s} {'rel err':>9s} {'spread W':>9s} {'W/|f*|':>9s} {'q':>7s}")
        rows = {}
        for Rv in (10, 12, 15, 20, 30, 50, 100, 300):
            i = int(np.argmin(np.abs(bohr_to_ang(R_eval) - Rv)))
            print(f"  {'':6s} {bohr_to_ang(R_eval[i]):7.1f} {abs(f_true[i]):9.2e} {mean[i]:+10.2e} {rel_err[i]:9.2e} {W[i]:9.2e} {relW[i]:9.2e} {q[i]:7.2f}")
            rows[Rv] = dict(mean=float(mean[i]), rel_err=float(rel_err[i]), W=float(W[i]), relW=float(relW[i]), q=float(q[i]))
        slope = ME.plateau_slope(bohr_to_ang(R_eval), relW, 30, 300)
        print(f"  {name}: slope of log(W/|f*|) vs log R on [30, 300] A = {slope:+.2f}  (branch 2 predicts 0; branch 1 p* - p_min; branch 3 -> -inf)  [{time.time() - t0:.0f} s]", flush=True)
        results[name] = dict(rows=rows, slope=slope)
    json.dump(results, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
