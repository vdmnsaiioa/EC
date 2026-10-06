"""
x2b_toy_regression.py -- the band-field path of the harness against the X2-B toy (truth T3, a/l_1 = 1/4):
per-atom response model with pinned charges, environment features from positions only, orders 0 and 2.
Reference (x2b-results.md, linear tier): order-0 ceiling 1.6e-3, order-2 ceiling 1.0e-4 relative to rms E_band.

    python scripts/x2b_toy_regression.py [--steps 1500] [--lbfgs 200] [--clusters 24]
"""
import argparse, time, sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sseft
from sseft import toys, model as M, train as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lbfgs", type=int, default=200)
    ap.add_argument("--clusters", type=int, default=24); ap.add_argument("--F", type=int, default=16)
    ap.add_argument("--orders", default="0,2"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="x2b_toy_regression.json")
    a = ap.parse_args()
    t0 = time.time()
    train = toys.x2b_dataset(n_clusters=a.clusters, N=(40,), seed=0)
    test = toys.x2b_dataset(n_clusters=8, N=(40,), densities=(0.18, 0.26), seed=99)
    y_te = np.array([s.energy for s in test]); y_tr = np.array([s.energy for s in train])
    print(f"{len(train)} training / {len(test)} test structures; rms E_band train {np.sqrt(np.mean(y_tr**2)):.3e}  [{time.time()-t0:.0f} s]", flush=True)
    res = {}
    for order in [int(x) for x in a.orders.split(",")]:
        rung = M.with_rung(M.RUNGS["M_S%d" % order], dispersion=False, pin_alpha=False, site_energies=False,
                           coulomb_energy=False, l1=2.0, e0={**M.RUNGS["M_S0"].e0, "r_cut": 6.0, "F": a.F})
        t = time.time()
        params, info = T.fit(rung, train, seed=a.seed, steps=a.steps, lr=3e-3, w_force=0.0, batch_size=48,
                             lbfgs_steps=a.lbfgs, verbose=True)
        E_tr = T.predict_aux(params, rung, train)["E_bf"]; E_te = T.predict_aux(params, rung, test)["E_bf"]
        rtr = np.sqrt(np.mean((E_tr - y_tr) ** 2)) / np.sqrt(np.mean(y_tr ** 2))
        rte = np.sqrt(np.mean((E_te - y_te) ** 2)) / np.sqrt(np.mean(y_te ** 2))
        print(f"order {order}: train rel rms {rtr:.3e}, test rel rms {rte:.3e}  (linear-tier ceilings 1.6e-3 / 1.0e-4)  [{time.time()-t:.0f} s]", flush=True)
        res[order] = dict(train=rtr, test=rte, time=time.time() - t)
    json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
