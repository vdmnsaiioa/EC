"""
E3-QP's held-out 2-body readings of the M_16 rungs, read again with their pins (e3qp-results.md section 3.1): the saved
models of results/e3qp, evaluated on the same raw CCSD(T) 2-body frames with the dispersion band they were trained with.

    python scratch/e3qp_reread_2b.py [results/e3qp] [external/qaqua/build_pol]

Seconds per model; the 3- and 4-body readings need no re-reading (the band is pairwise).
"""
import os, sys, pickle
import numpy as np
import jax.numpy as jnp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W
import e3_water as E3

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/e3qp"
root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "external", "qaqua", "build_pol")
K = 627.5094740631
truth = W.qaqua_pol_truth(root)
# configs/water_qaqua_pol.yaml's clusters part: the architecture and the held-out selection
a = type("A", (), dict(lA=4.0, rcut=6.0, F=16, l1=0.75, rs_over_l1=0.0, heldout="2b", heldout_oh_tol=0.03,
                       heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02, heldout_strict_angle_tol=3.0,
                       heldout_max_pair_kcal=2.0, heldout_max=0, rungs="M_16p,M_16S0p"))()
d = E3.prepare_heldout(a, truth)["2b"]
print(f"held-out 2b: {len(d['label'])} frames, labels rms {np.sqrt(np.mean(d['label'] ** 2)) * K:.4f} kcal/mol")
for name in ("M_16p", "M_16S0p"):
    rung = E3.configure(E3.RUNGS[name], a)
    for seed in range(4):
        path = os.path.join(out_dir, f"clusters_{name}_seed{seed}.params.pkl")
        if not os.path.exists(path):
            print(f"  {name} seed {seed}: no {path}")
            continue
        params = pickle.load(open(path, "rb"))
        params = __import__("jax").tree_util.tree_map(jnp.asarray, params)
        before = E3.heldout_metrics(E3.model_nbody(params, rung, d["pos"], d["charges"], 2), d)
        after = E3.heldout_metrics(E3.model_nbody(params, rung, d["pos"], d["charges"], 2, truth), d)
        print(f"  {name} seed {seed}: rmse {before['rmse_model'] * K:.4f} kcal/mol as read in E3-QP (no band) -> "
              f"{after['rmse_model'] * K:.4f} with the band; by largest O-O "
              + ", ".join(f"{b} A {v['rmse_model'] * K:.4f}" for b, v in after["by_largest_oo_angstrom"].items() if v["count"]))
