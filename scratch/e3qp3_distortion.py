"""How much of the raw sets' interaction energy comes from the monomers' small distortions (the frames are
near-rigid: O-H within 0.03 A, HOH within 5 deg of the training monomer): the truth at each frame against the truth at
the same frame with every monomer replaced by the rigid training monomer, best aligned (Kabsch, unweighted).

    python scratch/e3qp3_distortion.py [external/qaqua/build_pol]   # ~1 min on a CPU
"""
import os, sys, math
import numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W
import e3_water as E3
K = 627.5094740631
truth = W.qaqua_pol_truth(ROOT)
ref = W.monomer_positions()                                    # the rigid training monomer (A), O H H
def rigidify(P):
    out = []
    for m in range(len(P) // 3):
        X = P[3 * m:3 * m + 3]
        cx, cr = X.mean(0), ref.mean(0)
        H = (ref - cr).T @ (X - cx); U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T)); D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        out.append((ref - cr) @ R.T + cx)
    return np.concatenate(out)
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
for key, n in (("2b", 2), ("3b", 3)):
    a = type("A", (), dict(heldout=key, heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                           heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
    d = E3.prepare_heldout(a, truth)[key]
    P = np.asarray(d["pos"]); roo = d["roo"]; strict = d["strict"]
    if n == 2:
        e_act = np.array([truth.energy(p, 2) for p in P]); e_rig = np.array([truth.energy(rigidify(p), 2) for p in P])
    else:                                                       # the 3-body term of each, by the subset expansion
        def three(p):
            E = truth.energy(p, 3)
            for i, j in ((0, 1), (0, 2), (1, 2)):
                E -= truth.energy(np.concatenate([p[3 * i:3 * i + 3], p[3 * j:3 * j + 3]]), 2)
            return E
        sel = np.flatnonzero(roo < 7.0)
        P, roo, strict = P[sel], roo[sel], strict[sel]
        e_act = np.array([three(p) for p in P]); e_rig = np.array([three(rigidify(p)) for p in P])
    D = (e_act - e_rig) * K
    print(f"{key}: {len(P)} frames; truth(actual) - truth(rigid monomers): rms {rms(D):.4f} kcal/mol, strict {rms(D[strict]):.4f} "
          f"({strict.sum()}), the rest {rms(D[~strict]):.4f}; by largest O-O " + ", ".join(
              f"{lo}-{hi} A {rms(D[(roo >= lo) & (roo < hi)]):.4f} (n {((roo >= lo) & (roo < hi)).sum()})"
              for lo, hi in ((0, 3.2), (3.2, 5), (5, 7), (7, 99)) if ((roo >= lo) & (roo < hi)).any()), flush=True)
