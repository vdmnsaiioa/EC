"""The raw CCSD(T) 4-body frames at largest O-O 7-9 A (E3-QP2's selection): the PES's 4-body, TTM3-F alone's 4-body
(its many-body induction: electrostatics and the O-O term are pairwise) and the labels -- is the PES's 42 % there its
CCSD(T) 4-body correction or TTM3-F's induction?"""
import os, sys, itertools, math
import numpy as np
REPO, ROOT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W
import e3_water as E3
K = 627.5094740631; AUANG = 0.5291772083
truth = W.qaqua_pol_truth(ROOT)
a = type("A", (), dict(heldout="4b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d = E3.prepare_heldout(a, truth)["4b"]
qa = truth.qa
def ttm_int(P):                       # TTM3-F alone's interaction energy (hartree)
    t = qa.ttm(P); return t["total"] - t["intra"]
def nbody4(f, P):
    n = len(P) // 3; terms = {}
    sub = lambda idx: np.concatenate([P[3 * i:3 * i + 3] for i in idx])
    for k in range(2, n + 1):
        for idx in itertools.combinations(range(n), k):
            e = f(sub(idx))
            for j in range(2, k):
                e -= sum(terms[frozenset(s)] for s in itertools.combinations(idx, j))
            terms[frozenset(idx)] = e
    return sum(v for s, v in terms.items() if len(s) == 4)
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
for lo, hi in ((0, 5), (5, 7), (7, 9)):
    sel = np.flatnonzero((d["roo"] >= lo) & (d["roo"] < hi))
    lab = d["label"][sel]; pes = d["pes"][sel]
    ttm = np.array([nbody4(ttm_int, np.asarray(d["pos"][f]) / AUANG) for f in sel])
    print(f"4b {lo}-{hi} A: n {len(sel)}, labels rms {rms(lab) * K:.4f}; PES rmse {rms(pes - lab) * K:.4f} "
          f"({100 * rms(pes - lab) / rms(lab):.0f} %); TTM3-F alone rmse {rms(ttm - lab) * K:.4f} ({100 * rms(ttm - lab) / rms(lab):.0f} %); "
          f"PES - TTM3-F rms {rms(pes - ttm) * K:.4f}")
    if hi == 9:
        for f, l, p, t in zip(sel, lab, pes, ttm):
            print(f"   frame {f}: largest O-O {d['roo'][f]:.2f} A, label {l * K:+.4f}, PES {p * K:+.4f}, TTM3-F {t * K:+.4f}")
