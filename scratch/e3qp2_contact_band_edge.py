"""TTM3-F's smeared charge-charge electrostatics against the pinned Coulomb band, sum q_i q_j erf(r/l1)/r, for several
band edges l1: the contact mismatch E_0 must absorb in the *pm rungs, on the raw 2-body frames below 5 A."""
import os, sys, math
import numpy as np
from scipy.special import erf
REPO, ROOT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, qaqua as QA
import e3_water as E3
K = 627.5094740631; AUANG = QA.AUANG; KC = QA.TTM3F_CHARGECON ** 2 / QA.AUKCAL
truth = W.qaqua_pol_truth(ROOT); qa = truth.qa
a = type("A", (), dict(heldout="2b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d = E3.prepare_heldout(a, truth)["2b"]
sel = np.flatnonzero(d["roo"] < 5.0)
EDGES = (0.75, 1.0, 1.25, 1.5, 2.0)
smeared, bands, ccsd = [], [], []
for f in sel:
    P = np.asarray(d["pos"][f]) / AUANG; t = qa.ttm(P)
    mono = [qa.ttm(P[3 * i:3 * i + 3]) for i in range(2)]
    smeared.append(t["elec"] - sum(m["elec"] for m in mono))
    s = [[P[3 * i + 1] * AUANG, P[3 * i + 2] * AUANG, t["msites"][i] * AUANG] for i in range(2)]
    r = np.array([[np.linalg.norm(s[0][x] - s[1][y]) for y in range(3)] for x in range(3)])
    qq = np.outer(t["site_charges"][0], t["site_charges"][1])
    bands.append([KC * np.sum(qq * erf(r / l) / r) for l in EDGES])
smeared = np.array(smeared); bands = np.array(bands); roo = d["roo"][sel]
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
print("raw 2b below 5 A (n %d): smeared electrostatics - pinned Coulomb band, rms (mean) in kcal/mol" % len(sel))
for k, l in enumerate(EDGES):
    D = smeared - bands[:, k]
    parts = []
    for lo, hi in ((2.4, 2.7), (2.7, 3.0), (3.0, 3.5), (3.5, 5.0)):
        m = (roo >= lo) & (roo < hi)
        parts.append(f"{lo}-{hi} A {rms(D[m]) * K:.3f} ({np.mean(D[m]) * K:+.3f})")
    print(f"  l1 = {l:4.2f} A: all {rms(D) * K:.3f} ({np.mean(D) * K:+.3f});  " + ";  ".join(parts))
