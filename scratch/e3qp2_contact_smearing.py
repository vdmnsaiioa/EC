"""How far TTM3-F's Thole-smeared charge-charge electrostatics is from the bare Coulomb energy of the same site charges
(what the *pm rungs' pinned bands carry at contact: the band smoothing at l1 = 0.75 A is ~bare beyond 1.5 A), on the raw
CCSD(T) 2-body frames E3-QP2 reads -- against the CCSD(T) 2-body correction (PES - TTM3-F) and the labels."""
import os, sys, math
import numpy as np
REPO, ROOT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, qaqua as QA
import e3_water as E3
K = 627.5094740631; AUANG = QA.AUANG; KC = QA.TTM3F_CHARGECON ** 2 / QA.AUKCAL      # hartree * A / e^2
truth = W.qaqua_pol_truth(ROOT); qa = truth.qa
def pieces(P_ang):
    P = np.asarray(P_ang) / AUANG; t = qa.ttm(P); n = len(P) // 3
    sites = [[P[3 * i + 1] * AUANG, P[3 * i + 2] * AUANG, t["msites"][i] * AUANG] for i in range(n)]
    bare = sum(KC * t["site_charges"][i, a] * t["site_charges"][j, b] / np.linalg.norm(sites[i][a] - sites[j][b])
               for i in range(n) for j in range(i + 1, n) for a in range(3) for b in range(3))
    tm = [qa.ttm(np.asarray(P[3 * i:3 * i + 3])) for i in range(n)]
    ind = t["ind"] - sum(x["ind"] for x in tm)
    return t["elec"] - sum(x["elec"] for x in tm), bare, ind, t["total"] - t["intra"]
# units check: a dimer pulled to 10 A
a = type("A", (), dict(heldout="2b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d = E3.prepare_heldout(a, truth)["2b"]
P0 = np.array(d["pos"][0]); far = P0.copy(); far[3:] += np.array([10.0, 0, 0])
e_s, e_b, _, _ = pieces(far)
print(f"units check at +10 A: smeared {e_s * K:+.6f}, bare {e_b * K:+.6f} kcal/mol")
sel = np.flatnonzero(d["roo"] < 5.0)
rows = np.array([pieces(d["pos"][f]) for f in sel])
smeared, bare, ind, ttm = rows.T
lab, pes = d["label"][sel], d["pes"][sel]
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
D = smeared - bare
print(f"raw 2b, largest O-O < 5 A: {len(sel)} frames; labels rms {rms(lab) * K:.3f} kcal/mol")
print(f"  TTM3-F smeared electrostatics - bare Coulomb of the same charges: rms {rms(D) * K:.3f}, mean {np.mean(D) * K:+.3f}, "
      f"range [{D.min() * K:+.3f}, {D.max() * K:+.3f}] kcal/mol")
print(f"  TTM3-F's 2-body induction: rms {rms(ind) * K:.3f}; the CCSD(T) 2-body correction (PES - TTM3-F): rms {rms(pes - ttm) * K:.3f}")
roo = d["roo"][sel]
for lo, hi in ((2.4, 2.7), (2.7, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 5.0)):
    s = (roo >= lo) & (roo < hi)
    if s.any():
        print(f"  O-O {lo}-{hi} A: n {s.sum()}, smeared - bare rms {rms(D[s]) * K:.3f} (mean {np.mean(D[s]) * K:+.3f}), "
              f"CCSD(T) correction rms {rms((pes - ttm)[s]) * K:.3f}, labels rms {rms(lab[s]) * K:.3f}")
