"""
E3-QP3's numbers before training (notes/e3qp3-preregistration.md section 3a): the contact configurations of
configs/water_qaqua_pol3.yaml against the raw CCSD(T) dimers -- how many pairs at a compressed contact, their energies,
and how far the raw dimers lie from the training pairs with and without them (9 intermolecular atom distances, rms,
over the 8 symmetric orderings; a raw dimer counts as beyond the training data when its nearest training pair is
farther than 95 % of the held-out test dimers' are).  The far field (section 3b) is scratch/e3qp2_matched_response.py
with the E3-QP3 rungs:

    python scratch/e3qp3_preregistration.py [external/qaqua/build_pol]                                  # ~3 min, one CPU
    python scratch/e3qp2_matched_response.py external/qaqua/build_pol M_16pm_e1,M_16C2pm_e1 --edge=0.75   # ~10 min
"""
import itertools, math, os, sys, time
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scratch"))
from sseft import water as W
from sseft.units import bohr_to_ang
import e3_water as E3
from e3qp3_shortrange_diagnostic import pair_desc

K = 627.5094740631
t0 = time.time()
truth = W.qaqua_pol_truth(ROOT)
geometry = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
clusters = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, **geometry)      # E3-QP2's, bit for bit
dimers = W.cluster_dataset((2,), 24, seed=200, truth=truth, **geometry)                         # the held-out test dimers
contacts = (W.contact_configurations(2, 600, (2.3, 3.0), seed=700, truth=truth, max_pair_kcal=2.0, min_any=1.5)
            + W.contact_configurations(3, 200, (2.3, 3.0), seed=701, truth=truth, max_pair_kcal=2.0, min_any=1.5))
print(f"{len(clusters)} training clusters, {len(contacts)} contact configurations  [{time.time() - t0:.0f} s]", flush=True)


def pairs(structs):
    out, oo = [], []
    for st in structs:
        p = bohr_to_ang(np.asarray(st.positions))
        for i, j in itertools.combinations(range(len(p) // 3), 2):
            out.append(pair_desc(p[3 * i:3 * i + 3], p[3 * j:3 * j + 3])[0]); oo.append(np.linalg.norm(p[3 * i] - p[3 * j]))
    return np.array(out), np.array(oo)


tp_cl, oo_cl = pairs(clusters)
tp_co, oo_co = pairs(contacts)
E_co = np.array([s.energy for s in contacts]) * K
for label, oo in (("training clusters", oo_cl), ("contact configurations", oo_co)):
    print(f"{label}: {len(oo)} pairs; O-O below 2.7 A {int((oo < 2.7).sum())}, below 2.5 A {int((oo < 2.5).sum())}; "
          f"O-O quantiles 1 / 10 / 50 %: " + " / ".join(f"{q:.2f}" for q in np.quantile(oo, [0.01, 0.1, 0.5])), flush=True)
for n_mol in (2, 3):
    e = np.array([s.energy for s in contacts if s.n_atoms == 3 * n_mol]) * K
    print(f"  contact {'dimers' if n_mol == 2 else 'trimers'}: interaction energies {e.min():+.2f} .. {e.max():+.2f} kcal/mol "
          f"(mean {e.mean():+.2f}, rms {math.sqrt(np.mean(e ** 2)):.2f})", flush=True)

a = type("A", (), dict(heldout="2b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d = E3.prepare_heldout(a, truth)["2b"]
P, oo = np.asarray(d["pos"]), d["soo"]
D = np.array([pair_desc(p[:3], p[3:]) for p in P])                                  # (frames, 8, 9)


def nearest(tp, desc):
    out = np.empty(len(desc))
    for i0 in range(0, len(desc), 200):
        blk = desc[i0:i0 + 200]
        out[i0:i0 + 200] = np.sqrt(np.mean((blk[:, :, None, :] - tp[None, None]) ** 2, axis=-1)).min(axis=(1, 2))
    return out


for label, tp in (("clusters (E3-QP2)", tp_cl), ("clusters + contacts (E3-QP3)", np.concatenate([tp_cl, tp_co]))):
    ref = np.quantile(nearest(tp, np.array([pair_desc(*np.split(bohr_to_ang(np.asarray(s.positions)), 2)) for s in dimers])), 0.95)
    nn = nearest(tp, D)
    bins = ((2.3, 2.7), (2.7, 2.9), (2.9, 3.2), (3.2, 5.0))
    print(f"raw dimers beyond the training pairs ({label}; the test dimers' 95 % distance {ref:.3f} A): below 5 A "
          f"{100 * np.mean(nn[oo < 5] > ref):.0f} %; " + ", ".join(
              f"O-O {lo}-{hi} A {100 * np.mean(nn[(oo >= lo) & (oo < hi)] > ref):.0f} % of {int(((oo >= lo) & (oo < hi)).sum())}"
              for lo, hi in bins), flush=True)
print(f"done in {time.time() - t0:.0f} s")
