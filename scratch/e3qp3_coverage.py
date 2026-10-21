"""Where E3-QP2's short-range readings come from, on the data side (no model needed):
(1) how the raw CCSD(T) 2- and 3-body frames sit against the pairs and triples of the 300 training clusters
    (O-O distances, pair energies, a nearest-neighbour distance in the dimer's 9 intermolecular atom distances);
(2) how much of the training forces is rigid-body (net force and torque per molecule -- all that the energy of rigid
    monomers depends on) and how much internal (stretch and bend of the monomers: the dipole surface's charge flux and
    the PIP corrections' monomer dependence, which no reading tests).
The training clusters are regenerated exactly as configs/water_qaqua_pol2.yaml makes them.

    python scratch/e3qp3_coverage.py [external/qaqua/build_pol] [scratch]   # ~2 min on a CPU
"""
import os, sys, time, json, math, itertools, pickle
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "scratch")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W
import e3_water as E3
K = 627.5094740631; BOHR = 0.529177210903
truth = W.qaqua_pol_truth(ROOT)
geometry = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
cache = os.path.join(OUT, "e3qp3_coverage_data.pkl")
t0 = time.time()
if os.path.exists(cache):
    train, dimers = pickle.load(open(cache, "rb"))
else:
    train = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, **geometry)
    dimers = W.cluster_dataset((2,), 24, seed=200, truth=truth, **geometry)      # the held-out dimers (test_seed + 100)
    pickle.dump((train, dimers), open(cache, "wb"))
print(f"data: {len(train)} training clusters, {len(dimers)} test dimers [{time.time() - t0:.0f} s]", flush=True)

ang = lambda st: np.asarray(st.positions) * BOHR
def pairs_of(pos):
    n = len(pos) // 3
    return [(pos[3 * i:3 * i + 3], pos[3 * j:3 * j + 3]) for i, j in itertools.combinations(range(n), 2)]
def desc(A, B):
    """the 9 intermolecular atom distances, in its 8 symmetric orderings (H1<->H2 in each molecule, A<->B)."""
    out = []
    for a in ((0, 1, 2), (0, 2, 1)):
        for b in ((0, 1, 2), (0, 2, 1)):
            d = np.linalg.norm(A[list(a)][:, None] - B[list(b)][None], axis=-1)
            out.append(d.reshape(-1)); out.append(d.T.reshape(-1))
    return np.array(out)                                                         # (8, 9)

# ---- (1) pairs
tp = [p for st in train for p in pairs_of(ang(st))]
tp_oo = np.array([np.linalg.norm(A[0] - B[0]) for A, B in tp])
tp_E = np.array([truth.energy(np.concatenate([A, B]), 2) for A, B in tp]) * K
tp_desc = np.array([desc(A, B)[0] for A, B in tp])                               # (P, 9)
print(f"training pairs: {len(tp)}; O-O {tp_oo.min():.2f}-{tp_oo.max():.2f} A (quantiles 1/5/50 %: "
      + " / ".join(f"{q:.2f}" for q in np.quantile(tp_oo, [0.01, 0.05, 0.5])) + "); pair energy "
      f"{tp_E.min():+.2f} to {tp_E.max():+.2f} kcal/mol (99 % below {np.quantile(tp_E, 0.99):+.2f}); "
      f"pairs within 3.2 A: {(tp_oo < 3.2).sum()}", flush=True)

a = type("A", (), dict(heldout="2b,3b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
H = E3.prepare_heldout(a, truth)
d2 = H["2b"]; P2 = np.asarray(d2["pos"])
raw_oo = np.array([np.linalg.norm(p[0] - p[3]) for p in P2])
raw_E = d2["label"] * K
def nn(A, B):
    D = desc(A, B)                                                               # (8, 9)
    return float(np.min(np.sqrt(np.mean((D[:, None, :] - tp_desc[None]) ** 2, axis=-1))))
raw_nn = np.array([nn(p[:3], p[3:]) for p in P2])
dim_nn = np.array([nn(*pairs_of(ang(st))[0]) for st in dimers])
ref = np.quantile(dim_nn, 0.95)
print(f"test dimers (in distribution): nearest training pair at rms {np.median(dim_nn):.3f} A (median), 95 % below "
      f"{ref:.3f} A", flush=True)
print("raw 2-body frames by O-O: n | below the training O-O minimum | nearest training pair rms (median) | beyond the "
      "test dimers' 95 % | pair energy above the training pairs' 99 %")
hi = np.quantile(tp_E, 0.99)
rows = {}
for lo, up in ((2.3, 2.7), (2.7, 2.9), (2.9, 3.2), (3.2, 3.6), (3.6, 4.2), (4.2, 5.0), (5.0, 9.0)):
    m = (raw_oo >= lo) & (raw_oo < up)
    if not m.any():
        continue
    rows[f"{lo}-{up}"] = dict(n=int(m.sum()), below=int((raw_oo[m] < tp_oo.min()).sum()), nn=float(np.median(raw_nn[m])),
                              out=float(np.mean(raw_nn[m] > ref)), hiE=float(np.mean(raw_E[m] > hi)),
                              label_rms=float(np.sqrt(np.mean(raw_E[m] ** 2))))
    r = rows[f"{lo}-{up}"]
    print(f"  {lo}-{up} A: {r['n']:5d} | {r['below']:4d} | {r['nn']:.3f} | {100 * r['out']:3.0f} % | {100 * r['hiE']:3.0f} %   "
          f"(labels rms {r['label_rms']:.2f} kcal/mol)", flush=True)
allm = raw_oo < 5.0
print(f"  all below 5 A: {allm.sum()} frames, {100 * np.mean(raw_nn[allm] > ref):.0f} % beyond the test dimers' 95 % "
      f"nearest-pair distance; {(raw_oo < tp_oo.min()).sum()} below the training O-O minimum", flush=True)

# ---- trimers: the sorted O-O triangle against the training triples
tt = []
for st in train:
    p = ang(st); n = len(p) // 3
    for idx in itertools.combinations(range(n), 3):
        O = p[[3 * i for i in idx]]
        tt.append(np.sort([np.linalg.norm(O[0] - O[1]), np.linalg.norm(O[0] - O[2]), np.linalg.norm(O[1] - O[2])]))
tt = np.array(tt)
d3 = H["3b"]; P3 = np.asarray(d3["pos"])
rt = np.array([np.sort([np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[0] - p[6]), np.linalg.norm(p[3] - p[6])]) for p in P3])
nn3 = np.array([np.min(np.linalg.norm(tt - r, axis=1)) for r in rt])
ref3 = np.quantile([np.min(np.linalg.norm(np.delete(tt, i, 0) - tt[i], axis=1)) for i in range(0, len(tt), 7)], 0.95)
print(f"training triples: {len(tt)}; raw trimers: {len(rt)}; smallest O-O below the training minimum in "
      f"{(rt[:, 0] < tp_oo.min()).sum()}; O-O triangle farther from every training triple than 95 % of the training "
      f"triples are from each other ({ref3:.2f} A): " + ", ".join(
          f"largest O-O {lo}-{up} A {100 * np.mean(nn3[(rt[:, 2] >= lo) & (rt[:, 2] < up)] > ref3):.0f} %"
          for lo, up in ((0, 5), (5, 7), (7, 9), (9, 99))), flush=True)

# ---- (2) the training forces: rigid-body against internal, per molecule
def split(pos, F):
    """the forces of one rigid molecule split into the rigid-body part (projection on the 6 rigid displacements) and
    the rest (unweighted metric)."""
    c = pos.mean(axis=0)
    basis = []
    for k in range(3):
        e = np.zeros((3, 3)); e[:, k] = 1.0; basis.append(e.reshape(-1))
        w = np.zeros(3); w[k] = 1.0; basis.append(np.cross(w, pos - c).reshape(-1))
    Q, _ = np.linalg.qr(np.array(basis).T)
    f = F.reshape(-1); fr = Q @ (Q.T @ f)
    return float(fr @ fr), float((f - fr) @ (f - fr))
rig = intl = 0.0; per_size = {}
for st in train:
    p, F = ang(st), np.asarray(st.forces)
    n = len(p) // 3
    for m in range(n):
        r_, i_ = split(p[3 * m:3 * m + 3], F[3 * m:3 * m + 3])
        rig += r_; intl += i_
        per_size.setdefault(n, [0.0, 0.0]); per_size[n][0] += r_; per_size[n][1] += i_
print(f"training forces: the internal (non-rigid) part is {100 * intl / (rig + intl):.0f} % of the sum of squares "
      f"(rms per molecule: rigid {math.sqrt(rig / sum(len(ang(s)) // 3 for s in train)) * K / BOHR:.2f}, internal "
      f"{math.sqrt(intl / sum(len(ang(s)) // 3 for s in train)) * K / BOHR:.2f} kcal/mol/A); by size: "
      + ", ".join(f"{n}: {100 * v[1] / sum(v):.0f} %" for n, v in sorted(per_size.items())), flush=True)
json.dump(dict(pairs=dict(n=len(tp), oo_min=float(tp_oo.min()), E_max=float(tp_E.max()), E_q99=float(hi)),
               raw2b_by_oo=rows, dimer_nn_q95=float(ref), internal_force_share=intl / (rig + intl)),
          open(os.path.join(OUT, "e3qp3_coverage.json"), "w"), indent=1)
print(f"[{time.time() - t0:.0f} s]")
