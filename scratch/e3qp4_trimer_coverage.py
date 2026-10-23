"""
The raw CCSD(T) trimers against the training triples (notes/e3qp4-results.md): how far each of E3-QP3/QP4's 4673 held-out
trimers lies from every three-molecule subset of the training data -- the 300 relaxed clusters' triples (100 trimers,
4 per tetramer, 10 per pentamer) and the 200 contact trimers -- in the 27 intermolecular atom distances (rms, over the 48
orderings: the molecules' 6 permutations and the H swaps), against the same distance for the test hexamers' 480 triples
(relaxed, like the training clusters).  A raw trimer counts as beyond the training data when it lies farther than 95 % of
the hexamers' triples do: the 3-body analogue of the dimers' coverage that found E3-QP2's compressed dimers
(scratch/e3qp3_preregistration.py).

    python scratch/e3qp4_trimer_coverage.py [build_pol]          # ~5 min, one CPU
"""
import itertools, math, os, sys, time
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W
from sseft.units import bohr_to_ang
import e3_water as E3

t0 = time.time()
truth = W.qaqua_pol_truth(ROOT)
geometry = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
clusters = W.cluster_dataset((3, 4, 5), 100, seed=0, sigma=1e-9, truth=truth, **geometry)      # E3-QP2/3/4's, bit for bit
contacts = W.contact_configurations(3, 200, (2.3, 3.0), seed=701, truth=truth, max_pair_kcal=2.0, min_any=1.5)
hexamers = W.cluster_dataset((6,), 24, seed=100, truth=truth, **geometry)                      # the test hexamers
print(f"{len(clusters)} training clusters, {len(contacts)} contact trimers, {len(hexamers)} test hexamers  [{time.time() - t0:.0f} s]", flush=True)

PERMS = [(p, s) for p in itertools.permutations(range(3)) for s in itertools.product((False, True), repeat=3)]   # 48


def desc(mols):
    """the 27 intermolecular atom distances of three molecules (each (3, 3), O H H, A), pairs (0,1), (0,2), (1,2)."""
    return np.concatenate([np.linalg.norm(mols[i][:, None] - mols[j][None], axis=-1).ravel() for i, j in ((0, 1), (0, 2), (1, 2))])


def variants(mols):
    out = []
    for perm, swaps in PERMS:
        m = [mols[k][[0, 2, 1]] if sw else mols[k] for k, sw in zip(perm, swaps)]
        out.append(desc(m))
    return out


def triples(structs):
    for st in structs:
        p = bohr_to_ang(np.asarray(st.positions)); n = len(p) // 3
        for idx in itertools.combinations(range(n), 3):
            yield [p[3 * i:3 * i + 3] for i in idx]


bank = np.array([v for t in triples(clusters + contacts) for v in variants(t)])          # (triples x 48, 27)
n_train = len(bank) // 48
print(f"{n_train} training triples ({len(bank)} orderings)  [{time.time() - t0:.0f} s]", flush=True)
bank2 = np.sum(bank ** 2, axis=1)


def nearest(D):
    out = np.empty(len(D))
    for i0 in range(0, len(D), 256):
        blk = D[i0:i0 + 256]
        d2 = np.sum(blk ** 2, axis=1)[:, None] + bank2[None] - 2 * blk @ bank.T
        out[i0:i0 + 256] = np.sqrt(np.maximum(d2.min(axis=1), 0) / 27)
    return out


hex_nn = nearest(np.array([desc(t) for t in triples(hexamers)]))
ref = float(np.quantile(hex_nn, 0.95))
a = type("A", (), dict(heldout="3b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=0))()
d = E3.prepare_heldout(a, truth)["3b"]
P = np.asarray(d["pos"])
raw_nn = nearest(np.array([desc([p[0:3], p[3:6], p[6:9]]) for p in P]))
soo, roo = d["soo"], d["roo"]
print(f"the test hexamers' triples: nearest training triple {np.median(hex_nn):.3f} A (median), 95 % within {ref:.3f} A  "
      f"[{time.time() - t0:.0f} s]", flush=True)
print(f"raw trimers beyond the training triples (farther than {ref:.3f} A): {100 * np.mean(raw_nn > ref):.0f} % of {len(P)}; "
      f"median distance {np.median(raw_nn):.3f} A", flush=True)
for label, x, edges in (("closest O-O", soo, ((0.0, 2.7), (2.7, 3.0), (3.0, math.inf))),
                        ("largest O-O", roo, ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, math.inf)))):
    print(f"  by {label}: " + "; ".join(f"{lo:g}-{hi:g} A {100 * np.mean(raw_nn[(x >= lo) & (x < hi)] > ref):.0f} % of "
                                        f"{int(((x >= lo) & (x < hi)).sum())} (median {np.median(raw_nn[(x >= lo) & (x < hi)]):.3f} A)"
                                        for lo, hi in edges if ((x >= lo) & (x < hi)).any()), flush=True)
np.save(os.path.join(REPO, "scratch", "e3qp4_trimer_coverage.npy"), np.stack([raw_nn, soo, roo]))
print(f"done in {time.time() - t0:.0f} s")
