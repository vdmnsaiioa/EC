"""
E3-QP3's 3-body at contact (notes/e3qp3-results.md): the far-field-matched read-out of M_16C2pm (independent of E_0 and
of the seed, frozen while E_0 trains) read alone -- E_0 switched off -- on the raw CCSD(T) 3-body set, against TTM3-F's
3-body induction (the physics it stands for; TTM3-F's other terms are pairwise) and the labels.  The pinned bands are
pairwise, so the rung's 3-body with E_0 off is the read-out's.  What E_0 must learn of the 3-body is the labels less the
read-out's for the C rungs and the labels themselves for M_inf and the daggers: the script tells whether the read-out
makes that target smaller or larger, by the closest contact (smallest O-O) and by the largest O-O.

    python scratch/e3qp3_response_contact.py [build_pol] [matched read-out .pkl] [--max=N] [--edge=0.75] [--F=16]   # ~2 min

--edge: the read-out's band edge, with a read-out matched at that edge (scratch/e3qp2_matched_response.py M_16C2pm
--edge=...).

The read-out is E3-QP2's matched one (scratch/e3qp2_matched_response.py writes scratch/matched_M_16C2pm.pkl); E3-QP3's
is the same (its probe readings at 10-15 A equal E3-QP2's on every seed).
"""
import os, sys, time, math, pickle, json
import numpy as np
import jax, jax.numpy as jnp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, model as M
from sseft.units import ang_to_bohr
import e3_water as E3

K = 627.5094740631
args = [x for x in sys.argv[1:] if not x.startswith("--")]
root = args[0] if args else os.path.join(REPO, "external", "qaqua", "build_pol")
pkl = args[1] if len(args) > 1 else os.path.join(REPO, "scratch", "matched_M_16C2pm.pkl")
n_max = int(next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--max=")), 0))
edge = float(next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--edge=")), 0.75))     # the read-out's band edge (A)
width = int(next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--F=")), 16))          # E_0's width (the read-out's too)
t0 = time.time()
truth = W.qaqua_pol_truth(root)
rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")

a = type("A", (), dict(heldout="3b", heldout_oh_tol=0.03, heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02,
                       heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0, heldout_max=n_max,
                       lA=4.0, rcut=6.0, F=width, l1=0.75, rs_over_l1=0.0, response_edge=edge))()
d = E3.prepare_heldout(a, truth)["3b"]
P, lab, pes, soo, roo = d["pos"], d["label"], d["pes"], d["soo"], d["roo"]
print(f"{len(P)} raw trimers (E3-QP3's held-out 3-body set)  [{time.time() - t0:.0f} s]", flush=True)

# TTM3-F's 3-body: its induction (all orders, and first order in alpha) by the subset expansion; its total as a check
pairs = [(0, 1), (0, 2), (1, 2)]
sub = lambda p, idx: np.concatenate([p[3 * i:3 * i + 3] for i in idx])
ind3, tot3, ind3_1 = [], [], []
for p in P:
    pb = ang_to_bohr(p)
    full = truth.qa.ttm(pb)
    two = [truth.qa.ttm(sub(pb, ij)) for ij in pairs]
    one = [truth.qa.ttm(sub(pb, (i,))) for i in range(3)]
    ind3.append(full["ind"] - sum(t["ind"] for t in two) + sum(t["ind"] for t in one))
    tot3.append(full["total"] - sum(t["total"] for t in two) + sum(t["total"] for t in one))
    ind3_1.append(truth.qa.induction(pb, order=1)[0] - sum(truth.qa.induction(sub(pb, ij), order=1)[0] for ij in pairs))
ind3, tot3, ind3_1 = np.array(ind3), np.array(tot3), np.array(ind3_1)
print(f"TTM3-F's 3-body: induction rms {rms(ind3) * K:.4f} kcal/mol (first order in alpha {rms(ind3_1) * K:.4f}); its whole 3-body "
      f"less its induction's rms {rms(tot3 - ind3) * K:.1e} (pairwise otherwise)  [{time.time() - t0:.0f} s]", flush=True)


def readout_3b(name, mutual=None):
    rung = E3.configure(E3.RUNGS[name], a)
    if mutual is not None:
        rung = M.with_rung(rung, bf_mutual=mutual)
    params = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=1e-3)
    last = params["heads"]["energy"][-1]
    params["heads"]["energy"][-1] = {"w": 0 * last["w"], "b": 0 * last["b"]}         # E_0 off
    if rung.band_fields >= 0:
        params["readout"] = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(pkl, "rb")))
    t1 = time.time()
    e = np.asarray(E3.model_nbody(params, rung, P, d["charges"], 3, truth, rigid=True))
    print(f"  {name}{'' if mutual is None else f' (order {mutual})'}: 3-body with E_0 off, rms {rms(e) * K:.4f} kcal/mol  [{time.time() - t1:.0f} s]", flush=True)
    return e


check = readout_3b("M_16pm")                                     # the pinned bands alone: no 3-body
ro = readout_3b("M_16C2pm")                                      # the matched read-out, self-consistent
ro1 = readout_3b("M_16C2pm", mutual=1)                           # its first-order truncation

BINS = [("smallest O-O", soo, ((0.0, 2.7), (2.7, 3.0), (3.0, math.inf))),
        ("largest O-O", roo, ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, math.inf)))]
out = {"n": len(P), "pinned_bands_3b_rms": rms(check) * K, "bins": {}}
print("\nkcal/mol rms; slope = the read-out's 3-body regressed on TTM3-F's 3-body induction (through 0)")
print(f"{'bin':22s} {'n':>5s} {'labels':>7s} {'TTM ind':>8s} {'read-out':>8s} {'ro-ind':>7s} {'slope':>6s} {'ro1-ind1':>8s} {'slope1':>6s}"
      f" | E_0's target: {'labels':>7s} {'lab-ro':>7s} {'lab-ind':>7s} {'PES-ind':>7s}")
for label, x, edges in BINS:
    for lo, hi in ((0.0, math.inf),) + edges if label == "smallest O-O" else edges:
        s = (x >= lo) & (x < hi)
        if not s.any():
            continue
        slope = float(np.dot(ro[s], ind3[s]) / np.dot(ind3[s], ind3[s]))
        slope1 = float(np.dot(ro1[s], ind3_1[s]) / np.dot(ind3_1[s], ind3_1[s]))
        row = dict(n=int(s.sum()), labels=rms(lab[s]) * K, ttm_ind=rms(ind3[s]) * K, readout=rms(ro[s]) * K,
                   readout_less_ind=rms(ro[s] - ind3[s]) * K, slope=slope, readout1_less_ind1=rms(ro1[s] - ind3_1[s]) * K,
                   slope1=slope1, target_labels=rms(lab[s]) * K, target_labels_less_readout=rms(lab[s] - ro[s]) * K,
                   target_labels_less_ind=rms(lab[s] - ind3[s]) * K, pes_less_ind=rms(pes[s] - ind3[s]) * K,
                   readout_mean=float(np.mean(ro[s])) * K, ind_mean=float(np.mean(ind3[s])) * K, labels_mean=float(np.mean(lab[s])) * K)
        name = "all" if hi == math.inf and lo == 0.0 else f"{label} {lo:g}-{hi:g}"
        out["bins"][name] = row
        print(f"{name:22s} {row['n']:5d} {row['labels']:7.4f} {row['ttm_ind']:8.4f} {row['readout']:8.4f} {row['readout_less_ind']:7.4f} "
              f"{slope:6.2f} {row['readout1_less_ind1']:8.4f} {slope1:6.2f} |               {row['target_labels']:7.4f} "
              f"{row['target_labels_less_readout']:7.4f} {row['target_labels_less_ind']:7.4f} {row['pes_less_ind']:7.4f}", flush=True)
out["response_edge"] = edge; out["F"] = width
json.dump(out, open(os.path.join(REPO, "scratch", f"e3qp3_response_contact{'' if edge == 0.75 else f'_edge{edge:g}'}"
                                                  f"{'' if width == 16 else f'_F{width}'}.json"), "w"), indent=1)
print(f"done in {time.time() - t0:.0f} s")
