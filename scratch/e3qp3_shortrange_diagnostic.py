"""
Where E3-QP2's short-range error comes from: the saved models of results/e3qp2 read again, ~30 min on one GPU.

    python scratch/e3qp3_shortrange_diagnostic.py results2/e3qp2 external/qaqua/build_pol [--seeds 0] [--rungs ...]

The raw CCSD(T) frames are near-rigid, not rigid: their monomers are within 0.03 A and 5 deg of the training monomer,
and the truth's interaction energy moves with that -- 0.090 kcal/mol rms on the 2-body set, 0.064 on the 3-body
(scratch/e3qp3_distortion.py) -- while every training cluster has exactly rigid monomers. So each model's error on a
raw frame is split here into
    labels        model(frame) - label                    what E3-QP2 read
    truth         model(frame) - truth(frame)             what the model was trained to reproduce
    rigid         model(rigid) - truth(rigid)             every monomer replaced by the training monomer, best aligned
                                                          (Kabsch): the intermolecular error alone
    distortion    [model(frame) - model(rigid)] - [truth(frame) - truth(rigid)]: the response to the distortion
(truth = rigid + distortion, frame by frame), by O-O bin, and for the 2-body set by coverage: whether the dimer's nearest
training pair (9 intermolecular atom distances, rms, over the 8 symmetric orderings) is farther than 95 % of the
held-out test dimers' are.

And the pinned charges. E3-QP2 read the raw frames with the pinned charges recomputed at each frame's own monomer
geometry (the dipole-surface charges q3; TTM3-F's site charges), while in training every monomer had the rigid
monomer's charges and E_0 learned the flux of the truth's charges from its internal forces: on a distorted frame the
bands then carry the flux's energy and E_0 adds its learned flux response -- twice (0.069 kcal/mol rms of
electrostatics on the 2-body set with TTM3-F's charges, 0.051 with q3: scratch/e3qp3_flux_double_count.py). So every
pinned rung is also read with the rigid monomer's charges at the frame's positions ("fixed"), the model as trained.

On the data each model saw: the training clusters' energies, and their forces split into the rigid-body part (net
force and torque per molecule, all that a rigid monomer's energy depends on) and the internal part; the test
hexamers' energy error split into its 2-body part and the rest. One seed per rung by default (--seeds 0,1,2,3 for
all). Prints the readings; writes <results>/shortrange_diagnostic.json.
"""
import argparse, copy, itertools, json, math, os, pickle, sys, time
import numpy as np
import jax
import jax.numpy as jnp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, train as T
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
RUNGS = "M_inf,M_16p,M_16pm,M_16C0p,M_16S0pm,M_16C0pm,M_16S2pm,M_16C2pm"
OO_BINS = {"2b": ((2.3, 2.7), (2.7, 2.9), (2.9, 3.2), (3.2, 5.0), (5.0, 99.0)),
           "3b": ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 99.0)),
           "4b": ((0.0, 5.0), (5.0, 7.0), (7.0, 99.0))}
rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def rigidify(P):
    """every monomer of a frame (A, (3n, 3)) replaced by the rigid training monomer, best aligned (Kabsch, unweighted)."""
    ref = W.monomer_positions(); out = []
    for m in range(len(P) // 3):
        X = P[3 * m:3 * m + 3]; cx, cr = X.mean(0), ref.mean(0)
        U, _, Vt = np.linalg.svd((ref - cr).T @ (X - cx))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        out.append((ref - cr) @ (Vt.T @ D @ U.T).T + cx)
    return np.concatenate(out)


def pair_desc(A, B):
    """the 9 intermolecular atom distances of a dimer (A) in its 8 symmetric orderings."""
    out = []
    for a in ((0, 1, 2), (0, 2, 1)):
        for b in ((0, 1, 2), (0, 2, 1)):
            d = np.linalg.norm(A[list(a)][:, None] - B[list(b)][None], axis=-1)
            out.append(d.reshape(-1)); out.append(d.T.reshape(-1))
    return np.array(out)


def rigid_split(pos, F):
    """sums of squares of the rigid-body and the internal parts of one molecule's force vectors (projection on its six
    rigid displacements, unweighted)."""
    c = pos.mean(axis=0); basis = []
    for k in range(3):
        e = np.zeros((3, 3)); e[:, k] = 1.0; basis.append(e.reshape(-1))
        w = np.zeros(3); w[k] = 1.0; basis.append(np.cross(w, pos - c).reshape(-1))
    Q, _ = np.linalg.qr(np.array(basis).T)
    f = np.asarray(F).reshape(-1); fr = Q @ (Q.T @ f)
    return float(fr @ fr), float((f - fr) @ (f - fr))


def model_nbody(params, rung, frames, n, truth, fixed=None):
    """E3.model_nbody (the n-body energy by the subset expansion, every subset with the rung's pins), with the pinned
    charges either the truth's at each frame's own geometry (fixed None: as E3-QP2 read) or the rigid monomer's
    (fixed = (q3 of O, H, H; TTM3-F's H1, H2, M site charges)) at the frame's positions."""
    subsets = [c for k in range(2, n + 1) for c in itertools.combinations(range(n), k)]
    atoms = {c: np.concatenate([np.arange(3 * i, 3 * i + 3) for i in c]) for c in subsets}
    structs = []
    for p in frames:
        for c in subsets:
            sub = p[atoms[c]]
            q = np.tile(fixed[0], len(c)) if fixed is not None else truth.molecule_charges(sub)
            structs.append(W.water_structure(sub, charges=q))
    E3.pin_for(rung, structs, truth)
    if fixed is not None and rung.msites:
        qs = fixed[1]
        for s in structs:
            k = s.n_atoms // 3
            s.pinned = {**s.pinned, "q_site": np.tile([0.0, qs[0], qs[1]], k), "vsite_q": np.full(k, qs[2])}
    E = T.predict_energies(params, rung, structs).reshape(len(frames), len(subsets))
    out = np.zeros(len(frames))
    for f in range(len(frames)):
        term = {}
        for j, c in enumerate(subsets):
            term[c] = E[f, j] - sum(term[d] for k in range(2, len(c)) for d in itertools.combinations(c, k))
        out[f] = term[tuple(range(n))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("root")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--rungs", default=RUNGS)
    ap.add_argument("--sets", default="2b,3b,4b"); ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    truth = W.qaqua_pol_truth(args.root)
    rungs = args.rungs.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    # configs/water_qaqua_pol2.yaml's clusters part: the architecture and the held-out selection
    a = type("A", (), dict(lA=4.0, rcut=6.0, F=16, l1=0.75, rs_over_l1=0.0, response_edge=0.0, charges=None,
                           charge_neutrality="fragment", heldout=args.sets, heldout_oh_tol=0.03, heldout_angle_tol=5.0,
                           heldout_strict_oh_tol=0.02, heldout_strict_angle_tol=3.0, heldout_max_pair_kcal=2.0,
                           heldout_max=args.max_frames, rungs=",".join(rungs)))()
    mono = W.monomer_positions()
    fixed = (truth.molecule_charges(mono), truth.qa.ttm(ang_to_bohr(mono))["site_charges"][0])
    with open(os.path.join(args.results, "clusters_cache.pkl"), "rb") as fh:
        cached = pickle.load(fh)
    train = [s for s in cached["train"] if s.n_atoms > 3]
    hexamers, dimers = cached["tests"][6], cached["tests"][2]
    H = E3.prepare_heldout(a, truth)
    sets = {}
    for key, d in H.items():
        n = d["n"]; P = np.asarray(d["pos"]); R = np.array([rigidify(p) for p in P])
        t_rig = np.array([truth.qa.nbody(ang_to_bohr(r), max_order=n)[n] for r in R])
        sets[key] = dict(n=n, P=P, R=R, label=d["label"], t_act=d["pes"], t_rig=t_rig, roo=d["roo"], strict=d["strict"])
        if n == 2:
            sets[key]["oo"] = np.array([np.linalg.norm(p[0] - p[3]) for p in P])
        print(f"held-out {key}: {len(P)} frames; truth(frame) - truth(rigid) rms {rms(d['pes'] - t_rig) * K:.4f} kcal/mol "
              f"[{time.time() - t0:.0f} s]", flush=True)
    if "2b" in sets:                                     # coverage of the raw dimers by the training clusters' pairs
        tp = np.array([pair_desc(p[3 * i:3 * i + 3], p[3 * j:3 * j + 3])[0] for st in train
                       for p in [bohr_to_ang(np.asarray(st.positions))]
                       for i, j in itertools.combinations(range(len(p) // 3), 2)])
        nn = lambda A, B: float(np.min(np.sqrt(np.mean((pair_desc(A, B)[:, None, :] - tp[None]) ** 2, axis=-1))))
        ref = np.quantile([nn(*np.split(bohr_to_ang(np.asarray(st.positions)), 2)) for st in dimers], 0.95)
        s2 = sets["2b"]; s2["out"] = np.array([nn(p[:3], p[3:]) for p in s2["P"]]) > ref
        print(f"raw dimers beyond the test dimers' 95 % nearest-training-pair distance ({ref:.3f} A): "
              f"{100 * s2['out'].mean():.0f} %", flush=True)
    hex_pairs = [(h, i, j) for h in range(len(hexamers)) for i, j in itertools.combinations(range(6), 2)]
    hex_pos = [bohr_to_ang(np.asarray(s.positions)) for s in hexamers]
    pair_frames = np.array([np.concatenate([hex_pos[h][3 * i:3 * i + 3], hex_pos[h][3 * j:3 * j + 3]]) for h, i, j in hex_pairs])
    owner = [h for h, _, _ in hex_pairs]
    t_hex2 = np.bincount(owner, weights=np.array([truth.energy(f, 2) for f in pair_frames]), minlength=len(hexamers))
    t_hex = np.array([s.energy for s in hexamers])

    report = {}
    for name in rungs:
        rung = E3.configure(E3.RUNGS[name], a)
        pinned_charges = bool(rung.msites or (rung.charges and rung.pin_q))
        rows = []
        for seed in seeds:
            path = os.path.join(args.results, f"clusters_{name}_seed{seed}.params.pkl")
            if not os.path.exists(path):
                print(f"  {name} seed {seed}: no {path}", flush=True); continue
            params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(path, "rb")))
            row = {"seed": seed}
            for key, s in sets.items():
                m_act = model_nbody(params, rung, s["P"], s["n"], truth)
                m_rig = model_nbody(params, rung, s["R"], s["n"], truth)
                e = dict(labels=m_act - s["label"], truth=m_act - s["t_act"], rigid=m_rig - s["t_rig"],
                         distortion=(m_act - m_rig) - (s["t_act"] - s["t_rig"]))
                if pinned_charges:
                    m_fix = model_nbody(params, rung, s["P"], s["n"], truth, fixed=fixed)
                    e.update(fixed_labels=m_fix - s["label"], fixed_distortion=(m_fix - m_rig) - (s["t_act"] - s["t_rig"]))
                blk = lambda m: {**{k: rms(v[m]) * K for k, v in e.items()}, "n": int(m.sum())}
                r = {"all": blk(np.ones(len(m_act), bool)), "strict": blk(s["strict"])}
                dist = s["oo"] if s["n"] == 2 else s["roo"]
                for lo, hi in OO_BINS[key]:
                    m = (dist >= lo) & (dist < hi)
                    if m.any():
                        r[f"{lo:g}-{hi:g}"] = blk(m)
                if "out" in s:
                    r["covered"] = blk(~s["out"]); r["not covered"] = blk(s["out"])
                row[key] = r
            tr = E3.pin_for(rung, copy.deepcopy(train), truth)
            E, F = T.predict(params, rung, tr)
            rig_sq = int_sq = 0.0; n_mol = 0
            for st, f in zip(tr, F):
                p = np.asarray(st.positions); err = np.asarray(f) - np.asarray(st.forces)
                for m in range(len(p) // 3):
                    r_, i_ = rigid_split(p[3 * m:3 * m + 3], err[3 * m:3 * m + 3]); rig_sq += r_; int_sq += i_; n_mol += 1
            unit = K / bohr_to_ang(1.0)                                          # hartree/bohr -> kcal/mol/A
            row["train"] = {"energy": rms(E - np.array([st.energy for st in tr])) * K,
                            "force_rigid": math.sqrt(rig_sq / n_mol) * unit, "force_internal": math.sqrt(int_sq / n_mol) * unit}
            hx = E3.pin_for(rung, copy.deepcopy(hexamers), truth)
            m_hex = T.predict_energies(params, rung, hx)
            m_hex2 = np.bincount(owner, weights=model_nbody(params, rung, pair_frames, 2, truth), minlength=len(hexamers))
            row["hexamers"] = {"energy": rms(m_hex - t_hex) * K, "two_body": rms(m_hex2 - t_hex2) * K,
                               "rest": rms((m_hex - m_hex2) - (t_hex - t_hex2)) * K}
            rows.append(row)
            print(f"  {name} seed {seed}: training clusters E {row['train']['energy']:.3f} kcal/mol, F rigid "
                  f"{row['train']['force_rigid']:.3f} / internal {row['train']['force_internal']:.3f} kcal/mol/A; hexamers "
                  f"{row['hexamers']['energy']:.3f} (2-body {row['hexamers']['two_body']:.3f}, rest {row['hexamers']['rest']:.3f})"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
            for key in sets:
                cols = [c for c in ("labels", "truth", "rigid", "distortion", "fixed_labels", "fixed_distortion") if c in row[key]["all"]]
                print(f"      {key} ({' / '.join(cols)}, kcal/mol): " + "; ".join(
                    f"{b} n {v['n']} " + " / ".join(f"{v[c]:.4f}" for c in cols) for b, v in row[key].items()), flush=True)
        report[name] = rows
    with open(os.path.join(args.results, "shortrange_diagnostic.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"done [{time.time() - t0:.0f} s]; {os.path.join(args.results, 'shortrange_diagnostic.json')}")


if __name__ == "__main__":
    main()
