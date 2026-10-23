"""
Where a clusters run's short-range error comes from: its saved models read again, minutes on one GPU (E3-QP2's: 12 min
for one seed per rung on an RTX 3090).

    python scratch/e3qp3_shortrange_diagnostic.py results2/e3qp2 external/qaqua/build_pol [--seeds 0|0,1|all] [--rungs ...]
    python scratch/e3qp3_shortrange_diagnostic.py results/e3qp3 external/qaqua/build_pol --response_edge 0.75 --seeds all

The raw CCSD(T) frames are near-rigid, not rigid: their monomers are within 0.03 A and 5 deg of the training monomer,
and the truth's interaction energy moves with that -- 0.090 kcal/mol rms on the 2-body set, 0.064 on the 3-body
(scratch/e3qp3_distortion.py) -- while every training structure has exactly rigid monomers. So each model's error on a
raw frame is split here into
    labels        model(frame) - label, the pinned charges at the frame's own geometry (E3-QP2's reading)
    truth         model(frame) - truth(frame)             what the model was trained to reproduce
    rigid         model(rigid) - truth(rigid)             every monomer replaced by the training monomer, best aligned
                                                          (Kabsch): the intermolecular error alone
    distortion    [model(frame) - model(rigid)] - [truth(frame) - truth(rigid)]: the response to the distortion
and, for the rungs with pinned charges, read with the rigid monomer's charges at the frame's positions -- the models as
trained, the harness's reading from v0.0.12 with truth.heldout.pin_charges: rigid --
    fixed_labels, fixed_distortion
(truth = rigid + distortion, frame by frame; on a distorted frame the charges at its own geometry carry the flux of the
truth's charges, which E_0 learned from the internal forces, a second time: e3qp2-results.md sections 3.4-3.5), by O-O
bin (the 2-body set by its O-O, the others by their largest and their smallest O-O), and for the 2-body set by coverage:
whether the dimer's nearest training pair (9 intermolecular atom distances, rms, over the 8 symmetric orderings) is
farther than 95 % of the held-out test dimers' are. On the data each model saw: the training clusters' energies and
forces (the forces split into the rigid-body part -- net force and torque per molecule, all that a rigid monomer's energy
depends on -- and the internal part), and those of the contact configurations apart (v0.0.12); the test hexamers' energy
error split into its 2-body part and the rest.

The run's architecture comes from the flags (E3-QP2's by default) and is checked against the model records of
<results>/clusters.json where the run wrote them (v0.0.12). Prints the readings; writes <results>/<--out>, and with
--save_frames (v0.0.14) every frame's n-body energies -- each model's on the frame, on its rigidified twin and with the
pins as trained, the labels, the truth's, the frames' positions and O-O distances, the hexamers' energies and 2-body
parts -- to <results>/<--out stem>_frames.npz, so that the 3-body's strength and direction can be read without the GPU.
"""
import argparse, copy, glob, itertools, json, math, os, pickle, re, sys, time
import numpy as np
import jax
import jax.numpy as jnp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, train as T
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
E3QP2_RUNGS = "M_inf,M_16p,M_16pm,M_16C0p,M_16S0pm,M_16C0pm,M_16S2pm,M_16C2pm"
OO_BINS = {"2b": ((2.3, 2.7), (2.7, 2.9), (2.9, 3.2), (3.2, 5.0), (5.0, 99.0)),
           "3b": ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 99.0)),
           "4b": ((0.0, 5.0), (5.0, 7.0), (7.0, 99.0))}
CONTACT_BINS = ((0.0, 2.7), (2.7, 3.0), (3.0, 99.0))
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


def fit_on(params, rung, structs, truth):
    """energy rmse (kcal/mol per structure) and the forces' rigid-body and internal rms per molecule (kcal/mol/A)."""
    if not structs:
        return None
    tr = E3.pin_for(rung, copy.deepcopy(structs), truth)
    E, F = T.predict(params, rung, tr)
    rig_sq = int_sq = 0.0; n_mol = 0
    for st, f in zip(tr, F):
        p = np.asarray(st.positions); err = np.asarray(f) - np.asarray(st.forces)
        for m in range(len(p) // 3):
            r_, i_ = rigid_split(p[3 * m:3 * m + 3], err[3 * m:3 * m + 3]); rig_sq += r_; int_sq += i_; n_mol += 1
    unit = K / bohr_to_ang(1.0)                                                  # hartree/bohr -> kcal/mol/A
    return {"n": len(tr), "energy": rms(E - np.array([st.energy for st in tr])) * K,
            "force_rigid": math.sqrt(rig_sq / n_mol) * unit, "force_internal": math.sqrt(int_sq / n_mol) * unit}


def check_record(name, rung, record):
    """the configured rung against the run's model record (clusters.json, v0.0.12): the flags must describe the run."""
    if not record:
        return
    ang = lambda x: float(bohr_to_ang(x))
    pairs = [("band_edge_angstrom", ang(rung.l1)), ("r_cut_angstrom", ang(rung.e0["r_cut"])), ("features", rung.e0["F"]),
             ("analytic_envelope_angstrom", ang(rung.lA))]
    if rung.band_fields >= 0:
        pairs.append(("response_edge_angstrom", ang(rung.bf_l) if rung.bf_l > 0 else None))
    if rung.e0_field or "e0_field_taper_angstrom" in record:
        taper = record.get("e0_field_taper_angstrom") or [None, None]
        if not rung.e0_field or any(t is None or abs(t - ang(v)) > 1e-9 for t, v in zip(taper, (rung.e0_field_on, rung.e0_field_off))):
            raise ValueError(f"{name}: the field term's taper of the run ({taper}) is not the flags' -- give --rcut as in its "
                             "configuration")
    bad = [(k, record.get(k), v) for k, v in pairs
           if not ((record.get(k) is None and v is None) or (record.get(k) is not None and v is not None and abs(record[k] - v) < 1e-9))]
    if bad:
        raise ValueError(f"{name}: the flags do not describe the run (" + "; ".join(f"{k}: run {r}, flags {v}" for k, r, v in bad)
                         + "); give --l1 / --response_edge / --F / --rcut / --lA as in its configuration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("root")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds, or all")
    ap.add_argument("--rungs", default=None, help="comma-separated; omitted: the rungs of <results>/clusters.json (E3-QP2's without it)")
    ap.add_argument("--sets", default="2b,3b,4b"); ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--l1", type=float, default=0.75, help="the run's band edge (A)")
    ap.add_argument("--response_edge", type=float, default=0.0, help="the run's response edge (A; 0: the band edge)")
    ap.add_argument("--F", type=int, default=16); ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--lA", type=float, default=4.0)
    ap.add_argument("--out", default="shortrange_diagnostic.json")
    ap.add_argument("--save_frames", action="store_true", help="every frame's n-body energies to <out stem>_frames.npz")
    args = ap.parse_args()
    t0 = time.time()
    truth = W.qaqua_pol_truth(args.root)
    run_json = os.path.join(args.results, "clusters.json")
    run = json.load(open(run_json)) if os.path.exists(run_json) else {}
    run_rungs = [k for k, v in run.items() if isinstance(v, dict) and "clusters_by_size" in v]
    rungs = (args.rungs or ",".join(run_rungs) or E3QP2_RUNGS).split(",")
    if run.get("heldout_pin_charges"):
        print(f"the run read its held-out sets with the pinned charges: {run['heldout_pin_charges']}", flush=True)
    a = type("A", (), dict(lA=args.lA, rcut=args.rcut, F=args.F, l1=args.l1, rs_over_l1=0.0, response_edge=args.response_edge,
                           charges=None, charge_neutrality="fragment", heldout=args.sets, heldout_oh_tol=0.03,
                           heldout_angle_tol=5.0, heldout_strict_oh_tol=0.02, heldout_strict_angle_tol=3.0,
                           heldout_max_pair_kcal=2.0, heldout_max=args.max_frames, rungs=",".join(rungs)))()
    with open(os.path.join(args.results, "clusters_cache.pkl"), "rb") as fh:
        cached = pickle.load(fh)
    clusters = [s for s in cached["train"] if s.n_atoms > 3 and not s.info.get("contact")]
    contacts = [s for s in cached["train"] if s.info.get("contact")]
    hexamers, dimers = cached["tests"][6], cached["tests"][2]
    print(f"training structures: {len(clusters)} clusters, {len(contacts)} contact configurations", flush=True)
    H = E3.prepare_heldout(a, truth)
    sets = {}
    for key, d in H.items():
        n = d["n"]; P = np.asarray(d["pos"]); R = np.array([rigidify(p) for p in P])
        t_rig = np.array([truth.qa.nbody(ang_to_bohr(r), max_order=n)[n] for r in R])
        sets[key] = dict(n=n, P=P, R=R, q=d["charges"], qR=np.array([truth.molecule_charges(r) for r in R]), label=d["label"],
                         t_act=d["pes"], t_rig=t_rig, roo=d["roo"], soo=d.get("soo", d["roo"]), strict=d["strict"])
        print(f"held-out {key}: {len(P)} frames; truth(frame) - truth(rigid) rms {rms(d['pes'] - t_rig) * K:.4f} kcal/mol "
              f"[{time.time() - t0:.0f} s]", flush=True)
    if "2b" in sets:                                # coverage of the raw dimers by the training structures' pairs
        tp = np.array([pair_desc(p[3 * i:3 * i + 3], p[3 * j:3 * j + 3])[0] for st in clusters + contacts
                       for p in [bohr_to_ang(np.asarray(st.positions))]
                       for i, j in itertools.combinations(range(len(p) // 3), 2)])
        nn = lambda A, B: float(np.min(np.sqrt(np.mean((pair_desc(A, B)[:, None, :] - tp[None]) ** 2, axis=-1))))
        ref = np.quantile([nn(*np.split(bohr_to_ang(np.asarray(st.positions)), 2)) for st in dimers], 0.95)
        s2 = sets["2b"]; s2["out"] = np.array([nn(p[:3], p[3:]) for p in s2["P"]]) > ref
        close = s2["soo"] < 2.7
        print(f"raw dimers beyond the test dimers' 95 % nearest-training-pair distance ({ref:.3f} A): {100 * s2['out'].mean():.0f} %"
              + (f" (below O-O 2.7 A: {100 * s2['out'][close].mean():.0f} %)" if close.any() else ""), flush=True)
    frames = None
    if args.save_frames:
        frames = {f"{key}__{k}": np.asarray(s[k]) for key, s in sets.items()
                  for k in ("P", "label", "t_act", "t_rig", "roo", "soo", "strict")}
    hex_pairs = [(h, i, j) for h in range(len(hexamers)) for i, j in itertools.combinations(range(6), 2)]
    hex_pos = [bohr_to_ang(np.asarray(s.positions)) for s in hexamers]
    pair_frames = np.array([np.concatenate([hex_pos[h][3 * i:3 * i + 3], hex_pos[h][3 * j:3 * j + 3]]) for h, i, j in hex_pairs])
    pair_q = np.array([truth.molecule_charges(f) for f in pair_frames])
    owner = [h for h, _, _ in hex_pairs]
    t_hex2 = np.bincount(owner, weights=np.array([truth.energy(f, 2) for f in pair_frames]), minlength=len(hexamers))
    t_hex = np.array([s.energy for s in hexamers])
    if frames is not None:
        frames.update(hex__truth=t_hex, hex__truth_2b=t_hex2)

    report = {}
    for name in rungs:
        rung = E3.configure(E3.RUNGS[name], a)
        check_record(name, rung, (run.get(name) or {}).get("model"))
        pinned_charges = bool(rung.msites or (rung.charges and rung.pin_q))
        if args.seeds == "all":
            seeds = sorted(int(re.search(r"_seed(\d+)\.params\.pkl$", p).group(1))
                           for p in glob.glob(os.path.join(args.results, f"clusters_{name}_seed*.params.pkl")))
        else:
            seeds = [int(s) for s in args.seeds.split(",")]
        rows = []
        for seed in seeds:
            path = os.path.join(args.results, f"clusters_{name}_seed{seed}.params.pkl")
            if not os.path.exists(path):
                print(f"  {name} seed {seed}: no {path}", flush=True); continue
            params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(path, "rb")))
            row = {"seed": seed}
            for key, s in sets.items():
                m_act = E3.model_nbody(params, rung, s["P"], s["q"], s["n"], truth)
                m_rig = E3.model_nbody(params, rung, s["R"], s["qR"], s["n"], truth)
                e = dict(labels=m_act - s["label"], truth=m_act - s["t_act"], rigid=m_rig - s["t_rig"],
                         distortion=(m_act - m_rig) - (s["t_act"] - s["t_rig"]))
                if pinned_charges:
                    m_fix = E3.model_nbody(params, rung, s["P"], s["q"], s["n"], truth, rigid=True)
                    e.update(fixed_labels=m_fix - s["label"], fixed_distortion=(m_fix - m_rig) - (s["t_act"] - s["t_rig"]))
                if frames is not None:
                    tag = f"{name}__s{seed}__{key}"
                    frames.update({f"{tag}__act": m_act, f"{tag}__rig": m_rig})
                    if pinned_charges:
                        frames[f"{tag}__fix"] = m_fix
                blk = lambda m: {**{k: rms(v[m]) * K for k, v in e.items()}, "n": int(m.sum())}
                r = {"all": blk(np.ones(len(m_act), bool)), "strict": blk(s["strict"])}
                dist = s["soo"] if s["n"] == 2 else s["roo"]
                for lo, hi in OO_BINS[key]:
                    m = (dist >= lo) & (dist < hi)
                    if m.any():
                        r[f"{lo:g}-{hi:g}"] = blk(m)
                if s["n"] > 2:
                    for lo, hi in CONTACT_BINS:
                        m = (s["soo"] >= lo) & (s["soo"] < hi)
                        if m.any():
                            r[f"smallest {lo:g}-{hi:g}"] = blk(m)
                if "out" in s:
                    r["covered"] = blk(~s["out"]); r["not covered"] = blk(s["out"])
                row[key] = r
            row["train"] = fit_on(params, rung, clusters, truth)
            row["contacts"] = fit_on(params, rung, contacts, truth)
            hx = E3.pin_for(rung, copy.deepcopy(hexamers), truth)
            m_hex = T.predict_energies(params, rung, hx)
            m_hex2 = np.bincount(owner, weights=E3.model_nbody(params, rung, pair_frames, pair_q, 2, truth), minlength=len(hexamers))
            row["hexamers"] = {"energy": rms(m_hex - t_hex) * K, "two_body": rms(m_hex2 - t_hex2) * K,
                               "rest": rms((m_hex - m_hex2) - (t_hex - t_hex2)) * K}
            if frames is not None:
                frames.update({f"{name}__s{seed}__hex__E": np.asarray(m_hex), f"{name}__s{seed}__hex__2b": m_hex2})
            rows.append(row)
            fit_txt = lambda f: (f"E {f['energy']:.3f} kcal/mol, F rigid {f['force_rigid']:.3f} / internal {f['force_internal']:.3f} "
                                 f"kcal/mol/A") if f else "none"
            print(f"  {name} seed {seed}: training clusters {fit_txt(row['train'])}; contact configurations "
                  f"{fit_txt(row['contacts'])}; hexamers {row['hexamers']['energy']:.3f} (2-body {row['hexamers']['two_body']:.3f}, "
                  f"rest {row['hexamers']['rest']:.3f})  [{time.time() - t0:.0f} s]", flush=True)
            for key in sets:
                cols = [c for c in ("labels", "truth", "rigid", "distortion", "fixed_labels", "fixed_distortion") if c in row[key]["all"]]
                print(f"      {key} ({' / '.join(cols)}, kcal/mol): " + "; ".join(
                    f"{b} n {v['n']} " + " / ".join(f"{v[c]:.4f}" for c in cols) for b, v in row[key].items()), flush=True)
        report[name] = rows
    with open(os.path.join(args.results, args.out), "w") as fh:
        json.dump(report, fh, indent=1)
    if frames is not None:
        npz = os.path.join(args.results, os.path.splitext(args.out)[0] + "_frames.npz")
        np.savez_compressed(npz, **frames)
        print(f"every frame's n-body energies: {npz}", flush=True)
    print(f"done [{time.time() - t0:.0f} s]; {os.path.join(args.results, args.out)}")


if __name__ == "__main__":
    main()
