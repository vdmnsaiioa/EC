"""
Whether the 2-body split between E_0 and the field term's self terms matters off the data (E3-QP8's soft spot,
notes/e3qp8-results.md section 3.5). M_16C2pm_us's self terms carry a 1.3 kcal/mol rms 2-body on the raw dimers that
its E_0 cancels (M_16C2pm_ws's 0.28, M_16C2pm_w's 0.26): the data fix the sum, not the split. From the saved models, no
training (a few minutes, GPU or CPU: 16 models and 2000 scan dimers):

  (a) the raw CCSD(T) dimers (the decomposition's frames, <results>/decomposition_frames.npz; without them evaluated
      here), on and off the training pairs (the short-range diagnostic's measure: the nearest training pair by the 9
      intermolecular atom distances, beyond the test dimers' 95 % quantile);
  (b) a scan off the data: rigid dimers of the training monomer in random relative orientations, molecule B's O at
      fixed O-O distances (2.6-6.0 A) from A's, every intermolecular atom pair at least 1.5 A apart and the interaction
      at most 2 kcal/mol (the held-out sets' cap), against the PES the models were trained on (q-AQUA-pol).

On each: every model's 2-body error; the seeds' disagreement -- the rms over the dimers of the spread of a rung's seeds'
energies (sample standard deviation over the seeds), which grows off the data where the data do not fix the model; the
self terms' and E_0's sizes; and per seed the split against a twin -- S, how far the self terms moved (rms of their
difference), and R, what E_0 left of it (rms of the energy difference: the bands and the read-out are the seed's in
every rung, so on a dimer E_X - E_B = dE_0 + dself, checked). M_16C2pm_us against M_16C2pm_ws is the clean pair: the
same self reach, cross terms that give no dimer anything, self terms five times the size.

Registered before it runs (27 Oct): the split matters if, on the scan below O-O 5 A, M_16C2pm_us's 2-body error is
>= 20 % above M_16C2pm_ws's on >= 3 of 4 seeds, paired, and its seeds' disagreement >= 1.3 times M_16C2pm_ws's; it does
not if the error is within 10 % (the seeds' mean change) and the disagreement within 1.2 times; between, neither.
R/S on the scan against the raw dimers' says how: whether E_0 follows the self terms off the data as it does on them.

    python scratch/e3qp8_split_check.py results/e3qp8 external/qaqua/build_pol [--per_oo 200]

Writes <results>/split_check.json and the scan with every model's energies to <results>/split_check_frames.npz.
"""
import argparse, glob, itertools, json, os, pickle, re, sys, time
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, HERE)
from sseft import water as W
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3
import e3qp7_decomposition as D
from e3qp3_shortrange_diagnostic import check_record, pair_desc

K, rms = D.K, D.rms
OO = (2.6, 2.8, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
SCAN_BANDS = (("2.6-2.8", 2.5, 2.9), ("3.0-3.5", 2.9, 3.6), ("4.0-4.5", 3.6, 4.9), ("5.0-6.0", 4.9, 6.1))
TWINS = (("M_16C2pm_us", "M_16C2pm_ws"), ("M_16C2pm_ws", "M_16C2pm_w"), ("M_16C2pm_us", "M_16C2pm_w"),
         ("M_16C2pm_w", "M_16C2pm"), ("M_16C2pm_ws", "M_16C2pm"), ("M_16C2pm_us", "M_16C2pm"))
KEEP = ("E", "E0", "field", "self")


# ------------------------------------------------------------------------------------------------- the scan -------
def random_rotations(rng, n):
    """(n, 3, 3) rotation matrices, uniform on SO(3) (unit quaternions uniform on the sphere)."""
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], axis=1)


def scan_dimers(energy, oo, per_oo, rng, cap=2.0, min_any=1.5, max_tries=200):
    """rigid dimers of the training monomer (A: positions (6, 3), O H H twice) in random relative orientations with
    molecule B's O at each O-O distance from A's, every intermolecular atom pair at least min_any apart and
    energy(positions) (kcal/mol) at most cap; per distance the first per_oo accepted of at most max_tries * per_oo drawn.
    Returns positions (n, 6, 3), energies (n,), O-O (n,) and {distance: (accepted, drawn)}."""
    mono = W.monomer_positions() - W.monomer_positions()[0]
    P, E, O, tally = [], [], [], {}
    for d in oo:
        got = drawn = 0
        while got < per_oo and drawn < max_tries * per_oo:
            m = 64
            A = np.einsum("nij,aj->nai", random_rotations(rng, m), mono)
            u = rng.normal(size=(m, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            B = np.einsum("nij,aj->nai", random_rotations(rng, m), mono) + d * u[:, None, :]
            drawn += m
            apart = np.linalg.norm(A[:, :, None] - B[:, None, :], axis=-1).reshape(m, -1).min(axis=1) >= min_any
            for a, b in zip(A[apart], B[apart]):
                p = np.concatenate([a, b]); e = energy(p)
                if e <= cap:
                    P.append(p); E.append(e); O.append(d); got += 1
                    if got == per_oo:
                        break
        tally[d] = (got, drawn)
    return np.array(P), np.array(E), np.array(O), tally


# ----------------------------------------------------------------------------------------- the coverage -------
def training_pairs(cached):
    """the pair descriptors (9 intermolecular atom distances, A) of every pair of molecules the models trained on: the
    training clusters' and the contact configurations' (the short-range diagnostic's)."""
    structs = [s for s in cached["train"] if s.n_atoms > 3]
    return np.array([pair_desc(p[3 * i:3 * i + 3], p[3 * j:3 * j + 3])[0] for st in structs
                     for p in [bohr_to_ang(np.asarray(st.positions))]
                     for i, j in itertools.combinations(range(len(p) // 3), 2)])


def nearest(tp, P):
    """every dimer's (positions (n, 6, 3), A) distance to its nearest training pair: the rms over the 9 distances, the
    best of the 8 symmetric orderings."""
    out = np.empty(len(P))
    for k, p in enumerate(P):
        dsc = pair_desc(p[:3], p[3:])
        out[k] = np.sqrt(np.min(np.mean((dsc[:, None, :] - tp[None]) ** 2, axis=-1)))
    return out


# ------------------------------------------------------------------------------------------- the models -------
def model_quantities(rung, params, P, truth, chunk, built, pin_key):
    """the decomposition's per-dimer quantities (kcal/mol) with the pins as trained (the rigid monomer's charges)."""
    if pin_key not in built:
        built[pin_key] = D.batches_for(D.build_structures(rung, P, 2, truth, True), rung.K, chunk)
    Q = D.quantities(D.evaluate(params, rung, built[pin_key], len(P)), rung)
    return {k: np.asarray(v) for k, v in Q.items()}


def seed_spread(Es):
    """the rms over the dimers of the sample standard deviation of the seeds' energies."""
    A = np.array(Es)
    return rms(np.std(A, axis=0, ddof=1)) if len(A) > 1 else float("nan")


def readings(models, ref, masks):
    """per rung and mask: the error per seed, the seeds' disagreement, the self terms' and E_0's sizes; per twin pair
    and mask, per seed: S, R, the error change, and the check that the energy difference is E_0's and the field's."""
    out = {"models": {}, "twins": {}}
    for name, per_seed in models.items():
        seeds = sorted(per_seed)
        o = out["models"][name] = {}
        for b, m in masks.items():
            if not m.any():
                continue
            o[b] = dict(n=int(m.sum()), seeds=seeds,
                        error=[rms((per_seed[s]["E"] - ref)[m]) for s in seeds],
                        spread=seed_spread([per_seed[s]["E"][m] for s in seeds]),
                        E0=[rms(per_seed[s]["E0"][m]) for s in seeds],
                        self=[rms(per_seed[s]["self"][m]) if "self" in per_seed[s] else 0.0 for s in seeds])
    for X, B in TWINS:
        if X not in models or B not in models:
            continue
        seeds = sorted(set(models[X]) & set(models[B]))
        o = out["twins"][f"{X} - {B}"] = {}
        for b, m in masks.items():
            if not m.any() or not seeds:
                continue
            rows = []
            for s in seeds:
                qx, qb = models[X][s], models[B][s]
                dE = (qx["E"] - qb["E"])[m]
                dself = (qx.get("self", 0.0) - qb.get("self", 0.0))
                dself = np.broadcast_to(dself, qx["E"].shape)[m]
                dcomp = (qx["E0"] - qb["E0"])[m] + (np.broadcast_to(qx.get("field", 0.0), qx["E"].shape)
                                                    - np.broadcast_to(qb.get("field", 0.0), qb["E"].shape))[m]
                ex, eb = rms((qx["E"] - ref)[m]), rms((qb["E"] - ref)[m])
                rows.append(dict(seed=s, S=rms(dself), R=rms(dE), change=100 * (ex / eb - 1),
                                 check=float(np.max(np.abs(dE - dcomp)))))
            o[b] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("root")
    ap.add_argument("--rungs", default=None, help="comma-separated; omitted: the rungs of <results>/clusters.json")
    ap.add_argument("--per_oo", type=int, default=200, help="scan dimers per O-O distance")
    ap.add_argument("--oo", default=",".join(f"{d:g}" for d in OO), help="the scan's O-O distances (A)")
    ap.add_argument("--scan_seed", type=int, default=2710)
    ap.add_argument("--frames", default=None, help="the decomposition's frames (default <results>/decomposition_frames.npz)")
    ap.add_argument("--l1", type=float, default=0.75); ap.add_argument("--response_edge", type=float, default=0.75)
    ap.add_argument("--F", type=int, default=32); ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--lA", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    t0 = time.time()
    truth = W.qaqua_pol_truth(args.root)
    run = json.load(open(os.path.join(args.results, "clusters.json")))
    rungs = (args.rungs or ",".join(k for k, v in run.items() if isinstance(v, dict) and "clusters_by_size" in v)).split(",")
    a = type("A", (), dict(lA=args.lA, rcut=args.rcut, F=args.F, l1=args.l1, rs_over_l1=0.0,
                           response_edge=args.response_edge, charges=None, charge_neutrality="fragment"))()
    configured = {}
    for name in rungs:
        configured[name] = E3.configure(E3.RUNGS[name], a)
        check_record(name, configured[name], (run.get(name) or {}).get("model"))
    with open(os.path.join(args.results, "clusters_cache.pkl"), "rb") as fh:
        cached = pickle.load(fh)

    # the raw dimers: the decomposition's frames, or the diagnostic's
    fpath = args.frames or os.path.join(args.results, "decomposition_frames.npz")
    dec = dict(np.load(fpath)) if os.path.exists(fpath) else {}
    if "2b__P" in dec:
        rawP, raw_ref = dec["2b__P"], dec["2b__reference"]
        print(f"raw dimers: {len(rawP)} from {fpath}", flush=True)
    else:
        fr = np.load(os.path.join(args.results, "shortrange_diagnostic_frames.npz"))
        rawP, raw_ref, dec = fr["2b__P"], fr["2b__label"] * K, {}
        print(f"raw dimers: {len(rawP)} from the short-range diagnostic's frames (no decomposition frames)", flush=True)
    raw_oo = np.linalg.norm(rawP[:, 0] - rawP[:, 3], axis=-1)

    # the coverage, and the scan
    tp = training_pairs(cached)
    thr = np.quantile(nearest(tp, np.array([bohr_to_ang(np.asarray(s.positions)) for s in cached["tests"][2]])), 0.95)
    raw_nn = nearest(tp, rawP)
    raw_off = raw_nn > thr
    print(f"training pairs: {len(tp)}; the test dimers' 95 % nearest-training-pair distance {thr:.3f} A; raw dimers off the "
          f"training pairs: {100 * raw_off.mean():.0f} %  [{time.time() - t0:.0f} s]", flush=True)
    oo = tuple(float(x) for x in args.oo.split(","))
    scanP, scan_ref, scan_oo, tally = scan_dimers(lambda p: truth.qa.interaction_energy(ang_to_bohr(p)) * K, oo, args.per_oo,
                                                 np.random.default_rng(args.scan_seed))
    scan_nn = nearest(tp, scanP)
    print(f"scan: {len(scanP)} dimers (accepted / drawn by O-O: " + ", ".join(f"{d:g} {g}/{n}" for d, (g, n) in tally.items())
          + f"); the PES's interaction rms {rms(scan_ref):.3f} kcal/mol  [{time.time() - t0:.0f} s]", flush=True)
    print("  nearest training pair (A, median; share beyond the test dimers' 95 %): raw dimers "
          f"{np.median(raw_nn):.3f} ({100 * raw_off.mean():.0f} %); scan " + "; ".join(
              f"{lab} {np.median(scan_nn[(scan_oo >= lo) & (scan_oo < hi)]):.3f} "
              f"({100 * np.mean(scan_nn[(scan_oo >= lo) & (scan_oo < hi)] > thr):.0f} %)" for lab, lo, hi in SCAN_BANDS), flush=True)

    # every model on both sets
    models_raw, models_scan, built = {}, {}, {}
    for name in rungs:
        rung = configured[name]
        seeds = sorted(int(re.search(r"_seed(\d+)\.params\.pkl$", p).group(1))
                       for p in glob.glob(os.path.join(args.results, f"clusters_{name}_seed*.params.pkl")))
        pin_key = (bool(rung.pin_mu), bool(rung.pin_alpha), bool(rung.msites), rung.K)
        for seed in seeds:
            params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(
                os.path.join(args.results, f"clusters_{name}_seed{seed}.params.pkl"), "rb")))
            if all(f"{name}__s{seed}__2b__{q}" in dec for q in ("E", "E0")):
                qr = {q: dec[f"{name}__s{seed}__2b__{q}"] for q in KEEP if f"{name}__s{seed}__2b__{q}" in dec}
            else:
                qr = model_quantities(rung, params, rawP, truth, args.chunk, built, ("raw",) + pin_key)
            qs = model_quantities(rung, params, scanP, truth, args.chunk, built, ("scan",) + pin_key)
            models_raw.setdefault(name, {})[seed] = {q: v for q, v in qr.items() if q in KEEP}
            models_scan.setdefault(name, {})[seed] = {q: v for q, v in qs.items() if q in KEEP}
            print(f"  {name} seed {seed}: components sum on the scan to {np.max(np.abs(qs['_check_sum'])):.1e} kcal/mol  "
                  f"[{time.time() - t0:.0f} s]", flush=True)

    raw_masks = {"raw: all": np.ones(len(rawP), bool), "raw: on the training pairs": ~raw_off,
                 "raw: off them": raw_off, "raw: O-O 3.2-5 A": (raw_oo >= 3.2) & (raw_oo < 5.0)}
    scan_masks = {"scan: all": np.ones(len(scanP), bool), "scan: below O-O 5 A": scan_oo < 5.0,
                  **{f"scan: O-O {lab} A": (scan_oo >= lo) & (scan_oo < hi) for lab, lo, hi in SCAN_BANDS}}
    R_raw, R_scan = readings(models_raw, raw_ref, raw_masks), readings(models_scan, scan_ref, scan_masks)

    fmt = lambda v: f"{np.mean(v):.4f} [{np.min(v):.4f} {np.max(v):.4f}]"
    print("\nevery rung: the 2-body error (kcal/mol, seed mean [range]), the seeds' disagreement, the self terms' and "
          "E_0's sizes (seed means)")
    for R_, masks in ((R_raw, raw_masks), (R_scan, scan_masks)):
        for b in masks:
            first = next((R_["models"][r][b] for r in rungs if b in R_["models"].get(r, {})), None)
            if first is None:
                continue
            print(f"  [{b}, n {first['n']}]")
            for r in rungs:
                x = R_["models"].get(r, {}).get(b)
                if x:
                    print(f"    {r:<13} error {fmt(x['error'])}  disagreement {x['spread']:.4f}  self {np.mean(x['self']):.4f}  "
                          f"E0 {np.mean(x['E0']):.4f}")
    print("\nthe seeds' disagreement off the data (kcal/mol): on the training pairs / off them / the scan below O-O 5 A "
          "(and that against the first)")
    for r in rungs:
        on, off = (R_raw["models"].get(r, {}).get(b, {}).get("spread", float("nan")) for b in
                   ("raw: on the training pairs", "raw: off them"))
        sc = R_scan["models"].get(r, {}).get("scan: below O-O 5 A", {}).get("spread", float("nan"))
        print(f"  {r:<13} {on:.4f} / {off:.4f} / {sc:.4f}  ({sc / on:.1f} times)")
    print("\nthe split against a twin, per seed: S (the self terms' difference, rms), R (the energy difference = dE_0 + "
          "dself, rms), R/S; the error change X against B (%); seed means [range]")
    for R_, masks in ((R_raw, raw_masks), (R_scan, scan_masks)):
        for pair, by in R_["twins"].items():
            for b, rows in by.items():
                S, Rr = [x["S"] for x in rows], [x["R"] for x in rows]
                ratio = [x["R"] / x["S"] if x["S"] > 0 else float("nan") for x in rows]
                ch = [x["change"] for x in rows]
                print(f"  {pair:<26} {b:<28} S {np.mean(S):.4f}  R {np.mean(Rr):.4f}  R/S {np.nanmean(ratio):.2f}  "
                      f"error {np.mean(ch):+.0f} % [{np.min(ch):+.0f} {np.max(ch):+.0f}]  (check {max(x['check'] for x in rows):.0e})")

    verdict = None
    tw = R_scan["twins"].get("M_16C2pm_us - M_16C2pm_ws", {}).get("scan: below O-O 5 A")
    mu, mw = (R_scan["models"].get(r, {}).get("scan: below O-O 5 A") for r in ("M_16C2pm_us", "M_16C2pm_ws"))
    if tw and mu and mw:
        ch = [x["change"] for x in tw]
        k20 = sum(c >= 20 for c in ch)
        dis = mu["spread"] / mw["spread"]
        if k20 >= 3 and dis >= 1.3:
            verdict = "the split matters"
        elif abs(np.mean(ch)) <= 10 and dis <= 1.2:
            verdict = "the split does not matter"
        else:
            verdict = "neither"
        print(f"\nRegistered: on the scan below O-O 5 A, M_16C2pm_us against M_16C2pm_ws: error {[round(c) for c in ch]} % "
              f"(>= +20 % on {k20} of {len(ch)} seeds; mean {np.mean(ch):+.0f} %), the seeds' disagreement "
              f"{mu['spread']:.4f} against {mw['spread']:.4f} ({dis:.2f} times) -> {verdict}")
    report = {"threshold": float(thr), "tally": {f"{d:g}": v for d, v in tally.items()}, "raw": R_raw, "scan": R_scan,
              "verdict": verdict}
    with open(os.path.join(args.results, "split_check.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    out = {"scan__P": scanP, "scan__reference": scan_ref, "scan__oo": scan_oo, "scan__nn": scan_nn,
           "raw__nn": raw_nn, "raw__off": raw_off}
    for name, per_seed in models_scan.items():
        for s, q in per_seed.items():
            for k, v in q.items():
                out[f"{name}__s{s}__scan__{k}"] = v
    np.savez_compressed(os.path.join(args.results, "split_check_frames.npz"), **out)
    print(f"\ndone [{time.time() - t0:.0f} s]; {os.path.join(args.results, 'split_check.json')}")


if __name__ == "__main__":
    main()
