"""
e4_job16_6.py -- E4 job 16.6 (notes/e4-job16.6-preregistration.md; v0.0.46): is MACE-LES's size dependence LES's
k-space cutoff?  No training: the trained models' own latent charges, on the frames job 16.5 read.  NumPy (torch on the
GPU for the sums when it is there).

    python scripts/e4_job16_6.py tails      # every MACE-LES model of job 16.5, and the default's slabs of job 17
    python scripts/e4_job16_6.py read

LES's periodic Ewald (les 0.2.0, les/module/ewald.py's legacy path, the one baselines/mace/les.yaml leaves the models
on) sums the latent charges, as Gaussians of width sigma = 1 A, in reciprocal space alone, over 0 < |k| <= k_max =
2 pi / dl = pi / A (dl = 2 A):

    E_LES = (norm / V) sum_{0 < |k| <= k_max} exp(-sigma^2 k^2 / 2) / k^2 |S(k)|^2 - norm sum_i q_i^2 / (sigma (2 pi)^1.5)

with S(k) = sum_i q_i exp(i k r_i), norm = 90.4756 eV A, every pair of the cell (a molecule's own too) in it.  What it
leaves out, the tail T = (norm / V) sum_{k_max < |k| <= 2 k_max} ..., is about 0.05 kcal/mol per molecule of water:
exp(-pi^2 / 2) is not small, and S(k) near pi / A is the molecules' own charge distribution.  Its continuum part is the
same in any cell, and training absorbs it.  But the k vectors just inside pi / A lie on the cell's reciprocal lattice,
and how much of the shell they fill does not vary smoothly with the cell: a model trained at one size carries the
difference into the others,

    b_L0(L) = g_L0(L) + Tbar(L0) - T(L)

with g the error of the same model without the cutoff, Tbar(L0) the tail its training absorbed (the mean on its
training frames) and T(L) the tail on the frames read.  The corrected bias c_L0(L) = b_L0(L) + T(L) is then g + Tbar.

`tails` writes, per model, results/e4_job16_6/<run>_tails.npz: for every data set <split>_n<n> (and job17_slabs) of
its energies file, <set>__tail and <set>__les (F,), eV: T and E_LES per frame, from the charges energies.py stored.

`read` reads K16.6.1-K16.6.4 (kcal/mol per molecule; seed means; MACE-LES's own, unpaired bias on the frames job 16.5
reads as unseen):

    K16.6.1  the reversal is the tail: the corrected step from 256 to 1024 molecules, c_64(1024) - c_64(256), within
             K16_6_1 of zero (uncorrected, job 16.5: +0.0048)
    K16.6.2  most of the drift is the tail: |c_64(256) - c_64(64)| at most K16_6_2 times |b_64(256) - b_64(64)|
    K16.6.3  the training size's shift is the tail: c_128(n) - c_64(n) within K16_6_3 of Tbar_128(128) - Tbar_64(64)
             at every size, 64 to 1024 (uncorrected, job 16.5: +0.0040 to +0.0053)
    K16.6.4  job 17's slabs: the default MACE-LES's tail change between each gap and the widest, rms over the frames,
             at most K16_6_4 of MB-pol's E(g) - E(widest), at 9, 12 and 16 A, both seeds
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_job16_5 as J5                                    # noqa: E402

KCAL_PER_EV = 23.060547830619026
NORM, SIGMA, DL = 90.4756, 1.0, 2.0                        # LES's Ewald, baselines/mace/les.yaml
K_IN = 2 * np.pi / DL                                      # LES's cutoff, pi / A
K_OUT = 2 * K_IN                                           # the tail's outer edge: there exp(-k^2/2)/k^2 is 7e-11
SIZES = ("64", "128", "256", "512", "1024")
OUT_DIR = "results/e4_job16_6"
SLABS = "data/e4/job17_slabs_n64.npz"
SLAB_STEM = "job17_slabs"
K16_6_1, K16_6_2, K16_6_3, K16_6_4 = 0.0015, 0.5, 0.0015, 0.10
BACKEND = {"torch": None}


# ------------------------------------------------------------------ LES's reciprocal sum ----------------------------
def kvectors(cell, kmin, kmax):
    """the half-space k vectors with kmin < |k| <= kmax (k != 0), and |k|^2: a grid with |n_i| <= kmax |a_i| / 2 pi
    holds every one of them in any cell."""
    cell = np.asarray(cell, float)
    G = 2 * np.pi * np.linalg.inv(cell).T
    N = [int(np.ceil(kmax * np.linalg.norm(a) / (2 * np.pi))) + 1 for a in cell]
    g = [np.arange(-n, n + 1) for n in N]
    nv = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    k = nv.astype(float) @ G
    k2 = np.sum(k ** 2, axis=1)
    m = (k2 > kmin ** 2) & (k2 <= kmax ** 2) & (k2 > 0)
    nv, k, k2 = nv[m], k[m], k2[m]
    first = np.argmax(nv != 0, axis=1)
    half = nv[np.arange(len(nv)), first] > 0
    return k[half], k2[half]


def _sum_s2(q, r, k, w, chunk=4096):
    """sum_k w_k |S(k)|^2, S(k) = sum_i q_i exp(i k.r_i)."""
    t = BACKEND["torch"]
    if t is not None:
        dev = "cuda" if t.cuda.is_available() else "cpu"
        q_, r_ = t.as_tensor(q, dtype=t.float64, device=dev), t.as_tensor(r, dtype=t.float64, device=dev)
        k_, w_ = t.as_tensor(k, dtype=t.float64, device=dev), t.as_tensor(w, dtype=t.float64, device=dev)
        out = t.zeros((), dtype=t.float64, device=dev)
        for c in range(0, len(k_), chunk):
            kr = r_ @ k_[c:c + chunk].T
            out = out + t.sum(w_[c:c + chunk] * ((q_ @ t.cos(kr)) ** 2 + (q_ @ t.sin(kr)) ** 2))
        return float(out)
    out = 0.0
    for c in range(0, len(k), chunk):
        kr = r @ k[c:c + chunk].T
        out += float(np.sum(w[c:c + chunk] * ((q @ np.cos(kr)) ** 2 + (q @ np.sin(kr)) ** 2)))
    return out


def recip(q, r, cell, kmin=0.0, kmax=K_IN, sigma=SIGMA, norm=NORM):
    """(norm / V) sum_{kmin < |k| <= kmax} exp(-sigma^2 k^2 / 2) / k^2 |S(k)|^2, eV (both half-spaces)."""
    q, r = np.asarray(q, float).reshape(-1), np.asarray(r, float).reshape(-1, 3)
    k, k2 = kvectors(cell, kmin, kmax)
    if not len(k):
        return 0.0
    w = 2.0 * np.exp(-sigma ** 2 * k2 / 2) / k2
    return _sum_s2(q, r, k, w) / abs(np.linalg.det(np.asarray(cell, float))) * norm


def les_energy(q, r, cell, dl=DL, sigma=SIGMA, norm=NORM):
    """LES's energy of charges q at r in a periodic cell (its legacy path, self-interaction removed), eV."""
    q = np.asarray(q, float).reshape(-1)
    return recip(q, r, cell, 0.0, 2 * np.pi / dl, sigma, norm) - norm * np.sum(q ** 2) / (sigma * (2 * np.pi) ** 1.5)


def tail(q, r, cell, dl=DL, sigma=SIGMA, norm=NORM):
    """what LES's cutoff leaves out: the same sum over 2 pi / dl < |k| <= 4 pi / dl, eV."""
    return recip(q, r, cell, 2 * np.pi / dl, 4 * np.pi / dl, sigma, norm)


# ------------------------------------------------------------------ tails: the models' charges, per frame -----------
def set_tails(z, name, positions, cells):
    """per frame of data set `name` in an energies.py npz: (T, E_LES) in eV, from its stored latent charges."""
    q, off = np.asarray(z[f"{name}__q"], float), np.asarray(z[f"{name}__offsets"], int)
    if len(off) - 1 != len(positions):
        raise ValueError(f"{name}: {len(off) - 1} frames of charges for {len(positions)} frames of positions")
    T, E = np.zeros(len(positions)), np.zeros(len(positions))
    for f in range(len(positions)):
        qf, rf, cf = q[off[f]:off[f + 1]], np.asarray(positions[f], float), np.asarray(cells[f], float)
        T[f], E[f] = tail(qf, rf, cf), les_energy(qf, rf, cf)
    return T, E


def models(dirs=J5.DIRS):
    """{L0: {seed: (stem, energies npz path)}} of job 16.5's MACE-LES runs found."""
    out = {}
    for L0 in J5.L0S:
        for s in J5.SEEDS:
            stem = J5.STEMS["maceles"].format(L0=L0, seed=s)
            p = os.path.join(dirs["baselines"], stem, f"{stem}_stagetwo_{J5.TAG}.npz")
            if os.path.exists(p):
                out.setdefault(L0, {})[s] = (stem, p)
    return out


def run_tails(a):
    if a.backend in ("auto", "torch"):
        try:
            import torch
            BACKEND["torch"] = torch
        except ImportError:
            if a.backend == "torch":
                raise
    print(f"backend: {'torch, ' + ('cuda' if BACKEND['torch'].cuda.is_available() else 'cpu') if BACKEND['torch'] else 'numpy'}",
          flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    dirs = {**J5.DIRS, "data": a.data, "baselines": a.baselines}
    jobs = [(stem, path, None) for by in models(dirs).values() for stem, path in by.values()]
    for s in J5.SEEDS:                                     # the default MACE-LES's slabs (job 17)
        stem = f"maceles_n64_s{s}"
        p = os.path.join(a.baselines, stem, f"{stem}_stagetwo_job17.npz")
        if os.path.exists(p) and os.path.exists(a.slabs):
            jobs.append((stem + "_slabs", p, a.slabs))
    for stem, path, slabs in jobs:
        out = os.path.join(a.out_dir, f"{stem}_tails.npz")
        have = dict(np.load(out)) if os.path.exists(out) else {}
        z = np.load(path, allow_pickle=False)
        names = [SLAB_STEM] if slabs else sorted(k[:-len("__q")] for k in z.files if k.endswith("__q"))
        for name in names:
            if f"{name}__tail" in have:
                continue
            t0 = time.time()
            if slabs:
                S = np.load(slabs)
                pos, cells = S["positions"], S["cell"]
            else:
                d = np.load(os.path.join(a.data, f"{name}.npz"))
                pos, cells = d["positions"], np.broadcast_to(d["cell"], (len(d["positions"]), 3, 3))
            T, E = set_tails(z, name, pos, cells)
            have[f"{name}__tail"], have[f"{name}__les"] = T, E
            np.savez(out, **have)
            n = pos.shape[1] // 3
            print(f"  {stem} {name}: {len(T)} frames, tail {np.mean(T) / n * KCAL_PER_EV:.5f} kcal/mol per molecule "
                  f"(sd {np.std(T) / n * KCAL_PER_EV:.5f}), LES {np.mean(E) / n * KCAL_PER_EV:+.4f}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        print(f"written {out}", flush=True)


# ------------------------------------------------------------------ the readings ------------------------------------
def frame_tails(t, n, L0):
    """the tail per molecule (kcal/mol) on the frames job 16.5 reads as unseen at size n, in its order; None if any
    set is missing."""
    out = []
    for split in J5.unseen(L0, n):
        if f"{split}_n{n}__tail" not in t:
            return None
        out.append(np.asarray(t[f"{split}_n{n}__tail"], float) / int(n) * KCAL_PER_EV)
    return np.concatenate(out)


def trained_tail(t, L0):
    k = f"train_n{L0}__tail"
    return float(np.mean(np.asarray(t[k], float))) / int(L0) * KCAL_PER_EV if k in t else None


def evaluate(runs, tails, data_dir, slabs=None):
    """runs: job 16.5's (J5.load); tails: {L0: {seed: the tails npz as a dict}}; slabs: None or {"S": the slabs'
    npz, "z": {seed: job 17 energies npz}, "t": {seed: their tails npz}}."""
    ev = {"b": {}, "t": {}, "c": {}, "trained": {}}
    for L0 in J5.L0S:
        for key in ("b", "t", "c"):
            ev[key][L0] = {}
        ev["trained"][L0] = None
        tr = [trained_tail(tails.get(L0, {}).get(s, {}), L0) for s in J5.SEEDS]
        if all(x is not None for x in tr):
            ev["trained"][L0] = float(np.mean(tr))
        for n in SIZES:
            b, t = [], []
            for s in J5.SEEDS:
                r = runs.get(L0, {}).get(s) or {}
                e = J5.model_errors(r["z"], data_dir, n, L0) if r.get("z") is not None else None
                x = frame_tails(tails.get(L0, {}).get(s, {}), n, L0)
                if e is None or x is None or len(e) != len(x):
                    b = None
                    break
                b.append(float(np.mean(e))); t.append(float(np.mean(x)))
            ev["b"][L0][n] = float(np.mean(b)) if b else None
            ev["t"][L0][n] = float(np.mean(t)) if b else None
            ev["c"][L0][n] = ev["b"][L0][n] + ev["t"][L0][n] if b else None
    b, c = ev["b"][64], ev["c"][64]
    ok = lambda *v: all(x is not None for x in v)
    k1 = {"band": K16_6_1, "corrected": c["1024"] - c["256"] if ok(c["1024"], c["256"]) else None,
          "uncorrected": b["1024"] - b["256"] if ok(b["1024"], b["256"]) else None}
    k1["holds"] = bool(abs(k1["corrected"]) <= K16_6_1) if k1["corrected"] is not None else None
    k2 = {"band": K16_6_2, "corrected": c["256"] - c["64"] if ok(c["256"], c["64"]) else None,
          "uncorrected": b["256"] - b["64"] if ok(b["256"], b["64"]) else None}
    k2["holds"] = (bool(abs(k2["corrected"]) <= K16_6_2 * abs(k2["uncorrected"]))
                   if k2["corrected"] is not None and k2["uncorrected"] is not None else None)
    tb = ev["trained"]
    shift = tb[128] - tb[64] if ok(tb[128], tb[64]) else None
    diff = {n: ev["c"][128][n] - ev["c"][64][n] if ok(ev["c"][128][n], ev["c"][64][n]) else None for n in SIZES}
    raw = {n: ev["b"][128][n] - ev["b"][64][n] if ok(ev["b"][128][n], ev["b"][64][n]) else None for n in SIZES}
    k3 = {"band": K16_6_3, "absorbed": shift, "corrected": diff, "uncorrected": raw,
          "holds": (bool(all(abs(v - shift) <= K16_6_3 for v in diff.values()))
                    if shift is not None and all(v is not None for v in diff.values()) else None)}
    ev["K16.6.1"], ev["K16.6.2"], ev["K16.6.3"] = k1, k2, k3
    ev["K16.6.4"] = slab_reading(slabs) if slabs else {"band": K16_6_4, "holds": None, "ratio": {}}
    return ev


def slab_reading(slabs):
    """K16.6.4: the tail's change with the gap (per cell, kcal/mol) against MB-pol's E(g) - E(widest), and job 17's
    K17.8 with the tail added back, beside."""
    import e4_job17 as J17
    S = slabs["S"]
    frame, gap = np.asarray(S["slab_frame"]), np.asarray(S["slab_gap"], float)
    truth = J17.slab_changes(np.asarray(S["energy"], float) * KCAL_PER_EV * J17.EV_PER_HARTREE, frame, gap)
    out = {"band": K16_6_4, "ratio": {}, "k17_8": {}, "holds": None}
    for s, t in slabs["t"].items():
        if f"{SLAB_STEM}__tail" not in t or s not in slabs["z"]:
            continue
        T = np.asarray(t[f"{SLAB_STEM}__tail"], float) * KCAL_PER_EV
        dT = J17.slab_changes(T, frame, gap)
        out["ratio"][s] = {g: J17.rms(dT[g]) / J17.rms(truth[g]) for g in J17.SLAB_READ if g in dT}
        E = np.asarray(slabs["z"][s][f"{SLAB_STEM}__energy"], float) * KCAL_PER_EV
        dm, dc = J17.slab_changes(E, frame, gap), J17.slab_changes(E + T, frame, gap)
        out["k17_8"][s] = {g: (J17.rel_rmse(dm[g], truth[g]), J17.rel_rmse(dc[g], truth[g])) for g in J17.SLAB_READ
                           if g in dm}
    if len(out["ratio"]) == len(J5.SEEDS):
        out["holds"] = bool(all(v <= K16_6_4 for x in out["ratio"].values() for v in x.values()))
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev):
    f = lambda x, d=4: "-" if x is None else f"{x:+.{d}f}"
    L = ["E4, job 16.6 (notes/e4-job16.6-preregistration.md): MACE-LES's size dependence and LES's k-space cutoff",
         "  MACE-LES's own bias b, the tail its cutoff leaves out T, and b + T (kcal/mol per molecule, unseen frames, "
         "seed means)"]
    for L0 in J5.L0S:
        L.append(f"    L_0 = {L0} (its training frames' tail {f(ev['trained'][L0], 5)}): " + "; ".join(
            f"n {n} b {f(ev['b'][L0][n])} T {f(ev['t'][L0][n], 5)} b+T {f(ev['c'][L0][n])}" for n in SIZES))
    k = ev["K16.6.1"]
    L.append(f"  K16.6.1: {_verdict(k['holds'])} (the reversal is the tail: b + T from 256 to 1024 within "
             f"{k['band']:g} of zero): {f(k['corrected'])} (b alone {f(k['uncorrected'])})")
    k = ev["K16.6.2"]
    L.append(f"  K16.6.2: {_verdict(k['holds'])} (most of the drift is the tail: b + T from 64 to 256 at most "
             f"{k['band']:g} of b's): {f(k['corrected'])} against {f(k['uncorrected'])}")
    k = ev["K16.6.3"]
    L.append(f"  K16.6.3: {_verdict(k['holds'])} (the training size's shift is the tail: L_0 = 128 less 64, b + T, "
             f"within {k['band']:g} of the trained tails' difference {f(k['absorbed'], 5)}): "
             + ", ".join(f"n {n} {f(v)} (b alone {f(k['uncorrected'][n])})" for n, v in k["corrected"].items()))
    k = ev["K16.6.4"]
    L.append(f"  K16.6.4: {_verdict(k['holds'])} (job 17's slabs: the tail's change with the gap at most "
             f"{100 * k['band']:g} % of MB-pol's E(g) - E(widest)): " + "; ".join(
                 f"seed {s} " + " / ".join(f"{100 * v:.1f} %" for v in x.values()) for s, x in k["ratio"].items()))
    if k.get("k17_8"):
        L.append("    beside: K17.8's relative rmse at 9 / 12 / 16 A, as read and with the tail added: " + "; ".join(
            f"seed {s} " + " / ".join(f"{a:.3f} -> {b:.3f}" for a, b in x.values()) for s, x in k["k17_8"].items()))
    return "\n".join(L)


def load_tails(out_dir=OUT_DIR, dirs=J5.DIRS):
    out = {}
    for L0, by in models(dirs).items():
        for s, (stem, _) in by.items():
            p = os.path.join(out_dir, f"{stem}_tails.npz")
            if os.path.exists(p):
                out.setdefault(L0, {})[s] = dict(np.load(p))
    return out


def load_slabs(out_dir=OUT_DIR, baselines=J5.DIRS["baselines"], slabs=SLABS):
    if not os.path.exists(slabs):
        return None
    z, t = {}, {}
    for s in J5.SEEDS:
        stem = f"maceles_n64_s{s}"
        pz, pt = os.path.join(baselines, stem, f"{stem}_stagetwo_job17.npz"), os.path.join(out_dir, f"{stem}_slabs_tails.npz")
        if os.path.exists(pz) and os.path.exists(pt):
            z[s], t[s] = dict(np.load(pz)), dict(np.load(pt))
    return {"S": dict(np.load(slabs)), "z": z, "t": t} if t else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tails", "read"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=J5.DIRS["data"])
        p.add_argument("--baselines", default=J5.DIRS["baselines"])
        p.add_argument("--out_dir", default=OUT_DIR)
        p.add_argument("--slabs", default=SLABS)
    sub.choices["tails"].add_argument("--backend", default="auto", choices=("auto", "numpy", "torch"))
    for k in ("dagger64", "dagger128"):
        sub.choices["read"].add_argument(f"--{k}", default=J5.DIRS[k])
    sub.choices["read"].add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "tails":
        return run_tails(a)
    dirs = {"data": a.data, "baselines": a.baselines, "dagger64": a.dagger64, "dagger128": a.dagger128}
    ev = evaluate(J5.load(dirs), load_tails(a.out_dir, dirs), a.data, load_slabs(a.out_dir, a.baselines, a.slabs))
    print(report(ev))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(ev, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
