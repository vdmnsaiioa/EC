"""
e4_job16_5.py -- E4 job 16.5's registered readings K16.5.1-K16.5.4 (notes/e4-job16.5-preregistration.md; v0.0.44):
MACE-LES trained on cells of 128 molecules, job 16's L_0 test for the external baseline.  NumPy only.

    python scripts/e4_job16_5.py

MACE-LES at L_0 = 64 is the baseline (results/e4_baselines/maceles_n64_s<seed>); at L_0 = 128 it is trained the same
way on the 90 frames of 128 molecules (baselines/mace/train.sbatch, N0=128: results/e4_baselines/maceles_n128_s<seed>).
Each model's energies on every frame of 64, 128 and 256 molecules come from baselines/mace/energies.py --tag job16_5
(<model stem>_job16_5.npz, one entry per data set <split>_n<n>); its latent dipole and forces from eval.py's json.

The bias at size L is the mean energy error per molecule on the frames the training did not see: at L_0 the test split
(MACE's validation split chooses its checkpoints), at any other size every split.  delta_L0(L) is MACE-LES's bias less
the field-term dagger's trained at the same L_0 (job 11's at 64, job 16's at 128), on the same frames, seed by seed,
then the seed mean; its step from L_0 to L is s_L0(L) = delta_L0(L) - delta_L0(L_0), as K16.1 read M_inf's.

    K16.5.1  MACE-LES's size error is one function of the cells, the same at both L_0: s_128(64) against s_64(128) --
             opposite in sign, its magnitude within K16_5_1 (relative) of it; read when |s_64(128)| is at least
             K16_5_1_MIN (K16.1's construction)
    K16.5.2  its law: r = s_128(256) / s_64(256) -- "carried" (the training cell's energy carried, as M_inf's: r at most
             K16_5_2[0]; job 16: 0.23), "1/L" (an error in 1/L of the test cell: r in [K16_5_2[1], K16_5_2[2]]; 1/L gives
             0.44), "persists" (above K16_5_2[2]), "between" or "reversed" (r < 0)
    K16.5.3  its scale does not depend on L_0: the latent dipole (test frames of 64 molecules) at L_0 = 128 within
             K16_5_3 (relative) of L_0 = 64's, seed by seed
    K16.5.4  its forces at L_0 = 128 within K16_5_4 (relative) of L_0 = 64's on the test frames of 64, 128 and 256
             molecules (seed means)
    K16.5.5  the control, read if it was trained (MODEL=MACE): MACE without LES, a cutoff model as M_inf is, reads
             K16.5.2's "carried"

Beside them, not registered: K16.5.2's ratio as the L_0 = 64 runs alone predict it if K16.5.1 holds (1 - s_64(128) /
s_64(256)); the two L_0's difference on the same frames, MACE-LES at 128 less at 64, at every size to 1024 molecules
(one function of the cells makes it the same at every size; no dagger is needed, so it reaches the 512- and 1024-molecule
test frames the harness does not read); the net latent charge; MACE without LES's other readings.

Added beside before any job 16.5 number (28 Nov, after job 16's read, notes/e4-job16-results.md section 4: its frames'
standard errors are 2-3 times what K16.5.2 assumed): r's frame standard error; r from the frames off both L_0 alone,
(delta_64(256) - delta_64(128)) / (delta_128(256) - delta_128(64)); and the law where Form II reads it, MACE-LES's own
unpaired bias at 256, 512 and 1024 molecules: from 256 to 1024 an error in 1/L moves it LAW_FAR (0.63) times its step
from 64 to 256 again, a carried one about none.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_job7 as J7                                   # noqa: E402

KCAL = 627.5094740631
EV_PER_HARTREE = 27.211386245988
PERMANENT_DEBYE = 1.864
SIZES = ("64", "128", "256")
BIG = ("512", "1024")                                  # test frames only; read beside, MACE-LES against itself
SPLITS = ("train", "valid", "test")
SEEDS = (0, 1)
L0S = (64, 128)
TAG = "job16_5"
DAGGER = "M_16pm_us_d20"
DIRS = {"data": "data/e4", "baselines": "results/e4_baselines", "dagger64": "results/e4_job11",
        "dagger128": "results/e4_job16_128"}
STEMS = {"maceles": "maceles_n{L0}_s{seed}", "mace": "mace_n{L0}_s{seed}"}      # the registered model, and MACE beside
K16_5_1, K16_5_1_MIN, K16_5_2, K16_5_3, K16_5_4 = 0.4, 0.003, (0.30, 0.35, 0.80), 0.10, 0.05
LAW = {"1/L": (1 / 19.72 - 1 / 15.65) / (1 / 19.72 - 1 / 12.42), "1/N": (1 / 256 - 1 / 128) / (1 / 256 - 1 / 64)}
LAW_FAR = (1 / 31.30 - 1 / 19.72) / (1 / 19.72 - 1 / 12.42)  # 1/L: the step 256 -> 1024 over the step 64 -> 256


def unseen(L0, n):
    return ("test",) if int(n) == int(L0) or n in BIG else SPLITS


def truth(data_dir, n, split):
    return np.asarray(np.load(os.path.join(data_dir, f"{split}_n{n}.npz"))["energy"], float)


def model_errors(z, data_dir, n, L0):
    """the model's energy error per molecule (kcal/mol) on the unseen frames of size n, in split order; None when a set
    is missing."""
    out = []
    for split in unseen(L0, n):
        key = f"{split}_n{n}__energy"
        if key not in z:
            return None
        E = np.asarray(z[key], float) / EV_PER_HARTREE
        out.append((E - truth(data_dir, n, split)) * KCAL / int(n))
    return np.concatenate(out)


def dagger_errors(r, n, L0):
    cells = r["readings"]["cells"].get(str(n), {})
    out = []
    for split in unseen(L0, n):
        v = cells.get(split, {})
        if "energy_per_molecule_errors" not in v:
            return None
        out += list(v["energy_per_molecule_errors"])
    return np.asarray(out, float)


def load(dirs=DIRS, group="maceles"):
    """{L0: {seed: {"z": energies npz or None, "eval": eval json or None, "dagger": dagger run or None}}}"""
    dag = {64: J7.load_runs(dirs["dagger64"]).get(DAGGER, {}), 128: J7.load_runs(dirs["dagger128"]).get(DAGGER, {})}
    out = {}
    for L0 in L0S:
        for s in SEEDS:
            stem = STEMS[group].format(L0=L0, seed=s)
            base = os.path.join(dirs["baselines"], stem, f"{stem}_stagetwo")
            z = dict(np.load(base + f"_{TAG}.npz", allow_pickle=False)) if os.path.exists(base + f"_{TAG}.npz") else None
            ev = json.load(open(base + "_eval.json")) if os.path.exists(base + "_eval.json") else None
            out.setdefault(L0, {})[s] = {"z": z, "eval": ev, "dagger": dag[L0].get(s)}
    return out


def delta(runs, data_dir, n, L0):
    """the model's bias less the dagger's on the same unseen frames, seed by seed: (seed mean, sd, frame se) and the
    unpaired bias's seed mean; None when anything is missing."""
    d, se, raw = [], [], []
    for s in SEEDS:
        r = runs.get(L0, {}).get(s) or {}
        if r.get("z") is None or r.get("dagger") is None:
            return None
        a, b = model_errors(r["z"], data_dir, n, L0), dagger_errors(r["dagger"], n, L0)
        if a is None or b is None or len(a) != len(b):
            return None
        x = a - b
        d.append(float(np.mean(x))); se.append(float(np.std(x) / np.sqrt(len(x)))); raw.append(float(np.mean(a)))
    return {"mean": float(np.mean(d)), "sd": float(np.std(d)), "se": float(np.mean(se)), "bias": float(np.mean(raw))}


def between(runs, data_dir, n):
    """MACE-LES at L_0 = 128 less at 64 on the frames of size n neither saw, seed by seed, then (seed mean, sd, frame
    se); None when anything is missing."""
    d, se = [], []
    for s in SEEDS:
        a, b = (runs.get(L0, {}).get(s) or {} for L0 in (128, 64))
        if a.get("z") is None or b.get("z") is None:
            return None
        L0 = int(n) if n in ("64", "128") else 64      # the test split at 64 and 128; every split at 256
        x, y = model_errors(a["z"], data_dir, n, L0), model_errors(b["z"], data_dir, n, L0)
        if x is None or y is None or len(x) != len(y):
            return None
        d.append(float(np.mean(x - y))); se.append(float(np.std(x - y) / np.sqrt(len(x))))
    return {"mean": float(np.mean(d)), "sd": float(np.std(d)), "se": float(np.mean(se))}


def bias(runs, data_dir, n, L0):
    """MACE-LES's own (unpaired) bias at size n, trained at L0: (seed mean, frame se); None when anything is missing."""
    m, se = [], []
    for s in SEEDS:
        r = runs.get(L0, {}).get(s) or {}
        x = model_errors(r["z"], data_dir, n, L0) if r.get("z") is not None else None
        if x is None:
            return None
        m.append(float(np.mean(x))); se.append(float(np.std(x) / np.sqrt(len(x))))
    return {"mean": float(np.mean(m)), "se": float(np.mean(se))}


def _ratio(a, sa, b, sb):
    """a / b and its standard error, a and b independent."""
    if a is None or b is None or sa is None or sb is None or not b:
        return None, None
    return a / b, float(np.hypot(sa / b, a * sb / b ** 2))


def dipole(r):
    s = ((r or {}).get("eval") or {}).get("sets", {}).get("test_n64", {})
    return s["latent_dipole_debye_mean"] / PERMANENT_DEBYE if "latent_dipole_debye_mean" in s else None


def forces(runs, L0, n):
    v = [((runs.get(L0, {}).get(s) or {}).get("eval") or {}).get("sets", {}).get(f"test_n{n}", {}).get(
        "f_rmse_kcal_per_mol_A") for s in SEEDS]
    return float(np.mean(v)) if all(x is not None for x in v) else None


def evaluate(runs, data_dir):
    ev = {"delta": {L0: {n: delta(runs, data_dir, n, L0) for n in SIZES} for L0 in L0S}}
    d = ev["delta"]
    step = lambda L0, n: (d[L0][n]["mean"] - d[L0][str(L0)]["mean"]) if d[L0][n] and d[L0][str(L0)] else None
    s64_128, s128_64, s64_256, s128_256 = step(64, "128"), step(128, "64"), step(64, "256"), step(128, "256")
    ev["steps"] = {"s64(128)": s64_128, "s128(64)": s128_64, "s64(256)": s64_256, "s128(256)": s128_256}
    k1 = {"band": K16_5_1, "min": K16_5_1_MIN, "holds": None, "too_small": None}
    if s64_128 is not None and s128_64 is not None:
        k1["too_small"] = abs(s64_128) < K16_5_1_MIN
        k1["ratio"] = -s128_64 / s64_128 if s64_128 else None
        if not k1["too_small"]:
            k1["holds"] = bool(s128_64 * s64_128 < 0 and abs(s128_64 + s64_128) <= K16_5_1 * abs(s64_128))
    ev["K16.5.1"] = k1
    k2 = {"band": K16_5_2, "laws": LAW, "ratio": s128_256 / s64_256 if s64_256 and s128_256 is not None else None}
    r = k2["ratio"]
    k2["from_64"] = 1.0 - s64_128 / s64_256 if s64_256 and s64_128 is not None else None
    sse = lambda L0, n: float(np.hypot(d[L0][n]["se"], d[L0][str(L0)]["se"])) if d[L0][n] and d[L0][str(L0)] else None
    k2["se"] = _ratio(s128_256, sse(128, "256"), s64_256, sse(64, "256"))[1]
    dd = lambda L0, a, b: ((d[L0][a]["mean"] - d[L0][b]["mean"], float(np.hypot(d[L0][a]["se"], d[L0][b]["se"])))
                           if d[L0][a] and d[L0][b] else (None, None))
    k2["off_L0"] = dict(zip(("ratio", "se"), _ratio(*dd(64, "256", "128"), *dd(128, "256", "64"))))
    k2["verdict"] = (None if r is None else "reversed" if r < 0 else "carried" if r <= K16_5_2[0] else
                     "between" if r < K16_5_2[1] else "1/L" if r <= K16_5_2[2] else "persists")
    ev["K16.5.2"] = k2
    c = {L0: {s: dipole(runs.get(L0, {}).get(s)) for s in SEEDS} for L0 in L0S}
    rel = {s: c[128][s] / c[64][s] - 1.0 for s in SEEDS if c[64][s] is not None and c[128][s] is not None}
    ev["K16.5.3"] = {"band": K16_5_3, "c": c, "relative": rel,
                     "holds": bool(all(abs(x) <= K16_5_3 for x in rel.values())) if len(rel) == len(SEEDS) else None}
    f = {L0: {n: forces(runs, L0, n) for n in SIZES} for L0 in L0S}
    frel = {n: f[128][n] / f[64][n] - 1.0 for n in SIZES if f[64][n] and f[128][n]}
    ev["K16.5.4"] = {"band": K16_5_4, "forces": f, "relative": frel,
                     "holds": bool(all(abs(x) <= K16_5_4 for x in frel.values())) if len(frel) == len(SIZES) else None}
    ev["between"] = {n: between(runs, data_dir, n) for n in SIZES + BIG}
    far = {L0: {n: bias(runs, data_dir, n, L0) for n in ("64", "256") + BIG} for L0 in L0S}
    b64 = far[64]
    pred = (LAW_FAR * (b64["256"]["mean"] - b64["64"]["mean"]) if b64["256"] and b64["64"] else None)
    ev["far_law"] = {"bias": far, "law": LAW_FAR, "predicted_1/L": pred,
                     "step": {L0: ((by["1024"]["mean"] - by["256"]["mean"], float(np.hypot(by["1024"]["se"], by["256"]["se"])))
                                   if by["1024"] and by["256"] else None) for L0, by in far.items()}}
    ev["net_charge_rms"] = {L0: {s: (((runs.get(L0, {}).get(s) or {}).get("eval") or {}).get("sets", {})
                                     .get("test_n64", {}).get("latent_q_per_molecule_rms")) for s in SEEDS} for L0 in L0S}
    return ev


def control(mace_ev):
    """K16.5.5 from MACE without LES's readings (None: not trained)."""
    v = (mace_ev or {}).get("K16.5.2", {}).get("verdict")
    return {"verdict": v, "holds": None if v is None else v == "carried"}


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev, beside=None):
    f = lambda x, d=4: "-" if x is None else f"{x:+.{d}f}"
    g = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    L = ["E4, job 16.5 (notes/e4-job16.5-preregistration.md): MACE-LES trained on cells of 128 molecules",
         "  MACE-LES's bias less the dagger's (kcal/mol per molecule, unseen frames; seed mean, sd; frame se; the "
         "unpaired bias)"]
    for L0 in L0S:
        L.append(f"    L_0 = {L0}: " + "; ".join(
            f"n {n} " + (f"{v['mean']:+.4f} ({v['sd']:.4f}; {v['se']:.4f}; {v['bias']:+.4f})" if v else "-")
            for n, v in ev["delta"][L0].items()))
    st = ev["steps"]
    k = ev["K16.5.1"]
    L.append(f"  K16.5.1: {_verdict(k['holds'])} (its size error one function of the cells: s_128(64) opposite to s_64(128), "
             f"within {100 * k['band']:g} %; read when |s_64(128)| >= {k['min']:g}): s_64(128) {f(st['s64(128)'])}, "
             f"s_128(64) {f(st['s128(64)'])}" + (f", ratio {k['ratio']:.2f}" if k.get("ratio") is not None else "")
             + (" (too small to read)" if k["too_small"] else ""))
    k = ev["K16.5.2"]
    b = k["band"]
    L.append(f"  K16.5.2: {k['verdict'] or 'not read'} (its law, r = s_128(256) / s_64(256): carried at most {b[0]:g}, 1/L "
             f"{b[1]:g}-{b[2]:g}): s_64(256) {f(st['s64(256)'])}, s_128(256) {f(st['s128(256)'])}, r {g(k['ratio'], 2)} "
             f"(a 1/L law gives {k['laws']['1/L']:.2f}, 1/N {k['laws']['1/N']:.2f}; M_inf, job 16: 0.23)")
    k = ev["K16.5.3"]
    L.append(f"  K16.5.3: {_verdict(k['holds'])} (its latent dipole at L_0 = 128 within {100 * k['band']:g} % of 64's): "
             + "; ".join(f"seed {s} c {g(k['c'][64][s])} -> {g(k['c'][128][s])}" for s in SEEDS))
    k = ev["K16.5.4"]
    L.append(f"  K16.5.4: {_verdict(k['holds'])} (its forces at L_0 = 128 within {100 * k['band']:g} % of 64's, test "
             "frames): " + " / ".join(f"{100 * v:+.1f}" for v in k["relative"].values()) + " %"
             + " (" + "; ".join(f"L_0 = {L0} " + " / ".join(g(x) for x in k["forces"][L0].values()) for L0 in L0S) + ")")
    L.append("\n  Beside them, not registered")
    k2 = ev["K16.5.2"]
    L.append(f"    K16.5.2's r as the L_0 = 64 runs alone predict it (1 - s_64(128) / s_64(256)): {g(k2['from_64'], 2)}; "
             f"r's frame se {g(k2.get('se'), 2)}; r from the frames off both L_0 {g(k2['off_L0'].get('ratio'), 2)} "
             f"+- {g(k2['off_L0'].get('se'), 2)}")
    fl = ev["far_law"]
    L.append("    the law at Form II's sizes, MACE-LES's own bias (seed mean, frame se): " + "; ".join(
        f"L_0 = {L0} " + ", ".join(f"n {n} " + (f"{v['mean']:+.4f} ({v['se']:.4f})" if v else "-") for n, v in by.items())
        for L0, by in fl["bias"].items()))
    L.append("      the step 256 -> 1024: " + ", ".join(
        f"L_0 = {L0} " + (f"{v[0]:+.4f} +- {v[1]:.4f}" if v else "-") for L0, v in fl["step"].items())
        + f"; an error in 1/L gives {f(fl['predicted_1/L'])} ({fl['law']:.2f} of L_0 = 64's step from 64 to 256), "
        "a carried one about none")
    L.append("    MACE-LES at L_0 = 128 less at 64, the same frames (seed mean, sd; frame se): " + "; ".join(
        f"n {n} " + (f"{v['mean']:+.4f} ({v['sd']:.4f}; {v['se']:.4f})" if v else "-") for n, v in ev["between"].items()))
    L.append("    the net latent charge per molecule, rms (test frames of 64): " + "; ".join(
        f"L_0 = {L0} " + ", ".join(g(v, 4) for v in by.values()) for L0, by in ev["net_charge_rms"].items()))
    k = control(beside)
    L.insert(L.index("\n  Beside them, not registered"),
             f"  K16.5.5: {_verdict(k['holds'])} (the control: MACE without LES reads \"carried\"): "
             + (f"{k['verdict']}, r {g(beside['K16.5.2']['ratio'], 2)}" if k["verdict"] else "MACE not trained at 128 "
                "(MODEL=MACE sbatch run_e4_job16_5.sbatch)"))
    if beside:
        L.append("    MACE without LES: " + ", ".join(f"{k} {f(v)}" for k, v in beside["steps"].items())
                 + f"; K16.5.1's construction {_verdict(beside['K16.5.1']['holds'])}"
                 + "; its L_0's difference by size " + ", ".join(
                     f"{n} {f(v['mean']) if v else '-'}" for n, v in beside["between"].items())
                 + "; forces " + "; ".join(f"L_0 = {L0} " + " / ".join(g(x) for x in beside["K16.5.4"]["forces"][L0].values())
                                           for L0 in L0S))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for k, d in DIRS.items():
        ap.add_argument(f"--{k}", default=d)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = {k: getattr(a, k) for k in DIRS}
    ev = evaluate(load(dirs), a.data)
    mace = load(dirs, "mace")
    beside = evaluate(mace, a.data) if any(r["z"] is not None for by in mace.values() for r in by.values()) else None
    print(report(ev, beside))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"maceles": ev, "mace": beside, "K16.5.5": control(beside)}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
