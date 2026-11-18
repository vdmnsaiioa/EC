"""
e4_job7.py -- E4 job 7's registered readings J7.1-J7.5 (notes/e4-job7-preregistration.md; v0.0.31) from the training
jsons of scripts/e4_water.py train, without JAX.  J7.6, the warm-started charges' |b_T| on S0's frames, is read by
scripts/e4_sumrules.py read --job7_dir, with J7.4 from here (convergence).

    python scripts/e4_job7.py --out_dir results/e4_job7 --reference_dir results/e4_job6

The rigid forces are read on the unseen frames, job 1's convention: validation and test at the training size of 64
molecules, every split at 128 and 256.  Each split's rmse is pooled by its frames (every frame of a size has the same
molecules), and the readings are seed means.  M_inf's reference is its four seeds in job 6 (reference_dir); job 7's two
seeds of M_inf are its check.

    J7.1  M_1pm (MB-pol's site charges, no c6) within J7_1 of the dagger M_16pm (with c6), at every size
    J7.2  the scan's lowest forces (O/H charges at c times MB-pol's permanent dipole) at c >= 1, at every size
    J7.3  the scan's lowest at least J7_3 below M_inf's, at every size
    J7.4  the six warm-started runs' learned dipoles (test frames of 64) within J7_4 of their mean, each run's final
          training loss at or below the largest of M_inf's (job 6's and job 7's runs)
    J7.5  (read when J7.3 holds) the warm-started runs' mean forces at most J7_5 above the scan's lowest, at every size
"""
import argparse
import glob
import json
import os

import numpy as np

SIZES = ("64", "128", "256")
UNSEEN = {"64": ("valid", "test"), "128": ("train", "valid", "test"), "256": ("train", "valid", "test")}
SCAN = {"M_1q040": 0.4, "M_1q070": 0.7, "M_1q100": 1.0, "M_1q110": 1.1, "M_1q145": 1.45}
WARM = {"M_1w050": 0.5, "M_1w100": 1.0, "M_1w150": 1.5}
DAGGER, SITES, REF = "M_16pm", "M_1pm", "M_inf"
J7_1, J7_3, J7_4, J7_5 = 0.07, 0.10, 0.15, 0.07
SEEDS = (0, 1)
PERMANENT_DEBYE = 1.864          # MB-pol's rigid monomer (S0's perm); the O/H charges carry it (water.oh_equivalent_charges)


def load_runs(d):
    """{rung: {seed: json}} from a results directory (None or missing: {})."""
    runs = {}
    if not d:
        return runs
    for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        runs.setdefault(r["rung"], {})[int(r["seed"])] = r
    return runs


def pooled(r, n, key="force_rigid_rmse"):
    """a run's rmse on the unseen frames of size n, each split's pooled by its frames; None when a split is missing."""
    cells = r["readings"]["cells"].get(n, {})
    num = den = 0.0
    for split in UNSEEN[n]:
        v = cells.get(split, {})
        if key not in v:
            return None
        num += v["frames"] * v[key] ** 2
        den += v["frames"]
    return float(np.sqrt(num / den)) if den else None


def seed_mean(by_seed, n, key="force_rigid_rmse", seeds=None):
    """(mean, sd over seeds, seeds) of a rung's pooled rmse at size n; None when a seed's reading is missing."""
    seeds = sorted(by_seed) if seeds is None else list(seeds)
    v = [pooled(by_seed[s], n, key) if s in by_seed else None for s in seeds]
    if not v or any(x is None for x in v):
        return None
    return float(np.mean(v)), float(np.std(v)), seeds


def final_loss(r):
    fl = r.get("training", {}).get("final_loss")
    return None if fl is None else float(fl["total"])


def learned_dipole(r, n="64", split="test"):
    v = r["readings"]["cells"].get(n, {}).get(split, {}).get("learned_sources", {})
    return float(v["dipole_debye"][0]) if "dipole_debye" in v else None


def convergence(runs, reference=None):
    """J7.4: every warm-started run's learned dipole on the test frames of 64 molecules and its final training loss
    against the largest of M_inf's (job 7's runs and reference's).  Holds when all six runs are there, every one at or
    below that loss, and every dipole within J7_4 of the six runs' mean; None when a run is missing."""
    losses = [final_loss(r) for src in (runs, reference or {}) for r in src.get(REF, {}).values()]
    losses = [x for x in losses if x is not None]
    lmax = max(losses) if losses else None
    rows = {}
    for rung, c0 in WARM.items():
        for s in SEEDS:
            r = runs.get(rung, {}).get(s)
            if r is None:
                continue
            d, fl = learned_dipole(r), final_loss(r)
            rows[f"{rung} seed {s}"] = {"start": c0, "start_debye": c0 * PERMANENT_DEBYE, "dipole_debye": d,
                                        "final_loss": fl, "gate": bool(lmax is not None and fl is not None and fl <= lmax)}
    out = {"runs": rows, "loss_max": lmax, "band": J7_4, "holds": None}
    dips = [v["dipole_debye"] for v in rows.values()]
    if len(rows) == len(WARM) * len(SEEDS) and all(d is not None for d in dips):
        mean = float(np.mean(dips))
        out["mean_debye"] = mean
        out["spread"] = [d / mean - 1.0 for d in dips]
        out["holds"] = bool(all(v["gate"] for v in rows.values()) and all(abs(x) <= J7_4 for x in out["spread"]))
    return out


def evaluate(runs, reference=None):
    """J7.1-J7.5 from job 7's runs and job 6's (the reference for M_inf).  Every value is a seed mean on the unseen
    frames; a reading with a missing rung or size is reported as not read (holds None)."""
    ref = (reference or {}).get(REF) or runs.get(REF, {})
    F = {name: {n: seed_mean(by, n, seeds=SEEDS) for n in SIZES} for name, by in runs.items()}
    Fref = {n: seed_mean(ref, n) for n in SIZES} if ref else {n: None for n in SIZES}
    out = {}
    # J7.1: the site charges without c6 against the dagger
    rows, ok = {}, True
    for n in SIZES:
        a, b = F.get(SITES, {}).get(n), F.get(DAGGER, {}).get(n)
        if a is None or b is None:
            ok = None
            continue
        rows[n] = {"M_1pm": a[0], "dagger": b[0], "ratio": a[0] / b[0] - 1.0}
        ok = ok and abs(rows[n]["ratio"]) <= J7_1 if ok is not None else None
    out["J7.1"] = {"values": rows, "band": J7_1, "holds": None if ok is None else bool(ok)}
    # J7.2 and J7.3: the scan
    best, ok2, ok3 = {}, True, True
    for n in SIZES:
        vals = {name: F.get(name, {}).get(n) for name in SCAN}
        if any(v is None for v in vals.values()) or Fref[n] is None:
            ok2 = ok3 = None
            continue
        name = min(vals, key=lambda k: vals[k][0])
        best[n] = {"rung": name, "c": SCAN[name], "forces": vals[name][0], "M_inf": Fref[n][0],
                   "against_M_inf": vals[name][0] / Fref[n][0] - 1.0,
                   "curve": {str(SCAN[k]): v[0] for k, v in vals.items()}}
        if ok2 is not None:
            ok2 = ok2 and SCAN[name] >= 1.0
            ok3 = ok3 and vals[name][0] <= (1.0 - J7_3) * Fref[n][0]
    out["J7.2"] = {"values": best, "holds": None if ok2 is None else bool(ok2)}
    out["J7.3"] = {"values": best, "band": J7_3, "holds": None if ok3 is None else bool(ok3)}
    # J7.4: the warm starts converge
    out["J7.4"] = convergence(runs, reference)
    # J7.5: what they buy, read when J7.3 holds
    rows, ok = {}, True
    for n in SIZES:
        v = [pooled(runs[w][s], n) for w in WARM for s in SEEDS if s in runs.get(w, {})]
        if len(v) != len(WARM) * len(SEEDS) or any(x is None for x in v) or n not in best:
            ok = None
            continue
        rows[n] = {"warm": float(np.mean(v)), "scan_lowest": best[n]["forces"],
                   "ratio": float(np.mean(v)) / best[n]["forces"] - 1.0}
        ok = ok and rows[n]["ratio"] <= J7_5 if ok is not None else None
    read = out["J7.3"]["holds"] is True
    out["J7.5"] = {"values": rows, "band": J7_5, "read": read, "holds": (None if ok is None else bool(ok)) if read else None}
    # beside: job 7's M_inf against job 6's, seeds 0 and 1
    if reference and runs.get(REF):
        out["M_inf_check"] = {n: {s: (pooled(runs[REF][s], n), pooled(reference[REF][s], n))
                                  for s in SEEDS if s in runs[REF] and s in reference.get(REF, {})} for n in SIZES}
    out["forces"] = {name: {n: v for n, v in by.items()} for name, by in F.items()}
    out["M_inf_reference"] = Fref
    # beside: the c6's share of the dagger's gain, and the O/H form against MB-pol's sites at the permanent dipole
    get = lambda name, n: (F.get(name, {}).get(n) or (None,))[0]
    out["c6_share"] = {n: (get(SITES, n) - get(DAGGER, n)) / (Fref[n][0] - get(DAGGER, n))
                       for n in SIZES if None not in (get(SITES, n), get(DAGGER, n)) and Fref[n] is not None
                       and Fref[n][0] != get(DAGGER, n)}
    out["oh_against_sites"] = {n: get("M_1q100", n) / get(SITES, n) - 1.0 for n in SIZES
                               if None not in (get("M_1q100", n), get(SITES, n))}
    out["final_loss"] = {name: float(np.mean([x for x in (final_loss(r) for r in by.values()) if x is not None]))
                         for name, by in runs.items() if any(final_loss(r) is not None for r in by.values())}
    return out


def _verdict(h, read=True):
    return "not read" if (h is None or not read) else ("holds" if h else "FAILS")


def report(ev, energies=None):
    pct = lambda x: f"{100 * x:+.1f} %"
    lines = ["E4, job 7 (notes/e4-job7-preregistration.md): rigid forces on the unseen frames (kcal/mol/A), seed means"]
    b = ev["J7.1"]
    lines.append(f"  J7.1: {_verdict(b['holds'])} (M_1pm within {100 * J7_1:.0f} % of the dagger at every size)")
    for n, v in b["values"].items():
        lines.append(f"      n {n:>3s}: M_1pm {v['M_1pm']:.3f}, dagger {v['dagger']:.3f} ({pct(v['ratio'])})")
    lines.append(f"  J7.2: {_verdict(ev['J7.2']['holds'])} (the scan's lowest at c >= 1, at every size)")
    lines.append(f"  J7.3: {_verdict(ev['J7.3']['holds'])} (the scan's lowest at least {100 * J7_3:.0f} % below M_inf's, "
                 "at every size)")
    for n, v in ev["J7.2"]["values"].items():
        lines.append(f"      n {n:>3s}: lowest at c = {v['c']:g} ({v['rung']}), {v['forces']:.3f} against M_inf "
                     f"{v['M_inf']:.3f} ({pct(v['against_M_inf'])}); the curve "
                     + ", ".join(f"c {c}: {x:.3f}" for c, x in v["curve"].items()))
    c = ev["J7.4"]
    lines.append(f"  J7.4: {_verdict(c['holds'])} (the six warm-started dipoles within {100 * J7_4:.0f} % of their mean, "
                 f"every final training loss at or below M_inf's largest"
                 + (f", {c['loss_max']:.3e})" if c["loss_max"] is not None else ")"))
    for k, v in c["runs"].items():
        d = v["dipole_debye"]
        lines.append(f"      {k}: start {v['start_debye']:.3f} D -> "
                     + (f"{d:.3f} D" if d is not None else "no reading")
                     + (f", final loss {v['final_loss']:.3e} ({'passes' if v['gate'] else 'FAILS'})"
                        if v["final_loss"] is not None else ""))
    if "mean_debye" in c:
        lines.append(f"      mean {c['mean_debye']:.3f} D ({c['mean_debye'] / PERMANENT_DEBYE:.2f} of the permanent "
                     f"{PERMANENT_DEBYE} D); each against it " + ", ".join(pct(x) for x in c["spread"]))
    w = ev["J7.5"]
    lines.append(f"  J7.5: {_verdict(w['holds'], w['read'])} (the warm-started runs' mean at most {100 * J7_5:.0f} % above "
                 "the scan's lowest, at every size; read when J7.3 holds)")
    for n, v in w["values"].items():
        lines.append(f"      n {n:>3s}: warm starts {v['warm']:.3f} against {v['scan_lowest']:.3f} ({pct(v['ratio'])})")
    lines.append("\n  Beside them, not registered: every rung's forces (mean +- sd over seeds)")
    for name, by in sorted(ev["forces"].items()):
        lines.append(f"    {name:8s} " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}" for n, v in by.items() if v))
    lines.append("    M_inf (job 6, the reference) " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}"
                                                            for n, v in ev["M_inf_reference"].items() if v))
    if ev.get("c6_share"):
        lines.append("    the c6's share of the dagger's gain, (M_1pm - dagger) / (M_inf - dagger): "
                     + ", ".join(f"n {n} {pct(x)}" for n, x in ev["c6_share"].items()))
    if ev.get("oh_against_sites"):
        lines.append("    the O/H charges against MB-pol's sites at the permanent dipole, M_1q100 / M_1pm - 1: "
                     + ", ".join(f"n {n} {pct(x)}" for n, x in ev["oh_against_sites"].items()))
    if ev.get("final_loss"):
        lines.append("    final training loss, mean over seeds: " + "; ".join(
            f"{name} {x:.3e}" for name, x in sorted(ev["final_loss"].items())))
    if ev.get("M_inf_check"):
        lines.append("    job 7's M_inf against job 6's, seed by seed: " + "; ".join(
            f"n {n} " + ", ".join(f"seed {s} {a:.3f} / {b:.3f}" for s, (a, b) in by.items() if a is not None and b is not None)
            for n, by in ev["M_inf_check"].items()))
    if energies:
        lines.append("    energy per molecule (kcal/mol), unseen frames: " + "; ".join(
            f"{name} " + "/".join(f"{v[0]:.4f}" for v in by.values() if v) for name, by in sorted(energies.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", default="results/e4_job7")
    ap.add_argument("--reference_dir", default="results/e4_job6", help="job 6's results: M_inf's four seeds")
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    runs, ref = load_runs(a.out_dir), load_runs(a.reference_dir)
    if not runs:
        raise SystemExit(f"no results in {a.out_dir}")
    ev = evaluate(runs, ref)
    en = {name: {n: seed_mean(by, n, "energy_per_molecule_rmse", seeds=SEEDS) for n in SIZES} for name, by in runs.items()}
    print(report(ev, en))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"readings": ev, "energies": en}, fh, indent=1)


if __name__ == "__main__":
    main()
