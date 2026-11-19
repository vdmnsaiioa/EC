"""
e4_job8.py -- E4 job 8's registered readings K8.1-K8.5 and its model rule (notes/e4-job8-preregistration.md; v0.0.32)
from the training jsons of scripts/e4_water.py train, without JAX.  The pinned c6 band at contact: the band at its own
edge (Rung.disp_l), and without it, on the dagger, on the field-term dagger and on D' (job 5's read-out, held).

    python scripts/e4_job8.py --out_dir results/e4_job8 --reference_dir results/e4_job5

Rigid forces on the unseen frames (scripts/e4_job7.py's pooling: validation and test at 64 molecules, every split at
128 and 256), seed means over the seeds each rung ran (four; two for D' without the band).  D''s reference is its four
seeds in job 5 (reference_dir, rung M_16C2pm_us).

    K8.1  the dagger with its c6 band at 2.0 A (M_16pm_d20) within K8_1 of M_1pm, at every size
    K8.2  the field-term dagger without the band (M_1pm_us) within K8_2 of it with the band (M_16pm_us), at every size
    K8.3  M_1pm_us within K8_3 of M_1pm, at every size
    K8.4  D' with the band at 2.0 A (M_16C2pm_us_d20) at most K8_4 above D' (job 5), at every size
    K8.5  that rung's infrared as D''s: eps_inf on the held-out liquid within K8_5_EPS, the far probes' non-additive
          error at 10, 12 and 15 A and T5's three-body error within K8_5_POINTS percentage points
    model rule: if K8.4 and K8.5 hold, D' with the band at 2.0 A is E4's model
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_job7 as J7                                          # noqa: E402  (load_runs, pooled, seed_mean, final_loss)

SIZES = J7.SIZES
DAGGER, DAGGER_D, SITES = "M_16pm", "M_16pm_d20", "M_1pm"
FIELD, FIELD_D, FIELD_NO = "M_16pm_us", "M_16pm_us_d20", "M_1pm_us"
DPRIME, DPRIME_D, DPRIME_NO = "M_16C2pm_us", "M_16C2pm_us_d20", "M_1C2pm_us"
EDGES = {"M_16pm": 0.75, "M_16pm_d15": 1.5, "M_16pm_d20": 2.0, "M_16pm_d25": 2.5}
K8_1, K8_2, K8_3, K8_4, K8_5_EPS, K8_5_POINTS = 0.07, 0.05, 0.05, 0.02, 0.002, 0.5
PROBE_DISTANCES = ("10", "12", "15")


def means(runs):
    """{rung: {size: (mean, sd, seeds)}} of the rigid forces, every seed a rung ran."""
    return {name: {n: J7.seed_mean(by, n) for n in SIZES} for name, by in runs.items()}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def infrared(by_seed):
    """seed means of eps_inf on the held-out liquid (after training), the far probes' non-additive error (percent of
    the truth's rms) at 10, 12 and 15 A, and T5's three-body error (percent); with eps_inf on the read-out before
    training, the check that a held read-out is its reference's."""
    def eps(r, before=False):
        v = (r.get("matching") or {}).get("liquid_before_training") if before else r["readings"].get("liquid")
        return None if not v else float(v["eps_inf_model"])

    def probe(r, d):
        b = r["readings"].get("probes", {}).get("by_distance_angstrom", {}).get(d)
        return None if not b else 100.0 * b["nonadditive_rmse"] / b["nonadditive_rms"]

    def t5(r):
        v = r["readings"].get("far_trimers")
        return None if not v else 100.0 * v["three_body_rmse"] / v["three_body_rms"]

    rs = list(by_seed.values())
    return {"eps_inf": _mean([eps(r) for r in rs]), "eps_inf_before_training": _mean([eps(r, True) for r in rs]),
            "probes": {d: _mean([probe(r, d) for r in rs]) for d in PROBE_DISTANCES}, "T5": _mean([t5(r) for r in rs])}


def _within(a, b, band, one_sided=False):
    """{size: ratio - 1} of a against b and whether every size is in the band; None when a size is missing."""
    rows, ok = {}, True
    for n in SIZES:
        if not a or not b or a.get(n) is None or b.get(n) is None:
            return rows, None
        x = a[n][0] / b[n][0] - 1.0
        rows[n] = x
        ok = ok and (x <= band if one_sided else abs(x) <= band)
    return rows, bool(ok)


def evaluate(runs, reference=None):
    F = means(runs)
    ref = (reference or {}).get(DPRIME, {})
    Fd = {n: J7.seed_mean(ref, n) for n in SIZES} if ref else None
    out = {}
    for key, a, b, band in (("K8.1", DAGGER_D, SITES, K8_1), ("K8.2", FIELD_NO, FIELD, K8_2), ("K8.3", FIELD_NO, SITES, K8_3)):
        rows, ok = _within(F.get(a), F.get(b), band)
        out[key] = {"against": [a, b], "values": rows, "band": band, "holds": ok}
    rows, ok = _within(F.get(DPRIME_D), Fd, K8_4, one_sided=True)
    out["K8.4"] = {"against": [DPRIME_D, DPRIME + " (job 5)"], "values": rows, "band": K8_4, "holds": ok}
    ir_d, ir_ref = (infrared(runs[DPRIME_D]) if DPRIME_D in runs else None), (infrared(ref) if ref else None)
    ok5, diffs = None, {}
    if ir_d and ir_ref and ir_d["eps_inf"] is not None and ir_ref["eps_inf"] is not None:
        diffs["eps_inf"] = ir_d["eps_inf"] - ir_ref["eps_inf"]
        ok5 = abs(diffs["eps_inf"]) <= K8_5_EPS
        for d in PROBE_DISTANCES:
            x, y = ir_d["probes"][d], ir_ref["probes"][d]
            if x is None or y is None:
                ok5 = None
                break
            diffs[f"probes {d} A"] = x - y
            ok5 = ok5 and abs(x - y) <= K8_5_POINTS
        if ok5 is not None:
            if ir_d["T5"] is None or ir_ref["T5"] is None:
                ok5 = None
            else:
                diffs["T5"] = ir_d["T5"] - ir_ref["T5"]
                ok5 = ok5 and abs(diffs["T5"]) <= K8_5_POINTS
    out["K8.5"] = {"values": diffs, "this": ir_d, "D'": ir_ref, "holds": None if ok5 is None else bool(ok5)}
    h4, h5 = out["K8.4"]["holds"], out["K8.5"]["holds"]
    out["model_rule"] = None if h4 is None or h5 is None else bool(h4 and h5)
    # beside
    out["forces"] = F
    out["D'_reference"] = Fd
    out["edges"] = {e: F[name] for name, e in EDGES.items() if name in F}
    out["final_loss"] = {name: float(np.mean([x for x in (J7.final_loss(r) for r in by.values()) if x is not None]))
                         for name, by in runs.items() if any(J7.final_loss(r) is not None for r in by.values())}
    if DPRIME_NO in runs and ref:
        seeds = sorted(runs[DPRIME_NO])
        a = {n: J7.seed_mean(runs[DPRIME_NO], n) for n in SIZES}
        b = {n: J7.seed_mean(ref, n, seeds=seeds) for n in SIZES}
        out["D'_without_band"] = {"seeds": seeds, "values": _within(a, b, 1.0)[0], "infrared": infrared(runs[DPRIME_NO])}
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev, energies=None):
    pct = lambda x: f"{100 * x:+.1f} %"
    F = ev["forces"]
    lines = ["E4, job 8 (notes/e4-job8-preregistration.md): rigid forces on the unseen frames (kcal/mol/A), seed means"]
    texts = {"K8.1": f"the dagger with its c6 band at 2.0 A within {100 * K8_1:.0f} % of M_1pm",
             "K8.2": f"the field-term dagger without the band within {100 * K8_2:.0f} % of it with the band",
             "K8.3": f"the field-term dagger without the band within {100 * K8_3:.0f} % of M_1pm",
             "K8.4": f"D' with the band at 2.0 A at most {100 * K8_4:.0f} % above D' (job 5)"}
    for key, text in texts.items():
        b = ev[key]
        lines.append(f"  {key}: {_verdict(b['holds'])} ({text}, at every size)")
        a, r = b["against"]
        for n, x in b["values"].items():
            fa = F.get(a, {}).get(n)
            fb = (ev["D'_reference"] or {}).get(n) if key == "K8.4" else F.get(r, {}).get(n)
            lines.append(f"      n {n:>3s}: {a} {fa[0]:.3f} against {r} {fb[0]:.3f} ({pct(x)})")
    b = ev["K8.5"]
    lines.append(f"  K8.5: {_verdict(b['holds'])} (D' with the band at 2.0 A against D': eps_inf within {K8_5_EPS}, the far "
                 f"probes at 10-15 A and T5 within {K8_5_POINTS} points)")
    for k, x in b["values"].items():
        lines.append(f"      {k}: {x:+.4f}" if k == "eps_inf" else f"      {k}: {x:+.2f} points")
    this, dref = b["this"], b["D'"]
    if this and dref:
        lines.append(f"      eps_inf on the read-out before training: {this['eps_inf_before_training']} against "
                     f"D' (job 5) {dref['eps_inf_before_training']} (the held read-out's check)")
    rule = ev["model_rule"]
    lines.append("  the model rule (K8.4 and K8.5): " + ("not read" if rule is None else
                 ("D' with the c6 band at 2.0 A becomes E4's model" if rule else "D' stays E4's model")))
    lines.append("\n  Beside them, not registered:")
    if ev["edges"]:
        lines.append("    the dagger by the c6 band's edge: " + "; ".join(
            f"{e:g} A " + "/".join(f"{v[0]:.3f}" for v in by.values() if v) for e, by in sorted(ev["edges"].items()))
            + (f"; no band (M_1pm) " + "/".join(f"{v[0]:.3f}" for v in F[SITES].values() if v) if SITES in F else ""))
    for name, by in sorted(F.items()):
        lines.append(f"    {name:16s} " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}" for n, v in by.items() if v))
    if ev["D'_reference"]:
        lines.append("    D' (job 5, the reference) " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}"
                                                            for n, v in ev["D'_reference"].items() if v))
    if ev.get("D'_without_band"):
        w = ev["D'_without_band"]
        lines.append(f"    D' without the band (seeds {w['seeds']}) against D' on the same seeds: "
                     + ", ".join(f"n {n} {pct(x)}" for n, x in w["values"].items())
                     + f"; its far probes " + "/".join(f"{x:.1f}" for x in w["infrared"]["probes"].values() if x is not None)
                     + " %")
    if ev["final_loss"]:
        lines.append("    final training loss, mean over seeds: " + "; ".join(
            f"{name} {x:.3e}" for name, x in sorted(ev["final_loss"].items())))
    if energies:
        lines.append("    energy per molecule (kcal/mol), unseen frames: " + "; ".join(
            f"{name} " + "/".join(f"{v[0]:.4f}" for v in by.values() if v) for name, by in sorted(energies.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", default="results/e4_job8")
    ap.add_argument("--reference_dir", default="results/e4_job5", help="job 5's results: D''s four seeds")
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    runs, ref = J7.load_runs(a.out_dir), J7.load_runs(a.reference_dir)
    if not runs:
        raise SystemExit(f"no results in {a.out_dir}")
    ev = evaluate(runs, ref)
    en = {name: {n: J7.seed_mean(by, n, "energy_per_molecule_rmse") for n in SIZES} for name, by in runs.items()}
    print(report(ev, en))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"readings": ev, "energies": en}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
