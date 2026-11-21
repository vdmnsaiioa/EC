"""
e4_job9.py -- E4 job 9's registered readings K9.1-K9.5 (notes/e4-job9-preregistration.md; v0.0.33) from the training
jsons of scripts/e4_water.py train, without JAX.  K9.6, the charges' |b_T| on S0's frames, is read by
scripts/e4_sumrules.py read --job9_dir, with K9.3 from here (convergence).

    python scripts/e4_job9.py --out_dir results/e4_job9 --job7_dir results/e4_job7 --job6_dir results/e4_job6

Job 7's warm starts again, traced (BASE), and the same with the charge head at ten times the rest's learning rate (FIX;
ZERO started near zero).  Every run's json carries its charges' diagnostics (training.charges, e4_water.
charge_diagnostics).  The profile is job 7's scan: the final training loss of the pinned O/H charges at c times MB-pol's
permanent dipole, E_0 trained around each, its seed means at c = 0.7, 1.0, 1.1 and 1.45 fitted by a parabola L*(c)
(job 7's results, section 2).  A run whose charges carry c times the permanent dipole feels, along their overall scale s,
the profile's slope p = c dL*/dc if the rest of the model is at its best for those charges.

    K9.1  BASE: R = the objective's slope along the charges' scale at the end of Adam, the rest held, over p at the
          run's dipole; in K9_1 on at least MIN_RUNS of the four runs (the pull is there).  R within K9_1_LOCKED of 0 on
          at least MIN_RUNS: the runs sit at a stationary point of their own (locked)
    K9.2  BASE: |mean| / sd of the pull over the batches of Adam's second half at most K9_2 on at least MIN_RUNS
    K9.3  FIX, the four warm starts: the learned dipoles (test frames of 64) within K9_3 of their mean, every final
          training loss at or below the largest of M_inf's (job 6's and job 7's runs)
    K9.4  (read when K9.3 holds) their mean dipole within K9_4 of the profile's minimum, c* times the permanent dipole
    K9.5  their mean forces at most K9_5 above the scan's lowest, at every size
"""
import argparse
import json

import numpy as np

import e4_job7 as J7

BASE = {"M_1w050": 0.5, "M_1w150": 1.5}
FIX = {"M_1x050": 0.5, "M_1x150": 1.5}
ZERO = "M_1x000"
PROFILE_C = {"M_1q070": 0.7, "M_1q100": 1.0, "M_1q110": 1.1, "M_1q145": 1.45}   # job 7's results, section 2
K9_1, K9_1_LOCKED, K9_2, K9_3, K9_4, K9_5 = (0.5, 2.0), 0.25, 0.3, 0.15, 0.10, 0.07
MIN_RUNS = 3
SEEDS = (0, 1)
SIZES, PERMANENT_DEBYE, REF = J7.SIZES, J7.PERMANENT_DEBYE, J7.REF


def profile(job7):
    """job 7's scan as the profile: the seed means of the final training loss at PROFILE_C, the parabola through
    them (numpy.polyfit's coefficients, highest first), its minimum c* and L*(c*).  None when a scan rung is missing."""
    pts = {}
    for name, c in PROFILE_C.items():
        v = [J7.final_loss(r) for s, r in job7.get(name, {}).items() if s in SEEDS]
        v = [x for x in v if x is not None]
        if len(v) != len(SEEDS):
            return None
        pts[c] = float(np.mean(v))
    cs = sorted(pts)
    coef = np.polyfit(cs, [pts[c] for c in cs], 2)
    c_min = float(-coef[1] / (2 * coef[0]))
    return {"points": {str(c): pts[c] for c in cs}, "coef": [float(x) for x in coef], "c_min": c_min,
            "loss_min": float(np.polyval(coef, c_min))}


def profile_slope(prof, c):
    """p = d L*(s c) / ds at s = 1 = c dL*/dc."""
    a, b, _ = prof["coef"]
    return float(c * (2 * a * c + b))


def diagnostics(r):
    return r.get("training", {}).get("charges")


def pull_reading(runs, prof):
    """K9.1 and K9.2 on BASE's four runs: per run its dipole and c at the end of Adam (the traced frames), the slope
    along the charges' scale there, p, R, and the pull's |mean| / sd over Adam's second half."""
    rows = {}
    for rung in BASE:
        for s in SEEDS:
            r = runs.get(rung, {}).get(s)
            d = diagnostics(r) if r is not None else None
            if d is None or "adam" not in d["scale_curve"]:
                continue
            a = d["scale_curve"]["adam"]
            dip = a["charges"]["dipole_debye"]
            c = dip / PERMANENT_DEBYE
            p = profile_slope(prof, c) if prof is not None else None
            rows[f"{rung} seed {s}"] = {"dipole_debye": dip, "c": c, "slope": a["slope"], "profile_slope": p,
                                        "R": a["slope"] / p if p else None, "s_min_rest_held": a["s_min"],
                                        "snr": d["pull"]["second_half"]["snr"],
                                        "pull": (d["pull"]["second_half"]["mean"], d["pull"]["second_half"]["sd"])}
    complete = len(rows) == len(BASE) * len(SEEDS) and prof is not None
    lo, hi = K9_1
    inside = [k for k, v in rows.items() if v["R"] is not None and lo <= v["R"] <= hi]
    locked = [k for k, v in rows.items() if v["R"] is not None and abs(v["R"]) <= K9_1_LOCKED]
    quiet = [k for k, v in rows.items() if v["snr"] is not None and v["snr"] <= K9_2]
    k1 = {"runs": rows, "band": list(K9_1), "in_band": inside, "locked": locked,
          "holds": bool(len(inside) >= MIN_RUNS) if complete else None,
          "verdict": None if not complete else ("holds" if len(inside) >= MIN_RUNS else
                                               ("FAILS, low: locked" if len(locked) >= MIN_RUNS else "FAILS"))}
    k2 = {"runs": {k: v["snr"] for k, v in rows.items()}, "band": K9_2, "in_band": quiet,
          "holds": bool(len(quiet) >= MIN_RUNS) if len(rows) == len(BASE) * len(SEEDS) else None}
    return k1, k2


def m_inf_loss_max(*sources):
    losses = [J7.final_loss(r) for src in sources for r in (src or {}).get(REF, {}).values()]
    losses = [x for x in losses if x is not None]
    return max(losses) if losses else None


def convergence(runs, job7=None, job6=None):
    """K9.3: the four FIX runs' learned dipoles (test frames of 64) within K9_3 of their mean and every final training
    loss at or below M_inf's largest (job 6's and job 7's runs).  None when a run is missing."""
    lmax = m_inf_loss_max(job7, job6)
    rows = {}
    for rung, c0 in FIX.items():
        for s in SEEDS:
            r = runs.get(rung, {}).get(s)
            if r is None:
                continue
            d, fl = J7.learned_dipole(r), J7.final_loss(r)
            rows[f"{rung} seed {s}"] = {"start": c0, "start_debye": c0 * PERMANENT_DEBYE, "dipole_debye": d,
                                        "final_loss": fl, "gate": bool(lmax is not None and fl is not None and fl <= lmax)}
    out = {"runs": rows, "loss_max": lmax, "band": K9_3, "holds": None}
    dips = [v["dipole_debye"] for v in rows.values()]
    if len(rows) == len(FIX) * len(SEEDS) and all(d is not None for d in dips) and lmax is not None:
        mean = float(np.mean(dips))
        out["mean_debye"] = mean
        out["spread"] = [d / mean - 1.0 for d in dips]
        out["holds"] = bool(all(v["gate"] for v in rows.values()) and all(abs(x) <= K9_3 for x in out["spread"]))
    return out


def evaluate(runs, job7=None, job6=None):
    """K9.1-K9.5 and the readings beside them."""
    job7 = job7 or {}
    prof = profile(job7)
    out = {"profile": prof}
    out["K9.1"], out["K9.2"] = pull_reading(runs, prof)
    conv = convergence(runs, job7, job6)
    out["K9.3"] = conv
    # K9.4: where they converge, against the profile's minimum
    read = conv["holds"] is True and prof is not None
    target = prof["c_min"] * PERMANENT_DEBYE if prof is not None else None
    k4 = {"band": K9_4, "read": read, "target_debye": target, "holds": None}
    if "mean_debye" in conv and target:
        k4["mean_debye"] = conv["mean_debye"]
        k4["ratio"] = conv["mean_debye"] / target - 1.0
        if read:
            k4["holds"] = bool(abs(k4["ratio"]) <= K9_4)
    out["K9.4"] = k4
    # K9.5: what they buy, against the scan's lowest
    rows, ok = {}, True
    for n in SIZES:
        v = [J7.pooled(runs[w][s], n) for w in FIX for s in SEEDS if s in runs.get(w, {})]
        scan = {name: J7.seed_mean(job7.get(name, {}), n, seeds=SEEDS) if job7.get(name) else None for name in J7.SCAN}
        if len(v) != len(FIX) * len(SEEDS) or any(x is None for x in v) or any(x is None for x in scan.values()):
            ok = None
            continue
        best = min(scan, key=lambda k: scan[k][0])
        rows[n] = {"fix": float(np.mean(v)), "scan_lowest": scan[best][0], "scan_rung": best,
                   "ratio": float(np.mean(v)) / scan[best][0] - 1.0}
        ok = ok and rows[n]["ratio"] <= K9_5 if ok is not None else None
    out["K9.5"] = {"values": rows, "band": K9_5, "holds": None if ok is None else bool(ok)}
    # beside: the start near zero, against the warm starts' mean
    zero = {}
    for s, r in sorted(runs.get(ZERO, {}).items()):
        v = r["readings"]["cells"].get("64", {}).get("test", {}).get("learned_sources", {})
        zero[f"{ZERO} seed {s}"] = {"dipole_debye": J7.learned_dipole(r), "to_H_side": v.get("dipole_to_H_side"),
                                    "final_loss": J7.final_loss(r),
                                    "against_warm": (J7.learned_dipole(r) / conv["mean_debye"] - 1.0)
                                    if "mean_debye" in conv and J7.learned_dipole(r) is not None else None}
    out["zero"] = zero
    # beside: BASE against job 7's runs of the same rungs and seeds
    out["reproduction"] = {f"{rung} seed {s}": {"dipole_debye": (J7.learned_dipole(runs[rung][s]),
                                                                 J7.learned_dipole(job7[rung][s])),
                                                "forces_64": (J7.pooled(runs[rung][s], "64"), J7.pooled(job7[rung][s], "64"))}
                           for rung in BASE for s in SEEDS if s in runs.get(rung, {}) and s in job7.get(rung, {})}
    # beside: every traced run's path and the readings at the end of training
    paths = {}
    for rung, by in sorted(runs.items()):
        for s, r in sorted(by.items()):
            d = diagnostics(r)
            if d is None:
                continue
            tr = {v["step"]: v["dipole_debye"] for v in d["trace"]}
            steps = sorted(tr)
            mid = min(steps, key=lambda x: abs(x - steps[-1] / 2)) if steps else None
            total = tr[steps[-1]] - tr[steps[0]] if steps else None
            fin = d["scale_curve"]["final"]
            c_fin = fin["charges"]["dipole_debye"] / PERMANENT_DEBYE
            p_fin = profile_slope(prof, c_fin) if prof is not None else None
            paths[f"{rung} seed {s}"] = {
                "dipole": {str(k): tr[k] for k in steps if k % 500 == 0 or k == steps[-1]},
                "second_half_share": (tr[steps[-1]] - tr[mid]) / total if steps and total else None,
                "final": {"dipole_debye": fin["charges"]["dipole_debye"], "slope": fin["slope"], "profile_slope": p_fin,
                          "R": fin["slope"] / p_fin if p_fin else None, "s_min_rest_held": fin["s_min"]},
                "snr": d["pull"]["second_half"]["snr"]}
    out["paths"] = paths
    out["forces"] = {name: {n: J7.seed_mean(by, n, seeds=SEEDS) for n in SIZES} for name, by in runs.items()}
    out["final_loss"] = {name: float(np.mean([x for x in (J7.final_loss(r) for r in by.values()) if x is not None]))
                         for name, by in runs.items() if any(J7.final_loss(r) is not None for r in by.values())}
    return out


def _verdict(h, read=True):
    return "not read" if (h is None or not read) else ("holds" if h else "FAILS")


def report(ev, energies=None):
    pct = lambda x: f"{100 * x:+.1f} %"
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    lines = ["E4, job 9 (notes/e4-job9-preregistration.md): the learned charges' pull, and the charge head's own "
             "learning rate"]
    prof = ev["profile"]
    if prof is not None:
        lines.append(f"  the profile (job 7's scan, final training loss): " + ", ".join(
            f"c {c}: {v:.2f}" for c, v in prof["points"].items()) + f"; the parabola's minimum at c* = {prof['c_min']:.3f} "
            f"({prof['c_min'] * PERMANENT_DEBYE:.3f} D), {prof['loss_min']:.2f}")
    else:
        lines.append("  the profile: job 7's scan missing (--job7_dir)")
    k1 = ev["K9.1"]
    lo, hi = k1["band"]
    lines.append(f"  K9.1: {k1['verdict'] or 'not read'} (R, the slope along the charges' scale at the end of Adam over "
                 f"the profile's, in [{lo:g}, {hi:g}] on at least {MIN_RUNS} of 4)")
    for k, v in k1["runs"].items():
        lines.append(f"      {k}: dipole {v['dipole_debye']:.3f} D (c {v['c']:.3f}), slope {v['slope']:+.3e} against the "
                     f"profile's {f(v['profile_slope'], 2)}: R {f(v['R'])}; the minimum with the rest held at "
                     f"s = {v['s_min_rest_held']:.3f}")
    k2 = ev["K9.2"]
    lines.append(f"  K9.2: {_verdict(k2['holds'])} (the pull's |mean| / sd over Adam's second half at most {K9_2:g} on "
                 f"at least {MIN_RUNS} of 4): " + ", ".join(f"{k} {f(v)}" for k, v in k2["runs"].items()))
    c = ev["K9.3"]
    lines.append(f"  K9.3: {_verdict(c['holds'])} (the four warm starts at ten times the charge head's learning rate "
                 f"within {100 * K9_3:.0f} % of their mean, every final training loss at or below M_inf's largest"
                 + (f", {c['loss_max']:.3e})" if c["loss_max"] is not None else ")"))
    for k, v in c["runs"].items():
        d = v["dipole_debye"]
        lines.append(f"      {k}: start {v['start_debye']:.3f} D -> " + (f"{d:.3f} D" if d is not None else "no reading")
                     + (f", final loss {v['final_loss']:.3e} ({'passes' if v['gate'] else 'FAILS'})"
                        if v["final_loss"] is not None else ""))
    if "mean_debye" in c:
        lines.append(f"      mean {c['mean_debye']:.3f} D ({c['mean_debye'] / PERMANENT_DEBYE:.3f} of the permanent "
                     f"{PERMANENT_DEBYE} D); each against it " + ", ".join(pct(x) for x in c["spread"]))
    k4 = ev["K9.4"]
    lines.append(f"  K9.4: {_verdict(k4['holds'], k4['read'])} (their mean within {100 * K9_4:.0f} % of the profile's "
                 f"minimum; read when K9.3 holds)"
                 + (f": {k4['mean_debye']:.3f} against {k4['target_debye']:.3f} D ({pct(k4['ratio'])})"
                    if "ratio" in k4 else ""))
    w = ev["K9.5"]
    lines.append(f"  K9.5: {_verdict(w['holds'])} (their mean forces at most {100 * K9_5:.0f} % above the scan's lowest, "
                 "at every size)")
    for n, v in w["values"].items():
        lines.append(f"      n {n:>3s}: {v['fix']:.3f} against {v['scan_lowest']:.3f} ({v['scan_rung']}; {pct(v['ratio'])})")
    lines.append("\n  Beside them, not registered")
    for k, v in ev["zero"].items():
        lines.append(f"    the start near zero, {k}: {f(v['dipole_debye'])} D"
                     + (f" ({pct(v['against_warm'])} against the warm starts' mean)" if v["against_warm"] is not None else "")
                     + (f", {100 * v['to_H_side']:.0f} % to the H side" if v["to_H_side"] is not None else "")
                     + (f", final loss {v['final_loss']:.3e}" if v["final_loss"] is not None else ""))
    if ev["reproduction"]:
        lines.append("    job 7's warm starts again against job 7 (dipole D; forces at 64): " + "; ".join(
            f"{k} {f(a[0])} / {f(a[1])}, {f(b[0])} / {f(b[1])}" for k, (a, b) in
            ((k, (v["dipole_debye"], v["forces_64"])) for k, v in ev["reproduction"].items())))
    lines.append("    the paths (the learned dipole on the traced frames, D, by Adam step; the second half's share of the "
                 "change; at the end of training: the slope, R and the minimum with the rest held; the pull's |mean| / sd)")
    for k, v in ev["paths"].items():
        fin = v["final"]
        lines.append(f"      {k}: " + ", ".join(f"{d:.3f} at {s}" for s, d in v["dipole"].items())
                     + f"; second half {f(v['second_half_share'], 2)}; final {fin['dipole_debye']:.3f} D, slope "
                       f"{fin['slope']:+.3e}, R {f(fin['R'])}, minimum at s = {fin['s_min_rest_held']:.3f}; "
                       f"|mean| / sd {f(v['snr'])}")
    lines.append("    every rung's forces (mean +- sd over seeds)")
    for name, by in sorted(ev["forces"].items()):
        lines.append(f"      {name:8s} " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}" for n, v in by.items() if v))
    if ev.get("final_loss"):
        lines.append("    final training loss, mean over seeds: " + "; ".join(
            f"{name} {x:.3e}" for name, x in sorted(ev["final_loss"].items())))
    if energies:
        lines.append("    energy per molecule (kcal/mol), unseen frames: " + "; ".join(
            f"{name} " + "/".join(f"{v[0]:.4f}" for v in by.values() if v) for name, by in sorted(energies.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", default="results/e4_job9")
    ap.add_argument("--job7_dir", default="results/e4_job7", help="job 7's results: the scan (the profile), M_inf, "
                                                                  "and the warm starts it ran")
    ap.add_argument("--job6_dir", default="results/e4_job6", help="job 6's results: M_inf's four seeds (the gate)")
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    runs = J7.load_runs(a.out_dir)
    if not runs:
        raise SystemExit(f"no results in {a.out_dir}")
    ev = evaluate(runs, J7.load_runs(a.job7_dir), J7.load_runs(a.job6_dir))
    en = {name: {n: J7.seed_mean(by, n, "energy_per_molecule_rmse", seeds=SEEDS) for n in SIZES}
          for name, by in runs.items()}
    print(report(ev, en))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"readings": ev, "energies": en}, fh, indent=1)


if __name__ == "__main__":
    main()
