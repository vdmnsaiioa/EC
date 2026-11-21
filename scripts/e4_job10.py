"""
e4_job10.py -- E4 job 10's registered readings K10.1-K10.4 (notes/e4-job10-preregistration.md; v0.0.34) from the
training jsons of scripts/e4_water.py train, without JAX.  K10.5, the polished charges' |b_T| on S0's frames, is read by
scripts/e4_sumrules.py read --job10_dir, with K10.2 from here (convergence).

    python scripts/e4_job10.py --out_dir results/e4_job10 --job7_dir results/e4_job7 --job6_dir results/e4_job6

Job 10 starts from job 7's end states (training.start_from) and polishes each with L-BFGS on every training frame:
job 7's warm starts (M_1w050, M_1w100, M_1w150) and, as the control, the pinned O/H charges at the scan's best scale
(M_1q110), seeds 0 and 1.  The target is job 7's profile minimum (scripts/e4_job9.py's profile: c* = 1.098, 2.047 D).

    K10.1  the four off-scale starts (M_1w050, M_1w150): each covers at least K10_1 of its distance to the target,
           (d_end - d_start) / (d* - d_start), its learned dipole on the test frames of 64 molecules before (job 7) and
           after the polish
    K10.2  the six polished warm starts' learned dipoles within K10_2 of their mean; every final training loss at or
           below the largest of M_inf's (job 6's and job 7's runs)
    K10.3  (read when K10.2 holds) their mean within K10_3 of the target
    K10.4  their mean forces at most K10_4 above the polished control's, at every size
"""
import argparse
import json

import numpy as np

import e4_job7 as J7
import e4_job9 as J9

WARM = dict(J7.WARM)                                   # M_1w050, M_1w100, M_1w150: their starts c0
OFF = ("M_1w050", "M_1w150")
CONTROL = "M_1q110"
K10_1, K10_2, K10_3, K10_4 = 0.25, 0.15, 0.10, 0.07
SEEDS = (0, 1)
SIZES, PERMANENT_DEBYE = J7.SIZES, J7.PERMANENT_DEBYE


def target(prof):
    return prof["c_min"] * PERMANENT_DEBYE if prof is not None else None


def movement(runs, job7, prof):
    """K10.1: each off-scale start's share of its distance to the target covered by the polish."""
    t = target(prof)
    rows = {}
    for rung in OFF:
        for s in SEEDS:
            r, r0 = runs.get(rung, {}).get(s), job7.get(rung, {}).get(s)
            if r is None or r0 is None:
                continue
            d0, d1 = J7.learned_dipole(r0), J7.learned_dipole(r)
            share = (d1 - d0) / (t - d0) if None not in (d0, d1, t) and t != d0 else None
            rows[f"{rung} seed {s}"] = {"start_debye": d0, "end_debye": d1, "share": share}
    complete = len(rows) == len(OFF) * len(SEEDS) and all(v["share"] is not None for v in rows.values())
    return {"runs": rows, "band": K10_1, "target_debye": t,
            "holds": bool(all(v["share"] >= K10_1 for v in rows.values())) if complete else None}


def convergence(runs, job7=None, job6=None):
    """K10.2: the six polished warm starts' learned dipoles (test frames of 64) within K10_2 of their mean and every final
    training loss at or below M_inf's largest (job 6's and job 7's runs).  None when a run is missing."""
    lmax = J9.m_inf_loss_max(job7, job6)
    rows = {}
    for rung, c0 in WARM.items():
        for s in SEEDS:
            r = runs.get(rung, {}).get(s)
            if r is None:
                continue
            d, fl = J7.learned_dipole(r), J7.final_loss(r)
            rows[f"{rung} seed {s}"] = {"start": c0, "dipole_debye": d, "final_loss": fl,
                                        "gate": bool(lmax is not None and fl is not None and fl <= lmax)}
    out = {"runs": rows, "loss_max": lmax, "band": K10_2, "holds": None}
    dips = [v["dipole_debye"] for v in rows.values()]
    if len(rows) == len(WARM) * len(SEEDS) and all(d is not None for d in dips) and lmax is not None:
        mean = float(np.mean(dips))
        out["mean_debye"] = mean
        out["spread"] = [d / mean - 1.0 for d in dips]
        out["holds"] = bool(all(v["gate"] for v in rows.values()) and all(abs(x) <= K10_2 for x in out["spread"]))
    return out


def evaluate(runs, job7=None, job6=None):
    """K10.1-K10.4 and the readings beside them."""
    job7 = job7 or {}
    prof = J9.profile(job7)
    out = {"profile": prof, "K10.1": movement(runs, job7, prof)}
    conv = convergence(runs, job7, job6)
    out["K10.2"] = conv
    read = conv["holds"] is True and prof is not None
    k3 = {"band": K10_3, "read": read, "target_debye": target(prof), "holds": None}
    if "mean_debye" in conv and prof is not None:
        k3["mean_debye"] = conv["mean_debye"]
        k3["ratio"] = conv["mean_debye"] / target(prof) - 1.0
        if read:
            k3["holds"] = bool(abs(k3["ratio"]) <= K10_3)
    out["K10.3"] = k3
    rows, ok = {}, True
    for n in SIZES:
        v = [J7.pooled(runs[w][s], n) for w in WARM for s in SEEDS if s in runs.get(w, {})]
        c = J7.seed_mean(runs.get(CONTROL, {}), n, seeds=SEEDS) if runs.get(CONTROL) else None
        if len(v) != len(WARM) * len(SEEDS) or any(x is None for x in v) or c is None:
            ok = None
            continue
        rows[n] = {"warm": float(np.mean(v)), "control": c[0], "ratio": float(np.mean(v)) / c[0] - 1.0}
        ok = ok and rows[n]["ratio"] <= K10_4 if ok is not None else None
    out["K10.4"] = {"values": rows, "band": K10_4, "holds": None if ok is None else bool(ok)}
    # beside: every run before (job 7) and after the polish
    before_after = {}
    for rung, by in sorted(runs.items()):
        for s, r in sorted(by.items()):
            r0 = job7.get(rung, {}).get(s)
            st = r.get("training", {}).get("start", {})
            row = {"loss": (st.get("final_loss", {}).get("total"), J7.final_loss(r)),
                   "forces": {n: (J7.pooled(r0, n) if r0 is not None else None, J7.pooled(r, n)) for n in SIZES},
                   "train_forces_64": (r0["readings"]["cells"]["64"]["train"]["force_rigid_rmse"] if r0 is not None
                                       and "train" in r0["readings"]["cells"].get("64", {}) else None,
                                       r["readings"]["cells"]["64"].get("train", {}).get("force_rigid_rmse")),
                   "dipole": (J7.learned_dipole(r0) if r0 is not None else None, J7.learned_dipole(r))}
            d = J9.diagnostics(r)
            if d is not None:
                lb = d.get("lbfgs_trace", [])
                row["path"] = {str(v["step"]): (v["dipole_debye"], v["objective"]) for v in lb
                               if v["step"] % 250 == 0 or v is lb[-1]}
                curves = {}
                for label, c in d["scale_curve"].items():
                    L = dict(zip(c["s"], c["loss"]))
                    dip = c["charges"]["dipole_debye"]
                    curves[label] = {"dipole_debye": dip, "slope": c["slope"], "s_min": c["s_min"],
                                     "curvature": (L[1.01] - 2 * L[1.0] + L[0.99]) / 1e-4,
                                     "profile_curvature": (dip / PERMANENT_DEBYE) ** 2 * 2 * prof["coef"][0]
                                     if prof is not None else None,
                                     "profile_slope": J9.profile_slope(prof, dip / PERMANENT_DEBYE)
                                     if prof is not None else None}
                row["scale_curve"] = curves
            before_after[f"{rung} seed {s}"] = row
    out["before_after"] = before_after
    out["forces"] = {name: {n: J7.seed_mean(by, n, seeds=SEEDS) for n in SIZES} for name, by in runs.items()}
    out["forces_job7"] = {name: {n: J7.seed_mean(job7[name], n, seeds=SEEDS) if job7.get(name) else None for n in SIZES}
                          for name in runs}
    return out


def _verdict(h, read=True):
    return "not read" if (h is None or not read) else ("holds" if h else "FAILS")


def report(ev, energies=None):
    pct = lambda x: f"{100 * x:+.1f} %"
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    lines = ["E4, job 10 (notes/e4-job10-preregistration.md): job 7's end states polished with L-BFGS on every "
             "training frame"]
    prof = ev["profile"]
    lines.append(f"  the target: job 7's profile minimum, c* = {prof['c_min']:.3f} ({prof['c_min'] * PERMANENT_DEBYE:.3f} D)"
                 if prof is not None else "  the target: job 7's scan missing (--job7_dir)")
    k1 = ev["K10.1"]
    lines.append(f"  K10.1: {_verdict(k1['holds'])} (each of the four off-scale starts covers at least "
                 f"{100 * K10_1:.0f} % of its distance to the target)")
    for k, v in k1["runs"].items():
        lines.append(f"      {k}: {f(v['start_debye'])} -> {f(v['end_debye'])} D, "
                     + (f"{100 * v['share']:.0f} % of the way" if v["share"] is not None else "no reading"))
    c = ev["K10.2"]
    lines.append(f"  K10.2: {_verdict(c['holds'])} (the six polished warm starts within {100 * K10_2:.0f} % of their "
                 f"mean, every final training loss at or below M_inf's largest"
                 + (f", {c['loss_max']:.3e})" if c["loss_max"] is not None else ")"))
    for k, v in c["runs"].items():
        lines.append(f"      {k}: {f(v['dipole_debye'])} D"
                     + (f", final loss {v['final_loss']:.3e} ({'passes' if v['gate'] else 'FAILS'})"
                        if v["final_loss"] is not None else ""))
    if "mean_debye" in c:
        lines.append(f"      mean {c['mean_debye']:.3f} D; each against it " + ", ".join(pct(x) for x in c["spread"]))
    k3 = ev["K10.3"]
    lines.append(f"  K10.3: {_verdict(k3['holds'], k3['read'])} (their mean within {100 * K10_3:.0f} % of the target; "
                 "read when K10.2 holds)"
                 + (f": {k3['mean_debye']:.3f} against {k3['target_debye']:.3f} D ({pct(k3['ratio'])})"
                    if "ratio" in k3 else ""))
    w = ev["K10.4"]
    lines.append(f"  K10.4: {_verdict(w['holds'])} (their mean forces at most {100 * K10_4:.0f} % above the polished "
                 f"{CONTROL}'s, at every size)")
    for n, v in w["values"].items():
        lines.append(f"      n {n:>3s}: {v['warm']:.3f} against {v['control']:.3f} ({pct(v['ratio'])})")
    lines.append("\n  Beside them, not registered: every run before (job 7) and after the polish")
    for k, v in ev["before_after"].items():
        lines.append(f"    {k}: training loss {f(v['loss'][0], 2)} -> {f(v['loss'][1], 2)}; forces "
                     + "; ".join(f"n {n} {f(a)} -> {f(b)}" for n, (a, b) in v["forces"].items())
                     + f"; training frames at 64 {f(v['train_forces_64'][0])} -> {f(v['train_forces_64'][1])}"
                     + (f"; dipole {f(v['dipole'][0])} -> {f(v['dipole'][1])} D" if v["dipole"][1] is not None else ""))
        if "path" in v:
            lines.append("      the polish (dipole D, objective): " + ", ".join(
                f"{d:.3f}" + (f" ({o:.2f})" if o is not None else "") + f" at {s}" for s, (d, o) in v["path"].items()))
        for label, cv in v.get("scale_curve", {}).items():
            lines.append(f"      along the charges' scale, the rest held ({label}): curvature {cv['curvature']:.3e} against "
                         f"the profile's {f(cv['profile_curvature'], 1)}; slope {cv['slope']:+.3e} against "
                         f"{f(cv['profile_slope'], 1)}; minimum at s = {cv['s_min']:.3f}")
    lines.append("    every rung's forces after the polish (mean +- sd over seeds), and job 7's")
    for name, by in sorted(ev["forces"].items()):
        j7 = ev["forces_job7"].get(name, {})
        lines.append(f"      {name:8s} " + "; ".join(
            f"n {n} {v[0]:.3f} +- {v[1]:.3f}" + (f" (job 7 {j7[n][0]:.3f})" if j7.get(n) else "")
            for n, v in by.items() if v))
    if energies:
        lines.append("    energy per molecule (kcal/mol), unseen frames: " + "; ".join(
            f"{name} " + "/".join(f"{v[0]:.4f}" for v in by.values() if v) for name, by in sorted(energies.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", default="results/e4_job10")
    ap.add_argument("--job7_dir", default="results/e4_job7", help="job 7's results: the starts, the scan (the "
                                                                  "profile) and M_inf")
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
