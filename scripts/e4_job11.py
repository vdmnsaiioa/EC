"""
e4_job11.py -- E4 job 11's registered readings K11.1-K11.5 and its model rule (notes/e4-job11-preregistration.md;
v0.0.35) from the training jsons of scripts/e4_water.py train, without JAX.

    python scripts/e4_job11.py --out_dir results/e4_job11 --job7_dir results/e4_job7 --job8_dir results/e4_job8 \
        --job10_dir results/e4_job10

Every run of job 11 is a long L-BFGS polish of a trained run of the same rung and seed:

    (a) job 7's scan (M_1q070, M_1q100, M_1q145; M_1q110 polished in job 10) -- the profile under the long polish; and
        job 10's M_1w050 and M_1w150, continued
    (b) the field-term dagger at the 2.0 A c6 band (M_16pm_us_d20) and D' at it (M_16C2pm_us_d20, E4's model), from job 8

    K11.1  the polished profile's minimum c*_lb -- the parabola through the polished scan's final training losses at
           c = 0.7, 1.0, 1.1 (job 10) and 1.45, seed means -- within K11_1 of job 7's c* = 1.098
    K11.2  each of job 10's two M_1w050 runs moves its learned dipole (test frames of 64) up by at least K11_2 D
    K11.3  the polished field-term dagger's forces at least K11_3 below job 8's runs of the same seeds, at every size
    K11.4  the polished D''s forces at least K11_4 below job 8's runs of the same seeds, at every size
    K11.5  its infrared as job 8's: eps_inf on the held-out liquid within K11_5_EPS, the far probes' non-additive error
           at 10, 12 and 15 A and T5's within K11_5_POINTS percentage points
    model rule: if K11.4 and K11.5 hold, E4's model is D' at the 2.0 A band with the long polish
"""
import argparse
import json

import numpy as np

import e4_job7 as J7
import e4_job8 as J8
import e4_job9 as J9

SCAN_C = {"M_1q070": 0.7, "M_1q100": 1.0, "M_1q110": 1.1, "M_1q145": 1.45}   # M_1q110 from job 10
LOW, HIGH = "M_1w050", "M_1w150"
FIELD, DPRIME = "M_16pm_us_d20", "M_16C2pm_us_d20"
K11_1, K11_2, K11_3, K11_4, K11_5_EPS, K11_5_POINTS = 0.05, 0.06, 0.05, 0.03, 0.002, 0.5
C_STAR = 1.098                                     # job 7's profile minimum (scripts/e4_job9.py's profile)
SEEDS = (0, 1)
SIZES, PERMANENT_DEBYE = J7.SIZES, J7.PERMANENT_DEBYE


def polished_profile(runs, job10):
    """the scan polished: the seed means of the final training loss at SCAN_C (M_1q110 from job 10), the parabola
    through them, its minimum; and the forces' seed means at every size.  None when a scan run is missing."""
    src = lambda name: job10 if name == "M_1q110" else runs
    pts, forces = {}, {}
    for name, c in SCAN_C.items():
        by = src(name).get(name, {})
        v = [J7.final_loss(by[s]) for s in SEEDS if s in by]
        if len(v) != len(SEEDS) or any(x is None for x in v):
            return None
        pts[c] = float(np.mean(v))
        forces[str(c)] = {n: J7.seed_mean(by, n, seeds=SEEDS) for n in SIZES}
    cs = sorted(pts)
    coef = np.polyfit(cs, [pts[c] for c in cs], 2)
    c_min = float(-coef[1] / (2 * coef[0]))
    return {"points": {str(c): pts[c] for c in cs}, "coef": [float(x) for x in coef], "c_min": c_min,
            "loss_min": float(np.polyval(coef, c_min)), "forces": forces}


def paired(runs, ref, name):
    """{size: (polished mean, reference mean over the same seeds, ratio - 1)} and whether every size is there."""
    rows = {}
    for n in SIZES:
        a = J7.seed_mean(runs.get(name, {}), n, seeds=SEEDS) if runs.get(name) else None
        b = J7.seed_mean(ref.get(name, {}), n, seeds=SEEDS) if ref.get(name) else None
        if a is None or b is None:
            return rows, False
        rows[n] = (a[0], b[0], a[0] / b[0] - 1.0)
    return rows, True


def steps_done(by):
    return {s: r.get("training", {}).get("options", {}).get("lbfgs_steps_done") for s, r in sorted(by.items())}


def evaluate(runs, job7=None, job8=None, job10=None):
    job7, job8, job10 = job7 or {}, job8 or {}, job10 or {}
    out = {}
    prof = polished_profile(runs, job10)
    out["profile"] = prof
    out["profile_job7"] = J9.profile(job7)
    out["K11.1"] = {"band": K11_1, "c_min": prof["c_min"] if prof else None,
                    "holds": None if prof is None else bool(abs(prof["c_min"] - C_STAR) <= K11_1)}
    # K11.2: the low side continued
    rows, ok = {}, True
    for s in SEEDS:
        r, r0 = runs.get(LOW, {}).get(s), job10.get(LOW, {}).get(s)
        if r is None or r0 is None:
            ok = None
            continue
        d0, d1 = J7.learned_dipole(r0), J7.learned_dipole(r)
        rows[f"{LOW} seed {s}"] = {"start_debye": d0, "end_debye": d1, "moved": d1 - d0}
        ok = ok and (d1 - d0) >= K11_2 if ok is not None else None
    out["K11.2"] = {"runs": rows, "band": K11_2, "holds": None if ok is None else bool(ok)}
    # K11.3 and K11.4: the long polish for the E4-class models
    for key, name, band in (("K11.3", FIELD, K11_3), ("K11.4", DPRIME, K11_4)):
        rows, complete = paired(runs, job8, name)
        out[key] = {"rung": name, "values": rows, "band": band, "steps_done": steps_done(runs.get(name, {})),
                    "holds": bool(all(v[2] <= -band for v in rows.values())) if complete else None}
    # K11.5: D''s infrared against job 8's runs of the same seeds
    a = J8.infrared(runs[DPRIME]) if runs.get(DPRIME) else None
    b = J8.infrared({s: r for s, r in job8.get(DPRIME, {}).items() if s in SEEDS}) if job8.get(DPRIME) else None
    k5 = {"polished": a, "job8": b, "holds": None}
    if a and b and None not in (a["eps_inf"], b["eps_inf"], a["T5"], b["T5"]) \
            and all(a["probes"][d] is not None and b["probes"][d] is not None for d in J8.PROBE_DISTANCES):
        k5["holds"] = bool(abs(a["eps_inf"] - b["eps_inf"]) <= K11_5_EPS and abs(a["T5"] - b["T5"]) <= K11_5_POINTS
                           and all(abs(a["probes"][d] - b["probes"][d]) <= K11_5_POINTS for d in J8.PROBE_DISTANCES))
    out["K11.5"] = k5
    rule = out["K11.4"]["holds"] is True and k5["holds"] is True
    out["model_rule"] = {"fires": rule, "read": out["K11.4"]["holds"] is not None and k5["holds"] is not None}
    # beside: the from-above continuation, every run's start and end, the objective paths, the response after the polish
    target = prof["c_min"] * PERMANENT_DEBYE if prof else C_STAR * PERMANENT_DEBYE
    out["high"] = {f"{HIGH} seed {s}": {"start_debye": J7.learned_dipole(job10[HIGH][s]),
                                        "end_debye": J7.learned_dipole(runs[HIGH][s]), "target_debye": target}
                   for s in SEEDS if s in runs.get(HIGH, {}) and s in job10.get(HIGH, {})}
    paths = {}
    for name, by in sorted(runs.items()):
        for s, r in sorted(by.items()):
            tr = r.get("training", {}).get("lbfgs_trace", [])
            st = r.get("training", {}).get("start", {}).get("final_loss", {}).get("total")
            sel = [(k, v) for i, (k, v) in enumerate(tr) if v is not None and (k % 250 == 0 or i == len(tr) - 2)]
            d = J9.diagnostics(r)
            dip = {}
            if d is not None and d.get("lbfgs_trace"):
                dip = {str(v["step"]): v["dipole_debye"] for v in d["lbfgs_trace"]
                       if v["step"] % 250 == 0 or v is d["lbfgs_trace"][-1]}
            paths[f"{name} seed {s}"] = {"start_loss": st, "final_loss": J7.final_loss(r),
                                         "objective": {str(k): v for k, v in sel}, "dipole": dip,
                                         "steps_done": r.get("training", {}).get("options", {}).get("lbfgs_steps_done"),
                                         "train_forces_64": r["readings"]["cells"].get("64", {}).get("train", {})
                                         .get("force_rigid_rmse")}
    out["paths"] = paths
    resp = {}
    for n in SIZES:
        a = J7.seed_mean(runs.get(DPRIME, {}), n, seeds=SEEDS) if runs.get(DPRIME) else None
        b = J7.seed_mean(runs.get(FIELD, {}), n, seeds=SEEDS) if runs.get(FIELD) else None
        if a and b:
            resp[n] = a[0] / b[0] - 1.0
    out["response_after_polish"] = resp
    out["forces"] = {name: {n: J7.seed_mean(by, n, seeds=SEEDS) for n in SIZES} for name, by in runs.items()}
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev, energies=None):
    pct = lambda x: f"{100 * x:+.1f} %"
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    lines = ["E4, job 11 (notes/e4-job11-preregistration.md): long L-BFGS polishes -- the scan, the low side, E4's model"]
    prof, p7 = ev["profile"], ev["profile_job7"]
    if prof:
        lines.append("  the polished profile (final training loss): " + ", ".join(
            f"c {c}: {v:.2f}" for c, v in prof["points"].items()) + f"; its minimum at c*_lb = {prof['c_min']:.3f} "
            f"({prof['c_min'] * PERMANENT_DEBYE:.3f} D), {prof['loss_min']:.2f}"
            + (f"; job 7's: c* = {p7['c_min']:.3f}, {p7['loss_min']:.2f}" if p7 else ""))
    k1 = ev["K11.1"]
    lines.append(f"  K11.1: {_verdict(k1['holds'])} (the polished profile's minimum within {k1['band']:g} of c* = "
                 f"{C_STAR}): " + (f"{k1['c_min']:.3f}" if k1["c_min"] is not None else "not read"))
    k2 = ev["K11.2"]
    lines.append(f"  K11.2: {_verdict(k2['holds'])} (each of job 10's M_1w050 runs moves up by at least "
                 f"{K11_2:g} D in another polish)")
    for k, v in k2["runs"].items():
        lines.append(f"      {k}: {f(v['start_debye'])} -> {f(v['end_debye'])} D ({v['moved']:+.3f})")
    for key, word in (("K11.3", "the field-term dagger"), ("K11.4", "D' (E4's model)")):
        b = ev[key]
        lines.append(f"  {key}: {_verdict(b['holds'])} ({word}, polished, at least {100 * b['band']:.0f} % below job 8's "
                     f"runs of the same seeds, at every size; L-BFGS steps done "
                     + ", ".join(f"seed {s} {d}" for s, d in b["steps_done"].items()) + ")")
        for n, (a, r, x) in b["values"].items():
            lines.append(f"      n {n:>3s}: {a:.3f} against {r:.3f} ({pct(x)})")
    k5 = ev["K11.5"]
    lines.append(f"  K11.5: {_verdict(k5['holds'])} (D''s infrared as job 8's: eps_inf within {K11_5_EPS:g}, the far "
                 f"probes at 10 / 12 / 15 A and T5 within {K11_5_POINTS:g} points)")
    for label in ("polished", "job8"):
        v = k5[label]
        if v:
            lines.append(f"      {label:8s}: eps_inf {f(v['eps_inf'], 4)}; probes " + " / ".join(
                f"{f(v['probes'][d], 2)}" for d in J8.PROBE_DISTANCES) + f" %; T5 {f(v['T5'], 2)} %")
    m = ev["model_rule"]
    lines.append("  model rule: " + ("FIRES -- E4's model is D' at the 2.0 A band with the long polish" if m["fires"] else
                                     ("does not fire" if m["read"] else "not read")))
    lines.append("\n  Beside them, not registered")
    if prof:
        lines.append("    the polished scan's forces: " + "; ".join(
            f"c {c} " + "/".join(f"{v[n][0]:.3f}" for n in SIZES if v.get(n)) for c, v in prof["forces"].items()))
    for k, v in ev["high"].items():
        lines.append(f"    from above, continued, {k}: {f(v['start_debye'])} -> {f(v['end_debye'])} D (target "
                     f"{v['target_debye']:.3f} D)")
    if ev["response_after_polish"]:
        lines.append("    the response's bulk gain after the polish (D' against the field-term dagger): "
                     + ", ".join(f"n {n} {pct(x)}" for n, x in ev["response_after_polish"].items()))
    lines.append("    the polishes (objective by L-BFGS step; the learned dipole, D; steps done)")
    for k, v in ev["paths"].items():
        lines.append(f"      {k}: {f(v['start_loss'], 2)} -> {f(v['final_loss'], 2)}; " + ", ".join(
            f"{o:.2f} at {s}" for s, o in v["objective"].items())
            + ("; dipole " + ", ".join(f"{d:.3f} at {s}" for s, d in v["dipole"].items()) if v["dipole"] else "")
            + f"; {v['steps_done']} steps; training frames at 64 {f(v['train_forces_64'])}")
    lines.append("    every rung's forces (mean +- sd over seeds)")
    for name, by in sorted(ev["forces"].items()):
        lines.append(f"      {name:16s} " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}" for n, v in by.items() if v))
    if energies:
        lines.append("    energy per molecule (kcal/mol), unseen frames: " + "; ".join(
            f"{name} " + "/".join(f"{v[0]:.4f}" for v in by.values() if v) for name, by in sorted(energies.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", default="results/e4_job11")
    ap.add_argument("--job7_dir", default="results/e4_job7", help="job 7's results: the scan before the polish")
    ap.add_argument("--job8_dir", default="results/e4_job8", help="job 8's results: the E4-class models' starts")
    ap.add_argument("--job10_dir", default="results/e4_job10", help="job 10's results: M_1q110 polished; M_1w050, "
                                                                    "M_1w150's starts")
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    runs = J7.load_runs(a.out_dir)
    if not runs:
        raise SystemExit(f"no results in {a.out_dir}")
    ev = evaluate(runs, J7.load_runs(a.job7_dir), J7.load_runs(a.job8_dir), J7.load_runs(a.job10_dir))
    en = {name: {n: J7.seed_mean(by, n, "energy_per_molecule_rmse", seeds=SEEDS) for n in SIZES}
          for name, by in runs.items()}
    print(report(ev, en))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"readings": ev, "energies": en}, fh, indent=1)


if __name__ == "__main__":
    main()
