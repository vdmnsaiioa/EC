"""
e4_job12.py -- E4 job 12's registered readings K12.0-K12.3 and its model rule (notes/e4-job12-preregistration.md;
v0.0.36) from the training jsons of scripts/e4_water.py train, without JAX.

    python scripts/e4_job12.py --out_dir results/e4_job12 --job8_dir results/e4_job8 --job11_dir results/e4_job11

Job 12 polishes E4's model -- D' at the 2.0 A c6 band, M_16C2pm_us_d20 -- from job 8's runs of seeds 0-3 with 1000
full-batch L-BFGS steps, its response precomputed on the training frames (training.response_precomputed): the field-term
dagger M_16pm_us_d20, the same model less the read-out, trained against the remainder, the read-out held.

    K12.0  exactness, in every run: the response's constants unchanged by other shared parameters, and D''s objective
           and the dagger's against the remainder equal at the start and at the end, each within K12_0 (relative); and
           the objective at the start equal to job 11's D' at its step 0 (seeds 0 and 1), within K12_0
    K12.1  the polished D''s forces at least K12_1 below job 8's runs of the same seeds (seed means), at every size
    K12.2  its infrared as job 8's: eps_inf on the held-out liquid within K12_2_EPS, the far probes' non-additive error at
           10, 12 and 15 A and T5's within K12_2_POINTS percentage points
    K12.3  the response's bulk gain at equal steps: the polished D''s forces at 128 and 256 molecules at least K12_3
           below job 11's polished field-term dagger, seed means over seeds 0 and 1
    model rule: if K12.0, K12.1 and K12.2 hold, E4's model is D' at the 2.0 A band with the long polish
"""
import argparse
import json

import numpy as np

import e4_job7 as J7
import e4_job8 as J8

DPRIME, FIELD = "M_16C2pm_us_d20", "M_16pm_us_d20"
SEEDS, PAIRED = (0, 1, 2, 3), (0, 1)                  # job 8's seeds; job 11 polished seeds 0 and 1
K12_0, K12_1, K12_2_EPS, K12_2_POINTS, K12_3 = 1e-6, 0.03, 0.002, 0.5, 0.015
K12_3_SIZES = ("128", "256")
STEPS = 1000
SIZES = J7.SIZES


def _rel(a, b):
    return None if a is None or b is None else abs(a / b - 1.0)


def exactness(runs, job11):
    """K12.0, run by run: the invariance check, the objectives at the start and the end, and the start against job 11's
    D' at its step 0."""
    rows, ok = {}, True
    by = runs.get(DPRIME, {})
    if any(s not in by for s in SEEDS):
        ok = None
    for s, r in sorted(by.items()):
        pre = r.get("training", {}).get("response_precomputed")
        if not pre:
            rows[f"seed {s}"] = {"precomputed": False}
            ok = False if ok is not None else None
            continue
        inv = pre.get("invariance", {})
        st, en = pre.get("objective_start", {}), pre.get("objective_end", {})
        row = {"precomputed": True, "dagger": pre.get("dagger"), "invariance": [inv.get("energy"), inv.get("forces")],
               "start": _rel(st.get("dagger"), st.get("rung")), "end": _rel(en.get("dagger"), en.get("rung"))}
        tr = r.get("training", {}).get("lbfgs_trace") or []
        t11 = job11.get(DPRIME, {}).get(s, {}).get("training", {}).get("lbfgs_trace") or []
        if s in PAIRED:
            row["job11_step0"] = _rel(tr[0][1], t11[0][1]) if tr and t11 and tr[0][0] == t11[0][0] == 0 else None
        checks = row["invariance"] + [row["start"], row["end"]] + ([row["job11_step0"]] if s in PAIRED else [])
        if any(x is None for x in checks):
            ok = None if ok is not False else False
        elif not all(x <= K12_0 for x in checks) or pre.get("dagger") != FIELD:
            ok = False
        rows[f"seed {s}"] = row
    return {"band": K12_0, "runs": rows, "holds": ok if ok is None else bool(ok)}


def against(a_by, b_by, seeds, sizes=SIZES):
    """{size: (a's seed mean, b's seed mean over the same seeds, ratio - 1)} and whether every size is there."""
    rows = {}
    for n in sizes:
        a = J7.seed_mean(a_by, n, seeds=seeds) if a_by else None
        b = J7.seed_mean(b_by, n, seeds=seeds) if b_by else None
        if a is None or b is None:
            return rows, False
        rows[n] = (a[0], b[0], a[0] / b[0] - 1.0)
    return rows, True


def steps_done(r):
    o = r.get("training", {}).get("options", {})
    return o.get("lbfgs_steps_done", o.get("lbfgs_steps"))


def evaluate(runs, job8=None, job11=None):
    job8, job11 = job8 or {}, job11 or {}
    out = {"K12.0": exactness(runs, job11)}
    d12, d8 = runs.get(DPRIME, {}), job8.get(DPRIME, {})
    rows, complete = against(d12, d8, SEEDS)
    out["K12.1"] = {"values": rows, "band": K12_1, "steps_done": {s: steps_done(r) for s, r in sorted(d12.items())},
                    "holds": bool(all(v[2] <= -K12_1 for v in rows.values())) if complete else None}
    a = J8.infrared({s: d12[s] for s in SEEDS}) if all(s in d12 for s in SEEDS) else None
    b = J8.infrared({s: d8[s] for s in SEEDS}) if all(s in d8 for s in SEEDS) else None
    k2 = {"polished": a, "job8": b, "holds": None}
    if a and b and None not in (a["eps_inf"], b["eps_inf"], a["T5"], b["T5"]) \
            and all(a["probes"][d] is not None and b["probes"][d] is not None for d in J8.PROBE_DISTANCES):
        k2["holds"] = bool(abs(a["eps_inf"] - b["eps_inf"]) <= K12_2_EPS and abs(a["T5"] - b["T5"]) <= K12_2_POINTS
                           and all(abs(a["probes"][d] - b["probes"][d]) <= K12_2_POINTS for d in J8.PROBE_DISTANCES))
    out["K12.2"] = k2
    rows, complete = against(d12, job11.get(FIELD, {}), PAIRED, K12_3_SIZES)
    out["K12.3"] = {"values": rows, "band": K12_3,
                    "holds": bool(all(v[2] <= -K12_3 for v in rows.values())) if complete else None}
    hs = [out[k]["holds"] for k in ("K12.0", "K12.1", "K12.2")]
    out["model_rule"] = {"fires": all(h is True for h in hs), "read": all(h is not None for h in hs)}
    # beside: equal steps at 64, the objective paths against job 11's (D' and the dagger), the cost of a step, the
    # training frames' forces against the unseen (over-fitting), every run's forces
    out["equal_steps_64"] = against(d12, job11.get(FIELD, {}), PAIRED, ("64",))[0]
    out["job11_dprime"] = against(job11.get(DPRIME, {}), d8, PAIRED)[0]
    paths = {}
    for s, r in sorted(d12.items()):
        tr = r.get("training", {}).get("lbfgs_trace") or []
        mine = {k: v for k, v in tr if v is not None}
        t11 = {k: v for k, v in (job11.get(DPRIME, {}).get(s, {}).get("training", {}).get("lbfgs_trace") or [])
               if v is not None}
        tf = {k: v for k, v in (job11.get(FIELD, {}).get(s, {}).get("training", {}).get("lbfgs_trace") or [])
              if v is not None}
        n = steps_done(r)
        fit_time = r.get("training", {}).get("fit_time")
        paths[f"seed {s}"] = {
            "objective": {str(k): v for k, v in mine.items() if k % 250 == 0 or k == max(mine)} if mine else {},
            "job11_dprime": {str(k): (mine[k], t11[k], mine[k] / t11[k] - 1.0) for k in sorted(t11) if k in mine
                             and k % 50 == 0},
            "field_dagger": {str(k): (mine[k], tf[k]) for k in sorted(tf) if k in mine and k % 250 == 0},
            "steps_done": n, "seconds_per_step": fit_time / n if fit_time and n else None,
            "train_forces_64": r["readings"]["cells"].get("64", {}).get("train", {}).get("force_rigid_rmse"),
            "final_loss": J7.final_loss(r),
            "start_loss": r.get("training", {}).get("start", {}).get("final_loss", {}).get("total")}
    out["paths"] = paths
    out["forces"] = {label: {n: J7.seed_mean(by, n, seeds=seeds) for n in SIZES}
                     for label, by, seeds in (("job 12 " + DPRIME, d12, SEEDS), ("job 8 " + DPRIME, d8, SEEDS),
                                              ("job 11 " + FIELD, job11.get(FIELD, {}), PAIRED))
                     if by and all(s in by for s in seeds)}
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev, energies=None):
    pct = lambda x: f"{100 * x:+.1f} %"
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    e = lambda x: "-" if x is None else f"{x:.1e}"
    lines = ["E4, job 12 (notes/e4-job12-preregistration.md): E4's model polished, its response precomputed"]
    k0 = ev["K12.0"]
    lines.append(f"  K12.0: {_verdict(k0['holds'])} (exactness within {k0['band']:g}: the response's constants under "
                 "other shared parameters; D''s objective against the dagger's on the remainder, start and end; the "
                 "start against job 11's step 0)")
    for k, v in k0["runs"].items():
        if not v.get("precomputed"):
            lines.append(f"      {k}: the response was not precomputed")
            continue
        lines.append(f"      {k}: invariance {e(v['invariance'][0])} / {e(v['invariance'][1])}; start {e(v['start'])}, "
                     f"end {e(v['end'])}" + (f"; job 11's step 0 {e(v['job11_step0'])}" if "job11_step0" in v else ""))
    k1 = ev["K12.1"]
    lines.append(f"  K12.1: {_verdict(k1['holds'])} (D', polished, at least {100 * k1['band']:.0f} % below job 8's runs "
                 f"of the same seeds, at every size; L-BFGS steps done "
                 + ", ".join(f"seed {s} {d}" for s, d in k1["steps_done"].items()) + ")")
    for n, (a, r, x) in k1["values"].items():
        lines.append(f"      n {n:>3s}: {a:.3f} against {r:.3f} ({pct(x)})")
    k2 = ev["K12.2"]
    lines.append(f"  K12.2: {_verdict(k2['holds'])} (D''s infrared as job 8's: eps_inf within {K12_2_EPS:g}, the far "
                 f"probes at 10 / 12 / 15 A and T5 within {K12_2_POINTS:g} points)")
    for label in ("polished", "job8"):
        v = k2[label]
        if v:
            lines.append(f"      {label:8s}: eps_inf {f(v['eps_inf'], 4)}; probes " + " / ".join(
                f"{f(v['probes'][d], 2)}" for d in J8.PROBE_DISTANCES) + f" %; T5 {f(v['T5'], 2)} %")
    k3 = ev["K12.3"]
    lines.append(f"  K12.3: {_verdict(k3['holds'])} (at equal steps, D' at least {100 * k3['band']:.1f} % below job 11's "
                 f"polished field-term dagger at 128 and 256 molecules, seeds 0 and 1)")
    for n, (a, r, x) in k3["values"].items():
        lines.append(f"      n {n:>3s}: {a:.3f} against {r:.3f} ({pct(x)})")
    m = ev["model_rule"]
    lines.append("  model rule: " + ("FIRES -- E4's model is D' at the 2.0 A band with the long polish" if m["fires"] else
                                     ("does not fire" if m["read"] else "not read")))
    lines.append("\n  Beside them, not registered")
    if ev["equal_steps_64"]:
        a, r, x = ev["equal_steps_64"]["64"]
        lines.append(f"    at equal steps at 64 molecules: D' {a:.3f} against the field-term dagger {r:.3f} ({pct(x)})")
    if ev["job11_dprime"]:
        lines.append("    job 11's D' (277-290 steps) against job 8's, seeds 0 and 1: " + ", ".join(
            f"n {n} {pct(v[2])}" for n, v in ev["job11_dprime"].items()))
    lines.append("    the polishes (objective by step; seconds a step; the training frames' forces at 64)")
    for k, v in ev["paths"].items():
        lines.append(f"      {k}: {f(v['start_loss'], 2)} -> {f(v['final_loss'], 2)}; " + ", ".join(
            f"{o:.2f} at {s}" for s, o in v["objective"].items()) + f"; {v['steps_done']} steps, "
            f"{f(v['seconds_per_step'], 1)} s a step; training frames at 64 {f(v['train_forces_64'])}")
        if v["job11_dprime"]:
            lines.append("        against job 11's D' (job 12 / job 11): " + ", ".join(
                f"{s}: {a:.4f} / {b:.4f} ({x:+.1e})" for s, (a, b, x) in v["job11_dprime"].items()))
        if v["field_dagger"]:
            lines.append("        against job 11's field-term dagger: " + ", ".join(
                f"{s}: {a:.2f} / {b:.2f}" for s, (a, b) in v["field_dagger"].items()))
    lines.append("    forces (mean +- sd over seeds)")
    for label, by in ev["forces"].items():
        lines.append(f"      {label:26s} " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}" for n, v in by.items() if v))
    if energies:
        lines.append("    energy per molecule (kcal/mol), unseen frames: " + "; ".join(
            f"{name} " + "/".join(f"{v[0]:.4f}" for v in by.values() if v) for name, by in sorted(energies.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", default="results/e4_job12")
    ap.add_argument("--job8_dir", default="results/e4_job8", help="job 8's results: D''s starts, the reference")
    ap.add_argument("--job11_dir", default="results/e4_job11", help="job 11's results: the polished field-term dagger, "
                                                                    "D''s polish within its budget")
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    runs = J7.load_runs(a.out_dir)
    if not runs:
        raise SystemExit(f"no results in {a.out_dir}")
    ev = evaluate(runs, J7.load_runs(a.job8_dir), J7.load_runs(a.job11_dir))
    en = {name: {n: J7.seed_mean(by, n, "energy_per_molecule_rmse") for n in SIZES} for name, by in runs.items()}
    print(report(ev, en))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"readings": ev, "energies": en}, fh, indent=1)


if __name__ == "__main__":
    main()
