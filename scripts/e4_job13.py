"""
e4_job13.py -- E4 job 13's registered readings K13.1-K13.5 (notes/e4-job13-preregistration.md; v0.0.37) from the
training jsons of scripts/e4_water.py train, without JAX.

    python scripts/e4_job13.py --out_dir results/e4_job13 --stage_dir results/e4_job13s --job9_dir results/e4_job9 \
        --job10_dir results/e4_job10 --job11_dir results/e4_job11

Job 13 asks why learned charges end where their start puts them (jobs 7, 10 and 11: from 0.5 at 1.59-1.61 D, from 1.0 at
1.93-2.01 D, from 1.5 at 2.11-2.18 D, while the polished profile's minimum is c*_lb = 1.070, 1.995 D).  Three arms, every
run 1000 full-batch L-BFGS steps on job 7's settings:

    R  released from the polished profile (job 11's M_1q070, M_1q145): the pinned charges at c0 = 0.7 and 1.45 become
       learned ones at c0, the same model at the start -- by species alone, the charge head held (M_1zr070, M_1zr145),
       or with their environment dependence (M_1r070, M_1r145)
    T  the pinned charges moved to c = 1.1 with E_0 from the same states (M_1t070, M_1t145)
    S  learned charges by species alone from the initialisation at 0.5 and 1.5 (M_1z050, M_1z150): job 7's training
       (results/e4_job13s), then job 10's polish

    K13.1  R by species: every run's learned dipole (test frames of 64 molecules) within K13_C of c*_lb, in c
    K13.2  R with the head: the same
    K13.3  the head's part: by species nearer c*_lb than with the head on at least K13_3 of the 4 (start, seed) pairs
    K13.4  T: every run's final training loss at most K13_4 above the polished control's (job 10's M_1q110, same seed)
    K13.5  S: every run (after the polish) within K13_C of c*_lb
"""
import argparse
import json

import numpy as np

import e4_job7 as J7
import e4_job11 as J11

BY_SPECIES = {"M_1zr070": 0.7, "M_1zr145": 1.45}       # R, the charge head held
WITH_HEAD = {"M_1r070": 0.7, "M_1r145": 1.45}          # R, the charge head learned
TRANSPLANT = {"M_1t070": 0.7, "M_1t145": 1.45}         # T: pinned at 1.1, E_0 from the polished state at c0
SPECIES = {"M_1z050": 0.5, "M_1z150": 1.5}             # S
FLEXIBLE = {"M_1w050": 0.5, "M_1w150": 1.5}            # jobs 7-11's flexible warm starts
SOURCE = {0.7: "M_1q070", 1.45: "M_1q145"}             # the starts of R and T (job 11)
CONTROL = "M_1q110"                                    # T's reference: the polished control (job 10)
C_STAR_LB, K13_C, K13_3, K13_4 = 1.070, 0.05, 3, 0.05
SEEDS = (0, 1)
SIZES, PERMANENT_DEBYE = J7.SIZES, J7.PERMANENT_DEBYE


def end_c(r):
    d = J7.learned_dipole(r)
    return None if d is None else d / PERMANENT_DEBYE


def ends(runs, arm):
    """every run of an arm: its start c0, its end (test frames of 64), its distance from c*_lb; and whether all are within
    K13_C (None when a run or a reading is missing)."""
    rows = {}
    for rung, c0 in arm.items():
        for s in SEEDS:
            r = runs.get(rung, {}).get(s)
            if r is None:
                continue
            c = end_c(r)
            rows[f"{rung} seed {s}"] = {"start_c": c0, "end_c": c, "end_debye": J7.learned_dipole(r),
                                        "off": None if c is None else c - C_STAR_LB, "final_loss": J7.final_loss(r)}
    complete = len(rows) == len(arm) * len(SEEDS) and all(v["end_c"] is not None for v in rows.values())
    return {"runs": rows, "band": K13_C,
            "holds": bool(all(abs(v["off"]) <= K13_C for v in rows.values())) if complete else None}


def head_pairs(runs):
    """K13.3: for each start and seed, whether the run by species ends nearer c*_lb than the run with the head."""
    rows = {}
    for (zr, c0), (rr, _) in zip(BY_SPECIES.items(), WITH_HEAD.items()):
        for s in SEEDS:
            a, b = runs.get(zr, {}).get(s), runs.get(rr, {}).get(s)
            ca, cb = (end_c(a) if a else None), (end_c(b) if b else None)
            if ca is None or cb is None:
                continue
            rows[f"c0 {c0} seed {s}"] = {"species_c": ca, "head_c": cb,
                                         "species_nearer": bool(abs(ca - C_STAR_LB) < abs(cb - C_STAR_LB))}
    n = sum(v["species_nearer"] for v in rows.values())
    return {"pairs": rows, "species_nearer": n, "band": K13_3,
            "holds": bool(n >= K13_3) if len(rows) == len(BY_SPECIES) * len(SEEDS) else None}


def transplants(runs, job10):
    """K13.4: every transplant's final training loss against the polished control's of the same seed."""
    rows, ok = {}, True
    for rung, c0 in TRANSPLANT.items():
        for s in SEEDS:
            r, ref = runs.get(rung, {}).get(s), job10.get(CONTROL, {}).get(s)
            if r is None or ref is None or J7.final_loss(r) is None or J7.final_loss(ref) is None:
                ok = None
                continue
            a, b = J7.final_loss(r), J7.final_loss(ref)
            rows[f"{rung} seed {s}"] = {"from_c": c0, "final_loss": a, "control": b, "ratio": a / b - 1.0,
                                        "start_loss": r.get("training", {}).get("start", {}).get("final_loss", {})
                                        .get("total"),
                                        "forces": {n: J7.pooled(r, n) for n in SIZES},
                                        "control_forces": {n: J7.pooled(ref, n) for n in SIZES}}
            ok = ok and a / b - 1.0 <= K13_4 if ok is not None else None
    return {"runs": rows, "band": K13_4, "holds": ok if ok is None else bool(ok)}


def curvature(v):
    """the curvature of the objective along the charges' scale with the rest held, from s = 0.99, 1, 1.01."""
    s, L = v["s"], v["loss"]
    return (L[s.index(1.01)] - 2.0 * L[s.index(1.0)] + L[s.index(0.99)]) / 1e-4


def path(r, target_debye):
    """a learned run's dipole on the traced frames through the polish (every 250 steps and the last), the last 250
    steps' move against the distance left to the target there, and the objective; and the valley's curvature with the
    rest held at the start and the end."""
    d = r.get("training", {}).get("charges") or {}
    lb = d.get("lbfgs_trace") or []
    dip = {v["step"]: v["dipole_debye"] for v in lb}
    out = {"dipole": {str(k): v for k, v in dip.items() if k % 250 == 0 or k == max(dip)} if dip else {}}
    if dip:
        last = max(dip)
        before = max((k for k in dip if k <= last - 250), default=None)
        if before is not None:
            moved, left = dip[last] - dip[before], target_debye - dip[before]
            out["last_250"] = {"moved": moved, "left": left, "share": moved / left if left else None}
    tr = r.get("training", {}).get("lbfgs_trace") or []
    out["objective"] = {str(k): v for i, (k, v) in enumerate(tr) if v is not None and (k % 250 == 0 or i == len(tr) - 2)}
    out["curvature"] = {k: curvature(v) for k, v in (d.get("scale_curve") or {}).items()}
    return out


def evaluate(runs, stage=None, job9=None, job10=None, job11=None):
    stage, job9, job10, job11 = stage or {}, job9 or {}, job10 or {}, job11 or {}
    out = {"c_star_lb": C_STAR_LB}
    prof = J11.polished_profile(job11, job10)                 # job 11's polished profile, recomputed
    out["profile"] = prof
    out["K13.1"] = ends(runs, BY_SPECIES)
    out["K13.2"] = ends(runs, WITH_HEAD)
    out["K13.3"] = head_pairs(runs)
    out["K13.4"] = transplants(runs, job10)
    out["K13.5"] = ends(runs, SPECIES)
    # beside: the paths, the end objective against the polished profile at the end's c, the starts, S's first stage
    # against job 9's flexible warm starts and its polish against job 10's, every run's forces
    target = C_STAR_LB * PERMANENT_DEBYE
    paths, against_profile = {}, {}
    for arm in (BY_SPECIES, WITH_HEAD, SPECIES, TRANSPLANT):
        for rung in arm:
            for s, r in sorted(runs.get(rung, {}).items()):
                paths[f"{rung} seed {s}"] = path(r, target)
                c = end_c(r) if rung not in TRANSPLANT else 1.1
                if prof and c is not None and J7.final_loss(r) is not None:
                    at = float(np.polyval(prof["coef"], c))
                    against_profile[f"{rung} seed {s}"] = {"c": c, "final_loss": J7.final_loss(r), "profile": at,
                                                           "ratio": J7.final_loss(r) / at - 1.0}
    out["paths"], out["against_profile"] = paths, against_profile
    out["starts"] = {f"{name} seed {s}": {"final_loss": J7.final_loss(r), "forces": {n: J7.pooled(r, n) for n in SIZES}}
                     for name in SOURCE.values() for s, r in sorted(job11.get(name, {}).items()) if s in SEEDS}
    first = {}
    for rung, c0 in SPECIES.items():
        flex = [k for k, v in FLEXIBLE.items() if v == c0][0]
        for s in SEEDS:
            a, b = stage.get(rung, {}).get(s), job9.get(flex, {}).get(s)
            first[f"c0 {c0} seed {s}"] = {"species_debye": J7.learned_dipole(a) if a else None,
                                          "flexible_debye": J7.learned_dipole(b) if b else None,
                                          "species_loss": J7.final_loss(a) if a else None,
                                          "flexible_loss": J7.final_loss(b) if b else None}
    out["stage"] = first
    second = {}
    for rung, c0 in SPECIES.items():
        flex = [k for k, v in FLEXIBLE.items() if v == c0][0]
        for s in SEEDS:
            a, b, c = runs.get(rung, {}).get(s), job10.get(flex, {}).get(s), job11.get(flex, {}).get(s)
            second[f"c0 {c0} seed {s}"] = {"species_debye": J7.learned_dipole(a) if a else None,
                                           "flexible_job10_debye": J7.learned_dipole(b) if b else None,
                                           "flexible_job11_debye": J7.learned_dipole(c) if c else None,
                                           "species_loss": J7.final_loss(a) if a else None,
                                           "flexible_job10_loss": J7.final_loss(b) if b else None}
    out["polish"] = second
    out["forces"] = {name: {n: J7.seed_mean(by, n, seeds=SEEDS) for n in SIZES} for name, by in sorted(runs.items())
                     if all(s in by for s in SEEDS)}
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev, energies=None):
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    lines = [f"E4, job 13 (notes/e4-job13-preregistration.md): why learned charges end where their start puts them; "
             f"c*_lb = {C_STAR_LB} ({C_STAR_LB * PERMANENT_DEBYE:.3f} D)"]
    prof = ev["profile"]
    if prof:
        lines.append(f"  the polished profile (jobs 10-11), recomputed: minimum at c = {prof['c_min']:.3f}")
    for key, word in (("K13.1", "R by species, the charge head held (M_1zr070, M_1zr145)"),
                      ("K13.2", "R with the head (M_1r070, M_1r145)"),
                      ("K13.5", "S by species from the initialisation, polished (M_1z050, M_1z150)")):
        b = ev[key]
        lines.append(f"  {key}: {_verdict(b['holds'])} ({word}: every run within {b['band']:g} of c*_lb)")
        for k, v in b["runs"].items():
            off = "-" if v["off"] is None else f"{v['off']:+.3f}"
            lines.append(f"      {k}: from c {v['start_c']:g} to {f(v['end_c'])} ({f(v['end_debye'])} D, {off}); final "
                         f"loss {f(v['final_loss'], 2)}")
        if key == "K13.2":
            k3 = ev["K13.3"]
            lines.append(f"  K13.3: {_verdict(k3['holds'])} (by species nearer c*_lb than with the head on at least "
                         f"{k3['band']} of 4 pairs: {k3['species_nearer']})")
            for k, v in k3["pairs"].items():
                lines.append(f"      {k}: by species {v['species_c']:.3f}, with the head {v['head_c']:.3f}")
    k4 = ev["K13.4"]
    lines.append(f"  K13.4: {_verdict(k4['holds'])} (T: the pinned charges at 1.1 with E_0 from c0, every final loss at "
                 f"most {100 * k4['band']:.0f} % above the polished control M_1q110's of the same seed)")
    for k, v in k4["runs"].items():
        lines.append(f"      {k}: {f(v['start_loss'], 1)} at the start -> {v['final_loss']:.2f} against "
                     f"{v['control']:.2f} ({100 * v['ratio']:+.1f} %); forces "
                     + "/".join(f(v["forces"][n]) for n in SIZES) + " against "
                     + "/".join(f(v["control_forces"][n]) for n in SIZES))
    lines.append("\n  Beside them, not registered")
    for k, v in ev["starts"].items():
        lines.append(f"    the start {k}: final loss {f(v['final_loss'], 2)}; forces "
                     + "/".join(f(v["forces"][n]) for n in SIZES))
    lines.append("    the end's final loss against the polished profile at its c (T: at 1.1)")
    for k, v in ev["against_profile"].items():
        lines.append(f"      {k}: c {v['c']:.3f}, {v['final_loss']:.2f} against {v['profile']:.2f} "
                     f"({100 * v['ratio']:+.1f} %)")
    lines.append("    the paths (the learned dipole on the traced frames, D; the objective; the valley's curvature "
                 "with the rest held, start -> final)")
    for k, v in ev["paths"].items():
        l250 = v.get("last_250")
        lines.append(f"      {k}: " + (", ".join(f"{d:.3f} at {s}" for s, d in v["dipole"].items()) + "; "
                                       if v["dipole"] else "")
                     + (f"the last 250 steps {l250['moved']:+.3f} of {l250['left']:+.3f} D; " if l250 else "")
                     + ", ".join(f"{o:.2f} at {s}" for s, o in v["objective"].items())
                     + ("; curvature " + " -> ".join(f"{c:.2e}" for c in v["curvature"].values())
                        if v["curvature"] else ""))
    lines.append("    S's first stage (job 7's training) against job 9's flexible warm starts: the learned dipole (D), "
                 "the final loss")
    for k, v in ev["stage"].items():
        lines.append(f"      {k}: by species {f(v['species_debye'])} ({f(v['species_loss'], 1)}), flexible "
                     f"{f(v['flexible_debye'])} ({f(v['flexible_loss'], 1)})")
    lines.append("    S polished against the flexible warm starts polished (job 10) and continued (job 11)")
    for k, v in ev["polish"].items():
        lines.append(f"      {k}: by species {f(v['species_debye'])} ({f(v['species_loss'], 2)}); flexible "
                     f"{f(v['flexible_job10_debye'])} ({f(v['flexible_job10_loss'], 2)}), continued "
                     f"{f(v['flexible_job11_debye'])}")
    lines.append("    forces (mean +- sd over seeds)")
    for name, by in ev["forces"].items():
        lines.append(f"      {name:10s} " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}" for n, v in by.items() if v))
    if energies:
        lines.append("    energy per molecule (kcal/mol), unseen frames: " + "; ".join(
            f"{name} " + "/".join(f"{v[0]:.4f}" for v in by.values() if v) for name, by in sorted(energies.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", default="results/e4_job13")
    ap.add_argument("--stage_dir", default="results/e4_job13s", help="S's first stage (job 7's training)")
    ap.add_argument("--job9_dir", default="results/e4_job9", help="job 9's results: job 7's warm starts, traced")
    ap.add_argument("--job10_dir", default="results/e4_job10", help="job 10's results: M_1q110 and the warm starts "
                                                                    "polished")
    ap.add_argument("--job11_dir", default="results/e4_job11", help="job 11's results: the polished profile, the warm "
                                                                    "starts continued")
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    runs = J7.load_runs(a.out_dir)
    if not runs:
        raise SystemExit(f"no results in {a.out_dir}")
    ev = evaluate(runs, J7.load_runs(a.stage_dir), J7.load_runs(a.job9_dir), J7.load_runs(a.job10_dir),
                  J7.load_runs(a.job11_dir))
    en = {name: {n: J7.seed_mean(by, n, "energy_per_molecule_rmse", seeds=SEEDS) for n in SIZES}
          for name, by in runs.items()}
    print(report(ev, en))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"readings": ev, "energies": en}, fh, indent=1)


if __name__ == "__main__":
    main()
