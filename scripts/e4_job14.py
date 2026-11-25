"""
e4_job14.py -- E4 job 14's registered readings K14.1-K14.4 (notes/e4-job14-preregistration.md; v0.0.38) from the
training jsons of scripts/e4_water.py train, without JAX.

    python scripts/e4_job14.py --job9_dir results/e4_job9 --job10_dir results/e4_job10 --job11_dir results/e4_job11

Job 14 asks whether learned charges find the data's scale when E_0 cannot span the training cell.  E_0 has one
message-passing layer of r_cut = 4.5 A or 6 A (model.layers; job 7's has two of 6 A: a reach of 12 A in the 12.4 A
cells).  At each reach, job 7's scan (pinned O/H charges at c = 0.7, 1.0, 1.1, 1.45) and warm starts (learned charges
from 0.5 and 1.5) are trained as job 7 trained them (results/e4_job14s_<reach>) and polished as jobs 10 and 11 polished
them (results/e4_job14_<reach>).  The reference, a reach of 12 A, is job 11's polished profile (job 10's M_1q110) and
job 10's polished warm starts.

    K14.1  at 4.5 A, the polished profile's curvature at its minimum, c*^2 L*'' -- the part of the band's contribution E_0
           cannot absorb -- at least K14_1 times the 12 A profile's
    K14.2  at 6 A, at least K14_2 times
    K14.3  at 4.5 A, every polished warm start's learned dipole (test frames of 64) within K14_3 of the reach's own c*,
           which must lie inside the scan
    K14.4  the gap between the warm starts' ends (from 1.5 less from 0.5), seed by seed, smaller at 6 A than at 12 A and
           smaller at 4.5 A than at 6 A
"""
import argparse
import json

import numpy as np

import e4_job7 as J7
import e4_job11 as J11

REACHES = {"4.5": ("results/e4_job14s_r45", "results/e4_job14_r45"), "6": ("results/e4_job14s_r6", "results/e4_job14_r6")}
WARM = {"M_1w050": 0.5, "M_1w150": 1.5}
SCAN = J11.SCAN_C                                     # M_1q070, M_1q100, M_1q110, M_1q145
K14_1, K14_2, K14_3 = 5.0, 2.0, 0.05
WALL = 1.087e6                                        # job 13: the walls' curvature over c^4, E_0 held
SEEDS = (0, 1)
SIZES, PERMANENT_DEBYE = J7.SIZES, J7.PERMANENT_DEBYE


def profile(runs, control):
    """the polished profile (scripts/e4_job11.py's polished_profile: M_1q110 from `control`) with its curvature at the
    minimum, c*^2 L*''(c*), the objective's curvature along the charges' scale with E_0 re-fitted."""
    p = J11.polished_profile(runs, control)
    if p is None:
        return None
    c = p["c_min"]
    return {**p, "curvature": float(c * c * 2.0 * p["coef"][0]),
            "inside": bool(min(SCAN.values()) <= c <= max(SCAN.values()))}


def end_c(r):
    d = J7.learned_dipole(r) if r is not None else None
    return None if d is None else d / PERMANENT_DEBYE


def gaps(runs):
    """{seed: end c from 1.5 less end c from 0.5}; a seed with a run missing is left out."""
    out = {}
    for s in SEEDS:
        lo, hi = end_c(runs.get("M_1w050", {}).get(s)), end_c(runs.get("M_1w150", {}).get(s))
        if lo is not None and hi is not None:
            out[s] = hi - lo
    return out


def wall(r):
    """the warm start's curvature along the charges' scale with E_0 held at the end, over WALL c^4 (job 13's law for
    charges by species; the head's variation adds to it)."""
    v = (r.get("training", {}).get("charges") or {}).get("scale_curve", {}).get("final")
    c = end_c(r)
    if not v or c is None:
        return None
    s, L = v["s"], v["loss"]
    k = (L[s.index(1.01)] - 2.0 * L[s.index(1.0)] + L[s.index(0.99)]) / 1e-4
    return k / (WALL * c ** 4)


def evaluate(reach_runs, job9=None, job10=None, job11=None):
    """reach_runs: {reach: (first stage runs, polished runs)}."""
    job9, job10, job11 = job9 or {}, job10 or {}, job11 or {}
    ref = profile(job11, job10)
    out = {"reference": ref, "reaches": {}}
    for reach, (stage, runs) in reach_runs.items():
        p = profile(runs, runs)
        out["reaches"][reach] = {
            "profile": p,
            "ratio": p["curvature"] / ref["curvature"] if p and ref else None,
            "ends": {f"{w} seed {s}": {"start_c": c0, "end_c": end_c(runs.get(w, {}).get(s)),
                                       "stage_c": end_c(stage.get(w, {}).get(s)),
                                       "final_loss": J7.final_loss(runs[w][s]) if s in runs.get(w, {}) else None,
                                       "wall": wall(runs[w][s]) if s in runs.get(w, {}) else None}
                     for w, c0 in WARM.items() for s in SEEDS},
            "gaps": gaps(runs),
            "forces": {name: {n: J7.seed_mean(by, n, seeds=SEEDS) for n in SIZES}
                       for name, by in sorted(runs.items()) if all(s in by for s in SEEDS)}}
    out["gaps_12"] = gaps(job10)
    out["stage_12"] = {f"{w} seed {s}": end_c(job9.get(w, {}).get(s)) for w in WARM for s in SEEDS}
    r45, r6 = out["reaches"].get("4.5", {}), out["reaches"].get("6", {})
    ratio = lambda r: r.get("ratio") if r else None
    out["K14.1"] = {"band": K14_1, "ratio": ratio(r45), "holds": None if ratio(r45) is None else bool(ratio(r45) >= K14_1)}
    out["K14.2"] = {"band": K14_2, "ratio": ratio(r6), "holds": None if ratio(r6) is None else bool(ratio(r6) >= K14_2)}
    k3 = {"band": K14_3, "holds": None}
    if r45 and r45["profile"] and r45["profile"]["inside"]:
        c_star = r45["profile"]["c_min"]
        ends = [v["end_c"] for v in r45["ends"].values()]
        k3.update(c_star=c_star, off={k: (None if v["end_c"] is None else v["end_c"] - c_star)
                                      for k, v in r45["ends"].items()})
        if all(e is not None for e in ends):
            k3["holds"] = bool(all(abs(e - c_star) <= K14_3 for e in ends))
    elif r45 and r45["profile"]:
        k3["outside"] = True                                       # c* outside the scan: not read
    out["K14.3"] = k3
    g12, g6, g45 = out["gaps_12"], r6.get("gaps", {}) if r6 else {}, r45.get("gaps", {}) if r45 else {}
    complete = all(s in g for g in (g12, g6, g45) for s in SEEDS)
    out["K14.4"] = {"gaps": {"12": g12, "6": g6, "4.5": g45},
                    "holds": bool(all(g45[s] < g6[s] < g12[s] for s in SEEDS)) if complete else None}
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev):
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    lines = ["E4, job 14 (notes/e4-job14-preregistration.md): the scale and E_0's reach"]
    ref = ev["reference"]
    if ref:
        lines.append(f"  12 A (jobs 10-11): the polished profile's minimum c* = {ref['c_min']:.3f}, its curvature c*^2 L*'' "
                     f"{ref['curvature']:.1f}; points " + ", ".join(f"c {c}: {v:.2f}" for c, v in ref["points"].items()))
    for reach, v in ev["reaches"].items():
        p = v["profile"]
        if p:
            lines.append(f"  {reach} A: c* = {p['c_min']:.3f}{'' if p['inside'] else ' (outside the scan)'}, curvature "
                         f"{p['curvature']:.1f} ({f(v['ratio'], 1)} times 12 A's); points "
                         + ", ".join(f"c {c}: {x:.2f}" for c, x in p["points"].items()))
    k1, k2 = ev["K14.1"], ev["K14.2"]
    lines.append(f"  K14.1: {_verdict(k1['holds'])} (4.5 A: the profile's curvature at least {k1['band']:g} times 12 A's): "
                 f"{f(k1['ratio'], 1)}")
    lines.append(f"  K14.2: {_verdict(k2['holds'])} (6 A: at least {k2['band']:g} times): {f(k2['ratio'], 1)}")
    k3 = ev["K14.3"]
    lines.append(f"  K14.3: {_verdict(k3['holds'])} (4.5 A: every polished warm start within {k3['band']:g} of the reach's "
                 f"own c*{'; c* lies outside the scan' if k3.get('outside') else ''})")
    for k, x in k3.get("off", {}).items():
        lines.append(f"      {k}: {'-' if x is None else f'{x:+.3f}'}")
    k4 = ev["K14.4"]
    lines.append(f"  K14.4: {_verdict(k4['holds'])} (the gap between the warm starts' ends, seed by seed, 4.5 A < 6 A < 12 "
                 f"A): " + "; ".join(f"{r} A " + ", ".join(f"seed {s} {g:.3f}" for s, g in gs.items())
                                    for r, gs in k4["gaps"].items()))
    lines.append("\n  Beside them, not registered")
    for reach, v in ev["reaches"].items():
        lines.append(f"    {reach} A, the warm starts (c: after job 7's training -> polished; final loss; the walls over "
                     f"{WALL:.3e} c^4)")
        for k, e in v["ends"].items():
            lines.append(f"      {k}: from {e['start_c']:g}: {f(e['stage_c'])} -> {f(e['end_c'])}; {f(e['final_loss'], 2)}; "
                         f"{f(e['wall'], 2)}")
        for name, by in v["forces"].items():
            lines.append(f"      {name:8s} " + "; ".join(f"n {n} {x[0]:.3f} +- {x[1]:.3f}" for n, x in by.items() if x))
    lines.append("    12 A after job 7's training (job 9): " + ", ".join(f"{k} {f(c)}" for k, c in ev["stage_12"].items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for reach, (stage, pol) in REACHES.items():
        tag = reach.replace(".", "")
        ap.add_argument(f"--r{tag}_stage", default=stage)
        ap.add_argument(f"--r{tag}_dir", default=pol)
    ap.add_argument("--job9_dir", default="results/e4_job9", help="job 7's warm starts, traced: the 12 A first stage")
    ap.add_argument("--job10_dir", default="results/e4_job10", help="the 12 A control M_1q110 and warm starts, polished")
    ap.add_argument("--job11_dir", default="results/e4_job11", help="the 12 A scan, polished")
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    reach_runs = {}
    for reach in REACHES:
        tag = reach.replace(".", "")
        stage, runs = J7.load_runs(getattr(a, f"r{tag}_stage")), J7.load_runs(getattr(a, f"r{tag}_dir"))
        if runs or stage:
            reach_runs[reach] = (stage, runs)
    if not reach_runs:
        raise SystemExit("no results for either reach")
    ev = evaluate(reach_runs, J7.load_runs(a.job9_dir), J7.load_runs(a.job10_dir), J7.load_runs(a.job11_dir))
    print(report(ev))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(ev, fh, indent=1)


if __name__ == "__main__":
    main()
