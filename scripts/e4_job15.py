"""
e4_job15.py -- E4 job 15's registered readings K15.1-K15.10 (notes/e4-job15-preregistration.md; v0.0.40) from the
training jsons of scripts/e4_water.py train and MACE-LES's evaluations (baselines/mace/eval.py), without JAX.

    python scripts/e4_job15.py

Job 14 found learned charges mobile, and their ends independent of the start, with E_0 of one message-passing layer
(4.5 A; nearly at 6 A), frozen with two (12 A), though the profile along their scale barely sharpened.  Five arms:

  A  the profile's minimum at 4.5 A: the pinned O/H charges at c = 0.9, 0.95, 1.05, 1.2, 1.3 beside job 14's 0.7, 1.0,
     1.1, 1.45 (results/e4_job15_a, results/e4_job14_r45); and learned charges by species alone (the charge head held,
     job 13's M_1z050 and M_1z150) from 0.5 and 1.5 there
  B  depth alone, finished: job 14's 6 A warm starts polished another 1000 steps (results/e4_job15_b)
  C  depth against reach: E_0 of two layers of 3 A (a reach of 6 A), job 14's scan and warm starts (results/e4_job15_c)
  D  the route: B's warm starts, and job 14's pinned M_1q110 at 6 A, with E_0 grown to two layers of 6 A (job 7's E_0;
     training.start_from.grow), polished 1000 steps (results/e4_job15_d)
  E  MACE-LES at 64 molecules with its latent charges' unit (MACELES.les_output_scale, default 0.1) at 0.05 and 0.15:
     E1 trained from scratch at each unit (results/e4_job15_les/maceles_o050_n64_s<seed>, ..._o150_...; the default is
     the baseline, results/e4_baselines/maceles_n64_s<seed>); E2 the baseline continued 150 epochs at each unit, its
     charges scaled by 0.5, 1 or 1.5 at the start (results/e4_job15_les/release_o<unit>_n64_s<seed>)

The end of a run with learned charges is c, its learned dipole on the test frames of 64 molecules over MB-pol's
permanent 1.864 D (MACE-LES: the latent charges' molecular dipole, each molecule's net charge spread evenly).

    K15.1  A: the resolved profile's minimum c*_res -- the parabola in c through the five scan points around the lowest
           of the nine -- at least K15_1 above the largest of job 14's four flexible ends at 4.5 A
    K15.2  A: every species-only end within K15_2 of c*_res (read when c*_res lies inside its five points)
    K15.3  A: the species-only gap (from 1.5 less from 0.5), seed by seed, at most K15_3 times job 13's at 12 A
    K15.4  B: |the gap| at most K15_4 on both seeds
    K15.5  C: the polished gap on both seeds at most K15_5[0] ("reach": as one layer of 6 A) or at least K15_5[1]
           ("depth": as two layers of 6 A); otherwise mixed
    K15.6  D: every grown warm start's end within K15_6 of its start (B's end)
    K15.7  D: the grown warm starts' forces at least K15_7 below B's, every size, both starts (seed means)
    K15.8  D: the grown M_1q110's forces within K15_8 of job 10's M_1q110 (job 7's E_0 trained and polished), every size
    K15.9  E1: every latent dipole at 0.05 and 0.15 within K15_9 of the default's, same seed
    K15.10 E2: every released latent dipole at 0.05 and 0.15 within K15_10 of the continued control's, same seed
"""
import argparse
import json
import os

import numpy as np

import e4_job7 as J7
import e4_job14 as J14

PROFILE_A = {"M_1q070": 0.7, "M_1q090": 0.9, "M_1q095": 0.95, "M_1q100": 1.0, "M_1q105": 1.05, "M_1q110": 1.1,
             "M_1q120": 1.2, "M_1q130": 1.3, "M_1q145": 1.45}
NEW_A = ("M_1q090", "M_1q095", "M_1q105", "M_1q120", "M_1q130")          # job 15's; the rest job 14's
WINDOW = 5
FLEX = {"M_1w050": 0.5, "M_1w150": 1.5}
SPECIES = {"M_1z050": 0.5, "M_1z150": 1.5}
K15_1, K15_2, K15_3, K15_4, K15_5, K15_6, K15_7, K15_8, K15_9, K15_10 = (0.03, 0.05, 0.5, 0.05, (0.15, 0.25), 0.05,
                                                                          0.05, 0.03, 0.10, 0.10)
SEEDS = (0, 1)
SIZES, PERMANENT_DEBYE = J7.SIZES, J7.PERMANENT_DEBYE
UNITS = {"o050": 0.5, "o100": 1.0, "o150": 1.5}                          # MACE-LES's unit over its default
LES_SET = "test_n64"
DIRS = {"a_stage": "results/e4_job15s_a", "a": "results/e4_job15_a", "b": "results/e4_job15_b",
        "c_stage": "results/e4_job15s_c", "c": "results/e4_job15_c", "d": "results/e4_job15_d",
        "r45_stage": "results/e4_job14s_r45", "r45": "results/e4_job14_r45", "r6": "results/e4_job14_r6",
        "job9": "results/e4_job9", "job10": "results/e4_job10", "job11": "results/e4_job11", "job13": "results/e4_job13",
        "baselines": "results/e4_baselines", "les": "results/e4_job15_les"}
end_c = J14.end_c


def gaps(runs, lo, hi):
    """{seed: end c from `hi` less end c from `lo`}; a seed with a run missing is left out."""
    out = {}
    for s in SEEDS:
        a, b = end_c(runs.get(lo, {}).get(s)), end_c(runs.get(hi, {}).get(s))
        if a is not None and b is not None:
            out[s] = b - a
    return out


def resolved_profile(a_runs, r45_runs):
    """A's profile: the seed means of the final training loss at PROFILE_A (job 15's points from `a_runs`, job 14's from
    `r45_runs`), the parabola in c through the WINDOW points around the lowest, its minimum and curvature c*^2 L*''.
    None when a point is missing."""
    pts = {}
    for name, c in PROFILE_A.items():
        by = (a_runs if name in NEW_A else r45_runs).get(name, {})
        v = [J7.final_loss(by[s]) if s in by else None for s in SEEDS]
        if any(x is None for x in v):
            return None
        pts[c] = float(np.mean(v))
    cs = sorted(pts)
    i = int(np.argmin([pts[c] for c in cs]))
    lo = min(max(i - WINDOW // 2, 0), len(cs) - WINDOW)
    win = cs[lo:lo + WINDOW]
    a, b, k = np.polyfit(win, [pts[c] for c in win], 2)
    c_min = float(-b / (2.0 * a))
    whole = J14.fits(pts)
    return {"points": {str(c): pts[c] for c in cs}, "window": win, "lowest": cs[i], "c_min": c_min,
            "curvature": float(c_min * c_min * 2.0 * a), "loss_min": float(np.polyval([a, b, k], c_min)),
            "inside": bool(a > 0 and win[0] <= c_min <= win[-1]), "whole": whole}


def les_eval(path):
    """the MACE-LES evaluation json's reading on LES_SET: c (the latent dipole over 1.864 D), the latent charges by
    species, the net charge per molecule; and the forces at every set; None when the json is missing."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    s = d.get("sets", {}).get(LES_SET, {})
    if "latent_dipole_debye_mean" not in s:
        return None
    return {"c": s["latent_dipole_debye_mean"] / PERMANENT_DEBYE, "dipole_debye": s["latent_dipole_debye_mean"],
            "q_O": s.get("latent_q_O_mean"), "q_H": s.get("latent_q_H_mean"),
            "net_q_rms": s.get("latent_q_per_molecule_rms"),
            "forces": {n: v["f_rmse_kcal_per_mol_A"] for n, v in d["sets"].items()},
            "energy": {n: v["e_rmse_kcal_per_mol"] for n, v in d["sets"].items()}}


def les_runs(baselines, les):
    """{"E1": {unit: {seed: reading}}, "E2": {...}} from the result directories' evaluation jsons."""
    e1, e2 = {}, {}
    for u in UNITS:
        for s in SEEDS:
            p1 = (os.path.join(baselines, f"maceles_n64_s{s}", f"maceles_n64_s{s}_stagetwo_eval.json") if u == "o100"
                  else os.path.join(les, f"maceles_{u}_n64_s{s}", f"maceles_{u}_n64_s{s}_stagetwo_eval.json"))
            p2 = os.path.join(les, f"release_{u}_n64_s{s}", f"maceles_n64_s{s}_stagetwo_eval.json")
            for d, p in ((e1, p1), (e2, p2)):
                r = les_eval(p)
                if r is not None:
                    d.setdefault(u, {})[s] = r
    return {"E1": e1, "E2": e2}


def _against(e, ref_unit, band):
    """every reading at 0.05 and 0.15 against the reference unit's of the same seed: their ratios and whether every one
    lies within band; None until all four and their references are there."""
    ratios = {f"{u} seed {s}": e[u][s]["c"] / e[ref_unit][s]["c"]
              for u in ("o050", "o150") for s in SEEDS if s in e.get(u, {}) and s in e.get(ref_unit, {})}
    return {"band": band, "ratios": ratios,
            "holds": bool(all(abs(r - 1.0) <= band for r in ratios.values())) if len(ratios) == 4 else None}


def seed_forces(by):
    return {n: J7.seed_mean(by, n, seeds=SEEDS) for n in SIZES} if all(s in by for s in SEEDS) else None


def evaluate(R):
    """R: {key of DIRS: {rung: {seed: json}}}, and R["les"]: les_runs(...)."""
    g = lambda k: R.get(k) or {}
    out = {}
    # A
    prof = resolved_profile(g("a"), g("r45"))
    flex = [end_c(g("r45").get(w, {}).get(s)) for w in FLEX for s in SEEDS]
    flex = [x for x in flex if x is not None]
    out["profile_a"] = prof
    k1 = {"band": K15_1, "flexible_ends": flex, "holds": None}
    if prof is not None and len(flex) == 4:
        k1.update(c_star=prof["c_min"], above=prof["c_min"] - max(flex), inside=prof["inside"])
        k1["holds"] = bool(prof["inside"] and prof["c_min"] - max(flex) >= K15_1)
    out["K15.1"] = k1
    sp = {f"{w} seed {s}": end_c(g("a").get(w, {}).get(s)) for w in SPECIES for s in SEEDS}
    k2 = {"band": K15_2, "ends": sp, "holds": None}
    if prof is not None and prof["inside"]:
        k2["off"] = {k: (None if v is None else v - prof["c_min"]) for k, v in sp.items()}
        if all(v is not None for v in sp.values()):
            k2["holds"] = bool(all(abs(v) <= K15_2 for v in k2["off"].values()))
    out["K15.2"] = k2
    g45, g12 = gaps(g("a"), "M_1z050", "M_1z150"), gaps(g("job13"), "M_1z050", "M_1z150")
    out["K15.3"] = {"band": K15_3, "gaps": {"4.5": g45, "12": g12},
                    "holds": bool(all(g45[s] <= K15_3 * g12[s] for s in SEEDS))
                    if all(s in g45 and s in g12 for s in SEEDS) else None}
    # B
    gb = gaps(g("b"), "M_1w050", "M_1w150")
    out["K15.4"] = {"band": K15_4, "gaps": gb, "before": gaps(g("r6"), "M_1w050", "M_1w150"),
                    "holds": bool(all(abs(gb[s]) <= K15_4 for s in SEEDS)) if all(s in gb for s in SEEDS) else None}
    # C
    gc = gaps(g("c"), "M_1w050", "M_1w150")
    verdict = None
    if all(s in gc for s in SEEDS):
        v = [abs(gc[s]) for s in SEEDS]
        verdict = "reach" if max(v) <= K15_5[0] else ("depth" if min(v) >= K15_5[1] else "mixed")
    out["K15.5"] = {"band": list(K15_5), "gaps": gc, "stage_gaps": gaps(g("c_stage"), "M_1w050", "M_1w150"),
                    "verdict": verdict}
    # D
    moves = {f"{w} seed {s}": (end_c(g("d").get(w, {}).get(s)), end_c(g("b").get(w, {}).get(s)))
             for w in FLEX for s in SEEDS}
    k6 = {"band": K15_6, "ends": {k: {"start": b, "end": a, "move": None if a is None or b is None else a - b}
                                  for k, (a, b) in moves.items()}, "holds": None}
    if all(v["move"] is not None for v in k6["ends"].values()):
        k6["holds"] = bool(all(abs(v["move"]) <= K15_6 for v in k6["ends"].values()))
    out["K15.6"] = k6
    k7 = {"band": K15_7, "against_b": {}, "holds": None}
    for w in FLEX:
        fd, fb = seed_forces(g("d").get(w, {})), seed_forces(g("b").get(w, {}))
        if fd and fb and all(fd[n] and fb[n] for n in SIZES):
            k7["against_b"][w] = {n: fd[n][0] / fb[n][0] - 1.0 for n in SIZES}
    if len(k7["against_b"]) == len(FLEX):
        k7["holds"] = bool(all(x <= -K15_7 for v in k7["against_b"].values() for x in v.values()))
    out["K15.7"] = k7
    fq, f10 = seed_forces(g("d").get("M_1q110", {})), seed_forces(g("job10").get("M_1q110", {}))
    k8 = {"band": K15_8, "holds": None}
    if fq and f10 and all(fq[n] and f10[n] for n in SIZES):
        k8["against_job10"] = {n: fq[n][0] / f10[n][0] - 1.0 for n in SIZES}
        k8["holds"] = bool(all(abs(x) <= K15_8 for x in k8["against_job10"].values()))
    out["K15.8"] = k8
    # E
    les = R.get("les") or {"E1": {}, "E2": {}}
    out["les"] = les
    out["K15.9"] = _against(les["E1"], "o100", K15_9)
    k10 = _against(les["E2"], "o100", K15_10)
    k10["back"] = {k: (r - UNITS[k.split()[0]]) / (1.0 - UNITS[k.split()[0]]) for k, r in k10["ratios"].items()}
    out["K15.10"] = k10
    # beside: the profiles at C's reach and the 12 A reference; the paths; the forces
    ref = J14.profile(g("job11"), g("job10"))
    pc = J14.profile(g("c"), g("c"))
    out["profile_c"] = {"profile": pc, "fits": J14.fits(J14.scan_points(g("c"), g("c"))),
                        "ratio": pc["curvature"] / ref["curvature"] if pc and ref else None, "reference": ref}
    paths = {}
    for arm, stage, pol, rungs in (("A", "a_stage", "a", SPECIES), ("B", None, "b", FLEX), ("C", "c_stage", "c", FLEX),
                                   ("D", None, "d", FLEX)):
        for w in rungs:
            for s in SEEDS:
                st = g(stage).get(w, {}).get(s) if stage else None
                po = g(pol).get(w, {}).get(s)
                paths[f"{arm} {w} seed {s}"] = {"stage": J14.stage_path(st), "polish": J14.polish_path(po),
                                                "stage_c": end_c(st), "end_c": end_c(po),
                                                "final_loss": J7.final_loss(po) if po else None}
    out["paths"] = paths
    out["forces"] = {arm: {name: seed_forces(by) for name, by in sorted(g(k).items()) if seed_forces(by)}
                     for arm, k in (("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"))}
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev):
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    L = ["E4, job 15 (notes/e4-job15-preregistration.md): the learned charges' scale, five arms"]
    p = ev["profile_a"]
    if p:
        L.append("  A, 4.5 A: the profile " + ", ".join(f"c {c}: {v:.2f}" for c, v in p["points"].items())
                 + f"; lowest at c = {p['lowest']}; the window {p['window']}: c*_res = {p['c_min']:.3f}"
                 + ("" if p["inside"] else " (OUTSIDE its window)") + f", curvature {p['curvature']:.1f}")
    k = ev["K15.1"]
    L.append(f"  K15.1: {_verdict(k['holds'])} (c*_res at least {k['band']:g} above job 14's flexible ends at 4.5 A, "
             f"largest {f(max(k['flexible_ends']) if k['flexible_ends'] else None)}): {f(k.get('above'), 3)}")
    k = ev["K15.2"]
    L.append(f"  K15.2: {_verdict(k['holds'])} (A: every species-only end within {k['band']:g} of c*_res): "
             + "; ".join(f"{n} {f(v)}" + (f" ({k['off'][n]:+.3f})" if k.get("off") and k["off"][n] is not None else "")
                         for n, v in k["ends"].items()))
    k = ev["K15.3"]
    L.append(f"  K15.3: {_verdict(k['holds'])} (A: the species-only gap at most {k['band']:g} times job 13's at 12 A): "
             + "; ".join(f"{r} A " + ", ".join(f"seed {s} {v:.3f}" for s, v in gs.items()) for r, gs in k["gaps"].items()))
    k = ev["K15.4"]
    L.append(f"  K15.4: {_verdict(k['holds'])} (B: |the gap| at most {k['band']:g} after 2000 steps at 6 A): "
             + ", ".join(f"seed {s} {v:+.3f}" for s, v in k["gaps"].items())
             + " (after 1000: " + ", ".join(f"{v:+.3f}" for v in k["before"].values()) + ")")
    k = ev["K15.5"]
    L.append(f"  K15.5: {k['verdict'] or 'not read'} (C, two layers of 3 A: |the gap| at most {k['band'][0]:g} on both "
             f"seeds is reach, at least {k['band'][1]:g} depth): " + ", ".join(f"seed {s} {v:+.3f}" for s, v in k["gaps"].items())
             + " (after job 7's training: " + ", ".join(f"{v:+.3f}" for v in k["stage_gaps"].values()) + ")")
    k = ev["K15.6"]
    L.append(f"  K15.6: {_verdict(k['holds'])} (D: every grown warm start within {k['band']:g} of B's end): "
             + "; ".join(f"{n} {f(v['start'])} -> {f(v['end'])}" for n, v in k["ends"].items()))
    k = ev["K15.7"]
    L.append(f"  K15.7: {_verdict(k['holds'])} (D: the grown warm starts' forces at least {100 * k['band']:g} % below B's): "
             + "; ".join(f"{w} " + " / ".join(f"{100 * x:+.1f}" for x in v.values()) + " %"
                         for w, v in k["against_b"].items()))
    k = ev["K15.8"]
    L.append(f"  K15.8: {_verdict(k['holds'])} (D: the grown M_1q110 within {100 * k['band']:g} % of job 10's): "
             + (" / ".join(f"{100 * x:+.1f}" for x in k["against_job10"].values()) + " %" if "against_job10" in k else "-"))
    for key, label in (("K15.9", "E1: MACE-LES from scratch at 0.05 and 0.15, against the default's"),
                       ("K15.10", "E2: released at 0.05 and 0.15, against the continued control's")):
        k = ev[key]
        L.append(f"  {key}: {_verdict(k['holds'])} ({label}, within {100 * k['band']:g} %): "
                 + "; ".join(f"{n} {r:.3f}" for n, r in k["ratios"].items())
                 + ("; the share of the way back " + ", ".join(f"{n} {v:.2f}" for n, v in k["back"].items())
                    if k.get("back") else ""))

    L.append("\n  Beside them, not registered")
    pc = ev["profile_c"]
    if pc["profile"]:
        q = pc["profile"]
        L.append(f"    C's profile: c* = {q['c_min']:.3f}, curvature {q['curvature']:.1f} ({f(pc['ratio'], 2)} times 12 A's); "
                 + ", ".join(f"c {c}: {v:.2f}" for c, v in q["points"].items())
                 + (f"; in c^2: c* {f(pc['fits']['c2']['c_min'])}, {f(pc['fits']['c2']['curvature'], 1)}" if pc["fits"] else ""))
    if ev["profile_a"]:
        w = ev["profile_a"]["whole"]
        L.append(f"    A's nine points, every one: the parabola in c {f(w['c']['c_min'])} ({f(w['c']['curvature'], 1)}), "
                 f"in c^2 {f(w['c2']['c_min'])} ({f(w['c2']['curvature'], 1)})")
    les = ev["les"]
    for arm in ("E1", "E2"):
        for u, by in sorted(les[arm].items()):
            for s, r in sorted(by.items()):
                L.append(f"    MACE-LES {arm} {u} seed {s}: c {r['c']:.3f} ({r['dipole_debye']:.3f} D), q_O {f(r['q_O'])}, "
                         f"q_H {f(r['q_H'])} e, net q per molecule rms {f(r['net_q_rms'], 4)}; forces "
                         + " / ".join(f"{n} {v:.3f}" for n, v in r["forces"].items()))
    L.append("    the paths (c: through job 7's training at Adam steps, half by, after its L-BFGS; the pull's mean / sd in "
             "blocks of 500; through the polish every 250 steps, the last 250 steps' move; the objective)")
    for name, v in ev["paths"].items():
        st, po = v["stage"], v["polish"]
        if st is None and po is None and v["end_c"] is None:
            continue
        txt = f"      {name}: "
        if st:
            txt += (", ".join(f"{x:.3f} at {k}" for k, x in st["c"].items()) + f"; half by {st['half_step']}"
                    + (f"; {st['after_lbfgs']:.3f} after L-BFGS" if st["after_lbfgs"] is not None else "")
                    + "; pull " + ", ".join("-" if x is None else f"{x:+.2f}" for x in st["pull"]) + " | ")
        if po:
            l250 = "-" if po["last_250"] is None else f"{po['last_250']:+.3f}"
            txt += "polish " + ", ".join(f"{x:.3f} at {k}" for k, x in po["c"].items()) + f"; last 250 {l250}"
            if po["objective"]:
                txt += (f"; objective {po['objective'][0]:.2f} -> {po['objective'][1]:.2f}"
                        + ("" if po["objective_last_250"] is None else f" ({100 * po['objective_last_250']:+.1f} %)"))
        else:
            txt += f"end {f(v['end_c'])}"
        L.append(txt)
    L.append("    forces (mean +- sd over seeds)")
    for arm, by in ev["forces"].items():
        for name, x in by.items():
            L.append(f"      {arm} {name:8s} " + "; ".join(f"n {n} {v[0]:.3f} +- {v[1]:.3f}" for n, v in x.items() if v))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for k, d in DIRS.items():
        ap.add_argument(f"--{k}", default=d)
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    R = {k: J7.load_runs(getattr(a, k)) for k in DIRS if k not in ("baselines", "les")}
    R["les"] = les_runs(a.baselines, a.les)
    ev = evaluate(R)
    print(report(ev))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(ev, fh, indent=1)


if __name__ == "__main__":
    main()
