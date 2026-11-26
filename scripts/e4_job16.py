"""
e4_job16.py -- E4 job 16's registered readings K16.1-K16.5 (notes/e4-job16-preregistration.md; v0.0.41) from the
training jsons of scripts/e4_water.py train, without JAX.

    python scripts/e4_job16.py

Job 16 trains on cells of 128 molecules, the same atoms as job 7's 180 frames of 64 (90 frames, three per batch), and
the learned charges also on 256 (45 frames, one per batch):

  F  E4's L_0 sweep (Form II, e4-plan.md section 4): E_0 alone (M_inf), the field-term dagger (M_16pm_us_d20) and E4's
     model (D', M_16C2pm_us_d20 on job 5's read-out), trained and polished at L_0 = 128 as at 64
     (results/e4_job16_128, e4_job16_128d), against L_0 = 64: M_inf polished here (results/e4_job16_64), job 11's dagger
     and job 12's E4's model, seeds 0 and 1
  L  the learned charges' scale in larger cells: job 7's warm starts from 0.5 and 1.5 (two layers of 6 A, reaching
     12 A), trained and polished at L_0 = 128 (15.6 A cells, 99 % of which lie within 12 A of a point) and 256 (19.7 A,
     82 %), against L_0 = 64 (12.4 A, all of it; job 10: gaps 0.324, 0.288)

The bias of a rung at size L is its mean energy error per molecule on the frames of size L that its training did not
see: at L_0 every frame of the validation and test splits, at any other size every frame.  delta_L0(L) is M_inf's bias
less the dagger's, on the same frames, seed by seed, then the seed mean; its step from L_0 to L is s_L0(L) =
delta_L0(L) - delta_L0(L_0).

    K16.1  M_inf carries its training cell's long range: s_128(64) against s_64(128) -- opposite in sign, its magnitude
           within K16_1 (relative) of it; read when |s_64(128)| is at least K16_1_MIN
    K16.2  E4's model follows the cell: its bias at L_0 = 128 within K16_2 of its bias at L_0 = 64, at every size
    K16.3  E4's model's forces at L_0 = 128 within K16_3 (relative) of L_0 = 64's, at every size
    K16.4  L at 128: the polished gap (the end from 1.5 less the end from 0.5, in c) on both seeds at least K16_FROZEN
           ("frozen", as at 64), at most K16_MOBILE ("mobile"), else "partial"
    K16.5  L at 256: the same
"""
import argparse
import json

import numpy as np

import e4_job7 as J7
import e4_job14 as J14

SIZES = ("64", "128", "256")
SPLITS = ("train", "valid", "test")
SEEDS = (0, 1)
REF, DAGGER, MODEL = "M_inf", "M_16pm_us_d20", "M_16C2pm_us_d20"
WARM = {"M_1w050": 0.5, "M_1w150": 1.5}
K16_1, K16_1_MIN, K16_2, K16_3, K16_FROZEN, K16_MOBILE = 0.4, 0.003, 0.003, 0.05, 0.25, 0.15
COVERAGE = {"64": 1.00, "128": 0.99, "256": 0.82}     # the share of the cell within 12 A of a point (job 7's E_0's reach)
DIRS = {"dir128": "results/e4_job16_128", "stage128": "results/e4_job16s_128", "dir128d": "results/e4_job16_128d",
        "stage128d": "results/e4_job16s_128d", "dir256": "results/e4_job16_256", "stage256": "results/e4_job16s_256",
        "dir64": "results/e4_job16_64", "job9": "results/e4_job9", "job10": "results/e4_job10", "job11": "results/e4_job11",
        "job12": "results/e4_job12"}
end_c = J14.end_c


def unseen(L0, n):
    return ("valid", "test") if str(n) == str(L0) else SPLITS


def frame_errors(r, n, L0):
    """the run's energy error per molecule (kcal/mol) on every unseen frame of size n, in split order; None when a
    split is missing."""
    cells = r["readings"]["cells"].get(str(n), {})
    out = []
    for split in unseen(L0, n):
        v = cells.get(split, {})
        if "energy_per_molecule_errors" not in v:
            return None
        out += list(v["energy_per_molecule_errors"])
    return np.asarray(out, float)


def pooled(r, n, L0, key="force_rigid_rmse"):
    """the run's rmse on the unseen frames of size n, each split's pooled by its frames."""
    cells = r["readings"]["cells"].get(str(n), {})
    num = den = 0.0
    for split in unseen(L0, n):
        v = cells.get(split, {})
        if key not in v:
            return None
        num += v["frames"] * v[key] ** 2
        den += v["frames"]
    return float(np.sqrt(num / den)) if den else None


def bias(by, n, L0):
    """(seed mean, sd) of the bias at size n over SEEDS; None when a run or split is missing."""
    v = [frame_errors(by[s], n, L0) if s in by else None for s in SEEDS]
    if any(x is None for x in v):
        return None
    m = [float(np.mean(x)) for x in v]
    return float(np.mean(m)), float(np.std(m))


def forces(by, n, L0):
    v = [pooled(by[s], n, L0) if s in by else None for s in SEEDS]
    if any(x is None for x in v):
        return None
    return float(np.mean(v)), float(np.std(v))


def delta(ref, dag, n, L0):
    """M_inf's bias less the dagger's on the same unseen frames of size n, seed by seed, then (seed mean, sd); with the
    frame-to-frame standard error of the paired difference (seed mean)."""
    d, se = [], []
    for s in SEEDS:
        a = frame_errors(ref[s], n, L0) if s in ref else None
        b = frame_errors(dag[s], n, L0) if s in dag else None
        if a is None or b is None or len(a) != len(b):
            return None
        x = a - b
        d.append(float(np.mean(x)))
        se.append(float(np.std(x) / np.sqrt(len(x))))
    return {"mean": float(np.mean(d)), "sd": float(np.std(d)), "se": float(np.mean(se))}


def gaps(runs):
    out = {}
    for s in SEEDS:
        lo, hi = end_c(runs.get("M_1w050", {}).get(s)), end_c(runs.get("M_1w150", {}).get(s))
        if lo is not None and hi is not None:
            out[s] = hi - lo
    return out


def classify(g):
    if not all(s in g for s in SEEDS):
        return None
    v = [g[s] for s in SEEDS]
    return "frozen" if min(v) >= K16_FROZEN else ("mobile" if max(abs(x) for x in v) <= K16_MOBILE else "partial")


def evaluate(R):
    """R: {key of DIRS: {rung: {seed: json}}}."""
    g = lambda k: R.get(k) or {}
    sets = {64: {"ref": g("dir64").get(REF, {}), "dag": g("job11").get(DAGGER, {}), "model": g("job12").get(MODEL, {})},
            128: {"ref": g("dir128").get(REF, {}), "dag": g("dir128").get(DAGGER, {}),
                  "model": g("dir128d").get(MODEL, {})}}
    out = {"delta": {}, "bias": {}, "forces": {}}
    for L0, x in sets.items():
        out["delta"][L0] = {n: delta(x["ref"], x["dag"], n, L0) for n in SIZES}
        out["bias"][L0] = {name: {n: bias(x[k], n, L0) for n in SIZES} for name, k in ((REF, "ref"), (DAGGER, "dag"),
                                                                                     (MODEL, "model"))}
        out["forces"][L0] = {name: {n: forces(x[k], n, L0) for n in SIZES} for name, k in ((REF, "ref"), (DAGGER, "dag"),
                                                                                         (MODEL, "model"))}
    d64, d128 = out["delta"][64], out["delta"][128]
    step = lambda d, n, L0: (d[n]["mean"] - d[str(L0)]["mean"]) if d[n] and d[str(L0)] else None
    s64, s128 = step(d64, "128", 64), step(d128, "64", 128)
    k1 = {"s64_128": s64, "s128_64": s128, "s64_256": step(d64, "256", 64), "s128_256": step(d128, "256", 128),
          "band": K16_1, "min": K16_1_MIN, "holds": None}
    if s64 is not None and s128 is not None and abs(s64) >= K16_1_MIN:
        k1["ratio"] = -s128 / s64
        k1["holds"] = bool(s128 * s64 < 0 and abs(s128 + s64) <= K16_1 * abs(s64))
    elif s64 is not None and abs(s64) < K16_1_MIN:
        k1["too_small"] = True
    out["K16.1"] = k1
    b64, b128 = out["bias"][64][MODEL], out["bias"][128][MODEL]
    diff = {n: (b128[n][0] - b64[n][0]) if b64[n] and b128[n] else None for n in SIZES}
    out["K16.2"] = {"band": K16_2, "diff": diff, "holds": bool(all(abs(v) <= K16_2 for v in diff.values()))
                    if all(v is not None for v in diff.values()) else None}
    f64, f128 = out["forces"][64][MODEL], out["forces"][128][MODEL]
    rel = {n: (f128[n][0] / f64[n][0] - 1.0) if f64[n] and f128[n] else None for n in SIZES}
    out["K16.3"] = {"band": K16_3, "rel": rel, "holds": bool(all(abs(v) <= K16_3 for v in rel.values()))
                    if all(v is not None for v in rel.values()) else None}
    # L: the cell
    cells = {"64": (g("job9"), g("job10")), "128": (g("stage128"), g("dir128")), "256": (g("stage256"), g("dir256"))}
    out["cell"] = {}
    for n, (st, pol) in cells.items():
        L0 = int(n)
        out["cell"][n] = {"coverage": COVERAGE[n], "gaps": gaps(pol), "stage_gaps": gaps(st),
                          "ends": {f"{w} seed {s}": {"stage": end_c(st.get(w, {}).get(s)), "end": end_c(pol.get(w, {}).get(s))}
                                   for w in WARM for s in SEEDS},
                          "forces": {w: {m: forces(pol.get(w, {}), m, L0) for m in SIZES} for w in WARM},
                          "paths": {f"{w} seed {s}": {"stage": J14.stage_path(st.get(w, {}).get(s)),
                                                      "polish": J14.polish_path(pol.get(w, {}).get(s))}
                                    for w in WARM for s in SEEDS}}
    for key, n in (("K16.4", "128"), ("K16.5", "256")):
        out[key] = {"band": [K16_MOBILE, K16_FROZEN], "gaps": out["cell"][n]["gaps"],
                    "verdict": classify(out["cell"][n]["gaps"])}
    # beside: E4's model's infrared at L_0 = 128 against job 12's
    infra = {}
    for L0, by in ((64, sets[64]["model"]), (128, sets[128]["model"])):
        v = []
        for s in SEEDS:
            r = by.get(s)
            e = (r or {}).get("readings", {}).get("cells", {}).get(str(L0), {}).get("test", {}).get("eps_inf") if r else None
            if e:
                v.append(e["isotropic_mean"])
        infra[L0] = float(np.mean(v)) if v else None
    out["eps_inf"] = infra
    # beside: the L-BFGS steps each polish took (its budget may stop it early)
    out["steps"] = {k: {f"{rung} seed {s}": r.get("training", {}).get("options", {}).get("lbfgs_steps_done")
                        for rung, by in sorted(g(k).items()) for s, r in sorted(by.items())}
                    for k in ("dir64", "dir128", "dir128d", "dir256")}
    return out


def _verdict(h):
    return "not read" if h is None else ("holds" if h else "FAILS")


def report(ev):
    f = lambda x, d=4: "-" if x is None else f"{x:+.{d}f}"
    fm = lambda x, d=3: "-" if x is None else f"{x[0]:.{d}f} +- {x[1]:.{d}f}"
    c3 = lambda x: "-" if x is None else f"{x:.3f}"
    L = ["E4, job 16 (notes/e4-job16-preregistration.md): the training cells of 128 molecules"]
    L.append("  M_inf's bias less the dagger's (kcal/mol per molecule, unseen frames; seed mean, sd; frame se)")
    for L0, d in ev["delta"].items():
        L.append(f"    L_0 = {L0}: " + "; ".join(f"n {n} " + ("-" if v is None else f"{v['mean']:+.4f} ({v['sd']:.4f}; {v['se']:.4f})")
                                           for n, v in d.items()))
    k = ev["K16.1"]
    L.append(f"  K16.1: {_verdict(k['holds'])} (M_inf carries its training cell: s_128(64) opposite to s_64(128), within "
             f"{100 * k['band']:g} %; read when |s_64(128)| >= {k['min']:g}): s_64(128) {f(k['s64_128'])}, s_128(64) "
             f"{f(k['s128_64'])}" + (f", ratio {k['ratio']:.2f}" if "ratio" in k else "")
             + ("; s_64(128) too small to read" if k.get("too_small") else "")
             + f"; beside, the steps to 256: {f(k['s64_256'])} (L_0 64), {f(k['s128_256'])} (L_0 128)")
    k = ev["K16.2"]
    L.append(f"  K16.2: {_verdict(k['holds'])} (E4's model's bias at L_0 = 128 within {k['band']:g} of L_0 = 64's): "
             + ", ".join(f"n {n} {f(v)}" for n, v in k["diff"].items()))
    k = ev["K16.3"]
    L.append(f"  K16.3: {_verdict(k['holds'])} (E4's model's forces at L_0 = 128 within {100 * k['band']:g} % of 64's): "
             + " / ".join("-" if v is None else f"{100 * v:+.1f}" for v in k["rel"].values()) + " %")
    for key, n in (("K16.4", "128"), ("K16.5", "256")):
        k = ev[key]
        L.append(f"  {key}: {k['verdict'] or 'not read'} (the warm starts at L_0 = {n}, coverage "
                 f"{100 * ev['cell'][n]['coverage']:.0f} %: the gap on both seeds >= {K16_FROZEN:g} frozen, <= "
                 f"{K16_MOBILE:g} mobile): " + ", ".join(f"seed {s} {v:+.3f}" for s, v in k["gaps"].items()))
    L.append("\n  Beside them, not registered")
    L.append("    the bias by rung (seed mean +- sd) and the forces (pooled unseen frames)")
    for L0 in ev["bias"]:
        for name in ev["bias"][L0]:
            L.append(f"      L_0 = {L0} {name:16s} bias " + "; ".join(f"n {n} " + ("-" if v is None else f"{v[0]:+.4f} +- {v[1]:.4f}")
                                                                 for n, v in ev["bias"][L0][name].items())
                     + " | forces " + "; ".join(f"n {n} {fm(v)}" for n, v in ev["forces"][L0][name].items()))
    L.append(f"    E4's model's eps_inf (test frames of the training size): L_0 = 64 {ev['eps_inf'][64]}, "
             f"L_0 = 128 {ev['eps_inf'][128]}")
    L.append("    the polishes' L-BFGS steps: " + "; ".join(f"{k}: " + ", ".join(f"{n} {v}" for n, v in by.items())
                                                       for k, by in ev["steps"].items() if by))
    L.append("    the warm starts by cell (c: after the first stage -> polished; the gap after the first stage; forces)")
    for n, v in ev["cell"].items():
        L.append(f"      L_0 = {n} (coverage {100 * v['coverage']:.0f} %): first-stage gaps "
                 + ", ".join(f"seed {s} {x:+.3f}" for s, x in v["stage_gaps"].items()) + "; "
                 + "; ".join(f"{k} {c3(e['stage'])} -> {c3(e['end'])}" for k, e in v["ends"].items()))
        for w, by in v["forces"].items():
            L.append(f"        {w} forces " + "; ".join(f"n {m} {fm(x)}" for m, x in by.items()))
        for k, p in v["paths"].items():
            st, po = p["stage"], p["polish"]
            if not st and not po:
                continue
            txt = f"        {k}: "
            if st:
                txt += (", ".join(f"{x:.3f} at {s}" for s, x in st["c"].items()) + f"; half by {st['half_step']}; pull "
                        + ", ".join("-" if x is None else f"{x:+.2f}" for x in st["pull"]) + " | ")
            if po:
                l250 = "-" if po["last_250"] is None else f"{po['last_250']:+.3f}"
                txt += "polish " + ", ".join(f"{x:.3f} at {s}" for s, x in po["c"].items()) + f"; last 250 {l250}"
            L.append(txt)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for k, d in DIRS.items():
        ap.add_argument(f"--{k}", default=d)
    ap.add_argument("--out", default=None, help="also write the readings as json")
    a = ap.parse_args()
    R = {k: J7.load_runs(getattr(a, k)) for k in DIRS}
    ev = evaluate(R)
    print(report(ev))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(ev, fh, indent=1)


if __name__ == "__main__":
    main()
