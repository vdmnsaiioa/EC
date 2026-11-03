"""
e4_report.py -- E4's results read against the pre-registration (notes/e4-x3-preregistration.md), from the result
jsons of scripts/e4_water.py and the data sets' frames.  NumPy only: seconds on a login node, no GPU, no JAX.

    python scripts/e4_report.py --out_dir results/e4_job1 --data data/e4 [--json results/e4_job1/report.json]

It prints, in the pre-registration's order:
  1. the readings the matching fixes (section 4): eps_inf, T5 and the far-field probes of every run, against the values
     computed before training.  The probes at 10-15 A, T5 and eps_inf must agree across seeds and between the C rungs
     to round-off.  The 8 A probes are not fixed: 9 of their 20 test probes reach inside r_c of the core, where E_0 acts;
  2. the bulk table: energy per molecule (rmse and bias) and rigid-body forces, by size and frame set -- each split, and
     the unseen frames pooled (valid + test at the training size; every frame at the larger sizes);
  3. predictions 1 and 2 (section 5): the C rungs against the dagger, and the field term's rung against its ablation,
     paired by seed, at every size and frame set;
  4. prediction 3: each rung's energy bias from the training size to the largest, against its frame-to-frame standard
     error as registered, and the same with the frames' correlation in time (an addition, marked as such);
  5. the truth's energy per molecule by size and trajectory, and each rung's per-frame error against it;
  6. prediction 4 (Form III): M_G's errors by size;
  7. training: time and the Adam losses (the L-BFGS steps are in the logs only).
The registered criteria do not fix which frame set a bulk prediction is read on; every set is printed, and a verdict
that depends on the set says so.
"""
import argparse
import glob
import json
import math
import os

import numpy as np

KCAL = 627.5094740631                         # kcal/mol per hartree (sseft.units)
FIELD, ABLATION, DAGGER, GLOBAL = "M_16C2pm_us", "M_16C2pm", "M_16pm", "M_G"
C_RUNGS = (FIELD, ABLATION)
FIXED_DISTANCES = ("10", "12", "15")          # every test probe beyond r_c of its core (scripts/e4_water.py probes)
SPLITS = ("train", "valid", "test")

# notes/e4-x3-preregistration.md section 4, computed before training (hash 72b73e3aa6c7b53f)
PREREG = {
    "response": {"eps_inf": 1.8003, "eps_inf_se": 0.0021, "t5_pct": 3.7, "t5_slope": 1.026,
                 "nonadditive_pct": {"8": 1.9, "10": 2.5, "12": 2.6, "15": 2.0},
                 "interaction_pct": {"8": 1.4, "10": 0.5, "12": 0.7, "15": 0.5}},
    "pinned": {"t5_pct": 100.0, "t5_slope": 0.0,
               "nonadditive_pct": {"8": 100.0, "10": 100.0, "12": 100.0, "15": 100.0},
               "interaction_pct": {"8": 21.0, "10": 19.0, "12": 22.0, "15": 25.0}},
    "none": {"t5_pct": 100.0, "t5_slope": 0.0},
    "probes_hash": "72b73e3aa6c7b53f",
}


# ---------------------------------------------------------------- loading -------------------------------------------
def load_runs(out_dir):
    """{rung: {seed: result json}}"""
    runs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "*_seed*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        if "rung" in r and "readings" in r:
            runs.setdefault(r["rung"], {})[int(r["seed"])] = r
    return runs


def load_frames(data_dir, sizes):
    """{(split, n): {"u": the truth's energy per molecule (kcal/mol), "source": trajectory seed, "time_ps"}} for the
    data sets that exist."""
    out = {}
    for n in sizes:
        for split in SPLITS:
            p = os.path.join(data_dir, f"{split}_n{n}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=False)
            e = np.asarray(d["energy"], float) * KCAL / int(n)
            src = np.asarray(d["source"], int) if "source" in d.files else np.zeros(len(e), int)
            t = np.asarray(d["time_ps"], float) if "time_ps" in d.files else np.arange(len(e), dtype=float)
            out[(split, int(n))] = {"u": e, "source": src, "time_ps": t}
    return out


def kind(r):
    """the rung's IR sector from its model record: response (the C rungs), pinned (the dagger), learned, none."""
    m = r.get("model", {})
    if m.get("band_fields") is not None:
        return "response"
    if m.get("charges") == "pinned":
        return "pinned"
    if m.get("charges") == "learned" or m.get("polarisabilities") == "learned":
        return "learned"
    return "none"


def train_size(runs):
    for by in runs.values():
        for r in by.values():
            return int(r["config"]["data"]["train_size"])
    return 64


def sizes_of(runs):
    return sorted({int(n) for by in runs.values() for r in by.values() for n in r["readings"]["cells"]})


def ms(v, digits=4):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{np.mean(v):.{digits}f} +- {np.std(v, ddof=1):.{digits}f}"


# ---------------------------------------------------------------- 1. the fixed readings -----------------------------
def pct(v, a, b):
    return 100.0 * v[a] / v[b] if v.get(b) else float("nan")


def fixed_readings(runs):
    """every run's eps_inf, T5 and probes, by rung, with the pre-registered values and the spread across the runs that
    must agree."""
    rows, agree = {}, {}
    for name, by in runs.items():
        for seed, r in sorted(by.items()):
            rd, k = r["readings"], kind(r)
            row = {"kind": k, "probes_hash": r.get("probes_hash")}
            for n, splits in rd["cells"].items():
                for split, v in splits.items():
                    if "eps_inf" in v:
                        row["eps_inf"] = v["eps_inf"]["isotropic_mean"]
                        row["eps_inf_frames"] = v["eps_inf"]["frames"]
                        row["eps_inf_frame_sd"] = float(np.std(v["eps_inf"]["isotropic"], ddof=1)) \
                            if len(v["eps_inf"]["isotropic"]) > 1 else None
            if "far_trimers" in rd:
                t = rd["far_trimers"]
                row["t5_pct"] = 100.0 * t["three_body_rmse"] / t["three_body_rms"]
                row["t5_slope"] = t["slope"]
            if "probes" in rd:
                b = rd["probes"]["by_distance_angstrom"]
                row["nonadditive_pct"] = {d: pct(v, "nonadditive_rmse", "nonadditive_rms") for d, v in b.items()}
                row["interaction_pct"] = {d: pct(v, "interaction_rmse", "interaction_rms") for d, v in b.items()}
            before = (r.get("matching") or {}).get("probes_before_training")
            if isinstance(before, dict):
                row["before_nonadditive_pct"] = {d: pct(v, "nonadditive_rmse", "nonadditive_rms")
                                                 for d, v in before["by_distance_angstrom"].items()}
            rows.setdefault(name, {})[seed] = row
    # what must agree: every response run's eps_inf, T5 and 10-15 A probes; the pinned runs' T5 and probes
    for k in ("response", "pinned"):
        vals = [row for by in rows.values() for row in by.values() if row["kind"] == k]
        if not vals:
            continue
        a = {}
        for key in ("eps_inf", "t5_pct", "t5_slope"):
            x = [row[key] for row in vals if key in row]
            if x:
                a[key] = {"min": min(x), "max": max(x), "spread": max(x) - min(x), "runs": len(x)}
        for key in ("nonadditive_pct", "interaction_pct"):
            for d in FIXED_DISTANCES:
                x = [row[key][d] for row in vals if key in row and d in row[key]]
                if x:
                    a[f"{key}_{d}"] = {"min": min(x), "max": max(x), "spread": max(x) - min(x), "runs": len(x)}
        agree[k] = a
    return rows, agree


def print_fixed(rows, agree):
    print("1. The readings the matching fixes (pre-registration section 4)")
    hashes = sorted({row["probes_hash"] for by in rows.values() for row in by.values() if row["probes_hash"]})
    print(f"   probes hash: {', '.join(hashes)} (pre-registered {PREREG['probes_hash']})")
    for name, by in rows.items():
        k = next(iter(by.values()))["kind"]
        pre = PREREG.get(k, {})
        line = f"   {name} ({k}, seeds {sorted(by)}):"
        e = [row["eps_inf"] for row in by.values() if "eps_inf" in row]
        if e:
            line += f" eps_inf {ms(e)} (pre-registered {pre.get('eps_inf', '-')});"
        t = [row["t5_pct"] for row in by.values() if "t5_pct" in row]
        s = [row["t5_slope"] for row in by.values() if "t5_slope" in row]
        if t:
            line += f" T5 {ms(t, 2)} %, slope {ms(s)} (pre-registered {pre.get('t5_pct', '-')} %, {pre.get('t5_slope', '-')});"
        print(line)
        na = [row["nonadditive_pct"] for row in by.values() if "nonadditive_pct" in row]
        it = [row["interaction_pct"] for row in by.values() if "interaction_pct" in row]
        if na:
            ds = sorted(na[0], key=float)
            print("      probes, non-additive %: " + "; ".join(
                f"{d} A {ms([x[d] for x in na], 1)}" + (f" ({pre['nonadditive_pct'][d]:g})" if d in pre.get("nonadditive_pct", {}) else "")
                for d in ds))
            print("      probes, interaction %:  " + "; ".join(
                f"{d} A {ms([x[d] for x in it], 1)}" + (f" ({pre['interaction_pct'][d]:g})" if d in pre.get("interaction_pct", {}) else "")
                for d in ds))
    for k, a in agree.items():
        worst = max(a.items(), key=lambda kv: kv[1]["spread"]) if a else None
        if worst:
            print(f"   {k} runs: largest spread across runs that must agree: {worst[0]} {worst[1]['spread']:.2e} "
                  f"({worst[1]['runs']} runs)")
    print()


# ---------------------------------------------------------------- 2. the bulk table ---------------------------------
def unseen_splits(n, L0):
    return ("valid", "test") if n == L0 else SPLITS


def cell_reading(r, n, reading, L0):
    """one run's energy and force readings on a frame set: a split, or 'unseen' (pooled), or 'all' (pooled)."""
    cells = r["readings"]["cells"].get(str(n), {})
    if reading in SPLITS:
        v = cells.get(reading, {})
        if "energy_per_molecule_rmse" not in v:
            return None
        return {"frames": v["frames"], "errors": np.asarray(v["energy_per_molecule_errors"], float),
                "energy_rmse": v["energy_per_molecule_rmse"], "bias": v["energy_per_molecule_mean_error"],
                "force_rigid": v["force_rigid_rmse"], "net": v["net_force_rmse"], "torque": v["torque_rmse"],
                "atomic": v["force_atomic_rmse"]}
    parts = unseen_splits(n, L0) if reading == "unseen" else SPLITS
    vs = [cell_reading(r, n, s, L0) for s in parts]
    if any(v is None for v in vs):
        return None
    e = np.concatenate([v["errors"] for v in vs])
    w = np.array([v["frames"] for v in vs], float)          # every frame of a size has the same molecules: frame weights
    q = lambda key: float(np.sqrt(np.sum(w * np.square([v[key] for v in vs])) / np.sum(w)))
    return {"frames": int(len(e)), "errors": e, "energy_rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
            "force_rigid": q("force_rigid"), "net": q("net"), "torque": q("torque"), "atomic": q("atomic")}


def bulk_table(runs, L0):
    out = {}
    for name, by in runs.items():
        for n in sizes_of(runs):
            for reading in ("train", "valid", "test", "unseen"):
                vals = {seed: cell_reading(r, n, reading, L0) for seed, r in by.items()}
                vals = {s: v for s, v in vals.items() if v is not None}
                if vals:
                    out.setdefault(name, {}).setdefault(n, {})[reading] = vals
    return out


def print_bulk(table, L0):
    print("2. Bulk readings: energy per molecule rmse and bias (kcal/mol), rigid-body force rmse (kcal/mol/A);"
          " seed means +- sd")
    print(f"   'unseen' pools valid + test at the training size ({L0} molecules) and every frame at the larger sizes")
    for name, by_n in table.items():
        print(f"   {name}")
        for n, by_r in sorted(by_n.items()):
            for reading, vals in by_r.items():
                v = list(vals.values())
                print(f"     n{n:<5d} {reading:6s} ({v[0]['frames']:3d} frames): E {ms([x['energy_rmse'] for x in v])}, "
                      f"bias {ms([x['bias'] for x in v])}; F {ms([x['force_rigid'] for x in v], 3)} "
                      f"(net {ms([x['net'] for x in v], 3)}, torque {ms([x['torque'] for x in v], 3)}; "
                      f"atomic {ms([x['atomic'] for x in v], 2)})")
    print()


# ---------------------------------------------------------------- 3. predictions 1 and 2 ----------------------------
def paired(table, a, b, n, reading, key):
    """seeds where rung a's reading is at or below rung b's, the seeds compared, and the seed means."""
    va, vb = table.get(a, {}).get(n, {}).get(reading, {}), table.get(b, {}).get(n, {}).get(reading, {})
    seeds = sorted(set(va) & set(vb))
    if not seeds:
        return None
    wins = sum(va[s][key] <= vb[s][key] for s in seeds)
    ma, mb = float(np.mean([va[s][key] for s in seeds])), float(np.mean([vb[s][key] for s in seeds]))
    need = math.ceil(0.75 * len(seeds))
    return {"wins": int(wins), "seeds": len(seeds), "pass": bool(wins >= need), "mean_a": ma, "mean_b": mb,
            "change": ma / mb - 1.0 if mb else float("nan")}


def predictions_12(table, runs):
    comps = [("1", FIELD, DAGGER), ("1", ABLATION, DAGGER), ("2", FIELD, ABLATION)]
    out = []
    for pid, a, b in comps:
        if a not in table or b not in table:
            continue
        for n in sizes_of(runs):
            for reading in ("train", "valid", "test", "unseen"):
                for key, label in (("energy_rmse", "E"), ("force_rigid", "F")):
                    p = paired(table, a, b, n, reading, key)
                    if p is not None:
                        out.append(dict(p, prediction=pid, a=a, b=b, n=n, reading=reading, metric=label))
    return out


def print_predictions_12(rows, L0):
    print("3. Predictions 1 and 2: rung a at or below rung b, paired by seed (>= 3 of 4 seeds); seed means, change")
    print("   (at the training size the train split was fitted, so it is printed but not a reading)")
    groups = {}
    for r in rows:
        groups.setdefault((r["prediction"], r["a"], r["b"]), []).append(r)
    for (pid, a, b), rs in groups.items():
        print(f"   prediction {pid}: {a} <= {b}")
        for metric in ("E", "F"):
            cells = [r for r in rs if r["metric"] == metric]
            line = "; ".join(f"n{r['n']} {r['reading']} {r['wins']}/{r['seeds']}{'' if r['pass'] else ' FAIL'} "
                             f"({100 * r['change']:+.0f} %)" for r in cells
                             if not (r["n"] == L0 and r["reading"] == "train"))
            print(f"     {metric}: {line}")
        read = [r for r in rs if not (r["n"] == L0 and r["reading"] == "train")]
        for sets, label in ((("test",), "test split"), (("valid",), "valid split"), (("unseen",), "unseen pooled")):
            sel = [r for r in read if r["reading"] in sets]
            if sel:
                print(f"     on the {label} at every size: E {'holds' if all(r['pass'] for r in sel if r['metric'] == 'E') else 'fails'}, "
                      f"F {'holds' if all(r['pass'] for r in sel if r['metric'] == 'F') else 'fails'}")
    print()


# ---------------------------------------------------------------- 4. prediction 3 -----------------------------------
def segments(frames_meta, n, reading, L0):
    """the trajectory segment of every frame of a frame set (a split's frames of one source are one segment), in the
    order cell_reading concatenates them."""
    parts = (reading,) if reading in SPLITS else (unseen_splits(n, L0) if reading == "unseen" else SPLITS)
    seg = []
    for split in parts:
        m = frames_meta.get((split, n))
        if m is None:
            return None
        seg += [f"{split}:{s}" for s in m["source"]]
    return np.array(seg)


def lag1(e, seg):
    """the lag-1 autocorrelation of e within segments (frames in time order), pooled over segments."""
    num, den = 0.0, 0.0
    for s in np.unique(seg):
        x = e[seg == s]
        if len(x) < 3:
            continue
        x = x - x.mean()
        num += float(np.sum(x[1:] * x[:-1]))
        den += float(np.sum(x * x))
    return num / den if den > 0 else 0.0


def bias_stats(e, seg):
    """the mean, its frame-to-frame standard error (as registered) and the same with the lag-1 correlation in time
    (n_eff = n (1 - rho) / (1 + rho), rho clipped to [0, 0.95])."""
    n = len(e)
    sd = float(np.std(e, ddof=1)) if n > 1 else float("nan")
    rho = lag1(e, seg) if seg is not None else 0.0
    rc = min(max(rho, 0.0), 0.95)
    n_eff = n * (1 - rc) / (1 + rc)
    return {"bias": float(np.mean(e)), "se": sd / math.sqrt(n), "rho": rho, "n_eff": n_eff,
            "se_corr": sd / math.sqrt(max(n_eff, 1.0)), "frames": n}


def prediction_3(table, frames_meta, runs, L0):
    sizes = sizes_of(runs)
    big = max(sizes)
    out = {}
    if big == L0:
        return out
    for name, by_n in table.items():
        for reading in ("test", "valid", "unseen"):
            if reading not in by_n.get(L0, {}) or reading not in by_n.get(big, {}):
                continue
            a, b = by_n[L0][reading], by_n[big][reading]
            seeds = sorted(set(a) & set(b))
            if not seeds:
                continue
            sa, sb = segments(frames_meta, L0, reading, L0), segments(frames_meta, big, reading, L0)
            per_seed = []
            for s in seeds:
                x, y = bias_stats(a[s]["errors"], sa), bias_stats(b[s]["errors"], sb)
                d = y["bias"] - x["bias"]
                se, sec = math.hypot(x["se"], y["se"]), math.hypot(x["se_corr"], y["se_corr"])
                per_seed.append({"seed": s, "shift": d, "z": d / se if se > 0 else float("nan"),
                                 "z_corr": d / sec if sec > 0 else float("nan")})
            ea = np.mean([a[s]["errors"] for s in seeds], axis=0)       # the seeds share the frames: their mean error
            eb = np.mean([b[s]["errors"] for s in seeds], axis=0)
            x, y = bias_stats(ea, sa), bias_stats(eb, sb)
            d = y["bias"] - x["bias"]
            se, sec = math.hypot(x["se"], y["se"]), math.hypot(x["se_corr"], y["se_corr"])
            mid = {}
            for n in sizes:
                if n not in (L0, big) and reading in by_n.get(n, {}):
                    c = by_n[n][reading]
                    mid[n] = bias_stats(np.mean([c[s]["errors"] for s in seeds if s in c], axis=0),
                                        segments(frames_meta, n, reading, L0))
            out.setdefault(name, {})[reading] = {
                "from": x, "to": y, "mid": mid, "shift": d, "se": se, "z": d / se if se > 0 else float("nan"),
                "se_corr": sec, "z_corr": d / sec if sec > 0 else float("nan"), "holds": bool(abs(d) <= 2 * se),
                "holds_corr": bool(abs(d) <= 2 * sec), "per_seed": per_seed,
                "seeds_beyond_2se": int(sum(abs(p["z"]) > 2 for p in per_seed))}
    return out


def print_prediction_3(p3, L0, big):
    print(f"4. Prediction 3: the energy bias from {L0} to {big} molecules against twice its frame-to-frame standard "
          "error (seed-mean per-frame errors; the seeds share the frames)")
    print("   'corr': the same with the frames' lag-1 correlation in time (an addition, not the registered criterion)")
    for name, by in p3.items():
        for reading, v in by.items():
            mid = "; ".join(f"n{n} {m['bias']:+.4f}" for n, m in sorted(v["mid"].items()))
            print(f"   {name:12s} {reading:6s}: bias {v['from']['bias']:+.4f} -> {v['to']['bias']:+.4f}"
                  + (f" (via {mid})" if mid else "")
                  + f", shift {v['shift']:+.4f} +- {v['se']:.4f} (z {v['z']:+.1f}: {'holds' if v['holds'] else 'FAILS'});"
                  f" corr +- {v['se_corr']:.4f} (rho {v['from']['rho']:.2f} / {v['to']['rho']:.2f}; z {v['z_corr']:+.1f}:"
                  f" {'holds' if v['holds_corr'] else 'fails'}); seeds beyond 2 se {v['seeds_beyond_2se']}/{len(v['per_seed'])}")
    print()


# ---------------------------------------------------------------- 5. the truth, and the errors against it -----------
def truth_by_size(frames_meta):
    """the truth's energy per molecule: each trajectory's mean (valid and test of the last seed together), and the
    size's mean over trajectories with the trajectory-to-trajectory standard error."""
    out = {}
    for n in sorted({n for (_, n) in frames_meta}):
        u = np.concatenate([frames_meta[(s, n)]["u"] for s in SPLITS if (s, n) in frames_meta])
        src = np.concatenate([frames_meta[(s, n)]["source"] for s in SPLITS if (s, n) in frames_meta])
        traj = {int(t): {"mean": float(u[src == t].mean()), "sd": float(u[src == t].std(ddof=1)) if np.sum(src == t) > 1
                         else float("nan"), "frames": int(np.sum(src == t))} for t in np.unique(src)}
        m = np.array([v["mean"] for v in traj.values()])
        out[n] = {"trajectories": traj, "mean": float(m.mean()),
                  "se_trajectories": float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else float("nan")}
    return out


def error_vs_truth(table, frames_meta, runs, L0):
    """per rung and size: the slope and correlation of the seed-mean per-frame error against the truth's energy per
    molecule on the unseen frames of the size, both centred on their trajectory's mean (within-trajectory), and the
    bias on each trajectory."""
    out = {}
    for name, by_n in table.items():
        for n, by_r in by_n.items():
            parts = [s for s in unseen_splits(n, L0) if s in by_r and (s, n) in frames_meta]
            if not parts:
                continue
            seeds = sorted(set.intersection(*[set(by_r[s]) for s in parts]))
            e = np.concatenate([np.mean([by_r[s][k]["errors"] for k in seeds], axis=0) for s in parts])
            u = np.concatenate([frames_meta[(s, n)]["u"] for s in parts])
            src = np.concatenate([frames_meta[(s, n)]["source"] for s in parts])
            ec, uc = e.copy(), u.copy()
            for t in np.unique(src):
                ec[src == t] -= ec[src == t].mean()
                uc[src == t] -= uc[src == t].mean()
            den = float(np.sum(uc * uc))
            if den <= 0 or len(e) < 4:
                continue
            slope = float(np.sum(uc * ec) / den)
            res = ec - slope * uc
            dof = max(len(e) - len(np.unique(src)) - 1, 1)
            se = float(np.sqrt(np.sum(res ** 2) / dof / den))
            r = float(np.sum(uc * ec) / np.sqrt(den * np.sum(ec * ec))) if np.sum(ec * ec) > 0 else 0.0
            traj_bias = {int(t): float(e[src == t].mean()) for t in np.unique(src)}
            out.setdefault(name, {})[n] = {"slope": slope, "slope_se": se, "r": r, "frames": int(len(e)),
                                           "truth_sd_within": float(np.sqrt(den / len(e))),
                                           "bias_by_trajectory": traj_bias}
    return out


def print_truth(truth, evt, L0):
    print("5. The truth's energy per molecule (kcal/mol) by size: trajectory means, and their mean +- se over trajectories")
    base = truth.get(L0, {}).get("mean")
    for n, v in sorted(truth.items()):
        tr = ", ".join(f"s{t} {x['mean']:.4f} ({x['frames']})" for t, x in sorted(v["trajectories"].items()))
        rel = f"; minus n{L0}'s {v['mean'] - base:+.4f}" if base is not None and n != L0 else ""
        print(f"   n{n:<5d} {v['mean']:.4f} +- {v['se_trajectories']:.4f}{rel}  [{tr}]")
    print("   each rung's per-frame error against the truth on the unseen frames (within trajectories): slope, r;"
          " its bias by trajectory (s<seed>; at the training size the last seed's valid and test frames)")
    for name, by_n in evt.items():
        for n, v in sorted(by_n.items()):
            tb = ", ".join(f"s{t} {b:+.4f}" for t, b in sorted(v["bias_by_trajectory"].items()))
            print(f"   {name:12s} n{n:<5d} slope {v['slope']:+.4f} +- {v['slope_se']:.4f}, r {v['r']:+.2f} "
                  f"(truth sd {v['truth_sd_within']:.3f}, {v['frames']} frames); bias by trajectory {tb}")
    print()


# ---------------------------------------------------------------- 6, 7 ----------------------------------------------
def print_form3(table):
    if GLOBAL not in table:
        return
    print("6. Prediction 4 (Form III): M_G's bulk errors by size, unseen frames")
    for n, by_r in sorted(table[GLOBAL].items()):
        if "unseen" in by_r:
            v = list(by_r["unseen"].values())
            print(f"   n{n:<5d} E {ms([x['energy_rmse'] for x in v], 3)}, bias {ms([x['bias'] for x in v], 3)}; "
                  f"F {ms([x['force_rigid'] for x in v], 3)}")
    print()


def training(runs):
    out = {}
    for name, by in runs.items():
        for seed, r in sorted(by.items()):
            h = r.get("training", {}).get("history") or []
            out.setdefault(name, {})[seed] = {"hours": r.get("training", {}).get("time", float("nan")) / 3600.0,
                                              "adam": [[int(x[0]), float(x[1])] for x in h]}
    return out


def print_training(tr):
    print("7. Training: hours; the Adam loss at its recorded steps (the L-BFGS polish is in the logs only)")
    for name, by in tr.items():
        for s, v in sorted(by.items()):
            curve = " ".join(f"{x[1]:.1e}" for x in v["adam"])
            print(f"   {name:12s} s{s} {v['hours']:.2f} h; Adam {curve}")
    print()


# ---------------------------------------------------------------- main ----------------------------------------------
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def report(out_dir, data_dir, json_path=None, verbose=True):
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit(f"no results in {out_dir}")
    L0, sizes = train_size(runs), sizes_of(runs)
    frames_meta = load_frames(data_dir, sizes)
    rows, agree = fixed_readings(runs)
    table = bulk_table(runs, L0)
    p12 = predictions_12(table, runs)
    p3 = prediction_3(table, frames_meta, runs, L0)
    truth = truth_by_size(frames_meta) if frames_meta else {}
    evt = error_vs_truth(table, frames_meta, runs, L0) if frames_meta else {}
    tr = training(runs)
    if verbose:
        print(f"{out_dir}: {sum(len(b) for b in runs.values())} runs, rungs {sorted(runs)}; training size {L0}; "
              f"sizes {sizes}; frames from {data_dir}: {'yes' if frames_meta else 'no (sections 4-5 need them)'}\n")
        print_fixed(rows, agree)
        print_bulk(table, L0)
        print_predictions_12(p12, L0)
        if p3:
            print_prediction_3(p3, L0, max(sizes))
        if truth:
            print_truth(truth, evt, L0)
        print_form3(table)
        print_training(tr)
    strip = {name: {n: {rd: {s: {k: v for k, v in x.items() if k != "errors"} for s, x in vals.items()}
                        for rd, vals in by_r.items()} for n, by_r in by_n.items()} for name, by_n in table.items()}
    result = {"runs": {k: sorted(v) for k, v in runs.items()}, "train_size": L0, "sizes": sizes, "fixed": rows,
              "fixed_agreement": agree, "bulk": strip, "predictions_1_2": p12, "prediction_3": p3, "truth": truth,
              "error_vs_truth": evt, "training": tr, "prereg": PREREG}
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(jsonable(result), fh, indent=1)
        if verbose:
            print(f"written {json_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description="E4's results read against the pre-registration")
    ap.add_argument("--out_dir", default="results/e4_job1")
    ap.add_argument("--data", default="data/e4", help="the data sets (for the truth's energies and the trajectories)")
    ap.add_argument("--json", default=None, help="also write every number to this json")
    a = ap.parse_args()
    report(a.out_dir, a.data, a.json)


if __name__ == "__main__":
    main()
