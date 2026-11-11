"""
e4_report.py -- E4's results read against the pre-registration (notes/e4-x3-preregistration.md), from the result
jsons of scripts/e4_water.py and the data sets' frames.  NumPy only: seconds on a login node, no GPU, no JAX.

    python scripts/e4_report.py --out_dir results/e4_job1 --data data/e4 [--json results/e4_job1/report.json]
    python scripts/e4_report.py --job3 results/e4_job3/kread.json      # job 3's registered readings (v0.0.27)
    python scripts/e4_report.py --job4 results/e4_job4                 # job 4's, from its kread_*.json (v0.0.28)
    python scripts/e4_report.py --job5 results/e4_job5                 # job 5's, arm D' (v0.0.29)

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

# notes/e4-job2-preregistration.md: the readings fixed before training (sections 3-4; arm B's from the local calibration
# on 10 frames, Niflheim's on 30) and the bulk predictions (section 5) as (id, rung a, rung b, metric, rule); the
# readings beside them are printed the same way, unregistered
_ARM_A = {"eps_inf": 1.744, "t5_pct": 4.7, "t5_slope": 1.042, "nonadditive_pct": {"10": 2.3, "12": 2.1, "15": 2.1},
          "eps_inf_model": 1.743, "cell_dipole_rel_rmse": 0.089, "molecular_dipole_rel_rmse": 0.082,
          "molecular_response_rel_rmse": 0.092, "molecular_response_iso_rel_rmse": 0.087}
_ARM_B = {"eps_inf": 1.688, "t5_pct": 4.2, "t5_slope": 1.040, "nonadditive_pct": {"10": 2.1, "12": 2.6, "15": 2.8},
          "eps_inf_model": 1.688, "cell_dipole_rel_rmse": 0.021, "molecular_dipole_rel_rmse": 0.046,
          "molecular_response_rel_rmse": 0.022, "molecular_response_iso_rel_rmse": 0.017}
# arm C (notes/e4-job2c-preregistration.md section 3): the calibration on the order-0 columns alone, on the local frames
_ARM_C = {"eps_inf": 1.689, "t5_pct": 4.1, "t5_slope": 1.040, "nonadditive_pct": {"10": 2.3, "12": 2.2, "15": 2.7},
          "eps_inf_model": 1.6895, "cell_dipole_rel_rmse": 0.031, "molecular_dipole_rel_rmse": 0.050,
          "molecular_response_rel_rmse": 0.031, "molecular_response_iso_rel_rmse": 0.023}
# arm D (notes/e4-job4-preregistration.md section 8): job 4's matching level on Niflheim (e4_report --job4), arm C's
# Niflheim readings (job 3's kread.json) moved by the shifts D3-D5 report; chi: the finite-k response before training
_ARM_D = {"eps_inf": 1.689, "t5_slope": 1.040, "nonadditive_pct": {"10": 2.39, "12": 2.46, "15": 2.95},
          "eps_inf_model": 1.6893, "cell_dipole_rel_rmse": 0.023, "molecular_dipole_rel_rmse": 0.051,
          "molecular_response_rel_rmse": 0.029, "molecular_response_iso_rel_rmse": 0.015, "chi": {"1": 0.0133, "2": -0.0037}}
# arm D' (notes/e4-job5-preregistration.md section 8): job 5's matching level on Niflheim (e4_report --job5); its far
# probes are arm C's (arm D's less job 4's D5 shifts) moved by job 5's P5 shifts, to about 0.01 point
_ARM_DP = {"eps_inf": 1.689, "t5_slope": 1.040, "nonadditive_pct": {"10": 2.23, "12": 1.98, "15": 2.66},
           "eps_inf_model": 1.6893, "cell_dipole_rel_rmse": 0.016, "molecular_dipole_rel_rmse": 0.051,
           "molecular_response_rel_rmse": 0.029, "molecular_response_iso_rel_rmse": 0.015,
           "chi": {"1": 0.0132, "2": -0.0038}}
JOB2_PREREG = {"probes_hash": "72b73e3aa6c7b53f", "M_16C2pm_us [A]": _ARM_A, "M_16C2pm [A]": _ARM_A,
               "M_16C2pm_us [B]": _ARM_B, "M_16C2pm [B]": _ARM_B, "M_16C2pm_us [C]": _ARM_C, "M_16C2pm [C]": _ARM_C,
               "M_16C2pm_us [D]": _ARM_D, "M_16C2pm [D]": _ARM_D, "M_16C2pm_us [D']": _ARM_DP,
               "M_16C2pm [D']": _ARM_DP, "pinned": PREREG["pinned"]}
JOB2_PREDICTIONS = [
    ("J1", "M_16C2pm [B]", "M_16pm [job 1]", "force_rigid", "le"),
    ("J1a", "M_16C2pm [A]", "M_16pm [job 1]", "force_rigid", ("change_le", 0.02)),
    ("J2", "M_16C2pm_us [A]", "M_16C2pm_us [job 1]", "force_rigid", ("abs_change_le", 0.05)),
    ("J2", "M_16C2pm_us [B]", "M_16C2pm_us [A]", "force_rigid", ("abs_change_le", 0.05)),
    ("J3", "M_16C2pm_us [B]", "M_16pm_us [A]", "force_rigid", "le"),
]
# arm C's bulk predictions (notes/e4-job2c-preregistration.md section 4)
JOB2C_PREDICTIONS = [
    ("C1", "M_16C2pm [C]", "M_16C2pm [A]", "force_rigid", ("change_le", 0.01)),
    ("C2", "M_16C2pm [C]", "M_16C2pm [A]", "force_rigid", "le"),
    ("C3", "M_16C2pm [C]", "M_16pm [job 1]", "force_rigid", "le"),
    ("C4", "M_16C2pm_us [C]", "M_16C2pm_us [A]", "force_rigid", ("abs_change_le", 0.01)),
]
JOB2C_READINGS = [
    ("reading", "M_16C2pm [C]", "M_16C2pm [B]", "force_rigid", "le"),
    ("reading", "M_16C2pm_us [C]", "M_16pm_us [A]", "force_rigid", "le"),
    ("reading", "M_16C2pm [C]", "M_16C2pm [A]", "net", "le"),
] + [("energy", a, b, "energy_rmse", rule) for _, a, b, _, rule in JOB2C_PREDICTIONS]
# arm D's bulk predictions (notes/e4-job4-preregistration.md section 8, T1-T2): ('change_in', lo, hi) -- the seed
# means' change a / b - 1 within [lo, hi]; 'ge' -- a at or above b on >= 3 of 4 seeds, paired
JOB4_PREDICTIONS = [
    ("T1", "M_16C2pm_us [D]", "M_16C2pm_us [C]", "force_rigid", ("abs_change_le", 0.015)),
    ("T2", "M_16C2pm [D]", "M_16C2pm [C]", "force_rigid", ("change_in", 0.005, 0.045)),
    ("T2", "M_16C2pm [D]", "M_16C2pm [C]", "force_rigid", "ge"),
]
JOB4_READINGS = [
    ("reading", "M_16C2pm_us [D]", "M_16pm_us [A]", "force_rigid", "le"),
    ("reading", "M_16C2pm [D]", "M_16pm [job 1]", "force_rigid", "le"),
    ("reading", "M_16C2pm [D]", "M_16C2pm [C]", "net", "le"),
] + [("energy", a, b, "energy_rmse", rule) for _, a, b, _, rule in JOB4_PREDICTIONS[:2]]
# arm D''s bulk predictions (notes/e4-job5-preregistration.md section 8, T1'-T3'), and the model rule registered with
# them (M': D''s frozen form is the model if within +1.0 % of arm C's at every size)
JOB5_PREDICTIONS = [
    ("T1'", "M_16C2pm_us [D']", "M_16C2pm_us [C]", "force_rigid", ("change_in", 0.005, 0.040)),
    ("T2'", "M_16C2pm [D']", "M_16C2pm [C]", "force_rigid", ("change_in", 0.005, 0.040)),
    ("T3'", "M_16C2pm_us [D']", "M_16C2pm_us [D]", "force_rigid", "le"),
    ("T3'", "M_16C2pm [D']", "M_16C2pm [D]", "force_rigid", "le"),
    ("M'", "M_16C2pm_us [D']", "M_16C2pm_us [C]", "force_rigid", ("change_le", 0.010)),
]
JOB5_READINGS = [
    ("reading", "M_16C2pm_us [D']", "M_16pm_us [A]", "force_rigid", "le"),
    ("reading", "M_16C2pm [D']", "M_16pm [job 1]", "force_rigid", "le"),
    ("reading", "M_16C2pm [D']", "M_16C2pm [C]", "net", "le"),
] + [("energy", a, b, "energy_rmse", rule) for _, a, b, _, rule in JOB5_PREDICTIONS[:2]]
JOB2_READINGS = [
    ("reading", "M_16pm_us [A]", "M_16C2pm_us [job 1]", "force_rigid", "le"),
    ("reading", "M_16C2pm_us [A]", "M_16pm_us [A]", "force_rigid", "le"),
    ("reading", "M_16C2pm [A]", "M_16pm [job 1]", "net", "le"),
    ("reading", "M_16C2pm [B]", "M_16pm [job 1]", "net", "le"),
] + [("energy", a, b, "energy_rmse", rule) for _, a, b, _, rule in JOB2_PREDICTIONS]


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


def print_fixed(rows, agree, prereg=None, title="1. The readings the matching fixes (pre-registration section 4)"):
    prereg = PREREG if prereg is None else prereg
    print(title)
    hashes = sorted({row["probes_hash"] for by in rows.values() for row in by.values() if row["probes_hash"]})
    print(f"   probes hash: {', '.join(hashes)} (pre-registered {prereg.get('probes_hash', PREREG['probes_hash'])})")
    for name, by in rows.items():
        k = next(iter(by.values()))["kind"]
        pre = prereg.get(name, prereg.get(k, {}))
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


def print_prediction_3(p3, L0, big, label="4. Prediction 3"):
    print(f"{label}: the energy bias from {L0} to {big} molecules against twice its frame-to-frame standard "
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


# ---------------------------------------------------------------- job 2 (notes/e4-job2-preregistration.md) ----------
def job2_runs(job2a, job2b, job1=None, job2c=None, job2d=None, job5=None):
    """job 2's runs and the job-1 runs they are read against, {label: {seed: json}}: '<rung> [A]' (arm A, the mutual
    edge at 2.0 A), '<rung> [B]' (arm B, calibrated on the liquid), '<rung> [C]' (arm C, calibrated on the order-0
    columns alone), '<rung> [D]' (job 4's arm D, v0.0.28: arm C with its orders >= 1 on the finite-k response),
    "<rung> [D']" (job 5's arm D', v0.0.29: arm D's third stage anchored at arm C) and '<rung> [job 1]'."""
    out = {}
    for d, tag in ((job2a, "A"), (job2b, "B"), (job2c, "C"), (job2d, "D"), (job5, "D'"), (job1, "job 1")):
        if d and os.path.isdir(d):
            for name, by in load_runs(d).items():
                out[f"{name} [{tag}]"] = by
    return out


def liquid_rows(runs):
    """every run's held-out liquid readings, before training (on the matched read-out) and after."""
    out = {}
    for label, by in runs.items():
        for seed, r in sorted(by.items()):
            after = r["readings"].get("liquid")
            before = (r.get("matching") or {}).get("liquid_before_training")
            if after or before:
                out.setdefault(label, {})[seed] = {"before": before, "after": after,
                                                   "liquid_objective": (r.get("matching") or {}).get("liquid_objective"),
                                                   "k_before": (r.get("matching") or {}).get("k_before_training")}
    return out


def print_liquid(rows):
    print("J0. The liquid readings on MB-pol's held-out frames (fixed before training): eps_inf, the cells' induced dipole,"
          " every molecule's induced dipole and field response (rmse / rms)")
    keys = (("eps_inf_model", "eps", 4, 1), ("cell_dipole_rel_rmse", "cell M %", 1, 100),
            ("molecular_dipole_rel_rmse", "mu %", 1, 100), ("molecular_response_rel_rmse", "R %", 1, 100),
            ("molecular_response_iso_rel_rmse", "R iso %", 1, 100))
    for label, by in rows.items():
        pre = JOB2_PREREG.get(label, {})
        after = [v["after"] for v in by.values() if v["after"]]
        before = [v["before"] for v in by.values() if v["before"]]
        if not after:
            continue
        truth = after[0]["eps_inf_truth"]
        drift = max(abs(a["eps_inf_model"] - b["eps_inf_model"]) for a, b in
                    ((v["after"], v["before"]) for v in by.values() if v["after"] and v["before"])) if before else float("nan")
        print(f"   {label} (seeds {sorted(by)}; {after[0]['frames']} frames, MB-pol eps_inf {truth:.4f}):"
              f" largest change in training {drift:.1e}")
        print("      " + "; ".join(f"{name} {ms([scale * v[key] for v in after if key in v], d)}"
                                    + (f" ({pre[key] * scale:.{d}f})" if key in pre else "")
                                    for key, name, d, scale in keys))
        kb = [v["k_before"] for v in by.values() if v.get("k_before")]
        if kb:                                                   # job 4 (v0.0.28): the finite-k response before training
            print("      finite k, held out, before training: " + "; ".join(
                f"chi(k{n}) {ms([100 * x['harmonics'][n]['chi_rel_diff'] for x in kb], 2)} %"
                + (f" ({100 * pre['chi'][n]:+.2f})" if n in pre.get("chi", {}) else "")
                + f", molecules {ms([100 * x['harmonics'][n]['molecular_rel_rmse'] for x in kb], 1)} %"
                for n in sorted(kb[0]["harmonics"])))
    print()


def job2_predictions(table, runs, entries=None):
    """the registered comparisons (id, rung a, rung b, metric key, rule): rule 'le' -- a at or below b on >= 3 of 4
    seeds, paired; ('change_le', x) -- the seed means' change a / b - 1 at most x; ('abs_change_le', x) -- its size
    at most x."""
    out = []
    for pid, a, b, key, rule in (JOB2_PREDICTIONS if entries is None else entries):
        if a not in table or b not in table:
            out.append({"prediction": pid, "a": a, "b": b, "metric": key, "rule": rule, "missing": True})
            continue
        for n in sizes_of(runs):
            for reading in ("valid", "test", "unseen"):
                p = paired(table, a, b, n, reading, key)
                if p is None:
                    continue
                if rule == "le":
                    ok = p["pass"]
                elif rule == "ge":                                # a at or above b on >= 3 of 4 seeds (v0.0.28)
                    va, vb = (table[x][n][reading] for x in (a, b))
                    seeds = sorted(set(va) & set(vb))
                    ok = sum(va[s_][key] >= vb[s_][key] for s_ in seeds) >= math.ceil(0.75 * len(seeds))
                elif rule[0] == "change_in":
                    ok = rule[1] <= p["change"] <= rule[2]
                else:
                    ok = (p["change"] if rule[0] == "change_le" else abs(p["change"])) <= rule[1]
                out.append(dict(p, prediction=pid, a=a, b=b, n=n, reading=reading, metric=key, rule=rule, ok=bool(ok)))
    return out


def field_term_gains(table, runs, reading="unseen", key="force_rigid"):
    """J4: the field term's gain on the C rungs, 1 - F(frozen form) / F(ablation), seed means, for job 1, arm A and arm
    B at every size; and on the dagger, 1 - F(M_16pm_us) / F(M_16pm), a reading."""
    out = {}
    pairs = [(f"{FIELD} [{t}]", f"{ABLATION} [{t}]", t) for t in ("job 1", "A", "B", "C", "D", "D'")]
    pairs.append(("M_16pm_us [A]", f"{DAGGER} [job 1]", "dagger"))
    for a, b, tag in pairs:
        for n in sizes_of(runs):
            va = table.get(a, {}).get(n, {}).get(reading, {})
            vb = table.get(b, {}).get(n, {}).get(reading, {})
            if va and vb:
                out.setdefault(tag, {})[n] = 1.0 - np.mean([v[key] for v in va.values()]) / np.mean([v[key] for v in vb.values()])
    return out


def print_job2_predictions(rows, title=None, registered=True):
    print(title or "J1-J3 (arm C's C1-C4, arm D's T1-T2, arm D''s T1'-T3' and the model rule M'). The bulk predictions "
          "(notes/e4-job2-preregistration.md section 5; notes/e4-job2c-preregistration.md section 4; "
          "notes/e4-job4-preregistration.md section 8; notes/e4-job5-preregistration.md section 8), paired by seed; the "
          "registered reading is the unseen frames pooled at every size")
    groups = {}
    for r in rows:
        groups.setdefault((r["prediction"], r["a"], r["b"], r["metric"], str(r["rule"])), []).append(r)
    for (pid, a, b, key, rule), rs in groups.items():
        if rs[0].get("missing"):
            print(f"   {pid}: {a} vs {b} ({key}): not run")
            continue
        r0 = rs[0]["rule"]
        what = (f"{a} <= {b}" if rule == "le" else f"{a} >= {b}" if rule == "ge"
                else f"{a} at most {100 * r0[1]:+.0f} % from {b}" if r0[0] == "change_le"
                else f"{a} {100 * r0[1]:+.1f} to {100 * r0[2]:+.1f} % from {b}" if r0[0] == "change_in"
                else f"{a} within {100 * r0[1]:.1f} % of {b}")
        line = "; ".join(f"n{r['n']} {r['reading']} {r['wins']}/{r['seeds']} ({100 * r['change']:+.1f} %)"
                         + ("" if r["ok"] or not registered else " FAIL") for r in rs)
        verdict = all(r["ok"] for r in rs if r["reading"] == "unseen")
        print(f"   {pid}: {what}, {key}" + (f": {'holds' if verdict else 'FAILS'} on the unseen frames" if registered else ""))
        print(f"      {line}")
    print()


def report_job2(job2a, job2b, job1, data_dir, json_path=None, verbose=True, job2c=None, job2d=None, job5=None):
    runs = job2_runs(job2a, job2b, job1, job2c, job2d, job5)
    if not runs:
        raise SystemExit(f"no results in {job2a}, {job2b}")
    L0 = train_size(runs)
    table = bulk_table(runs, L0)
    fixed = {arm: fixed_readings({k: v for k, v in runs.items() if k.endswith(f"[{arm}]")})
             for arm in ("A", "B", "C", "D", "D'")}
    liq = liquid_rows(runs)
    arm_c, arm_d, arm_dp = (any(k.endswith(f"[{x}]") for k in runs) for x in ("C", "D", "D'"))
    pred = job2_predictions(table, runs, JOB2_PREDICTIONS + (JOB2C_PREDICTIONS if arm_c else [])
                            + (JOB4_PREDICTIONS if arm_d else []) + (JOB5_PREDICTIONS if arm_dp else []))
    readings = job2_predictions(table, runs, JOB2_READINGS + (JOB2C_READINGS if arm_c else [])
                                + (JOB4_READINGS if arm_d else []) + (JOB5_READINGS if arm_dp else []))
    gains = field_term_gains(table, runs)
    frames_meta = load_frames(data_dir, sizes_of(runs)) if data_dir and os.path.isdir(data_dir) else {}
    job2_only = {k: v for k, v in table.items() if not k.endswith("[job 1]")}
    p3 = prediction_3(job2_only, frames_meta, runs, L0) if frames_meta else {}
    if verbose:
        print(f"job 2: {job2a}, {job2b}" + (f", {job2c}" if job2c else "") + (f", job 4's arm D {job2d}" if job2d else "")
              + (f", job 5's arm D' {job5}" if job5 else "")
              + f", against {job1}: "
              f"{sum(len(b) for b in runs.values())} runs; labels {sorted(runs)}\n")
        for arm, (rows, agree) in fixed.items():
            if rows:
                print_fixed(rows, agree, JOB2_PREREG, f"J0. Arm {arm}: the readings the matching fixes (pre-registration "
                                                       "section 4; the probes at 10-15 A, T5 and eps_inf must agree)")
        print_liquid(liq)
        print_job2_predictions(pred)
        print_job2_predictions([r for r in readings if not r.get("missing")],
                               "   Beside them, not registered: seeds at or below, and the change of the seed means",
                               registered=False)
        if gains:
            print("J4. The field term's gain on the rigid forces, 1 - F(with) / F(without), seed means, unseen frames: "
                  "on the C rungs (job 1, arm A, arm B: registered to fall in that order at every size) and on the dagger")
            for tag, by in gains.items():
                print(f"   {tag:7s}: " + "; ".join(f"n{n} {100 * g:+.1f} %" for n, g in sorted(by.items())))
            ok = all(gains.get(x, {}).get(n, -1) < gains.get(y, {}).get(n, 1) for x, y in (("A", "job 1"), ("B", "A"))
                     for n in sizes_of(runs) if n in gains.get(x, {}) and n in gains.get(y, {}))
            print(f"   J4 {'holds' if ok else 'FAILS'} on the runs present\n")
        if p3:
            print_prediction_3(p3, L0, max(sizes_of(runs)), "J5. As job 1's prediction 3")
        print_bulk(table, L0)
    strip = {name: {n: {rd: {s: {k: v for k, v in x.items() if k != "errors"} for s, x in vals.items()}
                        for rd, vals in by_r.items()} for n, by_r in by_n.items()} for name, by_n in table.items()}
    result = {"runs": {k: sorted(v) for k, v in runs.items()}, "fixed": {a: f[0] for a, f in fixed.items()},
              "fixed_agreement": {a: f[1] for a, f in fixed.items()}, "liquid": liq, "predictions": pred,
              "readings": readings, "field_term_gains": gains, "prediction_5": p3, "bulk": strip, "prereg": JOB2_PREREG}
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(jsonable(result), fh, indent=1)
    return result


# ---------------------------------------------------------------- job 3 (v0.0.27) -------------------------------------
# notes/e4-job3-preregistration.md section 4: job 3's readings, from scripts/e4_water.py kread's kread.json, each with its
# registered band.  Variants: A, B, C (job 2's arms), C0 (C with its orders >= 1 off), L (the locality test).
JOB3_BANDS = {
    "K0": [("MB-pol's eps_L(k1), every cell", ("truth_eps", "1"), 1.762, 1.782),
           ("MB-pol's eps_L(k2), every cell", ("truth_eps", "2"), 2.068, 2.108),
           ("MB-pol's chi(k1) / chi(0+) - 1", ("truth_rise", "1"), 0.05, 0.09),
           ("MB-pol's chi(k2) / chi(0+) - 1", ("truth_rise", "2"), 0.24, 0.32)],
    "K1": [("C: chi(k1) against MB-pol's", ("chi", "C", "1"), -0.035, -0.005),
           ("C: chi(k2) against MB-pol's", ("chi", "C", "2"), -0.15, -0.08)],
    "K2": [("C: molecules at k1", ("mol", "C", "1"), 0.02, 0.06), ("C: molecules at k2", ("mol", "C", "2"), 0.03, 0.07),
           ("C0's molecules at k1 less C's", ("mol_gap", "1"), 0.0, math.inf),
           ("C0's molecules at k2 less C's", ("mol_gap", "2"), 0.0, math.inf)],
    "K3": [("C0: chi(k2) against MB-pol's", ("chi", "C0", "2"), 0.015, 0.055),
           ("|C's chi(k2)| less |C0's|", ("chi_gap", "2"), 0.0, math.inf)],
    "K4": [("A: chi(k1)", ("chi", "A", "1"), 0.015, 0.045), ("A: chi(k2)", ("chi", "A", "2"), -0.085, -0.02),
           ("B: chi(k1)", ("chi", "B", "1"), 0.015, 0.045), ("B: chi(k2)", ("chi", "B", "2"), 0.015, 0.075)],
    "K5": [("L: held-out molecules' dipoles", ("liq", "molecular_dipole_rel_rmse"), 0.0, 0.03),
           ("L: held-out molecules' response", ("liq", "molecular_response_rel_rmse"), 0.0, 0.02),
           ("L: held-out iso response r", ("liq", "molecular_response_iso_corr"), 0.9, math.inf),
           ("L: dipoles, held out less calibration", ("liq_gap", "molecular_dipole_rel_rmse"), -0.01, 0.01),
           ("L: response, held out less calibration", ("liq_gap", "molecular_response_rel_rmse"), -0.01, 0.01)],
    "K6": [("L: far probes, non-additive, 10 A", ("probe", "10"), 0.0, 0.105),
           ("L: far probes, non-additive, 12 A", ("probe", "12"), 0.0, 0.060),
           ("L: far probes, non-additive, 15 A", ("probe", "15"), 0.0, 0.068),
           ("L: T5 slope", ("t5",), 0.95, 1.10)],
}


def job3_value(k, key, split="heldout"):
    """one of job 3's readings from kread.json (k): the bands' keys of JOB3_BANDS."""
    kind = key[0]
    h = lambda lab, n, sp=split: k[lab][f"k_{sp}"]["harmonics"][n]
    if kind in ("truth_eps", "truth_rise"):
        lab = next(x for x in ("C", "A", "B", "C0", "L") if x in k)
        recs = [(k[lab][f"k_{sp}"], sp) for sp in ("heldout", "calibration") if f"k_{sp}" in k[lab]]
        w = [r["frames"] for r, _ in recs]
        if kind == "truth_eps":
            vals = [r["harmonics"][key[1]]["eps_L_truth"] for r, _ in recs]
        else:
            vals = [r["harmonics"][key[1]]["chi_truth"] / r["chi_screened_k0"]["truth"] - 1 for r, _ in recs]
        return float(np.dot(w, vals) / np.sum(w))
    if kind == "chi":
        return h(key[1], key[2])["chi_rel_diff"]
    if kind == "mol":
        return h(key[1], key[2])["molecular_rel_rmse"]
    if kind == "mol_gap":
        return h("C0", key[1])["molecular_rel_rmse"] - h("C", key[1])["molecular_rel_rmse"]
    if kind == "chi_gap":
        return abs(h("C", key[1])["chi_rel_diff"]) - abs(h("C0", key[1])["chi_rel_diff"])
    if kind == "liq":
        return k["L"][f"liquid_{split}"][key[1]]
    if kind == "liq_gap":
        return k["L"]["liquid_heldout"][key[1]] - k["L"]["liquid_calibration"][key[1]]
    if kind == "probe":
        v = k["L"]["probes"]["by_distance_angstrom"][key[1]]
        return v["nonadditive_rmse"] / v["nonadditive_rms"]
    if kind == "t5":
        return k["L"]["trimers"]["slope"]
    raise ValueError(f"unknown job 3 reading {key}")


def report_job3(path, verbose=True):
    """job 3's registered readings (K0-K6) from kread.json, each against its band; a K holds when all its parts do."""
    with open(path) as fh:
        k = json.load(fh)
    out = {}
    if verbose:
        print(f"E4 job 3 (notes/e4-job3-preregistration.md section 4), from {path}: read-outs "
              + ", ".join(x for x in k if x != "meta"))
    for kid, parts in JOB3_BANDS.items():
        rows = []
        for what, key, lo, hi in parts:
            try:
                v = job3_value(k, key)
            except KeyError:
                v = None
            rows.append({"what": what, "value": v, "band": [lo, hi], "holds": v is not None and lo <= v <= hi})
        out[kid] = {"holds": all(r["holds"] for r in rows), "parts": rows}
        if verbose:
            print(f"  {kid}: {'holds' if out[kid]['holds'] else 'FAILS'}")
            for r in rows:
                band = f"[{r['band'][0]:g}, {r['band'][1]:g}]"
                val = "missing" if r["value"] is None else f"{r['value']:+.4f}"
                print(f"      {r['what']:44s} {val:>9s}  band {band:18s} {'ok' if r['holds'] else 'outside'}")
    return out


# notes/e4-job4-preregistration.md section 4: job 4's readings (arm D at matching level, its control Dx, and H3 at the
# 1.5 and 2.5 A mutual edges), from scripts/e4_water.py kread's jsons, each with its band (lo, hi).  Held-out cells
# unless said; the stage objectives from the matching records.
_E = (["C0@1.5", "C0", "C0@2.5"], ["D@1.5", "D", "D@2.5"])              # H3: the order-0 sectors and arm D by edge
JOB4_BANDS = {
    "D1": [("D: chi(k1) against MB-pol's", ("chi", "D", "1"), 0.008, 0.019),
           ("D: chi(k2) against MB-pol's", ("chi", "D", "2"), -0.010, 0.002)],
    "D2": [("D: chi(k1), held out less calibration", ("chi_gap", "D", "1"), -0.005, 0.005),
           ("D: chi(k2), held out less calibration", ("chi_gap", "D", "2"), -0.005, 0.005)],
    "D3": [("D's eps_inf less C's (the same by construction)", ("eps_same", "D", "C"), 0.0, 1e-6),
           ("D's cells' dipole at k = 0 less C's", ("shift", "cell", "D", "C"), -math.inf, 0.020)],
    "D4": [("D's molecules at k1 less C's", ("shift", "mol1", "D", "C"), -math.inf, 0.020),
           ("D's molecules at k2 less C's", ("shift", "mol2", "D", "C"), -math.inf, 0.035),
           ("C0's molecules at k2 less D's", ("shift", "mol2", "C0", "D"), 0.0, math.inf)],
    "D5": [("D's far probes at 10 A less C's", ("shift", "probe10", "D", "C"), -math.inf, 0.010),
           ("D's far probes at 12 A less C's", ("shift", "probe12", "D", "C"), -math.inf, 0.010),
           ("D's far probes at 15 A less C's", ("shift", "probe15", "D", "C"), -math.inf, 0.010),
           ("D: T5 slope", ("t5", "D"), 0.95, 1.10),
           ("D: the probes' objective, stage 3 over stage 2", ("stage_ratio", "D", 3, 2), 0.3, 1.0)],
    "D6": [("Dx's chi(k2) less C's", ("shift", "chi2", "Dx", "C"), 0.06, math.inf),
           ("Dx's frame-to-frame sd of chi(k2) over D's", ("sd_ratio", "Dx", "D", "2"), 3.0, math.inf),
           ("|D's chi(k2)| less |Dx's|", ("abs_gap", "D", "Dx", "2"), -math.inf, 0.0)],
    "D7": [("D@1.5: chi(k1)", ("chi", "D@1.5", "1"), -0.001, 0.011), ("D@1.5: chi(k2)", ("chi", "D@1.5", "2"), -0.007, 0.004),
           ("D@2.5: chi(k1)", ("chi", "D@2.5", "1"), 0.023, 0.043), ("D@2.5: chi(k2)", ("chi", "D@2.5", "2"), -0.020, 0.0),
           ("H3: chi(k1)'s edge range, order 0 over D", ("edge_fall", "chi1") + _E, 2.0, math.inf),
           ("H3: chi(k2)'s edge range, order 0 over D", ("edge_fall", "chi2") + _E, 2.0, math.inf),
           ("H3: the molecules at k1, the same", ("edge_fall", "mol1") + _E, 1.2, 2.5),
           ("H3: the molecules at k2, the same", ("edge_fall", "mol2") + _E, 1.8, 4.0)],
}


# arm D' (notes/e4-job5-preregistration.md section 4): job 5's matching level, read from its kread json with the change at
# contact (kread.contact, against arm C). Da is arm D' -- arm D's third stage anchored at arm C --; D and Dx are job 4's
# arm and control from their caches, read again beside it.
JOB5_BANDS = {
    "P1": [("Da: chi(k1) against MB-pol's", ("chi", "Da", "1"), 0.008, 0.020),
           ("Da: chi(k2) against MB-pol's", ("chi", "Da", "2"), -0.010, 0.003)],
    "P2": [("Da: chi(k1), held out less calibration", ("chi_gap", "Da", "1"), -0.005, 0.005),
           ("Da: chi(k2), held out less calibration", ("chi_gap", "Da", "2"), -0.005, 0.005)],
    "P3": [("Da's eps_inf less C's (the same by construction)", ("eps_same", "Da", "C"), 0.0, 1e-6),
           ("Da's cells' dipole at k = 0 less C's", ("shift", "cell", "Da", "C"), -math.inf, 0.010)],
    "P4": [("Da's molecules at k1 less C's", ("shift", "mol1", "Da", "C"), -math.inf, 0.015),
           ("Da's molecules at k2 less C's", ("shift", "mol2", "Da", "C"), -math.inf, 0.025),
           ("Da's molecules at k2 less D's", ("shift", "mol2", "Da", "D"), -math.inf, 0.0)],
    "P5": [("Da's far probes at 10 A less C's", ("shift", "probe10", "Da", "C"), -math.inf, 0.003),
           ("Da's far probes at 12 A less C's", ("shift", "probe12", "Da", "C"), -math.inf, 0.003),
           ("Da's far probes at 15 A less C's", ("shift", "probe15", "Da", "C"), -math.inf, 0.003),
           ("Da: T5 slope less C's", ("t5_shift", "Da", "C"), -0.005, 0.005),
           ("Da: the probes' objective, stage 3 over stage 2", ("stage_ratio", "Da", 3, 2), 0.95, 1.15)],
    "P6": [("Da: rigid forces at contact against C's (rms)", ("contact", "Da"), 0.0, 0.30),
           ("Da's change at contact over D's", ("contact_ratio", "Da", "D"), 0.0, 0.70),
           ("Da's orders >= 1 change over D's", ("coef_ratio", "Da", "D"), 0.0, 0.70)],
    "P7": [("D: chi(k1) as job 4's (+1.33 %)", ("chi", "D", "1"), 0.01325, 0.01335),
           ("D: chi(k2) as job 4's (-0.37 %)", ("chi", "D", "2"), -0.00375, -0.00365),
           ("D: rigid forces at contact against C's (rms)", ("contact", "D"), 0.30, 0.60),
           ("Dx: rigid forces at contact against C's (rms)", ("contact", "Dx"), 0.30, 0.70)],
}


def stage_objectives(match):
    """the far probes' part of the matching objective after each stage (1: the probes; 2: and the liquid rows; 3: the
    orders >= 1 on the probes and the finite-k rows), from a matching record: {stage: value}."""
    obj = match.get("objective") or []
    n2 = len(match.get("liquid_objective") or [])
    n3 = int(match.get("finite_k_stage_steps") or 0)
    n1 = len(obj) - n2 - n3
    out = {}
    if n1 > 0:
        out[1] = obj[n1 - 1][1]
    if n2 > 0:
        out[2] = obj[n1 + n2 - 1][1] - match["liquid_objective"][-1][1]
    if n3 > 0:
        fk = match.get("finite_k_objective") or []
        out[3] = obj[-1][1] - (fk[-1][1] if fk else 0.0)
    return out


def job4_reading(k, field, lab, split="heldout"):
    """one variant's reading by name: chi1, chi2 (chi against MB-pol's), mol1, mol2 (the molecules at finite k), cell
    (the cells' dipole at k = 0), probe10/12/15 (the far probes' non-additive energy), t5 (T5's slope)."""
    v = k[lab]
    if field in ("chi1", "chi2"):
        return v[f"k_{split}"]["harmonics"][field[-1]]["chi_rel_diff"]
    if field in ("mol1", "mol2"):
        return v[f"k_{split}"]["harmonics"][field[-1]]["molecular_rel_rmse"]
    if field == "cell":
        return v[f"liquid_{split}"]["cell_dipole_rel_rmse"]
    if field.startswith("probe"):
        p = v["probes"]["by_distance_angstrom"][field[5:]]
        return p["nonadditive_rmse"] / p["nonadditive_rms"]
    if field == "t5":
        return v["trimers"]["slope"]
    raise ValueError(f"unknown job 4 reading {field}")


def job4_value(k, key, split="heldout"):
    """one of job 4's readings from the merged kread jsons (k): the bands' keys of JOB4_BANDS."""
    kind = key[0]
    h = lambda lab, n, sp=split: k[lab][f"k_{sp}"]["harmonics"][n]
    if kind == "chi":
        return h(key[1], key[2])["chi_rel_diff"]
    if kind == "chi_gap":                                         # held out less calibration
        return h(key[1], key[2])["chi_rel_diff"] - h(key[1], key[2], "calibration")["chi_rel_diff"]
    if kind == "shift":                                           # reading key[1]: variant key[2] less key[3]
        return job4_reading(k, key[1], key[2], split) - job4_reading(k, key[1], key[3], split)
    if kind == "abs_gap":                                         # |chi| of key[1] less |chi| of key[2]
        return abs(h(key[1], key[3])["chi_rel_diff"]) - abs(h(key[2], key[3])["chi_rel_diff"])
    if kind == "sd_ratio":                                        # the frames' sd of chi, key[1] over key[2]
        sd = lambda lab: float(np.std(h(lab, key[3])["chi_rel_diff_frames"], ddof=1))
        return sd(key[1]) / sd(key[2])
    if kind == "eps_same":
        return abs(k[key[1]][f"liquid_{split}"]["eps_inf_model"] - k[key[2]][f"liquid_{split}"]["eps_inf_model"])
    if kind == "t5":
        return k[key[1]]["trimers"]["slope"]
    if kind == "t5_shift":                                        # v0.0.29: T5's slope, key[1] less key[2]
        return k[key[1]]["trimers"]["slope"] - k[key[2]]["trimers"]["slope"]
    if kind == "contact":                                         # v0.0.29: the change at contact against the reference
        return k[key[1]]["contact"]["against_reference"]["rigid_rms"]
    if kind == "contact_ratio":
        c = lambda lab: k[lab]["contact"]["against_reference"]["rigid_rms"]
        return c(key[1]) / c(key[2])
    if kind == "coef_ratio":                                      # the orders >= 1 change, key[1]'s over key[2]'s
        c = lambda lab: k[lab]["contact"]["coefficients"]["orders_above_zero"]
        return c(key[1]) / c(key[2])
    if kind == "stage_ratio":                                     # the probes' objective, stage key[2] over key[3]
        o = stage_objectives(k[key[1]]["match"])
        return o[key[2]] / o[key[3]]
    if kind == "edge_fall":                                       # H3: a reading's range over the edges, the order-0
        rng = lambda labs: (lambda v: max(v) - min(v))([job4_reading(k, key[1], lab, split) for lab in labs])
        r0, rd = rng(key[2]), rng(key[3])                         # sectors' over arm D's
        return r0 / rd if rd > 0 else math.inf
    raise ValueError(f"unknown job 4 reading {key}")


def load_kread(paths):
    """kread jsons merged by variant: a directory (its kread*.json) or comma-separated files; the first meta kept."""
    files = []
    for p in paths.split(","):
        files += sorted(glob.glob(os.path.join(p, "kread*.json"))) if os.path.isdir(p) else [p]
    if not files:
        raise FileNotFoundError(f"no kread json in {paths}")
    k = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        for lab, v in d.items():
            if lab == "meta":
                k.setdefault("meta", v)
            elif lab in k:
                raise ValueError(f"{lab} is read in more than one of {files}")
            else:
                k[lab] = v
    return k, files


def report_job4(paths, verbose=True, bands=None, job=4):
    """job 4's registered readings (D1-D7) from kread's jsons, each against its band; a D holds when all its parts do.
    job=5 (v0.0.29): job 5's (P1-P7, JOB5_BANDS)."""
    k, files = load_kread(paths)
    bands = (JOB4_BANDS if job == 4 else JOB5_BANDS) if bands is None else bands
    out = {}
    if verbose:
        print(f"E4 job {job} (notes/e4-job{job}-preregistration.md section 4), from {', '.join(files)}: read-outs "
              + ", ".join(x for x in k if x != "meta"))
    for kid, parts in bands.items():
        rows = []
        for what, key, lo, hi in parts:
            try:
                v = job4_value(k, key)
            except (KeyError, IndexError, ZeroDivisionError):
                v = None
            rows.append({"what": what, "value": v, "band": [lo, hi], "holds": v is not None and lo <= v <= hi})
        out[kid] = {"holds": all(r["holds"] for r in rows), "parts": rows}
        if verbose:
            print(f"  {kid}: {'holds' if out[kid]['holds'] else 'FAILS'}")
            for r in rows:
                band = f"[{r['band'][0]:g}, {r['band'][1]:g}]"
                val = "missing" if r["value"] is None else f"{r['value']:+.4f}"
                print(f"      {r['what']:48s} {val:>9s}  band {band:18s} {'ok' if r['holds'] else 'outside'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="E4's results read against the pre-registration")
    ap.add_argument("--out_dir", default="results/e4_job1")
    ap.add_argument("--data", default="data/e4", help="the data sets (for the truth's energies and the trajectories)")
    ap.add_argument("--json", default=None, help="also write every number to this json")
    ap.add_argument("--job2", default=None, help="job 2: its arms' out_dirs, 'results/e4_job2a,results/e4_job2b"
                                                  "[,results/e4_job2c[,results/e4_job4/edge2.0[,results/e4_job5]]]' "
                                                  "(the fourth: job 4's arm D, v0.0.28; the fifth: job 5's arm D', "
                                                  "v0.0.29)")
    ap.add_argument("--job1", default="results/e4_job1", help="job 2: job 1's out_dir, read beside it")
    ap.add_argument("--job3", default=None, help="job 3: scripts/e4_water.py kread's kread.json")
    ap.add_argument("--job4", default=None, help="job 4: kread's jsons -- a directory (its kread*.json) or a list")
    ap.add_argument("--job5", default=None, help="job 5: kread's jsons, as --job4 (v0.0.29)")
    a = ap.parse_args()
    if a.job3 or a.job4 or a.job5:
        out = (report_job3(a.job3) if a.job3 else report_job4(a.job4) if a.job4 else report_job4(a.job5, job=5))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(jsonable(out), fh, indent=1)
    elif a.job2:
        ja, jb, jc, jd, je = (a.job2.split(",") + [None, None, None, None])[:5]
        report_job2(ja, jb, a.job1, a.data, a.json, job2c=jc, job2d=jd, job5=je)
    else:
        report(a.out_dir, a.data, a.json)


if __name__ == "__main__":
    main()
