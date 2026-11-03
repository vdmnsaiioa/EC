"""scripts/e4_report.py: job 1's report read on made-up results whose answers are known."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import e4_report as R  # noqa: E402

FR = {64: {"train": 12, "valid": 6, "test": 6}, 128: {"train": 9, "valid": 4, "test": 4},
      256: {"train": 6, "valid": 3, "test": 3}}
SEEDS = (0, 1, 2, 3)
MODELS = {"M_16C2pm_us": {"charges": "pinned", "band_fields": 2}, "M_16C2pm": {"charges": "pinned", "band_fields": 2},
          "M_16pm": {"charges": "pinned", "band_fields": None}, "M_inf": {"charges": "off", "band_fields": None},
          "M_G": {"charges": "off", "band_fields": None}}


def truth_frames(rng):
    """the truth's energy per molecule (kcal/mol) by split and size, with sources and times."""
    out = {}
    for n, by in FR.items():
        for split, k in by.items():
            if split == "train":
                src = np.repeat([0, 1, 2], k // 3)
            else:
                src = np.full(k, 3)
            u = -9.8 + 0.01 * (n // 64) + 0.05 * rng.standard_normal(k)
            out[(split, n)] = {"u": u, "source": src, "time_ps": np.arange(k) * 0.5}
    return out


def write_data(d, frames):
    for (split, n), v in frames.items():
        np.savez(os.path.join(d, f"{split}_n{n}.npz"), energy=v["u"] * n / R.KCAL, source=v["source"],
                 time_ps=v["time_ps"])


def errors(name, seed, split, n, u, src, rng):
    """the per-frame errors of each made-up rung: M_inf's bias moves by -0.02 from 64 to 256; the ablation's error
    follows the truth with slope 0.05 inside each trajectory; the rest scatter by 1e-3 about zero."""
    e = 1e-3 * rng.standard_normal(len(u))
    if name == "M_inf" and n > 64:
        e = e - 0.01 * math.log2(n // 64)
    if name == "M_16C2pm":
        uc = u.copy()
        for t in np.unique(src):
            uc[src == t] -= uc[src == t].mean()
        e = e + 0.05 * uc
    if name == "M_G":
        e = e - 0.1 * (n / 64) ** 2
    return e


def force_of(name, seed):
    base = {"M_16C2pm_us": 0.60, "M_16C2pm": 0.75, "M_16pm": 0.72, "M_inf": 0.90, "M_G": 3.0}[name]
    if name == "M_16C2pm" and seed in (2, 3):
        base = 0.70                                            # below the dagger on 2 of 4 seeds: prediction 1 fails
    return base + 0.001 * seed


def write_runs(d, frames, rng):
    for name, model in MODELS.items():
        for seed in SEEDS:
            cells = {}
            for n, by in FR.items():
                for split in by:
                    m = frames[(split, n)]
                    e = errors(name, seed, split, n, m["u"], m["source"], rng)
                    f = force_of(name, seed)
                    v = {"frames": len(e), "n_mol": n, "energy_per_molecule_errors": e.tolist(),
                         "energy_per_molecule_rmse": float(np.sqrt(np.mean(e ** 2))),
                         "energy_per_molecule_mean_error": float(np.mean(e)), "force_rigid_rmse": f,
                         "net_force_rmse": 1.5 * f, "torque_rmse": 0.5 * f, "force_atomic_rmse": 10 * f}
                    if model["band_fields"] is not None and split == "test" and n == 64:
                        v["eps_inf"] = {"frames": 3, "isotropic": [1.80, 1.80, 1.80], "isotropic_mean": 1.80}
                    cells.setdefault(str(n), {})[split] = v
            if model["band_fields"] is not None:
                nadd = {"8": 0.03 + 0.001 * seed, "10": 0.025, "12": 0.026, "15": 0.02}
                inter, t5 = {"8": 0.014, "10": 0.005, "12": 0.007, "15": 0.005}, (0.037, 1.026)
            else:
                nadd = {d_: 1.0 for d_ in ("8", "10", "12", "15")}
                inter, t5 = {d_: 0.2 for d_ in ("8", "10", "12", "15")}, (1.0, 0.0)
            probes = {"by_distance_angstrom": {d_: {"count": 20, "nonadditive_rmse": nadd[d_], "nonadditive_rms": 1.0,
                                                    "interaction_rmse": inter[d_], "interaction_rms": 1.0}
                                               for d_ in nadd}}
            r = {"rung": name, "seed": seed, "model": dict(model, polarisabilities="pinned"),
                 "config": {"data": {"train_size": 64}}, "probes_hash": "72b73e3aa6c7b53f",
                 "matching": None, "training": {"time": 3600.0 * (1 + seed), "history": [[0, 1e3, 1, 1], [9, 1.0, 1, 1]]},
                 "readings": {"cells": cells, "probes": probes,
                              "far_trimers": {"three_body_rmse": t5[0], "three_body_rms": 1.0, "slope": t5[1]}}}
            with open(os.path.join(d, f"{name}_seed{seed}.json"), "w") as fh:
                json.dump(r, fh)


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("e4_report")
    data, out = root / "data", root / "results"
    data.mkdir(); out.mkdir()
    frames = truth_frames(rng)
    write_data(str(data), frames)
    write_runs(str(out), frames, rng)
    res = R.report(str(out), str(data), json_path=str(root / "report.json"), verbose=True)
    return res, frames, root


def test_fixed_readings_agree(made):
    res, _, _ = made
    a = res["fixed_agreement"]["response"]
    assert a["eps_inf"]["spread"] == 0 and a["t5_slope"]["spread"] == 0
    assert all(a[f"nonadditive_pct_{d}"]["spread"] == 0 for d in R.FIXED_DISTANCES)
    assert res["fixed"]["M_16C2pm_us"][1]["nonadditive_pct"]["8"] == pytest.approx(3.1)   # 8 A is not fixed
    assert res["fixed"]["M_16pm"][0]["t5_slope"] == 0.0


def test_pooled_unseen(made):
    res, _, root = made
    with open(root / "results" / "M_inf_seed0.json") as fh:
        r = json.load(fh)
    e = np.concatenate([r["readings"]["cells"]["64"][s]["energy_per_molecule_errors"] for s in ("valid", "test")])
    v = res["bulk"]["M_inf"][64]["unseen"][0]
    assert v["frames"] == 12 and v["energy_rmse"] == pytest.approx(np.sqrt(np.mean(e ** 2)))
    e = np.concatenate([r["readings"]["cells"]["256"][s]["energy_per_molecule_errors"] for s in R.SPLITS])
    assert res["bulk"]["M_inf"][256]["unseen"][0]["bias"] == pytest.approx(np.mean(e))


def test_predictions_1_2(made):
    res, _, _ = made
    rows = res["predictions_1_2"]
    pick = lambda a, b, n, rd, m: next(r for r in rows if (r["a"], r["b"], r["n"], r["reading"], r["metric"]) == (a, b, n, rd, m))
    assert pick("M_16C2pm_us", "M_16pm", 128, "test", "F")["wins"] == 4
    p = pick("M_16C2pm", "M_16pm", 256, "unseen", "F")
    assert p["wins"] == 2 and not p["pass"]
    assert pick("M_16C2pm_us", "M_16C2pm", 64, "test", "F")["pass"]


def test_prediction_3(made):
    res, _, _ = made
    p3 = res["prediction_3"]
    assert p3["M_inf"]["unseen"]["shift"] == pytest.approx(-0.02, abs=2e-3) and not p3["M_inf"]["unseen"]["holds"]
    assert p3["M_inf"]["unseen"]["mid"][128]["bias"] == pytest.approx(-0.01, abs=2e-3)
    assert p3["M_16C2pm_us"]["test"]["holds"] and abs(p3["M_16C2pm_us"]["test"]["z"]) < 3
    assert p3["M_inf"]["unseen"]["seeds_beyond_2se"] == 4


def test_truth_and_slope(made):
    res, frames, _ = made
    u256 = np.concatenate([frames[(s, 256)]["u"] for s in R.SPLITS])
    src = np.concatenate([frames[(s, 256)]["source"] for s in R.SPLITS])
    means = [u256[src == t].mean() for t in range(4)]
    assert res["truth"][256]["mean"] == pytest.approx(np.mean(means))
    assert res["error_vs_truth"]["M_16C2pm"][256]["slope"] == pytest.approx(0.05, abs=0.01)
    assert abs(res["error_vs_truth"]["M_16C2pm_us"][256]["slope"]) < 0.02


def test_json_written(made):
    _, _, root = made
    with open(root / "report.json") as fh:
        j = json.load(fh)
    assert j["train_size"] == 64 and j["sizes"] == [64, 128, 256] and "prediction_3" in j
