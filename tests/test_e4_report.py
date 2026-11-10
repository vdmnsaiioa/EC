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


# ---------------------------------------------------------------- job 2 -----------------------------------------------
def _rewrite(d, name, new_name, force, liquid=None):
    """a made-up run of job 2 from a job-1 one: renamed, its rigid forces set to force + 0.001 seed (net 1.5 times),
    with the liquid readings before and after training."""
    for seed in SEEDS:
        with open(os.path.join(d, f"{name}_seed{seed}.json")) as fh:
            r = json.load(fh)
        os.remove(os.path.join(d, f"{name}_seed{seed}.json"))
        r["rung"] = new_name
        for by in r["readings"]["cells"].values():
            for v in by.values():
                v["force_rigid_rmse"] = force + 0.001 * seed
                v["net_force_rmse"] = 1.5 * v["force_rigid_rmse"]
        if liquid is not None:
            r["matching"] = {"liquid_before_training": liquid, "liquid_objective": [[1e-4, 1e-5]]}
            r["readings"]["liquid"] = liquid
        with open(os.path.join(d, f"{new_name}_seed{seed}.json"), "w") as fh:
            json.dump(r, fh)


def test_job2_report(tmp_path):
    """scripts/e4_report.py --job2: both arms read against job 1 -- the registered comparisons with their rules, the
    readings beside them, the field term's gains in order, the liquid readings before and after training."""
    rng = np.random.default_rng(3)
    frames = truth_frames(rng)
    data = tmp_path / "data"; data.mkdir()
    write_data(str(data), frames)
    dirs = {}
    for tag in ("job1", "a", "b"):
        dirs[tag] = tmp_path / tag; dirs[tag].mkdir()
        write_runs(str(dirs[tag]), frames, rng)
    liq = {"frames": 10, "eps_inf_model": 1.743, "eps_inf_truth": 1.6875, "cell_dipole_rel_rmse": 0.09,
           "molecular_dipole_rel_rmse": 0.08, "molecular_response_rel_rmse": 0.09, "molecular_response_iso_rel_rmse": 0.087}
    a, b = str(dirs["a"]), str(dirs["b"])
    for name in ("M_inf", "M_G"):
        for d in (a, b):
            for seed in SEEDS:
                os.remove(os.path.join(d, f"{name}_seed{seed}.json"))
    _rewrite(a, "M_16C2pm_us", "M_16C2pm_us", 0.60, liq)        # J2: as job 1's
    _rewrite(a, "M_16C2pm", "M_16C2pm", 0.715, liq)              # J1a: -0.9 % against the dagger's 0.7215
    _rewrite(a, "M_16pm", "M_16pm_us", 0.62)
    _rewrite(b, "M_16C2pm_us", "M_16C2pm_us", 0.61, {**liq, "eps_inf_model": 1.689})   # J2 +1.7 %; J3 below 0.62
    _rewrite(b, "M_16C2pm", "M_16C2pm", 0.70, {**liq, "eps_inf_model": 1.689})         # J1: 4 of 4 below the dagger
    for seed in SEEDS:
        os.remove(os.path.join(b, f"M_16pm_seed{seed}.json"))
    # arm C (v0.0.26): the ablation 0.712 (below arm A's 0.715 on every seed), the frozen form 0.603 (+0.5 %)
    dirs["c"] = tmp_path / "c"; dirs["c"].mkdir()
    write_runs(str(dirs["c"]), frames, rng)
    c = str(dirs["c"])
    for name in ("M_inf", "M_G", "M_16pm"):
        for seed in SEEDS:
            os.remove(os.path.join(c, f"{name}_seed{seed}.json"))
    _rewrite(c, "M_16C2pm_us", "M_16C2pm_us", 0.603, {**liq, "eps_inf_model": 1.6895})
    _rewrite(c, "M_16C2pm", "M_16C2pm", 0.712, {**liq, "eps_inf_model": 1.6895})
    res = R.report_job2(a, b, str(dirs["job1"]), str(data), json_path=str(tmp_path / "j2.json"), verbose=True, job2c=c)
    verdict = lambda pid, a_, n=256: next(r for r in res["predictions"] if r["prediction"] == pid and r["a"] == a_
                                          and r["n"] == n and r["reading"] == "unseen")
    assert verdict("J1", "M_16C2pm [B]")["wins"] == 4 and verdict("J1", "M_16C2pm [B]")["ok"]
    assert verdict("J1a", "M_16C2pm [A]")["ok"] and verdict("J1a", "M_16C2pm [A]")["change"] == pytest.approx(0.7165 / 0.7215 - 1)
    assert verdict("J2", "M_16C2pm_us [A]")["ok"] and verdict("J2", "M_16C2pm_us [B]")["ok"]
    assert verdict("J3", "M_16C2pm_us [B]")["ok"]
    g = res["field_term_gains"]
    assert g["job 1"][64] > g["A"][64] > g["B"][64]
    assert g["dagger"][64] == pytest.approx(1 - 0.6215 / 0.7215)
    assert res["liquid"]["M_16C2pm_us [B]"][0]["after"]["eps_inf_model"] == 1.689
    assert any(r["a"] == "M_16pm_us [A]" for r in res["readings"])
    assert "M_16C2pm_us [job 1]" in res["runs"] and "M_16pm_us [B]" not in res["runs"]
    assert verdict("C1", "M_16C2pm [C]")["ok"] and verdict("C2", "M_16C2pm [C]")["wins"] == 4
    assert verdict("C3", "M_16C2pm [C]")["ok"] and verdict("C4", "M_16C2pm_us [C]")["ok"]
    assert g["C"][64] == pytest.approx(1 - 0.6045 / 0.7135)
    assert res["fixed"]["C"]["M_16C2pm [C]"][0]["kind"] == "response"


def test_job3_report(tmp_path):
    """job 3's registered readings (v0.0.27) on the local reading committed with the pre-registration
    (scratch/e4_job3_local.json): every K holds there, the bands having been set around it; arm C closer to MB-pol at
    k2 fails K1 and K3; without the locality test's read-out K5 and K6 fail on missing readings instead of breaking."""
    import copy
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "e4_job3_local.json")
    out = R.report_job3(path, verbose=False)
    assert set(out) == {f"K{i}" for i in range(7)} and all(v["holds"] for v in out.values())
    k = json.load(open(path))
    moved = copy.deepcopy(k)
    moved["C"]["k_heldout"]["harmonics"]["2"]["chi_rel_diff"] = -0.02
    no_l = {x: v for x, v in copy.deepcopy(k).items() if x != "L"}
    for name, d in (("moved.json", moved), ("no_l.json", no_l)):
        with open(tmp_path / name, "w") as fh:
            json.dump(d, fh)
    m = R.report_job3(str(tmp_path / "moved.json"), verbose=False)
    assert not m["K1"]["holds"] and not m["K3"]["holds"] and m["K0"]["holds"] and m["K6"]["holds"]
    n = R.report_job3(str(tmp_path / "no_l.json"), verbose=True)
    assert not n["K5"]["holds"] and not n["K6"]["holds"] and n["K1"]["holds"]
    assert all(p["value"] is None for p in n["K6"]["parts"])


def test_job4_report(tmp_path):
    """job 4's registered readings (v0.0.28) on the local reading committed with the pre-registration
    (scratch/e4_job4_local.json): every D holds there; split by edge into three files in a directory, as the tasks write
    them, it reads the same; arm D at C's chi fails D1 and D6; a variant read twice is refused; without the H3 arms D7
    fails on missing readings instead of breaking."""
    import copy
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "e4_job4_local.json")
    out = R.report_job4(path, verbose=False)
    assert set(out) == {f"D{i}" for i in range(1, 8)} and all(v["holds"] for v in out.values())
    k = json.load(open(path))
    for edge, labels in (("2.0", ["C", "C0", "D", "Dx"]), ("1.5", ["C@1.5", "C0@1.5", "D@1.5"]),
                         ("2.5", ["C@2.5", "C0@2.5", "D@2.5"])):
        with open(tmp_path / f"kread_edge{edge}.json", "w") as fh:
            json.dump({"meta": k["meta"], **{x: k[x] for x in labels}}, fh)
    split = R.report_job4(str(tmp_path), verbose=False)
    assert all(split[d]["parts"][i]["value"] == out[d]["parts"][i]["value"] for d in out for i in range(len(out[d]["parts"])))
    moved = copy.deepcopy(k)
    moved["D"]["k_heldout"] = copy.deepcopy(k["C"]["k_heldout"])
    with open(tmp_path / "moved.json", "w") as fh:
        json.dump(moved, fh)
    m = R.report_job4(str(tmp_path / "moved.json"), verbose=False)
    assert not m["D1"]["holds"] and not m["D6"]["holds"] and m["D5"]["holds"]
    with pytest.raises(ValueError):
        R.report_job4(f"{tmp_path / 'kread_edge2.0.json'},{tmp_path / 'moved.json'}", verbose=False)
    n = R.report_job4(str(tmp_path / "kread_edge2.0.json"), verbose=True)
    assert not n["D7"]["holds"] and n["D1"]["holds"] and all(p["value"] is None for p in n["D7"]["parts"])


def test_job5_report(tmp_path):
    """job 5's registered readings (v0.0.29; scripts/e4_report.py --job5) on the local reading committed with the
    pre-registration (scratch/e4_job5_local.json): P1-P6 hold there; P7's chi parts are job 4's Niflheim values, which
    the local frames do not give (job 4's local +1.40 / -0.46 % they reproduce, read with those values); Da at C's
    chi fails P1; Da with D's change at contact fails P6; without the contact reading P6 fails on missing readings."""
    import copy
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "e4_job5_local.json")
    out = R.report_job4(path, verbose=True, job=5)
    assert set(out) == {f"P{i}" for i in range(1, 8)} and all(out[f"P{i}"]["holds"] for i in range(1, 7))
    p7 = out["P7"]["parts"]
    assert [x["holds"] for x in p7] == [False, False, True, True]
    local = {**R.JOB5_BANDS, "P7": [(w, key, lo, hi) for (w, key, lo, hi) in R.JOB5_BANDS["P7"][2:]]
             + [("D: chi(k1) as job 4's local (+1.40 %)", ("chi", "D", "1"), 0.01395, 0.01405),
                ("D: chi(k2) as job 4's local (-0.46 %)", ("chi", "D", "2"), -0.00465, -0.00455)]}
    assert R.report_job4(path, verbose=False, bands=local)["P7"]["holds"]
    k = json.load(open(path))
    moved = copy.deepcopy(k)
    moved["Da"]["k_heldout"] = copy.deepcopy(k["C"]["k_heldout"])
    moved["Da"]["contact"] = copy.deepcopy(k["D"]["contact"])
    with open(tmp_path / "moved.json", "w") as fh:
        json.dump(moved, fh)
    m = R.report_job4(str(tmp_path / "moved.json"), verbose=False, job=5)
    assert not m["P1"]["holds"] and not m["P6"]["holds"] and m["P3"]["holds"] and m["P5"]["holds"]
    bare = copy.deepcopy(k)
    for lab in ("C", "D", "Dx", "Da"):
        bare[lab].pop("contact")
    with open(tmp_path / "bare.json", "w") as fh:
        json.dump(bare, fh)
    n = R.report_job4(str(tmp_path / "bare.json"), verbose=False, job=5)
    assert not n["P6"]["holds"] and all(x["value"] is None for x in n["P6"]["parts"]) and n["P1"]["holds"]


def test_job4_training_report(tmp_path):
    """scripts/e4_report.py --job2 with job 4's arm D as a fourth directory (v0.0.28): T1 (the frozen form within 1.5 %
    of arm C's) and T2 (the ablation +0.5 to +4.5 % above, on >= 3 of 4 seeds) with their rules, arm D's field-term gain,
    and the finite-k response before training in the liquid rows; a frozen form 2 % worse fails T1."""
    rng = np.random.default_rng(5)
    frames = truth_frames(rng)
    data = tmp_path / "data"; data.mkdir()
    write_data(str(data), frames)
    dirs = {}
    for tag in ("job1", "a", "b", "c", "d"):
        dirs[tag] = tmp_path / tag; dirs[tag].mkdir()
        write_runs(str(dirs[tag]), frames, rng)
        if tag != "job1":
            for name in ("M_inf", "M_G", "M_16pm"):
                for seed in SEEDS:
                    os.remove(os.path.join(dirs[tag], f"{name}_seed{seed}.json"))
    liq = {"frames": 10, "eps_inf_model": 1.6893, "eps_inf_truth": 1.6885, "cell_dipole_rel_rmse": 0.015,
           "molecular_dipole_rel_rmse": 0.051, "molecular_response_rel_rmse": 0.029, "molecular_response_iso_rel_rmse": 0.015}
    for tag, f_us, f_ab in (("a", 0.60, 0.72), ("b", 0.61, 0.74), ("c", 0.59, 0.71), ("d", 0.595, 0.725)):
        _rewrite(str(dirs[tag]), "M_16C2pm_us", "M_16C2pm_us", f_us, liq)
        _rewrite(str(dirs[tag]), "M_16C2pm", "M_16C2pm", f_ab, liq)
    kb = {"frames": 10, "harmonics": {"1": {"chi_rel_diff": 0.0133, "molecular_rel_rmse": 0.046},
                                      "2": {"chi_rel_diff": -0.0037, "molecular_rel_rmse": 0.073}}}
    for name in ("M_16C2pm_us", "M_16C2pm"):
        for seed in SEEDS:
            path = os.path.join(dirs["d"], f"{name}_seed{seed}.json")
            r = json.load(open(path))
            r["matching"]["k_before_training"] = kb
            json.dump(r, open(path, "w"))
    args = [str(dirs[t]) for t in "abc"]
    res = R.report_job2(*args[:2], str(dirs["job1"]), str(data), verbose=True, job2c=args[2], job2d=str(dirs["d"]))
    pick = lambda pid, rule, n=256: next(r for r in res["predictions"] if r["prediction"] == pid and str(r["rule"]) == rule
                                         and r["n"] == n and r["reading"] == "unseen")
    t1 = pick("T1", str(("abs_change_le", 0.015)))
    assert t1["ok"] and t1["change"] == pytest.approx(0.5965 / 0.5915 - 1)
    t2 = pick("T2", str(("change_in", 0.005, 0.045)))
    assert t2["ok"] and t2["change"] == pytest.approx(0.7265 / 0.7115 - 1) and pick("T2", "ge")["ok"]
    assert res["field_term_gains"]["D"][64] == pytest.approx(1 - 0.5965 / 0.7265)
    assert res["liquid"]["M_16C2pm_us [D]"][0]["k_before"]["harmonics"]["2"]["chi_rel_diff"] == -0.0037
    assert "M_16C2pm [D]" in res["runs"] and res["fixed"]["D"]["M_16C2pm_us [D]"][0]["kind"] == "response"
    _rewrite(str(dirs["d"]), "M_16C2pm_us", "M_16C2pm_us", 0.6035, liq)                  # +2 % on arm C's frozen form
    res2 = R.report_job2(*args[:2], str(dirs["job1"]), str(data), verbose=False, job2c=args[2], job2d=str(dirs["d"]))
    assert not next(r for r in res2["predictions"] if r["prediction"] == "T1" and r["n"] == 256
                    and r["reading"] == "unseen")["ok"]
