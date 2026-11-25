"""
E4 job 14 (v0.0.38; notes/e4-job14-preregistration.md): E_0's message-passing layers in the config (model.layers), the
four configs at reaches of 4.5 and 6 A, and the readings K14.1-K14.4 (scripts/e4_job14.py).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, water as W
from sseft.units import ang_to_bohr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job14 as J14                                    # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402
from test_job7 import _run                                      # noqa: E402
from test_job10 import _train                                   # noqa: E402


def test_e0_layers_are_configured_and_recorded(tmp_path):
    """model.layers sets E_0's message-passing layers: one layer of 4.5 A builds one block and is recorded with its
    cutoff; the default (two) builds the rung exactly as before, with the record as before (no "layers")."""
    _fake_cells(tmp_path / "c.npz", 1, 16, 12.5, seed=3)
    st = W.e4_structures(str(tmp_path / "c.npz"), pins=True)
    cfg = E4.load_config(None)
    assert cfg["model"]["layers"] == 2
    two, _ = E4.rungs_for("M_1q070", cfg, st[0].cell)
    old = E4.E3.configure(E4.E3.RUNGS["M_1q070"], SimpleNamespace(**{k: v for k, v in vars(E4.e3_args(cfg)).items()
                                                                       if k != "layers"}))
    assert two.e0 == {**old.e0} and two.e0["T"] == 2
    rec2 = E4.model_record("M_1q070", cfg, two)
    assert "layers" not in rec2 and rec2["r_cut_angstrom"] == 6.0
    cfg["model"].update(r_cut_angstrom=4.5, layers=1)
    one, _ = E4.rungs_for("M_1q070", cfg, st[0].cell)
    assert one.e0["T"] == 1 and one.e0["r_cut"] == pytest.approx(ang_to_bohr(4.5))
    rec1 = E4.model_record("M_1q070", cfg, one)
    assert rec1["layers"] == 1 and rec1["r_cut_angstrom"] == 4.5 and rec1 != rec2
    p1, p2 = (M.init_params(jax.random.PRNGKey(0), r, energy_scale=1e-3) for r in (one, two))
    assert len(p1["e0"]["blocks"]) == 1 and len(p2["e0"]["blocks"]) == 2
    from sseft import structure as S
    b = {k: v[0] for k, v in S.pad_batch(st, n_freq=one.K).items()}
    assert np.isfinite(float(M.energy_single(p1, E4.in_cell(one, st[0].cell), b)[0]))


def test_driver_trains_and_polishes_at_a_shorter_reach(tmp_path):
    """the driver at one layer of 4.5 A: the json records the reach; its polish starts from the first stage, and a start
    trained at the default reach is refused as another model."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["seeds"], cfg["rungs"] = [0], ["M_1w150"]
    cfg["model"].update(features=8)
    cfg["training"]["adam"].update(steps=2, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    base = tmp_path / "base"
    _train(tmp_path, cfg, base)                                     # the default reach
    cfg["model"].update(r_cut_angstrom=4.5, layers=1)
    stage = tmp_path / "stage"
    _train(tmp_path, cfg, stage)
    r = json.load(open(stage / "M_1w150_seed0.json"))
    assert r["model"]["layers"] == 1 and r["model"]["r_cut_angstrom"] == 4.5 and r["config"]["model"]["layers"] == 1
    cfg["training"]["adam"]["steps"] = 0
    cfg["training"]["lbfgs"]["steps"] = 1
    cfg["training"]["start_from"] = {"dir": str(stage)}
    _train(tmp_path, cfg, tmp_path / "polish")
    rp = json.load(open(tmp_path / "polish" / "M_1w150_seed0.json"))
    assert rp["model"] == r["model"] and rp["training"]["start"]["from"] == str(stage / "M_1w150_seed0.params.pkl")
    cfg["training"]["start_from"] = {"dir": str(base)}
    with pytest.raises(ValueError, match="another model"):
        _train(tmp_path, cfg, tmp_path / "again")


def test_job14s_configs_agree():
    """the first stages are job 9's settings with E_0 of one layer at 4.5 and 6 A, job 7's scan and warm starts; the
    polishes are job 13's settings from them."""
    j9 = E4.load_config(os.path.join(REPO, "configs", "e4_job9.yaml"))
    j13 = E4.load_config(os.path.join(REPO, "configs", "e4_job13.yaml"))
    rungs = ["M_1q070", "M_1q100", "M_1q110", "M_1q145", "M_1w050", "M_1w150"]
    for tag, rc in (("r45", 4.5), ("r6", 6.0)):
        s = E4.load_config(os.path.join(REPO, "configs", f"e4_job14s_{tag}.yaml"))
        p = E4.load_config(os.path.join(REPO, "configs", f"e4_job14_{tag}.yaml"))
        for c in (s, p):
            assert c["rungs"] == rungs and c["seeds"] == [0, 1]
            assert c["model"] == {**j9["model"], "r_cut_angstrom": rc, "layers": 1}
            for key in ("data", "probes", "reading"):
                assert c[key] == j9[key]
        for key in ("adam", "lbfgs", "loss", "readout"):
            assert s["training"][key] == j9["training"][key]
        assert s["training"]["trace"] == {"every": 100, "frames": 4, "lbfgs_every": 25}
        for key in ("adam", "lbfgs", "loss", "readout", "trace"):
            assert p["training"][key] == j13["training"][key]
        assert p["training"]["start_from"] == {"dir": f"results/e4_job14s_{tag}"}
        stage, pol = J14.REACHES["4.5" if tag == "r45" else "6"]
        assert stage == f"results/e4_job14s_{tag}" and pol == f"results/e4_job14_{tag}"
    assert set(J14.SCAN) == set(rungs[:4]) and set(J14.WARM) == set(rungs[4:])


def _parabola(a, c0, L0):
    return {c: L0 + a * (c - c0) ** 2 for c in (0.7, 1.0, 1.1, 1.45)}


def _job14(a45=1000.0, c45=1.05, a6=300.0, c6=1.06, ends45=((1.03, 1.07), (1.04, 1.08)),
           ends6=((0.95, 1.15), (0.97, 1.12)), missing=()):
    """job 14's runs and its references on constructed jsons; ends: (from 0.5, from 1.5) by seed, in c."""
    sizes = J14.SIZES
    add = lambda d, r: d.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    job9, job10, job11 = {}, {}, {}
    for s in J14.SEEDS:
        for name, L in (("M_1q070", 30.75), ("M_1q100", 23.63), ("M_1q145", 31.44)):
            add(job11, _run(name, s, {n: 0.6 for n in sizes}, loss=L))
        add(job10, _run("M_1q110", s, {n: 0.52 for n in sizes}, loss=(21.09, 21.82)[s]))
        for w, d in (("M_1w050", (1.504, 1.648)), ("M_1w150", (2.108, 2.184))):
            add(job10, _run(w, s, {n: 0.55 for n in sizes}, dipole=d[s], loss=25.0))
            add(job9, _run(w, s, {n: 0.78 for n in sizes}, dipole=(1.214, 1.315, 2.461, 2.563)[2 * (w == "M_1w150") + s]))
    reach_runs = {}
    for reach, (a, c0, ends) in (("4.5", (a45, c45, ends45)), ("6", (a6, c6, ends6))):
        if reach in missing:
            continue
        stage, runs = {}, {}
        for s in J14.SEEDS:
            for (name, c), L in zip(J14.SCAN.items(), _parabola(a, c0, 40.0).values()):
                add(runs, _run(name, s, {n: 0.8 for n in sizes}, loss=L))
            for k, w in enumerate(J14.WARM):
                c = ends[s][k]
                r = _run(w, s, {n: 0.78 for n in sizes}, dipole=c * J14.PERMANENT_DEBYE, loss=45.0)
                d = 0.5e-4 * J14.WALL * c ** 4
                r["training"]["charges"] = {"scale_curve": {"final": {"s": [0.99, 1.0, 1.01], "loss": [45 + d, 45.0, 45 + d]}}}
                add(runs, r)
                add(stage, _run(w, s, {n: 0.9 for n in sizes}, dipole=(c - 0.1) * J14.PERMANENT_DEBYE))
        reach_runs[reach] = (stage, runs)
    return reach_runs, job9, job10, job11


def test_the_job14_readings():
    """K14.1-K14.4 on constructed jsons, then each band's edge, c* outside the scan and a missing reach."""
    ev = J14.evaluate(*_job14())
    ref = ev["reference"]
    assert ref["c_min"] == pytest.approx(1.0704, abs=1e-3) and ref["curvature"] == pytest.approx(142.8, rel=2e-3)
    p45 = ev["reaches"]["4.5"]["profile"]
    assert p45["c_min"] == pytest.approx(1.05) and p45["curvature"] == pytest.approx(1.05 ** 2 * 2000.0)
    assert ev["K14.1"]["holds"] and ev["K14.1"]["ratio"] == pytest.approx(1.05 ** 2 * 2000.0 / ref["curvature"])
    assert ev["K14.2"]["holds"] and ev["K14.3"]["holds"] and ev["K14.4"]["holds"]
    assert ev["K14.3"]["off"]["M_1w050 seed 0"] == pytest.approx(1.03 - 1.05)
    assert ev["K14.4"]["gaps"]["12"][0] == pytest.approx((2.108 - 1.504) / J14.PERMANENT_DEBYE)
    assert ev["reaches"]["4.5"]["ends"]["M_1w150 seed 1"]["wall"] == pytest.approx(1.0)
    assert ev["reaches"]["4.5"]["ends"]["M_1w050 seed 0"]["stage_c"] == pytest.approx(0.93)
    text = J14.report(ev)
    for k in ("K14.1: holds", "K14.2: holds", "K14.3: holds", "K14.4: holds", "12 A (jobs 10-11)"):
        assert k in text
    assert J14.evaluate(*_job14(a45=300.0))["K14.1"]["holds"] is False                # 4.7 times
    assert J14.evaluate(*_job14(a6=110.0))["K14.2"]["holds"] is False                 # 1.7 times
    assert J14.evaluate(*_job14(ends45=((0.99, 1.07), (1.04, 1.08))))["K14.3"]["holds"] is False
    ev = J14.evaluate(*_job14(c45=1.6, a45=60.0))                                    # c* beyond the scan
    assert ev["K14.3"]["holds"] is None and ev["K14.3"]["outside"] and "outside the scan" in J14.report(ev)
    assert J14.evaluate(*_job14(ends45=((1.0, 1.21), (1.04, 1.08))))["K14.4"]["holds"] is False   # 0.21 > 0.20
    ev = J14.evaluate(*_job14(missing=("6",)))
    assert ev["K14.2"]["holds"] is None and ev["K14.4"]["holds"] is None and ev["K14.1"]["holds"]
