"""
E4 job 11 (v0.0.35; notes/e4-job11-preregistration.md): the L-BFGS polish's wall-clock budget (train.fit's
lbfgs_max_seconds, training.lbfgs.max_minutes), the start by rung (training.start_from.by_rung), the objective traced
through the polish for any rung, and the readings K11.1-K11.5 and the model rule (scripts/e4_job11.py).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import sseft  # noqa: F401  (x64)
from sseft import train as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job11 as J11                                    # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402
from test_job7 import _run, _runs                              # noqa: E402
from test_job9 import _cells, _rung                            # noqa: E402
from test_job10 import KW, _train                              # noqa: E402


def test_the_polish_stops_when_its_budget_is_spent(tmp_path):
    """lbfgs_max_seconds: the polish stops after the step that spends its budget; the options record the steps it took
    and the trace ends there; a budget must be positive."""
    st = _cells(tmp_path)
    rung = _rung("M_1w150", st)
    p0, i0 = T.fit(rung, st, steps=2, lbfgs_steps=0, **KW)
    p1, i1 = T.fit(rung, st, steps=0, lbfgs_steps=5, params_init=p0, lbfgs_max_seconds=1e-9, scale=i0["scale"],
                   trace_fn=lambda p: {}, lbfgs_trace_every=1, **KW)
    assert i1["training_options"]["lbfgs_steps_done"] == 1 and i1["training_options"]["lbfgs_max_seconds"] == 1e-9
    assert [s for s, _, _ in i1["lbfgs_trace"]] == [0, 1]
    _, i2 = T.fit(rung, st, steps=0, lbfgs_steps=2, params_init=p0, lbfgs_max_seconds=3600.0, scale=i0["scale"], **KW)
    assert i2["training_options"]["lbfgs_steps_done"] == 2
    _, i3 = T.fit(rung, st, steps=0, lbfgs_steps=2, params_init=p0, scale=i0["scale"], **KW)
    assert "lbfgs_steps_done" not in i3["training_options"]
    with pytest.raises(ValueError, match="lbfgs_max_seconds"):
        T.fit(rung, st, steps=0, lbfgs_steps=1, params_init=p0, lbfgs_max_seconds=0.0, **KW)


def test_driver_starts_by_rung_and_traces_the_objective_of_any_rung(tmp_path):
    """training.start_from.by_rung: a rung starts from its own directory, the others from dir; with lbfgs_every the
    objective is traced through the polish for a pinned rung too (training.lbfgs_trace), and max_minutes reaches the
    polish (the options record the steps done)."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["seeds"] = [0]
    cfg["model"]["features"] = 8
    cfg["training"]["adam"].update(steps=2, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    first, second = tmp_path / "first", tmp_path / "second"
    cfg["rungs"] = ["M_1q100"]
    _train(tmp_path, cfg, first)
    cfg["rungs"] = ["M_1w150"]
    _train(tmp_path, cfg, second)
    cfg["rungs"] = ["M_1q100", "M_1w150"]
    cfg["training"]["adam"]["steps"] = 0
    cfg["training"]["lbfgs"].update(steps=2, max_minutes=10)
    cfg["training"]["start_from"] = {"dir": str(first), "by_rung": {"M_1w150": str(second)}}
    cfg["training"]["trace"] = {"every": 100, "frames": 1, "lbfgs_every": 1}
    out = tmp_path / "out"
    _train(tmp_path, cfg, out)
    rq, rw = (json.load(open(out / f"{n}_seed0.json")) for n in ("M_1q100", "M_1w150"))
    assert rq["training"]["start"]["from"] == str(first / "M_1q100_seed0.params.pkl")
    assert rw["training"]["start"]["from"] == str(second / "M_1w150_seed0.params.pkl")
    tr = rq["training"]["lbfgs_trace"]
    assert [s for s, _ in tr] == [0, 1, 2] and tr[-1][1] is None
    assert tr[0][1] == pytest.approx(rq["training"]["start"]["final_loss"]["total"], rel=1e-10)
    assert "charges" not in rq["training"] and rq["training"]["options"]["lbfgs_steps_done"] == 2
    assert rw["training"]["options"]["lbfgs_max_seconds"] == 600.0
    assert [v["step"] for v in rw["training"]["charges"]["lbfgs_trace"]] == [0, 1, 2]


def test_job11s_configs_and_reader_agree():
    """job 11's two configs: the polishes on job 7's settings (the scan from job 7, the warm starts from job 10, the
    field-term dagger from job 8) and D' on job 8's settings (from job 8, the read-out held from job 5)."""
    cfg = E4.load_config(os.path.join(REPO, "configs", "e4_job11.yaml"))
    assert cfg["rungs"] == ["M_1q070", "M_1q100", "M_1q145", J11.LOW, J11.HIGH, J11.FIELD] and cfg["seeds"] == [0, 1]
    tr = cfg["training"]
    assert tr["adam"]["steps"] == 0 and tr["lbfgs"]["steps"] == 1000 and tr["lbfgs"]["max_minutes"] == 300
    assert tr["start_from"] == {"dir": "results/e4_job7", "by_rung": {"M_1w050": "results/e4_job10",
                                                                      "M_1w150": "results/e4_job10",
                                                                      "M_16pm_us_d20": "results/e4_job8"}}
    assert tr["trace"] == {"every": 100, "frames": 4, "lbfgs_every": 25}
    j7 = E4.load_config(os.path.join(REPO, "configs", "e4_job7.yaml"))
    j8 = E4.load_config(os.path.join(REPO, "configs", "e4_job8.yaml"))
    for key in ("data", "model", "probes", "reading"):
        assert cfg[key] == j7[key] == j8[key]
    assert tr["loss"] == j7["training"]["loss"] and tr["readout"] == j7["training"]["readout"]
    d = E4.load_config(os.path.join(REPO, "configs", "e4_job11d.yaml"))
    j8d = E4.load_config(os.path.join(REPO, "configs", "e4_job8d.yaml"))
    assert d["rungs"] == [J11.DPRIME] and d["seeds"] == [0, 1]
    assert d["training"]["start_from"] == {"dir": "results/e4_job8"} and d["training"]["lbfgs"]["max_minutes"] == 600
    for key in ("data", "model", "probes", "reading", "liquid", "liquid_k"):
        assert d[key] == j8d[key]
    assert d["training"]["readout"] == j8d["training"]["readout"] and d["training"]["loss"] == j8d["training"]["loss"]
    assert set(J11.SCAN_C) == {"M_1q070", "M_1q100", "M_1q110", "M_1q145"}


def _infra(r, eps, probes, t5):
    r["readings"]["liquid"] = {"eps_inf_model": eps}
    r["matching"] = {"liquid_before_training": {"eps_inf_model": eps}}
    r["readings"]["probes"] = {"by_distance_angstrom": {d: {"nonadditive_rmse": p, "nonadditive_rms": 100.0}
                                                        for d, p in zip(("10", "12", "15"), probes)}}
    r["readings"]["far_trimers"] = {"three_body_rmse": t5, "three_body_rms": 100.0}
    return r


def _job11(losses=None, low=(1.60, 1.72), field=(0.455, 0.466, 0.468), dprime=(0.455, 0.461, 0.463),
           infra=(1.6893, (2.23, 1.98, 2.66), 4.31), sizes=("64", "128", "256")):
    """job 11's runs and its references (jobs 7, 8 and 10) on constructed jsons."""
    losses = losses or {0.7: 34.0, 1.0: 24.0, 1.45: 31.0}
    job7, _ = _runs(warm={0.5: (1.21, 1.32), 1.0: (1.93, 1.94), 1.5: (2.46, 2.56)})
    for c, L in {0.7: 43.1, 1.0: 31.1, 1.1: 27.5, 1.45: 40.3}.items():
        for s in J11.SEEDS:
            job7[E4.scan_rung(c)][s]["training"]["final_loss"]["total"] = L
    runs, job8, job10 = {}, {}, {}
    add = lambda d, r: d.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    for s in J11.SEEDS:
        for c, L in losses.items():
            add(runs, _run(E4.scan_rung(c), s, {n: 0.55 for n in sizes}, loss=L))
        add(job10, _run("M_1q110", s, {n: 0.52 for n in sizes}, loss=21.5))
        add(job10, _run(J11.LOW, s, {n: 0.62 for n in sizes}, dipole=(1.504, 1.648)[s], loss=30.0))
        add(job10, _run(J11.HIGH, s, {n: 0.55 for n in sizes}, dipole=(2.108, 2.184)[s], loss=25.0))
        add(runs, _run(J11.LOW, s, {n: 0.60 for n in sizes}, dipole=low[s], loss=28.0))
        add(runs, _run(J11.HIGH, s, {n: 0.54 for n in sizes}, dipole=(2.08, 2.15)[s], loss=24.0))
        add(job8, _run(J11.FIELD, s, dict(zip(sizes, (0.483, 0.495, 0.497))), loss=30.0))
        add(runs, _run(J11.FIELD, s, dict(zip(sizes, field)), loss=24.0))
        add(job8, _infra(_run(J11.DPRIME, s, dict(zip(sizes, (0.478, 0.485, 0.487))), loss=18.9),
                         1.6893, (2.23, 1.98, 2.66), 4.31))
        add(runs, _infra(_run(J11.DPRIME, s, dict(zip(sizes, dprime)), loss=15.0), infra[0], infra[1], infra[2]))
    for name in (J11.FIELD, J11.DPRIME):
        for s in J11.SEEDS:
            runs[name][s]["training"]["options"] = {"lbfgs_steps_done": 1000 if name == J11.FIELD else 340}
    return runs, job7, job8, job10


def test_the_job11_readings():
    """K11.1-K11.5 and the model rule on constructed jsons: the polished profile through job 11's scan and job 10's
    M_1q110; the low side's move; the E4-class models against job 8's runs of the same seeds; D''s infrared; then each
    band's edge."""
    runs, job7, job8, job10 = _job11()
    ev = J11.evaluate(runs, job7, job8, job10)
    pts = {0.7: 34.0, 1.0: 24.0, 1.1: 21.5, 1.45: 31.0}
    coef = np.polyfit(list(pts), list(pts.values()), 2)
    assert ev["profile"]["c_min"] == pytest.approx(-coef[1] / (2 * coef[0]))
    assert ev["K11.1"]["holds"] is (abs(ev["profile"]["c_min"] - 1.098) <= 0.05)
    assert ev["K11.2"]["holds"] and ev["K11.2"]["runs"]["M_1w050 seed 1"]["moved"] == pytest.approx(1.72 - 1.648)
    assert ev["K11.3"]["holds"] and ev["K11.3"]["values"]["64"][2] == pytest.approx(0.455 / 0.483 - 1)
    assert ev["K11.4"]["holds"] and ev["K11.4"]["steps_done"] == {0: 340, 1: 340}
    assert ev["K11.5"]["holds"] and ev["model_rule"]["fires"]
    assert ev["response_after_polish"]["64"] == pytest.approx(0.455 / 0.455 - 1)
    assert ev["high"]["M_1w150 seed 0"]["end_debye"] == 2.08
    text = J11.report(ev)
    for k in ("K11.2: holds", "K11.3: holds", "K11.4: holds", "K11.5: holds", "model rule: FIRES", "the polishes"):
        assert k in text
    assert J11.evaluate(*_job11(losses={0.7: 30.0, 1.0: 22.0, 1.45: 40.0}))["K11.1"]["holds"] is False  # minimum lower
    assert J11.evaluate(*_job11(low=(1.53, 1.70)))["K11.2"]["holds"] is False         # +0.026 D
    assert J11.evaluate(*_job11(field=(0.465, 0.466, 0.468)))["K11.3"]["holds"] is False  # -3.7 % at 64
    ev = J11.evaluate(*_job11(dprime=(0.470, 0.461, 0.463)))                           # -1.7 % at 64
    assert ev["K11.4"]["holds"] is False and not ev["model_rule"]["fires"]
    ev = J11.evaluate(*_job11(infra=(1.6893, (2.23, 1.98, 3.3), 4.31)))                 # the probes at 15 A move
    assert ev["K11.5"]["holds"] is False and not ev["model_rule"]["fires"] and "does not fire" in J11.report(ev)
    runs, job7, job8, job10 = _job11()
    del runs[J11.DPRIME]
    ev = J11.evaluate(runs, job7, job8, job10)
    assert ev["K11.4"]["holds"] is None and ev["model_rule"]["read"] is False
