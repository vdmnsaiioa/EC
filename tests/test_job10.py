"""
E4 job 10 (v0.0.34; notes/e4-job10-preregistration.md): a run started from a trained run's parameters (train.fit's
params_init, training.start_from), the learned charges traced through the L-BFGS polish (lbfgs_trace_every), the
driver's records, and the readings K10.1-K10.4 (scripts/e4_job10.py) and K10.5 (scripts/e4_sumrules.py read
--job10_dir).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, train as T, structure as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job7 as J7                                      # noqa: E402
import e4_job9 as J9                                      # noqa: E402
import e4_job10 as J10                                    # noqa: E402
import e4_sumrules as SR                                  # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402
from test_job7 import _run, _runs                              # noqa: E402
from test_job9 import _cells, _rung                            # noqa: E402

KW = dict(seed=0, lr=3e-3, w_energy=100.0, w_force=1.0, batch_size=2, force_mode="rigid", lbfgs_chunk=2)


def test_fit_starts_from_given_parameters_and_traces_the_polish(tmp_path):
    """params_init: the run starts from those parameters, not the seed's (the first traced objective is theirs, on every
    training frame); the L-BFGS trace reads the charges and the objective at every k-th step and after the last, and
    changes nothing; a structure that is not the rung's is refused, as is a trace without its reading."""
    st = _cells(tmp_path)
    rung = _rung("M_1w150", st)
    p0, i0 = T.fit(rung, st, steps=2, lbfgs_steps=0, **KW)
    reading = lambda p: float(p["heads"]["q_ref"][1])
    p1, i1 = T.fit(rung, st, steps=0, lbfgs_steps=3, params_init=p0, trace_fn=reading, lbfgs_trace_every=1,
                   scale=i0["scale"], **KW)
    tr = i1["lbfgs_trace"]
    assert [s for s, _, _ in tr] == [0, 1, 2, 3] and tr[-1][2] is None
    assert tr[0][1] == pytest.approx(float(p0["heads"]["q_ref"][1]), rel=1e-15)
    f = T.make_chunked_loss(rung, i0["scale"], S.pad_batch(st, n_freq=rung.K), 100.0, 1.0, chunk=2, force_mode="rigid")
    assert tr[0][2] == pytest.approx(float(f(p0)), rel=1e-12)
    assert tr[2][2] <= tr[1][2] <= tr[0][2]                        # the line search never goes up
    assert i1["training_options"]["params_init"] is True and i1["training_options"]["lbfgs_trace_every"] == 1
    q, _ = T.fit(rung, st, steps=0, lbfgs_steps=3, params_init=p0, scale=i0["scale"], **KW)
    same = lambda a, b: all(np.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    assert same(q, p1)                                             # the trace changes nothing
    fresh = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=i0["scale"])
    assert not same(p0, fresh)
    bad = {**p0, "heads": {**p0["heads"], "q_ref": np.zeros(3)}}
    with pytest.raises(ValueError, match="params_init"):
        T.fit(rung, st, steps=0, lbfgs_steps=1, params_init=bad, **KW)
    with pytest.raises(ValueError, match="lbfgs_trace_every"):
        T.fit(rung, st, steps=0, lbfgs_steps=1, params_init=p0, lbfgs_trace_every=1, **KW)
    with pytest.raises(ValueError, match="trace_fn"):
        T.fit(rung, st, steps=1, lbfgs_steps=0, trace_fn=reading, **KW)


def _train(tmp_path, cfg, out):
    import yaml
    path = tmp_path / f"{os.path.basename(out)}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    E4.run_train(SimpleNamespace(config=str(path), rungs=None, seeds=None, task=None, out_dir=str(out)))


def test_driver_starts_from_a_trained_run(tmp_path, capsys):
    """scripts/e4_water.py train with training.start_from: each run starts from the directory's run of the same rung
    and seed, records where from and the objective there (the earlier run's final loss), traces the learned charges
    through the polish from that objective on, and reads the objective along their scale at the start and the end; a
    pinned rung records its start alone; a start whose json records another model is refused."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["rungs"], cfg["seeds"] = ["M_1w150", "M_1q110"], [0]
    cfg["model"]["features"] = 8
    cfg["training"]["adam"].update(steps=2, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    first = tmp_path / "first"
    _train(tmp_path, cfg, first)
    cfg["training"]["adam"]["steps"] = 0
    cfg["training"]["lbfgs"]["steps"] = 2
    cfg["training"]["start_from"] = {"dir": str(first)}
    cfg["training"]["trace"] = {"every": 100, "frames": 1, "lbfgs_every": 1}
    out = tmp_path / "out"
    capsys.readouterr()
    _train(tmp_path, cfg, out)
    text = capsys.readouterr().out
    r0, r = json.load(open(first / "M_1w150_seed0.json")), json.load(open(out / "M_1w150_seed0.json"))
    st = r["training"]["start"]
    assert st["from"] == str(first / "M_1w150_seed0.params.pkl")
    assert st["final_loss"]["total"] == pytest.approx(r0["training"]["final_loss"]["total"], rel=1e-12)
    d = r["training"]["charges"]
    assert [v["step"] for v in d["lbfgs_trace"]] == [0, 1, 2] and d["lbfgs_trace"][-1]["objective"] is None
    assert d["lbfgs_trace"][0]["objective"] == pytest.approx(st["final_loss"]["total"], rel=1e-10)
    assert d["lbfgs_trace"][0]["dipole_debye"] == pytest.approx(st["charges"]["dipole_debye"], rel=1e-12)
    assert set(d["scale_curve"]) == {"start", "final"} and d["pull"]["second_half"]["mean"] is None
    assert d["scale_curve"]["start"]["charges"]["dipole_debye"] == pytest.approx(st["charges"]["dipole_debye"])
    assert r["training"]["options"]["params_init"] is True and r["training"]["options"]["steps"] == 0
    rq = json.load(open(out / "M_1q110_seed0.json"))
    assert "start" in rq["training"] and "charges" not in rq["training"] and "charges" not in rq["training"]["start"]
    assert "starting from" in text and "the learned dipole through the L-BFGS polish" in text
    assert "the objective along the charges' scale, the rest held (start)" in text
    r0["model"]["band_edge_angstrom"] = 1.5
    json.dump(r0, open(first / "M_1w150_seed0.json", "w"))
    with pytest.raises(ValueError, match="another model"):
        _train(tmp_path, cfg, tmp_path / "again")


def test_job10s_config_and_readers_agree():
    """job 10 polishes job 7's warm starts and the control from job 7's directory, on job 7's settings otherwise."""
    cfg = E4.load_config(os.path.join(REPO, "configs", "e4_job10.yaml"))
    assert cfg["rungs"] == [*J10.WARM, J10.CONTROL] and cfg["seeds"] == list(J10.SEEDS)
    assert set(J10.OFF) < set(J10.WARM) and J10.WARM == J7.WARM and J10.CONTROL in J7.SCAN
    tr = cfg["training"]
    assert tr["adam"]["steps"] == 0 and tr["lbfgs"]["steps"] == 1000 and tr["start_from"] == {"dir": "results/e4_job7"}
    assert tr["trace"] == {"every": 100, "frames": 4, "lbfgs_every": 25}
    j7 = E4.load_config(os.path.join(REPO, "configs", "e4_job7.yaml"))
    for key in ("data", "model", "probes", "reading"):
        assert cfg[key] == j7[key]
    assert tr["loss"] == j7["training"]["loss"] and tr["readout"] == j7["training"]["readout"]
    assert {k: v for k, v in tr["lbfgs"].items() if k != "steps"} == \
        {k: v for k, v in j7["training"]["lbfgs"].items() if k != "steps"}


def _job10(ends=None, forces=0.60, control=0.59, loss=30.0, sizes=("64", "128", "256")):
    """job 10's runs on constructed jsons, and job 7's (the scan, the warm starts, M_inf) and job 6's M_inf."""
    job7, job6 = _runs(warm={0.5: (1.21, 1.32), 1.0: (1.93, 1.94), 1.5: (2.46, 2.56)})
    for c, L in {0.7: 43.1, 1.0: 31.1, 1.1: 27.5, 1.45: 40.3}.items():
        for s in J7.SEEDS:
            job7[E4.scan_rung(c)][s]["training"]["final_loss"]["total"] = L
    ends = ends or {"M_1w050": (1.95, 2.0), "M_1w100": (2.0, 2.02), "M_1w150": (2.1, 2.12)}
    runs = {}
    add = lambda r: runs.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    for rung, dips in ends.items():
        for s in J10.SEEDS:
            r = _run(rung, s, {n: forces for n in sizes}, dipole=dips[s], loss=loss)
            r["training"]["start"] = {"final_loss": {"total": 45.0}}
            add(r)
    for s in J10.SEEDS:
        add(_run(J10.CONTROL, s, {n: control for n in sizes}, loss=26.0))
    return runs, job7, job6


def test_the_job10_readings():
    """K10.1-K10.4 on constructed jsons: each off-scale start's share of its way to the target (job 7's dipoles before,
    job 10's after), the six's convergence and gate, K10.3 read only when K10.2 holds, K10.4 against the polished
    control; then each band's edge."""
    runs, job7, job6 = _job10()
    ev = J10.evaluate(runs, job7, job6)
    t = ev["profile"]["c_min"] * 1.864
    k1 = ev["K10.1"]
    assert k1["holds"] and k1["runs"]["M_1w050 seed 0"]["share"] == pytest.approx((1.95 - 1.21) / (t - 1.21))
    assert k1["runs"]["M_1w150 seed 1"]["share"] == pytest.approx((2.12 - 2.56) / (t - 2.56))
    c = ev["K10.2"]
    assert c["holds"] and c["loss_max"] == 75 and c["mean_debye"] == pytest.approx(np.mean([1.95, 2.0, 2.0, 2.02, 2.1, 2.12]))
    assert ev["K10.3"]["read"] and ev["K10.3"]["holds"]
    assert ev["K10.4"]["holds"] and ev["K10.4"]["values"]["64"]["ratio"] == pytest.approx(0.60 / 0.59 - 1)
    assert ev["before_after"]["M_1w050 seed 0"]["loss"] == (45.0, 30.0)
    assert ev["before_after"]["M_1w050 seed 0"]["dipole"] == (1.21, 1.95)
    text = J10.report(ev)
    for k in ("K10.1: holds", "K10.2: holds", "K10.3: holds", "K10.4: holds", "every run before (job 7) and after"):
        assert k in text
    ev = J10.evaluate(*_job10(ends={"M_1w050": (1.3, 2.0), "M_1w100": (2.0, 2.02), "M_1w150": (2.1, 2.12)}))
    assert ev["K10.1"]["holds"] is False                          # 11 % of the way
    ev = J10.evaluate(*_job10(ends={"M_1w050": (1.5, 1.55), "M_1w100": (1.95, 1.96), "M_1w150": (2.3, 2.35)}))
    assert ev["K10.1"]["holds"] and ev["K10.2"]["holds"] is False and ev["K10.3"]["holds"] is None
    assert "K10.3: not read" in J10.report(ev)
    ev = J10.evaluate(*_job10(ends={"M_1w050": (2.3, 2.35), "M_1w100": (2.35, 2.4), "M_1w150": (2.4, 2.45)}))
    assert ev["K10.2"]["holds"] and ev["K10.3"]["holds"] is False          # converged, 16 % above the target
    assert J10.evaluate(*_job10(forces=0.64))["K10.4"]["holds"] is False    # 8.5 % above the control
    assert J10.evaluate(*_job10(loss=80.0))["K10.2"]["holds"] is False      # above M_inf's largest
    runs, job7, job6 = _job10()
    del runs[J10.CONTROL]
    assert J10.evaluate(runs, job7, job6)["K10.4"]["holds"] is None


def test_k105_reads_the_polished_warm_starts_in_l12s_band():
    """K10.5: the polished warm starts' model files (model_M_1w<c0>lb_s<seed>) in L1.2's band at both shells, at least
    four of the six; not read unless K10.2 holds; job 7's own files are not among them."""
    cand = lambda bT: {"b_T": (bT, 0.01), "r": (1.8, 0.02)}
    labels = [f"{w}lb_s{s}" for w in J10.WARM for s in J10.SEEDS]
    sizes = {n: {"shells": {"100": {"candidates": {**{lab: cand(v) for lab, v in
                                                      zip(labels, [1.3, 1.31, 1.33, 1.35, 1.5, 1.29])},
                                                    "M_1w050_s0": cand(1.3)}}}} for n in (128, 256)}
    out = SR.evaluate_job10(sizes, {"holds": True})["K10.5"]
    assert out["holds"] and len(out["runs_in_band"]) == 5 and "M_1w050_s0" not in out["runs_in_band"]
    assert out["gate_name"] == "K10.2" and SR.GATE_DIRS["K10.2"] == "job10_dir"
    assert SR.evaluate_job10(sizes, {"holds": False})["K10.5"]["verdict"] == "not read (K10.2 does not hold)"
    assert SR.evaluate_job7(sizes, {"holds": True})["J7.6"]["runs_in_band"] == ["M_1w050_s0"]
