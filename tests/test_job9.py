"""
E4 job 9 (v0.0.33; notes/e4-job9-preregistration.md): the learned charges' overall scale (model.scale_learned_charges,
model.charge_scale_slope), the charge head's own learning rate and the diagnostics in train.fit, the driver's records
(e4_water.charge_diagnostics), and the readings K9.1-K9.5 (scripts/e4_job9.py) and K9.6 (scripts/e4_sumrules.py read
--job9_dir).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, train as T, water as W, structure as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job7 as J7                                      # noqa: E402
import e4_job9 as J9                                      # noqa: E402
import e4_sumrules as SR                                  # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol, SITE_Q   # noqa: E402
from test_job7 import _run, _runs, _blocks                             # noqa: E402,F401


def _cells(tmp_path, n_frames=2, n_mol=16, seed=3):
    _fake_cells(tmp_path / "t.npz", n_frames, n_mol, 12.5, seed=seed)
    return W.e4_structures(str(tmp_path / "t.npz"), pins=True)


def _cfg(features=8):
    return E4.merge(E4.load_config(None), {"model": {"features": features}})


def _rung(name, st):
    return E4.warm_started(name, E4.rungs_for(name, _cfg(), st[0].cell)[0], st)


def test_the_charges_overall_scale(tmp_path):
    """scale_learned_charges multiplies every learned charge by s (the neutrality shift included) and leaves the rest of
    the model alone; charge_scale_slope is d/ds of a loss at s = 1, against a central difference; the Coulomb band is
    then s^2 times its value, so the objective is a quartic in s."""
    st = _cells(tmp_path)
    rung = _rung("M_1x000", st)
    p = M.init_params(jax.random.PRNGKey(2), rung, energy_scale=1e-4)
    b = {k: v[0] for k, v in S.pad_batch([st[0]], n_freq=rung.K).items()}
    _, aux = M.energy_single(p, rung, b)
    _, aux2 = M.energy_single(M.scale_learned_charges(p, 1.3), rung, b)
    assert np.allclose(np.asarray(aux2["q"]), 1.3 * np.asarray(aux["q"]), atol=1e-15)
    assert float(aux2["E_coul"]) == pytest.approx(1.69 * float(aux["E_coul"]), rel=1e-10)
    assert abs(float(np.sum(np.asarray(aux["q"])[:3]))) < 1e-14                                 # each molecule neutral
    sc = M.scale_learned_charges(p, 2.0)
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(sc["e0"]), jax.tree_util.tree_leaves(p["e0"])))
    loss = T.make_loss(rung, 1e-4, 100.0, 1.0, "rigid")
    data = S.pad_batch(st, n_freq=rung.K)
    g = jax.grad(lambda q: loss(q, data)[0])(p)
    f = lambda s: float(loss(M.scale_learned_charges(p, s), data)[0])
    assert float(M.charge_scale_slope(p, g)) == pytest.approx((f(1 + 1e-5) - f(1 - 1e-5)) / 2e-5, rel=1e-6)
    labels = M.charge_labels(p)
    assert labels["heads"]["q_ref"] == "charge" and set(jax.tree_util.tree_leaves(labels["heads"]["charge"])) == {"charge"}
    assert labels["heads"]["energy"][0]["w"] == "rest" and set(jax.tree_util.tree_leaves(labels["e0"])) == {"rest"}


def test_the_charge_heads_own_learning_rate_and_the_diagnostics(tmp_path):
    """train.fit with charge_lr_scale k: the charge head's first Adam step is k times the default's and the rest of the
    model's is the default's, digit for digit; the trace, the pull on every step and the parameters at the end of Adam
    change nothing; the options record them; the scale's options are refused on pinned charges."""
    st = _cells(tmp_path)
    rung = _rung("M_1x050", st)
    kw = dict(seed=0, steps=2, lr=3e-3, w_energy=100.0, w_force=1.0, batch_size=2, lbfgs_steps=0, force_mode="rigid")
    p1, i1 = T.fit(rung, st, **kw)
    seen = []
    p10, i10 = T.fit(rung, st, charge_lr_scale=10.0, trace_every=1, scale_gradient=True, snapshot_adam=True,
                     trace_fn=lambda p: seen.append(1) or float(p["heads"]["q_ref"][1]), **kw)
    k1, k10 = T.fit(rung, st, steps=1, **{k: v for k, v in kw.items() if k != "steps"}), \
        T.fit(rung, st, steps=1, charge_lr_scale=10.0, **{k: v for k, v in kw.items() if k != "steps"})
    p0 = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=i1["scale"])
    d1 = np.asarray(k1[0]["heads"]["q_ref"] - p0["heads"]["q_ref"])
    d10 = np.asarray(k10[0]["heads"]["q_ref"] - p0["heads"]["q_ref"])
    assert np.allclose(d10[[1, 8]], 10 * d1[[1, 8]], rtol=1e-12) and abs(d1[1]) == pytest.approx(3e-3, rel=1e-6)
    same = lambda a, b: all(np.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    assert same(k1[0]["e0"], k10[0]["e0"]) and same(k1[0]["heads"]["energy"], k10[0]["heads"]["energy"])
    assert [s for s, _ in i10["trace"]] == [0, 1, 2] and len(seen) == 3
    assert i10["trace"][0][1] == pytest.approx(0.5 * st[0].pinned["q"][1] / E4.Q_UNIT_WARM)
    assert i10["scale_gradient"].shape == (2,) and same(i10["params_adam"], p10)
    assert i10["training_options"]["charge_lr_scale"] == 10.0 and i10["training_options"]["trace_every"] == 1
    assert "charge_lr_scale" not in i1["training_options"] and "trace" not in i1
    pd, _ = T.fit(rung, st, trace_every=1, trace_fn=lambda p: 0.0, scale_gradient=True, snapshot_adam=True, **kw)
    assert same(pd, p1)                                          # the diagnostics change nothing
    pinned = E4.rungs_for("M_1q100", _cfg(), st[0].cell)[0]
    for bad in (dict(charge_lr_scale=10.0), dict(scale_gradient=True)):
        with pytest.raises(ValueError, match="learned charges"):
            T.fit(pinned, st, **bad, **kw)
    with pytest.raises(ValueError, match="trace_every"):
        T.fit(rung, st, trace_every=1, **kw)
    with pytest.raises(ValueError, match="charge_lr_scale"):
        T.fit(rung, st, charge_lr_scale=0.0, **kw)


def test_job9s_rungs_config_and_readers_agree():
    """the rungs: M_1x050 and M_1x150 start as job 7's warm starts, M_1x000 near zero, all three at ten times the charge
    head's learning rate; job 7's own rungs as before; the config runs job 7's warm starts beside them, traced, on job
    7's settings otherwise."""
    assert E4.CHARGE_LR == {"M_1x000": 10.0, "M_1x050": 10.0, "M_1x150": 10.0}
    assert {k: E4.STARTS[k] for k in J9.FIX} == J9.FIX and J9.ZERO not in E4.STARTS and E4.WARM == J7.WARM
    assert set(J9.BASE) <= set(J7.WARM) and set(J9.PROFILE_C) <= set(J7.SCAN)
    for name in (*J9.FIX, J9.ZERO):
        r = E4.E3.RUNGS[name]
        assert r.charges and not r.pin_q and not r.dispersion and r.q_unit == E4.Q_UNIT_WARM
    cfg = E4.load_config(os.path.join(REPO, "configs", "e4_job9.yaml"))
    assert cfg["rungs"] == [*J9.BASE, *J9.FIX, J9.ZERO] and cfg["seeds"] == list(J9.SEEDS)
    assert cfg["training"]["trace"] == {"every": 100, "frames": 4}
    j7 = E4.load_config(os.path.join(REPO, "configs", "e4_job7.yaml"))
    for key in ("data", "model", "probes", "reading"):
        assert cfg[key] == j7[key]
    assert {k: v for k, v in cfg["training"].items() if k != "trace"} == {k: v for k, v in j7["training"].items()
                                                                           if k != "trace"}


def test_driver_records_the_charges_diagnostics(tmp_path, capsys):
    """scripts/e4_water.py train with the trace on fake cells: the json records the trace from the start, the pull on
    every Adam step with its second half's |mean| / sd, and the objective along the charges' scale at the end of Adam
    and of training, with their slopes; the start at 1.5 times the O/H charges is traced from there; the options record
    the charge head's learning rate; a rung with pinned charges is not traced."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["rungs"], cfg["seeds"] = ["M_1x150", "M_1q100"], [0]
    cfg["model"]["features"] = 8
    cfg["training"]["adam"].update(steps=4, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=1, chunk_size=2)
    cfg["training"]["trace"] = {"every": 2, "frames": 1}
    import yaml
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmp_path / "out"
    E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=None, out_dir=str(out)))
    text = capsys.readouterr().out
    r = json.load(open(out / "M_1x150_seed0.json"))
    d = r["training"]["charges"]
    assert r["training"]["options"]["charge_lr_scale"] == 10.0 and d["frames"] == 1
    assert [v["step"] for v in d["trace"]] == [0, 2, 4]
    q = W.oh_equivalent_charges(SITE_Q, 0.426706882)
    P = np.load(tmp_path / "train_n16.npz")["positions"][0][:3]
    assert d["trace"][0]["dipole_debye"] == pytest.approx(1.5 * np.linalg.norm(q @ P) * 4.803204, rel=0.02)
    assert d["trace"][0]["to_H_side"] == 1.0 and d["trace"][0]["q_H"] == pytest.approx(1.5 * q[1], rel=1e-9)
    assert len(d["pull"]["steps"]) == 4 and d["pull"]["second_half"]["steps"] == [2, 4]
    h = d["pull"]["second_half"]
    assert h["snr"] == pytest.approx(abs(np.mean(d["pull"]["steps"][2:])) / np.std(d["pull"]["steps"][2:]))
    for label in ("adam", "final"):
        c = d["scale_curve"][label]
        assert c["s"] == list(E4.SCALE_GRID) and len(c["loss"]) == len(E4.SCALE_GRID)
        assert c["slope"] == pytest.approx((c["loss"][5] - c["loss"][3]) / 0.02) and 0.5 <= c["s_min"] <= 1.5
        assert "dipole_debye" in c["charges"]
    assert d["scale_curve"]["adam"]["charges"]["dipole_debye"] == d["trace"][-1]["dipole_debye"]
    assert "charges" not in json.load(open(out / "M_1q100_seed0.json"))["training"]
    assert "the learned dipole through Adam" in text and "the charge head's learning rate 10 times the rest's" in text
    assert "the objective along the charges' scale, the rest held (adam)" in text


def _diag(dip_adam, slope, snr=0.1, steps=3000, path=None, s_min=1.0, dip_final=None, slope_final=0.0):
    path = path or {0: 0.93, 1500: dip_adam - 0.02, 3000: dip_adam}
    return {"frames": 4, "trace": [{"step": s, "dipole_debye": v, "q_O": -0.6, "q_H": 0.3, "to_H_side": 1.0}
                                   for s, v in sorted(path.items())],
            "pull": {"window": 100, "mean": [], "sd": [], "steps": [],
                     "second_half": {"steps": [steps // 2, steps], "mean": snr, "sd": 1.0, "snr": snr}},
            "scale_curve": {"adam": {"s": list(E4.SCALE_GRID), "loss": [0.0] * 9, "slope": slope, "s_min": s_min,
                                     "charges": {"dipole_debye": dip_adam}},
                            "final": {"s": list(E4.SCALE_GRID), "loss": [0.0] * 9, "slope": slope_final, "s_min": s_min,
                                      "charges": {"dipole_debye": dip_final or dip_adam}}}}


def _job9(base=None, fix=None, zero=(2.0, 2.1), fix_forces=0.60, R=1.0, snr=0.1, loss=30.0, sizes=("64", "128", "256")):
    """job 9's runs on constructed jsons, and job 7's (the scan, M_inf, the warm starts) and job 6's M_inf."""
    job7, job6 = _runs(scan={0.4: 0.83, 0.7: 0.74, 1.0: 0.63, 1.1: 0.59, 1.45: 0.71})
    for c, L in {0.7: 43.1, 1.0: 31.1, 1.1: 27.5, 1.45: 40.3}.items():
        for s in J7.SEEDS:
            job7[E4.scan_rung(c)][s]["training"]["final_loss"]["total"] = L
    prof = J9.profile(job7)
    base = base or {"M_1w050": (1.21, 1.32), "M_1w150": (2.46, 2.56)}
    fix = fix or {"M_1x050": (2.0, 2.05), "M_1x150": (2.1, 2.15)}
    runs = {}
    add = lambda r: runs.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    for rung, dips in base.items():
        for s in J9.SEEDS:
            r = _run(rung, s, {n: 0.7 for n in sizes}, dipole=dips[s], loss=45.0)
            c = dips[s] / J9.PERMANENT_DEBYE
            r["training"]["charges"] = _diag(dips[s], R * J9.profile_slope(prof, c), snr=snr)
            add(r)
    for rung, dips in fix.items():
        for s in J9.SEEDS:
            r = _run(rung, s, {n: fix_forces for n in sizes}, dipole=dips[s], loss=loss)
            r["training"]["charges"] = _diag(dips[s], 0.5)
            add(r)
    for s in J9.SEEDS:
        r = _run(J9.ZERO, s, {n: 0.62 for n in sizes}, dipole=zero[s], loss=loss)
        r["readings"]["cells"]["64"]["test"]["learned_sources"]["dipole_to_H_side"] = 0.0
        r["training"]["charges"] = _diag(zero[s], 0.3)
        add(r)
    return runs, job7, job6


def test_the_job9_readings():
    """K9.1-K9.5 on constructed jsons: the profile from job 7's scan (job 7's parabola, minimum at c = 1.098); R per
    run against it; the pull's |mean| / sd; the four warm starts' convergence and gate; K9.4 read only when K9.3 holds;
    K9.5 against the scan's lowest; then each band's edge and the locked branch."""
    runs, job7, job6 = _job9()
    ev = J9.evaluate(runs, job7, job6)
    prof = ev["profile"]
    assert prof["c_min"] == pytest.approx(1.0981, abs=1e-3) and prof["points"]["1.1"] == 27.5
    assert J9.profile_slope(prof, 0.68) == pytest.approx(-51.9, abs=0.2)
    k1 = ev["K9.1"]
    assert k1["holds"] and k1["verdict"] == "holds" and len(k1["in_band"]) == 4
    assert k1["runs"]["M_1w050 seed 0"]["c"] == pytest.approx(1.21 / 1.864) and k1["runs"]["M_1w050 seed 0"]["R"] == \
        pytest.approx(1.0)
    assert ev["K9.2"]["holds"]
    c = ev["K9.3"]
    assert c["holds"] and c["loss_max"] == 75 and c["mean_debye"] == pytest.approx(np.mean([2.0, 2.05, 2.1, 2.15]))
    k4 = ev["K9.4"]
    assert k4["read"] and k4["holds"] and k4["target_debye"] == pytest.approx(prof["c_min"] * 1.864)
    assert ev["K9.5"]["holds"] and ev["K9.5"]["values"]["64"]["scan_rung"] == "M_1q110"
    assert ev["K9.5"]["values"]["128"]["ratio"] == pytest.approx(0.60 / 0.59 - 1)
    assert ev["zero"]["M_1x000 seed 1"]["against_warm"] == pytest.approx(2.1 / c["mean_debye"] - 1)
    assert ev["reproduction"]["M_1w050 seed 0"]["dipole_debye"] == (1.21, 2.0)
    assert ev["paths"]["M_1x050 seed 0"]["second_half_share"] == pytest.approx(0.02 / (2.0 - 0.93))
    text = J9.report(ev)
    for k in ("K9.1: holds", "K9.2: holds", "K9.3: holds", "K9.4: holds", "K9.5: holds", "the start near zero",
              "the paths", "the parabola's minimum at c* = 1.098"):
        assert k in text
    # the bands' edges and branches
    ev = J9.evaluate(*_job9(R=0.1))
    assert ev["K9.1"]["holds"] is False and ev["K9.1"]["verdict"] == "FAILS, low: locked"
    assert J9.evaluate(*_job9(R=2.5))["K9.1"]["verdict"] == "FAILS"
    assert J9.evaluate(*_job9(snr=0.4))["K9.2"]["holds"] is False
    ev = J9.evaluate(*_job9(fix={"M_1x050": (1.5, 1.6), "M_1x150": (2.3, 2.4)}))
    assert ev["K9.3"]["holds"] is False and ev["K9.4"]["read"] is False and ev["K9.4"]["holds"] is None
    assert "K9.4: not read" in J9.report(ev)
    ev = J9.evaluate(*_job9(fix={"M_1x050": (2.4, 2.45), "M_1x150": (2.5, 2.55)}))
    assert ev["K9.3"]["holds"] and ev["K9.4"]["holds"] is False                     # converged, 22 % above c*
    assert J9.evaluate(*_job9(fix_forces=0.64))["K9.5"]["holds"] is False           # 8.5 % above the scan's lowest
    assert J9.evaluate(*_job9(loss=80.0))["K9.3"]["holds"] is False                 # above M_inf's largest
    runs, job7, job6 = _job9()
    del runs["M_1x150"][1]
    assert J9.convergence(runs, job7, job6)["holds"] is None
    del job7["M_1q110"]
    ev = J9.evaluate(runs, job7, job6)
    assert ev["profile"] is None and ev["K9.1"]["holds"] is None and "K9.1: not read" in J9.report(ev)


def test_k96_reads_the_fixed_warm_starts_in_l12s_band_when_they_converged(tmp_path, monkeypatch, capsys):
    """K9.6 from read_size's readings: the four warm starts at ten times the charge head's learning rate, |b_T| in L1.2's
    band at both shells, at least three; not read unless K9.3 holds; the start near zero is not one of them; and the S0
    read prints it with --job9_dir."""
    cand = lambda bT: {"b_T": (bT, 0.01), "r": (1.8, 0.02)}
    labels = [f"{w}_s{s}" for w in J9.FIX for s in J9.SEEDS]
    sizes = {n: {"shells": {"100": {"candidates": {**{lab: cand(v) for lab, v in zip(labels, [1.3, -1.31, 1.5, 1.33])},
                                                    "M_1x000_s0": cand(1.3), "M_1w050_s0": cand(2.2)}}}}
             for n in (128, 256)}
    out = SR.evaluate_job9(sizes, {"holds": True})["K9.6"]
    assert out["holds"] and len(out["runs_in_band"]) == 3 and "M_1x000_s0" not in out["runs_in_band"]
    assert out["min_runs"] == 3 and out["gate_name"] == "K9.3"
    assert SR.evaluate_job9(sizes, {"holds": False})["K9.6"]["verdict"] == "not read (K9.3 does not hold)"
    assert SR.evaluate_job7(sizes, {"holds": True})["J7.6"]["runs_in_band"] == []        # job 7's reading unchanged
    from test_sumrules import _fake_sets, _FakeMB
    from sseft import mbpol as MB
    rng = np.random.default_rng(4)
    _fake_sets(tmp_path, ((128, 20.0), (256, 25.0)), rng)

    class Truth:
        def __init__(self, root):
            self.mb = _FakeMB()

        def describe(self):
            return {"name": "fake"}
    monkeypatch.setattr(W, "MBPolTruth", Truth)
    monkeypatch.setattr(MB, "MBPol", _FakeMB)
    s0 = tmp_path / "out"
    SR.run_mbpol(SimpleNamespace(mbx="x", data=str(tmp_path), out=str(s0), sizes=[128, 256], part=None))
    for n in (128, 256):
        zero = np.zeros((len(SR.load_frames(str(tmp_path), n)["positions"]), 4 * n, 3))
        for lab in labels:
            np.savez(s0 / f"model_{lab}_n{n}.npz", q=0.8 * np.tile(SITE_Q, n), mu_full=zero, mu_ind=zero,
                     meta=json.dumps({"field_blind": True}))
    runs, job7, job6 = _job9()
    dirs = {"job9": runs, "job7": job7, "job6": job6}
    for d, rs in dirs.items():
        (tmp_path / d).mkdir()
        for name, by in rs.items():
            for s, r in by.items():
                json.dump(r, open(tmp_path / d / f"{name}_seed{s}.json", "w"))
    capsys.readouterr()
    bands, _ = SR.run_read(SimpleNamespace(data=str(tmp_path), out=str(s0), json=str(s0 / "s0.json"), block=2,
                                           gate_dir=str(tmp_path / "job6"), job7_dir=str(tmp_path / "job7"),
                                           job9_dir=str(tmp_path / "job9")))
    text = capsys.readouterr().out
    assert bands["K9.6"]["converged"] is True and "K9.6: " in text and "of the four needed; K9.3 from" in text
