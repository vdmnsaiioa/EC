"""
E4 job 16 (v0.0.41; notes/e4-job16-preregistration.md): a read-out held across training sets of another size
(training.readout.held_from.train_size), the seven configs at L_0 = 64, 128 and 256, and the readings K16.1-K16.5
(scripts/e4_job16.py).
"""
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, water as W

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job16 as J16                                    # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402


def _two_sizes(tmp_path):
    cfg = _small_cfg(tmp_path)
    _fake_cells(tmp_path / "train_n16.npz", 2, 16, 12.5, seed=1)
    _fake_cells(tmp_path / "train_n32.npz", 2, 32, 15.75, seed=2)
    _fake_cells(tmp_path / "test_n32.npz", 1, 32, 15.75, seed=3)
    return cfg


def _cache(path, rung_open, scale, readout=None):
    with open(path, "wb") as fh:
        pickle.dump({"rung": E4.rung_key(rung_open), "scale": scale, "readout": readout or {"x": np.arange(3.0)},
                     "match": {"objective": [(1.0, 0.5)]}}, fh)


def test_a_readout_is_held_across_training_sizes(tmp_path):
    """held_from.train_size: a read-out by species cached with the training cells of 16 molecules (their cell and
    energy scale) is held for training on 32; without train_size the cache reads as another scale's; a read-out that is
    not by species is refused across sizes; at its own size train_size changes nothing."""
    cfg = _two_sizes(tmp_path)
    s16 = W.e4_structures(str(tmp_path / "train_n16.npz"), pins=True)
    s32 = W.e4_structures(str(tmp_path / "train_n32.npz"), pins=True)
    sc16, sc32 = E4.T.energy_scale(s16), E4.T.energy_scale(s32)
    assert sc16 != sc32
    probes = {"hash": "p0"}
    _, ref16 = E4.rungs_for("M_16C2pm_us", cfg, s16[0].cell)
    _, ref32 = E4.rungs_for("M_16C2pm_us", cfg, s32[0].cell)
    assert ref16.ewald_n_max != ref32.ewald_n_max and ref16.bf_species
    _cache(tmp_path / "matched_M_16C2pm_us_p0.pkl", ref16, sc16)
    held = {"rung": "M_16C2pm_us", "dir": str(tmp_path), "train_size": 16}
    cfg["data"]["train_size"] = 32
    ro, match = E4.held_readout(held, cfg, s32[0].cell, probes, sc32)
    assert np.array_equal(ro["x"], np.arange(3.0)) and match["held_from"]["train_size"] == 16
    with pytest.raises(ValueError, match="another rung"):
        E4.held_readout({k: v for k, v in held.items() if k != "train_size"}, cfg, s32[0].cell, probes, sc32)
    cfg["data"]["train_size"] = 16                                   # at its own size: as before
    ro, match = E4.held_readout(held, cfg, s16[0].cell, probes, sc16)
    assert np.array_equal(ro["x"], np.arange(3.0))
    _, s_open = E4.rungs_for("M_16S2p", cfg, s16[0].cell)
    assert not s_open.bf_species
    _cache(tmp_path / "matched_M_16S2p_p0.pkl", s_open, sc16)
    cfg["data"]["train_size"] = 32
    with pytest.raises(ValueError, match="by species"):
        E4.held_readout({"rung": "M_16S2p", "dir": str(tmp_path), "train_size": 16}, cfg, s32[0].cell, probes, sc32)


def test_driver_trains_at_another_size_on_the_held_readout(tmp_path, monkeypatch):
    """scripts/e4_water.py train for D' on cells of 32 molecules (one Adam step), the read-out held from a cache made
    with the cells of 16: nothing is matched, the json records where it was matched, the trained read-out is the
    cached one."""
    import yaml
    cfg = _two_sizes(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    cfg["data"].update(train_size=32, eval_sizes=[32])
    cfg["model"].update(features=8, mutual_edge_angstrom=2.0)
    cfg["training"]["adam"].update(steps=1, batch_size=1)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=1)
    cfg["reading"]["eps_frames"] = 1
    ref_dir = tmp_path / "job5"
    cfg["training"]["readout"]["held_from"] = {"rung": "M_16C2pm_us", "dir": str(ref_dir), "train_size": 16}
    cfg["rungs"], cfg["seeds"] = ["M_16C2pm_us_d20"], [0]
    s16 = W.e4_structures(str(tmp_path / "train_n16.npz"), pins=True)
    probes = E4.load_probes(cfg["data"]["probes"])
    _, ref_open = E4.rungs_for("M_16C2pm_us", cfg, s16[0].cell)
    p0 = M.init_params(jax.random.PRNGKey(3), ref_open, energy_scale=E4.T.energy_scale(s16))
    readout = jax.tree_util.tree_map(lambda x: 1.1 * np.asarray(x) + 0.01 if np.issubdtype(np.asarray(x).dtype, np.floating)
                                     else np.asarray(x), p0["readout"])
    os.makedirs(ref_dir)
    _cache(ref_dir / f"matched_M_16C2pm_us_{probes['hash']}.pkl", ref_open, E4.T.energy_scale(s16), readout)
    monkeypatch.setattr(E4, "match_readout", lambda *a, **k: pytest.fail("a held read-out is not matched"))
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmp_path / "out"
    E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=None, out_dir=str(out)))
    r = json.load(open(out / "M_16C2pm_us_d20_seed0.json"))
    assert r["matching"]["held_from"]["train_size"] == 16 and r["config"]["data"]["train_size"] == 32
    assert "32" in r["readings"]["cells"] and "eps_inf" in r["readings"]["cells"]["32"]["test"]
    with open(out / "M_16C2pm_us_d20_seed0.params.pkl", "rb") as fh:
        trained = pickle.load(fh)["readout"]
    for a, b in zip(jax.tree_util.tree_leaves(trained), jax.tree_util.tree_leaves(readout)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_job16_configs_agree():
    """the first stages are job 9's (job 7's training, traced) and job 8's D' at 128 and 256 molecules, with the same
    atoms per batch and chunk; the polishes are jobs 10-12's from them; M_inf at 64 is job 10's polish of job 7."""
    load = lambda n: E4.load_config(os.path.join(REPO, "configs", f"{n}.yaml"))
    j9, j10, j11, j8d, j12 = (load(n) for n in ("e4_job9", "e4_job10", "e4_job11", "e4_job8d", "e4_job12"))
    same = lambda a, b, skip=(): {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}
    for name, base, size, batch, rungs, start, maxm, frames in (
            ("e4_job16s_128", j9, 128, 3, ["M_inf", "M_16pm_us_d20", "M_1w050", "M_1w150"], None, None, 2),
            ("e4_job16_128", j11, 128, 3, ["M_inf", "M_16pm_us_d20", "M_1w050", "M_1w150"],
             {"dir": "results/e4_job16s_128"}, 2400, 2),
            ("e4_job16s_128d", j8d, 128, 3, ["M_16C2pm_us_d20"], None, None, None),
            ("e4_job16_128d", j12, 128, 3, ["M_16C2pm_us_d20"], {"dir": "results/e4_job16s_128d"}, 2400, 2),
            ("e4_job16s_256", j9, 256, 1, ["M_1w050", "M_1w150"], None, None, 1),
            ("e4_job16_256", j11, 256, 1, ["M_1w050", "M_1w150"], {"dir": "results/e4_job16s_256"}, 3000, 1)):
        c = load(name)
        assert c["data"] == {**base["data"], "train_size": size} and c["rungs"] == rungs and c["seeds"] == [0, 1], name
        assert c["model"] == base["model"] and c["probes"] == base["probes"] and c["reading"] == base["reading"], name
        t, b = c["training"], base["training"]
        assert t["adam"] == {**b["adam"], "batch_size": batch} and t["start_from"] == start, name
        assert t["lbfgs"] == {**b["lbfgs"], "chunk_size": batch, **({"max_minutes": maxm} if maxm else {})}, name
        assert same(t, b, ("adam", "lbfgs", "start_from", "trace", "readout")), name
        ho = b["readout"]["held_from"]
        assert t["readout"] == {**b["readout"], **({"held_from": {**ho, "train_size": 64}} if ho else {})}, name
        if frames is not None:
            assert t["trace"]["frames"] == frames and t["trace"]["lbfgs_every"] == 25, name
    assert batch * 256 == 256 and 3 * 128 == 6 * 64                 # the same atoms per Adam batch at 128
    c = load("e4_job16_64")
    assert c["rungs"] == ["M_inf"] and same(c, j10, ("rungs",))
    assert J16.DIRS["dir64"] == "results/e4_job16_64" and J16.DIRS["dir128d"] == "results/e4_job16_128d"


def _cells(sizes, L0, errors, force):
    """readings by size and split: frames 2 / 2 / 3 (train / valid / test) with the given per-frame energy errors
    (a function of size and split), the rigid forces' rmse a function of size."""
    out = {}
    for n in sizes:
        out[n] = {}
        for split, nf in (("train", 2), ("valid", 2), ("test", 3)):
            e = [errors(n, split)] * nf
            out[n][split] = {"frames": nf, "energy_per_molecule_errors": e, "energy_per_molecule_rmse": abs(e[0]),
                             "force_rigid_rmse": force(n)}
    return out


def _run(rung, seed, L0, errors, force=lambda n: 0.5, dipole=None, eps=None):
    cells = _cells(("64", "128", "256"), L0, errors, force)
    if dipole is not None:
        cells["64"]["test"]["learned_sources"] = {"dipole_debye": [dipole, 0.02]}
    if eps is not None:
        cells[str(L0)]["test"]["eps_inf"] = {"isotropic_mean": eps}
    return {"rung": rung, "seed": seed, "readings": {"cells": cells}, "training": {"final_loss": {"total": 30.0}}}


def _job16(lr=-0.009, lr128=None, model_shift=0.0, model_f=1.0, gaps=((0.67, 0.32), (0.60, 0.29), (0.55, 0.10)),
           seed_noise=0.0002):
    """constructed runs: the dagger's error per molecule +0.001 everywhere; M_inf's the dagger's plus the training
    cell's long-range energy, lr (kcal/mol per molecule) per step of size away from L_0 = 64 (at 128 lr128, default
    -lr: the same energy carried the other way); E4's model's bias -0.004 (+ model_shift at L_0 = 128), its forces 0.45
    (times model_f at 128); the warm starts' ends 1.0 -/+ gap / 2 by cell (64, 128, 256)."""
    lr128 = -lr if lr128 is None else lr128
    D = J16.J7.PERMANENT_DEBYE
    add = lambda d, r: d.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    R = {k: {} for k in J16.DIRS}
    for s in J16.SEEDS:
        eps = (-1) ** s * seed_noise
        dag = lambda n, split: 0.001 + eps
        ref64 = lambda n, split: 0.001 + eps + {"64": 0.0, "128": lr, "256": 1.1 * lr}[n]
        ref128 = lambda n, split: 0.001 + eps + {"64": lr128, "128": 0.0, "256": 0.1 * lr}[n]
        add(R["dir64"], _run("M_inf", s, 64, ref64))
        add(R["job11"], _run("M_16pm_us_d20", s, 64, dag))
        add(R["job12"], _run("M_16C2pm_us_d20", s, 64, lambda n, split: -0.004, force=lambda n: 0.45, eps=1.6893))
        add(R["dir128"], _run("M_inf", s, 128, ref128))
        add(R["dir128"], _run("M_16pm_us_d20", s, 128, dag))
        add(R["dir128d"], _run("M_16C2pm_us_d20", s, 128, lambda n, split: -0.004 + model_shift,
                               force=lambda n: 0.45 * model_f, eps=1.69))
        for (stage, pol), (g_stage, g_end) in zip((("job9", "job10"), ("stage128", "dir128"), ("stage256", "dir256")),
                                                  ((0.67, gaps[0][1]), (0.6, gaps[1][1]), (0.5, gaps[2][1]))):
            for w, sign in (("M_1w050", -1), ("M_1w150", 1)):
                add(R[pol], _run(w, s, 64, dag, force=lambda n: 0.55, dipole=(1.0 + sign * g_end / 2) * D))
                add(R[stage], _run(w, s, 64, dag, force=lambda n: 0.7, dipole=(1.0 + sign * g_stage / 2) * D))
    return R


def test_the_job16_readings():
    """K16.1-K16.5 on constructed jsons, then each band's edge and an arm missing."""
    ev = J16.evaluate(_job16())
    k = ev["K16.1"]
    assert k["s64_128"] == pytest.approx(-0.009) and k["s128_64"] == pytest.approx(0.009) and k["holds"]
    assert k["ratio"] == pytest.approx(1.0) and k["s64_256"] == pytest.approx(-0.0099)
    assert ev["delta"][128]["64"]["se"] == pytest.approx(0.0) and ev["delta"][64]["128"]["sd"] == pytest.approx(0.0)
    assert ev["K16.2"]["holds"] and ev["K16.3"]["holds"] and ev["K16.2"]["diff"]["256"] == pytest.approx(0.0)
    assert ev["K16.4"]["verdict"] == "frozen" and ev["K16.5"]["verdict"] == "mobile"
    assert ev["cell"]["64"]["gaps"][0] == pytest.approx(0.32) and ev["cell"]["128"]["stage_gaps"][1] == pytest.approx(0.6)
    assert ev["eps_inf"] == {64: pytest.approx(1.6893), 128: pytest.approx(1.69)}
    assert ev["bias"][128]["M_inf"]["64"][0] == pytest.approx(0.010)
    text = J16.report(ev)
    for t in ("K16.1: holds", "K16.4: frozen", "K16.5: mobile", "coverage 82 %", "L_0 = 128 M_16C2pm_us_d20"):
        assert t in text, t
    json.dumps(ev)
    # the edges
    assert J16.evaluate(_job16(lr128=0.0045))["K16.1"]["holds"] is False                   # half the energy: 50 % off
    assert J16.evaluate(_job16(lr128=0.0055))["K16.1"]["holds"] is True                    # 39 % off
    assert J16.evaluate(_job16(lr128=-0.009))["K16.1"]["holds"] is False                   # the same sign
    k = J16.evaluate(_job16(lr=-0.002))["K16.1"]
    assert k["holds"] is None and k["too_small"]
    assert J16.evaluate(_job16(model_shift=0.0035))["K16.2"]["holds"] is False
    assert J16.evaluate(_job16(model_f=1.06))["K16.3"]["holds"] is False
    assert J16.evaluate(_job16(gaps=((0.67, 0.32), (0.4, 0.20), (0.5, 0.30))))["K16.4"]["verdict"] == "partial"
    assert J16.evaluate(_job16(gaps=((0.67, 0.32), (0.4, 0.30), (0.5, 0.30))))["K16.5"]["verdict"] == "frozen"
    R = _job16()
    for k in ("dir128", "dir128d", "dir256"):
        R[k] = {}
    ev = J16.evaluate(R)
    assert all(ev[k]["holds"] is None for k in ("K16.1", "K16.2", "K16.3"))
    assert ev["K16.4"]["verdict"] is None and ev["K16.5"]["verdict"] is None and "not read" in J16.report(ev)
