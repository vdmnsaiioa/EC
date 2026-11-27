"""
E4 job 15 (v0.0.40; notes/e4-job15-preregistration.md): E_0 grown by a message-passing block that adds nothing at the
start (e4_water.grown, training.start_from.grow), job 7's scan resolved between 0.9 and 1.3, the six configs, the
readings K15.1-K15.10 (scripts/e4_job15.py), and MACE-LES's latent dipoles (baselines/mace/eval.py).
"""
import json
import os
import pickle
import sys

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, train as T, structure as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, os.path.join(REPO, "baselines", "mace"))
import e4_water as E4                                     # noqa: E402
import e4_job15 as J15                                    # noqa: E402
import eval as LES_EVAL                                   # noqa: E402  (baselines/mace/eval.py; torch only inside)
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402
from test_job7 import _run                                      # noqa: E402
from test_job9 import _cells                                    # noqa: E402
from test_job10 import _train                                   # noqa: E402

KW = dict(seed=0, lr=3e-3, w_energy=100.0, w_force=1.0, batch_size=2, force_mode="rigid", lbfgs_chunk=2)
leaves = jax.tree_util.tree_leaves


def _rungs(name, st, cfg):
    one = E4.merge(cfg, {"model": {"layers": 1}})
    return tuple(E4.warm_started(name, E4.rungs_for(name, c, st[0].cell)[0], st) for c in (one, cfg))


def test_a_grown_run_is_the_run_it_grew_from(tmp_path):
    """grown: a one-layer run's parameters with E_0 grown to two blocks give that run's energies and forces exactly,
    learned charges or pinned; the new block's radial weights and its update's last layer are zero, everything else is
    the run's, and the new block is reached by the gradient.  A run that has the blocks already is refused."""
    st = _cells(tmp_path)
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8}})
    for name in ("M_1w150", "M_1q110"):
        r1, r2 = _rungs(name, st, cfg)
        assert r1.e0["T"] == 1 and r2.e0["T"] == 2
        p1, _ = T.fit(r1, st, steps=3, lbfgs_steps=2, **KW)
        p2 = E4.grown(p1, r2, jax.random.PRNGKey(5))
        d = S.pad_batch(st, n_freq=r1.K)
        E1, F1 = (np.asarray(x) for x in M.energy_forces(p1, r1, d))
        E2, F2 = (np.asarray(x) for x in M.energy_forces(p2, r2, d))
        assert np.max(np.abs(E2 - E1)) <= 1e-12 * np.max(np.abs(E1)) and np.max(np.abs(F2 - F1)) <= 1e-12 * np.max(np.abs(F1))
        new = p2["e0"]["blocks"][1]
        assert not np.any(np.asarray(new["W"])) and not np.any(np.asarray(new["upd"][-1]["w"]))
        assert not np.any(np.asarray(new["upd"][-1]["b"])) and np.any(np.asarray(new["phi"][0]["w"]))
        assert all(np.array_equal(np.asarray(a), np.asarray(b))
                   for a, b in zip(leaves({k: v for k, v in p1.items() if k != "e0"}),
                                   leaves({k: v for k, v in p2.items() if k != "e0"})))
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(leaves(p1["e0"]["blocks"][0]),
                                                                                 leaves(p2["e0"]["blocks"][0])))
        grad = jax.grad(lambda p: jax.numpy.sum(M.energy_forces(p, r2, d)[1] ** 2))(p2)["e0"]["blocks"][1]
        assert np.any(np.asarray(grad["W"])) and np.any(np.asarray(grad["upd"][-1]["w"]))
        with pytest.raises(ValueError, match="blocks already"):
            E4.grown(p2, r2, jax.random.PRNGKey(5))


def test_driver_grows_a_run(tmp_path):
    """training.start_from.grow: a run starts from the same rung's run with one block fewer, grown, at the source's
    objective (checked against its json; a mismatch stops the run); the json records it and the two-layer model.  A
    source at another cutoff, a rung of one layer, and a grow that is not a boolean are refused."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["seeds"], cfg["rungs"] = [0], ["M_1w150", "M_1q110"]
    cfg["model"].update(features=8, layers=1)
    cfg["training"]["adam"].update(steps=2, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=1, chunk_size=2)
    one = tmp_path / "one"
    _train(tmp_path, cfg, one)
    cfg["model"]["layers"] = 2
    cfg["training"]["adam"]["steps"] = 0
    cfg["training"]["start_from"] = {"dir": str(one), "grow": True}
    cfg["training"]["trace"] = {"every": 100, "frames": 1, "lbfgs_every": 1}
    two = tmp_path / "two"
    _train(tmp_path, cfg, two)
    for name in cfg["rungs"]:
        r1, r2 = (json.load(open(d / f"{name}_seed0.json")) for d in (one, two))
        b = r2["training"]["start"]
        assert b["from"] == str(one / f"{name}_seed0.params.pkl") and b["grown"]["layers"] == [1, 2]
        assert b["grown"]["against_source"] <= E4.GROW_TOL and "from_rung" not in b
        assert b["final_loss"]["total"] == pytest.approx(r1["training"]["final_loss"]["total"], rel=1e-10)
        assert r1["model"]["layers"] == 1 and "layers" not in r2["model"]
        with open(two / f"{name}_seed0.params.pkl", "rb") as fh:
            assert len(pickle.load(fh)["e0"]["blocks"]) == 2
    assert two.joinpath("M_1w150_seed0.json").exists()
    rw = json.load(open(two / "M_1w150_seed0.json"))
    assert rw["training"]["charges"]["lbfgs_trace"][0]["objective"] == pytest.approx(
        rw["training"]["start"]["final_loss"]["total"], rel=1e-10)
    # a source whose recorded objective is not the grown start's
    cfg["rungs"] = ["M_1q110"]
    bad = tmp_path / "bad"
    bad.mkdir()
    r = json.load(open(one / "M_1q110_seed0.json"))
    r["training"]["final_loss"]["total"] *= 1.001
    json.dump(r, open(bad / "M_1q110_seed0.json", "w"))
    os.link(one / "M_1q110_seed0.params.pkl", bad / "M_1q110_seed0.params.pkl")
    cfg["training"]["start_from"] = {"dir": str(bad), "grow": True}
    with pytest.raises(RuntimeError, match="grown start's objective"):
        _train(tmp_path, cfg, tmp_path / "again")
    cfg["training"]["start_from"] = {"dir": str(one), "grow": True}
    cfg["model"]["r_cut_angstrom"] = 4.5                               # the source was trained at 6 A
    with pytest.raises(ValueError, match="another model"):
        _train(tmp_path, cfg, tmp_path / "again2")
    cfg["model"].update(r_cut_angstrom=6.0, layers=1)
    with pytest.raises(ValueError, match="growing needs two"):
        _train(tmp_path, cfg, tmp_path / "again3")
    cfg["model"]["layers"] = 2
    cfg["training"]["start_from"]["grow"] = "yes"
    with pytest.raises(ValueError, match="boolean"):
        _train(tmp_path, cfg, tmp_path / "again4")


def test_job15_configs_rungs_and_reader_agree():
    """A is job 14's 4.5 A settings for the resolved scan and the warm starts by species alone; B continues job 14's 6 A
    warm starts with its polish; C is job 14's settings at two layers of 3 A; D grows B's warm starts and job 14's 6 A
    M_1q110 to two layers of 6 A with the same polish; the new scan rungs pin c."""
    load = lambda n: E4.load_config(os.path.join(REPO, "configs", f"{n}.yaml"))
    s45, p45, p6 = load("e4_job14s_r45"), load("e4_job14_r45"), load("e4_job14_r6")
    rest = lambda c: {k: v for k, v in c.items() if k not in ("rungs", "model", "training")}
    tr = lambda c: {k: v for k, v in c["training"].items() if k != "start_from"}
    for name, base, model, rungs, start in (
            ("e4_job15s_a", s45, s45["model"], list(J15.NEW_A) + list(J15.SPECIES), None),
            ("e4_job15_a", p45, p45["model"], list(J15.NEW_A) + list(J15.SPECIES), {"dir": "results/e4_job15s_a"}),
            ("e4_job15_b", p6, p6["model"], list(J15.FLEX), {"dir": "results/e4_job14_r6"}),
            ("e4_job15s_c", s45, {**s45["model"], "r_cut_angstrom": 3.0, "layers": 2}, s45["rungs"], None),
            ("e4_job15_c", p45, {**p45["model"], "r_cut_angstrom": 3.0, "layers": 2}, p45["rungs"],
             {"dir": "results/e4_job15s_c"}),
            ("e4_job15_d", p6, {**p6["model"], "layers": 2}, list(J15.FLEX) + ["M_1q110"],
             {"dir": "results/e4_job15_b", "by_rung": {"M_1q110": "results/e4_job14_r6"}, "grow": True})):
        c = load(name)
        assert c["model"] == model and c["rungs"] == rungs and c["seeds"] == [0, 1], name
        assert rest(c) == rest(base) and tr(c) == tr(base) and c["training"]["start_from"] == start, name
    for name in J15.NEW_A:
        r = E4.E3.RUNGS[name]
        assert r.q_scale == J15.PROFILE_A[name] and r.pin_q and E4.scan_rung(J15.PROFILE_A[name]) == name
    assert set(J15.SPECIES) <= set(E4.SPECIES_ONLY) and all(E4.STARTS[n] == c for n, c in J15.SPECIES.items())
    assert J15.DIRS["a"] == "results/e4_job15_a" and J15.DIRS["d"] == "results/e4_job15_d"


def _parabola(names, a, c0, L0=40.0):
    return {n: L0 + a * (J15.PROFILE_A[n] - c0) ** 2 for n in names}


def _les_json(path, c, forces=0.9, latent=True):
    """an evaluation json; its bias -0.01 c^2 per molecule at 128 molecules and -0.02 c^2 at 256 above 64's (+0.001)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sets = {n: {"f_rmse_kcal_per_mol_A": forces, "e_rmse_kcal_per_mol": 0.03, "e_mean_kcal_per_mol": 0.001 - k * c ** 2}
            for n, k in (("valid_n64", 0.0), ("test_n64", 0.0), ("test_n128", 0.01), ("test_n256", 0.02))}
    if latent:
        sets["test_n64"].update(latent_q_O_mean=-0.6, latent_q_H_mean=0.3, latent_q_per_molecule_rms=1e-3,
                                latent_dipole_debye_mean=c * J15.PERMANENT_DEBYE)
    json.dump({"sets": sets}, open(path, "w"))


def _job15(tmp_path, a_c0=1.07, species=((1.04, 1.08), (1.05, 1.09)), b_ends=((0.96, 0.975), (0.95, 0.98)),
           c_ends=((0.93, 1.0), (0.94, 1.02)), d_shift=0.01, d_forces=0.54, q110=0.525, e1=(0.98, 1.04), e2=(0.97, 1.02)):
    """constructed jsons for every arm; ends: (from 0.5, from 1.5) by seed, in c."""
    D = J15.PERMANENT_DEBYE
    sizes = J15.SIZES
    add = lambda d, r: d.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    R = {k: {} for k in J15.DIRS if k not in ("baselines", "les")}
    old = [n for n in J15.PROFILE_A if n not in J15.NEW_A]
    for s in J15.SEEDS:
        for n, L in _parabola(old, 150.0, a_c0).items():
            add(R["r45"], _run(n, s, {x: 0.65 for x in sizes}, loss=L))
        for n, L in _parabola(J15.NEW_A, 150.0, a_c0).items():
            add(R["a"], _run(n, s, {x: 0.64 for x in sizes}, loss=L))
        for w, c in zip(J15.FLEX, ((0.974, 0.975), (0.987, 0.969))[s]):
            add(R["r45"], _run(w, s, {x: 0.6 for x in sizes}, dipole=c * D))
        for w, c in zip(J15.SPECIES, species[s]):
            add(R["a"], _run(w, s, {x: 0.66 for x in sizes}, dipole=c * D))
            add(R["a_stage"], _run(w, s, {x: 0.7 for x in sizes}, dipole=(c - 0.1) * D))
        for w, c in zip(J15.SPECIES, ((0.632, 1.347), (0.625, 1.342))[s]):
            add(R["job13"], _run(w, s, {x: 0.6 for x in sizes}, dipole=c * D))
        for w, c in zip(J15.FLEX, ((0.963, 0.975), (0.932, 1.035))[s]):
            add(R["r6"], _run(w, s, {x: 0.6 for x in sizes}, dipole=c * D))
        for w, c in zip(J15.FLEX, b_ends[s]):
            add(R["b"], _run(w, s, {x: 0.60 for x in sizes}, dipole=c * D))
            add(R["d"], _run(w, s, {x: d_forces for x in sizes}, dipole=(c + d_shift) * D))
        add(R["d"], _run("M_1q110", s, {x: q110 for x in sizes}))
        add(R["job10"], _run("M_1q110", s, {x: 0.52 for x in sizes}, loss=(21.09, 21.82)[s]))
        for n, L in (("M_1q070", 30.75), ("M_1q100", 23.63), ("M_1q145", 31.44)):
            add(R["job11"], _run(n, s, {x: 0.6 for x in sizes}, loss=L))
        for n, L in (("M_1q070", 45.0), ("M_1q100", 36.0), ("M_1q110", 35.0), ("M_1q145", 50.0)):
            add(R["c"], _run(n, s, {x: 0.75 for x in sizes}, loss=L))
        for w, c in zip(J15.FLEX, c_ends[s]):
            add(R["c"], _run(w, s, {x: 0.72 for x in sizes}, dipole=c * D))
            add(R["c_stage"], _run(w, s, {x: 0.8 for x in sizes}, dipole=(0.5 + 0.8 * (c - 0.5)) * D))
    bl, les = tmp_path / "baselines", tmp_path / "les"
    for s in J15.SEEDS:
        _les_json(str(bl / f"maceles_n64_s{s}" / f"maceles_n64_s{s}_stagetwo_eval.json"), 1.2 + 0.1 * s)
        _les_json(str(les / f"release_o100_n64_s{s}" / f"maceles_n64_s{s}_stagetwo_eval.json"), 1.25 + 0.1 * s)
        for u, k in (("o050", 0), ("o150", 1)):
            _les_json(str(les / f"maceles_{u}_n64_s{s}" / f"maceles_{u}_n64_s{s}_stagetwo_eval.json"),
                      e1[k] * (1.2 + 0.1 * s))
            _les_json(str(les / f"release_{u}_n64_s{s}" / f"maceles_n64_s{s}_stagetwo_eval.json"),
                      e2[k] * (1.25 + 0.1 * s))
    R["les"] = J15.les_runs(str(bl), str(les))
    return R


def test_the_job15_readings(tmp_path):
    """K15.1-K15.10 on constructed jsons, then each band's edge and an arm missing."""
    ev = J15.evaluate(_job15(tmp_path))
    p = ev["profile_a"]
    assert p["c_min"] == pytest.approx(1.07) and p["inside"] and p["window"] == [0.95, 1.0, 1.05, 1.1, 1.2]
    assert p["curvature"] == pytest.approx(1.07 ** 2 * 300.0) and p["lowest"] == 1.05
    assert ev["K15.1"]["holds"] and ev["K15.1"]["above"] == pytest.approx(1.07 - 0.987)
    for k in ("K15.2", "K15.3", "K15.4", "K15.6", "K15.7", "K15.8", "K15.9", "K15.10"):
        assert ev[k]["holds"], k
    assert ev["K15.5"]["verdict"] == "reach" and ev["K15.5"]["gaps"][1] == pytest.approx(0.08)
    assert ev["K15.2"]["off"]["M_1z150 seed 1"] == pytest.approx(0.02)
    assert ev["K15.3"]["gaps"]["12"][0] == pytest.approx(1.347 - 0.632)
    assert ev["K15.7"]["against_b"]["M_1w050"]["64"] == pytest.approx(0.54 / 0.60 - 1)
    assert ev["K15.8"]["against_job10"]["256"] == pytest.approx(0.525 / 0.52 - 1)
    assert ev["K15.9"]["ratios"]["o150 seed 1"] == pytest.approx(1.04)
    assert ev["K15.10"]["back"]["o050 seed 0"] == pytest.approx((0.97 - 0.5) / 0.5)
    assert ev["les"]["E1"]["o100"][1]["c"] == pytest.approx(1.3) and ev["les"]["E1"]["o100"][1]["q_O"] == -0.6
    far = ev["les_far"]                                                  # the bias's step: -0.02 c^2 at 256
    e1_c = [r["c"] for by in ev["les"]["E1"].values() for r in by.values()]
    assert far["E1"]["test_n256"]["runs"] == 6 and far["E1"]["test_n256"]["mean"] == pytest.approx(
        -0.02 * np.mean(np.square(e1_c))) and far["E1"]["test_n128"]["mean"] == pytest.approx(-0.01 * np.mean(np.square(e1_c)))
    assert ev["les"]["E2"]["o050"][0]["steps"]["test_n256"] == pytest.approx(-0.02 * (0.97 * 1.25) ** 2)
    for u, k in (("o050", 0.97), ("o150", 1.02)):                         # a step in c^2: the two ratios agree
        v = far["against_control"][u]
        assert v["c_squared"] == pytest.approx(k ** 2) and v["step"] == pytest.approx(k ** 2)
    assert ev["les_stale"] == []
    assert ev["profile_c"]["ratio"] is not None and ev["paths"]["B M_1w050 seed 0"]["end_c"] == pytest.approx(0.96)
    text = J15.report(ev)
    for k in ("K15.1: holds", "K15.5: reach", "K15.9: holds", "K15.10: holds", "C's profile", "MACE-LES E2 o150 seed 1",
              "bias valid_n64 +0.0010", "MACE-LES's far field", "the step 0.94 times, c^2 0.94"):
        assert k in text, k
    assert "has no latent dipole" not in text
    json.dumps(ev)
    # the edges
    assert J15.evaluate(_job15(tmp_path, a_c0=1.01))["K15.1"]["holds"] is False                 # 0.023 above
    assert J15.evaluate(_job15(tmp_path, species=((0.99, 1.08), (1.05, 1.09))))["K15.2"]["holds"] is False
    assert J15.evaluate(_job15(tmp_path, species=((0.68, 1.08), (1.05, 1.09))))["K15.3"]["holds"] is False  # 0.40 > 0.357
    assert J15.evaluate(_job15(tmp_path, b_ends=((0.96, 0.975), (0.93, 0.99))))["K15.4"]["holds"] is False
    assert J15.evaluate(_job15(tmp_path, c_ends=((0.80, 1.10), (0.80, 1.12))))["K15.5"]["verdict"] == "depth"
    assert J15.evaluate(_job15(tmp_path, c_ends=((0.80, 1.10), (0.94, 1.02))))["K15.5"]["verdict"] == "mixed"
    assert J15.evaluate(_job15(tmp_path, d_shift=0.06))["K15.6"]["holds"] is False
    assert J15.evaluate(_job15(tmp_path, d_forces=0.575))["K15.7"]["holds"] is False            # -4.2 %
    assert J15.evaluate(_job15(tmp_path, q110=0.54))["K15.8"]["holds"] is False                 # +3.8 %
    assert J15.evaluate(_job15(tmp_path, e1=(0.6, 1.04)))["K15.9"]["holds"] is False
    assert J15.evaluate(_job15(tmp_path, e2=(0.97, 1.3)))["K15.10"]["holds"] is False
    R = _job15(tmp_path)                                                  # a baseline evaluated before v0.0.40
    old = str(tmp_path / "baselines" / "maceles_n64_s0" / "maceles_n64_s0_stagetwo_eval.json")
    _les_json(old, 1.2, latent=False)
    R["les"] = J15.les_runs(str(tmp_path / "baselines"), str(tmp_path / "les"))
    ev = J15.evaluate(R)
    assert ev["les_stale"] == [old] and 0 not in ev["les"]["E1"]["o100"] and ev["K15.9"]["holds"] is None
    assert ev["K15.10"]["holds"] and ev["les_far"]["E1"]["test_n256"]["runs"] == 5
    assert f"MACE-LES: {old} has no latent dipole" in J15.report(ev)
    ev = J15.evaluate(_job15(tmp_path, a_c0=1.6))                                              # c* beyond the window
    assert ev["K15.1"]["holds"] is False and ev["K15.2"]["holds"] is None
    R = _job15(tmp_path)
    for k in ("a", "b", "c", "d"):
        R[k] = {}
    R["les"] = {"E1": {}, "E2": {}}
    ev = J15.evaluate(R)
    assert all(ev[k]["holds"] is None for k in ("K15.1", "K15.2", "K15.3", "K15.4", "K15.6", "K15.7", "K15.8", "K15.9",
                                                "K15.10")) and ev["K15.5"]["verdict"] is None
    assert "not read" in J15.report(ev)


def test_les_latent_dipoles():
    """baselines/mace/eval.py's latent_dipoles: each molecule's net charge spread over its atoms, its H atoms taken to
    their O by the minimum image; e A, so 4.803 D per e A."""
    cell = np.diag([10.0, 10.0, 12.0])
    pos = np.array([[0.1, 0.1, 0.1], [9.9, 0.1, 0.1], [0.1, 9.95, 0.1],
                    [5.0, 5.0, 11.9], [5.0, 5.0, 0.5], [5.8, 5.0, 11.9]])
    q = np.array([-0.8, 0.4, 0.4, -0.6, 0.35, 0.35])
    mu = LES_EVAL.latent_dipoles(q, pos, cell)
    assert mu[0] == pytest.approx([0.4 * -0.2, 0.4 * -0.15, 0.0])
    qh = 0.35 - 0.1 / 3                                                              # its net +0.1 e spread
    assert mu[1] == pytest.approx([qh * 0.8, 0.0, qh * 0.6])
    assert LES_EVAL.latent_dipoles(q + 0.3, pos, cell) == pytest.approx(mu)            # net charge spread evenly
    assert LES_EVAL.latent_dipoles(q, pos + np.array([10.0, 0, 0]) * [[1], [0], [0], [0], [0], [0]], cell) \
        == pytest.approx(mu)                                                           # an O taken into the next cell
    assert LES_EVAL.DEBYE_PER_E_ANGSTROM == pytest.approx(4.80320, abs=1e-5)
