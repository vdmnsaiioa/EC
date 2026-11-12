"""
E4's MD driver (scripts/e4_md.py; notes/e4-s1-preregistration.md): a run continued from its file reproduces the
uninterrupted one exactly; a run whose forces stop being finite stops and says so; a trained model's engine gives
kcal/mol and kcal/mol/A forces that are its energy's gradient, and its polarisation at the atoms; the read evaluates
S1.1 and S2.1 on trajectories written in the driver's format.
"""
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import e4_md as MD  # noqa: E402
import e4_sumrules as SR  # noqa: E402
from test_rigidmd import toy_force  # noqa: E402

GAMMA = 0.4267
SITE_Q = np.array([0.0, 0.55, 0.55, -1.10])


class ToyEngine:
    """the rigid-MD tests' toy potential, with a polarisation that is a smooth function of the positions."""

    def __init__(self, n, nan_after=None):
        self.f, self.calls, self.nan_after = toy_force(n), 0, nan_after
        self.info = {"engine": "toy"}

    def force(self, pos, cell):
        self.calls += 1
        E, F = self.f(pos, cell)
        if self.nan_after is not None and self.calls > self.nan_after:
            F = F * np.nan
        return E, F

    def sites(self, pos, cell):
        xyz = SR.site_positions(np.asarray(pos)[None], GAMMA)[0]
        return xyz, np.tile(SITE_Q, len(pos) // 3), 0.01 * np.sin(xyz)


def _box(n=16, side=12.0, seed=0):
    from sseft.mbpol import random_box
    cell = np.eye(3) * side
    return random_box(n, cell, np.random.default_rng(seed), min_oo=2.6, min_any=1.5), cell


def test_a_continued_run_reproduces_the_uninterrupted_one(tmp_path):
    """150 steps (50 of equilibration), a frame every 10: once straight through, once stopped after 73 steps and
    continued from its file -- the frames, the state and the log agree to the last bit, the centre-of-mass removal at
    step 100 included."""
    pos, cell = _box()
    kw = dict(equil_ps=0.1, sample_ps=0.2, save_every=10, ckpt_min=1e9, max_hours=1e9, say=lambda *x: None)
    seq = lambda: np.random.SeedSequence([0, 16, 7])
    a = MD.simulate(ToyEngine(16), pos, cell, str(tmp_path / "a.npz"), seq(), {"n": 16, "seed": 0}, **kw)
    m1 = MD.simulate(ToyEngine(16), pos, cell, str(tmp_path / "b.npz"), seq(), {"n": 16, "seed": 0},
                     stop_after_steps=73, **kw)
    assert not m1["completed"] and m1["step"] == 73
    b = MD.simulate(ToyEngine(16), pos, cell, str(tmp_path / "b.npz"), seq(), {"n": 16, "seed": 0}, **kw)
    assert a["completed"] and b["completed"] and b["continued"] == 1 and a["step"] == b["step"] == 150
    da, db = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
    assert da["time_ps"].shape == (10,) and da["mu"].shape == (10, 64, 3)
    for k in ("positions", "xyz", "q", "mu", "time_ps", "X", "V", "log"):
        assert np.array_equal(da[k], db[k]), k
    again = MD.simulate(ToyEngine(16), pos, cell, str(tmp_path / "a.npz"), seq(), {"n": 16, "seed": 0}, **kw)
    assert again["completed"] and again["step"] == 150                      # a completed run is left alone


def test_a_run_stops_when_its_forces_stop_being_finite(tmp_path):
    pos, cell = _box()
    kw = dict(equil_ps=0.02, sample_ps=0.04, save_every=5, ckpt_min=1e9, max_hours=1e9, say=lambda *x: None)
    m = MD.simulate(ToyEngine(16, nan_after=12), pos, cell, str(tmp_path / "c.npz"), 0, {"n": 16, "seed": 0}, **kw)
    assert not m["completed"] and "non-finite" in m["stopped"] and m["step"] == 12
    d = np.load(tmp_path / "c.npz")
    assert json.loads(str(d["meta"]))["stopped"] == m["stopped"]
    again = MD.simulate(ToyEngine(16), pos, cell, str(tmp_path / "c.npz"), 0, {"n": 16, "seed": 0}, **kw)
    assert again["stopped"] == m["stopped"]                                   # not continued


def test_model_engine_forces_are_its_energy_gradient_in_md_units(tmp_path):
    """the frozen form on a fake 16-molecule cell (init parameters with the read-out switched on, as "trained"): the
    forces are -dE/dx in kcal/mol/A to the finite difference's accuracy, the energy is the model's interaction energy
    in kcal/mol, and a saved frame carries the pinned charges and -dE/dE_ext,i at the atoms, nothing at the M-sites."""
    import jax
    import yaml
    import e4_water as E4
    from sseft import model as M, water as W
    from sseft.units import KCALMOL_PER_HARTREE
    from test_e4_train import _fake_cells
    from test_periodic import switched_on
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8, "mutual_edge_angstrom": 2.0}})
    data = tmp_path / "data"; data.mkdir()
    _fake_cells(data / "test_n16.npz", 2, 16, 12.5, seed=5)
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    st = W.e4_structures(str(data / "test_n16.npz"), pins=True)
    rung_cell, rung_open = E4.rungs_for("M_16C2pm_us", cfg, st[0].cell)
    params = M.init_params(jax.random.PRNGKey(0), rung_open, energy_scale=1e-3)
    params = E4.with_readout(params, switched_on(rung_cell)["readout"])
    with open(tmp_path / "p.pkl", "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)
    a = SimpleNamespace(engine="model", config=str(tmp_path / "c.yaml"), params=str(tmp_path / "p.pkl"),
                        rung="M_16C2pm_us", data=str(data), label="Dp", mbx=None)
    eng, pos0, cell, meta = MD.build(a, 16, 0)
    assert eng.response and meta["start_frame"] == 0
    E, F = eng.force(pos0, cell)
    E_ref = float(M.energy_single(params, E4.in_cell(rung_cell, st[0].cell),
                                  {k: v[0] for k, v in __import__("sseft.structure", fromlist=["pad_batch"]).pad_batch(
                                      [st[0]], n_freq=rung_cell.K).items()})[0]) * KCALMOL_PER_HARTREE
    assert E == pytest.approx(E_ref, rel=1e-9, abs=1e-9)
    h = 1e-4
    for i, k in ((0, 0), (7, 2), (30, 1)):
        p, m = pos0.copy(), pos0.copy(); p[i, k] += h; m[i, k] -= h
        fd = -(eng.force(p, cell)[0] - eng.force(m, cell)[0]) / (2 * h)
        assert fd == pytest.approx(F[i, k], rel=2e-4, abs=2e-5)
    xyz, q, mu = eng.sites(pos0, cell)
    assert xyz.shape == (64, 3) and np.allclose(q, np.tile(q[:4], 16)) and np.all(mu[3::4] == 0)
    assert np.abs(mu).max() > 1e-5
    m = MD.simulate(eng, pos0, cell, str(tmp_path / "t.npz"), 0, meta, equil_ps=0.01, sample_ps=0.01, save_every=5,
                    ckpt_min=1e9, max_hours=1e9, say=lambda *x: None)
    d = np.load(tmp_path / "t.npz")
    assert m["completed"] and d["mu"].shape == (1, 64, 3) and np.isfinite(d["log"]).all()


def _write(path, frames, n, side, seed, label, rng, sampled_ps):
    pos = np.array([_mols(n, side, rng) for _ in range(frames)])
    xyz = SR.site_positions(pos, GAMMA)
    mu = rng.normal(scale=0.02, size=xyz.shape)
    t = 10.0 + 0.05 * np.arange(1, frames + 1)
    log = np.column_stack([t, np.full(frames, 300.0), np.zeros(frames), np.full(frames, 1e-12), 1e-4 * t])
    meta = {"label": label, "n": n, "seed": seed, "site_gamma": GAMMA, "cell": (np.eye(3) * side).tolist(),
            "equil_ps": 10.0, "sampled_ps": sampled_ps, "completed": True, "stopped": None}
    MD._save(str(path), {"positions": pos, "xyz": xyz, "q": np.broadcast_to(np.tile(SITE_Q, n), xyz.shape[:2]),
                         "mu": mu, "time_ps": t, "log": log}, meta)
    return pos, xyz, mu


def _mols(n, side, rng):
    from sseft.water import monomer_positions, random_rotation
    m = monomer_positions() - monomer_positions()[0]
    return np.concatenate([m @ random_rotation(rng).T + rng.uniform(0, side, 3) for _ in range(n)])


def test_read_evaluates_s11_and_s21_on_the_drivers_files(tmp_path):
    """two engines with the same frames read the same L, so S1.1's difference is 0 and it holds; the dagger's L is read
    for S2.1; trajectories below the minimum sampling make S1.1 'not read'; tau_L and each run's stability are there."""
    rng = np.random.default_rng(8)
    for n, side in ((64, 12.43), (128, 15.66)):
        for seed in (0, 1):
            r = np.random.default_rng(100 * n + seed)
            _write(tmp_path / f"traj_mbpol_n{n}_s{seed}.npz", 24, n, side, seed, "mbpol", r, 60.0)
            r = np.random.default_rng(100 * n + seed)
            _write(tmp_path / f"traj_Dp_n{n}_s{seed}.npz", 24, n, side, seed, "Dp", r, 60.0)
    _write(tmp_path / "traj_dagger_n64_s0.npz", 24, 64, 12.43, 0, "dagger", rng, 60.0)
    a = SimpleNamespace(out=str(tmp_path), model="Dp", truth="mbpol", dagger="dagger", min_sample_ps=50.0,
                        json=str(tmp_path / "s1.json"))
    bands, res = MD.run_read(a)
    assert bands["S1.1"]["read"] and bands["S1.1"]["holds"]
    assert set(bands["S1.1"]["rows"]) == {"n64 (100)", "n128 (100)", "n128 (110)"}
    for v in bands["S1.1"]["rows"].values():
        assert v["difference"] == 0.0 and v["allowed"] >= 0.02
    assert bands["S2.1"]["read"] and isinstance(bands["S2.1"]["holds"], bool)
    g = res[("Dp", 64)]
    assert np.isfinite(g["tau_L_100_ps"]) and len(g["trajectories"]) == 2
    assert g["trajectories"][0]["drift_kcal_per_ps_per_molecule"] == pytest.approx(1e-4 / 64)
    assert json.load(open(tmp_path / "s1.json"))["bands"]["S1.1"]["holds"]
    a.min_sample_ps = 100.0
    assert not MD.run_read(a)[0]["S1.1"]["read"]
