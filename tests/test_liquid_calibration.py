"""
Job 2's liquid calibration (v0.0.24): the model's induced dipole and field response of a cell (model.induced_response,
model.molecular_response), the read-out solve with further rows (linear.solve_readout's extra), and the driver's
liquid response -- MB-pol's through MBX (scripts/e4_water.py response), the calibration rows and the held-out readings --
on a fake truth whose answers are known.
"""
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, linear as LIN, structure as S
from sseft.units import ang_to_bohr as A2B

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from test_periodic import frozen, switched_on, cluster, water_structure, one, in_cell  # noqa: E402


def _cell(n=8, L=12.5, seed=3):
    pos = cluster(n, seed=seed, box=L - 2.0) + L / 2
    return water_structure(pos, np.eye(3) * L)


def test_induced_response_is_the_dielectric_and_the_molecules_sum_to_it():
    """R = -d2E/dE_ext2 is (eps_inf - 1) V / 4 pi; M = -dE/dE_ext; the molecules' response tensors sum to the cell's (the
    orders >= 1 add nothing to the response to a uniform field), with every read-out order switched on."""
    rung = replace(frozen(), bf_solver="direct", bf_l_mutual=A2B(2.0))
    params = switched_on(rung)
    b = one(_cell())
    rc = in_cell(rung, np.asarray(b["cell"]))
    Mi, R = (np.asarray(x) for x in M.induced_response(params, rc, b))
    eps = np.asarray(M.dielectric_inf(params, rc, b))
    V = float(jnp.abs(jnp.linalg.det(b["cell"])))
    assert np.allclose(eps, np.eye(3) + 4 * np.pi * R / V, rtol=1e-10, atol=1e-12)
    assert np.allclose(Mi, np.asarray(M.induced_dipole(params, rc, {**b, "E_ext": jnp.zeros(3)})), rtol=1e-10, atol=1e-12)
    mu_mol, R_mol = (np.asarray(x) for x in M.molecular_response(params, rc, b))
    assert R_mol.shape[1:] == (3, 3) and mu_mol.shape[1:] == (3,)
    assert np.allclose(R_mol.sum(0), R, rtol=1e-8, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(0.5 * (R + R.T)) > 0)


def _probe_data(rung, n=6, seed=11):
    """small open clusters with made-up energies, as the far-field matching sees them."""
    structs = []
    rng = np.random.default_rng(seed)
    for k in range(n):
        st = water_structure(cluster(3, seed=seed + k, box=5.5))
        st.energy = float(rng.normal() * 1e-3)
        structs.append(st)
    return S.pad_batch(structs, n_freq=rung.K)


def test_solve_readout_extra_rows():
    """without extra rows the solve of before; from the switched-on read-out, whose energies the probe data are, rows
    asking for 0.8 of a cell's own response move the cell's response most of the way there (the steps halved where they
    overshoot), and the report keeps that part of the objective."""
    rung = replace(frozen(), bf_solver="direct", bf_l_mutual=A2B(2.0))
    params = switched_on(rung)
    data = _probe_data(rung)
    data["energy"] = M.energy_forces(params, rung, data)[0]                    # the probes as this read-out sees them
    kw = dict(w_force=0.0, ridge=1e-10, chunk=6, w_energy=100.0, orders="all", iterations=3)
    p0, rep0 = LIN.solve_readout(params, rung, data, 1e-3, **kw)
    assert rep0.history[-1][1] < 1e-12 and rep0.extra_history == [(0.0, 0.0)] * len(rep0.history)
    b = one(_cell())
    rc = in_cell(rung, np.asarray(b["cell"]))
    fn = lambda p, bb: jnp.concatenate([x.reshape(-1) for x in M.induced_response(p, rc, bb)])
    y0 = np.asarray(fn(params, b))
    target = np.concatenate([y0[:3], 0.8 * y0[3:]])
    w = np.concatenate([np.zeros(3), np.full(9, 1e-3 / np.mean(target[3:] ** 2))])
    p2, rep2 = LIN.solve_readout(params, rung, data, 1e-3, extra={"fn": fn, "batches": [b], "targets": [target],
                                                                     "weights": [w]}, **kw)
    y2 = np.asarray(fn(p2, b))
    assert np.linalg.norm(y2[3:] - target[3:]) < 0.2 * np.linalg.norm(y0[3:] - target[3:])
    assert rep2.extra_history[-1][1] < 0.1 * rep2.extra_history[0][0]


class FakeMB:
    """a fake MBX: site dipoles mu0 (fixed per frame) plus alpha times the external field, alpha 1.5 A^3 on the O site."""
    def __init__(self):
        self.ext = np.zeros(3)

    def set_external_field(self, pos, field, cell=None):
        self.ext = np.zeros(3) if field is None else np.asarray(field, float)

    def sites(self, pos, cell=None):
        n = len(pos) // 3
        mu = np.zeros((n, 4, 3))
        mu[:, 0] = 0.01 * np.asarray(pos).reshape(n, 3, 3)[:, 0] + 1.5 * self.ext
        return {"induced": mu.reshape(-1, 3)}


def test_liquid_response_round_trip_and_rows(tmp_path):
    """liquid_response on a fake MBX: the molecules' dipoles and an isotropic 1.5 A^3 response; the response file round
    trips against its frames (atomic units), refuses changed frames, and its calibration rows are 12 numbers a cell
    whose targets are the file's cell sums."""
    import e4_water as E4
    from test_e4_train import _fake_cells
    for split, nf in (("train", 4), ("test", 2)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg = E4.merge(E4.load_config(None), {"data": {"dir": str(tmp_path), "train_size": 16, "response": str(tmp_path / "r.npz")},
                                          "liquid": {"calibration_frames": 3, "heldout_frames": 2}})
    frames = E4.response_frames(cfg)
    assert list(frames["calibration"][1]) == [0, 2, 3] and list(frames["heldout"][1]) == [0, 1]
    mu, R = E4.liquid_response(FakeMB(), np.load(tmp_path / "train_n16.npz")["positions"][0], np.eye(3) * 12.5)
    assert mu.shape == (16, 3) and np.allclose(R, 1.5 * np.eye(3)[None], atol=1e-9)
    import yaml
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)

    class Truth:
        mb = FakeMB()
        def describe(self):
            return {"name": "fake"}
    real = E4.W.MBPolTruth
    E4.W.MBPolTruth = lambda root: Truth()
    try:
        E4.run_response(SimpleNamespace(config=str(tmp_path / "c.yaml"), mbx="none", out=None))
    finally:
        E4.W.MBPolTruth = real
    resp = E4.load_response(str(tmp_path / "r.npz"), cfg)
    assert len(resp["calibration"]["structures"]) == 3 and resp["heldout"]["R"].shape == (2, 16, 3, 3)
    assert np.allclose(resp["calibration"]["R"][0], A2B(A2B(A2B(1.5))) * np.eye(3)[None])
    rung_cell, _ = E4.rungs_for("M_16C2pm_us", cfg, resp["calibration"]["structures"][0].cell)
    lb = E4.liquid_block(rung_cell, resp, {"weight": 0.1})
    blk = lb["block"]
    assert len(blk["batches"]) == 3 and blk["targets"][0].shape == (12,) and blk["weights"][0].shape == (12,)
    assert np.allclose(blk["targets"][0][3:].reshape(3, 3), resp["calibration"]["R"][0].sum(0))
    params = switched_on(rung_cell)
    assert np.asarray(blk["fn"](params, blk["batches"][0])).shape == (12,)
    d = np.load(tmp_path / "train_n16.npz")
    moved = dict(d); moved["positions"] = d["positions"] + 0.01
    np.savez(tmp_path / "train_n16.npz", **moved)
    with pytest.raises(ValueError):
        E4.load_response(str(tmp_path / "r.npz"), cfg)
    assert json.loads(str(np.load(tmp_path / "r.npz")["meta"]))["splits"]["calibration"]["frames"] == 3


def test_driver_with_the_liquid_calibration(tmp_path):
    """scripts/e4_water.py train with training.readout.liquid on fake cells, fake probes and a fake MBX's response, for
    the frozen form at width 8: the matching's second stage records its liquid rows and is cached under its own name,
    the held-out liquid readings are taken before and after training -- the same, the read-out held -- and MB-pol's
    eps_inf is the fake's; a rung without a read-out is read without them; the summary prints."""
    import e4_water as E4
    from test_e4_train import _fake_cells, _small_cfg, FakeMBPol
    import yaml
    cfg = _small_cfg(tmp_path)
    cfg["probes"]["far_trimers"] = 2
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 3), ("valid", 1), ("test", 2)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["data"]["response"] = str(tmp_path / "r.npz")
    cfg["liquid"].update(calibration_frames=2, heldout_frames=1)
    cfg["model"]["features"] = 8
    cfg["rungs"], cfg["seeds"] = ["M_16C2pm_us", "M_16pm_us"], [0]
    cfg["training"]["adam"].update(steps=1, batch_size=1)
    cfg["training"]["lbfgs"].update(steps=0)
    cfg["training"]["readout"].update(liquid={"weight": 0.1}, solve_iterations=2)
    cfg["reading"]["splits"] = ["valid"]
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)

    class Truth:
        mb = FakeMB()
        def describe(self):
            return {"name": "fake"}
    real = E4.W.MBPolTruth
    E4.W.MBPolTruth = lambda root: Truth()
    try:
        E4.run_response(SimpleNamespace(config=str(tmp_path / "c.yaml"), mbx="none", out=None))
    finally:
        E4.W.MBPolTruth = real
    out = tmp_path / "out"
    E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=None, out_dir=str(out)))
    assert any(f.startswith("matched_M_16C2pm_us_") and "_liq" in f for f in os.listdir(out))
    r = json.load(open(out / "M_16C2pm_us_seed0.json"))
    m, liq = r["matching"], r["readings"]["liquid"]
    assert m["liquid_cells"] == 2 and len(m["liquid_objective"]) >= 1
    assert m["liquid_objective"][-1][1] <= m["liquid_objective"][0][0]
    V = 12.5 ** 3
    assert liq["eps_inf_truth"] == pytest.approx(1 + 4 * np.pi * 16 * 1.5 / V, rel=1e-9)
    before = m["liquid_before_training"]
    for k in ("eps_inf_model", "cell_dipole_rel_rmse", "molecular_dipole_rel_rmse", "molecular_response_rel_rmse"):
        assert np.isfinite(liq[k]) and liq[k] == pytest.approx(before[k], rel=1e-9, abs=1e-12), k
    d = json.load(open(out / "M_16pm_us_seed0.json"))
    assert "liquid" not in d["readings"] and d["matching"] is None
    assert d["model"]["charges"] == r["model"]["charges"]
    E4.run_summary(SimpleNamespace(out_dir=str(out)))


def test_job2_configs_are_job1s_but_for_the_liquid_and_the_mutual_edge():
    """configs/e4_job2a.yaml is configs/e4_water.yaml with the liquid response, the mutual edge at 2.0 A and its three
    rungs; e4_job2b.yaml is e4_job2a.yaml with the liquid calibration and the two C rungs; both build their rungs."""
    import e4_water as E4
    root = os.path.dirname(HERE)
    load = lambda name: E4.load_config(os.path.join(root, "configs", name))
    c1, ca, cb = load("e4_water.yaml"), load("e4_job2a.yaml"), load("e4_job2b.yaml")

    def strip(c):
        c = json.loads(json.dumps(c))
        for sec, key in (("data", "response"), ("model", "mutual_edge_angstrom"), ("training", "readout")):
            c[sec].pop(key, None)
        for key in ("liquid", "rungs"):
            c.pop(key, None)
        return c
    assert strip(ca) == strip(c1) and strip(cb) == strip(ca)
    assert {**ca["training"]["readout"], "liquid": None} == {**c1["training"]["readout"], "liquid": None}
    assert ca["model"]["mutual_edge_angstrom"] == cb["model"]["mutual_edge_angstrom"] == 2.0
    assert c1["model"]["mutual_edge_angstrom"] is None
    assert ca["rungs"] == ["M_16C2pm_us", "M_16C2pm", "M_16pm_us"] and cb["rungs"] == ["M_16C2pm_us", "M_16C2pm"]
    assert ca["training"]["readout"]["liquid"] is None and cb["training"]["readout"]["liquid"]["weight"] == 0.1
    assert ca["data"]["response"] == cb["data"]["response"] and ca["liquid"] == cb["liquid"] == E4.DEFAULTS["liquid"]
    cell = np.eye(3) * A2B(12.43)
    for name in ca["rungs"]:
        E4.rungs_for(name, ca, cell)
