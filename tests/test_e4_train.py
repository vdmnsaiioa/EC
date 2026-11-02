"""
E4's training (v0.0.21): the rigid-body projection of the force residual (train.rigid_force_sums, force_mode "rigid"),
a read-out matched elsewhere and held (train.fit's readout_init), the rigid site-model pins, MB-pol's probe file, and
the driver scripts/e4_water.py end to end on a fake data set (M_inf).  With an MBX build ($MBX_HOME or
external/mbx/build/MBX) also MB-pol's site model and charges.
"""
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, train as T, water as W, structure as S, periodic as PB
from sseft.mbpol import random_box
from sseft.units import ang_to_bohr, bohr_to_ang

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
from test_periodic import frozen, switched_on             # noqa: E402

MBX_ROOT = os.environ.get("MBX_HOME", os.path.join(REPO, "external", "mbx", "build", "MBX"))
needs_mbx = pytest.mark.skipif(not os.path.isfile(os.path.join(MBX_ROOT, "lib", "libmbx.so")),
                               reason="no MBX build (sh external/mbx/build.sh, or set MBX_HOME)")
SITE_Q = np.array([0.0, 0.577650024808747, 0.577650024808747, -1.155300049617494])     # MB-pol's rigid monomer
GAMMA = 0.426706882


def _box(n=8, L=12.43, seed=1):
    rng = np.random.default_rng(seed)
    return random_box(n, np.eye(3) * L, rng, min_oo=2.5, min_any=1.5), np.eye(3) * L


def _whole(pos, L):
    out = pos.copy()
    for m in range(len(pos) // 3):
        d = pos[3 * m:3 * m + 3] - pos[3 * m]; d -= np.round(d / L) * L; out[3 * m:3 * m + 3] = pos[3 * m] + d
    return out


def _project(e, pos):
    """the Euclidean projection of e on every molecule's rigid-body motions (whole molecules), by hand."""
    out = np.zeros_like(e)
    for m in range(len(pos) // 3):
        d = pos[3 * m:3 * m + 3] - pos[3 * m:3 * m + 3].mean(0); ee = e[3 * m:3 * m + 3]
        I = sum(np.dot(x, x) * np.eye(3) - np.outer(x, x) for x in d)
        w = np.linalg.solve(I, np.cross(d, ee).sum(0))
        out[3 * m:3 * m + 3] = ee.sum(0) / 3 + np.cross(w, d)
    return out


def _sums(e, pos, cell):
    n = len(pos) // 3
    fm = np.zeros((n, 3 * n), bool); fm[np.repeat(np.arange(n), 3), np.arange(3 * n)] = True
    s, c = T.rigid_force_sums(jnp.asarray(e), jnp.asarray(pos), jnp.ones(3 * n, bool), jnp.asarray(fm), jnp.asarray(cell))
    return float(s), float(c)


# ---------------------------------------------------------------- the rigid-body projection -------------------------
def test_rigid_force_sums_is_the_rigid_body_projection():
    """a rigid-body field is kept whole, a field with no net force and no torque on any molecule is invisible, a
    random field gives the squared norm of its projection; 6 degrees of freedom per water; wrapping atoms by lattice
    vectors and an open structure's 1e6 cell change nothing."""
    pos, cell = _box()
    L = cell[0, 0]; n = len(pos) // 3
    pw = _whole(pos, L)
    rng = np.random.default_rng(2)
    rig = np.zeros_like(pos)
    for m in range(n):
        c = pw[3 * m:3 * m + 3].mean(0)
        rig[3 * m:3 * m + 3] = rng.normal(size=3) + np.cross(rng.normal(size=3), pw[3 * m:3 * m + 3] - c)
    s, c = _sums(rig, pos, cell)
    assert c == 6 * n
    assert s == pytest.approx(np.sum(rig ** 2), rel=1e-12)
    e = rng.normal(size=pos.shape)
    internal = e - _project(e, pw)
    assert _sums(internal, pos, cell)[0] < 1e-20 * np.sum(e ** 2)
    ref = np.sum(_project(e, pw) ** 2)
    assert _sums(e, pos, cell)[0] == pytest.approx(ref, rel=1e-12)
    wrapped = pos + np.round(rng.uniform(-1, 1, size=(3 * n, 1))) * np.array([L, 0.0, 0.0])
    assert _sums(e, wrapped, cell)[0] == pytest.approx(ref, rel=1e-12)
    assert _sums(e, pw, np.eye(3) * 1e6)[0] == pytest.approx(ref, rel=1e-12)


def test_rigid_loss_ignores_internal_forces():
    """with force_mode "rigid" the loss is blind to an internal force field added to the labels (the flux of a truth's
    geometry-dependent charges on rigid molecules), and "atomic" sees it; padded atoms and fragments count nothing."""
    pos, _ = _box(n=3, L=30.0, seed=4)
    pos = _whole(pos, 30.0)
    rng = np.random.default_rng(5)
    F = rng.normal(size=pos.shape) * 1e-3
    internal = rng.normal(size=pos.shape) * 1e-3
    internal = internal - _project(internal, pos)
    rung = M.RUNGS["M_inf"]
    p = M.init_params(jax.random.PRNGKey(0), rung)
    small = W.water_structure(_whole(random_box(2, np.eye(3) * 30.0, np.random.default_rng(6)), 30.0),
                              energy=0.0, forces=np.zeros((6, 3)), pinned_q=False)   # pads the batch: 6 atoms of 9
    out = {}
    for mode in ("rigid", "atomic"):
        f = T.make_loss(rung, 1.0, 1.0, 1.0, force_mode=mode)
        vals = []
        for lab in (F, F + internal):
            st = W.water_structure(pos, energy=0.0, forces=lab, pinned_q=False)
            vals.append(float(f(p, S.pad_batch([st, small]))[0]))
        out[mode] = vals
    assert out["rigid"][0] == pytest.approx(out["rigid"][1], rel=1e-10)
    assert abs(out["atomic"][1] - out["atomic"][0]) > 1e-3 * abs(out["atomic"][0])
    with pytest.raises(ValueError):
        T.make_loss_sums(rung, 1.0, 1.0, force_mode="torque")


# ---------------------------------------------------------------- readout_init -------------------------------------
def test_fit_holds_a_matched_readout():
    """readout_init replaces the read-out's initialisation and survives Adam and the polish exactly; the polish may be
    skipped; a read-out of another shape, a rung without one, and the read-out's own solve or Adam are refused."""
    rng = np.random.default_rng(7)
    structs = []
    for k in range(4):
        pos = W.random_cluster(3, rng, mc_steps=0)
        E, Fz = W.PW_TRUTH.energy_forces(pos, 3)
        structs.append(W.water_structure(pos, energy=E, forces=Fz, pinned_q=False))
    W.pin_rigid_site_model(structs, SITE_Q, GAMMA)
    rung = frozen(r_cut=4.0, F=4)
    p0 = M.init_params(jax.random.PRNGKey(3), rung)
    ro = jax.tree_util.tree_map(lambda x: np.asarray(x) + 0.01 * np.random.default_rng(8).normal(size=np.shape(x)),
                                p0["readout"])
    for lbfgs in (0, 1):
        params, info = T.fit(rung, structs, seed=0, steps=2, batch_size=2, lbfgs_steps=lbfgs, lbfgs_chunk=2,
                             readout_init=ro, force_mode="rigid")
        for a, b in zip(jax.tree_util.tree_leaves(params["readout"]), jax.tree_util.tree_leaves(ro)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert info["training_options"]["readout_init"] and info["training_options"]["force_mode"] == "rigid"
    bad = jax.tree_util.tree_map(lambda x: np.zeros(np.shape(x) + (1,)), ro)
    with pytest.raises(ValueError):
        T.fit(rung, structs, seed=0, steps=1, readout_init=bad)
    with pytest.raises(ValueError):
        T.fit(M.RUNGS["M_inf"], structs, seed=0, steps=1, readout_init=ro)
    with pytest.raises(ValueError):
        T.fit(rung, structs, seed=0, steps=1, readout_init=ro, readout_adam=True)
    with pytest.raises(ValueError):
        T.fit(rung, structs, seed=0, steps=1, force_mode="rigid ", readout_init=ro)


# ---------------------------------------------------------------- pins, rungs ---------------------------------------
def test_rigid_site_model_pins_are_e4s(tmp_path):
    """pin_rigid_site_model gives open structures the pins e4_structures(pins=True) gives a cell's frames."""
    pos, cell = _box(n=4)
    path = tmp_path / "f.npz"
    np.savez(path, positions=pos[None], cell=cell, energy=np.zeros(1), forces=np.zeros((1, len(pos), 3)),
             site_q=SITE_Q, site_gamma=np.array(GAMMA))
    cellst = W.e4_structures(str(path), pins=True)[0]
    openst = W.pin_rigid_site_model([W.water_structure(pos, pinned_q=False)], SITE_Q, GAMMA)[0]
    for k in ("q_site", "vsite_w", "vsite_q", "vsite_frag", "alpha"):
        np.testing.assert_array_equal(cellst.pinned[k], openst.pinned[k])


def test_rungs_for_is_e3s_frozen_form():
    """the driver's M_16C2pm_us is E3-QP8's frozen form (test_periodic.frozen at width 32) in a cell, with GMRES and the
    cell's k-grid; its open twin differs only in periodic; M_16C2pm is the same rung without the field term."""
    cell = ang_to_bohr(np.eye(3) * 12.43)
    cfg = E4.load_config(None)
    rc, ro = E4.rungs_for("M_16C2pm_us", cfg, cell)
    l1 = frozen(F=32).l1
    ref = replace(frozen(F=32), periodic=True, ewald_b_max=4.0, bf_solver="gmres", ewald_n_max=PB.n_max_for(cell, l1, 4.0),
                  K=8, omega0=0.3)
    assert rc == ref
    assert ro == replace(rc, periodic=False)
    base, _ = E4.rungs_for("M_16C2pm", cfg, cell)
    assert base == replace(rc, name=base.name, e0_field=False, e0_field_whole=False, e0_field_on=base.e0_field_on,
                           e0_field_off=base.e0_field_off, e0_field_self_on=0.0, e0_field_self_off=0.0,
                           e0_field_cross_species=False)
    assert E4.rungs_for("M_16C0pm_us", cfg, cell)[0] == replace(rc, name="M_16C0pm_us", band_fields=0)


# ---------------------------------------------------------------- the probes and the driver ------------------------
class FakeMBPol:
    """the PW truth behind MB-pol's interface (site model, describe), for the probe file without MBX."""
    name = "fake_mbpol"
    site_charges, gamma_m = SITE_Q, GAMMA

    def energy(self, pos_ang, n_mol=None, cell=None):
        return W.PW_TRUTH.energy(pos_ang, len(pos_ang) // 3)

    def energy_forces(self, pos_ang, n_mol=None, cell=None):
        return W.PW_TRUTH.energy_forces(pos_ang, len(pos_ang) // 3)

    def molecule_charges(self, pos_ang):
        return W.PW_TRUTH.molecule_charges(pos_ang)

    def describe(self):
        return {"name": self.name}


def _small_cfg(data_dir):
    cfg = E4.load_config(None)
    cfg["data"].update(dir=str(data_dir), probes=str(data_dir / "probes.npz"), eval_sizes=[16])
    cfg["data"]["train_size"] = 16
    cfg["probes"].update(train_per_distance=1, train_cores=2, train_core_sizes=[2], test_per_distance=1, test_cores=1,
                         mc_steps=5, distances_angstrom=[9.0, 12.0])
    return cfg


def test_probe_file_round_trip(tmp_path):
    """the probe file keeps every probe: training probes come back labelled by their interaction (E - E_core, forces
    less the core's, the core and the probe as subtraction groups), test probes with their totals and the truth's
    readings; the hash is that of the positions."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    probes = E4.make_probes(truth, cfg)
    digest = E4.save_probes(cfg["data"]["probes"], probes, truth, cfg)
    back = E4.load_probes(cfg["data"]["probes"])
    assert back["hash"] == digest and back["meta"]["hash"] == digest
    assert len(back["train"]) == len(probes["train"]) == 4 and len(back["test"]) == len(probes["test"]) == 2
    for st, orig in zip(back["train"], probes["train"]):
        pos = bohr_to_ang(np.asarray(orig.positions)); n = len(pos) // 3
        E_c, F_c = truth.energy_forces(pos[:-3], n - 1)
        assert st.energy == pytest.approx(orig.energy - E_c, abs=1e-12)
        np.testing.assert_allclose(st.forces[:-3], np.asarray(orig.forces)[:-3] - F_c, atol=1e-12)
        np.testing.assert_allclose(st.forces[-3:], np.asarray(orig.forces)[-3:], atol=1e-12)
        assert list(st.sub) == [0] * (3 * n - 3) + [1] * 3
        assert "vsite_w" in st.pinned and "alpha" in st.pinned
    ref = back["test_ref"]
    for k, orig in enumerate(probes["test"]):
        pos = bohr_to_ang(np.asarray(orig.positions)); n = len(pos) // 3
        inter = orig.energy - orig.info["E_core"]
        pairs = sum(truth.energy(np.concatenate([pos[3 * i:3 * i + 3], pos[-3:]]), 2) for i in range(n - 1))
        assert ref["interaction"][k] == pytest.approx(inter, abs=1e-12)
        assert ref["nonadditive"][k] == pytest.approx(inter - pairs, abs=1e-12)


def test_far_readings_for_any_model(tmp_path):
    """scripts/e4_far.py, NumPy only: the truth's own energies of its frames read zero error and a T5 slope of 1, and a
    harness model's energies of the same frames read exactly what the driver reads for it (read_probes,
    read_trimers); the extxyz holds every frame in that order."""
    import e4_far
    cfg = _small_cfg(tmp_path)
    cfg["probes"]["far_trimers"] = 3
    truth = FakeMBPol()
    path = cfg["data"]["probes"]
    E4.save_probes(path, E4.make_probes(truth, cfg), truth, cfg)
    fr = e4_far.frames(path)
    assert [k for k, *_ in fr].count("trimer") == 3 and [k for k, *_ in fr].count("probe") == 2
    E_truth = np.array([truth.energy(pos, len(pos) // 3) for _, pos, _, _ in fr]) * e4_far.EV_PER_HARTREE
    out = e4_far.readings(path, E_truth)
    assert out["probes"]["nonadditive_rmse"] < 1e-12 and out["probes"]["interaction_rmse"] < 1e-12
    assert out["far_trimers"]["slope"] == pytest.approx(1.0, abs=1e-9)
    for kind, pos, E, _ in fr:
        if E is not None:
            assert E == pytest.approx(truth.energy(pos, len(pos) // 3), abs=1e-12)
    probes = E4.load_probes(path)
    rung = frozen(r_cut=4.0, F=4)
    params = switched_on(rung)                                       # every channel on: the response too
    structs = W.pin_rigid_site_model([W.water_structure(pos, pinned_q=False) for _, pos, _, _ in fr], SITE_Q, GAMMA)
    mine = e4_far.readings(path, T.predict_energies(params, rung, structs) * e4_far.EV_PER_HARTREE)
    theirs = E4.read_probes(params, rung, probes), E4.read_trimers(params, rung, probes)
    for key in ("interaction_rmse", "nonadditive_rmse"):
        assert mine["probes"][key] == pytest.approx(theirs[0][key], rel=1e-9, abs=1e-15)
    assert abs(theirs[1]["slope"]) > 1e-3                            # the response gives the trimers a three-body
    assert mine["far_trimers"]["slope"] == pytest.approx(theirs[1]["slope"], rel=1e-8)
    xyz = e4_far.write_extxyz(path)
    with open(xyz) as fh:
        assert sum(1 for line in fh if "pbc=" in line) == len(fr)


def _fake_cells(path, n_frames, n_mol, L, seed):
    rng = np.random.default_rng(seed)
    pos = np.array([random_box(n_mol, np.eye(3) * L, rng, min_oo=2.5, min_any=1.5) for _ in range(n_frames)])
    np.savez(path, positions=pos, cell=np.eye(3) * L, energy=rng.normal(size=n_frames) * 1e-3,
             forces=rng.normal(size=pos.shape) * 1e-3, site_q=SITE_Q, site_gamma=np.array(GAMMA),
             source=np.zeros(n_frames, int), time_ps=np.arange(n_frames) * 0.5)


def test_driver_end_to_end_on_fake_cells(tmp_path):
    """scripts/e4_water.py train and summary on fake 16-molecule cells (12.5 A, wide enough for r_c 6 A) and the fake
    probes, for M_inf: one json and one parameter file per rung and seed, every split read, the probes read, the summary
    printed; --task picks (rung, seed) as rung task // n_seeds, seed task % n_seeds."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 3), ("valid", 1), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["rungs"], cfg["seeds"] = ["M_inf", "M_1"], [0, 1]
    cfg["training"]["adam"].update(steps=1, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0)
    import yaml
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmp_path / "out"
    E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=1, out_dir=str(out)))
    files = sorted(os.listdir(out))
    assert files == ["M_inf_seed1.json", "M_inf_seed1.params.pkl"]
    r = json.load(open(out / "M_inf_seed1.json"))
    assert r["rung"] == "M_inf" and r["seed"] == 1 and r["training_frames"] == 3
    cells = r["readings"]["cells"]["16"]
    assert set(cells) == {"train", "valid", "test"}
    for v in cells.values():
        assert np.isfinite(v["energy_per_molecule_rmse"]) and np.isfinite(v["force_rigid_rmse"])
    by = r["readings"]["probes"]["by_distance_angstrom"]
    assert by["12"]["nonadditive_rmse"] == pytest.approx(by["12"]["nonadditive_rms"], rel=1e-9)    # M_inf: 100 %
    E4.run_summary(SimpleNamespace(out_dir=str(out)))


@needs_mbx
def test_mbpol_molecule_charges_are_neutral_and_its_pins_its_site_model():
    truth = W.MBPolTruth(MBX_ROOT)
    pos, _ = _box(n=3)
    q = truth.molecule_charges(pos)
    assert q.shape == (9,) and abs(q.reshape(3, 3).sum(axis=1)).max() < 1e-12
    np.testing.assert_allclose(truth.site_charges, SITE_Q, atol=1e-12)
    assert truth.gamma_m == pytest.approx(GAMMA, abs=1e-9)
    st = W.pin_rigid_site_model([W.water_structure(pos, pinned_q=False)], truth.site_charges, truth.gamma_m)[0]
    q_site, w, vq, vf = truth.site_model(pos)
    np.testing.assert_allclose(st.pinned["q_site"], q_site); np.testing.assert_allclose(st.pinned["vsite_w"], w)
    np.testing.assert_allclose(st.pinned["vsite_q"], vq); np.testing.assert_array_equal(st.pinned["vsite_frag"], vf)
