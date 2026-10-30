"""E4's data script (scripts/e4_data.py) and its loader (sseft.water.e4_structures). The format, the splits and the
extxyz run everywhere; a short MB-pol trajectory runs where an MBX build is ($MBX_HOME or external/mbx/build/MBX)."""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = os.environ.get("MBX_HOME", str(REPO / "external" / "mbx" / "build" / "MBX"))
needs_mbx = pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "lib", "libmbx.so")),
                               reason="no MBX build (sh external/mbx/build.sh, or set MBX_HOME)")
from sseft import water as W, rigidmd as R  # noqa: E402
from sseft.units import KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM, ang_to_bohr  # noqa: E402


@pytest.fixture(scope="module")
def ed():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("e4_data")


def fake_traj(n_mol, seed, frames, rng):
    """a trajectory file's arrays with made-up labels (the format only)."""
    from sseft.mbpol import random_box
    L = 10.0 + n_mol / 20.0
    pos = np.array([random_box(n_mol, np.eye(3) * L, rng, min_oo=2.4) for _ in range(frames)])
    meta = {"n_mol": n_mol, "seed": seed}
    if seed == 1:                                      # what run_md records (assemble prints it per trajectory)
        meta["diagnostics"] = {"T_mean": 301.0, "T_std": 20.0, "conserved_drift_per_mol_per_ps": 2e-3,
                               "g_oo": {"g_peak": 2.6, "r_peak": 2.8}}
    return dict(positions=pos, cell=np.eye(3) * L, energy=rng.normal(size=frames) * 0.01,
                forces=rng.normal(size=pos.shape) * 0.01, md_energy=rng.normal(size=frames),
                time_ps=11.0 + 0.5 * np.arange(1, frames + 1), source=np.full(frames, seed),
                site_q=np.array([0.0, 0.57765, 0.57765, -1.1553]), site_alpha=np.array([1.31, 0.294, 0.294, 0.0]),
                site_gamma=np.array(0.4267), log=np.zeros((3, 5)), g_oo=np.zeros((2, 5)),
                meta=np.array(json.dumps(meta)))


def read_extxyz(path):
    """a minimal reader for the writer's own format: per frame (lattice (3,3), energy, species, positions, forces)."""
    out = []
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line.strip():
                break
            n = int(line)
            head = fh.readline()
            lat = np.array([float(x) for x in head.split('Lattice="')[1].split('"')[0].split()]).reshape(3, 3)
            E = float(head.split("REF_energy=")[1].split()[0])
            rows = [fh.readline().split() for _ in range(n)]
            out.append((lat, E, [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows]),
                        np.array([[float(x) for x in r[4:7]] for r in rows])))
    return out


def test_cubic_side(ed):
    assert abs(ed.cubic_side(64) - 12.4296) < 1e-4 and abs(ed.cubic_side(512) / ed.cubic_side(64) - 2.0) < 1e-12


def test_assemble_splits_extxyz_and_loader(ed, tmp_path):
    rng = np.random.default_rng(0)
    for n, seeds, frames in ((8, (0, 1, 2, 3), 6), (16, (0, 1), 4)):
        for s in seeds:
            np.savez(tmp_path / f"traj_n{n}_s{s}.npz", **fake_traj(n, s, frames, rng))
    written = ed.assemble(str(tmp_path), train=(8,), test=(16,), verbose=False)
    assert written == {("train", 8): 18, ("valid", 8): 3, ("test", 8): 3, ("test", 16): 8}
    tr = np.load(tmp_path / "train_n8.npz")
    assert sorted(set(tr["source"].tolist())) == [0, 1, 2]
    va, te = np.load(tmp_path / "valid_n8.npz"), np.load(tmp_path / "test_n8.npz")
    src3 = np.load(tmp_path / "traj_n8_s3.npz")
    assert np.allclose(np.concatenate([va["positions"], te["positions"]]), src3["positions"])   # the last seed, split
    # the extxyz: eV, eV/A, the lattice, O H H
    fr = read_extxyz(tmp_path / "test_n16.extxyz")
    t16 = np.load(tmp_path / "test_n16.npz")
    assert len(fr) == 8
    lat, E, sym, pos, F = fr[3]
    assert np.allclose(lat, t16["cell"]) and sym[:3] == ["O", "H", "H"]
    assert np.isclose(E, t16["energy"][3] * ed.EV_PER_HARTREE, atol=1e-11)
    assert np.allclose(pos, t16["positions"][3], atol=1e-9)
    assert np.allclose(F, t16["forces"][3] * ed.EV_PER_HARTREE * BOHR_PER_ANGSTROM, rtol=1e-10, atol=1e-14)
    # the loader: bohr, hartree, one fragment per molecule, the cell, the site model on request
    sts = W.e4_structures(str(tmp_path / "test_n16.npz"), pins=True)
    assert len(sts) == 8
    s = sts[5]
    assert np.allclose(s.positions, ang_to_bohr(t16["positions"][5])) and np.allclose(s.cell, ang_to_bohr(t16["cell"]))
    assert s.pbc and np.isclose(s.energy, t16["energy"][5]) and np.allclose(s.forces, t16["forces"][5])
    assert list(s.frag[:6]) == [0, 0, 0, 1, 1, 1] and s.info["source"] in (0, 1)
    w = s.pinned["vsite_w"]
    assert w.shape == (16, 48) and np.allclose(w.sum(axis=1), 1.0) and np.isclose(w[1, 4], 0.5 * 0.4267)
    assert np.isclose(s.pinned["q_site"].sum() + s.pinned["vsite_q"].sum(), 0.0)
    assert "vsite_w" not in W.e4_structures(str(tmp_path / "test_n16.npz"))[0].pinned


def test_assemble_refuses_one_training_trajectory(ed, tmp_path):
    np.savez(tmp_path / "traj_n8_s0.npz", **fake_traj(8, 0, 4, np.random.default_rng(1)))
    with pytest.raises(ValueError):
        ed.assemble(str(tmp_path), train=(8,), test=(), verbose=False)


@needs_mbx
def test_md_smoke(ed, tmp_path):
    """a few steps of rigid MB-pol MD on 16 molecules: constraints held, every frame labelled with E4's truth, the
    file loads as structures."""
    out = ed.run_md(ROOT, 16, seed=3, frames=2, every_ps=0.01, equil_ps=0.02, warmup_ps=0.01, out_dir=str(tmp_path),
                    log_ps=0.01, verbose=False)
    assert out["positions"].shape == (2, 48, 3) and out["energy"].shape == (2,)
    assert out["log"][:, 3].max() < 1e-9                                       # rigid throughout
    assert out["log"].shape[1] == 5 and np.isfinite(out["log"][:, 4]).all()       # ... and the conserved energy logged
    X = out["positions"].reshape(2, 16, 3, 3)
    from sseft.mbpol import monomer_ang
    d = R.monomer_distances(monomer_ang())
    for fr in X:
        assert np.allclose([R.monomer_distances(m) for m in fr], d, atol=1e-8)
    from sseft.mbpol import MBPol
    t = MBPol(ROOT)
    e, f = t.energy_forces(out["positions"][1], out["cell"])
    assert np.isclose(e, out["energy"][1], rtol=0, atol=1e-11) and np.allclose(f, out["forces"][1], atol=1e-10)
    sts = W.e4_structures(str(tmp_path / "traj_n16_s3.npz"))
    assert len(sts) == 2 and sts[0].pbc
    meta = json.loads(str(np.load(tmp_path / "traj_n16_s3.npz")["meta"]))
    assert meta["n_mol"] == 16 and meta["dynamics_settings"]["replicate"] is False
