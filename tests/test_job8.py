"""
E4 job 8 (v0.0.32; notes/e4-job8-preregistration.md): the pinned c6 band at its own edge (Rung.disp_l), the rungs
without the band, a read-out held from another rung (training.readout.held_from, e4_water.held_readout) and the
readings K8.1-K8.5 with the model rule (scripts/e4_job8.py).
"""
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, water as W, structure as S, kernels as kn
from sseft.units import ang_to_bohr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job8 as J8                                      # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402


def _one(st, rung):
    return {k: v[0] for k, v in S.pad_batch([st], n_freq=rung.K).items()}


def _dimer(R=3.0):
    """two rigid waters R A apart (O-O), MB-pol's pins, open."""
    pos = np.vstack([W.monomer_positions(), W.monomer_positions() + np.array([0.0, 0.0, R])])
    return W.pin_rigid_site_model([W.water_structure(pos, pinned_q=False)], [0.0, 0.5777, 0.5777, -1.1553], 0.4267)[0]


def test_the_dispersion_band_has_its_own_edge():
    """with disp_l the c6 band is -1/2 sum C6_ij f6_long(r_ij, disp_l) over every pair, the other bands untouched;
    disp_l = 0 is the band at l1, as before; a wider edge damps the contact and keeps the tail; it is refused without a
    dispersion band or below l1."""
    cfg = E4.load_config(None)
    st = _dimer(2.9)
    cell = ang_to_bohr(np.eye(3) * 30.0)
    _, base = E4.rungs_for("M_16pm", cfg, cell)
    _, d20 = E4.rungs_for("M_16pm_d20", cfg, cell)
    assert d20.disp_l == pytest.approx(ang_to_bohr(2.0)) and base.disp_l == 0.0
    p = M.init_params(jax.random.PRNGKey(0), base, energy_scale=1e-4)
    b = _one(st, base)
    aux0 = M.energy_single(p, base, b)[1]
    aux2 = M.energy_single(p, d20, b)[1]
    alpha = np.asarray(aux0["alpha"])[:6]
    _, w = (np.asarray(x) for x in M.frequencies(base))
    X = np.asarray(st.positions)
    direct = {}
    for lab, l in (("base", base.l1), ("d20", ang_to_bohr(2.0))):
        e = 0.0
        for i in range(6):
            for j in range(6):
                if i != j:
                    c6 = float(kn.c6_from_alpha(alpha[i], alpha[j], w))
                    e -= 0.5 * c6 * float(kn.f6_long(np.linalg.norm(X[i] - X[j]), l))
        direct[lab] = e
    assert float(aux0["E_disp"]) == pytest.approx(direct["base"], rel=1e-10)
    assert float(aux2["E_disp"]) == pytest.approx(direct["d20"], rel=1e-10)
    assert float(aux2["E_coul"]) == float(aux0["E_coul"])
    assert abs(float(aux2["E_disp"])) < abs(float(aux0["E_disp"]))                        # damped at contact
    far = _dimer(9.0)
    e_far = [float(M.energy_single(p, r, _one(far, r))[0]) for r in (base, d20)]            # the tail: 0.04 % apart
    assert e_far[1] == pytest.approx(e_far[0], rel=1e-3)
    same = M.with_rung(base, disp_l=base.l1)
    assert float(M.energy_single(p, same, b)[0]) == float(M.energy_single(p, base, b)[0])
    with pytest.raises(ValueError, match="disp_l"):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(E4.E3.RUNGS["M_1pm"], disp_l=ang_to_bohr(2.0)))
    with pytest.raises(ValueError, match="disp_l"):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(base, disp_l=0.5 * base.l1))


def test_the_band_in_a_cell_is_the_open_band_with_its_images():
    """the periodic c6 band at its own edge: a dimer in a 25 A cell reads the open dimer's band to the images' share
    (C6/r^6 at >= 22 A), at 0.75 A and at 2.0 A alike."""
    cfg = E4.load_config(None)
    st = _dimer(3.0)
    cell = ang_to_bohr(np.eye(3) * 25.0)
    for name in ("M_16pm", "M_16pm_d20"):
        rc, ro = E4.rungs_for(name, cfg, cell)
        p = M.init_params(jax.random.PRNGKey(1), ro, energy_scale=1e-4)
        open_e = float(M.energy_single(p, ro, _one(st, ro))[1]["E_disp"])
        cst = _dimer(3.0)
        cst.cell, cst.pbc = cell, True
        cell_e = float(M.energy_single(p, rc, _one(cst, rc))[1]["E_disp"])
        assert cell_e == pytest.approx(open_e, rel=2e-3), name


def test_job8s_rungs():
    """the dagger at three edges; the field-term dagger and D' at 2.0 A and without the band; nothing else changed."""
    R = E4.E3.RUNGS
    for e in E4.DISP_EDGES:
        r = R[E4.disp_rung("M_16pm", e)]
        assert r.disp_l == pytest.approx(ang_to_bohr(e)) and r.dispersion and r.pin_alpha and r.msites
    for base, no in (("M_16pm_us", "M_1pm_us"), ("M_16C2pm_us", "M_1C2pm_us")):
        d = R[E4.disp_rung(base, 2.0)]
        assert d.disp_l == pytest.approx(ang_to_bohr(2.0)) and d == M.with_rung(R[base], name=d.name, disp_l=d.disp_l)
        assert R[no] == M.with_rung(R[base], name=no, dispersion=False, pin_alpha=False)
    c8 = E4.load_config(os.path.join(REPO, "configs", "e4_job8.yaml"))
    c8d = E4.load_config(os.path.join(REPO, "configs", "e4_job8d.yaml"))
    c5 = E4.load_config(os.path.join(REPO, "configs", "e4_job5.yaml"))
    assert set(c8["rungs"]) | set(c8d["rungs"]) == {*J8.EDGES, J8.SITES, J8.FIELD, J8.FIELD_D, J8.FIELD_NO, J8.DPRIME_D,
                                                    J8.DPRIME_NO}
    assert c8d["training"]["readout"]["held_from"] == {"rung": "M_16C2pm_us", "dir": "results/e4_job5"}
    for k in ("liquid", "finite_k"):                                    # the same cache tags as job 5's
        assert c8d["training"]["readout"][k] == c5["training"]["readout"][k]
    assert c8d["model"] == c5["model"] and c8d["data"] == c5["data"] and c8d["probes"] == c5["probes"]


def test_rung_key_is_the_key_written_before_the_new_fields():
    """a cache written before v0.0.30 keyed the rung without the fields added since: rung_key drops them at their
    defaults, and only then."""
    r = E4.E3.RUNGS["M_16C2pm_us"]
    key = E4.rung_key(r)
    for f, _ in E4.LATER_FIELDS:
        assert f"{f}=" not in key
    assert key.startswith("Rung(name='M_16C2pm_us'") and "frag_neutral=" in key and ", l1=" in key
    assert "disp_l=" in E4.rung_key(M.with_rung(r, disp_l=ang_to_bohr(2.0)))


def test_held_readout_loads_the_reference_and_refuses_others(tmp_path):
    """held_readout returns the reference rung's cached read-out under this config's tags, whether its key is today's
    or from before the new fields; it refuses a cache for another rung or energy scale and a missing one."""
    cfg = E4.load_config(None)
    cell = ang_to_bohr(np.eye(3) * 12.43)
    _, ref_open = E4.rungs_for("M_16C2pm_us", cfg, cell)
    probes = {"hash": "p0"}
    held = {"rung": "M_16C2pm_us", "dir": str(tmp_path)}
    path = tmp_path / "matched_M_16C2pm_us_p0.pkl"
    for key in (repr(ref_open), E4.rung_key(ref_open)):
        with open(path, "wb") as fh:
            pickle.dump({"rung": key, "scale": 0.5, "readout": {"x": np.arange(3.0)},
                         "match": {"objective": [(1.0, 0.5)]}}, fh)
        ro, match = E4.held_readout(held, cfg, cell, probes, 0.5)
        assert np.array_equal(ro["x"], np.arange(3.0)) and match["held_from"] == {"rung": "M_16C2pm_us", "path": str(path)}
    with pytest.raises(ValueError, match="another rung"):
        E4.held_readout(held, cfg, cell, probes, 0.25)
    with open(path, "wb") as fh:
        pickle.dump({"rung": repr(E4.rungs_for("M_16C2pm", cfg, cell)[1]), "scale": 0.5, "readout": {},
                     "match": {"objective": []}}, fh)
    with pytest.raises(ValueError, match="another rung"):
        E4.held_readout(held, cfg, cell, probes, 0.5)
    with pytest.raises(FileNotFoundError):
        E4.held_readout({"rung": "M_16C2pm_us", "dir": str(tmp_path / "none")}, cfg, cell, probes, 0.5)


def test_driver_trains_a_variant_on_the_held_readout(tmp_path, monkeypatch):
    """scripts/e4_water.py train for D' with its band at 2.0 A on fake cells (one Adam step), the read-out held from
    the frozen form's cache: nothing is matched, the json records the reference and the band's edge, and the trained
    read-out is the cached one."""
    import yaml
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["model"].update(features=8, mutual_edge_angstrom=2.0)
    cfg["training"]["adam"].update(steps=1, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    cfg["reading"]["eps_frames"] = 1
    ref_dir = tmp_path / "job5"
    cfg["training"]["readout"]["held_from"] = {"rung": "M_16C2pm_us", "dir": str(ref_dir)}
    cfg["rungs"], cfg["seeds"] = ["M_16C2pm_us_d20"], [0]
    train = W.e4_structures(str(tmp_path / "train_n16.npz"), pins=True)
    scale = E4.T.energy_scale(train)
    probes = E4.load_probes(cfg["data"]["probes"])
    _, ref_open = E4.rungs_for("M_16C2pm_us", cfg, train[0].cell)
    p0 = M.init_params(jax.random.PRNGKey(3), ref_open, energy_scale=scale)
    readout = jax.tree_util.tree_map(lambda x: 1.1 * np.asarray(x) + 0.01 if np.issubdtype(np.asarray(x).dtype, np.floating) else np.asarray(x), p0["readout"])
    os.makedirs(ref_dir)
    with open(ref_dir / f"matched_M_16C2pm_us_{probes['hash']}.pkl", "wb") as fh:
        pickle.dump({"rung": E4.rung_key(ref_open), "scale": scale, "readout": readout,
                     "match": {"objective": [(1.0, 0.5)]}}, fh)
    monkeypatch.setattr(E4, "match_readout", lambda *a, **k: pytest.fail("a held read-out is not matched"))
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmp_path / "out"
    E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=None, out_dir=str(out)))
    r = json.load(open(out / "M_16C2pm_us_d20_seed0.json"))
    assert r["matching"]["held_from"]["rung"] == "M_16C2pm_us" and r["model"]["dispersion_edge_angstrom"] == 2.0
    with open(out / "M_16C2pm_us_d20_seed0.params.pkl", "rb") as fh:
        trained = pickle.load(fh)["readout"]
    for a, b in zip(jax.tree_util.tree_leaves(trained), jax.tree_util.tree_leaves(readout)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def _run(rung, seed, f, eps=None, probes=None, t5=None, loss=30.0):
    cells = {n: {s: {"frames": nf, "force_rigid_rmse": f, "energy_per_molecule_rmse": 0.02}
                 for s, nf in (("train", 10), ("valid", 20), ("test", 30))} for n in J8.SIZES}
    rd = {"cells": cells}
    if eps is not None:
        rd["liquid"] = {"eps_inf_model": eps}
    if probes is not None:
        rd["probes"] = {"by_distance_angstrom": {d: {"nonadditive_rmse": x / 100, "nonadditive_rms": 1.0}
                                                 for d, x in probes.items()}}
    if t5 is not None:
        rd["far_trimers"] = {"three_body_rmse": t5 / 100, "three_body_rms": 1.0}
    m = {"liquid_before_training": {"eps_inf_model": 1.6893}} if eps is not None else None
    return {"rung": rung, "seed": seed, "readings": rd, "training": {"final_loss": {"total": loss}}, "matching": m}


def _job8(dagger=0.72, d20=0.57, sites=0.56, field=0.561, field_d=0.555, field_no=0.552, dprime_d=0.540, eps_d=1.6893,
          probes_d=None, t5_d=4.3):
    probes_ref, probes_d = {"10": 2.2, "12": 2.0, "15": 2.7}, probes_d or {"10": 2.3, "12": 2.0, "15": 2.6}
    runs = {}
    add = lambda r: runs.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    for s in range(4):
        for name, f in (("M_16pm", dagger), ("M_16pm_d20", d20), ("M_1pm", sites), ("M_16pm_us", field),
                        ("M_16pm_us_d20", field_d), ("M_1pm_us", field_no)):
            add(_run(name, s, f))
        add(_run("M_16C2pm_us_d20", s, dprime_d, eps=eps_d, probes=probes_d, t5=t5_d))
    for s in (0, 1):
        add(_run("M_1C2pm_us", s, 0.53, eps=1.6893, probes={"10": 5.0, "12": 6.0, "15": 7.0}, t5=4.3))
    ref = {"M_16C2pm_us": {s: _run("M_16C2pm_us", s, 0.541, eps=1.6893, probes=probes_ref, t5=4.3) for s in range(4)}}
    return runs, ref


def test_the_job8_readings_and_the_model_rule():
    """K8.1-K8.5 on constructed jsons, each band's edge, a missing rung (not read), and the model rule."""
    ev = J8.evaluate(*_job8())
    assert all(ev[k]["holds"] for k in ("K8.1", "K8.2", "K8.3", "K8.4", "K8.5")) and ev["model_rule"] is True
    assert ev["K8.1"]["values"]["64"] == pytest.approx(0.57 / 0.56 - 1)
    assert ev["K8.4"]["values"]["128"] == pytest.approx(0.540 / 0.541 - 1)
    assert ev["K8.5"]["values"]["probes 10 A"] == pytest.approx(0.1) and ev["K8.5"]["values"]["eps_inf"] == pytest.approx(0)
    assert ev["D'_without_band"]["seeds"] == [0, 1] and ev["D'_without_band"]["values"]["64"] == pytest.approx(0.53 / 0.541 - 1)
    text = J8.report(ev)
    assert "K8.1: holds" in text and "becomes E4's model" in text and "the dagger by the c6 band's edge" in text
    assert not J8.evaluate(*_job8(d20=0.61))["K8.1"]["holds"]                      # 8.9 % above M_1pm
    ev = J8.evaluate(*_job8(field_no=0.50))                                        # the field term and the band add up
    assert not ev["K8.2"]["holds"] and not ev["K8.3"]["holds"]
    ev = J8.evaluate(*_job8(dprime_d=0.555))                                       # 2.6 % above D'
    assert not ev["K8.4"]["holds"] and ev["model_rule"] is False and "D' stays E4's model" in J8.report(ev)
    assert J8.evaluate(*_job8(eps_d=1.6920))["K8.5"]["holds"] is False
    assert J8.evaluate(*_job8(probes_d={"10": 2.9, "12": 2.0, "15": 2.7}))["K8.5"]["holds"] is False
    assert J8.evaluate(*_job8(t5_d=5.0))["K8.5"]["holds"] is False
    runs, ref = _job8()
    del runs["M_16C2pm_us_d20"]
    ev = J8.evaluate(runs, ref)
    assert ev["K8.4"]["holds"] is None and ev["K8.5"]["holds"] is None and ev["model_rule"] is None
    assert "the model rule (K8.4 and K8.5): not read" in J8.report(ev)
