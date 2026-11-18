"""
E4 job 7 (v0.0.31; notes/e4-job7-preregistration.md): MB-pol's site model with its M-site's charge on the atoms (the
pinned O/H charges), the scan of their scale (Rung.q_scale), the warm start of learned charges (Rung.q_ref_start), the
driver's records, and the readings J7.1-J7.5 (scripts/e4_job7.py) and J7.6 (scripts/e4_sumrules.py read --job7_dir).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, water as W, structure as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_job7 as J7                                      # noqa: E402
import e4_sumrules as SR                                  # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol, SITE_Q, GAMMA   # noqa: E402

MBPOL_Q = np.array([0.0, 0.577650024808747, 0.577650024808747, -1.155300049617494])
MBPOL_GAMMA = 0.426706882


def _cell(tmp_path, n_frames=1, n_mol=16, seed=3, name="t.npz"):
    _fake_cells(tmp_path / name, n_frames, n_mol, 12.5, seed=seed)
    return W.e4_structures(str(tmp_path / name), pins=True)


def _cfg(features=8):
    return E4.merge(E4.load_config(None), {"model": {"features": features}})


def _one(st, rung):
    return {k: v[0] for k, v in S.pad_batch([st], n_freq=rung.K).items()}


def test_oh_charges_carry_the_site_models_charge_and_dipole():
    """the M-site's charge moved onto the atoms by the site's weights keeps the molecule's charge and its dipole at any
    rigid geometry, not its quadrupole; MB-pol's: O -0.6623, H +0.3312 e (1.864 D on MB-pol's monomer).  The pins carry
    them as the atoms' charges and leave a structure's own pinned charges alone."""
    q = W.oh_equivalent_charges(MBPOL_Q, MBPOL_GAMMA)
    assert q == pytest.approx([-0.66232557, 0.33116278, 0.33116278], abs=1e-8) and abs(q.sum()) < 1e-14
    rng = np.random.default_rng(2)
    for _ in range(3):
        R = W.random_rotation(rng)
        P = W.monomer_positions() @ R.T + rng.normal(size=3)
        M_site = P[0] + 0.5 * MBPOL_GAMMA * (P[1] + P[2] - 2 * P[0])
        mu_site = MBPOL_Q[:3] @ P + MBPOL_Q[3] * M_site
        assert np.allclose(q @ P, mu_site, atol=1e-12)
        quad = lambda qs, X: np.einsum("a,ai,aj->ij", qs, X - P[0], X - P[0])
        assert not np.allclose(quad(q, P), quad(MBPOL_Q, np.vstack([P, M_site])), atol=1e-3)
    st = W.water_structure(np.vstack([W.monomer_positions(), W.monomer_positions() + 5.0]), pinned_q=False)
    W.pin_rigid_site_model([st], MBPOL_Q, MBPOL_GAMMA)
    assert np.allclose(st.pinned["q"], np.tile(q, 2))
    own = W.water_structure(W.monomer_positions())                       # the PW truth's charges, pinned by the caller
    W.pin_rigid_site_model([own], MBPOL_Q, MBPOL_GAMMA)
    assert np.allclose(own.pinned["q"], W.monomer_charges())


def test_the_scan_scales_the_pinned_coulomb_band(tmp_path):
    """job 7's scan rungs pin the O/H charges at c times: the Coulomb band, bilinear in them, is c^2 times c = 1's with
    the same parameters; c = 1 is the pin exactly; and the scale is refused on a rung whose charges are learned."""
    st = _cell(tmp_path)[0]
    cfg = _cfg()
    E = {}
    for c in E4.SCAN_C:
        rung, _ = E4.rungs_for(E4.scan_rung(c), cfg, st.cell)
        assert rung.pin_q and not rung.msites and not rung.dispersion and rung.band_fields < 0 and rung.q_scale == c
        p = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=1e-4)
        _, aux = M.energy_single(p, rung, _one(st, rung))
        assert np.allclose(np.asarray(aux["q"])[:48], c * st.pinned["q"], atol=1e-15)
        E[c] = float(aux["E_coul"])
    for c in E4.SCAN_C:
        assert E[c] == pytest.approx(c ** 2 * E[1.0], rel=1e-10)
    with pytest.raises(ValueError, match="q_scale"):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(E4.rungs_for("M_1", cfg, st.cell)[0], q_scale=0.5))


def test_the_warm_start_starts_at_the_charges_exactly(tmp_path):
    """a warm-started rung's charges start at c0 times the pinned O/H charges on every atom, whatever the seed (the
    charge head's last layer at zero, the species offsets at q / unit), are not a stationary point of the energy, and
    move in units of Q_UNIT_WARM; a warm start is refused on pinned charges."""
    st = _cell(tmp_path)[0]
    cfg = _cfg()
    for c0 in E4.WARM_C:
        name = E4.warm_rung(c0)
        rung = E4.warm_started(name, E4.rungs_for(name, cfg, st.cell)[0], [st])
        assert rung.q_unit == E4.Q_UNIT_WARM and not rung.pin_q and E4.learns_sources(rung)
        assert dict(rung.q_ref_start) == pytest.approx({8: c0 * st.pinned["q"][0], 1: c0 * st.pinned["q"][1]})
        for seed in (0, 7):
            p = M.init_params(jax.random.PRNGKey(seed), rung, energy_scale=1e-4)
            assert not np.any(np.asarray(p["heads"]["charge"][-1]["w"]))
            _, aux = M.energy_single(p, rung, _one(st, rung))
            assert np.allclose(np.asarray(aux["q"])[:48], c0 * st.pinned["q"], rtol=1e-12, atol=1e-15)
        b = _one(st, rung)
        g = jax.grad(lambda hc: M.energy_single({**p, "heads": {**p["heads"], "charge": hc}}, rung, b)[0])(
            p["heads"]["charge"])
        assert float(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g))) > 0.0
    assert E4.warm_started("M_1u", E4.rungs_for("M_1u", cfg, st.cell)[0], [st]).q_ref_start == ()
    with pytest.raises(ValueError, match="q_ref_start"):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(E4.rungs_for("M_1q100", cfg, st.cell)[0],
                                                         q_ref_start=((8, -0.6), (1, 0.3))))


def test_job7s_rungs_and_the_readers_agree():
    """the driver's rungs are the reader's, in the config's order, and the sbatch's tasks map onto them."""
    assert {E4.scan_rung(c): c for c in E4.SCAN_C} == J7.SCAN and E4.WARM == J7.WARM
    assert E3_rung("M_1pm").msites and not E3_rung("M_1pm").dispersion and E3_rung("M_16pm").dispersion
    cfg = E4.load_config(os.path.join(REPO, "configs", "e4_job7.yaml"))
    assert cfg["rungs"] == ["M_inf", "M_16pm", "M_1pm", *J7.SCAN, *J7.WARM] and cfg["seeds"] == list(J7.SEEDS)


def E3_rung(name):
    return E4.E3.RUNGS[name]


def test_driver_records_the_scale_and_the_start(tmp_path, capsys):
    """scripts/e4_water.py train for a scan rung and a warm start on fake cells (one Adam step): the jsons record the
    pinned scale and the start; the warm start's learned dipole stays at about 1.5 times MB-pol's 1.864 D after one
    step; the summary prints the scan rung's final training loss and the warm start's gate."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["rungs"], cfg["seeds"] = ["M_inf", "M_1q070", "M_1w150"], [0]
    cfg["model"]["features"] = 8
    cfg["training"]["adam"].update(steps=1, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    import yaml
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmp_path / "out"
    E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=None, out_dir=str(out)))
    q = W.oh_equivalent_charges(SITE_Q, GAMMA)
    rq = json.load(open(out / "M_1q070_seed0.json"))
    assert rq["model"]["pinned_charge_scale"] == 0.7 and "charge_start_e" not in rq["model"]
    assert "learned_sources" not in rq["readings"]["cells"]["16"]["test"]
    rw = json.load(open(out / "M_1w150_seed0.json"))
    assert rw["model"]["charge_start_e"] == pytest.approx({"8": 1.5 * q[0], "1": 1.5 * q[1]})
    assert rw["model"]["charge_unit_e"] == E4.Q_UNIT_WARM and "pinned_charge_scale" not in rw["model"]
    v = rw["readings"]["cells"]["16"]["test"]["learned_sources"]
    P = np.load(tmp_path / "test_n16.npz")["positions"][0][:3]                    # the cells' own rigid monomer
    mu = 1.5 * np.linalg.norm(q @ P) * 4.803204
    assert v["dipole_debye"][0] == pytest.approx(mu, rel=0.05) and v["dipole_to_H_side"] == 1.0
    capsys.readouterr()
    E4.run_summary(SimpleNamespace(out_dir=str(out)))
    blocks = _blocks(capsys.readouterr().out)
    assert "final training loss" in blocks["M_1q070"] and "the gate" not in blocks["M_1q070"]
    assert "the gate (final training loss" in blocks["M_1w150"] and "learned sources: q_O" in blocks["M_1w150"]


def _blocks(text):
    """the summary's lines by rung."""
    out, cur = {}, None
    for line in text.splitlines():
        if line and not line.startswith(" ") and " (seeds " in line:
            cur = line.split(" (seeds ")[0]
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _run(rung, seed, forces, energy=0.02, loss=50.0, dipole=None):
    """a training json's readings: forces {size: rmse} on every split (frames 10 / 20 / 30 by split)."""
    cells = {}
    for n, f in forces.items():
        cells[n] = {}
        for split, nf in (("train", 10), ("valid", 20), ("test", 30)):
            cells[n][split] = {"frames": nf, "force_rigid_rmse": f, "energy_per_molecule_rmse": energy}
        if dipole is not None and n == "64":
            cells[n]["test"]["learned_sources"] = {"dipole_debye": [dipole, 0.02]}
    return {"rung": rung, "seed": seed, "readings": {"cells": cells}, "training": {"final_loss": {"total": loss}}}


def _runs(scan=None, sites=0.73, dagger=0.72, warm=None, warm_forces=0.74, ref=0.90, sizes=("64", "128", "256")):
    scan = scan or {0.4: 0.85, 0.7: 0.78, 1.0: 0.74, 1.1: 0.73, 1.45: 0.76}
    warm = warm or {0.5: (2.0, 2.05), 1.0: (2.06, 2.1), 1.5: (2.1, 2.2)}
    runs = {}
    add = lambda r: runs.setdefault(r["rung"], {}).__setitem__(r["seed"], r)
    for s in J7.SEEDS:
        add(_run("M_inf", s, {n: ref for n in sizes}, loss=60.0 + s))
        add(_run("M_1pm", s, {n: sites for n in sizes}))
        add(_run("M_16pm", s, {n: dagger for n in sizes}))
        for c, f in scan.items():
            add(_run(E4.scan_rung(c), s, {n: f for n in sizes}))
        for c0, d in warm.items():
            add(_run(E4.warm_rung(c0), s, {n: warm_forces for n in sizes}, dipole=d[s], loss=55.0))
    reference = {"M_inf": {s: _run("M_inf", s, {n: ref for n in sizes}, loss=x) for s, x in enumerate((57, 70, 75, 54))}}
    return runs, reference


def test_the_job7_readings():
    """J7.1-J7.5 on constructed jsons: the unseen frames pooled by frames (valid + test at 64, every split above), seed
    means, M_inf's reference from job 6; then each band's edge, a missing rung (not read), J7.5 read only when J7.3
    holds, and J7.4's spread and gate."""
    runs, ref = _runs()
    r = runs["M_inf"][0]
    r["readings"]["cells"]["64"]["valid"]["force_rigid_rmse"] = 0.6
    assert J7.pooled(r, "64") == pytest.approx(np.sqrt((20 * 0.36 + 30 * 0.81) / 50))
    r["readings"]["cells"]["64"]["valid"]["force_rigid_rmse"] = 0.9
    ev = J7.evaluate(runs, ref)
    assert ev["J7.1"]["holds"] and ev["J7.1"]["values"]["64"]["ratio"] == pytest.approx(0.73 / 0.72 - 1)
    assert ev["J7.2"]["holds"] and ev["J7.2"]["values"]["128"]["c"] == 1.1
    assert ev["J7.3"]["holds"] and ev["J7.3"]["values"]["256"]["against_M_inf"] == pytest.approx(0.73 / 0.90 - 1)
    c = ev["J7.4"]
    assert c["holds"] and c["loss_max"] == 75 and c["mean_debye"] == pytest.approx(np.mean([2.0, 2.05, 2.06, 2.1, 2.1, 2.2]))
    assert ev["J7.5"]["read"] and ev["J7.5"]["holds"] and ev["J7.5"]["values"]["64"]["ratio"] == pytest.approx(0.74 / 0.73 - 1)
    assert ev["c6_share"]["64"] == pytest.approx((0.73 - 0.72) / (0.90 - 0.72))
    assert ev["oh_against_sites"]["128"] == pytest.approx(0.74 / 0.73 - 1) and ev["final_loss"]["M_1w050"] == 55.0
    text = J7.report(ev)
    assert "J7.1: holds" in text and "J7.4: holds" in text and "lowest at c = 1.1" in text
    assert "the c6's share of the dagger's gain" in text and "final training loss, mean over seeds" in text
    # the bands' edges
    assert not J7.evaluate(*_runs(sites=0.78))["J7.1"]["holds"]                                  # 8.3 % above the dagger
    assert not J7.evaluate(*_runs(scan={0.4: 0.85, 0.7: 0.72, 1.0: 0.74, 1.1: 0.75, 1.45: 0.8}))["J7.2"]["holds"]
    ev = J7.evaluate(*_runs(scan={0.4: 0.9, 0.7: 0.88, 1.0: 0.85, 1.1: 0.86, 1.45: 0.89}))      # 5.6 % below M_inf
    assert not ev["J7.3"]["holds"] and ev["J7.5"]["read"] is False and ev["J7.5"]["holds"] is None
    assert "J7.5: not read" in J7.report(ev)
    ev = J7.evaluate(*_runs(warm={0.5: (1.2, 1.3), 1.0: (1.9, 2.0), 1.5: (2.6, 2.8)}, warm_forces=0.8))
    assert ev["J7.4"]["holds"] is False and not ev["J7.5"]["holds"]
    runs, ref = _runs()
    runs["M_1w100"][1]["training"]["final_loss"]["total"] = 80.0                                # above M_inf's largest
    assert J7.evaluate(runs, ref)["J7.4"]["holds"] is False
    del runs["M_1q110"]
    ev = J7.evaluate(runs, ref)
    assert ev["J7.2"]["holds"] is None and ev["J7.3"]["holds"] is None and "J7.2: not read" in J7.report(ev)
    del runs["M_1w150"][0]
    assert J7.convergence(runs, ref)["holds"] is None


def test_j76_reads_the_warm_starts_in_l12s_band_when_they_converged():
    """J7.6 from read_size's readings: every warm-started run's |b_T| on the first shells of 128 and 256 in L1.2's band
    (the sign free), at least four of the six; not read unless J7.4 holds."""
    cand = lambda bT: {"b_T": (bT, 0.01), "r": (1.8, 0.02)}
    vals = [1.30, -1.31, 1.29, 1.35, 1.5, 1.33]
    labels = [f"{w}_s{s}" for w in J7.WARM for s in J7.SEEDS]
    sizes = {n: {"shells": {"100": {"candidates": {**{lab: cand(v) for lab, v in zip(labels, vals)},
                                                    "M_1u_s0": cand(4.0), "perm": cand(1.455)}}}} for n in (128, 256)}
    out = SR.evaluate_job7(sizes, {"holds": True})["J7.6"]
    assert out["holds"] and out["verdict"] == "holds" and len(out["runs_in_band"]) == 5
    assert out["values"]["n128 (100), M_1w050_s1"][0] == pytest.approx(1.31)
    assert SR.evaluate_job7(sizes, {"holds": False})["J7.6"]["verdict"] == "not read (J7.4 does not hold)"
    assert SR.evaluate_job7(sizes, {"holds": None})["J7.6"]["verdict"] == "not read (J7.4 not read)"
    worse = {n: {"shells": {"100": {"candidates": {lab: cand(v) for lab, v in zip(labels, [1.5, 1.5, 1.5, 1.3, 1.3, 1.3])}}}}
             for n in (128, 256)}
    assert SR.evaluate_job7(worse, {"holds": True})["J7.6"]["verdict"] == "FAILS"
    assert SR.evaluate_job7({128: {"shells": {"100": {"candidates": {"C": cand(1)}}}}}, {"holds": True}) == {}


def test_the_s0_read_prints_j76(tmp_path, monkeypatch, capsys):
    """scripts/e4_sumrules.py read with --job7_dir: the warm-started runs' model files are read as candidates and J7.6
    is printed with its convergence (J7.4) from the training jsons."""
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
    out = tmp_path / "out"
    SR.run_mbpol(SimpleNamespace(mbx="x", data=str(tmp_path), out=str(out), sizes=[128, 256], part=None))
    for n in (128, 256):
        zero = np.zeros((len(SR.load_frames(str(tmp_path), n)["positions"]), 4 * n, 3))
        for lab in (f"{w}_s{s}" for w in J7.WARM for s in J7.SEEDS):
            np.savez(out / f"model_{lab}_n{n}.npz", q=0.8 * np.tile(SITE_Q, n), mu_full=zero, mu_ind=zero,
                     meta=json.dumps({"field_blind": True}))
    job7, job6 = tmp_path / "job7", tmp_path / "job6"
    job7.mkdir(); job6.mkdir()
    runs, ref = _runs()
    for d, rs in ((job7, runs), (job6, ref)):
        for name, by in rs.items():
            for s, r in by.items():
                json.dump(r, open(d / f"{name}_seed{s}.json", "w"))
    capsys.readouterr()
    bands, _ = SR.run_read(SimpleNamespace(data=str(tmp_path), out=str(out), json=str(out / "s0.json"), block=2,
                                           gate_dir=str(job6), job7_dir=str(job7)))
    text = capsys.readouterr().out
    assert bands["J7.6"]["converged"] is True and "J7.6: " in text
    assert "in the band at both shells" in text and "J7.4 from" in text and "holds)" in text
