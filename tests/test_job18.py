"""E4 job 18 (v0.0.43; notes/e4-job18-preregistration.md): the clusters, the truth's labels and response, training on
small clusters with the read-out calibrated on their response, and the readings K18.1-K18.7 (scripts/e4_job18.py)."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import sseft  # noqa: F401  (x64)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)

import e4_water as E4                                     # noqa: E402
import e4_job17 as J17                                    # noqa: E402
import e4_job18 as J                                      # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402

Q3 = np.array([-0.8, 0.4, 0.4])
ALPHA = 0.5                                                # A^3 per atom: the stand-in's induced dipoles, alpha E_ext


class Truth18:
    """a stand-in for MB-pol: open clusters, point charges Q3 between molecules and -600/R_OO^6 (kcal/mol), forces by
    central differences; the components as MBX names them; every atom's induced dipole ALPHA times the external field."""

    def __init__(self):
        self.field = np.zeros(3)

    def components(self, pos, cell=None):
        m = np.asarray(pos, float).reshape(-1, 3, 3)
        perm = disp = 0.0
        for a in range(len(m)):
            for b in range(a + 1, len(m)):
                r = np.linalg.norm(m[a][:, None] - m[b][None], axis=-1)
                perm += 332.0637 * (Q3[:, None] * Q3[None] / r).sum()
                disp -= 600.0 / r[0, 0] ** 6
        k = 1.0 / J.KCAL
        out = {t: 0.0 for t in J17.TERMS}
        out.update(dispersion=disp * k, electrostatics=perm * k, permanent=perm * k)
        return out

    def energy(self, pos, cell=None):
        c = self.components(pos)
        return c["dispersion"] + c["electrostatics"]

    def energy_forces(self, pos, cell=None):
        pos = np.asarray(pos, float)
        F = np.zeros_like(pos)
        h = 1e-4
        for i in range(len(pos)):
            for x in range(3):
                p, q = pos.copy(), pos.copy()
                p[i, x] += h; q[i, x] -= h
                F[i, x] = -(self.energy(p) - self.energy(q)) / (2 * h) / J.BOHR_PER_ANGSTROM
        return self.energy(pos), F

    def set_external_field(self, pos, field, cell_ang=None):
        self.field = np.zeros(3) if field is None else np.asarray(field, float)

    def sites(self, pos, cell=None):
        xyz = np.asarray(pos, float)
        return {"xyz": xyz, "charges": np.tile(Q3, len(xyz) // 3), "polarizabilities": np.full(len(xyz), ALPHA),
                "induced": np.tile(ALPHA * self.field, (len(xyz), 1))}

    def describe(self):
        return {"name": "truth18"}


SMALL = dict(TRAIN_SIZES=(2, 3), TRAIN_PER_SIZE=4, MONOMERS=2, VALID_EVERY=5, CALIBRATION=2, CAL_SIZES=(3,),
             TEST_PER_SIZE=2, PAIR_SUM_MAX=6, FAR_SIZES=(6,), TEST_SIZES=(3, 6))


@pytest.fixture
def small(tmp_path, monkeypatch):
    for k, v in SMALL.items():
        monkeypatch.setattr(J, k, v)
    src = str(tmp_path / "frames.npz")
    _fake_cells(src, 2, 40, 12.5, seed=7)
    a = SimpleNamespace(train_source=src, out=str(tmp_path / "job18.npz"), done=str(tmp_path / "done.json"),
                        probes=str(tmp_path / "none.npz"), far_xyz=str(tmp_path / "none.extxyz"),
                        test_sources={3: src, 6: src},
                        xyz={s: str(tmp_path / f"job18_{s}.extxyz") for s in ("train", "valid", "test")})
    J.make(Truth18(), a)
    return a


def test_job18_clusters(tmp_path):
    """a sparse cluster is whole molecules within the radius of its first; a compact one too wide for its cell is
    refused; the training set holds the monomers and, per size, compact and sparse clusters alternately; the test
    clusters come from distinct frame-centre pairs, per_size of each size."""
    from sseft.mbpol import random_box
    rng = np.random.default_rng(1)
    L = 12.5
    pos = random_box(40, np.eye(3) * L, rng, min_oo=2.5, min_any=1.5)
    c = J.cut_sparse(pos, L, 3, 4, rng, radius=6.0)
    m = c.reshape(-1, 3, 3)
    assert np.all(np.linalg.norm(m[:, 0] - m[0, 0], axis=1) <= 6.0 + 1e-9)
    assert np.allclose(np.linalg.norm(m[:, 1:] - m[:, :1], axis=-1), 0.9572, atol=1e-6)
    with pytest.raises(ValueError):
        J.compact(pos, L, 0, 40)
    tr = J.training_clusters(np.array([pos]), L, seed=0, sizes=(2, 3), per_size=4, monomers=2)
    assert [len(p) // 3 for p, *_ in tr] == [1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert [k for _, k, _, _ in tr][2:6] == ["compact", "sparse", "compact", "sparse"]
    src = str(tmp_path / "f.npz")
    _fake_cells(src, 2, 40, 12.5, seed=2)
    te = J.test_clusters({3: src, 5: src}, seed=0, per_size=3)
    assert [n for _, n, _, _ in te] == [3, 3, 3, 5, 5, 5] and all(len(p) == 3 * n for p, n, _, _ in te)
    assert len({(f, c) for _, n, f, c in te if n == 5}) == 3


def test_job18_make(small):
    """make: the training clusters with energies and forces, every fifth for validation, the calibration clusters'
    response (the stand-in's: ALPHA per atom), the test clusters with their components and pair sums; the extxyz files
    MACE reads; the done marker last."""
    C = J.load_clusters(small.out)
    d = np.load(small.out)
    assert len(d["train_energy"]) == 2 + 2 * 4 and int(d["train_valid"].sum()) == 2
    assert len(C["train"]) == 8 and len(C["calibration"]["positions"]) == 2
    for p, R in zip(C["calibration"]["positions"], C["calibration"]["R"]):
        assert R == pytest.approx(np.eye(3) * ALPHA * len(p), rel=1e-8)
    t = C["test"]
    assert sorted(t["size"].tolist()) == [3, 3, 6, 6]
    truth = Truth18()
    for p, e, n, ps in zip(t["positions"], t["energy"], t["size"], t["pair_sum"]):
        m = p.reshape(-1, 3, 3)
        assert e == pytest.approx(truth.energy(p)) and ps == pytest.approx(e)    # pairwise: the pairs are all of it
        assert len(m) == n
    tr = J17.read_extxyz(small.xyz["train"])
    assert len(tr) == 8 and "REF_forces" in tr[0][0]["Properties"]
    first = [ln for ln in open(small.xyz["train"]).read().splitlines()[2:3]][0].split()
    assert len(first) == 7
    p0, e0, f0 = C["train"][0]
    assert float(tr[0][0]["REF_energy"]) == pytest.approx(e0 * J.EV_PER_HARTREE)
    assert float(first[4]) == pytest.approx(f0[0, 0] * J.EV_ANG_PER_HARTREE_BOHR)
    assert len(J17.read_extxyz(small.xyz["valid"])) == 2 and len(J17.read_extxyz(small.xyz["test"])) == 4
    assert json.load(open(small.done))["hash"] == C["hash"]


def test_job18_trains_on_small_clusters(small, tmp_path):
    """train: E_0 alone and E4's model's rung on the training clusters, open; the latter's read-out matched on the
    probes and calibrated on the clusters' response (a cached read-out tagged by the calibration), held; each run
    read on the test clusters, the probes and T5, its calibration response beside."""
    cfg = _small_cfg(tmp_path)
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(FakeMBPol(), cfg), FakeMBPol(), cfg)
    raw = yaml.safe_load(open(os.path.join(REPO, "configs", "e4_job18.yaml")))
    raw["data"]["probes"] = cfg["data"]["probes"]
    raw["probes"] = cfg["probes"]
    raw["model"]["features"] = 8
    raw["training"]["adam"].update(steps=2, batch_size=4)
    raw["training"]["lbfgs"].update(steps=1, chunk_size=8)
    raw["job18"]["clusters"] = small.out
    path = tmp_path / "c.yaml"
    yaml.safe_dump(raw, open(path, "w"))
    out = tmp_path / "out"
    J.run_train(SimpleNamespace(config=str(path), rungs="M_inf,M_16C2pm_us_d20", seeds="0", out_dir=str(out),
                                chunk_atoms=3072))
    C = J.load_clusters(small.out)
    for name in ("M_inf", "M_16C2pm_us_d20"):
        r = json.load(open(out / f"{name}_seed0.json"))
        assert r["clusters_hash"] == C["hash"] and len(r["readings"]["test_energy"]) == 4
        assert np.all(np.isfinite(r["readings"]["test_energy"])) and "probes" in r["readings"]
        assert os.path.exists(out / f"{name}_seed0.params.pkl")
    r = json.load(open(out / "M_16C2pm_us_d20_seed0.json"))
    assert r["readings"]["calibration"]["trace_ratio_mean"] > 0 and r["job18"]["calibration"]["weight"] == 0.1
    assert r["matching"]["liquid_cells"] == 2 and r["matching"]["liquid_hash"].startswith("cl")
    assert any("_liqcl" in f for f in os.listdir(out))


def _run(C, scale, seed=0):
    t = C["test"]
    E = t["energy"] * (1 + scale * (t["size"] / t["size"].max()))     # an error growing with the size
    return {"seed": seed, "clusters_hash": C["hash"], "readings": {"test_energy": E.tolist()}}


def _mace(path, C, scale):
    t = C["test"]
    E = t["energy"] * (1 + scale * (t["size"] / t["size"].max()) ** 2) * J.EV_PER_HARTREE
    np.savez(path, job18_test__energy=E, job18_test__q=np.zeros(0), job18_test__offsets=np.zeros(len(E) + 1, int))
    return str(path)


def test_the_job18_readings(small, tmp_path):
    """K18.1-K18.7 on constructed models, and their edges."""
    C = J.load_clusters(small.out)
    t = C["test"]
    runs = {"model": {s: _run(C, 0.002, s) for s in (0, 1)}, "model_probes": {s: _run(C, 0.02, s) for s in (0, 1)},
            "dagger": {s: _run(C, 0.05, s) for s in (0, 1)}, "learned": {s: _run(C, 0.2, s) for s in (0, 1)}}
    mace = {"maceles": {s: _mace(tmp_path / f"m{s}.npz", C, 0.1) for s in (0, 1)}}
    ev = J.evaluate(C, runs, mace, far_path="none")
    for k in ("K18.2", "K18.3", "K18.4", "K18.5", "K18.6"):
        assert ev[k]["holds"], (k, ev[k])
    assert ev["K18.1"]["holds"] is False and ev["K18.7"]["holds"] is False
    err = (np.asarray(runs["model"][0]["readings"]["test_energy"]) - t["energy"]) * J.KCAL / t["size"]
    assert ev["models"]["model"][0]["sizes"][6]["rmse"] == pytest.approx(np.sqrt(np.mean(err[t["size"] == 6] ** 2)))
    assert ev["K18.3"]["ratio"] == pytest.approx(ev["seed_means"]["maceles"][6] / ev["seed_means"]["maceles"][3])
    text = J.report(ev)
    for k in ("K18.2: holds", "K18.1: FAILS", "maceles seed 1", "MB-pol: interaction per molecule"):
        assert k in text, k
    json.dumps(ev, default=float)
    mace1 = {"maceles": {s: _mace(tmp_path / f"n{s}.npz", C, 0.001) for s in (0, 1)}}
    ev1 = J.evaluate(C, runs, mace1, far_path="none")
    assert ev1["K18.1"]["holds"] and ev1["K18.2"]["holds"] is False
    runs4 = {**runs, "model": {s: _run(C, 1.0, s) for s in (0, 1)}}
    ev4 = J.evaluate(C, runs4, mace, far_path="none")
    assert ev4["K18.4"]["holds"] is False and ev4["K18.5"]["holds"] is False and ev4["K18.6"]["holds"] is False
    ev0 = J.evaluate(C, {}, {}, far_path="none")
    assert all(ev0[k]["holds"] is None for k in ("K18.1", "K18.2", "K18.3", "K18.4", "K18.5", "K18.6", "K18.7"))
    assert "not read" in J.report(ev0)
    bad = {"model": {0: {**_run(C, 0.0), "clusters_hash": "0" * 16}}}
    d = tmp_path / "res"; d.mkdir()
    json.dump(bad["model"][0] | {"rung": "M_inf"}, open(d / "M_inf_seed0.json", "w"))
    with pytest.raises(ValueError, match="other clusters"):
        J.load_runs({"cutoff": (str(d), "M_inf")}, C)
    assert J.mace_group("results/e4_job18_mace/maceles_s1/maceles_job18_s1_stagetwo_job18.npz") == ("maceles", 1)
    # v0.0.45: MACE's far probes and trimers, its isolated monomers subtracted when their energies are there
    import e4_far
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_job17 import _far_file
    far = _far_file(str(tmp_path / "probes.npz"))
    fr, mf = e4_far.frames(far), e4_far.monomer_frames(far)
    m_ev = 0.4 / J.KCAL * J.EV_PER_HARTREE
    for k, mono in ((0, True), (1, False)):
        path = tmp_path / f"far{k}_job18.npz"
        z = dict(np.load(mace["maceles"][k]))
        z["probes_mbpol_far__energy"] = np.array([m_ev * (len(pos) // 3) for _, pos, _, _ in fr])
        np.savez(path, **z)
        if mono:
            np.savez(tmp_path / f"far{k}_job18m.npz", probes_mbpol_far_monomers__energy=np.full(len(mf), m_ev))
        mace["maceles"][k] = str(path)
    evf = J.evaluate(C, runs, mace, far_path=far)
    r0, r1 = evf["models"]["maceles"][0], evf["models"]["maceles"][1]
    assert r0["monomer"] == pytest.approx(0.4) and np.allclose(r0["far_trimers"]["model"], 0.0, atol=1e-15)
    assert r1["monomer"] is None and np.allclose(np.asarray(r1["far_trimers"]["model"]) * J.KCAL, -1.2)
    text = J.report(evf)
    assert "its isolated monomer +0.4000 kcal/mol, subtracted" in text and "taken at zero" in text
