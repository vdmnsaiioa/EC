"""E4 job 17 (v0.0.42; notes/e4-job17-preregistration.md): the structures and their order, the truth's labels and their
check, a harness run's energies on them, and the readings K17.1-K17.9 (scripts/e4_job17.py)."""
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import jax
import pytest

import sseft  # noqa: F401  (x64)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)

import e4_water as E4                                     # noqa: E402
import e4_job17 as J                                      # noqa: E402
from sseft import model as M, water as W                  # noqa: E402
from sseft.units import BOHR_PER_ANGSTROM                 # noqa: E402
from test_e4_train import _fake_cells, SITE_Q, GAMMA      # noqa: E402

Q3 = np.array([-0.8, 0.4, 0.4])
SMALL = dict(SIZES=(2, 3), PAIRS=2, GAPS=(7.0, 9.0, 16.0, 25.0, 30.0), SLAB_FRAMES=2, SLAB_GAPS=(9.0, 12.0),
             SLAB_READ=(9.0,))


class Truth17:
    """a stand-in for MB-pol: open, point charges Q3 between molecules (permanent), -50/R_OO^4 (induced), -600/R_OO^6
    (dispersion) and a two-body term that ends at 6.5 A, in kcal/mol; periodic, -0.01/L_z hartree."""
    reach = 6.5

    def components(self, pos, cell=None):
        m = np.asarray(pos, float).reshape(-1, 3, 3)
        perm = ind = disp = two = 0.0
        for a in range(len(m)):
            for b in range(a + 1, len(m)):
                r = np.linalg.norm(m[a][:, None] - m[b][None], axis=-1)
                perm += 332.0637 * (Q3[:, None] * Q3[None] / r).sum()
                R = r[0, 0]
                ind -= 50.0 / R ** 4
                disp -= 600.0 / R ** 6
                two += max(0.0, self.reach - R) * 1e-3
        k = 1.0 / J.KCAL
        return {"one_body": 0.0, "two_body": two * k, "three_body": 0.0, "four_body": 0.0, "buckingham": 0.0,
                "dispersion": disp * k, "electrostatics": (perm + ind) * k, "permanent": perm * k, "induced": ind * k}

    def energy(self, pos, cell=None):
        c = self.components(pos)
        return sum(c[t] for t in ("one_body", "two_body", "three_body", "four_body", "buckingham", "dispersion",
                                  "electrostatics"))

    def energy_forces(self, pos, cell):
        return -0.01 / float(np.asarray(cell)[2, 2]), np.zeros_like(np.asarray(pos, float))

    def sites(self, pos, cell=None):                     # sseft.mbpol.MBPol.sites' dict
        xyz = np.asarray(pos, float)
        return {"xyz": xyz, "charges": np.tile(Q3, len(xyz) // 3), "polarizabilities": np.zeros(len(xyz)),
                "induced": np.zeros_like(xyz)}

    def describe(self):
        return {"name": "truth17"}


def _paths(tmp):
    return SimpleNamespace(source=str(tmp / "test_n16.npz"), out=str(tmp / "job17.npz"), slabs=str(tmp / "slabs.npz"),
                           clusters_xyz=str(tmp / "job17_clusters.extxyz"), slabs_xyz=str(tmp / "job17_slabs.extxyz"),
                           probes=str(tmp / "none.npz"), far_xyz=str(tmp / "none.extxyz"))


@pytest.fixture
def small(tmp_path, monkeypatch):
    for k, v in SMALL.items():
        monkeypatch.setattr(J, k, v)
    a = _paths(tmp_path)
    _fake_cells(a.source, 3, 16, 12.5, seed=3)
    J.make(Truth17(), a)
    return a


def test_job17_geometry():
    """a cluster is its centre's molecule and the nearest by the minimum image, whole; a placement leaves the smallest
    distance at the gap exactly and the centres of mass t apart; a slab keeps whole molecules, its O atoms in the
    cell's [0, L), with the gap's clearance."""
    rng = np.random.default_rng(0)
    L = 12.5
    from sseft.mbpol import random_box
    pos = random_box(16, np.eye(3) * L, rng, min_oo=2.5, min_any=1.5)
    A = J.cut_cluster(pos, L, 5, 4)
    m = J.whole_molecules(pos, L)
    dO = m[:, 0] - m[5, 0]; dO -= np.round(dO / L) * L
    nearest = np.sort(np.linalg.norm(dO, axis=1))[:4]
    got = np.sort(np.linalg.norm(A.reshape(-1, 3, 3)[:, 0] - A[0], axis=1))
    assert got == pytest.approx(nearest) and np.allclose(A[:3], m[5])
    oh = np.linalg.norm(A.reshape(-1, 3, 3)[:, 1:] - A.reshape(-1, 3, 3)[:, :1], axis=-1)
    assert np.allclose(oh, 0.9572, atol=1e-6)                                    # whole molecules
    A = A - J.com(A)
    B = (J.cut_cluster(pos, L, 9, 4) - J.com(J.cut_cluster(pos, L, 9, 4))) @ J.rotation(rng).T
    u = J.unit_vector(rng)
    ts = []
    for d in (7.0, 12.0, 30.0):
        t = J.placement(A, B, u, d)
        assert J.min_distance(A, B + t * u) == pytest.approx(d, abs=1e-9)
        assert np.linalg.norm(J.com(B + t * u) - J.com(A)) == pytest.approx(t, abs=1e-9)
        ts.append(t)
    assert ts[0] < ts[1] < ts[2]
    S, cell = J.slab(pos, L, 9.0)
    w = S.reshape(-1, 3, 3)
    assert np.all((w[:, 0] >= 0) & (w[:, 0] < L)) and np.allclose(np.linalg.norm(w[:, 1] - w[:, 0], axis=1), 0.9572)
    assert np.allclose(cell, np.diag([L, L, L + 9.0])) and J.slab_clearance(S, cell) >= 9.0 - 2 * 0.9572
    with pytest.raises(ValueError):
        J.cubic_length(np.array([[L, 1.0, 0], [0, L, 0], [0, 0, L]]))


def test_job17_make_with_a_stand_in_truth(small, tmp_path, monkeypatch):
    """make: the clusters' labels and their terms (the polynomials nothing between the clusters), one extxyz frame per
    A, B and AB in cluster_frames' order with the truth's energy in eV, the slabs in E4's data format; a polynomial that
    reaches between the clusters stops it."""
    T = J.load_truth(small.out)
    assert T["sizes"] == [2, 3] and list(T["gaps"]) == list(SMALL["GAPS"])
    truth = Truth17()
    for n in T["sizes"]:
        c = T["clusters"][n]
        assert c["A"].shape == (2, 3 * n, 3) and c["B"].shape == (2, 5, 3 * n, 3)
        inter = c["E_AB"] - c["E_A"][:, None] - c["E_B"][:, None]
        assert inter == pytest.approx(c["terms"]["electrostatics"] + c["terms"]["dispersion"], rel=1e-12, abs=1e-15)
        assert np.abs(c["terms"]["two_body"]).max() < 1e-18 and np.all(c["terms"]["induced"] < 0)
        for p in range(2):
            for g, d in enumerate(SMALL["GAPS"]):
                assert J.min_distance(c["A"][p], c["B"][p, g]) == pytest.approx(d, abs=1e-9)
            assert c["E_A"][p] == pytest.approx(truth.energy(c["A"][p]))
    fr = J.cluster_frames(T)
    assert len(fr) == sum(2 * (2 + 5) for _ in T["sizes"])
    xyz = J.read_extxyz(small.clusters_xyz)
    assert len(xyz) == len(fr)
    for (info, P), (n, p, kind, g, pos) in zip(xyz, fr):
        c = T["clusters"][n]
        E = c["E_A"][p] if kind == "A" else c["E_B"][p] if kind == "B" else c["E_AB"][p, g]
        assert float(info["REF_energy"]) == pytest.approx(E * J.EV_PER_HARTREE) and np.allclose(P, pos, atol=1e-9)
        assert info["pbc"] == "F F F" and info["kind"] == kind
    st = W.e4_structures(small.slabs, pins=True)
    S = np.load(small.slabs)
    assert len(st) == 4 and all(s.pbc for s in st) and list(S["slab_gap"]) == [9.0, 12.0, 9.0, 12.0]
    assert np.allclose(S["energy"], [-0.01 / (12.5 + g) for g in S["slab_gap"]])
    sx = J.read_extxyz(small.slabs_xyz)
    assert len(sx) == 4 and sx[1][0]["pbc"] == "T T T" and float(sx[1][0]["Lattice"].split()[-1]) == pytest.approx(24.5)
    a = _paths(tmp_path / "x"); os.makedirs(tmp_path / "x")
    _fake_cells(a.source, 3, 16, 12.5, seed=3)
    monkeypatch.setattr(J, "GAPS", (5.0, 9.0))
    with pytest.raises(RuntimeError, match="reaches between"):
        J.make(Truth17(), a)


def _run(tmp, name, source):
    """a run as scripts/e4_water.py train writes it, at its initialisation: <dir>/<rung>_seed0.json and .params.pkl."""
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8}})
    st = W.e4_structures(source, pins=True)
    rung, rung_open = E4.rungs_for(name, cfg, st[0].cell)
    params = M.init_params(jax.random.PRNGKey(3), rung_open, energy_scale=1e-3)
    d = tmp / "e4_jobX"; d.mkdir(exist_ok=True)
    stem = str(d / f"{name}_seed0")
    with open(stem + ".json", "w") as fh:
        json.dump({"rung": name, "seed": 0, "config": cfg, "readings": {"probes": {"x": 1}, "far_trimers": None}}, fh)
    with open(stem + ".params.pkl", "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)
    return stem


def _site_coulomb(A, B):
    """the pinned site model's Coulomb interaction of two clusters (hartree): atoms' SITE_Q[:3], an M-site per
    molecule at O + GAMMA/2 (H1 + H2 - 2 O) with SITE_Q[3]."""
    def sites(P):
        m = np.asarray(P).reshape(-1, 3, 3)
        Ms = m[:, 0] + GAMMA / 2 * (m[:, 1] + m[:, 2] - 2 * m[:, 0])
        return np.concatenate([m.reshape(-1, 3), Ms]), np.concatenate([np.tile(SITE_Q[:3], len(m)),
                                                                        np.full(len(m), SITE_Q[3])])
    (xa, qa), (xb, qb) = sites(A), sites(B)
    r = np.linalg.norm(xa[:, None] - xb[None], axis=-1) * BOHR_PER_ANGSTROM
    return float((qa[:, None] * qb[None] / r).sum())


def test_job17_harness_runs(small, tmp_path):
    """a harness run on the clusters and slabs: E_0 alone gives exactly nothing between clusters beyond r_c and
    nothing across a slab's gap; the pinned site model's Coulomb band is its charges' Coulomb interaction beyond r_c;
    the run's own infrared readings are copied."""
    out = tmp_path / "out"
    T = J.load_truth(small.out)
    for name in ("M_inf", "M_1pm"):
        stem = _run(tmp_path, name, small.source)
        J.run_harness(SimpleNamespace(run=stem, source=small.source, truth=small.out, slabs=small.slabs,
                                      out_dir=str(out), chunk=4))
        with open(out / f"e4_jobX__{name}__seed0.json") as fh:
            r = json.load(fh)
        assert r["truth_hash"] == T["hash"] and r["source_readings"]["probes"] == {"x": 1}
        for n in T["sizes"]:
            c, rc = T["clusters"][n], r["clusters"][str(n)]
            inter = np.asarray(rc["E_AB"]) - np.asarray(rc["E_A"])[:, None] - np.asarray(rc["E_B"])[:, None]
            if name == "M_inf":
                assert np.abs(inter).max() < 1e-12
            else:
                ref = np.array([[_site_coulomb(c["A"][p], c["B"][p, g]) for g in range(5)] for p in range(2)])
                assert inter == pytest.approx(ref, rel=1e-8)
        S = np.load(small.slabs)
        d = J.slab_changes(r["slabs"], S["slab_frame"], S["slab_gap"])
        assert set(d) == {9.0} and np.all(np.isfinite(d[9.0]))
        if name == "M_inf":
            assert np.abs(d[9.0]).max() < 1e-8


def _far_file(path):
    """two probes (a core of two molecules and the probe) and two far trimers, with made-up truths."""
    rng = np.random.default_rng(4)
    mol = lambda c: np.array([[0, 0, 0], [0.757, 0, 0.586], [-0.757, 0, 0.586]]) + c
    probes = [np.concatenate([mol([0, 0, 0]), mol([3, 0, 0]), mol([0, 0, d])]) for d in (10.0, 12.0)]
    trimers = np.array([np.concatenate([mol([0, 0, 0]), mol([8, 0, 0]), mol([0, 9, 0])]) for _ in range(2)])
    np.savez(path, test_offsets=np.array([0, 9, 18]), test_positions=np.concatenate(probes), test_d=np.array([10.0, 12.0]),
             test_energy=np.array([-0.002, -0.001]), test_E_core=np.array([-0.0015, -0.0008]),
             test_pair_energy=np.array([-0.0004, -0.0002]), trimer_positions=trimers,
             trimer_pairs=rng.normal(size=(2, 3)) * 1e-4, trimer_three_body=np.array([2e-6, -1e-6]))
    return path


def _les(path, T, S, far, c=0.6, net=0.05, seed=0, broken=0.0, slab_scale=0.5, far_exact=False, mono=0.7, monomers=True):
    """MACE-LES's file as baselines/mace/energies.py writes it, constructed: latent charges c Q3 plus a net charge per
    molecule (normal, sd net), its cluster interactions exactly its charges' (+ broken on one), the slabs at slab_scale
    times the truth, the far frames at zero (or the truth's: far_exact) plus mono kcal/mol for each molecule (its isolated
    monomer's energy, which the first read took at zero); with monomers, <path>m.npz: its energies of every molecule of
    the far frames alone (v0.0.45)."""
    path = str(path).replace(".npz", "_job17.npz")
    import e4_far
    rng = np.random.default_rng(seed)
    fr = J.cluster_frames(T)
    charges = {}
    E, Q = [], []
    for n, p, kind, g, pos in fr:
        if kind in ("A", "B"):
            q = (c * Q3[None] + rng.normal(scale=net, size=(n, 1)) / 3).reshape(-1)
            charges[(n, p, kind)] = q
            E.append(0.0); Q.append(q)
        else:
            qa, qb = charges[(n, p, "A")], charges[(n, p, "B")]
            A, B = T["clusters"][n]["A"][p], T["clusters"][n]["B"][p, g]
            E.append(J.coulomb_cross(qa, A, qb, B)); Q.append(np.concatenate([qa, qb]))
    E[-1] += broken
    off = np.concatenate([[0], np.cumsum([len(q) for q in Q])])
    out = {"job17_clusters__energy": np.array(E), "job17_clusters__q": np.concatenate(Q), "job17_clusters__offsets": off}
    Es = np.asarray(S["energy"]) * slab_scale * J.EV_PER_HARTREE
    qs = [np.tile(c * Q3, len(p) // 3) for p in S["positions"]]
    out.update({"job17_slabs__energy": Es, "job17_slabs__q": np.concatenate(qs),
                "job17_slabs__offsets": np.concatenate([[0], np.cumsum([len(q) for q in qs])])})
    ff = e4_far.frames(far)
    if far_exact:
        Ef = []
        for kind, _, Eref, tags in ff:
            Ef.append(0.0 if Eref is None else Eref * J.EV_PER_HARTREE)
        d = np.load(far)
        k = 0
        for kind, _, Eref, tags in ff:                    # the pairs: the truth's pair energy, on the first pair
            if kind == "pair" and tags["i"] == 0:
                Ef[k] = d["test_pair_energy"][tags["k"]] * J.EV_PER_HARTREE
            k += 1
    else:
        Ef = [0.0] * len(ff)
    m_ev = mono / J.KCAL_PER_EV
    Ef = [e + m_ev * (len(pos) // 3) for e, (_, pos, _, _) in zip(Ef, ff)]
    out.update({"probes_mbpol_far__energy": np.array(Ef), "probes_mbpol_far__q": np.zeros(0),
                "probes_mbpol_far__offsets": np.zeros(len(ff) + 1, int)})
    np.savez(path, model=np.array("x"), **out)
    if monomers:
        nm = len(e4_far.monomer_frames(far))
        np.savez(J.monomer_npz(path), model=np.array("x"), probes_mbpol_far_monomers__energy=np.full(nm, m_ev),
                 probes_mbpol_far_monomers__q=np.tile([-0.5, 0.26, 0.26], nm),
                 probes_mbpol_far_monomers__offsets=np.arange(nm + 1) * 3)
    return path


def _harness_run(T, S, scale_c=1.0, scale_s=1.0, hash_=None):
    cl = {}
    for n in T["sizes"]:
        c = T["clusters"][n]
        cl[str(n)] = {"E_A": c["E_A"].tolist(), "E_B": c["E_B"].tolist(),
                      "E_AB": (c["E_A"][:, None] + c["E_B"][:, None]
                               + (c["E_AB"] - c["E_A"][:, None] - c["E_B"][:, None]) * scale_c).tolist()}
    return {"truth_hash": hash_ or T["hash"], "clusters": cl, "slabs": (np.asarray(S["energy"]) * scale_s).tolist(),
            "source_readings": {"probes": None, "far_trimers": {"slope": 1.04}, "liquid": {"eps_inf_model": 1.6893,
                                                                                         "eps_inf_truth": 1.6885}}}


def test_the_job17_readings(small, tmp_path, monkeypatch):
    """K17.1-K17.9 on constructed models, each band's edge, and a model read on another truth refused."""
    monkeypatch.setattr(J, "TAIL_GAPS", (25.0, 30.0))
    T = J.load_truth(small.out)
    S = dict(np.load(small.slabs))
    far = _far_file(str(tmp_path / "probes.npz"))
    H = {"model": {s: _harness_run(T, S, 1.01, 1.02) for s in (0, 1)},
         "learned": {s: _harness_run(T, S, 0.8) for s in (0, 1)}}
    L = {"maceles": {s: _les(tmp_path / f"les{s}.npz", T, S, far, seed=s) for s in (0, 1)}}
    ev = J.evaluate(T, S, H, L, far_path=far)
    for k in ("K17.1", "K17.2", "K17.3", "K17.4", "K17.5", "K17.6", "K17.7", "K17.8", "K17.9"):
        assert ev[k]["holds"], (k, ev[k])
    assert ev["K17.1"]["worst"] < 1e-9 and ev["K17.1"]["charges_worst"] < 1e-12
    assert ev["K17.4"]["by_seed"][0][0] == pytest.approx(0.01)
    assert ev["K17.7"]["by_seed"][1][0] == pytest.approx(0.02)
    assert ev["K17.8"]["by_seed"][0][0] == pytest.approx(0.5)
    assert ev["harness"]["model"][0]["eps_inf"]["model"] == 1.6893
    assert ev["les"]["maceles"][0]["net_charge_rms"]["2"] > 0 and ev["les"]["maceles"][0]["probes"]["count"] == 2
    # v0.0.45: the isolated monomers at 0.7 kcal/mol: the first read's T5 is -3 x 0.7, exactly the trimers' monomers
    r = ev["les"]["maceles"][0]
    assert r["monomers"]["mean"] == pytest.approx(0.7) and r["monomers"]["net_q_rms"] == pytest.approx(0.02)
    assert ev["K17.2"]["worst"] < 1e-9 and ev["K17.2"]["uncorrected_worst"] == pytest.approx(2.1)
    assert r["uncorrected"]["t5_minus_monomers"] < 1e-9
    text = J.report(ev)
    for k in ("K17.1: holds", "K17.2: holds", "K17.6: holds", "K17.9: holds", "MACE-LES maceles seed 1",
              "model seed 0: clusters", "R^-3", "its isolated monomer +0.7000 kcal/mol", "the first read's T5"):
        assert k in text, k
    # without the monomers' energies K17.2 and K17.3 are not read
    Lm = {"maceles": {s: _les(tmp_path / f"m{s}.npz", T, S, far, seed=s, monomers=False) for s in (0, 1)}}
    evm = J.evaluate(T, S, H, Lm, far_path=far)
    assert evm["K17.2"]["holds"] is None and evm["K17.3"]["holds"] is None and evm["K17.1"]["holds"]
    assert evm["K17.2"]["uncorrected_worst"] == pytest.approx(2.1) and "run_e4_job17m.sbatch" in J.report(evm)
    json.dumps(ev, default=float)
    # the edges
    L1 = {"maceles": {0: _les(tmp_path / "b0.npz", T, S, far, broken=1e-6 / J.KCAL_PER_EV * 2), 1: L["maceles"][1]}}
    assert J.evaluate(T, S, H, L1, far_path=far)["K17.1"]["holds"] is False
    L3 = {"maceles": {s: _les(tmp_path / f"e{s}.npz", T, S, far, seed=s, far_exact=True) for s in (0, 1)}}
    assert J.evaluate(T, S, H, L3, far_path=far)["K17.3"]["holds"] is False
    L5 = {"maceles": {s: _les(tmp_path / f"f{s}.npz", T, S, far, c=1.0, net=0.0, seed=s, slab_scale=1.0) for s in (0, 1)}}
    ev5 = J.evaluate(T, S, H, L5, far_path=far)
    assert ev5["K17.6"]["holds"] is False and ev5["K17.8"]["holds"] is False
    assert ev5["K17.6"]["ratio"][0] == pytest.approx([1.0, 1.0])
    H4 = {**H, "model": {0: _harness_run(T, S, 1.06, 1.12)}}
    ev4 = J.evaluate(T, S, H4, L, far_path=far)
    assert ev4["K17.4"]["holds"] is False and ev4["K17.7"]["holds"] is False
    H9 = {**H, "learned": {0: _harness_run(T, S, 3.0)}}
    assert J.evaluate(T, S, H9, L, far_path=far)["K17.9"]["holds"] is False
    ev0 = J.evaluate(T, S, {}, {}, far_path=far)
    assert all(ev0[k]["holds"] is None for k in ("K17.1", "K17.2", "K17.3", "K17.4", "K17.5", "K17.6", "K17.7", "K17.8",
                                                 "K17.9"))
    assert "not read" in J.report(ev0)
    out = tmp_path / "o"; out.mkdir()
    with open(out / "e4_job12__M_16C2pm_us_d20__seed0.json", "w") as fh:
        json.dump(_harness_run(T, S, hash_="0" * 16), fh)
    with pytest.raises(ValueError, match="another truth"):
        J.load_harness(str(out), T)
    assert J.les_group("results/e4_job15_les/release_o050_n64_s1/maceles_n64_s1_stagetwo_job17.npz") == ("release_o050", 1)
    assert J.les_group("results/e4_baselines/maceles_n64_s0/x.npz") == ("maceles", 0)


def test_the_far_monomers(tmp_path):
    """v0.0.45: every molecule of the far frames, alone, in order; a model that is the sum of its isolated molecules
    (each with its own energy, a function of its coordinates) reads no interaction, no non-additive energy and no three-body energy once they are
    subtracted, and the first read's T5 is minus the trimer's monomers."""
    import e4_far
    far = _far_file(str(tmp_path / "probes.npz"))
    fr, mf = e4_far.frames(far), e4_far.monomer_frames(far)
    assert len(mf) == sum(len(p) // 3 for _, p, _, _ in fr) == 36
    path = e4_far.write_monomers(far, str(tmp_path / "mono.extxyz"))
    lines = open(path).read().splitlines()
    assert len(lines) == 5 * len(mf) and lines[1].startswith("Properties=") and 'pbc="F F F"' in lines[1]
    f, m, pos = mf[7]
    assert np.allclose(pos, fr[f][1][3 * m:3 * m + 3]) and lines[5 * 7 + 1].endswith(f"f={f} m={m}")
    e_mol = np.array([0.05 * np.sin(pos.sum()) for _, _, pos in mf])   # eV, each molecule's own, by its coordinates
    E = np.zeros(len(fr))
    np.add.at(E, [f for f, _, _ in mf], e_mol)
    out = e4_far.readings(far, E, monomers_ev=e_mol)
    b = out["probes"]
    ref = np.load(far)
    assert b["nonadditive_rmse"] == pytest.approx(b["nonadditive_rms"]) and b["interaction_rmse"] == pytest.approx(
        b["interaction_rms"])                                    # the model's readings are zero: the truth's rms left
    assert np.allclose(out["far_trimers"]["model"], 0.0, atol=1e-15)
    raw = e4_far.readings(far, E)
    tri = e4_far.monomer_sums(far, e_mol)[[k for k, (kind, *_) in enumerate(fr) if kind == "trimer"]]
    assert np.allclose(raw["far_trimers"]["model"], -tri, atol=1e-15)
    assert ref["trimer_three_body"].shape == (2,)
    with pytest.raises(ValueError, match="monomer energies"):
        e4_far.readings(far, E, monomers_ev=e_mol[:-1])
