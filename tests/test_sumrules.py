"""
E4's sum-rule reader S0 (scripts/e4_sumrules.py; notes/e4-s0-preregistration.md): the charge density of point charges
and point dipoles and the molecules' dipoles at their centres of mass give the same longitudinal polarisation at small
k; the normalisation of L and T on independent dipoles; the regression slopes; the shells and the unit factor against
the note's table; the jackknife and the intercept fit; MBX's step on a fake MBX; a frozen form's own dipoles (the
order-0 mu_ind when its orders >= 1 are off); the read on fake sets end to end.
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
import e4_sumrules as SR  # noqa: E402

GAMMA = 0.4267
SITE_Q = np.array([0.0, 0.55, 0.55, -1.10])


def _molecules(n, L, rng, spread=None):
    """n rigid waters at random places and orientations in a cubic cell of side L (A), molecules whole: (3n, 3)."""
    from sseft.water import monomer_positions, random_rotation
    m = monomer_positions() - monomer_positions()[0]
    out = []
    for _ in range(n):
        R = random_rotation(rng)
        out.append(m @ R.T + rng.uniform(0, spread or L, 3))
    return np.concatenate(out)


def test_unit_factor_and_shells_match_the_note():
    """beta e^2 / A at 300 K is 557.0; the first shells and 1/eps_inf(k) of E4's cells are the note's section 3.4."""
    assert SR.BETA_E2 == pytest.approx(557.0, abs=0.01)
    for side, shell, k, inv in ((12.43, "100", 0.5055, 0.564), (15.66, "110", 0.5674, 0.556), (19.73, "100", 0.3184, 0.581),
                                (24.86, "111", 0.4378, 0.571), (31.32, "110", 0.2837, 0.583)):
        kv = SR.kvectors(np.eye(3) * side, shell)
        assert np.allclose(np.linalg.norm(kv, axis=1), k, atol=2e-4)
        assert SR.inv_eps_inf(k) == pytest.approx(inv, abs=1e-3)
    assert len(SR.kvectors(np.eye(3), "100")) == 3 and len(SR.kvectors(np.eye(3), "110")) == 6


def test_charges_and_molecular_dipoles_give_one_longitudinal_polarisation_at_small_k():
    """P_L = i rho_k / |k| from the site charges and point dipoles equals k^.P from the molecules' dipoles at their
    centres of mass to O(ka); a point dipole's rho_k is -i (k.mu) e^{-ik.r} exactly."""
    rng = np.random.default_rng(0)
    side = 2000.0
    pos = _molecules(20, side, rng)[None]
    xyz = SR.site_positions(pos, GAMMA)
    q = np.tile(SITE_Q, 20)
    mu = rng.normal(scale=0.05, size=xyz.shape)
    com = SR.centres(pos)
    kv = SR.kvectors(np.eye(3) * side, "110")
    PL, PT = SR.channels(xyz, q, mu, com, kv)
    P = SR.polarisation_k(SR.molecular_dipoles(xyz, q, mu, com), com, kv)
    kh = kv / np.linalg.norm(kv, axis=1)[:, None]
    assert np.allclose(PL, np.einsum("fkx,kx->fk", P, kh), rtol=0, atol=2e-3 * np.abs(PL).max())
    assert np.allclose(np.einsum("fkx,kx->fk", PT, kh), 0, atol=1e-12)
    r0, m0 = np.array([[[1.0, 2.0, 3.0]]]), np.array([[[0.1, -0.2, 0.3]]])
    rho = SR.charge_density(r0, np.zeros(1), m0, kv[:1])
    assert rho[0, 0] == pytest.approx(-1j * (kv[0] @ m0[0, 0]) * np.exp(-1j * kv[0] @ r0[0, 0]))


def test_independent_dipoles_give_equal_channels_at_the_right_normalisation():
    """dipoles of fixed size and random orientation, independent from molecule to molecule: T / L = 1 and
    L = 4 pi beta e^2 N <d^2> / 3V, within the sampling error."""
    rng = np.random.default_rng(1)
    side, n, F, d = 30.0, 50, 300, 0.3
    pos = np.array([_molecules(n, side, rng) for _ in range(F)])
    xyz = SR.site_positions(pos, GAMMA)
    mu = np.zeros_like(xyz)
    u = rng.normal(size=(F, n, 3)); u *= d / np.linalg.norm(u, axis=2, keepdims=True)
    mu[:, 0::4] = u
    q = np.zeros(4 * n)
    com = SR.centres(pos)
    kv = SR.kvectors(np.eye(3) * side, "100")
    PL, PT = SR.channels(xyz, q, mu, com, kv)
    L = 4 * np.pi * SR.BETA_E2 / side ** 3 * np.mean(np.abs(PL) ** 2)
    T = 4 * np.pi * SR.BETA_E2 / side ** 3 * np.mean(0.5 * np.sum(np.abs(PT) ** 2, axis=2))
    expect = 4 * np.pi * SR.BETA_E2 * n * d ** 2 / (3 * side ** 3)
    assert L == pytest.approx(expect, rel=0.12) and T == pytest.approx(expect, rel=0.12)
    assert T / L == pytest.approx(1.0, rel=0.15)


def test_regression_slopes_and_the_jackknife():
    """P = 2Y gives b_L = b_T = 2 and r = 1 exactly; a constant statistic has no error; the jackknife of a mean is the
    blocks' standard error; blocks follow the trajectories and their time order."""
    rng = np.random.default_rng(2)
    side = 15.0
    pos = np.array([_molecules(8, side, rng) for _ in range(6)])
    xyz = SR.site_positions(pos, GAMMA)
    q = np.tile(SITE_Q, 8)
    mu = rng.normal(scale=0.05, size=xyz.shape)
    com = SR.centres(pos)
    kv = SR.kvectors(np.eye(3) * side, "100")
    PL, PT = SR.channels(xyz, 2 * q, 2 * mu, com, kv)
    YL, YT = SR.channels(xyz, q, mu, com, kv)
    bL = np.real(np.sum(PL * np.conj(YL))) / np.sum(np.abs(YL) ** 2)
    bT = np.real(np.sum(PT * np.conj(YT))) / np.sum(np.abs(YT) ** 2)
    assert bL == pytest.approx(2.0, rel=1e-12) and bT == pytest.approx(2.0, rel=1e-12)
    ids = SR.blocks(np.array([0, 0, 0, 0, 0, 1, 1, 1]), np.array([4, 3, 2, 1, 0, 0, 1, 2.0]), block=2)
    assert ids.tolist() == [2, 1, 1, 0, 0, 3, 3, 4]
    x = rng.normal(size=400)
    ids = np.repeat(np.arange(100), 4)
    v, e = SR.jackknife({"x": x, "one": np.ones(400)}, ids, lambda t: t["x"] / t["one"])
    blk = x.reshape(100, 4).mean(1)
    assert v == pytest.approx(x.mean()) and e == pytest.approx(blk.std(ddof=1) / 10, rel=1e-9)
    assert SR.jackknife({"x": np.ones(8), "one": np.ones(8)}, np.arange(8), lambda t: t["x"] / t["one"])[1] == 0.0
    A, sA, B, chi2 = SR.fit_intercept([(0.2, 0.6 + 0.3 * 0.04, 0.01), (0.3, 0.6 + 0.3 * 0.09, 0.01),
                                       (0.4, 0.6 + 0.3 * 0.16, 0.02)])
    assert A == pytest.approx(0.6) and B == pytest.approx(0.3) and chi2 == pytest.approx(0.0, abs=1e-20) and sA > 0


def test_site_positions_and_centres():
    """the M-site sits at r_O + gamma/2 (r_H1 + r_H2 - 2 r_O); the centre of mass weights O by 15.999 and H by 1.008."""
    rng = np.random.default_rng(3)
    pos = _molecules(3, 20.0, rng)[None]
    s = SR.site_positions(pos, GAMMA)[0].reshape(3, 4, 3)
    P = pos[0].reshape(3, 3, 3)
    assert np.allclose(s[:, :3], P)
    assert np.allclose(s[:, 3], P[:, 0] + 0.5 * GAMMA * (P[:, 1] + P[:, 2] - 2 * P[:, 0]))
    assert np.allclose(SR.centres(pos)[0], (15.999 * P[:, 0] + 1.008 * (P[:, 1] + P[:, 2])) / 18.015)


def _fake_sets(d, sizes, rng):
    """fake E4 sets: two trajectories of each size, train and test, molecules whole, at a low density."""
    from sseft.mbpol import random_box
    for n, side in sizes:
        for split, src in (("train", 0), ("test", 1)):
            pos = np.array([random_box(n, np.eye(3) * side, rng, min_oo=2.5, min_any=1.5) for _ in range(8)])
            np.savez(d / f"{split}_n{n}.npz", positions=pos, cell=np.eye(3) * side, energy=np.zeros(8),
                     forces=np.zeros_like(pos), site_q=SITE_Q, site_gamma=np.array(GAMMA),
                     source=np.full(8, src), time_ps=np.arange(8) * 0.5)


class _FakeMB:
    """MBX's interface for S0: zero-field sites, O, H, H, M, with induced dipoles 0.1 x the O position's direction."""
    def __init__(self, *a, **k):
        self.calls = 0

    def set_external_field(self, pos, field, cell=None):
        assert field is None

    def sites(self, pos, cell=None):
        self.calls += 1
        xyz = SR.site_positions(np.asarray(pos)[None], GAMMA)[0]
        mu = np.zeros_like(xyz)
        mu[0::4] = 0.1 * xyz[0::4] / np.linalg.norm(xyz[0::4], axis=1, keepdims=True)
        return {"xyz": xyz, "charges": np.tile(SITE_Q, len(pos) // 3), "induced": mu}

    def finalize(self):
        pass


def test_mbpol_step_on_a_fake_mbx_and_the_read_end_to_end(tmp_path, monkeypatch):
    """the mbpol step writes every frame's sites in two parts and checks the dynamics' settings on the first frame; a
    model file equal to MB-pol's polarisation reads r = 1 exactly (S0.4); the read evaluates S0.1-S0.4 on what is
    there and writes its json."""
    from sseft import water as W, mbpol as MB
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
    for part in ("0/2", "1/2"):
        SR.run_mbpol(SimpleNamespace(mbx="x", data=str(tmp_path), out=str(out), sizes=[64, 128, 256], part=part))
    assert sorted(os.listdir(out)) == ["mbpol_n128_p0of2.npz", "mbpol_n128_p1of2.npz", "mbpol_n256_p0of2.npz",
                                       "mbpol_n256_p1of2.npz"]
    meta = json.loads(str(np.load(out / "mbpol_n128_p0of2.npz")["meta"]))
    assert meta["dynamics_settings_rel_rms"] == 0.0 and meta["of"] == 16
    for n in (128, 256):                                  # arm C's file: MB-pol's own polarisation
        fr = SR.load_frames(str(tmp_path), n)
        mb = SR.load_mbpol(str(out), n, len(fr["positions"]))
        assert mb["missing"] == 0 and np.allclose(mb["xyz"], SR.site_positions(fr["positions"], GAMMA))
        np.savez(out / f"model_C_n{n}.npz", q=np.tile(SITE_Q, n), mu_full=mb["mu"], mu_ind=0.5 * mb["mu"])
    bands, sizes = SR.run_read(SimpleNamespace(data=str(tmp_path), out=str(out), json=str(out / "s0.json"), block=2))
    assert set(bands) == {"S0.1", "S0.2", "S0.4"} and bands["S0.4"]["holds"]
    for v in bands["S0.4"]["values"].values():
        assert v[0] == pytest.approx(1.0, rel=1e-12)
    c = sizes[128]["shells"]["100"]["candidates"]
    assert set(c) == {"perm", "C", "C_o0"} and c["C"]["b_L"][0] == pytest.approx(1.0, rel=1e-12)
    assert bands["S0.1"]["points"] == 5 and np.isfinite(bands["S0.1"]["value"])      # k <= 0.46: 2 shells at 20 A, 3 at 25
    j = json.load(open(out / "s0.json"))
    assert set(j["sizes"]) == {"128", "256"} and j["constants"]["block"] == 2


def test_model_step_gives_mu_ind_when_the_orders_above_zero_are_off(tmp_path):
    """the model step on fake 16-molecule cells: dipoles at every site (zero at the M-sites), in e A; with the read-out's
    orders >= 1 off, -dE/dE_ext,i is the order-0 sector's mu_ind exactly; with them on it differs."""
    import yaml
    import e4_water as E4
    from test_e4_train import _fake_cells
    from test_periodic import switched_on
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8, "mutual_edge_angstrom": 2.0}})
    data = tmp_path / "data"; data.mkdir()
    _fake_cells(data / "test_n16.npz", 2, 16, 12.5, seed=5)
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    st = __import__("sseft.water", fromlist=["e4_structures"]).e4_structures(str(data / "test_n16.npz"), pins=True)
    rung_cell, rung_open = E4.rungs_for("M_16C2pm_us", cfg, st[0].cell)
    on = switched_on(rung_cell)["readout"]
    import jax
    on = jax.tree_util.tree_map(np.asarray, on)
    for label, ro in (("on", on), ("off", E4.order_zero(on, 16))):
        with open(tmp_path / f"{label}.pkl", "wb") as fh:
            pickle.dump({"rung": repr(rung_open) + "_liqx", "scale": 1e-3, "readout": ro}, fh)
        SR.run_model(SimpleNamespace(config=str(tmp_path / "c.yaml"), readout=str(tmp_path / f"{label}.pkl"),
                                     rung="M_16C2pm_us", label=label, sizes=[16], data=str(data),
                                     out=str(tmp_path / "out")))
    off = np.load(tmp_path / "out" / "model_off_n16.npz")
    onf = np.load(tmp_path / "out" / "model_on_n16.npz")
    assert off["mu_full"].shape == (2, 64, 3) and np.all(off["mu_full"][:, 3::4] == 0)
    assert np.abs(off["mu_ind"]).max() > 1e-4
    assert np.allclose(off["mu_full"], off["mu_ind"], rtol=0, atol=1e-10 * np.abs(off["mu_ind"]).max())
    assert not np.allclose(onf["mu_full"], onf["mu_ind"], rtol=0, atol=1e-6 * np.abs(onf["mu_ind"]).max())
    assert np.allclose(off["q"], np.tile(SITE_Q, 16)) or off["q"].shape == (64,)
