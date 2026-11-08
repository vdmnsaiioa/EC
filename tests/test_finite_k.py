"""
Job 3's finite-k response (v0.0.27): the cell's plane-wave modes (sseft.modes), the model's response to a mode
(model.mode_response), MB-pol's through a per-site field (mbpol.set_site_fields; scripts/e4_water.py response_k, here on a
fake MBX whose answers are known), the reading (read_response_k), and the read-out solved on liquid rows alone (the
locality test: linear.solve_readout with extra rows only, their Jacobian by forward mode).
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
from sseft import model as M, linear as LIN, modes as MD
from sseft.units import ang_to_bohr as A2B

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from test_periodic import frozen, switched_on, cluster, water_structure, one, in_cell  # noqa: E402


def _cell(n=8, L=12.5, seed=3):
    pos = cluster(n, seed=seed, box=L - 2.0) + L / 2
    return water_structure(pos, np.eye(3) * L)


def test_mode_field_is_a_periodic_longitudinal_wave_with_its_derivatives():
    """u, g = du_a/dx_b and h = d^2 u_a/dx_b dx_c against central differences; the tensors symmetric (curl-free);
    u along k; the same at every lattice translate; k = 2 pi n b_axis; n = 0 refused."""
    cell = np.array([[11.0, 0, 0], [0.5, 12.0, 0], [0.2, -0.3, 13.0]])
    X = np.random.default_rng(0).uniform(0, 10, (5, 3))
    e = 1e-5
    for axis, n, phase in [(0, 1, 0), (1, 2, 1), (2, 1, 1)]:
        u, g, h = MD.mode_field(X, cell, axis, n, phase)
        for d in range(3):
            dx = np.zeros(3); dx[d] = e
            up, gp, _ = MD.mode_field(X + dx, cell, axis, n, phase)
            um, gm, _ = MD.mode_field(X - dx, cell, axis, n, phase)
            assert np.allclose((up - um) / (2 * e), g[:, :, d], atol=1e-8)
            assert np.allclose((gp - gm) / (2 * e), h[:, :, :, d], atol=1e-8)
        assert np.allclose(g, np.swapaxes(g, 1, 2)) and np.allclose(h, np.swapaxes(h, 1, 2))
        assert np.allclose(h, np.swapaxes(h, 2, 3)) and np.allclose(h, np.swapaxes(h, 1, 3))
        k = MD.wavevector(cell, axis, n)
        assert np.allclose(np.cross(u, k), 0.0) and np.allclose(cell @ k, 2 * np.pi * n * np.eye(3)[axis])
        for a in cell:
            assert np.allclose(MD.mode_field(X + a, cell, axis, n, phase)[0], u, atol=1e-12)
    with pytest.raises(ValueError):
        MD.mode_field(X, cell, 0, 0, 0)
    assert len(MD.DEFAULT_MODES) == 12 and MD.describe(np.eye(3) * 2 * np.pi, [(0, 1, 0)])[0]["k"] == pytest.approx(1.0)


def test_mode_response_uniform_limit_point_dipoles_and_the_gradient_terms():
    """a uniform 'mode' (u = y^, g = h = 0) gives molecular_response's R_f y^ and induced_response's R_yy; at read-out
    order 0 a mode's gradients do nothing and chi is the field's work on the atoms' dipoles, sum_i u_i . dmu_i; with the
    orders >= 1 on, the gradients move both chi and the molecules' response."""
    import e4_water as E4
    rung = replace(frozen(), bf_solver="direct", bf_l_mutual=A2B(2.0))
    params = switched_on(rung)
    b = one(_cell())
    rc = in_cell(rung, np.asarray(b["cell"]))
    N = b["positions"].shape[0]
    zg, zh = np.zeros((N, 3, 3)), np.zeros((N, 3, 3, 3))
    u = np.zeros((N, 3)); u[:, 1] = 1.0
    dmu, chi = (np.asarray(x) for x in M.mode_response(params, rc, b, u, zg, zh))
    _, R_mol = (np.asarray(x) for x in M.molecular_response(params, rc, b))
    _, R = (np.asarray(x) for x in M.induced_response(params, rc, b))
    assert np.allclose(dmu, R_mol[:, :, 1], rtol=1e-9, atol=1e-10) and float(chi) == pytest.approx(R[1, 1], rel=1e-9)
    uu, gg, hh = MD.mode_field(np.asarray(b["positions"]), np.asarray(b["cell"]), 0, 1, 0)
    p0 = {**params, "readout": E4.order_zero(params["readout"], 16)}
    d_full, c_full = (np.asarray(x) for x in M.mode_response(p0, rc, b, uu, gg, hh, per_atom=True))
    d_nog, c_nog = (np.asarray(x) for x in M.mode_response(p0, rc, b, uu, zg, zh, per_atom=True))
    assert np.allclose(d_full, d_nog, rtol=1e-10, atol=1e-12) and float(c_full) == pytest.approx(float(c_nog), rel=1e-10)
    assert float(c_full) == pytest.approx(float(np.sum(uu * d_full)), rel=1e-9)
    d1, c1 = (np.asarray(x) for x in M.mode_response(params, rc, b, uu, gg, hh))
    d0, c0 = (np.asarray(x) for x in M.mode_response(params, rc, b, uu, zg, zh))
    assert abs(float(c1) - float(c0)) > 1e-6 * abs(float(c1)) and not np.allclose(d1, d0, rtol=1e-6)
    assert float(c1) > 0 and np.all(np.isfinite(d1))


def test_mode_response_follows_the_cell_contents():
    """the contents shifted by a quarter wavelength along the mode's axis answer the cos mode as the unshifted ones
    answer minus the sin mode: chi the same, the molecules' response of the other sign."""
    rung = replace(frozen(), bf_solver="direct", bf_l_mutual=A2B(2.0))
    params = switched_on(rung)
    st = _cell()
    b = one(st)
    rc = in_cell(rung, np.asarray(b["cell"]))
    cell = np.asarray(b["cell"])
    shifted = one(replace(st, positions=np.asarray(st.positions) + cell[0] / 4))
    pos, pos_s = np.asarray(b["positions"]), np.asarray(shifted["positions"])
    d_sin, c_sin = (np.asarray(x) for x in M.mode_response(params, rc, b, *MD.mode_field(pos, cell, 0, 1, 1)))
    d_cos, c_cos = (np.asarray(x) for x in M.mode_response(params, rc, shifted, *MD.mode_field(pos_s, cell, 0, 1, 0)))
    assert float(c_cos) == pytest.approx(float(c_sin), rel=1e-8)
    assert np.allclose(d_cos, -d_sin, rtol=1e-7, atol=1e-9)


def test_solve_on_extra_rows_alone_with_forward_jacobians():
    """with w_energy = w_force = 0 the extra rows are the whole objective: from a read-out at zero, the order-0 columns
    solved on every molecule's dipole and response move them most of the way to a switched-on read-out's; the forward
    Jacobian in chunks gives the reverse-mode solve; without extra rows zero weights are refused."""
    rung = replace(frozen(), bf_solver="direct", bf_l_mutual=A2B(2.0))
    target_p = switched_on(rung)
    target_p["readout"]["theta"][-1]["w"] = target_p["readout"]["theta"][-1]["w"].at[:, 16:].set(0.0)
    b = one(_cell())
    rc = in_cell(rung, np.asarray(b["cell"]))
    n = 8

    def fn(p, bb):
        mu, R = M.molecular_response(p, rc, bb)
        return jnp.concatenate([mu[:n].reshape(-1), R[:n].reshape(-1)])
    y = np.asarray(fn(target_p, b))
    w = np.full(len(y), 1.0 / np.mean(y ** 2))
    p0 = M.init_params(__import__("jax").random.PRNGKey(0), rung)
    data = {k: v[None] for k, v in one(water_structure(cluster(2, seed=5))).items()}
    data["energy"] = jnp.zeros(1)
    kw = dict(w_force=0.0, w_energy=0.0, ridge=1e-10, orders="zero", iterations=3)
    extra = {"fn": fn, "batches": [b], "targets": [y], "weights": [w]}
    p_f, rep_f = LIN.solve_readout(p0, rung, data, 1e-3, extra={**extra, "jac": "forward", "jac_chunk": 7}, **kw)
    p_r, rep_r = LIN.solve_readout(p0, rung, data, 1e-3, extra=extra, **kw)
    assert rep_f.extra_history[-1][1] < 1e-3 * rep_f.extra_history[0][0]
    assert np.allclose(np.asarray(fn(p_f, b)), np.asarray(fn(p_r, b)), rtol=1e-6, atol=1e-9)
    assert [h[1] for h in rep_f.history] == pytest.approx([h[1] for h in rep_r.history], rel=1e-4)   # round-off
    with pytest.raises(ValueError):
        LIN.solve_readout(p0, rung, data, 1e-3, **kw)


class FakeMBK:
    """a fake MBX with a per-site field: four sites a molecule (O, H, H and an M-site on the O), induced dipoles 1.5 A^3
    times the field at the O site, mu0 = 0.01 x the O position; a 2 x 1 x 1 replica (the cell's own sites first)."""
    def __init__(self):
        self.F = None

    def evaluated_sites(self, pos, cell=None):
        P = np.asarray(pos, float).reshape(-1, 3, 3)
        S = np.concatenate([P, P[:, :1]], axis=1).reshape(-1, 3)
        return np.concatenate([S, S + np.asarray(cell)[0]])

    def set_external_field(self, pos, field, cell=None):
        assert field is None
        self.F = None

    def set_site_fields(self, pos, fields, cell=None):
        self.F = None if fields is None else np.asarray(fields, float)

    def sites(self, pos, cell=None):
        n = len(pos) // 3
        mu = np.zeros((n, 4, 3))
        mu[:, 0] = 0.01 * np.asarray(pos).reshape(n, 3, 3)[:, 0]
        if self.F is not None:
            mu[:, 0] += 1.5 * self.F[:4 * n].reshape(n, 4, 3)[:, 0]
        return {"induced": mu.reshape(-1, 3)}


def test_response_k_parts_merge_round_trip_and_reading(tmp_path):
    """liquid_response_k on the fake MBX: every molecule's response 1.5 u(O) and chi = 1.5 sum_O |u|^2; the frames in
    two parts, merged, equal the whole file; the file round trips in atomic units against its frames and refuses moved
    frames; read_response_k reads a model on it (finite, its MB-pol side the fake's)."""
    import e4_water as E4
    from test_e4_train import _fake_cells
    import yaml
    for split, nf in (("train", 4), ("test", 2)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg = E4.merge(E4.load_config(None), {"data": {"dir": str(tmp_path), "train_size": 16, "response": str(tmp_path / "r.npz"),
                                                   "response_k": str(tmp_path / "rk.npz")},
                                          "liquid": {"calibration_frames": 3, "heldout_frames": 2},
                                          "liquid_k": {"axes": [0, 2], "harmonics": [1, 2], "phases": [0, 1]}})
    modes = E4.k_modes(cfg)
    assert modes == [(0, 1, 0), (0, 1, 1), (2, 1, 0), (2, 1, 1), (0, 2, 0), (0, 2, 1), (2, 2, 0), (2, 2, 1)]
    pos = np.load(tmp_path / "train_n16.npz")["positions"][0]
    cell = np.eye(3) * 12.5
    dmu, chi = E4.liquid_response_k(FakeMBK(), pos, cell, modes)
    O = pos.reshape(16, 3, 3)[:, 0]
    for m, md in enumerate(modes):
        u = MD.mode_field(O, cell, *md)[0]
        assert np.allclose(dmu[m], 1.5 * u, atol=1e-9) and chi[m] == pytest.approx(1.5 * np.sum(u * u), rel=1e-9)
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)

    class Truth:
        mb = FakeMBK()
        def describe(self):
            return {"name": "fake"}
    real = E4.W.MBPolTruth
    E4.W.MBPolTruth = lambda root: Truth()
    try:
        args = dict(config=str(tmp_path / "c.yaml"), mbx="none", merge=0)
        E4.run_response_k(SimpleNamespace(out=str(tmp_path / "whole.npz"), part=None, **args))
        for i in range(2):
            E4.run_response_k(SimpleNamespace(out=None, part=f"{i}/2", **args))
    finally:
        E4.W.MBPolTruth = real
    E4.run_response_k(SimpleNamespace(config=str(tmp_path / "c.yaml"), mbx=None, out=None, part=None, merge=2))
    whole, merged = np.load(tmp_path / "whole.npz"), np.load(tmp_path / "rk.npz")
    for key in ("calibration_index", "calibration_dmu", "calibration_chi", "heldout_index", "heldout_dmu", "heldout_chi"):
        assert np.array_equal(whole[key], merged[key]), key
    mw, mm = json.loads(str(whole["meta"])), json.loads(str(merged["meta"]))
    assert mw["splits"] == mm["splits"] and mw["modes"] == mm["modes"] == [list(x) for x in modes]
    rk = E4.load_response_k(str(tmp_path / "rk.npz"), cfg)
    assert rk["modes"] == modes and rk["calibration"]["dmu"].shape == (3, 8, 16, 3)
    assert np.allclose(rk["heldout"]["chi"], A2B(A2B(A2B(merged["heldout_chi"]))))
    rung_cell, _ = E4.rungs_for("M_16C2pm_us", E4.merge(cfg, {"model": {"features": 8}}),
                                rk["heldout"]["structures"][0].cell)
    v = E4.read_response_k(switched_on(rung_cell), rung_cell, rk, "heldout", eps0=(1.7, 1.6))
    assert set(v["harmonics"]) == {"1", "2"} and v["frames"] == 2
    V = 12.5 ** 3
    for key, h in v["harmonics"].items():
        n = int(key)
        assert np.isfinite([h["chi_model"], h["chi_rel_diff"], h["molecular_rel_rmse"]]).all()
        assert h["k_inv_angstrom"] == pytest.approx(2 * np.pi * n / 12.5)
        cols = [m for m, md in enumerate(modes) if md[1] == n]             # two axes, two phases
        assert h["chi_truth"] == pytest.approx(np.mean(merged["heldout_chi"][:, cols].sum(1)) / 2 / V, rel=1e-9)
        assert h["eps_L_truth"] == pytest.approx(1 / (1 - 4 * np.pi * h["chi_truth"]), rel=1e-12)
    assert v["chi_screened_k0"]["truth"] == pytest.approx((1 - 1 / 1.6) / (4 * np.pi))
    assert "eps_L" in E4.k_line(v)
    d = np.load(tmp_path / "train_n16.npz")
    moved = dict(d); moved["positions"] = d["positions"] + 0.01
    np.savez(tmp_path / "train_n16.npz", **moved)
    with pytest.raises(ValueError):
        E4.load_response_k(str(tmp_path / "rk.npz"), cfg)


def test_liquid_rows_by_molecule():
    """liquid_block with rows "molecules": 12 numbers a molecule, MB-pol's per-molecule targets, the same weight per
    cell as the cell rows' convention, and a forward Jacobian; an unknown row kind is refused; the cell rows' hash is
    unchanged by the new option (job 2's cached read-outs keep their names)."""
    import e4_water as E4
    st = _cell(n=8)
    n = 8
    rng = np.random.default_rng(1)
    resp = {"hash": "abcd1234", "calibration": {"structures": [st, st], "mu": rng.normal(size=(2, n, 3)) * 0.1,
                                                 "R": np.tile(np.eye(3) * 10.0, (2, n, 1, 1))}}
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8}})
    rung_cell, _ = E4.rungs_for("M_16C2pm_us", cfg, st.cell)
    lb = E4.liquid_block(rung_cell, resp, {"weight": 0.1, "rows": "molecules"})["block"]
    assert lb["jac"] == "forward" and lb["targets"][0].shape == (12 * n,) and lb["weights"][0].shape == (12 * n,)
    assert np.allclose(lb["targets"][1][3 * n:].reshape(n, 3, 3), resp["calibration"]["R"][1])
    assert np.asarray(lb["fn"](switched_on(rung_cell), lb["batches"][0])).shape == (12 * n,)
    with pytest.raises(ValueError):
        E4.liquid_block(rung_cell, resp, {"weight": 0.1, "rows": "atoms"})
    import hashlib
    h = E4.liquid_block(rung_cell, resp, {"weight": 0.1})["hash"]
    assert h == hashlib.sha256(json.dumps({"weight": 0.1, "response": "abcd1234"}, sort_keys=True).encode()).hexdigest()[:8]


MBX = os.environ.get("MBX_HOME", os.path.join(os.path.dirname(HERE), "external", "mbx", "build", "MBX"))


@pytest.mark.skipif(not os.path.isfile(os.path.join(MBX, "lib", "libmbx.so")), reason="no MBX build")
def test_mbx_site_fields():
    """MBX: a uniform field set site by site is set_external_field's; a cell's plane wave on its replica answers
    linearly; cleared afterwards."""
    from sseft import mbpol as MP
    import e4_water as E4
    truth = MP.MBPol(MBX)
    rng = np.random.default_rng(2)
    cell = np.eye(3) * 12.43
    pos = MP.random_box(16, cell, rng, min_oo=2.6, min_any=1.6)
    X = truth.evaluated_sites(pos, cell)
    z0 = truth.sites(pos, cell)["induced"].copy()
    E = np.array([0.0, 2e-4, 0.0])
    truth.set_external_field(pos, E, cell); a = truth.sites(pos, cell)["induced"].copy()
    truth.set_site_fields(pos, np.tile(E, (len(X), 1)), cell); b = truth.sites(pos, cell)["induced"].copy()
    assert np.allclose(a, b, atol=1e-12)
    d1, c1 = E4.liquid_response_k(truth, pos, cell, [(1, 1, 0)], 1e-4)
    d2, c2 = E4.liquid_response_k(truth, pos, cell, [(1, 1, 0)], 4e-4)
    assert np.allclose(d1, d2, rtol=1e-6, atol=1e-9) and c1[0] == pytest.approx(c2[0], rel=1e-6) and c1[0] > 0
    assert np.allclose(truth.sites(pos, cell)["induced"], z0, atol=1e-12)             # cleared afterwards
    with pytest.raises(ValueError):
        truth.set_site_fields(pos, np.zeros((3, 3)), cell)


def test_job3_config_reads_job2s_frames_and_model():
    """configs/e4_job3.yaml: job 2c's data (plus the finite-k file), liquid frames and model; the 12 modes; the three
    arms' configs and caches; C read with its orders >= 1 off; the locality test's settings; its rung builds."""
    import e4_water as E4
    root = os.path.dirname(HERE)
    load = lambda name: E4.load_config(os.path.join(root, "configs", name))
    c3, cc = load("e4_job3.yaml"), load("e4_job2c.yaml")
    assert c3["data"] == cc["data"] and c3["data"]["response_k"] == "data/e4/response_k_mbpol.npz"   # the default
    assert c3["liquid"] == cc["liquid"] and c3["model"] == cc["model"]
    assert len(E4.k_modes(c3)) == 12 and c3["liquid_k"]["delta"] == 1e-4
    kc = c3["kread"]
    assert kc["rung"] == "M_16C2pm_us" and kc["order_zero_of"] == ["C"] and kc["splits"] == ["heldout", "calibration"]
    assert {a: s["config"] for a, s in kc["arms"].items()} == {a: f"configs/e4_job2{a.lower()}.yaml" for a in "ABC"}
    assert {a: s["out_dir"] for a, s in kc["arms"].items()} == {a: f"results/e4_job2{a.lower()}" for a in "ABC"}
    assert kc["locality"]["ridge"] == 1e-6 and kc["locality"]["iterations"] == 4
    E4.rungs_for(kc["rung"], c3, np.eye(3) * A2B(12.43))
