"""
Job 4's arm D (v0.0.28): the read-out's orders >= 1 solved alone (linear.solve_readout orders="higher"), the finite-k
rows (scripts/e4_water.py k_stage: every calibration cell's susceptibility in every mode, model.mode_chi), the third
matching stage and its cache chain (matched), kread's arms with overrides and order-zero labels, and job 4's config.
"""
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, linear as LIN, modes as MD
from sseft.units import ang_to_bohr as A2B

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from test_periodic import frozen, switched_on, cluster, water_structure, one, in_cell  # noqa: E402

N0 = 16                                         # the frozen form's order-0 columns: 2 invariants x 8 radial functions


def _cell(n=8, L=12.5, seed=3):
    pos = cluster(n, seed=seed, box=L - 2.0) + L / 2
    return water_structure(pos, np.eye(3) * L)


def _probe_data(rung, params, seeds=(5, 6, 7)):
    """three small clusters with the switched-on read-out's energies as targets (the probes' role)."""
    structs = [water_structure(cluster(3, seed=s)) for s in seeds]
    data = {k: jnp.concatenate([one(st)[k][None] for st in structs]) for k in one(structs[0])}
    E = np.array([float(M.energy_single(params, rung, {k: v[i] for k, v in data.items()})[0]) for i in range(len(seeds))])
    return {**data, "energy": jnp.asarray(E), "has_energy": jnp.ones(len(seeds), bool)}


def test_mode_chi_is_mode_responses_chi_and_linear_in_the_orders_above_zero():
    """model.mode_chi equals mode_response's chi; it is linear in the orders >= 1 columns (they sit outside the mutual
    solve); with them off it is the order-0 sector's."""
    import e4_water as E4
    rung = replace(frozen(), bf_solver="direct", bf_l_mutual=A2B(2.0))
    params = switched_on(rung)
    b = one(_cell())
    rc = in_cell(rung, np.asarray(b["cell"]))
    u, g, h = MD.mode_field(np.asarray(b["positions"]), np.asarray(b["cell"]), 1, 2, 1)
    c = float(M.mode_chi(params, rc, b, u, g, h))
    assert c == pytest.approx(float(M.mode_response(params, rc, b, u, g, h)[1]), rel=1e-12)
    th = params["readout"]["theta"][-1]
    d = np.zeros(th["w"].shape); d[:, N0:] = np.random.default_rng(0).normal(size=(d.shape[0], d.shape[1] - N0))
    at = lambda s: {**params, "readout": {**params["readout"], "theta": params["readout"]["theta"][:-1]
                                          + [{"w": th["w"] + s * 1e-2 * d, "b": th["b"]}]}}
    c1, c2 = (float(M.mode_chi(at(s), rc, b, u, g, h)) for s in (1.0, 2.0))
    assert abs(c1 - c) > 1e-6 * abs(c) and (c2 - c) == pytest.approx(2 * (c1 - c), rel=1e-8)
    p0 = {**params, "readout": E4.order_zero(params["readout"], N0)}
    dmu, c0 = M.mode_response(p0, rc, b, u, g, h, per_atom=True)
    assert float(M.mode_chi(p0, rc, b, u, g, h)) == pytest.approx(float(np.sum(u * np.asarray(dmu))), rel=1e-9)


def test_solve_on_the_orders_above_zero_alone():
    """orders="higher" moves the orders >= 1 columns alone -- the order-0 columns and the node term exactly as they were
    -- and lowers the objective; a read-out without orders >= 1 is refused."""
    rung = replace(frozen(), bf_solver="direct")
    target = switched_on(rung)
    data = _probe_data(rung, target)
    th = target["readout"]["theta"][-1]
    p0 = {**target, "readout": {**target["readout"], "theta": target["readout"]["theta"][:-1]
                                + [{"w": th["w"].at[:, N0:].multiply(0.3), "b": th["b"].at[N0:].set(0.0)}]}}
    p1, rep = LIN.solve_readout(p0, rung, data, 1e-3, w_force=0.0, ridge=1e-10, orders="higher", iterations=2)
    w0, w1 = np.asarray(p0["readout"]["theta"][-1]["w"]), np.asarray(p1["readout"]["theta"][-1]["w"])
    b0, b1 = np.asarray(p0["readout"]["theta"][-1]["b"]), np.asarray(p1["readout"]["theta"][-1]["b"])
    assert np.array_equal(w0[:, :N0], w1[:, :N0]) and np.array_equal(b0[:N0], b1[:N0])
    for k in ("w", "b"):
        assert np.array_equal(np.asarray(p0["readout"]["iso"][-1][k]), np.asarray(p1["readout"]["iso"][-1][k]))
    assert not np.allclose(w0[:, N0:], w1[:, N0:]) and rep.history[0][1] < 1e-3 * rep.history[0][0]
    rung0 = M.with_rung(rung, band_fields=0)
    p00 = M.init_params(jax.random.PRNGKey(0), rung0)
    with pytest.raises(ValueError):
        LIN.solve_readout(p00, rung0, data, 1e-3, w_force=0.0, orders="higher")


def _fake_resp_k(rung_cell, params, modes, n_cells=2):
    """a finite-k response file's calibration part whose chi_mode are the given read-out's own (in a cell)."""
    sts = [_cell(seed=3 + i) for i in range(n_cells)]
    r = in_cell(replace(rung_cell, bf_solver="direct"), np.asarray(sts[0].cell))
    chi = []
    for st in sts:
        b = one(st)
        pos, cell = np.asarray(b["positions"]), np.asarray(st.cell)
        chi.append([float(M.mode_chi(params, r, b, *MD.mode_field(pos, cell, *m))) for m in modes])
    return {"hash": "feedbeef", "modes": modes, "calibration": {"structures": sts, "chi": np.array(chi)}}


def test_k_stage_rows_weights_and_hash():
    """k_stage: 12 rows a cell (here 3 modes), the model's chi_mode in each, against the targets; weights
    weight / (cells x modes) / (MB-pol's mean chi_mode at the harmonic)^2; the hash follows the settings and the file;
    weight 0 builds no rows."""
    import e4_water as E4
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8, "mutual_edge_angstrom": 2.0}})
    st = _cell()
    rung_cell, _ = E4.rungs_for("M_16C2pm_us", cfg, st.cell)
    params = switched_on(rung_cell)
    modes = [(0, 1, 0), (0, 1, 1), (2, 2, 0)]
    rk = _fake_resp_k(rung_cell, params, modes)
    fk = E4.k_stage(rung_cell, rk, {"weight": 0.1})
    blk = fk["block"]
    chi = rk["calibration"]["chi"]
    assert set(fk["scale"]) == {1, 2} and fk["scale"][1] == pytest.approx(np.mean(chi[:, :2]))
    assert fk["scale"][2] == pytest.approx(np.mean(chi[:, 2]))
    assert np.allclose(blk["weights"][0], 0.1 / 6 / np.array([fk["scale"][1]] * 2 + [fk["scale"][2]]) ** 2)
    y = np.asarray(jax.jit(blk["fn"])(params, blk["batches"][1]))
    assert np.allclose(y, chi[1], rtol=1e-10) and len(blk["targets"]) == 2
    h2 = E4.k_stage(rung_cell, {**rk, "hash": "00000000"}, {"weight": 0.1})["hash"]
    assert h2 != fk["hash"] and E4.k_stage(rung_cell, rk, {"weight": 0.2})["hash"] != fk["hash"]
    none = E4.k_stage(rung_cell, rk, {"weight": 0.0})
    assert none["block"] is None and none["hash"] != fk["hash"]


def test_third_stage_fits_the_finite_k_rows_with_order_zero_held():
    """the third stage (orders="higher", the finite-k rows as extra): from a read-out whose orders >= 1 are off, it
    brings the cells' chi_mode to a read-out's whose orders >= 1 are on, the order-0 columns and the node term held."""
    import e4_water as E4
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8, "mutual_edge_angstrom": 2.0}})
    st = _cell()
    rung_cell, rung_open = E4.rungs_for("M_16C2pm_us", cfg, st.cell)
    target = switched_on(rung_cell)
    modes = [(0, 1, 0), (1, 1, 1), (2, 2, 0), (0, 2, 1)]
    rk = _fake_resp_k(rung_cell, target, modes)
    p0 = E4.with_readout(target, E4.order_zero(target["readout"], N0))
    fk = E4.k_stage(rung_cell, rk, {"weight": 1.0})
    data = _probe_data(rung_open, p0)                                     # the probes as p0 reads them: no pull
    p1, rep = LIN.solve_readout(p0, rung_open, data, 1e-3, w_force=0.0, ridge=1e-12, orders="higher", iterations=2,
                                extra=fk["block"])
    assert rep.extra_history[0][1] < 1e-2 * rep.extra_history[0][0]
    th0, th1 = p0["readout"]["theta"][-1], p1["readout"]["theta"][-1]
    assert np.array_equal(np.asarray(th0["w"])[:, :N0], np.asarray(th1["w"])[:, :N0])
    assert np.array_equal(np.asarray(p0["readout"]["iso"][-1]["w"]), np.asarray(p1["readout"]["iso"][-1]["w"]))


def test_matched_chains_the_third_stage_on_the_calibrated_read_out(tmp_path, monkeypatch):
    """matched with finite_k: the stage-2 read-out (liquid) from its cache or solved first, then the third stage from
    it, cached under its own name (..._liq<h>_k<h>.pkl) and found again; the liquid record carried over."""
    import e4_water as E4
    calls = []

    def fake_match(rung_open, probes, scale, cfg, verbose=True, liquid=None, start=None, finite_k=None):
        stage = 3 if finite_k is not None else (2 if liquid is not None else 1)
        calls.append((stage, None if start is None else int(start[0]["stage"])))
        rec = {"objective": [(1.0, 0.5)] * stage, "ridge": [0.1], "time": 1.0}
        if liquid is not None:
            rec.update(liquid_objective=[(1.0, 0.1)], liquid_cells=1, liquid_hash=liquid["hash"])
        elif start is not None:
            rec.update({k: v for k, v in start[1].items() if k.startswith("liquid_")})
        return {"readout": {"stage": np.array(stage)}}, rec
    monkeypatch.setattr(E4, "match_readout", fake_match)
    monkeypatch.setattr(E4, "read_probes", lambda params, rung, probes: {"fake": True})
    monkeypatch.setattr(E4.E3, "probe_summary", lambda m: "fake")
    rung = frozen()
    probes = {"hash": "p0"}
    liquid, fk = {"hash": "L1"}, {"hash": "K2", "block": None}
    ro, rec = E4.matched("R", rung, probes, 1.0, {}, str(tmp_path), liquid=liquid, finite_k=fk)
    assert calls == [(1, None), (2, 1), (3, 2)] and int(ro["stage"]) == 3 and rec["liquid_hash"] == "L1"
    assert sorted(os.listdir(tmp_path)) == ["matched_R_p0.pkl", "matched_R_p0_liqL1.pkl", "matched_R_p0_liqL1_kK2.pkl"]
    calls.clear()
    assert int(E4.matched("R", rung, probes, 1.0, {}, str(tmp_path), liquid=liquid, finite_k=fk)[0]["stage"]) == 3
    assert calls == []                                                    # every stage from its cache
    with open(tmp_path / "matched_R_p0_liqL1_kK2.pkl", "rb") as fh:
        assert pickle.load(fh)["match"]["probes_before_training"] == {"fake": True}


def test_job4_config_is_arm_c_and_the_third_stage(monkeypatch):
    """configs/e4_job4.yaml: job 2c's data, liquid frames, model, rungs, seeds and training with the finite-k stage added
    (weight 0.1), so arm C's cached stages carry over (the same liquid hash); job 3's modes; kread's arms -- C from job
    2c's cache, D and its control Dx at 2.0 A, C and D at 1.5 and 2.5 A -- with their overrides and order-zero labels."""
    import e4_water as E4
    root = os.path.dirname(HERE)
    monkeypatch.chdir(root)                                               # kread's arms name their configs from there
    c4, cc, c3 = (E4.load_config(os.path.join("configs", n)) for n in ("e4_job4.yaml", "e4_job2c.yaml", "e4_job3.yaml"))
    assert c4["data"] == c3["data"] and c4["liquid"] == cc["liquid"] and c4["model"] == cc["model"]
    assert c4["liquid_k"] == c3["liquid_k"] and c4["rungs"] == cc["rungs"] and c4["seeds"] == cc["seeds"]
    t4, tc = c4["training"], cc["training"]
    assert {k: v for k, v in t4.items() if k != "readout"} == {k: v for k, v in tc.items() if k != "readout"}
    no_k = lambda ro: {k: v for k, v in ro.items() if k != "finite_k"}
    assert no_k(t4["readout"]) == no_k(tc["readout"])
    assert t4["readout"]["finite_k"] == {"weight": 0.1} and tc["readout"]["finite_k"] is None
    arms, oz = E4.kread_arms(c4["kread"])
    by = {a: (cfg, out) for a, cfg, out in arms}
    assert set(by) == {"C", "D", "Dx", "C@1.5", "D@1.5", "C@2.5", "D@2.5"}
    assert oz == {"C": "C0", "C@1.5": "C0@1.5", "C@2.5": "C0@2.5"}
    assert by["C"][1] == "results/e4_job2c" and by["D"][1] == by["Dx"][1] == "results/e4_job4/edge2.0"
    assert by["D"][0]["training"] == t4 and by["Dx"][0]["training"]["readout"]["finite_k"] == {"weight": 0.0}
    for e in (1.5, 2.5):
        for arm in ("C", "D"):
            cfg, out = by[f"{arm}@{e}"]
            assert cfg["model"]["mutual_edge_angstrom"] == e and out == f"results/e4_job4/edge{e}"
            assert {k: v for k, v in cfg["model"].items() if k != "mutual_edge_angstrom"} == \
                {k: v for k, v in cc["model"].items() if k != "mutual_edge_angstrom"}
        assert by[f"C@{e}"][0]["training"]["readout"] == tc["readout"]
        assert by[f"D@{e}"][0]["training"]["readout"] == t4["readout"]
    assert c4["kread"]["rung"] == "M_16C2pm_us" and not c4["kread"].get("locality")
    with pytest.raises(ValueError):
        E4.kread_arms({"arms": {}, "order_zero_of": ["C"]})
