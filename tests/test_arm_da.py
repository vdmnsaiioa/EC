"""
Job 5's arm D' (v0.0.29; the label Da): arm D's third stage anchored at arm C's read-out -- linear.solve_readout's anchor
(the ridge toward the starting coefficients) and hold (the energy and force rows held at the starting model's own
values), the anchored stage in scripts/e4_water.py (k_stage, match_readout), kread's change at contact (read_contact),
and job 5's config.
"""
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import sseft  # noqa: F401  (x64)
from sseft import model as M, linear as LIN
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from test_periodic import frozen, switched_on  # noqa: E402
from test_arm_d import N0, _cell, _probe_data, _fake_resp_k  # noqa: E402


def _higher(params):
    """the read-out's orders >= 1 output columns, flat."""
    th = params["readout"]["theta"][-1]
    return np.concatenate([np.asarray(th["w"])[:, N0:].ravel(), np.asarray(th["b"])[N0:]])


def _scaled(params, s):
    """params with the read-out's orders >= 1 output columns scaled by s."""
    th = params["readout"]["theta"][-1]
    theta = params["readout"]["theta"][:-1] + [{"w": th["w"].at[:, N0:].multiply(s), "b": th["b"].at[N0:].multiply(s)}]
    return {**params, "readout": {**params["readout"], "theta": theta}}


def test_anchor_pulls_the_ridge_toward_the_start():
    """anchor: under a ridge that dominates the rows the solve stays at the start, where without it the orders >= 1 go
    to zero; under a small ridge both fit the rows (three probes, many more coefficients)."""
    rung = replace(frozen(), bf_solver="direct")
    target = switched_on(rung)
    data = _probe_data(rung, target)
    p0 = _scaled(target, 0.3)
    c0 = _higher(p0)
    big = dict(w_force=0.0, ridge=1e6, orders="higher", iterations=1)
    pa, _ = LIN.solve_readout(p0, rung, data, 1e-3, anchor=True, **big)
    pz, _ = LIN.solve_readout(p0, rung, data, 1e-3, **big)
    assert np.linalg.norm(_higher(pa) - c0) < 1e-4 * np.linalg.norm(c0)
    assert np.linalg.norm(_higher(pz)) < 1e-4 * np.linalg.norm(c0)
    small = dict(w_force=0.0, ridge=1e-14, orders="higher", iterations=1)
    (pa, ra), (pz, rz) = (LIN.solve_readout(p0, rung, data, 1e-3, anchor=x, **small) for x in (True, False))
    assert ra.history[0][1] < 1e-6 * ra.history[0][0] and rz.history[0][1] < 1e-6 * rz.history[0][0]


def test_hold_keeps_the_rows_where_the_start_puts_them():
    """hold: the rows' targets are the start's own values, so with no other rows the anchored solve does not move;
    .data_history reads the objective on data's own targets (unchanged), and without hold it is .history."""
    rung = replace(frozen(), bf_solver="direct")
    target = switched_on(rung)
    data = _probe_data(rung, target)
    p0 = _scaled(target, 0.3)
    p1, rep = LIN.solve_readout(p0, rung, data, 1e-3, w_force=0.0, ridge=1e-10, orders="higher", iterations=2,
                                anchor=True, hold=True)
    c0 = _higher(p0)
    assert np.abs(_higher(p1) - c0).max() < 1e-9 * np.abs(c0).max()
    assert rep.history[0][0] == pytest.approx(0.0, abs=1e-20)
    _, plain = LIN.solve_readout(p0, rung, data, 1e-3, w_force=0.0, ridge=1e-10, orders="higher", iterations=1)
    assert plain.history[0][0] > 0 and rep.data_history[0][0] == pytest.approx(plain.history[0][0], rel=1e-12)
    assert rep.data_history[-1][1] == pytest.approx(rep.data_history[0][0], rel=1e-8)
    assert plain.data_history == plain.history


def test_the_anchored_stage_fits_the_finite_k_rows_and_holds_the_probes():
    """arm D' (anchor and hold) against arm D's stage from the same start, with probe targets that pull elsewhere: both
    bring the cells' chi_mode to a target read-out's; the anchored stage leaves the probes' energies where the start put
    them and moves its orders >= 1 less; its .data_history starts where arm D's .history does."""
    import e4_water as E4
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8, "mutual_edge_angstrom": 2.0}})
    st = _cell()
    rung_cell, rung_open = E4.rungs_for("M_16C2pm_us", cfg, st.cell)
    target = switched_on(rung_cell)
    modes = [(0, 1, 0), (1, 1, 1), (2, 2, 0), (0, 2, 1)]
    rk = _fake_resp_k(rung_cell, target, modes)
    p0 = _scaled(target, 0.5)                                 # the start: its chi_mode off the target's
    data = _probe_data(rung_open, switched_on(rung_open, key=1))   # the probes' targets: another read-out's
    fk = E4.k_stage(rung_cell, rk, {"weight": 1.0})
    kw = dict(w_force=0.0, ridge=1e-8, orders="higher", iterations=2, extra=fk["block"])
    pd, rd = LIN.solve_readout(p0, rung_open, data, 1e-3, **kw)
    pa, ra = LIN.solve_readout(p0, rung_open, data, 1e-3, anchor=True, hold=True, **kw)
    assert ra.extra_history[-1][1] < 1e-2 * ra.extra_history[0][0]
    E = lambda p: np.array([float(M.energy_single(p, rung_open, {k: v[i] for k, v in data.items()})[0])
                            for i in range(len(data["energy"]))])
    e0 = E(p0)
    assert np.abs(E(pa) - e0).max() < 0.2 * np.abs(E(pd) - e0).max()
    c0 = _higher(p0)
    assert np.linalg.norm(_higher(pa) - c0) < np.linalg.norm(_higher(pd) - c0)
    assert ra.data_history[0][0] == pytest.approx(rd.history[0][0], rel=1e-10)


def test_k_stage_anchor_flag_and_hash():
    """k_stage: settings without "anchor" hash as v0.0.28's did, so arm D's cache carries over; anchor true is another
    hash and is returned, with the rows or without them (weight 0)."""
    import e4_water as E4
    cfg = E4.merge(E4.load_config(None), {"model": {"features": 8, "mutual_edge_angstrom": 2.0}})
    st = _cell()
    rung_cell, _ = E4.rungs_for("M_16C2pm_us", cfg, st.cell)
    rk = _fake_resp_k(rung_cell, switched_on(rung_cell), [(0, 1, 0)], n_cells=1)
    old = hashlib.sha256(json.dumps({"weight": 0.1, "response_k": rk["hash"]}, sort_keys=True).encode()).hexdigest()[:8]
    d = E4.k_stage(rung_cell, rk, {"weight": 0.1})
    a = E4.k_stage(rung_cell, rk, {"weight": 0.1, "anchor": True})
    assert d["hash"] == old and d["anchor"] is False and a["anchor"] is True and a["hash"] != old
    z = E4.k_stage(rung_cell, rk, {"weight": 0.0, "anchor": True})
    assert z["block"] is None and z["anchor"] is True


def test_job5_config_is_arm_ds_anchored(monkeypatch):
    """configs/e4_job5.yaml: job 4's config with training.readout.finite_k.anchor true and nothing else; kread's arms --
    C (job 2c's config), D and Dx (job 4's) and Da (this one) -- all in results/e4_job5 at 2.0 A, with the change at
    contact against C, so that arm C's and arm D's cached stages carry over."""
    import e4_water as E4
    monkeypatch.chdir(os.path.dirname(HERE))                             # kread's arms name their configs from there
    c5, c4, cc = (E4.load_config(os.path.join("configs", n)) for n in ("e4_job5.yaml", "e4_job4.yaml", "e4_job2c.yaml"))
    for key in ("data", "liquid", "liquid_k", "model", "rungs", "seeds", "probes", "reading"):
        assert c5[key] == c4[key], key
    t5, t4 = c5["training"], c4["training"]
    assert {k: v for k, v in t5.items() if k != "readout"} == {k: v for k, v in t4.items() if k != "readout"}
    no_k = lambda ro: {k: v for k, v in ro.items() if k != "finite_k"}
    assert no_k(t5["readout"]) == no_k(t4["readout"])
    assert t5["readout"]["finite_k"] == {**t4["readout"]["finite_k"], "anchor": True}
    arms, oz = E4.kread_arms(c5["kread"])
    by = {a: (cfg, out) for a, cfg, out in arms}
    assert set(by) == {"C", "D", "Dx", "Da"} and oz == {}
    assert all(out == "results/e4_job5" for _, out in by.values())
    assert by["C"][0]["training"]["readout"] == cc["training"]["readout"]
    assert by["D"][0]["training"] == t4 and by["Da"][0]["training"] == t5
    assert by["Dx"][0]["training"]["readout"]["finite_k"] == {"weight": 0.0}
    assert all(cfg["model"]["mutual_edge_angstrom"] == 2.0 for cfg, _ in by.values())
    assert c5["kread"]["contact"] == {"reference": "C", "frames": "test"} and c5["kread"]["rung"] == "M_16C2pm_us"


def test_kread_reads_the_anchored_arm_and_the_change_at_contact(tmp_path):
    """scripts/e4_water.py kread on fake cells, probes and responses (MB-pol's stand-ins), the frozen form at width 8:
    arm C (the liquid stage), arm D (then the third stage) and Da (the third stage anchored), all in one directory, with
    the change at contact against C.  The third stages move the orders >= 1 alone; eps_inf is C's in both; C's change
    against itself is zero; both stages' records read the probes against their own targets, so stage 3 starts where
    stage 2 ended; arm D re-fits the training probes and Da holds them; every reading is in the json.  (Whether Da's
    orders >= 1 move less than D's depends on the data: on these fakes both chase an unreachable finite-k target.)"""
    import e4_water as E4
    import yaml
    from test_e4_train import _fake_cells, _small_cfg, FakeMBPol
    from test_liquid_calibration import FakeMB
    from test_finite_k import FakeMBK
    from e4_report import stage_objectives
    cfg = _small_cfg(tmp_path)
    cfg["probes"]["far_trimers"] = 2
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 3), ("valid", 1), ("test", 2)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["data"].update(response=str(tmp_path / "r.npz"), response_k=str(tmp_path / "rk.npz"))
    cfg["liquid"].update(calibration_frames=2, heldout_frames=1)
    cfg["liquid_k"].update(axes=[0], harmonics=[1, 2], phases=[0, 1])
    cfg["model"].update(features=8, mutual_edge_angstrom=2.0)
    cfg["training"]["readout"].update(liquid={"weight": 0.1, "orders": "zero"}, solve_iterations=2)
    paths = {k: str(tmp_path / f"{k}.yaml") for k in ("C", "D", "k")}
    arm_d = E4.merge(cfg, {"training": {"readout": {"finite_k": {"weight": 0.1}}}})
    edge = str(tmp_path / "edge2.0")
    kread = {"rung": "M_16C2pm_us", "splits": ["heldout"], "contact": {"reference": "C", "frames": "test"},
             "arms": {"C": {"config": paths["C"], "out_dir": edge}, "D": {"config": paths["D"], "out_dir": edge},
                      "Da": {"config": paths["D"], "out_dir": edge,
                             "set": {"training": {"readout": {"finite_k": {"anchor": True}}}}}}}
    for k, c in (("C", cfg), ("D", arm_d), ("k", E4.merge(arm_d, {"kread": kread}))):
        with open(paths[k], "w") as fh:
            yaml.safe_dump(c, fh)

    class Truth:
        def __init__(self, mb):
            self.mb = mb
        def describe(self):
            return {"name": "fake"}
    real = E4.W.MBPolTruth
    try:
        E4.W.MBPolTruth = lambda root: Truth(FakeMB())
        E4.run_response(SimpleNamespace(config=paths["k"], mbx="none", out=None))
        E4.W.MBPolTruth = lambda root: Truth(FakeMBK())
        E4.run_response_k(SimpleNamespace(config=paths["k"], mbx="none", out=None, part=None, merge=0))
    finally:
        E4.W.MBPolTruth = real
    out = tmp_path / "out" / "kread.json"
    E4.run_kread(SimpleNamespace(config=paths["k"], out_dir=str(tmp_path / "out"), only=None, out=str(out)))
    k = json.load(open(out))
    assert {"C", "D", "Da"} <= set(k) and k["meta"]["contact"]["reference"] == "C"
    assert k["meta"]["contact"]["n_frames"] == 2
    c = {lab: k[lab]["contact"] for lab in ("C", "D", "Da")}
    assert c["C"]["against_reference"] == {"energy_per_molecule": 0.0, "rigid_rms": 0.0}
    assert c["C"]["coefficients"] == {"orders_above_zero": 0.0, "order_zero": 0.0}
    for lab in ("D", "Da"):
        assert c[lab]["coefficients"]["order_zero"] == 0.0 and c[lab]["coefficients"]["orders_above_zero"] > 0
        assert c[lab]["against_reference"]["rigid_rms"] > 0 and c[lab]["own"]["rigid_rms"] > 0
        eps = lambda x: k[x]["liquid_heldout"]["eps_inf_model"]
        assert eps(lab) == pytest.approx(eps("C"), rel=1e-9)
        assert np.isfinite([h["chi_rel_diff"] for h in k[lab]["k_heldout"]["harmonics"].values()]).all()
        m = k[lab]["match"]
        o = stage_objectives(m)
        n3 = int(m["finite_k_stage_steps"])
        start3 = m["objective"][-n3][0] - m["finite_k_objective"][0][0]
        assert start3 == pytest.approx(o[2], rel=1e-8)
    o_d, o_a = stage_objectives(k["D"]["match"]), stage_objectives(k["Da"]["match"])
    assert o_d[3] < 1e-3 * o_d[2]                                   # arm D re-fits the training probes (four here)
    assert o_a[3] == pytest.approx(o_a[2], rel=1e-6)                # Da holds them where arm C put them
    assert k["Da"]["match"]["finite_k_anchor"] is True and "finite_k_anchor" not in k["D"]["match"]
    assert k["Da"]["match"]["finite_k_held_objective"][0][0] == pytest.approx(k["Da"]["match"]["finite_k_objective"][0][0],
                                                                            rel=1e-9)
    files = sorted(f for f in os.listdir(edge) if f.startswith("matched_"))
    assert len(files) == 4 and len({f.split("_k")[-1] for f in files if "_k" in f}) == 2
