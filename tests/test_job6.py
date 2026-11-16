"""
E4 job 6 (v0.0.30; notes/e4-job6-preregistration.md): the learned sources started near zero -- the charges in a small
unit (Rung.q_unit), the polarisabilities at a small softplus start (Rung.alpha_ref0) -- the end-of-training loss and the
gate against M_inf, the learned sources' readings, and the S0 model step and read for learned charges (L1.2, L1.3).
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
from sseft import model as M, train as T, water as W, structure as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import e4_water as E4                                     # noqa: E402
import e4_sumrules as SR                                  # noqa: E402
from test_e4_train import _fake_cells, _small_cfg, FakeMBPol   # noqa: E402


def _cell(tmp_path, n_frames=2, n_mol=16, seed=3, name="t.npz"):
    _fake_cells(tmp_path / name, n_frames, n_mol, 12.5, seed=seed)
    return W.e4_structures(str(tmp_path / name), pins=True)


def _cfg(features=8):
    return E4.merge(E4.load_config(None), {"model": {"features": features}})


def _one(st, rung):
    return {k: v[0] for k, v in S.pad_batch([st], n_freq=rung.K).items()}


def test_learned_sources_start_near_zero(tmp_path):
    """with the same parameters, M_1u's charges are 0.02 x M_1's exactly (the neutrality shift is linear), so its
    Coulomb band, quadratic in them, is 4e-4 of M_1's; M_16u's polarisabilities start at softplus(-6 + x) instead of
    softplus(1 + x); every other parameter is as in M_inf, so both start close to E_0 alone."""
    st = _cell(tmp_path)[0]
    cfg = _cfg()
    out = {}
    for name in ("M_inf", "M_1", "M_1u", "M_16", "M_16u"):
        rung, _ = E4.rungs_for(name, cfg, st.cell)
        p = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=1e-4)
        E, aux = M.energy_single(p, rung, _one(st, rung))
        out[name] = (float(E), aux)
    q1, q1u = (np.asarray(out[k][1]["q"])[:st.n_atoms] for k in ("M_1", "M_1u"))
    assert np.abs(q1).max() > 1e-2 and np.allclose(q1u, E4.Q_UNIT * q1, rtol=1e-12, atol=1e-15)
    for i in range(16):                                                   # every molecule neutral
        assert abs(q1u[3 * i:3 * i + 3].sum()) < 1e-14
    d1, d1u = out["M_1"][0] - out["M_inf"][0], out["M_1u"][0] - out["M_inf"][0]
    assert d1u == pytest.approx(E4.Q_UNIT ** 2 * d1, rel=1e-9)
    a16, a16u = (np.asarray(out[k][1]["alpha"])[:st.n_atoms] for k in ("M_16", "M_16u"))
    assert a16.min() > 0.5 and 0 < a16u.max() < 1e-2                  # softplus(-6 + x), x the MLP's start
    assert abs(out["M_16u"][0] - out["M_inf"][0]) < 1e-4 * abs(out["M_16"][0] - out["M_inf"][0])


def test_zero_charges_are_a_stationary_point_and_small_ones_are_not(tmp_path):
    """the Coulomb band is bilinear in the charges, so at zero charges the energy's gradient in the charge head
    vanishes -- the reason job 6 starts them small, not at zero -- while at M_1u's start it does not."""
    st = _cell(tmp_path)[0]
    cfg = _cfg()
    norms = {}
    for unit in (0.0, E4.Q_UNIT):
        rung = M.with_rung(E4.rungs_for("M_1", cfg, st.cell)[0], q_unit=unit)
        p = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=1e-4)
        b = _one(st, rung)
        g = jax.grad(lambda hc: M.energy_single({**p, "heads": {**p["heads"], "charge": hc}}, rung, b)[0])(
            p["heads"]["charge"])
        norms[unit] = float(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    assert norms[0.0] == 0.0 and norms[E4.Q_UNIT] > 0.0


def test_final_loss_is_the_polish_loss_and_the_gate(tmp_path):
    """final_loss is the L-BFGS polish's full-batch objective, exactly; gate_status (scripts/e4_water.py) and
    job6_gate (scripts/e4_sumrules.py, without JAX) read the same verdicts from the jsons."""
    structs = _cell(tmp_path, n_frames=4)
    cfg = _cfg()
    rung, _ = E4.rungs_for("M_1u", cfg, structs[0].cell)
    p = M.init_params(jax.random.PRNGKey(2), rung, energy_scale=T.energy_scale(structs))
    scale = T.energy_scale(structs)
    fl = E4.final_loss(p, rung, structs, scale, 100.0, 1.0, 2, "rigid")
    ref = T.make_chunked_loss(rung, scale, S.pad_batch(structs, n_freq=rung.K), 100.0, 1.0, chunk=2, force_mode="rigid")
    assert fl["total"] == pytest.approx(float(ref(p)), rel=1e-10)
    assert fl["total"] == pytest.approx(100.0 * fl["energy"] + fl["forces"], rel=1e-12)
    out = tmp_path / "runs"; out.mkdir()
    losses = {("M_inf", 0): 5.0, ("M_inf", 1): 5.0, ("M_inf", 2): 5.0, ("M_1u", 0): 4.0, ("M_1u", 1): 6.0,
              ("M_1u", 2): 5.0, ("M_1u", 3): 1.0}
    for (name, s), v in losses.items():
        json.dump({"rung": name, "seed": s, "training": {"final_loss": {"total": v}}}, open(out / f"{name}_seed{s}.json", "w"))
    g = E4.gate_status(str(out), "M_1u")
    assert g == {0: (True, 4.0, 5.0), 1: (False, 6.0, 5.0), 2: (True, 5.0, 5.0)}          # seed 3: no reference
    assert SR.job6_gate(str(out)) == {s: v[0] for s, v in g.items()}


def test_learned_sources_reading(tmp_path):
    """charges set by the species offsets alone (the charge MLP's output layer zero): q_O = -2a, q_H = +a on every
    molecule, so the dipole is a |r_H1 + r_H2 - 2 r_O| about any origin, all to the H side for a > 0 and none for a < 0;
    with polarisabilities flat at alpha on every node, C6_ij = (3/pi) sum_k w_k alpha^2 for every pair."""
    structs = _cell(tmp_path, n_frames=2)
    cfg = _cfg()
    rung, _ = E4.rungs_for("M_16u", cfg, structs[0].cell)
    p = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=1e-4)
    last = p["heads"]["charge"][-1]
    zero = jax.tree_util.tree_map(jnp.zeros_like, last)
    from sseft.units import bohr_to_ang
    P = bohr_to_ang(np.asarray(structs[0].positions))[:3]                  # the frames' rigid monomer, whole
    span = np.linalg.norm(P[1] + P[2] - 2 * P[0])                          # A
    for a in (0.4, -0.4):
        q_ref = jnp.zeros_like(p["heads"]["q_ref"]).at[8].set(-2 * a / E4.Q_UNIT).at[1].set(a / E4.Q_UNIT)
        alpha_last = jax.tree_util.tree_map(jnp.zeros_like, p["heads"]["alpha"][-1])
        heads = {**p["heads"], "charge": p["heads"]["charge"][:-1] + [zero], "q_ref": q_ref,
                 "alpha": p["heads"]["alpha"][:-1] + [alpha_last],
                 "alpha_ref": jnp.zeros_like(p["heads"]["alpha_ref"]) + 2.0}
        v = E4.learned_sources({**p, "heads": heads}, rung, structs)
        assert v["q_O"][0] == pytest.approx(-2 * a, rel=1e-9) and v["q_H"][0] == pytest.approx(a, rel=1e-9)
        assert v["q_O"][1] < 1e-9 and v["dipole_debye"][1] < 1e-6
        assert v["dipole_debye"][0] == pytest.approx(abs(a) * span * 4.803204, rel=1e-6)
        assert v["dipole_to_H_side"] == (1.0 if a > 0 else 0.0)
    alpha = float(np.log1p(np.exp(2.0)))
    w = np.asarray(M.frequencies(rung)[1])
    assert v["C6_au"]["OO"] == pytest.approx(3 / np.pi * w.sum() * alpha ** 2, rel=1e-9)
    assert v["C6_au"]["molecule"] == pytest.approx(9 * v["C6_au"]["OO"], rel=1e-9)
    assert v["C6_over_mbpol"]["OH"] == pytest.approx(v["C6_au"]["OH"] / E4.MBPOL_C6_AU["OH"])


def test_driver_trains_the_learned_charges_and_prints_the_gate(tmp_path, capsys):
    """scripts/e4_water.py train for M_inf and M_1u on fake cells (one Adam step): each json carries the final training
    loss, M_1u's readings its learned sources, and the summary prints the gate seed by seed."""
    cfg = _small_cfg(tmp_path)
    truth = FakeMBPol()
    E4.save_probes(cfg["data"]["probes"], E4.make_probes(truth, cfg), truth, cfg)
    for split, nf in (("train", 2), ("test", 1)):
        _fake_cells(tmp_path / f"{split}_n16.npz", nf, 16, 12.5, seed=len(split))
    cfg["rungs"], cfg["seeds"] = ["M_inf", "M_1u"], [0]
    cfg["model"]["features"] = 8
    cfg["training"]["adam"].update(steps=1, batch_size=2)
    cfg["training"]["lbfgs"].update(steps=0, chunk_size=2)
    import yaml
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmp_path / "out"
    E4.run_train(SimpleNamespace(config=str(tmp_path / "c.yaml"), rungs=None, seeds=None, task=None, out_dir=str(out)))
    r = json.load(open(out / "M_1u_seed0.json"))
    assert r["model"]["charge_unit_e"] == E4.Q_UNIT and r["training"]["final_loss"]["total"] > 0
    v = r["readings"]["cells"]["16"]["test"]["learned_sources"]
    assert 0 < v["dipole_debye"][0] < 1.0 and 0.0 <= v["dipole_to_H_side"] <= 1.0
    assert "learned_sources" not in json.load(open(out / "M_inf_seed0.json"))["readings"]["cells"]["16"]["test"]
    capsys.readouterr()
    E4.run_summary(SimpleNamespace(out_dir=str(out)))
    text = capsys.readouterr().out
    assert "the gate (final training loss at or below M_inf's" in text and "seed 0" in text
    assert "learned sources: q_O" in text


def test_model_step_writes_the_learned_charges_per_frame(tmp_path):
    """scripts/e4_sumrules.py model --params for a field-blind rung: the learned charges of every frame at the atoms
    (nothing at the M-sites), zero point dipoles, the model's own charges exactly, and the file marked field-blind; a
    rung with a response is refused."""
    import yaml
    cfg = _cfg()
    data = tmp_path / "data"; data.mkdir()
    _fake_cells(data / "test_n16.npz", 2, 16, 12.5, seed=5)
    with open(tmp_path / "c.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    structs = W.e4_structures(str(data / "test_n16.npz"), pins=True)
    rung, _ = E4.rungs_for("M_1u", cfg, structs[0].cell)
    p = jax.tree_util.tree_map(np.asarray, M.init_params(jax.random.PRNGKey(4), rung, energy_scale=1e-4))
    with open(tmp_path / "p.pkl", "wb") as fh:
        pickle.dump(p, fh)
    args = dict(config=str(tmp_path / "c.yaml"), readout=None, params=str(tmp_path / "p.pkl"), rung="M_1u",
                label="M_1u_s0", sizes=[16], data=str(data), solver=None, out=str(tmp_path / "out"))
    SR.run_model(SimpleNamespace(**args))
    d = np.load(tmp_path / "out" / "model_M_1u_s0_n16.npz")
    assert d["q"].shape == (2, 64) and np.all(d["q"][:, 3::4] == 0) and not np.any(d["mu_full"])
    assert json.loads(str(d["meta"]))["field_blind"] is True and "c_iso" not in d.files
    pj = jax.tree_util.tree_map(jnp.asarray, p)
    q = np.asarray(M.energy_single(pj, rung, _one(structs[1], rung))[1]["q"])[:48]
    assert np.allclose(d["q"][1].reshape(16, 4)[:, :3].ravel(), q, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match="not a field-blind rung"):
        SR.run_model(SimpleNamespace(**{**args, "rung": "M_16C2pm_us"}))


def test_read_regresses_on_learned_charges_and_the_job6_bands():
    """per-frame charges as a candidate: MB-pol's permanent charges scaled by 1/c read b_L and b_T c times perm's and the
    same r, with no order-0 row; the dipole reading gives |mu| / c, all to the H side, and flips with the sign. Then
    L1.2 and L1.3 hold on three gated seeds in band at both shells, and fail when one of them fails the gate."""
    from test_sumrules import _frames
    rng = np.random.default_rng(8)
    n, side, F = 30, 22.0, 80
    fr, mb = _frames(n, side, F, rng, noise=0.03)
    c = 1.3
    zero = np.zeros_like(mb["mu"])
    models = {"M_1u_s0": {"q": mb["q"] / c, "mu_full": zero, "mu_ind": zero, "field_blind": True},
              "M_1u_s1": {"q": -mb["q"] / c, "mu_full": zero, "mu_ind": zero, "field_blind": True}}
    rd = SR.read_size(fr, mb, models, block=1)
    for sh, row in rd["shells"].items():
        perm, m0, m1 = (row["candidates"][k] for k in ("perm", "M_1u_s0", "M_1u_s1"))
        assert m0["b_T"][0] == pytest.approx(c * perm["b_T"][0], rel=1e-10)
        assert m0["b_L"][0] == pytest.approx(c * perm["b_L"][0], rel=1e-10)
        assert m0["r"][0] == pytest.approx(perm["r"][0], rel=1e-10) and m1["r"][0] == pytest.approx(perm["r"][0], rel=1e-10)
        assert m1["b_T"][0] == pytest.approx(-m0["b_T"][0], rel=1e-10)
        assert "M_1u_s0_o0" not in row["candidates"]
    dp = rd["dipoles"]
    assert dp["M_1u_s0"]["mean_debye"] == pytest.approx(dp["perm"]["mean_debye"] / c, rel=1e-10)
    assert dp["perm"]["to_H_side"] == 1.0 and dp["M_1u_s0"]["to_H_side"] == 1.0 and dp["M_1u_s1"]["to_H_side"] == 0.0
    cand = lambda bT, r: {"b_T": (bT, 0.01), "r": (r, 0.02)}
    reg = [(1.31, 1.75), (-1.30, 1.8), (1.33, 1.7), (1.5, 1.9)]
    sizes = lambda ctl: {n_: {"shells": {"100": {"candidates": {
        **{f"M_1u_s{s}": cand(*v) for s, v in enumerate(reg)},
        **{f"M_1uc_s{s}": cand(x, 1.8) for s, x in enumerate(ctl)}}}}} for n_ in (128, 256)}
    all4 = {0: True, 1: True, 2: True, 3: True}
    same = sizes([1.36, 1.35, 1.37, 1.38])                     # the control's mean within 5 % of M_1u's 1.36
    bands = SR.evaluate_job6(same, all4, all4)
    assert bands["L1.2"]["seeds_in_band"] == [0, 1, 2] and bands["L1.2"]["converged"] is True    # |b_T|: the sign is free
    assert bands["L1.2"]["holds"] and bands["L1.3"]["holds"]
    assert bands["L1.2"]["values"]["n128 (100), seed 1"][0] == pytest.approx(1.30)
    assert bands["L1.2"]["control_means"]["n128 (100)"] == pytest.approx((1.36, 1.365))
    off = SR.evaluate_job6(sizes([1.20, 1.21, 1.19, 1.22]), all4, all4)                         # 12 % below: not converged
    assert off["L1.2"]["converged"] is False and not off["L1.2"]["holds"] and off["L1.3"]["holds"]
    assert SR.evaluate_job6(same, all4, {})["L1.2"]["converged"] is None                         # no gated control
    bands = SR.evaluate_job6(same, {0: True, 1: False, 2: True, 3: True}, all4)
    assert not bands["L1.2"]["holds"] and bands["L1.2"]["gate"]["1"] is False
    assert SR.evaluate_job6({128: {"shells": {"100": {"candidates": {"C": cand(1, 1)}}}}}, {}) == {}
