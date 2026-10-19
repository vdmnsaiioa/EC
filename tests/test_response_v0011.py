"""v0.0.11 without a q-AQUA build: the induced dipoles' mutual fields (bandfields.mutual_response, Rung.bf_mutual), a
site model's virtual sites as pinned sources (Rung.msites), subtraction groups (Structure.sub: a probe's interaction
energy), and the read-out solve of every order, iterated, with a ridge chosen on held-out groups (sseft.linear)."""
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sseft import bandfields as BF, kernels as kn, linear, model as M, structure as S, train, water as W
from sseft.units import ang_to_bohr

L1 = ang_to_bohr(0.75)
E0CFG = {**M.E0.default_config(), "r_cut": ang_to_bohr(6.0), "F": 8, "n_rbf": 6, "n_rbf_readout": 4}
GAMMA = 0.46
Q_H, Q_M = 0.6127136, -1.2254272


def cluster(n, seed=0, far=None):
    pos = W.random_cluster(n, np.random.default_rng(seed), mc_steps=0)
    if far is not None:                                     # move the last molecule far away
        pos[-3:] += np.array([far, 0.0, 0.0])
    return pos


def site_pins(st):
    """a TTM3-F-like site model: H charges on the H atoms, the M-site at (1 - gamma) r_O + gamma/2 (r_H1 + r_H2)."""
    n = st.n_atoms // 3
    q = np.tile([0.0, Q_H, Q_H], n); w = np.zeros((n, 3 * n))
    for m in range(n):
        w[m, 3 * m:3 * m + 3] = [1 - GAMMA, GAMMA / 2, GAMMA / 2]
    st.pinned = {**st.pinned, "q_site": q, "vsite_w": w, "vsite_q": np.full(n, Q_M), "vsite_frag": np.arange(n)}
    return st


def rung(base="M_S2p", **kw):
    r = {"M_S2p": M.with_rung(M.RUNGS["M_1"], pin_q=True, band_fields=2),
         "M_S0p": M.with_rung(M.RUNGS["M_1"], pin_q=True, band_fields=0),
         "M_1pm": M.with_rung(M.RUNGS["M_1"], pin_q=True, msites=True)}[base]
    return M.with_rung(r, l1=L1, bf_rs=0.0, e0=E0CFG, **kw)


def random_readout(params, seed=7, scale=0.05):
    leaves, td = jax.tree_util.tree_flatten(params["readout"])
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    params["readout"] = jax.tree_util.tree_unflatten(td, [l + scale * jax.random.normal(k, l.shape) for l, k in zip(leaves, keys)])
    return params


def single(st, **kw):
    return {k: v[0] for k, v in S.pad_batch([st], **kw).items()}


# ---------------------------------------------------------------- the mutual response ---------------------------------
def test_mutual_response_series_and_solve():
    rng = np.random.default_rng(0)
    N = 5
    A = rng.normal(size=(N, 3, 3)); C = 0.3 * (A + A.transpose(0, 2, 1)) + 2.0 * np.eye(3)
    R = rng.normal(size=(N, 3)) * 4.0
    T = np.array(BF.band_dipole_tensor(jnp.asarray(R), jnp.asarray(~np.eye(N, dtype=bool)), 1.0))
    E = rng.normal(size=(N, 3)) * 0.01
    e1, mu1 = BF.mutual_response(jnp.asarray(C), jnp.asarray(T), jnp.asarray(E), 1)
    assert abs(float(e1) + 0.5 * np.einsum("ia,iab,ib->", E, C, E)) < 1e-15
    np.testing.assert_allclose(mu1, np.einsum("iab,ib->ia", C, E), rtol=1e-13)
    c = np.zeros((3 * N, 3 * N))
    for i in range(N):
        c[3 * i:3 * i + 3, 3 * i:3 * i + 3] = C[i]
    t = T.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    mu = np.linalg.solve(np.eye(3 * N) - c @ t, c @ E.reshape(-1))
    ex, mux = BF.mutual_response(jnp.asarray(C), jnp.asarray(T), jnp.asarray(E), -1)
    np.testing.assert_allclose(mux.reshape(-1), mu, rtol=1e-12)
    assert abs(float(ex) + 0.5 * E.reshape(-1) @ mu) < 1e-15
    e2, _ = BF.mutual_response(jnp.asarray(C), jnp.asarray(T), jnp.asarray(E), 2)
    second = -0.5 * E.reshape(-1) @ c @ t @ c @ E.reshape(-1)
    assert abs(float(e2) - float(e1) - second) < 1e-16
    e30, _ = BF.mutual_response(jnp.asarray(C), jnp.asarray(T), jnp.asarray(E), 30)
    assert abs(float(e30) - float(ex)) < 1e-12 * abs(float(ex))


def test_band_dipole_tensor_is_the_kernels_hessian():
    R = jnp.array([[0.0, 0.0, 0.0], [1.3, -0.4, 2.2], [3.0, 1.0, -1.0]])
    pm = jnp.array([[False, True, False], [True, False, True], [False, True, False]])
    T = BF.band_dipole_tensor(R, pm, L1)
    hess = jax.hessian(lambda x: kn.g_long(jnp.linalg.norm(x - R[1]), L1))(R[0])
    np.testing.assert_allclose(T[0, 1], hess, rtol=1e-12)
    np.testing.assert_allclose(T[0, 1], T[1, 0], rtol=1e-14)
    assert float(jnp.max(jnp.abs(T[0, 2]))) == 0.0 and float(jnp.max(jnp.abs(T[1, 1]))) == 0.0


def test_first_order_through_the_tensor_is_the_read_out():
    """bf_mutual = 1 (the order-0 sector as polarisabilities, no mutual fields) reproduces the read-out of before to
    rounding, the orders 1-2 included; the series through order k approaches the self-consistent energy."""
    st = W.water_structure(cluster(4), energy=0.0, forces=np.zeros((12, 3)))
    base = rung("M_S2p")
    p = random_readout(M.init_params(jax.random.PRNGKey(1), base, energy_scale=1e-3))
    b = single(st)
    e0 = float(M.energy_single(p, base, b)[0])
    e1 = float(M.energy_single(p, M.with_rung(base, bf_mutual=1), b)[0])
    assert abs(e0 - e1) <= 1e-13 * abs(e0)
    ex = float(M.energy_single(p, M.with_rung(base, bf_mutual=-1), b)[0])
    e12 = float(M.energy_single(p, M.with_rung(base, bf_mutual=12), b)[0])
    assert abs(ex - e0) > 1e-9 and abs(e12 - ex) < 1e-6 * abs(ex - e0)


def test_mutual_response_forces_and_fragments():
    st = W.water_structure(cluster(3), energy=0.0, forces=np.zeros((9, 3)))
    r = rung("M_S0p", bf_mutual=-1)
    p = random_readout(M.init_params(jax.random.PRNGKey(2), r, energy_scale=1e-3))
    b = single(st)
    E, F = M.energy_forces_single(p, r, b)
    v = np.random.default_rng(3).normal(size=b["positions"].shape); v /= np.linalg.norm(v); h = 1e-4
    f = lambda s: float(M.energy_single(p, r, {**b, "positions": b["positions"] + s * v})[0])
    assert abs((f(h) - f(-h)) / (2 * h) + float(np.vdot(F, v))) < 1e-8 * max(1.0, float(np.abs(F).max()))
    # one molecule alone: no other fragment, no field, no response
    mono = single(W.water_structure(cluster(1), energy=0.0))
    assert float(M.aux_batched(p, r, {k: v[None] for k, v in mono.items()})["E_bf"][0]) == 0.0


def test_mutual_response_needs_the_pair_read_out():
    with pytest.raises(ValueError):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(M.RUNGS["M_1"], bf_mutual=-1))
    with pytest.raises(ValueError):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(M.RUNGS["M_1"], band_fields=0, readout="node", bf_mutual=-1))


# ---------------------------------------------------------------- the site model --------------------------------------
def explicit_sites(pos_bohr):
    """(positions, charges, molecule) of the site model's charges, for explicit sums."""
    X, q, mol = [], [], []
    for m in range(len(pos_bohr) // 3):
        O, H1, H2 = pos_bohr[3 * m:3 * m + 3]
        for x, c in ((H1, Q_H), (H2, Q_H), ((1 - GAMMA) * O + GAMMA / 2 * (H1 + H2), Q_M)):
            X.append(x); q.append(c); mol.append(m)
    return np.array(X), np.array(q), np.array(mol)


def test_site_model_coulomb_band_and_fields():
    from scipy.special import erf
    st = site_pins(W.water_structure(cluster(3, seed=4), energy=0.0))
    r = rung("M_1pm")
    p = M.init_params(jax.random.PRNGKey(0), r, energy_scale=1e-3)
    b = single(st)
    aux = M.aux_batched(p, r, {k: v[None] for k, v in b.items()})
    X, q, mol = explicit_sites(np.asarray(st.positions))
    e = 0.0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = np.linalg.norm(X[i] - X[j]); e += q[i] * q[j] * erf(d / L1) / d
    # E_coul of the whole structure includes the molecules' own (constant) pairs; the fragment subtraction removes them
    assert abs(float(aux["E_coul"][0]) - e) < 1e-12 * abs(e)
    inter = sum(q[i] * q[j] * erf(np.linalg.norm(X[i] - X[j]) / L1) / np.linalg.norm(X[i] - X[j])
                for i in range(len(X)) for j in range(i + 1, len(X)) if mol[i] != mol[j])
    E_model = float(M.energy_single(p, r, b)[0])
    E0_only = float(M.energy_single(p, M.with_rung(r, coulomb_energy=False), b)[0])
    assert abs((E_model - E0_only) - inter) < 1e-12 * abs(inter)
    # the band field at the atoms: the other molecules' site charges
    rs = M.with_rung(r, band_fields=0)
    ps = M.init_params(jax.random.PRNGKey(0), rs, energy_scale=1e-3)
    Ef = np.array(M.aux_batched(ps, rs, {k: v[None] for k, v in b.items()})["Efield"][0])
    pos = np.asarray(st.positions)
    for a in range(len(pos)):
        ref = np.zeros(3)
        for j in range(len(X)):
            if mol[j] == a // 3:
                continue
            Rv = pos[a] - X[j]; d = np.linalg.norm(Rv)
            gp = (2 / math.sqrt(math.pi) * math.exp(-(d / L1) ** 2) / L1 - erf(d / L1) / d) / d
            ref += -q[j] * gp * Rv / d
        np.testing.assert_allclose(Ef[a], ref, rtol=1e-11, atol=1e-15)


def test_site_model_forces_and_isolated_molecule():
    st = site_pins(W.water_structure(cluster(3, seed=5), energy=0.0))
    r = M.with_rung(rung("M_1pm"), band_fields=0)
    p = random_readout(M.init_params(jax.random.PRNGKey(4), r, energy_scale=1e-3))
    b = single(st)
    E, F = M.energy_forces_single(p, r, b)
    v = np.random.default_rng(6).normal(size=b["positions"].shape); v /= np.linalg.norm(v); h = 1e-4
    f = lambda s: float(M.energy_single(p, r, {**b, "positions": b["positions"] + s * v})[0])
    assert abs((f(h) - f(-h)) / (2 * h) + float(np.vdot(F, v))) < 1e-8 * max(1.0, float(np.abs(F).max()))
    mono = site_pins(W.water_structure(cluster(1), energy=0.0))
    assert abs(float(M.energy_single(p, r, single(mono))[0])) < 1e-15


def test_site_model_needs_its_pins_and_fragments():
    st = W.water_structure(cluster(2), energy=0.0)
    with pytest.raises(AssertionError):
        M.energy_single(M.init_params(jax.random.PRNGKey(0), rung("M_1pm")), rung("M_1pm"), single(st))
    bad = site_pins(W.water_structure(cluster(2), energy=0.0)); bad.frag = None
    with pytest.raises(ValueError, match="fragments"):
        S.pad_batch([bad])


# ---------------------------------------------------------------- subtraction groups ----------------------------------
def test_subtraction_groups_give_the_probes_interaction():
    pos = cluster(4, seed=8, far=8.0)
    st = W.water_structure(pos, energy=0.0)
    r = rung("M_S2p", bf_mutual=-1)
    p = random_readout(M.init_params(jax.random.PRNGKey(5), r, energy_scale=1e-3))
    E_all = float(M.energy_single(p, r, single(st))[0])
    E_core = float(M.energy_single(p, r, single(W.water_structure(pos[:9], energy=0.0)))[0])
    grouped = W.water_structure(pos, energy=0.0); grouped.sub = np.repeat([0, 1], [9, 3])
    assert abs(float(M.energy_single(p, r, single(grouped))[0]) - (E_all - E_core)) < 1e-15
    # a batch mixing grouped and ungrouped structures: the ungrouped keep their fragments
    data = S.pad_batch([grouped, st])
    E = jax.vmap(lambda bb: M.energy_single(p, r, bb)[0])(data)
    assert abs(float(E[0]) - (E_all - E_core)) < 1e-15 and abs(float(E[1]) - E_all) < 1e-15
    split = W.water_structure(pos, energy=0.0); split.sub = np.array([0, 0, 1] + [1] * 9)
    with pytest.raises(ValueError, match="whole fragments"):
        S.pad_batch([split])


# ---------------------------------------------------------------- the solve -------------------------------------------
def _toy_solve(monkeypatch, nonlinear=False, n=40, seed=0):
    """a model whose energy is f(sum_k b_k phi_k(structure)) in the read-out's last-layer biases b (every column):
    linear (f = identity) or mildly not (f = x + 0.05 x^2, as the mutual response is in its polarisabilities); forces
    zero."""
    rng = np.random.default_rng(seed)
    ncol = 6
    phi = rng.normal(size=(n, ncol))
    b_true = rng.normal(size=ncol)
    lin = phi @ b_true
    energy = lin + (0.05 * lin ** 2 if nonlinear else 0.0) + 1e-3 * rng.normal(size=n)
    params = {"readout": {"theta": [{"w": jnp.zeros((1, ncol)), "b": jnp.zeros(ncol)}],
                          "iso": [{"w": jnp.zeros((1, 1)), "b": jnp.zeros(1)}]}, "gnn": {"w": jnp.ones(())}}
    data = {"mask": jnp.ones((n, 1), dtype=bool), "weight": jnp.ones(n), "has_energy": jnp.ones(n, dtype=bool),
            "has_forces": jnp.zeros(n, dtype=bool), "energy": jnp.asarray(energy), "forces": jnp.zeros((n, 1, 3)),
            "phi": jnp.asarray(phi)}

    def energy_forces(p, r, batch):
        x = batch["phi"] @ p["readout"]["theta"][-1]["b"]
        E = x + (0.05 * x ** 2 if nonlinear else 0.0)
        return E, jnp.zeros((batch["phi"].shape[0], 1, 3))
    monkeypatch.setattr(linear.M, "energy_forces", energy_forces)
    return params, SimpleNamespace(e0={"n_rbf": 1, "n_rbf_readout": 1}), data, b_true


def test_solve_of_every_order_recovers_all_columns(monkeypatch):
    params, r, data, b_true = _toy_solve(monkeypatch)
    zero, rep0 = linear.solve_readout(params, r, data, 1.0, w_force=0.0, ridge=1e-12, chunk=7)
    allc, rep = linear.solve_readout(params, r, data, 1.0, w_force=0.0, ridge=1e-12, chunk=7, orders="all")
    np.testing.assert_allclose(allc["readout"]["theta"][-1]["b"], b_true, atol=2e-3)
    assert np.all(np.asarray(zero["readout"]["theta"][-1]["b"][2:]) == 0.0)       # order 0: the first 2 nr columns only
    assert rep[1] < rep0[1] and rep == (rep.history[0][0], rep.history[-1][1])


def test_gauss_newton_iterations_converge(monkeypatch):
    params, r, data, b_true = _toy_solve(monkeypatch, nonlinear=True)
    one, rep1 = linear.solve_readout(params, r, data, 1.0, w_force=0.0, ridge=1e-12, chunk=8, orders="all")
    many, rep = linear.solve_readout(params, r, data, 1.0, w_force=0.0, ridge=1e-12, chunk=8, orders="all", iterations=8)
    assert rep[1] < rep1[1] and len(rep.history) >= 2
    # every accepted step lowered the objective; a step that does not is the stop, recorded and not taken
    accepted = [h for h in rep.history[1:] if h[1] < h[0]]
    assert len(accepted) >= 1 and rep[1] == min(h[1] for h in [rep.history[0]] + accepted)
    np.testing.assert_allclose(many["readout"]["theta"][-1]["b"], b_true, atol=5e-3)


def test_ridge_chosen_on_held_out_groups(monkeypatch):
    params, r, data, _ = _toy_solve(monkeypatch, n=60)
    groups = np.repeat(np.arange(12), 5)
    _, rep = linear.solve_readout(params, r, data, 1.0, w_force=0.0, chunk=10, orders="all",
                                  ridge_grid=linear.RIDGE_GRID, groups=groups)
    assert rep.ridge[0] in linear.RIDGE_GRID and rep.ridge[0] < 1e-3      # a well-determined problem: a small ridge
    with pytest.raises(ValueError):
        linear.solve_readout(params, r, data, 1.0, w_force=0.0, chunk=10, ridge_grid=linear.RIDGE_GRID, groups=np.zeros(60))


# ---------------------------------------------------------------- the fit's options -----------------------------------
@pytest.fixture
def staged(monkeypatch):
    initial = {"gnn": {"weight": jnp.array(0.0)}, "readout": {"weight": jnp.array(0.0)}}
    monkeypatch.setattr(train.M, "init_params", lambda *args, **kwargs: initial)
    monkeypatch.setattr(train, "pad_batch", lambda *args, **kwargs: {"x": jnp.ones((2, 1)), "weight": jnp.ones(2)})

    def loss(params, batch):
        g = (params["gnn"]["weight"] - 3.0) ** 2; ro = (params["readout"]["weight"] - 4.0) ** 2
        return g + ro, (g, ro)
    monkeypatch.setattr(train, "make_loss", lambda *args, **kwargs: loss)
    monkeypatch.setattr(train, "make_chunked_loss", lambda *args, **kwargs: lambda params: loss(params, None)[0])
    return SimpleNamespace(K=1), [object(), object()]


def test_fit_forwards_the_solve_options_and_freezes_the_read_out(staged, monkeypatch):
    rung_, structures = staged
    seen = []

    def solve(params, rung, data, scale, **kwargs):
        seen.append((np.asarray(data["weight"]), kwargs))
        return {**params, "readout": {"weight": jnp.array(2.0)}}, (1.0, 0.5)
    monkeypatch.setattr(linear, "solve_readout", solve)
    params, info = train.fit(rung_, structures, steps=2, lr=0.1, scale=1.0, lbfgs_steps=5, readout_orders="all",
                             readout_solve_iterations=3, readout_ridge_selection="groups", readout_solve_groups=[0, 1],
                             readout_solve_weights=[1.0, 300.0], lbfgs_freeze_readout=True)
    weights, kwargs = seen[0]
    np.testing.assert_array_equal(weights, [1.0, 300.0])
    assert kwargs["orders"] == "all" and kwargs["iterations"] == 3 and list(kwargs["groups"]) == [0, 1]
    assert kwargs["ridge_grid"] == linear.RIDGE_GRID
    assert float(params["readout"]["weight"]) == 2.0                 # kept through the polish
    assert abs(float(params["gnn"]["weight"]) - 3.0) < 1e-3          # everything else polished
    assert info["training_options"]["lbfgs_freeze_readout"] is True


@pytest.mark.parametrize("kwargs", [
    {"readout_orders": "some"}, {"readout_ridge_selection": "cv"}, {"readout_solve_iterations": 0},
    {"lbfgs_freeze_readout": True, "lbfgs_readout_only": True}, {"lbfgs_freeze_readout": 1},
])
def test_fit_rejects_bad_solve_options(staged, kwargs):
    rung_, structures = staged
    with pytest.raises(ValueError):
        train.fit(rung_, structures, steps=1, scale=1.0, lbfgs_steps=1, **kwargs)


# ---------------------------------------------------------------- the species read-out, the response edge -------------
def test_species_read_out_is_independent_of_the_seed_and_of_E0():
    r = M.with_rung(rung("M_S0p"), bf_species=True, bf_mutual=-1)
    p1 = M.init_params(jax.random.PRNGKey(1), r, energy_scale=1e-3)
    p2 = M.init_params(jax.random.PRNGKey(2), r, energy_scale=5e-3)
    for a_, b_ in zip(jax.tree_util.tree_leaves(p1["readout"]), jax.tree_util.tree_leaves(p2["readout"])):
        np.testing.assert_array_equal(a_, b_)
    p1 = random_readout(p1, seed=3); p2["readout"] = p1["readout"]
    b = single(W.water_structure(cluster(3, seed=9), energy=0.0))
    e1 = M.aux_batched(p1, r, {k: v[None] for k, v in b.items()})["E_bf"]
    e2 = M.aux_batched(p2, r, {k: v[None] for k, v in b.items()})["E_bf"]
    assert abs(float(e1[0]) - float(e2[0])) < 1e-15 and abs(float(e1[0])) > 0
    with pytest.raises(ValueError):
        M.init_params(jax.random.PRNGKey(0), M.with_rung(M.RUNGS["M_1"], bf_species=True))


def test_response_edge_smears_the_read_outs_field():
    st = W.water_structure(cluster(3, seed=10), energy=0.0)
    b = single(st)
    l2 = ang_to_bohr(1.25)
    r = M.with_rung(rung("M_S0p"), bf_l=l2)
    p = M.init_params(jax.random.PRNGKey(0), r, energy_scale=1e-3)
    Ef = np.asarray(M.aux_batched(p, r, {k: v[None] for k, v in b.items()})["Efield"][0])
    fm = np.asarray(b["frag_mask"]).astype(float); same = (fm.T @ fm) > 0.5
    pos = jnp.asarray(b["positions"]); pm = jnp.asarray(~np.eye(9, dtype=bool))
    ref = BF.band_field_tensors(pos, b["pin_q"], pm, l2, 0, 0.0, src_mask=jnp.asarray(~same))[0]
    np.testing.assert_allclose(Ef, np.asarray(ref), rtol=1e-12, atol=1e-16)
    ref075 = BF.band_field_tensors(pos, b["pin_q"], pm, L1, 0, 0.0, src_mask=jnp.asarray(~same))[0]
    assert np.max(np.abs(np.asarray(ref075) - Ef)) > 1e-6


def test_fit_matches_the_read_out_first_and_keeps_it(staged, monkeypatch):
    rung_, structures = staged
    order = []

    def solve(params, rung, data, scale, **kwargs):
        order.append(("solve", float(params["gnn"]["weight"]), kwargs.get("w_force"), np.asarray(data["weight"]).shape))
        return {**params, "readout": {"weight": jnp.array(2.5)}}, (1.0, 0.5)
    monkeypatch.setattr(linear, "solve_readout", solve)
    params, info = train.fit(rung_, structures, steps=3, lr=0.1, scale=1.0, lbfgs_steps=5, readout_first=True,
                             readout_solve_subset=[1], readout_solve_w_force=0.0, lbfgs_freeze_readout=True)
    assert order == [("solve", 0.0, 0.0, (1,))]                      # before Adam (the gnn untouched), on the subset
    assert float(params["readout"]["weight"]) == 2.5 and abs(float(params["gnn"]["weight"]) - 3.0) < 1e-3
    assert info["training_options"]["readout_first"] is True and info["training_options"]["readout_solve_structures"] == 1
    with pytest.raises(ValueError, match="readout_first"):
        train.fit(rung_, structures, steps=1, scale=1.0, lbfgs_steps=1, readout_first=True, readout_solve_subset=[1])
