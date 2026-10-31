"""
The periodic path (sseft/periodic.py, v0.0.20): the chunked reciprocal sums, the site model's band, band field and
field term in a cell, the response's periodic dipole tensor and its solves, the external field; and the model built on
them -- a cluster in a large cell against its open value plus the dipole image term, the supercell identity, forces
against differences, invariance under wrapping and translation.
"""
import math
from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import sseft  # noqa: F401  (x64)
from sseft import periodic as PB, ewald as EW, bandfields as BF, kernels as kn, model as M, structure as S, water as W
from sseft.mbpol import random_box
from sseft.units import ang_to_bohr as A2B

GAMMA, QH = 0.4267, 0.57765                    # MB-pol's rigid-monomer site model (M-site on the bisector, H charge)


def water_structure(pos_ang, cell_ang=None):
    """rigid waters (A) with MB-pol's site model and dispersion pins, open or in a cell."""
    n = len(pos_ang) // 3
    w = np.zeros((n, 3 * n))
    for k in range(n):
        w[k, 3 * k:3 * k + 3] = [1 - GAMMA, GAMMA / 2, GAMMA / 2]
    Z = np.tile([8, 1, 1], n)
    pinned = {"q_site": np.tile([0.0, QH, QH], n), "vsite_w": w, "vsite_q": np.full(n, -2 * QH),
              "vsite_frag": np.arange(n), "alpha": W.mbpol_dispersion_pins(Z)}
    return S.Structure(A2B(np.asarray(pos_ang, float)), Z, cell=None if cell_ang is None else A2B(np.asarray(cell_ang)),
                       pbc=cell_ang is not None, frag=np.repeat(np.arange(n), 3), pinned=pinned)


def one(st):
    return {k: v[0] for k, v in S.pad_batch([st], n_freq=8).items()}


def frozen(r_cut=6.0, F=8, **kw):
    """M_16C2pm_us (E3's frozen model, scripts/e3_water.py) at E3-QP8's settings, E_0's width F."""
    e0 = {**M.RUNGS["M_1"].e0, "r_cut": A2B(r_cut), "F": F, "n_rbf_readout": 8}
    return M.with_rung(M.RUNGS["M_1"], name="M_16C2pm_us", pin_q=True, msites=True, dispersion=True, pin_alpha=True,
                       band_fields=2, bf_species=True, bf_mutual=-1, e0_field=True, e0_field_whole=True,
                       e0_field_on=A2B(r_cut - 1.0), e0_field_off=A2B(r_cut), e0_field_self_on=A2B(r_cut - 2.0),
                       e0_field_self_off=A2B(r_cut - 1.0), e0_field_cross_species=True, l1=A2B(0.75), bf_l=A2B(0.75),
                       bf_rs=0.0, e0=e0, **kw)


def switched_on(rung, key=0):
    """parameters with every channel on: an isotropic species polarisability (C = 5 bohr^3 on every atom), small random
    read-out orders 1-2 and pair terms, the field term's MLP and its species cross terms."""
    p = M.init_params(jax.random.PRNGKey(key), rung)
    k = jax.random.split(jax.random.PRNGKey(key + 1), 3)
    last = p["readout"]["theta"][-1]
    p["readout"]["theta"][-1] = {"w": 0.01 * jax.random.normal(k[0], last["w"].shape), "b": last["b"]}
    il = p["readout"]["iso"][-1]
    p["readout"]["iso"][-1] = {"w": jnp.zeros_like(il["w"]), "b": jnp.full_like(il["b"], 0.5)}
    fl = p["field"]["mlp"][-1]
    p["field"]["mlp"][-1] = {"w": 0.05 * jax.random.normal(k[1], fl["w"].shape), "b": fl["b"]}
    p["field_x"] = {"a": jnp.full_like(p["field_x"]["a"], 0.1), "c": jnp.full_like(p["field_x"]["c"], 0.05)}
    return p


def cluster(n, seed=0, box=6.0):
    pos = random_box(n, np.eye(3) * box, np.random.default_rng(seed), min_oo=2.6, min_any=1.6)
    return pos - pos.mean(axis=0)


def in_cell(rung, cell_bohr, b_max=4.0, **kw):
    return replace(rung, periodic=True, ewald_n_max=PB.n_max_for(cell_bohr, rung.l1, b_max), ewald_b_max=b_max, **kw)


# ---------------------------------------------------------------- the chunked sums -----------------------------
def test_chunked_sums_equal_the_dense_ones():
    """coulomb_pbc (charges and dipoles), the p = 6 / 8 dispersion bands and the field tensors equal ewald.py's and
    bandfields.py's dense sums over the whole grid, at any chunk size."""
    rng = np.random.default_rng(0)
    a, l, N = 12.0, 2.0, 6
    cell = jnp.asarray(np.array([[a, 0, 0], [0.7, a, 0], [0.3, -0.5, 0.9 * a]]))
    X = jnp.asarray(rng.uniform(0, a, (N, 3)))
    q = rng.normal(size=N); q = jnp.asarray(q - q.mean())
    mu = jnp.asarray(rng.normal(size=(N, 3)) * 0.4)
    on = jnp.ones(N, bool)
    ref = float(EW.coulomb_band_pbc(q, mu, X, cell, on, l, 12, 5.0))
    for chunk in (0, 97, 5000):
        assert abs(float(PB.coulomb_pbc(X, q, on, cell, l, 12, 5.0, mu=mu, chunk=chunk)) - ref) < 1e-12 * abs(ref)
    om, w = kn.casimir_polder_grid(8)
    alpha = jnp.asarray(rng.uniform(1, 5, (N, 1)) / (1 + (np.asarray(om)[None] / 0.6) ** 2))
    a2 = alpha * 3.0
    assert np.isclose(float(PB.dispersion_band_pbc(alpha, w, X, on, cell, l, 12, 5.0, chunk=211)),
                      float(EW.dispersion_band_pbc(alpha, w, X, cell, on, l, 12, 5.0)), rtol=1e-12, atol=0)
    assert np.isclose(float(PB.dispersion8_band_pbc(alpha, a2, w, X, on, cell, l, 12, 5.0)),
                      float(EW.dispersion8_band_pbc(alpha, a2, w, X, cell, on, l, 12, 5.0)), rtol=1e-12, atol=0)
    # the field tensors: bandfields' periodic version leaves the own charge out (its gradient analytically)
    Ef, gE, ggE = PB.field_tensors_pbc(X, X, q, on, cell, l, 2, 12, 5.0, chunk=150)
    Er, gEr, ggEr = BF.band_field_tensors_pbc(X, q, on, cell, l, 2, 12, 5.0)
    self_g = (q * 4.0 / (3.0 * math.sqrt(math.pi) * l ** 3))[:, None, None] * jnp.eye(3)
    assert np.allclose(Ef, Er, rtol=1e-11, atol=1e-14)
    assert np.allclose(gE - self_g, gEr, rtol=1e-11, atol=1e-14)
    assert np.allclose(ggE, ggEr, rtol=1e-11, atol=1e-14)


def test_n_max_too_small_returns_nan():
    cell = jnp.eye(3) * 20.0
    X = jnp.asarray(np.random.default_rng(1).uniform(0, 20, (4, 3))); q = jnp.asarray([0.5, -0.5, 0.2, -0.2])
    n = PB.n_max_for(np.eye(3) * 20.0, 1.5, 4.0)
    assert np.isfinite(float(PB.coulomb_pbc(X, q, jnp.ones(4, bool), cell, 1.5, n, 4.0)))
    assert np.isnan(float(PB.coulomb_pbc(X, q, jnp.ones(4, bool), cell, 1.5, n - 1, 4.0)))


# ---------------------------------------------------------------- the dipole tensor ----------------------------
def test_dipole_matvec_is_the_band_energy_of_dipoles():
    """-1/2 mu . T mu, T leaving out only each dipole's own field at n = 0, is the periodic band energy of the dipoles
    alone (coulomb_pbc with q = 0: an independent sum); the explicit tensor is the matvec's; the own-fragment exclusion
    removes exactly the direct pairs."""
    rng = np.random.default_rng(2)
    a, l, N = 11.0, 1.6, 5
    cell = jnp.asarray(np.diag([a, 1.1 * a, 0.95 * a]))
    X = jnp.asarray(rng.uniform(0, a, (N, 3))); mu = jnp.asarray(rng.normal(size=(N, 3)))
    mask = jnp.ones(N, bool)
    n = PB.n_max_for(np.asarray(cell), l, 5.0)
    none_idx, none_on = jnp.zeros((N, 1), int), jnp.zeros((N, 1), bool)
    Tmu = PB.dipole_matvec_pbc(mu, X, mask, none_idx, none_on, cell, l, n, 5.0)
    E_band = float(PB.coulomb_pbc(X, jnp.zeros(N), mask, cell, l, n, 5.0, mu=mu))
    assert abs(-0.5 * float(jnp.sum(mu * Tmu)) - E_band) < 1e-11 * abs(E_band)
    # two fragments: atoms (0, 1, 2) and (3, 4)
    own_idx = jnp.asarray([[1, 2], [0, 2], [0, 1], [4, 4], [3, 3]])
    own_on = jnp.asarray([[True, True]] * 3 + [[True, False]] * 2)
    T = PB.dipole_tensor_pbc(X, mask, own_idx, own_on, cell, l, n, 5.0)
    Tmu_x = PB.dipole_matvec_pbc(mu, X, mask, own_idx, own_on, cell, l, n, 5.0)
    assert np.allclose(jnp.einsum("ijab,jb->ia", T, mu), Tmu_x, rtol=1e-11, atol=1e-13)
    direct = PB.own_dipole_field(mu, X, own_idx, own_on, l)
    assert np.allclose(Tmu - Tmu_x, direct, rtol=1e-11, atol=1e-13)
    d = X[0] - X[1]
    assert np.allclose(PB._T_direct(d, l), BF.band_dipole_tensor(X[:2], jnp.asarray([[False, True], [True, False]]), l)[0, 1])


@pytest.mark.parametrize("solver", ["direct", "gmres"])
def test_clausius_mossotti(solver):
    """isotropic point polarisabilities on a simple cubic lattice in a uniform field: with tin-foil boundary conditions
    the induced dipoles feel the Lorentz field (4 pi / 3V) mu, so eps = (1 + 8 pi n alpha / 3) / (1 - 4 pi n alpha / 3)
    exactly (the band kernel's smearing enters as exp(-a^2 / l^2), ~1e-22, the sum's truncation as exp(-b_max^2)); one
    atom per cell or a 2x2x2 supercell."""
    a, l, alpha = 10.0, 1.4, 25.0
    x = 4 * math.pi * alpha / (3 * a ** 3)
    eps_cm = (1 + 2 * x) / (1 - x)
    Eext = jnp.asarray([0.0, 0.0, 1e-3])
    for reps in (1, 2):
        g = np.arange(reps)
        X = jnp.asarray(np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * a + 0.3)
        N = len(X)
        cell = jnp.eye(3) * a * reps
        C = jnp.broadcast_to(alpha * jnp.eye(3), (N, 3, 3))
        n = PB.n_max_for(np.asarray(cell), l, 6.0)
        _, mu = PB.mutual_response_pbc(C, jnp.broadcast_to(Eext, (N, 3)), X, jnp.ones(N, bool), jnp.zeros((N, 1), int),
                                       jnp.zeros((N, 1), bool), cell, l, n, 6.0, solver=solver, tol=1e-13)
        chi = float(jnp.sum(mu[:, 2])) / (float(jnp.linalg.det(cell)) * 1e-3)
        assert abs(1 + 4 * math.pi * chi - eps_cm) < 1e-10, (reps, 1 + 4 * math.pi * chi, eps_cm)
        assert float(jnp.max(jnp.abs(mu[:, :2]))) < 1e-15


# ---------------------------------------------------------------- the site model's band field ------------------
def _antiparallel_pair():
    """two rigid waters, the second the first's inversion: the cell's dipole is zero, so image sums of the field
    converge absolutely (each image cell a quadrupole)."""
    m = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.92662721, 0.0]])
    rot = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    A = m @ rot.T + np.array([3.1, 3.4, 2.9])
    c = np.array([5.0, 5.2, 4.7])
    return np.concatenate([A, 2 * c - A])


def test_site_band_field_against_image_sums():
    """the band field tensors at the atoms of a site model in a cell, every source and image except the atom's own
    molecule at n = 0, against explicit sums over a sphere of images (absolutely convergent: the cell's dipole is
    zero)."""
    pos = _antiparallel_pair(); L = 9.0
    st = water_structure(pos, np.eye(3) * L)
    b = one(st)
    l = A2B(0.75)
    rung = in_cell(frozen(r_cut=3.0), np.asarray(b["cell"]), 5.0)
    p = b["positions"]
    X_src = jnp.concatenate([p, b["vs_w"] @ p]); q_src = jnp.concatenate([b["pin_q_site"], b["vs_q"]])
    on_src = jnp.concatenate([b["mask"], b["vs_mask"]])
    sX, sq, son = PB.grouped_sources(p, b)
    _, _, slots = PB.own_fragment(b, sX.shape[1])
    af, sl = b["atom_frag"], b["atom_slot"]
    Ef, gE, ggE = PB.site_band_field_pbc(p, b["mask"], X_src, q_src, on_src, sX[af], sq[af], son[af] & slots,
                                         sq[af, sl] * son[af, sl], b["cell"], l, 2, rung.ewald_n_max, 5.0)
    Lb = float(b["cell"][0, 0])
    n = 7
    r = np.arange(-n, n + 1)
    imgs = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    imgs = imgs[np.linalg.norm(imgs, axis=1) <= n] * Lb
    Xs, Qs = np.asarray(X_src), np.asarray(q_src)
    frag_src = np.array([0, 0, 0, 1, 1, 1, 0, 1])                     # atoms, then the two M-sites
    P_all = (Xs[None] + imgs[:, None]).reshape(-1, 3); Q_all = np.tile(Qs, len(imgs))
    zero = np.all(imgs == 0, axis=1).repeat(len(Xs))
    F_all = np.tile(frag_src, len(imgs))

    def V(x, f):
        keep = jnp.asarray(~(zero & (F_all == f)))
        d = jnp.where(keep[:, None], x[None] - jnp.asarray(P_all), 1.0)
        rr = jnp.sqrt(jnp.sum(d * d, axis=-1))
        return jnp.sum(keep * jnp.asarray(Q_all) * kn.g_long(rr, l))
    E = lambda x, f: -jax.grad(V)(x, f)
    for i in range(6):
        f = i // 3
        x = p[i]
        # the sphere's remainder falls as 1/R, 1/R^2, 1/R^3 for the three tensors (a quadrupole lattice)
        assert np.allclose(Ef[i], E(x, f), rtol=3e-3, atol=3e-7), (i, Ef[i], E(x, f))
        assert np.allclose(gE[i], jax.jacfwd(E)(x, f), rtol=1e-3, atol=2e-8), (i, gE[i], jax.jacfwd(E)(x, f))
        assert np.allclose(ggE[i], jax.jacfwd(jax.jacfwd(E))(x, f), rtol=3e-4, atol=2e-9)


def test_local_field_in_a_large_cell_is_the_open_one():
    """E_0's view of the pinned field in a cell wider than its reach is the open one, part by part (every fragment
    whole, the self terms' shorter reach), wherever the molecules sit across the cell's faces."""
    pos = cluster(5, seed=4)
    L = 20.0
    rung = frozen()
    st_o = water_structure(pos)
    bo = one(st_o)
    p = bo["positions"]
    X_src = jnp.concatenate([p, bo["vs_w"] @ p]); q_src = jnp.concatenate([bo["pin_q_site"], bo["vs_q"]])
    on_src = jnp.concatenate([bo["mask"], bo["vs_mask"]])
    frag_src = jnp.concatenate([bo["frag_mask"], bo["vs_frag_mask"]], axis=1)
    same = (bo["frag_mask"].astype(float).T @ frag_src.astype(float)) > 0.5
    src = bo["mask"][:, None] & on_src[None] & ~same
    ref = BF.local_field_tensors(p, X_src, q_src, src, rung.l1, rung.e0_field_on, rung.e0_field_off, frag_src=frag_src,
                                 per_frag=True, self_taper=(rung.e0_field_self_on, rung.e0_field_self_off))
    # in a cell, the cluster straddling a corner (molecules whole or not)
    shift = np.array([-1.0, 0.5, -2.0])
    pc = (pos + shift) % L
    st_c = water_structure(pc, np.eye(3) * L)
    bc = one(st_c)
    pw = PB.make_whole(bc["positions"], bc["atom_frag"], bc["frag_atoms"], bc["cell"])
    sX, sq, son = PB.grouped_sources(pw, bc)
    out = PB.local_field_pbc(pw, bc["atom_frag"], bc["mask"], sX, sq, son, bc["cell"], rung.l1, rung.e0_field_on,
                             rung.e0_field_off, per_frag=True, self_taper=(rung.e0_field_self_on, rung.e0_field_self_off))
    for a, b_ in zip(ref[:3], out[:3]):
        assert np.allclose(a, b_, rtol=1e-12, atol=1e-15)
    for a, b_ in zip(ref[3], out[3]):
        assert np.allclose(a, b_, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------- the model in a cell ----------------------------
def test_frozen_model_cluster_in_a_large_cell():
    """M_16C2pm_us with every channel on: a cluster in cubic cells of 30 and 40 A against its open energy.  E_0 and the
    field term are the open ones exactly; the rest differs by the tin-foil dipole image term, -(2 pi / 3V)
    |M_perm + M_ind|^2, the open cluster's whole dipole, up to O(L^-5) (extrapolated in 1 / L^2)."""
    pos = cluster(4, seed=0)
    rung = frozen()
    params = switched_on(rung)
    st = water_structure(pos)
    bo = one(st)
    E_open, aux_o = M.energy_single(params, rung, bo)
    p = np.asarray(bo["positions"])
    Mp = st.pinned["q_site"] @ p + st.pinned["vsite_q"] @ (st.pinned["vsite_w"] @ p)
    Mtot = Mp + np.asarray(aux_o["mu_ind"]).sum(0)
    vals = []
    for L in (30.0, 40.0):
        b = one(water_structure(pos + L / 2, np.eye(3) * L))
        rc = in_cell(rung, np.asarray(b["cell"]), 5.0)
        E, aux = jax.jit(lambda pp, bb: M.energy_single(pp, rc, bb))(params, b)
        assert abs(float(aux["E0"]) - float(aux_o["E0"])) < 1e-12 * abs(float(aux_o["E0"]))
        assert abs(float(aux["E_field"]) - float(aux_o["E_field"])) < 1e-12 * abs(float(aux_o["E_field"]))
        vals.append(float(jnp.linalg.det(b["cell"])) * (float(E) - float(E_open)))
    limit = (16 * vals[1] - 9 * vals[0]) / 7
    assert abs(limit / (-(2 * math.pi / 3) * Mtot @ Mtot) - 1) < 3e-3, (vals, limit, -(2 * math.pi / 3) * Mtot @ Mtot)


def test_supercell_identity_forces_and_wrapping():
    """a periodic water cell: its 2x1x1 supercell has twice its energy; forces are the energy's gradient (direct and
    matrix-free solves alike); the energy does not change when a molecule is split across a face, or when every atom
    is translated (the cell at E_0's smallest width: r_c = 3 A in an 8 A cell)."""
    rung = frozen(r_cut=3.0)
    params = switched_on(rung, key=3)
    L = 8.0
    pos = random_box(4, np.eye(3) * L, np.random.default_rng(5), min_oo=2.6, min_any=1.6)
    b = one(water_structure(pos, np.eye(3) * L))
    rc = in_cell(rung, np.asarray(A2B(np.diag([2 * L, L, L]))), 4.0)            # n_max for the supercell holds the cell
    E1 = float(M.energy_single(params, rc, b)[0])
    sup = np.concatenate([pos, pos + np.array([L, 0, 0])])
    b2 = one(water_structure(sup, np.diag([2 * L, L, L])))
    E2 = float(M.energy_single(params, rc, b2)[0])
    assert abs(E2 - 2 * E1) < 1e-10 * abs(E1), (E1, E2)
    # a hydrogen wrapped across a face, every atom translated
    p3 = pos.copy(); p3[1] += np.array([L, 0, 0])
    assert abs(float(M.energy_single(params, rc, one(water_structure(p3, np.eye(3) * L)))[0]) - E1) < 1e-12 * abs(E1)
    p4 = (pos + np.array([0.37, -2.1, 5.3])) % L
    assert abs(float(M.energy_single(params, rc, one(water_structure(p4, np.eye(3) * L)))[0]) - E1) < 1e-11 * abs(E1)
    # forces against differences; the matrix-free solve gives the same
    _, F = jax.jit(lambda pp, bb: M.energy_forces_single(pp, rc, bb))(params, b)
    rg = replace(rc, bf_solver="gmres", bf_tol=1e-13)
    Eg, Fg = jax.jit(lambda pp, bb: M.energy_forces_single(pp, rg, bb))(params, b)
    assert abs(float(Eg) - E1) < 1e-12 * abs(E1) and np.allclose(Fg, F, rtol=1e-9, atol=1e-12)
    Ef = jax.jit(lambda x: M.energy_single(params, rc, {**b, "positions": x})[0])
    h = 1e-5                           # E_0's random weights at r_c = 3 A curve fast: the stencil is off by 1e-5 at h = 1e-4
    for i, c in [(0, 0), (5, 1), (10, 2)]:
        v = [float(Ef(b["positions"].at[i, c].add(k * h))) for k in (2, 1, -1, -2)]
        fd = -(-v[0] + 8 * v[1] - 8 * v[2] + v[3]) / (12 * h)
        assert abs(fd - float(F[i, c])) < 1e-6 * abs(fd) + 1e-9, (i, c, fd, float(F[i, c]))


def test_one_molecule_in_a_cell_keeps_its_own_images():
    """the interaction energy of a single molecule in a cell is its interaction with its own images (the fragment
    alone is subtracted open): tending to the tin-foil dipole image energy -(2 pi / 3V) |mu|^2 of its permanent plus
    induced dipole; E_0 and the field term give nothing."""
    m = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.92662721, 0.0]])
    rung = frozen()
    params = switched_on(rung, key=4)
    vals = []
    for L in (24.0, 32.0):
        b = one(water_structure(m + L / 2, np.eye(3) * L))
        rc = in_cell(rung, np.asarray(b["cell"]), 5.0)
        E, aux = M.energy_single(params, rc, b)
        assert abs(float(aux["E_field"])) < 1e-15
        vals.append(float(jnp.linalg.det(b["cell"])) * float(E))
    st = water_structure(m)
    p = A2B(m)
    mu = st.pinned["q_site"] @ p + st.pinned["vsite_q"] @ (st.pinned["vsite_w"] @ p)
    limit = (16 * vals[1] - 9 * vals[0]) / 7
    # the molecule's own induced dipole in its images' field is second order; the permanent dipole sets the limit
    assert abs(limit / (-(2 * math.pi / 3) * mu @ mu) - 1) < 1e-2, (vals, limit, -(2 * math.pi / 3) * mu @ mu)


def test_external_field_and_dielectric():
    """-dE/dE_ext is the read-out's induced dipole; the first-order response gives eps_inf = 1 + 4 pi sum_i C_i / V, the
    self-consistent one more; the dielectric tensor is symmetric; without a read-out it is 1 exactly."""
    rung = frozen(r_cut=3.0)
    params = switched_on(rung, key=6)
    L = 8.0
    pos = random_box(4, np.eye(3) * L, np.random.default_rng(7), min_oo=2.6, min_any=1.6)
    b = one(water_structure(pos, np.eye(3) * L))
    rc = in_cell(rung, np.asarray(b["cell"]), 4.0)
    eps_c = np.asarray(M.dielectric_inf(params, rc, b))
    assert np.allclose(eps_c, eps_c.T, atol=1e-10)
    r1 = replace(rc, bf_mutual=0)
    eps_1 = np.asarray(M.dielectric_inf(params, r1, b))
    # the first-order response: chi = (1/V) sum_i C_i, C = iso 1 - 2 sum_j (w0 + w1 n n) from the read-out
    _, aux = M.energy_single(params, rc, b)
    V = float(jnp.linalg.det(b["cell"]))
    mu1 = np.asarray(M.induced_dipole(params, r1, {**b, "E_ext": jnp.asarray([0.0, 0.0, 1e-4])}))
    mu0 = np.asarray(M.induced_dipole(params, r1, b))
    assert np.allclose(4 * math.pi * (mu1 - mu0) / (1e-4 * V), eps_1[:, 2] - np.eye(3)[:, 2], atol=1e-8)
    assert eps_c[2, 2] > eps_1[2, 2] > 1.0
    r0 = replace(rc, band_fields=-1, bf_mutual=0, bf_species=False)
    p0 = M.init_params(jax.random.PRNGKey(0), r0)
    assert np.allclose(np.asarray(M.dielectric_inf(p0, r0, b)), np.eye(3), atol=1e-14)


def test_mbpol_dispersion_pins_give_its_c6():
    """the pinned alpha's Casimir-Polder C6 is MB-pol's long-range c6_i c6_j (rank one), in atomic units."""
    from sseft.units import KCALMOL_PER_HARTREE, BOHR_PER_ANGSTROM
    a = W.mbpol_dispersion_pins([8, 1])
    om, w = kn.casimir_polder_grid(8)
    C6 = np.asarray(kn.c6_from_alpha(jnp.asarray(a)[:, None], jnp.asarray(a)[None, :], w))
    unit = BOHR_PER_ANGSTROM ** 6 / KCALMOL_PER_HARTREE
    c6 = np.array([W.MBPOL_C6_LR[8], W.MBPOL_C6_LR[1]])
    assert np.allclose(C6 / unit, np.outer(c6, c6), rtol=1e-12)
    assert abs(C6[0, 0] / unit - 237.32) < 0.01


def test_a_batch_of_cells_is_its_cells():
    """two cells of different sizes and molecule counts in one padded batch: the batched energies and forces are the
    single cells' (the fragment grouping padded; the k grid of the larger cell serves both)."""
    rung = frozen(r_cut=3.0)
    params = switched_on(rung, key=8)
    sts = [water_structure(random_box(n, np.eye(3) * L, np.random.default_rng(10 + n), min_oo=2.6, min_any=1.6), np.eye(3) * L)
           for n, L in ((4, 8.0), (5, 9.0))]
    rc = in_cell(rung, np.asarray(A2B(np.eye(3) * 9.0)), 4.0)
    batch = S.pad_batch(sts, n_freq=8)
    E, F = jax.jit(lambda p, bb: M.energy_forces(p, rc, bb))(params, batch)
    for k, st in enumerate(sts):
        Ek, Fk = M.energy_forces_single(params, rc, one(st))
        n = st.n_atoms
        assert abs(float(E[k]) - float(Ek)) < 1e-12              # hartree: a small difference of O(1) terms
        assert np.allclose(F[k, :n], Fk[:n], rtol=1e-10, atol=1e-13)
        assert np.all(np.asarray(F[k, n:]) == 0)
