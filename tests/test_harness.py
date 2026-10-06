"""
pytest -q tests/

Fast checks of the harness: analytic kernels against numerical derivatives, Casimir-Polder against the London
formula, forces of every rung against finite differences, the band-field tensors against finite differences,
isolated atoms and the cutoff (interaction energy exactly zero beyond r_c), and the exact structure of the
analytic-kernel family.  The dimer-ladder and X2-B toy regressions are scripts (scripts/), too slow for pytest.
"""
import numpy as np
import jax, jax.numpy as jnp
import pytest
import sseft
from sseft import kernels as kn, model as M, structure as S, bandfields as BF, data as D


def _cluster(seed=0, n=10):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        x = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(x - p) > 2.5 for p in pos): pos.append(x)
    pos = np.array(pos) * 1.8897; Z = rng.choice([10, 18, 36], n)
    om, w = kn.casimir_polder_grid(8)
    alpha = np.tile(11.1 / (1 + (np.array(om) / 0.6) ** 2), (n, 1))
    q = rng.normal(size=n) * 0.1; q -= q.mean()
    st = S.Structure(pos, Z, energy=-0.01, forces=np.zeros((n, 3)), pinned={"q": q, "alpha": alpha})
    batch = S.pad_batch([st], n_max=n + 2, n_freq=8)
    return {k: v[0] for k, v in batch.items()}


def test_kernel_derivatives():
    l = 2.5
    for r in (1.3, 3.0, 7.0):
        g = lambda x: kn.g_long(x, l)
        assert abs(float(jax.grad(g)(r)) - float(kn.dg_long(r, l))) < 1e-10
        assert abs(float(jax.grad(jax.grad(g))(r)) - float(kn.d2g_long(r, l))) < 1e-9
    # p = 6 closed form against the incomplete gamma
    for r in (0.5, 2.0, 6.0):
        assert abs(float(kn.f6_long(r, l)) - float(kn.k_long(r, l, 6))) < 1e-12 * max(1.0, float(kn.k_long(r, l, 6)))


def test_casimir_polder_london():
    om, w = kn.casimir_polder_grid(8)
    a0a, wa, a0b, wb = 11.1, 0.6, 2.7, 1.0
    aa = a0a / (1 + (om / wa) ** 2); ab = a0b / (1 + (om / wb) ** 2)
    c6 = float(kn.c6_from_alpha(aa, ab, w)); exact = float(kn.london_c6(a0a, wa, a0b, wb))
    assert abs(c6 - exact) / exact < 2e-3


def test_analytic_family_is_laplacian_family():
    polys = kn.gaussian_family_polys(3)
    l = 3.0
    def g(r, n):
        x = (r / l) ** 2; P = polys[n]
        return float(sum(c * x ** k for k, c in enumerate(P)) * np.exp(-x))
    def lap(f, r, h=1e-4):
        return (f(r + h) - 2 * f(r) + f(r - h)) / h ** 2 + 2 / r * (f(r + h) - f(r - h)) / (2 * h)
    for n in (0, 1, 2):
        r = 4.0
        assert abs(g(r, n + 1) - (-l ** 2) * lap(lambda rr: g(rr, n), r)) < 1e-5 * max(1.0, abs(g(r, n + 1)))


@pytest.mark.parametrize("name", list(M.RUNGS))
def test_forces_finite_difference(name):
    b = _cluster()
    rung = M.with_rung(M.RUNGS[name], dipoles=M.RUNGS[name].charges and not M.RUNGS[name].pin_q)
    params = M.init_params(jax.random.PRNGKey(1), rung)
    _, F = M.energy_forces_single(params, rung, b)
    def E_at(i, a, dx):
        return float(M.energy_single(params, rung, {**b, "positions": b["positions"].at[i, a].add(dx)})[0])
    # four-point stencil, truncation error O(h^4).  The Bessel basis (n <= 20 on r_c = 6 A) makes the fifth
    # derivative of E large, so h = 1e-4 is not yet converged at every coordinate (checked: the FD error falls
    # as h^4 from 7e-6 at h = 1e-3 to 2e-11 at h = 3e-5, onto the autodiff value); h = 3e-5 balances
    # truncation (~2e-11) against roundoff (~3e-12) in float64.
    for (i, a) in [(0, 0), (3, 2), (7, 1)]:
        h = 3e-5
        fd = -(-E_at(i, a, 2 * h) + 8 * E_at(i, a, h) - 8 * E_at(i, a, -h) + E_at(i, a, -2 * h)) / (12 * h)
        assert abs(fd - float(F[i, a])) < 1e-6 * abs(fd) + 1e-10, (name, i, a, fd, float(F[i, a]))


def test_band_field_tensors():
    b = _cluster()
    D_, r, pmask = S.pair_geometry(b["positions"], b["mask"], b["cell"])
    l = 2.83
    Ef, gE, ggE = BF.band_field_tensors(b["positions"], b["pin_q"], pmask, l, 2)
    i = 2; x0 = b["positions"][i]
    def V(x):
        d = x[None, :] - b["positions"]; rr = jnp.sqrt(jnp.sum(d * d, axis=-1)); rr = jnp.where(pmask[i], rr, 1.0)
        return jnp.sum(pmask[i] * b["pin_q"] * kn.g_long(rr, l))
    h = 1e-4
    E_fd = -np.array([(V(x0.at[a].add(h)) - V(x0.at[a].add(-h))) / (2 * h) for a in range(3)])
    assert np.allclose(np.array(Ef[i]), E_fd, rtol=1e-6, atol=1e-12)
    assert np.allclose(gE[i], gE[i].T)
    assert np.allclose(ggE[i], np.transpose(ggE[i], (1, 0, 2))) and np.allclose(ggE[i], np.transpose(ggE[i], (0, 2, 1)))
    # padded atoms carry no field
    assert float(jnp.sum(jnp.abs(Ef[-1]))) == 0.0


def test_isolated_atoms_and_cutoff():
    """interaction energy vanishes identically for a dimer beyond r_c under M_inf, and for an isolated atom."""
    rung = M.with_rung(M.RUNGS["M_inf"], e0={**M.RUNGS["M_inf"].e0, "r_cut": 10.0})
    params = M.init_params(jax.random.PRNGKey(3), rung, energy_scale=1e-4)
    far = D.dimer_structures(18, np.array([12.0, 30.0]), lambda R: 0.0)
    iso = [D.isolated_atom(18)]
    batch = S.pad_batch(far + iso, n_freq=8)
    E = jax.vmap(lambda bb: M.energy_single(params, rung, bb)[0])(batch)
    assert np.allclose(np.array(E), 0.0, atol=1e-14)
    near = D.dimer_structures(18, np.array([5.0]), lambda R: 0.0)
    En = M.energy_single(params, rung, {k: v[0] for k, v in S.pad_batch(near, n_freq=8).items()})[0]
    assert abs(float(En)) > 0.0


def test_bands_two_body_limits():
    """pinned sources on a dimer: the Coulomb band (charges + dipoles) and the dispersion band reduce to the
    closed-form two-body expressions, and to the classical multipole energies far beyond the band edge."""
    l = M.RUNGS["M_16p"].l1
    om, w = kn.casimir_polder_grid(8)
    a0, w0 = 11.1, 0.6
    q1, q2, m1, m2 = 0.3, -0.3, 0.2, -0.5
    for R in (4.0, 8.0, 60.0):
        pos = np.array([[0, 0, -R / 2], [0, 0, R / 2]])
        st = S.Structure(pos, np.array([18, 18]), pinned={"q": np.array([q1, q2]), "mu": np.array([[0, 0, m1], [0, 0, m2]]),
                                                        "alpha": np.tile(a0 / (1 + (np.array(om) / w0) ** 2), (2, 1))})
        b = {k: v[0] for k, v in S.pad_batch([st], n_freq=8).items()}
        # Coulomb band alone, charges and dipoles pinned
        rung = M.with_rung(M.RUNGS["M_16p"], dispersion=False, site_energies=False, dipoles=True, pin_mu=True)
        E = float(M.energy_single(params := M.init_params(jax.random.PRNGKey(0), rung), rung, b)[0])
        g, dg, d2g = (float(f(R, l)) for f in (kn.g_long, kn.dg_long, kn.d2g_long))
        exact = q1 * q2 * g + (q1 * m2 - q2 * m1) * dg - m1 * m2 * d2g
        assert abs(E - exact) < 1e-12 * max(1.0, abs(exact)), (R, E, exact)
        if R > 20:   # classical limit:  q q / R  -  (q1 m2 - q2 m1) / R^2  -  2 m1 m2 / R^3
            classical = q1 * q2 / R - (q1 * m2 - q2 * m1) / R ** 2 - 2 * m1 * m2 / R ** 3
            assert abs(E - classical) < 1e-8 * abs(classical)
        # dispersion band alone, alpha pinned
        rung = M.with_rung(M.RUNGS["M_16p"], charges=False, site_energies=False)
        E = float(M.energy_single(M.init_params(jax.random.PRNGKey(0), rung), rung, b)[0])
        c6 = float(kn.c6_from_alpha(b["pin_alpha"][0], b["pin_alpha"][1], w))
        assert abs(E + c6 * float(kn.f6_long(R, l))) < 1e-12 * c6 / R ** 6
        if R > 20:
            assert abs(E + 0.75 * a0 ** 2 * w0 / R ** 6) < 3e-3 * 0.75 * a0 ** 2 * w0 / R ** 6   # London, K = 8 quadrature


def test_rotation_translation_equivariance():
    """energies invariant and forces covariant under a rigid motion, for the rungs with vector / tensor channels
    (learned dipoles; band-field tensors through order 2)."""
    from scipy.spatial.transform import Rotation
    b = _cluster(seed=1)
    Rm = jnp.asarray(Rotation.from_rotvec([0.3, -1.1, 0.7]).as_matrix()); shift = jnp.array([1.0, -2.0, 0.5])
    for name, kw in (("M_16", dict(dipoles=True)), ("M_S2", {}), ("M_A", {})):
        rung = M.with_rung(M.RUNGS[name], **kw)
        params = M.init_params(jax.random.PRNGKey(2), rung)
        E, F = M.energy_forces_single(params, rung, b)
        b2 = {**b, "positions": b["positions"] @ Rm.T + shift}
        E2, F2 = M.energy_forces_single(params, rung, b2)
        assert abs(float(E2 - E)) < 1e-12 * max(1.0, abs(float(E))), name
        assert np.allclose(np.array(F2), np.array(F @ Rm.T), rtol=1e-10, atol=1e-13), name


def test_neutrality_shift():
    b = _cluster()
    rung = M.RUNGS["M_1"]
    params = M.init_params(jax.random.PRNGKey(5), rung)
    _, aux = M.energy_single(params, rung, b)
    assert abs(float(jnp.sum(aux["q"]))) < 1e-12
