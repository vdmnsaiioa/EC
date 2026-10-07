"""
Periodic bands against known lattice sums.  The band is the long (erf-type) part of each kernel; adding the
short remainder by a direct real-space sum over images must give the full lattice sum.
"""
import math
import numpy as np
import jax.numpy as jnp
from scipy.special import erfc
import sseft
from sseft import ewald as EW, kernels as kn


def _images(n):
    r = np.arange(-n, n + 1)
    return np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)


def test_nacl_madelung():
    """E_cell of the 8-ion NaCl cell = 4 pairs x (-1.747565 / r_nn), r_nn = a/2 (atomic units, q = +-1)."""
    a = 10.0; l = 2.0
    fcc = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pos = np.concatenate([fcc, fcc + [0.5, 0, 0]]) * a
    q = np.array([1, 1, 1, 1, -1, -1, -1, -1], float)
    cell = np.eye(3) * a
    E_long = float(EW.coulomb_band_pbc(jnp.asarray(q), None, jnp.asarray(pos), jnp.asarray(cell), jnp.ones(8, bool), l, n_max=10, b_max=5.0))
    # short part: 1/2 sum_{i,j,L}' q_i q_j erfc(r/l)/r, direct, converges like exp(-r^2/l^2)
    L = _images(2) * a
    E_short = 0.0
    for i in range(8):
        for j in range(8):
            d = pos[i] - pos[j] + L
            r = np.linalg.norm(d, axis=1)
            m = r > 1e-9
            E_short += 0.5 * q[i] * q[j] * np.sum(erfc(r[m] / l) / r[m])
    E_ref = -4 * 1.747564594633 * 2.0 / a
    assert abs(E_long + E_short - E_ref) < 1e-9 * abs(E_ref), (E_long, E_short, E_ref)


# simple-cubic lattice sums sum' |n|^-p (sphere of radius 120 plus the spherical tail, converged to 1e-9)
SC_SUMS = {6: 8.4019239750, 8: 6.9458079272}


def test_dispersion_sc_lattice():
    """one atom per cubic cell, C = 1: E = -1/2 sum' |L|^-6 (and |L|^-8)."""
    a = 6.0; l = 2.5
    pos = np.zeros((1, 3)); cell = np.eye(3) * a; mask = jnp.ones(1, bool)
    om, w = kn.casimir_polder_grid(8)
    # pick alpha(i w) = const c so that C6 = (3/pi) c^2 sum w = 1
    c = math.sqrt(math.pi / (3.0 * float(jnp.sum(w))))
    alpha = jnp.full((1, 8), c)
    E6 = float(EW.dispersion_band_pbc(alpha, w, jnp.asarray(pos), jnp.asarray(cell), mask, l, n_max=12, b_max=5.0))
    L = _images(6) * a; r = np.linalg.norm(L, axis=1); r = r[r > 0]
    short6 = -0.5 * np.sum(1.0 / r ** 6 - np.array(kn.f6_long(jnp.asarray(r), l)))
    ref6 = -0.5 * SC_SUMS[6] / a ** 6
    assert abs(E6 + short6 - ref6) < 1e-8 * abs(ref6), (E6, short6, ref6)
    # p = 8 with alpha1 = alpha2 = c8 so that C8 = (15/pi) c8^2 sum w = 1
    c8 = math.sqrt(math.pi / (15.0 * float(jnp.sum(w))))
    a1 = jnp.full((1, 8), c8)
    E8 = float(EW.dispersion8_band_pbc(a1, a1, w, jnp.asarray(pos), jnp.asarray(cell), mask, l, n_max=12, b_max=5.0))
    short8 = -0.5 * np.sum(1.0 / r ** 8 - np.array(kn.f8_long(jnp.asarray(r), l)))
    ref8 = -0.5 * SC_SUMS[8] / a ** 8
    assert abs(E8 + short8 - ref8) < 1e-8 * abs(ref8), (E8, short8, ref8)


def test_dipoles_as_charge_pairs():
    """point dipoles in the periodic band equal the limit of +-q pairs at separation d (error O(d^2))."""
    rng = np.random.default_rng(0)
    a = 12.0; l = 2.0; N = 6
    pos = rng.uniform(0, a, (N, 3)); cell = np.eye(3) * a
    q = rng.normal(size=N); q -= q.mean()
    mu = rng.normal(size=(N, 3)) * 0.5
    E_dip = float(EW.coulomb_band_pbc(jnp.asarray(q), jnp.asarray(mu), jnp.asarray(pos), jnp.asarray(cell), jnp.ones(N, bool), l, n_max=12, b_max=5.0))
    errs = []
    for d in (0.08, 0.04):
        # each site: the charge q at r, plus +m/d at r + d mu_hat/2 and -m/d at r - d mu_hat/2
        m = np.linalg.norm(mu, axis=1); u = mu / m[:, None]
        P = np.concatenate([pos, pos + 0.5 * d * u, pos - 0.5 * d * u])
        Q = np.concatenate([q, m / d, -m / d])
        E_q = float(EW.coulomb_band_pbc(jnp.asarray(Q), None, jnp.asarray(P), jnp.asarray(cell), jnp.ones(3 * N, bool), l, n_max=12, b_max=5.0))
        # remove the intra-site charge-charge terms that are not part of the dipole model: q*(+-m/d) pairs and the (+,-) pair,
        # all at distances d/2 and d, through the same erf kernel (and their images are already consistent)
        g = lambda r: math.erf(r / l) / r
        intra = 0.0
        for i in range(N):
            intra += q[i] * (m[i] / d) * g(0.5 * d) + q[i] * (-m[i] / d) * g(0.5 * d) - (m[i] / d) ** 2 * g(d)
        errs.append(abs((E_q - intra) - E_dip))
    assert errs[1] < 0.3 * errs[0] and errs[1] < 1e-3 * max(1.0, abs(E_dip)), (errs, E_dip)


def test_periodic_rungs_against_explicit_images():
    """the periodic M_6 / M_68 / M_16p bands equal the open-cluster bands summed over an explicit block of images
    (absolutely convergent for p = 6, 8, and for p = 1 with a neutral, dipole-free charge set); forces by autodiff
    match finite differences in the periodic model."""
    import jax
    from sseft import model as M, structure as S
    rng = np.random.default_rng(1)
    a = 14.0; N = 4
    pos = np.array([[2, 2, 2], [9, 2, 2], [2, 9, 2], [9, 9, 2.5]], float) + rng.normal(size=(N, 3)) * 0.2
    q = np.array([0.4, -0.4, -0.4, 0.4])                                  # neutral, zero dipole (quadrupole)
    om, w = kn.casimir_polder_grid(8)
    alpha = np.array([11.1 / (1 + (np.array(om) / 0.7) ** 2)] * N); alpha2 = alpha * 4.5
    st = S.Structure(pos, np.full(N, 18), cell=np.eye(3) * a, pbc=True, pinned={"q": q, "alpha": alpha, "alpha2": alpha2})
    b = {k: v[0] for k, v in S.pad_batch([st], n_freq=8).items()}
    base = M.with_rung(M.RUNGS["M_16p"], dispersion8=True, pin_alpha2=True, site_energies=False, periodic=True,
                       ewald_n_max=14, ewald_b_max=5.0, e0={**M.RUNGS["M_16p"].e0, "r_cut": 6.0})
    params = M.init_params(jax.random.PRNGKey(0), base)
    _, aux = M.energy_single(params, base, b)
    # explicit images with the open-cluster kernels: 1/2 sum_{i,j,L}' (...) over |L_a| <= n
    def image_sum(kernel, n, tail_p=None, C=None):
        """images inside the sphere |L| <= n a, plus the spherical tail 4 pi / ((p - 3) a^3 R^p-3) for p = 6, 8."""
        tot = 0.0
        Ls = _images(n) * a; Ls = Ls[np.linalg.norm(Ls, axis=1) <= n * a + 1e-9]
        for L in Ls:
            d = pos[:, None, :] - pos[None, :, :] + L; r = np.linalg.norm(d, axis=-1)
            m = r > 1e-9
            tot += 0.5 * np.sum(np.where(m, kernel(np.where(m, r, 1.0)), 0.0))
        if tail_p is not None:
            R = n * a
            tot += 0.5 * np.sum(C) * 4 * math.pi / ((tail_p - 3) * a ** 3 * R ** (tail_p - 3))
        return tot
    l = base.l1
    C6 = np.array(kn.c6_from_alpha(jnp.asarray(alpha)[:, None], jnp.asarray(alpha)[None, :], w))
    C8 = np.array(kn.c8_from_alpha(jnp.asarray(alpha)[:, None], jnp.asarray(alpha2)[:, None], jnp.asarray(alpha)[None, :], jnp.asarray(alpha2)[None, :], w))
    E6 = -image_sum(lambda r: C6 * np.array(kn.f6_long(jnp.asarray(r), l)), 8, 6, C6)
    E8 = -image_sum(lambda r: C8 * np.array(kn.f8_long(jnp.asarray(r), l)), 8, 8, C8)
    E1 = image_sum(lambda r: np.outer(q, q) * np.array(kn.g_long(jnp.asarray(r), l)), 12)
    assert abs(float(aux["E_disp"]) - E6) < 1e-5 * abs(E6), (float(aux["E_disp"]), E6)       # the explicit sum is limited by its discrete sphere boundary (~3e-6); the kernel itself is checked to 1e-8 above
    assert abs(float(aux["E_disp8"]) - E8) < 1e-7 * abs(E8), (float(aux["E_disp8"]), E8)
    assert abs(float(aux["E_coul"]) - E1) < 2e-3 * abs(E1), (float(aux["E_coul"]), E1)     # the quadrupole image sum converges only as 1/n^2; the kernel is checked to 1e-9 by the Madelung test
    # forces of the periodic model (E_0 by minimum image + Ewald bands) against finite differences
    rung = M.with_rung(base, site_energies=True)
    _, F = M.energy_forces_single(params, rung, b)
    def E_at(i, c, dx):
        return float(M.energy_single(params, rung, {**b, "positions": b["positions"].at[i, c].add(dx)})[0])
    h = 3e-5
    for (i, c) in [(0, 0), (3, 2)]:
        fd = -(-E_at(i, c, 2 * h) + 8 * E_at(i, c, h) - 8 * E_at(i, c, -h) + E_at(i, c, -2 * h)) / (12 * h)
        assert abs(fd - float(F[i, c])) < 1e-6 * abs(fd) + 1e-10, (i, c, fd, float(F[i, c]))


def test_band_field_tensors_pbc_against_images():
    """E, grad E, grad grad E at the atoms of a periodic quadrupolar charge set = the open-cluster tensors summed over
    an explicit block of images (absolutely convergent: the field of a quadrupole falls as r^-4)."""
    import jax
    from sseft import bandfields as BF
    a = 14.0; l = 2.0
    pos = np.array([[2, 2, 2], [9, 2, 2], [2, 9, 2], [9, 9, 2.5]], float) + np.array([[0.1, -0.2, 0.3], [0, 0.1, -0.1], [0.2, 0, 0], [-0.1, 0.1, 0.2]])
    q = np.array([0.4, -0.4, -0.4, 0.4])
    N = len(q)
    Ef, gE, ggE = BF.band_field_tensors_pbc(jnp.asarray(pos), jnp.asarray(q), jnp.ones(N, bool), jnp.asarray(np.eye(3) * a), l, 2, n_max=14, b_max=5.0)
    # explicit images: potential at x from every charge and image except the own charge at L = 0
    n = 8
    Ls = _images(n) * a; Ls = Ls[np.linalg.norm(Ls, axis=1) <= n * a + 1e-9]
    P = (pos[None, :, :] + Ls[:, None, :]).reshape(-1, 3); Q = np.tile(q, len(Ls))
    own = np.where(np.all(Ls == 0, axis=1))[0][0] * N                      # index of the L = 0 copy of atom 0
    def V(x, i):
        w = jnp.arange(len(Q)) != own + i
        d = jnp.where(w[:, None], x[None, :] - jnp.asarray(P), 1.0)        # the own charge gets a dummy vector before the sqrt
        rr = jnp.sqrt(jnp.sum(d * d, axis=-1))
        return jnp.sum(w * jnp.asarray(Q) * kn.g_long(rr, l))
    E = lambda x, i: -jax.grad(V)(x, i)
    # the charge set carries a small net dipole M, so the spherically summed field differs from the tin-foil
    # (Ewald) field by the uniform depolarisation field -(4 pi / 3V) M; the gradients are unaffected
    Mdip = q @ pos; depol = -(4 * math.pi / (3 * a ** 3)) * Mdip
    for i in range(N):
        x = jnp.asarray(pos[i])
        E_ref = np.array(E(x, i)) - depol; gE_ref = np.array(jax.jacfwd(E)(x, i)); ggE_ref = np.array(jax.jacfwd(jax.jacfwd(E))(x, i))
        # the explicit sums converge as 1/R (field of a quadrupole), 1/R^2, 1/R^3 for the three tensors
        assert np.allclose(np.array(Ef[i]), E_ref, rtol=3e-3, atol=2e-6), (i, np.array(Ef[i]), E_ref)
        assert np.allclose(np.array(gE[i]), gE_ref, rtol=1e-3, atol=2e-7), (i, np.array(gE[i]), gE_ref)
        assert np.allclose(np.array(ggE[i]), ggE_ref, rtol=3e-4, atol=2e-8), (i, np.array(ggE[i]), ggE_ref)


def test_periodic_band_field_rung_forces():
    """M_S2 under periodic boundary conditions (Ewald bands + Ewald band-field tensors + pair read-out): forces
    against finite differences."""
    import jax
    from sseft import model as M, structure as S
    a = 14.0; N = 4
    pos = np.array([[2, 2, 2], [9, 2, 2], [2, 9, 2], [9, 9, 2.5]], float) + np.array([[0.1, -0.2, 0.3], [0, 0.1, -0.1], [0.2, 0, 0], [-0.1, 0.1, 0.2]])
    q = np.array([0.4, -0.4, -0.4, 0.4])
    om, w = kn.casimir_polder_grid(8)
    alpha = np.array([11.1 / (1 + (np.array(om) / 0.7) ** 2)] * N)
    st = S.Structure(pos, np.full(N, 18), cell=np.eye(3) * a, pbc=True, pinned={"q": q, "alpha": alpha})
    b = {k: v[0] for k, v in S.pad_batch([st], n_freq=8).items()}
    rung = M.with_rung(M.RUNGS["M_S2"], periodic=True, ewald_n_max=10, e0={**M.RUNGS["M_S2"].e0, "r_cut": 6.0, "F": 8})
    params = M.init_params(jax.random.PRNGKey(0), rung)
    # switch the higher orders on (they start at zero) so that the test exercises them
    last = params["readout"]["theta"][-1]
    params["readout"]["theta"][-1] = {"w": last["w"] + 0.05 * jax.random.normal(jax.random.PRNGKey(1), last["w"].shape), "b": last["b"]}
    _, F = M.energy_forces_single(params, rung, b)
    def E_at(i, c, dx):
        return float(M.energy_single(params, rung, {**b, "positions": b["positions"].at[i, c].add(dx)})[0])
    h = 3e-5
    for (i, c) in [(0, 1), (2, 2)]:
        fd = -(-E_at(i, c, 2 * h) + 8 * E_at(i, c, h) - 8 * E_at(i, c, -h) + E_at(i, c, -2 * h)) / (12 * h)
        assert abs(fd - float(F[i, c])) < 1e-6 * abs(fd) + 1e-10, (i, c, fd, float(F[i, c]))


def test_analytic_band_pbc_against_images():
    """the periodic M_A kernel (polynomial in k^2 times the Gaussian envelope) equals the open-cluster family summed
    over images (the family is Gaussian in r: a 3 x 3 x 3 block of images is converged to machine precision)."""
    from sseft import bands as B
    rng = np.random.default_rng(3)
    a = 12.0; N = 5; lA = 3.0
    pos = rng.uniform(0, a, (N, 3)); s = rng.normal(size=N); coeffs = jnp.asarray([1.0, -0.5, 0.3, 0.1, -0.05])
    E_pbc = float(EW.analytic_band_pbc(jnp.asarray(s), coeffs, jnp.asarray(pos), jnp.asarray(np.eye(3) * a), jnp.ones(N, bool), lA, n_max=10, b_max=6.0))
    tot = 0.0
    for L in _images(3) * a:
        d = pos[:, None, :] - pos[None, :, :] + L; r = np.linalg.norm(d, axis=-1)
        m = r > 1e-9
        G = np.array(kn.analytic_family(jnp.asarray(np.where(m, r, 1.0)), lA, 4) / kn.analytic_family_scales(4))
        K = np.sum(G * np.array(coeffs), axis=-1)
        tot += 0.5 * np.sum(np.where(m, np.outer(s, s) * K, 0.0))
    assert abs(E_pbc - tot) < 1e-10 * max(1.0, abs(tot)), (E_pbc, tot)
