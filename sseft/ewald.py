"""
The IR bands under periodic boundary conditions: the long parts of the split kernels summed over all images in
reciprocal space (Ewald), the single-band scheme S = 1 with the band edge l as the Ewald width.

The model's band energy IS the long part of the kernel (erf-type), summed over every image; the short remainder
(erfc-type, within r_c) is E_0's.  So the periodic band is the reciprocal-space sum minus the self terms --
there is no real-space sum at all, and the sum converges like exp(-k^2 l^2 / 4).

    charges + dipoles (p = 1):   E = (2 pi / V) sum_{k != 0} e^{-k^2 l^2/4} / k^2 |S(k)|^2
                                     - sum_i [ q_i^2 / (sqrt(pi) l) + 2 |mu_i|^2 / (3 sqrt(pi) l^3) ],
                                 S(k) = sum_j (q_j + i k . mu_j) e^{i k . r_j};   tin-foil boundary conditions
                                 (the k = 0 term is dropped; a net charge is neutralised by a background and
                                 the corresponding constant is not included).
    dispersion (p = 6, 8):       E = - (1 / 2V) sum_k ghat_p(k) sum_kappa |S_kappa(k)|^2 + (1/2) sum_i C_ii g_p(0),
                                 with C_ij = sum_kappa B_i^kappa B_j^kappa the Casimir-Polder sum (one structure
                                 factor per imaginary frequency), ghat_p the 3-D Fourier transform of the long
                                 kernel [1 - e^{-x} P(x)] / r^p, x = r^2 / l^2, beta = 1 / l:
                                     ghat_6(k) = (pi^{3/2} beta^3 / 3) [ (1 - 2 b^2) e^{-b^2} + 2 sqrt(pi) b^3 erfc(b) ]
                                     ghat_8(k) = (pi^{3/2} beta^5 / 15) [ (1 - 2 b^2 / 3 + 4 b^4 / 3) e^{-b^2} - (4/3) sqrt(pi) b^5 erfc(b) ]
                                 b = k l / 2, and g_6(0) = beta^6 / 6, g_8(0) = beta^8 / 24.  The k = 0 term is
                                 finite and included (there is no neutrality for dispersion).
The reciprocal lattice is a fixed integer grid |n_a| <= n_max (static, for jit) masked by |k| <= k_max = 2 b_max / l.
Validated in tests/test_ewald.py against the Madelung constant of NaCl (p = 1), against direct lattice sums
(p = 6, 8), and the dipole terms against the point-charge limit.
"""
import math
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc


def k_vectors(cell, n_max):
    """reciprocal vectors k = 2 pi n B^-T for integers |n_a| <= n_max (the n = 0 entry included), shape (M, 3)."""
    n = jnp.arange(-n_max, n_max + 1)
    N = jnp.stack(jnp.meshgrid(n, n, n, indexing="ij"), axis=-1).reshape(-1, 3).astype(cell.dtype)
    recip = 2.0 * math.pi * jnp.linalg.inv(cell).T          # rows: reciprocal lattice vectors
    return N @ recip


def _volume(cell):
    return jnp.abs(jnp.linalg.det(cell))


def coulomb_band_pbc(q, mu, positions, cell, mask, l, n_max=8, b_max=4.0):
    """periodic p = 1 band on charges q (N,) and optional dipoles mu (N,3); positions (N,3) bohr, cell rows."""
    k = k_vectors(cell, n_max)                                  # (M,3)
    k2 = jnp.sum(k * k, axis=-1)
    kmax2 = (2.0 * b_max / l) ** 2
    live = (k2 > 1e-12) & (k2 <= kmax2)
    k2s = jnp.where(live, k2, 1.0)
    phase = k @ positions.T                                     # (M,N)
    qm = q * mask
    amp_re = qm[None, :] * jnp.cos(phase); amp_im = qm[None, :] * jnp.sin(phase)
    if mu is not None:
        kmu = k @ (mu * mask[:, None]).T                        # (M,N): k . mu_j
        # (q_j + i k.mu_j) e^{i phase} = (q cos - k.mu sin) + i (q sin + k.mu cos)
        amp_re = amp_re - kmu * jnp.sin(phase); amp_im = amp_im + kmu * jnp.cos(phase)
    S2 = jnp.sum(amp_re, axis=1) ** 2 + jnp.sum(amp_im, axis=1) ** 2
    V = _volume(cell)
    E = (2.0 * math.pi / V) * jnp.sum(jnp.where(live, jnp.exp(-0.25 * k2s * l * l) / k2s * S2, 0.0))
    E = E - jnp.sum(qm * qm) / (math.sqrt(math.pi) * l)
    if mu is not None:
        E = E - 2.0 * jnp.sum(jnp.sum(mu * mu, axis=-1) * mask) / (3.0 * math.sqrt(math.pi) * l ** 3)
    return E


def ghat6(k2, l):
    b = 0.5 * jnp.sqrt(k2) * l; beta = 1.0 / l
    return (math.pi ** 1.5 * beta ** 3 / 3.0) * ((1.0 - 2.0 * b * b) * jnp.exp(-b * b) + 2.0 * math.sqrt(math.pi) * b ** 3 * erfc(b))


def ghat8(k2, l):
    b = 0.5 * jnp.sqrt(k2) * l; beta = 1.0 / l
    return (math.pi ** 1.5 * beta ** 5 / 15.0) * ((1.0 - 2.0 * b * b / 3.0 + 4.0 * b ** 4 / 3.0) * jnp.exp(-b * b)
                                                  - (4.0 / 3.0) * math.sqrt(math.pi) * b ** 5 * erfc(b))


def _dispersion_pbc(Bi, Bj, positions, cell, mask, l, ghat, g0, n_max, b_max):
    """E = -(1/2V) sum_k ghat(k) sum_kappa Re[S_i^kappa(k) conj(S_j^kappa(k))] + (1/2) sum_i C_ii g0, with
    C_ij = sum_kappa (Bi_i^kappa Bj_j^kappa + Bj_i^kappa Bi_j^kappa) / 2 symmetrised: Bi, Bj (N,K)."""
    k = k_vectors(cell, n_max)
    k2 = jnp.sum(k * k, axis=-1)
    live = k2 <= (2.0 * b_max / l) ** 2
    phase = k @ positions.T                                      # (M,N)
    c, s = jnp.cos(phase), jnp.sin(phase)
    Bi = Bi * mask[:, None]; Bj = Bj * mask[:, None]
    Si_re = c @ Bi; Si_im = s @ Bi                               # (M,K)
    Sj_re = c @ Bj; Sj_im = s @ Bj
    SS = jnp.sum(Si_re * Sj_re + Si_im * Sj_im, axis=-1)         # sum_kappa Re[S_i conj S_j]  (M,)
    V = _volume(cell)
    E = -(0.5 / V) * jnp.sum(jnp.where(live, ghat(k2, l) * SS, 0.0))
    Cii = jnp.sum(Bi * Bj, axis=-1)
    return E + 0.5 * jnp.sum(Cii * mask) * g0


def dispersion_band_pbc(alpha, w, positions, cell, mask, l, n_max=8, b_max=4.0):
    """periodic p = 6 band: C6_ij = (3/pi) sum_k w_k alpha_i alpha_j = sum_k B_i B_j, B = sqrt(3 w / pi) alpha."""
    B = jnp.sqrt(3.0 * w / math.pi)[None, :] * alpha
    return _dispersion_pbc(B, B, positions, cell, mask, l, ghat6, (1.0 / l) ** 6 / 6.0, n_max, b_max)


def dispersion8_band_pbc(alpha1, alpha2, w, positions, cell, mask, l, n_max=8, b_max=4.0):
    """periodic p = 8 band: C8_ij = (15/2pi) sum_k w_k [a1_i a2_j + a2_i a1_j] = sum_k (Bi_i Bj_j + Bj_i Bi_j)
    with Bi = sqrt(15 w / 2pi) alpha1, Bj = sqrt(15 w / 2pi) alpha2; the symmetrised product below."""
    f = jnp.sqrt(15.0 * w / (2.0 * math.pi))[None, :]
    B1 = f * alpha1; B2 = f * alpha2
    # sum_kappa Re[S1 conj S2] + Re[S2 conj S1] = 2 Re[S1 conj S2]; C_ii = 2 B1 B2
    return 2.0 * _dispersion_pbc(B1, B2, positions, cell, mask, l, ghat8, (1.0 / l) ** 8 / 24.0, n_max, b_max)


def analytic_band_pbc(s, coeffs, positions, cell, mask, lA, n_max=8, b_max=4.0):
    """periodic rung M_A: K(r) = sum_n c_n G_n(r) / rms_n with G_n = (-lA^2 lap)^n exp(-r^2/lA^2); in reciprocal space
    Ghat_n(k) = (k^2 lA^2)^n pi^{3/2} lA^3 exp(-k^2 lA^2 / 4), so
        E = (1 / 2V) sum_k Khat(k) |S(k)|^2 - (1/2) sum_i s_i^2 K(0),   S(k) = sum_j s_j e^{i k . r_j},
    with K(0) = sum_n c_n P_n(0) / rms_n (the polynomials' constant terms) and the k = 0 term included."""
    from . import kernels as kn
    NA = coeffs.shape[0] - 1
    scales = kn.analytic_family_scales(NA)
    polys = kn.gaussian_family_polys(NA)
    K0 = sum(float(P[0]) * coeffs[n] / scales[n] for n, P in enumerate(polys))
    k = k_vectors(cell, n_max)
    k2 = jnp.sum(k * k, axis=-1)
    live = k2 <= (2.0 * b_max / lA) ** 2
    x = k2 * lA * lA
    Khat = math.pi ** 1.5 * lA ** 3 * jnp.exp(-0.25 * x) * sum(coeffs[n] / scales[n] * x ** n for n in range(NA + 1))
    phase = k @ positions.T
    sm = s * mask
    S2 = jnp.sum(sm[None, :] * jnp.cos(phase), axis=1) ** 2 + jnp.sum(sm[None, :] * jnp.sin(phase), axis=1) ** 2
    V = _volume(cell)
    return (0.5 / V) * jnp.sum(jnp.where(live, Khat * S2, 0.0)) - 0.5 * jnp.sum(sm * sm) * K0
