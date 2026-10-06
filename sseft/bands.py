"""
IR band energies for open systems (direct pair sums) in the single-band scheme S = 1: the long part of each
channel's kernel at the band edge l_1 acts on the learned sources; the short remainders are E_0's.

    charges / dipoles :  g(r) = erf(r/l)/r and its derivatives (the Ewald real-space split, here in full)
    dispersion        :  -1/2 sum_{i!=j} C6_ij [1 - e^-x (1 + x + x^2/2)]/r^6,   x = r^2/l^2
    analytic rung M_A :  1/2 sum_{i!=j} s_i s_j sum_n c_n G_n(r; l_A)  -- a learned kernel analytic in q^2

Pair energy of two multipole sites with the smoothed kernel (each pair once, rhat_ij = (r_i - r_j)/r):
    E_ij = q_i q_j g + (q_j mu_i - q_i mu_j) . rhat_ij g' - mu_i . H . mu_j,   H = g'' rr + (g'/r)(1 - rr).
Periodic (Ewald) versions are the week-2 item.
"""
import jax.numpy as jnp
from . import kernels as kn


def coulomb_band(q, mu, D, r, pmask, l):
    """q (N,), mu (N,3) or None, D (N,N,3) = r_i - r_j, r (N,N), pmask (N,N)."""
    g = kn.g_long(r, l) * pmask
    E = 0.5 * jnp.sum(q[:, None] * q[None, :] * g)
    if mu is not None:
        rhat = D / r[..., None]
        dg = kn.dg_long(r, l) * pmask
        d2g = kn.d2g_long(r, l) * pmask
        cross = (q[None, :, None] * mu[:, None, :] - q[:, None, None] * mu[None, :, :])      # (i,j,a): q_j mu_i - q_i mu_j
        E = E + 0.5 * jnp.sum(jnp.sum(cross * rhat, axis=-1) * dg)
        mr_i = jnp.sum(mu[:, None, :] * rhat, axis=-1)                                      # mu_i . rhat_ij
        mr_j = jnp.sum(mu[None, :, :] * rhat, axis=-1)                                      # mu_j . rhat_ij
        mm = jnp.sum(mu[:, None, :] * mu[None, :, :], axis=-1)
        muHmu = d2g * mr_i * mr_j + (dg / r) * (mm - mr_i * mr_j)
        E = E - 0.5 * jnp.sum(muHmu * pmask)
    return E


def dispersion_band(alpha, w, r, pmask, l):
    """alpha (N,K) at the Casimir-Polder frequencies with weights w (K,)."""
    C6 = kn.c6_from_alpha(alpha[:, None, :], alpha[None, :, :], w)                           # (N,N)
    return -0.5 * jnp.sum(C6 * kn.f6_long(r, l) * pmask)


def analytic_band(s, coeffs, r, pmask, lA):
    """rung M_A: learned scalar sources s (N,) and kernel coefficients coeffs (N_A + 1,) of the normalised
    family G_n / rms(G_n) (kernels.analytic_family_scales)."""
    NA = coeffs.shape[0] - 1
    G = kn.analytic_family(r, lA, NA) / kn.analytic_family_scales(NA)                          # (N,N,N_A+1)
    K = jnp.sum(G * coeffs, axis=-1) * pmask
    return 0.5 * jnp.sum(s[:, None] * s[None, :] * K)
