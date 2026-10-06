"""
The split kernels of the proposal's section 1.3 and the Casimir-Polder quadrature.

    r^-p = r^-p Q(p/2, r^2/l^2) + r^-p P(p/2, r^2/l^2)       (short + long; P, Q regularised incomplete gammas)

The long part is the IR band kernel for the single-band scheme (S = 1); the short part is Gaussian-decaying
beyond ~3 l and is E_0's business.  p = 1: erf(r/l)/r (Ewald); p = 6: [1 - e^-x (1 + x + x^2/2)]/r^6.
Dipole kernels are the first and second derivatives of the p = 1 kernel.  Everything is written so that
jax can differentiate through it (forces, band fields).
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf, gammainc

SQRT_PI = math.sqrt(math.pi)


# ---------------------------------------------------------------- p = 1 and its derivatives ------------------
def g_long(r, l):
    """erf(r/l)/r"""
    return erf(r / l) / r


def dg_long(r, l):
    """d/dr erf(r/l)/r"""
    return (2.0 / (l * SQRT_PI)) * jnp.exp(-(r / l) ** 2) / r - erf(r / l) / r ** 2


def d2g_long(r, l):
    e = jnp.exp(-(r / l) ** 2)
    return -(2.0 / (l * SQRT_PI)) * e * (2.0 / l ** 2 + 2.0 / r ** 2) + 2.0 * erf(r / l) / r ** 3


def g_short(r, l):
    """erfc(r/l)/r = 1/r - g_long"""
    return 1.0 / r - g_long(r, l)


# ---------------------------------------------------------------- general p ----------------------------------
def k_long(r, l, p):
    """r^-p P(p/2, r^2/l^2); finite at r -> 0 (-> l^-p / Gamma(p/2 + 1))."""
    x = (r / l) ** 2
    return gammainc(p / 2.0, x) / r ** p


def k_short(r, l, p):
    return (1.0 - gammainc(p / 2.0, (r / l) ** 2)) / r ** p


def f6_long(r, l):
    """the p = 6 long kernel in closed form (dispersion PME split): [1 - e^-x (1 + x + x^2/2)] / r^6."""
    x = (r / l) ** 2
    small = x ** 3 / 6.0 - x ** 4 / 8.0 + x ** 5 / 20.0 - x ** 6 / 72.0
    P = jnp.where(x < 1e-2, small, 1.0 - jnp.exp(-x) * (1.0 + x + 0.5 * x * x))
    return P / r ** 6


def f8_long(r, l):
    """the p = 8 long kernel: [1 - e^-x (1 + x + x^2/2 + x^3/6)] / r^8."""
    x = (r / l) ** 2
    small = x ** 4 / 24.0 - x ** 5 / 30.0 + x ** 6 / 72.0 - x ** 7 / 252.0
    P = jnp.where(x < 1e-2, small, 1.0 - jnp.exp(-x) * (1.0 + x + 0.5 * x * x + x ** 3 / 6.0))
    return P / r ** 8


# ---------------------------------------------------------------- analytic-kernel family (rung M_A) -----------
def gaussian_family_polys(N):
    """coefficients (in x = r^2/l^2) of P_n with (-l^2 lap)^n exp(-x) = P_n(x) exp(-x) in 3-D, n = 0..N."""
    from fractions import Fraction
    polys = [[Fraction(1)]]
    for _ in range(N):
        P = polys[-1]; deg = len(P) - 1
        dP = [P[k] * k for k in range(1, deg + 1)] or [Fraction(0)]
        d2P = [dP[k] * k for k in range(1, len(dP))] or [Fraction(0)]
        new = [Fraction(0)] * (deg + 2)
        for k, c in enumerate(d2P): new[k + 1] -= 4 * c
        for k, c in enumerate(dP): new[k + 1] += 8 * c; new[k] -= 6 * c
        for k, c in enumerate(P): new[k] += 6 * c; new[k + 1] -= 4 * c
        polys.append(new)
    return [np.array([float(c) for c in P]) for P in polys]


def analytic_family(r, l, N):
    """G_n(r) = P_n(r^2/l^2) exp(-r^2/l^2), n = 0..N, stacked on the last axis: the real-space pair functions of
    the kernel class sum_n c_n q^{2n} exp(-q^2 l^2/4) (up to constants), p_min = infinity."""
    x = (r / l) ** 2
    polys = gaussian_family_polys(N)
    e = jnp.exp(-x)
    cols = []
    for P in polys:
        val = jnp.zeros_like(x)
        for k in range(len(P) - 1, -1, -1):
            val = val * x + P[k]
        cols.append(val * e)
    return jnp.stack(cols, axis=-1)


def analytic_family_scales(N):
    """rms of G_n over r in [l/2, 5l/2] (a fixed diagonal preconditioner: the raw family's columns differ by
    10^4 in scale and its window Gram matrix has condition number ~10^10, which no first-order optimiser
    resolves; normalised it is ~10^5, which L-BFGS does)."""
    x = np.linspace(0.25, 6.25, 400)
    out = []
    for P in gaussian_family_polys(N):
        val = np.polyval(P[::-1], x) * np.exp(-x)
        out.append(np.sqrt(np.mean(val ** 2)))
    return jnp.asarray(out)


# ---------------------------------------------------------------- Casimir-Polder ------------------------------
def casimir_polder_grid(K=8, omega0=0.3):
    """imaginary frequencies and weights for C6_ij = (3/pi) sum_k w_k alpha_i(i w_k) alpha_j(i w_k):
    Gauss-Legendre on t in [-1, 1] mapped to w = omega0 (1 - t)/(1 + t)."""
    t, wgl = np.polynomial.legendre.leggauss(K)
    omega = omega0 * (1.0 - t) / (1.0 + t)
    w = wgl * 2.0 * omega0 / (1.0 + t) ** 2
    return jnp.asarray(omega), jnp.asarray(w)


def c6_from_alpha(alpha_i, alpha_j, w):
    """alpha_* of shape (..., K); returns C6 = (3/pi) sum_k w_k alpha_i alpha_j."""
    return (3.0 / math.pi) * jnp.sum(w * alpha_i * alpha_j, axis=-1)


def c8_from_alpha(alpha1_i, alpha2_i, alpha1_j, alpha2_j, w):
    """dipole-quadrupole dispersion: C8 = (15/pi) sum_k w_k [alpha1_i alpha2_j + alpha2_i alpha1_j] / 2
    (Casimir-Polder: C8_AB = (15/2 pi) int [a1_A a2_B + a2_A a1_B] d omega)."""
    return (15.0 / (2.0 * math.pi)) * jnp.sum(w * (alpha1_i * alpha2_j + alpha2_i * alpha1_j), axis=-1)


def london_c6(alpha0_a, omega_a, alpha0_b, omega_b):
    """closed form for one-oscillator alpha(i w) = alpha0/(1 + w^2/omega^2): C6 = (3/2) a0_a a0_b wa wb/(wa + wb)."""
    return 1.5 * alpha0_a * alpha0_b * omega_a * omega_b / (omega_a + omega_b)
