"""
Synthetic truths with exact answers, used as regression tests of the harness.

X2-B dielectric toy (claude/x2-protocol.md section 3, truth T3): sites with a short-range, non-local
response to the band field above l_1,
    E_band = -1/2 F^T chi_3 F,    chi_3 = a0 (1 + a0 H_sr + a0^2 H_sr^2),    H_sr = grad grad erfc(r/a)/r,
    F_i = -grad V(r_i),  V = sum_{k != i} q_k erf(|r - r_k|/l_1)/|r - r_k|.
The per-atom band-field model with pinned charges must recover it at order 0 to ~1e-3 and better at order 2
(x2b-results.md).  Units are arbitrary (the toy is unitless); the harness treats them as atomic units.
"""
import numpy as np
from scipy.special import erf
from .structure import Structure

SQPI = np.sqrt(np.pi)


def _g1(r, s): return (2 / (s * SQPI)) * np.exp(-(r / s) ** 2) / r - erf(r / s) / r ** 2
def _g2(r, s):
    e = np.exp(-(r / s) ** 2)
    return -(2 / (s * SQPI)) * e * (2 / s ** 2 + 2 / r ** 2) + 2 * erf(r / s) / r ** 3


def cluster(N, density, shape, rng, dmin=1.0):
    vol = N / density
    if shape == "sphere":
        Rc = (3 * vol / (4 * np.pi)) ** (1 / 3); semi = np.array([Rc] * 3)
    elif shape == "oblate":
        c = (3 * vol / (4 * np.pi * 4)) ** (1 / 3); semi = np.array([2 * c, 2 * c, c])
    else:
        c = (3 * vol / (4 * np.pi * 2)) ** (1 / 3); semi = np.array([c, c, 2 * c])
    pts = []
    for _ in range(400000):
        if len(pts) == N: break
        x = rng.uniform(-1, 1, 3) * semi
        if np.sum((x / semi) ** 2) > 1: continue
        if pts and np.min(np.linalg.norm(np.array(pts) - x, axis=1)) < dmin: continue
        pts.append(x)
    return np.array(pts)


def charge_patterns(R, rng, n_random=3, n_mod=7):
    N = len(R); pats = []
    for _ in range(n_random):
        q = rng.uniform(-0.5, 0.5, N); q -= q.mean(); pats.append(q)
    ext = (R.max(axis=0) - R.min(axis=0)).mean()
    for _ in range(n_mod):
        k = rng.normal(size=3); k /= np.linalg.norm(k)
        lam = rng.uniform(0.6, 2.0) * ext
        q = 0.5 * np.cos(2 * np.pi * (R @ k) / lam + rng.uniform(0, 2 * np.pi)) + 0.1 * rng.normal(size=N)
        q -= q.mean(); pats.append(q)
    return pats


def t3_truth(R, q, a, alpha0, l1):
    """E_band of the Neumann-3 truth for one cluster and charge pattern."""
    N = len(R)
    D = R[:, None, :] - R[None, :, :]
    r = np.linalg.norm(D, axis=-1); np.fill_diagonal(r, np.inf)
    rhat = D / np.where(np.isfinite(r), r, 1.0)[:, :, None]
    fin = np.isfinite(r); rr = np.where(fin, r, 1.0)
    f1 = np.where(fin, -1 / rr ** 2 - _g1(rr, a), 0.0); f2 = np.where(fin, 2 / rr ** 3 - _g2(rr, a), 0.0)
    outer = rhat[:, :, :, None] * rhat[:, :, None, :]
    Hb = (f1 / rr)[:, :, None, None] * (np.eye(3)[None, None] - outer) + f2[:, :, None, None] * outer
    H = np.transpose(Hb, (0, 2, 1, 3)).reshape(3 * N, 3 * N)
    # band field above l1 at the sites, own charge excluded
    g1 = np.where(fin, _g1(rr, l1), 0.0)
    F = -np.einsum("ik,ika->ia", q[None, :] * g1, rhat).reshape(-1)
    HF = H @ F; HHF = H @ HF
    return -0.5 * alpha0 * (F @ F + alpha0 * F @ HF + alpha0 ** 2 * HF @ HF)


def x2b_dataset(n_clusters=40, N=(40, 60), densities=(0.15, 0.22, 0.30), a=0.5, alpha0=0.8, l1=2.0, seed=0,
                n_random=3, n_mod=7, Z=18):
    """structures with pinned charges and energies E_band (T3); all sites the same species (charge-blind E_0)."""
    rng = np.random.default_rng(seed)
    out = []
    combos = [(n, d, s) for n in N for d in densities for s in ("sphere", "oblate")]
    for k in range(n_clusters):
        n, d, s = combos[k % len(combos)]
        R = cluster(n, d, s, rng)
        for q in charge_patterns(R, rng, n_random, n_mod):
            E = t3_truth(R, q, a, alpha0, l1)
            out.append(Structure(R.copy(), np.full(n, Z), energy=E, pinned={"q": q}, info={"cluster": k}))
    return out
