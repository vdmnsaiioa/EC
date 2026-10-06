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


def taylor_class_features(R, q, a, l1, order=2):
    """The exact Taylor class of the T3 truth at a given order (x2-protocol.md 1.3): expand the field at the
    neighbours about each site, F_j = sum_n (d_ij . grad)^n F_i / n!, in both the pair term F H F and the
    chain term F H H F.  Returns the per-structure features (summed over atoms), one per (term, order):
        n = 0: F.B.F, F.(HB).F          B_i = sum_j H_ij,  (HB)_i = sum_j H_ij B_j
        n = 1: F . [sum_j H_ij d_ij^c] . d_c F,   and the chain analogue with d_ik
        n = 2: F . [sum_j H_ij d_ij^c d_ij^d] . d_c d_d F / 2,   and the chain analogue
    plus F.F (the bare alpha0 term).  An OLS fit of these to E_band gives the ceiling of the order-k class on
    the data set at hand (the ceiling depends on the clusters: it is a property of the design, not a constant)."""
    N = len(R)
    Dm = R[None, :, :] - R[:, None, :]                                        # d_ij = r_j - r_i
    r = np.linalg.norm(Dm, axis=-1); np.fill_diagonal(r, np.inf)
    fin = np.isfinite(r); rr = np.where(fin, r, 1.0); rhat = -Dm / rr[:, :, None]   # (r_i - r_j)/r, as in t3_truth
    f1 = np.where(fin, -1 / rr ** 2 - _g1(rr, a), 0.0); f2 = np.where(fin, 2 / rr ** 3 - _g2(rr, a), 0.0)
    outer = rhat[:, :, :, None] * rhat[:, :, None, :]
    H = (f1 / rr)[:, :, None, None] * (np.eye(3)[None, None] - outer) + f2[:, :, None, None] * outer   # (i,j,a,b)
    # band field and its gradients at the sites (own charge excluded): F = -grad V, V = sum_k q_k g(|r - r_k|; l1)
    g1 = np.where(fin, _g1(rr, l1), 0.0)
    F = -np.einsum("ik,ika->ia", q[None, :] * g1, rhat)
    # grad F_i (a,c) = -sum_k q_k d_a d_c g = -sum_k q_k [g'' rhat_a rhat_c + (g'/r)(delta_ac - rhat_a rhat_c)]
    g2 = np.where(fin, _g2(rr, l1), 0.0)
    gF = -np.einsum("ik,ikac->iac", q[None, :] * g2, outer) - np.einsum("ik,ikac->iac", q[None, :] * np.where(fin, g1 / rr, 0.0), np.eye(3)[None, None] - outer)
    # grad grad F_i (a,c,d) by finite differences of gF in the probe point (cheap and exact enough for a ceiling)
    def gF_at(x, i):
        d = R - x[None, :]; rr_ = np.linalg.norm(d, axis=-1); m = np.ones(N, bool); m[i] = False
        rr_ = np.where(m, rr_, 1.0); rh = -d / rr_[:, None]; ou = rh[:, :, None] * rh[:, None, :]
        G1 = np.where(m, _g1(rr_, l1), 0.0); G2 = np.where(m, _g2(rr_, l1), 0.0)
        return -np.einsum("k,kac->ac", q * G2, ou) - np.einsum("k,kac->ac", q * np.where(m, G1 / rr_, 0.0), np.eye(3)[None] - ou)
    h = 1e-4
    ggF = np.zeros((N, 3, 3, 3))
    for i in range(N):
        for dd in range(3):
            e = np.zeros(3); e[dd] = h
            ggF[i, :, :, dd] = (gF_at(R[i] + e, i) - gF_at(R[i] - e, i)) / (2 * h)
    B = H.sum(axis=1)                                                          # (i,a,b)
    feats = [np.sum(F * F), np.einsum("ia,iab,ib->", F, B, F), np.einsum("ia,ijab,jbc,ic->", F, H, B, F)]
    if order >= 1:
        T1 = np.einsum("ijab,ijc->iabc", H, Dm)                                # sum_j H_ij d_ij^c
        C1 = np.einsum("ijab,jkbe,ikc->iaec", H, H, Dm)                        # sum_jk H_ij H_jk d_ik^c
        feats += [np.einsum("ia,iabc,ibc->", F, T1, gF), np.einsum("ia,iabc,ibc->", F, C1, gF)]
    if order >= 2:
        T2 = np.einsum("ijab,ijc,ijd->iabcd", H, Dm, Dm)
        C2 = np.einsum("ijab,jkbe,ikc,ikd->iaecd", H, H, Dm, Dm)
        feats += [0.5 * np.einsum("ia,iabcd,ibcd->", F, T2, ggF), 0.5 * np.einsum("ia,iabcd,ibcd->", F, C2, ggF)]
    return np.array(feats)


def taylor_class_ceiling(train, test, a=0.5, l1=2.0, orders=(0, 1, 2)):
    """OLS fit of the exact Taylor class on `train`, relative rms on train and test, per order."""
    ytr = np.array([s.energy for s in train]); yte = np.array([s.energy for s in test])
    Xtr = np.array([taylor_class_features(s.positions, s.pinned["q"], a, l1, max(orders)) for s in train])
    Xte = np.array([taylor_class_features(s.positions, s.pinned["q"], a, l1, max(orders)) for s in test])
    out = {}
    for k in orders:
        n = 3 + 2 * k
        c = np.linalg.lstsq(Xtr[:, :n], ytr, rcond=None)[0]
        out[k] = dict(train=float(np.sqrt(np.mean((Xtr[:, :n] @ c - ytr) ** 2)) / np.sqrt(np.mean(ytr ** 2))),
                      test=float(np.sqrt(np.mean((Xte[:, :n] @ c - yte) ** 2)) / np.sqrt(np.mean(yte ** 2))), coef=c.tolist())
    return out


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
