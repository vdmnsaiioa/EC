"""
Where E3-QP's band-field rungs lose the far-field many-body energy, computed from the truth before any new training:

    python scratch/e3qp_decomposition.py [external/qaqua/build_pol]

TTM3-F's induction is re-evaluated here from its sites (the M-site charge and polarisability, the H charges, the Thole
smearing of ttm3f_mod.f90) and checked against the library; then, on the test probes and the test hexamers of
configs/water_qaqua_pol.yaml (same seeds):

 (1) the non-additive induction split into its first-order part (the cross terms of the permanent fields, -alpha E_j.E_k)
     and the mutual remainder (the induced dipoles' fields: alpha^2 T and beyond), and piece (2) of the rehearsal (the
     core's own self-consistent dipoles against the probe's permanent field);
 (2) how well first-order forms with the harness's band field reproduce the truth's first-order part: the pinned
     dipole-surface charges q3 on the atoms (E3-QP), and TTM3-F's multipoles on the atoms (charges -1.2254 / +0.6127 on
     O / H and an O dipole q_M (r_M - r_O), which is the M-site charge to dipole order about O);
 (3) the pinned self-consistent response a harness rung could carry: TTM3-F's alpha at O, the band field of the TTM3-F
     multipoles and the dipole-dipole tensor of the band kernel erf(r/l)/r between the molecules, solved exactly.
"""
import math, os, sys, time
import numpy as np
from scipy.special import erf

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, qaqua as QA
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
truth = W.qaqua_pol_truth(root)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
t_start = time.time()
A3 = QA.AUANG ** 3
ALPHA = 1.444 / A3                                     # TTM3-F's polarisability on the M-site, bohr^3
POLFAC_H, POLFAC_M, A_SMEAR = 0.496 / A3, 0.837 / A3, 0.175      # its Thole factors (bohr^3) and the smearing a


# ----------------------------------------------------------------------- TTM3-F's induction from its sites --------
def ttm_sites(pos_ang):
    """the M-sites (n, 3, bohr) and the site charges (n, 3: H1, H2, M) from the library."""
    t = truth.qa.ttm(ang_to_bohr(pos_ang))
    return t["msites"], t["site_charges"], t


def _thole(r, pol12, a):
    e = np.exp(-a * (r / pol12 ** (1 / 6)) ** 3)
    ts1 = (1 - e) / r ** 3
    ts2 = (ts1 - e * a / pol12 ** 0.5) / r ** 2
    return ts1, ts2


def ttm_induction(pos_ang, mols=None, order=None):
    """TTM3-F's induction of the molecules `mols` (default all) at pos_ang (O, H, H per molecule, A) from its sites:
    the permanent fields at the M-sites (charge-dipole Thole smearing, other molecules only), the M-M dipole tensor,
    and the induction energy -1/2 mu.E_q with mu = (1/alpha - T)^-1 E_q (order None), or the Neumann series
    truncated after `order` terms (1: first order).  Returns (energy hartree, mu (m, 3), E_q (m, 3))."""
    pos_ang = np.asarray(pos_ang, float)
    n = len(pos_ang) // 3
    mols = list(range(n)) if mols is None else list(mols)
    sub = np.concatenate([pos_ang[3 * m:3 * m + 3] for m in mols])
    RM, qs, _ = ttm_sites(sub)
    X = ang_to_bohr(sub); m_ = len(mols)
    Eq = np.zeros((m_, 3)); T = np.zeros((3 * m_, 3 * m_))
    for i in range(m_):
        for j in range(m_):
            if i == j:
                continue
            for site, q, pol in ((X[3 * j + 1], qs[j, 0], POLFAC_H), (X[3 * j + 2], qs[j, 1], POLFAC_H), (RM[j], qs[j, 2], POLFAC_M)):
                R = RM[i] - site; r = np.linalg.norm(R)
                ts1, _ = _thole(r, pol * POLFAC_M, A_SMEAR)
                Eq[i] += ts1 * q * R
            R = RM[i] - RM[j]; r = np.linalg.norm(R)
            ts1, ts2 = _thole(r, POLFAC_M ** 2, A_SMEAR)
            T[3 * i:3 * i + 3, 3 * j:3 * j + 3] = 3 * ts2 * np.outer(R, R) - ts1 * np.eye(3)
    e = Eq.reshape(-1)
    if order is None:
        mu = np.linalg.solve(np.eye(3 * m_) / ALPHA - T, e)
    else:
        mu = np.zeros_like(e); term = ALPHA * e
        for _ in range(order):
            mu = mu + term; term = ALPHA * (T @ term)
    return -0.5 * float(mu @ e), mu.reshape(m_, 3), Eq


# ----------------------------------------------------------------------- the harness's band field ------------------
def band_kernel(r, l):
    """g = erf(r/l)/r and its radial derivatives g', g''."""
    g = erf(r / l) / r
    gp = (2 / math.sqrt(math.pi) * np.exp(-(r / l) ** 2) / l - g) / r
    gpp = -2 / math.sqrt(math.pi) * np.exp(-(r / l) ** 2) * 2 * r / l ** 3 / r - 2 * gp / r
    return g, gp, gpp


def hessian_g(R, l):
    r = np.linalg.norm(R); n = R / r
    _, gp, gpp = band_kernel(r, l)
    return gpp * np.outer(n, n) + gp / r * (np.eye(3) - np.outer(n, n))


def band_field(X, q, mu, frag, l, at):
    """the band field (erf(r/l)/r kernel) at the points X[at] of the charges q and point dipoles mu of the OTHER
    fragments; X bohr (N, 3).  Returns (len(at), 3)."""
    out = np.zeros((len(at), 3))
    for a, i in enumerate(at):
        for j in range(len(X)):
            if frag[j] == frag[i]:
                continue
            R = X[i] - X[j]; r = np.linalg.norm(R)
            _, gp, _ = band_kernel(r, l)
            out[a] += -q[j] * gp * R / r                   # -grad of q g
            if mu is not None:
                out[a] += hessian_g(R, l) @ mu[j]          # grad (mu . grad g)
    return out


def ttm_multipoles(pos_ang):
    """TTM3-F's site charges as atom-centred sources: O carries the M-site charge, each H its own, and O a dipole
    q_M (r_M - r_O) (the M-site to dipole order about O); bohr, e, e bohr."""
    RM, qs, _ = ttm_sites(pos_ang)
    X = ang_to_bohr(np.asarray(pos_ang, float)); n = len(X) // 3
    q = np.zeros(3 * n); mu = np.zeros((3 * n, 3))
    for m in range(n):
        q[3 * m:3 * m + 3] = [qs[m, 2], qs[m, 0], qs[m, 1]]
        mu[3 * m] = qs[m, 2] * (RM[m] - X[3 * m])
    return X, q, mu


def harness_induction(pos_ang, mols, sources, l, scf=True, site="O", alpha=ALPHA):
    """a pinned response a harness rung could carry: polarisability alpha at O (or at the M-site), the band field at
    l of the other molecules' sources, and (scf) the induced dipoles' mutual fields through the band kernel's dipole
    tensor between molecules.  sources: 'q3' (the dipole-surface charges) or 'ttm' (TTM3-F's multipoles)."""
    pos_ang = np.asarray(pos_ang, float)
    sub = np.concatenate([pos_ang[3 * m:3 * m + 3] for m in mols]); m_ = len(mols)
    if sources == "q3":
        X = ang_to_bohr(sub); q = truth.molecule_charges(sub); mu = None
    else:
        X, q, mu = ttm_multipoles(sub)
    frag = np.repeat(np.arange(m_), 3)
    if site == "O":
        P = X[0::3]
    else:
        P = ttm_sites(sub)[0]
    # the field at the polarisable sites: evaluate with the site as an extra point of its own fragment
    Xe = np.vstack([X, P]); fe = np.concatenate([frag, np.arange(m_)])
    qe = np.concatenate([q, np.zeros(m_)]); mue = None if mu is None else np.vstack([mu, np.zeros((m_, 3))])
    E = band_field(Xe, qe, mue, fe, l, range(len(X), len(X) + m_))
    e = E.reshape(-1)
    if not scf:
        return -0.5 * alpha * float(e @ e)
    T = np.zeros((3 * m_, 3 * m_))
    for i in range(m_):
        for j in range(m_):
            if i != j:
                T[3 * i:3 * i + 3, 3 * j:3 * j + 3] = hessian_g(P[i] - P[j], l)
    mu_ind = np.linalg.solve(np.eye(3 * m_) / alpha - T, e)
    return -0.5 * float(mu_ind @ e)


def nonadditive(fn, pos_ang, n_core):
    """the probe's non-additive part of a molecular energy function fn(pos, mols): E(core + probe) - E(core) -
    sum_k E(k + probe) (the monomers' own terms vanish for induction)."""
    n = n_core + 1
    return fn(pos_ang, range(n)) - fn(pos_ang, range(n_core)) - sum(fn(pos_ang, [k, n_core]) for k in range(n_core))


def cluster_nonadditive(fn, pos_ang, n):
    """a cluster's induction beyond the pairs: E(all) - sum_{pairs} E(pair)."""
    return fn(pos_ang, range(n)) - sum(fn(pos_ang, [i, j]) for i in range(n) for j in range(i + 1, n))


# ----------------------------------------------------------------------- 0. the re-evaluation against the library --
print(f"q-AQUA-pol at {truth.qa.root}")
hexa_geo = dict(mc_steps=300, T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
hexamers = W.cluster_dataset([6], 24, seed=100, truth=truth, **hexa_geo)
worst = 0.0
for s in hexamers[:6]:
    pos = bohr_to_ang(s.positions)
    e_lib = truth.qa.ttm(s.positions)["ind"]; e_me, mu_me, _ = ttm_induction(pos)
    mu_lib = truth.qa.ttm(s.positions)["mu"]
    worst = max(worst, abs(e_me - e_lib) / abs(e_lib), float(np.max(np.abs(mu_me - mu_lib))) / float(np.max(np.abs(mu_lib))))
print(f"TTM3-F's induction from its sites against the library on 6 hexamers: worst relative difference {worst:.1e}")

# ----------------------------------------------------------------------- 1-3. the probes ---------------------------
core_geo = {k: v for k, v in hexa_geo.items() if k != "mc_steps"}
dists = (8.0, 10.0, 12.0, 15.0)
probes = W.probe_configurations(3, dists, 10, seed=500, mc_steps=300, n_cores=2, truth=truth, **core_geo)[0]
ref = E3.probe_truth(probes, truth)
l75 = ang_to_bohr(0.75)
rows = []
for s in probes:
    pos = bohr_to_ang(s.positions)
    ind = nonadditive(lambda p, m: ttm_induction(p, m)[0], pos, 3)
    first = nonadditive(lambda p, m: ttm_induction(p, m, order=1)[0], pos, 3)
    second = nonadditive(lambda p, m: ttm_induction(p, m, order=2)[0], pos, 3)
    # piece (2): the core's own self-consistent (and first-order) dipoles against the probe's permanent field
    _, mu_core, _ = ttm_induction(pos, range(3)); _, mu_core1, _ = ttm_induction(pos, range(3), order=1)
    _, _, Eq_all = ttm_induction(pos, range(4), order=1)
    _, _, Eq_core = ttm_induction(pos, range(3), order=1)
    EP = Eq_all[:3] - Eq_core                               # the probe's permanent field at the core's M-sites
    piece2 = -float(np.sum(mu_core * EP)); piece2_1 = -float(np.sum(mu_core1 * EP))
    q3_first = nonadditive(lambda p, m: harness_induction(p, m, "q3", l75, scf=False), pos, 3)
    ttm_first_O = nonadditive(lambda p, m: harness_induction(p, m, "ttm", l75, scf=False), pos, 3)
    ttm_scf_O = nonadditive(lambda p, m: harness_induction(p, m, "ttm", l75, scf=True), pos, 3)
    ttm_scf_M = nonadditive(lambda p, m: harness_induction(p, m, "ttm", l75, scf=True, site="M"), pos, 3)
    rows.append((ind, first, second, piece2, piece2_1, q3_first, ttm_first_O, ttm_scf_O, ttm_scf_M))
rows = np.array(rows)
nadd = ref["nonadditive"]
print(f"\n=== the test probes (2 cores x 4 distances x 10)  [{time.time() - t_start:.0f} s]")
print("share of the truth's non-additive rms (rmse of each form against the truth's non-additive energy, % of its rms):")
labels = ("TTM3-F induction, self-consistent", "first order (alpha E_j.E_k)", "through second order (alpha^2 T)",
          "piece (2): core's SCF dipoles x probe field", "piece (2) with first-order core dipoles")
for d in dists:
    m = ref["d"] == d; r0 = rms(nadd[m])
    print(f"  {d:4.0f} A (rms {r0 * K:.5f} kcal/mol): " + "; ".join(
        f"{lab} {100 * rms(nadd[m] - rows[m, k]) / r0:.1f} %" for k, lab in enumerate(labels)))
print("first-order forms with the harness's band field (l_1 = 0.75 A) against the truth's FIRST-ORDER part, % of its rms:")
for d in dists:
    m = ref["d"] == d; f1 = rows[m, 1]
    fit = lambda col: float(np.sum(col * f1) / np.sum(col * col))
    c_q3 = fit(rows[m, 5] / ALPHA); c_t = fit(rows[m, 6] / ALPHA)
    print(f"  {d:4.0f} A: q3 at the atoms' O, alpha fitted ({c_q3:.2f} a.u.) {100 * rms(f1 - c_q3 * rows[m, 5] / ALPHA) / rms(f1):.1f} %; "
          f"TTM3-F multipoles, alpha = 9.745 at O {100 * rms(f1 - rows[m, 6]) / rms(f1):.1f} % (alpha fitted {c_t:.2f}: "
          f"{100 * rms(f1 - c_t * rows[m, 6] / ALPHA) / rms(f1):.1f} %)")
print("the pinned self-consistent response against the truth's non-additive energy, % of its rms:")
for d in dists:
    m = ref["d"] == d; r0 = rms(nadd[m])
    print(f"  {d:4.0f} A: TTM3-F multipoles + alpha at O + band dipole tensor {100 * rms(nadd[m] - rows[m, 7]) / r0:.1f} %; "
          f"the same at the M-site {100 * rms(nadd[m] - rows[m, 8]) / r0:.1f} %; first order at O {100 * rms(nadd[m] - rows[m, 6]) / r0:.1f} %")

# ----------------------------------------------------------------------- the hexamers -------------------------------
print(f"\n=== the 24 test hexamers  [{time.time() - t_start:.0f} s]")
hx = []
for s in hexamers:
    pos = bohr_to_ang(s.positions)
    na = s.energy - W.pair_additive_energy(pos, 6, truth)
    ind = cluster_nonadditive(lambda p, m: ttm_induction(p, m)[0], pos, 6)
    first = cluster_nonadditive(lambda p, m: ttm_induction(p, m, order=1)[0], pos, 6)
    second = cluster_nonadditive(lambda p, m: ttm_induction(p, m, order=2)[0], pos, 6)
    scf_O = cluster_nonadditive(lambda p, m: harness_induction(p, m, "ttm", l75, scf=True), pos, 6)
    first_O = cluster_nonadditive(lambda p, m: harness_induction(p, m, "ttm", l75, scf=False), pos, 6)
    q3_first = cluster_nonadditive(lambda p, m: harness_induction(p, m, "q3", l75, scf=False), pos, 6)
    hx.append((na, ind, first, second, scf_O, first_O, q3_first))
hx = np.array(hx); r0 = rms(hx[:, 0])
print(f"non-additive rms {r0 * K:.3f} kcal/mol; TTM3-F's many-body induction {rms(hx[:, 1]) * K:.3f}, its first-order part "
      f"{rms(hx[:, 2]) * K:.3f}, mutual remainder {rms(hx[:, 1] - hx[:, 2]) * K:.3f} ({100 * rms(hx[:, 1] - hx[:, 2]) / rms(hx[:, 1]):.0f} % "
      f"of the induction), through second order leaves {rms(hx[:, 1] - hx[:, 3]) * K:.3f}")
for k, lab in ((1, "TTM3-F induction (SCF)"), (2, "its first order"), (4, "pinned SCF at O, TTM3-F multipoles, band tensor"),
               (5, "pinned first order at O, TTM3-F multipoles"), (6, "pinned first order at O, q3")):
    print(f"  the non-additive energy minus {lab}: {100 * rms(hx[:, 0] - hx[:, k]) / r0:.0f} %")
print(f"\ndone in {time.time() - t_start:.0f} s")
