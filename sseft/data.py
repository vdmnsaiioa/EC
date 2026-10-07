"""
Data builders: analytic dimer truths for the ladder (Tang-Toennies with C6, C8, C10 -- the synthetic truth of
the programme; the published ab initio potentials plug in here as further `truth` callables), windows and
grids per `phase2-reference-data.md`, and isolated-atom references.
"""
import math
import numpy as np
from scipy.special import gammainc
from .structure import Structure
from .units import ang_to_bohr

# Tang-Toennies rare-gas parameters (TT 2003 coefficients; A, b fitted to Re, De -- long-range-atomistic-ml.md).
# C's in E_h a0^2n, A in E_h, b in 1/a0.
TT = {
    "He2": dict(C=(1.461, 14.11, 183.5), A=32.5, b=2.478, Z=2),
    "Ne2": dict(C=(6.383, 90.34, 1536.0), A=198.1, b=2.456, Z=10),
    "Ar2": dict(C=(64.30, 1623.0, 49060.0), A=725.8, b=2.027, Z=18),
    "Kr2": dict(C=(129.6, 4187.0, 155500.0), A=766.5, b=1.854, Z=36),
    "Xe2": dict(C=(285.9, 12810.0, 619800.0), A=873.0, b=1.671, Z=54),
}


def tt_potential(name):
    """V(R) in hartree for R in bohr (damped Tang-Toennies, N = 3 dispersion terms)."""
    p = TT[name]
    def V(R):
        R = np.asarray(R, float); x = p["b"] * R
        v = p["A"] * np.exp(-x)
        for k, c in enumerate(p["C"]):
            n = 6 + 2 * k
            v = v - gammainc(n + 1, x) * c / R ** n
        return v
    return V


# ----------------------------------------------------------------------------------------------------------
# Published ab initio pair potentials (the first real truths of E1/E2, phase-d-build-plan.md section 2).
#
# All in the Rostock form (R in nm, V in K):
#     V(R) = A exp(a1 R + a2 R^2 + a_-1 / R + a_-2 / R^2)
#            - sum_{n=3}^{8} C_2n / R^2n * [1 - exp(-b R) sum_{k=0}^{2n} (b R)^k / k!]
#
# PROVENANCE.  The parameter sets below were entered from memory of the papers, not copied from them, and
# are accepted only because each reproduces the paper's own well depth and position to all quoted digits
# (tests/test_data.py: Ar2 eps/k = 143.123 K at 0.37618 nm, Ne2 eps/k = 42.153 K at 0.30894 nm), which
# twelve parameters cannot do by accident.  They should still be checked against Table 2 of each paper
# before the paper phase; Kr2 (Jaeger, Hellmann, Bich, Vogel, J. Chem. Phys. 144, 114304 (2016)) and Xe2
# (Hellmann, Jaeger, Bich, J. Chem. Phys. 147, 034304 (2017)) use the same form and are added by passing
# their tables to `rostock_potential` (a json file with the keys below).
PUBLISHED = {
    "Ar2": dict(   # Jaeger, Hellmann, Bich, Vogel, Mol. Phys. 107, 2181 (2009)
        A=4.61330146e7, a1=-2.98337630e1, a2=-9.71208881, am1=2.75206827e-2, am2=-1.01489050e-2, b=4.02517211e1,
        C={6: 4.42812017e-1, 8: 3.26707684e-2, 10: 2.45656537e-3, 12: 1.88246247e-4, 14: 1.47012192e-5, 16: 1.17006343e-6},
        Z=18, eps_K=143.123, Re_nm=0.37618, source="Jaeger et al. Mol. Phys. 107, 2181 (2009); parameters recalled, verified by eps/Re"),
    "Ne2": dict(   # Hellmann, Bich, Vogel, Mol. Phys. 106, 133 (2008)
        A=4.02915058383e7, a1=-4.28654039586e1, a2=-3.33818674327, am1=-5.34644860719e-2, am2=5.01774999419e-3, b=4.92438731676e1,
        C={6: 4.40676750157e-2, 8: 1.64892507701e-3, 10: 7.90473640524e-5, 12: 4.85489170103e-6, 14: 3.82012334054e-7, 16: 3.85106552963e-8},
        Z=10, eps_K=42.153, Re_nm=0.30894, source="Hellmann et al. Mol. Phys. 106, 133 (2008); parameters recalled, verified by eps/Re"),
}

KELVIN_PER_HARTREE = 315775.02480407
NM_PER_BOHR = 0.052917721067


def rostock_potential(p):
    """V(R) in hartree for R in bohr, from a Rostock-form parameter dict (R in nm, V in K inside)."""
    from math import factorial
    ns = sorted(p["C"])
    fact = {n: np.array([factorial(k) for k in range(n + 1)], float) for n in ns}
    def V(R):
        R = np.asarray(R, float) * NM_PER_BOHR
        v = p["A"] * np.exp(p["a1"] * R + p["a2"] * R ** 2 + p["am1"] / R + p["am2"] / R ** 2)
        x = p["b"] * R
        for n in ns:
            k = np.arange(n + 1)
            damp = 1.0 - np.exp(-x) * np.sum(x[..., None] ** k / fact[n], axis=-1)
            v = v - p["C"][n] / R ** n * damp
        return v / KELVIN_PER_HARTREE
    return V


def published_potential(name):
    return rostock_potential(PUBLISHED[name])


def published_c2n(name, n):
    """C_2n of a published potential in E_h a0^2n (its own asymptotic coefficient, not DOSD)."""
    return PUBLISHED[name]["C"][n] / KELVIN_PER_HARTREE / NM_PER_BOHR ** n


# Sealed matching targets for E2 (never shown to a model).  DOSD C6 in E_h a0^6 (Kumar & Meath 1985; the
# homonuclear values are the ones the TT truth uses), static dipole polarisabilities in a0^3.  The mixed
# C6 values are recalled and must be verified against the DOSD tables before E2 is reported.
DOSD_C6 = {("Ne", "Ne"): 6.383, ("Ar", "Ar"): 64.30, ("Kr", "Kr"): 129.6, ("Xe", "Xe"): 285.9,
           ("Ne", "Ar"): 19.50, ("Ar", "Kr"): 91.13}
DOSD_C8 = {("Ne", "Ne"): 90.34, ("Ar", "Ar"): 1623.0, ("Kr", "Kr"): 4187.0, ("Xe", "Xe"): 12810.0}   # E_h a0^8 (TT 2003 table; verify against Kumar & Meath)
ALPHA_STATIC = {"Ne": 2.669, "Ar": 11.083, "Kr": 16.78, "Xe": 27.32}
SYMBOL = {2: "He", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe"}


def dimer_structures(Z, R_bohr, truth, sigma=0.0, rng=None, with_forces=True):
    """homonuclear dimer along z at separations R (bohr); energies from `truth` (+ Gaussian noise sigma),
    forces from the numerical derivative of the noiseless truth."""
    rng = rng or np.random.default_rng(0)
    out = []
    for R in np.atleast_1d(R_bohr):
        E = float(truth(R)) + (rng.normal() * sigma if sigma > 0 else 0.0)
        h = 1e-4 * R
        dV = (truth(R + h) - truth(R - h)) / (2 * h)
        F = np.array([[0, 0, +dV], [0, 0, -dV]]) if with_forces else None   # F_1 = -dE/dz_1 with z_1 = -R/2: dE/dz_1 = -dV
        pos = np.array([[0, 0, -R / 2], [0, 0, R / 2]])
        out.append(Structure(pos, np.array([Z, Z]), energy=E, forces=F, info={"R": float(R)}))
    return out


def isolated_atom(Z):
    return Structure(np.zeros((1, 3)), np.array([Z]), energy=0.0, forces=np.zeros((1, 3)), info={"R": np.inf})


def window_design(R_minus_ang, R_plus_ang, n=40, log_uniform=True):
    lo, hi = ang_to_bohr(R_minus_ang), ang_to_bohr(R_plus_ang)
    return np.exp(np.linspace(np.log(lo), np.log(hi), n)) if log_uniform else np.linspace(lo, hi, n)


def eval_grid(R_plus_ang, R_max_ang=500.0, n=61):
    return np.exp(np.linspace(np.log(ang_to_bohr(R_plus_ang * 1.02)), np.log(ang_to_bohr(R_max_ang)), n))
