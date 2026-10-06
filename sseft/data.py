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
