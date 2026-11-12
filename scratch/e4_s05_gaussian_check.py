"""S0.5's account of S0.3 (notes/e4-s0-preregistration.md section 6), checked by Monte Carlo on its Gaussian model:
the bare permanent polarisation Y and residual eta independent, the Coulomb weight exp(-Pi_L^2 / 2 eps_inf) on the
longitudinal channel only (Pi = A Y + eta), P_L = Pi_L / eps_inf and P_T = Pi_T; variances in L's units. Then
r = b_T / b_L = eps_inf + v_eta, b_L = A / (eps_inf + v_eta), and the transverse regression's residual is v_eta.

    python scratch/e4_s05_gaussian_check.py
"""
import numpy as np

rng = np.random.default_rng(0)
for eps, A, vY, veta in ((1.6885, 1.455, 25.0, 0.3), (1.7, 1.2, 40.0, 0.8), (2.0, 1.5, 5.0, 0.05)):
    lam = np.diag([1 / vY, 1 / veta]) + np.outer([A, 1.0], [A, 1.0]) / eps     # the longitudinal precision matrix
    n = 400000
    Yl, el = rng.multivariate_normal([0, 0], np.linalg.inv(lam), size=n).T
    PL = (A * Yl + el) / eps
    Yt, et = rng.normal(scale=np.sqrt(vY), size=n), rng.normal(scale=np.sqrt(veta), size=n)
    PT = A * Yt + et
    bL, bT = np.mean(PL * Yl) / np.mean(Yl ** 2), np.mean(PT * Yt) / np.mean(Yt ** 2)
    print(f"eps_inf {eps}, A {A}, v_Y {vY}, v_eta {veta}: r {bT / bL:.4f} against eps_inf + v_eta {eps + veta:.4f}; "
          f"the transverse residual {np.mean((PT - bT * Yt) ** 2):.4f}; b_L {bL:.4f} against A / (eps_inf + v_eta) "
          f"{A / (eps + veta):.4f}; L {np.mean(PL ** 2):.4f}")
