"""linear-class minimum of the charge scale lambda under the training objective (energies + forces, w_force = 1,
per-atom energies) on the window nodes where E_0 is silent -- as in e3-water-preregistration.md section 3 -- for
two definitions of 'silent': R_OO > r_c (what section 3 used) and min intermolecular pair distance > r_c."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from sseft import water as W, model as M, train as T, data as D, structure as S
from sseft.units import ang_to_bohr, bohr_to_ang
WINDOW = (1.35 * 2.86, 1.35 * 2.86 * 1.8)
R_train = bohr_to_ang(D.window_design(*WINDOW, 40))
rng = np.random.default_rng(0)
train = W.dimer_scan(R_train, sigma=1e-9, rng=rng)
rung = M.with_rung(M.RUNGS["M_1"], pin_q=True, e0={**M.RUNGS["M_1"].e0, "r_cut": ang_to_bohr(6.0)})
params = M.init_params(jax.random.PRNGKey(0), rung, 1e-3)
E_b, F_b = T.predict(params, rung, train)      # band with the truth's charges (lambda = 1) + E_0 noise...
# E_0 at init is not zero inside r_c; isolate the band: evaluate with site energies off
rung0 = M.with_rung(rung, site_energies=False)
E_b, F_b = T.predict(params, rung0, train); E_b = np.array(E_b); F_b = np.array(F_b)[:, :6]
E_t = np.array([s.energy for s in train]); F_t = np.array([s.forces for s in train])
n = 6.0; N_E = 44.0; N_F = 44 * 6 * 3.0   # the run's counts (40 dimers + 4 monomers)
def lam_hat(sel, w_force=1.0):
    e_num = np.sum(E_b[sel] * E_t[sel]) / n**2 / N_E; e_den = np.sum(E_b[sel] ** 2) / n**2 / N_E
    f_num = np.sum(F_b[sel] * F_t[sel]) / N_F; f_den = np.sum(F_b[sel] ** 2) / N_F
    lam2 = (e_num + w_force * f_num) / (e_den + w_force * f_den)
    lam2_e = e_num / e_den
    return np.sqrt(lam2), np.sqrt(lam2_e)
for rc in (6.0, 5.0):
    sel_OO = R_train > rc
    dmin = np.array([np.linalg.norm(s.positions[:3, None] - s.positions[None, 3:], axis=-1).min() for s in train]) * bohr_to_ang(1.0)
    sel_pair = dmin > rc
    for label, sel in (("R_OO > r_c", sel_OO), ("min pair > r_c", sel_pair)):
        if sel.sum() == 0: print(f"r_c = {rc}: {label}: no nodes"); continue
        l, le = lam_hat(sel)
        print(f"r_c = {rc} A, {label}: {sel.sum()} nodes, lambda_hat = {l:.4f} (energies only {le:.4f}) -> mu_hat = {1.855 * l:.3f} D")
