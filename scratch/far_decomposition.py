"""decomposition of the far-field test's 'induction' (probe interaction minus point-charge electrostatics) into
(2) the interaction of the core's pre-existing induced dipoles with the probe's charges (first order, no 1/2) and the
remainder (1) + (3), the responses of probe and core to each other (second order).  Reads the cached data sets of
the eight-core run."""
import sys, os, pickle; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sseft import water as W, kernels as kn
from sseft.units import bohr_to_ang
base, far_train, E_core_tr, test, E_core = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/cache_far_8cores.pkl", "rb"))
rows = []
for s, Ec in zip(test, E_core):
    pos = bohr_to_ang(s.positions); c_all = W.pw_components(pos, 4); c_core = W.pw_components(pos[:9], 3)
    y = s.energy - Ec
    es = (c_all["es"] - c_core["es"]) + (c_all["rep"] - c_core["rep"]) + (c_all["disp"] - c_core["disp"])
    ind = y - es
    # piece (2): - sum_j mu_j(core alone) . E_probe(r_j), the probe's charges through the truth's smeared kernel
    mu_core = c_core["mu_ind"]                                   # (9,3) induced dipoles of the core alone (O rows nonzero)
    x = s.positions; q = np.tile(W.monomer_charges(), 4)
    E_probe = np.zeros((9, 3))
    for j in range(9):
        for i in range(9, 12):
            D = x[j] - x[i]; r = np.linalg.norm(D)
            E_probe[j] += -q[i] * float(kn.dg_long(jnp.asarray(r), W.PW["lam"])) * D / r
    piece2 = -np.sum(mu_core * E_probe)
    rows.append((s.info["d"], y, ind, piece2, ind - piece2))
rows = np.array(rows)
print("d (A)   rms interaction   rms induction   rms piece (2)   rms remainder (1)+(3)   remainder / induction")
for d in sorted(set(rows[:, 0])):
    r = rows[rows[:, 0] == d]; rms = lambda v: np.sqrt(np.mean(v ** 2)) * 627.5
    print(f"{d:5.1f}   {rms(r[:,1]):12.4f}   {rms(r[:,2]):12.4f}   {rms(r[:,3]):12.4f}   {rms(r[:,4]):14.4f}   {rms(r[:,4]) / rms(r[:,2]) * 100:12.1f} %")
rms = lambda v: np.sqrt(np.mean(v ** 2)) * 627.5
print(f"all     {rms(rows[:,1]):12.4f}   {rms(rows[:,2]):12.4f}   {rms(rows[:,3]):12.4f}   {rms(rows[:,4]):14.4f}   {rms(rows[:,4]) / rms(rows[:,2]) * 100:12.1f} %")
