"""how much of each reading the order-0 band-field read-out with PERMANENT sources can carry: the first-order
induction E_ind1 = -1/2 sum_i alpha_i |E_q,i|^2 (the field of the other molecules' permanent charges at each polarisable
site, including the cross terms between two neighbours' fields -- three-body at first order) against the truth's
self-consistent E_ind.  (1) hexamers: the non-additive energy NA = E_ind(hex) - sum_dimers E_ind, and its first-order
part NA1; (2) the water-probe far-field test: the probe's induction and its first-order part; (3) the ion test."""
import sys, os, pickle; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from sseft import water as W
from sseft.units import bohr_to_ang
K = 627.509; rms = lambda v: np.sqrt(np.mean(np.asarray(v) ** 2)) * K
# (1) hexamers
train, test6, test2 = pickle.load(open("results/cache_clusters.pkl", "rb"))
NA, NA1 = [], []
for s in test6:
    pos = bohr_to_ang(s.positions); c = W.pw_components(pos, 6)
    e_ind, e_ind1 = c["ind"], c["ind1"]
    for i in range(6):
        for j in range(i + 1, 6):
            cd = W.pw_components(np.concatenate([pos[3*i:3*i+3], pos[3*j:3*j+3]]), 2)
            e_ind -= cd["ind"]; e_ind1 -= cd["ind1"]
    NA.append(e_ind); NA1.append(e_ind1)
NA, NA1 = np.array(NA), np.array(NA1)
print(f"hexamers (8): non-additive energy rms {rms(NA):.3f} kcal/mol; its first-order part rms {rms(NA1):.3f} = {rms(NA1)/rms(NA)*100:.0f} %; "
      f"residual (self-consistent correction) rms {rms(NA-NA1):.3f} = {rms(NA-NA1)/rms(NA)*100:.0f} % of NA")
# (2) water-probe test (cached eight-core data)
base, far_train, E_core_tr, test, E_core = pickle.load(open("results/cache_far_8cores.pkl", "rb"))
rows = []
for s, Ec in zip(test, E_core):
    pos = bohr_to_ang(s.positions); ca = W.pw_components(pos, 4); cc = W.pw_components(pos[:9], 3)
    ind = ca["ind"] - cc["ind"]; ind1 = ca["ind1"] - cc["ind1"]
    rows.append((s.info["d"], ind, ind1))
rows = np.array(rows)
print("water probe at d from a trimer: induction rms, first-order part, residual (kcal/mol, % of the induction)")
for d in (6.0, 7.0, 8.0, 10.0):
    r = rows[rows[:, 0] == d]
    print(f"  {d:5.1f}  {rms(r[:,1]):.4f}  {rms(r[:,2]):.4f} ({rms(r[:,2])/rms(r[:,1])*100:.0f} %)  {rms(r[:,1]-r[:,2]):.4f} ({rms(r[:,1]-r[:,2])/rms(r[:,1])*100:.0f} %)")
# (3) ion test: the ion's induction (response + piece 2) and its first-order part
dists = (5.0, 6.0, 7.0, 8.0, 10.0, 12.0)
test_i, _ = W.ion_configurations(3, dists, 4, seed=300, mc_steps=300, n_cores=4)
rows = []
for s in test_i:
    pos = bohr_to_ang(s.positions); ca = W.pw_components(pos, 3, n_ion=1); cc = W.pw_components(pos[:-1], 3)
    rows.append((s.info["d"], ca["ind"] - cc["ind"], ca["ind1"] - cc["ind1"]))
rows = np.array(rows)
print("ion at d from a trimer: induction (piece 2 + response) rms, first-order part, residual (kcal/mol, % of the induction)")
for d in dists:
    r = rows[rows[:, 0] == d]
    print(f"  {d:5.1f}  {rms(r[:,1]):.4f}  {rms(r[:,2]):.4f} ({rms(r[:,2])/rms(r[:,1])*100:.0f} %)  {rms(r[:,1]-r[:,2]):.4f} ({rms(r[:,1]-r[:,2])/rms(r[:,1])*100:.0f} %)")
