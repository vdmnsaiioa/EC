"""the embedded dagger's probe interaction beyond r_c is training-independent (E_0 is exactly silent): evaluate it with
untrained parameters on the d = 10 A test configurations and compare with the truth, piece by piece."""
import sys, os, pickle; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from sseft import water as W, model as M, train as T
from sseft.units import bohr_to_ang, ang_to_bohr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import e3_water as E3
base, far_train, E_core_tr, test, E_core = pickle.load(open("results/cache_far_8cores.pkl", "rb"))
rung = M.with_rung(E3.RUNGS["M_1pemb"], e0={**M.RUNGS["M_1"].e0, "r_cut": ang_to_bohr(6.0)})
params = M.init_params(jax.random.PRNGKey(0), rung, 1e-3)
D_SEL = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
sel = [i for i, s in enumerate(test) if s.info["d"] == D_SEL]
ts = [test[i] for i in sel]; E3.pin_embedded(ts, n_core=3)
cores = [W.water_structure(bohr_to_ang(s.positions)[:9]) for s in ts]; E3.pin_embedded(cores)
Et, _ = T.predict(params, rung, ts); Ec, _ = T.predict(params, rung, cores)
yp = np.array(Et) - np.array(Ec)
print(f"d = {D_SEL} A configurations: model (untrained, pinned embedded sources, E_0 as initialised) vs truth, kcal/mol")
print(" k   truth y    model yp   diff      es_truth   induction  disp     diff-(-ind-disp)")
rows = []
for s, y_model in zip(ts, yp):
    pos = bohr_to_ang(s.positions); ca = W.pw_components(pos, 4); cc = W.pw_components(pos[:9], 3)
    y = s.energy - s.info["E_core"]
    es = ca["es"] - cc["es"]; rep = ca["rep"] - cc["rep"]; disp = ca["disp"] - cc["disp"]; ind = y - es - rep - disp
    # piece (2) with the truth's kernel
    x = s.positions; q = np.tile(W.monomer_charges(), 4); mu_core = cc["mu_ind"]
    from sseft import kernels as kn; import jax.numpy as jnp
    Ep = np.zeros((9, 3))
    for j in range(9):
        for i in range(9, 12):
            D = x[j] - x[i]; r = np.linalg.norm(D); Ep[j] += -q[i] * float(kn.dg_long(jnp.asarray(r), W.PW["lam"])) * D / r
    p2 = -np.sum(mu_core * Ep)
    rows.append((y, y_model, es, p2, ind, disp))
    print(f"{s.info['k']:2d} {y*627.5:+9.5f} {y_model*627.5:+9.5f} {(y_model-y)*627.5:+9.5f} {es*627.5:+9.5f} {ind*627.5:+9.5f} {disp*627.5:+8.5f}   model-(es+p2) = {(y_model-es-p2)*627.5:+.6f}")
rows = np.array(rows); rms = lambda v: np.sqrt(np.mean(v**2)) * 627.5
print(f"rms: diff {rms(rows[:,1]-rows[:,0]):.5f} = {rms(rows[:,1]-rows[:,0])/rms(rows[:,4])*100:.1f} % of the induction; "
      f"remainder (ind - p2) {rms(rows[:,4]-rows[:,3]):.5f} = {rms(rows[:,4]-rows[:,3])/rms(rows[:,4])*100:.1f} %; "
      f"disp {rms(rows[:,5]):.5f} = {rms(rows[:,5])/rms(rows[:,4])*100:.1f} %; model - (es + p2): {rms(rows[:,1]-rows[:,2]-rows[:,3]):.6f}")
