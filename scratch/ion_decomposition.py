"""the band-field response test on an ion probe, truth only: for relaxed water trimers plus a unit charge at d from
the centre of mass, the ion's interaction with the core decomposed into point-charge electrostatics (+ repulsion),
piece (2) = the core's embedded induced dipoles against the ion's field, and the remainder = the core's
self-consistent response to the ion -- the band-field response that F_s has to carry."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from sseft import water as W
dists = (5.0, 6.0, 7.0, 8.0, 10.0, 12.0)
n_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 4; n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
test, _ = W.ion_configurations(3, dists, n_per, seed=300, mc_steps=300, n_cores=n_cores)
rows = []
for st in test:
    d = W.ion_response_decomposition(st); rows.append((st.info["d"], d["y"], d["es"], d["piece2"], d["response"]))
rows = np.array(rows); K = 627.509
print(f"{n_cores} trimer cores x {n_per} placements per distance; kcal/mol")
print("  d (A)   rms interaction   rms electrostatics   rms piece (2)   rms response   response / |y|   response / piece (2)")
for d in dists:
    r = rows[rows[:, 0] == d]; rms = lambda v: np.sqrt(np.mean(v ** 2)) * K
    print(f"  {d:5.1f}   {rms(r[:,1]):14.4f}   {rms(r[:,2]):16.4f}   {rms(r[:,3]):12.4f}   {rms(r[:,4]):12.4f}   {rms(r[:,4])/rms(r[:,1])*100:10.1f} %   {rms(r[:,4])/rms(r[:,3])*100:10.1f} %")
# power law of the response
mids = np.array(dists); rr = np.array([np.sqrt(np.mean(rows[rows[:, 0] == d][:, 4] ** 2)) for d in dists])
print(f"  response rms power law on 6-12 A: d^({np.polyfit(np.log(mids[1:]), np.log(rr[1:]), 1)[0]:.2f})  (first-order response to a monopole field: -4)")
# a single monomer at 20 A: the response is exactly -1/2 alpha |E|^2 (one polarisable site)
pos = np.vstack([W.monomer_positions(), [[0.0, 0.0, 20.0]]])
c = W.pw_components(pos, 1, n_ion=1); E = np.linalg.norm(c["Eq"][0])
print(f"  monomer + ion at 20 A: response {c['ind'] * K:.6f} kcal/mol = -1/2 alpha |E|^2 with alpha = {-2 * c['ind'] / E**2:.4f} (truth 9.72)")
