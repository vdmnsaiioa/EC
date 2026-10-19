"""
The numbers of notes/e3qp2-preregistration.md that are not the matched responses (those: scratch/e3qp2_matched_response.py),
computed from q-AQUA-pol before any training:

    python scratch/e3qp2_preregistration.py [external/qaqua/build_pol]

(a) the dimer: the exact daggers M_1pm and M_16pm at the reading rows, evaluated with the harness's own rungs (E_0 has
    nothing beyond r_c): what is left is TTM3-F's induction (both) and its O-O C6 + exponential term (M_1pm);
(b) the truth's energy: the energy-only routine against the gradient routine on relaxed clusters and in the 3-body
    blend band, and the PES's own error on the raw 3-body set by largest O-O with the gradient routine's energies;
(c) the training probes of configs/water_qaqua_pol2.yaml: counts by core size and distance, the share beyond r_c of
    their cores, and their signal (interaction, non-additive energy, the probe's induction).
"""
import math, os, sys, time
import numpy as np
import jax

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, qaqua as QA, data as D, model as M, train as T
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "external", "qaqua", "build_pol")
truth = W.qaqua_pol_truth(root)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
t0 = time.time()
print(f"q-AQUA-pol at {truth.qa.root}: gamma_M = {truth.gamma_m}; site charges at the rigid geometry "
      f"{np.round(truth.site_model(W.monomer_positions())[0][1:3], 7)} (H), {truth.site_model(W.monomer_positions())[2][0]:+.7f} (M)")

print("\n=== (a) the exact daggers on the dimer tail (phi = 0; the dimer part's band edge 1.5 A, r_c 5 A)")
Re = truth.r_e; hi = 1.35 * Re * 1.8
R_eval = bohr_to_ang(D.eval_grid(hi, 500.0, 61))
rows = np.array([R_eval[int(np.argmin(np.abs(R_eval - Rv)))] for Rv in (8, 10, 12, 15, 20, 30, 50, 100, 300)])
test = W.dimer_scan(rows, truth=truth)
a = type("A", (), dict(lA=4.0, rcut=5.0, F=16, l1=0.0, rs_over_l1=0.0, rungs="M_1pm,M_16pm"))()
E3.pin_sources(a, test, truth=truth)
f_true = np.array([s.energy for s in test])
pieces = [truth.qa.ttm(s.positions) for s in test]
for name in ("M_1pm", "M_16pm"):
    rung = E3.configure(E3.RUNGS[name], a)
    params = M.init_params(jax.random.PRNGKey(0), rung)
    E = T.predict_energies(params, rung, test)
    print(f"  {name}: relative error at the rows " + " / ".join(f"{abs(e / f - 1):.3g}" for e, f in zip(E, f_true)))
print("  the truth's pieces at the rows (share of the energy): induction " + " / ".join(f"{p['ind'] / f:.3g}" for p, f in zip(pieces, f_true))
      + "; the O-O vdW term " + " / ".join(f"{p['vdw'] / f:.3g}" for p, f in zip(pieces, f_true)))
kappa = QA.TTM3F_CHARGECON ** 2 / (QA.AUKCAL * QA.AUANG)
print(f"  TTM3-F's Coulomb constant over the atomic unit's (in the potential's own units): {kappa:.8f}")

print(f"\n=== (b) the truth's energy routines  [{time.time() - t0:.0f} s]")
rng = np.random.default_rng(11)
for n in (3, 4, 6):
    d = []
    for _ in range(12):
        pos = W.random_cluster(n, rng, mc_steps=300, energy=truth.sampling_energy, min_OO=2.7, min_any=1.7)
        d.append(truth.qa.interaction_energy_sampling(ang_to_bohr(pos)) - truth.energy(pos, n))
    d = np.abs(d) * K
    print(f"  relaxed {n}-mers: |energy-only - gradient routine| max {d.max():.2e} kcal/mol, nonzero on {np.sum(d > 1e-9)} of 12")
pos, lab = QA.read_raw(os.path.join(truth.qa.root, "q-AQUA", "3b_data.xyz"), 3)
keep = np.flatnonzero(QA.near_rigid(pos, W.R_OH, math.degrees(W.THETA), 0.03, 5.0))
worst = np.array([max(truth.energy(np.concatenate([pos[f][3 * i:3 * i + 3], pos[f][3 * j:3 * j + 3]]), 2)
                      for i in range(3) for j in range(i + 1, 3)) for f in keep])
sel = keep[worst * K <= 2.0]
P, L = pos[sel], lab[sel]
pes = np.array([truth.qa.nbody(ang_to_bohr(p), max_order=3)[3] for p in P])
pes_old = np.array([truth.qa.interaction_energy_sampling(ang_to_bohr(p)) - sum(
    truth.qa.interaction_energy_sampling(ang_to_bohr(np.concatenate([p[3 * i:3 * i + 3], p[3 * j:3 * j + 3]])))
    for i in range(3) for j in range(i + 1, 3)) for p in P])
roo = QA.largest_oo(P)
for lo, hi_ in ((0, 5), (5, 7), (7, 9), (9, math.inf)):
    m = (roo >= lo) & (roo < hi_)
    print(f"  raw 3-body, largest O-O {lo}-{hi_} A: {m.sum()} frames, labels rms {rms(L[m]) * K:.4f}; the PES's rmse "
          f"{rms(pes[m] - L[m]) * K:.4f} (gradient routine) / {rms(pes_old[m] - L[m]) * K:.4f} (energy-only, as E3-QP read it)")
m = (roo >= 5.0) & (roo <= 5.5)
print(f"  in the blend band 5.0-5.5 A: {m.sum()} frames, PES rmse {rms(pes[m] - L[m]) * K:.4f} / {rms(pes_old[m] - L[m]) * K:.4f}; "
      f"all {len(L)}: {rms(pes - L) * K:.4f} / {rms(pes_old - L) * K:.4f}")

print(f"\n=== (c) the training probes  [{time.time() - t0:.0f} s]")
geo = dict(T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
dists = (8.0, 10.0, 12.0, 15.0)
probes = []
for k, n_core in enumerate((3, 4, 5)):
    made = W.probe_configurations(n_core, dists, 2, seed=400 + 1000 * k, mc_steps=300, n_cores=16, truth=truth, **geo)[0]
    for st in made:
        st.info.update(n_core=n_core, core_id=1000 * k + st.info["core"])
    probes += made
blind = E3.far_field_blind(probes, 6.0)
ref = E3.probe_truth(probes, truth)
size = np.array([s.info["n_core"] for s in probes])
for n_core in (3, 4, 5):
    m = size == n_core
    print(f"  cores of {n_core}: {m.sum()} probes; beyond r_c by d " + ", ".join(
        f"{d:g} A {np.mean(blind[m & (ref['d'] == d)]) * 100:.0f} %" for d in dists)
          + "; interaction / non-additive / induction rms by d (kcal/mol) " + "; ".join(
        f"{rms(ref['interaction'][m & (ref['d'] == d)]) * K:.4f} / {rms(ref['nonadditive'][m & (ref['d'] == d)]) * K:.4f} / "
        f"{rms(ref['induction'][m & (ref['d'] == d)]) * K:.4f}" for d in dists))
print(f"  matched on {blind.sum()} of {len(probes)} (beyond r_c of their cores)")
print(f"\ndone in {time.time() - t0:.0f} s")
