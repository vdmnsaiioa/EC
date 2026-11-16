"""E4 job 6's check on synthetic cells (notes/e4-job6-preregistration.md section 8; v0.0.30, CPU, about an hour): a
teacher made of MB-pol's rigid site charges alone (O 0, H +0.578, M -1.155 e; the Coulomb band in the cell, no E_0, no
dispersion) labels random cells of 32 molecules (12.5 A); M_inf and M_1 with its charges in units of 1 e (job 1's), 0.02 e and
0.05 e (job 6's M_1u) learn it with E4's training settings, the features reduced to 16, Adam alone. Printed: the final
full-batch training loss and, for the learned charges, the molecular dipole and its sign. The teacher has no electrons,
so the charges that reproduce it carry its 1.864 D. A rung name `<rung>@<unit>` sets its charge unit (e).

    python scratch/e4_job6_teacher_student.py . 600 12 32 M_inf,M_1@1,M_1@0.02,M_1@0.05 16   # repo, steps, cells, molecules, rungs, F
"""
import os, sys, time, json
import numpy as np
REPO = sys.argv[1]
STEPS = int(sys.argv[2]); NCELL = int(sys.argv[3]); NMOL = int(sys.argv[4]); RUNGS = sys.argv[5].split(",")
FEAT = int(sys.argv[6]) if len(sys.argv) > 6 else 16
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "tests"))
import jax, jax.numpy as jnp
import sseft
from dataclasses import replace
from sseft import model as M, train as T, water as W, structure as S
from sseft.mbpol import random_box
import e4_water as E4
SITE_Q = np.array([0.0, 0.577650024808747, 0.577650024808747, -1.155300049617494]); GAMMA = 0.426706882
L = 12.5
import tempfile
out = os.path.join(tempfile.gettempdir(), f"e4_job6_cells_{NCELL}x{NMOL}.npz")      # labelled once, reused
cfg = E4.merge(E4.load_config(None), {"model": {"features": FEAT}})
if not os.path.exists(out):
    rng = np.random.default_rng(11)
    pos = np.array([random_box(NMOL, np.eye(3) * L, rng, min_oo=2.6, min_any=1.6) for _ in range(NCELL)])
    np.savez(out, positions=pos, cell=np.eye(3) * L, energy=np.zeros(NCELL), forces=np.zeros_like(pos),
             site_q=SITE_Q, site_gamma=np.array(GAMMA), source=np.zeros(NCELL, int), time_ps=np.arange(NCELL) * 0.5)
    st = W.e4_structures(out, pins=True)
    tc, _ = E4.rungs_for("M_16pm", cfg, st[0].cell)
    tc = replace(tc, site_energies=False, dispersion=False)
    p0 = M.init_params(jax.random.PRNGKey(99), tc, energy_scale=1.0)
    f = jax.jit(lambda p, b: M.energy_forces_single(p, tc, b))
    E, F = [], []
    for s in st:
        b = {k: v[0] for k, v in S.pad_batch([s], n_freq=tc.K).items()}
        e, g = f(p0, b); E.append(float(e)); F.append(np.asarray(g)[:s.n_atoms])
    np.savez(out, positions=pos, cell=np.eye(3) * L, energy=np.array(E), forces=np.array(F) / 1.0,
             site_q=SITE_Q, site_gamma=np.array(GAMMA), source=np.zeros(NCELL, int), time_ps=np.arange(NCELL) * 0.5)
    print(f"teacher: {NCELL} cells of {NMOL}, energy per molecule {np.mean(E) / NMOL * 627.5:+.3f} +- "
          f"{np.std(E) / NMOL * 627.5:.3f} kcal/mol", flush=True)
train = W.e4_structures(out, pins=True)
scale = T.energy_scale(train)
res = {}
for name in RUNGS:
    base, _, unit = name.partition("@")
    rc, _ = E4.rungs_for(base, cfg, train[0].cell)
    if unit:
        rc = replace(rc, q_unit=float(unit))
    t0 = time.time()
    params, info = T.fit(rc, train, seed=0, steps=STEPS, lr=3e-3, w_energy=100.0, w_force=1.0, batch_size=4,
                         lbfgs_steps=0, lr_final_ratio=0.01, scale=scale, force_mode="rigid", verbose=False)
    fl = E4.final_loss(params, rc, train, scale, 100.0, 1.0, 4, "rigid")
    src = E4.learned_sources(params, rc, train) if E4.learns_sources(rc) else {}
    hist = [round(h[1], 1) for h in info["history"]]
    res[name] = {"final": fl, "sources": src, "hist": hist, "time": time.time() - t0}
    print(f"{name:6s} final {fl['total']:.3e} (E {fl['energy']:.2e}, F {fl['forces']:.2e})  "
          + (f"dipole {src['dipole_debye'][0]:.3f} +- {src['dipole_debye'][1]:.3f} D, H side {src['dipole_to_H_side']:.2f}, "
             f"q_O {src['q_O'][0]:+.3f}, q_H {src['q_H'][0]:+.3f}" if 'dipole_debye' in src else "")
          + f"  hist {hist}  [{time.time() - t0:.0f} s]", flush=True)
