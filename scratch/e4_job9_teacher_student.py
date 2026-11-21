"""E4 job 9's check on synthetic cells (notes/e4-job9-preregistration.md section 8; v0.0.33, CPU, about 40 minutes a
run): job 6's and job 7's teacher and cells (scratch/e4_job7_teacher_student.py: MB-pol's rigid site charges alone
through the Coulomb band, 12 random cells of 32 molecules, 12.5 A) and students trained with E4's settings, the features
reduced to 16, Adam alone, the charge head at its own learning rate (train.fit's charge_lr_scale):

    M_1w@<unit>@<c0>@<k>    learned O/H charges in units of <unit> e, started at c0 times the teacher's O/H charges
                            (job 7's warm start), the charge head's learning rate k times the rest's
    M_1u@<unit>@<k>         the same started near zero (the head's random start, no species offsets: job 6's start)

Printed: the final full-batch training loss, the molecular dipole at the start and the end, its trace every 50 steps,
and the pull on the charges' scale (model.charge_scale_slope) over the second half of Adam, mean and sd over batches.
The teacher has no electrons, so the charges that reproduce it carry its 1.864 D.

    python scratch/e4_job9_teacher_student.py . 600 12 32 M_1w@0.15@0.5@10,M_1w@0.15@1.5@10,M_1u@0.15@10 16
"""
import os, sys, time, json
import numpy as np
REPO = sys.argv[1]
STEPS = int(sys.argv[2]); NCELL = int(sys.argv[3]); NMOL = int(sys.argv[4]); RUNS = sys.argv[5].split(",")
FEAT = int(sys.argv[6]) if len(sys.argv) > 6 else 16
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import jax
from dataclasses import replace
from sseft import model as M, train as T, water as W
import e4_water as E4
import tempfile
out = os.path.join(tempfile.gettempdir(), f"e4_job6_cells_{NCELL}x{NMOL}.npz")      # job 6's cells, labelled once
if not os.path.exists(out):
    raise SystemExit(f"{out} missing: run scratch/e4_job7_teacher_student.py first (it labels job 6's cells)")
cfg = E4.merge(E4.load_config(None), {"model": {"features": FEAT}})
train = W.e4_structures(out, pins=True)
scale = T.energy_scale(train)
q_oh = np.asarray(train[0].pinned["q"], float)[:3]
print(f"O/H charges with the teacher's dipole: O {q_oh[0]:+.4f}, H {q_oh[1]:+.4f} e", flush=True)
for spec in RUNS:
    parts = spec.split("@")
    rc, _ = E4.rungs_for("M_1", cfg, train[0].cell)
    if parts[0] == "M_1w":
        unit, c0, k = float(parts[1]), float(parts[2]), float(parts[3])
        rc = replace(rc, q_unit=unit, q_ref_start=((8, c0 * q_oh[0]), (1, c0 * 0.5 * (q_oh[1] + q_oh[2]))))
    else:
        unit, k = float(parts[1]), float(parts[2])
        rc = replace(rc, q_unit=unit)
    t0 = time.time()
    tracer = E4.charge_tracer(rc, train[:2])
    params, info = T.fit(rc, train, seed=0, steps=STEPS, lr=3e-3, w_energy=100.0, w_force=1.0, batch_size=4,
                         lbfgs_steps=0, lr_final_ratio=0.01, scale=scale, force_mode="rigid", verbose=False,
                         charge_lr_scale=k, trace_every=50, trace_fn=tracer, scale_gradient=True)
    fl = E4.final_loss(params, rc, train, scale, 100.0, 1.0, 4, "rigid")
    src = E4.learned_sources(params, rc, train)
    tr = [(s, round(v["dipole_debye"], 3)) for s, v in info["trace"]]
    g = info["scale_gradient"][STEPS // 2:]
    line = (f"{spec:20s} final {fl['total']:.3e} (E {fl['energy']:.2e}, F {fl['forces']:.2e})  dipole "
            f"{tr[0][1]:.3f} -> {src['dipole_debye'][0]:.3f} +- {src['dipole_debye'][1]:.3f} D, H side "
            f"{src['dipole_to_H_side']:.2f}, q_O {src['q_O'][0]:+.3f}, q_H {src['q_H'][0]:+.3f}; pull over the second "
            f"half {np.mean(g):+.3e} +- {np.std(g):.3e}")
    print(line + f"  [{time.time() - t0:.0f} s]\n    trace {tr}", flush=True)
    with open(os.path.join(tempfile.gettempdir(), "e4_job9_check.jsonl"), "a") as fh:
        fh.write(json.dumps({"run": spec, "steps": STEPS, "final": fl, "end": src, "trace": tr,
                             "pull_second_half": [float(np.mean(g)), float(np.std(g))],
                             "time": time.time() - t0}) + "\n")
