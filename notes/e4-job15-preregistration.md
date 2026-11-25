# E4, job 15: the learned charges' scale after job 14 — pre-registration

*25 Nov 2026, registered at this commit's time, before any run of job 15 and before any of its numbers. Vadic's go of 25
Nov ("prepare all of these experiments", the five proposals of `e4-job14-results.md` §10). Read against:*

- *`e4-job14-results.md`: the profile barely sharpens with E₀'s reach; the scale moves two to three times further with
  one layer; at 4.5 Å it settles at 0.97–0.99 from both sides, at 6 Å it is still converging;*
- *`e4-job13-results.md` §4 and §8: by species alone the scale moves less under Adam; the scale freezes as E₀ trains;*
- *`e4-job6-results.md`: learned charges' scale followed their unit (L1.2).*

## 1. Why

Job 14 left five questions.

- **A. Where is the pinned minimum at 4.5 Å, and is the learned charges' lower end their head's?** The four scan points
  put c* anywhere from 1.006 to 1.11 depending on the curve. The flexible learned charges end at 0.969–0.987.
- **B. Does depth alone give one scale?** At 6 Å (one layer against job 7's two of 6 Å) three of the four ends were
  still moving toward each other after 1000 steps.
- **C. Is it depth or reach?** Job 14 changed both at 4.5 Å. Two layers of 3 Å have the reach of one layer of 6 Å and
  the depth of job 7's E₀.
- **D. Can the scale be found shallow and kept deep?** If a one-layer E₀ finds the scale and a second layer is added
  after, the model may get both the scale and the depth's accuracy.
- **E. Does MACE-LES freeze its scale too?** It has two message-passing layers of 6 Å in the same 12.4 Å cells. Job 14
  predicts that it does.

## 2. What changes (v0.0.40), fixed before any run

**In the harness:**

- **`e4_water.grown` and `training.start_from.grow`.** A run starts from the same rung's run with one E₀ block fewer.
  - The new block has its radial weights and its update's last layer at zero. E₀'s blocks are residual, so it adds
    nothing: the grown model is the source model, energies and forces exactly (tested to 10⁻¹²; exactly 0 on CPU).
  - The gradient reaches the new block's radial weights and update.
  - The driver checks the grown start's objective against the source json's to 10⁻⁸ (relative) and stops the run
    otherwise. The source's record must be the rung's at one layer fewer.
- **The scan resolved:** M_1q090, M_1q095, M_1q105, M_1q120, M_1q130, the pinned O/H charges at c = 0.9, 0.95, 1.05,
  1.2, 1.3 (as job 7's M_1q rungs).
- **Six configs:**
  - `e4_job15s_a.yaml`, `e4_job15_a.yaml`: job 14's 4.5 Å settings, for the new scan points and job 13's warm starts
    by species alone (M_1z050, M_1z150: the charge head held).
  - `e4_job15_b.yaml`: job 14's 6 Å polish, continuing job 14's 6 Å warm starts for 1000 more steps.
  - `e4_job15s_c.yaml`, `e4_job15_c.yaml`: job 14's settings at two layers of 3 Å, job 7's scan and warm starts.
  - `e4_job15_d.yaml`: B's warm starts, and job 14's 6 Å M_1q110 as the control, grown to two layers of 6 Å (job 7's
    E₀), then polished 1000 steps.

**For MACE-LES (`baselines/mace`):**

- **`les.yaml` writes out MACE-LES's unit for the latent charges, `output_scale: 0.1`** (`MACELES.les_output_scale`,
  its default; not trained). `les_o050.yaml` and `les_o150.yaml` set it to 0.05 and 0.15.
- **`train.sh`** holds the baseline's training call and hyper-parameters, moved unchanged from `train.sbatch` (which
  now calls it).
- **`eval.py` reads the latent charges' molecular dipole** (each molecule's net charge spread evenly over its atoms, its
  H atoms taken to their O by the minimum image; debye) and the mean latent charge on O and H. LES's charges are in e:
  its Ewald sum has 1/(4πε₀) = 14.40 eV Å.
- **The release:** the baseline's trained run continued from its last checkpoint (`--restart_latest`) with the unit
  changed. Its latent charges are scaled by exactly 0.5 or 1.5 at the restart (checked on a CPU smoke run: 2×10⁻¹⁶).

**The reader:** `scripts/e4_job15.py` reads K15.1–K15.10, without JAX or PyTorch.

## 3. The runs

**`run_e4_job15.sbatch`, arms A–D, eleven tasks:**

- 0, 1: A, the new scan points, seeds 0 and 1;
- 2: A, by species alone, both seeds;
- 3–6: B then D, one warm start and seed each;
- 7: D, the grown M_1q110, both seeds;
- 8, 9: C, the scan, seeds 0 and 1;
- 10: C, the warm starts, both seeds.

**`run_e4_job15_les.sbatch`, arm E, six tasks** (MACE-LES at 64 molecules, seeds 0 and 1, the baseline's
hyper-parameters):

- 0, 1: the baseline (E1 at the default unit, `results/e4_baselines/maceles_n64_s<seed>`), then its three releases at
  0.05, 0.1 and 0.15 (E2), 150 epochs each;
- 2–5: E1 from scratch at 0.05 and at 0.15.

A smoke run first: `SMOKE=1 sbatch --array=0 -p sm3090el9_devel -t 00:30:00 run_e4_job15_les.sbatch`.

Then:

```sh
python scripts/e4_job15.py
python scripts/e4_water.py summary --out_dir results/e4_job15_a     # and _c, _d
```

## 4. Registered

**The readings.**

- **The end of a run with learned charges is c,** its learned dipole on the test frames of 64 molecules over 1.864 D.
  For MACE-LES it is the latent charges' molecular dipole there over 1.864 D.
- **The gap** is a seed's end from 1.5 less its end from 0.5.
- **The resolved profile (A)** has nine points, the seed means of the final training loss at c = 0.7, 0.9, 0.95, 1.0,
  1.05, 1.1, 1.2, 1.3, 1.45 (job 14's four, job 15's five). c*_res is the minimum of the parabola in c through the five
  points around the lowest. It must lie inside them.
- **Forces** are seed means of the rigid forces on the unseen frames, at 64, 128 and 256 molecules.

| | arm | reading | band | prediction |
|---|---|---|---|---|
| **K15.1** | A | c*_res against the largest of job 14's four flexible ends at 4.5 Å (0.987) | at least 0.03 above | holds |
| **K15.2** | A | every species-only end | within ±0.05 of c*_res | fails |
| **K15.3** | A | the species-only gap, seed by seed | at most half of job 13's at 12 Å (0.715, 0.717) | holds |
| **K15.4** | B | \|the gap\| after 2000 steps, both seeds | at most 0.05 | holds |
| **K15.5** | C | \|the gap\|, both seeds | at most 0.15: "reach"; at least 0.25: "depth"; else "mixed" | reach |
| **K15.6** | D | every grown warm start's end against its start (B's end) | within ±0.05 | holds |
| **K15.7** | D | the grown warm starts' forces against B's, both starts, every size | at least 5 % lower | holds |
| **K15.8** | D | the grown M_1q110's forces against job 10's M_1q110 (0.520 / 0.536 / 0.541) | within ±3 %, every size | holds |
| **K15.9** | E1 | MACE-LES's latent dipole at units 0.05 and 0.15 against the default's, seed by seed | within ±10 % | fails |
| **K15.10** | E2 | released at 0.05 and 0.15 against the continued control's, seed by seed | within ±10 % | fails |

**Why these bands, and these predictions.**

- **K15.1.** At every reach L(1.1) < L(1.0) by more than the seeds' spread, so a convex profile has its minimum above
  1.0. A parabola in c² puts it at 1.07, the cubic at 1.11.
  - 0.03 is a little more than the four flexible ends' spread (0.018).
  - Predicted: c*_res at 1.04–1.09, so the learned charges' minimum lies below the pinned one.
- **K15.2.** By species alone the model is the pinned model with one free scale, so its own minimum is c*_res.
  - The ±0.05 band is jobs 11–14's.
  - Predicted to fail from below. At 12 Å the species offsets alone moved the scale 4–7 times less than the head under
    Adam (job 13 §4).
- **K15.3** asks only whether the short reach helps the species-only charges as it helped the flexible ones. Half is a
  loose band: job 14 cut the flexible gap by a factor of ten or more.
- **K15.4.** 0.05 is the band of K14.3. In job 14's last 250 steps, the pairs closed by 0.03 (seed 0) and 0.06 (seed 1).
- **K15.5.** The two edges are job 14's 6 Å gaps (0.01–0.10, read with margin as 0.15) and its 12 Å gaps (0.29–0.32,
  read with margin as 0.25).
  - The prediction is weak. Within one layer the shorter cutoff was the more mobile (4.5 Å against 6 Å), which points
    to reach.
  - Two layers of 3 Å also see little of the first shell's O···O (2.8 Å, where the cutoff's taper leaves 1 %).
- **K15.6.** ±0.05 again. At 12 Å the polish moved a frozen scale 0.05–0.07 in 1000 steps (job 13's R), and the grown
  E₀ starts trained.
- **K15.7.** Depth is worth 8–13 % in forces at the same c (job 14 §6, the pinned scan). 5 % is about half of that,
  reached in 1000 steps.
- **K15.8.** A grown start should do no worse than job 7's training. ±3 % is about the seed spread of the polished
  pinned runs at 12 Å.
- **K15.9.** Job 6 found our learned charges' scale followed their unit (M_1u against M_1uc). With Adam the charges'
  absolute step is proportional to the unit, so a smaller unit should leave the charges smaller when MACE's layers
  commit.
- **K15.10.** Released from a trained state, a deep model moved a scale 0.05–0.07 in c in 1000 steps (job 13's R). The
  release runs in MACE's stage two, at its lower learning rate.
  - Predicted: the released dipoles end nearer their perturbed start than the control. The share of the way back is
    below one half.

## 5. Read beside, not registered

- **A's whole profile:** every curve through the nine points, the curvature, the species-only first stages and paths.
- **C's profile and c*,** its curvature against 12 Å's and 6 Å's; C's first stages and paths (Adam's move against job
  14's 6 Å and 12 Å); the forces at two layers of 3 Å.
- **B's paths:** the last 250 steps' move, the objective.
- **D's paths, and the grown warm starts against job 10's M_1w100** (learned, at c*, polished: 0.491 / 0.507 / 0.510).
- **MACE-LES:**
  - its forces and energies at 64, 128 and 256 molecules (the baseline's first reading);
  - its latent charges by species and per molecule;
  - its latent dipole against 1.864 D and against our learned ends;
  - E2's share of the way back.

## 6. What would change the reading

- **A.**
  - K15.1 and K15.2 hold: by species the learned charges find the pinned minimum, and the flexible ones' lower end is
    their head's. The head's environment dependence changes the optimal mean dipole.
  - K15.1 holds and K15.2 fails: the head is what moves the scale (job 13), and without it the short reach is not
    enough.
  - K15.1 fails: the pinned minimum is where the flexible ends are, and job 14's K14.3 holds in substance.
- **B.**
  - K15.4 holds: one layer against two at the same cutoff separates a scale found from a scale frozen.
  - K15.4 fails: one layer of 6 Å slows short of one scale; the cutoff matters as well as the depth.
- **C.**
  - "Reach": what freezes the scale is the reach over the cell, not message passing.
  - "Depth": message passing freezes it at any reach.
  - "Mixed": both.
- **D.**
  - K15.6 and K15.7 hold: the route works. Find the scale with a shallow E₀, then add depth.
  - K15.6 fails: the deep E₀ moves the scale away. Read where it goes.
  - K15.7 fails: depth added late does not pay in 1000 steps.
  - K15.8 fails: growing is a worse start than training from scratch.
- **E.**
  - K15.9 and K15.10 fail: MACE-LES's scale is set by its unit and its history, as ours with two layers. The claim of
    `les-vs-eft-what-is-provable.md` §5 then stands on the model it is about.
  - Either holds: MACE-LES finds its scale, and the depth story does not carry to it. Read why: its readout, its
    smearing (σ = 1 Å), its stage two.

## 7. Not tested here

- **The cell:** training at 128 molecules.
- **LES without message passing** (the water model of the LES papers).
- **MACE-LES with one interaction layer.**

## 8. Cost

- **The harness, about 35 GPU hours:**
  - A: 14 runs at 4.5 Å, 0.4–0.6 h each, stage and polish;
  - B: four 6 Å polishes, 1.2–1.6 h;
  - C: 12 runs at two layers of 3 Å, not timed on the GPU. On the CPU an objective and gradient cost what two layers
    of 6 Å cost (6.2 s against 6.0 s on 4 cells of 64); on the GPU, job 14's short cutoffs ran far cheaper than the
    CPU suggested.
  - D: six 12 Å polishes, 1.7–2.4 h.
- **MACE-LES, not timed on the GPU:** two trainings of 500 epochs, four more, and six releases of 150 epochs. It asks
  24 h a task.

## 9. Numbers before any run

- **At 4.5 Å (job 14):**
  - the profile 47.47 / 35.34 / 33.42 / 63.55 at c = 0.7 / 1.0 / 1.1 / 1.45;
  - c* 1.006 (parabola in c), 1.069 (in c²), 1.112 (cubic);
  - the flexible ends 0.974, 0.987 (from 0.5), 0.975, 0.969 (from 1.5).
- **At 12 Å (job 13), by species alone:** the ends 0.632 / 0.625 (from 0.5) and 1.348 / 1.342 (from 1.5); the gaps
  0.715 / 0.717.
- **At 6 Å (job 14), B's starts:**
  - the ends 0.963 / 0.932 (from 0.5) and 0.975 / 1.035 (from 1.5); the gaps 0.012 / 0.104;
  - the forces from 0.5 0.679 / 0.680 / 0.683, from 1.5 0.565 / 0.573 / 0.577;
  - M_1q110 0.565 / 0.582 / 0.584.
- **At 12 Å:** job 10's M_1q110 0.520 / 0.536 / 0.541, and M_1w100 0.491 / 0.507 / 0.510.
- **The grown model** reproduces its source to 0 in energies and forces on CPU test cells (M_1w150 and M_1q110).
- **On CPU, the objective and gradient on 4 cells of 64 (M_1q110):** one layer of 4.5 Å 4.7 s; two layers of 3 Å 6.2 s;
  one of 6 Å 4.1 s; two of 6 Å 6.0 s. E₀ has 20,032 parameters with one layer and 38,272 with two.
- **MACE-LES** has not been run on E4's data. The smoke runs (CPU, random labels) checked the pipeline end to end:
  training, the release from the checkpoints, the evaluation's latent dipoles.
