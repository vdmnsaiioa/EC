# E4, job 10: following the valley — job 7's end states polished with L-BFGS — pre-registration

*21 Nov 2026, registered at 19:05 CET (this commit's time), before any run of job 10 and before any of its numbers.
Vadic's go of 21 Nov ("let's do a first"). Read against:*

- *`e4-job9-results.md` §2–§4 and §9 (a): the valley, the pull, the paths; this job is option (a) there;*
- *`e4-job7-results.md` §2 and §4: the scan (the profile) and the warm starts' end states, this job's starts.*

## 1. Why

**Job 9's finding.**

- With E₀ held, the training objective along the learned charges' overall scale has a curvature 7,000–19,000 times the
  profile's. The profile is the objective with E₀ re-trained at each scale: job 7's scan.
- The charges' Coulomb band contributes 40–100 times the residual. E₀ cancels it, exactly only at the scale it was
  trained with.
- Adam, on batches of six frames, crawls along that valley at its learning rate's pace. Its batch-averaged pull has the
  profile's sign, buried 50–100 times under the batch noise the same stiffness makes.
- **The 50 full-batch L-BFGS steps of the polish moved all four runs toward the data's scale.** They covered 2–9 % of the
  distance (§8).

**The question.** Is the obstacle the optimiser alone? If so, a curvature-aware optimiser on every training frame,
given enough steps, walks the valley to the data's scale from both sides.

**Two hypotheses.**

- **H_O, the optimiser.** The valley's floor leads to the profile's minimum, c* = 1.098 (2.047 D). Full-batch L-BFGS
  follows it: the charges keep moving toward c*, from both sides, and arrive.
- **H_M, minima of its own.** The joint objective has minima away from c*: E₀'s basins around the charges it grew up
  with. A full-batch optimiser stalls short of c*, at a scale that still depends on the start.

**What separates them.** Whether the off-scale starts keep moving toward c* over 1000 L-BFGS steps (K10.1), and
whether the six warm starts end at one scale (K10.2), c*'s (K10.3).

## 2. What changes (v0.0.34), fixed before any run

- **A start from a trained run** (`train.fit(params_init=...)`, `training.start_from`).
  - Each run starts from the parameters of the same rung and seed in a results directory: here job 7's,
    `results/e4_job7/<rung>_seed<seed>.params.pkl`.
  - It does not start from the seed's initialisation.
  - The directory's json must record the same model as the rung, or the run is refused.
  - The objective and the learned charges at the start are recorded (`training.start`).
- **The learned charges traced through the L-BFGS polish** (`train.fit`'s `lbfgs_trace_every`, `polish_lbfgs`'s
  trace; `training.trace.lbfgs_every`).
  - The learned dipole on 4 training frames, and the objective on every training frame, every 25 L-BFGS steps and
    after the last.
  - The trace changes nothing in the polish. The tests check this.
- **The objective along the charges' scale, with the rest held,** is read at the start and the end, as in job 9
  (`e4_water.charge_diagnostics`, the label "start" when there is no Adam).
- **`configs/e4_job10.yaml`:** job 7's settings, with these changes:
  - no Adam;
  - L-BFGS for 1000 steps (job 7's polish, chunks and memory, 20 times as long);
  - `start_from` `results/e4_job7`;
  - the trace;
  - the rungs M_1w050, M_1w100 and M_1w150 (job 7's warm starts) and M_1q110 (the control);
  - seeds 0 and 1.
- **The control, M_1q110:** job 7's O/H charges pinned at c = 1.1, the scan's best. Polished the same way, it shows
  what the long polish does to E₀ with no charge scale to find.
- **The readers.**
  - `scripts/e4_job10.py` reads K10.1–K10.4, without JAX.
  - `scripts/e4_sumrules.py read --job10_dir` reads K10.5, on model files labelled `M_1w<c0>lb_s<seed>`, which
    leave job 7's alone.

## 3. The runs

`run_e4_job10.sbatch` has two tasks of four runs, one GPU each:

- **0:** M_1w050, M_1w150, M_1w100 and M_1q110, seed 0;
- **1:** the same, seed 1.

Each warm start then writes its charges on S0's frames (`e4_sumrules.py model --params`), for K10.5. Then:

```sh
python scripts/e4_job10.py --out_dir results/e4_job10 --job7_dir results/e4_job7 --job6_dir results/e4_job6
python scripts/e4_sumrules.py read --gate_dir results/e4_job6 --job7_dir results/e4_job7 --job10_dir results/e4_job10
python scripts/e4_water.py summary --out_dir results/e4_job10
```

## 4. Registered

**The target** is job 7's profile minimum, from `scripts/e4_job9.py`'s parabola through the scan's final training
loss: c* = 1.098, 2.047 D.

**The readings.** The learned dipole on the test frames of 64 molecules; rigid forces on the unseen frames (job 7's
pooling); seed means over seeds 0 and 1. The starts are job 7's runs of the same rung and seed.

| | reading | band | prediction |
|---|---|---|---|
| **K10.1** | the four off-scale starts (M_1w050, M_1w150): the share of each one's distance to the target covered by the polish, (d_end − d_start) / (2.047 − d_start) | at least 0.25 on each of the four | holds (H_O): the valley leads to c* |
| **K10.2** | the six polished warm starts (M_1w050, M_1w100, M_1w150): their learned dipoles | every one within ±15 % of their mean, and every final loss at or below M_∞'s largest (74.7) | holds |
| **K10.3** | their mean against the target, 2.047 D | within ±10 %; read when K10.2 holds | holds |
| **K10.4** | their mean forces against the polished control's (M_1q110) | at most +7 % at every size | holds |
| **K10.5** | their \|b_T\| on S0's first shells of 128 and 256 molecules | 1.22–1.40 at both shells on at least 4 of 6; read when K10.2 holds | holds |

**Why these bands.**

- **K10.1's 0.25.** The polish's 50 steps in job 9 covered 2–9 % of the distance (§8). At that pace, 1000 steps would
  cover 36–180 %. A quarter allows the pace to fall by about a third and still separates moving from stalled. Under
  H_M the charges stop within the first steps, as Adam's did at a stationary point.
- **K10.2's ±15 %** is J7.4's and K9.3's band. M_∞'s largest final loss is from job 6's and job 7's runs.
- **K10.3's ±10 %** is K9.4's.
- **K10.4's +7 %** is J7.5's and K9.5's. It is read against the polished control, not job 7's, so that what the long
  polish does to E₀ by itself, better or worse, is in both.
- **K10.5** is J7.6's band and count.
- **No reading here rests on a slope at the end state.** Job 9's K9.1 did, and the valley's curvature made it read the
  end state's offset from the valley's bottom.

**The least certain prediction is K10.2.** 1000 steps may move the charges most of the way without their arriving
within ±15 % of each other: the case "K10.1 holds, K10.2 fails" of §6.

## 5. Read beside, not registered

- **Every run before and after the polish:**
  - the training objective (its start is job 7's final loss);
  - forces on the training frames and on the unseen ones (whether a long polish on every training frame over-fits);
  - the learned dipole.
- **The control's own change:** what 1000 steps do to E₀ with no charge scale to find.
- **The paths:** the learned dipole and the objective every 25 steps (printed every 250).
- **Along the charges' scale with the rest held,** at the start and the end:
  - the curvature against the profile's;
  - the slope against the profile's;
  - the minimum.
- **The polished objective at each run's end scale,** against the profile's there, L*(c_end).
- **The energies, the probes, T5.**

## 6. What would change the reading

- **K10.1–K10.3 hold.** H_O: the obstacle was the optimiser.
  - Learned charges reach the data's scale from both sides once polished on every frame. The no-pins recipe is Adam,
    then a long full-batch L-BFGS.
  - K10.4 says whether they then fit as well as the pinned charges at that scale, and K10.5 whether the scale is the
    lemma's.
- **K10.1 holds and K10.2 fails.** Moving, not arrived. The traces give the pace, and so the steps it would take.
- **K10.1 fails.** The full-batch polish stalls short of c*: H_M, minima of the joint objective away from the data's
  scale. Next would be job 9's option (b), the valley made shallower by the charges' band damped at contact, or
  training along the profile.
- **K10.1 fails on one side only.** The valley's floor is not level. The side that moves tells which way it tilts.
- **K10.4 fails with K10.2 and K10.3 holding.** Learned charges at the right scale fit worse than pinned ones there: the
  learned shape, or the head, costs.
- **The control.**
  - If the long polish lowers its unseen forces by more than a few per cent, a part of any gain in the warm starts is
    the polish's alone. K10.4 already compares against it.
  - If it lowers the training frames' forces and raises the unseen ones, the polish over-fits. Read the same in the
    warm starts.

## 7. Cost

- **An L-BFGS step** is one or two value-and-gradient evaluations on every training frame, in chunks with
  rematerialisation: about 30 of Adam's batch-of-six steps each.
- **1000 steps** are about 15 times job 7's 3000 Adam steps: about 1.5–2 h a run, with the readings and diagnostics.
- **Two tasks of four runs:** 6–9 h each. The sbatch asks 14 h, as job 8's did, and a run already done is skipped on
  resubmission.

## 8. Numbers before any run

**The starts** (job 7's ends; learned dipole on the test frames of 64 molecules):

| run | dipole (D) | its distance to 2.047 D | final loss | forces at 64 |
|---|---|---|---|---|
| M_1w050 seed 0 / 1 | 1.214 / 1.315 | +0.833 / +0.732 | 46.1 / 48.8 | 0.767 / 0.792 |
| M_1w100 seed 0 / 1 | 1.927 / 1.941 | +0.120 / +0.106 | 24.8 / 30.1 | 0.594 (mean) |
| M_1w150 seed 0 / 1 | 2.461 / 2.562 | −0.414 / −0.515 | 38.2 / 39.8 | 0.696 / 0.711 |
| M_1q110 (control) | 2.05, pinned | — | 27.5 (mean) | 0.592 (mean) |

**Job 9's 50-step polish** moved the four off-scale runs (dipole on the traced frames, from the end of Adam to the end
of training):

- M_1w050 by +0.015 and +0.045 D: 2 % and 6 % of the distance;
- M_1w150 by −0.041 and −0.012 D: 9 % and 2 %.

**The profile at the starts' scales** (§4's parabola): L*(0.65) = 47.2, L*(0.71) = 42.6, L*(1.32) = 33.3, L*(1.37) =
35.6, against the starts' final losses of 46.1, 48.8, 38.2 and 39.8. At c* the profile is 28.85. Job 7's warm starts at
1.0, already near c*, reached 24.8 and 30.1.
