# E4, job 9: why the learned charges stop short, and the charge head's own learning rate — pre-registration

*21 Nov 2026, registered at 02:01 CET (this commit's time), before any run of job 9 on MB-pol's data and before any of
its numbers. Vadic's go of 20 Nov. Read against:*

- *`e4-job7-results.md` §2 and §4 (the scan; the warm starts stopped short), §8 (this job is J7.4's registered branch);*
- *`e4-job6-results.md` §2 and §4 (the learned scale followed the path).*

## 1. Why

**Job 7's outcome.**

- **The data set a field-blind scale.** The scan of pinned O/H charges is lowest at c = 1.1 in forces and in the
  training objective. A parabola through the objective puts its minimum at c* = 1.098, 2.047 D (§8).
- **Learned charges do not reach it.** Started at 0.5 and 1.5 times the O/H charges, they moved 25–45 % of the way and
  stopped. Where they stopped depended on the start.
- **On synthetic cells they did.** Where MB-pol's charges are the only signal, the same settings converged from both
  sides in 600 steps.
- **Where they stopped, E₀ had adapted to them.** Their final losses lie near the scan's curve at their own scale: 46–49
  at c ≈ 0.68 against the profile's 43–47, and 38–40 at c ≈ 1.35 against 33–36.

**The puzzle.**

- Suppose the rest of the model is at its best for the charges it has. Then the objective's slope along the charges'
  overall scale, with the rest held, equals the slope of the profile: the objective with E₀ re-trained at each scale,
  which is what the scan measured (the envelope theorem).
- The profile is steep where the runs stopped: −52 per unit of scale at c = 0.68, and +54 to +68 at 1.32–1.37 (§8).
- So something hid that pull from Adam, or stopped the charges following it.

**Two hypotheses.**

- **H_N, noise.**
  - The pull is there, but on a batch of six frames it is small against the batch-to-batch noise of E₀'s short-range
    errors.
  - Adam moves a parameter by about its learning rate times the gradient's mean over its rms. A noisy direction moves
    at a fraction of the learning rate, and the cosine schedule's decay to 1 % then freezes it.
  - Job 7's travel was 6–10 % of Adam's budget in the species offsets (§8). Job 6's M_1uc also stopped well inside its
    budget, as H_N expects.
- **H_L, lock-in.**
  - E₀'s two layers reach 12 Å, the whole training cell. While E₀ fits, it absorbs what the charges get wrong and builds
    a basin around them.
  - The run then sits at a stationary point of its own. The slope along the charges' scale, with the rest held, is near
    zero, though the profile is steep.

**What separates them.** Two diagnostics, neither of which changes the training:

- **R**, the slope along the charges' scale with the rest held, on every training frame at the end of Adam, over the
  profile's slope at the run's scale. H_N predicts about 1, H_L about 0.
- **The pull's batch-to-batch ratio**, |mean| / sd over Adam's second half. H_N predicts it small.

**The fix job 7 registered** (§6 there: "J7.4 fails, with J7.3 holding") is the charge head's own learning rate.

- Under H_N it multiplies the charges' travel.
- Under H_L it makes the charges the fast variable. They may reach the data's scale before E₀ builds its basin. It is
  not the cure designed for H_L.

## 2. What changes (v0.0.33), fixed before any run

- **The charge head's own learning rate** (`train.fit(charge_lr_scale=k)`).
  - The charge head's layers and its species offsets (`model.charge_labels`) train at k times the learning rate, on
    the same cosine schedule.
  - The rest of the model's updates are the default's, digit for digit. The tests check this.
- **The diagnostics** (`train.fit`'s `trace_fn`, `scale_gradient` and `snapshot_adam`; `e4_water.charge_diagnostics`).
  None of them changes the training.
  - **The trace:** the learned dipole on 4 training frames, every 100 Adam steps from the start.
  - **The pull:** on every Adam batch, the derivative of the batch loss along the charges' overall scale s, from the
    gradient (`model.charge_scale_slope`).
    - The scale is `model.scale_learned_charges`: the charge head's last layer and the species offsets times s.
    - That multiplies every learned charge by s exactly. The head's last layer is linear, and the neutrality shift is
      linear in it.
  - **The objective along that scale, with the rest held,** on every training frame, at s = 0.5 to 1.5, at the end of
    Adam and at the end of training.
    - Its slope at s = 1 is taken from s = 0.99 and 1.01.
    - The minimum is that of the quartic through the grid. The objective is exactly quartic in s: E₀ does not read the
      charges, and the Coulomb band is quadratic in them.
- **The rungs.**
  - **M_1x050 and M_1x150:** job 7's M_1w050 and M_1w150 with the charge head at 10 times the learning rate
    (`e4_water.CHARGE_LR`).
  - **M_1x000:** the same started near zero, from job 6's M_1uc start (unit 0.15 e, no species offsets, the head's
    random start).
  - **M_1w050 and M_1w150 again,** traced: they give K9.1 and K9.2.
- **`configs/e4_job9.yaml`:** job 7's settings, the five rungs, seeds 0 and 1, and `training.trace` (every 100 steps,
  4 frames).
- **The readers.**
  - `scripts/e4_job9.py` reads K9.1–K9.5, without JAX. Job 7's scan is its profile.
  - `scripts/e4_sumrules.py read --job9_dir` reads K9.6.
- **Everything else is job 7's.**
  - The 64-molecule cells.
  - Adam for 3000 steps (batch 6, learning rate 3×10⁻³, cosine to 1 %), then L-BFGS for 50.
  - The loss weights, the rigid forces, the charges' unit of 0.15 e.
  - The readings on every frame of 64–256 molecules, the probes and T5.

## 3. The runs

`run_e4_job9.sbatch` has two tasks of five runs, one GPU each:

- **0:** M_1w050 and M_1w150, seeds 0 and 1; M_1x000 seed 0.
- **1:** M_1x050 and M_1x150, seeds 0 and 1; M_1x000 seed 1.

Each M_1x run then writes its charges on S0's frames (`e4_sumrules.py model --params`), for K9.6. Then:

```sh
python scripts/e4_job9.py --out_dir results/e4_job9 --job7_dir results/e4_job7 --job6_dir results/e4_job6
python scripts/e4_sumrules.py read --gate_dir results/e4_job6 --job7_dir results/e4_job7 --job9_dir results/e4_job9
python scripts/e4_water.py summary --out_dir results/e4_job9
```

## 4. Registered

**The profile.**

- Job 7's scan: the seed means of the final training loss at c = 0.7, 1.0, 1.1 and 1.45 (43.1, 31.1, 27.5, 40.3).
- The parabola through them is L*(c) = 91.31 c² − 200.54 c + 138.94, with its minimum at c* = 1.098 (2.047 D).
- A run whose learned dipole is c times the permanent 1.864 D has the profile's slope p = c L*′(c) along s.
- The reader fits the parabola from job 7's jsons.

**The readings.** Rigid forces on the unseen frames (job 7's pooling), seed means over seeds 0 and 1.

| | reading | band | prediction |
|---|---|---|---|
| **K9.1** | job 7's warm starts again (M_1w050, M_1w150; four runs): R, the slope along the charges' scale at the end of Adam with the rest held, over p at the run's dipole | R in [0.5, 2] on at least 3 of 4 | holds (H_N): the pull is there |
| **K9.2** | the same runs: the pull's \|mean\| / sd over the batches of Adam's second half (steps 1500–2999) | at most 0.3 on at least 3 of 4 | holds (H_N): it is buried in noise |
| **K9.3** | the warm starts at 10 times the charge head's learning rate (M_1x050, M_1x150; four runs): the learned dipole on the test frames of 64 molecules | every one within ±15 % of their mean, and every final loss at or below M_∞'s largest (74.7) | holds |
| **K9.4** | their mean dipole against the profile's minimum (2.047 D) | within ±10 %; read when K9.3 holds | holds |
| **K9.5** | their mean forces against the scan's lowest (M_1q110: 0.592 / 0.606 / 0.612) | at most +7 % at every size | holds |
| **K9.6** | their \|b_T\| on S0's first shells of 128 and 256 molecules | 1.22–1.40 at both shells on at least 3 of 4; read when K9.3 holds | holds |

**The diagnosis, registered with them.**

- **K9.1 and K9.2 hold:** H_N. The pull was there, buried in noise.
- **K9.1 fails low,** with |R| ≤ 0.25 on at least 3 of 4: H_L. The runs are locked at stationary points of their own.
- **K9.1 holds and K9.2 fails:** the pull is there and clear, and Adam did not follow it. Neither hypothesis; read the
  paths.
- **K9.1 fails in any other way** (R above 2, or of the wrong sign): neither; read the paths.

**Why these bands.**

- **K9.1's [0.5, 2].** The profile is the scan's: O/H charges pinned, E₀ trained from scratch at each scale, then
  polished. The run's slope is taken before the polish, with its own learned charges, whose shape varies with the
  environment. A factor of two either way allows for that. 0.25 marks a pull mostly gone.
- **K9.2's 0.3.** Adam's travel along a direction is its budget times about this ratio. Job 7's travel was 6–10 % of the
  budget in the species offsets (§8), so H_N needs a ratio well below 1. The head's last layer also carries the scale,
  so 0.3 allows a factor of three.
- **K9.3's ±15 %** is J7.4's band. M_∞'s largest final loss is from job 6's and job 7's runs.
- **K9.4's ±10 %.** The scan's minimum is at c = 1.07–1.10 in forces and 1.10 in the objective, and the lemma's at
  1.10–1.11. The learned charges' shape may move it a little. 10 % is about the distance from c = 1.0 to c*.
- **K9.5's 7 %** is J7.5's band.
- **K9.6** is J7.6's band, on 3 of the 4 runs (J7.6 asked 4 of 6).
- **K9.3–K9.6 read the four warm starts.** The start near zero is read beside. From zero the charges must also leave a
  stationary point, since the Coulomb band is quadratic in them, and their sign is free.

## 5. Read beside, not registered

- **The start near zero:** its dipole against the warm starts' mean, its sign, its path.
- **Every run's path:** the dipole every 500 steps, and the share of its change in Adam's second half. Under H_N the
  share follows the schedule's: 19 % of Adam's budget lies in the second half (§8).
- **R after the polish,** and **the minimum with the rest held** at the end of Adam and of training. Under H_L the
  minimum sits near s = 1.
- **Job 7's warm starts again, against job 7.** The seeds and code path are the same, so their dipoles and forces should
  match job 7's to the printed digit.
- **The rest:** the fixed runs' forces at every size, energies, probes, T5 and final losses.

## 6. What would change the reading

- **K9.1, K9.2 and K9.3 hold.** H_N, and the fix works.
  - The learned charges reach a scale set by the data (K9.4). K9.6 reads whether it is the lemma's.
  - The no-pins version of E4's model then has charges at the data's scale. The comparison should read MACE-LES's
    charges the same way.
- **K9.1 holds and K9.3 fails.** The pull is there, and ten times is not enough or not stable. Read the paths:
  - still moving at the end: a larger multiplier, or no decay for the charge head;
  - oscillating: the noise amplified; next, larger batches.
- **K9.1 fails low and K9.3 holds.** E₀ locked the charges, but charges ten times faster reach the data's scale before
  E₀'s basin forms.
- **K9.1 fails low and K9.3 fails.** Locked, and a faster head does not help. Next would be training along the
  profile: the scale set from outside, as in the scan, and the shape learned. Or two timescales.
- **K9.3 holds and K9.4 fails.** The learned charges converge, but not to the pinned O/H charges' scale. Their
  environment dependence carries part of what the scale does. K9.6 then says whether it is the lemma's.
- **K9.5 fails with K9.3 and K9.4 holding.** The scale is right and the forces are not: the learned shape costs.
- **The start near zero.**
  - Ends with the warm starts: no one-molecule property is needed, and the comparison needs none.
  - Does not: the stationary point at zero is a second obstacle.

## 7. Cost

- **Ten runs** of 12–20 minutes each, with the diagnostics and the model step. Job 7's warm starts took 8–14 minutes
  with every reading.
- **Two tasks** of 1–1.7 h each, asking 3 h: about 3 GPU hours in all.

## 8. Numbers before any run

**(a) The profile and the pull it predicts** (§4's parabola through job 7's scan; no training):

| c | 0.65 | 0.68 | 0.71 | 1.32 | 1.35 | 1.37 |
|---|---|---|---|---|---|---|
| L*(c) | 47.2 | 44.8 | 42.6 | 33.3 | 34.6 | 35.6 |
| p = c L*′(c) | −53.2 | −51.9 | −50.3 | +53.5 | +62.1 | +68.0 |

Job 7's warm starts ended at c = 0.65–0.71 (from 0.5) and 1.32–1.37 (from 1.5), with final losses of 46.1 and 48.8,
and of 38.2 and 39.8.

**(b) Adam's budget and job 7's travel.**

- Over 3000 steps on the cosine schedule (3×10⁻³ to 1 %), Adam moves a parameter at most 4.55 in total, 81 % of it in
  the first half. At a unit of 0.15 e, that is 0.68 e in the species offsets.
- **The start at 0.5** moved its H charge by 0.050–0.070 e: 0.33–0.46 units, 7–10 % of the budget.
- **The start at 1.5** moved its H charge by 0.043–0.060 e: 0.29–0.40 units, 6–9 % of the budget.
- These count the species offsets only. The head's last layer also moves the charges.
- **From 0.5 to c*** is 0.20 e, 1.3 units. At ten times the learning rate the budget is 45.5 units.
  - If the ratio stayed at job 7's 0.06–0.10, the charges could travel 2.7–4.5 units.
  - That is two to three times the distance, with most of it in the schedule's first half.

**(c) The check on synthetic cells** (`scratch/e4_job9_teacher_student.py . 600 12 32 <runs> 16`, CPU, about 40
minutes a run). The teacher and cells are job 6's and job 7's: MB-pol's rigid site charges alone, through the Coulomb
band, on 12 random cells of 32 molecules. The students train with E4's settings, 16 features, and Adam for 600 steps
with no polish, the charge head at 10 times the learning rate.

| run | final training loss | learned dipole, start → step 100 → end (of the teacher's 1.864 D) | sign |
|---|---|---|---|
| started at 0.5, charge head ×10 | 2.21 | 0.93 → 1.87 → 1.89 D (1.01) | physical |
| started at 1.5, charge head ×10 | 1.58 | 2.80 → 2.03 → 1.93 D (1.03) | physical |
| started near zero, charge head ×10 | 2.17 | 0.10 → 1.15 → 1.74 D (0.93), still rising | flipped |
| job 7's, started at 0.5, ×1 | 1.80 | 0.93 → 1.77 D (0.95) | physical |
| job 7's, started at 1.5, ×1 | 1.65 | 2.80 → 2.05 D (1.10) | physical |
| job 7's O/H charges pinned at c = 1 | 1.09 | 1.864 D | — |

M_∞ reads 6.95 here (job 6's §8).

- **The warm starts converge faster and closer.**
  - At ten times, both are within 10 % of the teacher's dipole by step 100, and end 2 % apart, at 1.01 and 1.03.
  - At job 7's learning rate they ended at 0.95 and 1.10, 15 % apart.
- **Ten times is stable.** The traces settle without oscillation, every 50 steps to the end. The pull over Adam's
  second half averages −0.18 ± 4.6 and +0.41 ± 5.9 per batch: |mean| / sd 0.04 and 0.07 at convergence, where the
  profile is flat.
- **From near zero, the charges leave zero and nearly arrive.** They end 7 % below the teacher, still rising, with the
  sign flipped, which energies and forces cannot fix.
- **The loss is not lower at ten times.** Started at 0.5 it is higher than job 7's (2.21 against 1.80); started at 1.5,
  lower (1.58 against 1.65). Both stay above the pinned charges' 1.09. On 600 steps the charges' scale is no longer
  what limits the loss.
  - So K9.5 reads whether the faster head costs forces on E4's data.
- **The multiplier is fixed at ten.** The synthetic check gives no reason to change it.
