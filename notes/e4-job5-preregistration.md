# E4, job 5: arm D′, arm D's third stage anchored at arm C — pre-registration

*10 Nov 2026, v0.0.29. Vadic's go of 10 Nov, after job 4's training read (`e4-job4-results.md` §2): D′ at matching
level, then training if it reads. The local master of this note is `notes/` in the repository.*

*When each part was written:*

- *The expectations (§3a) were committed at 14:06 CET on 10 Nov (`scratch/e4_job5_expectations.md`, its own commit).
  That was before the anchored stage was solved on any liquid frame. Seen before them:*
  - *the anchored solve's tests on synthetic clusters and cells;*
  - *job 4's numbers.*
- *§3's numbers were computed after that, on job 3's local frames.*
- *§4's bands for Niflheim were written after §3's numbers, before the job ran.*
- *§8, the training predictions, was written after Niflheim's matching level was read (`e4-job5-results.md` §1)
  and before the training sbatch existed.*

## 1. The question

Job 4's arm D brought the cells' finite-k susceptibility to the frozen form's ceiling (χ +1.33 / −0.37 % at k₁ / k₂).
It cost the bulk 6–7 % on both rungs and every seed, and the field term absorbed none of it (`e4-job4-results.md`
§2). Its third stage did two things at once:

- it re-solved the orders ≥ 1 on the far probes beside the calibrated order 0, with the ridge pulling toward zero. That
  is what the control Dx does alone;
- it added the finite-k rows.

The training did not separate the two. Before training, the change at contact put most of arm D's change in the first:
Dx moved the read-out's forces in the liquid 0.49 kcal/mol/Å rms from arm C's, and arm D 0.42.

Job 5 asks, at matching level, before any training:

- **(i)** Anchored at arm C, does the third stage still reach the ceiling on held-out cells? Here the ridge pulls toward
  arm C's orders ≥ 1, and the far probes are held at arm C's own predictions.
- **(ii)** How much does it then change at contact, against arm D?
- **(iii)** What does it cost the far probes, T5, the molecules at finite k and the cells' dipole at zero field?

If (i) holds, D′ trains. That separates arm D's two moves: if D′ trains to arm C's bulk, arm D's cost was the re-fit.

## 2. Design

**Arm D′ (the label Da in the code).** It is arm C's read-out, then arm D's third stage anchored at arm C.

- It keeps arm D's stage otherwise: the orders ≥ 1 columns alone (3168 coefficients), the training probes' rows, the
  finite-k rows (12 χ_mode a calibration cell, relative to MB-pol's mean at the harmonic, weight 0.1) and stage 1's
  ridge.
- **The ridge pulls toward arm C's orders ≥ 1 instead of zero:** (A + λI)c = r + λc_C.
- **The training probes' rows are held at arm C's own predictions** instead of MB-pol's energies. They then resist any
  change of what arm C predicts for those probes.
- So the orders ≥ 1 move from arm C's values only where the finite-k rows need it. The stage is linear in them, so one
  Gauss–Newton step solves it.
- The far test probes (10, 12 and 15 Å), T5 and the liquid are not in the stage. They are read on it.

**Read beside it**, all at the 2.0 Å mutual edge:

- arm C, the reference;
- arm D and the control Dx, from job 4's caches.

**A new reading: the change at contact.** On the 30 test frames of 64 molecules (3 locally), it gives:

- each read-out's own energy per molecule and rigid forces, the model with E₀ and the field term at initialisation, less
  the same with the read-out switched off;
- the change of both against arm C's;
- the read-out's orders ≥ 1 coefficients' change, relative to arm C's norm.

This is `scratch/e4_job2_calibration_orders.py`'s reading, now inside `kread`. Arm B's change was 0.47 rms, arm C's
0.33 and arm D's 0.42; its size did not price the training cost (`e4-job4-results.md` §2c(ii)). It is read here to
measure what the anchoring changes, not to predict the training.

**The run.** `run_e4_job5.sbatch` is one GPU task:

- it copies job 4's caches at 2.0 Å into `results/e4_job5`;
- it solves D′ from arm C's stage 2, which is cached as `..._liq<h>_k<h>.pkl` with its own hash;
- it reads all four read-outs and their change at contact.

That is about an hour. Locally, on two CPU cores, it took 40 minutes.

**The code (v0.0.29).**

- `sseft/linear.py`, `solve_readout(anchor=, hold=)`:
  - anchor: the ridge toward the start. The step is formed from the residuals, (A + λI)(c − c₀) = g − λ(c₀ − c_start),
    so A c₀ is not cancelled against r;
  - hold: the energy and force rows' targets are the starting model's own values;
  - `.data_history` is the objective on data's own targets, for the records.
- `scripts/e4_water.py`:
  - `training.readout.finite_k.anchor`. Settings without it hash as before, so arm D's cache carries over;
  - the anchored stage in `match_readout`. Its record's objective is the probes' against MB-pol, as arm D's is, and the
    held objective is kept beside it as `finite_k_held_objective`;
  - `read_contact` with `kread.contact: {reference, frames}`;
  - an arm's cache directory is made if it is new.
- `scripts/e4_report.py --job5`: the bands P1–P7.
- `configs/e4_job5.yaml`: job 4's config with `finite_k.anchor: true`, and `kread`'s four arms in one directory.
- Tests:
  - `tests/test_arm_da.py`, six tests. One is the whole `kread` path on fake cells, probes and responses: arms C, D and
    Da, and the change at contact;
  - `tests/test_e4_report.py::test_job5_report`.

## 3. The local numbers

These are job 3's local frames: 10 calibration cells and 3 held out, and 3 test frames of 64 molecules for the change
at contact (`scratch/e4_job5_local.yaml`; readings `scratch/e4_job5_local.json` and `.log`). Held-out cells unless said.

| reading | arm C | arm D | Dx | **D′** |
|---|---|---|---|---|
| χ(k₁); χ(k₂) | −1.90; −11.49 % | +1.40; −0.46 % | +1.74; +0.66 % | **+1.44; −0.32 %** |
| the same on the calibration cells | −1.85; −11.16 % | +1.36; −0.41 % | +2.13; +2.17 % | +1.36; −0.41 % |
| the frames' sd of χ(k₂) | 0.29 points | 0.07 | 0.83 | 0.05 |
| the molecules at k₁; k₂ | 3.7; 4.7 % | 4.6; 7.1 % | 4.8; 8.8 % | 4.4; 6.5 % |
| ε₀ | 1.68948 | 1.68948 | 1.68948 | 1.68948 |
| the cells' dipole at k = 0 | 3.1 % | 4.1 % | 4.4 % | 3.5 % |
| far probes, 10 / 12 / 15 Å | 2.27 / 2.21 / 2.73 % | 2.59 / 2.71 / 2.89 % | 2.76 / 2.77 / 2.96 % | **2.23 / 2.05 / 2.63 %** |
| T5 slope | 1.0396 | 1.0390 | 1.0390 | 1.0395 |
| the probes' objective, stage 3 ÷ stage 2 | — | 0.53 | 0.50 | 1.016 |
| the change at contact against arm C: energy per molecule; rigid forces | — | −0.218 kcal/mol; 0.422 rms | −0.237; 0.485 | **−0.019; 0.216** |
| the read-out's own rigid forces at contact | 2.465 kcal/mol/Å rms | 2.699 | 2.716 | 2.564 |
| its orders ≥ 1 moved, ÷ arm C's norm | 0 | 1.42 | 1.30 | 0.58 |

The far probes are the non-additive energy's rmse ÷ rms. D′'s training probes are held to 2.6×10⁻⁸ against the stage's
probe objective of 1.7×10⁻⁵, which is 0.15 % of it.

### (a) The expectations against the numbers

| | expected | read | |
|---|---|---|---|
| 1 | χ within ±0.5 points of arm D's +1.40 / −0.46 % | +1.44 / −0.32 % | holds |
| 2 | held out within 0.5 points of calibration | +0.09 / +0.08 points | holds |
| 3 | ε₀ arm C's to every digit; the cells' dipole at most +1.0 point | the same; +0.39 | holds |
| 4 | the molecules within +0.9 / +2.3 points of arm C's | +0.72 / +1.76 | holds |
| 5 | the far probes within ±0.1 point of arm C's; T5 ±0.005; the probes' objective 0.9–1.3 | −0.04 / **−0.16 / −0.10**; −0.0001; 1.016 | **fails at 12 and 15 Å, by an improvement** |
| 6 | the change at contact 0.05–0.25 rms, at most 0.6 of arm D's; the orders ≥ 1 moved less than arm D's | 0.216, 0.51 of arm D's; 0.58 against 1.42 | holds |
| 7 | arms C, D and Dx as job 4's, to the printed digit; arm D 0.422, Dx 0.485 at contact | as job 4's; 0.422, 0.485 | holds |

Six of the seven hold. Expectation 5 fails at 12 and 15 Å because the far probes got better than arm C's by more than
the ±0.1 point expected.

### (b) What the numbers say

**(i) The anchored stage reaches the ceiling, and holding the probes costs it nothing.**

- On the calibration cells, D′ reads arm D's +1.36 / −0.41 % to the printed digit: both reach the γ-only least-squares
  point.
- Its finite-k rows end at 1.029×10⁻⁵ against arm D's 1.038×10⁻⁵.
- On the held-out cells it reads +1.44 / −0.32 %, with the smallest frame scatter of the four (0.05 points).
- So the direction the finite-k response needs is one the training probes do not see. This is Lemma 3 again: the far
  probes do not identify the orders ≥ 1.

**(ii) It changes the read-out half as much at contact, and its energy a tenth as much.**

- The rigid forces move 0.216 rms against arm D's 0.422, and the energy −0.019 kcal/mol per molecule against −0.218.
- The orders ≥ 1 move 0.58 of arm C's norm against arm D's 1.42.
- The rest of arm D's change was the re-fit on the probes.

**(iii) The far probes improve.**

- D′'s far test probes are 0.04 / 0.16 / 0.10 points better than arm C's at 10 / 12 / 15 Å. Arm D's were 0.3 / 0.5 / 0.2
  points worse, and Dx's 0.5 / 0.6 / 0.2.
- Arm D and Dx re-fitted the training probes (their objective fell to 0.5 of stage 2's), and the test probes got worse.
  That re-fit over-fits the 16 training cores.
- D′ holds the training probes, and the test probes get better. So the direction the finite-k response needs is right
  for the far field too.

**(iv) The rest lies between arm C's and arm D's.**

- The molecules at finite k are 4.4 / 6.5 %, against arm D's 4.6 / 7.1 % and arm C's 3.7 / 4.7 %.
- The cells' dipole at k = 0 is 3.5 %, against 4.1 % and 3.1 %.
- ε₀ and T5 are arm C's.

**(v) What only training can tell.**

- D′'s change at contact is still half of arm D's: 0.22 rms, against arm C's 0.33 from arm A, which gained.
- The size of such a change does not price its cost (`e4-job4-results.md` §2c(ii)).
- So whether the finite-k fix itself costs the bulk is read only in training.

## 4. Registered for Niflheim

The bands are for Niflheim's frames: 30 calibration cells and 10 held out, and 30 test frames of 64 molecules for the
change at contact. Held-out cells unless said. `scripts/e4_report.py --job5 results/e4_job5` reads them.

| | reading | band | local |
|---|---|---|---|
| **P1** | D′'s χ(k₁); χ(k₂) | +0.8 to +2.0 %; −1.0 to +0.3 % | +1.44; −0.32 |
| **P2** | held out less calibration, k₁; k₂ | ±0.5 points each | +0.09; +0.08 |
| **P3** | ε₀ less arm C's; the cells' dipole at k = 0 less arm C's | < 10⁻⁶; ≤ +1.0 point | 0; +0.39 |
| **P4** | the molecules less arm C's, k₁; k₂; the molecules at k₂ less arm D's | ≤ +1.5; ≤ +2.5; < 0 points | +0.72; +1.76; −0.57 |
| **P5** | the far probes less arm C's, 10 / 12 / 15 Å; T5's slope less arm C's; the probes' objective, stage 3 ÷ stage 2 | ≤ +0.3 points each; ±0.005; 0.95–1.15 | −0.04 / −0.16 / −0.10; −0.0001; 1.016 |
| **P6** | D′'s change at contact against arm C (rigid forces); the same ÷ arm D's; the orders ≥ 1 moved, ÷ arm D's | ≤ 0.30 kcal/mol/Å rms; ≤ 0.70; ≤ 0.70 | 0.216; 0.51; 0.40 |
| **P7** | arm D as job 4's: χ(k₁); χ(k₂); arm D's and Dx's change at contact against arm C | +1.325 to +1.335 %; −0.375 to −0.365 %; 0.30–0.60 and 0.30–0.70 rms | (job 4's local +1.40; −0.46, read to the digit); 0.422; 0.485 |

- **P1 and P2** are D1 and D2's bands for arm D. P1's upper ends are raised by 0.1 point for the local difference
  between D′ and arm D (+0.04 / +0.14 points).
- **P5** asks only that the far probes are not worse than arm C's by more than 0.3 points. The local improvement is
  not registered.
- **P6's** 30 frames may move the rms against the 3 local frames' by about a fifth. The bands allow that.
- **P7** is the check that nothing but the anchoring changed. Its χ parts are job 4's Niflheim readings, so locally
  they are read against job 4's local ones (`tests/test_e4_report.py::test_job5_report`).

## 5. What would change the reading

- **All seven hold.** D′ trains. Its predictions T0–T2 are registered after this read and before training (§8, to
  come), as job 4's were.
- **P1 or P2 fails.** Holding the probes conflicts with the finite-k fix on 30 cells, so the far probes see γ there.
  D′ does not train. The next variant is the anchor without the hold.
- **P6 fails, and P1–P5 hold.** The finite-k rows themselves move the contact as much as arm D's whole change did, and
  the re-fit was not the larger part on Niflheim. D′ still trains. It is then the finite-k fix with no re-fit, and its
  cost in training is the fix's own.
- **P5 fails on the far probes.** The anchoring holds the training probes and not the test ones. D′ trains only if P6
  holds.
- **P7 fails.** Something besides the anchoring changed between job 4 and job 5. That is found before P1–P6 are read.

## 6. Training (after §4's read)

D′'s frozen form and its ablation, four seeds each, with the read-out held. This is job 4's training with D′'s
read-out: two GPU tasks of four seeds, about 26 GPU hours. The predictions are registered after §4's read, from the
numbers on Niflheim.

## 7. Readings beside them, not registered

- Every read-out's own energy and rigid forces at contact, and Dx's change.
- D′'s far probes against arm D's (−0.36 / −0.66 / −0.27 points locally).
- The molecules' correlation with MB-pol's at finite k: 0.9990 at k₂ for D′, against 0.9983 for arm D and 0.9989 for
  arm C.
- The frames' sd of χ(k₂), arm by arm.
- The held objective of D′'s stage (`finite_k_held_objective`): how far the training probes move.

## 8. Training: registered 11 Nov, 15:19 CET (this commit's time), after the matching level and before any training

*Written after Niflheim's matching level was read (all seven bands hold; `e4-job5-results.md` §1) and before
`run_e4_job5_train.sbatch` was written. Arm C's trained readings are `e4-job2c-results.md`'s; arm D's are
`e4-job4-results.md` §2's.*

**What D′ changes before training, against arm D.** These are Niflheim's 30 test frames of 64 molecules.

- The read-out's rigid forces in the liquid move 0.156 kcal/mol/Å rms from arm C's: 0.48 of arm D's 0.325.
- Its energy moves −0.054 kcal/mol per molecule, 0.29 of arm D's −0.187.
- Its orders ≥ 1 move 0.46 as far as arm D's.
- Per unit of coefficient change, D′ and arm D move the contact alike (0.25 and 0.24 rms per arm C's norm). Dx, the
  re-fit alone, moves it 0.36.
- Its χ is arm D's to 0.01 point. Every other reading is closer to arm C's than arm D's is.

**The predictions.** They are paired by seed, on the unseen frames pooled, as for jobs 2c and 4.
`scripts/e4_report.py --job2` reads them with job 5's directory as a fifth.

| | reading | registered |
|---|---|---|
| **T0′** | before training, every run of both rungs: the held-out liquid's ε_∞ and χ on the matched read-out | the matching job's D′: ε_∞ 1.6893, χ +1.32 / −0.38 %, to the printed digit |
| | ε_∞, T5 and the far probes after training | as before it |
| **T1′** | the frozen form, D′ against arm C: rigid forces, seed means, at 64, 128 and 256 molecules | +0.5 to +4.0 % at every size |
| **T2′** | the ablation, D′ against arm C: the same | +0.5 to +4.0 % at every size |
| **T3′** | D′ against arm D, both rungs | at or below it on at least 3 of 4 seeds, at every size |

**The model rule, registered with them.** D′'s frozen form becomes E4's model if it is within +1.0 % of arm C's at every
size (seed means, unseen frames). Otherwise arm C stays the model, and D′ is kept as the variant with the finite-k
response right.

Beside them, not registered: the field term's gain (arm C's and arm D's 23 %), the energies, and D′ against the runs
without the read-out's response.

**Why these.**

- **T1′ and T2′: arm D's cost, scaled by D′'s share of the change.**
  - Job 4 showed that the size of a change at contact does not price it across kinds of change: arms B, C and D gave
    three outcomes in no order of their size.
  - D′ and arm D are one kind. Both move the orders ≥ 1 with the same finite-k rows, and they move the contact alike
    per unit of coefficient change.
  - So scale arm D's cost (+6.4 to +7.5 %) by D′'s share of its change at contact, 0.48. Linearly that is +3.1 to
    +3.6 %. In quadrature, as an added error independent of the rest, it is +1.5 to +1.8 %.
  - The part D′ leaves out is the re-fit. Per unit of coefficient change it moves the contact most, and it made the far
    test probes worse while it fitted the training probes. If anything it was the costlier part.
  - Hence +0.5 to +4.0 %, centred near +2 %.
- **Both rungs alike.** Arm D cost both rungs the same fraction, and the field term absorbed none of it. Nothing in D′'s
  change is of the k = 0 kind the field term took from arm B. So T1′ and T2′ share their band.
- **T3′.** D′ has arm D's finite-k fix without the re-fit, and moves the contact half as much.

**What would change the reading.**

- **T1′ fails low:** the frozen form within +0.5 % of arm C's, or below it. The re-fit carried arm D's cost, and the
  finite-k fix is free in the bulk. By the model rule D′'s frozen form is then the model: arm C's bulk, with χ right at
  k₁ and k₂.
- **T1′–T3′ hold.** The finite-k fix itself costs the bulk, about in its share of the change at contact. Arm C stays the
  model, unless D′ is within the model rule's +1.0 %. The next question is whether the fix can be made in directions the
  liquid's forces do not see: D″, with the read-out's forces in the calibration cells held as rows too.
- **T1′ or T2′ fails high, and T3′ holds.** The fix costs more than its share of the change at contact. Arm C stays,
  and D″ is next, as above.
- **T3′ fails.** The re-fit was not where arm D's cost came from, and the finite-k fix costs the bulk however it is
  anchored. Arm C stays the model. The frozen form's finite-k response at this edge is then recorded as a limit of the
  read-out.

**The run.** `run_e4_job5_train.sbatch` is two GPU tasks, one per rung, with the four seeds in sequence, as job 4's.

- The frozen form's D′ read-out comes from the matching job's cache.
- The ablation's third stage is solved once, by its first seed, from arm C's cached stages, then cached for the other
  three.
- About 26 GPU hours, as arms C and D.
