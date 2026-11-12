# E4, job 5: the results

*10–12 Nov 2026. Job 5 ran as pre-registered (`e4-job5-preregistration.md`) on Niflheim. The matching level ran at
d6c801f, the v0.0.29 patch's tree: `run_e4_job5.sbatch`, one GPU task of 27 minutes (17:07–17:35 CET on 10 Nov). The
training ran on 11–12 Nov as eight GPU tasks.*

*§1, the matching level, is read from `scripts/e4_report.py --job5 results/e4_job5` and the task's log. D′'s training
predictions were registered after this read and before any training run (pre-registration §8). §2, the training, was
read on 12 Nov.*

## 1. The matching level

### (a) The bands

| | reading (held-out cells unless said) | registered | Niflheim | local | |
|---|---|---|---|---|---|
| **P1** | D′'s χ(k₁); χ(k₂) | +0.8 to +2.0 %; −1.0 to +0.3 % | **+1.32 %; −0.38 %** | +1.44; −0.32 | holds |
| **P2** | held out less calibration, k₁; k₂ | ±0.5 points | −0.03; +0.03 | +0.09; +0.08 | holds |
| **P3** | ε₀ less arm C's; the cells' dipole at k = 0 less arm C's | < 10⁻⁶; ≤ +1.0 point | 0; +0.10 | 0; +0.39 | holds |
| **P4** | the molecules less arm C's, k₁; k₂; at k₂ less arm D's | ≤ +1.5; ≤ +2.5; < 0 points | +0.48; +1.08; −0.68 | +0.72; +1.76; −0.57 | holds |
| **P5** | the far probes less arm C's, 10 / 12 / 15 Å; T5's slope less arm C's; the probes' objective, stage 3 ÷ stage 2 | ≤ +0.3 points each; ±0.005; 0.95–1.15 | +0.02 / −0.15 / −0.08; −0.0002; 1.023 | −0.04 / −0.16 / −0.10; −0.0001; 1.016 | holds |
| **P6** | D′'s change at contact against arm C (rigid forces); the same ÷ arm D's; its orders ≥ 1 moved ÷ arm D's | ≤ 0.30 kcal/mol/Å rms; ≤ 0.70; ≤ 0.70 | **0.156; 0.48; 0.46** | 0.216; 0.51; 0.40 | holds |
| **P7** | arm D's χ(k₁); χ(k₂); arm D's and Dx's change at contact against arm C | job 4's +1.33; −0.37 %; 0.30–0.60 and 0.30–0.70 rms | +1.33; −0.37; 0.325 and 0.442 | (job 4's local +1.40; −0.46); 0.422 and 0.485 | holds |

**All seven hold.**

### (b) The four read-outs on Niflheim

These are 30 calibration cells and 10 held out, and the 30 test frames of 64 molecules at contact. ε₀ is 1.6893 for all
four (MB-pol's 1.6885).

| reading | arm C | arm D | Dx | **D′** |
|---|---|---|---|---|
| χ(k₁); χ(k₂) | −1.84; −10.96 % | +1.33; −0.37 % | +3.32; +6.29 % | **+1.32; −0.38 %** |
| the same on the calibration cells | −1.92; −11.33 % | +1.35; −0.40 % | +3.20; +5.77 % | +1.35; −0.40 % |
| the molecules at k₁; k₂ | 3.8; 5.5 % | 4.6; 7.3 % | 5.2; 10.1 % | 4.3; 6.6 % |
| the cells' dipole at k = 0 | 1.5 % | 2.3 % | 2.5 % | 1.6 % |
| the far probes against arm C's, 10 / 12 / 15 Å | — | +0.18 / +0.33 / +0.21 points | — | +0.02 / −0.15 / −0.08 |
| T5's slope (rmse) | 1.0405 (4.4 %) | 1.0401 (4.1 %) | 1.0401 (4.2 %) | 1.0403 (4.3 %) |
| the probes' objective, stage 3 ÷ stage 2 | — | 0.563 | 0.525 | 1.023 |
| the finite-k rows after the stage (6.651×10⁻⁴ before) | — | 1.050×10⁻⁵ | — | 1.032×10⁻⁵ |
| at contact, the read-out's own energy; rigid forces | −3.674 kcal/mol; 2.454 kcal/mol/Å rms | −3.861; 2.595 | −3.875; 2.689 | −3.729; 2.506 |
| the change at contact against arm C: energy; rigid forces | — | −0.187; 0.325 | −0.200; 0.442 | **−0.054; 0.156** |
| the orders ≥ 1 moved, ÷ arm C's norm | 0 | 1.354 | 1.238 | 0.622 |

Arm D's far probes are job 4's D5; Dx's are not in the report's bands. Energies at contact are per molecule.

### (c) Reading

**(i) D′ reaches the ceiling on 30 cells as on 10, and its χ is arm D's to 0.01 point.**

- Held out it reads +1.32 / −0.38 % against arm D's +1.33 / −0.37 %.
- On the calibration cells both read +1.35 / −0.40 %: the same γ-only least-squares point.
- Its finite-k rows end a little below arm D's (1.032 against 1.050 ×10⁻⁵).
- So holding the 328 training probes costs the fit nothing.

**(ii) It changes the read-out at contact half as much as arm D does.**

- On the 30 test frames the rigid forces move 0.156 rms against arm D's 0.325: 0.48 of it, against 0.51 on the 3 local
  frames.
- The energy moves −0.054 against −0.187 kcal/mol per molecule, 0.29 of arm D's. Locally it was 0.09, so "a tenth" in
  the pre-registration's §3b was the local frames' figure.
- Its orders ≥ 1 move 0.46 of arm D's (0.62 against 1.35 of arm C's norm).
- Every arm's change at contact is smaller on the 30 frames than on the 3 local ones: arm D 0.325 against 0.422, Dx
  0.442 against 0.485. The ratios hold.
- Per unit of coefficient change, D′ and arm D move the contact alike: 0.25 and 0.24 rms per arm C's norm. Dx, the re-fit
  alone, moves it 0.36.

**(iii) Every other reading is closer to arm C's than arm D's is.**

- The cells' dipole at k = 0 is +0.10 point from arm C's, against arm D's +0.84.
- The molecules at finite k are +0.48 / +1.08 points, against arm D's +0.80 / +1.76.
- The far probes are +0.02 / −0.15 / −0.08 points at 10 / 12 / 15 Å, against arm D's +0.18 / +0.33 / +0.21. T5's slope
  is 1.0403 against arm C's 1.0405.
- The probes' matched objective is held at stage 2's (×1.023). Arm D's re-fit lowered it to 0.56 and Dx's to 0.53, while
  their far test probes got worse: the re-fit over-fits the training cores, here as locally.

**(iv) Dx reproduces job 4's swing.** It reads +3.32 / +6.29 % (job 4: +6.3 % at k₂).

**Next.** P1–P7 hold, so D′ trains (pre-registration §5). Its predictions T0′–T3′ were committed at 15:19 CET on 11 Nov, before
`run_e4_job5_train.sbatch` was written (§8 there).

- T0′: every run's read-out readings are the matching job's.
- T1′ and T2′: the frozen form and the ablation each +0.5 to +4.0 % above arm C's.
- T3′: both at or below arm D's.

`scripts/e4_report.py --job2 results/e4_job2a,results/e4_job2b,results/e4_job2c,results/e4_job4/edge2.0,results/e4_job5
--job1 results/e4_job1` reads them.

## 2. Training

*Read 12 Nov from `scripts/e4_report.py --job2 results/e4_job2a,results/e4_job2b,results/e4_job2c,results/e4_job4/edge2.0,
results/e4_job5 --job1 results/e4_job1` and the tasks' accounting (`sacct`).*

*There are eight runs: the frozen form M_16C2†ᵐ_us and its ablation M_16C2†ᵐ, four seeds each, on D′'s read-out. Arm
C's runs are job 2c's and arm D's are job 4's. Every comparison is paired by seed. Forces are rigid-body rmse in
kcal/mol/Å on the unseen frames pooled, as in jobs 2c and 4. The task logs and the driver's summary were not read, so
there are no losses here.*

### (a) The predictions

| | reading | registered | Niflheim | |
|---|---|---|---|---|
| **T0′** | before training, all eight runs: the held-out liquid's ε_∞; χ(k₁); χ(k₂) | the matching job's 1.6893; +1.32 %; −0.38 %, to the printed digit | 1.6893; +1.32 %; −0.38 % on every run | **holds** |
| | after training: ε_∞, T5 and the far probes | as before training | unchanged. The liquid's largest change in training is 0. T5's largest spread over the eight runs is 1×10⁻⁷ points. The far probes are 2.2 / 2.0 / 2.7 % at 10 / 12 / 15 Å, against the matching's 2.23 / 1.98 / 2.66 | **holds** |
| **T1′** | the frozen form, D′ against arm C, at 64 / 128 / 256 molecules | +0.5 to +4.0 % at every size | **−0.6 / −0.7 / −0.7 %**; 3 of 4 seeds at or below arm C's at every size | **fails low** |
| **T2′** | the ablation, the same | +0.5 to +4.0 % at every size | **−0.6 / −1.0 / −1.1 %**; 4 of 4 seeds at or below at every size | **fails low** |
| **T3′** | D′ against arm D, both rungs | at or below it on at least 3 of 4 seeds, at every size | the frozen form −7.5 / −6.9 / −6.7 %, the ablation −7.4 / −7.1 / −7.1 %; every seed below, at every size | **holds** |
| **M′** | the model rule: D′'s frozen form within +1.0 % of arm C's at every size | — | −0.6 / −0.7 / −0.7 % | **D′'s frozen form is the model** |

Every split reads the same:

- the frozen form: −0.5 / −0.6 % at 64 molecules (valid / test), −0.5 / −0.7 % at 128, −0.8 / −0.6 % at 256;
- the ablation: −0.5 / −0.8 %, −1.0 / −1.4 %, −0.8 / −1.4 %.

**The registered reading of this branch** (pre-registration §8, T1′ failing low): the re-fit carried arm D's cost, and
the finite-k fix is free in the bulk. By the model rule, D′'s frozen form is the model: arm C's bulk, with χ right at k₁
and k₂.

### (b) Beside them

These are the report's readings, not registered. They are paired by seed where the report pairs them. The rest are
changes of the bulk table's seed means.

- **Net forces and torques**, both lower:
  - the frozen form's net forces −1.0 / −1.1 / −0.8 % and torques −0.8 / −0.8 / −1.0 %;
  - the ablation's net forces −0.1 / −0.6 / −0.7 % (paired: 2, 4 and 4 of 4 seeds at or below) and torques −0.9 /
    −1.2 / −1.4 %.
- **The field term's gain** is 23.3 / 22.7 / 22.7 %, against arm C's 23.3 / 22.9 / 23.0 % and arm D's 23.2 / 22.9 /
  23.0 %. It is level.
- **Atomic forces, which are not trained**, are arm C's:
  - the frozen form 7.33 / 7.34 / 7.29 against 7.32 / 7.33 / 7.29;
  - the ablation 3.96 / 3.97 / 3.97 against 3.97 / 3.99 / 3.99.

  Arm D had moved the frozen form's by +8.3 to +8.5 %.
- **Energies are level with arm C's**: the frozen form −1.5 / +2.2 / +0.3 % and the ablation −0.7 / +0.6 / −1.9 %
  (paired). Arm D's were +5.5 to +15.2 %. J5 holds for both rungs (unseen frames, z +0.9 and −0.5).
- **Against the runs without the read-out's response.**
  - D′'s frozen form is below M_16†ᵐ_us (the dagger with the field term) by 3.6 / 4.1 / 4.3 %, on 3 of 4 seeds at each
    size. Arm C's is below it by 3.0 / 3.4 / 3.6 %.
  - D′'s ablation is below job 1's dagger by 2.2 / 3.1 / 3.2 % (3 of 4 seeds). Arm C's is below it by 1.5 / 2.1 /
    2.1 %.
- **The far probes after training.**
  - At 10–15 Å they are as before training.
  - At 8 Å they move with E₀: 2.3 ± 0.2 % for the frozen form and 3.2 ± 1.0 % for the ablation, against arm C's 2.5 ±
    0.3 and 3.3 ± 1.7 %.
- **The readings fixed in training, beside arm C's.**
  - T5 is 4.31 % (slope 1.0403) against 4.35 % (1.0405).
  - The cells' dipole is 1.6 % against 1.5 %.
  - The molecules' dipole and field response are arm C's to the printed digit: 5.1 %, 2.9 %, iso 1.5 %.
  - χ is +1.32 / −0.38 % against −1.84 / −10.96 %. The molecules at k₁ / k₂ are 4.3 / 6.6 % against 3.8 / 5.5 %.

### (c) What the readings say

**(i) The finite-k fix is free in the bulk, and arm D's cost was its re-fit.**

- D′ keeps arm D's χ and trains to arm C's bulk, or just below it, on both rungs, at every size and on every split.
  Everything else training reads is arm C's: the field term's gain, the atomic forces, and the energies to within
  ±2 %.
- D′ and arm D share the finite-k rows. They differ in what D′ leaves out: the re-solve of the orders ≥ 1 on the far
  probes, with its ridge toward zero.
- So arm D's 6–7 % came from that re-fit. Job 4's (c)(iv) left open whether the finite-k response itself costs the
  bulk. It does not.
- No Dx run was trained, so the re-fit's own cost is inferred from the difference, not measured.

**(ii) The size of a change at contact did not price it, this time within one kind.**

- T1′ and T2′ scaled arm D's cost by D′'s 0.48 share of its change at contact, on the ground that D′ and arm D are one
  kind: per unit of coefficient change they move the contact alike, 0.25 and 0.24 rms per arm C's norm. That priced
  +1.5 to +3.6 %. The result is −0.6 to −1.1 %.
- The registration called the re-fit "if anything the costlier part". It was all of the cost.
- At matching level, what separated the two was the far probes, not the contact.
  - Arm D's re-fit lowered the training probes' objective to 0.56 of stage 2's while its far test probes got worse
    (+0.2–0.3 points). It over-fitted the training cores.
  - D′ held the objective (×1.02), and its far test probes improved (−0.15 to +0.02 points).
  - In training, the over-fitted read-out cost the bulk and the held one did not. That is two cases, not a rule.

**(iii) The response pays its way again, a little more than arm C's.** The frozen form is 3.6–4.3 % below the dagger
with the field term, against arm C's 3.0–3.6 %. The ablation is 2.2–3.2 % below job 1's dagger, against 1.5–2.1 %.

**(iv) E4's model is D′'s frozen form.** This is M_16C2†ᵐ_us at the 2.0 Å mutual edge, with D′'s read-out
(`configs/e4_job5.yaml`; the anchored third stage, cached in `results/e4_job5`). Its readings:

- rigid forces 0.541 / 0.551 / 0.552 kcal/mol/Å at 64 / 128 / 256 molecules, against arm C's 0.545 / 0.555 / 0.557;
- energies 0.024 / 0.019 / 0.010 kcal/mol per molecule;
- ε_∞ 1.6893, with MB-pol's at 1.6885;
- χ +1.32 / −0.38 % at k₁ / k₂, against arm C's −1.8 / −11.0 %;
- the far probes 2.2 / 2.0 / 2.7 % at 10 / 12 / 15 Å;
- T5 4.3 %.

### (d) Notes to the pre-registration

- **The predictions** were committed at 15:19 CET on 11 Nov (6b7f0d1 on GitHub). That was before
  `run_e4_job5_train.sbatch` and `run_e4_job5_train_4h.sbatch` were written and before either ran.
- **The runs.**
  - They ran as eight tasks on the 4-hour GPUs (`run_e4_job5_train_4h.sbatch`, `train --task i`), not as two tasks of
    four seeds. The pairing is unchanged: `fit` draws E₀'s initialisation and the batches from the seed alone.
  - Wall times were 2:57–3:35 for the frozen form and 3:11–3:40 for the ablation. The ablation's tasks start by solving
    its anchored third stage, unless another task has already cached it. The longest task finished 20 minutes inside
    the limit.
- **The report's count is seeds at or below arm C's.** It is 3 of 4 for T1′ and 4 of 4 for T2′, at every size. The
  report does not print each seed's change, so T1′ and T2′ are read on the change of the seed means, as registered.
- **J4's line in the report** ("FAILS on the runs present") is job 2's registered order of the gains, which failed
  there. D′ does not enter it.

### (e) Next

1. **The model is D′'s frozen form** (`configs/e4_job5.yaml`). Everything downstream reads it.
2. **S0.4 is registered on arm C's polarisation**, as before. The model step reads D′'s beside it from its cache. Now
   that D′ is the model, that is the reading of the model's own response, though not a registered one.
3. **Proposed, not registered: the molecules' finite-k rows beside the cells'.** This was job 4's registered next step
   for this branch. D′'s molecules at k₁ / k₂ are 4.3 / 6.6 %, against arm C's 3.8 / 5.5 %. The cells' χ is right; the
   single molecules' finite-k response is 0.5–1.1 points worse. Whether that is worth a job is Vadic's call against
   item 4.
4. **Then the rest of `e4-plan.md`**: the readers on the model, Form I and the L₀ sweep.
