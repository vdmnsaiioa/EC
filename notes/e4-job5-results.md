# E4, job 5: the results

*10–11 Nov 2026. Job 5 ran as pre-registered (`e4-job5-preregistration.md`) on Niflheim at d6c801f, the v0.0.29 patch's
tree: `run_e4_job5.sbatch`, one GPU task of 27 minutes (17:07–17:35 CET on 10 Nov).*

*§1, the matching level, is read from `scripts/e4_report.py --job5 results/e4_job5` and the task's log. D′'s training
predictions were registered after this read and before any training run (pre-registration §8). §2, the training, is
to come.*

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

*To come.*
