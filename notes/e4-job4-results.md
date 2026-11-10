# E4, job 4: the results

*9 Nov 2026. Job 4 ran as pre-registered (`e4-job4-preregistration.md`) on Niflheim at 71831ca, the v0.0.28 patch's
tree: `run_e4_job4.sbatch`, three GPU tasks, one per mutual edge.*

*Read from `scripts/e4_report.py --job4 results/e4_job4`. That report prints the band readings; the per-frame values
and stage records are in the three `kread_edge*.json`. This note is the matching level (§1). Arm D's training, whose
predictions were registered after this read and before any run (pre-registration §8), is §2, to come.*

## 1. The matching level

### (a) The bands

| | reading (held-out cells unless said) | registered | Niflheim | local | |
|---|---|---|---|---|---|
| **D1** | arm D's χ(k₁); χ(k₂) | +0.8 to +1.9 %; −1.0 to +0.2 % | **+1.33 %; −0.37 %** | +1.40; −0.46 | holds |
| **D2** | held out less calibration, k₁; k₂ | ±0.5 points | −0.03; +0.04 | +0.04; −0.05 | holds |
| **D3** | ε₀ less arm C's; the cells' dipole at k = 0 less arm C's | < 10⁻⁶; ≤ +2.0 points | 0; +0.84 | 0; +1.04 | holds |
| **D4** | the molecules less arm C's, k₁ / k₂; C0's at k₂ less arm D's | ≤ +2.0 / ≤ +3.5 points; > 0 | +0.80 / +1.76; +4.06 | +0.90 / +2.33; +4.31 | holds |
| **D5** | far probes 10 / 12 / 15 Å less arm C's; T5; the probes' objective, stage 3 ÷ stage 2 | ≤ +1.0 each; 0.95–1.10; 0.3–1.0 | +0.18 / +0.33 / +0.21; 1.0401; 0.563 | +0.31 / +0.51 / +0.16; 1.0390; 0.528 | holds |
| **D6** | Dx's χ(k₂) less arm C's; Dx's frame sd of χ(k₂) ÷ arm D's; \|D's χ(k₂)\| less \|Dx's\| | ≥ +6 points; ≥ 3; < 0 | +17.25; 8.2; −5.92 | +12.15; 11.6; −0.21 | holds |
| **D7** | D@1.5: χ(k₁); χ(k₂) | −0.1 to +1.1 %; −0.7 to +0.4 % | +0.44; −0.07 | +0.49; −0.16 | holds |
| | D@2.5: χ(k₁); χ(k₂) | +2.3 to +4.3 %; −2.0 to 0 % | +3.20; −0.96 | +3.30; −1.05 | |
| | H3: χ's edge range, order 0 ÷ arm D, k₁; k₂ | ≥ 2; ≥ 2 | 3.35; 23.2 | 3.32; 23.7 | |
| | H3 out of sample: the molecules' edge range, the same, k₁; k₂ | 1.2–2.5; 1.8–4.0 | 1.72; 2.75 | 1.83; 2.60 | |

**All seven hold.**

### (b) Reading

**(i) Arm D sits where the frozen form's ceiling puts it, on 30 cells as on 10.**

- The γ-only least-squares point, computed before the job from job 3's Niflheim numbers, was +1.32 / −0.39 %. Arm D
  reads +1.33 / −0.37 %.
- The held-out cells read as the calibration cells, to 0.04 points.
- At 1.5 and 2.5 Å it reads within 0.1 points of the local values.

**(ii) The costs are a little smaller than locally.**

- The far probes are 0.2–0.3 points worse at 10–15 Å, and the probes' matched objective falls to 0.56 of stage 2's.
- The cells' zero-field dipole is 0.8 points worse.
- The molecules at finite k are 0.8 and 1.8 points worse. That puts them at 4.6 and 7.3 %, from arm C's 3.8 and 5.5 %.

**(iii) Dx: the far probes alone do not pin the orders ≥ 1, and do not even reproduce.**

- Re-solved on the probes beside the calibrated order-0 sector, Dx's χ(k₂) is +6.3 % on Niflheim, against +0.7 %
  locally.
- Its frame scatter is 8 times arm D's.
- The order-0 sector it is solved beside was calibrated on 30 cells there and 10 here. Small differences in it swing the
  probes' solution by 5.6 points of χ(k₂) (+0.7 → +6.3 %), while arm D moves by 0.1.
- This is arm B's K4 failure (job 3) seen from the other side: data that do not identify the orders ≥ 1 leave them to
  whatever else is in the solve.

**(iv) H3 reads as registered.**

- The edge dependence of χ falls 3.4- and 23-fold, by construction at the fitted k.
- Out of sample, the molecules' falls 1.7-fold at k₁ and 2.7-fold at k₂. H3's ≥ 2 holds at k₂ and fails narrowly at k₁,
  as the local reading registered.

**Next.** D1–D5 hold, so arm D trains (pre-registration §5). Its predictions T0–T2 were committed at 14:40 CET on 9 Nov
(§8 there), before `run_e4_job4_train.sbatch` started.

- T0: every run's read-out readings are the matching job's.
- T1: the frozen form within ±1.5 % of arm C's.
- T2: the ablation +0.5 to +4.5 % above arm C's.

`scripts/e4_report.py --job2 results/e4_job2a,results/e4_job2b,results/e4_job2c,results/e4_job4/edge2.0 --job1
results/e4_job1` reads them.

## 2. Training

*To come.*
