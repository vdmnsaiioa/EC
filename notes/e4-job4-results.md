# E4, job 4: the results

*9–10 Nov 2026. Job 4 ran as pre-registered (`e4-job4-preregistration.md`) on Niflheim at 71831ca, the v0.0.28 patch's
tree. The matching level ran as `run_e4_job4.sbatch`, three GPU tasks, one per mutual edge. Arm D's training then ran
as `run_e4_job4_train.sbatch`, two GPU tasks, one per rung, with the four seeds in sequence.*

*Niflheim's tree also carried a two-line local fix in `matched()`. It makes the cache directory when it is new, which the
sbatch had not done for the 1.5 and 2.5 Å tasks. It was found on 10 Nov, when v0.0.29 was pulled. It changes no number,
and it has been in the repository since 11 Nov.*

*§1, the matching level, is read from `scripts/e4_report.py --job4 results/e4_job4`. That report prints the band
readings; the per-frame values and stage records are in the three `kread_edge*.json`. §2, the training, is read from
`scripts/e4_report.py --job2` with arm D's directory as a fourth. Its predictions were registered after §1's read and
before any training run (pre-registration §8).*

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

*Read 10 Nov from `scripts/e4_report.py --job2 results/e4_job2a,results/e4_job2b,results/e4_job2c,results/e4_job4/edge2.0
--job1 results/e4_job1`. It was run on Niflheim after the matching read's commits (0ea8c64, e87b0a1) were applied there.*

*There are eight runs: the frozen form M_16C2†ᵐ_us and its ablation M_16C2†ᵐ, four seeds each, on arm D's read-out.
Arm C's runs are job 2c's (`e4-job2c-results.md`), and every comparison is paired by seed. Forces are rigid-body rmse
in kcal/mol/Å on the unseen frames pooled, as there. The two task logs were not read, so there are no wall times or
losses here.*

### (a) The predictions

| | reading | registered | Niflheim | |
|---|---|---|---|---|
| **T0** | before training, all eight runs: the held-out liquid's ε_∞; χ(k₁); χ(k₂) | the matching job's 1.6893; +1.33 %; −0.37 %, to the printed digit | 1.6893; +1.33 %; −0.37 % on every run | **holds** |
| | after training: ε_∞, T5 and the far probes | as before training | unchanged. The liquid's largest change in training is 0; T5's largest spread over the eight runs is 6×10⁻⁸ points | **holds** |
| **T1** | the frozen form, arm D against arm C, at 64 / 128 / 256 molecules | within ±1.5 % at every size | **+7.5 / +6.6 / +6.4 %**; every seed above arm C's | **fails high** |
| **T2** | the ablation, the same | +0.5 to +4.5 % at every size, and above on at least 3 of 4 seeds | **+7.3 / +6.6 / +6.4 %**; every seed above | **fails above its band** |
| T3, beside | the field term's gain, 1 − F(frozen form) ÷ F(ablation), at least arm C's | follows from T1 and T2; not registered | 23.2 / 22.9 / 23.0 % against arm C's 23.3 / 22.9 / 23.0 % | level |

Every split reads the same:

- the frozen form: +7.7 / +7.2 % at 64 molecules (valid / test), +7.0 / +6.7 % at 128, +6.3 / +6.7 % at 256;
- the ablation: +7.5 / +7.1 %, +6.7 / +6.4 %, +6.2 / +6.9 %.

**The registered reading of this branch** (pre-registration §8, T1 failing high): the finite-k fix costs the bulk; arm C
stays the model, and the molecules' finite-k rows come next. (c)(iv) qualifies the first clause.

### (b) Beside them

These are the report's readings, not registered. They are paired by seed, except the frozen form's net forces, both
rungs' torques and the atomic forces, which are changes of the bulk table's seed means.

- **Net forces and torques.**
  - The ablation's net forces are +9.4 / +8.0 / +7.7 %, its torques +7.3 / +6.6 / +6.6 %.
  - The frozen form's are +8.8 / +7.3 / +7.3 % and +7.3 / +6.8 / +6.5 %.
  - The field term's gain on each is arm C's to within 0.6 points: on the net forces 21.8 / 21.7 / 20.9 % against 21.4
    / 21.2 / 20.6 %; on the torques 20.5 / 20.3 / 21.3 % against 20.5 / 20.5 / 21.2 %.
- **Atomic forces, which are not trained.**
  - The ablation's are arm C's: 3.98 / 3.99 / 3.99 against 3.97 / 3.99 / 3.99.
  - The frozen form's are +8.3 to +8.5 %: 7.94 / 7.94 / 7.90 against 7.32 / 7.33 / 7.29.
  - Arm B moved the frozen form's by +9 % too against arm A, and the ablation's by +5 %.
- **Energies.**
  - They are worse at every size on both rungs, on every seed, and more at the larger sizes: the frozen form by +5.8 /
    +12.2 / +14.2 %, the ablation by +5.5 / +12.9 / +15.2 %.
  - No bias moves with size: J5 holds for both rungs (unseen, z +0.6 and −0.5).
- **Against the runs without the read-out's response.**
  - Arm D's frozen form is above M_16†ᵐ_us (the dagger with the field term) by 4.3 / 3.0 / 2.6 %, at or below it on 0
    / 1 / 1 of 4 seeds. Arm C's is below it by 3.0 / 3.4 / 3.6 %.
  - Arm D's ablation is above job 1's dagger by 5.7 / 4.3 / 4.1 % (0 / 1 / 1 of 4 seeds). Arm C's is below it by 1.5 /
    2.1 / 2.1 % (C3).
- **The far probes after training.**
  - At 10–15 Å they are as before training: 2.4 / 2.5 / 3.0 %.
  - At 8 Å they move with E₀, as in arm C. The non-additive energy is 2.8 ± 1.0 % (ablation) and 2.2 ± 0.3 % (frozen
    form), against arm C's 3.3 ± 1.7 and 2.5 ± 0.3 %.
- **The readings fixed in training, beside arm C's.**
  - T5 is 4.13 % (slope 1.0401) against 4.35 % (1.0405).
  - The cells' dipole is 2.3 % against 1.5 %.
  - The molecules' dipole and field response are arm C's to the printed digit: 5.1 %, 2.9 %, iso 1.5 %.

### (c) What the readings say

**(i) The cost is under the field term, and the field term takes none of it.**

- The field term's gain is arm C's to 0.1 point on the rigid forces, and to within 0.6 points on each component. So
  both rungs lose the same fraction, at every size and split: T1 and T2 agree to 0.2 points.
- In arm B the field term took the change:
  - its frozen form moved +0.2 to +0.7 % from arm A's while its ablation paid +3.3 to +3.6 %;
  - the gain rose from 23 to 25 %.
- Arm D's change is one that neither the field term nor E₀ takes up. The short-range model, trained on top of it, is
  left 6–7 % worse on both rungs.

**(ii) The size of a change at contact does not price it.** This is the change of the read-out's own rigid forces in
the liquid, made by each calibration and computed before training (`scratch/e4_job2_calibration_orders.py`, on the 3
local test frames). The sources are `e4-job2-results.md` §4b for arm B, §4f for arm C (the local order-0 solve), and
`scratch/e4_job4_contact.log` for arm D.

| change | at contact, rms | the ablation after training | the frozen form |
|---|---|---|---|
| arm B against arm A, every order | 0.47 kcal/mol/Å (its orders ≥ 1 alone 0.70) | +3.3 to +3.6 % | +0.2 to +0.7 % |
| arm C against arm A, the order 0 | 0.33 | −1.4 to −1.6 % | −1.2 to −1.4 % |
| arm D against arm C, the orders ≥ 1 | 0.42 | +6.4 to +7.3 % | +6.4 to +7.5 % |

- Three calibrations gave three outcomes, in no order of the size. The smallest change gained, the middle one cost both
  rungs most, and the largest cost only the ablation.
- T1 and T2 were priced by this size, as job 2c's reading (ii) was. Both times it was wrong.
  - In job 2c the sign was wrong: a cost of +1.8 % was priced, and the ablation gained.
  - Here T2's basis priced +1–3 % and the ablation paid 2–7 times that. T1's basis expected the field term to absorb
    the change, and it absorbed none.
- What does price a change is presumably where it sits: what the short-range model and the field term can represent.
  The matching level does not read that.

**(iii) Arm D's response no longer pays its way in the bulk.**

- With arm C the response was worth 1.5–3.6 %, with and without the field term (job 2c §4d).
- With arm D both rungs are above the runs without it, by 2.6–5.7 %.
- Its infrared is arm C's but for the far probes, which are 0.2–0.3 points worse: ε_∞ 1.6893, T5 1.040, the far probes
  2.4–3.0 %. Its finite-k response is right: χ +1.33 / −0.37 % against arm C's −1.8 / −11.0 %.

**(iv) What the training does not separate.** Arm D's third stage did two things at once:

- it re-solved the orders ≥ 1 on the far probes beside the calibrated order 0, with the ridge pulling toward zero
  rather than toward arm C's values. That is what Dx does alone;
- it added the finite-k rows.

The contact log puts most of arm D's change at contact in the first. Dx moves the read-out's forces in the liquid 0.49
rms from arm C's; the finite-k rows then move them 0.29 rms from Dx's, to arm D's 0.42. But (ii) says the size of a
change does not price it, and no Dx run was trained. So the registered reading's first clause, that the finite-k fix
costs the bulk, is read here as: **arm D's calibration costs the bulk.** Whether the finite-k response itself does is
open.

**(v) Arm C stays E4's model**, by the registered branch. It is M_16C2†ᵐ_us at the 2.0 Å mutual edge: rigid forces
0.545 / 0.555 / 0.557 kcal/mol/Å at 64 / 128 / 256 molecules, ε_∞ 1.6893, and χ −1.8 / −11.0 % at k₁ / k₂.

### (d) Notes to the pre-registration

- **The predictions reached Niflheim after training.** They were committed at 14:40 CET on 9 Nov (0ea8c64) and
  delivered in `ec-e4-job4-matching-read.patch` at 14:43. Both were before `run_e4_job4_train.sbatch` was written
  (15:01) and run. The patch was applied on Niflheim on 10 Nov, for this report.
- **The report's count is seeds at or below arm C's.**
  - For T1 and T2 it is 0 of 4 at every size and split: every seed is above arm C's.
  - T2's "above on at least 3 of 4 seeds" holds; its band does not.
  - The report does not print each seed's change, so T1 is read, as registered, on the change of the seed means.
- **T3 was not registered.** Arm D's gain is 0.1 point below arm C's at 64 molecules and level at 128 and 256, which is
  within the rounding of the seed means.
- **J4's line in the report** ("FAILS on the runs present") is job 2's registered order of the gains (job 1, A, B),
  which failed there. Arm D does not enter it.
- **The runs.** There were two GPU tasks of four seeds each instead of eight tasks. The pairing is unchanged: `fit`
  draws E₀'s initialisation and the batches from the seed alone.

### (e) Next

1. **The model stays arm C's frozen form** (`configs/e4_job2c.yaml`).
2. **Proposed, not registered: D′, to separate arm D's two moves** before anything more is calibrated at finite k.
   - It uses the same finite-k rows, and moves the orders ≥ 1 from arm C's values only where those rows need it: the
     ridge pulls toward arm C's orders ≥ 1 instead of zero, and the far probes' rows are held at arm C's own predictions
     instead of re-fitted to MB-pol's.
   - It is read at matching level first, with the read-out's change at contact and the far probes as bands, and
     trained only after that read.
   - If D′ trains to arm C's bulk with χ right, the cost was the re-fit, and D′'s frozen form is the model.
   - If it pays as arm D did, the finite-k response itself costs the bulk with this read-out. Then item 3 tests whether
     a finer calibration moves less at contact.
3. **The molecules' finite-k rows beside the cells'**, the registered next step for this branch.
4. **Then the rest of `e4-plan.md`**: the readers on the model, Form I and the L₀ sweep.
