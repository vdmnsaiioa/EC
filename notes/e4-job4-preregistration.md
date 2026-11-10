# E4, job 4: arm D, the orders ≥ 1 calibrated on the finite-k response — pre-registration

*8–9 Nov 2026, v0.0.28. Vadic's go of 8 Nov, after job 3's read (`e4-job3-results.md` §8): arm D at matching level,
with H3 at two more mutual edges beside it, then training if it holds. The local master of this note is `notes/` in the
repository.*

*Timing of what was written when:*

- *The expectations (§3a) were committed at 22:49 CET on 8 Nov (`scratch/e4_job4_expectations.md`, its own commit),
  before the third stage was solved on any frame. Seen before them:*
  - *a check of the finite-k rows on one calibration cell with arm C's read-out: they equal `model.mode_response`'s
    χ_mode and are linear in the orders ≥ 1;*
  - *job 3's numbers.*
- *§3's numbers were computed after that, on job 3's local frames.*
- *§4's bands for Niflheim were written after §3's numbers, before the job ran.*

## 1. The question

Job 3 read arm C, E4's model, on MB-pol's finite-k response (`e4-job3-results.md`).

- **Arm C's orders ≥ 1 miss MB-pol's response at the cutoff's scale.** Its χ(k₂) is −11.0 %, against its order-0
  sector's +3.3 %. They are the far probes' solution, and the far probes do not identify them at any mutual edge
  (§6e there; qualified by this job's control, §3b).
- **In the cells' χ(k) the orders ≥ 1 are one number.** They add exactly γk². The k⁴ term is the order-0 sector's,
  fixed by the frozen form's mutual edge: −8 % of χ(0⁺) at k₂ at 2.0 Å, while MB-pol's χ(k) is k² to k₂.
- **So the derivative counting's second step can be taken, with a known ceiling.**
  - The k = 0 data set the order-0 coefficient (arm C).
  - The k-dependence sets the orders ≥ 1 (arm D).
  - At 2.0 Å no γ brings χ closer than about ±1.1 % at k₁ and k₂ together.

Job 4 asks, at matching level, before any training:

- **(i)** Calibrated on the calibration cells' finite-k response, do arm C's orders ≥ 1 reach that ceiling on held-out
  cells?
- **(ii)** What does it cost what the orders ≥ 1 also carry: the far probes, T5, the molecules' finite-k response, and the
  cells' dipole at zero field?
- **(iii)** How much of the change is the finite-k data's? The control Dx re-solves the orders ≥ 1 on the probes alone,
  after arm C's calibration of order 0.
- **(iv) H3 on the finite-k response.** Does the regulator dependence of the readings fall from the order-0 sector to arm
  D, at the 1.5 and 2.5 Å mutual edges beside 2.0 Å?

## 2. Design

**Arm D** (`configs/e4_job4.yaml`, `training.readout.finite_k`) is arm C's read-out followed by a third matching stage
(`scripts/e4_water.py match_readout`).

- **What moves.** The orders ≥ 1 columns alone: 3168 coefficients, `linear.solve_readout` with `orders="higher"`. The
  order-0 columns and the node term stay arm C's, exactly.
- **The rows.**
  - The far probes' rows, as in the first two stages, with the probes' ridge.
  - Every calibration cell's susceptibility in each of job 3's 12 modes, χ_mode = −d²E/da² (`model.mode_chi`, `k_stage`),
    against MB-pol's.
  - Each mode's row is relative to MB-pol's mean χ_mode at its harmonic over the calibration cells, at weight 0.1. A 1 %
    error then costs about the probes' matched objective, as job 2's liquid weight does at k = 0.
- **What is left out: the liquid's k = 0 rows.** Job 3's proposal (§8 there) kept them; they are left out here.
  - The uniform field's response R does not depend on the orders ≥ 1: a uniform field has no gradient.
  - The cells' dipole at zero field does, through the gradients of the permanent field. Fitting it was what moved arm
    B's orders ≥ 1, and contact energy with them (`e4-job2-results.md` §4b).
  - So ε₀ is arm C's by construction, and the cells' dipole at k = 0 is a reading.
- **The stage is linear.** The orders ≥ 1 sit outside the mutual solve, so the energy is linear in their columns (checked
  to 10⁻¹³): one Gauss–Newton step solves it. The Jacobian of the 12 rows is taken by reverse mode, 9 s a cell on two
  CPU cores.

**Dx, the control.** The same third stage with the finite-k rows' weight at 0: the orders ≥ 1 re-solved on the probes
alone, with arm C's calibrated order-0 sector beside them. Arm D less Dx is what the finite-k data do.

**H3 at 1.5 and 2.5 Å.**

- At each edge: arm C matched there (the far probes, then the liquid at order 0), its order-0 sector, and arm D.
- These are `kread` arms with overrides (`set: {model: {mutual_edge_angstrom: 1.5}}`).
- The edge stays the a-priori 2.0 Å (`e4-response-regulator.md` §3). The other two read how much of the edge's
  dependence the calibrated orders absorb.

**The readings** (`scripts/e4_water.py kread`, as job 3's, without training):

- the cells' χ(k₁), χ(k₂) and the molecules' finite-k response, on the held-out and the calibration cells;
- ε₀, the cells' dipole and the molecules at k = 0;
- the far probes and T5;
- the stages' objectives.

**Code (v0.0.28).**

- `model.mode_chi`: one mode's χ alone.
- `linear.solve_readout(orders="higher")`.
- `scripts/e4_water.py`:
  - `k_stage`;
  - the third stage in `match_readout` and `matched`, cached as `..._liq<h>_k<h>.pkl` after arm C's stages;
  - `train` with `training.readout.finite_k`, which reads the finite-k response before training too;
  - `kread`'s arms with `set` overrides, `order_zero_of` as a mapping to labels, and `--out`.
- `scripts/e4_report.py --job4`.
- `tests/test_arm_d.py`.

**Cost on Niflheim.**

- **The matching level:** `run_e4_job4.sbatch`, three GPU tasks, one per edge.
  - At 2.0 Å arm C's cached stages are copied in, so arm D starts from exactly the read-out arm C trained with.
  - At 1.5 and 2.5 Å the probes' and the liquid's stages are solved first.
  - About half an hour a task.
- **Training,** only after the matching level is read: `run_e4_job4_train.sbatch`, eight tasks of one run, about 26 GPU
  hours.

## 3. The local numbers

13 local cells (10 calibration, 3 held out) with job 3's local frames and read-outs (`scratch/e4_job4_local.yaml`;
readings in `scratch/e4_job4_local.json`). At 1.5 and 2.5 Å the far-probe stage comes from the local mutual-edge sweep.
Held-out cells. On χ the calibration cells read within 0.1 points of them for arm D at every edge, within 0.5 for arm
C and its order-0 sectors, and 0.4 and 1.5 points apart for Dx.

**At the 2.0 Å edge:**

| read-out | χ(k₁) | χ(k₂) | χ(k₂)'s frame sd | molecules k₁ / k₂ | the cells' dipole at k = 0 | far probes 10 / 12 / 15 Å | T5 | the probes' objective after its last stage |
|---|---|---|---|---|---|---|---|---|
| C | −1.9 % | −11.5 % | 0.29 | 3.7 / 4.7 % | 3.1 % | 2.3 / 2.2 / 2.7 % | 1.040 | 1.72×10⁻⁵ |
| C0 | +2.6 % | +3.5 % | 0.27 | 5.9 / 11.4 % | 3.1 % | 7.7 / 6.0 / 6.8 % | 1.029 | — |
| Dx | +1.7 % | +0.7 % | 0.83 | 4.8 / 8.8 % | 4.4 % | 2.8 / 2.8 / 3.0 % | 1.039 | 8.61×10⁻⁶ |
| **D** | **+1.4 %** | **−0.5 %** | **0.07** | 4.6 / 7.1 % | 4.1 % | 2.6 / 2.7 / 2.9 % | 1.039 | 9.07×10⁻⁶ |

- **ε₀ is 1.6895 for all four** (MB-pol 1.6870), as constructed.
- **The probes' objective.** Arm A's first stage left 9.90×10⁻⁶. Arm C's calibration of order 0 raised it 1.73-fold.
  Arm D's third stage brings it to 0.92 of arm A's while fitting the finite-k rows. Those rows fall 62-fold, from
  6.46×10⁻⁴ to 1.04×10⁻⁵: 1.0 % rms a mode.
- **One Gauss–Newton step.** It solves the stage, as the linearity says. Three further steps changed the objective by
  6×10⁻¹¹ relative. The orders ≥ 1 columns moved by 1.4 times their norm.

**H3, by mutual edge:**

| edge | C0: χ(k₁) / χ(k₂) | C | D | D: χ(k₂)'s frame sd | D: molecules k₁ / k₂ | D: far probes 10 / 12 / 15 Å | D: T5 | D: the finite-k rows after |
|---|---|---|---|---|---|---|---|---|
| 1.5 Å | −1.5 / −6.7 % | −5.8 / −21.1 % | **+0.5 / −0.2 %** | 0.04 | 3.7 / 5.2 % | 2.0 / 2.8 / 2.6 % | 1.044 | 1.36×10⁻⁶ (0.4 %) |
| 2.0 Å | +2.6 / +3.5 % | −1.9 / −11.5 % | **+1.4 / −0.5 %** | 0.07 | 4.6 / 7.1 % | 2.6 / 2.7 / 2.9 % | 1.039 | 1.04×10⁻⁵ (1.0 %) |
| 2.5 Å | +7.9 / +14.2 % | +2.7 / −3.2 % | **+3.3 / −1.0 %** | 0.09 | 7.6 / 11.7 % | 3.2 / 2.8 / 3.3 % | 1.030 | 5.81×10⁻⁵ (2.4 %) |

- At every edge the order-0 calibration raised the probes' objective 1.71–1.80-fold. Arm D's stage brought it to
  0.91–0.93 of the first stage's.
- The order-0 sectors' molecules at 1.5 Å are 4.0 / 5.8 %, against 5.9 / 11.4 % at 2.0 Å and 11.0 / 22.7 % at 2.5 Å.

### (a) The expectations against the numbers

1. **Arm D near the γ-only least-squares point, +1.3 / −0.4 % each within ±1.0 point: holds.** It is +1.40 / −0.46 %.
   Computed from these frames' own arm C and MB-pol, that point is +1.41 / −0.42 %.
2. **The held-out cells within 0.5 points of the calibration cells: holds.** They are within 0.04 and 0.05 points.
3. **ε₀ arm C's to every digit, and the cells' dipole moving less than 2 points: holds.** ε₀ is 1.68948 in both, and
   the dipole moves +1.0 point (3.1 → 4.1 %).
4. **The molecules at 3–7 %, not worse than arm C's by more than 2 points: fails narrowly at k₂.** They are 4.6 and
   7.1 %, worse by 0.9 and 2.3 points.
5. **The far probes pay something: holds on the probes and T5, fails by an improvement on the objective.**
   - The test probes are 2.6 / 2.7 / 2.9 % at 10–15 Å, inside 2.5–5 %, and T5's slope is 1.039.
   - The stage's probe objective fell to 0.53 of where it started, instead of rising 1.2–3-fold.
6. **Dx moving χ(k₂) by less than 3 points: fails.** Dx moves it by +12.1 points, to +0.7 % (held out) and +2.2 %
   (calibration). Its probe objective is 0.87 of arm A's, inside the expected ×1.2.
7. **H3 following the ceilings: holds.**
   - D is +0.49 / −0.16 % at 1.5 Å (expected about +0.4 / −0.1) and +3.30 / −1.05 % at 2.5 Å (about +3.2 / −1.0).
   - The edge range falls 3.3-fold on χ(k₁) and 24-fold on χ(k₂).

Four of seven hold. Expectation 4 fails narrowly at k₂ (7.05 %, 2.3 points over arm C), and 5 fails in the improving
direction. Expectation 6's failure is the finding.

### (b) What the numbers say

**(i) Arm D reaches the frozen form's ceiling, and pins it.**

- Its χ is §6's γ-only least-squares point at every edge, to 0.06 points. At 1.5 and 2.5 Å the point is +0.50 /
  −0.15 % and +3.32 / −0.99 %.
- Its χ is the same on held-out cells as on calibration cells.
- Its frame-to-frame sd is 0.04–0.09 points at k₂, below the order-0 sector's own (0.14–0.36). The finite-k rows set the
  response cell by cell, not only on average.
- What is left is the frozen form's k⁴ term, shared between the two k by the fit.

**(ii) Most of arm C's −11 % was arm C's sequence, not the far probes alone (Dx).**

- Arm C's orders ≥ 1 were matched in the first stage beside arm A's order-0 sector, whose ε₀ is 1.743. The second
  stage then changed that sector and left them as they were.
- Re-solved on the far probes beside the calibrated sector, they bring χ(k₂) to +0.7 % (held out) and +2.2 %
  (calibration).
- But the probes do not pin them. Dx's χ(k₂) scatters with a frame sd of 0.8–1.4 points, twelve to twenty times arm D's.
  Dx sits 1.8 points further from MB-pol's than arm D at k₂ on the calibration cells (0.2 on the held-out ones). Its
  molecules are worse: 8.8 % against 7.1 % at k₂.
- So job 3's reading stands with a qualification (added to `e4-job3-results.md` §6e):
  - the far probes do not identify the orders ≥ 1 cell by cell;
  - their mean finite-k response follows the order-0 sector they were matched beside;
  - the finite-k data pin it.

**(iii) What arm D costs.**

- **The far probes:** +0.2–0.5 points at 10–15 Å over arm C. The matched objective itself falls.
- **The cells' dipole at zero field:** +1.0 point. The orders ≥ 1 reach it through the permanent field's gradients.
- **The molecules at finite k:** 4.6 / 7.1 % against arm C's 3.7 / 4.7 %. The 12 cell rows constrain the orders ≥ 1 in
  sum, and they moved by 1.4 times their norm.

**(iv) H3 on the finite-k response.**

- **On χ.** The edge dependence falls 3.3-fold at k₁ and 24-fold at k₂, from the order-0 sectors to arm D. This is by
  construction at the fitted k. The residual is the k⁴ term: χ(k₁) +0.5 to +3.3 % and χ(k₂) −0.2 to −1.0 % from 1.5 to
  2.5 Å.
- **Out of sample, on the molecules.** The edge range falls 1.8-fold at k₁ (7.1 → 3.9 points) and 2.6-fold at k₂
  (16.9 → 6.5 points). So H3's ≥ 2 holds at k₂ and fails narrowly at k₁.
- **The far probes and T5** hardly depend on the edge in either.
- **The 1.5 Å edge reads best on everything finite-k:**
  - arm D's χ at +0.5 / −0.2 %;
  - its molecules at 3.7 / 5.2 %, arm C's 2.0 Å level;
  - the probes at 2.0–2.8 %.
- The edge stays 2.0 Å by the a-priori rule (`e4-response-regulator.md` §3). The 1.5 Å reading is recorded, not acted
  on.

## 4. Registered for Niflheim

Job 4 on Niflheim (`run_e4_job4.sbatch`, three tasks), with the 40 cells' finite-k response from job 3 and arm C's
cached read-out at 2.0 Å.

- Arm D's stage there uses 30 calibration cells, not 10.
- §6's ceiling, computed from job 3's Niflheim numbers, puts arm D's γ-only least-squares point at +1.32 / −0.39 %.
- Bands that compare arm D with arm C are relative, so that arm C's own Niflheim values (cells' dipole 1.5 %,
  molecules 3.8 / 5.5 %) carry over.

| | reading (held-out cells unless said) | registered |
|---|---|---|
| **D1** | arm D's χ(k₁); χ(k₂) | +0.8 to +1.9 %; −1.0 to +0.2 % |
| **D2** | arm D's χ, held out less calibration, at k₁ and k₂ | within ±0.5 points |
| **D3** | arm D's ε₀ less arm C's; its cells' dipole at k = 0 less arm C's | < 10⁻⁶; ≤ +2.0 points |
| **D4** | arm D's molecules less arm C's, at k₁ / k₂; C0's at k₂ less arm D's | ≤ +2.0 / ≤ +3.5 points; > 0 |
| **D5** | arm D's far probes at 10 / 12 / 15 Å less arm C's; T5; the probes' objective, stage 3 ÷ stage 2 | each ≤ +1.0 point; 0.95–1.10; 0.3–1.0 |
| **D6** | Dx's χ(k₂) less arm C's; Dx's frame sd of χ(k₂) ÷ arm D's; \|D's χ(k₂)\| less \|Dx's\| | ≥ +6 points; ≥ 3; < 0 |
| **D7** | D@1.5's χ(k₁); χ(k₂) | −0.1 to +1.1 %; −0.7 to +0.4 % |
| | D@2.5's χ(k₁); χ(k₂) | +2.3 to +4.3 %; −2.0 to 0 % |
| | H3: the edge range of χ(k₁) and χ(k₂), order-0 sectors ÷ arm D | each ≥ 2 |
| | H3 out of sample: the same for the molecules at k₁; at k₂ | 1.2–2.5; 1.8–4.0 |

`scripts/e4_report.py --job4 results/e4_job4` prints them. Every D holds on the local reading
(`tests/test_e4_report.py`).

- D1, D2 and D6 are the main prediction: the finite-k data pin the orders ≥ 1 at the frozen form's ceiling, and the far
  probes alone do not.
- D3–D5 are the costs.
- D7 is H3. The molecules' k₁ band registers the narrow H3 failure that the local reading shows.

## 5. What would change the reading

- **D1–D5 hold: train arm D.**
  - `run_e4_job4_train.sbatch`: its two C rungs, 4 seeds, eight tasks.
  - The bulk predictions for training are registered with the matching level's read, before training starts.
- **D1 fails toward arm C** (χ(k₂) below −1 %). Niflheim's stage does not reach the ceiling; read the stage's
  objectives and Dx before anything trains.
- **D4 or D5 fail** (the molecules or the probes worse than registered).
  - The finite-k rows on the cells' sums move the orders ≥ 1 in directions the other data need.
  - Next would be the molecules' finite-k rows beside the cells', before training.
- **D6 fails with Dx at arm D's χ.** Then the far probes, beside the calibrated order-0 sector, already set the orders
  ≥ 1. Arm D would add nothing, and arm C re-solved (Dx) would be the model to train.
- **D7's out-of-sample part fails at k₂.** The calibrated orders do not absorb the edge's dependence beyond the rows
  they are fitted on: H3 fails on the finite-k response.

## 6. Training (after §4's read)

- `run_e4_job4_train.sbatch` trains arm D's two C rungs, as arm C's were (`configs/e4_job4.yaml`, out_dir
  `results/e4_job4/edge2.0`).
- The frozen form's read-out comes from the matching job's cache. The ablation's third stage starts from arm C's cached
  ablation read-out, which is copied in first.
- Read against arm C, paired by seed, with `scripts/e4_report.py --job2`'s tables.

## 7. Readings beside them, not registered

- the calibration cells against the held-out ones, for every variant;
- arm C at 1.5 and 2.5 Å, and its probes (the far-probe orders ≥ 1 at other edges);
- the stages' objectives at every edge;
- the molecules' dipole size at finite k;
- the matching's cost on the GPU (the third stage, a Jacobian of 12 rows a cell).

## 8. Training: registered 9 Nov, 14:40 CET (this commit's time), after the matching level and before any training

*Written after Niflheim's matching level was read (all seven bands hold; `e4-job4-results.md` §1) and before
`run_e4_job4_train.sbatch` started. Arm C's trained readings are `e4-job2c-results.md`'s.*

**What arm D changes at contact.** This was computed locally before this section, without training
(`scratch/e4_job4_contact.log`: `scratch/e4_job2_calibration_orders.py` on the 3 local test frames at 2.0 Å, E₀ and
the field term at initialisation cancelling).

- **Arm D against arm C.** Arm D's orders ≥ 1 change the read-out's own energy in the liquid by −0.22 kcal/mol per
  molecule and its rigid forces by 0.42 kcal/mol/Å rms. Its own energy is −3.97 against −3.75; its rigid forces 2.70
  against 2.47 rms.
- **Most of it is the re-solve, not the finite-k rows.** Dx against arm C is −0.24 kcal/mol and 0.49 rms. Arm D against
  Dx is +0.02 kcal/mol and 0.29 rms.
- **Arm B for scale** (`e4-job2-results.md` §4b). Its orders ≥ 1 moved −0.79 kcal/mol and 0.70 rms against arm A's.
  - That cost the ablation 2–5 % on every seed.
  - The frozen form was unchanged: its field term absorbed it.
  - Arm D's change is about a quarter of that in energy and 60 % in the forces.

**The predictions** (paired by seed; the unseen frames pooled, as job 2's; `scripts/e4_report.py --job2` with arm D's
directory as a fourth):

| | reading | registered |
|---|---|---|
| **T0** | before training, every run of both rungs: the held-out liquid's ε_∞ and the finite-k χ on the matched read-out | as the matching job's arm D: ε_∞ 1.6893, χ +1.33 / −0.37 %, to the printed digit |
| | ε_∞, T5 and the far probes after training | as before it |
| **T1** | the frozen form, arm D against arm C: rigid forces, seed means, at 64, 128 and 256 molecules | within ±1.5 % at every size |
| **T2** | the ablation, arm D against arm C: the same | +0.5 to +4.5 % at every size, and above on at least 3 of 4 seeds |

T3 follows from T1 and T2 and is read beside them, not registered: the field term's gain, 1 − F(frozen form) ÷
F(ablation), at least arm C's (23 %).

**Why these.**

- **T1.** The field term absorbed arm B's larger change at contact, and arm D's is smaller. Fixing the cells'
  response at the cutoff's scale (−11 % → −0.4 % at k₂) is a long-wavelength gain, and the bulk forces at 64–256
  molecules are mostly contact. The molecules' finite-k pattern also gets worse (+1.8 points at k₂). Neither
  direction is favoured.
- **T2.** E₀ alone did not absorb arm B's change. Scaled by the rigid forces, arm D's change predicts about +1–3 %.

**What would change the reading.**

- **T1 holds, T2 holds.** The finite-k calibration is neutral for the frozen form's bulk, and its orders ≥ 1 carry
  contact energy that only the field term absorbs, as arm B's did.
  - Arm D's frozen form is then the model: the same bulk as arm C's, with the response right at k₁ and k₂.
  - The ablation's cost says the orders ≥ 1 should next be calibrated with the molecules' finite-k rows beside the
    cells', so that they move less at contact.
- **T1 fails low** (the frozen form gains more than 1.5 %). The finite-k response matters for the bulk, and arm D's
  frozen form is the model.
- **T1 fails high** (it loses more than 1.5 %). The finite-k fix costs the bulk; arm C stays the model, and the
  molecules' finite-k rows come next.
- **T2 fails low** (the ablation pays less than 0.5 %). E₀ absorbs arm D's change at contact, unlike arm B's. The cost of
  moving the orders ≥ 1 was arm B's k = 0 rows, not the moving itself.
