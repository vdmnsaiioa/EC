# E4, job 2: the response regulated, and calibrated in the liquid — pre-registration

*5 Nov 2026, v0.0.24. This supersedes §6 of `e4-x3-preregistration.md` before any of it was trained: that form lost the
far field (`e4-response-regulator.md` §2).*

*Timing of what was written when:*

- *§3's numbers and §4's were computed before training (`scratch/e4_job2_preregistration.py`, local frames).*
- *§4's expectations were written after §3's numbers and before §4's were computed. They are copied verbatim from
  `scratch/e4_job2_armB_expectations.md`, written 14:54 CET; arm B started at 14:55.*
- *§5's predictions were written before any job-2 training, after the first of §4's numbers (order 0 at 2.0 Å) had
  been read.*

*The local master of this note is `notes/` in the repository.*

## 1. What job 2 asks

Job 1 left three linked failures (`e4-job1-results.md` §4a):

- the response without the field term bought nothing in bulk: rigid forces +3 to +4 % over the dagger, net forces
  +8 to +11 %;
- ε_∞ was 1.80 against MB-pol's 1.69;
- the induction was too strong at contact (local-field enhancement 1.30 against 1.17).

Read through the EFT of large-scale structure (Baumann, Nicolis, Senatore and Zaldarriaga, arXiv:1004.2488), the
response's regulator sat below the non-linear scale of the dipole–dipole coupling. Placed above it, at the a-priori
2.0 Å (`e4-response-regulator.md`), ε_∞'s error halves before training. What remains is a counterterm the far probes do
not see. Job 2 takes the paper's two moves into the bulk:

- **Arm A, the regulator** (`configs/e4_job2a.yaml`).
  - The mutual tensor is smeared at 2.0 Å; the read-out is matched on the far probes alone, as in job 1.
  - Three rungs: the frozen form M_16C2†ᵐ_us; its field-term ablation M_16C2†ᵐ; and M_16†ᵐ_us, new: the dagger with the
    frozen form's field term, which is the frozen form less its read-out.
- **Arm B, the calibration** (`configs/e4_job2b.yaml`).
  - Arm A's read-out is then calibrated on MB-pol's induced dipoles and field response in 30 liquid cells. In the
    paper's words: "calibrated by computing the small-scale dynamics with numerical N-body simulations".
  - The two C rungs.
- **Job 1's other rungs are read beside job 2's, not rerun.** M_∞, the dagger M_16†ᵐ, M_1, M_16 and M_G have no
  mutual response.

## 2. Design

**MB-pol's liquid response** (`scripts/e4_water.py response`, `run_e4_response.sbatch`).

- MBX's induced dipoles at clamped nuclei, each molecule's four sites summed.
- Taken at zero field and at ±δ along each axis, with δ = 10⁻⁴ e/Å². The response comes from central differences; it
  is linear, so the step only sets the rounding. That is 7 dipole solves a frame.
- 30 calibration frames are spread evenly over the 180 training frames of 64 molecules (three trajectories), and 10
  held-out frames over the 30 test frames (the fourth trajectory's second half).
- The file records its frames' hash, and the train step refuses it if the frames have changed. It runs in under an hour
  on CPU.

**Arm B's matching** is a second stage of `match_readout`, starting from the far-field solution.

- Its rows are the probes' energies, plus each calibration cell's induced dipole M (3 numbers) and field response R (3×3)
  from `model.induced_response`.
- The probes' ridge is kept. Gauss-Newton steps are halved where they overshoot, up to 5 times, with 4 iterations.
- The weights are 0.1 / n_frames × 1/(3 s_M²) on the M rows and × 1/(9 s_R²) on the R rows.
  - s_R = n α_ref, where α_ref = 1.4075 Å³ is MB-pol's monomer polarisability.
  - s_M = √n μ_ref, with μ_ref = 0.15 e Å.
  - At these weights a 1 % error of the cells' response costs about as much as the probes' matched objective.

**Training** is job 1's: 3000 Adam steps, 50 L-BFGS steps, rigid-body forces, the read-out held.

**Readings.**

- Job 1's readings: every frame of 64, 128 and 256 molecules; the far probes; T5; ε_∞ on 10 test frames.
- For the C rungs, on the 10 held-out liquid frames before and after training:
  - ε_∞ and the cells' induced dipole;
  - each molecule's induced dipole and field response (rmse / rms);
  - the spread of the molecules' isotropic response against MB-pol's, and its correlation;
  - the size of the molecules' dipoles against MB-pol's, and their correlation.
- `scripts/e4_report.py --job2` reads both arms against job 1 and against §3–5.

**Cost.**

- Five runs a seed: the frozen form in arms A and B, the ablation in arms A and B, then M_16†ᵐ_us.
- About 17–18 h a job, 70 GPU hours in all.
- Without the two arm-B runs (`RUNS=...`), about 14 h a job.

## 3. Arm A: fixed before training

The local frames are the X3 pre-registration's two short MB-pol trajectories of 64 molecules (`data: {dir: <local>,
response: <local>}`, `liquid: {calibration_frames: 10, heldout_frames: 3}`):

- calibration: 10 training frames of one trajectory;
- held out: 3 test frames of the other, where MB-pol's ε_∞ is 1.6870;
- 6 frames that no calibration uses (those 3 plus 3 validation frames), where MB-pol's ε_∞ is 1.6871.

The numbers are in `scratch/e4_job2_preregistration.json`. Its `sseft` field reads 0.0.23: the version string was
bumped after the runs, on the code that made them.

Each read-out is matched on the far probes alone. The probes have job 1's hash, 72b73e3aa6c7b53f. The molecules'
errors are rmse / rms over every molecule of the held-out frames.

| edge (Å), read-out order | ε_∞, 3 / 6 frames | cell M | molecules' μ | molecules' R (iso) | iso R spread ÷ MB-pol's (r) | \|μ\| ÷ MB-pol's (r) | probes 10 / 12 / 15 Å | T5 slope (rmse) |
|---|---|---|---|---|---|---|---|---|
| 0.75, 0 | 1.8198 / 1.8208 | 17.4 % | 17.5 % | 22.0 (20.0) % | 3.78 (0.45) | 1.146 (0.996) | 7.0 / 4.7 / 4.2 % | 0.943 (7.1 %) |
| 1.5, 0 | 1.7664 / 1.7674 | 10.9 % | 11.3 % | 14.0 (12.4) % | 3.42 (0.57) | 1.078 (0.997) | 6.8 / 4.3 / 4.2 % | 0.956 (5.9 %) |
| 2.0, 0 | 1.7214 / 1.7219 | 6.5 % | 8.6 % | 7.9 (6.3) % | 3.04 (0.62) | 1.027 (0.997) | 7.0 / 4.0 / 4.5 % | 0.980 (4.0 %) |
| 2.5, 0 | 1.6912 / 1.6913 | 6.3 % | 8.8 % | 5.6 (3.7) % | 2.94 (0.62) | 0.992 (0.996) | 7.3 / 4.0 / 4.8 % | 0.997 (3.2 %) |
| 0.75, 2 (job 1) | 1.7990 / 1.7996 | 16.2 % | 19.5 % | 18.1 (16.6) % | 1.99 (0.02) | 1.172 (0.996) | 2.5 / 2.6 / 2.0 % | 1.026 (3.7 %) |
| 1.5, 2 | 1.7727 / 1.7732 | 12.3 % | 12.3 % | 13.4 (12.8) % | 1.63 (0.02) | 1.105 (0.998) | 2.2 / 2.1 / 1.8 % | 1.034 (4.1 %) |
| **2.0, 2 (arm A)** | **1.7429 / 1.7432** | 8.9 % | 8.2 % | 9.2 (8.7) % | 1.71 (−0.07) | 1.057 (0.998) | 2.3 / 2.1 / 2.1 % | 1.042 (4.7 %) |
| 2.5, 2 | 1.7200 / 1.7201 | 6.1 % | 6.9 % | 6.4 (5.8) % | 1.74 (−0.12) | 1.023 (0.998) | 2.9 / 2.3 / 2.4 % | 1.048 (5.0 %) |

The ε_∞ values reproduce the scan's on its 8 frames (`e4-response-regulator.md`) to 0.001.

**What this adds to the scan.**

- **The molecules' induced dipoles follow MB-pol's almost exactly.** Their correlation is 0.996–0.998 in direction and
  relative size. Their error is mostly a single scale, which falls with the edge along with ε_∞: the dipoles' excess is
  about 0.7 of the susceptibility's.
  - Job 1's model at 0.75 Å over-polarises each molecule by 15–17 %. This is §4a's contact over-polarisation, now
    measured molecule by molecule.
  - At 2.0 Å the excess is 3–6 %.
- **The molecules' response to a uniform field does not follow MB-pol's.**
  - Its spread from molecule to molecule is 3–3.8 times MB-pol's at order 0 (r 0.45–0.62), and 1.6–2 times at order 2
    (r about 0). The edge hardly changes it.
  - The cause is the read-out's polarisability. It is built from pairs with the neighbours
    (`bandfields.response_energy_pair`; the isolated monomer's is exactly zero), so it varies with the environment.
    MB-pol's site polarisabilities are fixed, so its molecules vary only through the local field.
  - MB-pol's ε_∞ is 1.6870–1.6871 on all three frames. The model's varies by 0.002–0.009 between frames.

**Fixed for job 2's arm A on Niflheim** (the order-2 rows at 2.0 Å):

- ε_∞ ≈ 1.744 on job 1's 10 test frames, and ≈ 1.743 on the held-out liquid frames;
- far probes 2.3 / 2.1 / 2.1 % at 10 / 12 / 15 Å;
- T5 slope 1.042;
- molecules' μ about 8 %;
- the same for the ablation, which shares the read-out.

## 4. Arm B: fixed before training

### 4.1 The expectations, as written before arm B ran

From `scratch/e4_job2_armB_expectations.md`, as written (each edge 1.5 / 2.0 / 2.5 Å, read-out orders 0 and 2; the second
stage on the 10 local calibration frames, weight 0.1):

> B1. The calibration reaches the bulk response: eps_inf on the 3 held-out frames and on the 6 frames no calibration uses
>     within 0.010 of MB-pol's, at every edge and order. Its edge dependence (1.5 -> 2.5 A) then <= 0.01, against 0.053
>     (order 2) and 0.075 (order 0) in arm A.
> B2. The far probes do not see the counterterm: the probes' part of the matching objective rises by less than 2x in
>     the second stage; the probes' non-additive error at 10, 12, 15 A at most 1 point above arm A's at order 2 and 2
>     points at order 0, at every edge; T5's slope within 0.90-1.10.
> B3. The molecules' dipoles take the right scale: |mu| / MB-pol's within 0.98-1.02 at every edge and order, r stays
>     >= 0.995, and the molecules' dipole error falls at every edge and order (at 2.0 A to <= 7 % at order 2, <= 8 % at
>     order 0).
> B4. The molecules' response spread is not the cell sums' to fix: its spread stays >= 1.5 times MB-pol's and its
>     correlation with MB-pol's <= 0.7, at every edge and order. (Falsified if the calibration brings the spread below 1.5:
>     then the sums do constrain the environment dependence.)
> B5. The per-molecule response error (R, full tensor) falls at every edge and order but stays >= 4 %: what remains is
>     the spread (B4).

### 4.2 The numbers

The same frames as §3 (MB-pol 1.6870 on the 3 held-out frames, 1.6871 on the 6). "Probes' objective" is the probes'
share of the matching objective after the second stage, against the far-field solution's.

| edge (Å), read-out order | ε_∞, 3 / 6 frames | cell M | molecules' μ | molecules' R (iso) | iso R spread ÷ MB-pol's (r) | \|μ\| ÷ MB-pol's (r) | probes 10 / 12 / 15 Å | T5 slope (rmse) | probes' objective |
|---|---|---|---|---|---|---|---|---|---|
| 1.5, 0 | 1.6886 / 1.6879 | 3.4 % | 4.0 % | 3.2 (2.1) % | 1.89 (0.72) | 0.988 (0.999) | 7.0 / 4.3 / 4.1 % | 0.998 (2.1 %) | ×1.33 |
| 2.0, 0 | 1.6893 / 1.6885 | 2.8 % | 4.3 % | 2.9 (2.1) % | 1.86 (0.67) | 0.994 (0.999) | 7.1 / 4.2 / 4.4 % | 1.004 (2.0 %) | ×1.22 |
| 2.5, 0 | 1.6897 / 1.6889 | 2.6 % | 5.5 % | 3.0 (2.2) % | 1.83 (0.63) | 0.998 (0.998) | 7.2 / 4.2 / 4.7 % | 1.006 (2.0 %) | ×1.17 |
| 1.5, 2 | 1.6877 / 1.6875 | 1.9 % | 4.7 % | 2.2 (1.5) % | 1.21 (0.60) | 0.969 (0.999) | 1.7 / 2.7 / 2.5 % | 1.038 (4.2 %) | ×1.26 |
| **2.0, 2 (arm B)** | **1.6882 / 1.6880** | 2.1 % | 4.6 % | 2.2 (1.7) % | 1.22 (0.49) | 0.968 (0.999) | 2.1 / 2.6 / 2.8 % | 1.040 (4.2 %) | ×1.22 |
| 2.5, 2 | 1.6885 / 1.6883 | 2.4 % | 5.6 % | 2.5 (1.9) % | 1.23 (0.40) | 0.969 (0.999) | 2.8 / 2.8 / 3.2 % | 1.041 (4.2 %) | ×1.21 |

The liquid rows fall 12–2600-fold in the second stage: 1.3×10⁻⁴–1.9×10⁻³ → 1.1–1.2×10⁻⁵ at order 0, and
2.7×10⁻⁴–2.0×10⁻³ → 7.8–9.5×10⁻⁷ at order 2. On the calibration frames themselves the cells' ε_∞ is within 0.0014
of MB-pol's (cell M 0.2–1.1 %).

### 4.3 Verdicts

| | verdict | the numbers |
|---|---|---|
| **B1** | **holds** | ε_∞ within 0.0027 of MB-pol's on the 3 held-out frames and 0.0018 on the 6, at every edge and order. From 1.5 to 2.5 Å it moves 0.0011 (order 0) and 0.0008 (order 2), against 0.075 and 0.053 in arm A |
| **B2** | **holds** | The probes' objective rises ×1.17–1.33. The probes at 10–15 Å are within +0.8 points of arm A's at order 2 and ±0.2 at order 0. T5's slope is 0.998–1.041 |
| **B3** | **holds in part** | The molecules' dipole error falls at every edge and order, to 4.0–5.6 % (4.6 % and 4.3 % at 2.0 Å), with r 0.998–0.999. Their scale is right at order 0 (0.988–0.998) but **3 % small at order 2 (0.968–0.969)**: the scale clause fails there |
| **B4** | **fails at order 2** | The spread falls to 1.21–1.23 times MB-pol's (arm A 1.63–1.74). At order 0 the spread stays 1.83–1.89, but r reaches 0.72 at 1.5 Å |
| **B5** | **fails** | The molecules' response error falls to 2.2–3.2 % at every edge and order, below the 4 % expected |

**Reading.**

- **The matching does what the paper's does.** Calibrated on the liquid, the long-wavelength readings no longer
  depend on the regulator:
  - ε_∞ is 1.6877–1.6897 from 1.5 to 2.5 Å at both orders;
  - the far field stays as arm A had it;
  - the probes barely feel the new rows (×1.2–1.3 on their objective). They did not see this counterterm, and
    calibrating it costs them little.
- H3 as registered in the X3 pre-registration (the edge dependence of ε_∞ falling with the read-out's order) is moot
  after calibration: ε_∞ no longer depends on the edge at either order.
- **The cell sums carry more than I expected (B4, B5 wrong).** The zero-field cell dipole M is Σ_i α_i E_i over
  molecules whose local fields differ in size and direction. Across frames, M therefore constrains how the
  polarisability varies with the environment.
  - At order 2 the molecules' response comes within 2.2 % of MB-pol's.
  - Its spread from molecule to molecule falls to 1.2 times MB-pol's.
- **What remains at order 2 is a scale the two kinds of row trade against each other.** The molecules' dipoles end 3 %
  small (0.968–0.969) while the cells' M is 1.9–2.4 % off.
  - Job 1's contact over-polarisation, +17 % per molecule, becomes −3 %.
  - The order-0 read-out lands at the right scale (0.99–1.00) but keeps a wider spread (1.8–1.9×).

**Fixed for job 2's arm B on Niflheim** (the order-2 row at 2.0 Å; there the second stage has 30 calibration frames,
not 10, so the numbers move a little):

- ε_∞ within 0.01 of MB-pol's on the 10 held-out frames;
- far probes about 2.1 / 2.6 / 2.8 % at 10 / 12 / 15 Å;
- T5 slope about 1.04;
- the molecules' μ about 4.6 % and R about 2.2 %;
- the same for the ablation, which shares the read-out.

## 5. Predictions for training

Read as in job 1:

- the rigid-body forces (rmse of each molecule's net force and torque);
- on the unseen frames pooled (valid + test at 64 molecules, every frame at 128 and 256);
- seed by seed for the paired rules, as seed means otherwise.

Job 1's rungs are read from `results/e4_job1`. With job 1's seed spreads, a seed-mean difference resolves about ±4 %
for the ablation (sd 0.06) and about ±3 % for the frozen form (sd 0.03). A single comparison near that size can fail
by noise; the order across the three levels in J4 is the sturdier reading.

| | prediction | rule | job 1 |
|---|---|---|---|
| **J0** | The matched readings hold through training. ε_∞ on the held-out liquid frames is the same before and after training, to 10⁻⁶. The probes at 10–15 Å and T5 agree across seeds and between an arm's two C rungs. | consistency, not a prediction | as job 1 |
| **J1** | With its contact error calibrated away, the response no longer costs in bulk: arm B's ablation is at or below the dagger (job 1's M_16†ᵐ). | ≥ 3 of 4 seeds, paired, at 64, 128 and 256 | ablation 2/4 at every size, +3 to +4 % |
| **J1a** | The regulator alone removes at least half of that cost: arm A's ablation is at most +2 % above the dagger. | seed means, every size | +4.2, +3.4, +3.1 % |
| **J2** | The field term had absorbed most of the contact error, so the frozen form moves little. Arm A's frozen form is within ±5 % of job 1's, and arm B's within ±5 % of arm A's. | seed means, every size | 0.605, 0.619, 0.618 |
| **J3** | Regulated and calibrated, the response adds to the field term: arm B's frozen form is at or below M_16†ᵐ_us. | ≥ 3 of 4 seeds, paired, every size | (new rung) |
| **J4** | The field term's gain on the C rungs, 1 − F(frozen) / F(ablation), falls as the contact error does: job 1 > arm A > arm B. | seed means, every size | 19 % |
| **J5** | As job 1's prediction 3: no job-2 rung's energy bias moves from 64 to 256 molecules by more than 2 frame-to-frame standard errors. | as job 1 | holds but for M_∞ |

**Notes on the predictions.**

- **J3 is the least certain.** In E3-QP6's clusters the dagger with the whole-molecule field term was ahead of the C rung
  with it at contact (strict near-rigid 3-body 0.043 against 0.059): "the read-out's near field still costs there".
  J3 asks whether, once regulated and calibrated, it stops costing.
- **M_16†ᵐ_us against job 1's frozen form is a reading, not a prediction.** It says whether the response ever helped the
  bulk.
- **The field term's gain on the dagger, 1 − F(M_16†ᵐ_us) / F(M_16†ᵐ), is also a reading.** E3-QP6 found it larger
  than on the C rungs in clusters.
- **Energies are reported with the same rules but not registered.** On the pooled frames at 128 and 256 molecules they
  separate the rungs by less than their seed spread (job 1 §4c).

## 6. What would change the reading

- **J1 fails** with arm B's molecules' dipoles within 3 % of MB-pol's (§4). The response's bulk cost is then not its
  contact over-polarisation. It could lie in:
  - the molecules' response spread, which arm B leaves 1.2 times MB-pol's at order 2;
  - the read-out's energies at orders ≥ 1;
  - the 3 % under-polarisation that arm B leaves, of the other sign.
- **J2 fails by an improvement beyond 5 %.** The field term had not absorbed the contact error, and the frozen form gains
  with it.
- **J3 fails.** In bulk the field term does what the regulated response does. The response's worth is then the infrared
  (ε_∞, T5, the probes, Form I), and the choice of model rests on those readings and on the interfaces (X2-C).
- **The fixed readings do not hold (J0)**, or arm B's ε_∞ on Niflheim's 10 held-out frames is not within 0.01 of MB-pol's
  (B1 on 30 calibration frames). That is an implementation error, to be found before the bulk is read.
