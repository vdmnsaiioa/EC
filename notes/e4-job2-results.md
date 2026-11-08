# E4, job 2: the results

*6 Nov 2026. Job 2 ran as pre-registered (`e4-job2-preregistration.md`) on Niflheim at fc7900d (v0.0.24): MB-pol's
liquid response first, then one job per seed, each training five runs in turn. All 20 runs finished, in 13.0–14.3 h
per job, about 55 GPU hours in all (70 estimated).*

*Read from the four seed logs. The pooled readings are frame-weighted from the logged splits, the way
`scripts/e4_report.py` pools them. Job 1's rungs are taken from its report. Two readings need the result jsons and come
from `scripts/e4_report.py --job2` when it is run:*

- *J5 exactly;*
- *the seed-by-seed comparisons with job 1's rungs at 128 and 256 molecules on every frame. Job 1's training splits at
  those sizes were not transcribed, so those comparisons use the validation and test frames here.*

*Notation as in the pre-registration. Arm A is the 2.0 Å mutual edge, matched on the far probes. Arm B is arm A's
read-out calibrated on MB-pol's liquid response. "Job 1" is the 0.75 Å edge. The frozen form is M_16C2†ᵐ_us, the
ablation M_16C2†ᵐ, the dagger M_16†ᵐ (job 1's run), and M_16†ᵐ_us the dagger with the frozen form's field term. Forces
are rigid-body rmse in kcal/mol/Å on the unseen frames pooled: valid + test at 64 molecules, every frame at 128 and 256.
Values are seed means ± sd over 4 seeds.*

## 1. The readings fixed before training (J0): as computed

| reading | pre-registered | arm A | arm B |
|---|---|---|---|
| ε_∞, 10 held-out liquid frames (MB-pol 1.6885) | 1.743; B within 0.01 of MB-pol's | 1.7434 | **1.6887** |
| ε_∞, job 1's 10 test frames | 1.744 | 1.7443 | 1.6881 |
| molecules' induced dipole; field response (iso) | 8 %; B 4.6 %, 2.2 % | 8.3 %; 9.0 (8.5) % | 4.4 %; 2.0 (1.3) % |
| \|μ\| ÷ MB-pol's (r); iso response spread ÷ MB-pol's (r) | — | 1.060 (0.998); 1.58 (−0.15) | 0.979 (0.999); 1.13 (0.71) |
| cells' induced dipole | — | 7.0 % | 0.9 % |
| far probes, non-additive, 10 / 12 / 15 Å | 2.3 / 2.1 / 2.1 %; B 2.1 / 2.6 / 2.8 % | 2 / 2 / 2 % | 2 / 2 / 3 % |
| T5 slope (rmse) | 1.042; B about 1.04 | 1.0424 (5 %) | 1.0273 (3 %) |
| the probes' objective in the calibration | B ×1.2 | — | ×1.24 |

- **Every run of an arm reads the same, to the printed digit.** That holds across seeds and between the two C rungs.
  The liquid readings before and after training are identical: the read-out is held, and nothing else sees a uniform
  field.
- **Arm B's calibration on 30 cells.** The liquid rows fell 500-fold (1.2×10⁻³ → 2.4×10⁻⁶). It ends closer to MB-pol's
  than the local calibration on 10 cells did: dipoles 4.4 % and 2 % small, response 2.0 %, spread 1.13 with r 0.71.
- **J0 holds.**

## 2. The bulk readings

| run | F 64 | F 128 | F 256 | E 64 | E 128 | E 256 |
|---|---|---|---|---|---|---|
| M_∞ (job 1) | 0.907 ± 0.069 | 0.917 ± 0.070 | 0.929 ± 0.066 | 0.0329 | 0.0254 | 0.0200 |
| dagger (job 1) | 0.721 ± 0.022 | 0.735 ± 0.021 | 0.738 ± 0.018 | 0.0309 | 0.0221 | 0.0140 |
| M_16†ᵐ_us | 0.561 ± 0.018 | 0.574 ± 0.018 | 0.577 ± 0.017 | 0.0269 | 0.0201 | 0.0135 |
| ablation, job 1 | 0.751 ± 0.061 | 0.760 ± 0.059 | 0.761 ± 0.062 | 0.0335 | 0.0228 | 0.0134 |
| ablation, A | 0.721 ± 0.052 | 0.731 ± 0.050 | 0.733 ± 0.054 | 0.0321 | 0.0230 | 0.0133 |
| ablation, B | 0.747 ± 0.049 | 0.756 ± 0.047 | 0.757 ± 0.051 | 0.0346 | 0.0248 | 0.0136 |
| frozen form, job 1 | 0.605 ± 0.032 | 0.619 ± 0.032 | 0.618 ± 0.030 | 0.0259 | 0.0212 | 0.0122 |
| **frozen form, A** | **0.551 ± 0.032** | **0.563 ± 0.030** | **0.563 ± 0.029** | **0.0244** | 0.0187 | **0.0099** |
| frozen form, B | 0.555 ± 0.031 | 0.564 ± 0.028 | 0.566 ± 0.026 | 0.0265 | **0.0182** | 0.0103 |

Energies are pooled rmse per molecule, kcal/mol. Their seed sds are 0.0013–0.0034.

**Paired by seed.** Every run of a seed shares E₀'s initialisation and batches across the jobs and arms, and the paired
changes are far tighter than the seed spread:

| change of the rigid forces, per seed (64 / 128 / 256 molecules) | seeds 0, 1, 2, 3 |
|---|---|
| frozen form, A against job 1 (valid + test) | −8.5, −8.4, −8.5, −10.2 / −7.6, −8.0, −8.6, −8.7 / −8.2, −7.9, −9.3, −9.1 % |
| ablation, A against job 1 (valid + test) | −3.9, −5.3, −2.8, −3.5 / −3.5, −4.8, −2.0, −2.8 / −3.3, −4.6, −2.6, −3.0 % |
| frozen form, B against A | +0.3, +0.8, −0.2, +1.9 / −0.1, +0.3, −0.8, +1.6 / +0.3, +0.4, −0.5, +1.7 % |
| ablation, B against A | +4.4, +2.1, +4.4, +3.7 / +3.9, +1.8, +4.9, +2.8 / +3.7, +1.8, +4.7, +3.0 % |
| frozen form, B against M_16†ᵐ_us | −5.3, +7.6, −4.7, −1.9 / −4.5, +6.5, −5.4, −3.3 / −4.7, +5.2, −4.9, −3.3 % |

The training losses agree.

- The frozen form's final loss is the same in both arms: 24.1, 29.3, 26.4, 22.3 (A) against 24.1, 29.3, 26.3, 23.2
  (B).
- The ablation's is 4–10 % higher in arm B on every seed (39.7, 50.2, 40.8, 35.2 against 42.8, 52.1, 45.0, 37.0).

These are differences in what the model class can fit, not in how it generalises.

## 3. The predictions

| | prediction | reading (unseen frames) | verdict |
|---|---|---|---|
| **J0** | the matched readings hold through training | §1 | **holds** |
| **J1** | arm B's ablation at or below the dagger, ≥ 3 of 4 seeds, every size | 1 of 4 at every size (valid + test); seed means +3.6, +2.8, +2.5 % | **fails** |
| **J1a** | arm A's ablation at most +2 % above the dagger | +0.0, −0.5, −0.7 %; 3 of 4 seeds below it at every size | **holds** |
| **J2** | arm A's frozen form within ±5 % of job 1's; arm B's within ±5 % of arm A's | A **−8.9, −9.1, −8.9 %** (every seed −8 to −10 %); B +0.7, +0.2, +0.4 % | **fails by an improvement** (A); holds (B) |
| **J3** | arm B's frozen form at or below M_16†ᵐ_us, ≥ 3 of 4 seeds, every size | 3 of 4 at every size, −1.1, −1.7, −2.0 % (seed 1 +5 to +8 %) | **holds** |
| **J4** | the field term's gain on the C rungs falls: job 1 > A > B | 19.4 → 23.6 → 25.7 % at 64 molecules (18.6 → 23.1 → 25.4, 18.8 → 23.1 → 25.2) | **fails**: it rises |
| **J5** | no bias moves from 64 to 256 molecules by more than 2 standard errors | per seed, from the logged splits: \|z\| ≤ 1.5 on every run | holds on the logs; the report reads it exactly |

The two failures that matter are both §6 readings of the pre-registration:

- **J2 failed by an improvement.** "The field term had not absorbed the contact error, and the frozen form gains with
  it."
- **J1 failed with arm B's dipoles within 2 % of MB-pol's.** "The response's bulk cost is then not its contact
  over-polarisation." The candidates named there were the spread, the energies at orders ≥ 1, and the
  under-polarisation. §4b finds the energies at orders ≥ 1.

## 4. What the readings say

**(a) The regulator is E4's largest bulk gain so far.** The mutual edge at 2.0 Å was placed by the EFT-of-LSS hierarchy
and fixed before any of its numbers were read.

- The frozen form gains 9 % on the rigid forces at every size, on every seed: 0.605 → 0.551 at 64 molecules.
  - Net forces −11 %, torques −8 %.
  - Energies −6, −12, −19 % at 64, 128, 256 molecules.
  - Against the cutoff model M_∞ it is now −39 % on the forces.
- The ablation gains 4 % and is level with the dagger: 0.721 against 0.721, below it on 3 of 4 seeds. In job 1 it was
  +3 to +4 %, on 2 of 4 seeds.
- So the contact over-polarisation — 17 % per molecule at 0.75 Å, 6 % at 2.0 Å (§1) — was a bulk error. The field
  term had not absorbed it.
- Read through the workflow note: the regulator sits on the right side of the hierarchy, and the long-wavelength
  theory no longer carries contact physics it cannot express.

**(b) The calibration is exact in the infrared and buys nothing in bulk.** It costs the ablation, and it traded contact
energy between the read-out's orders.

- **The frozen form is unchanged:** +0.2 to +0.7 %, within ±2 % on every seed. It gets ε_∞ exact and T5 1.027. The
  energies move +8, −2 and +4 % at 64, 128 and 256 molecules, within their seed spread, and the atomic forces, which
  are not trained, +9 %.
- **The ablation pays 2–5 % on every seed**, which undoes the regulator's gain: arm B's ablation is job 1's to 0.6 %.
  The calibration therefore changed something at contact that the field term absorbs and E₀ alone does not.
- **What changed** (no training; `scratch/e4_job2_calibration_orders.py` on the 3 local test frames at 2.0 Å, order 2,
  E₀ and the field term at initialisation cancelling):
  - The read-out's own energy in the liquid is −3.96 kcal/mol per molecule in arm A and −4.37 in arm B. Its own rigid
    forces are 2.70 and 2.76 kcal/mol/Å rms.
  - The calibration changes the energy by −0.41 kcal/mol per molecule and the forces by 0.47 rms.
  - Taking only the order-0 columns from B: +0.38 kcal/mol and 0.52 rms.
  - Taking only the orders ≥ 1 from B: **−0.79 kcal/mol and 0.70 rms**.
- **The reading.** The liquid lowered the order-0 polarisability: the molecules' dipoles go from 6 % too large to 2 %
  too small. The solve moved the orders ≥ 1 as well, the field-gradient couplings. They fit the liquid rows ten times
  better and the probes a little better (§4f), and in the liquid they carry 0.8 kcal/mol per molecule of contact energy
  that neither kind of row identifies.
  - The field term reads the potential, field and field gradient at each atom and absorbs it.
  - E₀ does not, and the ablation fits worse in training too.
- **This is the derivative counting of the workflow note** (`eft-of-lss-workflow.md` §5): the k = 0 response identifies
  the order-0 coefficient, and the order-2 coefficients belong to the response's k-dependence, never to far probes.
  The stage-2 solve moved every order. §4f reads the calibration with the order-0 columns alone.

**(c) Regulated, the response pays its way beside the field term, barely.**

- The frozen form is below M_16†ᵐ_us on 3 of 4 seeds at every size: −1.8 to −2.4 % in arm A, −1.1 to −2.0 % in arm B.
  Seed 1 is 5–8 % above, as seed 1 was the frozen form's worst seed in job 1 too.
- M_16†ᵐ_us is 7 % below job 1's frozen form. So the unregulated response cost 7 % in bulk even with the field term,
  and regulated it gains 2 %.
- The response's case still rests mainly on the infrared: ε_∞, T5, the far field and Form I, which M_16†ᵐ_us does not
  have at all (probes 100 %, T5 0).

**(d) The field term is not mainly a repair.** Its gain grows as the response's contact error falls: 19 → 24 → 26 %. On
the dagger it is 22 %, the same size. It carries what neither the dagger nor the response carries, as E3-QP6 found in
clusters (the dagger with the field term had the best contact 3-body). J4's premise — that most of the gain repaired
the response's error — is wrong.

**(e) Net forces and torques.**

- The ablation's net forces were job 1's worst sign, +11 % over the dagger. They are +3.8 % in arm A and +6.9 % in arm B.
- Torques: arm A −1.3 % against the dagger, arm B +2.5 %.

**(f) The calibration on the order-0 columns alone (matching only, local frames).**

Run on the local frames at 2.0 Å with the second stage restricted to the order-0 columns and the node term
(`training.readout.liquid.orders: zero`, v0.0.25). The expectations were written before its numbers
(`scratch/e4_job2_order0_expectations.md`). Read-out order 2; at order 0 the two calibrations are the same solve and
read the same, to the digit.

| reading (MB-pol ε_∞ 1.6870) | arm A | arm B, every order | order 0 alone |
|---|---|---|---|
| ε_∞ | 1.7429 | 1.6882 | 1.6895 |
| molecules' μ; R (iso) | 8.2 %; 9.2 (8.7) % | 4.6 %; 2.2 (1.7) % | 5.0 %; 3.1 (2.3) % |
| \|μ\| ÷ MB-pol's; iso response spread (r) | 1.057; 1.71 (−0.07) | 0.968; 1.22 (0.49) | **0.991**; 1.14 (0.04) |
| far probes, 10 / 12 / 15 Å; T5 slope | 2.3 / 2.1 / 2.1 %; 1.042 | 2.1 / 2.6 / 2.8 %; 1.040 | 2.3 / 2.2 / 2.7 %; 1.040 |
| the probes' objective; the liquid rows' residual | —; 7.9×10⁻⁴ | ×1.22; 7.8×10⁻⁷ | ×1.73; 8.2×10⁻⁶ |
| the read-out's change in the liquid: energy per molecule; rigid forces | — | −0.41 kcal/mol; 0.47 rms | **+0.21 kcal/mol; 0.33 rms** |

The expectations against the numbers:

1. ε_∞ within 0.010 of MB-pol's: **holds** (1.6895).
2. The trade lands on the far probes: **fails**. Their objective rises ×1.73, not ×2 or more, and the non-additive error
   stays 2.2–2.7 %. The far field does not need the orders ≥ 1; in arm B they mostly fit the liquid rows.
3. The read-out's change in the liquid: **holds in part**. The energy is +0.21 kcal/mol per molecule, the other sign of
   arm B's −0.41 and smaller. The forces change by 0.33 rms, below the 0.52 expected: the order-0 columns move a fifth
   as far as in arm B (2 % against 10 % of their norm).

**Reading.** The order-0 columns alone do the calibration's job:

- ε_∞ and the dipoles' scale come right: \|μ\| 0.991, against arm B's 0.968;
- the far field and T5 are kept;
- the contact forces change by a third less than in arm B, and nothing is moved into the field-gradient couplings.

What the orders ≥ 1 bought in arm B — a ten-times smaller liquid residual and the molecules' response following
MB-pol's (r 0.49 against 0.04) — is what the k = 0 data do not identify as theirs. Whether it also gives back the
ablation's 2–5 % is a training question: §6.

**(g) Cost.**

- C-rung runs took 3.0–3.4 h each with their readings; M_16†ᵐ_us took 0.5 h.
- Arm B's second matching stage took 6–7 minutes on the GPU, once per arm and rung, shared by the seeds.

## 5. Notes to the pre-registration

- **The resolution in §5 was too pessimistic.** Seeds share E₀'s initialisation and batches across arms, so the paired
  arm-to-arm changes resolve about 1 % (B against A: ±2 % for the frozen form, +2 to +5 % for the ablation, every seed).
  The ±4 % from the seed spread applies only to seed means.
- **J1's paired reading against job 1's dagger** used valid + test at 128 and 256 molecules. The verdict (1 of 4) does
  not depend on it: the seed means are +2.5 to +3.6 %.
- **J2's band of ±5 % was the right size for a null.** The regulator moved the frozen form by twice that, on every seed.

## 6. Next

1. **The model.** The frozen form at the 2.0 Å mutual edge: 0.551 kcal/mol/Å (A) or 0.555 (B) on the rigid forces.
   Arm B has the infrared exactly (ε_∞ 1.6887, T5 1.027). Arm B's small costs — energies at 64 molecules, atomic
   forces — are §4b's orders ≥ 1.
2. **The calibration by derivative counting.** The second stage on the order-0 columns alone (`liquid.orders: zero`,
   v0.0.25). At the matching level it does the calibration's job and keeps the far field (§4f). Retrain the two C
   rungs with it: 2 runs × 4 seeds, about 26 GPU hours. J1's question — whether the calibrated response stops costing
   the ablation — is asked again there, with the contact forces moved by 0.33 kcal/mol/Å rms instead of 0.47 and
   nothing moved into the orders ≥ 1. *Registered 6 Nov as arm C: `e4-job2c-preregistration.md`. Read 7 Nov
   (`e4-job2c-results.md`): the ablation 1.4–1.6 % below arm A's and the frozen form 1.2–1.5 % below, on every seed,
   with ε_∞ 1.6893. Arm B's cost was the orders ≥ 1, and arm C is the model.*
3. **The locality test of the workflow note** (§5): calibrate on the liquid alone and read the far probes as an
   out-of-state prediction. Matching only.
4. **Then** the finite-k response (the order-2 coefficients, X2-C) with H3 read on it; the drift's attribution check;
   M_1 and M_16 again; the readers on neighbour lists; MACE's rigid forces beside these.
