# E4, job 2, arm C: the results

*7 Nov 2026. Arm C ran as pre-registered (`e4-job2c-preregistration.md`) on Niflheim at e000642 (v0.0.26). Vadic ran
its eight runs as one task each (`scripts/e4_water.py train --task i`), to fit the GPUs' time limit, all on one node.
All eight finished, in 2.9–3.4 h each: 25.7 GPU hours (26 estimated).*

*Read from the eight task logs, pooled as in `e4-job2-results.md` (frame-weighted over the logged splits, the way
`scripts/e4_report.py` pools them). Arms A and B are job 2's runs; job 1's rungs are taken from its report. As for job
2, two readings come exactly from `scripts/e4_report.py --job2 results/e4_job2a,results/e4_job2b,results/e4_job2c`
when it is run:*

- *J5;*
- *the seed-by-seed comparisons with job 1's dagger at 128 and 256 molecules on every frame. Here they use the
  validation and test frames.*

*Notation as in the pre-registration. The frozen form is M_16C2†ᵐ_us, the ablation M_16C2†ᵐ, the dagger M_16†ᵐ (job
1's run), and M_16†ᵐ_us the dagger with the frozen form's field term (arm A).*

- *Arm A: the 2.0 Å mutual edge, matched on the far probes.*
- *Arm B: arm A's read-out calibrated on MB-pol's liquid response, every order.*
- *Arm C: the same calibration on the order-0 columns alone.*

*Forces are rigid-body rmse in kcal/mol/Å on the unseen frames pooled: valid + test at 64 molecules, every frame at 128
and 256. Values are seed means ± sd over 4 seeds.*

## 1. The readings fixed before training (C0): as computed

| reading | pre-registered (local, 10 cells) | arm C | arm A | arm B |
|---|---|---|---|---|
| ε_∞, 10 held-out liquid frames (MB-pol's) | 1.6895 (1.6870) | **1.6893** (1.6885) | 1.7434 | 1.6887 |
| ε_∞, job 1's 10 test frames | — | 1.6889 | 1.7443 | 1.6881 |
| cells' induced dipole | 3.1 % | 1.5 % | 7.0 % | 0.9 % |
| molecules' induced dipole; field response (iso) | 5.0 %; 3.1 (2.3) % | 5.1 %; 2.9 (1.5) % | 8.3 %; 9.0 (8.5) % | 4.4 %; 2.0 (1.3) % |
| \|μ\| ÷ MB-pol's (r) | 0.991 (0.999) | **0.991** (0.999) | 1.060 (0.998) | 0.979 (0.999) |
| iso response spread ÷ MB-pol's (r) | 1.14 (0.04) | 0.84 (0.42) | 1.58 (−0.15) | 1.13 (0.71) |
| far probes, non-additive, 10 / 12 / 15 Å | 2.3 / 2.2 / 2.7 % | 2 / 2 / 3 % | 2 / 2 / 2 % | 2 / 2 / 3 % |
| T5 slope (rmse) | 1.040 (4.1 %) | 1.0405 (4 %) | 1.0424 (5 %) | 1.0273 (3 %) |
| the probes' objective in the calibration | ×1.73 | ×1.62 | — | ×1.24 |
| the liquid rows' objective, from 1.21×10⁻³ | — | 1.29×10⁻⁵ | — | 2.4×10⁻⁶ |

The probes' objective is the stage's objective less the liquid rows', against stage 1's 9.696×10⁻⁶.

- **All eight runs read the same, to the printed digit.**
  - Stage 1 reached 9.696×10⁻⁶, as in arm A, and stage 2 followed the same path in every task.
  - The liquid and T5 readings are the same before and after training: the read-out is held, and nothing else sees a
    uniform field.
- **C0 holds on every criterion:**
  - ε_∞ is within 0.0008 of MB-pol's;
  - \|μ\| is 0.991;
  - the far probes are within +1 point of arm A's (3 % at 15 Å against 2 %);
  - T5's slope is 1.0405;
  - the probes' objective rose ×1.62.
- **On 30 cells.** The order-0 columns lower the liquid rows 93-fold; arm B's every-order solve lowered them 500-fold.
  - The dipoles' scale is the local check's to the digit.
  - The cells' dipole is closer (1.5 % against 3.1 %).
  - The molecules' response pattern differs from the local one (spread 0.84, r 0.42, against 1.14 and 0.04): with 30
    cells the order-0 columns shape it too. This was not registered.

## 2. The bulk readings

| run | F 64 | F 128 | F 256 | E 64 | E 128 | E 256 |
|---|---|---|---|---|---|---|
| M_∞ (job 1) | 0.907 ± 0.069 | 0.917 ± 0.070 | 0.929 ± 0.066 | 0.0329 | 0.0254 | 0.0200 |
| dagger (job 1) | 0.721 ± 0.022 | 0.735 ± 0.021 | 0.738 ± 0.018 | 0.0309 | 0.0221 | 0.0140 |
| M_16†ᵐ_us | 0.561 ± 0.018 | 0.574 ± 0.018 | 0.577 ± 0.017 | 0.0269 | 0.0201 | 0.0135 |
| ablation, job 1 | 0.751 ± 0.061 | 0.760 ± 0.059 | 0.761 ± 0.062 | 0.0335 | 0.0228 | 0.0134 |
| ablation, A | 0.721 ± 0.052 | 0.731 ± 0.050 | 0.733 ± 0.054 | 0.0321 | 0.0230 | 0.0133 |
| ablation, B | 0.747 ± 0.049 | 0.756 ± 0.047 | 0.757 ± 0.051 | 0.0346 | 0.0248 | 0.0136 |
| **ablation, C** | **0.710 ± 0.057** | **0.720 ± 0.054** | **0.723 ± 0.057** | 0.0317 | 0.0226 | 0.0127 |
| frozen form, job 1 | 0.605 ± 0.032 | 0.619 ± 0.032 | 0.618 ± 0.030 | 0.0259 | 0.0212 | 0.0122 |
| frozen form, A | 0.551 ± 0.032 | 0.563 ± 0.030 | 0.563 ± 0.029 | 0.0244 | 0.0187 | 0.0099 |
| frozen form, B | 0.555 ± 0.031 | 0.564 ± 0.028 | 0.566 ± 0.026 | 0.0265 | 0.0182 | 0.0103 |
| **frozen form, C** | **0.545 ± 0.028** | **0.554 ± 0.025** | **0.557 ± 0.025** | **0.0243** | **0.0181** | **0.0096** |

Energies are pooled rmse per molecule, kcal/mol. Their seed sds are 0.0009–0.0034.

**Paired by seed.** Every run of a seed shares E₀'s initialisation and batches across the arms and the tasks. `fit`
draws both from the seed alone (`jax.random.PRNGKey(seed)`, `np.random.default_rng(seed)`), so splitting the runs into
tasks leaves the pairing intact.

| change of the rigid forces, per seed (64 / 128 / 256 molecules) | seeds 0, 1, 2, 3 |
|---|---|
| ablation, C against A | −1.5, −0.6, −1.6, −2.6 / −1.8, −0.8, −1.3, −2.5 / −1.5, −0.9, −1.1, −2.0 % |
| frozen form, C against A | −0.3, −2.1, −1.5, −0.8 / −0.5, −2.6, −1.7, −0.9 / −0.1, −2.0, −1.6, −1.0 % |
| ablation, C against B | −5.7, −2.6, −5.7, −6.0 / −5.5, −2.5, −5.9, −5.1 / −5.0, −2.6, −5.6, −4.8 % |
| frozen form, C against B | −0.6, −2.9, −1.3, −2.6 / −0.4, −2.8, −0.9, −2.5 / −0.4, −2.3, −1.1, −2.6 % |
| ablation, C against the dagger (valid + test) | −2.6, +8.4, −5.2, −6.8 / −3.6, +6.9, −4.6, −7.8 / −2.9, +5.7, −4.4, −7.8 % |
| frozen form, C against M_16†ᵐ_us | −5.8, +4.5, −6.0, −4.5 / −4.9, +3.5, −6.3, −5.7 / −5.0, +2.8, −6.0, −5.8 % |

- The paired changes against arm A have a seed sd of 0.5–0.9 %, so their means (−1.2 to −1.6 %) lie 3–6 standard
  errors from zero.
- The training losses fall with them:
  - the frozen form's final loss is 24.1, 28.1, 25.7, 22.0, against 24.1, 29.3, 26.4, 22.3 in arm A;
  - the ablation's is 38.5, 49.0, 39.9, 33.7, against 39.7, 50.2, 40.8, 35.2: 2–4 % lower on every seed, where arm B was
    4–10 % higher.
- The gap from the training frames to the unseen ones at 64 molecules is the same in every arm: −0.1 % for the frozen
  form, +2.1 to +2.5 % for the ablation.

## 3. The predictions

| | prediction | reading (unseen frames) | verdict |
|---|---|---|---|
| **C0** | the calibration's fixed readings on Niflheim | §1: ε_∞ 1.6893 (MB-pol 1.6885), \|μ\| 0.991, probes 2 / 2 / 3 %, T5 1.0405, probes' objective ×1.62. The same in all eight runs, before and after training | **holds** |
| **C1** | arm C's ablation within +1 % of arm A's (seed means). Reading (ii) predicted +1.8 % | **−1.5, −1.6, −1.4 %** | **holds** |
| **C2** | arm C's ablation at or below arm A's: reading (i) | 4 of 4 seeds at every size (−0.6 to −2.6 %) | **holds** |
| **C3** | J1 again: arm C's ablation at or below the dagger | 3 of 4 seeds at every size (valid + test); seed means −1.5, −2.0, −2.1 % | **holds** |
| **C4** | arm C's frozen form within ±1 % of arm A's (seed means) | **−1.2, −1.5, −1.2 %**; every seed at or below (−0.1 to −2.6 %) | **fails by an improvement** |

J5 is read on the logs too: no run's bias moves from 64 to 256 molecules by more than 1.1 standard errors.

## 4. What the readings say

**(a) Reading (i): arm B's cost was what moved, not how much.**

- Arm C has the infrared arm B had: ε_∞ within 0.001 of MB-pol's and the dipoles within 2 %. Yet its ablation is
  4.5–4.9 % below arm B's and 1.4–1.6 % below arm A's, on every seed.
- Reading (ii) priced arm C's smaller change at contact (0.33 kcal/mol/Å rms against 0.47) at +1.8 %. The ablation
  gained instead.
- **The size of arm B's extra moves.** Arm B moved the read-out beyond arm C's solution in two ways
  (`e4-job2-results.md` §4b, §4f): it moved the orders ≥ 1, and it moved the order-0 columns five times as far, the
  two together.
  - That costs the ablation +4.7 to +5.2 % and the frozen form +1.6 to +1.9 %.
  - The field term absorbs about two-thirds of it, which is why arm B's frozen form looked unchanged.
- **The derivative counting holds in training.**
  - The k = 0 response identifies the order-0 coefficient.
  - Letting it set the field-gradient couplings costs bulk accuracy that no infrared or far-field reading shows: arm
    B's ε_∞, far probes and T5 were as good as arm C's.

**(b) The dipoles' scale was a bulk error: C4 fails by an improvement.**

- **Both rungs gain alike.** The calibration takes the molecules' dipoles in the liquid from 6 % too large to 1 % too
  small.
  - The frozen form gains 1.2–1.5 %, the ablation 1.4–1.6 %, on every seed.
  - Net forces and torques gain 1–2 % on both.
- **Neither E₀ nor the field term had absorbed it.** The field term's gain is unchanged: 23.3, 23.0, 23.0 % against arm
  A's 23.6, 23.1, 23.1 %. This is the reading the pre-registration's §5 gave for this outcome.
- **It changes the fit, not the generalisation.** The training losses fall with it, and the gap to the unseen frames is
  unchanged.
- **It is small beside the regulator.**
  - The regulator took the dipoles from 17 % to 6 % too large, and bought the frozen form 9 % and the ablation 4 %.
  - The calibration takes them from 6 % too large to 1 % too small, and buys 1.2–1.5 % and 1.4–1.6 %.

**(c) Arm C is the model.**

- **The bulk.** Rigid forces 0.545 / 0.554 / 0.557 kcal/mol/Å at 64 / 128 / 256 molecules, energies 0.0243 / 0.0181 /
  0.0096 kcal/mol per molecule.
  - It is below arms A and B on every seed and at every size.
  - Against job 1's frozen form: −10 % on the forces (−8 to −11 % on every seed) and −6, −15, −21 % on the energies.
  - Against M_∞: −40 %.
- **The infrared.** ε_∞ 1.6893 against MB-pol's 1.6885, the dipoles' scale 0.991, T5 1.040, the far probes 2 / 2 / 3 %.
- **Calibration in the state of use**, restricted to what its data identify, gets the infrared exactly and gains in bulk.

**(d) The response now pays its way with and without the field term, by 2–3 %.**

- **Without the field term.** The ablation is below the dagger: −1.5, −2.0, −2.1 % (3 of 4 seeds).
  - That was +3 to +4 % in job 1, 0.0 to −0.7 % in arm A and +2.5 to +3.6 % in arm B.
  - Its net forces are level with the dagger's (+1.4, −0.4, −0.5 %; job 1's were +11 % at 64 molecules). Its torques
    are 2.5–2.8 % below.
- **With the field term.** The frozen form is below M_16†ᵐ_us by 3.0, 3.4 and 3.5 % (3 of 4 seeds), against arm A's
  1.8–2.4 %.
- **Seed 1 is above in both comparisons** (+3 to +8 %), as in arms A and B. It was the frozen form's worst seed in job
  1 too.
- **The margins are small.** The response's case still rests mainly on the infrared: ε_∞, T5, the far field and Form I.
  The dagger has none of these.

**(e) The rest.**

- **Energies.** Arm C is below arm A at every size on both rungs (−0.7 to −4.0 %), within the seed spread.
- **Atomic forces**, which are not trained. The frozen form is +3 % against arm A (arm B was +9 %); the ablation −1.5 %.
- **The far probes after training** behave as in arms A and B.
  - At 10–15 Å they are unchanged.
  - At 8 Å they move with E₀: non-additive 2–6 %, the ablation's seed 2 the largest, as in arm A.

**(f) Cost.**

- The runs took 2.9–3.4 h each: 25.7 GPU hours.
- The eight tasks started together, so none found the matched read-out's cache, and each matched the read-out itself:
  15–17 minutes a task, about 2.1 of the 25.7 GPU hours.

## 5. Notes to the pre-registration

- **C4's band of ±1 % was at the resolution.**
  - The paired changes' seed sd is 0.8–0.9 % for the frozen form. A mean change of −1.2 % is three standard errors.
  - The failure is by 0.2–0.5 points, with every seed on the same side.
- **C3's paired reading** used valid + test at 128 and 256 molecules, as J1's did. The three seeds below the dagger are
  below by 2.6–7.8 %, so the full frames will not change the count.
- **The comparison column for arm B in §4** (+3.6, +3.4, +3.3 %) was computed from rounded seed means. From the runs'
  own values it is +3.6, +3.3, +3.3 %.

## 6. Next

1. **The model.** Arm C's frozen form: M_16C2†ᵐ_us at the 2.0 Å mutual edge, its read-out matched on the far probes and
   its order-0 columns calibrated on MB-pol's liquid response (`configs/e4_job2c.yaml`). The readers, Form I and the L₀
   sweep are read on it.
2. **The order-2 coefficients, by the finite-k response.** This is the workflow note's move 5 for the derivative terms,
   and X2-C's modulation response.
   - **Why.** Arm C's orders ≥ 1 are still the far probes' solution under the ridge, and the far probes do not
     identify them (Lemma 3).
   - **The data.** MB-pol's induced dipoles under a longitudinal field ∝ k̂ cos(k·r), on the same liquid cells, at the
     cells' lowest wavevectors: 2π/L ≈ 0.5 Å⁻¹ at 64 molecules, ka ≈ 0.5.
     - MBX takes a field per site through the call the uniform response already uses
       (`set_potential_and_electric_field_on_sites_`).
     - So the data are a CPU job like `run_e4_response.sbatch`, which took 7 solves a frame and under an hour. Here it
       is two phases per wavevector, six solves for the three axes.
   - **The model** needs the external field's tensors per atom up to order 2, added to the band field's: the read-out's
     orders 1 and 2 read the field's first and second derivatives.
   - **Order of work.** Read it on arm C first, as a prediction: matching level, no training. Only if the prediction
     misses, calibrate the orders ≥ 1 on these data and train.
3. **The locality test** (workflow note §5): calibrate the order-0 columns on the liquid alone, then read the far probes
   as out-of-state predictions. Matching only; it can run beside item 2's reading.
4. **Then:**
   - H3 on the finite-k response at two edges;
   - the readers: Form I, the near field, and 512–1024 molecules on neighbour lists;
   - the drift's attribution;
   - M_1 and M_16 again;
   - MACE's rigid forces beside these.
