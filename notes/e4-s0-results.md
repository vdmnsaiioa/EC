# E4, S0: the polarisation sum rules on MB-pol's trajectories — results

*Read 12 Nov 2026 against `e4-s0-preregistration.md`. Its bands and the reader's choices were committed at 22:31 CET on
11 Nov, before the reader had run on any MB-pol frame. §1 is MB-pol's own readings: S0.1–S0.3 and the readings beside
them. §2, S0.4 on arm C's polarisation, is to come: the model step stopped at 128 molecules on the first run because of
a bug in the reader, fixed since.*

## 1. MB-pol's polarisation

**The run.** `run_e4_s0_mbpol.sbatch` ran at f8788ff as five CPU tasks of 1–5 minutes each. The read ran on the login
node.

- Every frame of `data/e4` was read: 240, 120, 60, 60 and 52 frames at 64–1024 molecules.
- With the dynamics' settings, each size's first frame has induced dipoles within 0.001–0.002 % of the truth's.

| | reading | band | prediction | Niflheim |
|---|---|---|---|---|
| **S0.1** | L(k → 0), the intercept over the nine shells with k ≤ 0.46 Å⁻¹ of 128–1024 | 0.48–0.68 | 0.572–0.582 | **0.610 ± 0.030** ✓ (B = +0.31 Å², χ²/dof 0.47) |
| **S0.2** | T/L on those nine shells | ≥ 20 | ≈ ε_∞ε_s, of order 100 | **86–114** ✓ |
| **S0.3** | r for P_perm on the first shells of 512 and 1024 | 1.45–1.95 | 1.709, 1.701 | **1.979 ± 0.032, 1.992 ± 0.045** ✗ (high) |
| **S0.4** | r for arm C's polarisation on the first shells of 128 and 256 | 0.90–1.10 | 1 | §2 |

**S0.1 and S0.2 hold.** MB-pol's Born–Oppenheimer polarisation obeys the nuclear sum rules on E4's own trajectories.

- **The intercept** is 0.610 ± 0.030, 0.9σ above the predicted 0.572–0.582. The smallest shells read L = 0.612 ± 0.029
  (512, k = 0.25 Å⁻¹) and 0.622 ± 0.047 (1024, 0.20 Å⁻¹).
- **T/L** is 96–114 at k ≤ 0.36 Å⁻¹. Since T/L = ε_∞ε_s at small k, this gives ε_s ≈ 57–68.
- **ε_∞ from L.** Taking ε_s ≈ 60, the intercept gives 1/ε_∞ = 0.627 ± 0.030, so ε_∞ = 1.60 ± 0.08. The clamped responses'
  1.6885 (jobs 2–3) is 1.2σ from that, and 2.0 is 4σ from it. So the sum rule confirms the ε_∞ that the matching rests
  on, independently of S0.3.

**S0.3 fails high.** For MB-pol's own permanent sources, r is 1.98–1.99 at the first shells of 512 and 1024. The
registered band was 1.45–1.95 and the lemma's prediction ε_∞(k) = 1.70. The failure lies in b_L, not in b_T:

- **b_T is 1.454–1.461 on all fifteen shells, at every size.** This is the liquid's dipole enhancement, the note's
  a ≈ 1.4: MB-pol's molecular dipoles are 1.455 times their permanent part, in the regression's sense, at every k
  read.
- **b_L is 0.731–0.761 at k ≤ 0.36 Å⁻¹**, and 0.731–0.736 on the five shells of 512 and 1024 there. If the same
  enhancement were screened by ε_∞, it would be b_T/ε_∞(k) = 0.84–0.86. Read as b_L ε_∞(k), the longitudinal channel's
  enhancement rises with k:
  - 1.24–1.31 at k ≤ 0.36 Å⁻¹;
  - 1.30–1.38 at 0.40–0.57;
  - 1.49–1.50 at 0.69–0.71;
  - 1.65 at 0.88.
- **So r falls with k while ε_∞(k) rises.** r is 1.91–2.00 at k ≤ 0.36 Å⁻¹, 1.84–1.96 at 0.40–0.57 and 1.74–1.82 at
  0.69–0.88. Over the same range ε_∞(k) goes from 1.70 to 1.97. The two cross near 0.65 Å⁻¹, where b_L ε_∞(k) reaches
  b_T.

**What the failure is, and is not.**

- **It is not the case the registration feared.** §4 of the pre-registration asked to stop if r read near 1, which
  would mean MB-pol's zero-field polarisation lacked the electronic screening. Instead, b_L < b_T/ε_∞: the longitudinal
  channel is screened more than ε_∞ alone accounts for, and S0.1 confirms ε_∞ itself.
- **What it does say.** The lemma's r = ε_∞ needs a source whose companion polarisation is the same multiple of it in
  both channels, up to a residual uncorrelated with it in each. MB-pol's own permanent sources are not such a source.
  - The transverse multiple is constant at every k and size.
  - The longitudinal one is about 15 % smaller at the smallest k, and grows with k.
- **Two candidate readings, neither tested:**
  - *The screening of the residual.* MB-pol's induced dipoles follow each molecule's local field, and the source's
    k-component does not fix that field. In the longitudinal channel only, the long-range energy couples this residual
    to the source. That would anti-correlate the two and lower b_L most where the screening is strongest, at small k.
  - *A mean-field longitudinal channel.* On the five shells of 512 and 1024 below 0.36 Å⁻¹, b_L is within about 1 % of
    Clausius–Mossotti's longitudinal factor (ε_∞ + 2)/3ε_∞ = 0.728. b_T exceeds that model's transverse factor,
    (ε_∞ + 2)/3 = 1.23, by 18 %. In the Clausius–Mossotti model the two factors' ratio is exactly ε_∞.
- **For the architecture**, the lemma's use survives with a new scale. Field-blind sources regress onto the truth at
  r ≈ 2.0 at these k, not 1.7. A model that carries the response should read 1. The test separates the two more
  sharply than registered, but r cannot be read on the lemma's scale between 1 and ε_∞. The path-integral note's §5
  should state the lemma's assumption and this scale.

**Readings beside, not registered.**

- **MB-pol's static longitudinal function.** 1/ε_s(k) = 1/ε_∞(k) − L(k) is consistent with 1/ε_s ≈ 0.015 at
  k ≤ 0.25 Å⁻¹: −0.027 ± 0.029 (512) and −0.034 ± 0.047 (1024). It turns negative from about 0.3 Å⁻¹:
  - −0.03 to −0.12 at 0.28–0.45 (−0.089 ± 0.034 at 0.28, in the 1024 cell);
  - −0.17 at 0.51–0.57;
  - −0.26 to −0.29 at 0.69–0.71;
  - −0.45 ± 0.03 at 0.88.

  So MB-pol overscreens from about 0.3 Å⁻¹.
- **ε_s by the tin-foil formula** is 49 ± 5 to 82 ± 12 across the sizes, with only 52–240 frames. From T at the
  smallest shells it is 62–71.
- **The 64 cell's first shell** (k = 0.51 Å⁻¹) reads r = 1.87 ± 0.02 for P_perm.
- **Arm C at 64 molecules.** The model step wrote this cell before it stopped (§2). S0.4 registers 128 and 256, so this
  is a reading beside, seen before S0.4's sizes were read.
  - Arm C: r = 0.99–1.00 on all three shells, with b_L = 1.006–1.009 and b_T = 1.001–1.003.
  - Its order-0 sector alone: the same to 0.01.

## 2. S0.4: arm C's polarisation (to come)

The first run of `run_e4_s0_model.sbatch` wrote arm C's 64-molecule file and stopped at 128 molecules.

- **The cause.** The model step checked that the read-out was matched for this rung by comparing the rung's whole repr.
  That repr includes the reciprocal grid, `ewald_n_max`, which follows the cell: 22, 27 and 34 at 64, 128 and 256
  molecules. The matching ran in the 64-molecule cell, so every larger cell was refused.
- **The fix.** The check leaves the grid out (`tests/test_sumrules.py::test_model_step_reads_cells_other_than_the_
  matching_one_with_the_configured_solve`). The step also now:
  - evaluates both dipole sets in one pass;
  - solves the mutual response as the config's model does, with GMRES to 1e-10, where it had used the dense solve. The
    dense solve's cost grows as N³. On the test cells the two agree to 1e-7.

  None of this changes what S0.4 reads: the same read-out, the same frames, the same −∂E/∂E_ext,i.

## 3. S0.5 and the nuclear EFT's readings (to come)

S0.5 was registered after §1 and before any of its numbers (pre-registration §6, 12 Nov, 14:48 CET). It is the
nuclear EFT note's account of S0.3:

- the Coulomb term (2π/ε_∞V)|Π_L|² couples the induced dipoles' local-field residual to the sources in the
  longitudinal channel only;
- so r = ε_∞ + v_η, with v_η the transverse regression's residual.

The band is 0.7–1.4 on ρ = (r − ε_∞(k))/v_η, at S0.3's shells. S3's normalisation was amended with it, to
ε_∞(k) − 1. The read step computes ρ and the readings beside it from the files S0.1–S0.3 were read on.
