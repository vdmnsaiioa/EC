# E4, job 16.5: MACE-LES trained on cells of 128 molecules — results

*Read 28 Nov 2026 against `e4-job16.5-preregistration.md`, registered on 28 Nov before any run. Its §5 gained three
readings, beside the registered ones, after job 16's read and before any job 16.5 number. `run_e4_job16_5.sbatch` ran
on Niflheim: two tasks, each training MACE-LES at L₀ = 128 in about 6.1 h (21994 s and 22287 s) and writing both L₀'s
energies in about 3–4 minutes a model; no step failed and no frame went to the CPU. MACE without LES was not run.*

- *b is MACE-LES's own bias, the mean energy error per molecule on the frames the training did not see (at L₀ its test
  split, elsewhere every frame), kcal/mol per molecule, seed means. δ is b less the dagger's at the same L₀.*

## 1. The registered readings

| | band | Niflheim | verdict | predicted |
|---|---|---|---|---|
| **K16.5.1** | s₁₂₈(64) opposite to s₆₄(128), within 40 % | −0.0065, +0.0045 (ratio 0.69) | **holds** | holds |
| **K16.5.2** | r = s₁₂₈(256) / s₆₄(256): carried ≤ 0.30, 1/L 0.35–0.80 | 0.65 | **"1/L"** | 1/L |
| **K16.5.3** | the latent dipole at 128 within ±10 % of 64's | 0.604 → 0.644, 0.592 → 0.623 (+6.6, +5.2 %) | **holds** | holds |
| **K16.5.4** | forces at 128 within ±5 % of 64's | +9.7 / +9.5 / +9.2 % | **fails** | holds |
| K16.5.5 | plain MACE "carried" | not trained at 128 | not read | holds |

Three of the four read hold or read as predicted, and K16.5.4 fails. K16.5.2's verdict carries no weight (§2).

## 2. What the readings do and do not say

- **K16.5.2's r is 0.65 ± 0.62.** Its steps lean on the bias at the training size, read on the test split alone: 15
  frames of one trajectory at 128, where the two seeds differ by 0.0034. The estimators added beside read 0.20 ± 0.19
  (from the frames off both L₀) and 0.24 (from the 64-molecule runs alone).
- **The training size shifts MACE-LES by a constant.** At L₀ = 128 less at 64, on the same frames: +0.0053 / +0.0041 /
  +0.0040 / +0.0043 / +0.0040 at 64 / 128 / 256 / 512 / 1024 molecules. One function of the cells, to 1024.
- **K16.5.4 fails on the training, most likely, not the cell.** MACE's batch stayed at two frames at both sizes, so the
  128-molecule runs took half the optimizer steps (45 batches an epoch against 90, 500 epochs). Job 16 kept the atoms
  per batch for our model; this job should have set `BATCH=1` at 128. The verdict stands; its reading is not the
  cell's.

## 3. Beyond 256 molecules MACE-LES's bias turns back

| b (kcal/mol per molecule) | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| MACE-LES, L₀ = 64 | +0.0002 | −0.0043 | −0.0077 | −0.0019 | −0.0029 |
| MACE-LES, L₀ = 128 | +0.0052 | +0.0003 | −0.0037 | +0.0024 | +0.0011 |
| plain MACE, L₀ = 64, seed 0 (valid, test) | −0.0013, −0.0004 | −0.0013, −0.0004 | −0.0028, −0.0038 | −0.0037 | −0.0039 |

- **From 256 to 1024 molecules MACE-LES's bias rises by +0.0048 ± 0.0005 at both L₀.** An error in 1/L moves it −0.0050
  there and a carried one about none. Neither law has it.
- **It is not the frames' noise.** By trajectory (four a size, seed means, L₀ = 64): −0.0083, −0.0089, −0.0072, −0.0064
  at 256; −0.0021, −0.0017, −0.0016, −0.0021 at 512; −0.0035, −0.0023, −0.0031, −0.0027 at 1024. Every trajectory at
  256 lies 0.004–0.007 below every one at 512.
- **It is not the frames shifting every model.** Plain MACE, the same architecture without LES, drifts −0.003 to 256
  and stays there at 512 and 1024 (−0.0037, −0.0039): a cutoff model's carried error, as M_inf's.
- **So it is LES's own term.** It goes with the charges squared (job 15) and is a function of the test cell alone
  (K16.5.1, the constant L₀ shift).

## 4. A candidate: LES's k-space cutoff

LES's periodic Ewald (les 0.2.0, the legacy path these models use) sums its Gaussian charges (σ = 1 Å) in reciprocal
space alone, up to |k| = 2π / dl = π / Å (dl = 2 Å). Beyond that it leaves out about 0.05 kcal/mol per molecule of
water: exp(−π²/2) is not small, and S(k) near π / Å is each molecule's own charge distribution. Training absorbs the
continuum part, the same in any cell. But the k vectors just inside π / Å lie on the cell's reciprocal lattice, and how
much of the shell they fill does not vary smoothly with the cell's size.

A deterministic estimate uses each rigid molecule's orientation-averaged charge form factor and q_H = 0.2 e (c ≈ 0.6),
on the k lattice of each cell. It gives the shift of a model trained at 64 molecules, −(T(L) − T(64)):

| | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| the estimate, q_H = 0.2 | −0.0029 | −0.0052 | −0.0011 | −0.0022 |
| the same, its scale fitted (× 1.5, q_H ≈ 0.245) | −0.0044 | −0.0077 | −0.0016 | −0.0033 |
| MACE-LES, L₀ = 64, b less b(64) | −0.0045 | −0.0079 | −0.0021 | −0.0031 |

- **The pattern is the observed one,** non-monotonic, deepest at 256 and back at 512, with one fitted scale.
- **It would explain job 15's readings too:** a term quadratic in the charges, and the ratio of the steps to 256 and 128
  (1.75 here, 1.8 read) that looked like 1/L.
- **It would explain K16.5.1 and the constant L₀ shift:** a model trained at 128 absorbs T(128) instead of T(64), about
  +0.0044 everywhere (read: +0.0040 to +0.0053).
- **Not yet shown.** The estimate's scale is fitted, its form factor is a rigid molecule's, and the slabs of job 17 are
  periodic too (their tail changes with the gap). Job 16.6 (`e4-job16.6-preregistration.md`) computes the tail of
  every trained model on every frame, from their own latent charges, without training. Its predictions are registered
  before it runs.

## 5. What it means, if job 16.6 confirms it

- **MACE-LES's bulk size dependence would be its implementation's cutoff, not its class.** The plan's Form II row
  (MACE-LES ∝ 1/L₀) and job 15 §10's reading ("in LES's own term, consistent with 1/L") would both be revised. The
  paper could not claim a Form II difference against MACE-LES from these runs.
- **Job 17's clusters stand:** open frames go through LES's direct sum, which has no cutoff.
- **LES's default dl = 2 Å would be too coarse for water at σ = 1 Å.** dl = 1 Å takes the tail to about 10⁻⁹.
- **Our Form II result stands against the cutoff models** (M_inf, plain MACE: carried); ours follows the cell.

## 6. Next

1. **Job 16.6:** the tails, no training, about half an hour on one GPU.
2. *If the control is wanted:* MACE without LES at 128, and MACE-LES again, both with `BATCH=1` (the atoms per step
   matched); about 6 h a seed each.
