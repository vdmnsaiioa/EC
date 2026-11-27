# E4, job 16.5: MACE-LES trained on cells of 128 molecules — pre-registration

*28 Nov 2026, registered at this commit's time, before any run of job 16.5 and before any of its numbers. Vadic's go of
28 Nov ("register it as job 16.5": job 15 §12(b), MACE-LES's Form II beside job 16). Read against:*

- *`e4-plan.md` §4: the ladder's Form II row, MACE-LES's saturated error ∝ L₀⁻¹ (its net charge free, p_min = 1);*
- *`e4-job15-results.md` §10–§11: MACE-LES drifts −0.0085 from 64 to 256 molecules, with its charges squared, in a
  size law consistent with 1/L;*
- *`e4-job16-preregistration.md` and job 16's first reading: K16.1 held for M_inf (the step reversed, ratio 1.01).*

## 1. Why

**Job 16 asks of M_inf whether its size error is tied to its training cell. This asks the same of MACE-LES.**

- **Trained at 64 molecules, MACE-LES drifts nearly as much as the cutoff model.** Its energy per molecule moves
  −0.0047 kcal/mol to 128 molecules and −0.0085 ± 0.0006 to 256 (job 15, unpaired), against M_inf's −0.0101 and the
  dagger's +0.0008.
- **Job 15 put the drift in LES's own term.** Its released runs drift with their charges squared (0.30 times at c × 0.6,
  0.91 times at c × 0.96). The ratio of the steps, 1.8, is what an error in 1/L gives (1.79); 1/N gives 1.5.
- **Two readings of the same drift.**
  - *Carried*, as M_inf's: the short-range part absorbs the 64-molecule cell's long-range energy and carries it.
  - *In 1/L*, the plan's law for this class: an error that is a function of the test cell, f(L) ∝ 1/L, zeroed at the
    training cell. Its saturated Form II bias is then −f(L₀) ∝ 1/L₀.
- **Both reverse the step when L₀ moves (K16.1's construction).** Any error that is one function of the cells does:
  trained at L₀, the bias at L is f(L) − f(L₀).
- **They differ in how much of the drift to 256 a model trained at 128 keeps.** r = s₁₂₈(256) / s₆₄(256) is
  (f(256) − f(128)) / (f(256) − f(64)):
  - 0.44 for 1/L (cells of 12.42, 15.65 and 19.72 Å);
  - 0.33 for 1/N;
  - 0.23 for M_inf, read (job 16; 0.13 from its 64-molecule runs alone).
- **What it means for the paper.** E4's model's bias is flat in L₀ (K16.2 held: −0.0014 / −0.0023 / −0.0014). If
  MACE-LES's is 1/L₀, training it on larger cells shrinks its bulk error and never removes it: at 128 molecules, to
  0.79 times 64's.

## 2. What changes (v0.0.44), fixed before any run

- **`scripts/e4_job16_5.py`** reads K16.5.1–K16.5.5, NumPy only. It reads:
  - each model's energies from `baselines/mace/energies.py --tag job16_5` (`<run>_stagetwo_job16_5.npz`), against
    the truth in `data/e4/<split>_n<n>.npz`;
  - its latent dipole and forces from `eval.py`'s json;
  - the dagger's errors on the same frames from the harness jsons: job 11's at L₀ = 64, job 16's (task 0) at 128.
- **`baselines/mace/energies.py`:** a frame that does not fit on the GPU goes to the CPU, as `eval.py` does
  (the 1024-molecule cells). `--max_frames` is for a smoke test.
- **`run_e4_job16_5.sbatch`:** one task per seed (0, 1).
  - It trains MACE-LES at L₀ = 128 with the baseline's hyper-parameters (`baselines/mace/train.sh`, N0=128: the 90
    frames of 128 molecules, the same atoms as the 180 of 64) and evaluates it.
  - It writes both models' energies, the L₀ = 64 baseline's (job 15's runs) and the new one's.
  - `MODEL=MACE` runs the control, MACE without LES, the same way (K16.5.5).
- **Nothing else.** The L₀ = 64 MACE-LES baselines and the daggers at both L₀ are runs already made.

## 3. The readings

- **The bias at size L** is the mean energy error per molecule on the frames the training did not see.
  - At L₀ that is the test split alone: MACE's validation split chooses its checkpoints (job 16 used validation and
    test for the harness).
  - At any other size, every frame.
- **δ_L₀(L)** is MACE-LES's bias less the field-term dagger's trained at the same L₀, on the same frames, seed by
  seed, then the seed mean. The pairing removes the frames' own difficulty (the truth's energy per molecule scatters 6–12
  times more across trajectories than the drift).
- **Its step** from L₀ to L is s_L₀(L) = δ_L₀(L) − δ_L₀(L₀), as K16.1 read M_inf's.
- **The latent dipole, c,** is MACE-LES's latent charges' mean molecular dipole on the test frames of 64 molecules,
  over 1.864 D (`eval.py`, job 15).

## 4. Registered

| | reading | band | prediction |
|---|---|---|---|
| **K16.5.1** | s₁₂₈(64) against s₆₄(128) | opposite in sign, \|s₁₂₈(64) + s₆₄(128)\| ≤ 0.4 \|s₆₄(128)\|; read when \|s₆₄(128)\| ≥ 0.003 | holds |
| **K16.5.2** | r = s₁₂₈(256) / s₆₄(256) | ≤ 0.30 "carried"; 0.30–0.35 "between"; 0.35–0.80 "1/L"; > 0.80 "persists"; < 0 "reversed" | 1/L |
| **K16.5.3** | c at L₀ = 128 against L₀ = 64, seed by seed | within ±10 % | holds |
| **K16.5.4** | forces at L₀ = 128 against L₀ = 64, test frames of 64, 128 and 256 molecules (seed means) | within ±5 % | holds |
| **K16.5.5** | the control, if trained: MACE without LES, its r | ≤ 0.30, "carried" | holds |

**Why these bands, and these predictions.**

- **K16.5.1** is K16.1's band, so the two models are read alike.
  - Predicted to hold: job 15 puts the drift in LES's term, a function of the test cell's charges. Training at 128
    should move only the constant.
  - *The risk is noise.* The step to 128 is about half M_inf's (−0.005 to −0.006 paired, against −0.0098). 40 % of it is
    about 0.002: one to two frame standard errors of the sum (about 0.001 a step). A failure within that is reported as
    one, with the numbers.
- **K16.5.2** classifies the law; it is not a band around one value.
  - The edges sit between the laws: 0.30 above M_inf's 0.23, 0.35 above 1/N's 0.33, 0.80 well above 1/L's 0.44.
  - Predicted "1/L": job 15's unpaired steps, read as one function of the cells, give r = 1 − 0.0047 / 0.0085 = 0.45.
  - *The frames separate "carried" from "1/L" by about 1.5 standard errors.* With s₆₄(256) about −0.010 and about
    0.001 on each step, r carries about ±0.1–0.15. A "between" reading is no verdict.
- **K16.5.3.** Arm E: MACE-LES ends at c = 0.59–0.61 across units 0.05–0.15 and two seeds. 10 % is three to six times
  that spread. Predicted to hold: its scale did not follow its unit, so the cell is not expected to move it.
- **K16.5.4** is K16.3's band. The same atoms, half the frames of twice the size. MACE-LES's forces at 64 are 0.20–0.22
  kcal/mol/Å.
- **K16.5.5.** Plain MACE reaches 12 Å (two layers of 6 Å), as M_inf's E₀ does. If it is the cutoff that carries, it
  carries as M_inf does. Its drift at 64 has not been read before.

## 5. Read beside, not registered

- **K16.5.2's r as the 64-molecule runs alone predict it,** 1 − s₆₄(128) / s₆₄(256). Equal to r if the error is one
  function of the cells.
- **MACE-LES at L₀ = 128 less at 64, on the same frames, at every size to 1024 molecules.**
  - It is f(64) − f(128) at every size if the error is one function of the cells.
  - It needs no dagger, so it reaches the test frames of 512 and 1024 molecules, which the harness does not read.
- **The net latent charge per molecule** at both L₀ (job 15: 0.021–0.025 e rms).
- **MACE without LES:** its steps, K16.1's construction, the L₀ difference by size and its forces.
- **Each δ's unpaired bias,** and the seed and frame spreads.

## 6. What would change the reading

- **K16.5.1 holds and K16.5.2 reads "1/L"** (predicted): MACE-LES's bulk error is the plan's law for its class, ∝ 1/L in
  the test cell and ∝ 1/L₀ in Form II. With job 15's c² law, it sits in LES's own term.
  - Training on larger cells shrinks it and never removes it. Ours is flat in L₀ (K16.2): that is the paper's Form II
    row.
- **K16.5.2 reads "carried":** MACE-LES's drift behaves as M_inf's. That conflicts with job 15's c² law. Read the
  attribution without training (job 15 §12(a)).
- **"Between":** 1/N or a mixture; the frames do not separate them. Read r from the 64-molecule runs and the L₀
  difference at 512 and 1024 molecules.
- **"Persists" or "reversed" with K16.5.1 holding:** the error is not one function of the cells after all. Read the L₀
  difference by size.
- **K16.5.1 fails near zero:** trained at 128, MACE-LES has no step back to 64. Its drift was tied to the 64-molecule
  cell (its 12 Å reach spanning the 12.4 Å cell), not to the test cell.
- **K16.5.1 fails with the same sign:** the error grows away from the training cell in either direction. That is an
  extrapolation in the cell, not a law in L.
- **K16.5.3 fails:** MACE-LES's scale depends on its training cell, though not on its unit or seed. Then K16.5.1–K16.5.2
  compare different c². Read the steps scaled by c² (job 15's law).
- **K16.5.4 fails:** the training cell changes MACE-LES's forces, where it did not change ours (K16.3).
- **K16.5.5 fails:** plain MACE does not carry as M_inf does. M_inf's carry is then E₀'s, not every cutoff model's. If
  plain MACE also reads "1/L", the law is not LES's.

## 7. Not tested here

- **MACE-LES at L₀ = 256.**
- **The drift's attribution** (job 15 §12(a): LES's energy with each molecule's net latent charge removed, or only the
  cell's).
- **The plan's four seeds;** two here, as job 16.
- **MACE-LES with neutral molecules,** a sum rule it does not have.

## 8. Cost

- **Training at 128:** the same atoms per epoch as at 64 (90 frames of 384 atoms, 45 batches of two), 500 epochs.
  About 6–8 h a seed (estimated; not timed at 128).
- **The energies:** 15 files for each of two models. Minutes on the GPU up to 256 molecules. The 1024-molecule cells
  may go to the CPU, which could take hours.
- **Two GPU tasks.** The sbatch asks 2 days a task; a task that runs out of time is finished by submitting it again.
- **The control (MODEL=MACE):** two more tasks, less than MACE-LES's.

## 9. Numbers before any run

| | s₆₄(128) | s₆₄(256) | s₁₂₈(64) | s₁₂₈(256) | r |
|---|---|---|---|---|---|
| M_inf (job 16, paired against the dagger) | −0.0098 | −0.0113 | +0.0098 | −0.0026 | 0.23 |
| MACE-LES (job 15 E1's means, unpaired) | −0.0047 | −0.0085 | — | — | 0.45 if one function of the cells |
| 1/L, scaled to MACE-LES's s₆₄(256) | −0.0047 | −0.0085 | +0.0047 | −0.0038 | 0.44 |

- **The dagger's own drift, 64 to 256:** +0.0008 (job 15), +0.0021 (job 16's reading). Paired, s₆₄(256) is then about
  −0.009 to −0.011.
- **MACE-LES at 64:** c 0.59–0.61 (1.10–1.13 D); forces 0.20–0.22 kcal/mol/Å; net latent charge 0.021–0.025 e rms per
  molecule.
- **The cells:** 12.42, 15.65 and 19.72 Å for 64, 128 and 256 molecules; 24.84 and 31.30 Å for 512 and 1024.
