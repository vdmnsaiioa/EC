# E4, job 16: the training cells of 128 molecules — results

*Read 28 Nov 2026 against `e4-job16-preregistration.md`, registered on 26 Nov before any run. `run_e4_job16.sbatch`'s
three tasks ran on Niflheim (RTX 3090s) at v0.0.41: task 0 (E4's model at 128) in about 24 h, task 1 (the dagger and
M_inf at 128, M_inf polished at 64, the warm starts at 128) in about 10 h, task 2 (the warm starts at 256) last. Read with
`scripts/e4_job16.py`. Every polish ran its 1000 L-BFGS steps inside its budget. The reader finds no step count in the
64-molecule M_inf polish's jsons, whose readings are there.*

- *Bias is the mean energy error per molecule on the unseen frames (at L₀ the validation and test splits, at any other
  size every frame), in kcal/mol per molecule. δ_L₀(L) is M_inf's bias less the dagger's on the same frames, seed by
  seed, then the seed mean. s_L₀(L) = δ_L₀(L) − δ_L₀(L₀).*
- *The gap is a seed's polished end from 1.5 less its end from 0.5, in c, the learned dipole on the test frames of 64
  molecules over 1.864 D.*

## 1. The registered readings

| | band | Niflheim | verdict | predicted |
|---|---|---|---|---|
| **K16.1** | s₁₂₈(64) opposite to s₆₄(128), within 40 % | −0.0098 and +0.0098, ratio 1.01 | **holds** | holds |
| **K16.2** | E4's model's bias at L₀ = 128 within ±0.003 of L₀ = 64's, at 64 / 128 / 256 | −0.0014 / −0.0023 / −0.0014 | **holds** | holds |
| **K16.3** | E4's model's forces within ±5 % | +1.7 / −0.3 / +0.4 % | **holds** | holds |
| **K16.4** | the gap at 128 (coverage 99 %): ≥ 0.25 frozen, ≤ 0.15 mobile | 0.211, 0.247 | **partial** | frozen |
| **K16.5** | the gap at 256 (coverage 82 %) | 0.195, 0.268 | **partial** | partial |

Four of five as predicted. K16.4 reads partial where frozen was predicted, seed 1 by 0.003.

## 2. Form II: the cutoff model carries its training cell; ours follows the cell

| δ_L₀(L): M_inf less the dagger | n 64 | n 128 | n 256 |
|---|---|---|---|
| L₀ = 64 | −0.0011 (frame se 0.0032) | −0.0109 (0.0018) | −0.0124 (0.0016) |
| L₀ = 128 | +0.0110 (0.0017) | +0.0012 (0.0031) | −0.0014 (0.0017) |
| L₀ = 128 less L₀ = 64 | +0.0121 | +0.0121 | +0.0110 |

- **K16.1: the step reverses exactly.** Trained at 128, M_inf under-binds the 64-molecule cells by as much as the
  64-trained model over-binds 128. The two L₀ differ by a constant, +0.011 to +0.012 at every size. Its error is one
  function of the cells, zeroed at the training cell.
  - *The frames' standard errors are 2–3 times the 0.001 the pre-registration assumed.* Taken as independent, the sum
    s₆₄(128) + s₁₂₈(64) carries about 0.005, more than the band's 0.004. It reads 0.0000, and the L₀ difference is
    constant to 0.001. The two L₀'s models err alike frame by frame, so their differences are far better fixed than
    the frame errors say.
- **The law: most of the drift lies between 64 and 128 molecules.**
  - r = s₁₂₈(256) / s₆₄(256) = 0.23 ± 0.32.
  - From the frames off both L₀, (δ₆₄(256) − δ₆₄(128)) / (δ₁₂₈(256) − δ₁₂₈(64)) = 0.12 ± 0.20.
  - From the 64-molecule runs alone, 1 − s₆₄(128) / s₆₄(256) = 0.13.
  - An error in 1/L gives 0.44 and one in 1/N 0.33. The readings favour a training-cell effect that is spent by 128
    molecules, at one to two standard errors.
  - That fits job 1 §4b's reading: E₀'s 12 Å reach spans the 12.4 Å training cell and absorbs its long-range energy.
- **E4's model follows the cell (K16.2).** Within each L₀ its bias varies by at most 0.0021 across sizes. Trained at
  128 it is lower by a near-uniform 0.0014–0.0023.

| bias by rung, seed mean ± sd | n 64 | n 128 | n 256 | forces n 64 / 128 / 256 |
|---|---|---|---|---|
| M_inf, L₀ = 64 | −0.0032 ± 0.0015 | −0.0095 ± 0.0002 | −0.0124 ± 0.0014 | 0.731 / 0.748 / 0.763 |
| M_inf, L₀ = 128 | +0.0074 ± 0.0000 | −0.0044 ± 0.0022 | −0.0039 ± 0.0008 | 0.718 / 0.734 / 0.752 |
| dagger, L₀ = 64 | −0.0021 ± 0.0001 | +0.0014 ± 0.0002 | +0.0000 ± 0.0003 | 0.440 / 0.453 / 0.456 |
| dagger, L₀ = 128 | −0.0036 ± 0.0008 | −0.0056 ± 0.0028 | −0.0025 ± 0.0017 | 0.448 / 0.451 / 0.457 |
| E4's model, L₀ = 64 | +0.0000 ± 0.0001 | +0.0016 ± 0.0005 | −0.0005 ± 0.0003 | 0.443 / 0.445 / 0.448 |
| E4's model, L₀ = 128 | −0.0014 ± 0.0008 | −0.0007 ± 0.0010 | −0.0019 ± 0.0011 | 0.450 / 0.444 / 0.450 |

- **The dagger at 128 is noisier between seeds** (sd up to 0.0028), though its forces are the same.
- **M_inf's forces are 1.4–1.9 % better trained at 128.** The pinned rungs' forces do not move.
- **E4's model's ε_∞,** on the test frames of the training size: 1.6889 at L₀ = 64 and 1.6893 at 128, against
  MB-pol's 1.6885. The read-out is held.
- **For the paper's Form II row:** the cutoff model's bulk error is tied to L₀; the plan's "flat in L₀" for M_∞ is wrong,
  as job 1 said. E4's model's is flat in L₀ and follows the cell. MACE-LES's is job 16.5's question.

## 3. The cell test: in larger cells the scale keeps moving, slowly

| cell | coverage | first-stage gap | ends from 0.5 | ends from 1.5 | gap | gap at 500 steps | its change in the last 250 |
|---|---|---|---|---|---|---|---|
| 64 (12.4 Å) | 100 % | 0.669, 0.669 | 0.807, 0.884 | 1.131, 1.171 | 0.324, 0.288 | 0.364, 0.357 | −0.029, −0.020 |
| 128 (15.6 Å) | 99 % | 0.628, 0.623 | 0.868, 0.872 | 1.079, 1.119 | 0.211, 0.247 | 0.358, 0.327 | −0.057, −0.033 |
| 256 (19.7 Å) | 82 % | 0.659, 0.604 | 0.878, 0.906 | 1.074, 1.175 | 0.195, 0.268 | 0.359, 0.310 | −0.066, −0.036 |

(seeds 0, 1; the 64-molecule runs are jobs 9–10's)

- **Adam's first stage closes about a third of the gap in every cell:** from 1.0 at the start to 0.60–0.67 after its
  3000 steps (job 9: 0.669 at 64). The polish closes most of the rest.
- **Halfway through the polish the cells read alike:** gaps of 0.31–0.36 at 500 steps everywhere.
- **Then the 64-molecule cell stalls and the larger cells do not.** In the last 250 steps the gap closes by 0.020–0.029
  at 64 and by 0.033–0.066 at 128 and 256, about twice as fast. Every end is still moving when the budget ends.
- **Coverage is not the lever.** At 99 % and 82 % the gaps and their rates read alike. What separates the 64-molecule
  cell is its size against E₀'s 12 Å reach.
  - Of the pre-registration §6's readings this is nearest "mobile at 128: the 64-molecule cell's own geometry froze it",
    but slower: within 1000 steps no cell reaches one scale.
- **"Partial" is the low start lagging.**
  - From 1.5 the ends come down to 1.07–1.12 at 128 and 1.07–1.18 at 256. That is about the pinned profile's minimum
    at 64 molecules (c* = 1.070, job 11), measured only in the 64-molecule cells.
  - From 0.5 they reach only 0.87–0.91, and in the larger cells they are the ones still moving fastest (+0.016 to
    +0.053 in the last 250 steps).
  - The start still marks the fit: forces from 0.5 are 6–12 % worse than from 1.5 in every cell.
- **The warm starts' forces barely depend on the cell.** From 1.5: 0.554 / 0.562 / 0.566 at n 64 for L₀ = 64 / 128 /
  256. From 0.5: 0.619 / 0.600 / 0.635.
- *The verdicts read the ends at a fixed budget. "Partial" here is a scale still moving, not one held at two values.*

## 4. For job 16.5 (registered 28 Nov, not yet read)

**Job 16.5's K16.5.2 assumed about 0.001 of frame error on each step.** Job 16 measures 0.0016–0.0032 on each δ, so the
registered r carries about ±0.3: M_inf's reads 0.23 ± 0.32. "Carried" (≤ 0.30) and "1/L" (0.35–0.80) are not separated
by it. K16.5.1 should still read: the reversal is fixed far better than the frame errors say (§2).

**Added beside to `scripts/e4_job16_5.py` before any job 16.5 number** (the registered readings unchanged):

- r's frame standard error;
- r from the frames off both L₀ (M_inf: 0.12 ± 0.20);
- **the law where Form II reads it.** This is MACE-LES's own bias at 256, 512 and 1024 molecules, which job 16.5
  computes and which needs no dagger. From 256 to 1024 an error in 1/L moves the bias by 0.63 of its step from 64 to
  256, about −0.005 kcal/mol per molecule for MACE-LES (job 15's −0.0085). A carried error moves it about none. The
  step's frame error should be about 0.001–0.002 (one run's mean at 256 molecules, job 15), so the two laws sit two to
  four standard errors apart.

## 5. What job 16 says

- **Form II separates the classes by L₀.**
  - The cutoff model's bulk error is its training cell's long-range energy: trained at 128 its step reverses exactly
    (K16.1).
  - E4's model's bias and forces do not depend on the training cell (K16.2–K16.3).
  - The size law is consistent with a training-cell effect spent by 128 molecules; 1/L is disfavoured at one to two
    standard errors.
- **The learned charges' scale is not fixed by bulk energies and forces within these budgets, in any cell.**
  - It moves twice as fast in cells larger than E₀'s reach, and still does not close.
  - Coverage of the cell does not matter.
  - That is the identifiability argument on our own model: the data constrain the scale weakly. It argues for learned
    sources fixed by matching data (dipoles, the response), not for longer training on the liquid.

## 6. Next

1. **Job 16.5** (MACE-LES at L₀ = 128): its law beside the registered readings, at 512 and 1024 molecules (§4).
2. *Proposed, not registered:* the warm starts' polishes continued 2000 more L-BFGS steps at 64, 128 and 256, to see
   whether the ends meet, and where. A few GPU hours.
