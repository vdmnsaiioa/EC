# E4, job 16: the training cells of 128 molecules — pre-registration

*26 Nov 2026, registered at this commit's time, before any run of job 16 and before any of its numbers. Vadic's go of 26
Nov ("start with 1": the L₀ sweep at 128 molecules with the learned charges' cell test; the cell test at 256 molecules
too, his choice after the coverage question). Read against:*

- *`e4-plan.md` §3–§4: the L₀ sweep (Form II), the same atoms at every size;*
- *`e4-job1-results.md` §4b: M_∞ carries the 64-molecule cell's long-range energy;*
- *`e4-job15-results.md` §4 and §7: learned charges freeze when E₀ reaches across the cell.*

## 1. Why

**Two questions, one set of runs.**

- **E4's L₀ sweep (Form II).** The plan trains every rung on cells of one size, the same atoms at every size (180 frames
  of 64 molecules, 90 of 128, 45 of 256), and reads the bulk errors by size.
  - Job 1 found the cutoff model's energy moving with the cell: trained at 64, M_∞ less the dagger reads +0.0006,
    −0.0089 ± 0.0008 and −0.0102 ± 0.0016 kcal/mol per molecule at 64, 128 and 256 molecules.
  - Its reading: E₀, whose two layers reach 12 Å in the 12.4 Å cell, carries the training cell's long-range energy into
    larger cells. The rungs with explicit electrostatics follow the cell.
  - If so, M_∞ trained at 128 carries the 128-molecule cell's long-range energy, and under-binds the 64-molecule cells
    by about the amount the 64-trained model over-binds 128.
- **The learned charges' scale and the cell.** Job 15 found every E₀ reaching 6 Å or less mobile, and job 7's E₀
  (12 Å) frozen in 12.4 Å cells. The coverage of the cell matters, not only the cell's length:

| training cell | L | share of the cell within 12 Å of a point | within 6 Å |
|---|---|---|---|
| 64 molecules | 12.4 Å | 100 % | 47 % |
| 128 molecules | 15.6 Å | 99 % | 24 % |
| 256 molecules | 19.7 Å | 82 % | 12 % |

- **At 128 molecules job 7's E₀ still sees nearly the whole cell.** The cell test there separates "coverage" from "the
  cell's length". At 256 it is the test proper.

## 2. What changes (v0.0.41), fixed before any run

- **`training.readout.held_from.train_size`.**
  - E4's model holds job 5's read-out, matched with the 64-molecule cells: its energy scale and its cell's n_max are in
    the cache's key.
  - With `train_size: 64` the cache is checked against that training set's cell and energy scale, and held for
    training on cells of another size.
  - It is refused unless the read-out is by species. D′'s is: its unit is fixed (1 mhartree), not the site energies'
    data-derived scale, so it is the same physical read-out at any training size.
- **Seven configs:**
  - `e4_job16s_128.yaml`, `e4_job16_128.yaml`: M_inf, the field-term dagger and job 7's warm starts at L₀ = 128, job
    7's training traced as job 9 traced it, then jobs 10–11's polish.
  - `e4_job16s_128d.yaml`, `e4_job16_128d.yaml`: E4's model at L₀ = 128, job 8's training, then job 12's polish (the
    response precomputed).
  - `e4_job16_64.yaml`: job 7's M_inf polished as job 10 polished job 7's runs, the L₀ = 64 reference.
  - `e4_job16s_256.yaml`, `e4_job16_256.yaml`: the warm starts at L₀ = 256.
- **The same atoms.**
  - At 128: three frames per Adam batch and per L-BFGS chunk, the atoms of job 7's six and the pairs and k-space terms
    of its chunk of twelve.
  - At 256: one frame per batch and chunk (256 molecules, two thirds of job 7's batch); one frame holds about 1.3 times
    the pairs of job 7's chunk.
  - Everything else is the 64-molecule runs' settings: 3000 Adam steps on the same schedule, 50 L-BFGS steps, then
    1000. The polishes have budgets of 1200 minutes (128) and 1500 (256).
- **The reader:** `scripts/e4_job16.py` reads K16.1–K16.5, without JAX.
  - Bias is the mean energy error per molecule on the frames the training did not see. At L₀ that is the validation and
    test splits; at any other size it is every frame.

## 3. The runs

`run_e4_job16.sbatch`, thirteen tasks, one GPU each, seeds 0 and 1:

- 0, 1: E4's model at 128, the first stage then the polish;
- 2, 3: the dagger at 128;
- 4: M_inf at 128, then M_inf at 64 polished;
- 5–8: the warm starts at 128, one start and seed each;
- 9–12: the warm starts at 256.

The L₀ = 64 references are runs already made:

- job 11's polished dagger (seeds 0, 1);
- job 12's polished E4's model (seeds 0, 1 of its four);
- job 10's polished warm starts and job 9's traced first stages.

## 4. Registered

**The readings.**

- δ_L₀(L) is M_inf's bias less the dagger's, on the same unseen frames of size L, seed by seed, then the seed mean.
- Its step from L₀ to L is s_L₀(L) = δ_L₀(L) − δ_L₀(L₀).
- The gap is a seed's polished end from 1.5 less its end from 0.5, in c (the learned dipole on the test frames of 64
  molecules over 1.864 D).

| | reading | band | prediction |
|---|---|---|---|
| **K16.1** | s_128(64) against s_64(128) | opposite in sign, \|s_128(64) + s_64(128)\| ≤ 0.4 \|s_64(128)\|; read when \|s_64(128)\| ≥ 0.003 | holds |
| **K16.2** | E4's model's bias at L₀ = 128 against L₀ = 64, at 64, 128 and 256 molecules | within ±0.003 kcal/mol per molecule | holds |
| **K16.3** | E4's model's forces at L₀ = 128 against L₀ = 64, at every size | within ±5 % | holds |
| **K16.4** | the warm starts' gap at L₀ = 128, both seeds | ≥ 0.25: frozen; ≤ 0.15: mobile; else partial | frozen |
| **K16.5** | the same at L₀ = 256 | the same | partial |

**Why these bands, and these predictions.**

- **K16.1.** Job 1's step was −0.0095 (64 to 128). A model carrying its training cell's long-range energy makes the
  step the other way from 128 down to 64, by the same amount.
  - 40 % (about 0.004) is a few times the paired bias's frame-to-frame standard error (about 0.001), and allows for the
    polish.
  - Not read under 0.003, where there is no step to reverse.
  - Predicted to hold: at 128 molecules E₀ still sees 99 % of its training cell.
- **K16.2.** E4's model carries its electrostatics explicitly, so its bias follows the cell, not L₀. ±0.003 is about
  three times the seed spread of a rung's bias (job 1, jobs 11–12).
- **K16.3.** The same atoms, half the frames of twice the size. ±5 % is about five times the seed spread of E4's
  model's forces (0.004).
- **K16.4 and K16.5.** The edges are job 15's: the gaps of the mobile models (0.00–0.10, read with margin as 0.15) and
  of job 7's E₀ at 64 (0.29–0.32, read with margin as 0.25).
  - At 128 E₀ sees 99 % of the cell: predicted frozen.
  - At 256 it sees 82 %: predicted partial. Nothing yet says how much coverage freezes the scale.

## 5. Read beside, not registered

- **The steps to 256**, s_64(256) and s_128(256): the share of the drift between 128 and 256.
- **Every rung's bias and forces by size, at both L₀:** M_inf's and the dagger's Form II, beside E4's model's.
- **E4's model's ε_∞** on the test frames of the training size, against job 12's 1.6893 at 64. It is the read-out's,
  held, so it should not move beyond the frames' difference.
- **The warm starts by cell:**
  - the first stage's gap (Adam's closure: job 9's 0.669 and 0.670 at 64);
  - the paths;
  - the ends' place;
  - the forces at every size.
- **The polishes' L-BFGS steps** (a budget may stop one early).

## 6. What would change the reading

- **K16.1.**
  - It holds: Form II for a cutoff model is its training cell's long-range energy, moved by L₀. Its saturated error is
    not flat in L₀ but tied to it, as job 1 corrected the plan's table.
  - It fails with the same sign: the drift is not the training cell's energy carried along. It lies in the larger
    cells themselves; read the dagger's own bias by size.
  - It fails near zero: M_inf trained at 128 follows the cells. Its 64-molecule drift came from E₀ spanning its cell,
    which it no longer does at 128.
- **K16.2 fails:** E4's model's energy depends on its training cell. Read E₀'s share against the dagger's.
- **K16.3 fails:** the training cell's size changes E4's accuracy. Read the size by size forces.
- **K16.4 and K16.5.**
  - Frozen at 128 and partial or mobile at 256: coverage of the cell is the lever. The scale freezes when E₀ sees
    (nearly) the whole training cell.
  - Mobile at 128: the 64-molecule cell's own geometry froze it (a reach of 12 Å against a cell of 12.4 Å), not
    coverage.
  - Frozen at both: within these cells the lever is E₀'s reach itself, 12 Å, not its reach against the cell. Job 15's
    one-layer models were mobile for being short, not for being short against the cell.

## 7. Not tested here

- **E4's model and the dagger at L₀ = 256.**
- **MACE and MACE-LES at 128 molecules.** `baselines/mace/train.sbatch` takes N0=128; their Form II is a separate run.
- **The 512- and 1024-molecule tests,** which the harness does not read (it has no neighbour lists).

## 8. Cost

- **On the CPU the same atoms cost 1.8–2.6 times more in cells of 128 molecules:** per objective and gradient, M_inf
  2.5 → 6.3 s, the dagger 8.7 → 16.9 s, a warm start 9.8 → 17.9 s. Peak memory grew 1.6–1.7 times.
- **On the GPU, scaling jobs 8–12's 64-molecule runs:**
  - E4's model 13–18 h a task (its first stage 3–3.6 h at 64, its polish 3.6 h);
  - the dagger 8–10 h;
  - M_inf 8 h;
  - a warm start at 128 about 5 h, at 256 about 10–14 h (extrapolated).
- **About 125 GPU hours.** The sbatch asks 36 h a task. Job 15's tasks varied by up to ten times between nodes.

## 9. Numbers before any run

- **Job 1 (64 molecules, unpolished):** M_∞ less the dagger +0.0006 / −0.0089 ± 0.0008 / −0.0102 ± 0.0016 kcal/mol per
  molecule at 64 / 128 / 256.
- **The 64-molecule references:**
  - the polished dagger (job 11, seeds 0, 1): forces 0.440 / 0.453 / 0.456;
  - E4's model (job 12, four seeds): 0.444 / 0.449 / 0.451, ε_∞ 1.6893;
  - the warm starts (job 10): gaps 0.324 and 0.288, after job 9's first stage 0.669 and 0.670.
- **The coverage of the cell within 12 Å:** 100 / 99 / 82 % at 64 / 128 / 256 molecules; within 6 Å, 47 / 24 / 12 %.
