# E4, job 17: what a field-blind model with free latent charges cannot do — pre-registration

*27 Nov 2026, registered at this commit's time, before any run of job 17 and before any of its numbers. Vadic's go of 27
Nov: shift the focus to experiments that show what our model does and MACE-LES cannot, rows a–c of that day's proposal,
as one pre-registered job without training. Read against:*

- *`les-vs-eft-what-is-provable.md`: §2, Proposition 1 (beyond r_c MACE-LES is a frozen-charge force field); §5 (a free
  net charge per fragment gives a 1/R imposter); §7, T4 and T5;*
- *`e4-job15-results.md` §9–§11: MACE-LES's scale (c = 0.59–0.61), its net latent charge per molecule (0.021–0.024 e
  rms), its far field drifting with the cell as its charges squared;*
- *`e4-job8-results.md` and `e4-job12-results.md`: E4's model's infrared (ε_∞, the far probes, T5), fixed by its held
  read-out.*

## 1. Why

**The paper's case against MACE-LES has four parts,** each a statement about the class, not a benchmark:

1. its electronic response to a field is none (ε_∞ = 1);
2. its many-body induction beyond r_c is none (T5 = 0);
3. as deployed it has no dispersion beyond r_c;
4. its far field is not identified: each molecule's net latent charge is free.

**In bulk water these barely show.** Screening hides most of the far field, and a fixed-charge model mimics it by scaling
its charges (`les-vs-eft` §8, caveat 2). Job 15 found MACE-LES's bulk energy drifting with the cell by only 0.0085
kcal/mol per molecule from 64 to 256 molecules.

**This job reads them where nothing screens them,** on every model at once, without training:

- **a, the response to a uniform field (ε_∞):** read from the runs, not computed here. E4's model has 1.6893 against
  MB-pol's 1.6885 (jobs 8 and 12); every model without a field term — the dagger, our learned charges, the cutoff model,
  MACE-LES as deployed — has 1 by construction.
- **b, the far probes and the far trimers (T5):** the probe file all of E4 uses. The harness's runs read them in training;
  MACE-LES has not been read on them.
- **c, far fields in vacuum, new:**
  - **c1, two clusters pulled apart:** the interaction of two clusters cut from the liquid, at smallest atom–atom
    distances of 7 to 30 Å;
  - **c2, a slab:** a liquid frame opened into a slab, and its energy's change with the vacuum gap.

## 2. What changes (v0.0.42), fixed before any run

- **`scripts/e4_job17.py`:**
  - `make` (CPU, MBX) builds the structures and MB-pol's labels;
  - `harness` (GPU) evaluates one trained harness run on them, its rung rebuilt from the run's json as
    `scripts/e4_water.py train` built it;
  - `read` (NumPy only) reads K17.1–K17.9.
- **`baselines/mace/energies.py`:** a MACE-LES model's energies and latent charges on any extxyz, open or periodic.
- **`tests/test_job17.py`.**
- **Nothing in the harness's models or training changes.**

## 3. The structures

Every structure is cut from the 30 test frames of 64 molecules (`data/e4/test_n64.npz`), with seed 1700.

**c1, the clusters** (`data/e4/job17_mbpol.npz`, `job17_clusters.extxyz`):

- **Clusters of 8 and 16 molecules,** 12 pairs of each. A cluster is a molecule and its nearest neighbours by the O–O
  minimum image, whole.
- **A and B come from two different frames.** B is rotated uniformly about its centre of mass, put on A's and moved along
  a uniform direction until the smallest atom–atom distance is 7, 9, 12, 16, 20, 25 and 30 Å.
- **The reading is the interaction** E(AB) − E(A) − E(B), open, in kcal/mol.
- **MB-pol labels it with its components by term.** `make` stops if MB-pol's 2-, 3- or 4-body polynomials give anything
  between the clusters at any gap. The truth's far field there is then electrostatics (permanent and induced) and
  dispersion alone.

**c2, the slabs** (`data/e4/job17_slabs_n64.npz`, in E4's data format; `job17_slabs.extxyz`):

- **The first 8 test frames,** each with its molecules made whole and their O atoms wrapped into the cell.
- **The cell is lengthened along z** to L + g, for vacuum gaps g = 9, 12, 16 and 24 Å. The slab's clearance from its image
  across the gap is checked at 6.5 Å or more.
- **The reading is E(g) − E(24 Å)** per frame, in kcal/mol per cell. E₀ and MB-pol's polynomials see the same slab at every
  gap, so the change is the long range alone.
- **MB-pol** is periodic, with PME and tin-foil boundaries, as E4's data. The cell's dipole along z, permanent and induced,
  is recorded beside.

**b, the far probes and trimers:**

- `data/e4/probes_mbpol.npz` (hash 72b73e3aa6c7b53f): 80 test probes at 8–15 Å and 40 far trimers with every pair beyond
  6.5 Å.
- MACE-LES reads them through `data/e4/probes_mbpol_far.extxyz` (`scripts/e4_far.py`).

## 4. The models, none trained here

| role | run | seeds | what it is |
|---|---|---|---|
| E4's model | `results/e4_job12`, M_16C2pm_us_d20 | 0–3 | MB-pol's charges and c6 pinned, the response matched, polished |
| the dagger | `results/e4_job11`, M_16pm_us_d20 | 0, 1 | the same without the response |
| our learned charges | `results/e4_job10`, M_1w100 | 0, 1 | learned, each molecule neutral, no pins: the same data as MACE-LES |
| the cutoff model | `results/e4_job7`, M_inf | 0, 1 | E₀ alone |
| **MACE-LES** | `results/e4_baselines/maceles_n64_s0`, `_s1` | 0, 1 | the baseline, registered |
| MACE-LES, beside | `results/e4_job15_les/*` | 0, 1 | job 15's other units (0.05, 0.15) and its releases |

## 5. Registered

Relative rmse is the rmse of the model's reading less the truth's over the pairs (both sizes) or the frames, over the
truth's rms. "The default MACE-LES" is the baseline, both seeds.

| | reading | band | prediction |
|---|---|---|---|
| **K17.1** | every MACE-LES model, the clusters at every gap: its interaction from its energies against its latent charges' Coulomb interaction, k Σ q_i q_j erf(r/√2σ)/r over i in A, j in B | within 10⁻⁶ kcal/mol | holds |
| **K17.2** | every MACE-LES model, T5: its three-body energy of the far trimers | within 10⁻⁶ kcal/mol of zero | holds |
| **K17.3** | the default MACE-LES, the far probes: the share of the non-additive energy it leaves at 10, 12 and 15 Å | at least 0.5 | holds |
| **K17.4** | E4's model, the clusters: relative rmse at every gap from 9 Å, every seed | at most 0.05 | holds |
| **K17.5** | the default MACE-LES, the clusters: relative rmse at every gap from 9 Å | at least 0.30 | holds |
| **K17.6** | the default MACE-LES, the clusters at 25 and 30 Å: its error over its error with each molecule's net latent charge removed (spread evenly over its atoms) | at least 1.5 | holds |
| **K17.7** | E4's model, the slabs: relative rmse of E(g) − E(24 Å) at 9, 12 and 16 Å | at most 0.10 | holds |
| **K17.8** | the default MACE-LES, the slabs: the same | at least 0.30 | holds |
| **K17.9** | the clusters, seed means: our learned charges' relative rmse against the default MACE-LES's at every gap from 9 Å | at most MACE-LES's | holds |

**Why these bands, and these predictions.**

- **K17.1 and K17.2 are Proposition 1 and Corollary 1.2 of `les-vs-eft`,** read on the trained models.
  - Beyond MACE's 6 Å graph the two clusters are separate components, so their latent charges are those of each alone.
  - Their interaction is then LES's pairwise Coulomb sum, and a far trimer's three-body energy is zero.
  - The sum was checked on job 15's smoke model (8-molecule cells, random labels): it equals the energies' interaction to
    10⁻¹⁴ kcal/mol at 7–30 Å, and differs at 5 Å, inside r_c.
  - 10⁻⁶ is float64's rounding with room.
  - If K17.1 fails, the latent charges written are not the ones LES uses, and K17.6 cannot be read.
- **K17.3.**
  - The probes' non-additive energy is the core's induction, internal and mutual.
  - MACE-LES's environment-dependent charges can carry part of the core's internally induced dipole (job 1's correction
    to the plan). But they sit at 0.6 of the permanent scale and were never asked for it.
  - The dagger and the cutoff model leave 100 %; E4's model leaves 2.0–2.7 %.
  - Half is a weak band, set where carrying most of it would change the paper's claim.
- **K17.4.**
  - E4's model has MB-pol's own permanent charges and c6, and a response matched on the far probes and the liquid.
  - Its error should be about the induced share times the response's error on the probes (2–3 %), well under 5 %.
  - Its far field does not depend on the seed: E₀ cancels beyond r_c, and the read-out is held.
- **K17.5.**
  - MACE-LES's dipole is 0.6 of MB-pol's permanent one, so its permanent tail is about 0.36 of the truth's.
  - It has no induction and, beyond 6 Å, no dispersion.
  - Its free net charges add a 1/R term.
  - 30 % is about half of what the scale alone predicts.
- **K17.6.**
  - In the liquid MACE-LES's molecules carry 0.021–0.024 e of net latent charge each (job 15).
  - Uncorrelated, a cluster of 8–16 would carry 0.06–0.09 e. Their monopole–monopole term at 30 Å (332 Q_A Q_B / R) would
    then be several times the dipolar tail.
  - The prediction is uncertain in one way. The molecules' net charges may be correlated and cancel within a cluster, and a
    cluster in vacuum is not the liquid. A ratio near 1 would say the free charges do not matter here.
- **K17.7 and K17.8.**
  - The slab's energy changes with the gap through its dipole's interaction with its images and their dispersion. The
    dipole includes the induced dipoles, which answer the images' field.
  - E4's model has both, MB-pol's own. MACE-LES has 0.36 of the permanent part, no induced part and no dispersion beyond
    6 Å.
  - A slab with a net latent charge adds, under tin-foil boundaries, a term that grows with the gap.
  - 10 % and 30 % are the clusters' bands, loosened for the slab's surfaces.
- **K17.9: the comparison on the same data, without pins.**
  - Our learned charges are each molecule neutral and carry 1.03–1.08 of the permanent dipole (job 10). They lack
    induction and dispersion.
  - MACE-LES lacks the same, and has 0.6 of the dipole and free net charges.
  - If K17.9 fails, the clusters' gap is the pins' alone, and the per-molecule sum rule buys nothing here.

## 6. Read beside, not registered

- **MB-pol's interaction by gap:** its rms and its shares (permanent, induced, dispersion).
- **The interaction along each ray, 30 Å over 16 Å (median),** against the same pairs' (R₁₆/R₃₀)³ for a dipolar tail
  and R₁₆/R₃₀ for a 1/R one (R the centres' distance).
- **Every model's clusters and slabs by gap.**
  - The dagger's error against the truth's induced share; the cutoff model's 1 exactly beyond r_c.
  - MACE-LES with each molecule's, and each cluster's, net latent charge removed.
  - The clusters' and slabs' net latent charges.
  - Job 15's other units and releases.
- **The far probes:** MACE-LES's interaction error, beside the non-additive one; every model's T5 slope.
- **ε_∞** from the harness's runs, and MB-pol's on the same frames.

## 7. What would change the reading

- **K17.1 or K17.2 fails:** MACE-LES as trained is not the class Proposition 1 describes. Read the charges' change
  between A alone and in AB.
- **K17.3 fails:** MACE-LES's environment-dependent charges carry most of the core's induction. Then §5's claim of the
  provability note narrows to T5 and ε_∞.
- **K17.4 or K17.7 fails:** E4's model's matched far field does not carry to clusters in vacuum or to surfaces. Read the
  induced share against its error.
- **K17.5 holds and K17.6 fails:** MACE-LES's far error is its scale and its missing physics, not its free charges. The
  sum-rule ablation is then not worth running.
- **K17.6 holds:** the free net charges dominate MACE-LES's far tail. The per-molecule sum rule (the ablation proposed on
  27 Nov) should remove most of it.
- **K17.9 fails:** on these structures our learned charges are no better than MACE-LES. The gap is the pins', not the sum
  rule's.

## 8. Not tested here

- **The sum-rule ablation** (MACE-LES retrained with each molecule neutral).
  K17.6's evaluation-time counterfactual is not a retrained model.
- **Ions (T2), a field applied in MD,** and spectra.
- **MACE-LES with LES's induced-dipole extension,** which is a different class.

## 9. The run and its cost

`run_e4_job17.sbatch`, one GPU task:

1. `make` on the node's CPUs, if its files are not there already.
2. The ten harness runs.
3. The MACE-LES models, in the mace-les env: job 15's twelve, wherever they are found.
4. `read`.

**Estimated cost:**

- `make`: about 15 minutes. The clusters' 430 MBX calls take seconds each or less. The 32 slab evaluations are periodic, on
  a 2 × 2 replica.
- The harness runs: 2–5 minutes each on the GPU.
- MACE-LES: 1–2 minutes a model.
- About 2 hours in all; the task asks 12.
- A step already done (its output file) is skipped, so a task that stops is finished by submitting it again.

## 10. Numbers before any run

- **E4's model:**
  - ε_∞ 1.6893 against MB-pol's 1.6885;
  - the far probes' non-additive energy 2.23 / 1.98 / 2.66 % left at 10 / 12 / 15 Å;
  - T5 4.31 %, slope 1.0403.
  - The read-out is held, so these are the same on every seed (jobs 8 and 12).
- **The dagger and the cutoff model:** the probes 100 %, T5 0, ε_∞ 1, by construction.
- **MACE-LES (job 15):**
  - c = 0.592–0.605;
  - each molecule's net latent charge 0.021–0.024 e rms on the test frames;
  - its energy per molecule drifting −0.0085 ± 0.0006 kcal/mol from 64 to 256 molecules.
- **Proposition 1 on the smoke model:**
  - three molecules against three, at 7, 12 and 30 Å, it holds to 6 × 10⁻¹⁵ kcal/mol, with the charges unchanged to 10⁻¹⁷
    e;
  - at 5 Å, inside r_c, it does not (4 × 10⁻⁴).
