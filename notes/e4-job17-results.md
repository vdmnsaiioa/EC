# E4, job 17: what a field-blind model with free latent charges cannot do — results

*Read 28 Nov 2026 against `e4-job17-preregistration.md`, registered on 27 Nov before any run. `run_e4_job17.sbatch` ran on
Niflheim (s001, RTX 3090) at 4f842d3, sseft 0.0.43, from 00:21 to 00:51 CET on 28 Nov: `make` had run already; ten harness
runs, 49–231 s each; twelve MACE-LES models, 40–121 s each; the read, 2 s. Half an hour in all (estimated 2 hours).*

- *Relative rmse is the rmse of a model's reading less MB-pol's, over the pairs (both sizes) or the frames, over
  MB-pol's rms. The clusters' gap is the smallest atom–atom distance between them; the slabs' reading is E(g) − E(24 Å).*
- *"The default MACE-LES" is the baseline, `maceles_n64_s0` and `_s1`. Beside it, job 15's other units (0.05, 0.15)
  and its releases, two seeds each.*

## 1. The registered readings

| | band | Niflheim | verdict | predicted |
|---|---|---|---|---|
| **K17.1** | MACE-LES's cluster interaction equals its latent charges' Coulomb sum within 10⁻⁶ kcal/mol, every model and gap | worst 1.0 × 10⁻⁸; the charges unchanged in AB to 1.7 × 10⁻¹¹ e | **holds** | holds |
| **K17.2** | MACE-LES's T5 within 10⁻⁶ of zero | worst 5.47 kcal/mol | *FAILS as first read: the reading's error (§2)* | holds |
| **K17.3** | the default MACE-LES leaves ≥ 0.5 of the probes' non-additive energy at 10 / 12 / 15 Å | 361 / 673 / 1219 (seed 0), 503 / 938 / 1699 (seed 1) | *holds as first read; the numbers are the same error (§2)* | holds |
| **K17.4** | E4's model, clusters, ≤ 0.05 from 9 Å, every seed | 0.006–0.007, all four seeds alike | **holds** | holds |
| **K17.5** | the default MACE-LES, clusters, ≥ 0.30 from 9 Å | 4.58 → 24.3 (seed 0), 1.86 → 9.29 (seed 1), growing with the gap | **holds** | holds |
| **K17.6** | its error over its error with each molecule neutral, ≥ 1.5 at 25 and 30 Å | 25.6 / 34.2 (seed 0), 9.6 / 12.8 (seed 1) | **holds** | holds |
| **K17.7** | E4's model, slabs, ≤ 0.10 at 9 / 12 / 16 Å | 0.005 / 0.005 / 0.005, every seed | **holds** | holds |
| **K17.8** | the default MACE-LES, slabs, ≥ 0.30 | 1.56 / 2.00 / 2.54 (seed 0), 0.320 / 0.455 / 0.611 (seed 1) | **holds** | holds |
| **K17.9** | our learned charges at most the default MACE-LES, seed means, from 9 Å | 0.245 → 0.234 against 3.22 → 16.8 | **holds** | holds |

Seven hold as predicted. K17.8 holds on seed 1 at 9 Å by 0.02. K17.2 and K17.3 are read again, completed (§2), before
they count.

## 2. K17.2 and K17.3: the first read took MACE-LES's isolated monomer at zero

**The probes' and trimers' reading (`scripts/e4_far.py readings`) assumes the truth's convention:** every energy is an
interaction energy, the isolated monomer at zero. The harness's rungs keep it; the dagger reads T5 = 0 and 100 % of the
probes exactly. The far file has no monomer frames, so a three-body energy is read as E(ABC) − E(AB) − E(AC) − E(BC). For
a model whose isolated monomer sits at e:

- the trimers' three-body energy reads −3e (exactly, if its true three-body energy is zero);
- a probe's non-additive energy (a core of three molecules) reads −5e plus its own;
- its interaction reads +e plus its own.

**MACE-LES was trained on the liquid alone, with E₀s of zero.** Nothing fixes its isolated monomer at zero.

**One number per model accounts for both readings.** If T5 reads −3e on every trimer, its slope against the truth is −3e
× Σref / Σref², and the probes' non-additive rmse is about 5|e|. Their ratio is then the same for every model:

| model | T5 slope | probes left at 10 Å | slope / left | \|e\| from the worst T5 (kcal/mol) |
|---|---|---|---|---|
| default, seed 0 | −2128.65 | 360.77 | −5.900 | 0.962 |
| default, seed 1 | −2965.05 | 502.58 | −5.900 | 1.340 |
| unit 0.05, seeds 0 / 1 | −1868.42 / −2158.60 | 316.24 / 365.73 | −5.908 / −5.902 | 0.844 / 0.976 |
| unit 0.15, seeds 0 / 1 | −2170.84 / −2708.02 | 367.77 / 459.12 | −5.903 / −5.898 | 0.981 / 1.224 |
| released at 0.05, seeds 0 / 1 | **+158.33** / −1215.37 | 26.39 / 206.19 | **+6.001** / −5.894 | **0.072** / 0.549 |
| released at 0.1, seeds 0 / 1 | −2119.05 / −2970.22 | 359.14 / 503.45 | −5.900 / −5.900 | 0.958 / 1.342 |
| released at 0.15, seeds 0 / 1 | −3581.86 / −4036.14 | 607.20 / 684.29 | −5.899 / −5.898 | 1.619 / **1.824** |

- **The ratio is −5.90 on eleven models, to 0.2 %.** On the twelfth (released at 0.05, seed 0) e has the other sign and is
  thirteen times smaller, and the ratio is +6.00.
- **\|e\| is set from the worst T5:** 5.4726 kcal/mol = 3 × 1.824 on the model with the largest slope, the others in
  proportion to their slopes.
- **Every model implies the same truth.** 5|e| over the share left gives MB-pol's non-additive rms at 10 Å as 0.01333
  kcal/mol on eleven models (0.01356 on the twelfth). At 12 and 15 Å the same construction gives 0.00714 and 0.00394 on
  both default seeds.
- **So K17.2's 5.47 kcal/mol is 3e, not a three-body energy.** K17.3's shares are 5e over the truth's non-additive
  energy, and say nothing yet about MACE-LES's.
- **Nothing else is touched.** The clusters' interaction (K17.1, K17.4–K17.6, K17.9) is E(AB) − E(A) − E(B), and the
  slabs' readings (K17.7–K17.8) compare the same molecules: e cancels in both.

**The fix (v0.0.45).**

- `scripts/e4_far.py monomers` writes every molecule of every far frame alone: 1400 open frames of three atoms.
- `readings(..., monomers_ev=)` subtracts each frame's monomers first.
- `scripts/e4_job17.py read` reads K17.2 and K17.3 only with each model's monomer energies there (`<run>_stagetwo_job17m.npz`,
  from `run_e4_job17m.sbatch`, about 15 minutes). It prints the first read's numbers beside them.
- `scripts/e4_job18.py read` does the same for its MACE models (`_job18m.npz`); its registered readings do not use the
  probes.

**Registered now, before the monomers are computed:**

| | reading | band | prediction |
|---|---|---|---|
| **K17.2** (completed) | every MACE-LES model's T5, its monomers subtracted | within 10⁻⁶ kcal/mol of zero | holds |
| **K17.3** (completed) | the default MACE-LES's share of the probes' non-additive energy left at 10 / 12 / 15 Å, its monomers subtracted | at least 0.5 | holds |

- **K17.2.** The trimers' every intermolecular atom pair lies beyond 6.5 Å, outside MACE's 6 Å graph. Proposition 1
  (K17.1, to 10⁻⁸) then makes the three-body energy exactly zero.
- **K17.3.** Each probe is an isolated component, carrying its monomer's latent charges and their net charge. The core's
  molecules carry other net charges together than alone. Their monopole cross term survives the pair sum and falls as
  1/R, against the truth's R⁻³.
  - The released model at 0.05, whose e is small, departs from the common ratio by 1.7 %. A genuine non-additive error
    of the order of the truth's would do that.
- **Beside, predicted from the first read:**
  - each model's isolated monomer \|e\| as in the table, within 1 %; one sign on eleven models, the other on the released
    model at 0.05, seed 0;
  - its spread over the 1400 molecules below 10⁻³ kcal/mol (E4's monomers are rigid);
  - the first read's T5 plus the trimer's monomers within 10⁻⁶ (the same statement as K17.2).

## 3. The clusters: MACE-LES's far field is a monopole tail

| gap (Å) | 7 | 9 | 12 | 16 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| MB-pol, rms (kcal/mol) | 0.443 | 0.288 | 0.169 | 0.095 | 0.059 | 0.035 | 0.023 |
| its induced share | 0.285 | 0.284 | 0.283 | 0.283 | 0.284 | 0.286 | 0.287 |
| E4's model (every seed) | 0.008 | 0.007 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| the dagger | 0.287 | 0.285 | 0.283 | 0.284 | 0.284 | 0.286 | 0.287 |
| our learned charges, seeds 0 / 1 | 0.272 / 0.251 | 0.257 / 0.233 | 0.250 / 0.224 | 0.247 / 0.220 | 0.247 / 0.219 | 0.247 / 0.219 | 0.248 / 0.220 |
| the cutoff model | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MACE-LES, default, seed 0 | 3.53 | 4.58 | 6.43 | 9.40 | 12.95 | 18.19 | 24.33 |
| MACE-LES, default, seed 1 | 1.50 | 1.86 | 2.53 | 3.64 | 4.98 | 6.97 | 9.29 |
| seed 0, each molecule neutral | 0.713 | 0.710 | 0.710 | 0.710 | 0.710 | 0.711 | 0.711 |
| seed 0, each cluster neutral | 0.406 | 0.393 | 0.385 | 0.381 | 0.380 | 0.378 | 0.377 |

**The ladder reads as its parts.**

- **The cutoff model** has nothing beyond r_c: 1 exactly.
- **The dagger** has MB-pol's permanent charges and dispersion, and no response. Its error equals the truth's induced
  share at every gap, to 0.002.
- **E4's model** adds the matched response. It is within 0.6–0.8 % at every gap and seed: about 2 % of the induced
  share, the response's error on the far probes (2.0–2.7 %).
- **Our learned charges** (per-molecule neutral, unpinned, the same data as MACE-LES) read 0.22–0.27. That is the
  dagger's error, a little less: their environment dependence carries a little of the induction.

**MACE-LES's error grows with the gap, 1.9 to 24 times the truth at 9–30 Å.**

- **Along each ray its interaction falls as 1/R.** From 16 to 30 Å it keeps 0.593 / 0.650 (8 / 16 molecules); 1/R gives
  0.607 / 0.634. MB-pol keeps 0.222 / 0.251 (R⁻³: 0.223 / 0.255).
- **The clusters carry net latent charges.**
  - Default seed 0: 0.16 e rms in a cluster of 8, 0.30 e in one of 16. Seed 1: 0.11 and 0.19 e.
  - In the liquid the molecules carry 0.021–0.024 e each (job 15). Uncorrelated, that is 0.06 and 0.09 e for 8 and 16.
    Here the charges are 1.7–3.4 times that, and they grow nearly as the number of molecules (×1.75–1.87 from 8 to 16,
    against √2).
  - The molecules at a surface carry charge of one sign. In vacuum the whole cluster is surface, an environment the
    liquid never showed it.
- **The error goes as the net charge squared** (the 16-molecule clusters' charges). Default seed 0 over seed 1 at 30 Å:
  24.3 / 9.29 = 2.62; (0.300 / 0.185)² = 2.62. Unit 0.05 seed 0 over default seed 0: (0.402 / 0.300)² = 1.79, errors
  44.0 / 24.3 = 1.81.
  - In absolute terms, seed 0's error at 30 Å is about 0.56 kcal/mol rms, where MB-pol's whole interaction is 0.023.
- **With each molecule's net charge removed** it reads 0.71–0.74 on every model but the two released at 0.05 (0.90).
  That is about the scale's error alone. At c ≈ 0.6 its dipolar tail is 0.36 of the permanent one, and 1 − 0.36 × 0.78
  (the permanent share) = 0.72. Released at 0.05, c ≈ 0.36: 1 − 0.13 × 0.78 = 0.90.
- **With only each cluster's net charge removed** the default reads 0.38–0.46, better than with each molecule's. The
  charge moved between molecules within a cluster adds to its dipole, the part the scale lost.
- **No variant of MACE-LES escapes it.** All twelve read 1.1–47 from 9 Å. The best, released at 0.15 seed 1, has the
  smallest net charges (0.08 / 0.14 e) and still reads 1.08 → 5.0.
- **K17.9:** our learned charges, with the same data and no pins, are 13–72 times closer than the default MACE-LES. On
  these structures the per-molecule sum rule is most of the difference.

## 4. The slabs

| | 9 Å | 12 Å | 16 Å |
|---|---|---|---|
| MB-pol, E(g) − E(24 Å) rms (kcal/mol per cell) | 6.716 | 4.373 | 2.375 |
| E4's model (every seed) | 0.005 | 0.005 | 0.005 |
| the dagger | 0.238 | 0.216 | 0.191 |
| our learned charges, seeds 0 / 1 | 0.209 / 0.159 | 0.174 / 0.126 | 0.141 / 0.100 |
| MACE-LES, default, seeds 0 / 1 | 1.557 / 0.320 | 2.001 / 0.455 | 2.541 / 0.611 |

- **E4's model is within 0.5 %** of a slab's energy as its vacuum gap closes, on every seed.
- **The dagger misses the induced part.** MB-pol's slab dipole along z is 3.85 e Å rms permanent and 0.65 e Å induced.
- **MACE-LES's error grows with the gap.**
  - Its 64-molecule slabs carry a net latent charge of 0.85 e rms (seed 0) and 0.41 e (seed 1). The liquid's cells
    carry about 0.17 e if each molecule's 0.02 e is uncorrelated.
  - Under tin-foil boundaries a charged slab's energy changes with the cell's length.
  - Its cell dipole (4.2 e Å rms) depends on the origin when the cell is charged, and is not compared.

## 5. Rows a and b for the other models

- **ε_∞:** E4's model 1.6893 against MB-pol's 1.6885 (the read-out's, held). Every model without a field term has 1 by
  construction.
- **The far probes' non-additive energy left at 10 / 12 / 15 Å:**
  - E4's model 0.022 / 0.020 / 0.027;
  - the dagger and the cutoff model 1.000;
  - our learned charges 0.93–0.97: their environment dependence carries 3–7 %.
- **T5's slope:** E4's model 1.0403; the dagger, the learned charges and the cutoff model 0.0000.
- **MACE-LES:** read after the completion (§2).

## 6. What job 17 says

- **Proposition 1 holds on trained MACE-LES.** On all twelve models, beyond r_c its interaction is its latent charges'
  Coulomb sum to 10⁻⁸ kcal/mol, the charges of each cluster alone. `les-vs-eft` §2's frozen-charge force field is now
  measured, not argued.
- **Where nothing screens it, MACE-LES's far field is its free net charges.** In vacuum its molecules take net charges
  of one sign at a surface. Its clusters' interaction then falls as 1/R and misses MB-pol by 1.9–24 times (default) at
  9–30 Å. That is `les-vs-eft` §5's imposter, measured on the trained models. Neutralised molecule by molecule, what is
  left (0.71–0.74) is about its scale.
- **E4's model is within 1 % everywhere,** clusters and slabs, every seed. The ladder's steps read as its parts: the
  cutoff 1, the pins without response 0.28 (the induced share), the matched response 0.006.
- **The per-molecule sum rule is most of the gap to MACE-LES** (K17.9): our learned charges, on the same data without
  pins, read 0.22–0.27 against its 1.9–24.
- **A model trained on the liquid alone does not fix its gas-phase reference.** MACE-LES's isolated monomer sits 0.07–1.8
  kcal/mol from zero. No registered reading but K17.2–K17.3 sees it.
- *Every structure here is out of the training distribution for every model: vacuum surfaces and gas-phase clusters.
  That is the design, the far field with nothing screening it. MACE-LES trained on clusters is job 18's question.*

## 7. Next

1. **`run_e4_job17m.sbatch`** (v0.0.45, about 15 minutes): the monomers, then K17.2 and K17.3 read as completed (§2). It
   also computes job 18's MACE models' monomers if they are there.
2. **The sum-rule ablation** (pre-registration §8, proposed, not registered): MACE-LES retrained with each molecule's net
   latent charge removed. K17.6 says it should take the clusters' error from 2–24 to about 0.7, and job 15 asks whether
   it removes the drift with the cell.
