# E4, job 18: from a few molecules to many, MB-pol standing in for CCSD(T) — pre-registration

*27 Nov 2026, registered at this commit's time, before any run of job 18 and before any of its numbers. Vadic's go of 27
Nov: compare MACE-LES and our model on extrapolating from CCSD(T) on a few molecules to CCSD(T) on many, first as a
pilot with MB-pol as the truth, and find out what DLPNO-CCSD(T) on Niflheim would take (§9). Read against:*

- *`les-vs-eft-what-is-provable.md`: layer 1 (no induction, no many-body non-additivity beyond r_c for MACE-LES) and
  layer 3 (the gap shows on size-extrapolating observables);*
- *`e3-summary.md` §2: trained on clusters, the frozen form's far field holds; the many-body energy needs mutual
  induction;*
- *`e4-job1-results.md` and `e4-job5-results.md`: matched on the far probes alone, the read-out gives the liquid
  ε_∞ = 1.80 against MB-pol's 1.69 (χ 16 % too high), and calibrated on the liquid's response it gives 1.6893.*

## 1. Why

**CCSD(T) is affordable on a few molecules, not on many.** A model for condensed water at that level has to learn from
small clusters and carry what it learns to systems it never saw at that level.

- **Our model takes its long range from the single molecule:** its charges, polarisability and C₆. Its response then
  gives the induction to all orders.
- **MACE-LES must learn everything from the small clusters:** its latent charges, and any many-body energy.

**Two things are tested at once:**

- **Many-body extrapolation.** In clusters of 1–6 molecules every molecule has at most five neighbours. In the liquid it
  has four hydrogen bonds and about 240 molecules within 12 Å. MB-pol's energy beyond its 2- and 3-body polynomials is
  induction alone, so what no small cluster shows is cooperative, many-body induction.
- **The long range.** The clusters of 128 and 256 molecules are 19 and 24 Å across, beyond MACE's 12 Å receptive field.

**MB-pol is the pilot's truth because it is built the way a CCSD(T) model would be:** 2- and 3-body polynomials fitted
to CCSD(T), plus a polarisable many-body electrostatics. MBX labels any geometry, so the whole protocol costs about a
day. If the gap is large here, the CCSD(T) version is worth paying for (§9).

## 2. What changes (v0.0.43), fixed before any run

- **`scripts/e4_job18.py`:**
  - `make` (CPU, MBX) builds the clusters and MB-pol's labels and response;
  - `train` (GPU) trains a harness rung on the clusters, open;
  - `read` (NumPy only) reads K18.1–K18.7.
- **`configs/e4_job18.yaml`, `configs/e4_job18p.yaml`.**
- **`baselines/mace/train.sh`:** TRAIN, VALID and BATCH may replace the E4 sets and the batch of 2.
- **`baselines/mace/energies.py`:** `--tag`.
- **`tests/test_job18.py`.**
- **Nothing in the harness's models or the 64-molecule training changes.**

## 3. The clusters (`data/e4/job18_clusters.npz`, seed 1800)

**Training, from the 45 training frames of 256 molecules:**

- 20 isolated monomers, and 400 clusters of each size from 2 to 6 molecules.
- Half of each size compact (a molecule and its nearest neighbours by the O–O minimum image). Half sparse (a molecule
  and others drawn at random within 9 Å of it), for the dimer and trimer tails.
- MB-pol's interaction energies and forces. Its 1-body polynomial is off, so a monomer is zero.
- Every tenth cluster is held out as MACE's validation set (202). The other 1818 train every model.
- CCSD(T) gradients of clusters this small are affordable, so the forces are not an advantage the real version would
  lack.

**Calibration:** 60 compact training clusters of 3–6 molecules, with MB-pol's induced dipole at zero field and its
response to a uniform field (clamped nuclei, MBX, central differences of 10⁻⁴ e/Å²). This is the small-cluster
response that a CCSD calculation affords.

**Test:** 16 compact clusters of each of 6, 16, 32, 64, 128 and 256 molecules.

- They are cut from the test frames of 256 molecules (up to 64), 512 (128) and 1024 (256), each within half its cell.
- MB-pol's interaction energy and its components.
- Up to 64 molecules, the sum of its pair energies, so that the part beyond pairs is read.

## 4. The models

Every model is trained on the same 1818 clusters, open, with two seeds (0, 1). The harness's runs use E4's settings:

- losses: energy weight 100, forces 1, each molecule's net force and torque;
- Adam: 3000 steps in batches of 32, then 500 L-BFGS steps over all clusters.

| role | rung or model | what it has |
|---|---|---|
| **our model** | M_16C2pm_us_d20, `configs/e4_job18.yaml` | MB-pol's monomer charges and c6 pinned; the response matched on the far probes (cores of 3–5 molecules and one more), then calibrated on the 60 clusters' response; held |
| our model, probes alone | the same, `configs/e4_job18p.yaml` | the read-out matched on the far probes alone, as job 1's was |
| the dagger | M_16pm_us_d20 | the pins, no response |
| our learned charges | M_1w100 | learned charges, each molecule neutral; no pins, no response |
| the cutoff model | M_inf | E₀ alone |
| **MACE-LES** | baselines/mace, the baseline's settings | 300 epochs (stage two from 200), batches of 16 |
| MACE | the same without LES | |

## 5. Registered

The error per molecule is (model − MB-pol) / n in kcal/mol, its rms over the 16 clusters of each size. "Ours" is our
model (`configs/e4_job18.yaml`); seed means unless stated.

| | reading | band | prediction |
|---|---|---|---|
| **K18.1** | in distribution, 6 molecules: MACE-LES's error against ours | at most ours | holds |
| **K18.2** | at 128 and 256 molecules: ours against MACE-LES's | below MACE-LES's at both | holds |
| **K18.3** | MACE-LES: its error at 256 molecules over its error at 6 | at least 2 | holds |
| **K18.4** | ours, every size, every seed | at most 0.15 kcal/mol per molecule | holds |
| **K18.5** | the dagger's error at 256 molecules over ours | at least 2 | holds |
| **K18.6** | at 256 molecules, the mean error: the read-out on the probes alone against calibrated on the clusters | larger in magnitude | holds |
| **K18.7** | at 128 and 256 molecules: our learned charges against MACE-LES | at most MACE-LES's at both | **fails** |

**Why these bands, and these predictions.**

- **K18.1.** In distribution MACE's local accuracy should win, as it does in bulk at the training size (job 15: half our
  forces). The paper concedes it. If it fails, our model is the more accurate one even where MACE-LES should be at home.
- **K18.2, the headline.**
  - At 128 and 256 molecules MACE-LES meets environments no training cluster showed, and cooperative induction it can
    only have learned up to six molecules. Beyond 12 Å it has fixed charges and nothing else.
  - Our model's induction comes from the response at any size.
  - Its E₀ meets the same new environments, but carries less: MB-pol's short-range polynomials, not its electrostatics
    or induction.
- **K18.3.** If MACE-LES has not learned the cooperative induction, its error grows with the number of molecules that
  see dense environments. Twice is modest: surfaces keep part of every cluster small-cluster-like.
- **K18.4.**
  - 0.15 kcal/mol per molecule is about 2 % of the interaction per molecule in the large clusters (MB-pol's liquid,
    −9.8).
  - The prediction is the least certain of the seven, for two reasons.
    - Job 1's read-out, matched on the probes alone, over-polarised the liquid by 16 % in χ. The clusters' calibration
      may not carry the liquid's local field.
    - E₀, trained on molecules with at most five neighbours, may misread dense environments.
  - If K18.4 fails and K18.2 holds, our model extrapolates better than MACE-LES but not well. Read K18.6 and the dagger to
    tell which part fails.
- **K18.5.** The dagger has the pins and no response. Its E₀ can learn the induction of up to six molecules, not the
  cooperative induction of 256. Twice is set where the response clearly matters.
- **K18.6.** Job 1's probes-only read-out gave ε_∞ = 1.80 against 1.69 in the liquid, so it should over-bind the large
  clusters. Calibration on clusters of 3–6 should pull that toward MB-pol.
- **K18.7, predicted to fail.**
  - Our learned charges are neutral per molecule but have no response.
  - Like the dagger, they must learn induction from small clusters.
  - Predicted no better than MACE-LES: what the paper would claim is the response's, not the learned charges'.
  - If K18.7 holds, the per-molecule sum rule alone carries part of the extrapolation.

## 6. Read beside, not registered

- **Every model's error per molecule by size,** rms and mean, and the interaction's relative rmse.
- **MB-pol:**
  - the interaction per molecule by size;
  - its induced share;
  - up to 64 molecules, its share beyond pairs.
- **MACE without LES.**
- **The calibration clusters' response on our model** against MB-pol's.
- **Every model's far probes and T5:** MACE and MACE-LES through `scripts/e4_far.py`. This is MACE-LES's T5 trained on
  clusters, beside job 17's.
- **The training clusters' energy error per molecule** of each harness run.

## 7. What would change the reading

- **K18.2 fails:** from six molecules MACE-LES extrapolates at least as well as our model. The paper's
  size-extrapolation case for the response does not stand on MB-pol, and the CCSD(T) version is not worth running.
- **K18.2 holds, K18.4 fails:** read where the error is.
  - The dagger and the probes-only run separate the response's error from E₀'s.
  - A response error points to calibrating on larger clusters' response, which CCSD affords up to about 20 molecules.
  - An E₀ error points to training on larger clusters.
- **K18.3 fails:** MACE-LES's error does not grow with size. Its error is then a constant offset learned from small
  clusters, not a failure to extrapolate.
- **K18.7 holds:** the per-molecule sum rule alone helps. The sum-rule ablation of MACE-LES becomes worth running.

## 8. Not tested here

- **CCSD(T) itself** (§9).
- **Clusters larger than 256 molecules, the liquid in a cell, dynamics.**
- **Training sets of other sizes:** up to 3 molecules, or up to 10.

## 9. Toward CCSD(T): what it would take

**The truth for the large clusters.**

- Canonical CCSD(T) near the basis-set limit is rare beyond about ten molecules. For (H₂O)₂₀ the best references combine
  MP2-F12 on the whole cluster with CCSD(T) many-body corrections.
- Sylvetsky and Martin computed the four (H₂O)₂₀ isomers with PNO-LCCSD(T)-F12b in Molpro.
  - With tight PNO domains they reproduce the reference association energies to about 0.25 %: within about 0.5
    kcal/mol of about 215 at their best settings, up to about 1 kcal/mol at others.
  - Default domains are two to six times worse.
  - One isomer took 6.6 CPU-days with tight domains, 23 CPU-hours with default ones. The canonical n-body reference took
    over a CPU-year.
- At 256 molecules, 0.25 % is about 0.02 kcal/mol per molecule, small against K18.4's 0.15. Default domains would not
  be.

**The codes.**

- **ORCA** (DLPNO-CCSD(T)) is free for academic use, but each user registers and downloads a personal copy (FACCTs; CSC
  runs it that way, ORCA 6.1.1).
  - Niflheim's EasyBuild documentation names no quantum-chemistry code; `module avail ORCA` on Niflheim would say.
  - Otherwise Vadic installs his own copy.
- **MRCC's LNO-CCSD(T)** (also free for academics) is the alternative.

**The cost, estimated, not measured:**

- Local CCSD(T) scales nearly linearly at these sizes, with one tight run of (H₂O)₂₀ costing about a week of CPU.
- (H₂O)₆₄ would then cost one to four weeks of CPU per cluster with tight settings and a triple-ζ basis. 16 clusters of
  each large size is hundreds of CPU-days.
- The training clusters (1–6 molecules, about 2000) at canonical CCSD(T)/aug-cc-pVTZ are cheap by comparison, forces
  included.

**The plan, if the pilot warrants it:**

- the same protocol;
- the training clusters at canonical CCSD(T);
- 8 test clusters of each of 16, 32 and 64 molecules at tight DLPNO- or LNO-CCSD(T);
- the published (H₂O)₂₀ references (BEGDB, WATER27) as a check.

## 10. The run and its cost

`run_e4_job18.sbatch`, three GPU tasks:

- 0: `make` (about 30 minutes on the node's CPUs), then our model (both read-outs) and the dagger;
- 1: after `make`, our learned charges and E₀ alone, then MACE-LES trained and read;
- 2: after `make`, MACE without LES trained and read.

**Estimated 6–10 hours a task;** each asks a day. A step whose output exists is skipped, so a task that stops is finished
by submitting it again.

## 11. Numbers before any run

- **MB-pol's liquid:** −9.78 to −9.81 kcal/mol per molecule at every size (E4's data).
- **The read-out on the far probes alone:** ε_∞ = 1.80 in the liquid against MB-pol's 1.69 (job 1). Calibrated on liquid
  cells: 1.6893 (job 5).
- **MACE-LES trained on the 64-molecule cells:**
  - forces 0.20–0.22 kcal/mol/Å at 64–256 molecules;
  - latent dipole c = 0.59–0.61;
  - energy drifting −0.0085 kcal/mol per molecule from 64 to 256 molecules (job 15).
