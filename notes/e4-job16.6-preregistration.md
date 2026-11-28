# E4, job 16.6: MACE-LES's size dependence and LES's k-space cutoff — pre-registration

*28 Nov 2026, registered at this commit's time, before any run of job 16.6 and before any of its numbers. It follows
job 16.5's read (`e4-job16.5-results.md` §3–§4), from which its question and its estimate come. No training. Read
against:*

- *`e4-job16.5-results.md`: MACE-LES's bias falls to 256 molecules and turns back at 512 and 1024, on every trajectory;
  plain MACE does not turn back; the training size shifts MACE-LES by a constant;*
- *`e4-job15-results.md` §10: its drift goes with its charges squared;*
- *les 0.2.0, `les/module/ewald.py`: the legacy periodic path the baselines take (`baselines/mace/les.yaml`).*

## 1. Why

**LES's Ewald leaves out a tail whose share depends on the cell.**

- LES sums its latent charges, Gaussians of σ = 1 Å, in reciprocal space alone, over 0 < |k| ≤ 2π / dl = π / Å.
- What it leaves out, π < |k| ≤ 2π, is about 0.05 kcal/mol per molecule of water. exp(−π²/2) is 0.007, and S(k)
  there is the molecules' own charge distribution.
- Its continuum part is the same in any cell; training absorbs it. The k vectors just inside π / Å are the cell's
  reciprocal lattice, and how much of the shell they fill changes with the cell's size, not smoothly.
- **A model trained at L₀ then reads b_L₀(L) = g(L) + T̄(L₀) − T(L).** Here g is the error of the same model without the
  cutoff, T̄(L₀) the tail its training absorbed, and T(L) the tail on the frames read.

**The estimate fits job 16.5's pattern with one scale.** It uses a rigid molecule's form factor, on each cell's k
lattice: −0.0044 / −0.0077 / −0.0016 / −0.0033 at 128 / 256 / 512 / 1024 against −0.0045 / −0.0079 / −0.0021 / −0.0031
read. That is not a test: the scale is fitted, the form factor is not the liquid's, and the charges are not the model's.
This job computes T for every frame from each model's own latent charges.

## 2. What changes (v0.0.46), fixed before any run

- **`scripts/e4_job16_6.py`.**
  - `tails`: for each MACE-LES model of job 16.5 (L₀ = 64 and 128, seeds 0 and 1), on every frame of every data set its
    energies file holds (64–256 molecules, every split; the test frames of 512 and 1024), and for the default
    MACE-LES's slabs of job 17:
    - T, LES's sum over π < |k| ≤ 2π / Å;
    - E_LES, LES's own energy;
    - both from the latent charges `baselines/mace/energies.py` stored, into `results/e4_job16_6/<run>_tails.npz`.
  - `read`: K16.6.1–K16.6.4.
  - The sum is LES's, in NumPy (torch on the GPU when it is there). It equals les 0.2.0's own Ewald to 10⁻⁸ on cubic,
    orthorhombic and triclinic cells, numpy and torch alike (`tests/test_job16_6.py`, where les is installed). It
    matches a hand-computed case, and a 2 × 2 × 2 replica holds eight times its energy and tail.
- **`run_e4_job16_6.sbatch`:** the tails, then the read. One GPU task, about half an hour.

## 3. Registered

- **b_L₀(n)** is MACE-LES's own bias on the frames job 16.5 reads as unseen at size n, and **T_L₀(n)** the tail per
  molecule on the same frames with that model's charges; seed means. **c = b + T.**
- **T̄(L₀)** is the mean tail on the model's training frames.

| | reading | band | prediction |
|---|---|---|---|
| **K16.6.1** | c₆₄(1024) − c₆₄(256): the reversal | within ±0.0015 of zero (b alone: +0.0048) | holds |
| **K16.6.2** | \|c₆₄(256) − c₆₄(64)\| against \|b₆₄(256) − b₆₄(64)\| (−0.0079) | at most 0.5 of it | holds |
| **K16.6.3** | c₁₂₈(n) − c₆₄(n) against T̄₁₂₈(128) − T̄₆₄(64), at 64, 128, 256, 512 and 1024 | within ±0.0015 | holds |
| **K16.6.4** | job 17's slabs: the default MACE-LES's tail change from the widest gap to 9, 12 and 16 Å, rms over the frames, against MB-pol's E(g) − E(widest) | at most 10 %, both seeds | holds |

**Why these bands, and these predictions.**

- **K16.6.1.** The estimate puts −(T(1024) − T(256)) at +0.0044 of the +0.0048 read, leaving +0.0004. ±0.0015 is about
  three trajectory spreads at 1024 molecules (0.0005).
- **K16.6.2.** If the tail is the drift, little is left: the estimate leaves −0.0002 of −0.0079. Half is generous: a
  physical drift as large as plain MACE's (−0.003) would still hold it, and one as large as M_inf's would not.
- **K16.6.3.** A model trained at 128 absorbs T̄(128) instead of T̄(64). The estimate gives +0.0044, the shift read
  +0.0040 to +0.0053.
- **K16.6.4.** The slab's cell changes along z with the gap, and its k lattice with it. The estimate shifts E(g) − E(24
  Å) by +0.31 / +0.23 / +0.12 kcal/mol per cell, 4.6 / 5.3 / 5.1 % of MB-pol's 6.7 / 4.4 / 2.4. It is the least
  certain: the slab's charge structure is not the liquid's at small k.

## 4. Read beside, not registered

- **T and E_LES per molecule by size,** and T's spread over frames.
- **c at every size and both L₀;** c's step from 256 to 1024 at L₀ = 128.
- **Job 17's K17.8 with the tail added back,** seed by seed.

## 5. What would change the reading

- **All hold:** MACE-LES's bulk size dependence in jobs 15 and 16.5 is LES's cutoff.
  - Its corrected bias g + T̄ is then MACE-LES's own Form II, and it is nearly flat.
  - The plan's Form II row for MACE-LES (∝ 1/L₀) and job 15 §10's reading are revised.
  - LES's dl = 2 Å is too coarse for water at σ = 1 Å.
- **K16.6.2 holds and K16.6.1 fails:** the tail is the drift to 256, but something else turns it back. Read c by size
  and the trajectories.
- **K16.6.1 holds and K16.6.2 fails:** the tail is the reversal, but a physical drift remains under it. Its size law is
  c's, read on the frames off L₀.
- **K16.6.3 fails:** the training size changes more than the tail it absorbs. Read the corrected difference by size.
- **K16.6.4 fails:** job 17's slab reading for MACE-LES carries the cutoff too. K17.8 with the tail added is beside it.

## 6. Not tested here

- **MACE-LES retrained with a finer dl.** That would show the same with a model, at a model's cost.
- **Plain MACE at 128,** and MACE-LES at 128 with the atoms per step matched (`BATCH=1`; job 16.5's K16.5.4).
- **Our harness's own Ewald,** whose reciprocal sum is cut at b_max with a real-space complement. E4's model's bias is
  flat with the cell (K16.2) and L₀ (job 16), so nothing like this shows there.

## 7. Numbers before any run

- **The estimate:**
  - each rigid molecule's orientation-averaged form factor, q_O = −2 q_H, on each cell's k lattice;
  - the tail over π < |k| ≤ 10.5 / Å, about 0.050–0.055 kcal/mol per molecule at q_H = 0.2;
  - the model's shift against 64 molecules, −(T(L) − T(64)): −0.0029 / −0.0052 / −0.0011 / −0.0022 at q_H = 0.2;
    × 1.5 fitted, −0.0044 / −0.0077 / −0.0016 / −0.0033;
  - on random rigid waters, with LES's sum itself (`dl` 2 against 0.9): 0.049 / 0.053 / 0.056 / 0.050 at 64 / 128 /
    256 / 512, ± 0.002–0.004 between configurations.
- **Job 16.5:**
  - b₆₄: +0.0002 / −0.0043 / −0.0077 / −0.0019 / −0.0029;
  - b₁₂₈: +0.0052 / +0.0003 / −0.0037 / +0.0024 / +0.0011;
  - their difference: +0.0040 to +0.0053.
