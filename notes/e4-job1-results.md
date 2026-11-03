# E4, job 1: the results, first read

*3 Nov 2026. Job 1 ran as pre-registered (`notes/e4-x3-preregistration.md`) on Niflheim at e6746b3 (v0.0.21): one job
per seed, each training the 7 rungs in turn. All 28 runs finished, in 8.2–9.2 h per job.*

*This read uses the per-seed lines of the four logs. Prediction 3 and the readings on the pooled frames need the
per-frame errors in the result jsons. `scripts/e4_report.py` (v0.0.22) computes them; §3's prediction 3 and §6 are
completed from its output.*

*Notation as in the pre-registration: M_16†ᵐ is the dagger (MB-pol's sources pinned, no response), M_16C2†ᵐ_us the
frozen form, M_16C2†ᵐ the frozen form without its field term (the ablation). "C rungs" are the last two.*

## 1. The infrared readings: as computed before training

| reading | pre-registered (§4) | job 1 |
|---|---|---|
| ε_∞, C rungs | 1.8003 ± 0.0021 (8 frames of two validation trajectories) | 1.8009 on all 8 runs (the first 10 test frames) |
| T5 slope, C rungs | 1.026 | 1.0260 on all 8 runs |
| far probes, non-additive, 10 / 12 / 15 Å, C rungs | 2.5 / 2.6 / 2.0 % | 2 / 3 / 2 % on all 8 runs (the log prints whole per cent) |
| the dagger: far probes; T5 | 100 %; 0 | 100 %; 0 on all 4 runs |
| every rung without a response: T5 | 0 | 0 on all 16 runs |

- The two C rungs read the same, seed by seed, as their shared matched read-out requires. Training did not move the
  infrared: no implementation error.
- ε_∞ stays the recorded failure: 1.80 against MB-pol's 1.69, χ 16 % too high. MB-pol's value (1.6876 ± 0.0016 on 8
  frames) was not recomputed on these test frames, but the model's own value moved by 0.0006 between the two frame
  sets, so the gap of 0.11 stands.

**A correction to §4.** Its 8 Å column was listed as fixed. It is not:

- 9 of the 20 test probes at 8 Å have an atom within r_c (6 Å) of their core, where E₀ acts, so training moves them.
  The frozen form's 8 Å probe interaction reads 2–6 % across seeds, against 1.4 % before training.
- The test probes at 10–15 Å all lie beyond r_c (closest atom pairs 6.6–14.5 Å) and read as computed.
- The matching was not affected: it already used only the training probes beyond r_c (328 of 384).

## 2. The bulk readings

The valid and test frames at each size are the two halves of one trajectory: 30 / 30 frames at 64 molecules, 15 / 15
at 128, 7 / 8 at 256, 0.5 ps apart. At 128 and 256 every frame is unseen by a model trained at 64.

**Rigid-body forces** (rmse of each molecule's net force and torque, kcal/mol/Å; mean ± sd over 4 seeds):

| rung | test 64 | valid 64 | test 128 | valid 128 | test 256 | valid 256 |
|---|---|---|---|---|---|---|
| M_∞ | 0.898 ± 0.068 | 0.915 ± 0.071 | 0.927 ± 0.074 | 0.897 ± 0.068 | 0.926 ± 0.072 | 0.947 ± 0.059 |
| M_16†ᵐ (dagger) | 0.718 ± 0.023 | 0.724 ± 0.022 | 0.737 ± 0.015 | 0.722 ± 0.017 | 0.738 ± 0.015 | 0.736 ± 0.021 |
| M_16C2†ᵐ (ablation) | 0.739 ± 0.055 | 0.762 ± 0.068 | 0.744 ± 0.058 | 0.750 ± 0.062 | 0.756 ± 0.056 | 0.756 ± 0.058 |
| M_16C2†ᵐ_us (frozen form) | **0.603 ± 0.034** | **0.607 ± 0.030** | **0.608 ± 0.029** | **0.611 ± 0.032** | **0.615 ± 0.033** | **0.615 ± 0.033** |
| M_1 | 1.683 ± 0.306 | | | | 1.716 ± 0.287 | |
| M_16 | 2.595 ± 0.235 | | | | 2.642 ± 0.234 | |
| M_G | 2.988 ± 0.560 | | | | | |

**Energy per molecule** (rmse, kcal/mol):

| rung | test 64 | valid 64 | test 128 | valid 128 | test 256 | valid 256 |
|---|---|---|---|---|---|---|
| M_∞ | 0.0319 ± 0.0023 | 0.0337 ± 0.0031 | 0.0261 ± 0.0015 | 0.0274 ± 0.0009 | 0.0188 ± 0.0030 | 0.0180 ± 0.0010 |
| M_16†ᵐ (dagger) | 0.0315 ± 0.0017 | 0.0303 ± 0.0007 | 0.0191 ± 0.0008 | 0.0251 ± 0.0019 | 0.0159 ± 0.0008 | 0.0112 ± 0.0015 |
| M_16C2†ᵐ (ablation) | 0.0378 ± 0.0053 | 0.0285 ± 0.0039 | 0.0221 ± 0.0030 | 0.0231 ± 0.0023 | 0.0140 ± 0.0008 | 0.0061 ± 0.0029 |
| M_16C2†ᵐ_us (frozen form) | 0.0300 ± 0.0033 | 0.0210 ± 0.0034 | 0.0187 ± 0.0027 | 0.0249 ± 0.0017 | 0.0108 ± 0.0033 | 0.0142 ± 0.0042 |
| M_1 | 0.0538 ± 0.0227 | | | | 0.0333 ± 0.0125 | |
| M_16 | 0.0963 ± 0.0063 | | | | 0.0494 ± 0.0055 | |
| M_G | 0.10 ± 0.02 | | 9.6 ± 2.6 | | 28 ± 11 | |

(M_1, M_16 and M_G: only the cells kept in this read; the report prints every one.)

## 3. The predictions

| prediction (§5) | reading | verdict |
|---|---|---|
| 1. The C rungs at or below the dagger, energy and forces, every size, ≥ 3 of 4 seeds | **Frozen form**: forces 4/4 at every size and split (−15 to −17 %); energies 3–4/4 on test at every size and on valid at 64 and 128, 1/4 on valid 256. **Ablation**: forces 1–3/4 (+1 to +5 %); energies 0/4 on test 64, 2/4 on valid 64, 1/4 on test 128, 4/4 beyond | **fails**: the ablation is no better than the dagger, and the frozen form fails one energy reading |
| 2. The frozen form at or below the ablation, both, every size | Forces 4/4 at every size and split (−18 to −20 %). Energies 4/4 at 64 (test and valid), 3/4 on test 128 and 256, 0/4 on valid 128 and 256 | **fails on energies** (valid at 128 and 256); holds on forces everywhere |
| 3. No bias moves from 64 to 256 molecules by more than 2 frame-to-frame SE, M_G apart | needs the per-frame errors | **pending** (§6) |
| 4. M_G's error grows with size (Form III) | 0.10 → 9.6 → 28 kcal/mol per molecule at 64 → 128 → 256, bias ≈ −rmse on every seed | **holds** |
| M_1, M_16: far probes between the C rungs' and 100 %; T5 exactly 0 | probes 91–99 % at 10–15 Å; T5 0 | as predicted, but these rungs did not train (§4c): it does not count |

## 4. What the readings say

**(a) The forces separate the sectors, the same way at every size.**

- Long range with pinned sources, M_∞ → the dagger: −19 to −22 %.
- The response matched on the far field, without the field term, the dagger → the ablation: +1 to +5 %. Nothing gained.
- The field term, the ablation → the frozen form: −18 to −20 % (−15 to −17 % against the dagger).
- From 64 to 256 molecules the seed-mean forces change by between −1 and +3.5 %, and the gaps between rungs stay.
  These are effects at contact, read in a liquid.

The ablation's null is the ε_∞ failure, seen at contact:

- The response's dipole tensor at the 0.75 Å edge is nearly bare at a hydrogen bond (97–100 %), where MB-pol's Thole
  damping keeps 12–60 % of it. In the liquid the induction is too strong (local-field enhancement 1.30 against 1.17).
- On this reading, E₀ alone cannot absorb that error of the induction, while the field term — E₀ reading the field
  at each atom, the variable the error depends on — can. On q-AQUA-pol the field term did the same at the 2- and
  3-body level (E3-QP8: raw 2-body −25 %, raw 3-body −35 %).
- What the field term cannot reach is the response to a uniform field: that reaches only the read-out. So ε_∞ is
  1.80 on both C rungs.

**(b) The energies do not separate the rungs beyond 64 molecules at this precision.**

- Every rung but M_1, M_16 and M_G is within 0.006–0.04 kcal/mol per molecule.
- At 64 molecules (one trajectory, 60 frames): frozen form 0.030 / 0.021 (test / valid), dagger 0.032 / 0.030,
  M_∞ 0.032 / 0.034, ablation 0.038 / 0.029.
- At 128 and 256 every rung's rmse falls, as a local error averaged over more molecules should. The comparisons then
  flip between the two halves of the same trajectory. At 256 molecules the frozen form reads best on the test half
  (0.011, the ablation 0.014) and third of four on the valid half (0.014), where the ablation reads best (0.006).
- Each half holds 15 (128) or 7–8 (256) frames 0.5 ps apart, so a comparison on one half rests on few independent
  samples of the error's slow part.
- The reading to use at 128 and 256 is the pooled one: train + valid + test, four trajectories, all unseen by a model
  trained at 64. The report gives it.

**(c) M_1 and M_16 did not train.**

- Each contains M_∞ (learned charges, and for M_16 learned α(iω), added to E₀), yet each reads worse than it. On test
  64: energies 0.054 ± 0.023 and 0.096, forces 1.7 and 2.6 kcal/mol/Å, against M_∞'s 0.032 and 0.90. M_1's seeds
  spread 2.3× (0.037–0.086).
- M_16's initial losses were 3.5×10⁷–2.8×10⁸, and its L-BFGS polish was still descending at its last step.
- The cause is its start (measured here at initialisation, seed 0, on a local 64-molecule frame):
  - its learned α(iω) starts flat at softplus(1) = 1.31 au on all 8 frequency nodes, so its C₆ is 4× MB-pol's
    long-range value for O–O and 49× for H–H;
  - at the band's 0.75 Å edge that is a dispersion energy of about −36 kcal/mol per atom, against an energy scale of
    0.06 (the training frames' spread per atom): an energy term of 4×10⁷ in the loss.
- M_1 starts close to M_∞: +4.5 against +3.2 kcal/mol per atom (the missing offset), forces 32 against 8 kcal/mol/Å,
  the energy term 6×10⁵ against 3×10⁵. Its failure is not its start alone; its Adam curves (in the jsons; the report
  prints them) are the next thing to read.
- So their readings measure the optimiser, not the rungs. The fix proposed: start the learned sources small — the
  charge head's output layer at zero, so that M_1 starts exactly as M_∞ (checked), and α(iω) small enough that its
  C₆ starts below MB-pol's — and check that each reaches M_∞'s training error before reading it. It is a training
  change, registered before they run again.

**(d) M_G: Form III, measured.**

- Its pair function saw pairs up to L₀/2 = 6.2 Å in training. At 128 and 256 molecules it is evaluated out to 7.8
  and 9.9 Å.
- Its energy error grows about 100-fold from 64 to 128 molecules and about 300-fold to 256, the bias ≈ −rmse on every
  seed.
- On the open probes, whose pairs reach 15 Å, its non-additive error is 5–22 times the truth's rms.

**(e) The atomic forces.** The rigid loss leaves the forces' internal components free:

- the frozen form's atomic rmse is 9–12 kcal/mol/Å against 0.6 for the rigid part; the ablation's is 4.6–5.8;
- the field term carries internal forces that a rigid trajectory never feels.

Rungs and baselines are compared on the rigid part. For MACE: `scripts/e4_far.py cells` (v0.0.22).

**(f) Cost.**

- A C rung took 3.1–3.8 h: Adam about 1.3 s per step on 6 cells, the L-BFGS polish about 1.8 h. The estimate was 5 h.
- M_16 took 0.5 h, the dagger 0.45, M_1 0.35, M_G 0.3, M_∞ 0.23.
- A job took 8.2–9.2 h against 16 h estimated: about 35 GPU hours in all.

## 5. Corrections and notes to the pre-registration

- §4's 8 Å column (§1 above).
- §1's cost (§4f above).
- **Prediction 1 was registered for both C rungs.** It fails on the ablation: the response without the field term
  buys nothing in bulk. That is information about the field term and E₀, as §7 of the pre-registration reads a bulk
  failure, not about the infrared.
- **§5 did not fix the frame set** a bulk prediction is read on. Every set is reported here and in the report. The
  verdicts on forces do not depend on it; those on energies at 128 and 256 do.

## 6. To complete, and next

**The report**, seconds on a login node:

```sh
python scripts/e4_report.py --out_dir results/e4_job1 --data data/e4 --json results/e4_job1/report.json
```

It gives:

- prediction 3, with the frame-to-frame standard error as registered, and with the frames' correlation in time beside
  it;
- predictions 1 and 2 on the pooled frames;
- the truth's energy per molecule by size and trajectory, and each rung's per-frame error against it.

**One thing to look for.** M_∞'s bias goes from about 0 on test 64 to −0.014 to −0.018 on test 128. On the 45
training frames at 256 molecules (three trajectories, unseen by a model trained at 64) it is −0.011 to −0.014, on
every seed.

- Against the frame-to-frame standard error, the shift at 128 molecules is about 2 SE: borderline.
- The seeds' agreement does not count as evidence: they share the frames.
- If the pooled reading confirms it, it is the long-range part of the truth's energy per molecule changing with the
  cell, which a cutoff model cannot follow. The report shows every rung's bias by size on the same frames.

**Next** (proposed):

1. Complete this note from the report.
2. Re-run M_1 and M_16 with their sources starting small, about 1 GPU hour per seed for both.
3. Before job 2: the liquid-cluster response matching (pre-registration §4) — the remedy for ε_∞, and by (a) the
   response's error at contact too. Then the readers on neighbour lists.
