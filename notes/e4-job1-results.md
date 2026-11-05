# E4, job 1: the results

*3 Nov 2026. Job 1 ran as pre-registered (`notes/e4-x3-preregistration.md`) on Niflheim at e6746b3 (v0.0.21): one job
per seed, each training the 7 rungs in turn. All 28 runs finished, in 8.2–9.2 h per job.*

*Read with `scripts/e4_report.py` (v0.0.22) on the 28 result jsons and the data sets' frames (`results/e4_job1/
report.json`). A first read from the logs came before it; the report agrees with it split by split, to rounding.*

*Notation as in the pre-registration: M_16†ᵐ is the dagger (MB-pol's sources pinned, no response), M_16C2†ᵐ_us the
frozen form, M_16C2†ᵐ the frozen form without its field term (the ablation). "C rungs" are the last two. Errors are
seed means ± sd over 4 seeds unless a trajectory-level standard error is given.*

## 1. The infrared readings: as computed before training

| reading | pre-registered (§4) | job 1 |
|---|---|---|
| ε_∞, C rungs | 1.8003 ± 0.0021 (8 frames of two validation trajectories) | 1.8009 on all 8 runs (the first 10 test frames) |
| T5, C rungs: rmse / rms; slope | 3.7 %; 1.026 | 3.74 %; 1.0260 on all 8 runs |
| far probes, non-additive, 10 / 12 / 15 Å, C rungs | 2.5 / 2.6 / 2.0 % | 2.5 / 2.6 / 2.0 % on all 8 runs |
| far probes, interaction, 10 / 12 / 15 Å, C rungs | 0.5 / 0.7 / 0.5 % | 0.5 / 0.7 / 0.5 % |
| the dagger: probes, non-additive; interaction; T5 | 100 %; 19 / 22 / 25 %; 0 | 100.0 %; 18.6 / 22.2 / 25.3 %; 0 |
| every rung without a response: T5 | 0 | 0 on all 20 runs (round-off) |

- Across the 8 C-rung runs the largest spread of a reading that must agree is 2×10⁻⁸ (T5, in per cent). The two C
  rungs read the same, seed by seed, as their shared matched read-out requires. Training did not move the infrared:
  no implementation error.
- ε_∞ stays the recorded failure: 1.80 against MB-pol's 1.69, χ 16 % too high. MB-pol's value (1.6876 ± 0.0016 on 8
  frames) was not recomputed on these test frames, but the model's own value moved by 0.0006 between the two frame
  sets, so the gap of 0.11 stands.

**A correction to §4.** Its 8 Å column was listed as fixed. It is not:

- 9 of the 20 test probes at 8 Å have an atom within r_c (6 Å) of their core, where E₀ acts, so training moves them.
  After training the 8 Å probe interaction reads 4.2 ± 1.7 % (frozen form) and 4.6 ± 3.0 % (ablation), against 1.4 %
  before; the non-additive part 1.9 ± 0.1 % and 3.1 ± 1.4 %, against 1.9 %.
- The test probes at 10–15 Å all lie beyond r_c (closest atom pairs 6.6–14.5 Å) and read as computed.
- The matching was not affected: it already used only the training probes beyond r_c (328 of 384).

## 2. The bulk readings

The frame sets: at 64 molecules 180 training frames (three trajectories) and the fourth trajectory's two halves as
valid and test (30 + 30); at 128 and 256 molecules the same split (90 + 15 + 15, 45 + 7 + 8), every frame unseen by a
model trained at 64. Frames are 0.5 ps apart.

The reading used here is the unseen frames pooled: valid + test at 64 molecules (60 frames, one trajectory), every
frame at 128 and 256 (120 and 60 frames, four trajectories). The first read named it as the one to use before it was
computed. The report prints every split as well.

**Rigid-body forces** (rmse of each molecule's net force and torque, kcal/mol/Å) and **energy per molecule** (rmse and
mean error, kcal/mol), unseen frames:

| rung | F 64 | F 128 | F 256 | E 64 | E 128 | E 256 | bias 64 | bias 128 | bias 256 |
|---|---|---|---|---|---|---|---|---|---|
| M_∞ | 0.907 ± 0.069 | 0.917 ± 0.070 | 0.929 ± 0.066 | 0.0329 | 0.0254 | 0.0200 | −0.0026 | **−0.0097** | **−0.0127** |
| M_16†ᵐ (dagger) | 0.721 ± 0.022 | 0.735 ± 0.021 | 0.738 ± 0.018 | 0.0309 | 0.0221 | 0.0140 | −0.0032 | −0.0007 | −0.0024 |
| M_16C2†ᵐ (ablation) | 0.751 ± 0.061 | 0.760 ± 0.059 | 0.761 ± 0.062 | 0.0335 | 0.0228 | 0.0134 | −0.0021 | −0.0038 | −0.0026 |
| M_16C2†ᵐ_us (frozen form) | **0.605 ± 0.032** | **0.619 ± 0.032** | **0.618 ± 0.030** | **0.0259** | **0.0212** | **0.0122** | −0.0036 | +0.0002 | +0.0014 |
| M_1 | 1.695 ± 0.302 | 1.711 ± 0.298 | 1.713 ± 0.294 | 0.0555 | 0.0500 | 0.0329 | −0.0149 | −0.0204 | −0.0140 |
| M_16 | 2.607 ± 0.239 | 2.661 ± 0.246 | 2.643 ± 0.240 | 0.0896 | 0.0694 | 0.0470 | +0.0083 | −0.0141 | −0.0070 |
| M_G | 2.99 ± 0.56 | 4.46 ± 0.26 | 10.7 ± 1.5 | 0.103 | 9.66 | 28.1 | +0.015 | −9.65 | −28.1 |

(Energy sds over seeds: 0.0005–0.005 for the first four rungs, 0.004–0.016 for M_1 and M_16.)

## 3. The predictions

| prediction (§5) | reading (unseen frames; the single splits in brackets) | verdict |
|---|---|---|
| 1. The C rungs at or below the dagger, energy and forces, every size, ≥ 3 of 4 seeds | **Frozen form**: forces 4/4 at every size (−16 %); energies 4/4, 3/4, 4/4 (−16, −4, −13 %) [fails on valid 256: 1/4]. **Ablation**: forces 2/4 at every size (+3 to +4 %); energies 2/4, 0/4, 3/4 (+8, +3, −5 %) | **fails**: the ablation is no better than the dagger |
| 2. The frozen form at or below the ablation, both, every size | Forces 4/4 at every size (−19 %); energies 4/4 at every size (−23, −7, −9 %) [fails on valid 128 and 256: 0/4] | **holds** on the unseen frames |
| 3. No bias moves from 64 to 256 molecules by more than 2 frame-to-frame standard errors, M_G apart | M_∞: −0.0101 ± 0.0041 (z −2.4; test −2.3, valid −1.1). Every other rung within 2 SE (\|z\| ≤ 1.5; valid 1.8 for the frozen form) | **fails on M_∞** (§4b) |
| 4. M_G's error grows with size (Form III) | 0.10 → 9.7 → 28 kcal/mol per molecule at 64 → 128 → 256, bias ≈ −rmse on every seed | **holds** |
| M_1, M_16: far probes between the C rungs' and 100 %; T5 exactly 0 | non-additive 94–96 % (M_1) and 98 % (M_16) at 10–15 Å; T5 0 | as predicted, but these rungs did not train (§4d): it does not count |

- The frame-to-frame standard errors hold with the frames' correlation in time too: the lag-1 correlation of the
  errors at 0.5 ps is −0.6 to +0.4 and changes no verdict.
- The two single-split failures of predictions 1 and 2 (the validation halves at 128 and 256 molecules: 15 and 7
  frames of one trajectory) do not survive the pooled frames, as the first read expected.

## 4. What the readings say

**(a) The forces separate the sectors, the same way at every size** (unseen frames):

| step | rigid forces | net forces | torques |
|---|---|---|---|
| long range with pinned sources: M_∞ → dagger | −20 to −21 % | −10 to −11 % | −19 to −20 % |
| the response without the field term: dagger → ablation | +3 to +4 % | +8 to +11 % | +2 % |
| the field term: ablation → frozen form | −19 % | −17 to −18 % | −17 % |

- The pinned long range acts mostly on the torques: a field from beyond 6 Å turns a molecule's dipole far more than it
  pushes it.
- The response without the field term buys nothing and worsens the net forces by 8–11 %. The field term repairs both.
- From 64 to 256 molecules every rung's forces but M_G's rise by 1–2.4 %, and the gaps stay. These are effects at
  contact, read in a liquid.

The ablation's null reads as the ε_∞ failure, seen at contact:

- The response's dipole tensor at the 0.75 Å edge is nearly bare at a hydrogen bond (97–100 %), where MB-pol's Thole
  damping keeps 12–60 % of it. In the liquid the induction is too strong (local-field enhancement 1.30 against 1.17).
- On this reading, E₀ alone cannot absorb that error of the induction, while the field term — E₀ reading the field
  at each atom, the variable the error depends on — can. On q-AQUA-pol the field term did the same at the 2- and
  3-body level (E3-QP8: raw 2-body −25 %, raw 3-body −35 %).
- What the field term cannot reach is the response to a uniform field, which reaches only the read-out. So ε_∞ is
  1.80 on both C rungs.

**(b) The cutoff model's energy moves with the cell; the long-range rungs' does not.** This is prediction 3's failure,
and the clearest size effect in job 1.

- M_∞'s mean error is −0.003 at 64 molecules (its unseen trajectory), −0.010 at 128 and −0.013 at 256; at 128 and
  256 on every seed.
- Paired with the dagger on the same frames, trajectory by trajectory, the truth's own scatter cancels:

  | M_∞ − dagger, mean error per molecule | 64 | 128 | 256 |
  |---|---|---|---|
  | by trajectory (kcal/mol) | +0.0006 (one; the fitted 180 frames: +0.0004) | −0.0101, −0.0093, −0.0067, −0.0097 | −0.0144, −0.0067, −0.0101, −0.0097 |
  | mean ± se over trajectories | | **−0.0089 ± 0.0008** | **−0.0102 ± 0.0016** |

- The C rungs against the dagger, the same way, at 128 and 256 molecules: the ablation −0.0030 ± 0.0031 and
  −0.0002 ± 0.0021, no drift; the frozen form +0.0009 ± 0.0038 and +0.0039 ± 0.0019, the latter at 2 se and of the
  other sign (its registered reading holds: z 1.5).
- So a model whose long range is all in E₀ carries the 64-molecule cell's long-range energy into larger cells: it
  over-binds them by about 0.01 kcal/mol per molecule. The rungs that carry the electrostatics explicitly follow the
  cell.
- Most of the drift comes between 64 and 128 molecules. The two points cannot tell 1/N from a faster fall-off (the
  ratio of the shifts is 1.15 ± 0.21; 1/N gives 1.5).
- The truth's energy per molecule cannot show it on its own: by size it is −9.814 ± 0.015, −9.795 ± 0.023 and
  −9.781 ± 0.027 kcal/mol, the trajectories' means spanning 0.06–0.12 kcal/mol at each size, 6–12 times the drift.
  The models' errors subtract the truth frame by frame, which is why they resolve 0.001.
- *Attribution, not yet checked.* The dagger differs from M_∞ by its pinned bands (the site charges' Coulomb and the
  c₆ dispersion) and by its E₀. Dispersion beyond 6 Å has no size dependence in a uniform liquid, so the drift points
  at the electrostatics. How M_∞ carries it is open: E₀'s two message-passing layers reach 12 Å, which in the
  64-molecule cell (L₀ = 12.4 Å) is the whole cell through its images, so E₀ may have learned the training cell's
  long range as a function of the cell itself. A first check costs no training (§6).
- M_∞'s per-frame error also follows the truth's energy within a trajectory, more so at larger cells (slope −0.016,
  −0.028, −0.054; r −0.09, −0.21, −0.36); the dagger's does not (|r| ≤ 0.08). Among the other trained rungs the one
  correlation above noise is the frozen form's at 256 molecules (r +0.36), unexplained. M_1's and M_16's errors follow
  the truth's energy at every size (r −0.2 to −0.5): their energies are too flat.

**(c) On the pooled frames the energies rank the rungs as the forces do.**

- At every size the frozen form is lowest (0.026, 0.021, 0.012). At 128 and 256 the dagger and the ablation follow
  (0.022–0.023, 0.013–0.014) and M_∞ is highest (0.025, 0.020), most of its excess at 256 its bias.
- Every rung's rmse falls with size, as a local error averaged over more molecules should.
- A single split at 128 or 256 molecules is half of one trajectory, 7–15 frames. Comparisons on one split flip
  between the halves; on the pooled frames they do not.

**(d) M_1 and M_16 did not train.**

- Each contains M_∞ (learned charges, and for M_16 learned α(iω), added to E₀), yet each reads worse than it: forces
  1.7 and 2.6 kcal/mol/Å against 0.91, energies 0.056 and 0.090 against 0.033 at 64 molecules.
- The Adam curves (the loss at every 300th step) show two failures:
  - **M_16 starts far off.** Its loss is 3×10⁶–3×10⁸ at step 0, against M_∞'s 3×10⁵. Its learned α(iω) starts flat at
    softplus(1) = 1.31 au on all 8 frequency nodes: a C₆ 4× MB-pol's long-range value for O–O and 49× for H–H, about
    −36 kcal/mol per atom at the band's 0.75 Å edge (measured here at initialisation on a local frame). It ends at
    7×10²–1.1×10³, 11–16× M_∞'s final loss, seed by seed.
  - **M_1 starts where M_∞ does and stalls.** Its loss is 2.3–6.8×10⁵ at step 0 and 0.7–1.2×10³ at step 300 (M_∞:
    3×10⁵ and 0.5–0.6×10³), then falls slowly to 1.9–4.0×10² (M_∞: 0.6–0.8×10²). Its worst seed in loss is its worst
    on the test frames.
- A plausible reason for the stall: at this energy scale (the training frames' spread, 0.06 kcal/mol per atom) a
  small change of a learned charge moves the energy by a large step of the loss, so the charge head's Adam steps are
  far too large. Not tested.
- So their readings measure the optimiser, not the rungs. The fix proposed: the learned sources start at zero and move
  in small units (an output scale on the charge and polarisability heads, M_1 then starting exactly as M_∞), with a
  gate — each must reach M_∞'s training loss before it is read. It is a training change, registered before they run
  again.

**(e) M_G: Form III, measured.**

- Its pair function saw pairs up to L₀/2 = 6.2 Å in training. At 128 and 256 molecules it is evaluated out to 7.8
  and 9.9 Å.
- Its energy error grows 94-fold from 64 to 128 molecules and 270-fold to 256, the bias ≈ −rmse on every seed; its
  forces 1.5- and 3.6-fold.
- On the open probes, whose pairs reach 15 Å, its non-additive error is 5.5–19 times the truth's rms at 10–15 Å.

**(f) The atomic forces.** The rigid loss leaves the forces' internal components free. Atomic rmse: M_∞ 3.8, dagger
4.2, M_16 4.9, ablation 5.3, M_1 7.5, frozen form 10.4, M_G 14 kcal/mol/Å. The field term carries internal forces
that a rigid trajectory never feels. Rungs and baselines are compared on the rigid part: for MACE, `scripts/e4_far.py
cells` (v0.0.22).

**(g) Cost.** Training alone took 2.8–3.7 h for a C rung (Adam about 1.3 s per step on 6 cells), 0.36–0.49 h for the
dagger and M_16, 0.17–0.30 h for M_∞, M_1 and M_G. A job took 8.2–9.2 h with the readings, against 16 h estimated:
about 35 GPU hours in all.

## 5. Corrections and notes to the pre-registration

- §4's 8 Å column (§1 above).
- §1's cost (§4g above).
- **Prediction 1 was registered for both C rungs.** It fails on the ablation: the response without the field term
  buys nothing in bulk. That is information about the field term and E₀, as §7 of the pre-registration reads a bulk
  failure, not about the infrared.
- **§5 did not fix the frame set** a bulk prediction is read on. Every set is in the report. The pooled unseen frames
  are the reading here, named before they were computed. The verdicts on forces do not depend on the set; those on
  energies at 128 and 256 molecules do, on single splits only.
- **§5 said bulk errors "separate no classes".** Their size dependence does: the cutoff model's energy drifts with the
  cell and the rungs with explicit electrostatics do not. For the L₀ sweep (Form II, e4-plan.md §4) this means M_∞'s
  bulk bias — and a cutoff MACE's — depends on L₀, not "flat in L₀" as the plan's table has it: it is the long-range
  energy of the training cell, carried along.

## 6. Next

1. **The attribution check** (§4b), no training: MB-pol's site charges' electrostatic energy from atom pairs beyond
   6 Å, per molecule, on every frame of 64, 128 and 256 molecules, by trajectory. If its size dependence is the +0.010
   kcal/mol per molecule M_∞ misses, the drift is that energy.
2. **M_1 and M_16 again**, with their sources starting at zero in small units and the training-loss gate (§4d): about
   1 GPU hour per seed for both.
3. **Before job 2**: the liquid-cluster response matching (pre-registration §4) — the remedy for ε_∞ and, by §4a,
   for the response's error at contact. Then the readers on neighbour lists. *Revised 4 Nov*
   (`e4-response-regulator.md`): the mutual tensor's own edge at 2.0 Å, fixed in advance, halves ε_∞'s error before
   training; the liquid calibration is for what remains.
