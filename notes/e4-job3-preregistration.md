# E4, job 3: the finite-k response, and the locality test — pre-registration

*7–8 Nov 2026, v0.0.27. Vadic's go of 7 Nov, after arm C's read (`e4-job2c-results.md` §6): build the finite-k response,
read arm C on it as a prediction before any training, with the locality test beside it. No training: every reading here
is fixed by read-outs job 2 already matched. The local master of this note is `notes/` in the repository.*

*Timing of what was written when:*

- *The expectations (§3a) were written at 22:30 CET on 7 Nov, before MB-pol's finite-k response was computed on any
  frame and before the locality test ran (`scratch/e4_job3_expectations.md`). Seen before them: a timing check of the
  model's mode response on one local frame with arm C's read-out, no MB-pol number.*
- *§3's numbers were computed after that, on the local frames (13 cells, the local read-outs of job 2's pre-registration).*
- *The locality test's ridge, 10⁻⁶, was set before its fit. It was checked on the local held-out cells' liquid readings
  — as good as the calibration cells' — before its far probes were read. No other ridge was tried.*
- *§4's readings for Niflheim were written after §3's numbers, before the job ran.*

## 1. The question

**Arm C is E4's model** (`e4-job2c-results.md`). It has two kinds of read-out coefficients, from two kinds of data:

- Its order-0 coefficient — the polarisability — is calibrated in the state of use, on MB-pol's response of liquid
  cells to a uniform field.
- Its orders ≥ 1 — the couplings to the field's gradients — are still the far probes' solution under the ridge.

The far probes show every core a far field of one kind, nearly uniform across it, so they do not identify the orders ≥ 1
(Lemma 3; the workflow note's derivative counting, `eft-of-lss-workflow.md` §5). The order-2 coefficients belong to the
response's k-dependence. **Job 3 measures that k-dependence in MB-pol and reads the model's against it.** Three
questions:

- **(i)** Does arm C predict MB-pol's finite-k response?
- **(ii)** What do its orders ≥ 1 do there, against its order-0 sector alone? And what do arm B's do, moved by k = 0
  data?
- **(iii) The locality test** (the workflow note's §5). Calibrate the order-0 coefficient on the liquid alone. Do the
  vacuum cores' far probes then come out as out-of-state predictions?

## 2. Design

**The modes** (`sseft/modes.py`). Longitudinal plane waves of the cell, per unit amplitude:

- u(x) = k̂ cos(k·x − pπ/2), with k = 2πn b_axis along each axis, harmonics n = 1, 2, phases p = 0 (cos) and 1 (sin);
- that is 12 modes a frame, at k₁ = 0.506 and k₂ = 1.011 Å⁻¹ in the 64-molecule cells (L = 12.43 Å);
- curl-free, the field of an external potential, so its gradient tensors are symmetric, as the read-out reads them.

**MB-pol's response** (`scripts/e4_water.py response_k`).

- MBX solves the induced dipoles with the mode's field set at every site it evaluates, the replica's copies included
  (`mbpol.set_site_fields`).
- For each mode it records every molecule's induced dipole per unit amplitude and the mode's susceptibility
  χ_mode = Σ_s u_s·dμ_s/da over the cell's sites. Central differences, 24 dipole solves a frame.
- The frames are job 2's: 30 calibration and 10 held-out cells.

**The model's response** (`model.mode_response`).

- The mode's field and its first and second derivatives at the atoms are added to the band field's tensors that the
  read-out reads.
- χ_mode = −d²E/da²: everything the read-out's energy does in the mode.
- Each molecule's induced dipole is Σ_i −∂E/∂E_ext,i at its atoms, the derivatives held: the order-0 sector's dipoles
  and the field-gradient terms of the orders ≥ 1.

**The readings** (`read_response_k`).

- **The cells' susceptibility.** χ(k) = (χ_cos + χ_sin)/V averaged over the axes, and ε_L(k) = 1/(1 − 4πχ(k)), for the
  model and for MB-pol. Their relative difference is taken frame by frame, then averaged.
  - At k → 0 under tin-foil boundary conditions the uniform field is the macroscopic field, so χ(0) = (ε₀ − 1)/4π.
  - At finite k a longitudinal field is screened, and the limit is χ(0⁺) = (1 − 1/ε₀)/4π.
  - The model and MBX treat this alike: both sum the induced dipoles' fields by Ewald, without the k = 0 term.
- **Every molecule's induced-dipole response:** rmse ÷ rms over molecules and modes, the correlation r, and the ratio of
  the norms.

**The read-outs read** (`kread`; no training, so the seeds play no part):

| | read-out | what it carries |
|---|---|---|
| A | job 2's arm A | every order from the far probes |
| B | job 2's arm B | A's, then every order on the liquid's uniform response (cell sums) |
| C | job 2's arm C, E4's model | A's, then the order-0 columns on the same |
| C0 | arm C with its orders ≥ 1 off | arm C's order-0 sector alone |
| L | the locality test | order 0 from zero on every calibration molecule's induced dipole and field response alone |

**The locality test's read-out (L).**

- **The fit.** `linear.solve_readout` with the probes' weights at zero, so the liquid rows are the whole objective
  (v0.0.27).
  - The rows are every calibration molecule's induced dipole and field response, in units of μ_ref = 0.15 e Å and
    α_ref = 1.4075 Å³: 12 numbers a molecule, 768 a cell.
  - Job 2's calibration used the cells' sums, 12 numbers a cell, each an average over 64 molecules.
  - Here no probe constrains the coefficient function's ~66 effective numbers (4 species pairs × 2 invariants × 8
    radial functions, and 2 node terms). So the molecules' own rows are used instead; the cells' sums alone were not
    tried.
  - Their Jacobian is taken by forward mode, in chunks of the coefficients.
  - 4 Gauss–Newton steps, with a ridge of 10⁻⁶ relative to these rows.
- **Its far probes and T5 are out-of-state predictions.** Compare them with C0's (order 0, matched on the probes, then
  on the liquid's cell sums) and with job 2's order-0 read-out matched on the probes alone.

**Cost on Niflheim.**

- The response: 40 frames × 24 MBX solves, in two CPU tasks of 20 frames (`run_e4_response_k.sbatch`). Locally a frame
  took 100 s on 2 cores; there, under an hour a task.
- The reading: one GPU job (`run_e4_job3.sbatch`), which joins the two parts first. It reads arms A, B and C from their
  caches in `results/e4_job2{a,b,c}`, and solves L. About an hour.

## 3. The local numbers

13 local cells (10 calibration, 3 held out) and the local read-outs of job 2's pre-registration
(`scratch/e4_job3_local.json`; `kread` with `scratch/e4_job3_local.yaml`'s settings). The readings agree frame to frame
to 0.2–0.5 % (the chi differences' sd), so 13 cells read them closely. Held-out cells:

| read-out | ε₀ (MB-pol 1.6870) | ε_L(k₁) (MB-pol 1.7718) | χ(k₁) | ε_L(k₂) (MB-pol 2.0904) | χ(k₂) | molecules k₁ / k₂ |
|---|---|---|---|---|---|---|
| A | 1.7429 | 1.8153 | +3.1 % | 1.9756 | −5.3 % | 7.8 / 8.2 % |
| B | 1.6882 | 1.8113 | +2.8 % | 2.1998 | +4.6 % | 3.3 / 4.4 % |
| **C** | 1.6895 | **1.7462** | **−1.9 %** | **1.8577** | **−11.5 %** | 3.7 / 4.7 % |
| C0 | 1.6895 | 1.8077 | +2.6 % | 2.1726 | +3.5 % | 5.9 / 11.4 % |
| L | 1.6873 | 1.8048 | +2.4 % | 2.1654 | +3.2 % | 4.9 / 11.1 % |

The calibration cells read within 0.3 points of these. Beside them, on the same read-outs:

| read-out | far probes, non-additive, 8 / 10 / 12 / 15 Å | T5 slope (rmse) | liquid at k = 0, per molecule: μ; R (iso); spread (r) |
|---|---|---|---|
| A | 1.9 / 2.3 / 2.1 / 2.1 % | 1.042 (4.7 %) | 8.2 %; 9.2 (8.7) %; 1.71 (−0.07) |
| B | 1.9 / 2.1 / 2.6 / 2.8 % | 1.040 (4.2 %) | 4.6 %; 2.2 (1.7) %; 1.22 (0.49) |
| C | 2.1 / 2.3 / 2.2 / 2.7 % | 1.040 (4.1 %) | 5.0 %; 3.1 (2.3) %; 1.14 (0.04) |
| C0 | 7.0 / 7.7 / 6.0 / 6.8 % | 1.029 (4.2 %) | as C |
| L | 6.0 / 6.7 / 5.4 / 5.6 % | 1.031 (3.7 %) | **1.9 %; 1.1 (0.4) %; 0.99 (0.97)** |
| order 0, matched on the probes (job 2) | 4.9 / 7.0 / 4.0 / 4.5 % | 0.980 | — |

### (a) The expectations against the numbers

1. **MB-pol's χ(k₁) exceeds its screened limit by 2–10 %, and χ(k₂) by more: holds.** It is +7.0 % at k₁ and +28 %
   at k₂ (ε_L 1.772 and 2.090, against ε₀ 1.687).
2. **Arm C within ±3 % at k₁ and ±6 % at k₂: holds at k₁, fails at k₂.** It is −1.9 % at k₁ and −11.5 % at k₂. Arm C
   gives half of MB-pol's k-dependence: χ(k)/χ(0⁺) − 1 is +4.7 % and +13 %, against +7.0 % and +28 %.
3. **The orders ≥ 1 move χ(k₁) away from MB-pol's: fails.** At k₁ they move it toward MB-pol's and past it (C0
   +2.6 % → C −1.9 %). At k₂ they move it away (+3.5 % → −11.5 %).
4. **The molecules about 5 % at k₁ and 10 % at k₂: fails by an improvement.** They are 3.7 % and 4.7 %. Here the
   orders ≥ 1 help: the order-0 sector alone is 5.9 % and 11.4 %.
5. **Arm A 3–5 % above at k₁: holds** (+3.1 %).
6. **Arm B no closer than C at k₁, by at least 1 point: fails at the margin.** It is 0.9 points farther at k₁
   (+2.8 % against −1.9 %), and far closer at k₂ (+4.6 % against −11.5 %). It is the best of the arms per molecule.
7. **The locality test: holds.** L's far probes, 6.0 / 6.7 / 5.4 / 5.6 %, are within 1.5× of the probe-matched order-0
   read-out's at every distance (1.2×, 1.0×, 1.4×, 1.2×), and below C0's. T5's slope is 1.031.

### (b) What the numbers say

**(i) Arm C misses MB-pol's response at the cutoff's scale, through its orders ≥ 1.**

- At k₁ (wavelength 12 Å) every read-out is within 3.1 % on the cells' susceptibility: order 0, set by ε₀, carries it.
- At k₂ (wavelength 6 Å, the cutoff's scale) arm C is 11.5 % low.
  - Its order-0 sector alone is 3.5 % high.
  - Arm C's far-probe orders ≥ 1 take 15 points.
- **The decomposition** (`scratch/e4_job3_decompose.py`, held-out frame 0). Of arm C's −11.7 % there:
  - −7.0 points are in the dipoles' share, Σ u_i·dμ_i/da;
  - the rest is the energy of the higher moments the orders ≥ 1 induce: the quadrupole terms, which have no
    counterpart in MB-pol's point dipoles.
- **Per molecule** the same orders ≥ 1 help (4.7 % against the order-0 sector's 11.4 % at k₂). MB-pol's molecules answer
  a field that varies across them, and the order-0 class answers it less well than with a gradient coupling. Arm B's
  orders ≥ 1, moved by k = 0 data, happen to keep both: χ +4.6 %, molecules 4.4 %.
- The far probes need the orders ≥ 1 (C0's probes are 6–8 %, C's 2–3 %). In the liquid at finite k, their far-probe
  values are wrong. The derivative counting says the finite-k response is the data that identify them.

**(ii) The order-0 coefficient is local at this cutoff (the locality test).**

- **L in its own state.** Calibrated on every liquid molecule alone, it reproduces the liquid molecules' response to
  1.1 % and their induced dipoles to 1.9 %, held-out cells as well as the calibration cells. The spread is 0.99 and r
  0.97, against arm C's 1.14 and 0.04.
- **L out of state.** It predicts the vacuum cores' far field to 5.4–6.7 %, the order-0 class's own floor (4.0–7.0 %
  matched on the probes themselves), with T5 at 1.031.
- **For E4's model:**
  - The far probes and the liquid ask the same of the order-0 coefficient, so arm C's joint matching is consistent.
  - Calibrating the liquid's molecules one by one, not by cell sums, is available at order 0. It would fix the pattern
    arm C leaves (spread and r).

## 4. Registered for Niflheim

Job 3 on Niflheim (`run_e4_response_k.sbatch`, then `run_e4_job3.sbatch`): MB-pol's finite-k response on job 2's 40
cells, then `kread` with Niflheim's read-outs, which job 2 matched and calibrated on 30 cells.

- Arm A's read-out is the same matching on the same probes as the local one. Their objectives differ slightly
  (9.696×10⁻⁶ against 9.904×10⁻⁶ locally): the training cells' energy scale differs.
- Arms B and C were calibrated on 30 cells, not 10. Arm C on Niflheim reads the molecules' pattern differently: spread
  0.84 and r 0.42, against 1.14 and 0.04 locally.

The tolerances below allow for this.

| | reading (held-out cells unless said) | registered |
|---|---|---|
| **K0** | MB-pol's ε_L(k₁), ε_L(k₂), over the 40 cells | 1.772 ± 0.010, 2.088 ± 0.020; χ(k)/χ(0⁺) − 1 in 5–9 % and 24–32 % |
| **K1** | arm C's χ(k₁), χ(k₂) against MB-pol's | −3.5 to −0.5 %; **−15 to −8 %** |
| **K2** | arm C's molecules at k₁ / k₂ | 2–6 % / 3–7 %, and below C0's at both |
| **K3** | C0 (arm C's orders ≥ 1 off) at k₂ | χ +1.5 to +5.5 %, closer to MB-pol's than arm C's |
| **K4** | arms A and B: χ(k₁); χ(k₂) | A +1.5 to +4.5 %; −8.5 to −2 %. B +1.5 to +4.5 %; +1.5 to +7.5 % |
| **K5** | L, in state: the held-out molecules' dipoles; response; r | ≤ 3 %; ≤ 2 %; ≥ 0.9, within 1 point of the calibration cells |
| **K6** | L, out of state: the far probes at 10 / 12 / 15 Å; T5 | ≤ 10.5 / 6.0 / 6.8 % (1.5× the probe-matched order 0's); 0.95–1.10 |

A K0 failure is MB-pol's or the data's and is read first. K1–K4 are this note's main prediction: arm C's orders ≥ 1
mispredict MB-pol's finite-k response at k₂, and its order-0 sector does not. K5–K6 are the locality test.

## 5. What would change the reading

- **K1 and K3 hold.** Arm C's orders ≥ 1, from the far probes, mispredict MB-pol's response at the cutoff's scale.
  - **Next is arm D.** Calibrate the orders ≥ 1 on the calibration cells' finite-k response, with the order-0 columns
    as arm C's and the probes' rows kept: the derivative counting's second step.
  - Matching level first: the cells' χ(k), the molecules, and what the probes pay. Then train the two C rungs, about
    26 GPU hours.
- **K1 fails toward MB-pol** (|χ(k₂)| below 8 %). Niflheim's read-out answers differently from the local one. Read the
  orders ≥ 1 as adequate at this resolution; H3 is next.
- **K5 and K6 hold.** The order-0 coefficient is local at this cutoff, and arm C's joint matching is consistent.
  - L's per-molecule calibration fits the liquid's molecules three times better than arm C's cell sums.
  - So it is a candidate for arm C's order-0 sector, read in training beside arm D.
- **K6 fails** (L's probes above 1.5× the probe-matched order 0's). The order-0 coefficient is not local at this
  cutoff: risk 8, read cleanly. The joint matching is then a compromise between states, not a consistency.

## 6. Readings beside them, not registered

- the calibration cells against the held-out ones;
- the per-frame spread of the χ differences;
- ε_L against the model's own ε₀;
- L's finite-k response;
- the cells' induced dipole at k = 0 for every read-out;
- the matching's cost (L's Gauss–Newton steps, its time on the GPU).
