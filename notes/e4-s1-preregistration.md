# E4, S1 and S2: the model's own liquid in MD — pre-registration

*12 Nov 2026, 15:18 CET (this commit's time). The bands were registered in the project's `path-integral-ir-sector.md`
§4.2–4.3, written on 10 Nov 2026 at 16:58 CET, before the harness had an MD driver. This note copies them. It fixes the
runs' and the reader's choices before the driver exists, so before any trajectory exists, and it lists the readings
beside them. The model is D′'s frozen form, E4's model by job 5's model rule (`e4-job5-results.md` §2). The direction
is Vadic's go of 12 Nov: close S0, then S1 and S2 as the main line. The local master of this note is `notes/` in the
repository.*

## 1. What S1 reads

S0 read the sum rules on MB-pol's own trajectories and found the longitudinal channel where they put it
(`e4-s0-results.md` §1). S1 asks the same of the model's own liquid. Run D′'s frozen form as the Hamiltonian of rigid
water at 300 K, and compare its L(k) with MB-pol's on runs of the same sizes and lengths.

- **The electronic part of the difference is known before the run.**
  - D′'s χ(k₁) is +1.32 % against MB-pol's. That moves 1/ε_∞(k₁) by −0.006 on the 64 cell's first shell.
  - Interpolating D′'s χ error in k² between k = 0 (+0.07 %), k₁ and k₂ (−0.38 %) gives −0.004 and −0.005 on the
    128 cell's first two shells.
- **Anything beyond that is the model's orientational structure at long wavelength.** Nothing in E4 has tested that
  yet. Every reading so far was made on MB-pol's frames.
- **S2 is the control.** A field-blind long-range sector — the dagger, whose sources are MB-pol's pinned charges with no
  response — has no electronic screening of its own. Its own L therefore obeys the path-integral note's §3.3 second row,
  L(0⁺) = 1 − 1/ε_s′, close to 1.

## 2. The bands

| | reading | band | prediction |
|---|---|---|---|
| **S1.1** | \|L_D′(k) − L_MB-pol(k)\| on the 64 cell's (100) shell and the 128 cell's (100) and (110) | ≤ 0.02 + 2σ at each, with σ the two runs' block errors combined | the electronic part: −0.006 (64, (100)); −0.004, −0.005 (128, (100), (110)) |
| **S2.1** | the dagger's own L(k₁) on the 64 cell | ≥ 0.90 | 1 − 1/ε_s′(k₁): about 1, or above it if the dagger's liquid overscreens at k₁ as MB-pol's does |

- **S1.1's MB-pol side** is MB-pol's S1 runs, not S0's frames. For reference, S0 read L = 0.730 ± 0.022 on the 64
  cell's (100) shell, and 0.689 ± 0.038 and 0.727 ± 0.029 on the 128 cell's first two.
- **The path-integral note's "near 0.56"** for the frozen form's own L at k₁ came before S0. S0 found MB-pol
  overscreening already at that shell (1/ε_s(k₁) = −0.17), so 0.73 is where both should sit, and the gap S2 shows is
  smaller than that note's 0.3–0.4.

## 3. The runs, fixed here

**The integrator** is `sseft/rigidmd.py`, as for E4's data:

- velocity Verlet with SHAKE/RATTLE for rigid monomers at the harness's geometry;
- 2 fs steps, the Bussi thermostat at 300 K with τ = 100 fs, and the centre-of-mass velocity removed every 100 steps;
- 10 ps of equilibration, which is not read, then 100 ps of sampling.

**The polarisation** is saved every 25 steps (0.05 ps), with the positions, the site positions and the site charges.

| engine | Hamiltonian | polarisation saved | sizes | trajectories |
|---|---|---|---|---|
| D′ | D′'s frozen form, M_16C2†ᵐ_us at the 2.0 Å mutual edge (`configs/e4_job5.yaml`), trained seed s for trajectory s | its pinned site charges, and at every atom −∂E/∂E_ext,i at zero external field (S0's model step, the configured GMRES solve) | 64, 128 | 4 (s = 0–3) |
| MB-pol | MBX with the dynamics' settings (`scripts/e4_data.py` DYNAMICS), as E4's data | MBX's site charges and induced dipoles at zero field, with the same settings | 64, 128 | 4 |
| the dagger | job 1's M_16†ᵐ (`configs/e4_water.yaml`, M_16pm), trained seed s | its pinned site charges (it has no response) | 64 | 2 (s = 0, 1) |

- **The start.** Trajectory s of every engine starts from the same MB-pol frame of its size: the first frame of the
  s-th quarter of `data/e4`'s frames of that size, with train, valid and test together. Each engine draws its own
  velocities and thermostat noise from the seed.
- **Continuation.** The runs are checkpointed and continued over 4-hour jobs. A continued run reproduces the
  uninterrupted one exactly, because the checkpoint carries the thermostat's random state.
- **Instability.** A trajectory stops if its energy or forces become non-finite. It is read on what it has, and its
  stop is reported.
- **Too short.** S1.1 is read on an engine and size only if at least 2 of its trajectories reach 50 ps of sampling.
  Otherwise it is reported as not read.
- **The timing comes first** (`scripts/e4_md.py time`). If a 128-molecule trajectory of D′ needs more than three 4-hour
  jobs, the 128 runs are kept to their 100 ps over more jobs, not shortened.

## 4. The reader, fixed here

- **The estimators are S0's** (`e4-s0-preregistration.md` §3):
  - L(k) from the charge density of the site charges and point dipoles;
  - T(k) from the molecules' dipoles at their centres of mass;
  - on the (100), (110) and (111) shells of each cell, with βe²/Å = 557.0 at 300 K.
- **The frames** are every saved frame of the 100 ps of sampling, all of an engine's trajectories of a size pooled.
- **The errors** are jackknife over blocks of 10 ps (200 frames) within a trajectory. That is 10 blocks per trajectory,
  longer than τ_D ≈ 8 ps.
- **S1.1's σ** is the two engines' errors in quadrature.

## 5. Read beside them, not registered

- T(k), T/L, and ε_s by the tin-foil formula, for each engine and size. T needs nanoseconds, so 400 ps reads it
  roughly.
- The proper stiffnesses 1/A_T and 1/A_L on every shell (`e4-s0-preregistration.md` §6).
- **The longitudinal relaxation time.** τ_L(k) is the integral of P_L(k, t)'s normalised autocorrelation to its first
  zero, for each engine. This is the nuclear EFT note's fast dynamic reading. It converges as fast as L does, and
  Fröhlich puts it near 0.5 ps.
- **Stability**, for each trajectory:
  - the temperature;
  - the conserved energy's drift, in kcal/mol/ps per molecule;
  - the largest constraint error;
  - any stop.
- **The dagger's own T(k) and T/L.** For a field-blind sector T/L → ε_s′, not ε_∞ε_s.

## 6. What would change the reading

These are the path-integral note's §4.5, with S0's results.

- **S1.1 holds.** The model's own liquid carries MB-pol's long-wavelength polarisation structure: the electronic
  screening from its solve, the nuclear part from its training. This is the programme's central claim, read in MD for
  the first time.
- **S1.1 fails while S0 holds.** The model's long-wavelength orientational structure differs from MB-pol's. Its size
  and k-dependence, set against T(k), say whether ε_s is the cause.
- **S2.1 fails.** The dagger's own polarisation would then carry screening it has no mechanism for. Look first at the
  reader, which S0 has tested on MB-pol and on the frozen form, and at the dagger's sources.
- **A model trajectory goes unstable.** That is a finding about the frozen form as a simulator, read before anything
  else. MB-pol's runs are read on their own.

## 7. The run

1. **`run_e4_s1_time.sbatch`.** One GPU task: seconds a step for D′ at 64 and 128 molecules and for the dagger at 64.
2. **`run_e4_s1_model.sbatch`.** GPU tasks, one per trajectory (D′ at 64 and 128 with four seeds, and the dagger at 64
   with two), each continued from its checkpoint until it is done.
3. **`run_e4_s1_mbpol.sbatch`.** Eight CPU tasks, MB-pol at 64 and 128, four seeds each.
4. **The read**, on the login node with NumPy only:

   ```sh
   python scripts/e4_md.py read
   ```
