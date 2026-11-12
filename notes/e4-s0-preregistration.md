# E4, S0: the polarisation sum rules on MB-pol's trajectories — pre-registration

*11 Nov 2026. S0's bands were registered in the project's note `path-integral-ir-sector.md` §4.1. That note was written
on 10 Nov 2026 at 16:58 CET, after Vadic's question of 10 Nov and before any S0 reading. This note copies those bands
into the repository, with the note's §4.5 readings of them. It adds §3, the reader's choices, fixed here before any S0
number existed: `scripts/e4_sumrules.py` had run on no MB-pol frame when this was committed. Vadic's go of 11 Nov. The
local master of this note is `notes/` in the repository.*

## 1. What S0 reads

Let P be the Born–Oppenheimer polarisation of a configuration: the site charges and the induced dipoles. Let P_k be its
extensive Fourier component, P_L its part along k̂ and P_T the part normal to k. For classical nuclei in canonical
equilibrium,

    L(k)  =  (4πβ/V) ⟨|P_L,k|²⟩      =  1/ε_∞(k) − 1/ε_s(k)   →   1/ε_∞ − 1/ε_s ,
    T(k)  =  (4πβ/V) ⟨|ê·P_T,k|²⟩    =  ε_s,T(k) − ε_∞,T(k)   →   ε_s − ε_∞     (each transverse direction ê) .

- **The longitudinal channel** is read from the charge density, P_L = iρ_k/|k|, with
  ρ_k = Σ_a q_a e^{−ik·r_a} − i Σ_s (k·μ_s) e^{−ik·r_s}. This needs no molecular convention.
- **Units.** In E4's units L = 4π × 557.0 × ⟨|P_L|²⟩/V at 300 K, with P in e Å and V in Å³.
- **The regression.** For a candidate polarisation Y, MB-pol's P is regressed on Y channel by channel:
  b_L = Re⟨P_L Y_L*⟩/⟨|Y_L|²⟩, b_T the same over both transverse directions, and r = b_T/b_L.
  - A local, field-blind source gives r → ε_∞(k) (the lemma of the path-integral note, §5).
  - A model that carries the electronic response gives r → 1.
- **The expected value.** MB-pol's ε_∞ is 1.6885, and job 3 fitted χ(k) = χ(0⁺)(1 + c₂k²) with c₂ = 0.274 Å². With
  these, L(0⁺) = 0.572–0.582 for ε_s = 50–100.

## 2. The bands

| | reading | band | prediction |
|---|---|---|---|
| **S0.1** | MB-pol's L(k → 0): the intercept A of L = A + Bk², fitted over the shells with k ≤ 0.46 Å⁻¹ of the 128–1024 cells | 0.48–0.68 | 0.572–0.582 (ε_s 50–100) |
| **S0.2** | MB-pol's T/L on each of those shells | ≥ 20 | ≈ ε_∞(k) ε_s(k), of order 100 |
| **S0.3** | r for the dagger's sources, Y = P_perm (MB-pol's pinned site charges, no response), on the first shells of 512 and 1024 | 1.45–1.95 | ε_∞(k) = 1.709, 1.701 |
| **S0.4** | r for the frozen form's own polarisation (arm C's read-out), on the first shells of 128 and 256 | 0.90–1.10 | 1 |

**Read beside them, not registered:**

- MB-pol's static longitudinal function, 1/ε_s(k) = 1/ε_∞(k) − L(k), on every shell, and so where overscreening sets
  in;
- b_T for P_perm: the liquid's dipole enhancement;
- ε_s by the tin-foil formula ε_s = ε_∞ + 4πβ⟨M²⟩/3V, and from T(k);
- the 64 cell's r at k₁.

## 3. The reader's choices, registered here before any reading

- **The frames.** Every frame of `data/e4`'s sets of a size, with train, valid and test together. That is 240, 120,
  60, 60 and 52 frames of 64–1024 molecules: four trajectories each, frames 0.5 ps apart.
- **MB-pol's polarisation.** MBX's site charges and its converged induced dipoles at zero external field, with the
  truth's settings (`sseft.mbpol.DEFAULTS`), O, H, H, M per molecule.
  - The note asks for the dynamics' settings to be checked on one frame. Each size's first frame is solved again with
    them (`scripts/e4_data.py` DYNAMICS), and the induced dipoles' relative rms difference is printed. It is read, not
    banded.
- **The temperature** is 300 K, the thermostat's, so βe²/Å = 557.0.
- **The shells** are (100), (110) and (111) of every cell, one of each ±k pair: 3, 6 and 4 vectors.
  - S0.1's fit uses the shells with k ≤ 0.46 Å⁻¹ in the 128–1024 cells. There are nine of them: (100) at 128; (100)
    and (110) at 256; all three at 512 and 1024.
  - The fit is weighted by the inverse squares of the shells' errors, and its error is the fit's.
- **The transverse channel's convention.** Each molecule's total dipole sits at its centre of mass: the site charges
  about the centre of mass plus the induced dipoles, with masses 15.999 and 1.008.
- **The errors** are jackknife over blocks of 4 consecutive frames (2 ps) within a trajectory, for every quantity,
  ratios included.
- **P_perm** is MB-pol's site charges at its sites, with no induced dipoles.
- **The frozen form's own polarisation** (S0.4) is arm C's.
  - The rung is M_16C2†ᵐ_us at the 2.0 Å mutual edge (`configs/e4_job2c.yaml`), with its read-out from job 2c's cache
    (stage 2).
  - It is the pinned site charges, and at every atom the induced dipole −∂E/∂E_ext,i at zero external field. That is
    the order-0 sector's μ_ind together with the orders ≥ 1's field-gradient terms.
  - Its order-0 μ_ind alone is read beside, and so is D′'s polarisation if its file is there.
- **1/ε_∞(k)** in the readings beside is job 3's fit above.

## 4. What would change the reading

These are the path-integral note's §4.5.

- **S0.1 fails high (above 0.68), or S0.2 fails.** Three checks come before anything physical:
  - the frames' temperature;
  - the induced dipoles: present in ρ_k, converged at zero field, and with the sign of §1. Leaving them out raises L to
    about (ε_∞/a)² L ≈ 0.8, with a ≈ 1.4 the liquid's dipole enhancement;
  - the 557.0 unit factor.

  If all three pass, MB-pol's 1/ε_s(k) is strongly negative already at 0.2–0.46 Å⁻¹. That would be overscreening at the
  long-wavelength end: a finding about the truth.
- **S0.1 fails low (below 0.48).** Look first at the normalisation: β, V and the division by k². At k → 0 the physics
  lets L fall below 1/ε_∞ only by 1/ε_s.
- **S0.3 reads near 1.** MB-pol's zero-field polarisation would then not carry the electronic screening that the
  clamped responses of jobs 2–3 measure. The two data sets would disagree on ε_∞, which the matching rests on, so stop
  and resolve it.
- **S0.4 departs from 1.** The model's solve, or the reader, misses the k ≠ 0 part of the dipole tensor.

## 5. The run

1. **`run_e4_s0_mbpol.sbatch`.** Five CPU tasks, one per size, MBX only, into `results/e4_s0/mbpol_n<n>.npz`. MBX's
   zero-field solve takes seconds a frame, so each task should take well under an hour.
2. **`run_e4_s0_model.sbatch`.** One GPU task: arm C's dipoles on the 64–256 sets, into
   `results/e4_s0/model_C_n<n>.npz`. It takes minutes.
3. **The read**, on the login node with NumPy only:

   ```sh
   python scripts/e4_sumrules.py read
   ```

   It prints S0.1–S0.4 and the readings beside them, and writes `results/e4_s0/s0.json`.

The path-integral note's check script, `path_integral_check.py` (its §6), was left in the 10 Nov session's outputs and
is not in the repository yet.

## 6. S0.5 and the nuclear EFT's readings: registered 12 Nov, 14:48 CET (this commit's time), after S0.1–S0.3's read and before any number below

*This section was written after `e4-s0-results.md` §1 was committed, and before the reader computed anything in it.
S0.4 has not been read. The theory is the project's `nuclear-eft-ir-sector.md` (11 Nov): its field Π, its protected
Coulomb term and its §8.2 readings. The derivation below is this note's, at the EFT's tree level. Vadic's go of
12 Nov.*

**The account of S0.3.**

- **Where the lemma is exact.** The lemma's r = ε_∞ is exact for a source proportional to Π, the short-circuit
  polarisation: P_L = Π_L/ε_∞ and P_T = Π_T (the path-integral note's (9)). Π includes the dipoles that the sources
  induce through local fields. P_perm, the sources alone, is only part of it.
- **The split.** Write Π = A P_perm + η. A and η are defined in the ensemble without the long-range Coulomb term,
  where η is uncorrelated with P_perm and its statistics are the same in every direction.
- **What the Coulomb term does.** The nuclear EFT's Coulomb term, (2π/ε_∞V)|Π_L|², acts on all of Π_L, and only in
  the longitudinal channel.
  - Transverse: there is no such term. So b_T = A, and the regression's residual is η itself.
  - Longitudinal: at Gaussian level the term couples η_L to P_perm,L. Let v_η = (4πβ/V)⟨|η|²⟩ per component, in L's
    units. Then the slope is b_L = A/(ε_∞ + v_η).
- **The prediction.** Hence

      r  =  ε_∞  +  v_η ,        v_η  =  (4πβ/V) ⟨|P_T − b_T P_perm,T|²⟩  per transverse direction .          (S0.5)

  - The left side is S0.3's reading.
  - The right side is a residual of the transverse regression, which the reader has not computed.
  - Nothing is fitted.
- **Why the excess persists at small k.** This accounts for the shape the results' §1 describes: r/ε_∞(k) stays near
  1.15 down to the smallest shells. A finite-ka effect would make it fall toward 1 there instead.

**The derivation, in L's units.**

- Let x = (4πβ/V)^{1/2} Π_L. The Coulomb weight is exp(−x²/2ε_∞).
- Take P_perm,L and η_L to be Gaussian, with variances v_Y and v_η, and uncorrelated without the Coulomb term.
- With it, their precision matrix is diag(1/v_Y, 1/v_η) + (1/ε_∞)(A, 1)ᵀ(A, 1).
- So the regression of η_L on P_perm,L is −A v_η/(ε_∞ + v_η), and that of Π_L is Aε_∞/(ε_∞ + v_η). P_L = Π_L/ε_∞ then
  gives b_L.
- The transverse precision matrix has no second term.
- A Monte Carlo of this Gaussian model reproduces r = ε_∞ + v_η to 0.1 % and the transverse residual to 0.3 %
  (`scratch/e4_s05_gaussian_check.py`), for ε_∞ 1.69–2.0, A 1.2–1.5 and v_η 0.05–0.8.

**S0.5, registered.** It is read on S0.3's shells, the first shells of 512 and 1024:

| | reading | band | prediction |
|---|---|---|---|
| **S0.5** | ρ = (r − ε_∞(k)) / v_η, with r and v_η for P_perm, and ε_∞(k) job 3's as in S0.3 | 0.7–1.4 at both | 1 |

The band is about ±2σ of the expected errors. r − ε_∞(k) carries 12–15 %, and v_η should carry well under 10 %.

**What would change the reading.**

- **ρ in the band at both shells.** S0.3's excess is the local-field residual, screened in the longitudinal channel by
  the Coulomb term the EFT protects. The lemma stands as stated, for sources proportional to Π. The dagger's sources
  are not such a source.
- **ρ below 0.7.** The residual is larger than the excess. Then η is partly correlated with P_perm without the Coulomb
  term, or its longitudinal statistics differ from its transverse ones. The Gaussian account is then incomplete.
- **ρ above 1.4.** The residual is too small to carry the excess. Then the enhancement A itself differs between the
  channels, and the results' mean-field reading becomes the candidate.

**An amendment to S3, before it runs.**

- S3 (the path-integral note §4.4) registers m = (r_LES − 1)/(r_dagger − 1), with r_dagger S0.3's reading at the same
  shell, and m = 1 as the lemma's field-blind limit.
- S0.3 read 1.98–1.99. A latent source at the lemma's limit is proportional to Π, so by (9) it reads r = ε_∞(k),
  whatever S0.5 says. Its m would then be about 0.7.
- The denominator is therefore ε_∞(k) − 1. m = 1 is a source proportional to Π, and m = 0 is a model that carries the
  response. r_dagger is read beside it.

**Read beside them, not registered** (the nuclear EFT note's §8.2):

- **ρ and v_η on every shell**, for P_perm and for every model file. For a model, v_η is the part of MB-pol's
  transverse polarisation that the model's misses.
- **The proper stiffnesses on every shell.**
  - 1/A_T(k) = 4π/T(k).
  - 1/A_L(k) = 4πx/(1 − ε_∞(k)x), with x = 1/ε_∞(k) − L(k). It passes through zero where 1/ε_s,L(k) does.
  - Each channel is fitted as a + bk² over S0.1's nine shells (k ≤ 0.46 Å⁻¹, 128–1024 molecules).
  - The two intercepts are read against each other, and against 4π/(ε_s − ε_∞) from each cell's own dipole.
  - b_L, b_T, and the overscreening zero k_os = (−a_L/b_L)^{1/2}.
- **The number–number and number–charge structure factors on every shell.** The density is the molecules' centres of
  mass, and ρ_k is (8)'s.
  - S_nn(k) = ⟨|n_k|²⟩/N. Its small-k level is n k_BT κ_T.
  - S_nZ(k)/k² = Re⟨n_k ρ_k*⟩/Nk². The EFT makes it constant at small k, with λ in its coefficient.
