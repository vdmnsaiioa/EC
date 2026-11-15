# E4, S0: the polarisation sum rules on MB-pol's trajectories — results

*Read 12 and 15 Nov 2026 against `e4-s0-preregistration.md`. Its bands and the reader's choices were committed at
22:31 CET on 11 Nov, before the reader had run on any MB-pol frame. S0.5 was committed at 14:48 CET on 12 Nov (its §6),
before any of its numbers. §1 is MB-pol's own readings, read 12 Nov. The rest was read 15 Nov:*

- *§2: S0.4, the frozen form's polarisation;*
- *§3: S0.5, the nuclear EFT's account of S0.3, with the EFT's readings beside it;*
- *§4: the tie of induction and dispersion;*
- *§5: what they change.*

*S0 is closed: S0.1, S0.2, S0.4 and S0.5 hold. S0.3 failed high, and S0.5 accounts for its excess with nothing
fitted.*

## 1. MB-pol's polarisation

**The run.** `run_e4_s0_mbpol.sbatch` ran at f8788ff as five CPU tasks of 1–5 minutes each. The read ran on the login
node.

- Every frame of `data/e4` was read: 240, 120, 60, 60 and 52 frames at 64–1024 molecules.
- With the dynamics' settings, each size's first frame has induced dipoles within 0.001–0.002 % of the truth's.

| | reading | band | prediction | Niflheim |
|---|---|---|---|---|
| **S0.1** | L(k → 0), the intercept over the nine shells with k ≤ 0.46 Å⁻¹ of 128–1024 | 0.48–0.68 | 0.572–0.582 | **0.610 ± 0.030** ✓ (B = +0.31 Å², χ²/dof 0.47) |
| **S0.2** | T/L on those nine shells | ≥ 20 | ≈ ε_∞ε_s, of order 100 | **86–114** ✓ |
| **S0.3** | r for P_perm on the first shells of 512 and 1024 | 1.45–1.95 | 1.709, 1.701 | **1.979 ± 0.032, 1.992 ± 0.045** ✗ (high) |
| **S0.4** | r for arm C's polarisation on the first shells of 128 and 256 | 0.90–1.10 | 1 | §2 |

**S0.1 and S0.2 hold.** MB-pol's Born–Oppenheimer polarisation obeys the nuclear sum rules on E4's own trajectories.

- **The intercept** is 0.610 ± 0.030, 0.9σ above the predicted 0.572–0.582. The smallest shells read L = 0.612 ± 0.029
  (512, k = 0.25 Å⁻¹) and 0.622 ± 0.047 (1024, 0.20 Å⁻¹).
- **T/L** is 96–114 at k ≤ 0.36 Å⁻¹. Since T/L = ε_∞ε_s at small k, this gives ε_s ≈ 57–68.
- **ε_∞ from L.** Taking ε_s ≈ 60, the intercept gives 1/ε_∞ = 0.627 ± 0.030, so ε_∞ = 1.60 ± 0.08. The clamped responses'
  1.6885 (jobs 2–3) is 1.2σ from that, and 2.0 is 4σ from it. So the sum rule confirms the ε_∞ that the matching rests
  on, independently of S0.3.

**S0.3 fails high.** For MB-pol's own permanent sources, r is 1.98–1.99 at the first shells of 512 and 1024. The
registered band was 1.45–1.95 and the lemma's prediction ε_∞(k) = 1.70. The failure lies in b_L, not in b_T:

- **b_T is 1.454–1.461 on all fifteen shells, at every size.** This is the liquid's dipole enhancement, the note's
  a ≈ 1.4: MB-pol's molecular dipoles are 1.455 times their permanent part, in the regression's sense, at every k
  read.
- **b_L is 0.731–0.761 at k ≤ 0.36 Å⁻¹**, and 0.731–0.736 on the five shells of 512 and 1024 there. If the same
  enhancement were screened by ε_∞, it would be b_T/ε_∞(k) = 0.84–0.86. Read as b_L ε_∞(k), the longitudinal channel's
  enhancement rises with k:
  - 1.24–1.31 at k ≤ 0.36 Å⁻¹;
  - 1.30–1.38 at 0.40–0.57;
  - 1.49–1.50 at 0.69–0.71;
  - 1.65 at 0.88.
- **So r falls with k while ε_∞(k) rises.** r is 1.91–2.00 at k ≤ 0.36 Å⁻¹, 1.84–1.96 at 0.40–0.57 and 1.74–1.82 at
  0.69–0.88. Over the same range ε_∞(k) goes from 1.70 to 1.97. The two cross near 0.65 Å⁻¹, where b_L ε_∞(k) reaches
  b_T.

**What the failure is, and is not.**

- **It is not the case the registration feared.** §4 of the pre-registration asked to stop if r read near 1, which
  would mean MB-pol's zero-field polarisation lacked the electronic screening. Instead, b_L < b_T/ε_∞: the longitudinal
  channel is screened more than ε_∞ alone accounts for, and S0.1 confirms ε_∞ itself.
- **What it does say.** The lemma's r = ε_∞ needs a source whose companion polarisation is the same multiple of it in
  both channels, up to a residual uncorrelated with it in each. MB-pol's own permanent sources are not such a source.
  - The transverse multiple is constant at every k and size.
  - The longitudinal one is about 15 % smaller at the smallest k, and grows with k.
- **Two candidate readings, neither tested:**
  - *The screening of the residual.* MB-pol's induced dipoles follow each molecule's local field, and the source's
    k-component does not fix that field. In the longitudinal channel only, the long-range energy couples this residual
    to the source. That would anti-correlate the two and lower b_L most where the screening is strongest, at small k.
  - *A mean-field longitudinal channel.* On the five shells of 512 and 1024 below 0.36 Å⁻¹, b_L is within about 1 % of
    Clausius–Mossotti's longitudinal factor (ε_∞ + 2)/3ε_∞ = 0.728. b_T exceeds that model's transverse factor,
    (ε_∞ + 2)/3 = 1.23, by 18 %. In the Clausius–Mossotti model the two factors' ratio is exactly ε_∞.
- **For the architecture**, the lemma's use survives with a new scale. Field-blind sources regress onto the truth at
  r ≈ 2.0 at these k, not 1.7. A model that carries the response should read 1. The test separates the two more
  sharply than registered, but r cannot be read on the lemma's scale between 1 and ε_∞. The path-integral note's §5
  should state the lemma's assumption and this scale.

**Readings beside, not registered.**

- **MB-pol's static longitudinal function.** 1/ε_s(k) = 1/ε_∞(k) − L(k) is consistent with 1/ε_s ≈ 0.015 at
  k ≤ 0.25 Å⁻¹: −0.027 ± 0.029 (512) and −0.034 ± 0.047 (1024). It turns negative from about 0.3 Å⁻¹:
  - −0.03 to −0.12 at 0.28–0.45 (−0.089 ± 0.034 at 0.28, in the 1024 cell);
  - −0.17 at 0.51–0.57;
  - −0.26 to −0.29 at 0.69–0.71;
  - −0.45 ± 0.03 at 0.88.

  So MB-pol overscreens from about 0.3 Å⁻¹.
- **ε_s by the tin-foil formula** is 49 ± 5 to 82 ± 12 across the sizes, with only 52–240 frames. From T at the
  smallest shells it is 62–71.
- **The 64 cell's first shell** (k = 0.51 Å⁻¹) reads r = 1.87 ± 0.02 for P_perm.
- **Arm C at 64 molecules.** The model step wrote this cell before it stopped (§2). S0.4 registers 128 and 256, so this
  is a reading beside, seen before S0.4's sizes were read.
  - Arm C: r = 0.99–1.00 on all three shells, with b_L = 1.006–1.009 and b_T = 1.001–1.003.
  - Its order-0 sector alone: the same to 0.01.

## 2. S0.4: the frozen form's polarisation

**The runs.**

- **The first run** of `run_e4_s0_model.sbatch` wrote arm C's 64-molecule file and stopped at 128 molecules.
  - *The cause.* The model step checked that the read-out was matched for this rung by comparing the rung's whole
    repr. That repr includes the reciprocal grid, `ewald_n_max`, which follows the cell: 22, 27 and 34 at 64, 128 and
    256 molecules. The matching ran in the 64-molecule cell, so every larger cell was refused.
  - *The fix.* The check leaves the grid out (`tests/test_sumrules.py::test_model_step_reads_cells_other_than_the_
    matching_one_with_the_configured_solve`). The step also now evaluates both dipole sets in one pass, and solves the
    mutual response as the config's model does: GMRES to 1e-10, where it had used the dense solve, whose cost grows as
    N³. On the test cells the two agree to 1e-7. None of this changes what S0.4 reads: the same read-out, the same
    frames, the same −∂E/∂E_ext,i.
- **The second run** wrote arm C's and D′'s files at 64, 128 and 256 molecules.
- **The read** on 15 Nov printed every band, then stopped in the tie reading (§4), before writing the json. The
  reading stacked each cell size's polarisabilities, which have 192, 384 and 768 atoms per frame. It now pools the
  sizes atom by atom (`tests/test_sumrules.py::test_the_tie_reading_pools_cell_sizes_atom_by_atom`). Nothing
  registered moved, and the rerun wrote `results/e4_s0/s0.json`.

| | reading | band | prediction | Niflheim |
|---|---|---|---|---|
| **S0.4** | r for arm C's polarisation on the first shells of 128 and 256 | 0.90–1.10 | 1 | **0.995 ± 0.002, 0.994 ± 0.002** ✓ |

**S0.4 holds.** The frozen form's polarisation — the pinned charges, and −∂E/∂E_ext,i at every atom — is MB-pol's in
both channels on MB-pol's own frames.

- **On every shell of 64–256 molecules**, r = 0.99–1.00 for arm C and for D′. Their order-0 sector alone reads the same
  to 0.01. D′'s order 0 is arm C's (job 5 re-solved only the orders ≥ 1), so the two order-0 rows are identical.
- **What separates r from 1 is half a percent in the longitudinal channel.** b_L is 1.004–1.012 and b_T 1.000–1.003,
  for both models.
  - Arm C's and D′'s clamped χ(k₁) errors have opposite signs: −1.84 % and +1.32 %.
  - Through P_L = Π_L/ε_∞(k), those errors would put b_L near 0.99 and 1.01 on the 64 cell's first shell. Both read
    1.006–1.007.
  - So S0.4 does not read the finite-k response to an external wave. It reads the response to the liquid's own fields,
    and the order 0 carries that.
- **The transverse residual** of MB-pol's P on the model's, v_η, is:
  - 0.002–0.004 for the two models;
  - 0.005–0.007 for their order 0 alone;
  - 0.30–0.335 for the permanent sources.

  The frozen form leaves about 1 % of the transverse polarisation that the sources miss.
- **The scale.** On the same frames, field-blind sources regress at r ≈ 2.0 (S0.3) and the frozen form at 1.00.

## 3. S0.5: the nuclear EFT's account of S0.3

| | reading | band | prediction | Niflheim |
|---|---|---|---|---|
| **S0.5** | ρ = (r − ε_∞(k))/v_η for P_perm, on the first shells of 512 and 1024 | 0.7–1.4 at both | 1 | **0.910 ± 0.118, 0.945 ± 0.162** ✓ |

- The parts are r − ε_∞(k) = 0.270 ± 0.032 and 0.291 ± 0.045, and v_η = 0.297 ± 0.018 and 0.308 ± 0.017.
- The errors are the ones the registration expected: 12 and 15 % on the excess, 6 % on v_η.

**S0.5 holds.** The pre-registration's branch for ρ in the band:

- S0.3's excess is the local-field residual, screened in the longitudinal channel by the Coulomb term the EFT protects.
- The lemma stands as stated, for sources proportional to Π. The dagger's sources are not such a source.
- Nothing was fitted. v_η is a residual of the transverse regression, and ε_∞(k) is job 3's.
- Of §1's two candidate readings, the first is this one, now with its number. The mean-field reading is not needed.

**Beside it, not registered: ρ on every shell, for P_perm.**

- **0.84–0.95** on the five shells of 512 and 1024 with k ≤ 0.36 Å⁻¹.
- **0.27–0.64** at 0.32–0.57 Å⁻¹: the smaller cells' shells, and 512's (111) at 0.44.
- **Negative above 0.65 Å⁻¹** (−0.11 to −0.74), where r falls below ε_∞(k) (§1).
- v_η is 0.30–0.335 on every shell at every size: the permanent sources miss the same transverse share at every k.

So the account holds at the smallest k read and fails from about 0.4 Å⁻¹.

For the models the ρ column (−145 to −380) is not a reading. Their r ≈ 1 lies below ε_∞(k), and S0.5's account is
about field-blind sources.

**A reading added after the numbers (mine; not registered).**

- **The idea.** §6's derivation assumed that, without the Coulomb term, η and P_perm have the same statistics in both
  channels. The EFT's own gradient terms break that at O(k²): the proper stiffness is 1/A_L in one channel and 1/A_T in
  the other.
- **The prediction.** If that difference acts on Π as the Coulomb term does, the same algebra gives, shell by shell
  from the printed stiffnesses with nothing fitted,

      ρ  =  1 + ε_∞(k) (1/A_L − 1/A_T) / 4π .

- **Against what was read:**

  | shells | predicted | read |
  |---|---|---|
  | the five of 512 and 1024 with k ≤ 0.36 Å⁻¹ | 0.84–0.93 | 0.84–0.95 |
  | 256's (100), 0.32 Å⁻¹ | 0.88 | 0.58 ± 0.14 |
  | 0.40–0.57 Å⁻¹ | 0.73–0.83 | 0.27–0.64 |
  | 0.69–0.88 Å⁻¹ | 0.48–0.63 | −0.74 to −0.11 |

- **What it says.** The stiffnesses' anisotropy is the right size for the dip below 1 where the account holds. It
  does not rescue the account beyond 0.4 Å⁻¹. There the split Π = A P_perm + η itself depends on the channel: b_L ε_∞(k)
  rises with k (§1), while b_T does not.
- **The caveat.** The prediction and ρ share L, so their errors are correlated. This is a reading, not a test.

**Beside it, not registered: the proper stiffnesses** (the nuclear EFT note's (7)). The fits run over S0.1's nine
shells.

| channel | a | b (Å²) | χ²/dof |
|---|---|---|---|
| 1/A_T = 4π/T | 0.188 ± 0.010 | +0.06 ± 0.07 | 1.38 |
| 1/A_L | −0.31 ± 0.32 | −4.0 ± 2.2 | 0.48 |
| 1/A_L with a fixed at 0.188 (mine, after the numbers) | 0.188 | −7.2 ± 0.8 | 0.72 |

- **The transverse intercept gives ε_s = ε_∞ + 4π/a = 68.5 ± 3.6.**
  - The cells' own dipoles give a = 0.156–0.264, weighted 0.184 ± 0.011 (χ²/dof 2.7). The 512 cell's 0.264 ± 0.028 is
    2.9σ high; its tin-foil ε_s is 49 ± 5.
  - The two readings of the one intercept agree to 0.3σ, as (7) says.
  - In the EFT variable, T gives ε_s to 5 %, where the tin-foil formula scatters from 49 to 82 between cells.
- **Against the literature.** Reddy et al. (2016) report ε_s = 68.4 for MB-pol at 298 K: classical MD, the cell
  dipole's fluctuations, flexible monomers, 256 molecules. Experiment is 78.5. E4's frames are rigid, so the agreement
  is a consistency check, not a test.
- **b_T is zero within its errors** up to 0.46 Å⁻¹. It is a convention: the EFT note's R2, the molecular centre.
- **The longitudinal intercept is consistent with the common one but does not test it.** It lies 1.5σ below 0.188.
  S0.1's own ±0.03 in L moves it by about ±0.4 (the EFT note's §3.4).
- **The overscreening zero.**
  - 1/A_L is negative on all nine shells, by 2σ or more at 0.28, 0.36, 0.40, 0.44 and 0.45 Å⁻¹.
  - With the transverse intercept imposed, its zero is at k_os = (a/−b_L)^{1/2} = 0.16 Å⁻¹ (0.15–0.17). The free fit's
    slope would put it at 0.22 Å⁻¹.
  - So MB-pol's static ε_L(k) has its pole at 0.16–0.22 Å⁻¹, a wavelength of 29–39 Å, and overscreens beyond it.
  - This refines §1's "from about 0.3 Å⁻¹". That was read on 1/ε_s(k) directly, where the shells below 0.28 Å⁻¹ could
    not tell zero from 1/ε_s ≈ 0.015.
  - Only the 1024 cell's first shell, at 0.20 Å⁻¹, is read near the pole. Every other shell lies beyond it.

**Beside it, not registered: the structure factors.**

- **S_nn.**
  - It is 0.049–0.076, rising with k. S(0) + bk² over the fifteen shells gives S(0) = 0.057 ± 0.002 (χ²/dof 2.0).
  - So κ_T = S(0)/nk_BT = (4.1 ± 0.2)×10⁻⁵ bar⁻¹, with the error scaled by √(χ²/dof).
  - Experiment is 4.5×10⁻⁵ bar⁻¹ at 300 K, and Reddy et al. find flexible MB-pol in agreement with it. E4's rigid
    frames read 9 % lower.
- **S_nZ/k².**
  - It is −0.0087 ± 0.0002 e Å² averaged over the fifteen shells (χ² 19.8 for 14). A + Bk² gives −0.0079 ± 0.0005,
    with slope −0.0018 ± 0.0010 Å².
  - So it is constant to about 10 % from 0.2 to 0.9 Å⁻¹: the EFT's k² law, from per-molecule neutrality.
  - With S(0) = 0.057 and ε_s = 68.5, its coefficient gives λ = (S_nZ/k²)(aε_∞ + 4π)/S(0) ≈ −1.8 e Å² in (6)'s
    Gaussian units.
  - Most of it is each molecule's own charge second moment. Decoupling orientations from positions,
    −(1/6)Σ_a q_a d_a² × S_nn ≈ −0.0089, with Σ_a q_a d_a² ≈ 0.93 e Å² about the centre of mass. That sum is my
    estimate, from the rigid monomer's 1.8636 D and the M-site at γ = 0.4267.

## 4. The tie of induction and dispersion (not registered; `v2-one-field-proposal.md` §5.1)

**The reading** is on arm C's order-0 polarisabilities, over 420 frames of 64, 128 and 256 molecules. D′'s order 0 is
arm C's, so its reading is identical.

| | O | H | molecule |
|---|---|---|---|
| order-0 α(0) = tr C/3, mean ± sd over the liquid (bohr³) | 7.567 ± 0.135 | 1.205 ± 0.038 | 9.977 ± 0.141 |
| the dispersion pins' α(0) (bohr³) | 6.777 | 1.972 | 10.721 |
| MB-pol's long-range C₆, au | 17.22 (OO) | 1.458 (HH) | 43.10 |
| one oscillator on that α(0): ω = 4C₆/3α² (E_h) | 0.401 | 1.339 | 0.577 |
| the implied C₆'s spread at fixed ω (sd/mean of α²) | 3.6 % | 6.3 % | 2.8 % |

The implied cross term C₆_OH is 0.842 of MB-pol's 5.01 au.

**Per molecule, the tie costs little.**

- **ω = 0.577 E_h,** against the gas-phase monomer's 0.65–0.67. That range is C₆ = 45.4 au with α = 9.64 au, or with
  MB-pol's 9.50 bohr³ (the path-integral note's §2.3 (c)).
- **The gap is two known factors and nothing else:**
  - MB-pol's long-range c6 sum to C₆ = 43.1 au per molecule, 5 % below 45.4;
  - the order-0 α in the liquid is 9.98 bohr³ (1.478 Å³), 5 % above MB-pol's gas-phase monomer (1.4075 Å³), and ω
    goes as 1/α².

  Indeed 0.671 × 0.950 × 0.906 = 0.577, and 0.651 × 0.950 × 0.934 = 0.577.
- **The spread at fixed ω is 2.8 % per molecule.**
- **So one ω per molecule** reproduces MB-pol's molecular C₆ in the liquid. So does one ω for every site. A tied model
  would add a 3 % environmental spread that MB-pol's fixed c6 does not have.

**Per site, MB-pol is not one field.**

- **The frequencies.** MB-pol's c6 per site needs ω_O = 0.40 and ω_H = 1.34 E_h on the model's α, a factor 3.3 apart.
  - One oscillator per site then gives the cross term 2(ω_Oω_H)^{1/2}/(ω_O + ω_H) = 0.842 of MB-pol's product rule.
  - It gives the molecular C₆ at 0.927 of MB-pol's.
- **One common ω** keeps the molecular C₆ exact and moves the site pairs instead: OO +44 %, OH −21 %, HH −57 %.
- **The partitions differ.** The share of the molecule's α on O is:
  - 0.76 in the model's induction;
  - 0.69 in MB-pol's bare Thole values (1.310 and 0.294 Å³);
  - 0.63 in its c6 at equal ω (the pins).

**What it means for v2** (the proposal's §5.1 rule).

- **Per molecule,** ω is near the gas-phase value once MB-pol's own C₆ and the liquid's α are accounted for, and the
  spread is 3 %. Tying a molecule's dispersion to its induction costs little against MB-pol.
- **Per site,** an ω per species, as the proposal's §4 designs it, cannot reproduce MB-pol's site c6 on the model's
  partition.
  - Against MB-pol, the tie can be tested with one ω for every site. That keeps the C₆ tail and moves the site
    partition, which is the anisotropic part at intermediate range.
  - The site-resolved tie needs a truth built as one field: §5.2's MB-pol₁.

## 5. What S0 changes

- **The lemma** (the path-integral note's §5) stands for sources proportional to Π. Its (ii) forces such sources on a
  field-blind model whose long-wavelength energetics are right.
  - Sources that are not proportional to Π, like MB-pol's own permanent charges, read r = ε_∞(k) + v_η, with v_η their
    transverse residual.
  - So the scale on which a source is read runs from 1 (it carries the response) to ε_∞(k) + v_η (it is field-blind).
    On E4's frames that is 1.00 against 2.0.
- **S3**, amended on 12 Nov, normalises m by ε_∞(k) − 1. Read beside it, not registered: ρ_LES = (r_LES −
  ε_∞(k))/v_η,LES. S0.5 says a field-blind latent source reads 1 there at small k.
- **The EFT's tree level holds on the truth's frames:**
  - the sum rules;
  - one intercept from both readings of ε_s, and ε_s at MB-pol's published value;
  - the k² law of S_nZ;
  - the account of S0.3 below 0.36 Å⁻¹.

  The account fails from about 0.4 Å⁻¹, where the local-field split stops being the same in both channels.
- **For S1,** the main line:
  - S0.4 shows that D′'s polarisation is MB-pol's to half a percent on the same frames. So S1.1's difference reads D′'s
    liquid structure, not its electronics.
  - MB-pol's own S1 runs should reproduce §1's L on S1.1's shells: 0.730 ± 0.022 (64, (100)), and 0.689 ± 0.038 and
    0.727 ± 0.029 (128, (100) and (110)). That checks the driver against the frames E4 was trained on.
- **For v2,** the tie per molecule is testable on MB-pol, and the tie per site waits for MB-pol₁ (§4).
