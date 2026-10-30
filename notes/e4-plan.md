# E4 — the periodic water box on MB-pol: plan

*27 Oct 2026. Set up after Vadic's decisions of 27 Oct:*

- *E3 is closed (`e3-summary.md`).*
- *MB-pol through MBX is the truth for the boxes.*
- *DFT comes later, at one or two sizes; the code is still to be decided.*

*This note replaces the E4 paragraph of `phase-d-build-plan.md` §4, whose DFT labels move to §8 here. The X3
pre-registration, computed from the truth, comes before any periodic training (§9). The local master of this note is
`notes/` in the repository.*

## 1. What E4 is for

E4 tests the claims that clusters cannot, which are what make the model an EFT rather than "LES plus induction":

- **X3, size extrapolation** (`phase-a-size-theorem-3.md`).
  - *Form I*: the finite-size tail Δ(L) = E(L) − E(∞) of fixed contents in growing cells, in a cubic and a
    tetragonal (L, L, 2L) shape. E(∞) is the open value only where MBX's open and periodic dispersion agree (§2).
  - *Form II*: the bulk per-molecule error at sizes L above the training cells' L₀, with an L₀ sweep.
  - Gate C: if M_G matches the pinned rungs on Form I, the EFT structure buys nothing on these data.
- **X4b, matching in a crystal-like cell.** The clamped-nuclei ε_∞ and the Born charges Z* come from the model's
  response, never trained on. They are compared against the truth's own: MB-pol's induced dipoles in a uniform field,
  through MBX, and DFPT later (§8).
- **X2-C, the band-edge level on real data.** It is read on the long-wavelength modulation response, model against
  model, at two band edges.
- **Theorem 2's tests T4 and T6** (`les-vs-eft-what-is-provable.md` §7). For MACE-LES, ε_∞ = 1 exactly, and there is
  no image polarisation.
- **E3's carry-over**: the read-out's near field in a liquid, where every molecule has close contacts
  (`e3-summary.md` §4).

What E4 will not show: on bulk water, energy and force errors do not separate the classes (`les-vs-eft` §8, caveat 2;
the size note's benchmark corollary). E4's evidence is the observables above, not a leaderboard.

## 2. The truth: MB-pol via MBX v1.4.0

**What MB-pol is** (`sseft/mbpol.py`):

- Electrostatics as in TTM4-F: charges on the H atoms and on an M-site, from the Partridge–Schwenke dipole surface.
- Thole-damped point polarisabilities on O and H, solved self-consistently.
- CCSD(T)-fitted 2-body and 3-body polynomials inside 6.5 and 4.5 Å.
- A damped C₆ dispersion.
- In cells, particle-mesh Ewald for electrostatics and dispersion.

**Why MB-pol.** It is periodic, polarisable like q-AQUA-pol, and cheap. Its long range is a known site model, so E3's
method carries over:

- numbers computed from the truth before training;
- exact daggers;
- probes and scans on demand;
- the truth's own ε_∞ and Z* from its induced dipoles under a uniform field (MBX exposes the field on the sites).

**As used:**

- The interaction energy is MBX's energy with the 1-body polynomial off, so an isolated monomer is zero at any
  geometry (E3's convention).
- Rigid monomers (§3).
- MBX is called through ctypes from `sseft/mbpol.py`; `external/mbx/build.sh` builds it.

**Checked here, 27 Oct.** MBX v1.4.0 was built in the container at −O0 for speed; these are readings of the code, not
of water. The commands were `python sseft/mbpol.py --check` and `scratch/e4_mbx_checks.py`, and `tests/test_mbpol.py`
holds the same checks as tests.

- **MBX's own MB-pol example** (a trimer, open and in a 20 Å box) reproduces MBX's printed energies to 2×10⁻⁹ kcal/mol
  and its gradients to 8×10⁻⁹.
- **The interaction energy**: a lone monomer at any geometry gives 0.
- **Forces** agree with central differences to 1.3×10⁻⁶ relative.
- **The site model.** Charges +0.5777 e on each H and −1.1553 e on the M-site, 0.25 Å from O, with none on O.
  Polarisabilities O 1.310, H 0.294, M 0 Å³. The rigid monomer's dipole is 1.8636 D, and its polarisability through
  the external field is 1.4075 Å³ isotropic (eigenvalues 1.313, 1.414, 1.496).
- **Tin-foil boundary conditions.** One rigid molecule in cubic cells of 20, 30, 40 and 60 Å gives
  E(L) / [−(2π/3)|μ|²/L³] = 0.965, 0.984, 0.991 and 0.996. The remainder is (1 − ratio) L² → 14.8 Å², the quadrupole
  images. Under vacuum conditions the ratio would tend to 0.
- **The supercell identity is exact** when the cell's half-width clears every cutoff. For 32 molecules in a 13.5 Å
  cell against its 2×2×2 replica, every term agrees to 3×10⁻¹¹ kcal/mol.
- **MBX takes the minimum image.**
  - The 64-molecule cell at liquid density (12.43 Å, half-width 6.21 Å) misses 1.6×10⁻² kcal/mol of its 2-body
    against its replica; everything else agrees to 10⁻⁸.
  - The driver therefore evaluates a cell narrower than twice the 6.5 Å reach on its smallest replica that clears it.
    That makes the 64-molecule labels exact: the 2×2×2 and 3×3×3 replicas agree to 4×10⁻⁸ kcal/mol. MD for sampling
    can run on the cell itself.
- **PME convergence** (32 molecules, r_c 6.5 Å). MBX's example settings (α 0.6, grid 2.5 Å⁻¹, order 6) are off by
  3×10⁻³ kcal/mol in the electrostatics. At α 0.7, a grid of 6 Å⁻¹ and order 8 the result is converged to 2×10⁻⁷, and
  the real-space truncation is erfc(4.55).
- **The dispersion's real-space cutoff is part of MBX's periodic Hamiltonian.**
  - Inside r_c, MBX uses MB-pol's damped pair C₆, switched off over r_c − 1 to r_c.
  - Beyond r_c, PME uses per-site c₆ whose O–H product, 69.06 kcal mol⁻¹ Å⁶, is 17 % below the pair value of 83.50.
    O–O and H–H agree.
  - The 64-molecule cell's dispersion is −354.87 kcal/mol at r_c 6.5 Å and −355.65 at 9 Å. The electrostatics are
    unchanged to 10⁻⁶.
  - **E4's truth fixes r_c = 6.5 Å in every cell**: the 2-body reach, with α 0.7, grid 6 Å⁻¹ and order 8 (the
    driver's defaults). Beyond 6.5 Å its dispersion is then the rank-1 c₆ model that a pinned dispersion band
    reproduces.
  - Open structures use the pair C₆ at every distance. Their energies differ from a cell's in the O–H pairs beyond
    5.5 Å, but their non-additive energies do not, because MB-pol's dispersion is pairwise. So Form I reads
    differences between cell sizes, or an extrapolated E(∞), not E(L) − E(open), wherever the contents have pairs
    that far apart.
- **The external field in a cell.** A random 128-molecule packing gives ε_∞ = 1.681. That lies between the first-order
  1 + 4πnα = 1.590 and Clausius–Mossotti's 1.734 with the monomer's α.
- **Timings** (−O0, 2 threads; not representative), energy and forces with E4's settings: 2.1 / 3.8 / 9.0 s at
  128 / 256 / 512 molecules, and 7.5 s for 64 molecules through their 512-molecule replica.

**On Niflheim, 28 Oct** (`run_e4_mbx_checks.sbatch` on s001, 8 threads, MBX at −O2, commit d6ebb08):

- Every check above came out as here, to the printed digit: the optimised build changes no number. The 8 tests pass.
- The build took about 40 minutes.
- Energy and forces with E4's truth take 0.37, 0.67 and 1.45 s for 128, 256 and 512 molecules, and 1.41 s for
  64 molecules through their 512-molecule replica.
- MBX's example PME settings are 4–5 times cheaper (0.09, 0.17 and 0.33 s). Those are good enough for sampling, where
  the accuracy of the energy does not matter.

## 3. The data

**Sampling:**

- MB-pol MD at 300 K and 0.997 g cm⁻³ through MBX (NVT), with the monomers held rigid at the harness's geometry
  (r_OH 0.9572 Å, 104.52°). The integrator is our own (`sseft/rigidmd.py`): velocity Verlet with SHAKE/RATTLE and the
  Bussi thermostat, which samples the canonical ensemble. *Changed 28 Oct* from flexible MD at 0.5 fs followed by
  rigidifying every frame (Kabsch): rigid MD takes 2 fs steps, so it is four times cheaper, and it samples the same
  rigid-monomer model the labels describe.
- MBX's example PME settings for the dynamics; E4's truth for the labels. The 64-molecule cell runs on itself, under
  MBX's minimum image; at the validation frames its forces agree with the truth's to 0.004 kcal/mol/Å rms.
- The thermostat's time constant is 100 fs. At 500 fs, one unlucky noise stream kept a 64-molecule run 20 K hot for
  picoseconds, with its energy conserved (RESULTS.md, v0.0.19). The conserved energy, K + E_pot plus the heat the
  thermostat took out, is logged; its drift is each trajectory's check that the forces are the energy's gradient.
- Frames every 0.5 ps after 10 ps of equilibration.

Rigid monomers do three things:

- They keep the pinned charges one set per molecule. E3's lesson: charges recomputed at every geometry count the
  charge flux twice (`water.pin_site_model`).
- They keep E₀ off the monomer's own surface.
- They keep E3's conventions.

**Cells at 0.997 g cm⁻³:**

| molecules | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| cubic side (Å) | 12.43 | 15.66 | 19.73 | 24.86 | 31.32 | 39.46 |
| the bands' k-vectors at ℓ₁ = 0.75 Å | 3.9×10⁴ | 7.9×10⁴ | 1.6×10⁵ | 3.1×10⁵ | 6.3×10⁵ | 1.3×10⁶ |

E₀'s minimum image (r_c 6 Å) holds from 64 molecules. MBX labels the 64-molecule cells through their 2×2×2 replica,
which is exact (§2) and costs 8×; from 128 molecules it labels the cell itself.

**The sets:**

- **Training, by L₀** (the Form II sweep). Each rung trains on cells of one size, the same number of atoms at every
  size: 180 frames of 64, 90 of 128 and 45 of 256 molecules, some 35 000 atoms and 100 000 force components each.
  - As built (`scripts/e4_data.py`, `run_e4_data.sbatch`): four independent trajectories (seeds) per size, with 60,
    30 and 15 frames per seed.
  - Seeds 0–2 are training. Seed 3 is split into validation (its first half) and test (its second half), so no
    test frame is correlated with a training frame.
- **The read-out's identification data**, computed open as in E3. MB-pol's far-field probes (a core with a molecule
  at 10–15 Å) and far trimers match the read-out before training. Its pins are MB-pol's site charges, M-sites and
  polarisabilities, read from MBX.
- **Test, Form II**: 60 frames of 512 and 52 of 1024 molecules (four seeds each), from their own MD.
- **Test, Form I**: fixed contents in cubic cells L = 15–60 Å on a log grid, and in (L, L, 2L) cells.
  - A polar cluster: an E3 hexamer re-relaxed on MB-pol. This is the charge channel, p⋆ = 3. The dipole image term
    leads, and its level for the dagger is set by the induced share of the cluster dipole.
  - A non-polar cluster: the S4 octamer, μ = 0. Quadrupole images go as L⁻⁵ and dispersion images as L⁻⁶, which is
    where the learned-charge rungs part from the pinned ones.
  - One molecule, as the check of the boundary condition.
- **ε_∞ and Z***: 20 frames each of 128 and 512 molecules at clamped nuclei.
- **The near field**: the liquid's compact tetramers (closest O–O below 2.7 Å) cut from the 512 frames, with their
  4-body computed open by MBX.

**Cost**, from the Niflheim timings (§2):

- *Labels.* Every label in the plan (the training sets, the 512- and 1024-molecule tests) takes about 10 minutes of
  MBX. The ε_∞ frames take about 3 more.
- *MD.* MD dominates. A trajectory is a 1 ps warm-up at 0.5 fs, 10 ps of equilibration at 2 fs, then 30, 15, 7.5,
  7.5 and 6.5 ps of sampling for 64, 128, 256, 512 and 1024 molecules. On 8 threads that takes about 0.3, 0.4, 0.5,
  1.0 and 2 h. The 64-molecule figure (the cell on itself) and the 1024-molecule one are extrapolated. Flexible MD at
  0.5 fs would take four times as long.
- *What costs most.* The truth is cheap. E4's cost lies in the harness's reciprocal-space sums at ℓ₁ = 0.75 Å
  (the table above), which is where the timing has to be watched.

## 4. The ladder

Every rung is trained with 4 seeds. The Form I and Form II predictions follow the size note's §6. The X3
pre-registration replaces them with numbers for these contents.

| rung | IR sector | why it is here | Form I | Form II, saturated error |
|---|---|---|---|---|
| M_∞ | none (E₀ to 6 Å) | the short-range control | relative error → 1 | flat in L₀ |
| M_1 | learned charges, neutral per molecule | p_min = 3 | at threshold (charge channel), slope 3 (dispersion) | ∝ L₀^−(3+2s_δ) |
| M_16 | + learned α(iω) | the free dispersion channel | as M_1 | as M_1 |
| M_16†ᵐ | MB-pol's charges, M-sites and α pinned, no response | the dagger | slope 0, level set by the missing induction | floor ∝ L₀⁻³ |
| M_16C2†ᵐ_us | E3's frozen form, retrained | the model | slope 0, small level | floor |
| M_G | a global function | Gate C | measured (Form III) | measured |
| MACE, r_c 6 Å | none | external short-range baseline | → 1 | flat |
| MACE-LES | latent charges, no sum rule | Theorem 2's class | net charge free: p_min = 1 | ∝ L₀⁻¹ |

## 5. The harness build: what is not periodic yet

**Already in cells:**

- E₀ by the minimum image;
- the Ewald bands, reciprocal sums only (p = 1 with dipoles, p = 6, 8), tested against the Madelung constant and
  direct sums;
- the band-field tensors to order 2, with the fragment exclusion (`band_field_tensors_pbc`);
- M_A and M_G.

**Missing for the ladder:**

1. **The site model in cells.** `Rung.msites` refuses a cell (`model.py` line 222). The M-sites' charges are needed as
   sources of the Ewald band and of the band field, with the fragment exclusion and molecules made whole across the
   boundary.
2. **The self-consistent response in cells.** `bf_mutual` refuses a cell (line 308). It needs:
   - the band's dipole tensor under Ewald, as matrix-free products through structure factors;
   - its fragment exclusion and self term;
   - the solve, either by conjugate gradient with implicit differentiation, or direct up to 512 molecules (4608
     unknowns).
3. **The field term in cells**: each molecule's local pinned field within the taper (4–6 Å) by the minimum image, the
   sources whole.
4. **A uniform external field** at clamped nuclei, added to the band field. This gives dM/dE, hence ε_∞, and Z* by
   differences (X4b and T4).
5. **The readers**:
   - Form I: fixed contents in a cell sequence, against the largest cell or an extrapolated E(∞) (§2).
   - Form II: per-molecule errors by size.
   - The modulation response (X2-C).
   - The near-field set.
6. **The data pipeline** (`scripts/e4_data.py`): rigid MD, MBX labels and pins, and extxyz for the
   baselines. *Done in v0.0.19.*

Each item comes with tests:

- a cluster in a large cell against its open value plus the image terms;
- the supercell identity;
- explicit image sums for the dipole tensor;
- equivariance;
- forces against differences.

**Cost.** The reciprocal sums are O(N · N_k), with N_k ≈ 3 V/bohr³ at ℓ₁ = 0.75 Å (the table in §3). That is cheap for
training at 64–256 molecules, and seconds per test frame at 512–1024 with the mutual solve. A particle-mesh version is
needed only if 2048 molecules are wanted.

## 6. Predictions the X3 pre-registration will compute

- **Form I.** The truth's Δ⋆(L) for each content and cell shape (MBX). Each rung's predicted slope and, for the pinned
  rungs, its level: the dagger's from the induced share of the cluster dipole, the C rung's from the matched response.
- **Form II.** The L₀ law per rung: M_∞ flat; pinned rungs on the analytic floor ∝ L₀⁻³; MACE-LES ∝ L₀⁻¹.
- **ε_∞.**
  - The truth's comes from MBX on the 128 and 512 frames.
  - The C rung's is predicted within a few per cent: the response is matched and self-consistent.
  - A first-order response (S rungs) gives 1 + 4πnα, below the truth's.
  - M_∞, the dagger and MACE-LES give exactly 1.
- **The near field.** The C rung's 4-body on the liquid's compact tetramers against MBX's. E3's compact tetramers had
  0.57 against the labels' 0.34.
- **Theorem 2 on the open probes.** MACE-LES reads 100 % on the far non-additive energy, the daggers' value.

## 7. Baselines

MACE (short range, r_c 6 Å, 2 layers) and MACE-LES:

- trained on the same sets and splits, from the pipeline's extxyz;
- on Vadic's GPU, in their own PyTorch env;
- first on E3-QP8's clusters, which needs no new data (T1 on the rare-gas dimers of E1/E2, T5 on the far probes);
- then on the boxes.

MACE+D3 is optional. EFA/RANGE come later (`long-range-models-survey.md`).

## 8. DFT, later

The same rigid-monomer frames at one or two sizes (64 or 128 molecules), revPBE-D3 or SCAN, with DFPT ε_∞ and Z*. The
code is still open: CP2K, GPAW, VASP or QE.

## 9. Order

1. **This patch.**
   - Here: the MBX build script, the driver and its checks (§2), this plan, and E3's summary.
   - On Vadic's side: build MBX on Niflheim with `sh external/mbx/build.sh`. In the conda env,
     `conda install -c conda-forge fftw autoconf automake libtool`, or `module load FFTW`. The clone needs the login
     node; the compile takes about 20–40 min on 8 cores.
   - Then run `run_e4_mbx_checks.sbatch`, which repeats §2's checks at −O2 with the real timings, and send back its
     log. *Done 28 Oct (§2).*
2. **Next patches, here.**
   - The data pipeline; Vadic runs the MD on CPU nodes. *Done in v0.0.19 (`run_e4_data.sbatch`).*
   - The periodic path, items 1–4 of §5, with tests.
3. **In parallel, on Vadic's GPU**: the MACE/MACE-LES env and the baselines on E3's data (§7).
4. **The X3 pre-registration** with the numbers from MBX.
5. **E4's first job**: the ladder at one L₀, read on Form I, ε_∞ and the near field.
6. **The L₀ sweep** (Form II).
