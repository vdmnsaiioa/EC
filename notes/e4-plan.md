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
  - *Produced 30 Oct* on Niflheim (3b343b9). All 20 trajectories are clean: their temperatures are canonical and
    the conserved energy drifts at most 6e-5 kcal/mol/ps per molecule. The energy per molecule is −9.78 to −9.81
    kcal/mol at every size.
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
| M_∞ | none (E₀ to 6 Å) | the short-range control | relative error → 1 | flat in L₀ (but see †) |
| M_1 | learned charges, neutral per molecule | p_min = 3 | at threshold (charge channel), slope 3 (dispersion) | ∝ L₀^−(3+2s_δ) |
| M_16 | + learned α(iω) | the free dispersion channel | as M_1 | as M_1 |
| M_16†ᵐ | MB-pol's charges, M-sites and α pinned, no response | the dagger | slope 0, level set by the missing induction | floor ∝ L₀⁻³ |
| M_16C2†ᵐ | the same without the field term | the field-term ablation (2 Nov) | as M_16C2†ᵐ_us | floor |
| M_16C2†ᵐ_us | E3's frozen form, retrained | the model | slope 0, small level | floor |
| M_G | a global pair function | the null | zero tail by construction in a cell* | measured (Form III) |
| MACE, r_c 6 Å | none | external short-range baseline | → 1 | flat (but see †) |
| MACE-LES | latent charges, no sum rule | Theorem 2's class | net charge free: p_min = 1 | ∝ L₀⁻¹ |

† *Added 3 Nov, from job 1* (`e4-job1-results.md` §4b): trained at 64 molecules, M_∞'s mean error per molecule shifts
by −0.009 to −0.010 kcal/mol at 128–256 molecules against the dagger's, on every trajectory. Its bulk error at L is
the training cell's long-range energy carried along, so its Form II bias depends on L₀ rather than being flat in it.

\* *Added 2 Nov.* In a cell M_G's pair function runs over minimum-image pairs, so a cluster in a cell wider than twice
its size has no image term at all: its Form I relative error is 1 by construction, and Gate C cannot be read on it. Its
informative reading is Form III, its bulk error by size. Gate C as an open question needs a learned global function that
sees the images (a filter on the k-grid); none is in the harness (`e4-x3-preregistration.md` §5).

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

   *Items 1–4 done in v0.0.20* (`sseft/periodic.py`, `tests/test_periodic.py`):
   - The interaction energy of a cell is its periodic energy less every molecule alone, open. A molecule keeps its
     interactions with its own images, as MBX's labels do.
   - The response's band field and dipole tensor leave out the own molecule at n = 0 only, so a molecule is polarised
     by its own images.
   - The dipole tensor is explicit for a direct solve, or a matrix-free product for GMRES with implicit derivatives;
     the two agree to round-off.
   - The field term takes every other molecule whole at its nearest image.
   - The dispersion pins are MB-pol's long-range c₆, rank one (`water.mbpol_dispersion_pins`).
   - All Ewald sums are chunked half-space sums with bounded memory. They return NaN when `ewald_n_max` falls short of
     k_max, and the energy is NaN in cells narrower than 2 r_c.
   - Checked:
     - a cluster in 30–40 Å cells against its open value plus −(2π/3V)|M_perm + M_ind|², within 0.3 %;
     - Clausius–Mossotti to 1e-10;
     - the supercell identity to 1e-10;
     - forces against differences;
     - invariance under wrapping and translation;
     - one molecule in a cell against its own dipole image energy.
   - Still dense in N: E₀'s and the pair read-out's pair arrays. That suits training at 64–256 molecules.
   - *Measured 2 Nov* on an RTX 3090 (`run_e4_periodic_timing.sbatch`): a training step takes 0.54, 1.6 and 6.3 s at
     64, 128 and 256 molecules with GMRES (peak 0.8, 3.1 and 11 GB), and GMRES equals the direct solve to 5e-16 in the
     forces. 512 and 1024 molecules run out of the card's 24 GB, and it is E₀, not the Ewald sums: compiled with parts
     switched off (`scratch/e4_periodic_memory.py`), E₀ alone holds 7.5 of the 8.5 GB of a 512-molecule energy's
     buffers, the read-out 0.8, the field term 0.2, the chunked bands nothing measurable. About 90 atoms lie within
     6 Å of each atom, so neighbour lists shrink E₀'s arrays 17× at 512 and 34× at 1024: they come with the readers.
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
  *Corrected 2 Nov* (`e4-x3-preregistration.md` §5): a probe's core lies inside r_c, so environment-dependent charges
  can carry the core's internally induced dipoles, the leading R⁻³ part of the probe's non-additive energy; MACE-LES
  need not read 100 % there. The clean test is T5, far trimers with every pair beyond r_c, where every rung without a
  response, MACE-LES included, gives exactly zero.
- *Added 2 Nov.* Since a uniform field reaches only the read-out and the read-out is matched on the probes and held,
  ε_∞, the far field, T5 and the Form I tail of contents whose images lie beyond r_c are fixed before training: the
  pre-registration computes them, it does not estimate them.

## 7. Baselines

MACE (short range, r_c 6 Å, 2 layers) and MACE-LES:

- trained on the same sets and splits, from the pipeline's extxyz;
- on Vadic's GPU, in their own PyTorch env;
- on the boxes only (Vadic's decision of 2 Nov), with MB-pol's far-field probes and far trimers read open (T5).
  E3-QP8's clusters are skipped: E3 is closed.

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
   - The periodic path, items 1–4 of §5, with tests. *Done in v0.0.20.*
   - The training driver and job 1 (`scripts/e4_water.py`, `configs/e4_water.yaml`). *Done in v0.0.21.*
   - Next: the readers (§5 item 5), with E₀ and the read-out on neighbour lists for the 512- and 1024-molecule
     frames, while job 1 runs.
3. **In parallel, on Vadic's GPU**: the MACE/MACE-LES env and the baselines on the boxes (§7).
4. **The X3 pre-registration** with the numbers from MBX. *Done in v0.0.21* (`e4-x3-preregistration.md`).
5. **E4's first job**: the ladder at L₀ = 12.4 Å (64 molecules), 4 seeds, with the field-term ablation beside the
   frozen form (Vadic's decision of 2 Nov); read on every frame of 64–256 molecules, the probes, T5 and ε_∞, then by
   the readers on Form I, the near field and 512–1024 molecules. *Run 3 Nov (8–9 h per seed); read in
   `e4-job1-results.md`: the infrared as computed, the field term −19 % on the rigid forces at every size, the
   response without it nothing; M_∞'s energy drifts with the cell (−0.010 kcal/mol per molecule against the dagger
   at 128–256 molecules); M_1 and M_16 did not train.*
6. **Job 2: the band-edge sweep** (X2-C on real data; Vadic's decision of 2 Nov): ℓ₁ = 0.75 and 1.0 Å at read-out
   orders 0 and 2, pre-registered with job 1. *Its matched part is computed* (`e4-x3-preregistration.md` §6): ε_∞
   is set by the matching's regularisation, not identified by the probes. *4 Nov* (`e4-response-regulator.md`): by the
   EFT-of-LSS hierarchy the regulator belongs above the coupling's non-linear scale. Smearing the bands' edge with it
   loses the far field; smearing only the mutual tensor, at the a-priori 2.0 Å, brings ε_∞ to 1.72–1.74 (MB-pol 1.69)
   with the far field intact. Proposed: job 2 at that edge, and the remaining dependence on the edge calibrated on
   MB-pol's response in liquid frames. *Re-registered 5 Nov* (`e4-job2-preregistration.md`, v0.0.24; Vadic's go of
   5 Nov): arm A the frozen form, its ablation and the dagger with the field term at the 2.0 Å mutual edge; arm B the
   frozen form calibrated on MB-pol's induced dipoles and field response in 30 liquid cells (`run_e4_response.sbatch`,
   then `run_e4_job2.sbatch`). *Run 5–6 Nov; read in `e4-job2-results.md`: the regulator −9 % on the frozen form's
   rigid forces at every size (0.551 kcal/mol/Å at 64 molecules), the ablation level with the dagger; the calibration
   exact in the infrared and neutral for the frozen form's bulk, but it moved contact energy into the field-gradient
   orders (the ablation +2–5 %). Proposed: the calibration on the order-0 columns alone (v0.0.25), retrained.*
   *Registered 6 Nov as job 2's arm C (`e4-job2c-preregistration.md`, v0.0.26; Vadic's go of 6 Nov):
   `run_e4_job2c.sbatch`, about 26 GPU hours.* *Run 7 Nov, eight tasks of 2.9–3.4 h; read in `e4-job2c-results.md`.
   Arm B's cost was the orders ≥ 1: on the order-0 columns alone the calibration gains on both rungs, on every seed —
   the ablation 1.4–1.6 % below arm A's (now 1.5–2.1 % below the dagger), the frozen form 1.2–1.5 % (0.545 kcal/mol/Å
   at 64 molecules) — with ε_∞ 1.6893 against MB-pol's 1.6885. Arm C's frozen form is the model. Proposed next: the
   finite-k response, read on arm C as a prediction of its orders ≥ 1 before it calibrates them.*
   *Built 7–8 Nov as job 3 (v0.0.27, `e4-job3-preregistration.md`; Vadic's go of 7 Nov). It needs no training: MB-pol's
   induced dipoles under longitudinal plane waves of the cell (k = 0.51 and 1.01 Å⁻¹) on job 2's 40 liquid cells, read
   against arms A, B and C, arm C's order-0 sector, and the locality test. Locally (13 cells): MB-pol's ε_L is 1.772 and
   2.090 (ε₀ 1.687). Arm C is −1.9 % and −11.5 % on the cells' susceptibility, through its far-probe orders ≥ 1; its
   order-0 sector alone is +2.6 % and +3.5 %. The order-0 coefficient calibrated on every liquid molecule alone fits them
   to 1–2 % and predicts the vacuum cores' far field at the order-0 class's floor: local at this cutoff.
   `run_e4_response_k.sbatch`, then `run_e4_job3.sbatch`. Proposed after it: arm D, the orders ≥ 1 calibrated on the
   finite-k response.* *Run 8 Nov at a7cf750; the reading took 16 minutes on one GPU. Read in `e4-job3-results.md`: six
   of the seven bands hold.*
   - *MB-pol's ε_L is 1.773 and 2.092 on the 40 cells, k² up to k₂ (c₂ = 0.274 Å²).*
   - *Arm C is −1.8 % and −11.0 %, through its far-probe orders ≥ 1. Its order-0 sector is +2.4 % and +3.3 %.*
   - *The locality test holds in state and out of state.*
   - *K4 fails on arm B. Its orders ≥ 1, moved by k = 0 data, do not reproduce from 10 to 30 calibration cells (+12.6 %
     at k₂ against +4.6 %).*
   - *Not registered: in the cells' χ(k) the orders ≥ 1 add exactly γk². The k⁴ term is the order-0 sector's, set by
     the mutual edge: −8 % of χ(0⁺) at k₂ at 2.0 Å. So arm D's ceiling at that edge is ±1.1 % at k₁ and k₂ together.*
   - *Proposed: job 4, arm D at matching level, with H3 read at 1.5 and 2.5 Å beside it.*
   - *Built 8–9 Nov as job 4 (v0.0.28, `e4-job4-preregistration.md`; Vadic's go of 8 Nov). Arm D is arm C's read-out,
     then its orders ≥ 1 alone on the far probes and the calibration cells' finite-k susceptibilities. Locally it reaches
     the frozen form's ceiling at every edge: χ +1.4 / −0.5 % at 2.0 Å, with a frame sd of 0.07 points. Its costs are
     the probes +0.2–0.5 points and the molecules at k₂ 7.1 % against 4.7 %.*
   - *The control Dx (the orders ≥ 1 re-solved on the probes alone) shows that most of arm C's −11 % came from its
     orders ≥ 1 being matched beside the uncalibrated order 0. The far probes do not pin them; the finite-k data do.*
   - *`run_e4_job4.sbatch` (three GPU tasks), then the read; training (`run_e4_job4_train.sbatch`, about 26 GPU hours)
     only after it.*
   - *Matching level run 9 Nov at 71831ca; read in `e4-job4-results.md` §1. All seven bands hold: arm D +1.33 / −0.37 %
     (predicted +1.32 / −0.39 %). Dx swings to +6.3 % (+0.7 % locally), so the far probes alone do not pin the orders
     ≥ 1. Training predictions T0–T2 were registered at 14:40 CET (pre-registration §8), then training started.*
   - *Trained 9–10 Nov (two GPU tasks of four seeds); read in `e4-job4-results.md` §2. T0 holds; T1 and T2 fail high.
     Both rungs are 6.4–7.5 % above arm C's on every seed, and the field term's gain is unchanged at 23 %. Arm C stays
     the model. Arm D's change at contact was smaller than arm B's, so its size does not price the training cost.*
   - *Proposed, not registered: D′, the same finite-k rows with the orders ≥ 1 anchored at arm C's (the ridge toward
     them, and the probes held at arm C's predictions), to separate arm D's re-fit on the probes from the finite-k fix.
     Then the molecules' finite-k rows, the registered next step for this branch.*
   - *Built 10 Nov as job 5 (v0.0.29, `e4-job5-preregistration.md`; Vadic's go of 10 Nov). Locally D′ reaches arm D's
     ceiling (χ +1.44 / −0.32 %). Its change at contact is half of arm D's (0.22 against 0.42 rms) and its energy a
     tenth, and its far test probes improve on arm C's. `run_e4_job5.sbatch` (one GPU task) runs the matching level;
     training only after its read.*
   - *Matching level run 10 Nov at d6c801f; read in `e4-job5-results.md` §1. All seven bands hold. D′ reads +1.32 /
     −0.38 %, arm D's χ to 0.01 point. At contact it moves 0.156 rms against arm D's 0.325, and every other reading is
     closer to arm C's. Training predictions T0′–T3′ and the model rule (D′ is the model if within +1.0 % of arm C's)
     were registered at 15:19 CET on 11 Nov (pre-registration §8); `run_e4_job5_train.sbatch` follows.*
   - *Trained 11–12 Nov as eight tasks on the 4-hour GPUs (`run_e4_job5_train_4h.sbatch`, 2:57–3:40 each). Read in
     `e4-job5-results.md` §2.*
     - *T0′ and T3′ hold. T1′ and T2′ fail low: D′ is 0.6–0.7 % below arm C's rigid forces (frozen form) and 0.6–1.1 %
       below them (ablation), every split. So arm D's re-fit carried its cost, and the finite-k fix is free in the bulk.*
     - *E4's model is now D′'s frozen form (model rule M′; `configs/e4_job5.yaml`): 0.541 / 0.551 / 0.552 kcal/mol/Å at
       64 / 128 / 256 molecules, ε_∞ 1.6893, χ +1.32 / −0.38 %.*
     - *Next, proposed: the molecules' finite-k rows, since D′'s molecules at k₁ / k₂ are 0.5–1.1 points worse than arm
       C's. Or the readers on the new model, Form I and the L₀ sweep (item 7).*
   - *S0 registered 11 Nov (`e4-s0-preregistration.md`; the bands of the project's `path-integral-ir-sector.md` §4.1;
     Vadic's go of 11 Nov). It reads the polarisation sum rules on MB-pol's own trajectories:
     - L(k → 0) = 1/ε_∞ − 1/ε_s and T/L ≈ ε_∞ε_s;
     - the regression ratio r = b_T/b_L, which is ε_∞ for field-blind sources and 1 for a model that carries the
       response, on the dagger's sources and on arm C.
     It needs no training: MBX's sites at zero field on every frame (five CPU tasks), arm C's own polarisation (one GPU
     task), then the read with NumPy (`scripts/e4_sumrules.py`).*
   - *MB-pol's part read 12 Nov at f8788ff (`e4-s0-results.md` §1).*
     - *S0.1 and S0.2 hold: L(k → 0) = 0.610 ± 0.030 and T/L = 86–114. The sum rule confirms ε_∞ (1.60 ± 0.08).*
     - *S0.3 fails high. MB-pol's own sources regress at r = 1.98–1.99, not ε_∞ = 1.70: b_T = 1.455 at every k, and
       b_L sits about 15 % below b_T/ε_∞ at small k. The lemma's scale between 1 and ε_∞ does not hold for MB-pol, but
       field-blind (≈ 2.0) and response-carrying (1) sources are still told apart.*
     - *MB-pol overscreens from about 0.3 Å⁻¹. Arm C at 64 molecules reads r = 0.99–1.00, not registered.*
     - *S0.4 waits for the model step, which stopped at 128 molecules on a reader bug, since fixed.*
   - *Direction chosen 12 Nov (Vadic's go): close S0 first, then the MD driver and S1/S2 (the path-integral note's
     §4.2–4.3) as the main line. The molecules' finite-k rows, the L₀ sweep and Form I are deferred.*
     - *S0.5 was registered at 14:48 CET (`e4-s0-preregistration.md` §6). It is the nuclear EFT note's account of
       S0.3: r = ε_∞ + v_η, with v_η the transverse regression's residual. Its band is 0.7–1.4 on
       ρ = (r − ε_∞(k))/v_η.*
     - *S3's normalisation was amended to ε_∞(k) − 1.*
     - *The EFT's readings were added beside it: 1/A_T and 1/A_L with their fits and the overscreening zero, S_nn and
       S_nZ. All of it is on the existing frames.*
     - *S1 and S2 were registered at 15:18 CET (`e4-s1-preregistration.md`), before the MD driver existed.*
       - *S1.1: |L_D′ − L_MB-pol| ≤ 0.02 + 2σ on the 64 cell's (100) shell and the 128 cell's (100) and (110).*
       - *S2.1: the dagger's own L(k₁) ≥ 0.90.*
       - *The runs: D′ at 64 and 128 with four seeds, MB-pol the same, and the dagger at 64 with two; 10 + 100 ps.*
       - *The driver is `scripts/e4_md.py`, on `sseft/rigidmd.py`, checkpointed over 4-hour jobs. A timing job comes
         first.*
     - *Vadic's question of 12 Nov — induction and dispersion from one path integral — became
       `v2-one-field-proposal.md`.*
       - *It argues for v2's long-range sector as one field: α_i(iξ) = C_i(0)/(1 + ξ²/ω_Z²) for both channels, and the
         Tr ln's many-body remainder on the induction's mutual tensor.*
       - *Its first test, the tie read on the frozen form, rides with S0's model step.*
       - *The real tests are a truth built as one field (MB-pol₁), after S1, and the DFT phase.*
7. **The L₀ sweep** (Form II).
