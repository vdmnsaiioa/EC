# Results log

Numbers behind the README's claims, one entry per commit that changes them.  Every entry names the command
that produced it; wrong predictions are recorded as such.

## v0 (harness-v0, Oct 2026) — regressions against the synthetic truths

### Tests (`pytest -q tests/`, 21 tests)

Forces of every rung against four-point finite differences at h = 3e-5 bohr: |ΔF| < 1e-6 |F| + 1e-10 (the
largest observed discrepancy 2.3e-11 on a force of 4e-4; the autodiff value is the converged one — the FD
error falls as h⁴ from 7e-6 at h = 1e-3).  Band-field tensors against finite differences; Casimir–Polder
against London to 0.3 % at K = 8; the Laplacian family against a finite-difference Laplacian; the Coulomb
band (charges + dipoles) and the dispersion band against their closed-form two-body expressions to 1e-12
and against the classical multipole / London limits at 60 bohr; energies invariant and forces covariant
under rigid motions (M_16 with learned dipoles, M_S2, M_A) to 1e-12; interaction energy exactly 0 for a
dimer beyond r_c and for an isolated atom under M_inf; Σq_i = 0 to 1e-12; the published Ar₂ / Ne₂
parameter sets reproduce ε/k_B = 143.123 / 42.153 K at R_ε = 0.37618 / 0.30894 nm.

### Dimer limit, Tang–Toennies Ar₂ (`scripts/dimer_ladder.py --system Ar2 --truth tt`)

Window [5.15, 9.26] Å, 40 nodes, σ = 1e-9 E_h, r_c = 6 Å, 1500 Adam steps, lr 3e-3, energies + forces.

| rung | seeds | window rmse (E_h) | rel. error 10 / 20 / 100 Å | W/\|f*\| 10 / 20 / 100 Å | q 10 / 20 / 100 Å | plateau slope [30, 300] Å |
|---|---|---|---|---|---|---|
| M_inf | 4 | ~1e-6 (cannot fit beyond 6 Å) | 1.000 / 1.000 / 1.000 | 0 / 0 / 0 | – | – (branch 3) |
| M_6 | 4 | ~1e-8 | 0.16 / 0.22 / 0.25 | 1.2e-3 / 1.2e-3 / 1.2e-3 | 6.00 / 6.00 / 6.00 | 0.00 (branch 2) |
| M_A (ℓ = 4 Å, N = 4), + L-BFGS 300 | 2 | 5.2e-8, 2.8e-8 | 0.03 / 1.00 / 1.00 | 0.21 / 8e-4 / float floor | −7.5 / 38.6 / – | – (collapsed) |

M_6: learned C₆ = 80.4 E_h a₀⁶ against 64.3 (Ĉ₆/C₆ = 1.25; the linear L6a prediction on the nodes beyond
r_c is 1.20), relative error 16–25 % = the C₈, C₁₀ truncation of a C₆-only band, spread flat at 1.2e-3 —
the branch-2 plateau at the κ set by training variance.

M_A: the three regimes of `x1-prediction.md` §2.1 — burst beyond R₊ (W/|f*| 0.2–0.7 at 10–12 Å, q = −7.5
at 10 Å), q rising through 6 at ≈ 12 Å (predicted ℓ√(N+3) = 10.6 Å plus 1–3 Å), then growing linearly in
R′²: between 15.0 and 19.6 Å, Δq/ΔR² = (38.6 − 17.6)/(384 − 225) = 0.132 Å⁻² against the predicted
2/ℓ² = 0.125 (6 %); the mean's relative error 1.00 from 20 Å on; the spread at the float64 floor (1e-20)
beyond 29 Å.  Window rmse/σ = 30–50, not the ≈ 1 of criterion (i): with Adam alone (1500 steps) the raw
family left rmse 3e-5 — the window Gram matrix of the raw family has condition number 3.5e10 — and the
column-normalised family with the L-BFGS polish brings it to 3e-8.  Optimisation-limited; recorded.

### X2-B toy (`scripts/x2b_toy_regression.py`)

Truth T3 (`x2-protocol.md` §3; a = 0.5, α₀ = 0.8, ℓ₁ = 2), 24 training clusters × 10 charge patterns
(N = 40, densities 0.15 / 0.22 / 0.30), 8 test clusters (densities 0.18 / 0.26, other seed).  Pinned charges,
all sites one species; E₀ cutoff 2.0 (= 4a); F = 16.  The pass mark is the ceiling of this split — the
exact Taylor class of the truth (one LEC per term and order, OLS on the training set):

| class | order 0 (train / test) | order 1 | order 2 |
|---|---|---|---|
| exact Taylor class (ceiling of this split) | 3.70e-3 / 4.33e-3 | 1.64e-3 / 2.12e-3 | 5.82e-4 / 7.40e-4 |

(The fitted LECs at order 2 are −0.400, −0.327, −0.242, −0.317, −0.230, −0.299, −0.254 against the exact
−0.4, −0.32, −0.256, −0.32, −0.256, −0.32, −0.256: the class is built correctly.  The ceiling depends on the
clusters: `x2b-results.md`'s 2.3e-3 → 1.2e-4 was for N ∈ {40, 60}, 40 clusters.)

Harness, read-out variants (24 training clusters unless stated; 1000–1500 Adam steps + 150–300 L-BFGS):

| read-out | E₀ cutoff | order 0 (train / test) | order 2 (train / test) | verdict |
|---|---|---|---|---|
| node (L ≤ 2 channels), raw derivative tensors | 6.0 | 1.14e-3 / 7.16e-3 | 1.38e-3 / 7.13e-3 | fails: fully connected E₀ overfits the training clusters; no gain 0 → 2 |
| node, tensors in units of ℓ₁, higher orders zero-initialised | 2.0 | 2.25e-3 / 6.67e-3 | 4.41e-3 / 1.15e-2 | fails: order 2 under-optimised, and L ≤ 2 node channels cannot carry the rank-4 environment tensors of the order-2 operators |
| pair (Taylor class on the structure), n_rbf 20 | 2.0 | 1.71e-3 / 4.20e-3 | 1.01e-3 / 8.40e-3 | order 0 at the ceiling (4.33e-3); order 2 overfits (24 clusters) |
| **pair, n_rbf_readout 8, 48 training clusters** (ceiling of that split 4.04e-3 / 4.01e-3 and 6.25e-4 / 7.69e-4) | 2.0 | **2.27e-3 / 3.74e-3** | **1.32e-3 / 2.44e-3** | order 0 below the ceiling; order 2 improves on it (gain 1.5 on test, the exact class gains 5.2) with the training loss still falling under L-BFGS (3.5e-5 → 4.0e-6 in 300 steps, halving every ~150): wired correctly, under-converged at this budget |

The node read-out's failure is informative: the order-2 operators F_a T_{abcd} ∂_c∂_d F_b need
Σ_j w(r_ij) n̂_ij^{⊗4}, an L = 4 environment tensor, which scalar / vector / rank-2 node channels cannot
represent — the body-ordered pair read-out (radial functions times powers of the pair direction, the
Taylor class on the physical structure of `x2-protocol.md` §1.3) is the right implementation of F_s.
With a 900-step polish (`results/x2b_toy_pair_48clusters_order2_lbfgs900`, the memory-bounded L-BFGS): order 2
**train 6.31e-4 (ceiling 6.25e-4), test 1.63e-3 (ceiling 7.69e-4)** — the training error reaches the exact class's,
the test error is 2.1× it with a gain of 2.3 over order 0 (the exact class gains 5.2).  The band-field path is
wired correctly; what remains is a generalisation gap of the learned coefficient functions (7 LECs in the exact
class against ~10⁴ parameters here, 48 training clusters), which more clusters or weight decay would close.
The loss was still falling (9.2e-7 at step 899, halving every ~250 steps).

The node read-out's failure is informative: the order-2 operators F_a T_{abcd} ∂_c∂_d F_b need
Σ_j w(r_ij) n̂_ij^{⊗4}, an L = 4 environment tensor, which scalar / vector / rank-2 node channels cannot
represent — the body-ordered pair read-out (radial functions times powers of the pair direction, the
Taylor class on the physical structure of `x2-protocol.md` §1.3) is the right implementation of F_s.

## v0.0.2 — E1 / E2 on the published Ar₂ and Ne₂ potentials (K = 4)

`scripts/dimer_ladder.py --truth published --seeds 4 --steps 1500 --lbfgs 300`; logs and json in `results/`
(`e1_Ar2_published_K4`, `e1_Ne2_published_K4`, `e2_NeAr_joint_M6_freeK`, `e2_NeAr_joint_M6_oneosc`).  Full
discussion against the pre-registration in the project note `e1-e2-results.md`.

| rung | Ar₂: rel. err 20 / 300 Å, W/\|f*\| 20 Å, q 20 / 300 Å, slope | Ne₂: the same | branch |
|---|---|---|---|
| M_inf | 1.000 / 1.000, 0, –, – | 1.000 / 1.000, 0, –, – | 3 ✓ |
| M_G | 1.5e4 / 1e13, 3.6e3, −4.0 / −1.03, +7.4 | 1.2e4 / 5e12, 3.5e3, −2.7 / −0.92, +7.6 | 1, q → −1 ✓ |
| M_A (4 Å, N = 4) | 0.998 / 1.000, 1.4e-4, 38.7 / float floor | 1.000 / 1.000, 3.9e-4, 43.4 / floor | 3 after the burst ✓; dq/dR′² = 0.131 (Ar₂), 0.129 (Ne₂) vs 2/ℓ² = 0.125 |
| M_6 | 0.241 / 0.265, 6.2e-4, 6.00 / –, +0.00 | 0.153 / 0.163, 5.9e-2, 6.00 / 6.00, +0.00 | 2 ✓; Ĉ₆/C₆ = 1.265, 1.163 (predicted 1.21, 1.10 ± 0.05: +5 % systematic ✗) |
| M_16 | identical to M_6 seed by seed; max\|q_i\| = 0 | – | ≡ M_6 ✓ (Coulomb channel dead on homonuclear dimers) |
| M_16p (q = 0, α from static α₀ + DOSD C₆) | 0.0126 / 0.0068, 0 | 0.0050 / 0.0039, 0 | 2 with zero fibre ✓ |

E2 (joint Ne₂ + Ar₂ fit of M_6; Casimir–Polder C₆ from the learned free-atom α(iω)):
free K = 8: Ne–Ne 7.36 ± 0.37, Ar–Ar 81.70 ± 0.79, **Ne–Ar (unseen) 18.78 ± 3.18** (DOSD 6.383, 64.30, 19.50);
one-oscillator: 7.21 ± 0.14, 81.13 ± 0.10, **21.30 ± 1.00**.  The cross C₆'s seed spread (17 % free, 5 %
one-oscillator) is the measured gauge freedom; α₀ is not identified from dimer energies (Ne 2.66, Ar 5.28
against 2.67, 11.08 with the one-oscillator prior).

**The +5 % C₆ offset, resolved** (`results/c6bias_Ar2_M6_rcut{5.0,7.0}`): trained Ĉ₆/C₆ at r_c = 5.0 / 6.0 / 7.0 Å
= 1.372 / 1.265 / 1.186 (two seeds agree to four digits).  A scan of the training loss with α pinned puts its
minimum at 1.37 / 1.26 / 1.18 — the energies + forces objective — while the energies-only minimum is at
1.31 / 1.21 / 1.15, the numbers the pre-registration quoted.  The forces (∝ R⁻⁷) weight the inner window where
C₈/C₆R² is largest; at r_c = 5.0 Å E₀ sees no pair in the window, so nothing else is involved.  Ne₂: objective
minimum 1.125 (energies only 1.105); seeds at 1.120, 1.133, 1.149 and one unconverged seed at 1.251.
Linear-class predictions must be computed under the training objective, forces included, and seeds need a
convergence criterion before averaging.

## v0.0.3 — the L6b rung M_68 (C₆ + C₈ bands) on Ar₂ and Ne₂ (K = 4)

Pre-registered in the project note `e1-m68-preregistration.md` with the linear-class minimum computed under the
training objective (energies + forces, nodes beyond r_c): Ar₂ Ĉ₆/C₆ = 0.958, Ĉ₈/C₈ = 1.455; Ne₂ 0.984, 1.321.
`results/e1_{Ar2,Ne2}_published_M68_K4`.

| system | window rmse | Ĉ₆/C₆ per seed | Ĉ₈/C₈ per seed | rel. err 20 / 300 Å | W/\|f*\| 20 / 300 Å, slope | q 10 → 300 Å |
|---|---|---|---|---|---|---|
| Ar₂ | 1.2–1.7e-8 | 0.957–0.963 (pred. 0.958 ± 0.03 ✓) | 1.433–1.460 (pred. 1.455 ± 0.10 ✓) | 0.031 / 0.040 (pred. 0.034 / 0.042 ✓) | 2.4e-3 / 2.7e-3, +0.02 | 4.84 → 6.00 |
| Ne₂ | 1.1–2.4e-8 | 1.02, 1.04, 0.954, 0.965 | 0.87, 0.87, 1.56, 1.50 | 0.0035 / 0.0052 (pred. 0.013 / 0.016) | 3.8e-2 / 4.2e-2, +0.01 | 5.07 → 6.00 |

Ar₂: every number inside its tolerance; the C₁₀–C₁₆ truncation loads onto C₈ (+45 %) and pulls C₆ 4 % low, as
computed.  Ne₂: the two parameters on 16 nodes (0.34 octaves) beyond r_c are ill-conditioned — the seeds split
into two groups along the C₆–C₈ valley (the better-converged pair at 0.96 / 1.53, the other at 1.03 / 0.87),
the C₈ prediction fails (1.32 ± 0.10 against 1.53 for the converged pair) while the asymptotic C₆ holds to 3 %;
reported as such.  One wrong prediction on both systems: q was predicted between 6 and 8 at 10–20 Å and is
instead 4.8–5.8, approaching 6 from below — the seeds' δC₆ and δC₈ are anticorrelated (the fibre runs along the
valley), so the spread |δC₆/R⁶ + δC₈/R⁸| is partially cancelled at short R′; the uncorrelated estimate was wrong.
Window rmse 50× below M_6's (the C₈ term absorbs the L6a misfit); the plateau level 2.6e-3 on Ar₂ is 4× M_6's,
the wider two-parameter fibre.

## v0.0.5 → v0.0.6 — E3 rehearsed without data: the synthetic polarisable water truth (`scripts/e3_water.py`, `sseft/water.py`)

Pre-registered in `notes/e3-water-preregistration.md` (§3; §6–§9 are the addenda of the far-field sequence,
each written before the run it predicts); the results note is `notes/e3-rehearsal-results.md`.  Tests: 34
(`pytest -q tests/`; the water truth, the fragment references, the near-source exclusion and the embedded dagger's
identity added).  Two
design changes the molecular case forced, both in the package: fragment references (`E_int = E(all) −
Σ_f E(f)`, the model's own fragment energies at their own geometries; without them a learned monomer energy
left a flat 10⁻³ E_h tail under every rung) and the near-source switch in the band-field input
(`|E_near + E_far|²` carries a term linear in the far field with an environment-dependent coefficient — a
spurious permanent dipole of ~0.5 kcal/mol at 3 Å for a rigid water).

**Dimer, φ = 0, p⋆ = 3, K = 4** (`results/e3_pw_dimer_K4_part{1,2}.log`; window [3.86, 6.95] Å, r_c = 6 Å):

| rung | window rmse | rel. err 20 / 300 Å | W/\|f⋆\| 20 / 300 Å, slope | q 10 → 300 Å | monomer dipole (truth 1.855 D) | pre-registered → verdict |
|---|---|---|---|---|---|---|
| M_∞ | 1.5e-4 | 1.000 / 1.000 | floor | – | – | branch 3 ✓ |
| M_G | 5.6e-6 (2 of 4 seeds excluded) | 415 / 1.1e8 | 1.2e3 / 1.6e8, +4.12 | −6.4 → −1.04 | – | branch 1, q → −1 ✓ |
| M_1† | 7.7–8.5e-6 | 1.53e-3 / 4.98e-7 | 0 | – | pinned | 1.5e-3 at 20 Å falling as R⁻³, spread 0 ✓ |
| M_1 | 8.1–11e-6 (1 excluded) | 3.4e-2 / 3.3e-2 | 1.63e-2 / 1.63e-2, +0.00 | 3.13 → 3.00 | 1.822 ± 0.013 D | branch 2 ✓; 1.857 D ± 3 %: magnitude ✓, sign of the bias ✗ |
| M_1μ | 4.7–7.8e-6 (2 excluded) | 1.7e-2 / 1.3e-2 | 2.9e-2 / 3.0e-2, +0.01 | 3.09 → 3.00 | 1.876 ± 0.019 D, q_O = +0.04 … +0.08 | same tail, split unidentified ✓ |
| M_A (4 Å, N = 4) | 2.2–3.5e-5 (1 excluded) | 1.07 at 15.5 Å, 1.00 beyond | floor beyond 27 Å | burst, q = 3 crossed at ~10 Å, dq/dR′² = 0.133 | – | R_q3 = 9.4 + (1–3) Å ✓, 0.125 ± 20 % ✓, branch 3 after the burst ✓ |

The wrong sign of M_1's dipole bias (measured λ = 0.982, class minimum under the objective 1.0065,
`scratch/lambda_hat.py`) has a verified cause: at every window node an intermolecular atom pair (the donor
H to the acceptor O, at R_OO − 0.96 Å) is inside r_c = 6 Å, so E₀ is silent at no node and absorbs part of
the band's R⁻³ energy on the identifying nodes.  Pre-registered check with r_c = 5 Å (11 nodes with every
pair beyond r_c): μ̂ = 1.867 D, the bias changing sign.  **Result: 1.864, 1.862, 1.858 D (one seed unconverged and
excluded) = 1.861 ± 0.003 D**, +0.3 %, the tail error of the mean 0.67 % at 300 Å (was 3.3 %), q = 3.00, slope +0.00;
M_1μ 1.870 ± 0.020 D (`results/e3_pw_dimer_rcut5.0_K4.log`).  The identifying nodes of a molecular dimer are
those with every intermolecular pair beyond r_c.

**Clusters, K = 2** (`results/e3_pw_clusters_K2*.log`; 36 training clusters n = 3–5, 8 hexamers held out;
the hexamers' non-additive energy has rms 1.287 kcal/mol = 9.1 % of the interaction energy): non-additive
rmse as a share of it — M_1† 88, 93 %; M_S0† 85, 73 % (pre-registered ≤ 30 % ✗); M_S2† 114, 120 % (no gain
over order 0 ✓, in fact a loss).  Small-cluster induction is near-field induction, E₀'s, and the training set
holds nothing that identifies the band-field coefficient (the dimer's far nodes carry ~10⁻⁶ E_h of
induction, below the window rmse).

**Far-field induction test, K = 2** (`results/e3_pw_far*.log`; a probe monomer at 6 / 7 / 8 / 10 Å from a
relaxed trimer, 8 placements per distance; the "induction" — the probe's interaction minus point-charge
electrostatics — is 13.9 % of the interaction at every distance; the reading is the rmse as a share of it):

| run | far-field training configurations | M_1† | M_S0† | M_S2† | M_1μ | M_S0μ |
|---|---|---|---|---|---|---|
| A (§6) | none | 112, 111 % | 80, 89 % (pred. ≤ 20 % ✗) | 238, 299 % (pred. = M_S0† ✗) | – | – |
| B (§7) | 16 from one core | 106, 99 % | 92, 82 % (pred. ≤ 30 % ✗) | 297, 370 % (pred. ≤ 1.5 × M_S0† ✗) | – | – |
| C (§8) | 16 from one core | – | – | – | 120, 79 % (pred. ≤ 30 % ✗) | 117, 65 % (pred. ≤ 15 % ✗) |
| D (§9) | 32 from eight cores | 98, 114 % | – | – | 86, 116 % (pred. ≤ 50 % ✗) | 80, 108 % |
| E (§10) | 32 from eight cores, weight ×10 | – | – | – | 124, 90 % | – |
| F (§11) | 32 from eight cores | M_1†ᵉᵐᵇ (embedded dipoles, no self energy): 140, 162 % (6 / 7 / 8 / 10 Å: 174–202 / 64–65 / 53–56 / 17 %) | – | – | – | M_S0†ᵉᵐᵇ 149, 158 % |
| G (§12) | 32 from eight cores | M_1†ᵉᵐᵇ + self energy: 48, 39 % (49 / 46 / 50 / 17; 39 / 37 / 49 / 17); **M_16†ᵉᵐᵇ: 21, 18 % (25 / 13 / 16 / 5; 20 / 16 / 9 / 5)** | – | – | – | – |

The constant 13.9 % share identifies what this "induction" is: not the probe's R⁻⁶ response to the core's
field but, dominantly, the interaction of the core's *pre-existing* induced dipoles (0.1–0.3 a.u. per oxygen
in a hydrogen-bonded trimer) with the probe's charges, ∝ R⁻³ — an environment-dependent source, v1's learned
μ_i, which a pinned-charge dagger cannot carry by construction and no degree-2 read-out at the probe can
either.  Run C's single training core showed the source map one environment and asked for a second (the
test was at fault; §9); run D corrects the design and the prediction fails again — for a reason the new training-
residual diagnostic shows: the learned-source rungs' residual on the far-field *training* configurations is
134–163 % of their own induction, and weighting those configurations ×10 (run E) leaves it at 113–170 %.  The
far-field signal (0.03–0.08 kcal/mol on a 0.1–0.6 kcal/mol interaction) sits below a floor of ~0.05 kcal/mol that
the learned-source class does not get under here, weighting or not.  The exact decomposition
(`scratch/far_decomposition.py`): piece (2) is 91 / 100 / 93 / 98 % of the induction rms at 6 / 7 / 8 / 10 Å, the
mutual response 40 / 16 / 14 / 5 %.  The embedded dagger (q and atomic dipoles pinned to the truth's
self-consistent induced dipoles of each molecule in its own cluster) reproduces the band's share exactly — beyond
r_c its probe interaction equals electrostatics + piece (2) to 1e-6 kcal/mol with untrained parameters
(`scratch/emb_check.py`) — but needs two more pinned inputs to be a dagger: the polarisation work ½|μ|²/α per atom
(`pin_eself`; without it the band double counts the induction inside the cluster and the 6 Å error is 174–202 %),
and the dispersion band on a pinned α_O(iω) (without it 12.5 % of the induction at 10 Å is missing).  With all three,
M_16†ᵉᵐᵇ's error is the mutual response alone: 5 % at 10 Å (computed 4.6), 9–16 % at 7–8 Å, 20–25 % at 6 Å.

**The embedded dagger on the dimer and the clusters, and the band edge** (`results/e3_pw_dimer_embself_K2.log`,
`results/e3_pw_clusters_embself*_K2.log`): dimer window rmse 9.5e-7 E_h (10× below every other rung), relative
error beyond r_c 8.4e-7 at 8 Å, 5.9e-8 at 20 Å, 9e-10 at 98 Å, spread 0.  Clusters at ℓ₁ = 1.5 Å: non-additive
rmse 80–81 % of the non-additive energy (pre-registered ≤ 25 % ✗) with dimer rmse 0.05 kcal/mol — the
hydrogen-bond partner's field at 1.9 Å is only 64 % long-range at that band edge, so the cluster's induction is
E₀'s and unlearned; **at ℓ₁ = 0.75 Å (pre-registered ≤ 25 %): non-additive rmse 9 and 4 %, hexamer rmse 0.078 /
0.061 kcal/mol (every rung at ℓ₁ = 1.5 Å: 0.50–0.88), dimer rmse 0.02 / 0.01.**  The band edge of a
hydrogen-bonded system sits below the hydrogen bond; the real E3 runs at ℓ₁ = 0.75 Å.

## v0.0.6 → v0.0.7 — E3-R: the band-field response tested explicitly (`scripts/e3_water.py --part ion`, `sseft/linear.py`)

Pre-registered in `notes/e3r-bandfield-response-preregistration.md` (§3: the order-0 ceilings computed from the
truth; §4: predictions; §6–§10: five addenda, each written between the runs); results in `notes/e3r-results.md`.
Design: permanent sources (the isolated monomer's charges and the ion's charge) pinned with the dispersion band
on the pinned α_O, the band edge ℓ₁ = 0.75 Å, the own fragment excluded from the band-field input (new,
`Rung.bf_fragments`, tested open and periodic), and an ion probe in the truth (a non-polarisable unit charge;
`sseft.water.ION`) whose induction signal at 6–12 Å from a trimer is 0.15–1.0 kcal/mol, 10–60× the fit floors.
The order-0 read-out's computed ceiling is the first-order induction E_ind1: 71 % of the hexamers' non-additive
energy, 85–93 % of a water probe's induction, 86–98 % of the ion's.

**Training the read-out took five rules, each bought with a failed run** (the runs are in `results/e3r_ion_l1_0.75_K2_*.log`):
field tensors in units of E_ref = 0.01 a.u. and the output in units of e_scale (without: a_eff/α = 0.21, the
coefficient never reached); output layers starting at zero (without: the fit collapsed on the ion at 2 Å, clusters
at 3.3 kcal/mol); the read-out frozen during Adam (without: Adam's fixed steps wreck the near configurations,
clusters 1.9 kcal/mol); its order-0 sector set by the weighted linear least-squares solve of the objective
(`sseft.linear.solve_readout`, exact against `train.make_chunked_loss`; the full order-2 solve over-fits);
L-BFGS polish of everything (300 steps).

| reading (K = 2) | M_16† (no read-out) | M_16S0† (order 0) | M_16S2† (order 2) | ceiling / floor from the truth |
|---|---|---|---|---|
| hexamers' non-additive rmse, % of 1.287 kcal/mol — joint Adam, read-out unscaled, L-BFGS 150 | 106, 90 % | **36, 15 %** | 22, 19 % | first order 71 %, remainder 30 % |
| same, final schedule (frozen Adam → solve → L-BFGS 300) | 107, 91 % | 44, 51 % | 52, 51 % (order-0 solve; the full order-2 solve: 126, 69 %) | |
| hexamer rmse, kcal/mol (joint Adam / final) | 0.66, 0.59 / 0.65, 0.60 | 0.21, 0.16 / 0.32, 0.27 | 0.15, 0.14 / 0.40, 0.32 | |
| ion at 6 / 7 / 8 / 10 / 12 Å: rmse / induction, final schedule | 93, 87 % (100 % beyond r_c, as computed) | 42, 50 % | – | first order 95 / 95 / 93 / 89 / 86 % |
| same, + 72 configurations with the ion beyond r_c of every atom | – | **32, 34 %** (30 / 34 / 35 / 55 / 43; 34 / 33 / 31 / 40 / 36) | – | |
| response coefficient a_eff/α (monomer + ion at 20 Å) | – | 0.65, 0.61 → **1.05, 1.00** with the identification data | – | 1 |
| water probe at 6 / 7 / 8 / 10 Å: rmse / induction | 101, 101 % | 68, 68 % (floor-limited: the induction is 1–4× the fit floor) | – | first order 85–93 % |

The read-out learns the band-field response: the response coefficient read on an isolated monomer with
nothing else in play is 1.05 and 1.00 of the truth's α once the training set holds configurations in which the
source is beyond r_c of every atom (14 such structures left it at 0.6–0.65, 86 brought it within 5 %); the
ion's induction error falls from 87–95 % to 32–34 %, the hexamers' non-additive error from 90–106 % to 15–51 %
(schedule-dependent). The errors above the computed floors are fit residuals on the far-field configurations
(33–45 % of their induction, 0.03–0.05 kcal/mol, five to ten times below the clusters' residual) that the
uniform objective does not reduce; the water probe's induction is at the floor. The learned-source tier is
unreadable on a charged system (the charge head's uniform neutrality shift spreads the ion's +1 over every atom:
M_1μ at 591 %) and is deferred until the ion's charge is pinned. Wrong predictions: eight, listed in the results
note; the one that mattered was the §3 "floor" of 30 % on the hexamers, beaten by one seed (15 %) because the
coefficients are environment-dependent.

## v0.0.7 → v0.0.8 — YAML configuration of the experiments; the neutrality shift per fragment (`sseft/heads.py`)

No experiment re-run; this entry records what changed in the code and which recorded numbers it leaves untouched.

**Configuration.** Every experiment is selected and configured from a YAML file (`configs/`, `scripts/run_experiment.py`,
`scripts/experiment_config.py`): the experiment, the water cluster data, the water model (architecture) and the training
schedule, each field mapped to the scripts' existing flags, explicit flags overriding YAML. `configs/water.yaml` is the
E3-R cluster run R1 under the final schedule (`M_16p`, `M_16S0p`, `M_16S2p`; ℓ₁ = 0.75 Å, no radial switch; the
pre-registered clusters; K = 2, Adam 1000 with the read-out frozen → order-0 linear solve → L-BFGS 300; batch 12, chunk 6),
i.e. the command of `notes/e3r-results.md` §1. Every rung's result now records the architecture as run under `model`
(sources pinned / learned / off, the neutrality shift, the band-field order, ℓ₁, the switch radius, r_c, F, ℓ_A) next to
the resolved `training` and `dataset` settings; cluster outputs and caches reject a dataset mismatch. One latent flag bug
fixed on the way: the near-source switch radius was `rs_over_l1` × the rung's *default* band edge (1.5 Å) rather than the
band edge in use — immaterial to every recorded run (all E3-R runs have `--rs_over_l1 0`; the rehearsal runs the default
ℓ₁), recorded as `near_source_switch_angstrom` from now on.

**The neutrality shift per fragment.** The charge head shifted the learned charges uniformly over the whole structure
(`Σ_i q_i = Q`), and the fragment references `E(f alone)` were shifted to the *structure's* total charge. On structures
of several fragments that is inconsistent in two ways: a molecule's charges depend on the other molecules' environment-
dependent `dq` (an intermolecular monopole of either sign; for a probe beyond r_c of a trimer the probe acquires
−Δ/4 and the trimer +Δ/4, a 1/R term the truth does not have), and on a charged system every fragment of the reference
is charged — the M_1μ ion result of v0.0.7 (591 % of the induction, `notes/e3r-results.md` §4(7)). Now the shift is taken
fragment by fragment to each fragment's own total charge (`Structure.frag_charges`; the waters 0, the ion +1;
`Rung.frag_neutral`, `--charge_neutrality fragment|structure`): a fragment beyond r_c of everything else carries
exactly its isolated charges, the ion exactly +1, and `E_int = E(all) − Σ_f E(f)` cancels the intramolecular band energy
exactly (tests: `tests/test_fragment_neutrality.py` — per-molecule sums < 1e-12; a monomer + ion at 40 Å: the monomer's
charges equal the isolated monomer's to 1e-12 and E_int equals the inter-fragment Coulomb band to 1e-12, neither of
which holds under the old shift; forces against finite differences; atomic structures unchanged to 1e-14).

**What it touches.** Nothing pinned: every E3-R number (the daggers `M_16†`, `M_16S0†`, `M_16S2†`, `M_16†ᵉᵐᵇ`) and the
atomic ladders (E1, E2, L6b) are bit-for-bit the same models. The learned-source rungs on multi-fragment structures —
`M_1`, `M_1μ` on the water dimer and clusters, `M_1μ` / `M_S0μ` on the far-field tests of v0.0.6, `M_1μ` on the ion — are
now different models; their v0.0.5–v0.0.7 numbers stand as recorded under the old shift (reproducible with
`--charge_neutrality structure`), and the learned-source tier on the ion, deferred in v0.0.7, is runnable.

### Tests (`pytest -q tests/`, 235 tests)

The 34 physics tests of v0.0.7, 196 configuration tests (selection, data, model, training, dispatch — the expensive
parts mocked) and 5 fragment-neutrality tests.

## v0.0.8 → v0.0.9 — E3-Q set up: q-AQUA v1 as the truth (`sseft/qaqua.py`, `external/qaqua/`, `configs/water_qaqua.yaml`)

No experiment run yet; pre-registered in `notes/e3q-preregistration.md` with the numbers computed from the PES
(`scratch/e3q_preregistration.py`).

**The truth.** q-AQUA v1 (Yu et al. 2022) built with gfortran through a C interface that initialises once and re-sizes
q-AQUA's work arrays for any number of waters (`external/qaqua/build.sh`, `qaqua_capi.f90`): the authors' trimer and
prism-hexamer benchmarks reproduced to the printed 1e-8 in every energy and gradient component; two monomers 12 Å
apart interact by exactly their LTP2011 point charges (4e-13); the energy's rounding noise is 2e-11 (dimer) to 8e-11
(hexamer) hartree, so finite differences below 1e-3 bohr are noise-dominated while the gradient integrates to the
energy to within the noise. On rigid monomers at the harness geometry the interaction energy is q-AQUA minus its 1-body
term (`sseft.water.QAquaTruth`); the generators of `sseft.water` take a truth (PW by default, bit-for-bit as before:
`tests/test_water_truths.py` holds values recorded from v0.0.8).

**The experiment.** `configs/water_qaqua.yaml` runs (a) the dimer ladder at φ = 0 (R_e = 2.9346 Å; window
[3.96, 7.13] Å; r_c = 5 Å for 12 identifying nodes; M_∞, M_G, M_A, M_1, M_1μ, M_1†; K = 8) and (c) the clusters (the
E3-R design and schedule at ℓ₁ = 0.75 Å; M_∞, M_1†, M_S0†, M_S2†, M_1μ; K = 4), every cluster model also read on the raw
CCSD(T) 2-/3-/4-body sets of the q-AQUA repository (near-rigid frames inside the training domain: 4940 / 4673 / 267),
the 3-body set by largest O–O distance — beyond the PES's switch at 7–9 Å the CCSD(T) labels are not zero, the PES is.
The harness's configuration gained several water parts in one file (`experiment.part` as a list, per-part overrides
under `parts`, `run_experiment.py --out_dir` running them in sequence after `--validate_only` checks every part), the
water `truth` mapping and the dimer design as a `data` mapping; trained parameters are saved per seed on q-AQUA.

**The predictions that matter** (§4 of the note): M_1† exact on the tail (≤ 1e-8 from the 8 Å row); M_1's learned
dipole biased by the window, not the class — the identifying nodes lie in the CCSD(T) fit, 13–24 % away from the point
charges, and the class minimum with a linear charge response is 1.763 D against 1.851 (registered 1.71–1.82 D), the
alternatives 1.47 D (the response not learned) and 2.10 D (forces ignored) being diagnostic; M_S0† at 25–55 % of the
hexamers' non-additive energy (0.915 kcal/mol; fixed-coefficient floor 20 %); and on the raw CCSD(T) trimers beyond
the PES's switch (7–9 Å: the PES 48 %; beyond 9 Å: 100 %), M_S0† below the PES — a model trained on q-AQUA carrying
many-body physics that q-AQUA drops.

### Tests (`pytest -q tests/`, 279 tests; the 10 q-AQUA tests are skipped without the build)

v0.0.8's 236, the q-AQUA driver and truth (benchmarks, atom order, separated monomers, the gradient by path
integration, the exact many-body decomposition, the pre-registered R_e / dipole coefficient / charges / raw-set count,
the model-side subset expansion vanishing for a pairwise model), the multi-part configuration (merging, refusals,
`--validate_only`, the dispatcher's validate-then-run sequence), and the PW generators against v0.0.8's values.

### Build fix: the q-AQUA library carries its Fortran runtime (no numerical change; 285 tests)

The first HPC run stopped at `ctypes.CDLL`: `libgfortran.so.5: cannot open shared object file`. `libqaqua.so` was
linked against the runtime of the compiler that built it, which the build's shell could see (the compiler's module or
conda env) and the run's could not. `external/qaqua/build.sh` now copies the runtime the library needs (libgfortran,
and libquadmath where the compiler uses it) to `<build>/runtime`, links with that directory in a DT_RPATH
(`$ORIGIN/runtime`: it also serves libgfortran's own libquadmath and is not overridden by `LD_LIBRARY_PATH`), and fails
unless `ldd` resolves every library with `LD_LIBRARY_PATH` unset; `--relink` repairs an earlier build in seconds (no
download, q-AQUA not recompiled). `sseft.qaqua.load_library` preloads the bundled runtime should the RPATH not serve
and otherwise says how to repair the build, and `qaqua.py --check` prints which runtime was loaded. Checked with the
system's libgfortran and libquadmath removed (a runtime needing libquadmath, as GCC ≤ 12's): the benchmarks and the
q-AQUA tests pass on the relinked build. Tests: `tests/test_qaqua_loader.py` (the loader with a fake `ctypes.CDLL`,
the script's syntax) and the build's runtime resolution in `tests/test_qaqua.py`.

## v0.0.9 → v0.0.10 — E3-Q read; E3-QP set up: q-AQUA-pol as the truth, hundreds of training clusters, the probe reading

**E3-Q read** (`notes/e3q-results.md`, the run of 14 Oct): the far-field reading registered as the sharpest held —
beyond 9 Å, where q-AQUA's 3-body term is zero, M_S0† and M_S2† cut the error on the raw CCSD(T) trimers from 100 to
69 % on every seed — but everywhere the truth has many-body energy it lies within E₀'s reach, and M_∞ beat every
long-range rung on the hexamers (54 % against M_S0†'s 70 %); M_1's dipole (1.683 ± 0.058 D against 1.71–1.82) tracked
the window fit across seeds (unconverged at L-BFGS 300). q-AQUA v1 has no polarisation at long range, so it cannot be
the truth of a long-range polarisation test.

**The truth.** q-AQUA-pol (Yu et al., JCTC 2023; github.com/jmbowma/q-AQUA-pol, CC0): TTM3-F everywhere — site
charges on H and an M-site, one Thole-smeared polarisable site per molecule with self-consistent induced dipoles and
no cutoff, an O–O C₆ term — plus CCSD(T) corrections switched off by 7 Å. Built by `external/qaqua/build_pol.sh`
through `qaqua_pol_capi.f90` (v1's names and conventions, plus TTM3-F's pieces and induced dipoles); the authors'
hexamer reproduced (6e-9 kcal/mol, gradients within 5e-9), TTM3-F's electrostatics, induction, C₆ term and induced
dipoles at 12 Å recomputed independently to 2e-13. Against the raw CCSD(T) trimers its error is 22 % at 7–9 Å and
21 % beyond 9 Å (v1: 48 and 100 %). `sseft.qaqua.QAquaPol`, `sseft.water.QAquaPolTruth` (`truth.name: qaqua_pol`);
the pinned charges are the dipole-surface charges (TTM3-F's dipole, not its M-site quadrupole: no dagger is exact on
this tail, and the daggers' errors are computed), the M_16 daggers pin TTM3-F's α and C₆, the embedded daggers are
refused. The build scripts share `external/qaqua/runtime.sh` (the runtime bundling of the v0.0.9 fix).

**The experiment** (`configs/water_qaqua_pol.yaml`, `notes/e3qp-preregistration.md`): the dimer ladder as E3-Q plus
M_16†, L-BFGS 1000; the clusters with 300 training clusters and 160 probe configurations (a relaxed trimer and a
molecule at 8–15 Å; `data.probe_*`, any truth), 24 test hexamers and dimers and 80 test probes, rungs M_∞, M_1†,
M_S0†, M_S2†, M_1μ, M_16†, M_16S0†, Adam 2000 (batch 32) → linear solve → L-BFGS 500. The probe reading — the probe's
interaction and its non-additive part, the long-range many-body induction — is new; the raw CCSD(T) readings are
E3-Q's.

### Tests (`pytest -q tests/`, 306 tests; the 18 q-AQUA and q-AQUA-pol tests are skipped without their builds)

`tests/test_qaqua_pol.py` (the benchmark and TTM3-F's long range, the corrections vanishing beyond 7 Å, the gradient
by path integration, the pre-registered constants, the probe reading on the truth, the held-out counts) and
`tests/test_water_pol_config.py` (the refusals, the probe data's validation and settings, the shipped config, the
pinned oscillator with a truth's α and C₆, the PW probe generator against values recorded at 4a81a84).

## v0.0.10 → v0.0.11 — E3-QP read; E3-QP2 set up: the truth's charges exactly, the response matched on the far field, the mutual response

**E3-QP read** (`notes/e3qp-results.md`, the run of 15 Oct): the read-out carries far-field many-body induction — on
the probes beyond r_c every read-out rung removes 40–50 % of the non-additive energy that M_∞ and the daggers cannot
touch, and beyond 9 Å M_S0† cuts the CCSD(T) trimers' error from 100 to 57 % — but at half strength, learned from the
clusters rather than the far field (worse than a fixed first-order form, training probes = test probes, order 2 ≈
order 0). The daggers reproduced their computed tails; M_1μ's dipole and tail are the best learned-source result so
far (1.90 D; 1–4 % from 10 Å). Found after the run: the M_16 rungs' raw 2-body readings lacked their dispersion band
(a reading error, fixed: every reading applies the rung's pins; `scratch/e3qp_reread_2b.py` re-reads the saved models);
q-AQUA-pol's energy-only routine leaves the switch of its 3-body blend (largest O–O 5.0–5.5 Å) unset — the labels came
from the gradient routine and are unaffected, the PES's reference error in the 3-body 5–7 Å bin was 0.0097 for 0.0084
kcal/mol; and, computed on the truth, the probes' non-additive energy is TTM3-F's induction, first order to 90 %, through
α²T to 99 %.

**The harness.**
- *Site models as pinned sources* (`Rung.msites`; `Structure.pinned` q_site, vsite_w, vsite_q, vsite_frag): charges on
  the atoms and on virtual sites placed linearly from them, in the Coulomb band and the band field (sources other than
  the evaluation points: `bandfields.band_field_tensors_from`); `QAquaPolTruth.site_model` gives TTM3-F's H charges and
  M-site. On q-AQUA-pol the dagger `M_16pm` misses the dimer tail by the induction alone (0.6 % at 7.8 Å, against 14 %).
- *The mutual response* (`Rung.bf_mutual`; `bandfields.band_dipole_tensor`, `mutual_response`): the order-0 read-out
  written as local polarisability tensors C_i, their induced dipoles coupled through the band kernel's dipole tensor
  between fragments — self-consistent (−1) or the series through order k; at order 1 it is the read-out of before to
  rounding.
- *The species read-out* (`Rung.bf_species`, `bandfields.init_readout_species`): coefficients from fixed species
  features with a fixed unit, independent of E₀ and of the seed.
- *Subtraction groups* (`Structure.sub`): the model's energy E(all) − Σ_g E(group g) — a probe's interaction energy —
  and `loss.far_labels: interaction` labels the training probes so.
- *The solve* (`sseft.linear`): every order's columns (`orders="all"`), Gauss–Newton steps (`iterations`), the ridge
  chosen on held-out groups and kept (`ridge_grid`, `groups`), the Jacobian by reverse mode per structure (1 + 3N
  passes instead of one per coefficient; energies only: one); `T.fit` takes structure weights, a subset, the forces'
  weight in the solve, `readout_first` (the solve before Adam) and `lbfgs_freeze_readout`; `readout.stage: far_first`
  matches the read-out on the training probes beyond r_c, energies only.
- *The truth*: q-AQUA-pol's energy is its gradient routine's (`QAquaPol.interaction_energy`); the energy-only routine
  drives the Metropolis relaxation only (`sampling_energy`), so the data sets reproduce; `QAquaPol.induction` re-evaluates
  TTM3-F's induction from its sites, self-consistent or through order k in α (to 1e-15 of the library).
- `model.response_edge_angstrom` (the read-out's own band edge), training probes on cores of several sizes
  (`data.probe_train_core_sizes`), the C rungs' probes read at first and second order of their own response.

**The experiment** (`configs/water_qaqua_pol2.yaml`, `notes/e3qp2-preregistration.md`): the dimer's exact daggers;
the clusters with E3-QP's data plus 384 training probes on 48 cores of 3–5 molecules, rungs M_∞, M_16†, M_16†ᵐ, M_16C0†
(the control: q3), M_16S0†ᵐ, M_16C0†ᵐ, M_16S2†ᵐ, M_16C2†ᵐ. The matched responses are computed before the run: on the
test probes at 10–15 Å they leave 3–11 % of the non-additive energy (M_16C2†ᵐ 4 / 5 / 3 %; E3-QP's read-outs 50–61 %),
on the CCSD(T) trimers beyond 9 Å 17–25 % (the PES 21 %), and the same response on q3 19–24 %.

## v0.0.11 → v0.0.12 — E3-QP2 read; the short range diagnosed; E3-QP3 set up

**E3-QP2 read** (`notes/e3qp2-results.md`, the run of 19–20 Oct): every far-field reading computed before the run came
out on every seed — the exact daggers' dimer tails, the probes at 10–15 Å for every rung (M_16C2†ᵐ 4 / 5 / 3 %; E3-QP's
read-outs 50–61 %), the C rungs' truncations — and on the CCSD(T) trimers beyond 9 Å the read-outs sit at the PES's own
error (17–25 % against 21 %; E3-QP 57–59 %). The self-consistent response pays off through E₀: C beats S on every seed
on the hexamers (M_16C2†ᵐ 15.8 %, the only rung clearly below M_∞'s 22.5) and on the raw 4-body (63 against 110–139 %).
Short of it: the raw 2- and 3-body readings below 5 Å, where every pinned rung trailed M_∞.

**The short range diagnosed** (`scratch/e3qp3_shortrange_diagnostic.py` on the saved models, 21 Oct; §3.4–3.5 of the
results): the raw frames are near-rigid, not rigid (the truth moves 0.090 kcal/mol rms on the 2-body set with the
monomers' distortions); the raw sets were read with the pinned charges at each frame's own geometry while E₀ learned the
truth's charge flux from the internal forces (72 % of the training forces) — a double count, 0.02–0.04 kcal/mol of the
raw 2-body reading; read with the charges the models were trained with, every pinned rung beats M_∞ on the raw dimers at
O–O ≥ 2.7 Å (0.103–0.151 against 0.159 kcal/mol), and 56–82 % of their squared error sits in the 5.6 % of compressed
dimers at 2.3–2.7 Å, where the training clusters hold 28 pairs; E₀ fits its own training clusters to 0.17–0.26 kcal/mol,
the test errors' size, the loss's per-atom energy term two orders of magnitude below its force term.

**The harness.**
- *Contact configurations* (`W.contact_configurations`; `data.contact_dimers`, `contact_trimers`, `contact_oo_angstrom`,
  `contact_seed`, `contact_max_pair_kcal`, `contact_min_atom_distance_angstrom`): rigid dimers and trimers at a
  hydrogen-bond contact or closer under a pair-energy cap, added to the clusters part's training set; without them the
  dataset settings are those of before.
- *The pins as trained* (`truth.heldout.pin_charges: rigid`, `--heldout_pin_charges`; `model_nbody(rigid=True)`,
  `W.pin_site_model(rigid=True)`, `W.rigid_molecule_charges`): the held-out frames read with the rigid monomer's charges;
  `frame` (the default) keeps v0.0.11's reading.
- *Readings by the closest contact* (`QA.smallest_oo`; `heldout_metrics` `by_smallest_oo_angstrom`: below 2.7, 2.7–3.0,
  beyond 3.0 Å).
- *Rungs with their own band edge* (`e3_water.RUNG_BAND_EDGE`): `M_16pm_e1`, `M_16C2pm_e1` at ℓ₁ = 1.0 Å, the read-out at
  the run's band edge.
- `scratch/e3qp3_shortrange_diagnostic.py` reads any clusters run: its rungs, seeds and sets from the command line or
  the run's `clusters.json`, its flags checked against each rung's `model` record.

**The experiment** (`configs/water_qaqua_pol3.yaml`, `notes/e3qp3-preregistration.md`): E3-QP2's data and matching plus
600 contact dimers and 200 contact trimers at O–O 2.3–3.0 Å, the energy weight 100, Adam 4000 and L-BFGS 1000, the pins as
trained in every reading; rungs M_∞, M_16†ᵐ, M_16†ᵐ_e1, M_16C2†ᵐ, M_16C2†ᵐ_e1, 4 seeds; the diagnostic on every model after
the run (`run_e3qp3.sbatch`).

Tests: `tests/test_water_pol3_config.py` (the shipped config, the contact generator and its refusals and validation, the
rungs' own band edge, the readings' pins as trained, the bins by the closest contact); 362 tests, 340 pass and 22 are
skipped without the q-AQUA builds (14 min on one CPU).

## v0.0.12 → v0.0.13 — E3-QP3 read; E3-QP4 set up

**E3-QP3 read** (`notes/e3qp3-results.md`, the run of 22 Oct): the 2-body short range is fixed — the compressed raw
dimers (O–O 2.3–2.7 Å) at 0.11–0.13 kcal/mol for every rung (E3-QP2 0.41–0.95), every pinned rung below M_∞ on the raw
dimers on every seed (0.090–0.098 against 0.113) — and the far field came out exactly as computed. Short of their
registered values: E₀'s own training fit (0.13–0.16 for the pinned rungs; the energy weight traded the forces for the
energies), the hexamers (M_16C2†ᵐ 0.189), the test dimers and the raw 3-body, where the pinned rungs trail M_∞ (0.14–0.15
against 0.10), most at the closest contacts. Computed after the run: the matched read-out's 3-body is TTM3-F's induction
beyond 3 Å but 0.83 of it at the hydrogen-bond contact and 0.68 below 2.7 Å, and the loss's 1 / n_atoms² weighting let
the 800 contact configurations carry 58–77 % of the energy term.

**The harness.**
- *The contact configurations' energy weight* (`training.loss.contact_energy_weight`, `--w_contact_energy`;
  `Structure.energy_weight`, applied by `e3_water.weight_contact_energies`): a weight on their energies alone — the
  energy term is a weighted mean over structures in the loss (`train.make_loss_sums`) and in the read-out's solve
  (`linear.solve_readout`); their forces keep their weight. Refused without contact configurations.
- *A rung with the read-out's own band edge* (`e3_water.RUNG_RESPONSE_EDGE`): `M_16C2pm_r1`, M_16C2pm's read-out at
  1.0 Å, the pinned bands at the run's.
- `scratch/e3qp2_matched_response.py` and `scratch/e3qp3_response_contact.py` take E₀'s width (`--F=`; the species
  read-out's random features share it) and the read-out's band edge (`--edge=`).

**The experiment** (`configs/water_qaqua_pol4a.yaml`, `configs/water_qaqua_pol4b.yaml`, `notes/e3qp4-preregistration.md`):
two jobs on E3-QP3's data, matching and readings, rungs M_∞, M_16†ᵐ, M_16C2†ᵐ, M_16C2†ᵐ_r1, 4 seeds. Job A: E₀'s width
32 (E3-QP3: 16). Job B: job A with the contact configurations' energies at weight 0.125 (equal weight with the relaxed
clusters at equal errors). Computed before training: at width 32 the matched far field is E3-QP3's at the printed digit
(M_16C2†ᵐ_r1: 3 / 3 / 2 % at 10 / 12 / 15 Å), the read-out's near field a little farther from TTM3-F's (the hexamers'
many-body 29 against 21 %), and the 1.0 Å edge brings it closer (15 %).

Tests: `tests/test_water_pol4_config.py` (the two shipped configs and that each job is one change, the rung with the
read-out's own band edge, the contact energy weight's field, validation and application, the weighted energy term in
the loss and the solve, the contacts weighted alone in a cluster run); 371 tests, 349 pass and 22 are skipped without
the q-AQUA builds (14 min on one CPU).

## v0.0.13 → v0.0.14 — E3-QP4 read; E3-QP5 set up

**E3-QP4 read** (`notes/e3qp4-results.md`, both jobs of 22 Oct on one H200 each; 43 of 55 checks as registered).
Job A (E₀'s width 32): every rung's training energies and forces improved together (−10 to −19 %), the test errors less
(−3 to −16 %), the raw 4-body rose — the width a small limit. Job B (the contact configurations' energies at weight
0.125): every pinned rung improved on all but the contacts — the relaxed training clusters −26 to −33 %, their forces −8
to −12 %, the hexamers −7 to −10 %, the raw 3-body −4 to −8 % — and the compressed dimers held; M_∞ traded its 2-body
(+19 %) for its clusters, its hexamers flat. M_16C2†ᵐ of job B is the best model of the series: hexamers 0.160 kcal/mol
(E3-QP2 0.201), their non-additive energy 13.1 % (M_∞ 22.8 %), raw 2-body 0.082 (M_∞ 0.113). The pinned rungs' raw
3-body over M_∞'s did not move with either change (1.33–1.53): M_16†ᵐ's deficit is E₀'s, largest on the near-rigid
trimers' rigid part (1.51×), narrowed there by the C rungs' read-out (1.25 / 1.15×), which adds its own below O–O 2.7 Å.
The read-out at 1.0 Å (M_16C2†ᵐ_r1) is a trade: better on the raw sets on 7–8 of 8 seeds, worse on the hexamers on 6–7
of 8. The far field as computed in both jobs, to the same digits.

**The harness.**
- *E₀'s view of the pinned field* (`Rung.e0_field`, `e0_field_on`, `e0_field_off`, `e0_field_unit`): E₀'s site energies
  read the potential, field and field gradient of the other fragments' site-model charges at every atom
  (`bandfields.local_field_tensors`: the band kernel at ℓ₁, analytic, every source's share weighted by `bandfields.taper`,
  a C² smoothstep that ends inside E₀'s cutoff), through `heads.field_energy`: an MLP of E₀'s invariants, the potential,
  E₀'s vector and tensor channels contracted with the field and its gradient, and the two squares, less the same MLP at
  zero field — the term vanishes where no source is in reach and starts at zero (its output layer), its parameters
  (`params["field"]`) drawn on their own key so that every other parameter is the base rung's. Refused without a site
  model.
- *Rungs* (`e3_water`): `M_16pm_f`, `M_16C2pm_f` (the bases with the field term) and `M_inf_f` (the field term on the
  pinned charges with no bands, `Rung.coulomb_energy` off); the taper at r_c − 1.5 to r_c − 0.5 Å
  (`e3_water.E0_FIELD_TAPER`, set by `configure`); the model record carries the taper, the unit and an absent Coulomb band.
- `scratch/e3qp3_shortrange_diagnostic.py`: `--save_frames` writes every frame's n-body energies (each model's on the
  frame, its rigidified twin and with the pins as trained; the labels, the truth's, the positions and O–O distances; the
  hexamers' energies and 2-body parts) to `<out>_frames.npz`; the run's taper checked against the flags.

**The experiment** (`configs/water_qaqua_pol5.yaml`, `notes/e3qp5-preregistration.md`): E3-QP4's job B with rungs
M_16†ᵐ, M_16†ᵐ_f, M_16C2†ᵐ, M_16C2†ᵐ_f, M_∞_f, 4 seeds; M_∞ is job B's, the bases run again as the controls; the
diagnostic with `--save_frames` after the run (`run_e3qp5.sbatch`). Computed before training
(`scratch/e3qp5_preregistration.py`): every probe at 10–15 Å lies beyond the field term's reach, so the far field there is
the bases'; on the 4673 raw trimers one effective polarisability per species on the input's first-order cross term
leaves 0.114 of the 3-body's 0.623 kcal/mol rms (82 % explained; below O–O 2.7 Å 0.143 of 1.268) — as well as M_16†ᵐ's
E₀ does (0.115) — where TTM3-F's whole 3-body induction explains 54 %; a training step costs 0.9–1.1 times the base's
(`scratch/e3qp5_field_timing.py`).

Tests: `tests/test_e0_field.py` (the rungs that existed against v0.0.13's golden energies and forces; the field rungs
configured and recorded; a field rung at initialisation its base, digit for digit; the term local — a probe beyond r_c
and an isolated molecule as with the base rung, with every parameter switched on; the tapered shares against the band
field by autodiff and cut beyond the taper; the taper C²; forces against central differences; invariance under rotation,
translation and the molecules' order) and `tests/test_water_pol5_config.py` (the shipped config is job B's but for its
rungs; refused without the site model); 382 tests, 360 pass and 22 are skipped without the q-AQUA builds (13 min on
one CPU).

## v0.0.14 → v0.0.15 — E3-QP5 read; E3-QP6 set up

**E3-QP5 read** (`notes/e3qp5-results.md`, two jobs of 23 Oct split by rung, one H200 each, 2.5 and 2.8 h; 18 of 24
checks as registered). (b) holds: given the other molecules' pinned field, M_16†ᵐ_f's raw 3-body on the near-rigid
trimers 0.110 → 0.062 kcal/mol (paired −32 to −48 %), all raw trimers 0.117 → 0.078, below O–O 2.7 Å 0.195 → 0.127 —
level with or below M_∞'s at every contact — at no registered price (raw 2-body −12 %, hexamers −11 %, training clusters
−10 %, raw 4-body 120 → 82 %); M_∞_f's 3-body no better (strict +15 %), its 2-body halved: the pinned rungs' piece. The C
rung gained at contact (raw 3-body −31 %, 2-body −29 %, training clusters −17 %) and lost its hexamers (0.155 → 0.178 on
every seed), its 4-body (57 → 81 %) and the trimers beyond 9 Å (27 → 42 %); M_16C2†ᵐ stays the best model. The far field
as computed. Two faults of the field term, read after the run (`scratch/e3qp5_reach_scan.py`, from the PES and the data
alone): each source tapered by its own distance leaves a molecule straddling the taper part charged (27–38 % of the
atom–molecule pairs, 0.3 e rms) — two polarisabilities on that input predict the extended trimers' 3-body worse than
none, and M_16†ᵐ_f's 3-body there equals their residual to the printed digit; and the pinned field rungs' 2-body just
past the contact data's edge (3.0–3.2 Å +72 / +47 %), which is not the taper's.

**The harness.**
- *Every molecule whole* (`Rung.e0_field_whole`; `bandfields.local_field_tensors(..., frag_src=)`): all of a fragment's
  sources share one weight at a point, the product of their tapers — 1 when every site lies within the taper's start, 0
  when any lies beyond its end — so that a fragment enters with its charges in their proportions (a neutral molecule
  neutral). The rungs' taper r_c − 1 to r_c (`e3_water.E0_FIELD_TAPER_WHOLE`, set by `configure`): no source beyond r_c
  in reach.
- *The field term at degree 2* (`Rung.e0_field_form: "quadratic"`, `heads.init_field_head2`, `heads.field_energy2`):
  −½ e_scale (E·A·E + G:B:G), A = a + Σ b_f v_f v_f + Σ b′_f t_f and G:B:G = c |G|² + Σ d_f (t_f:G)², the coefficients
  an MLP of E₀'s invariants starting at zero — the read-out's order-0 form on the local field, and its square in the
  field gradient; exactly quadratic in the pinned charges. `"mlp"` (the default) is v0.0.14's term, unchanged.
- *Rungs* (`e3_water`): `M_16pm_w`, `M_16C2pm_w` (the field term with every molecule whole), `M_16C2pm_w2` (and at
  degree 2); the model record carries `e0_field_weights` and `e0_field_form` where they differ from v0.0.14's, so that
  E3-QP5's records read as before. `scratch/e3qp3_shortrange_diagnostic.py` checks them against the rung.

**The experiment** (`configs/water_qaqua_pol6.yaml`, `notes/e3qp6-preregistration.md`): E3-QP5 with its field rungs
replaced — M_16†ᵐ, M_16†ᵐ_w, M_16C2†ᵐ, M_16C2†ᵐ_w, M_16C2†ᵐ_w2, 4 seeds; the bases run again as the controls; the
diagnostic with `--save_frames` after the run (`run_e3qp6.sbatch`; two jobs, each base with its field rungs). Computed
before training (`scratch/e3qp6_preregistration.py`, with the harness's own tensors): every probe at 10–15 Å lies beyond
the whole-molecule term's reach, so the far field there is the bases'; on the 4673 raw trimers one polarisability per
species on the cross term leaves 0.078 of the 3-body's 0.623 kcal/mol rms with every molecule whole (E3-QP5's taper
0.114, none 0.066), and nothing worse than predicting no 3-body where the farthest molecule is 7 Å or more away (0.023
against the labels' 0.027 at 7–9 Å; E3-QP5's 0.039); below O–O 3.2 Å every dimer's partner is whole in reach, so the
input at the contact data's edge is E3-QP5's. A training step costs 1.00–1.04 times the base's
(`scratch/e3qp6_field_timing.py`).

Tests: `tests/test_e0_field_whole.py` (E3-QP5's field rungs against v0.0.14's golden energies and forces; the new rungs
configured and recorded, E3-QP5's records unchanged; a new rung at initialisation its base, digit for digit; the tensors
with every molecule whole against each molecule's whole field times the product of its sites' tapers, computed apart, and
every molecule neutral as seen from any atom; the term local — a probe beyond r_c and an isolated molecule as with the
base rung; the degree-2 term exactly quadratic, in the head and in the rung's charges; forces against central
differences with a molecule straddling the taper; invariance) and `tests/test_water_pol6_config.py` (the shipped config
is E3-QP5's but for its rungs; refused without the site model); 394 tests, 372 pass and 22 are skipped without the
q-AQUA builds (21 min on one CPU).

## v0.0.15 → v0.0.16 — E3-QP6 read; E3-QP7 set up

**E3-QP6 read** (`notes/e3qp6-results.md`, one job of 25 Oct on an A100, 7.8 h + 1.6 h; 29 of 38 checks as
registered). (i) holds: with every molecule whole the extended trimers, the trimers beyond 9 Å, the C rung's 8 Å probes
and its hexamers are back at the bases'. The edge fault (ii) went with them (O–O 3.0–3.2 Å −16 / +4 % against E3-QP5's
+72 / +47 %): it was the part charges' too, through what the MLP learned elsewhere; the 2-body at 5–7 Å is the MLP
form's. M_16†ᵐ_w: near-rigid 3-body 0.111 → 0.043 kcal/mol (M_∞ 0.072) at no registered price. M_16C2†ᵐ_w is the best
model so far: raw 3-body −38 %, raw 2-body −39 %, training clusters −26 %, hexamers level, the far field as computed; its
costs the raw 4-body (58 → 75 %) and the 2-body at 5–7 Å. M_16C2†ᵐ_w2 (degree 2) keeps the 4-body (55 %) and loses the
2-body past the contact data (3.0–3.2 Å +87 %, off the training pairs +56 %) and the hexamers (+8 %). Computed after the
run (`scratch/e3qp6_self_term.py`): a function of the summed field ties its self terms (a 2-body) to its cross terms
(the 3-body); at degree 2 one coefficient sets both, and the self term the trimers' two numbers bring has a spread on
the raw dimers 3.6–7.9 times the base's 2-body error at every contact bin.

**The harness.**
- *Every molecule's share apart* (`bandfields.local_field_tensors(..., frag_src=, per_frag=True)`): the potential, field
  and field gradient of each source fragment at every point, with its whole-molecule weight, and where it is in reach;
  the totals then the sums of the parts, so that where one fragment is in reach the total is its part to the last digit.
- *The field term without its 2-body* (`Rung.e0_field_cross`, with `e0_field_whole`; `heads.field_energy(..., parts=)`,
  `heads.field_energy2(..., parts=)`): the term less its value on each source fragment alone, ε_i = h(s_i; x_i) −
  Σ_B h(s_i; x_i(B)) — nothing to any dimer, only what the fragments' fields do together; at degree 2 the cross terms
  −½ Σ_{B≠C} [E_i(B)·A_i·E_i(C) + G_i(B):B_i:G_i(C)], the leading non-additive induction. The rungs without it unchanged
  to the last digit.
- *Rungs* (`e3_water`): `M_16C2pm_wx` (M_16C2pm_w less each molecule's own value) and `M_16C2pm_w2x` (M_16C2pm_w2 with
  the cross terms only); the model record carries `e0_field_cross` where set, so that E3-QP5's and E3-QP6's records
  read as before; `scratch/e3qp3_shortrange_diagnostic.py` checks it against the rung.

**The experiment** (`configs/water_qaqua_pol7.yaml`, `notes/e3qp7-preregistration.md`): E3-QP6 with its rungs replaced —
M_16C2†ᵐ, M_16C2†ᵐ_w, M_16C2†ᵐ_wx, M_16C2†ᵐ_w2x, 4 seeds, paired in one job; the diagnostic with `--save_frames` after
the run (`run_e3qp7.sbatch`). Computed before training (`scratch/e3qp7_preregistration.py`, through the per-molecule
tensors): the trimers' two numbers on the cross term (0.078 of the 3-body's 0.623 kcal/mol rms left, E3-QP6's to the
printed digit) carry 83 % of the test hexamers' non-additive energy's rms (correlation +0.974, 0.166 of 0.997 left) and
81–83 % of the training clusters'; the self terms they bring, which M_16C2†ᵐ_w2's E₀ had to cancel, spread more than the
non-additive energy itself in every set (the clusters 0.72–1.11 against 0.43–0.89 kcal/mol, the hexamers 1.31 against
0.997); on the 4937 raw dimers the cross term is zero to the last digit. A training step costs 1.08–1.10 times the
base's on the CPU (`scratch/e3qp7_field_timing.py`); about 8.3 h on an A100.

Tests: `tests/test_e0_field_cross.py` (E3-QP6's field rungs against v0.0.15's golden energies and forces; the cross
rungs configured and recorded, the records before unchanged; a cross rung at initialisation its base, digit for digit;
the per-molecule tensors against each molecule's whole field times its weight, computed apart; the heads with parts
against the cross terms written out and against the head without parts, and nothing where one fragment is in reach; a
cross rung's dimers — at contact, straddling the taper, beyond it — and monomer M_16C2pm's, energy and every force; the
term local; the degree-2 cross term exactly quadratic in the rung's charges; forces against central differences with a
molecule straddling the taper; invariance) and `tests/test_water_pol7_config.py` (the shipped config is E3-QP6's but for
its rungs; refused without the site model); 408 tests, 386 pass and 22 are skipped without the q-AQUA builds (27 min
on two CPU cores).

## v0.0.16 → v0.0.17 — E3-QP7 read and decomposed; E3-QP8 set up

**E3-QP7 read** (`notes/e3qp7-results.md`, one job of 25–26 Oct on an A100, 8.1 h + 1.4 h; 33 of 39 checks as
registered). The MLP form's self terms carried its whole 2-body gain (M_16C2†ᵐ_wx's raw 2-body 0.084, M_16C2†ᵐ's;
M_16C2†ᵐ_w 0.051) and part of its many-body gain (the hexamers' non-additive part 15.7 against 11.3 %, every seed); its
5–7 Å loss went with them. The degree-2 cross terms are the best contact 3-body of the C rungs (raw 3-body 0.078, below
O–O 2.7 Å 0.142), most of the degree-2 rung's edge loss gone with its self terms, but the hexamers' non-additive part
15.1 % and the raw 4-body 76 % (58 %). M_16C2†ᵐ_w stays the best model, reproduced within 5 %.

**The decomposition** (`scratch/e3qp7_decomposition.py`, `notes/e3qp7-decomposition.md`; one job of 26 Oct, 11.5 min,
all 36 models of E3-QP7 and E3-QP6): every model's n-body energies on the test hexamers' sub-clusters and the raw sets
split into E₀, the pinned bands, the read-out and the field term — the field term into its self and cross terms and into
its value at the isolated molecule's features (fixed coefficients) and the rest. Every check held (the components sum to
the energy and to the diagnostic's, ≤ 1.6e-12 kcal/mol; no 3-body in the bands or the self terms at fixed coefficients,
no 4-body in the degree-2 cross terms there). M_16C2†ᵐ_w2x's 4-body loss is its cross terms' coefficients, not E₀ beside
them (without the field term its 4-body error 0.073 is below the base's 0.076); every field rung's 4-body loss is its
field term's own. The MLP self terms are a 0.27 rms 2-body that E₀ cancels at −0.98; their 5–7 Å loss is their response
to the potential there at fixed coefficients (still 12 % of its contact value at 5–6 Å;
`scratch/e3qp7_field_inputs.py`), the degree-2 self terms 90 times smaller there. Their own many-body in the hexamers is
small. The read-out, the same in every C rung of a seed, carries 0.93 of the hexamers' 0.98 3-body and overshoots the
compressed tetramers' 4-body (0.57 against 0.34).

**The harness.**
- *The self terms at their own reach* (`Rung.e0_field_self_on / _off`, with `e0_field_whole` and the MLP form;
  `bandfields.local_field_tensors(..., self_taper=)`: every fragment's own whole-molecule weight over the shorter reach;
  `heads.field_energy_split`): ε_i = h(s_i; x_i) − Σ_B (reach_iB − w_iB) h(s_i; x_i(B)) — with w = reach
  `field_energy`'s term, with w = 0 its cross terms alone.
- *Cross terms by species* (`Rung.e0_field_cross_species`, `heads.field_cross_species`, parameters `field_x` starting at
  zero): the MLP's cross terms replaced by −½ e_scale Σ_{B≠C} [a_Z E_i(B)·E_i(C) + c_Z G_i(B):G_i(C)] beside its self
  terms — strictly 3-body, no gradient into E₀ through them.
- *Rungs* M_16C2pm_ws (M_16C2pm_w, its self terms over the last 2–1 Å inside r_c: `E0_FIELD_SELF_TAPER`) and M_16C2pm_us
  (those self terms beside the species cross terms); the model record carries `e0_field_self_taper_angstrom` and
  `e0_field_cross_form` where set, so that the earlier records read as before; the short-range diagnostic checks them
  and the decomposition reads the new rungs' channels (`F_self`, `F_cross` in place of v0.0.16's `F_full`, `F_own`).
  E3-QP7's cross rungs and every rung before unchanged to the last digit (golden values from v0.0.16).

**The experiment** (`configs/water_qaqua_pol8.yaml`, `notes/e3qp8-preregistration.md`): E3-QP7 with its rungs replaced —
M_16C2†ᵐ, M_16C2†ᵐ_w, M_16C2†ᵐ_ws, M_16C2†ᵐ_us, 4 seeds, paired in one job; the diagnostic with `--save_frames` and the
decomposition after the run (`run_e3qp8.sbatch`). Computed before training (`scratch/e3qp8_preregistration.py`): the
4–5 Å self reach cuts 98 % of the potential the self terms read on the raw dimers at O–O 5–6 Å and at most 1.6 % below
3.2 Å (12 % at 3.2–4 Å; 10–17 % in the training clusters and the hexamers); four numbers by species on the cross terms
leave 0.0767 of the raw trimers' 0.6225 kcal/mol 3-body (two: 0.0775) and 0.160 of the hexamers' 0.997 non-additive
energy. A training step costs 1.38 (M_16C2†ᵐ_ws) and 1.18 times (M_16C2†ᵐ_us) the base's on the CPU
(`scratch/e3qp8_field_timing.py`); about 9 h on an A100.

Tests: `tests/test_e0_field_split.py` (E3-QP7's cross rungs against v0.0.16's golden energies and forces; the split
rungs configured and recorded, the records before unchanged; refused where undefined; at initialisation the base, digit
for digit, M_16C2pm_ws's parameters M_16C2pm_w's; the split head's limits (w = reach, w = 0, between) and the species
cross terms against the cross sums written out; the self weights against each molecule's taper computed apart; a dimer
beyond the self reach the base's, energy and every force; on a dimer the self terms M_16C2pm_w's times the weight; the
species cross terms nothing on a dimer and no 4-body; no gradient into E₀ through them; local; forces against central
differences; invariance), the decomposition's tests extended to the new rungs, and `tests/test_water_pol8_config.py`
(the shipped config is E3-QP7's but for its rungs; refused without the site model); 436 tests, 414 pass and 22 are
skipped without the q-AQUA builds (40 min on two CPU cores). A smoke run on tiny data (CPU): both new rungs through
training, the diagnostic with the frames, the decomposition (every check at 1e-12 kcal/mol or below, the species cross
terms nothing on the dimers and the 4-body) and the self-term check; pre-registration (c) on a stand-in for E3-QP7's
run, the same with and without its frames file.

## v0.0.17 → v0.0.18 — E3-QP8 read, E3 closed; E4 set up: MB-pol through MBX

**E3-QP8 read** (`notes/e3qp8-results.md`; one job of 26–27 Oct on an A100, 8.1 h of training; 31 of 35 checks as
registered).

- Cross terms by species ended the 4-body loss. M_16C2†ᵐ_us has a raw 4-body of 51.7 %, and the hexamers' 4-body
  0.042 and 5-body 0.0029: the lowest many-body tail so far.
- The self terms within 4–5 Å took most of the 5–7 Å tail, at a price:
  - half the 3.2–5 Å gain;
  - +18 % at 2.7–2.9 Å;
  - +15 % on the training internal forces.

**The split check** (`scratch/e3qp8_split_check.py`, §5 there; 91 s): the E₀–self-term split of the 2-body is a gauge.

- On 2000 rigid dimers in random orientations, M_16C2†ᵐ_us's 2-body error below 5 Å is M_16C2†ᵐ_ws's (−3 to +4 %
  paired).
- Its seeds disagree less (0.93×).
- Twin models' energy differences do not grow with their self terms' (R/S 0.02).

**E3 closed** (`notes/e3-summary.md`). What E3 established:

- The far field is set before training, and it holds. The read-out leaves 3–5 % of the far probes' non-additive
  energy, and 26.7 % on the CCSD(T) trimers beyond 9 Å. Frozen sources leave 100 %.
- Mutual induction is needed for the many-body energy.
- The field term closed the pins' cost at contact.
- The E₀–self-term split is a gauge.

The frozen model is M_16C2†ᵐ_us. Still open:

- the read-out's near field at the closest contacts, registered for E4;
- the self terms' reach;
- the coarse-grained density.

**E4 set up** (`notes/e4-plan.md`): the periodic water box, with MB-pol as the truth and DFT later.

**The truth** (`sseft/mbpol.py`, `external/mbx/build.sh`). MBX v1.4.0 (0e01b75) is called through its C entry points.
It gives, for open clusters and periodic cells:

- interaction energies, with the 1-body polynomial off;
- forces;
- the site model: charges, M-sites, polarisabilities and induced dipoles;
- the field at points;
- a uniform external field on the sites, and the induced dipoles' response to it.

E4's settings, which are the driver's defaults:

- a real-space cutoff of 6.5 Å in every cell, α 0.7 Å⁻¹, a grid of 6 Å⁻¹ and splines of order 8;
- cells narrower than twice the 6.5 Å reach are evaluated on their smallest replica (MBX takes the minimum image) and
  reported per cell.

**Checked** (`python sseft/mbpol.py --check`, `scratch/e4_mbx_checks.py`, on a −O0 build in the container):

- MBX's MB-pol example reproduced to 2e-9 kcal/mol.
- Tin-foil boundary conditions: one molecule in cubic cells gives 0.965 → 0.996 of −(2π/3)|μ|²/L³ at 20–60 Å, with an
  O(L⁻²) remainder.
- The supercell identity holds to 3e-11 kcal/mol where the cutoffs fit the cell.
- MBX's minimum image: the 64-molecule liquid cell misses 1.6e-2 kcal/mol of 2-body. The replica labels agree to 4e-8
  (2×2×2 against 3×3×3).
- PME convergence: MBX's example settings are 3e-3 kcal/mol off, E4's 2e-7.
- The dispersion's real-space cutoff is part of the periodic Hamiltonian. The long-range O–H c₆ product is 17 % below
  the pair C₆, so the 64-molecule cell's dispersion is −354.87 kcal/mol at r_c 6.5 Å and −355.65 at 9 Å.
- The monomer's polarisability is 1.4075 Å³.
- A random 128-molecule packing gives ε_∞ = 1.681 (first order 1.590, Clausius–Mossotti 1.734).

**Tests.** `tests/test_mbpol.py` runs the geometry helpers everywhere. With a build it also tests:

- MBX's example;
- the site model;
- units and forces in a cell against differences;
- a narrow cell through its replica, against the 3×3×3 replica;
- the tin-foil dipole image;
- the external field's linear, symmetric response.

448 tests: 426 pass and 22 are skipped without the q-AQUA builds (49 min on two CPU cores). Without an MBX build, 6
more are skipped.

## v0.0.18 → v0.0.19 — E4's data pipeline: rigid MB-pol MD and the frame sets

**The MBX checks on Niflheim** (`run_e4_mbx_checks.sbatch`, 28 Oct; s001, 8 threads, MBX at −O2, commit d6ebb08):

- Every check of v0.0.18 came out the same, to the printed digit. The 8 MBX tests pass.
- Energy and forces with E4's truth take 0.37, 0.67 and 1.45 s for 128, 256 and 512 molecules, and 1.41 s for 64
  molecules through their replica.
- MBX's example PME settings take 0.09, 0.17 and 0.33 s, 4–5 times less. The dynamics use them.

**Rigid-water MD** (`sseft/rigidmd.py`, NumPy only):

- velocity Verlet with SHAKE/RATTLE, holding the three distances of every molecule (O–H, O–H, H–H), vectorised over
  the molecules;
- the Bussi–Donadio–Parrinello thermostat, which samples the canonical ensemble;
- `heat`, the kinetic energy the thermostat takes out, so that `conserved()` = K + E_pot + heat is constant up to the
  integrator's error;
- molecules whole and unwrapped along the trajectory; the forces and the saved frames see them wrapped, each O in the
  cell;
- 6n − 3 degrees of freedom, with the centre-of-mass velocity removed every 100 steps.

**The data script** (`scripts/e4_data.py`):

- `md`, one trajectory per size and seed.
  - Rigid monomers are packed at random at 0.997 g/cm³.
  - A 1 ps warm-up at 0.5 fs (τ 50 fs), then 2 fs steps at 300 K (τ 100 fs) with MBX's example PME settings.
  - Frames every 0.5 ps after 10 ps, labelled with E4's truth at the end.
  - It prints the temperature, the constraints, the conserved energy's drift, g_OO, the labels against the
    dynamics' energies, and timings.
- `assemble`, the sets, after one line per trajectory (its temperature, the conserved energy's drift, flagged
  above 1e-3 kcal/mol/ps per molecule, and g_OO):
  - training (seeds 0–2), and validation and test (seed 3, halved) for 64, 128 and 256 molecules;
  - test for 512 and 1024 molecules;
  - an npz for the harness, read by `sseft.water.e4_structures` (which pins MB-pol's site model on request);
  - an extxyz for MACE, in eV and eV/Å.
- `sseft.water.MBPolTruth`: the truth with its site model read from MBX.

**Checks in the container** (−O0 build, 2 threads):

- The integrator on a toy potential (`tests/test_rigidmd.py`):
  - its forces are the energy's gradient;
  - the energy holds without the thermostat;
  - the thermostat's temperature is right, and its conserved energy holds;
  - Bussi's factor has the right limits;
  - the initial velocities, the wrapping, and an ideal gas's g(r) = 1.
- A 64-molecule MB-pol trajectory (seed 0, τ 500 fs; 1 + 4 + 5 ps):
  - constraints held to 3e-13;
  - g_OO's first peak is 2.49 at 2.88 Å;
  - the labels lie −0.0027 ± 0.0001 kcal/mol per molecule from the dynamics' energies;
  - the dynamics' forces are 0.004 kcal/mol/Å rms (at most 0.027) from the truth's;
  - 0.39 s per step.
- That trajectory's sampling averaged 320.7 K, with a stretch at 330–380 K around 6–7 ps. **The thermostat's random
  stream caused it, not the dynamics:**
  - NVE from its last frame drifts −0.0011 kcal/mol/ps per molecule over 0.4 ps. Its energy fluctuates by 0.0028
    kcal/mol per molecule at 2 fs and by 0.0008 at 1 fs. Converging the dipoles to 1e-16 changes nothing.
  - The conserved energy, logged every step, stays within ±0.2 kcal/mol (0.003 per molecule) over 3 ps of a fresh NVT
    run and over 7.7 ps of a replay of the trajectory.
  - The replay draws the same thermostat noise. It heated at the same time, with its energy conserved.
  - Seed 0's stream sums its normal deviates to +4.8σ over the 500 steps from 6.1 ps. That is the largest excursion
    in its first 10⁵ steps; seeds 1–5 peak at 3.0–3.8σ. A 64-molecule cell's total energy has a canonical spread of
    about 15 kcal/mol, and the excursion pushed it up by about 50.
- The dynamics stay as they were, with three changes:
  - τ is 100 fs instead of 500 fs. The total energy of a 64-molecule cell then relaxes in about 0.3 ps instead of
    1.5 ps, so an excursion of the noise is shorter-lived and frames 0.5 ps apart are less correlated. Both sample
    the canonical ensemble.
  - The packing and the dynamics draw from two streams of SeedSequence([seed, n_mol]). Before, both were
    default_rng(seed): the packing and the velocities drew the same random numbers, and every size reused seed s's
    stream.
  - The conserved energy is logged, and its drift is reported for every trajectory.
- A second 64-molecule trajectory with these settings (seed 1; 0.5 + 2 + 3 ps):
  - the sampling averages 298.5 ± 16.2 K (300.7 K from 1.5 ps on);
  - the conserved energy drifts −0.0007 kcal/mol/ps per molecule and stays within 0.34 kcal/mol (0.005 per molecule)
    over 5 ps;
  - constraints held to 3e-13;
  - the labels lie −0.0027 ± 0.0001 kcal/mol per molecule from the dynamics' energies;
  - g_OO's first peak is 2.85 at 2.83 Å (6 frames).

**Tests.** `tests/test_rigidmd.py` (7) and `tests/test_e4_data.py` (4; the last runs 16 molecules of MB-pol MD with an
MBX build). 459 tests: 437 pass and 22 are skipped without the q-AQUA builds (70 min on two CPU cores, sharing them
with an MD run). Without an MBX build, 7 more are skipped.

**To run on Niflheim:** `mkdir -p logs logs_error && sbatch run_e4_data.sbatch` (20 CPU tasks, at most about 2.5 h
each), then `python scripts/e4_data.py assemble`.
