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
