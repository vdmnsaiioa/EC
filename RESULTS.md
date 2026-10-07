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
