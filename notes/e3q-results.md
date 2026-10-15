# E3-Q results — E3 on q-AQUA v1, read against `e3q-preregistration.md`

*15 Oct 2026. One run of `configs/water_qaqua.yaml` (both parts), 14 Oct, Niflheim node s001, one RTX 3090, JAX 0.10.2,
harness v0.0.9 + the q-AQUA runtime build fix (commit 4a81a84). Dimer part 1.7 h (6 rungs × 8 seeds), clusters 1.4 h
(5 rungs × 4 seeds); the authors' benchmarks passed in the job. No seed excluded on the clusters; M_G and M_A lost 5
of 8 seeds each to the 2×-best rule. Percentages are rmse as a share of the signal rms, means over seeds. The local
master of this note is `notes/` in the repository.*

## 1. The readings

| reading | registered | result |
|---|---|---|
| M_1† tail | ≤ 1e-8 at 7.8 Å, ≤ 1e-10 beyond 10 Å | ✓ 7.5e-9 at 7.8 Å, ≤ 3.4e-11 to 100 Å; 1.3e-8 at 305 Å = 4.5e-17 E_h absolute, the double-precision floor of E(all) − Σ E(f) — the registered bound did not allow for it (a bound error, not physics) |
| M_∞ / M_G / M_A tails | branches 3 / 1 / 3 | ✓ (M_A: q crosses 3 at ≈ 10.4 Å, curvature 0.11–0.13 Å⁻² vs 0.125); M_∞ window rmse 4.8e-4 E_h vs "~1e-4" |
| M_1 tail form | q = 3.00, flat | ✓ q 3.00–3.04 beyond 30 Å, slope −0.00 |
| M_1 bias, dipole | 0.06–0.13; 1.71–1.82 D, spread ≤ 3 % | ✗ 0.173; 1.683 ± 0.058 D (3.4 %). The dipole tracks the window fit across seeds (Pearson −0.99 with window rmse; best seed 1.757 D, linear trend reaches 1.763 D at rmse 1.40e-4 vs the best seed's 1.50e-4): unconverged optimisation, not the class |
| M_1μ dipole | 1.47–2.10 D, spread > M_1's | range ✓ 1.892 ± 0.006 D (+2.2 %); spread ✗ (0.3 %); the charge/dipole split is unidentified (q_O −0.21 … +0.38) but the total is not |
| hexamer non-additive (0.915 kcal/mol) | M_S0† 25–55 %; M_∞, M_1† 70–110 %; M_S2† ≤ 1.3 × M_S0†; M_1μ 50–100 % | M_S0† ✗ 70 % (seeds 51–98); M_∞ ✗ 54 % — the best rung; M_1† ✓ 91 %; M_S2† ✓ 85 %; M_1μ ✓ 73 %. M_S0†'s hexamer energy rmse is not half of M_1†'s (0.56 vs 0.55 kcal/mol) ✗ |
| held-out dimers | every rung ≤ 0.3 kcal/mol | ✓ except M_1μ 0.34 |
| raw 2-body | 0.1–0.6 kcal/mol, none within 5 × PES | ✓ 0.31–0.42 (PES 0.018) |
| raw 3-body, all 4673 | M_S0† 25–55 %; M_∞, M_1† 50–100 % | M_S0† ✗ 67 %; M_∞ 68 %, M_1† 83 %, M_S2† 65 %, M_1μ 58 % |
| raw 3-body, 7–9 Å (311) | M_S0† ≤ 60 %, below the PES (48 %) | ✗ 73 % (M_S2† 76, M_1μ 84, M_∞ 99, M_1† 106) |
| raw 3-body, > 9 Å (108) | M_S0† ≤ 80 %, ≥ 3/4 seeds < 100 %; M_∞, M_1† ~100 % | ✓ 69 %, 4/4 seeds (M_S2† 69, M_1μ 81; M_∞, M_1† 100; PES 100) |
| raw 4-body (267) | every rung 50–120 % | ✗ M_S0† 154 %, M_S2† 139 % (one seed each at 246 / 223 %); M_∞ 96, M_1† 110, M_1μ 97 |

No §5 surprise except the M_1† bound at 305 Å, which is a rounding floor.

## 2. What it says

- **The one place the long-range rungs beat M_∞ is where the truth has nothing.** Beyond 9 Å, where q-AQUA's
  3-body term is zero, M_S0† and M_S2† cut the error on the CCSD(T) trimers from 100 % to 69 % on every seed: a model
  trained on q-AQUA carries part of the CCSD(T) many-body physics that q-AQUA drops. The reading the experiment was
  built around came out as registered (at 7–9 Å it did not: the PES's switched fit, 48 %, stays ahead).
- **Inside ~7 Å E₀ covers what the truth has.** q-AQUA's many-body energy is mostly induction (the fixed-coefficient
  induction form leaves 20 % of the hexamers'), but it lives where the short-range part reaches, so the read-out is
  not needed and 36 clusters do not identify it: M_∞ beats every long-range rung on the hexamers, ties M_S0† on all
  trimers. Below 5 Å every rung's 3-body error is 58–83 % of the CCSD(T) signal (PES 2 %) — the near-field many-body
  energy is barely learned from 36 clusters by any rung, and that is what the hexamer reading measures.
- **q-AQUA v1 cannot test long-range polarisation as a truth**: no induction or dispersion beyond 7.8 Å, no
  many-body energy beyond 9 Å (anticipated in `e3-data-q-aqua.md`, which kept the far-field reading for a polarisable
  truth). The conclusion after the run: a truth with polarisation at long range is needed.
- Smaller: M_1's dipole miss is convergence (more L-BFGS, or a tighter rule); the source rungs' read-out produces a
  spurious 4-body energy; M_1μ's total dipole is tightly identified even though its split is not.

## 3. Next: E3-QP on q-AQUA-pol

`github.com/jmbowma/q-AQUA-pol` (CC0, June 2023): TTM3-F everywhere plus CCSD(T) corrections switched off by 7 Å —
a far field with many-body induction and an R⁻⁶ tail, the same raw CCSD(T) sets as held-out, the build and driver of
v1 nearly unchanged, so v1 vs pol isolates the long range of the training truth. Set up in v0.0.10 with 300 training
clusters, probe configurations for the far field and longer L-BFGS on the dimer: `notes/e3qp-preregistration.md`.
Alternative / follow-up: MB-pol via MBX (`paesanilab/MBX`, UC licence free for research and non-profit use; MB-nrg
potentials for alkali ions in water) — the ion part with a real polarisable truth.
