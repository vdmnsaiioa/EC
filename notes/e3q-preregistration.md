# E3-Q — E3 on q-AQUA v1: pre-registration

*13 Oct 2026, written before any training. The first E3 on a published CCSD(T) potential — the water analogue of
E1 on the published Ar₂ / Ne₂ curves, with the truth chosen in `e3-data-q-aqua.md`. Harness v0.0.9;
`configs/water_qaqua.yaml` holds every value of the experiment; every number below is computed from the PES by
`scratch/e3q_preregistration.py` (q-AQUA v1 at commit 91c90d2, built with gfortran 13.3 by
`external/qaqua/build.sh`). The local master of this note is `notes/` in the repository.*

## 1. The truth, and what its construction implies

q-AQUA v1 (Q. Yu, C. Qu, P. L. Houston, R. Conte, A. Nandi, J. M. Bowman, J. Phys. Chem. Lett. 13, 5068 (2022)) is
a 4-body-truncated many-body expansion (`pes_shell.f90`): the Partridge–Schwenke monomer; a 2-body polynomial fit to
CCSD(T)/CBS for O–O < 6.5 Å, switched over 6.5–7.8 Å to the bare point-charge Coulomb of the monomers' LTP2011
charges and that alone beyond 7.8 Å; a 3-body fit (CCSD(T)-F12a, BSSE-corrected) switched to zero as the largest
O–O distance goes from 7 to 9 Å; a 4-body fit (CCSD(T)-F12) switched to zero over 5.8–7.5 Å.

**The build.** gfortran (`-fdefault-real-8 -fdefault-double-8`, the authors' `-r8`) with a C interface
(`external/qaqua/qaqua_capi.f90`) that initialises once and re-sizes q-AQUA's work arrays for any number of waters.
The authors' two benchmarks (README of `qAQUA_v1`: the trimer and the prism hexamer) are reproduced to the printed
1e-8 hartree in energy and 1e-8 hartree/bohr in every gradient component; two monomers 12 Å apart interact by
exactly the point charges of their LTP2011 charges (relative difference 4e-13). The energy carries rounding noise
of 2e-11 (dimer) to 8e-11 (hexamer) hartree rms — cancellation in the polynomial sums — ten times below the label
noise σ = 1e-9; finite differences with steps below 1e-3 bohr are dominated by it (recorded so that nobody
re-diagnoses it as a gradient error), and the gradient integrates to the energy difference to within the noise.

**The truth as used.** Rigid monomers at the harness geometry (r_OH 0.9572 Å, HOH 104.52°), interaction energy =
q-AQUA minus its 1-body term (`sseft.qaqua`, `sseft.water.QAquaTruth`), forces its exact gradient (q-AQUA's own
switching and charge derivatives are central differences: ~1e-7 relative). At this geometry the LTP2011 charges
are q_O = −0.65772, q_H = +0.32886, the monomer dipole **1.8509 D**.

Three consequences for E3:

1. **The dimer tail beyond 7.8 Å is exactly Σ q_i q_j / r_ij** of the truth's own charges, so the dagger M_1† with
   the LTP2011 charges pinned is exact on it — the first E3 dagger exact on its truth's tail (on PW the dagger
   missed the induction, 1.5e-3 at 20 Å).
2. **Below 7.8 Å the truth is the CCSD(T) fit and its switch**, 13–24 % away from the point charges at the
   window's identifying nodes (6.0–7.1 Å): the multipoles of the real charge density beyond the point charges'
   dipole, induction, dispersion. The identifying nodes are not in the asymptotic regime — the water analogue of
   E1's C₈ / C₁₀ on the Ar₂ nodes beyond r_c, which biased M_6's learned C₆ by +25 %.
3. **No many-body energy beyond 9 Å and no induction or dispersion tail.** The far-field many-body readings of the
   rehearsal and E3-R have no target on this truth (the far and ion parts are refused on it), and the rungs that pin
   sources of the PW truth's polarisable model (M_16†, M_16S0†, the embedded daggers) have no inputs (refused): the
   ladder is charge-only. The raw CCSD(T) trimers of the q-AQUA repository, whose 3-body energy at 7–12 Å the PES
   sets to zero, are the far-field check (§2(d)).

## 2. Design (`configs/water_qaqua.yaml`)

**(a) The dimer ladder** at φ = 0 (acceptor dipole along the O–O axis): R_e = 2.9346 Å, D_e = 4.552 kcal/mol
(rigid monomers); window [1.35 R_e, 1.8 × that] = [3.962, 7.131] Å, 40 log-uniform nodes, σ = 1e-9 E_h, four
isolated monomers at zero; evaluation to 500 Å (61 points). **r_c = 5 Å**: 12 identifying nodes (every
intermolecular pair beyond r_c), from 6.04 Å; at r_c = 6 Å there would be 2, against the ≥ 10 the rule of 10 Oct
requires. Rungs M_∞, M_G, M_A (ℓ = 4 Å, N = 4), M_1, M_1μ, M_1†; **K = 8**; Adam 1500 (full batch) + L-BFGS 300.

**(b) Matching**: the learned monomer dipole of M_1 and M_1μ against the LTP2011 dipole, 1.8509 D.

**(c) Clusters**: the design of the rehearsal and E3-R — n = 3, 4, 5 (12 each) for training, 8 hexamers and 8
dimers held out, random placement relaxed by 300 rigid-body Metropolis steps at 300 K on the truth; the non-additive
energy is E(hexamer) minus the model's own 15 dimer energies, against the truth's (its 3- and 4-body terms).
ℓ₁ = 0.75 Å, no radial switch (the own fragment excluded from the band-field input), r_c = 6 Å. Rungs M_∞, M_1†,
M_S0†, M_S2†, M_1μ; **K = 4**; the E3-R schedule (Adam 1000, batch 12, the read-out frozen → its order-0 linear
solve → L-BFGS 300 on everything).

**(d) The held-out CCSD(T) sets**: every cluster model is read on the raw n-body sets of the q-AQUA repository
(2-body CCSD(T)/CBS, 3-body CCSD(T)-F12a, 4-body CCSD(T)-F12) — the model's n-body energy by the subset expansion,
the same decomposition as the truth's — on the frames (i) whose every monomer is within 0.03 Å / 5° of the rigid
geometry (the models never saw another; the strict subset, 0.02 Å / 3°, is reported separately) and (ii) whose
every pair has a q-AQUA 2-body energy ≤ +2 kcal/mol (the training domain: the training clusters' pairs span −4.8
to +1.2 kcal/mol, the raw 2-body set reaches +37 kcal/mol at its 99th percentile; the filter is defined on the
training truth, not on the labels). The PES's own error on the same frames is the reference; the 3-body set is
also read by its largest O–O distance, the PES's switching variable.

## 3. Numbers from the PES, before training

| quantity | value |
|---|---|
| R_e, D_e (φ = 0, rigid) | 2.9346 Å, 4.552 kcal/mol |
| window; identifying nodes at r_c = 5 / 6 Å | [3.962, 7.131] Å; 12 (from 6.04 Å) / 2 |
| truth beyond the point charges at the identifying nodes | 13–24 % of the energy |
| point-dipole coefficient of the tail; E R³ at 500 Å | −0.6491; −0.6504 hartree bohr³ (the charges' higher multipoles) |
| M_1†'s exact error at 8 / 10–100 / 300 Å | 2.3e-9 / ≤ 2e-12 / 3e-11 (the band edge's erfc at 8 Å, then rounding); the "8 Å" reading row is the grid point 7.805 Å, 0.005 Å beyond the switch (the truth there is the point charges to 7e-15) |
| **M_1's class minimum** on the identifying nodes (charge scale λ, energies per atom and forces, w_force = 1) | |
| — charges with a linear response to the monomer geometry (three parameters by symmetry; the class the charge head can express) | **λ² = 0.907: dipole 1.763 D (−4.8 %), tail error 0.093** |
| — charges fixed (the response not learned) | λ² = 0.635: 1.475 D (−20 %), tail error 0.365 |
| — energies only | λ² = 1.289: 2.101 D (+13.5 %), tail error 0.289 |
| training clusters: interaction energy rms | 10.37 kcal/mol |
| test hexamers: interaction / non-additive rms (3-body, 4-body) | 15.72 / **0.915** kcal/mol (0.880, 0.048): 5.8 % of the interaction |
| test dimers: interaction rms | 3.19 kcal/mol |
| fixed-coefficient order-0 floor (the non-additive energy fitted by −½ Σ_Z a_Z Σ_{i∈Z} Σ_{j≠k} E_ij·E_ik, the band field at ℓ₁ = 0.75 Å of the other molecules' charges) | **20 %** on the test hexamers, 22 % on 200 more (a_O = 6.20, a_H = 5.65 a.u.); PW's was 30 % |
| held-out 2-body: frames (near-rigid of all; strict), label rms, PES rmse | 4940 (5952 of 71892; 4410), 3.46 kcal/mol, 0.018 (0.5 %) |
| held-out 3-body | 4673 (4699 of 45332; 1450), 0.622 kcal/mol, 0.015 (2 %) |
| — by largest O–O 0–5 / 5–7 / 7–9 / ≥ 9 Å: frames; label rms; PES rmse | 2739 / 1515 / 311 / 108; 0.779 / 0.312 / 0.027 / 0.011; 2 / 5 / **48 / 100 %** |
| — the same fixed-coefficient form: fitted per bin / with the hexamers' coefficients | 17 / 22 / 27 / 24 %; 18 / 22 / **42 / 36 %** |
| held-out 4-body | 267 (275 of 3692; 57), 0.130 kcal/mol, 0.014 (10 %); 7–9 Å: 17 frames, PES 99 % |

## 4. Predictions

**(a) Dimer, p⋆ = 3** (relative error of the mean, W/|f⋆|, local exponent q at 8–300 Å; plateau slope on [30, 300] Å):

- **M_∞**: relative error 1.000 at every reading row, spread 0, window rmse ~1e-4 E_h (the identifying nodes
  unfitted). Branch 3.
- **M_G**: q → −1 or 0; relative error > 10 at 20 Å. Branch 1.
- **M_A** (ℓ = 4 Å, N = 4): the burst beyond R₊, q rising through 3 near ℓ√(N + 3/2) = 9.4 Å (+1–3 Å), then growing
  in R′² with slope 2/ℓ² = 0.125 Å⁻² ± 20 %; relative error → 1 by 15–20 Å, float floor beyond ~30 Å. Branch 3.
- **M_1†: exact** — relative error ≤ 1e-8 at the 8 Å row (7.805 Å) and ≤ 1e-10 from 10 Å out, spread exactly 0
  beyond r_c.
- **M_1**: q = 3.00 beyond 30 Å, W/|f⋆| flat (slope 0 ± 0.1), branch 2 — **with a bias set by the window, not
  by the class**: the mean's relative error constant beyond ~10 Å at |λ̂² − 1| = **0.09 (registered 0.06–0.13)**.
- **M_1μ**: the same tail form (q = 3.00, flat); its total dipole read below.

**(b) Matching**: M_1's learned |μ| = **1.763 D, registered 1.71–1.82 D (the class minimum ± 3 %), seed spread ≤ 3 %**
(the rehearsal's M_1 sat within 0.3 % of its class minimum at r_c = 5 Å). The two alternatives are diagnostic: |μ|
near 1.47 D means the charge head did not learn the charges' response to the monomer geometry (the forces then pull
the scale down by 20 %); near 2.10 D, the forces did not constrain the scale. The charges' sign is in the Z₂ gauge of
neutral water (Coulomb energies are products q_i q_j): q_O of either sign, |μ| is the reading. M_1μ: total |μ| within
1.47–2.10 D with a seed spread larger than M_1's (the split between charges and atomic dipoles is not identified).

**(c) Clusters** (8 test hexamers; non-additive rmse as a share of 0.915 kcal/mol, means over the converged seeds):

- **M_∞, M_1†**: 70–110 % (E3-R's dagger without read-out: 90–106 %; E₀ learns some near-field many-body energy from
  36 clusters).
- **M_S0†: 25–55 %**, hexamer energy rmse at most half of M_1†'s — the read-out carries the many-body induction of
  real water as it did on PW (E3-R, same schedule: 44, 51 %); the fixed-coefficient floor is 20 %, and environment-
  dependent coefficients can go below it (one E3-R seed did).
- **M_S2†**: within 1.3× of M_S0† (R1b).
- **M_1μ**: 50–100 %.
- Held-out dimers: every rung ≤ 0.3 kcal/mol.

**(d) The held-out CCSD(T) sets** (rmse as a share of the label rms; the PES's own error is the floor where it is
small, since the models learn the PES):

- **2-body**: every rung 0.1–0.6 kcal/mol, none within 5× of the PES's 0.018.
- **3-body, all 4673 trimers**: M_∞ and M_1† 50–100 %; **M_S0† 25–55 %** (the hexamers' quantity, read on CCSD(T)
  labels); M_S2† within 1.3× of M_S0†; M_1μ 50–100 %.
- **3-body beyond the PES's switch — the sharpest reading.** At 7–9 Å (311 trimers, the PES 48 %) and beyond 9 Å
  (108, the PES 100 %: its 3-body term is zero there), M_∞ and M_1† stay at ~100 % beyond 9 Å (nothing in them
  carries a 3-body energy across r_c), while **M_S0† is below the PES in both bins: ≤ 60 % at 7–9 Å and ≤ 80 %
  beyond 9 Å, with at least 3 of 4 seeds below 100 % beyond 9 Å** — a model trained on q-AQUA carrying CCSD(T)
  many-body physics that q-AQUA drops. A fixed-coefficient order-0 term with the hexamers' coefficients would reach
  42 and 36 %; the bounds are looser because the read-out's coefficient in the environment of a distant monomer is
  not identified by compact clusters (E3-R: 0.6 of α without far-field identification data, 1.0 with).
- **4-body** (267 tetramers): every rung 50–120 % (the read-out has no first-order 4-body term).

## 5. Surprises, and what is not tested

A surprise would be: M_1† above 1e-8 at any reading row (the truth or the driver is not what §1 says); M_1's |μ|
outside 1.47–2.10 D (outside every limit of its class); M_S0† above M_1† on the hexamers or the 3-body set; M_S0† at
or above 100 % beyond 9 Å on every seed (the read-out's far-field coefficient collapses without far-field data —
which would make far-field identification data a requirement of the real E3, not an option).

Not tested here: flexible monomers (the held-out filter reads only near-rigid frames); a many-body response beyond
9 Å against the truth (there is none); seeds beyond K = 4 on the clusters. The convergence rule of E1 (seeds with a
window or training rmse above 2× the best seed's reported and excluded) applies to both parts.
