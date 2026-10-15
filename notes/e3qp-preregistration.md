# E3-QP — E3 on q-AQUA-pol with hundreds of training clusters: pre-registration

*15 Oct 2026, written before any training. The controlled follow-up of E3-Q (`e3q-results.md`): the same harness,
rungs and raw CCSD(T) held-out sets, a truth whose long range has many-body induction and dispersion at every distance,
and 300 training clusters instead of 36, plus 160 probe configurations for the far field. Harness v0.0.10;
`configs/water_qaqua_pol.yaml` holds every value of the experiment; every number below is computed from the PES by
`scratch/e3qp_preregistration.py` (q-AQUA-pol at commit 3a38dc4, built with gfortran 13.3 by
`external/qaqua/build_pol.sh`; the raw sets from q-AQUA at 91c90d2). The local master of this note is `notes/` in the
repository.*

## 1. The truth, and what its construction implies

q-AQUA-pol (Q. Yu, C. Qu, P. L. Houston, A. Nandi, R. Conte, J. M. Bowman, J. Chem. Theory Comput. 19, 3446 (2023);
CC0) is TTM3-F everywhere plus CCSD(T) corrections (`pes_shell.f90`): the Partridge–Schwenke monomer; the charges of
the Partridge–Schwenke dipole surface moved to the H atoms and an M-site on the bisector (γ = 0.46: 0.2695 Å from O;
q_H = +0.6127, q_M = −1.2254 e at the rigid geometry); one polarisable site per molecule, the M-site, 1.444 Å³
(9.7446 a.u.), Thole-smeared, induced dipoles iterated to self-consistency with no cutoff; an O–O −C₆/R⁶ (C₆ = 52.469
a.u.) + exponential term; and permutationally invariant polynomial corrections to CCSD(T) — 2-body switched off over
6.5–7.0 Å (O–O), 3-body over 6.5–7.0 Å (largest O–O), 4-body over 5.5–6.5 Å. Energies are TTM3-F's kcal/mol over the
potential's own 627.51.

**The build.** gfortran with a C interface (`external/qaqua/qaqua_pol_capi.f90`, the names and conventions of v1's)
and the compiler's runtime bundled (`external/qaqua/runtime.sh`). The authors' hexamer benchmark is reproduced: total
energy −45.7738923224 kcal/mol against the README's −45.7738923287 (6e-9, rounding), every gradient component within
4.9e-9 hartree/bohr of the printed 1e-8. At 12 Å the interaction energy is TTM3-F's exactly (the corrections vanish),
and its electrostatics, induction, C₆ term and induced dipoles are reproduced by an independent evaluation from the
site charges, the M-sites and α to ≤ 2e-13. The energy's rounding noise at R_e is 5e-13 hartree (v1: 2e-11).

**The truth as used.** Rigid monomers at the harness geometry; interaction energy = q-AQUA-pol minus its
Partridge–Schwenke 1-body term (`sseft.qaqua.QAquaPol`, `sseft.water.QAquaPolTruth`). The pinned charges are the
dipole-surface charges on O, H, H, q_O = −0.66173, q_H = +0.33087: their dipole is TTM3-F's (**1.8622 D**), their
quadrupole is not — TTM3-F's site quadrupole about O has diagonal (0.536, −0.517, −0.019) e Å², the atom-centred
charges' (0.266, −0.303, 0.038); no symmetric neutral set of three atom-centred charges does better at fixed dipole
(one free charge). The M_16 daggers pin TTM3-F's α and C₆ (a one-oscillator α(iω) whose K = 8 quadrature gives C₆ =
52.4699).

Three consequences for E3:

1. **The dimer tail beyond 7 Å is TTM3-F**: at 7.8 / 14.7 / 98.7 Å the site-charge electrostatics is 97.6 / 99.6 /
   100.0 % of the energy, the induction 0.62 / 0.10 / 0.00 %, the C₆ term 1.8 / 0.32 / 0.00 %. The atom-centred
   charges' Coulomb is 83.8 / 91.7 / 98.8 % of it (the quadrupole): **no dagger of this harness is exact on this
   tail** — M_1† misses it by −16 % at 7.8 Å, −4.0 % at 29.8 Å, −0.38 % at 305 Å, M_16† by −14 % … −0.38 % (§3). A model with atomic
   dipoles (M_1μ) can carry the M-site's quadrupole; one with atom-centred charges alone (M_1) cannot.
2. **The identifying nodes** (12 at r_c = 5 Å, from 6.03 Å) are 16–18 % away from the point charges, mostly the
   quadrupole: M_1's class minimum with a linear charge response is λ² = 0.905, 1.7715 D (−4.9 %), and its tail error
   at that minimum is 0.24 at 7.8 Å falling to 0.10 at 300 Å — not constant, as on v1.
3. **Many-body induction at every distance, and realistic.** On the raw CCSD(T) trimers the PES's error is 22 % at
   7–9 Å and 21 % beyond 9 Å (v1: 48 and 100 %), on the tetramers at 7–9 Å 42 % (v1: 99 %). A trimer and a fourth
   molecule 8 / 10 / 12 / 15 Å away have a non-additive energy of 0.019 / 0.0084 / 0.0046 / 0.0031 kcal/mol rms (the
   test probes), nearly all of it TTM3-F's induction. This is the quantity v1 could not supply.

## 2. Design (`configs/water_qaqua_pol.yaml`)

**(a) The dimer ladder** at φ = 0 as E3-Q: R_e = 2.9299 Å, D_e = 4.563 kcal/mol (rigid), window [3.955, 7.120] Å,
40 log-uniform nodes, σ = 1e-9, four monomers, evaluation to 500 Å; r_c = 5 Å (12 identifying nodes); rungs M_∞, M_G,
M_A, M_1, M_1μ, M_1† and **M_16†**; K = 8; Adam 1500 (full batch) + **L-BFGS 1000** (E3-Q: 300, which left M_1's
charge scale short of its class minimum).

**(b) Matching**: the learned monomer dipole of M_1 and M_1μ against 1.8622 D.

**(c) Clusters**: **300 training clusters** (100 each of n = 3, 4, 5; E3-Q 36), relaxed as before (300 rigid-body
Metropolis steps at 300 K on the truth), four monomers, and **160 probe configurations** (8 relaxed trimers × probes
at d = 8, 10, 12, 15 Å × 5, weight 1); held out: **24 hexamers, 24 dimers** (E3-Q 8 and 8) and **80 probe
configurations** (2 other trimers × 4 distances × 10). The non-additive energy as before (E minus the model's own
dimer energies). r_c = 6 Å, ℓ₁ = 0.75 Å, no radial switch. Rungs M_∞, M_1†, M_S0†, M_S2†, M_1μ (E3-Q's) and **M_16†,
M_16S0†** (TTM3-F's α and C₆ pinned in the dispersion band); K = 4. Schedule: Adam 2000 (batch 32, read-out frozen)
→ the read-out's order-0 linear solve → L-BFGS 500 on everything (chunk 64).

**(d) The held-out CCSD(T) sets** as E3-Q (near-rigid within 0.03 Å / 5°, strict 0.02 Å / 3°, every pair's truth
2-body energy ≤ +2 kcal/mol — on q-AQUA-pol's pair energies: 4937 / 4673 / 267 frames, three 2-body frames fewer than
on v1), the 3-body set by largest O–O.

**(e) The probe reading** (new): for every test probe, the probe's interaction E(core + probe) − E(core) and its
non-additive part (minus the probe's three dimer energies) against the truth's, by distance — the long-range many-body
induction, with nothing short-range in it beyond r_c.

## 3. Numbers from the PES, before training

| quantity | value |
|---|---|
| R_e, D_e (φ = 0, rigid) | 2.9299 Å, 4.563 kcal/mol |
| window; identifying nodes at r_c = 5 / 6 Å | [3.955, 7.120] Å; 12 (from 6.03 Å) / 2 |
| the truth beyond the point charges at the identifying nodes | 16–18 % of the energy |
| point-dipole coefficient | −0.6571 hartree bohr³ |
| the reading rows: the evaluation grid's points nearest 8, 10, …, 300 Å | 7.793 / 10.333 / 11.898 / 14.702 / 19.494 / 29.764 / 48.766 / 98.727 / 305.173 Å |
| the tail at those rows: the point charges' Coulomb over the truth | 0.838 / 0.880 / 0.897 / 0.917 / 0.939 / 0.960 / 0.976 / 0.988 / 0.996 |
| **M_1†'s exact relative error** at those rows (the harness's own bands) | −0.1624 / −0.1200 / −0.1033 / −0.08257 / −0.06148 / −0.03971 / −0.02400 / −0.01175 / −0.003765 |
| **M_16†'s** | −0.1444 / −0.1116 / −0.09754 / −0.07941 / −0.06006 / −0.03930 / −0.02390 / −0.01174 / −0.003764 |
| **M_1's class minimum** on the identifying nodes (charge scale λ, energies per atom and forces, w_force = 1) | |
| — charges with a linear response to the monomer geometry | **λ² = 0.905: 1.7715 D (−4.9 %)**; the tail error at the rows 0.242 / 0.204 / 0.188 / 0.170 / 0.151 / 0.131 / 0.117 / 0.106 / 0.098 |
| — charges fixed / energies only | λ² = 0.724: 1.584 D (−15 %) / λ² = 1.204: 2.044 D (+9.7 %) |
| training clusters (300): interaction energy rms; their pairs | 10.83 kcal/mol; −4.70 to +1.84 kcal/mol |
| test hexamers (24): interaction / non-additive rms | 16.53 / **0.997** kcal/mol (6.0 %): 3-body 0.982, 4-body 0.033, 5- and 6-body 0.009; TTM3-F's many-body induction 0.772, the CCSD(T) corrections' non-additive part 0.274 |
| test dimers (24): interaction rms | 2.92 kcal/mol |
| fixed-coefficient order-0 floor (the non-additive energy by −½ Σ_Z a_Z Σ_{i∈Z} Σ_{j≠k} E_ij·E_ik, the band field at ℓ₁ = 0.75 Å of the other molecules' pinned charges) | **21 %** on 200 more hexamers (a_O = 7.49, a_H = 4.62 a.u.), **27 %** on the 24 test hexamers (E3-Q: 20 %) |
| test probes at 8 / 10 / 12 / 15 Å: interaction rms; non-additive rms; induction rms (kcal/mol) | 0.183 / 0.083 / 0.050 / 0.021; **0.0185 / 0.0084 / 0.0046 / 0.0031**; 0.0189 / 0.0085 / 0.0046 / 0.0032 |
| — the probe's nearest core atom beyond r_c = 6 Å | 65 / 100 / 100 / 100 % (training probes: 50 / 98 / 100 / 100 %) |
| — the non-additive part by the fixed form with the hexamers' coefficients | 34 / 45 / 41 / 42 % (training probes: 55 / 57 / 61 / 61 %) |
| — the interaction by the pinned charges' Coulomb (with the C₆ term) | 22 (24) / 13 (13) / 13 (13) / 20 (19) % |
| held-out 2-body: frames (near-rigid of all; strict), label rms, PES rmse | 4937 (5952 of 71892; 4407), 3.465 kcal/mol, 0.011 (0.3 %) |
| held-out 3-body | 4673 (4699 of 45332; 1450), 0.622 kcal/mol, 0.012 (2 %) |
| — by largest O–O 0–5 / 5–7 / 7–9 / ≥ 9 Å: frames; label rms; PES rmse | 2739 / 1515 / 311 / 108; 0.779 / 0.312 / 0.027 / 0.011; 2 / 3 / **22 / 21 %** (v1: 2 / 5 / 48 / 100 %) |
| — the fixed form: fitted per bin / with the hexamers' coefficients | 17 / 22 / 27 / 24 %; 19 / 22 / 45 / 38 % |
| held-out 4-body | 267 (275 of 3692; 57), 0.130 kcal/mol, 0.012 (9 %); 7–9 Å: 17 frames, PES 42 % |

## 4. Predictions

**(a) Dimer, p⋆ = 3** (relative error of the mean, W/|f⋆|, local exponent q at 8–300 Å; plateau slope on [30, 300] Å):

- **M_∞**: relative error 1.000 at every reading row, window rmse ~5e-4 E_h (E3-Q 4.8e-4). Branch 3.
- **M_G**: relative error > 10 at 20 Å, q → −1 or 0. Branch 1.
- **M_A** (ℓ = 4 Å, N = 4): q rising through 3 at 10–12.5 Å, then growing in R′² with slope 0.125 Å⁻² ± 20 %;
  relative error → 1 by 15–20 Å. Branch 3.
- **M_1† and M_16†: their computed values** (§3) at every reading row to 1e-6 relative, or within the double-precision
  floor of E(all) − Σ E(f), 1e-16 hartree absolute (E3-Q's lesson); spread exactly 0 beyond r_c. (A 10-step smoke run
  of M_16† reproduced all nine to the printed three digits — the tail is the pinned bands whatever E₀ learns.) What they test here is
  the harness — the physics they miss (the M-site quadrupole, the induction, and for M_1† the dispersion) is computed.
- **M_1**: q = 3.00 ± 0.05 beyond 30 Å, W/|f⋆| flat (slope 0 ± 0.1), branch 2; the mean's relative error within
  ± 0.03 of the class minimum's 0.242 / 0.204 / 0.188 / 0.170 / 0.151 / 0.131 / 0.117 / 0.106 / 0.098 at the rows.
- **M_1μ**: the same tail form, and below M_1's relative error at every row from 10 to 30 Å by at least a factor 2
  (its atomic dipoles can carry the M-site's quadrupole, which is ~14 % of the energy at the identifying nodes).

**(b) Matching**: M_1's learned |μ| = **1.7715 D, registered 1.72–1.82 D (the class minimum ± 3 %), seed spread
≤ 2 %** (L-BFGS 1000: E3-Q's dipole tracked the window fit, r = −0.99, at 300 steps). M_1μ: total |μ| within
1.79–1.94 D (± 4 % of 1.862; E3-Q +2.2 %), seed spread ≤ 1 %. The alternatives 1.58 D (the response not learned)
and 2.04 D (forces ignored) remain diagnostic.

**(c) Clusters** (24 test hexamers; non-additive rmse as a share of 0.997 kcal/mol, means over the converged seeds):

- Every rung's hexamer energy rmse below 0.4 kcal/mol (E3-Q, 36 clusters: 0.35–0.97); held-out dimers ≤ 0.2.
- **M_S0†, M_16S0†: 20–45 %** (E3-R on PW with 36 clusters: 44, 51 %; the fixed-coefficient floor on these hexamers
  27 %, which environment-dependent coefficients can go below).
- **M_S2†**: within 1.3 × M_S0†.
- **M_∞: 30–70 %; M_1†, M_16†: 40–90 %; M_1μ: 30–80 %.**
- **Open, with a stated expectation**: M_S0† and M_16S0† at or below M_∞ on at least 3 of 4 seeds — on E3-Q (36
  clusters, a truth whose many-body energy ends at 7–9 Å) M_∞ beat both; here the truth's induction reaches every
  distance and E₀ has eight times the data.

**(e) The probes** (80 test configurations; rmse as a share of the truth's rms at each distance):

- **Non-additive part — the sharpest reading.** M_∞, M_1†, M_16†: **100 % at 10, 12 and 15 Å** (structural: beyond
  r_c of every core atom nothing in them is many-body, and every test probe there is), 70–105 % at 8 Å.
  **M_S0†, M_16S0†, M_S2†: below 100 % at every distance on every seed, and 25–65 % at 10–15 Å** (the fixed form with
  the hexamers' coefficients: 41–45 %; with far-field configurations in the training set E3-R's read-out reached the
  response coefficient, 1.00–1.05 of α). **M_1μ: 50–100 %** (environment-dependent dipoles carry the core's
  pre-polarisation in the probe's permanent field, not the response to the probe).
- **Interaction**: M_∞ 100 % at 10–15 Å; M_1† and M_16† within ± 5 points of the pinned charges' 13–20 % at 10–15 Å
  (§3; the quadrupole); the read-out rungs at or below M_16†.

**(d) The held-out CCSD(T) sets** (rmse as a share of the label rms; the PES's own error is the floor where it is
small, since the models learn the PES):

- **2-body**: every rung 0.05–0.4 kcal/mol, none within 5 × of the PES's 0.011.
- **3-body, all 4673 trimers**: M_S0†, M_16S0†: 20–50 % (E3-Q 67 %); M_∞: 30–70 %.
- **3-body at 7–9 Å** (311): M_S0†, M_16S0† ≤ 60 % (the PES 22 %; E3-Q's M_S0† 73 %).
- **3-body beyond 9 Å** (108): **M_S0†, M_16S0† ≤ 60 % on at least 3 of 4 seeds** (the PES 21 %; E3-Q's M_S0†, trained
  on a truth with nothing there, 69 %); M_∞, M_1†, M_16† 95–105 %.
- **4-body** (267): every rung 50–130 % (E3-Q's source rungs 139–154 %).

## 5. Surprises, and what is not tested

A surprise would be: M_1† or M_16† off their computed tail values by more than 1e-6 relative beyond the rounding floor
(the bands or the truth are not what §1 says); M_∞, M_1† or M_16† below 98 % on the probes' non-additive energy at 12
or 15 Å; a band-field rung at or above 100 % on the probes' non-additive energy at every distance on every seed (the
read-out does not carry the far-field induction even with far-field training data and a truth that has it); M_S0†
above M_1† on the hexamers or the 3-body set; M_1's |μ| outside 1.58–2.04 D.

Not tested here: flexible monomers; ions (q-AQUA-pol has none; MB-pol's MB-nrg ions are the candidate); the embedded
daggers (TTM3-F's induced dipoles sit on the M-site, which the harness has no site for — refused); an exact dagger
(the M-site charge would need virtual sites in the harness; its bias is computed instead); seeds beyond K = 4 on the
clusters. The convergence rule of E1 (seeds with a window or training rmse above 2× the best seed's reported and
excluded) applies to both parts. Expected cost on one RTX 3090: ~3.5 h for the dimer part, ~5 h for the clusters.
