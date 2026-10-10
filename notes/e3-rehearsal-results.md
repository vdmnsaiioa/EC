# E3 rehearsal — results on the synthetic polarisable water truth (harness v0.0.5)

*10 Oct 2026. Pre-registration: `e3-water-preregistration.md` (§3 for the dimer and the clusters, §6–§9
for the far-field sequence). Script `scripts/e3_water.py`, truth `sseft/water.py`; every number below
is in a log under `results/e3_pw_*` in the repository, and the command that produced it is given with
each table. The CCSD(T) data do not exist yet: this is the rehearsal of the E3 protocol on a truth with
known answers, and its product is the list of things the protocol got wrong (§5) and what the real E3
has to do differently (§6).*

## 1. Dimer, p⋆ = 3 (K = 4) — `results/e3_pw_dimer_K4_part{1,2}.log`

    python scripts/e3_water.py --part dimer --rungs M_inf,M_G --seeds 4 --steps 1500 --lbfgs 300
    python scripts/e3_water.py --part dimer --rungs M_1p,M_1,M_1mu,M_A --seeds 4 --steps 1500 --lbfgs 300

Window [3.86, 6.95] Å (O–O), 40 log-uniform nodes, σ = 10⁻⁹ E_h, four isolated monomers at zero, r_c = 6 Å,
Adam 1500 + L-BFGS 300, w_force = 1; evaluation on 61 nodes to 500 Å; the truth's E R³ → −0.6533 at 300 Å
(dipole–dipole −0.652). Seeds with window rmse above twice the best seed's are excluded before the ensemble
statistics (the convergence rule); the spread is set to NaN below 10⁻¹⁴ max|E_train| (float floor).

| rung | window rmse (seeds) | rel. err of the mean, 20 / 300 Å | W/\|f⋆\| 20 / 300 Å, slope on [30, 300] | q 10 → 300 Å | monomer dipole | predicted (§3) → verdict |
|---|---|---|---|---|---|---|
| M_∞ | 1.50–1.54e-4 | 1.000 / 1.000 | 0 (floor) / 0, – | – | – | error 1, spread 0, branch 3 → ✓ |
| M_G | 5.6e-6 (2 of 4 excluded: 1.1e-5, 1.6e-5, 2.1e-5) | 415 / 1.1e8 | 1.2e3 / 1.6e8, +4.12 | −6.4 → −1.04 | – | q → −1 or 0, error > 10 at 20 Å, branch 1 → ✓ (the MLP in r grows like R¹) |
| M_1† (q pinned) | 7.7–8.5e-6 | 1.53e-3 / 4.98e-7 | 0 / 0, – | – | pinned | 1.5e-3 at 20 Å falling as R⁻³, 1e-5 at 100 Å, spread 0 → ✓ (1.47e-5 at 97.8 Å; the ratio 20.5 → 304 Å is 3070 against (304/20.5)³ = 3270) |
| M_1 (q learned) | 8.1, 9.7, 11.2e-6 (1 excluded: 1.8e-5) | 3.41e-2 / 3.26e-2 | 1.63e-2 / 1.63e-2, +0.00 | 3.13 → 3.00 | 1.822 ± 0.013 D (1.814–1.842; two seeds with the sign of every charge flipped) | q = 3.00, slope 0 ± 0.1, branch 2 → ✓; μ̂ = 1.857 D ± 3 % → magnitude ✓ (−1.8 %), **sign of the bias ✗**, see §5(1) |
| M_1μ (q, μ learned) | 4.7, 7.8e-6 (2 excluded: 1.4e-5, 2.6e-5) | 1.73e-2 / 1.32e-2 | 2.86e-2 / 3.00e-2, +0.01 | 3.09 → 3.00 | 1.876 ± 0.019 D; q_O = +0.035 … +0.080 | same tail, same dipole within 3 %, split unidentified → ✓ (the charges carry 5–12 % of the dipole, the atomic dipoles the rest) |
| M_A (ℓ = 4 Å, N = 4) | 2.15–3.5e-5 (1 excluded: 4.4e-5) | 1.00 at 20.5 Å (1.07 at 15.5) | floor beyond 27 Å | burst (q = −6.0 at 8.2 Å), q = 3 crossed at 9.4–10.1 Å, then 2.83, 9.2, 20.5 at 10.1, 10.9, 11.7 Å | – | R_q3 = 9.4 + (1–3) Å ✓; dq/dR′² = 0.125 ± 20 % → 0.133 (fit on 15–26 Å; 0.129 on 17–26, 0.136 on 14–26) ✓; error 1 from 15–20 Å ✓; floor by ~30 Å ✓ |

The asymptotic line of M_A's exponent is q = 2R²/ℓ² − 2N = 0.125 R² − 8 (the highest power of the Laplacian
family dominates the spread); the measured q approaches it from below (68.1 at 25.4 Å against 72.8), so a
fit over a finite window overestimates the slope slightly, as on Ar₂ and Ne₂ (0.131, 0.129).

**The learned charge scale (§5(1)).** The prediction was computed for the linear class beyond r_c — a
one-parameter family, all multipoles ∝ λ — minimised under the training objective on "the ten nodes beyond
6 Å"; §3 quoted λ̂ = 1.001 (μ̂ = 1.857 D). That number cannot be reproduced from the files; the kept script
`scratch/lambda_hat.py` (the run's objective: per-atom energies, forces with w = 1, the run's counts) gives
λ̂ = 1.0065 (energies only 1.022), i.e. μ̂ = 1.867 D, and supersedes it — induction and dispersion pull the
scale up either way. The measured scale is λ = 0.982 (μ = 1.822 D; the relative error of the mean, 3.26 %
at 300 Å, is 1 − λ² to the digit), 2.4 % below the class minimum with a seed spread of 0.7 %. The premise
was wrong, not the arithmetic: **at every node of the window
at least one intermolecular atom pair is inside r_c** — the donor H to the acceptor O sits at R_OO − 0.96 Å,
5.04 Å at R_OO = 6.0 and 5.99 Å at 6.95 Å — so E₀ is silent at no node, and the H···O message, small but
free, absorbs part of the band's R⁻³ energy on the identifying nodes and biases the scale down. "Beyond r_c"
for a molecular dimer means the smallest intermolecular pair distance beyond r_c, not the centre distance.

*Pre-registered check (before the run):* the same M_1 and M_1μ with r_c = 5.0 Å, under which the 11 nodes
with R_OO > 5.96 Å have every intermolecular pair beyond r_c. Prediction from the same computation on those
11 nodes: λ̂ = 1.0067, **μ̂ = 1.867 D**, seed spread ≤ 3 %, and in particular **the bias changes sign**
(μ̂ > 1.855 D). If the dipole stays at 1.82 D the mechanism is something else and §5(1) is wrong.

**Result of the check** (`results/e3_pw_dimer_rcut5.0_K4.log`, run after the prediction was on record): M_1 at
r_c = 5 Å gives **1.864, 1.862, 1.858 D** (seed 3 unconverged — window rmse 7.0e-5 against 1.8–2.3e-5 — and
excluded by the rule, at 1.695 D): **1.861 ± 0.003 D**, +0.3 % against the truth and 0.3 % below the class
minimum 1.867 D; the bias changed sign as predicted, the spread fell from 0.7 % to 0.2 %, and the tail
error of the mean fell from 3.3 % to 0.67 % at 300 Å (= λ² − 1 to the digit, λ = 1.0032), W/|f⋆| from 1.6e-2
to 3.2e-3, q = 3.00, slope +0.00. M_1μ at r_c = 5 Å: 1.889, 1.849, 1.872 D (one seed excluded at 1.926):
1.870 ± 0.020 D, +0.8 %, q_O = −0.03 … +0.07 — the dipole gauge of the atomic-dipole channel sets its
wider spread. The window rmse rises from 8e-6 to 2e-5 (E₀ loses the O···O message beyond 5 Å), which is
the price and is irrelevant to the identification. §5(1) stands as diagnosed.

## 2. Clusters (K = 2) — `results/e3_pw_clusters_K2.log`, `results/e3_pw_clusters_K2_MS2p.log`

    python scripts/e3_water.py --part clusters --rungs M_1p,M_S0p --seeds 2 --steps 1000 --lbfgs 150 --batch 12 --lbfgs_chunk 6
    python scripts/e3_water.py --part clusters --rungs M_S2p --seeds 2 --steps 1000 --lbfgs 150 --batch 12 --lbfgs_chunk 6

Training: 12 clusters each of n = 3, 4, 5 (random rigid packings relaxed by 300 rigid-body Metropolis steps
at 300 K on the truth) plus the dimer window; held out: 8 hexamers and 8 dimers. rms interaction energy of
the training clusters 8.30 kcal/mol; the hexamers' non-additive energy (E minus the truth's 15 dimer
energies) has rms 1.287 kcal/mol, 9.1 % of the interaction energy. The model's non-additive energy is its
hexamer energy minus its own 15 dimer energies, so the comparison is free of the dimer error.

| rung | hexamer rmse (kcal/mol), seeds | dimer rmse | non-additive rmse, % of 1.287 | predicted (§3) → verdict |
|---|---|---|---|---|
| M_1† | 0.547, 0.602 | 0.112, 0.190 | 1.130, 1.194 → **88, 93 %** | 50–100 % → ✓ |
| M_S0† (order 0) | 0.625, 0.504 | 0.119, 0.104 | 1.089, 0.945 → **85, 73 %** | ≤ 30 %, hexamer rmse ≤ ½ v1's → **✗** (§5(2)) |
| M_S2† (order 2) | 0.881, 0.737 | 0.208, 0.219 | 1.464, 1.551 → **114, 120 %** | no gain over order 0 → ✓ as "no gain"; it is a loss of a factor 1.5 |

The band field at an oxygen is the field of the sources *through the ℓ₁-smeared kernel*: the H-bond partner's
H at 1.9 Å contributes 64 % of its true field, a source at 2.8 Å 93 %. In a hexamer every molecule is a near
source of every other, so the many-body induction of a small cluster is near-field induction — E₀'s
territory, which 36 clusters do not teach it — and the band-field read-out sees a smeared, partial field
with a coefficient nothing in the training set identifies (the dimer's far nodes carry ~10⁻⁶ E_h of
induction, below the window rmse). The order-2 terms, equally unidentified, extrapolate as noise: the loss.

## 3. The far-field induction test (K = 2): four pre-registered runs

    python scripts/e3_water.py --part far [--train_far N --train_far_cores C] --rungs ... --seeds 2 --steps 1000 --lbfgs 150

Test: a probe monomer at d = 6, 7, 8, 10 Å from a relaxed trimer (seed 300), 8 random placements per
distance; the probe's interaction with the core, E(core + probe) − E(core), has rms 0.3516 kcal/mol, and its
"induction" — the same minus the point-charge electrostatics of the truth's charges (repulsion and
dispersion are negligible there) — rms 0.0488 kcal/mol, 13.9 % of it, at every distance (a constant share,
see below). Training: the dimer window + 36 clusters, plus, from run B on, far-field configurations from
*other* cores (seed 400). The reading is the model's rmse on the probe's interaction as a percentage of the
induction, per distance: 100 % means "electrostatics only".

| run | training far-field configs | rung | rmse / induction, per seed (6 / 7 / 8 / 10 Å of the better seed) | predicted → verdict |
|---|---|---|---|---|
| A (§6) | none | M_1† | 112, 111 % (111 / 109 / 123 / 106) | ~100 % ✓ |
| | | M_S0† | 80, 89 % (81 / 69 / 88 / 96) | ≤ 20 % **✗** |
| | | M_S2† | 238, 299 % (261 / 224 / 126 / 118) | = M_S0† **✗** |
| B (§7) | 16, one core | M_1† | 106, 99 % (98 / 92 / 119 / 106) | ~100 % ✓ |
| | | M_S0† | 92, 82 % (83 / 72 / 89 / 99) | ≤ 30 % **✗** |
| | | M_S2† | 297, 370 % (349 / 224 / 102 / 115) | ≤ 1.5 × M_S0† **✗** |
| C (§8) | 16, one core | M_1μ | 120, 79 % (88 / 58 / 66 / 98) | ≤ 30 % **✗** (test at fault, §9) |
| | | M_S0μ | 117, 65 % (72 / 53 / 46 / 68) | ≤ 15 % **✗** (same) |
| D (§9) | 32, eight cores | M_1† | 98, 114 % (86 / 111 / 124 / 106) | ~100 % ✓ |
| | | M_1μ | 86, 116 % (94 / 67 / 73 / 90) | ≤ 50 % **✗** (the floor, §10) |
| | | M_S0μ | 80, 108 % (90 / 56 / 65 / 81) | below M_1μ with a gap at 6–7 Å only → lower at every distance by 4–11 points, both at the floor: **not read** |
| E (§10) | 32, eight cores, **weight 10** | M_1μ | 124, 90 % (102 / 54 / 72 / 111) | far-field training residual < 30 % of its induction → **170, 113 % ✗**; the test reading therefore void |
| F (§11) | 32, eight cores | M_1†ᵉᵐᵇ (embedded dipoles pinned) | RESULT_F_M1PEMB | 4.6 ± 0.5 % at 10 Å, ≈ 14 / 16 % at 8 / 7 Å (± 5), ≤ 40 % at 6 Å |
| | | M_S0†ᵉᵐᵇ | RESULT_F_MS0PEMB | = M_1†ᵉᵐᵇ within the seed spread |

**Run D's training residuals** (the diagnostic added in §10; `results/e3_pw_far_8cores_K2_part2.log`): on the
training clusters 0.23 / 0.33 kcal/mol (M_1μ), 0.22 / 0.36 (M_S0μ); on the 32 far-field *training*
configurations 0.055 / 0.045 kcal/mol (M_1μ) and 0.046 / 0.054 (M_S0μ) = **134–163 % of their own induction**
(0.0336 kcal/mol rms). The far-field signal was never fitted: with every structure weighted alike, the
optimiser trades a 0.05 kcal/mol residual on a far-field configuration against the same residual on a cluster
whose interaction energy is 8 kcal/mol, and the cluster floor of 0.2–0.4 kcal/mol is where both end up. Runs
A–D therefore measured the fit's floor at the far-field configurations, not the generalisation of anything,
and §9's prediction was wrong for a third reason, below the two it named. (The run itself had to be restarted
from cached data after the container's memory limit killed it half-way; the restarted seeds reproduce the
first run's numbers to the digit.)

**Run E, the weighted re-fit** (`results/e3_pw_far_8cores_wfar10_K2.log`): with the 32 far-field training
configurations weighted ×10 in energies and forces, their residual is 0.057 / 0.038 kcal/mol = 170 / 113 % of
their induction — it does not move — and the test error is 124 / 90 %. The floor is not the loss weighting.
Something in the model class or the optimisation leaves ~0.05 kcal/mol on a configuration whose
electrostatics is 0.35 kcal/mol; the learned sources' far field is wrong by about the induction itself. The
candidate is the one free multipole: a learned charge-plus-dipole monomer has a quadrupole that nothing in
the training set pins (the dimer's far nodes are dipole–dipole, the clusters are degenerate with E₀), and at
6–10 Å the quadrupole–dipole term is a tenth of the electrostatics. Not tested further here; the real E3
pins the sources instead (§6).

**The exact decomposition of the test's induction** (`scratch/far_decomposition.py`, on the cached test set):

| d (Å) | rms interaction | rms induction | rms piece (2): embedded dipoles × probe charges | rms remainder (1) + (3) | remainder / induction |
|---|---|---|---|---|---|
| 6 | 0.5598 | 0.0760 | 0.0688 | 0.0306 | 40.3 % |
| 7 | 0.3471 | 0.0495 | 0.0494 | 0.0077 | 15.6 % |
| 8 | 0.2318 | 0.0315 | 0.0292 | 0.0044 | 13.9 % |
| 10 | 0.0827 | 0.0179 | 0.0175 | 0.0008 | 4.6 % |
| all | 0.3516 | 0.0488 | 0.0456 | 0.0159 | 32.6 % |

(kcal/mol; piece (2) = −Σ_j μ_j(core alone)·E_probe(r_j) with the truth's kernel.) The far-field many-body
energy of a hydrogen-bonded trimer at 7–10 Å is, to 85–95 %, the field of its embedded induced dipoles.

RESULT_F_TEXT

**What the "induction" of this test is** (§8, verified by the constant 13.9 % share from 6 to 10 Å): the
probe's interaction minus point-charge electrostatics contains (1) the probe's own response to the core's
field, −½α|E|² ∝ R⁻⁶; (2) the interaction of the core's *pre-existing* induced dipoles — set by its own near
field, 0.1–0.3 a.u. per oxygen in a hydrogen-bonded trimer — with the probe's charges, ∝ R⁻³ like the
electrostatics itself; (3) the core's further response to the probe, ∝ R⁻⁶. (2) dominates, which is why the
share does not fall as R⁻³. And (2) is not a response to the band field: it is an environment-dependent
source, the induced dipole of each core oxygen as a function of its local environment — v1's learned
sources, not v1.5's read-out. A pinned-charge dagger cannot carry it by construction, and no degree-2
read-out at the probe can either, because at the probe the term is linear in the probe's field.

## 4. Verification record

* `pytest -q tests/`: 30 tests (forces of every rung against finite differences, the Ewald versions of every
  band, the water truth: dipole 1.855 D by construction, E R³ → −0.652 at φ = 0, forces against finite
  differences to 10⁻⁷, the fragment identity E = E₀ + E_coul − ΣE_frag with |E(300 Å)| < 10⁻⁸).
* The window geometry claim of §1 (`dimer_positions`): intermolecular pairs at R_OO = 6.0 Å are
  5.04, 5.68, 5.68, 6.00, 6.31, 6.63, 6.63, 6.93, 6.93 Å; at 6.95 Å the smallest is 5.99 Å; the first node with
  every pair beyond 6 Å would be R_OO = 6.96 Å, outside the window.
* The linear-class minima of §1 (`scratch/lambda_hat.py`, kept out of the package): λ̂ = 1.0065 on the 10
  nodes with R_OO > 6 Å, 1.0067 on the 11 nodes with every pair beyond 5 Å, 1.0108 on the 22 nodes with
  R_OO > 5 Å (energies only 1.022, 1.023, 1.033).
* M_A's exponent fit (`results/e3_pw_dimer_K4.json`, grid of the spread): q = −d ln W / d ln R by centred
  differences on the log grid; least squares of q against R² on the monotone segment.

## 5. Wrong predictions, in order

1. **M_1's learned dipole: −1.8 % where +0.1 % (recomputed: +0.6 %) was predicted** (within the ±3 %
   tolerance, wrong sign, 2.4 % below the class minimum at a spread of 0.7 %). Cause verified: no window
   node is E₀-silent for a molecular dimer at r_c = 6 Å (the H···O pair). Check pre-registered above
   (r_c = 5 Å → 1.867 D): **1.861 ± 0.003 D, sign flipped, 0.3 % from the computed minimum ✓.**
2. **Clusters, v1.5 order 0 ≤ 30 % → 73–85 %.** The many-body induction of a small cluster is near-field;
   the band-field inputs see the smeared far part and nothing identifies their coefficient (§2).
3. **Far-field run A, M_S0† ≤ 20 % → 80–89 %; M_S2† = M_S0† → 240–300 %.** No training configuration
   resolves the response to the band field above the fit's floor (§7).
4. **Far-field run B, M_S0† ≤ 30 % → 82–92 %.** Identification was not the limit; the "induction" is
   dominated by environment-dependent sources, not by a response (§8). The estimate "induction ~0.3–0.7 %
   of the probe's interaction" in §6 was wrong by a factor 20 for the same reason.
5. **Far-field run C, M_1μ ≤ 30 % → 79–120 %.** The 16 far-field training configurations shared a single
   core: one environment shown, a second asked for (§9).
6. **Far-field run D (eight cores), M_1μ ≤ 50 % → 86–116 %; M_S0μ "below M_1μ by the R⁻⁶ share" → both at
   the floor.** The training residual on the far-field configurations is 134–163 % of their induction: the
   signal was never fitted (§10(a) ✓).
7. **Run E (weight 10): the far-field training residual < 30 % → 113–170 %** (§10(b) ✗). The floor is the
   model class or the optimisation, not the weighting; the test reading of §10(c) is void.
8. RESULT_F_WRONG

## 6. What the rehearsal settles for the real E3

The protocol that goes to the CCSD(T) data is the rehearsed one with these changes, each tied to a numbered
wrong prediction above.

1. **Identifying nodes (§5(1)).** "Beyond r_c" means the smallest intermolecular atom-pair distance beyond r_c,
   and the dimer window must hold ≥ 10 such nodes. With the Phase 2 window [1.35 R_e, 1.8 × 1.35 R_e] that is
   r_c ≤ 5 Å for the dimer part (11 nodes) — verified: the learned dipole moves from 1.822 ± 0.013 D to
   1.861 ± 0.003 D against a computed class minimum of 1.867 D. The prediction for the learned monomer
   multipoles is the linear-class minimum under the training objective on those nodes, computed from the
   CCSD(T) curve before training by the construction of `scratch/lambda_hat.py`.
2. **The fixed orientation is φ = 0** (acceptor dipole along the O–O axis). At the near-equilibrium orientation
   the dipole–dipole coefficient of the rigid-monomer dimer vanishes and the curve is p = 4 to ~900 Å; a
   CCSD(T) scan computed there would be a quadrupole experiment, which M_1 with charges alone cannot represent.
3. **Fragment references and the near-source switch stay on** (`Rung.fragments`, `bf_rs = ℓ₁`); both were
   forced by failures in the first smoke tests (§1 of the pre-registration).
4. **The cluster reading is a near-field reading (§5(2)).** The non-additive energy of hexamers is induction
   among near sources, E₀'s territory; at 36 training clusters it separates nothing between v1 and v1.5 and is
   kept as what it is — a test of E₀ with the band's sources — with the prediction form "v1.5 = v1 within the
   seed spread", the opposite of §3's.
5. **The far-field test is the v1 / v1.5 reading, and it is a test of the sources (§5(3)–(5)).** At 7–10 Å from
   a hydrogen-bonded trimer, 85–95 % of the many-body energy is the field of the cluster's embedded induced
   dipoles; the mutual response (1) + (3) — the only part a band-field read-out can carry — is 5–16 % there and
   40 % at 6 Å. So the far-field test reads v1's claim (environment-dependent sources) first and v1.5's only
   in the remainder, and the remainder is at or below the fit floors reached here (0.05 kcal/mol on a probe
   configuration, independent of weighting, §5(6)–(7)).
6. **The dagger must pin embedded multipoles.** M_1† with isolated-monomer sources has the error "100 % of the
   induction" by construction and is the baseline; the dagger that tests the architecture is M_1†ᵉᵐᵇ with the
   distributed multipoles of each monomer *in its cluster* (from the CCSD(T) density or a DFT density embedded
   in the cluster — a decision for the data plan), RESULT_F_CONSEQ Learned sources (M_1μ) have to recover
   the same map from energies and forces; on the synthetic truth they do not at this data scale, and the gap
   between M_1μ and M_1†ᵉᵐᵇ on the real data is the price of learning the sources — report it as such rather
   than as a failure of either.
7. **Data for the far-field part.** The signal is 0.03–0.08 kcal/mol per configuration on a 0.1–0.6 kcal/mol
   interaction. For CCSD(T)/CBS that is within reach only with counterpoise correction and a tight basis
   extrapolation on the whole tetramer-sized structure; a cheaper and adequate truth for this *part* is a
   hybrid-functional DFT (far-field induction is not a correlation problem), with CCSD(T) kept for the dimer
   curve and the clusters. Either way: ≥ 8 different cores, probes at several distances with d defined from
   the core's centre of mass (at d = 6 Å the nearest atoms are 3–4.5 Å from the core — inside r_c), and the
   training-residual diagnostic reported alongside the test.
8. **Seeds.** K = 2 cannot separate rungs whose errors differ by less than ~30 % (the seed spreads above are
   that large); the real E3 runs K ≥ 4 with the convergence rule and reports the per-distance pattern with the
   rmse. The E3 runs are memory-bound on this container (one far-field run at a time, 3.5 GB; the data sets
   are cached); the K = 4 batch belongs on the group's GPU like the K = 16 batch of E1.
9. **Not rehearsed:** the dispersion channel on water (the truth's C₆ is O–O only and damped; on the real data
   α(iω) matching is E2's problem and M_16† needs the monomer's α(iω)), the periodic case, and the real sources
   themselves — the embedded distributed multipoles at the CCSD(T) level, which the data plan has to supply
   together with the curve and the clusters.

The pre-registration's §4 ("predictions for the real data, in this form") is therefore amended: the dimer
predictions are computed on the correctly defined identifying nodes; the cluster prediction is "v1.5 = v1";
the far-field prediction is stated for M_1†ᵉᵐᵇ (the remainder share computed from a polarisable-model
decomposition of the actual configurations, as in §3 here) and for M_1μ (the open question, with the
rehearsal's answer as the prior); and the band-field response is read only where the remainder is above the
fit floor, which the training-residual diagnostic establishes per run.
