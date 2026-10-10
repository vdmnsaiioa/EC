# E3 rehearsal — results on the synthetic polarisable water truth (harness v0.0.5)

*10 Oct 2026. Pre-registration: `e3-water-preregistration.md` (§3 for the dimer and the clusters, §6–§13
for the far-field sequence, the embedded dagger and the band edge — eight addenda, each written before the
run it predicts). Script `scripts/e3_water.py`, truth `sseft/water.py`; every number below is in a log under
`results/e3_pw_*` in the repository, and the command that produced it is given with each table. The CCSD(T)
data do not exist yet: this is the rehearsal of the E3 protocol on a truth with known answers, and its
product is the list of things the protocol got wrong (§5) and what the real E3 has to do differently (§6).*

**In one paragraph.** The dimer ladder behaved as pre-registered on every branch assignment, and its one
wrong number (the learned monomer dipole 1.8 % low instead of 0.6 % high) had a verified cause — for a
molecular dimer "beyond r_c" is the smallest intermolecular atom pair, not the centre distance — and was put
right by a pre-registered check (1.861 ± 0.003 D against a computed 1.867). The cluster and far-field parts
falsified the v1.5 predictions and, through eight pre-registered runs, located why: the many-body energy of
a hydrogen-bonded cluster seen from outside is, to 85–95 %, the field of its *embedded induced dipoles*
(environment-dependent sources, v1's claim), not a response to the band field; a pinned-charge dagger cannot
carry it, and the learned-source rungs did not recover it from energies and forces at this data scale, their
residual on the far-field training configurations never falling below the signal. A dagger with three
pinned inputs per fragment — embedded multipoles, the polarisation work and the dispersion coefficients —
carries it with nothing fitted beyond r_c: dimer tail to 6e-8 at 20 Å, far-field test to the mutual response
(5 % at 10 Å, as computed), and, once the band edge is moved below the hydrogen bond (ℓ₁ = 0.75 Å), the
hexamers' non-additive energy to 4–9 % where every rung at ℓ₁ = 1.5 Å sat at 73–120 %. The real E3's
protocol changes accordingly (§6).

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
| M_1† (q pinned) | 7.7–8.5e-6 | 1.53e-3 / 4.98e-7 | 0 / 0, – | – | pinned | 1.5e-3 at 20 Å falling as R⁻³, 1e-5 at 100 Å, spread 0 → numbers ✓ (1.47e-5 at 97.8 Å; the ratio 20.5 → 304 Å is 3070 against (304/20.5)³ = 3270); the attribution "the induction" ✗: two thirds of it is the dimer's dispersion, −C₆/R⁶ against −0.652/R³ = 1.14e-3 at 20.5 Å (§5(9)) |
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

## 3. The far-field induction test (K = 2): eight pre-registered runs, A–H

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
| F (§11) | 32, eight cores | M_1†ᵉᵐᵇ (embedded dipoles pinned, no self energy) | 140, 162 % (174 / 64 / 53 / 17; seed 1: 202 / 65 / 56 / 17) | 4.6 ± 0.5 % at 10 Å, ≈ 14 / 16 % at 8 / 7 Å, ≤ 40 % at 6 Å → **✗ twice over**: the dispersion was left out of the prediction (12.5 % at 10 Å) and the pinned induced dipoles double count the polarisation work inside r_c (§12) |
| | | M_S0†ᵉᵐᵇ | 149, 158 % (187 / 60 / 40 / 21; 200 / 53 / 41 / 23) | = M_1†ᵉᵐᵇ within the seed spread → ✓ at 6–7 Å; at 8 Å 40 % against 53–56 and at 10 Å 21–23 % against 17: the read-out's unidentified coefficients add a little beyond r_c |
| G (§12) | 32, eight cores | M_1†ᵉᵐᵇ + self energy | 48, 39 % (49 / 46 / 50 / 17; 39 / 37 / 49 / 17) → ✓ at every distance | 16.8 ± 0.3 % at 10 Å (computed), ≤ 54 / 64 % at 8 / 7 Å, < 174 % at 6 Å |
| | | **M_16†ᵉᵐᵇ** (+ dispersion on pinned α_O) | **21, 18 % (25 / 13 / 16 / 5; 20 / 16 / 9 / 5)** → 10 Å ✓ to the digit, 7 Å ✓, 8 Å ✓ within the seed spread (16 / 9 against ≤ 14), 6 Å ✓ | **4.6 ± 0.5 % at 10 Å, ≤ 14 / 16 % at 8 / 7 Å**, ≤ 60 % at 6 Å |

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

**Run F, the embedded dagger without the polarisation work** (`results/e3_pw_far_8cores_emb_K2.log`): at 10 Å
both seeds give 17 % — training-independent, and equal to the untrained band's 16.8 % (`scratch/emb_check.py`:
the band equals electrostatics + piece (2) to 10⁻⁶ kcal/mol on every configuration; the 16.8 % is the
remainder 4.6 % ⊕ the dispersion 12.5 % that the test subtracts from the truth and no M_1-type rung carries).
At 7 and 8 Å the trained values (64–65 %, 53–56 %) equal the untrained band's (64 %, 54 %): E₀ adds nothing
there (the nearest probe–core pairs sit at 4.3–6.6 Å, under the cosine tail of the cutoff). At 6 Å the
dagger is *worse* than M_1† (174–202 % against 86–114 %): pinned induced dipoles treated as permanent double
count the polarisation work inside the cluster, and E₀ had to unlearn ~1 kcal/mol per molecule — the cluster
training residual is 0.48–0.49 kcal/mol against 0.23–0.37 with learned sources. Both effects are bookkeeping
of the prediction, not of the model, and both are repaired in run G.

**Run G, the embedded dagger with the polarisation work, and with the dispersion band**
(`results/e3_pw_far_8cores_embself_K2.log`). With the pinned self energy, M_1†ᵉᵐᵇ falls from 140–162 % to
**48 and 39 %** (6 Å: 49 / 39 % against 174–202 %; 7 Å 46 / 37; 8 Å 50 / 49; 10 Å 17 / 17 — the computed
16.8 % to the digit, the dispersion and the mutual response), and the cluster training residual from 0.48 to
0.35–0.36 kcal/mol (predicted < 0.3: slightly missed; it is now repulsion, the short part of the ℓ₁-split
kernels and the dispersion inside r_c that E₀ learns). **M_16†ᵉᵐᵇ, with the dispersion band on the pinned
one-oscillator α_O, gives 21 and 18 % overall and 25 / 13 / 16 / 5 and 20 / 16 / 9 / 5 % at 6 / 7 / 8 / 10 Å**
— at 10 Å the mutual response (1) + (3) alone, 4.6 % as computed; at 7–8 Å the computed 16 / 14 % within the
seed spread; at 6 Å 20–25 % against the band-only 40 % (E₀, now with only short-range terms to learn, fits
part of the response inside r_c). The far-field many-body energy of the cluster is carried, with no fitting
beyond r_c, by three pinned inputs per fragment — the embedded multipoles, the polarisation work and the
dispersion coefficients — and what is left is the second-order mutual response, which is what the band-field
read-out would have to carry and which is 0.0008–0.03 kcal/mol here, below every fit floor reached in this
rehearsal. The far-field training residual of the embedded daggers (42–63 % of their induction) is, for the
first time in the sequence, below the signal.

**Run G on the dimer and the clusters** (`results/e3_pw_dimer_embself_K2.log`, `results/e3_pw_clusters_embself_K2.log`).
Dimer: M_16†ᵉᵐᵇ's window rmse is 9.3–9.7e-7 E_h, ten times below every other rung's (the band now carries
electrostatics, induction and dispersion; E₀ only the short-range remainder), and its relative error beyond
r_c is **8.4e-7 at 8.2 Å, 5.9e-8 at 20.5 Å, 9.2e-10 at 97.8 Å** and 3.9e-9 at 304 Å (the float floor:
1.3e-17 E_h), spread exactly 0 — the predicted ≤ 1e-4 by four orders of magnitude. The p⋆ = 3 tail with its
induction and dispersion corrections is reproduced with nothing fitted beyond r_c. Clusters: **the prediction
failed** — non-additive rmse 1.043 / 1.032 kcal/mol = **81 / 80 %** of the non-additive energy (predicted
≤ 25 %), hexamer rmse 0.766 / 0.749 (≤ 0.3), while the dimer rmse is 0.048 / 0.056 kcal/mol against
0.10–0.22 for every other rung: the pair part is nearly exact and the many-body part is not. The
prediction's error was the "7 % at the hydrogen-bond distance", which is the short part of the *potential*;
the induction runs on the *field*, whose long part at r/ℓ₁ = 1.27 is 0.64, and on the field gradient for the
dipole couplings, less. With ℓ₁ = 1.5 Å, a third of the hydrogen-bond partner's field and most of the
induced-dipole couplings inside a compact hexamer are short-range by the band's own definition: the
self-consistent induction of a hydrogen-bonded cluster, non-additive part included, is largely E₀'s even
with exact embedded sources, and E₀ does not learn it from 36 clusters. The pre-registered run H (§13)
tests the diagnosis by moving the band edge below the hydrogen bond.

**Run H, the band edge below the hydrogen bond** (`results/e3_pw_clusters_embself_l1_0.75_K2.log`; M_16†ᵉᵐᵇ with
ℓ₁ = 0.75 Å, everything else as in G): **non-additive rmse 0.110 / 0.055 kcal/mol = 9 / 4 % of the non-additive
energy** (predicted ≤ 25 %), **hexamer rmse 0.078 / 0.061 kcal/mol** (≤ 0.3; every other rung 0.50–0.88), dimer
rmse 0.021 / 0.011 (≤ 0.06). The diagnosis holds: with the band edge at half the hydrogen-bond length the band
carries the self-consistent induction of the cluster — pair and many-body alike — and E₀ is left with
repulsion and the dispersion damping, which 36 clusters teach it. The cluster part of E3 is therefore not a
near-field dead end but a statement about ℓ₁: at 1.5 Å the hydrogen bond is inside the band edge and the
induction is E₀'s; at 0.75 Å it is the band's, and the sources, pinned or learned, carry it.

The table of the cluster part, completed:

| rung (ℓ₁) | hexamer rmse (kcal/mol) | dimer rmse | non-additive rmse, % of 1.287 |
|---|---|---|---|
| M_1† (1.5 Å) | 0.547, 0.602 | 0.112, 0.190 | 88, 93 % |
| M_S0† (1.5 Å) | 0.625, 0.504 | 0.119, 0.104 | 85, 73 % |
| M_S2† (1.5 Å) | 0.881, 0.737 | 0.208, 0.219 | 114, 120 % |
| M_16†ᵉᵐᵇ (1.5 Å) | 0.766, 0.749 | 0.048, 0.056 | 81, 80 % |
| **M_16†ᵉᵐᵇ (0.75 Å)** | **0.078, 0.061** | **0.021, 0.011** | **9, 4 %** |

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

* `pytest -q tests/`: 32 tests (forces of every rung against finite differences, the Ewald versions of every
  band, the water truth: dipole 1.855 D by construction, E R³ → −0.652 at φ = 0, forces against finite
  differences to 10⁻⁷, the fragment identity E = E₀ + E_coul − ΣE_frag with |E(300 Å)| < 10⁻⁸; the embedded
  dagger's identity band(q, μ_emb; ℓ₁ = λ) + ½Σ|μ|²/α = E_es + E_ind of the truth to 10⁻¹⁰ on a relaxed trimer,
  and off by more than the many-body energy without the self term).
* The far-field decomposition (`scratch/far_decomposition.py`) and the untrained-band check of the embedded
  dagger (`scratch/emb_check.py`: band = electrostatics + piece (2) to 10⁻⁶ kcal/mol at d = 10 Å on all eight
  configurations; band-only errors 247 / 64 / 54 / 16.8 % at 6 / 7 / 8 / 10 Å, the 6 Å value including E₀ at
  initialisation), both on the cached data sets of the eight-core run.
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
8. **Run F, the embedded dagger M_1†ᵉᵐᵇ: 4.6 % at 10 Å → 17 %; ≤ 40 % at 6 Å → 174 %.** Two bookkeeping errors in
   the prediction, both verified (§12 of the pre-registration; `scratch/emb_check.py`): the test's "induction"
   has the truth's dispersion subtracted, which no M_1-type rung carries beyond r_c (12.5 % of the induction
   rms at 10 Å; 4.6 ⊕ 12.5 = 16.8 %, the measured 17 % to the digit, with the band equal to electrostatics +
   piece (2) to 10⁻⁶ kcal/mol), and pinned *induced* dipoles in a Coulomb band double count the polarisation
   work inside r_c, which E₀ then has to unlearn (cluster training residual 0.48 kcal/mol against 0.23–0.37).
9. **§3's M_1† item attributed its tail error to the induction**; the number (1.5e-3 at 20 Å) was right, the
   attribution wrong — dispersion is two thirds of it. Same omission as 8.
10. **Run G: the cluster training residual of M_1†ᵉᵐᵇ + self < 0.3 kcal/mol → 0.35–0.36**; M_16†ᵉᵐᵇ at 8 Å
    ≤ 14 % → 16 and 9 % (one seed above). Everything else in run G as computed, the 10 Å numbers to the digit.
11. **Run G, clusters: M_16†ᵉᵐᵇ's non-additive error ≤ 25 % → 80–81 %.** The short-range share of the
    induction at ℓ₁ = 1.5 Å was taken from the potential's split (7 % at 1.9 Å) instead of the field's (36 %)
    and the field gradient's; the hydrogen-bonded cluster's induction is mostly inside the band edge.
12. Run H: every number inside its bound (9 / 4 % against ≤ 25 %; 0.078 / 0.061 against ≤ 0.3 kcal/mol).

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
4. **The cluster reading depends on the band edge (§5(2), (11), (12)).** At ℓ₁ = 1.5 Å the non-additive
   energy of hexamers is induction among sources inside the band edge, E₀'s territory, and 36 clusters do not
   teach it to any rung (73–120 %). At ℓ₁ = 0.75 Å it is the band's: the embedded dagger reaches 4–9 % with
   the same data. The real E3 runs the cluster part at ℓ₁ = 0.75 Å with M_16†ᵉᵐᵇ as the ceiling (pre-registered
   form: ≤ 25 % of the non-additive energy), the isolated-monomer dagger as the baseline (~90 %), and the
   learned-source and band-field rungs between them as the measurement — those were not run at ℓ₁ = 0.75 Å
   here, so their numbers are the open question of the real cluster part, with the far-field answer (learned
   sources do not recover the embedded map at this data scale) as the prior.
5. **The far-field test is the v1 / v1.5 reading, and it is a test of the sources (§5(3)–(5)).** At 7–10 Å from
   a hydrogen-bonded trimer, 85–95 % of the many-body energy is the field of the cluster's embedded induced
   dipoles; the mutual response (1) + (3) — the only part a band-field read-out can carry — is 5–16 % there and
   40 % at 6 Å. So the far-field test reads v1's claim (environment-dependent sources) first and v1.5's only
   in the remainder, and the remainder is at or below the fit floors reached here (0.05 kcal/mol on a probe
   configuration, independent of weighting, §5(6)–(7)).
6. **The dagger must pin embedded multipoles — and the polarisation work, and the dispersion.** M_1† with
   isolated-monomer sources has the error "100 % of the induction" by construction and is the baseline; the
   dagger that tests the architecture is M_16†ᵉᵐᵇ with three matching inputs per fragment: the distributed
   multipoles of each monomer *in its cluster* (from the CCSD(T) density or a DFT density embedded in the
   cluster), the first-order induction (polarisation) energy that goes with them (SAPT's E_ind, or ½Σ|μ|²/α in a
   polarisable-model reading) as a pinned per-atom interaction-energy term — without it the band double counts
   the induction inside the cluster — and the dispersion coefficients (α(iω) of the embedded monomer, E2's
   matching). With the three inputs and the band edge below the hydrogen bond, this dagger reproduced the
   dimer tail to 6e-8 at 20 Å, the far-field test to the mutual response (5 % at 10 Å) and the hexamers'
   non-additive energy to 4–9 %. Learned sources (M_1μ) have to recover
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
9. **The band edge of a hydrogen-bonded system sits below the hydrogen bond (§5(11)–(12)).** At ℓ₁ = 1.5 Å
   the partner's field at 1.9 Å is 64 % long-range and the induced-dipole couplings less, so the induction of
   a compact cluster is E₀'s and is not learned from 36 clusters (every rung at 73–120 % of the non-additive
   energy, embedded sources included); at ℓ₁ = 0.75 Å the band carries it and the embedded dagger reaches
   4–9 %. The real E3 runs at ℓ₁ = 0.75 Å (or scans 0.5–1.0 Å with the level criterion of X2 as the reading);
   the periodic case pays with k_max = 2 b_max/ℓ₁, eight times the k-vectors of ℓ₁ = 1.5 Å, which the Ewald
   path supports (`ewald_n_max`). The near-source switch of the band-field input scales with ℓ₁.
10. **Not rehearsed:** the dispersion channel on water (the truth's C₆ is O–O only and damped; on the real data
   α(iω) matching is E2's problem and M_16† needs the monomer's α(iω)), the periodic case, and the real sources
   themselves — the embedded distributed multipoles at the CCSD(T) level, which the data plan has to supply
   together with the curve and the clusters.

The pre-registration's §4 ("predictions for the real data, in this form") is therefore amended: the dimer
predictions are computed on the correctly defined identifying nodes at r_c ≤ 5 Å; the cluster part runs at
ℓ₁ = 0.75 Å with the embedded dagger's ≤ 25 % as the ceiling and the learned-source rungs as the open
measurement; the far-field prediction is stated for M_16†ᵉᵐᵇ (the mutual-response share computed from a
polarisable-model decomposition of the actual configurations, as in §3 here) and for M_1μ (open, with the
rehearsal's answer as the prior); and the band-field response is read only where the remainder is above the
fit floor, which the training-residual diagnostic establishes per run.
