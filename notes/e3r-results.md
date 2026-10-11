# E3-R — the band-field response, tested explicitly: results

*10–11 Oct 2026. Pre-registration: `e3r-bandfield-response-preregistration.md` (§3 has the ceilings computed
from the truth, §4 the predictions, §6–§10 the five addenda written between the runs). Harness v0.0.6 with
the own-fragment exclusion, the ion probe, the scaled and zero-initialised read-out, the linear solve of its
order-0 sector and the frozen-Adam → solve → L-BFGS schedule; every number is in `results/e3r_*` with the
command that produced it. K = 2 throughout (the convergence rule did not exclude any seed).*

**In one paragraph.** The read-out does carry the band-field response when it is given the right job and
the right optimiser: at the band edge ℓ₁ = 0.75 Å with permanent sources, order 0 brings the hexamers'
non-additive error from 90–106 % of the non-additive energy (no read-out) to 15–51 % depending on the
training schedule, and on the ion probe — where the signal is 10–60× the fit floor — from 87–95 % of the
induction to 42–50 %, with the response coefficient read directly on an isolated monomer at 0.60–0.65 of
the truth's α. Getting there took four successive repairs of the *training*, not of the model — units,
initialisation, optimiser, identification data — each pre-registered before its run and each wrong about
the next; the remaining gap to the computed ceiling (86–98 % of the ion's induction is first order in the
band field) is the identification question of §10, run as R2b. The learned-source tier is unreadable on a
charged system for a reason of its own (the neutrality shift spreads the ion's charge over every atom).

## 1. R1 — clusters at the band edge ℓ₁ = 0.75 Å

    python scripts/e3_water.py --part clusters --rungs M_16p,M_16S0p,M_16S2p --seeds 2 --steps 1000 --lbfgs 150 \
        --batch 12 --lbfgs_chunk 6 --l1 0.75 --rs_over_l1 0 --cache results/cache_clusters.pkl      (joint Adam, unscaled read-out)
    python scripts/e3_water.py --part clusters --rungs M_16S0p,M_16S2p --seeds 2 --steps 1000 --lbfgs 300 ...  (final schedule)

Permanent sources (the isolated monomer's charges) pinned, the dispersion band on the pinned α_O, the band
field at an atom being the field of the *other* molecules' charges; 36 training clusters (n = 3–5), 8 hexamers
and 8 dimers held out; the hexamers' non-additive energy has rms 1.287 kcal/mol, of which the first-order
induction E_ind1 is 71 % (the order-0 read-out's computed ceiling for a fixed coefficient) and the
self-consistent correction 30 %.

| rung, schedule | hexamer rmse (kcal/mol), seeds | dimer rmse | non-additive rmse, % of 1.287 | predicted → verdict |
|---|---|---|---|---|
| M_16† (no read-out), L-BFGS 150 | 0.659, 0.588 | 0.178, 0.176 | **106, 90 %** | 80–95 %, hexamer 0.5–0.7 → ✓ (one seed at 106) |
| M_16S0†, joint Adam (read-out in physical units, L-BFGS 150) | **0.206, 0.159** | 0.064, 0.015 | **36, 15 %** | ≤ 45 %, hexamer ≤ 0.35 → **✓** |
| M_16S2†, joint Adam | 0.148, 0.144 | 0.020, 0.026 | 22, 19 % | within 1.3× of order 0 → ✓ (means 0.265 against 0.327 kcal/mol) |
| M_16S0†, final schedule (frozen Adam → linear solve → L-BFGS 300) | 0.317, 0.271 | 0.084, 0.115 | **44, 51 %** | ≤ 45 % → one seed above; still 2× below the baseline |
| M_16S2†, final schedule with the full order-2 solve | 0.996, 0.687 | 0.095, 0.056 | 126, 69 % | the solve of ~1900 coefficients on ~1700 equations over-fits (§4(4)); re-run with the order-0 solve: R1b |

**R1b** (`results/e3r_clusters_final_K2_S2order0solve.log`; the final schedule with the linear solve restricted
to the order-0 sector, the orders ≥ 1 switching on in the polish): M_16S2† **52, 51 %** (hexamer 0.398, 0.319;
dimer 0.092, 0.086) against order 0's 44, 51 % under the same schedule — within 1.3× (means 0.660 against
0.610 kcal/mol, ratio 1.08): the pre-registered control holds once the solve is well-posed.

With the same 36 clusters that left every rung of the rehearsal at 73–120 %, the read-out carries the
many-body induction of the hexamers. The best result (15 %, joint Adam) sits *below* the 30 % "floor" of §3:
that floor was for a fixed coefficient α; the read-out's coefficients are environment-dependent (a(s_i) and
the pair terms w(r_ij; s_i, s_j)(n̂·E)²), so the local self-consistent enhancement of the response — the
effective polarisability of an oxygen in its hydrogen-bond environment — is partly representable too. The
two schedules bracket a real trade-off: training the read-out jointly with E₀ from the start lets it take the
cluster induction where the fields are strong (0.01–0.05 a.u.) and E₀ adapts around it; training E₀ first and
solving the read-out on the residual leaves E₀ holding part of the induction inside r_c, which the solve and
300 L-BFGS steps do not fully redistribute. The former cannot identify the far-field coefficient (§2), the
latter can; a schedule that does both is the open optimisation question, not a question about the class.

## 2. R2 — the ion probe (`results/e3r_ion_final_K2_*.log`; the failed schedules in `results/e3r_ion_l1_0.75_K2_*.log`)

    python scripts/e3_water.py --part ion --rungs M_16p,M_16S0p --seeds 2 --steps 1000 --lbfgs 300 --batch 12 --lbfgs_chunk 6 \
        --l1 0.75 --rs_over_l1 0 --train_far 1 --train_far_cores 8 --test_cores 4 --n_test 4 --cache results/cache_ion.pkl

A unit charge at d = 6, 7, 8, 10, 12 Å from four relaxed trimers (80 configurations); its interaction with the
core minus the point-charge electrostatics of the pinned charges — the **induction**, rms 1.04 / 0.57 / 0.35 /
0.20 / 0.15 kcal/mol — of which 86–98 % is first order in the band field of the permanent sources (the
order-0 ceiling) and the core's self-consistent response proper 0.39 kcal/mol overall. Training: the dimer
window + 36 clusters + 54 ion–monomer configurations (2–8 Å) + 48 ion–trimer configurations from 8 other
cores at d = 4–10 Å. The ion is beyond r_c = 6 Å of every atom only from d ≈ 8 Å (test) and at d = 10 Å (train).

| rung, schedule | rmse / induction, seeds (6 / 7 / 8 / 10 / 12 Å of seed 0) | far training residual (d ≥ 6) | a_eff/α (monomer + ion, 20 Å) | predicted → verdict |
|---|---|---|---|---|
| M_16† (no read-out), L-BFGS 150 | 95, 89 % (91 / 102 / 100 / 100 / 100) | 104, 97 % | – | 100 ± 3 % at 8–12 Å → ✓ |
| M_16† , L-BFGS 300 (parity) | 93, 87 % (89 / 101 / 100 / 100 / 100) | 99, 94 % | – | ✓ |
| M_16S0†, joint Adam, unscaled | 69 % (67 / 75 / 72 / 70 / 69) [1 seed] | 76 % | 0.209 | ≤ 20–25 % ✗ — the floor: units (§6) |
| M_16S0†, scaled, random init | 107 % [1 seed]; clusters' residual 3.3 kcal/mol | 79 % | 0.124 | void — the fit collapsed (§7) |
| M_16S0†, scaled, zero init, Adam | 86 % [1 seed]; clusters' residual 1.9 | 70 % | 0.020 | void — Adam's steps (§8) |
| M_16S0†, frozen Adam → L-BFGS 300 | 46 % [1 seed] (46 / 46 / 43 / 52 / 47) | 40 % | 0.655 | moving; not converged (§9) |
| **M_16S0†, frozen Adam → linear solve → L-BFGS 300** | **42, 50 %** (36 / 54 / 43 / 63 / 54; 43 / 63 / 56 / 67 / 62) | **30, 43 %** | **0.648, 0.605** | ≤ 25 % at 6–8, ≤ 30 % at 10–12 Å → ✗; the signal fitted to the bound, the coefficient at two thirds (§10) |
| M_1μ (learned q, μ) | 591 % [1 seed] | – | – | unreadable: the neutrality shift spreads the ion's +1 over all ten atoms (§4(5)) |

The read-out halves the ion's induction error (87–95 → 42–50 %) once the far-field signal is actually fitted
(training residual 30–43 % of it), and the coefficient read on an isolated monomer with the ion at 20 Å —
where nothing but the read-out acts beyond the electrostatics — is 0.60–0.65 of the truth's α: identified,
to two thirds. The error is largest where only the band carries the ion (10–12 Å: 54–67 %), smallest where
E₀ shares (6 Å: 36–43 %). The band-field coefficient is shared with E₀ in every training configuration but
the eight at d = 10 Å (and six ion–monomer configurations at 8 Å): the identification data were thin, which
R2b (§10) tests directly by adding 72 configurations with the ion beyond r_c of every atom.

**R2b — with the identification data** (`results/e3r_ion_final_extra_K2_S0.log`; `--extra_far 2`: 72 configurations
with the ion beyond r_c of every atom added to the training set — 8 further relaxed cores × d = 10, 12, 14 Å ×
2 placements and 24 ion–monomer configurations at 8–14 Å; same test, same schedule, K = 2):

| rung | rmse / induction, seeds (6 / 7 / 8 / 10 / 12 Å) | pure far-field training residual | a_eff/α | predicted (§10) → verdict |
|---|---|---|---|---|
| **M_16S0†** | **32, 34 %** (30 / 34 / 35 / 55 / 43; 34 / 33 / 31 / 40 / 36) | 33, 45 % | **1.052, 1.000** | a_eff/α = 1.00 ± 0.15 → **✓**; test ≤ 25 % at 6–8 Å → 30–35 % ✗ by 5–10 points; ≤ 30 % at 10–12 Å → 36–55 % ✗; residual ≤ 25 % → 33–45 % ✗ |

The response coefficient is now identified: read on an isolated monomer with the ion at 20 Å, where the read-out
is the only term beyond the electrostatics, it is **1.05 and 1.00 of the truth's α** (0.65 and 0.61 without the
72 configurations, 0.21 with the read-out in physical units). The ion's induction error falls to a third
(87–95 % → 32–34 %), the per-distance pattern is flat at 30–35 % from 6 to 8 Å and 36–55 % at 10–12 Å, and the
clusters' training residual is unchanged (0.36–0.44 kcal/mol). What remains above the computed floor (7–14 %
of the induction is the self-consistent correction the order-0 operator cannot carry) is a fit residual of
33–45 % on the pure far-field *training* configurations themselves — 0.03–0.05 kcal/mol per structure, five to
ten times below the clusters' residual, which the objective, weighting every structure alike, has no reason to
reduce further. With the signal fitted and the coefficient right, that is the legitimate place for the weight
of the far-field configurations (`--w_far`), K ≥ 4, and more cores — the next pre-registration, not this one.



## 3. R3 — the water probe at ℓ₁ = 0.75 Å (`results/e3r_far_final_K2.log`)

    python scripts/e3_water.py --part far --rungs M_16p,M_16S0p --seeds 2 --steps 1000 --lbfgs 300 --batch 12 --lbfgs_chunk 6 \
        --l1 0.75 --rs_over_l1 0 --train_far 1 --train_far_cores 8 --cache results/cache_far_8cores.pkl

The rehearsal's far-field test (a water probe at 6–10 Å from a trimer; induction 0.076 / 0.050 / 0.032 /
0.018 kcal/mol, 85–93 % of it first order), with permanent sources at the new band edge and the final schedule.

| rung | rmse / induction, seeds (6 / 7 / 8 / 10 Å) | far training residual | clusters' residual (kcal/mol) | predicted → verdict |
|---|---|---|---|---|
| M_16† | 101, 101 % (97 / 112 / 102 / 100; 102 / 100 / 98 / 100) | 113, 96 % | 0.31, 0.32 | 100 ± 5 % at 10 Å → ✓ |
| M_16S0† | **68, 68 %** (71 / 68 / 53 / 48; 78 / 49 / 45 / 46) | 71, 64 % | **0.13, 0.14** | ≤ 30 % at 6–7 Å, ≤ 60 % at 8 Å, residual ≤ 50 % → ✗: floor-limited, as the §3 arithmetic allowed (the signal is 1–4× the floor) |

The read-out takes a third of the water probe's induction and cannot take more here: the far-field training
residual, 0.022–0.024 kcal/mol, is the fit floor of this data set and equals the probe's induction at 8–10 Å.
The one clean number is the clusters' training residual, which the read-out halves (0.31 → 0.13 kcal/mol):
inside r_c the read-out represents the pairwise and first-order many-body induction that E₀ otherwise has
to learn. The ion probe, with a signal 10–60× the floor, is the test that reads the far field; the water
probe's induction at 7–10 Å is a quantity for K ≥ 4 and a tighter floor.


## 4. Wrong predictions

1. M_16†'s non-additive error 80–95 % → 90 and 106 % (one seed above): immaterial; the baseline.
2. The §3 "floor" of 30 % for order 0 on the hexamers → 15 % for one seed: the floor was for a fixed
   coefficient, the class has environment-dependent ones.
3. **R2, every schedule before the last two: M_16S0† ≤ 20–25 % → 69, 107, 86, 46 %**, each with the training
   residual on the far-field configurations above the pre-registered 30 % — the floors of §6–§9 (units,
   initialisation, optimiser), each diagnosed and each wrong about the next; recorded as such.
4. **R2, final schedule: M_16S0† ≤ 25 % → 42, 50 %, a_eff/α = 1.00 ± 0.15 → 0.65, 0.61**, with the far training
   residual at the bound (30, 43 %): the identification data, not the fit (§10).
5. **R2b: a_eff/α ✓ (1.05, 1.00); the test numbers 5–25 points above their bounds** with the pure far-field
   training residual 33–45 % against ≤ 25 %: the objective's weighting of a signal that is now fitted.
6. **R3, M_16S0† ≤ 30 % at 6–7 Å → 68–78 %**: the water probe's induction is at the fit floor, as the §3
   arithmetic allowed (1–4×); the test is not readable at K = 2.
7. **The learned-source tier on the ion (M_1μ ≥ 60 %, M_S0μ 30 points below) → 591 % and not run**: the
   charge head's uniform neutrality shift spreads the ion's +1 over all ten atoms and the electrostatics is
   off by multiples of the induction; a design fault of the head for charged systems, nothing to do with the
   read-out. The learned-source tier needs the ion's charge pinned (or a localised shift) and is deferred.
8. **R1, final schedule: M_16S2† within 1.3× of order 0 → 126, 69 %**: the full order-2 linear solve (~1900
   coefficients on ~1700 equations) over-fits; with the solve restricted to the order-0 sector: R1b.
9. R1b: within 1.3× ✓ (ratio 1.08).

## 5. What this settles

1. **The band-field response is learned, and the evidence is now direct.** With permanent sources, the own
   fragment excluded from the input and the band edge below the hydrogen bond, the order-0 read-out carries
   the first-order induction: the hexamers' many-body energy (non-additive error 15–51 % against 90–106 %
   without it), the ion's induction (32–34 % against 87–95 %), and — the sharpest reading — the response
   coefficient of an isolated monomer, 1.05 and 1.00 of α, read with nothing else in play. The X2-B toy's
   result stands on a molecular truth.
2. **Where the remaining error is.** The computed floor of order 0 is the self-consistent correction (30 % of
   the hexamers' non-additive energy, 7–14 % of the ion's induction); the measured errors sit above it by a
   fit residual that is five to ten times smaller than the clusters' and that the uniform objective does not
   reduce. The far-field reading at 10–12 Å (36–55 %) is where this shows; the next step there is the weight
   of the far-field configurations and K ≥ 4, both legitimate now that the signal is fitted.
3. **Five rules for training the read-out, each bought with a failed run:** the field tensors in units of a
   reference field and the output in units of the site-energy scale; the output layers start at zero; the
   read-out is frozen during Adam; its order-0 sector is set by its weighted linear solve; L-BFGS polishes
   everything. In one sentence: a linear-in-coefficients term inside the model is initialised by its linear
   solve and never by gradient descent — the rule M_A taught, applied late.
4. **The identification rule, sharpened:** the configurations that identify the band-field coefficient are
   those in which the source is beyond r_c of every atom; 14 such structures left the coefficient at two
   thirds, 86 brought it to within 5 %. For the real E3 that is a set of ion–water (or probe–water)
   configurations with the probe beyond r_c of every atom, from many cores, in the training set — cheap at
   any level of theory, and the far-field induction of a point charge is not a correlation problem.
5. **Order 2 is a control, not a reading, on far-field data** (§5 of the pre-registration), and its solve must
   be regularised or restricted (R1b); on the clusters with joint Adam it matched order 0 (19–22 % against
   15–36 %) — no gain and no loss, as predicted for a zero-range truth.
6. **Two things this does not settle.** Whether learned sources and the read-out can share the induction
   consistently (the tier was unreadable here for a charge-assignment reason and is deferred until the ion's
   charge is pinned); and the trade-off between the joint-Adam schedule, which was best on the clusters
   (15–36 %), and the frozen schedule, which identifies the far-field coefficient — a schedule that does both
   (joint Adam at the near scale, then the solve on the far-field residual) is the obvious candidate and is
   not run here.
7. **What goes into the real E3's protocol from here:** `M_16S0†` with permanent (isolated-monomer) multipoles
   as the v1.5 rung, trained on the five rules, with an ion-probe set in the training data and the response
   coefficient read on the isolated monomer against the CCSD(T) polarisability as the matching number (the
   molecular analogue of E2's α); the embedded dagger `M_16†ᵉᵐᵇ` as the ceiling and `M_16†` as the baseline.
