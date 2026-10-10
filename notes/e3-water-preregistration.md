# E3 — Water: pre-registration (rehearsal on a synthetic polarisable truth, then the real data)

*10 Oct 2026, written before the run. The CCSD(T) data do not exist yet, so E3 is rehearsed end to end
on a synthetic truth with known answers; the same script runs on the real data when they arrive
(`scripts/e3_water.py`, `sseft/water.py`, harness v0.0.5). What is pre-registered here is (i) the
design decisions that the molecular case forced, (ii) the predictions for the rehearsal, with numbers,
and (iii) the predictions for the real data in the form they will take once the monomer multipoles
and the dimer curve are in hand. §6–§9 are addenda, each written after the runs before it and
before the run it pre-registers. The local master of this note is `notes/` in the repository.*

## 1. Two design decisions the molecular case forced

**(a) Fragment references.** The harness's interaction energy was "total minus the model's own
isolated-atom energies", exact at infinity for atoms. For molecules that leaves E_model(monomer) as
a learned constant, and a learned constant δ puts 2δ at infinity — the first smoke test showed a
flat +10⁻³ E_h tail under every rung. The model now subtracts its own energy of each fragment at
its own geometry, E_int = E(all) − Σ_f E(fragment f alone), evaluated with the same parameters;
non-interacting molecules give exactly zero again (checked to 10⁻¹⁰ at 100 Å), and the intramolecular
band energy cancels exactly. Implemented generally (`Structure.frag`, `b["frag_mask"]`,
`Rung.fragments`); the atomic case is the special case of one atom per fragment.

**(b) The near field must not enter the band-field input.** The degree-2 read-out contains
|E_near + E_far|² ⊃ 2 E_near·E_far, a term *linear in the far field* with an environment-dependent
coefficient — the operator class the degree-2 rule excludes because it is degenerate with a dipole
source (`x2-protocol.md` §2.1). For a rigid molecule E_near — the ℓ₁-smeared field of its own
charges at its own atoms — is a fixed vector in the molecular frame, so the read-out would produce a
spurious permanent dipole α E_near; for water at 3 Å the term is ~0.5 kcal/mol, a tenth of the
binding energy, and nothing in the model can cancel it with pinned sources. The band-field input now
carries a smooth near-source switch, w(r) = 1 − exp(−(r/r_s)¹⁰) with r_s = ℓ₁ (1.5 Å: the own H's at
0.96 Å enter at 1 %, the H-bond partner at 1.9 Å at 100 %); the near sources are E₀'s. The X2-B toy
is unaffected (r_s = 0 there, as in its truth). Periodic version: Ewald minus the switched-off near
part in real space (tested against explicit images).

A third point, recorded as a trap: at the global-minimum-like orientation (acceptor bisector tilted
57° from the O–O axis) the dipole–dipole coefficient of the rigid-monomer dimer *vanishes*
(1 − 3 cos 52° cos 57° ≈ 0) and the curve is dipole–quadrupole, p = 4, out to ~900 Å. The scan
orientation is therefore the acceptor dipole along the axis (φ = 0), where E R³ → −0.652 a.u. and
p⋆ = 3 is clean. The real-data scan must use the same orientation.

## 2. The synthetic truth PW (`sseft.water`)

Rigid gas-phase monomers (r_OH = 0.9572 Å, 104.52°), atom-centred charges reproducing μ = 1.855 D
(q_H = 0.3296), intermolecular Coulomb smeared at 0.5 Å, an isotropic polarisable site on each O
(α = 9.72 a.u.) responding self-consistently to the other molecules' charges and induced dipoles,
exponential O–O / O–H / H–H repulsion and a damped O–O C₆ = 45.4 a.u. Dimer minimum at φ = 0:
R_e = 2.86 Å, D_e = 4.1 kcal/mol; forces by autodiff (checked against finite differences to 10⁻⁹).
Its many-body energy is pure induction; the non-additive part of a hexamer is the self-consistent
dipole term.

Design, as E1: 40 log-uniform O–O nodes on [1.35 R_e, 1.8 × that] = [3.86, 6.95] Å, σ = 10⁻⁹ E_h,
isolated monomers at zero, E₀ cutoff 6 Å, Adam 1500 + L-BFGS 300, K = 4; evaluation to 500 Å.
Clusters: n = 3, 4, 5 (12 each) for training, 8 hexamers and 8 dimers held out, each a random
rigid packing relaxed by 300 rigid-body Metropolis steps at 300 K on the truth.

## 3. Predictions for the rehearsal

**Dimer, p⋆ = 3** (relative error of the mean, W/|f⋆|, q on the far grid; the plateau slope on
[30, 300] Å):

- M_∞: window rmse ≫ σ; relative error 1.000 beyond 6 Å; spread 0. Branch 3.
- M_G: q → −1 or 0; relative error > 10 at 20 Å. Branch 1.
- M_A(ℓ = 4 Å, N = 4): the burst, then q rising through 3 at R_q3 ≈ ℓ√(N + 3/2) = 9.4 Å (plus 1–3 Å)
  and growing in R′² with slope 2/ℓ² = 0.125 Å⁻² ± 20 %; relative error 1 from ~15–20 Å; float floor
  beyond ~30 Å. Branch 3 after the burst.
- **M_1 (q learned).** Beyond r_c each monomer carries its isolated-monomer charges, neutral by the
  shift, so the tail is the Coulomb interaction of two learned charge sets: q = 3.00 beyond 30 Å,
  W/|f⋆| flat (slope 0 ± 0.1), branch 2. **The learned monomer dipole is identified**: the class
  beyond r_c is a one-parameter family (the charge scale λ, all multipoles ∝ λ), and the minimum of
  the training objective on the 10 nodes beyond 6 Å is λ̂ = 1.001 → **μ̂ = 1.857 D against 1.855**,
  with a seed spread ≤ 3 % (ten identifying nodes). Energies-only the bias would be +1.7 % (induction
  pulls the scale up); the forces pull it back.
- M_1μ (q and atomic dipoles learned): the same tail and the same total monomer dipole (1.857 D,
  spread ≤ 3 %); the split between charges and atomic dipoles is not identified (q_O scattered across
  seeds by more than the dipole is).
- **M_1† (q pinned to the truth's charges).** The electrostatic tail is exact; what is missing is the
  induction, ∝ R⁻⁶: relative error ≈ 1.5×10⁻³ at 20 Å falling as R⁻³ (10⁻⁵ at 100 Å), spread
  exactly 0 beyond r_c.

**Clusters** (rmse over the 8 held-out hexamers; the non-additive energy is E(hexamer) minus the
model's own 15 dimer energies, compared with the truth's):

- v1 dagger (M_1†): the whole non-additive energy must come from E₀ within 6 Å; with 36 training
  clusters, non-additive rmse 50–100 % of the non-additive energy (the induction is learnable by a
  local model in principle; it is data-limited here).
- v1.5 dagger at order 0 (M_S0†): the −½ a(env)|E|² structure is the truth's own form (a point
  polarisability on O); non-additive rmse ≤ 30 % of the non-additive energy, hexamer rmse below
  v1's by a factor ≥ 2.
- v1.5 at order 2 (M_S2†): **no measurable gain over order 0** (within the seed spread): the truth's
  response has zero range, so the derivative orders carry nothing — the counting's "order 0 carries
  almost all" statement, with a point response as the extreme case. The order-2 run is a control.
- With learned charges (M_S0 vs M_1): the same ordering, each a little worse than its dagger.

## 4. Predictions for the real data (to be filled in when they exist, in this form)

Same rungs and design on the CCSD(T)/CBS fixed-orientation curve (φ = 0) and the n = 2–6 clusters.
M_1†'s sources will be the monomer's distributed multipoles at the same level of theory (charges and
atomic dipoles; a dagger with charges alone cannot reproduce the quadrupole). Predictions: the learned
total monomer dipole from M_1 / M_1μ within a few % of the CCSD(T) monomer dipole, with the bias
computed under the objective from the curve itself before training; M_1† relative error at 20 Å set
by the induction plus dispersion tails relative to dipole–dipole (to be computed from the SAPT
components); order 0 carrying ≥ 90 % of the many-body induction on the clusters and the band-edge
level flat within the seed spread. Numbers go into this note before the real run, not after.

## 5. What would count as a surprise

A learned monomer dipole off by more than 5 % with a seed spread below that (identification failing
for a reason other than too few nodes); q ≠ 3.00 beyond 30 Å for M_1; an order-2 gain of more than
a factor 1.5 over order 0 on the clusters (a non-zero-range response where the truth has none); a
v1 non-additive error below 30 % (E₀ learning induction from 36 clusters better than expected).

## 6. Addendum (after the dimer and cluster runs, before the far-field run)

**The cluster prediction for v1.5 was wrong, and the reason is a statement about the design, not a
bug.** M_S0† gave a non-additive rmse of 73–85 % of the non-additive energy, the same as v1's
88–93 %, not the predicted ≤ 30 %. The band field at an oxygen is the field of the *far* sources
through the ℓ₁-smeared kernel: the H-bond partner's H at 1.9 Å contributes only 64 % of its true
field to it (erf(r/ℓ₁) minus its derivative term at r/ℓ₁ = 1.27), and a source at 2.8 Å 93 %. In a
hexamer every molecule is a near source of every other, so the induction there is near-field
induction — E₀'s territory, which the band-field inputs were never meant to carry and E₀ cannot
learn from 36 clusters. The counting's "order 0 carries almost all of the response *to the band
field*" was conflated with "almost all of the induction"; on small clusters the two are different
things. The claim the v1.5 inputs actually make is about the response to sources beyond ~2–3 ℓ₁,
i.e. beyond the cutoff region in practice, and that is what has to be tested.

**The far-field induction test, pre-registered now.** Training: the dimer window (its ten nodes
beyond r_c are the only data that teach the response to the band field, since there E₀ sees
nothing) plus the 36 clusters. Test: a probe monomer at d = 6, 7, 8, 10 Å from a relaxed trimer, in
random directions and orientations (8 per distance); the probe's interaction energy with the core,
E(core + probe) − E(core), and its induction part, the same minus the point-charge electrostatics
(plus the negligible repulsion and dispersion at these distances). Beyond r_c the probe's interaction
in the model is the band alone: for v1 that is electrostatics only, so **M_1†'s error equals the
induction, 100 % of it at every distance** (the electrostatics itself is exact to the kernel
difference, 10⁻⁸ at 6 Å); for v1.5 the order-0 term −½ a(s_O)|E_band|² with a(s_O) learned from the
dimer's far nodes represents it, so **M_S0†'s error is ≤ 20 % of the induction at every distance**
(the tolerance covers how well ten nodes pin a(s_O) ≈ α); **M_S2† the same as M_S0† within the seed
spread** (zero-range truth, the order-2 terms carry nothing). The induction is ~0.3–0.7 % of the
probe's interaction at these distances, so the test is on a small quantity; the per-distance ratios
are the reading.

## 7. Second addendum (after the far-field run of §6, before the run with far-field training data)

**The §6 prediction for v1.5 was wrong too, and this time the reason is identification, not
representability.** On the far-field test M_1† erred by 111–112 % of the induction (as predicted,
~100 %), M_S0† by 80–89 % (predicted ≤ 20 %), and M_S2† by 240–300 % — worse than having no
response term at all. The induction at the dimer window's ten far nodes is ~10⁻⁶ E_h, below the
fit's own floor there (window rmse 8×10⁻⁶): the training set contains **no configuration in which
the response to the band field is resolvable**, so the order-0 coefficient a(s_O) is unconstrained,
and the order-2 coefficients, equally unconstrained, extrapolate as noise. (The induction turned out
to be 14 % of the probe's interaction at 6–10 Å, not the 0.3–0.7 % estimated in §6 — the trimer's net
dipole and its own polarisation in the probe's field were left out of that estimate — so the test is
not on a small quantity after all.)

**Consequence for the real E3, stated now:** a dimer curve plus small clusters cannot identify the
far-field response; the training set must contain far-field-dominated configurations (probes beyond
r_c, or periodic cells), or the response coefficients must come from matching data (SAPT induction
components). This is the molecular analogue of E2's gauge statement: energies in the regime where
E₀ sees everything do not pin the band-field response.

**Pre-registered run:** the same test with 4 probe configurations per distance from a *different*
trimer (seed 400) added to the training set (16 structures, ~15 % of the set). Predictions: M_1†
unchanged, ~100 % of the induction (it has nothing to represent it with); **M_S0† ≤ 30 % of the
induction at every distance**; M_S2† within a factor 1.5 of M_S0† (its extra coefficients are now
constrained by the same data, and the zero-range truth gives them nothing to do).

## 8. Third addendum (after the run of §7, before the learned-source run it pre-registers)

**The §7 prediction was wrong as well: with 16 far-field configurations in the training set, M_1†
gives 99–106 % of the induction, M_S0† 82–92 %, M_S2† 300–370 % (worse at 6–7 Å, where probe–core
pairs fall just inside r_c and the order-2 pair terms extrapolate). Identification was not the
limiting factor; the decomposition of the "induction" was.** The probe's interaction minus the
point-charge electrostatics contains three pieces: (1) the probe's own response to the core's far
field, −½α|E_core|², ∝ R⁻⁶; (2) the interaction of the core's *pre-existing* induced dipoles — set
by its own near field, 0.1–0.3 a.u. per oxygen in a hydrogen-bonded trimer — with the probe's
charges, ∝ R⁻³ like the electrostatics itself; (3) the core's additional response to the probe's
field, ∝ R⁻⁶. The test is dominated by (2): that is why the "induction" is a constant 14 % of the
interaction from 6 to 10 Å instead of falling as R⁻³ relative to it. And (2) is not a response to
the band field at all — it is an **environment-dependent source**: the induced dipole of each core
oxygen is a function of its local environment, exactly what v1's learned sources are. A pinned-charge
dagger cannot carry it by construction (its sources are the isolated monomer's), and no degree-2
read-out can either, because at the probe the term is linear in the probe's field. The band-field
response (1) + (3) is the R⁻⁶ remainder, a few per cent of the induction at these distances.

So the far-field test separates the two things the proposal keeps distinct: sources that run with
the environment (v1) and the response to the band field (v1.5). The right comparison is between
rungs with learned q and μ with and without the band fields.

**Pre-registered run:** M_1μ (learned q and atomic dipoles, no band fields) and M_S0μ (the same with
the order-0 band-field read-out), same training set as §7 (dimer window, 36 clusters, 16 far-field
configurations). Predictions: **M_1μ captures piece (2): error ≤ 30 % of the induction**, falling
with distance in absolute terms as R⁻³ (the residual is (1) + (3), ∝ R⁻⁶, so the per-distance ratio
drops from 6 to 10 Å); **M_S0μ ≤ 15 %**, the band-field term removing most of the remainder; the
learned induced dipoles are identified only by the far-field configurations (inside r_c, E₀ and the
band are degenerate), so with 16 of them the seed spread is the measurement of how well.

**For the real E3 this changes the dagger:** M_1† must pin environment-dependent sources (distributed
multipoles of the monomer *in the cluster*, not of the isolated monomer) or be read as the
"isolated-monomer sources" baseline it is; the environment dependence of the sources — induction at
first order — is the v1 claim and is what the learned μ_i test.

## 9. Fourth addendum (after the run of §8, before the re-run with a corrected training design)

**The §8 prediction failed, and this time the test was at fault.** M_1μ gave 79–120 % of the induction
and M_S0μ 65–117 % (seed 1 of M_S0μ: 72 / 53 / 46 / 68 % at 6 / 7 / 8 / 10 Å, the best any rung has
done; seed 0: 138 / 76 / 72 / 65 %). The 16 far-field training configurations of §7–§8 all used **one
and the same core trimer** (seed 400) with 16 probe placements; the test uses another core (seed
300). The induced-dipole map μ_i(environment) was therefore shown exactly one environment and asked
to generalise to a second — which it cannot, whatever the architecture. The test, not the model,
was under-determined. (The per-distance pattern of the best seed — errors smallest at 7–8 Å, larger
at 6 Å where probe–core pairs enter r_c and at 10 Å where the signal is smallest — is what a
partially learned map looks like.)

**Pre-registered re-run:** far-field training configurations from **8 different relaxed trimers**
(seed 400), one probe per distance each (32 structures), test on the seed-300 trimer as before;
rungs M_1†, M_1μ, M_S0μ, K = 2. Predictions: M_1† ~100 % (unchanged); **M_1μ ≤ 50 %** of the
induction (eight environments are few for a map from local structure to induced dipole; the number
is a guess with that caveat, and the seed spread is part of the reading); **M_S0μ below M_1μ by at
least the R⁻⁶ share, i.e. a visible gap at 6–7 Å and none at 10 Å**. If M_1μ stays near 100 % with
eight cores, the learned-source map is not learnable from energies and forces at this data scale,
and the real E3 needs matching targets for the embedded monomer multipoles rather than more
configurations.

## 10. Fifth addendum (after the first half of the §9 run, before its completion and the weighted re-run)

**Status of §9.** The eight-core run was killed by the container's memory limit half-way (a smoke
test of a script change ran beside it; nothing else may run beside a far-field run). What it had
produced: M_1† 98 % and 114 % of the induction (as predicted); **M_1μ seed 0: 86 %** (94 / 67 / 73 /
90 % at 6 / 7 / 8 / 10 Å) — not the ≤ 50 % predicted, and the same per-distance shape as the best
seed of §8. The remaining seeds and M_S0μ are re-run from the cached data sets (`--cache`; the
generation is seeded, so the sets are identical).

**Before concluding that the source map is not learnable (the §9 fallback), the floor has to be
measured.** The training loss weights every structure alike; the clusters' residual (≳ 0.1 kcal/mol
per structure in the earlier runs) is larger than the far-field configurations' whole induction
(0.05 kcal/mol), so the optimiser may never have fitted the far-field signal at all — the §7
lesson again, at the level of the fit rather than the data. The script now reports, per seed, the
residual on the training clusters and on the far-field *training* configurations as a share of
their own induction, and takes a weight `--w_far` for those configurations.

**Pre-registered:** (a) at weight 1 the residual on the far-field training configurations is
**≥ 50 % of their induction** (the signal was below the floor; if it is ≤ 20 % the floor is not the
explanation and the map is simply not generalising); (b) at weight 10 that residual falls **below
30 %**; (c) the test error at weight 10 is then the reading of generalisation from eight cores:
**≤ 50 %** means the map is learnable from energies and forces at this scale and the real E3 needs
far-field configurations with a weight, not matching targets; **≥ 70 %** means it is not, and the
real E3 needs the embedded-monomer multipoles as matching targets (§9's conclusion, now earned).
Between the two the question stays open and K = 4 decides.

## 11. Sixth addendum (after the completed §9 run and the first seed of the weighted run; before the embedded dagger)

**§10's prediction (a) held and (b) failed.** Completed eight-core run: M_1† 98 / 114 %, M_1μ 86 / 116 %,
M_S0μ 80 / 108 % of the induction; the residual on the far-field *training* configurations is 134–163 %
of their own induction for every seed of every learned-source rung — the far-field signal was indeed
never fitted (a). But with the far-field configurations weighted ×10 the residual does not move
(170 % for the first seed; test 124 %): the floor is not the weighting. Something in the model class
or the optimisation leaves ~0.05 kcal/mol on a probe configuration whose electrostatics is 0.35 and
whose induction 0.03–0.08 kcal/mol — the learned sources' far field is wrong by about the induction
itself (a learned charge-plus-dipole monomer has a free quadrupole that nothing in the training set
pins, is the candidate), and no reweighting of a signal the class cannot fit helps.

**The decomposition, computed exactly** (`scratch/far_decomposition.py`, on the cached test set):
piece (2), the core's pre-existing induced dipoles against the probe's charges, is 91 / 100 / 93 /
98 % of the induction rms at 6 / 7 / 8 / 10 Å; the remainder (1) + (3), the mutual response, is
**40 / 16 / 14 / 5 %** (0.031 / 0.008 / 0.004 / 0.0008 kcal/mol), 33 % overall, dominated by 6 Å
where the probe's nearest atoms sit 3–4.5 Å from the core.

**Pre-registered: the embedded dagger M_1†ᵉᵐᵇ** (`M_1pemb`: q pinned to the monomer charges, atomic
dipoles pinned to the truth's self-consistent induced dipoles of each molecule *in its own cluster*;
a probe outside the cluster carries the isolated monomer's zero; the core's dipoles are the same in
E(core + probe) and E(core), so the core's internal band cancels in the difference). Beyond r_c its
probe interaction is electrostatics + piece (2) with no training involved, so its error is the
remainder: **4.6 % ± 0.5 at 10 Å** (every probe–core pair beyond r_c there: a test of the
implementation to the digit), **≈ 14 % at 8 Å and ≈ 16 % at 7 Å** (E₀ sees a few pairs; ± 5), and
**≤ 40 % at 6 Å** (E₀ can fit part of the response there). Same training set as §9 (dimer window,
36 clusters, 32 far-field configurations, all with embedded dipoles pinned), K = 2. Control:
M_S0†ᵉᵐᵇ (the same plus the order-0 read-out), predicted equal to M_1†ᵉᵐᵇ within the seed spread —
the response coefficient is still unidentified at this floor, and the remainder it would have to
carry (0.016 kcal/mol overall) is below it.

**What this settles for the real E3, whichever way the learned-source question goes:** the far-field
many-body energy of a hydrogen-bonded cluster at 6–10 Å is, to 85–95 %, the field of its embedded
induced dipoles. A dagger with embedded multipoles (distributed multipoles of each monomer in its
cluster, from the CCSD(T) or a DFT density) carries it with no fitting; learned sources have to
recover the same map from energies and forces, and on the synthetic truth they do not at this data
scale. The real E3 should run both and read the gap between them as the price of learning the
sources.
