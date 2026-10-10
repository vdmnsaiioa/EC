# E3-R — the band-field response, tested explicitly (pre-registration)

*10 Oct 2026, written before the runs. The E3 rehearsal (`e3-rehearsal-results.md`) left the v1.5
claim — that the response to the band field is carried by degree-2 operators in the field tensors at the
atom with environment-dependent coefficients (`x2-protocol.md` §2) — with weak evidence on the molecular
harness: on every water run the read-out was either given no resolvable signal or was asked to carry the
wrong thing. This note says what went wrong in the test designs, fixes them with everything the rehearsal
taught, computes from the truth what an order-0 read-out can carry on each reading, and pre-registers the
numbers. Harness v0.0.6 + the changes of §2; script `scripts/e3_water.py` (`--part clusters|far|ion`),
truth `sseft/water.py`; the computations behind the numbers are `scratch/first_order_share.py` and
`scratch/ion_decomposition.py`.*

## 1. Why the rehearsal could not read F_s

The rehearsal's band-field rungs failed for four separate reasons, each now understood:

1. **Band edge.** At ℓ₁ = 1.5 Å the band field at an oxygen carries 64 % of the hydrogen-bond partner's
   field (the field's long part at r/ℓ₁ = 1.27) and less of the induced-dipole couplings: the input the
   read-out saw was not the field that polarises the molecule (§13 of the E3 pre-registration, run H).
2. **Identification.** With every structure weighted alike, the far-field configurations' induction
   (0.02–0.08 kcal/mol) sat below the fit's floor on the clusters (0.2–0.4 kcal/mol), so the response
   coefficient was never constrained (§10); the order-2 coefficients, equally unconstrained, extrapolated
   as noise (M_S2† at 240–370 %).
3. **Wrong target.** With *embedded* sources (the self-consistent induced dipoles pinned), the band already
   carries the response to the cluster's internal field, and a read-out of −½a|E_total|² double counts it
   and adds a spurious term linear in the external field (run F: the read-out's coefficients added noise
   beyond r_c). The read-out is consistent only with *permanent* sources, whose band field it is a
   response to.
4. **Near field in the input.** The own molecule's charges entered the band-field input through a radial
   switch at r_s = ℓ₁, which at ℓ₁ = 0.75 Å would let the own hydrogens in (0.96 Å). The clean rule for
   molecules is exclusion by fragment.

What the proposal's F_s actually promises, read against the truth: with permanent sources, the order-0
operator −½ a(s_i)|E_band,i|² at each polarisable site, with E_band the field of the *other* fragments'
permanent charges, is exactly the truth's **first-order induction** E_ind1 = −½ Σ_i α_i |E_q,i|² — which is
not pairwise: |Σ_j E_j|² contains the cross terms E_j·E_k between two neighbours' fields, the leading
three-body induction, and for a probe or an ion outside a cluster the cross term 2 E_core·E_probe *is*
piece (2) of the rehearsal's decomposition (the core's induced dipoles against the probe's field, to first
order). What order 0 cannot carry is the self-consistent correction E_ind − E_ind1 (the induced dipoles'
mutual fields, O(α²T)). So the prediction for every reading is computable from the truth: the share of
E_ind1 in the quantity measured is the ceiling of the order-0 read-out, and the remainder is its floor.

## 2. Changes to the harness for this test

- **Own-fragment exclusion** in the band-field input (`Rung.bf_fragments`, default on): the band field at
  an atom is the field of the other fragments' sources; open and periodic (the excluded sources' long-part
  field subtracted in real space); tested against explicit sums and against images.
- **Band edge ℓ₁ = 0.75 Å** (`--l1 0.75`), no radial switch (`--rs_over_l1 0`), r_c = 6 Å.
- **Permanent-source daggers with the dispersion band**: `M_16p` (the isolated monomer's charges pinned,
  the dispersion band on the pinned one-oscillator α_O; no read-out — the baseline), `M_16S0p` (+ order 0),
  `M_16S2p` (+ order 2). The learned-source tier: `M_1mu`, `M_S0mu` at the same band edge.
- **The ion probe** (`--part ion`): the truth gains a non-polarisable unit charge (sodium-like: exponential
  repulsion tuned to R_e(ion–O) = 2.39 Å, D_e = 21.5 kcal/mol for the ion–water dimer; it polarises the
  waters through the same smeared kernel and is itself unpolarisable), with its own fragment and its
  charge pinned. Training: the dimer window + 36 clusters + an ion–monomer scan (9 distances from 2 to
  8 Å × 6 orientations, for E₀'s short range) + ion–trimer configurations from 8 relaxed cores at
  d = 4, 5, 6, 7, 8, 10 Å from the centre of mass (one each; the far ones identify the coefficient, the
  lesson of §9–§10). Test: 4 other cores × d = 6, 7, 8, 10, 12 Å × 4 placements. Reading: the ion's
  interaction with the core minus its point-charge electrostatics with the pinned charges — the
  **induction**, i.e. the core's embedded induced dipoles against the ion's field plus the core's
  response to it — and the model's error as a share of it per distance; the training residual on the
  ion–core configurations (above their induction the test measures the floor, §10); and the
  **response coefficient** a_eff/α read directly from a monomer with the ion at 20 Å, where the read-out
  is the only term beyond electrostatics and the truth's response is exactly −½α|E|².

## 3. What the order-0 read-out can carry, computed from the truth

| reading | quantity (rms, kcal/mol) | first-order part E_ind1 | remainder (self-consistent) |
|---|---|---|---|
| hexamers (8 held out): non-additive energy | 1.287 | 0.913 = **71 %** | 0.381 = 30 % |
| water probe at 6 / 7 / 8 / 10 Å from a trimer: its induction | 0.076 / 0.050 / 0.032 / 0.018 | 93 / 85 / 88 / 89 % | 10 / 15 / 13 / 12 % |
| ion at 6 / 7 / 8 / 10 / 12 Å from a trimer: its induction | 1.27 / 0.52 / 0.34 / 0.17 / 0.15 | 95 / 95 / 93 / 89 / 86 % | 7 / 9 / 11 / 13 / 14 % |

(The ion numbers are from 4 cores × 4 placements with the seed of the test set; the script prints the
test set's own. Of the ion's induction, the core's *response* — as opposed to its pre-existing dipoles
against the ion's field — is 0.97 / 0.38 / 0.22 / 0.078 / 0.038 kcal/mol, falling as d⁻⁴·⁶; the floors of
the rehearsal's fits were 0.015–0.02 kcal/mol on a pinned-source far-field configuration and ~0.05 with
learned sources, so the ion's signal is 10–60× the floor at 6–8 Å and 2–4× at 10–12 Å, the water probe's
1–4×.)

## 4. Predictions

**R1 — clusters at ℓ₁ = 0.75 Å** (`--part clusters --rungs M_16p,M_16S0p,M_16S2p --l1 0.75 --rs_over_l1 0`,
K = 2, cached data of run G/H): M_16† non-additive rmse 80–95 % of the non-additive energy (as M_1†'s 88–93 %;
the dispersion band is pairwise), hexamer rmse 0.5–0.7 kcal/mol; **M_16S0† ≤ 45 %** (the computed floor is
30 %; success means clearly below the baseline and within a factor 1.5 of the floor), hexamer rmse
≤ 0.35 kcal/mol; M_16S2† within 1.3× of M_16S0† in both numbers (zero-range truth: nothing for the
order-2 operators to carry, and with the signal resolvable their coefficients stay near zero rather than
extrapolating as noise). A gain of order 2 over order 0 by more than a factor 1.5 would be a surprise.

**R2 — the ion** (`--part ion --rungs M_16p,M_16S0p,M_1mu,M_S0mu --l1 0.75 --rs_over_l1 0 --train_far 1
--train_far_cores 8 --test_cores 4`, K = 2): M_16† **100 ± 3 % of the induction at 8, 10, 12 Å** (the
ion beyond r_c of every atom from ~8 Å; the error is the induction by construction) and 80–110 % at 6–7 Å
(E₀ shares); **M_16S0† ≤ 20 % at 6, 7, 8 Å and ≤ 25 % at 10, 12 Å** (computed floors 7–14 %), with the
training residual on the ion–core configurations at d ≥ 6 Å **≤ 30 % of their induction** and
**a_eff/α = 1.00 ± 0.10**. Learned sources: M_1μ ≥ 60 % at 10–12 Å (the response alone is 46 / 26 % of the
induction there, and the learned sources' far field has its own ~0.05 kcal/mol floor, 30 % of the induction
at 10–12 Å) and ≥ 50 % at 6–8 Å; **M_S0μ below M_1μ by ≥ 30 points at 6–10 Å** (the response is far
above the floor there) — if it is not, the read-out does not help a model whose sources are free, which
would say that learned sources and the read-out compete for the same term, and the real E3 must pin the
sources to read F_s.

**R3 — the water probe at ℓ₁ = 0.75 Å** (`--part far --rungs M_16p,M_16S0p --l1 0.75 --rs_over_l1 0
--train_far 1 --train_far_cores 8`, K = 2, cached data of run D): M_16† 100 ± 5 % at 10 Å, 95–115 % at
6–8 Å; **M_16S0† ≤ 30 % at 6 and 7 Å, ≤ 60 % at 8 Å**; at 10 Å the floor (~0.015 kcal/mol) equals the
induction (0.018) and the number is reported, not read; the far-field training residual ≤ 50 % of its
induction (it was 134–175 % with learned sources and 42–63 % for the embedded daggers).

**What would count as failure of the v1.5 claim on the molecular harness:** M_16S0† not below 60 % on the
clusters, or above 40 % at 8–12 Å on the ion, or a_eff/α outside 0.75–1.25 with the training residual
below 30 % (the signal fitted, the coefficient wrong). Any of these with the training residual *above* its
induction is the floor again, not the claim, and is reported as such.

## 5. What is deliberately not tested here

The order-2 operators' own content — the response to the field gradient — is ∝ (C/α)/R² relative to
order 0 (C the quadrupole polarisability) and is a few per cent for sources beyond r_c; on far-field data
it cannot be resolved against the order-0 floor, and inside r_c it is E₀'s. The X2-B toy (order 2 below
order 0's ceiling on a truth with a finite-range response) stands as the evidence for the order-2
operators; here order 2 is a control. The periodic version of the read-out (Ewald band fields with the
fragment exclusion) is wired and tested but not exercised.
