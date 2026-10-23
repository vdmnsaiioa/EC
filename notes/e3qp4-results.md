# E3-QP4 — E₀'s width, the energy term's balance, the read-out's near field: results

*23 Oct 2026. Both jobs of 22 Oct (sd657, one H200 each, commit c7bcafe = v0.0.13 as applied; in each the clusters part
3.4 h, the diagnostic 48 min), read against `e3qp4-preregistration.md`: job A (E₀'s width 32) against E3-QP3, job B (job
A with the contact configurations' energies at weight 0.125) against job A. Seed means over the 4 seeds unless said,
kcal/mol; `scratch/e3qp4_summarise_log.py` on the two job logs, against E3-QP3 (`scratch/e3qp3_summary.json`). §3.6 is
computed from the PES and the data alone (`scratch/e3qp4_trimer_coverage.py`). The local master of this note is
`notes/` in the repository.*

## 1. Verdict

- **The balance is a limit for the pinned rungs, not for M_∞** (job B). With the contacts' energies at 0.125 (their
  share of the energy term at equal errors 50 % against 89 %) every pinned rung improved on everything but the contacts
  themselves: the relaxed training clusters −26 to −33 %, their forces −8 to −12 % (rigid and internal alike), the
  hexamers −7 to −10 %, the test dimers −7 to −12 %, the raw 3-body −4 to −8 %; the compressed dimers held (−1 to
  −4 %) with an eighth of the contacts' energy weight — the pinned bands carry the 2-body at contact. M_∞ traded: its
  clusters −16 %, its 3-body −7 %, its 4-body −21 %, but its raw 2-body +19 % (compressed +16 %), its forces and its
  hexamers flat (+1 %) — its 2-body at contact is learned from the contacts' energies.
- **M_16C2†ᵐ of job B is the best model of the series**: hexamers 0.160 (E3-QP2 0.201, E3-QP3 0.189, job A 0.178),
  their non-additive energy 13.1 % (M_∞ 22.8 %), test dimers 0.046, raw 2-body 0.082 (M_∞ 0.113), raw 3-body beyond
  O–O 3 Å 0.032 (M_∞ 0.051), the far field exactly as computed.
- **The width is a limit, but a small one** (job A): every rung's training energies and forces improved together
  (energies −14 to −18 %, rigid-body forces −10 to −11 %, internal −17 to −19 %), the test sets less (the raw 2-body −6
  to −16 %, the raw 3-body −9 to −13 %, the hexamers −3 to −8 %); the raw 4-body got worse (§3.4).
- **The pinned rungs' 3-body gap to M_∞ is neither the width's nor the balance's.** Their raw 3-body over M_∞'s:
  M_16†ᵐ 1.40 / 1.37 / 1.36 and M_16C2†ᵐ 1.53 / 1.48 / 1.53 in E3-QP3 / job A / job B, M_16C2†ᵐ_r1 1.35 / 1.33. Two
  deficits (§3.3): M_16†ᵐ's is E₀'s, in the rigid part and largest on the strict near-rigid subset (1.51× M_∞'s in B),
  where the C rungs' read-out narrows it (1.25×, 1.15×); the C rungs add the read-out's near field below O–O 2.7 Å, in
  the rigid part and in the distortion part (twice M_∞'s).
- **The read-out at 1.0 Å is a trade, not the better C rung.** Against M_16C2†ᵐ, M_16C2†ᵐ_r1 is better on the raw
  3-body on 8 of 8 seeds (below 2.7 Å on 7) and on the raw 4-body on 7, worse on the hexamers' energy on 7 and their
  non-additive energy on 6 (15.1 against 13.1 % in B, 16.0 against 14.2 % in A).
- **The far field is as computed** on every seed in both jobs, to the same digits in both: M_16C2†ᵐ 4 / 5 / 3 % and
  M_16C2†ᵐ_r1 3 / 3 / 2 % at 10 / 12 / 15 Å, M_16†ᵐ's interaction 10 / 9 / 15 %; the CCSD(T) trimers beyond 9 Å at
  26–27 % for the C rungs and 100 % for the others.

## 2. The predictions

Job A:

| registered (§4) | job A (M_∞ / M_16†ᵐ / M_16C2†ᵐ / M_16C2†ᵐ_r1) | |
|---|---|---|
| far field at 10–15 Å on every seed | as computed (§1) | ✓ |
| training clusters ≥ 15 % below E3-QP3's, forces not worse: M_∞, M_16†ᵐ | −13.6 %, −17.6 %; forces −11 / −17 % for both | M_∞ ✗ (by a hair), M_16†ᵐ ✓ |
| training clusters: M_16C2†ᵐ ≤ 0.11 | 0.171 / 0.128 / 0.111 / 0.104 | ✗ (by a hair) |
| hexamers: M_16†ᵐ ≤ 0.24, M_16C2†ᵐ ≤ 0.17, M_∞ 0.35–0.50 | 0.431 / 0.247 / 0.178 / 0.197 | ✗ ✗ ✓ |
| test dimers: pinned ≤ 0.05, M_∞ ≤ 0.07 | 0.063 / 0.051 / 0.053 / 0.051 | ✗ (by a hair) / ✓ |
| raw dimers: pinned ≤ 0.085 and below M_∞ on every seed; M_∞ ≤ 0.105 | 0.095 / 0.082 / 0.088 / 0.079; 15 of 16 (M_16†ᵐ seed 3: 0.0874 against 0.0868) | ✗ / ✓ |
| compressed dimers (2.3–2.7 Å) ≤ 0.12 for every rung | 0.098 / 0.103 / 0.115 / 0.109 | ✓ |
| raw 3-body: M_∞ ≤ 0.095, M_16†ᵐ ≤ 0.13, M_16C2†ᵐ ≤ 0.145 | 0.091 / 0.124 / 0.134 / 0.122 | ✓ |
| r1: raw 3-body below 2.7 Å lower on ≥ 3 of 4 seeds | 3 of 4 | ✓ |
| r1: hexamers' non-additive at or below M_16C2†ᵐ's | 16.0 against 14.2 % | ✗ |
| kept: raw 4-body M_16C2†ᵐ ≤ 55 % | 65 / 132 / 67 / 45 % | ✗ |
| kept: hexamers' non-additive M_16C2†ᵐ ≤ 17 % | 23.8 / 26.5 / 14.2 / 16.0 % | ✓ |
| kept: trimers beyond 9 Å, C rungs 24–28 %, others 95–105 % | 99.8 / 100.0 / 26.7 / 25.8 % | ✓ |

19 of 28 checks as registered.

Job B:

| registered (§4) | job B (M_∞ / M_16†ᵐ / M_16C2†ᵐ / M_16C2†ᵐ_r1), against job A | |
|---|---|---|
| far field at 10–15 Å on every seed, the same in both jobs | as computed, job A's digits | ✓ |
| relaxed training clusters ≥ 10 % below A's | 0.143 / 0.086 / 0.081 / 0.077 against 0.171 / 0.128 / 0.111 / 0.104 | ✓ ✓ ✓ ✓ |
| contacts above A's, ≤ 0.15 | 0.090 / 0.073 / 0.080 / 0.079 against 0.063 / 0.060 / 0.064 / 0.062 | ✓ ✓ ✓ ✓ |
| compressed dimers within 0.02 of A's | 0.114 / 0.099 / 0.111 / 0.107 against 0.098 / 0.103 / 0.115 / 0.109 | ✓ ✓ ✓ ✓ |
| hexamers below A's | 0.435 / 0.226 / 0.160 / 0.184 against 0.431 / 0.247 / 0.178 / 0.197 | ✗ ✓ ✓ ✓ |
| r1: raw 3-body below 2.7 Å lower on ≥ 3 of 4 seeds | 4 of 4 | ✓ |
| r1: hexamers' non-additive at or below M_16C2†ᵐ's | 15.1 against 13.1 % | ✗ |
| kept: raw 4-body M_16C2†ᵐ ≤ 55 % | 51 / 121 / 57 / 45 % | ✗ |
| kept: hexamers' non-additive M_16C2†ᵐ ≤ 17 % | 22.8 / 26.0 / 13.1 / 15.1 % | ✓ |
| kept: trimers beyond 9 Å, C rungs 24–28 %, others 95–105 % | 100.0 / 99.8 / 26.7 / 25.8 % | ✓ |

24 of 27 checks as registered; 43 of 55 in E3-QP4. No registered surprise: the far field exact and the same in both
jobs; the width's gain above 5 % with the forces improving too; the compressed dimers not 0.05 above A's (they fell for
the pinned rungs and rose 0.016 for M_∞); the relaxed clusters better than A's.

## 3. Reading

**3.1 What the width bought**, job A against E3-QP3, change of the seed means:

| | training E; F rigid / internal | contacts | raw 2-body | raw 3-body | hexamers; non-additive | test dimers | raw 4-body |
|---|---|---|---|---|---|---|---|
| M_∞ | −14; −11 / −17 % | −24 % | −16 % | −9 % | −3; −1 % | −15 % | +10 % |
| M_16†ᵐ | −18; −11 / −17 % | −24 % | −12 % | −11 % | −8; −15 % | −13 % | +27 % |
| M_16C2†ᵐ | −14; −10 / −19 % | −20 % | −6 % | −13 % | −6; −11 % | −7 % | +37 % |

For the C rung the width also moved the read-out's near field (computed: the hexamers' many-body 29 against 21 % from
TTM3-F's at width 16), so M_∞ and M_16†ᵐ are the clean reading: training errors down by a sixth buy a tenth to a sixth
on the raw sets and little on the hexamers.

**3.2 What the balance bought**, job B against job A, change of the seed means:

| | training E; F rigid / internal | contacts | raw 2-body; compressed | raw 3-body; below 2.7 Å | hexamers; non-additive | test dimers | raw 4-body |
|---|---|---|---|---|---|---|---|
| M_∞ | −16; +1 / +2 % | +43 % | +19; +16 % | −7; −13 % | +1; −4 % | +7 % | −21 % |
| M_16†ᵐ | −33; −11 / −12 % | +21 % | −8; −4 % | −8; −1 % | −8; −2 % | −8 % | −8 % |
| M_16C2†ᵐ | −27; −11 / −10 % | +24 % | −7; −4 % | −4; −5 % | −10; −8 % | −12 % | −15 % |
| M_16C2†ᵐ_r1 | −26; −8 / −11 % | +26 % | +2; −1 % | −8; −15 % | −7; −6 % | −7 % | +1 % |

and job B's readings:

| job B | training E; F rigid / internal | contacts | hexamers (2-body, rest); non-additive | test dimers | raw 2-body; compressed | raw 3-body | raw 4-body |
|---|---|---|---|---|---|---|---|
| M_∞ | 0.143; 0.533 / 0.552 | 0.090 | 0.435 (0.332, 0.228); 22.8 % | 0.068 | 0.113; 0.114 | 0.084 | 51 % |
| M_16†ᵐ | 0.086; 0.393 / 0.491 | 0.073 | 0.226 (0.148, 0.259); 26.0 % | 0.047 | 0.075; 0.099 | 0.115 | 121 % |
| M_16C2†ᵐ | 0.081; 0.410 / 0.526 | 0.080 | 0.160 (0.129, 0.131); 13.1 % | 0.046 | 0.082; 0.111 | 0.129 | 57 % |
| M_16C2†ᵐ_r1 | 0.077; 0.404 / 0.521 | 0.079 | 0.184 (0.124, 0.151); 15.1 % | 0.048 | 0.080; 0.107 | 0.112 | 45 % |

At weight 1 the 800 contact configurations held 89 % of the energy term at equal errors (the 1 / n² weight), at 0.125
half. For the pinned rungs their energies were more burden than lever: the contacts' own error rose a fifth to a
quarter and everything else fell, the forces with the energies — the contacts' energies had pulled E₀ away from the
clusters' forces — and the compressed dimers did not rise: the pinned bands carry the 2-body at contact, the contacts'
forces (weight 1) the rest. M_∞ has no pinned 2-body: its contacts' error rose 43 %, its raw 2-body 19 % (back to
E3-QP3's 0.113), its forces did not move, and the clusters' gain (−16 %; the 3-body −7 %, the 4-body −21 %) left its
hexamers where they were (their 2-body part 0.310 → 0.332, the rest 0.237 → 0.228). The physics in the pins is what
lets the pinned rungs take the contacts' energies at an eighth of the weight.

**3.3 The 3-body below 3 Å.** Job B's raw 3-body by part (the trimers rigidified, and the distortion from them;
`scratch/e3qp3_shortrange_diagnostic.py`, the pins as trained), seed means:

| job B | all | rigid | distortion | strict (1450) | below 2.7 Å: all / rigid / distortion | 2.7–3.0 Å: all / rigid | ≥ 3 Å |
|---|---|---|---|---|---|---|---|
| M_∞ | 0.084 | 0.081 | 0.015 | 0.072 | 0.126 / 0.109 / 0.038 | 0.086 / 0.084 | 0.051 |
| M_16†ᵐ | 0.115 | 0.102 | 0.023 | 0.109 | 0.192 / 0.148 / 0.061 | 0.115 / 0.106 | 0.056 |
| M_16C2†ᵐ | 0.129 | 0.106 | 0.032 | 0.090 | 0.267 / 0.200 / 0.087 | 0.122 / 0.105 | 0.032 |
| M_16C2†ᵐ_r1 | 0.112 | 0.092 | 0.029 | 0.083 | 0.225 / 0.167 / 0.077 | 0.108 / 0.092 | 0.031 |
| the labels' rms | 0.623 | | 0.061 | | 1.268 | 0.596 | 0.145 |

Two deficits. **M_16†ᵐ's is E₀'s**: in the rigid part (0.102 against 0.081; 1.26× at 2.7–3.0 Å, 1.36× below), and
largest on the strict near-rigid subset (0.109, its rigid part 0.108, against 0.072: 1.51×; job A 1.60×), where the
truth's charge flux has no part. M_16†ᵐ fits its training clusters far better than M_∞ (0.086 against 0.143) and generalises its 3-body
worse: what E₀ is given, not how much of it there is (§3.1). **The C rungs' read-out narrows it on the near-rigid
trimers** (0.090 and 0.083: 1.25× and 1.15×; job A 1.29×, 1.15×) — an explicit induction helps E₀ there — **and adds its
own below 2.7 Å**: the rigid part 0.200 / 0.167 against M_16†ᵐ's 0.148 and M_∞'s 0.109, the distortion part 0.087 /
0.077 against 0.061 and 0.038 — the read-out's response at the compressed contact, regulated by its edge (the 1.0 Å
edge lowers both). Beyond 3 Å the C rungs lead (0.031–0.032 against 0.051–0.056). Job A reads the same (M_∞ 0.091,
M_16†ᵐ 0.124, M_16C2†ᵐ 0.134, M_16C2†ᵐ_r1 0.122; below 2.7 Å 0.145 / 0.194 / 0.280 / 0.264). Neither the width
(§3.1), nor the balance (§3.2), nor coverage (§3.6), nor the pinned bands' edge (E3-QP3's _e1 rungs) moved the ratio.

**3.4 The 4-body.** The 4-body set is small (267 tetramers, 15 with a contact below 2.7 Å) and its error moves with E₀'s
many-body freedom: the wider E₀ raised it for every rung (job A: M_∞ 59 → 65 %, M_16†ᵐ 104 → 132 %, M_16C2†ᵐ 49 →
67 %), the balance lowered it again but for M_16C2†ᵐ_r1 (job B: 51 / 121 / 57 / 45 %). M_16†ᵐ, with nothing but E₀ for
it, puts more error into the 4-body than the 4-body holds (below 2.7 Å 0.45–0.51 against M_∞'s 0.14–0.19); the C
rungs' read-out carries the many-body induction, and the 1.0 Å read-out's 4-body is the best of the run (45 % in both
jobs; below 2.7 Å 0.15–0.16).

**3.5 The read-out at 1.0 Å, both jobs.** M_16C2†ᵐ_r1 against M_16C2†ᵐ, per seed (8): the raw 3-body lower on 8
(below 2.7 Å on 7, at 2.7–3.0 Å on 8), the raw 2-body and the raw 4-body on 7, the hexamers' 2-body part on 6; the
hexamers' energy higher on 7 and their non-additive energy on 6 (by −0.008 to +0.054 kcal/mol). The raw trimers and the
hexamers disagree at the same contacts (2.7–3.0 Å, where r1 is better on every seed on the raw set). One reading: the
hexamers' many-body sums 20 trimers over relaxed hydrogen-bond networks, so a bias the trimers' rms hardly sees adds up
there. What the two edges trade is the read-out's near-field form (its damping), not tested.

**3.6 The trimers are not outside the training data** (computed). Against every three-molecule subset of the training
data — the 300 relaxed clusters' 1500 triples and the 200 contact trimers — in the 27 intermolecular atom distances (rms,
over the 48 orderings), the raw trimers lie no farther than the test hexamers' own triples do: median 0.27 against
0.35 Å; beyond 95 % of the hexamers' triples (0.51 Å) only 7 % of the raw trimers, all of them extended (largest O–O
7–9 Å 67 %, beyond 9 Å 100 %, where the 3-body is a few hundredths), against 2 % below O–O 2.7 Å and 3 % at 2.7–3.0 Å.
Unlike E3-QP2's compressed dimers, the 3-body at contact is not an extrapolation beyond the training data on this
measure; the triples' space is sparsely sampled for the test hexamers as much as for the raw trimers.

## 4. Next

- **Carried forward**: E₀'s width 32, the contacts' energies at 0.125, M_16C2†ᵐ (the read-out at 0.75 Å) as the C
  rung — the hexamers' many-body, which adds over hydrogen-bond networks as the liquid's does, prefers it on 6–7 of 8
  seeds; M_16C2†ᵐ_r1's gain on the raw sets belongs to the read-out's near-field form. M_∞ stays the reference (on its
  own it would keep the contacts' energies at weight 1, for its 2-body).
- **The pinned rungs' 3-body at contact** is the one reading M_∞ still wins. Of job A's two candidates, (b) leads: E₀
  given the pinned charges' field at each atom, from the other molecules within r_c. With the pins rigid it is a fixed
  function of the geometry — no information E₀ could not build, only its shape — and local, so the far field stays as
  computed. For it: M_16†ᵐ's deficit is largest on the near-rigid trimers' rigid part, and the read-out's explicit
  induction narrows it there (1.51 → 1.25 / 1.15×). (a), a loss on the rigid-body forces only, recedes: the deficit is
  largest where the charge flux plays no part in the energies, and the loss would drop 72 % of the force signal to test
  an indirect mechanism.
- **Before a run, a diagnostic on the 32 saved models** (no training; under an hour on one GPU): (i) each model's
  per-trimer 3-body against the truth's on the rigidified trimers, by contact: its slope (the 3-body's strength) and
  the residual along TTM3-F's 3-body induction (the CCSD(T) correction runs along it, correlation 0.87 on these trimers,
  so the induction's direction is read, not the two terms apart); (ii) a linear probe of each E₀'s vector features for
  the pinned field at the atom. The field input is worth a run if M_16†ᵐ's E₀ encodes the field less than M_∞'s does and
  its missing 3-body lies along the induction; if neither, the cause is elsewhere.
