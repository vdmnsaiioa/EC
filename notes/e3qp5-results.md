# E3-QP5 — E₀'s view of the pinned field: results

*24 Oct 2026. The two jobs of 23 Oct (one H200 each, commit 0acf387 = v0.0.14 as applied), the run split by rung as the
sbatch allows: job a (sd656) M_16†ᵐ, M_16†ᵐ_f, M_∞_f, 2.5 h; job b (sd655) M_16C2†ᵐ, M_16C2†ᵐ_f, 2.8 h — each with
the short-range diagnostic and its saved frames. Read against `e3qp5-preregistration.md`. Seed means over the 4 seeds
unless said, kcal/mol; "paired" compares a field rung with its base seed by seed in the same job; M_∞ is E3-QP4 job B's.
`scratch/e3qp5_summarise_log.py` on the two job logs and job B's (the run part and the diagnostic agree to every printed
digit in all three). §3.5 is computed from the PES and the data alone (`scratch/e3qp5_reach_scan.py`, 15 s on one CPU).
The local master of this note is `notes/` in the repository.*

## 1. Verdict

- **(b) holds: the field is what the pinned rungs' E₀ lacked.** M_16†ᵐ_f's raw 3-body on the strict near-rigid trimers
  0.110 → 0.062 (paired −32 / −44 / −48 / −48 %), its rigid part 0.104 → 0.071, all raw trimers 0.117 → 0.078, below
  O–O 2.7 Å 0.195 → 0.127 — level with or below M_∞'s on every one (0.072, 0.081, 0.084, 0.126). The gap E3-QP4 left
  (1.36–1.53× M_∞) is closed and reversed. At none of the registered prices: its raw 2-body −12 %, hexamers −11 %,
  training clusters −10 %, raw 4-body 120 → 82 %.
- **The pinned rungs' piece, not a better input for any E₀.** M_∞_f's 3-body is no better than M_∞'s (strict +15 %, all
  +5 %): M_∞'s E₀ had built the field for itself, as (b) has it. The field gives M_∞ what it lacked instead, its 2-body at
  contact: raw dimers 0.113 → 0.059.
- **The C rung gains at contact and loses the hexamers.** M_16C2†ᵐ_f: raw 3-body −31 % (below 2.7 Å −35 %, on every
  seed), raw 2-body −29 %, training clusters −17 % (forces −10 to −14 %), test dimers −17 %; but hexamers 0.155 → 0.178
  (+16 %, worse on every seed), their non-additive energy 12.7 → 14.5 %, raw 4-body 57 → 81 %, the CCSD(T) trimers beyond
  9 Å 26.7 → 41.6 %, the 8 Å test probes' non-additive energy 4–6 → 21–31 %. M_16C2†ᵐ stays the best model on the
  hexamers and at long range.
- **The far field is as computed** on every seed: every field rung's at 10–15 Å its base's, M_16C2†ᵐ_f's read-out solves
  M_16C2†ᵐ's step for step.
- **The field term has two faults (§3.5).** *(i) The per-source taper leaves a molecule part charged.* A molecule that
  straddles the taper enters with a net charge — 27 % of the raw trimers' (atom, other molecule) pairs, rms 0.30 e, up
  to 0.79 e; 29–38 % of the training clusters' — a monopole's field where the truth has a dipole's. Two polarisabilities
  on the tapered cross term predict the 3-body of the extended trimers worse than zero, and M_16†ᵐ_f's 3-body there is,
  to the printed digit, what they leave: largest O–O 7–9 Å 0.0388, beyond 9 Å 0.0132 (121 % of the labels' rms, the
  registered 121.6 %). Weighting every source molecule whole removes it and keeps more of the cross term: 0.114 → 0.078
  at the same locality. *(ii) The 2-body past the contact data's edge*, in the pinned rungs only: 3.0–3.2 Å +72 / +47 %
  (M_16†ᵐ_f / M_16C2†ᵐ_f) against −42 / −51 % at 2.9–3.0 Å, the dimers off the training pairs +37 / +25 % against −33 /
  −43 % on them. The taper plays no part there (1 % of the atoms see their partner straddle it at 3.0–3.2 Å).

## 2. The predictions

| registered (§4) | E3-QP5 | |
|---|---|---|
| *Computed* | | |
| M_16C2†ᵐ_f's test probes at 10 / 12 / 15 Å: non-additive 4 / 5 / 3 %, order 1 15 / 15 / 14 %, order 2 5 / 5 / 4 % | as computed, every seed | ✓ |
| M_16†ᵐ_f: interaction 10 / 9 / 15 %, non-additive 100 % | as computed, every seed | ✓ |
| M_∞_f: 100 % | as computed, every seed | ✓ |
| M_16C2†ᵐ_f's read-out solve M_16C2†ᵐ's, step for step | 4 of 4 solves | ✓ |
| *Expected* | | |
| (b): M_16†ᵐ_f's strict-subset 3-body ≥ 15 % below M_16†ᵐ's on ≥ 3 of 4 seeds, paired | 4 of 4 (−32 to −48 %) | ✓ |
| (b): its mean ≤ 0.09; the rigid part ≤ 0.09; all raw trimers ≤ 0.10 | 0.062; 0.071; 0.078 | ✓ ✓ ✓ |
| no price: raw 2-body ≤ M_16†ᵐ's + 0.005; hexamers and training clusters at or below M_16†ᵐ's; raw 4-body not above | 0.067 / 0.076; 0.206 / 0.233; 0.078 / 0.087; 82 / 120 % | ✓ ✓ ✓ ✓ |
| the pinned rungs' piece: M_∞_f's strict-subset gain (against M_∞) less than half M_16†ᵐ_f's | +15 % (a loss) against −44 % | ✓ |
| the C rung: raw 3-body below 2.7 Å ≥ 15 % below M_16C2†ᵐ's on ≥ 3 of 4 seeds, paired | 4 of 4 (−19 to −44 %) | ✓ |
| the C rung: hexamers, and their non-additive energy, at or below M_16C2†ᵐ's | 0.178 / 0.155; 14.5 / 12.7 % | ✗ ✗ |
| *Kept* | | |
| hexamers' non-additive M_16C2†ᵐ ≤ 17 % | 12.7 % | ✓ |
| trimers beyond 9 Å: C rungs 24–28 %, the others 95–105 % | M_16C2†ᵐ 26.7, M_16†ᵐ 99.8; M_16C2†ᵐ_f 41.6, M_16†ᵐ_f 121.6, M_∞_f 105.3 % | ✓ ✓ ✗ ✗ ✗ |
| the controls: not bit-identical to job B (the GPU's arithmetic), so every seed mean within job B's range | M_16†ᵐ all inside; M_16C2†ᵐ all but its 3-body distortion part, 0.0313 against 0.0316–0.0329 | ✓ ✗ (by a hair) |

18 of 24 checks as registered. Of §5's surprises only the control's hair: no field rung's training clusters improved
without its 3-body (both did, together), no field rung's raw 2-body rose (−12 %, −29 %), and M_∞_f did not gain as
M_16†ᵐ_f did.

## 3. Reading

**3.1 The 3-body at contact.** The raw trimers (4673; the labels' rms and the PES's own error alongside):

| raw trimers | labels | M_16†ᵐ | M_16†ᵐ_f | M_16C2†ᵐ | M_16C2†ᵐ_f | M_∞ | M_∞_f | PES |
|---|---|---|---|---|---|---|---|---|
| all | 0.623 | 0.117 | 0.078 | 0.126 | 0.087 | 0.084 | 0.089 | 0.012 |
| strict near-rigid (1450) | | 0.110 | 0.062 | 0.089 | 0.065 | 0.072 | 0.083 | |
| rigid part / distortion part | | 0.104 / 0.022 | 0.071 / 0.012 | 0.104 / 0.031 | 0.075 / 0.021 | 0.081 / 0.015 | 0.085 / 0.012 | |
| smallest O–O < 2.7 Å (371) | 1.268 | 0.195 | 0.127 | 0.260 | 0.167 | 0.126 | 0.119 | 0.018 |
| 2.7–3.0 Å (3361) | 0.596 | 0.117 | 0.078 | 0.120 | 0.084 | 0.086 | 0.091 | 0.012 |
| from 3.0 Å (941) | 0.145 | 0.057 | 0.048 | 0.033 | 0.033 | 0.051 | 0.061 | 0.009 |
| largest O–O < 5 Å (2739) | 0.779 | 0.131 | 0.088 | 0.156 | 0.106 | 0.098 | 0.090 | 0.014 |
| 5–7 Å (1515) | 0.312 | 0.103 | 0.067 | 0.072 | 0.051 | 0.068 | 0.095 | 0.008 |
| 7–9 Å (311) | 0.027 | 0.026 | **0.039** | 0.0083 | **0.0136** | 0.023 | **0.049** | 0.006 |
| from 9 Å (108) | 0.011 | 0.0108 | **0.0132** | 0.0029 | **0.0045** | 0.0109 | 0.0114 | 0.002 |

The field's gain is in the rigid part and the distortion part alike (M_16†ᵐ_f −31 and −46 %), at every contact, and
largest on the strict near-rigid subset, where E3-QP4 located the deficit (−43 %). M_16C2†ᵐ_f gains as much (−31 %, below
2.7 Å −35 %) from a higher start, so below 2.7 Å it stays above M_∞ (0.167 against 0.126): the read-out's near field
still costs at the closest contacts. From O–O 3 Å the C rung is unchanged (0.033) and the best. Where the farthest
molecule is 7 Å or more from another, every field rung is worse than its base, and M_16†ᵐ_f worse than predicting no
3-body at all (the labels' rms): §3.5.

**3.2 The 2-body.** The raw dimers (4937), by O–O and by coverage (whether the dimer's nearest training pair lies within
the test dimers' 95 % nearest-training-pair distance, 0.116 Å; 24 % do not):

| raw dimers | M_16†ᵐ | M_16†ᵐ_f | M_16C2†ᵐ | M_16C2†ᵐ_f | M_∞ | M_∞_f |
|---|---|---|---|---|---|---|
| all (PES 0.011) | 0.076 | 0.067 | 0.085 | 0.061 | 0.113 | 0.059 |
| O–O 2.3–2.7 Å (278) | 0.100 | 0.076 | 0.109 | 0.082 | 0.114 | 0.081 |
| 2.7–2.9 Å (930) | 0.071 | 0.058 | 0.070 | 0.052 | 0.087 | 0.052 |
| 2.9–3.0 Å (2241) | 0.076 | 0.044 | 0.098 | 0.048 | 0.117 | 0.042 |
| 3.0–3.2 Å (761) | 0.067 | **0.113** | 0.067 | **0.093** | 0.117 | 0.048 |
| 3.2–5 Å (638) | 0.080 | 0.067 | 0.066 | 0.056 | 0.118 | 0.082 |
| 5–7 Å (52) | 0.021 | **0.046** | 0.025 | **0.039** | 0.179 | 0.204 |
| on the training pairs | 0.075 | 0.051 | 0.090 | 0.051 | 0.108 | 0.051 |
| off them | 0.076 | **0.103** | 0.069 | **0.085** | 0.128 | 0.078 |

On the data the field term's 2-body gains everywhere: at the contacts' O–O (2.3–3.0 Å) −18 to −51 %, on the training
pairs −33 / −43 %. Past the contacts' edge at 3.0 Å the pinned field rungs lose (+72 / +47 %, on every seed), and off the
training pairs (+37 / +25 %); M_∞_f gains there too (3.0–3.2 Å −59 %). One reading: in the pinned rungs the term fits a
correction to the bands at contact (the point charges' penetration error), and nothing past 3.0 Å tells it to vanish; in
M_∞_f it carries the electrostatics itself, whose shape continues. At 5–7 Å (52 dimers) the pinned field rungs double
their small error (0.021 → 0.046, 0.025 → 0.039): there 71 % of the atoms see their partner straddle the taper (§3.5).

**3.3 Many-body beyond the contact.** *The 4-body*: M_16†ᵐ_f 120 → 82 %, M_16C2†ᵐ_f 57 → 81 %, M_∞_f 51 → 101 % of
the labels' rms — every field rung ends at 80–100 % whichever its base, better for the one rung whose E₀ was worse,
worse for the two that were better. *The hexamers*: M_16†ᵐ_f 0.233 → 0.206 (their 2-body part −22 %, their non-additive
part −26 %); M_16C2†ᵐ_f 0.155 → 0.178 (2-body part −7 %, non-additive part +16 %: −11 / +32 / +33 / +11 % by seed),
M_∞_f 0.435 → 0.449. One reading: the hexamers' non-additive energy sums the 3-body at contact, which improved, with the
4-body and the part-charged second shell's share (§3.5: 38 % of the hexamers' atom–molecule pairs), which got worse; for
the C rung the second wins. *The 8 Å probes* (non-additive energy, by seed): test M_16†ᵐ
99 / 98 / 103 / 100 → M_16†ᵐ_f 104 / 110 / 104 / 100 %, M_16C2†ᵐ 5 / 6 / 4 / 4 → M_16C2†ᵐ_f 21 / 22 / 31 / 27 %;
training 95–99 → 138–159 % and 8–11 → 41–50 %; M_∞_f's training probes 165–211 %. Two of the 20 test and 34 of the 96
training probes at 8 Å lie within the field term's reach, 7 and 59 within r_c, where E₀ — retrained with the term —
changed too: the 8 Å readings do not separate the two.

**3.4 M_∞_f.** Against M_∞: the 3-body +5 % (strict +15 %, below 2.7 Å −5 %), the raw 2-body −48 % (2.9–3.2 Å −63 %),
the contacts' energies −18 %, the training clusters −1 %, forces ±2 %, test dimers −4 %, the hexamers +3 % (non-additive
22.8 → 24.8 %), the raw 4-body 51 → 101 %, the trimers with the farthest molecule at 7–9 Å +111 %. M_∞'s E₀ supplies its
3-body without the field; with it, its 2-body at contact is no longer learned from the contacts' energies alone.

**3.5 The taper's fault** (computed). E3-QP5's pre-registration fitted one polarisability per species on the field input's
first-order cross term, −Σ_i a_Z(i) E_i(B)·E_i(C), every source's share of E_i tapered by its own distance to the atom
(4.5–5.5 Å): 0.114 of the raw trimers' 0.623 kcal/mol rms left. Refitted for other weightings:

| two numbers on the cross term | all | S < 2.7 | S 2.7–3 | S ≥ 3 | L < 5 | L 5–7 | L 7–9 | L ≥ 9 |
|---|---|---|---|---|---|---|---|---|
| the labels | 0.623 | 1.268 | 0.596 | 0.145 | 0.779 | 0.312 | 0.0272 | 0.0109 |
| no taper | 0.066 | 0.096 | 0.068 | 0.034 | 0.082 | 0.036 | 0.0051 | 0.0025 |
| per source 4.5–5.5 Å (E3-QP5) | 0.114 | 0.143 | 0.119 | 0.074 | 0.108 | 0.137 | **0.0388** | **0.0132** |
| per source 4–5 / 3.5–4.5 / 3–4 Å | 0.152 / 0.201 / 0.330 | | | | | | | |
| whole: product of its sources' tapers, 4.5–5.5 Å | 0.098 | 0.135 | 0.102 | 0.052 | 0.110 | 0.086 | 0.0252 | 0.0109 |
| whole: product, 5–6 Å | 0.078 | 0.112 | 0.081 | 0.040 | 0.089 | 0.065 | 0.0230 | 0.0109 |
| whole: atom to the molecule's O, 4.5–5.5 Å | 0.074 | 0.108 | 0.077 | 0.038 | 0.085 | 0.061 | 0.0227 | 0.0109 |
| whole: the two molecules' O–O, 5–6 Å | 0.070 | 0.100 | 0.073 | 0.036 | 0.082 | 0.054 | 0.0197 | 0.0108 |
| the triple's three O–O, 5.5–6.5 Å (on the cross term) | 0.072 | 0.097 | 0.076 | 0.042 | 0.082 | 0.063 | 0.0272 | 0.0109 |

(S, L: the smallest and the largest O–O, Å.) What the per-source taper does: a molecule straddling it — some of its
sources (two H atoms at +0.613 e, an M-site at −1.225 e) inside, some in the taper or beyond — enters with a net charge:
in the raw trimers 27 % of the (atom, other molecule) pairs, rms 0.30 e, up to 0.79 e; in the training clusters 29 / 35 /
38 % (trimers / tetramers / pentamers), in the test hexamers 38 %, rms 0.28–0.30 e. Its field falls as a monopole's
where the truth's falls as a dipole's, and so the fitted cross term, which tracks the 3-body to correlation +0.99 with no
taper wherever the farthest molecule is, tracks it to +0.94 at 5–7 Å, +0.54 at 7–9 Å (its rms 0.045 against the
labels' 0.027) and +0.10 beyond 9 Å. M_16†ᵐ_f's 3-body in the last two bins equals the fit's residual to the printed digit
(0.0388, 0.0132); M_16C2†ᵐ_f's grows by less (0.0083 → 0.0136, 0.0029 → 0.0045). Shortening the
reach makes it worse (the third molecule's field from 4–6 Å carries much of the 3-body at contact: 3–4 Å leaves 0.330);
weighting every source molecule whole mends it at the same reach and recovers most of the untapered cross term. The
product of its sources' tapers at 5–6 Å keeps E3-QP5's locality as it was — no source beyond r_c in reach, so a
molecule enters only when E₀'s graph holds all its charged sites — and leaves 0.078, nothing worse than zero beyond
7 Å. In the raw dimers the partner straddles the taper for 0.2–6 % of the atoms below O–O 3.2 Å (net charge rms ≤ 0.01
e), 24 % at 3.2–5 Å and 71 % at 5–7 Å: fault (ii) at 3.0–3.2 Å is not the taper's.

**3.6 The controls.** M_16†ᵐ and M_16C2†ᵐ run again are not job B's bit for bit (1 of 34 readings identical on every
seed — the GPU's arithmetic: autotuning picks among kernels), and every seed mean lies within job B's range over its
seeds but M_16C2†ᵐ's 3-body distortion part, 0.0313 against 0.0316–0.0329. The paired comparison is unaffected.

## 4. Next

- **E3-QP6 — every source molecule whole, and the term's degree in the field**: the field term with one weight per
  (atom, source molecule), the product of its sources' tapers at r_c − 1 to r_c (5–6 Å), in two forms:
  `_w` (E3-QP5's MLP otherwise as it was) on M_16†ᵐ and M_16C2†ᵐ; `_w2` on M_16C2†ᵐ, the term of degree 2 in the field —
  −½ E_i·A_i·E_i − ½ G_i:B_i:G_i, A and B set by E₀'s features (the read-out's form, local): no term linear in φ, E or G,
  no MLP over the field. The bases run again as the controls; five rungs in two jobs as E3-QP5's (about 2.5–3 h each on
  an H200). It reads (i) whether the extended trimers, the 2-body at 5–7 Å, the 8 Å probes and the C rung's hexamers
  return with whole molecules, the contact gain kept or grown (computed: the cross term 0.114 → 0.078); (ii) whether the
  2-body past 3.0 Å and the 4-body return only at degree 2. The numbers before training, the far field's reach among
  them (no probe source within 6 Å of a core atom at 10–15 Å), come with its pre-registration.
- **Without the GPU**: `results/e3qp5a/` and `results/e3qp5b/shortrange_diagnostic_frames.npz` hold every frame's n-body
  energies for every model; with them the 3-body's strength and its direction against TTM3-F's induction, and which
  dimers at 3.0–3.2 Å the pinned field rungs miss (by orientation), can be read here.
- **Carried forward**: M_16C2†ᵐ (job B's settings) remains the best model; M_16†ᵐ_f is the evidence for (b), not a
  candidate (no read-out, so no far field).
