# E3-QP7 — the field term without its 2-body: pre-registration

*25 Oct 2026, written before any training. The follow-up of E3-QP6 (`e3qp6-results.md` §4), set up after Vadic's "set it
up". Harness v0.0.16; `configs/water_qaqua_pol7.yaml` holds every value. The numbers of §3 are computed before training
from the PES and the data alone, with the harness's own per-molecule field tensors (`scratch/e3qp7_preregistration.py`),
and from a timing of the training step (`scratch/e3qp7_field_timing.py`). The local master of this note is `notes/` in
the repository.*

## 1. The question, and the change

E3-QP6 made M_16C2†ᵐ_w the best model so far — the field term within 6 Å, every molecule whole: raw 3-body −38 %, raw
2-body −39 %, training clusters −26 % against M_16C2†ᵐ, hexamers level — at two costs: the raw 4-body 58 → 75 %, the
2-body at O–O 5–7 Å 0.025 → 0.047. Its degree-2 form kept both (55 %, +1 %) and lost the 2-body past the contact data
(O–O 3.0–3.2 Å +87 %, off the training pairs +56 %) and the hexamers (+8 %). The reading (`e3qp6-results.md` §3.4): a
field term that is a function of the field summed over the other molecules gives a dimer its *self terms* and a trimer its
*cross terms*, and at degree 2 one coefficient sets both — the self term −½ E(B)·A·E(B), a 2-body, comes with the cross
term −E(B)·A·E(C) that carries the 3-body. With the two numbers that fit the trimers, the self term's spread on the raw
dimers is 3.6–7.9 times the base's 2-body error at every contact bin; E₀ must cancel it, and does only where it has data.

**The change** (`Rung.e0_field_cross`, with every molecule whole): the term less its value on each source molecule alone,

    eps_i = h(s_i; x_i) - sum_B h(s_i; x_i(B)),      x_i = sum_B x_i(B)      (x: the potential, field and field gradient),

every molecule's share apart from `bandfields.local_field_tensors(..., per_frag=True)` (the totals then the sums of the
parts). By construction, and tested (`tests/test_e0_field_cross.py`):

- *nothing to any dimer*: where one molecule is in reach the two terms cancel to the last digit — a cross rung's dimer
  (at contact, straddling the taper, beyond it) is M_16C2†ᵐ's, energy and every force, with every parameter switched on;
  so E₀, the bands and the read-out carry the whole 2-body, as before E3-QP5;
- *at degree 2 the cross terms only*: −½ e_scale Σ_{B≠C} [E_i(B)·A_i·E_i(C) + G_i(B):B_i:G_i(C)] (checked against the
  bilinear forms written out) — the leading non-additive induction, A and B still E₀'s; no products of three fields;
- *the MLP form less its single-molecule values*: h(total) − Σ_B h(part), checked against the head without parts;
- the base rung at the start, digit for digit; E3-QP6's rungs (and those before) unchanged to the last digit (golden
  values from v0.0.15); local (a probe 7.3 Å from its core interacts as with the base rung); C² in the positions, the
  forces the energy's gradient (central differences); invariant under rotation, translation and reordering.

## 2. Design

| rung | what it is | what it reads |
|---|---|---|
| M_16C2†ᵐ | the base, run again | the paired comparison; whether it reproduces |
| M_16C2†ᵐ_w | E3-QP6's best model, run again | the reference the cross rungs must match or beat |
| M_16C2†ᵐ_w2x | M_16C2†ᵐ_w2 with the cross terms only | the self-term reading: the edge and the hexamers back, the 3-body and the 4-body kept |
| M_16C2†ᵐ_wx | M_16C2†ᵐ_w less each molecule's own value | what the MLP form's 2-body gain, its 5–7 Å loss and its 4-body loss were |

Everything else is E3-QP6's bit for bit (`configs/water_qaqua_pol7.yaml` differs from `configs/water_qaqua_pol6.yaml` in
its rungs alone, tested): E3-QP4 job B's data, training, matching and readings, 4 seeds, the short-range diagnostic with
every frame's n-body energies. The four rungs share their seeds and start from the same parameters: the comparisons are
paired, seed by seed, in one job. E3-QP6's numbers (A100) are the reference for the changes (not paired).

## 3. Numbers before training

**(a) The reach** is E3-QP6's (`e3qp6-preregistration.md` §3a): every probe at 10–15 Å lies beyond the whole-molecule
term's reach, so the far field there is the base's; at 8 Å 73 of the 96 training and 18 of the 20 test probes lie beyond
it. Where a probe is in reach the cross term adds only products of its field with a core molecule's — no probe–atom
2-body.

**(b) The cross term** (`scratch/e3qp7_preregistration.py`, the per-molecule tensors). On E3-QP4's 4673 raw trimers the
two-number fit −Σ_i a_Z(i) E_i(B)·E_i(C) leaves 0.0775 of the 0.6225 kcal/mol rms with a_O 4.66, a_H 4.18 bohr³ —
E3-QP6's numbers to the printed digit through the new tensors. With those two numbers fixed, on what the cross rungs are
read and trained on (rebuilt bit for bit):

| | non-additive energy rms | the cross term rms | correlation | left |
|---|---|---|---|---|
| the 24 test hexamers | 0.997 | 0.961 | +0.974 | 0.166 |
| the training clusters of 3 / 4 / 5 (100 each) | 0.430 / 0.741 / 0.888 | 0.457 / 0.764 / 0.924 | +0.980 / +0.979 / +0.976 | 0.082 / 0.129 / 0.157 |

Refitted on the 300 clusters' non-additive energy: a_O 5.72, a_H 3.45, leaving 0.116 of 0.712 (the hexamers then 0.172,
the raw trimers 0.0879). Two numbers on the cross term carry about five sixths of the non-additive energy's rms in the
hexamers and the clusters (81–84 %; the raw trimers 88 %), and transfer from the trimers to the hexamers; the trained
models do better on the hexamers (non-additive part M_16C2†ᵐ 0.121, M_16C2†ᵐ_w 0.115, M_16C2†ᵐ_w2 0.153) — E₀ and the
read-out carry the rest.

**(c) The self terms** the same two numbers bring, −½ Σ_i a_Z(i) Σ_B |E_i(B)|² — what M_16C2†ᵐ_w2's E₀ had to cancel and
the cross rungs do not carry. Their spread (rms about the mean) is larger than the non-additive energy the cross term is
to carry, in every set: the raw dimers 0.51 kcal/mol (mean −0.73; by O–O in `e3qp6-results.md` §3.4), the training
clusters 0.72 / 0.95 / 1.11 (against their non-additive 0.43 / 0.74 / 0.89), the test hexamers 1.31 (against 0.997). On
the 4937 raw dimers the cross term itself is zero to the last digit.

**(d) The cost.** One training step at width 32 (a batch of 32 clusters, CPU): M_16C2†ᵐ_wx 1.10×, M_16C2†ᵐ_w2x 1.08× the
base (M_16C2†ᵐ_w 1.07×, M_16C2†ᵐ_w2 0.97× in the same timing). On E3-QP6's A100 the field rungs cost about 0.06 more than
on the CPU (1.10 against 1.04, 1.08 against 1.00): about 1.15×, 1650 s a seed. The job: M_16C2†ᵐ 1440 s, M_16C2†ᵐ_w
1579 s and the two cross rungs a seed, 4 seeds, about 7.0 h, and the diagnostic about 1.3 h — about 8.3 h on an A100
(E3-QP6: 9.4 h for five rungs), about 5 h on an H200.

**(e) The baselines**, E3-QP6 (seed means, A100; M_16C2†ᵐ_w2 the reference for M_16C2†ᵐ_w2x):

| | M_16C2†ᵐ | M_16C2†ᵐ_w | M_16C2†ᵐ_w2 |
|---|---|---|---|
| raw 3-body: all; strict near-rigid; below O–O 2.7 Å | 0.127; 0.091; 0.259 | 0.078; 0.059; 0.151 | 0.083; 0.063; 0.160 |
| raw 2-body: all; O–O 3.0–3.2 Å; 5–7 Å; off the training pairs | 0.082; 0.054; 0.025; 0.061 | 0.050; 0.056; 0.047; 0.054 | 0.085; 0.092; 0.026; 0.091 |
| hexamers: energy; non-additive | 0.1530; 12.1 % | 0.1525; 11.6 % | 0.1625; 15.3 % |
| raw 4-body; the 8 Å test probes' non-additive | 58 %; 4–6 % | 75 %; 4–8 % | 55 %; 4–6 % |
| training clusters' energy; the trimers beyond 9 Å | 0.078; 26.7 % | 0.058; 26.7 % | 0.075; 26.7 % |

## 4. Predictions

**Computed** (to the printed digit, on every seed): the far field at 10 / 12 / 15 Å of every field rung is the base's —
M_16C2†ᵐ_w's, M_16C2†ᵐ_wx's and M_16C2†ᵐ_w2x's test probes' non-additive 4 / 5 / 3 %, through order 1 15 / 15 / 14 %,
order 2 5 / 5 / 4 % — and their read-out solves M_16C2†ᵐ's, step for step.

**Expected** (seed means unless said; kcal/mol; "paired" compares a rung with M_16C2†ᵐ seed by seed, in this run):

- *(i) No self terms, no edge.* M_16C2†ᵐ_w2x's raw 2-body at O–O 3.0–3.2 Å, and off the training pairs, at most 10 %
  above M_16C2†ᵐ's (M_16C2†ᵐ_w2 +70 %, +50 %); the same for M_16C2†ᵐ_wx (M_16C2†ᵐ_w +3 %, −10 %). Both cross rungs' raw
  2-body at O–O 5–7 Å within 25 % of M_16C2†ᵐ's (M_16C2†ᵐ_w +88 %, M_16C2†ᵐ_w2 +1 %) — nothing of the term reaches a
  dimer.
- *(ii) The 3-body kept.* Both cross rungs' raw 3-body below O–O 2.7 Å ≥ 15 % below M_16C2†ᵐ's on ≥ 3 of 4 seeds, paired
  (M_16C2†ᵐ_w2 −38 %, M_16C2†ᵐ_w −42 %), and their raw 3-body ≤ 0.090 (0.083, 0.078; M_16C2†ᵐ 0.127).
- *(iii) The hexamers.* M_16C2†ᵐ_w2x's hexamers, and their non-additive energy, at or below M_16C2†ᵐ's (M_16C2†ᵐ_w2 0.163,
  15.3 % against 0.153, 12.1 %); the same for M_16C2†ᵐ_wx.
- *(iv) The training clusters.* Both cross rungs' training clusters' energy ≥ 10 % below M_16C2†ᵐ's on ≥ 3 of 4 seeds,
  paired (M_16C2†ᵐ_w −24 to −28 %, every seed; M_16C2†ᵐ_w2 on 1 of 4): with no self terms to cancel, two numbers on the
  cross term carry about five sixths of the clusters' non-additive energy (§3b).
- *(v) The 4-body.* M_16C2†ᵐ_w2x's raw 4-body ≤ 65 % (M_16C2†ᵐ_w2 55 %), M_16C2†ᵐ_wx's above 65 % (M_16C2†ᵐ_w 75 %): the
  MLP's products of three and more fields stay when its self terms go. Read together:

  | M_16C2†ᵐ_w2x | M_16C2†ᵐ_wx | the C rung's 4-body loss with the MLP form is |
  |---|---|---|
  | ≤ 65 % | > 65 % | its terms beyond degree 2 in the field (E3-QP6's reading) |
  | ≤ 65 % | ≤ 65 % | its self terms (through E₀'s features or its linear channels) |
  | > 65 % | either | not the form alone: the cross terms' coefficients, set by E₀'s features, carry a 4-body too |

- *(vi) What the MLP form's 2-body gain was* (read, not a check): M_16C2†ᵐ_wx's raw 2-body at most 0.060 — E₀ freed from
  the non-additive energy (M_16C2†ᵐ_w 0.050); at least 0.075 — the term's own 2-body, its self terms, carried the gain
  (M_16C2†ᵐ 0.082); between, both. M_16C2†ᵐ_w2x's alongside.
- *Kept*: M_16C2†ᵐ's seed means within the range of E3-QP4 job B's, E3-QP5's and E3-QP6's twelve seeds; M_16C2†ᵐ_w as in
  E3-QP6 — hexamers ≤ 0.17, raw 3-body ≤ 0.09, raw 2-body ≤ 0.06; the hexamers' non-additive energy of M_16C2†ᵐ ≤ 17 %;
  the trimers beyond 9 Å 24–28 % for all four rungs; the field rungs' raw 3-body at largest O–O 7–9 Å ≤ 0.011; their 8 Å
  test probes' non-additive ≤ 12 % on every seed.

E3-QP6's reading is refuted if M_16C2†ᵐ_w2x's raw 2-body at O–O 3.0–3.2 Å is ≥ 30 % above M_16C2†ᵐ's on ≥ 3 of 4 seeds,
paired: the degree-2 rung's loss past the contact data was then not its self terms.

## 5. Surprises, and what is not tested

A surprise would be: a far-field reading at 10–15 Å off §4; a control's seed mean outside the combined range;
M_16C2†ᵐ_w2x's contact gain (raw 3-body below 2.7 Å, against M_16C2†ᵐ, paired) less than half M_16C2†ᵐ_w2's (38 %) —
the cross terms carry the 3-body, the self terms only through E₀'s features; a cross rung's 2-body at O–O 5–7 Å outside
25 % of M_16C2†ᵐ's — nothing of the term reaches a dimer, only E₀ retrained could move it.

Not tested: the cross term on the pinned rung without the read-out (M_16†ᵐ_w carried the best contact 3-body); the
read-out's near field at the closest contacts (the C rungs' 3-body below 2.7 Å 0.151–0.160 against M_16†ᵐ_w's 0.109); a
coarse-grained density beside the pinned charges (after this); contact data past O–O 3.0 Å; the field of the induced
dipoles; E₀'s backbone. Expected cost: about 8.3 h as one job on an A100, about 5 h on an H200.
