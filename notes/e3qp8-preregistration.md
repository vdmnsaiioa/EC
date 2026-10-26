# E3-QP8 — the field term's two channels apart: pre-registration

*26 Oct 2026, written before any training. The follow-up of the E3-QP7 decomposition (`e3qp7-decomposition.md` §6),
set up after Vadic's "yes". Harness v0.0.17; `configs/water_qaqua_pol8.yaml` holds every value. The numbers of §3 are
computed before training from the PES and the data alone, with the harness's own per-molecule field tensors
(`scratch/e3qp8_preregistration.py`), from a timing of the training step (`scratch/e3qp8_field_timing.py`), and — in
the job, before its training — from E3-QP7's saved M_16C2†ᵐ_w (§3c). The local master of this note is `notes/` in the
repository.*

## 1. The question, and the change

M_16C2†ᵐ_w is the best model so far (E3-QP6, E3-QP7: raw 2-body 0.051, raw 3-body 0.079, hexamers' non-additive part
11.3 %), at two costs: the raw 4-body (74 % of the labels' rms against M_16C2†ᵐ's 58 %) and the 2-body at O–O 5–7 Å
(0.048 against 0.025). The decomposition of every saved model (`e3qp7-decomposition.md`) put each cost on one channel of
the field term:

- *the 5–7 Å loss is the self terms' tail.* The self terms h(s_i; x_i(B)) are a 2-body of 0.27 kcal/mol rms on the raw
  dimers that E₀ cancels to −0.98; beyond 5 Å they are 0.073 rms, all at fixed coefficients — the MLP's response to the
  potential there, still 12 % of its contact value at 5–6 Å (the squared field 0.4 %) — and E₀ follows only 0.045;
- *the 4-body loss is the cross terms' own*: E₀'s change alone leaves every field rung's raw 4-body below the base's;
  at degree 2 it is all the coefficients' (a fourth molecule moving the E₀ features that set them; none at fixed
  coefficients), and without the self terms to tie them to the 2-body they moved more (M_16C2†ᵐ_w2 → _w2x: 0.053 →
  0.074 rms).

**The changes** (v0.0.17, with every molecule whole and the MLP form):

    M_16C2†ᵐ_ws:  eps_i = h(s_i; x_i) - sum_B (reach_iB - w_iB) h(s_i; x_i(B))
    M_16C2†ᵐ_us:  eps_i = sum_B w_iB h(s_i; x_i(B)) - 1/2 sum_{B != C} [ a_Z(i) E_i(B).E_i(C) + c_Z(i) G_i(B):G_i(C) ]

with w_iB molecule B's own whole-molecule weight at atom i over 4–5 Å (the product of its sites' tapers;
`Rung.e0_field_self_on / _off`, the last 2–1 Å inside r_c: `E0_FIELD_SELF_TAPER`) and the term's reach over 5–6 Å as
before. M_16C2†ᵐ_ws is M_16C2†ᵐ_w with its self terms cut beyond 4–5 Å and its cross terms h(x_i) − Σ_B h(x_i(B)) as
they were (`heads.field_energy_split`); M_16C2†ᵐ_us keeps those self terms and replaces the MLP's cross terms by
degree-2 cross terms with coefficients by species alone (`Rung.e0_field_cross_species`, `heads.field_cross_species`:
four numbers starting at zero, no feature of E₀ in them). By construction, and tested (`tests/test_e0_field_split.py`):

- *the limits*: with w = reach the split head is `field_energy`'s term (M_16C2†ᵐ_w), with w = 0 its cross terms alone
  (M_16C2†ᵐ_wx), and in between the cross terms plus Σ_B w_iB h(x_i(B)), each h from the head on that molecule alone; on
  a dimer M_16C2†ᵐ_ws's term is M_16C2†ᵐ_w's atom by atom times the weight, and a dimer beyond the self reach but in the
  term's (one H atom in reach, every weight zero) is the base's, energy and every force;
- *the species cross terms strictly 3-body*: nothing on any dimer and no 4-body on a tetramer's sub-clusters (to the
  last digits; the self terms beside them carry one through E₀'s features), checked against the cross sums written out;
  E₀'s gradient the same with and without them — no gradient into E₀ through them;
- the base rung at the start, digit for digit, and M_16C2†ᵐ_ws's parameters M_16C2†ᵐ_w's; E3-QP7's cross rungs (and
  those before) unchanged to the last digit (golden values from v0.0.16); local (a probe 7.3 Å from its core interacts
  as with the base rung); the forces the energy's gradient (central differences, a molecule straddling the taper);
  invariant under rotation, translation and reordering.

## 2. Design

| rung | what it is | what it reads |
|---|---|---|
| M_16C2†ᵐ | the base, run again | the paired comparison; whether it reproduces |
| M_16C2†ᵐ_w | the best model, run again | the reference the split rungs must keep |
| M_16C2†ᵐ_ws | M_16C2†ᵐ_w, its self terms within 4–5 Å | whether the 5–7 Å loss is the self terms' tail, at no other cost |
| M_16C2†ᵐ_us | those self terms beside species cross terms | whether strictly 3-body cross terms keep the 3-body without the 4-body loss, and what the hexamers do |

Everything else is E3-QP7's bit for bit (`configs/water_qaqua_pol8.yaml` differs from `configs/water_qaqua_pol7.yaml` in
its rungs alone, tested): E3-QP4 job B's data, training, matching and readings, 4 seeds, the short-range diagnostic with
every frame's n-body energies, and now the decomposition of every saved model after it. The four rungs share their seeds
and start from the same parameters: the comparisons are paired, seed by seed, in one job; E3-QP7's numbers are the
reference for the changes (not paired).

## 3. Numbers before training

**(a) Where the self reach cuts** (`scratch/e3qp8_preregistration.py`; every atom–molecule pair in the term's reach, the
rigid monomer's TTM3-F charges; the share of the potential, and of the squared field, the self terms read that the cut
takes, Σ (reach − w)|x| / Σ reach |x|):

| | pairs in reach | w = 1 | mean w | potential cut | squared field cut |
|---|---|---|---|---|---|
| raw dimers O–O 2.3–2.7 Å (278) | 1668 | 89 % | 0.989 | 0.3 % | 0.09 % |
| 2.7–2.9 Å (930) | 5580 | 73 % | 0.908 | 1.6 % | 0.93 % |
| 2.9–3.0 Å (2241) | 13446 | 86 % | 0.993 | 0.3 % | 0.10 % |
| 3.0–3.2 Å (761) | 4566 | 92 % | 0.987 | 1.2 % | 0.37 % |
| 3.2–4 Å (552) | 3312 | 57 % | 0.822 | 12.4 % | 5.5 % |
| 4–5 Å (86) | 493 | 6 % | 0.226 | 63.9 % | 46.5 % |
| 5–6 Å (46) | 150 | 0 | 0.007 | **97.6 %** | 97.5 % |
| 6–7 Å (6); beyond 7 Å (36) | 2; none | 0 | 0 | 100 % | 100 % |
| the raw trimers (4673) | 72889 | 57 % | 0.721 | 5.7 % | 0.81 % |
| the 24 test hexamers | 1534 | 37 % | 0.549 | 17.3 % | 3.5 % |
| the training clusters of 3 / 4 / 5 (100 each) | 1693 / 3149 / 4735 | 52 / 46 / 40 % | 0.71 / 0.66 / 0.58 | 10.5 / 12.7 / 15.8 % | 1.9 / 2.3 / 2.8 % |
| the contact dimers / trimers (600 / 200) | 3600 / 3459 | 87 / 67 % | 0.973 / 0.779 | 1.3 / 7.2 % | 0.27 / 0.90 % |

The cut takes the tail — 98 % of the potential the self terms read at 5–6 Å — and below O–O 3.2 Å at most 1.6 %: the
pairs it weights down there are the far atoms', whose potential is small. Between, it takes 12 % at 3.2–4 Å and 64 % at
4–5 Å, and in the training clusters 10–16 % and the hexamers 17 % (their second neighbours). A 4.5–5.5 Å reach would
keep 3.2–4 Å (2 %) and the clusters (4–7 %) but leave a third of the tail (67 % cut at 5–6 Å): the test of the tail
wants it gone.

**(b) The cross terms by species** (the same tensors). On the 4673 raw trimers' 3-body labels (0.6225 kcal/mol rms): two
numbers, a_O 4.66 and a_H 4.18 bohr³ on −Σ_i a_Z Σ_{B<C} E_i(B)·E_i(C), leave 0.0775 (E3-QP6's and E3-QP7's to the
printed digit); four, with c_Z on the field gradients' products (a_O 5.20, a_H 3.82 bohr³, c_O 0.9, c_H −0.1 bohr⁵),
leave 0.0767 — the gradients add little; by closest O–O, below 2.7 Å 0.112 (of 1.268), 2.7–3.0 Å 0.080 (0.596), from
3.0 Å 0.039 (0.145). With the four numbers fixed, the test hexamers' non-additive energy (0.997 rms) is left at 0.160,
the training clusters' of 3 / 4 / 5 at 0.079 / 0.124 / 0.151 (of 0.430 / 0.741 / 0.888); refitted on the 300 clusters
(a_O 6.23, a_H 2.96, c_O 1.9, c_H 0.0) 0.113 of 0.712, the hexamers then 0.171, the raw trimers 0.0842. Four numbers by
species carry the trimers' 3-body as well as M_16C2†ᵐ_w2x's learned coefficients did (0.078), and five sixths of the
clusters' and the hexamers' non-additive energy.

**(c) The cut alone, not retrained** — in the job, before its training (`--e3qp7 results/e3qp7`): M_16C2†ᵐ_w's saved
models read with their self terms cut to 4–5 Å (the same parameters read as M_16C2†ᵐ_ws), the raw dimers' 2-body by O–O
against M_16C2†ᵐ_w as trained and M_16C2†ᵐ. A reading, not a check: what the cut does before E₀ adapts.

**(d) The cost** (`scratch/e3qp8_field_timing.py`, one training step at width 32 on a batch of 32 clusters, CPU):
M_16C2†ᵐ_w 1.17×, M_16C2†ᵐ_wx 1.26×, M_16C2†ᵐ_ws 1.38×, M_16C2†ᵐ_us 1.18× the base. E3-QP7 on its A100: M_16C2†ᵐ 1449 s
a seed, M_16C2†ᵐ_w 1594 s (1.10×), M_16C2†ᵐ_wx 1858 s (1.28×). Scaled: M_16C2†ᵐ_ws about 1950 s, M_16C2†ᵐ_us about 1700
s, 4 seeds about 7.4 h; the diagnostic 1.4 h, the decomposition 5 min — about 9 h on an A100, about 5.5 h on an H200.

**(e) The baselines**, E3-QP7 (seed means, A100):

| | M_16C2†ᵐ | M_16C2†ᵐ_w | M_16C2†ᵐ_wx | M_16C2†ᵐ_w2x |
|---|---|---|---|---|
| raw 2-body: all; O–O 2.3–2.7; 2.7–2.9; 2.9–3.0; 3.0–3.2 Å | 0.084; 0.111; 0.072; 0.095; 0.063 | 0.051; 0.076; 0.042; 0.051; 0.054 | 0.084; 0.114; 0.076; 0.094; 0.053 | 0.081; 0.119; 0.077; 0.080; 0.070 |
| raw 2-body: 3.2–5 Å; 5–7 Å (52); off the training pairs | 0.066; 0.025; 0.067 | 0.041; 0.048; 0.053 | 0.078; 0.026; 0.067 | 0.083; 0.024; 0.080 |
| raw 3-body: all; below O–O 2.7 Å; strict near-rigid | 0.128; 0.264; 0.092 | 0.079; 0.151; 0.061 | 0.091; 0.176; 0.070 | 0.078; 0.142; 0.061 |
| raw 4-body (% of 0.130); closest O–O 2.7–3.0 Å (242) | 58 %; 0.044 | 74 %; 0.055 | 68 %; 0.063 | 76 %; 0.067 |
| hexamers: energy; 2-body part; non-additive | 0.158; 0.130; 13.9 % | 0.153; 0.125; 11.3 % | 0.141; 0.138; 15.7 % | 0.150; 0.144; 15.1 % |
| training clusters' energy; test dimers; the trimers beyond 9 Å | 0.081; 0.046; 26.7 % | 0.059; 0.035; 26.7 % | 0.067; 0.047; 26.7 % | 0.067; 0.049; 26.7 % |

## 4. Predictions

**Computed** (to the printed digit, on every seed): the far field at 10 / 12 / 15 Å of both split rungs is M_16C2†ᵐ_w's
— their test probes' non-additive 4 / 5 / 3 %, through order 1 15 / 15 / 14 %, order 2 5 / 5 / 4 % (every probe there
lies beyond the term's reach) — and their read-out solves M_16C2†ᵐ's, step for step.

**Expected** (seed means unless said; kcal/mol; "paired" compares a rung with M_16C2†ᵐ, or with M_16C2†ᵐ_w where named,
seed by seed, in this run):

- *(i) The tail gone.* M_16C2†ᵐ_ws's raw 2-body at O–O 5–7 Å within 20 % of M_16C2†ᵐ's (M_16C2†ᵐ_w +90 % paired, every
  seed +75 to +117 %).
- *(ii) The contact kept.* M_16C2†ᵐ_ws's raw 2-body in each bin below O–O 3.2 Å within 10 % of M_16C2†ᵐ_w's (paired
  against it), and overall at most 0.058 (M_16C2†ᵐ_w 0.051, M_16C2†ᵐ 0.084).
- *(iii) The rest of M_16C2†ᵐ_w kept.* M_16C2†ᵐ_ws's raw 3-body ≤ 0.085, hexamers' non-additive part ≤ 13 %, training
  clusters ≤ 0.065, and raw 4-body within 10 % of M_16C2†ᵐ_w's paired (its cross terms untouched).
- *(iv) No 4-body from the cross terms.* M_16C2†ᵐ_us's raw 4-body ≤ 65 % (M_16C2†ᵐ 58, M_16C2†ᵐ_w 74, M_16C2†ᵐ_w2x 76
  %), and on the 242 uncompressed tetramers (closest O–O 2.7–3.0 Å) at most 10 % above M_16C2†ᵐ's, paired (M_16C2†ᵐ_w
  +27 %, M_16C2†ᵐ_w2x +53 %).
- *(v) The 3-body kept by species.* M_16C2†ᵐ_us's raw 3-body ≤ 0.090, and below O–O 2.7 Å ≥ 30 % below M_16C2†ᵐ's on ≥ 3
  of 4 seeds, paired (M_16C2†ᵐ_w −43 %, M_16C2†ᵐ_w2x −46 %).
- *(vi) Its 2-body M_16C2†ᵐ_ws's.* M_16C2†ᵐ_us's raw 2-body within 10 % of M_16C2†ᵐ_ws's overall and at 5–7 Å (the same
  self terms; its cross terms give no dimer anything).
- *(vii) The hexamers* (read, not a check): M_16C2†ᵐ_us's non-additive part at or below M_16C2†ᵐ's 13.9 % — the channels
  apart keep the hexamers; above M_16C2†ᵐ_w2x's 15.1 % — coefficients by species cost them (four numbers alone leave
  16.0 %); between, neither. M_16C2†ᵐ_ws's 2-body at O–O 3.2–5 Å alongside (read): at most 0.050 — the cut kept the gain
  there (M_16C2†ᵐ_w 0.041); at least 0.060 — the self terms at 4–5 Å carried it (M_16C2†ᵐ 0.066).
- *Kept*: M_16C2†ᵐ's seed means within the range of E3-QP4 job B's, E3-QP5's, E3-QP6's and E3-QP7's sixteen seeds;
  M_16C2†ᵐ_w as in E3-QP7 — hexamers ≤ 0.17, raw 3-body ≤ 0.09, raw 2-body ≤ 0.06; the hexamers' non-additive part of
  M_16C2†ᵐ ≤ 17 %; the trimers beyond 9 Å 24–28 % for all four rungs; the field rungs' raw 3-body at largest O–O 7–9 Å ≤
  0.011; their 8 Å test probes' non-additive ≤ 12 % on every seed.

The decomposition's reading of the 5–7 Å loss is refuted if M_16C2†ᵐ_ws's raw 2-body at O–O 5–7 Å is ≥ 50 % above
M_16C2†ᵐ's on ≥ 3 of 4 seeds, paired: with 98 % of the potential the self terms read there cut, the loss was then not
theirs. Its reading of the 4-body loss is refuted if M_16C2†ᵐ_us's raw 4-body is at or above M_16C2†ᵐ_w's 74 %: with no
coefficient in its cross terms the loss was then not their coefficients'.

## 5. Surprises, and what is not tested

A surprise would be: a far-field reading at 10–15 Å off §4; a control's seed mean outside the combined range;
M_16C2†ᵐ_ws off M_16C2†ᵐ_w by more than 10 % on its raw 3-body or its hexamers (the cut takes 6 % of the potential the
self terms read on the trimers, 17 % on the hexamers, and their many-body part is small: `e3qp7-decomposition.md` §4);
M_16C2†ᵐ_us's 4-body on the 242 uncompressed tetramers above M_16C2†ᵐ_w's (the species cross terms have none).

Not tested: the self terms without their linear channels (inputs even in the sources) — the alternative to the reach,
if the cut costs the 3.2–5 Å gain; the read-out's near field at the closest contacts (its 4-body on the 15 compressed
tetramers 0.57 against the labels' 0.34: the largest 4-body error of every C rung there); the species cross terms on the
pinned rung without the read-out; a coarse-grained density beside the pinned charges; contact data past O–O 3.0 Å; the
field of the induced dipoles; E₀'s backbone. Expected cost: about 9 h as one job on an A100, about 5.5 h on an H200.
