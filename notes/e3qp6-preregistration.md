# E3-QP6 — every molecule whole, and the field term's degree: pre-registration

*24 Oct 2026, written before any training. The follow-up of E3-QP5 (`e3qp5-results.md` §4), set up after Vadic's "yes
please" to its plan. Harness v0.0.15; `configs/water_qaqua_pol6.yaml` holds every value. The numbers of §3 are computed
before training from the PES and the data alone, with the harness's own field tensors (`scratch/e3qp6_preregistration.py`),
and from a timing of the training step (`scratch/e3qp6_field_timing.py`). The local master of this note is `notes/` in
the repository.*

## 1. The question, and the two changes

E3-QP5 confirmed (b): given the other molecules' pinned field, the pinned rungs' E₀ closes its 3-body gap at contact
(M_16†ᵐ_f's near-rigid 3-body 0.110 → 0.062, below M_∞'s 0.072). It also found two faults of its field term
(`e3qp5-results.md` §3.5):

- *(i) part-charged molecules.* Each source tapered by its own distance leaves a molecule that straddles the taper part
  charged — 27–38 % of the (atom, other molecule) pairs in the raw trimers, the training clusters and the hexamers, with
  a net charge of 0.3 e rms — so the input carries a monopole's field where the truth has a dipole's. Two polarisabilities
  on that input predict the extended trimers' 3-body worse than predicting none, and M_16†ᵐ_f's 3-body there equals their
  residual to the printed digit (largest O–O 7–9 Å 0.0388, beyond 9 Å 0.0132).
- *(ii) the contact data's edge.* The pinned field rungs' 2-body just past it: O–O 3.0–3.2 Å +72 / +47 %, the dimers off
  the training pairs +37 / +25 %, where the taper plays no part.

And the C rung lost its hexamers (0.155 → 0.178), its 4-body (57 → 81 %) and the trimers beyond 9 Å (27 → 42 %) — by
(i), by the MLP's terms beyond degree 2 in the field, or by the term's overlap with the read-out at intermediate range:
E3-QP5 cannot tell them apart.

**Change 1** (`Rung.e0_field_whole`): every molecule whole. All of a molecule's sources — its O (no charge in TTM3-F), its
two H atoms and its M-site — share one weight at an atom i, the product of their tapers,

    w_iB = prod_{k in B} taper(r_ik; r_c - 1, r_c)      (5-6 A),

1 when every site of B lies within 5 Å of i, 0 when any lies beyond 6 Å (`bandfields.local_field_tensors` with `frag_src`).
By construction, and tested (`tests/test_e0_field_whole.py`):

- *neutral*: a molecule straddling the taper enters with all its charges in their proportions, never part charged — its
  field a dipole's and higher, as the truth's;
- *local*: no source beyond r_c in reach, so a molecule enters an atom's term only when all its sites lie within E₀'s
  graph of the atom; every probe beyond the reach interacts as with the base rung, and the far field is the base rung's;
- *the base rung at the start*, digit for digit; E3-QP5's rungs (and those before) unchanged, to 1e-10 (golden values from
  v0.0.14);
- C² in the positions (a product of C² tapers), its forces the energy's gradient (central differences, h = 1e-5 bohr).

**Change 2** (`Rung.e0_field_form: quadratic`; on the C rung only, with change 1): the term at degree 2 in the field,

    eps_i = -1/2 e_scale [ E_i . A_i . E_i + G_i : B_i : G_i ]
    E.A.E = a |E|^2 + sum_f b_f (E . v_f)^2 + sum_f b'_f E . t_f . E,      G:B:G = c |G|^2 + sum_f d_f (t_f : G)^2,

the coefficients (a, b, b′, c, d) an MLP of E₀'s invariants s_i whose output layer starts at zero (`heads.field_energy2`):
the read-out's order-0 form (`bandfields.response_energy`) on the local field, and its square in the field gradient. No
potential, no term linear in the field, no function of the field beyond its square — exactly quadratic in the pinned
charges (tested: the charges doubled, the term's energy ×4). It keeps the cross term that carries the 3-body at contact
(§3b) and drops what E3-QP5's MLP could add besides: the potential (a charge-like channel q(s_i)·φ_i, the input's
longest tail), the first-order multipole corrections (v·E, t:G) and the products of three and more fields.

## 2. Design

| rung | what it is | what it reads |
|---|---|---|
| M_16†ᵐ, M_16C2†ᵐ | E3-QP5's controls, run again | the paired comparison; whether they reproduce |
| M_16†ᵐ_w | M_16†ᵐ with E3-QP5's field term, every molecule whole | (i) where E₀ carries the whole 3-body; the contact gain kept |
| M_16C2†ᵐ_w | M_16C2†ᵐ with the same | (i) for the C rung: its hexamers, its 4-body, the extended trimers |
| M_16C2†ᵐ_w2 | M_16C2†ᵐ_w at degree 2 in the field | (ii); and the C rung's losses, if they belong to the MLP's higher orders |

Everything else is E3-QP5's bit for bit (`configs/water_qaqua_pol6.yaml` differs from `configs/water_qaqua_pol5.yaml` in
its rungs alone, tested): E3-QP4 job B's data, training, matching and readings, 4 seeds, the short-range diagnostic with
every frame's n-body energies. A field rung and its base share their seeds and start from the same parameters: the
comparison is paired, seed by seed. E3-QP5's field rungs are not run again; their seed means are the reference for the
changes (not paired). Two jobs, each base with its field rungs: a = M_16†ᵐ, M_16†ᵐ_w; b = M_16C2†ᵐ, M_16C2†ᵐ_w,
M_16C2†ᵐ_w2.

## 3. Numbers before training

**(a) The reach.** Every probe configuration at 10, 12 and 15 Å (288 training, 60 test, rebuilt bit for bit) lies beyond
the whole-molecule term's reach — no core atom with all of the probe's sites within 6 Å, no probe atom with all of a core
molecule's — so the registered far field is the bases'. At 8 Å, 73 of the 96 training probes and 18 of the 20 test probes
lie beyond it (E3-QP5's reach: 62 and 18; beyond r_c, E₀ blind: 37 and 13). Within the clusters, of the (atom, other
molecule) pairs the term sees whole (weight 1), in the taper, or not at all: the test hexamers 51 / 20 / 29 %, the
training clusters 65 / 19 / 16 %.

**(b) The cross term.** On E3-QP4's 4673 raw trimers (the rigid monomer's charges, as the models read them), one effective
polarisability per species on the input's first-order cross term, label_3b ≈ −Σ_i a_Z(i) E_i(B)·E_i(C), refitted for each
weighting (kcal/mol rms left; S, L the smallest and the largest O–O, Å):

| weighting | all | S < 2.7 | S 2.7–3 | S ≥ 3 | L < 5 | L 5–7 | L 7–9 | L ≥ 9 |
|---|---|---|---|---|---|---|---|---|
| the labels | 0.623 | 1.268 | 0.596 | 0.145 | 0.779 | 0.312 | 0.0272 | 0.0109 |
| no taper | 0.066 | 0.096 | 0.068 | 0.034 | 0.082 | 0.036 | 0.0051 | 0.0025 |
| each source by its own distance, 4.5–5.5 Å (E3-QP5) | 0.114 | 0.143 | 0.119 | 0.074 | 0.108 | 0.137 | 0.0388 | 0.0132 |
| every molecule whole, 5–6 Å (E3-QP6) | 0.078 | 0.112 | 0.081 | 0.040 | 0.089 | 0.065 | 0.0230 | 0.0109 |

With every molecule whole the input carries more of the 3-body at every contact than E3-QP5's (88 against 82 % of its rms,
the untapered limit 89 %), and nowhere does it push the 3-body the wrong way: where the farthest molecule is 7–9 Å from
another the fitted 3-body (rms 0.0087, correlation with the labels +0.61) leaves less than the labels (0.023 against
0.027; E3-QP5's 0.045 rms at correlation +0.54 left 0.039), and beyond 9 Å it is nil (0.0001 rms). The numbers repeat
`scratch/e3qp5_reach_scan.py`'s (numpy) through the harness's own tensors, to the printed digit.

**(c) The raw dimers.** Where the term sees the partner, by O–O, of the atoms (whole in reach / in the taper / beyond):
below 3.2 Å 100 / 0 / 0 % in every bin (2.3–2.7, 2.7–2.9, 2.9–3.0, 3.0–3.2 Å) — the input there is the untapered field,
as E3-QP5's was (1 % of the atoms or fewer saw a straddling partner); 3.2–5 Å 90 / 9 / 1 %; 5–7 Å 6 / 42 / 51 %; beyond
7 Å none.

**(d) The cost.** One training step at width 32 (a batch of 32 clusters, CPU): M_16†ᵐ_w 1.01×, M_16C2†ᵐ_w 1.04×,
M_16C2†ᵐ_w2 1.00× their bases (E3-QP5's field rungs 0.96×, 1.07× in the same timing). With E3-QP5's times on an H200
(M_16†ᵐ 475 s, M_16†ᵐ_f 555 s, M_16C2†ᵐ 875 s, M_16C2†ᵐ_f 950 s a seed; the diagnostic about 4 min a model): job a about
1.7 h, job b about 3.9 h.

**(e) The baselines**, E3-QP5 (seed means; the field rungs the reference for the changes, the bases for the pairs):

| | M_16†ᵐ | M_16†ᵐ_f | M_16C2†ᵐ | M_16C2†ᵐ_f |
|---|---|---|---|---|
| raw 3-body: strict near-rigid; all; below O–O 2.7 Å | 0.110; 0.117; 0.195 | 0.062; 0.078; 0.127 | 0.089; 0.126; 0.260 | 0.065; 0.087; 0.167 |
| raw 3-body, largest O–O 7–9 Å; the trimers beyond 9 Å | 0.026; 99.8 % | 0.039; 121.6 % | 0.0083; 26.7 % | 0.0136; 41.6 % |
| raw 2-body: all; O–O 3.0–3.2 Å; 5–7 Å | 0.076; 0.067; 0.021 | 0.067; 0.113; 0.046 | 0.085; 0.067; 0.025 | 0.061; 0.093; 0.039 |
| hexamers: energy; non-additive | 0.233; 26.6 % | 0.206; 19.5 % | 0.155; 12.7 % | 0.178; 14.5 % |
| raw 4-body; the 8 Å test probes' non-additive | 120 %; 98–103 % | 82 %; 100–110 % | 57 %; 4–6 % | 81 %; 21–31 % |
| training clusters' energy | 0.087 | 0.078 | 0.080 | 0.066 |

## 4. Predictions

**Computed** (to the printed digit, on every seed): the far field at 10 / 12 / 15 Å of every new rung is its base's —
M_16C2†ᵐ_w's and M_16C2†ᵐ_w2's test probes' non-additive 4 / 5 / 3 %, through order 1 15 / 15 / 14 %, order 2 5 / 5 /
4 %; M_16†ᵐ_w's interaction 10 / 9 / 15 %, non-additive 100 % — and M_16C2†ᵐ_w's and M_16C2†ᵐ_w2's read-out solves
M_16C2†ᵐ's, step for step.

**Expected** (seed means unless said; kcal/mol; "paired" compares a new rung with its base seed by seed, in this run):

- *(i): no part charges, no wrong 3-body beyond the contact.* M_16†ᵐ_w's raw 3-body with the largest O–O 7–9 Å at or
  below M_16†ᵐ's on ≥ 3 of 4 seeds, paired (M_16†ᵐ_f was above on 4 of 4); the CCSD(T) trimers beyond 9 Å back within the
  bands, M_16†ᵐ_w 95–105 %, M_16C2†ᵐ_w and M_16C2†ᵐ_w2 24–28 %; the C rungs' raw 3-body at 7–9 Å ≤ 0.011 (M_16C2†ᵐ 0.0083,
  M_16C2†ᵐ_f 0.0136); every new rung's raw 2-body at O–O 5–7 Å within 25 % of its base's (E3-QP5's +121 / +52 %);
  M_16C2†ᵐ_w's and M_16C2†ᵐ_w2's 8 Å test probes' non-additive energy ≤ 12 % on every seed (M_16C2†ᵐ_f 21–31 %; 2 of
  the 20 lie within the term's reach, 7 within r_c — a miss with the rest of (i) met puts E3-QP5's 8 Å loss on E₀ as
  retrained, not on the input).
- *The contact gain kept*: M_16†ᵐ_w's strict-subset 3-body ≥ 15 % below M_16†ᵐ's on ≥ 3 of 4 seeds, paired, its mean
  ≤ 0.065 (M_16†ᵐ_f 0.062); M_16C2†ᵐ_w's and M_16C2†ᵐ_w2's raw 3-body below O–O 2.7 Å ≥ 15 % below M_16C2†ᵐ's on ≥ 3 of
  4 seeds, paired. At no price for M_16†ᵐ_w, as E3-QP5 registered for M_16†ᵐ_f: its raw 2-body at most 0.005 above
  M_16†ᵐ's, its hexamers and its training clusters at or below M_16†ᵐ's, its raw 4-body not above.
- *The C rung's losses*: M_16C2†ᵐ_w's hexamers, and their non-additive energy, at or below M_16C2†ᵐ's; the same for
  M_16C2†ᵐ_w2; the raw 4-body ≤ 65 % for both (M_16C2†ᵐ 57 %, M_16C2†ᵐ_f 81 %). Read together:

  | M_16C2†ᵐ_w | M_16C2†ᵐ_w2 | the C rung's losses in E3-QP5 were |
  |---|---|---|
  | ✓ | ✓ | the part charges (fault (i)) |
  | ✗ | ✓ | the MLP's terms beyond degree 2 in the field |
  | ✗ | ✗ | the term's overlap with the read-out at intermediate range: next, a term that adds to the read-out only what it lacks |
  | ✓ | ✗ | the part charges, and the quadratic form too narrow for the rest |

- *(ii): the edge is not the taper's.* M_16†ᵐ_w's and M_16C2†ᵐ_w's raw 2-body at O–O 3.0–3.2 Å ≥ 20 % above their bases'
  on ≥ 3 of 4 seeds, paired — whole molecules leave their input there as it was (§3c) — and M_16C2†ᵐ_w2's mean at most
  10 % above M_16C2†ᵐ's: if the leak is the potential's and the linear terms' (a correction to the pinned bands at
  contact that nothing past 3.0 Å tells to vanish), degree 2 drops it.
- *Kept*: the controls' seed means within the range of E3-QP4 job B's and E3-QP5's eight seeds together; the hexamers'
  non-additive energy of M_16C2†ᵐ ≤ 17 %; the trimers beyond 9 Å M_16†ᵐ 95–105 %, M_16C2†ᵐ 24–28 %.

(i) is refuted if M_16†ᵐ_w's raw 3-body at 7–9 Å is not below M_16†ᵐ_f's E3-QP5 mean (0.0388) on every seed, or its
trimers beyond 9 Å not back within 95–105 %: the part charges were then not what spoilt the extended trimers.

## 5. Surprises, and what is not tested

A surprise would be: a far-field reading at 10–15 Å off §4; a control's seed mean outside the combined range; M_16†ᵐ_w's
contact gain (strict subset, against M_16†ᵐ) less than half M_16†ᵐ_f's — the input carries more of the 3-body (§3b), so
whole molecules should not cost the contact; M_16C2†ᵐ_w2 missing its contact prediction — degree 2 carries the cross term
by construction.

Not tested: the field in E₀'s message passing; a term built to add to the read-out only what it lacks; contact data past
O–O 3.0 Å (fault (ii)'s other remedy); the product's end inside r_c (a shorter reach costs the contact: E3-QP5's scan,
3–4 Å per source 0.330); the field of the induced dipoles; E₀'s backbone. Expected cost: about 1.7 and 3.9 h on one
H200 each (5.6 h as one job), about 19 h as one job on a 3090.
