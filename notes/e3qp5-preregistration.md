# E3-QP5 — E₀'s view of the pinned field: pre-registration

*23 Oct 2026, written before any training. The follow-up of E3-QP4 (`e3qp4-results.md` §4), set up after Vadic's "yes"
to its plan. Harness v0.0.14; `configs/water_qaqua_pol5.yaml` holds every value. The numbers of §3 are computed before
training from the PES and the data alone (`scratch/e3qp5_preregistration.py`) and from a timing of the training step
(`scratch/e3qp5_field_timing.py`). The local master of this note is `notes/` in the repository.*

## 1. The question, and the one change

E3-QP4 left one reading where M_∞ wins: the pinned rungs' 3-body at contact. Their raw 3-body is 1.33–1.53× M_∞'s
whatever E₀'s width or the energy term's balance, below O–O 3 Å, in the rigid part, and largest for M_16†ᵐ on the
near-rigid trimers (1.51×), where the charge flux plays no part; the raw trimers are no farther from the training
triples than the test hexamers' own (`e3qp4-results.md` §3.3, §3.6). M_16†ᵐ has no read-out and pairwise bands, so its
3-body is E₀'s alone, and its E₀ fits the training clusters far better than M_∞'s (0.086 against 0.143) and generalises
the 3-body worse: what E₀ is given, not how much of it. The C rungs' read-out — an explicit induction — narrows the gap on
the near-rigid trimers (1.25×, M_16C2†ᵐ_r1 1.15×). **The hypothesis (b)**: M_∞'s E₀ must carry the electrostatics and
builds field-like features with them, from which the induction-like 3-body follows; the pinned rungs' E₀ has the bands
for the electrostatics and no such need. Given the field, it should close the gap.

**The change** (`Rung.e0_field`): E₀'s site energies read, at every atom i, the potential φ_i, field E_i and field gradient
G_i of the other molecules' pinned charges — TTM3-F's site charges on the H atoms and the M-sites, as in the bands —
through the band kernel at ℓ₁ = 0.75 Å, every source's share tapered to zero between r_c − 1.5 and r_c − 0.5 Å (4.5–5.5 Å;
a C² smoothstep; `bandfields.local_field_tensors`), through

    ε_i^F = e_scale [ MLP(s_i, φ_i, (W_v v_i)·E_i, (W_t t_i):G_i, |E_i|², |G_i|²) − MLP(s_i, 0, 0, 0, 0, 0) ]

(φ, E, G in units of 0.01 a.u.; `heads.field_energy`): E₀'s invariants, its vector and tensor channels contracted with
the field and its gradient, and the two squares — MACE-POLAR's non-local read-out (Batatia et al., arXiv:2602.19411),
here local. By construction, and tested (`tests/test_e0_field.py`):

- *local*: no share from a source beyond 5.5 Å, and an M-site lies 0.27 Å from its O, so every source in reach belongs to a
  molecule that E₀'s graph (6 Å) already connects to the atom: the term adds no connection, every probe beyond r_c of its
  core interacts exactly as with the base rung, and the far field is the base rung's;
- *zero at zero field*: an isolated molecule, or one beyond the taper of every other, has no field term;
- *the base rung at the start*: the MLP's output layer starts at zero and the term's parameters are drawn on their own
  key, so a field rung starts as its base rung digit for digit; the rungs that existed are unchanged (their energies and
  forces at fixed parameters as v0.0.13's to 1e-10).

Unlike the read-out, the term may be linear in the field: the read-out's degree-2 rule keeps a far field from being
degenerate with a dipole source, and this input has no far field — as E₀ itself has none.

## 2. Design

| rung | what it is | what it reads |
|---|---|---|
| M_16†ᵐ, M_16C2†ᵐ | job B's rungs, run again | the controls of the paired comparison; whether job B reproduces |
| M_16†ᵐ_f | M_16†ᵐ with the field term | (b) where E₀ carries the whole 3-body (no read-out) |
| M_16C2†ᵐ_f | M_16C2†ᵐ with the field term | whether E₀ given the field corrects the read-out's near field |
| M_∞_f | M_∞ with the field term (the pinned charges source the input alone: no bands) | the field as a better input for any E₀, or the pinned rungs' missing piece |

M_∞ is job B's. Everything else is job B's bit for bit (`configs/water_qaqua_pol5.yaml` differs from
`configs/water_qaqua_pol4b.yaml` in its rungs alone): the 300 training clusters and 4 monomers, the 384 training probes,
the 600 contact dimers and 200 contact trimers at weight 0.125 in the energy term, E₀'s width 32, the read-out matched
first on the training probes beyond r_c and held, Adam 4000 (batch 32), L-BFGS 1000 (chunk 64), energies 100, forces 1,
4 seeds, the pins as trained in every reading; then `scratch/e3qp3_shortrange_diagnostic.py --save_frames` on every model,
which now keeps every frame's n-body energies, so that the 3-body's strength and its direction against TTM3-F's induction
can be read after the run without the GPU. A field rung and its base share their seeds and start from the same
parameters: the comparison is paired, seed by seed.

## 3. Numbers before training

**(a) The reach.** The M-site sits 0.270 Å from its O (γ = 0.46). Of E3-QP4's probe configurations (rebuilt bit for bit),
every one at 10, 12 and 15 Å lies beyond the field term's reach — no probe source within 5.5 Å of a core atom and no core
source within 5.5 Å of a probe atom (288 training, 60 test) — so the registered far field at 10–15 Å is the base rungs'.
At 8 Å, 62 of the 96 training probes and 18 of the 20 test probes lie beyond it (37 and 13 beyond r_c): the 8 Å readings
may move.

**(b) The input's size** on the training data (the rms of each quantity's magnitude over the atoms, and the 99th
percentile of the field's; a.u., against the unit 0.01):

| | potential O / H | field O / H (99 %) | field gradient O / H |
|---|---|---|---|
| relaxed clusters | 0.037 / 0.040 | 0.012 / 0.014 (0.025 / 0.034) | 0.009 / 0.011 |
| contact configurations | 0.023 / 0.024 | 0.010 / 0.012 (0.024 / 0.033) | 0.009 / 0.012 |

**(c) How much of the 3-body the input carries.** On E3-QP4's 4673 raw trimers (the rigid monomer's charges, as the
models read them), one effective polarisability per species on the input's first-order cross term,
label_3b ≈ −Σ_i a_Z(i) E_i(B)·E_i(C) (every atom i of one molecule, B and C the other two, tapered as the rungs read it),
fitted by least squares, leaves 0.114 of the 3-body's 0.623 kcal/mol rms — 82 % of it explained by two numbers (a_O 2.1,
a_H 4.7 bohr³); TTM3-F's whole 3-body induction explains 54 % (0.287 left). By the closest O–O:

| | below 2.7 Å (371) | 2.7–3.0 Å (3361) | from 3.0 Å (941) | all |
|---|---|---|---|---|
| the labels' 3-body | 1.268 | 0.596 | 0.145 | 0.623 |
| left by the two-number fit | 0.143 | 0.119 | 0.074 | 0.114 |
| job B's M_∞ / M_16†ᵐ / M_16C2†ᵐ | 0.126 / 0.192 / 0.267 | 0.086 / 0.115 / 0.122 | 0.051 / 0.056 / 0.032 | 0.084 / 0.115 / 0.129 |

Two numbers on the input do as well as M_16†ᵐ's E₀ on the 3-body, and better below 2.7 Å. Without the taper the fit
leaves 0.072 below 3 Å (0.122 with it): the third molecule's field from beyond 5.5 Å carries part of the 3-body at
contact, which a local term cannot see — nor can E₀ (6 Å); in the C rungs it is the read-out's.

**(d) The cost.** One training step at width 32 (a batch of 32 clusters, CPU): M_16†ᵐ_f 1.09×, M_16C2†ᵐ_f 1.04× their
bases, M_∞_f 0.92× M_∞ (within the timing's noise). With job B's times on the H200 (M_∞ 520 s, M_16†ᵐ 470 s, M_16C2†ᵐ
900 s a seed) the run is about 4 h and the diagnostic 1 h; about 17 h on a 3090.

**(e) The baselines**, job B (seed means, kcal/mol):

| | M_∞ | M_16†ᵐ | M_16C2†ᵐ |
|---|---|---|---|
| training clusters: energy; forces rigid-body / internal (kcal/mol/Å) | 0.143; 0.53 / 0.55 | 0.086; 0.39 / 0.49 | 0.081; 0.41 / 0.53 |
| raw 3-body: all; strict (1450); its rigid part | 0.084; 0.072; 0.081 | 0.115; 0.109; 0.102 | 0.129; 0.090; 0.106 |
| compressed dimers (O–O 2.3–2.7 Å); all raw dimers | 0.114; 0.113 | 0.099; 0.075 | 0.111; 0.082 |
| hexamers: energy; non-additive | 0.435; 22.8 % | 0.226; 26.0 % | 0.160; 13.1 % |
| test dimers; raw 4-body | 0.068; 51 % | 0.047; 121 % | 0.046; 57 % |

## 4. Predictions

**Computed** (to the printed digit, on every seed): the far field at 10 / 12 / 15 Å of each field rung is its base's —
M_16C2†ᵐ_f's test probes' non-additive 4 / 5 / 3 %, through order 1 15 / 15 / 14 %, order 2 5 / 5 / 4 %; M_16†ᵐ_f's
interaction 10 / 9 / 15 %; M_∞_f 100 % — and M_16C2†ᵐ_f's read-out solve the same as M_16C2†ᵐ's, step for step.

**Expected** (seed means unless said; kcal/mol; "paired" compares a field rung with its base seed by seed, in this run):

- *(b), where E₀ carries the whole 3-body*: M_16†ᵐ_f's raw 3-body on the strict subset at least 15 % below M_16†ᵐ's on
  at least 3 of 4 seeds, paired, its mean ≤ 0.09 (job B 0.109; M_∞ 0.072); the rigid part ≤ 0.09 (0.102); all raw
  trimers ≤ 0.10 (0.115).
- *At no price*: M_16†ᵐ_f's raw 2-body at most 0.005 above M_16†ᵐ's, its hexamers and its training clusters' energy at or
  below M_16†ᵐ's, its raw 4-body not above M_16†ᵐ's.
- *The pinned rungs' piece, not a better input for any E₀*: M_∞_f's strict-subset 3-body gains less than half as much,
  relative to job B's M_∞, as M_16†ᵐ_f's does relative to M_16†ᵐ.
- *The C rung*: M_16C2†ᵐ_f's raw 3-body below 2.7 Å at least 15 % below M_16C2†ᵐ's (job B 0.267) on at least 3 of 4
  seeds, paired; its hexamers and their non-additive energy at or below M_16C2†ᵐ's.
- *Kept*: the hexamers' non-additive energy of M_16C2†ᵐ ≤ 17 %; the CCSD(T) trimers beyond 9 Å 24–28 % for the C rungs,
  95–105 % for the others; the controls as job B's to the printed digit if the run's GPU repeats job B's arithmetic (the
  same model and software), and otherwise every seed mean within job B's range over its seeds.

(b) is refuted if M_16†ᵐ_f's strict-subset 3-body is not below M_16†ᵐ's on at least 3 of 4 seeds: the field is then not
what the pinned rungs' E₀ lacks, and E₀'s backbone (its body order) is the next suspect.

## 5. Surprises, and what is not tested

A surprise would be: a far-field reading at 10–15 Å off §4, or a control's seed mean outside job B's range; a field
rung's training clusters 10 % or more better than its base's with its strict-subset 3-body not better (the field fitted,
not used); a field rung's raw 2-body more than 10 % above its base's; M_∞_f gaining as much as M_16†ᵐ_f (the field a
better input for any E₀ — the gap then not explained by it).

Not tested: the field in E₀'s message passing (as an initial vector or tensor feature) rather than at its site energies;
other tapers, units or field orders (the potential alone, the field alone); the field of the induced or of learned
sources; the read-out's near-field form; E₀'s backbone. Expected cost: about 5 h on one H200, 17 h on a 3090.
