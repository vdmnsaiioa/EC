# E3-QP4 — E₀'s width, the energy term's balance, the read-out's near field: pre-registration

*22 Oct 2026, written before any training. The follow-up of E3-QP3 (`e3qp3-results.md` §3–4), set up after Vadic's
"yes" to its proposal. Harness v0.0.13; `configs/water_qaqua_pol4a.yaml` (job A) and `configs/water_qaqua_pol4b.yaml`
(job B) hold every value of the two jobs. The numbers of §3 are computed before training: the matched read-out at width
32 and its readings with E₀ switched off (`scratch/e3qp2_matched_response.py M_16C2pm --F=32 --edge=0.75 / 1.0`), its
3-body on the raw trimers against TTM3-F's (`scratch/e3qp3_response_contact.py --F=32`), the loss's weights, and a
timing of E₀'s training step (`scratch/e3qp4_width_timing.py`). The local master of this note is `notes/` in the repository.*

## 1. What limited E3-QP3, and the three changes

1. **E₀'s width.** E3-QP3's energy weight (×100) and doubled training cut the training clusters' energy error by a
   quarter and raised their force errors by 8–25 %: E₀ (16 features, 2 layers) moved along a front, and the test sets
   gained little. **Job A: `model.features` 16 → 32**, the one change against E3-QP3 (its rungs aside). The species
   read-out's random features share the width, so its response is matched again (§3a): the far field the same to a
   point, the near field not quite (§3b).
2. **The energy term's balance.** The loss weighs a structure's energy by 1 / n_atoms²: E3-QP3's 800 contact
   configurations outweighed its 300 relaxed clusters eightfold at equal errors and carried 58–77 % of the energy term.
   **Job B: job A with the contact configurations' energies at weight 0.125** (`training.loss.contact_energy_weight`;
   Σ n⁻² over the clusters / over the contacts = 2.37 / 19.14 = 0.124: equal weight at equal errors). Their forces keep
   weight 1 — what held the compressed dimers — and they stay in the data.
3. **The read-out's near field.** The matched read-out's 3-body is TTM3-F's induction beyond 3 Å, but 0.83 of it at the
   hydrogen-bond contact and 0.68 below O–O 2.7 Å, and E₀ did not absorb that: M_16C2†ᵐ's raw 3-body did not improve from
   E3-QP2 while M_16†ᵐ's did. **A factor in both jobs: `M_16C2pm_r1`**, M_16C2†ᵐ with the read-out's own band edge
   1.0 Å (`e3_water.RUNG_RESPONSE_EDGE`), the pinned bands at the run's 0.75 Å.

The _e1 rungs (the pinned bands' edge) are dropped: no lever in E3-QP3.

## 2. Design

| | job A (`configs/water_qaqua_pol4a.yaml`) | job B (`configs/water_qaqua_pol4b.yaml`) |
|---|---|---|
| E₀'s width | 32 (E3-QP3: 16) | 32 |
| the contact configurations' energy weight | 1 | 0.125 |
| rungs | M_∞, M_16†ᵐ, M_16C2†ᵐ, M_16C2†ᵐ_r1 | the same |

Everything else is E3-QP3's, bit for bit: the 300 training clusters and 4 monomers, 384 training probes, 600 contact
dimers and 200 contact trimers, the test hexamers, dimers and probes and the raw sets; the read-out matched first on the
320 training probes beyond r_c and held; Adam 4000 (batch 32), L-BFGS 1000 (chunk 64); energies 100, forces 1; 4 seeds;
the pins as trained in every reading; then `scratch/e3qp3_shortrange_diagnostic.py` on every model. Job A against
E3-QP3 reads the width — cleanly for M_∞ and M_16†ᵐ, which have no read-out; for the C rung the width also moves the
read-out's near field (§3b). Job B against job A reads the balance. M_16C2†ᵐ_r1 against M_16C2†ᵐ reads the read-out's
near field, in each job.

## 3. Numbers before training

**(a) The far field at width 32** (the read-out solved as `T.fit` does, read with E₀ off; beyond r_c the trained models
read exactly these, E₀ cancelling in the probe's interaction). The contacts are not in the solve, so the two jobs share
it, and the solve's history is the same at every seed:

| rung | test probes' non-additive at 10 / 12 / 15 Å | interaction | through order 1; order 2 | 8 Å beyond r_c | trimers 7–9 / ≥ 9 Å (E₀ off) |
|---|---|---|---|---|---|
| M_16C2†ᵐ (read-out at 0.75 Å) | 4 / 5 / 3 % (E3-QP3: the same) | 0 / 0 / 0 % | 15 / 15 / 14; 5 / 5 / 4 % | 7 % | 26 / 25 % |
| M_16C2†ᵐ_r1 (1.0 Å) | 3 / 3 / 2 % | 0 / 0 / 0 % | 14 / 15 / 13; 3 / 3 / 3 % | 6 % | 26 / 24 % |
| M_16†ᵐ (no read-out) | 100 % (interaction 10 / 9 / 15 %) | | | | 100 / 100 % |
| M_∞ | 100 % | | | | 100 / 100 % |

**(b) The read-out's near field** (E₀ off; kcal/mol rms; the 3-body on E3-QP3's 4673 raw trimers by the closest O–O):

| | width 16, 0.75 Å (E3-QP3) | width 32, 0.75 Å | width 32, 1.0 Å (r1) |
|---|---|---|---|
| hexamers: the response's many-body energy against TTM3-F's | 21 % | 29 % | 15 % |
| hexamers: its whole energy against TTM3-F's induction, rmse | 0.55 | 0.56 | 0.36 |
| test dimers: the same, rmse | 0.146 | 0.151 | 0.093 |
| raw trimers: read-out less TTM3-F's 3-body induction, below 2.7 / 2.7–3.0 / from 3.0 Å | 0.375 / 0.141 / 0.022 | 0.399 / 0.149 / 0.025 | 0.331 / 0.147 / 0.024 |
| the read-out on TTM3-F's (slope), the same bins | 0.68 / 0.83 / 1.01 | 0.66 / 0.83 / 1.03 | 0.64 / 0.75 / 1.00 |
| E₀'s 3-body target (labels less read-out), all / below 2.7 Å | 0.409 / 0.925 | 0.413 / 0.945 | 0.410 / 0.909 |

Matching on the far field leaves the near field to the random features: at width 32 the 0.75 Å read-out sits a little
farther from TTM3-F than at 16. The 1.0 Å edge brings it closer on the hexamers (15 against 29 %), on the dimers (0.093
against 0.151) and on the compressed trimers (0.33 against 0.40), not at the hydrogen-bond contact (0.147 against
0.149; slope 0.75 against 0.83).

**(c) The energy term.** At equal errors the contacts weigh Σ n⁻² = 19.14 against the clusters' 2.37 (the probes' 1.85);
at weight 0.125, 2.39. At E3-QP3's errors their share against the clusters falls from 58 / 68 / 76 % (M_∞ / M_16†ᵐ /
M_16C2†ᵐ) to 15 / 21 / 28 %.

**(d) The cost.** E₀'s training step at width 32 takes 1.7–1.9 times that at 16 (a batch of 32 clusters, CPU, two
timings). With E3-QP3's times on a 3090 that is about 14 h per job including the diagnostic; the two jobs run in
parallel.

**(e) The baselines**, E3-QP3 (seed means, kcal/mol):

| | M_∞ | M_16†ᵐ | M_16C2†ᵐ |
|---|---|---|---|
| training clusters: energy; forces rigid-body / internal (kcal/mol/Å) | 0.198; 0.59 / 0.66 | 0.155; 0.50 / 0.67 | 0.129; 0.51 / 0.72 |
| contact configurations: energy | 0.082 | 0.079 | 0.080 |
| compressed dimers (O–O 2.3–2.7 Å); all raw dimers | 0.114; 0.113 | 0.117; 0.093 | 0.119; 0.093 |
| raw 3-body: all; closest O–O below 2.7 / 2.7–3.0 Å | 0.100; 0.156 / 0.102 | 0.140; 0.246 / 0.138 | 0.153; 0.316 / 0.146 |
| hexamers: energy; non-additive | 0.443; 24.1 % | 0.267; 31.2 % | 0.189; 16.0 % |
| test dimers; raw 4-body | 0.074; 59 % | 0.058; 104 % | 0.057; 49 % |

## 4. Predictions

**Computed** (to the printed digit, on every seed, the same in both jobs): §3a.

**Expected** (seed means unless said; kcal/mol):

- *Job A, E₀'s fit — the width's signature*: M_∞'s and M_16†ᵐ's training clusters' energy error at least 15 % below
  E3-QP3's (≤ 0.17 and ≤ 0.13), with both their force parts at or below E3-QP3's; M_16C2†ᵐ's ≤ 0.11.
- *Job A, the test sets*: hexamers M_16†ᵐ ≤ 0.24, M_16C2†ᵐ ≤ 0.17, M_∞ 0.35–0.50; test dimers ≤ 0.05 for the pinned
  rungs, ≤ 0.07 for M_∞; all raw dimers ≤ 0.085 for the pinned rungs and below M_∞ on every seed, ≤ 0.105 for M_∞;
  compressed dimers ≤ 0.12 for every rung; raw 3-body M_∞ ≤ 0.095, M_16†ᵐ ≤ 0.13, M_16C2†ᵐ ≤ 0.145.
- *Job B against job A, every rung*: the relaxed training clusters' energy error at least 10 % lower; the contact
  configurations' higher, but ≤ 0.15; the compressed dimers within 0.02 of A's; the hexamers lower.
- *The read-out's near field* (M_16C2†ᵐ_r1 against M_16C2†ᵐ, in each job): the raw 3-body below O–O 2.7 Å lower on
  at least 3 of 4 seeds; the hexamers' non-additive energy at or below; at 2.7–3.0 Å open (the read-out's error there
  is the same, §3b).
- *Kept*: the raw 4-body M_16C2†ᵐ ≤ 55 %; the hexamers' non-additive M_16C2†ᵐ ≤ 17 %; the CCSD(T) trimers beyond 9 Å
  24–28 % for the C rungs, 95–105 % for the others.

## 5. Surprises, and what is not tested

A surprise would be: a far-field reading at 10–15 Å off §3a, or differing between seeds or between the jobs; in job A,
M_∞'s or M_16†ᵐ's training-cluster energy error less than 5 % below E3-QP3's, or their forces above E3-QP3's (the width
not the limit — then the optimiser: its learning rate and length); in job B, the compressed dimers more than 0.05 above
A's (the contacts' energies were needed at full weight), or the relaxed clusters no better than A's (the balance not a
limit); M_16C2†ᵐ_r1 above M_16C2†ᵐ below 2.7 Å on 3 or more seeds in both jobs.

Not tested: widths beyond 32, E₀'s depth (2 layers) and cutoff (6 Å); the energy weight (100, kept); other contact
weights; the read-out's near field beyond its edge (a damping form, or matching on near-field data); flexible
monomers. Expected cost: about 14 h per job on one 3090, the two in parallel.
