# E3-QP2 results — the response matched on the far field, read against `e3qp2-preregistration.md`

*20 Oct 2026. One run of `configs/water_qaqua_pol2.yaml` (both parts), 19–20 Oct, Niflheim node sd653, one A100,
harness v0.0.11 (commit aeb0a68). Dimer part 0.4 h (2 rungs × 4 seeds), clusters 4.06 h (8 rungs × 4 seeds), as
estimated; the q-AQUA-pol check passed in the job. Percentages are rmse as a share of the signal rms, means over the 4
seeds. The raw sets' per-bin shares use each bin's label rms, computed with the run's selection
(`scratch/e3qp2_label_bins.py`: the PES's per-bin errors reproduce the log's); the seed means come from the log
(`scratch/e3qp2_summarise_log.py`). Section 3 holds what was computed after the run. The local master of this note is
`notes/` in the repository.*

## 1. The readings

| reading | registered (§4) | result |
|---|---|---|
| dimer: M_1†ᵐ, M_16†ᵐ tails | the §3a values to 1e-6, seed spread 0 | ✓ every row to the printed digits (M_16†ᵐ 6.20e-3 at 7.8 Å … 3.05e-5 at 305 Å; M_1†ᵐ 2.41e-2 …); spread 6e-17–9e-17 E_h, the double-precision floor |
| probes at 10–15 Å, read-out rungs | the §3d values to the printed digit, every seed | ✓ every rung, distance and seed — non-additive and interaction, and the C rungs' first- and second-order truncations |
| probes at 10–15 Å, M_∞ and the daggers | 100 % non-additive; the §3d interactions | ✓ 100 %; interactions 13 / 13 / 19 % (M_16†), 10 / 9 / 15 % (M_16†ᵐ) |
| probes at 8 Å | within ± 6 points of §3d | ✓ M_16S0†ᵐ 9 (§3d 9), C0†ᵐ 8–10 (8), S2†ᵐ 4–6 (4), C2†ᵐ 4–7 (4), C0† 25–26 (25) |
| CCSD(T) trimers ≥ 9 Å (108) | `*pm` read-outs ± 5 of §3d and < 30 % on every seed; M_16C0† 28–38 %; M_∞, daggers 95–105 % | ✓ 20 / 17 / 22 / 25 % (S0, C0, S2, C2: the §3d values), 17–26 % on every seed; C0† 33 %; M_∞ and the daggers 100 % |
| CCSD(T) trimers 7–9 Å (311) | `*pm` read-outs 20–35 % | ✓ 30 / 30 / 32 / 32 % (the PES 22; E3-QP's read-outs 58–60) |
| hexamers, non-additive (0.997 kcal/mol) | `*pm` read-outs 12–25 %; M_∞ 18–30; M_16†ᵐ 25–45; M_16C0† 20–40 | C2†ᵐ 15.8 ✓, S2†ᵐ 23.1 ✓, C0†ᵐ 23.7 ✓, **S0†ᵐ 31.2 ✗** (seeds 28 / 23 / 28 / 46); M_∞ 22.5 ✓; **M_16†ᵐ 45.8 ✗** (just); C0† 33.6 ✓; M_16† 36.0 |
| open: M_16C2†ᵐ ≤ M_∞ on ≥ 3 of 4 seeds | stated expectation | ✓ 3 of 4 (0.139 / 0.158 / 0.149 / 0.186 against 0.213 / 0.166 / 0.353 / 0.165 kcal/mol) |
| hexamer energy; held-out dimers | < 0.4 kcal/mol but M_∞; ≤ 0.2 | ✓ 0.18–0.33 (M_∞ 0.45); dimers 0.056–0.096 |
| raw 2-body | every rung 0.05–0.4 kcal/mol | ✓ 0.17–0.28 |
| raw 4-body (267) | 50–130 % | **✗ M_16S2†ᵐ 139 %** (126–159 by seed); the rest 63–110 % |

No §5 surprise: no read-out rung's far-field reading moved off its computed value or between seeds, M_16†ᵐ is its §3a
values, no `*pm` read-out rung reaches 40 % on the trimers beyond 9 Å, and every `*pm` read-out rung is below M_16†ᵐ on
the hexamers in the mean (on seed 3 alone M_16S0†ᵐ, 0.457 kcal/mol, is above M_16†ᵐ's 0.440). The three misses sit on
the same side: E₀'s many-body energy at short range.

Seed means (probes and trimers: share of the non-additive energy and of the label rms; hexamers, dimers and 2-body in
kcal/mol):

| rung | probes, non-additive 10 / 12 / 15 Å | trimers 7–9 / ≥ 9 Å | hexamers, non-additive | hexamer energy | dimers | 4-body | 2-body | 3-body, all / < 5 Å |
|---|---|---|---|---|---|---|---|---|
| M_∞ | 100 / 100 / 100 | 78 / 100 % | 0.224 (22.5 %) | 0.451 | 0.096 | 77 % | 0.171 | 18 / 17 % |
| M_16† | 100 / 100 / 100 | 97 / 100 % | 0.358 (36.0 %) | 0.332 | 0.089 | 96 % | 0.242 | 27 / 26 % |
| M_16†ᵐ | 100 / 100 / 100 | 101 / 100 % | 0.457 (45.8 %) | 0.305 | 0.076 | 95 % | 0.271 | 31 / 31 % |
| M_16C0† | 19 / 24 / 23 | 38 / 33 % | 0.335 (33.6 %) | 0.285 | 0.079 | 79 % | 0.205 | 25 / 24 % |
| M_16S0†ᵐ | 11 / 6 / 9 | 30 / 20 % | 0.311 (31.2 %) | 0.227 | 0.071 | 110 % | 0.269 | 22 / 22 % |
| M_16C0†ᵐ | 8 / 5 / 7 | 30 / 17 % | 0.236 (23.7 %) | 0.181 | 0.074 | 63 % | 0.277 | 25 / 25 % |
| M_16S2†ᵐ | 6 / 7 / 3 | 32 / 22 % | 0.230 (23.1 %) | 0.274 | 0.067 | 139 % | 0.241 | 29 / 29 % |
| M_16C2†ᵐ | 4 / 5 / 3 | 32 / 25 % | 0.158 (15.8 %) | 0.201 | 0.056 | 63 % | 0.229 | 25 / 25 % |
| the PES | — | 22 / 21 % | — | — | — | 9 % | 0.011 | 2 / 2 % |
| *E3-QP's read-outs* | *50–61* | *58–60 / 57–59 %* | *22–27 %* | *0.31–0.37* | *0.08–0.11* | *101–109 %* | *0.17–0.25* | *21–25 % / —* |

The C rungs' own truncations on the test probes at 10 / 12 / 15 Å (first order; through second; all orders):
M_16C0†ᵐ 16 / 14 / 17; 9 / 6 / 9; 8 / 5 / 7 %. M_16C2†ᵐ 15 / 15 / 14; 5 / 5 / 4; 4 / 5 / 3 %. M_16C0† 22 / 26 / 21;
18 / 24 / 21; 19 / 24 / 23 % (on q3 the full series is no better than its second order). The truth's: 9–11; 1.0–1.3; 0.

By bin, the raw sets: 2-body at 5–7 / 7–9 Å, every `*pm` rung 0.027–0.029 / 0.0074–0.0075 kcal/mol (the PES 0.0102 /
0.0074: exact beyond 7 Å), q3 0.044–0.048 / 0.0161, M_∞ 0.168 / 0.105; below 5 Å M_∞ 0.171, the `*pm` rungs
0.231–0.280. 3-body at 5–7 Å: the `*pm` read-outs 23–25 %, M_∞ 27, the daggers 34–38, the PES 3. 4-body below 5 / 5–7 /
7–9 Å: M_16C0†ᵐ 47 / 96 / 29 %, M_16C2†ᵐ 68 / 43 / 7, M_16S0†ᵐ 101 / 134 / 106, M_16S2†ᵐ 145 / 117 / 100, M_∞ 78 / 72 /
99, the PES 7 / 12 / 42.

## 2. What it says

- **The far field is solved, and it is computable.** Every far-field reading — the dimer daggers, the probes at
  10–15 Å for every rung, the C rungs' truncations — is the value computed before training, on every seed. The response
  matched where E₀ is blind and then held does not depend on E₀ or the seed, as the matching argument says. It leaves
  3–11 % of the probes' many-body induction (M_16C2†ᵐ 3–5 %; E3-QP 50–61 %), and on CCSD(T) data it was never
  fitted to — the trimers beyond 9 Å — it is at the PES's own error (17–25 % against 21 %; E3-QP 57–59 %).
- **The self-consistent response pays off through E₀.** With E₀ off (§3d of the pre-registration) the mutual fields
  were worth 1–3 points on the probes and nothing at order 0 on the hexamers. Trained, the C rungs beat the S rungs on
  every seed on the hexamers (C2†ᵐ 15.8 against S2†ᵐ 23.1 %; C0†ᵐ 23.7 against S0†ᵐ 31.2) and on the raw 4-body
  (63 against 110–139 %). A first-order response leaves the mutual induction — its α²T part of the 3-body energy and
  all of the 4-body — to E₀, which learns it on clusters of 3–5 molecules and carries it badly to hexamers and to the
  raw tetramers (the S rungs' 4-body below 5 Å: 101–145 %; C0†ᵐ 47). The C rungs are also the most seed-stable:
  M_16C0†ᵐ 20–25 % on the hexamers and 60–64 % on the 4-body over the seeds, M_∞ 17–35 and 45–103.
  **M_16C2†ᵐ is the best rung on the hexamers' non-additive energy and the probes**: 15.8 % (14–19 by seed), the only
  rung clearly below M_∞ (E3-QP's best, S2†, tied it: 22 against 23 %); probes 3–5 %; 4-body 63 %, tied with
  M_16C0†ᵐ, which has the best hexamer energy (0.18 kcal/mol against 0.20).
- **The short range did not move, and there the pins cost.** On the raw 2- and 3-body sets below 5 Å — most of their
  frames — every pinned rung is behind M_∞: 2-body 0.21–0.28 against 0.17 kcal/mol, 3-body 22–31 against 17 %, so M_∞ still reads
  the best 3-body overall (18 %; E3-QP: 17). The cost grows with the pinned charges: M_∞, q3, TTM3-F's give 0.171 /
  0.244 / 0.273 kcal/mol on the 2-body below 5 Å and 22.5 / 36 / 46 % on the hexamers' non-additive energy for the
  daggers. The read-outs recover part of it (S0†ᵐ 22 % on the close trimers against the dagger's 31). But on rigid,
  in-distribution dimers the pinned rungs are the better ones (§3.4): the deficit is on the raw, near-rigid frames.
- **The C rungs' mutual part is stronger than TTM3-F's.** Their first-order truncation leaves 14–17 % on the probes
  where the truth's first order leaves 9–11 %: the matching, on a signal 90 % first order, trades the first-order
  response for a mutual one about 1.5 times TTM3-F's. §3.2 shows the same on the 4-body.

## 3. Found after the run

**3.1 The pinned charges at contact** (`scratch/e3qp2_contact_smearing.py`, `scratch/e3qp2_contact_band_edge.py`).
TTM3-F smears its charge–charge interactions (Thole, a = 0.175, with the H and M factors of `ttm3f_mod.f90`); the
pinned Coulomb band, Σ q_i q_j erf(r/ℓ₁)/r, is bare at contact for ℓ₁ = 0.75 Å. On the raw dimers below 5 Å the
difference, TTM3-F's smeared electrostatics less the band's, is 0.45 kcal/mol rms (mean +0.25), 1.31 at O–O
2.4–2.7 Å and 0.39 at 2.7–3.0 Å — as large as the CCSD(T) 2-body correction itself (PES − TTM3-F: 0.54; 1.41; 0.42).
A steep term at the hydrogen bond, which E₀ must learn in every pinned rung and not in M_∞. By the band edge: ℓ₁ = 0.75
/ 1.0 / 1.25 / 1.5 / 2.0 Å give 0.45 / 0.30 / 0.35 (sign reversed) / 1.27 / 2.86 kcal/mol rms — smallest near 1.0 Å,
and large but smooth beyond 1.25 Å, where the band leaves the contact electrostatics to E₀ as M_∞ does. A hypothesis
for the short-range cost (the dispersion band at contact is not included), not yet a test.

**3.2 The far 4-body is one scan** (`scratch/e3qp2_fourbody_far.py`). 15 of the 17 tetramers at largest O–O 7–9 Å are
one geometry pulled out from 7.0 to 8.4 Å. There the PES's 4-body is TTM3-F's alone (its CCSD(T) 4-body correction is
switched off by 6.5 Å), and the CCSD(T) 4-body is 1.7 times it all along the scan (−0.0276 against −0.0160 kcal/mol at
7.03 Å): hence the PES's 42 %. TTM3-F alone is 43–45 % off the CCSD(T) 4-body below 7 Å too, where q-AQUA-pol's
correction repairs it (7–12 %). The S rungs, M_∞ and the daggers read 99–106 % there (no 4-body at all); M_16C0† 41 %
(TTM3-F's strength); M_16C0†ᵐ 29 and M_16C2†ᵐ 7 % — a mutual 4-body up to 1.7 times their truth's, which happens to be
CCSD(T)'s on this scan. Not accuracy: the over-strong mutual coupling of §2.

**3.3 Harness notes.** (1) The first-order rungs ran 2–4 Gauss–Newton steps (registered: one); the steps after the first
change nothing, and how many are taken varies by seed (acceptance at the fixed point, decided by rounding). (2) The
matched read-out is solved again for every seed though it is the same: cache it. (3) The order-2 rungs and M_16C0† chose
the largest ridge of the grid (1e-1): extend the grid upwards. (4) The dimer's tail-slope line is computed on a seed
spread at the double-precision floor (W ≈ 1e-16, so W/|f*| ∝ R³): suppress it there. (5) The E3-QP M_16 2-body
re-read: M_16† 0.195–0.266, M_16S0† 0.217–0.251 kcal/mol with the band (1.09–1.13 as read in E3-QP); 7–9 Å 0.0161
(q3's quadrupole; TTM3-F's charges 0.0074).

**3.4 The short range, read again** (20 Oct; `scratch/e3qp3_distortion.py`, `e3qp3_coverage.py`,
`e3qp3_flux_double_count.py`, all on the truth and the data, no model). Four facts change §2's third bullet:

1. *The raw frames are near-rigid, not rigid.* The truth at each frame less the truth at the same frame with every
   monomer replaced by the rigid training monomer (Kabsch-aligned): 0.090 kcal/mol rms on the 2-body set (strict
   0.080), 0.064 on the 3-body (strict 0.020). Every training monomer is exactly rigid, so a model exact on rigid
   monomers that ignored the distortions would read ~0.09 kcal/mol on the 2-body set — half of M_∞'s 0.17.
2. *On rigid, in-distribution dimers the pinned rungs are the better ones*: the held-out test dimers read 0.056–0.089
   kcal/mol for every pinned rung against M_∞'s 0.096, the hexamer energies 0.18–0.33 against 0.45. The deficit is on
   the raw frames only — not the contact mismatch of §3.1 at work in general.
3. *A reading inconsistency.* In training the pinned charges are the rigid monomer's, and E₀ learns the flux of the
   truth's charges from the internal forces — 72 % of the training forces' sum of squares is internal (stretch and
   bend; rms 8.1 against 5.1 kcal/mol/Å rigid-body, per molecule). The raw frames were read with the charges
   recomputed at each frame's own geometry (q3; TTM3-F's site charges), so the bands carry the flux's energy and E₀
   adds its learned flux response on top. The bands' part: 0.069 kcal/mol rms on the 2-body set with TTM3-F's charges,
   0.051 with q3, none for M_∞ — the order of the raw 2-body readings (0.23–0.28 / 0.20–0.24 / 0.17). It is pairwise,
   so it touches the 2-body reading only; the close trimers' deficit needs another cause.
4. *Coverage.* 29 % of the raw dimers below 5 Å are farther from every training pair (9 intermolecular distances,
   rms) than 95 % of the test dimers are: 57 % at O–O 2.3–2.7 Å, 67–72 % at 3.2–5 Å, 10–24 % in the hydrogen-bonded
   bulk at 2.7–3.2 Å. The raw trimers' O–O triangles below 7 Å are covered (0–2 %). Every rung meets this.

`scratch/e3qp3_shortrange_diagnostic.py` reads the saved models with each raw error split into its rigid
(intermolecular) part and its distortion response, the pinned rungs also with the rigid monomer's charges (the model
as trained), by O–O bin and by coverage; plus the training fit (energies; the forces' rigid-body and internal parts)
and the hexamers' 2-body error against the rest.

## 4. Next

- **The short range first** (20 Oct, Vadic: "we need to address the short range prediction"): the diagnostic on the
  saved E3-QP2 models (~30 min on one GPU, one seed per rung); then E3-QP3 by what it shows — a consistent pin
  convention if the double count is confirmed (the reading alone, no retraining); training monomers sampled within the
  raw frames' tolerance if the distortion response dominates; broader pair orientations (3.2–5 Å, compressed contacts)
  if the uncovered frames do; E₀'s width and the force loss's rigid-body/internal balance if the training fit is the
  limit. The band-edge scan of §3.1 waits on it.
- **M_16C2†ᵐ forward** as the response rung (M_16C0†ᵐ the cheaper one). The over-strong mutual coupling is the thing to
  watch in the ion part (MB-pol/MB-nrg), where the fields at contact are strong and the mutual share large; matching
  data with a larger mutual share would pin it.
- Then as planned: the ion part, rigid-body force projections, the K = 16 batch of E1/E2, the MACE baseline.
