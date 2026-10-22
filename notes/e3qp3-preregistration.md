# E3-QP3 — the short range on q-AQUA-pol: pre-registration

*22 Oct 2026, written before any training. The follow-up of E3-QP2's short-range diagnostic (`e3qp2-results.md`
§3.4–3.5), after Vadic's "we need to address the short range prediction". Harness v0.0.12; `configs/water_qaqua_pol3.yaml`
holds every value of the experiment. The numbers of §3 are computed from the PES, the data and the harness before
training: `scratch/e3qp3_preregistration.py` (§3a), `scratch/e3qp2_matched_response.py` with the E3-QP3 rungs (§3b),
`scratch/e3qp2_contact_band_edge.py` (§3c) and the diagnostic of E3-QP2's saved models of 21 Oct (§3d). The local master
of this note is `notes/` in the repository.*

## 1. What limited E3-QP2 at short range, and the four changes

E3-QP2's diagnostic (seed 0 of every rung) found four things; each change answers one.

1. **The reading.** The raw frames are near-rigid (O–H within 0.03 Å, HOH within 5° of the training monomer) and the
   truth's interaction energy moves with that: 0.090 kcal/mol rms on the 2-body set. E3-QP2 read them with the pinned
   charges recomputed at each frame's geometry, while the models were trained with the rigid monomer's and E₀ learned
   the truth's charge flux from the internal forces: the flux counted twice (0.069 / 0.051 kcal/mol rms of
   electrostatics with TTM3-F's charges / q3). **Change: the pins as trained** (`truth.heldout.pin_charges: rigid`) — a
   reading only. On E3-QP2's models it lowers every pinned rung's raw 2-body reading by 0.02–0.04 kcal/mol and its
   distortion error from 0.070–0.102 to 0.027–0.046 (M_∞'s: 0.012).
2. **The contact.** Read so, every pinned rung beats M_∞ on the raw dimers at O–O ≥ 2.7 Å and loses below it: 56–82 %
   of their squared 2-body error sits in the 278 compressed dimers at O–O 2.3–2.7 Å (5.6 %), where E3-QP2's 300 relaxed training
   clusters hold 28 pairs (one below 2.5 Å) and never bring two atoms of different molecules nearer than 1.7 Å (the raw
   sets: 1.53 Å). **Change: contact data** — 600 rigid dimers and 200 rigid trimers at O–O 2.3–3.0 Å (drawn uniformly),
   orientations uniform, every pair under the raw sets' 2 kcal/mol cap, no two atoms of different molecules nearer than
   1.5 Å (`data.contact_*`, `W.contact_configurations`).
3. **E₀'s fit.** On its own training clusters E₀ is off by 0.17–0.26 kcal/mol, the size of the test errors: it
   underfits. At these errors the loss's per-atom energy term is ~250 times smaller than its force term — the energies
   are learned through the forces, 72 % of whose signal is internal (the monomers' stretch and bend), and every reading
   is an energy. **Change: E₀ fitted to the energies** — `loss.energy_weight` 1 → 100 (the two terms then comparable)
   and twice the training (Adam 4000, L-BFGS 1000).
4. **The pinned bands at contact.** At ℓ₁ = 0.75 Å the pinned Coulomb band is bare at a hydrogen bond while TTM3-F
   smears its charges (Thole): between them 1.31 kcal/mol rms of electrostatics on the raw dimers at O–O 2.4–2.7 Å, a
   steep term E₀ must learn there. At ℓ₁ = 1.0 Å it is 0.85; larger edges overshoot. **Change, as a factor: the _e1
   rungs** — M_16pm and M_16C2pm at ℓ₁ = 1.0 Å, the read-out's band edge held at 0.75 Å (`e3_water.RUNG_BAND_EDGE`,
   `model.response_edge_angstrom: 0.75`), so that their matched response is E3-QP2's.

## 2. Design (`configs/water_qaqua_pol3.yaml`)

The clusters part only: E3-QP2's 300 training clusters and 4 monomers, 384 training probes, 24 test hexamers and
dimers, 80 test probes and raw sets (all bit for bit), plus the 800 contact configurations — 1488 training structures.

| rung | the pinned bands' edge ℓ₁ | read-out |
|---|---|---|
| M_∞ | — | — |
| M_16†ᵐ (`M_16pm`) | 0.75 Å | — |
| M_16†ᵐ_e1 (`M_16pm_e1`) | 1.0 Å | — |
| M_16C2†ᵐ (`M_16C2pm`) | 0.75 Å | orders 0–2, the order-0 part self-consistent; its edge 0.75 Å |
| M_16C2†ᵐ_e1 (`M_16C2pm_e1`) | 1.0 Å | the same; its edge 0.75 Å |

r_c = 6 Å, F = 16, 4 seeds. Schedule: the read-out matched as in E3-QP2 (the interaction energies of the 320 training
probes beyond r_c, 4 Gauss–Newton steps, the ridge on held-out cores) → Adam 4000 (batch 32, read-out frozen) → L-BFGS
1000 (chunk 64, read-out frozen); loss: energies 100, forces 1 (per atom, as before), the probes' interaction labels at
weight 1. Readings: E3-QP2's, with the pins as trained and every held-out set also by the frame's smallest O–O (below
2.7 / 2.7–3.0 / beyond 3.0 Å); then `scratch/e3qp3_shortrange_diagnostic.py` on every model (`run_e3qp3.sbatch`): the
raw errors split into their rigid and distortion parts, by bin and by coverage, the fit on the training clusters and on
the contact configurations, the hexamers' 2-body part and the rest.

## 3. Numbers before training

**(a) The contact data** (`scratch/e3qp3_preregistration.py`): 1200 pairs, 277 below O–O 2.7 Å and 42 below 2.5 Å
(E3-QP2's clusters: 28 and 1 of 1900); the dimers' interaction energies −4.35 … +2.00 kcal/mol (rms 1.68), the
trimers' −7.08 … +3.42 (rms 2.54). The raw dimers farther from every training pair than 95 % of the test dimers are
(0.116 Å; 9 intermolecular atom distances, rms):

| O–O | raw dimers | beyond E3-QP2's training pairs | beyond E3-QP3's |
|---|---|---|---|
| 2.3–2.7 Å | 278 | 57 % | 11 % |
| 2.7–2.9 Å | 930 | 10 % | 2 % |
| 2.9–3.2 Å | 3002 | 24 % | 20 % |
| 3.2–5.0 Å | 638 | 71 % | 70 % |
| below 5 Å | 4848 | 29 % | 23 % |

**(b) The far field** (the read-out solved as `T.fit` does, read with E₀ switched off). Beyond r_c the _e1 rungs'
pinned bands are their base rungs' (erfc(6 Å / 1.0 Å) ≈ 2e-17) and their read-out is too, so their matched response and
far-field readings are E3-QP2's; the energy weight scales the solve's objective, not its minimum. Computed for the
_e1 rungs with the read-out's edge given explicitly (`scratch/e3qp2_matched_response.py ... M_16pm_e1,M_16C2pm_e1
--edge=0.75`, 9 min on two CPUs): every reading equals E3-QP2's base rung's to 3e-10 relative or better, the solve's
history to 1e-12 (the ridge 1e-1 at every step), so

| rung | test probes, non-additive 8 / 10 / 12 / 15 Å | interaction 8 / 10 / 12 / 15 Å | trimers 7–9 / ≥ 9 Å | through order 1; order 2 |
|---|---|---|---|---|
| M_16C2†ᵐ, M_16C2†ᵐ_e1 | 4 / 4 / 5 / 3 % (8 Å beyond r_c: 7 %) | 4 / 0 / 0 / 0 % | 26 / 25 % | 14 / 15 / 15 / 14; 5 / 5 / 5 / 4 % |
| M_16†ᵐ, M_16†ᵐ_e1 | 100 % at every distance | 9 / 10 / 9 / 15 % | 100 / 100 % | — |

with E₀ switched off (the trained models read the 8 Å probes inside r_c and the trimers with E₀; beyond r_c, at 10–15 Å,
E₀ cancels in the probe's interaction). The response's many-body energy on the hexamers leaves 21 % of TTM3-F's
many-body induction. The objective values the run prints differ from these by the energy weight and the data's energy
scale; the ridge and the readings do not.

**(c) The pinned bands at contact** (`e3qp2-results.md` §3.1): TTM3-F's smeared electrostatics less the pinned Coulomb
band on the raw dimers below 5 Å, rms: 0.45 kcal/mol at ℓ₁ = 0.75 Å, 0.30 at 1.0 Å; at O–O 2.4–2.7 Å 1.31 and 0.85; at
2.7–3.0 Å 0.39 and 0.26.

**(d) The baselines**: E3-QP2's models, seed 0, read with the pins as trained (the diagnostic of 21 Oct), kcal/mol:

| | M_∞ | M_16†ᵐ | M_16C2†ᵐ |
|---|---|---|---|
| raw 2-body, all (4937) | 0.186 | 0.258 | 0.204 |
| O–O 2.3–2.7 Å (278) | 0.407 | 0.946 | 0.708 |
| from 2.7 Å (4658) | 0.159 | 0.128 | 0.119 |
| raw 3-body, all (4673) | 0.131 | 0.218 | 0.164 |
| largest O–O below 5 Å (2739) | 0.155 | 0.267 | 0.206 |
| raw 4-body, all (267) | 0.059 | 0.146 | 0.089 |
| hexamers: energy (its 2-body part; the rest) | 0.518 (0.461; 0.213) | 0.328 (0.206; 0.421) | 0.185 (0.171; 0.139) |
| test dimers | 0.101 | 0.081 | 0.062 |
| training clusters: energy | 0.258 | 0.204 | 0.175 |

The rows by O–O are the diagnostic's bins; one raw dimer lies below 2.3 Å, in neither (the harness's new bin below
2.7 Å holds 279 frames, the diagnostic prints both). Over E3-QP2's four seeds (the pins at the frames' geometry): raw 2-body M_∞ 0.171, M_16†ᵐ 0.271, M_16C2†ᵐ 0.229 kcal/mol;
hexamer energies 0.451 / 0.305 / 0.201 (non-additive 22.5 / 45.8 / 15.8 %); raw 4-body 77 / 95 / 63 %.

## 4. Predictions

**Computed** (to the printed digit, on every seed, seed spread 0): the far-field readings of §3b for every rung at
10–15 Å and the C rungs' truncations there; each _e1 rung's equal to its base rung's.

**Expected** (seed means unless said; kcal/mol):

- *E₀'s fit*: the training clusters' energy error ≤ 0.10 for the pinned rungs and ≤ 0.15 for M_∞ (whose clusters carry
  electrostatics beyond its cutoff), on every seed; the contact configurations' ≤ 0.15; the training forces' rigid-body
  part ≤ 1.0 kcal/mol/Å per molecule.
- *The compressed dimers* (raw 2-body, O–O 2.3–2.7 Å; the same for the harness's bin below 2.7 Å): every rung ≤ 0.30
  (E3-QP2: 0.41–0.95), the pinned rungs within 0.05 of M_∞ or below.
- *The rest of the raw dimers* (from 2.7 Å): the pinned rungs ≤ 0.10, M_∞ ≤ 0.13; the pinned rungs below M_∞ on
  every seed.
- *Raw 2-body, all*: every rung ≤ 0.15; the pinned rungs below M_∞ on at least 3 of 4 seeds.
- *Raw 3-body*: every rung ≤ 0.12 overall and ≤ 0.14 on the compact trimers (largest O–O below 5 Å). Open, with a
  stated expectation: the pinned rungs at or below M_∞ on the compact trimers (E3-QP2: 0.21–0.27 against 0.16).
- *Hexamers*: energy ≤ 0.15 for the pinned rungs (M_16C2†ᵐ ≤ 0.12), M_∞ 0.35–0.55 (its 2-body part beyond the cutoff
  stays); non-additive M_16C2†ᵐ ≤ 16 %, M_16†ᵐ ≤ 40 %, M_∞ ≤ 25 %.
- *Test dimers*: the pinned rungs ≤ 0.05, M_∞ ≤ 0.08.
- *Raw 4-body*: M_16C2†ᵐ ≤ 63 %.
- *The far field beyond the computed readings*: the probes at 8 Å within ± 6 points of E3-QP2's; the CCSD(T) trimers
  beyond 9 Å M_16C2†ᵐ 25 % ± 3, M_∞ and the daggers 95–105 %.
- *The factor* (open, stated expectation): each _e1 rung at or below its base rung on the compressed dimers on at least
  3 of 4 seeds, and within 0.01 of it from O–O 3.0 Å.

## 5. Surprises, and what is not tested

A surprise would be: a far-field reading at 10–15 Å off its computed value, or differing between seeds or between an
_e1 rung and its base rung; the training clusters' energy error above 0.15 for every pinned rung on every seed (E₀'s fit
not the lever — then its width); the compressed dimers' error above 0.4 for a pinned rung (the contact data not the
lever); a pinned rung worse than M_∞ on the raw dimers from 2.7 Å.

Not tested: the contact data's and the energy weight's separate effects (one run; the bins and the training fit tell
them apart in part — the compressed dimers are mostly the data's, the training fit the loss's); E₀'s width (16
features); flexible monomers (the distortion response learned from energies); TTM3-F's Thole smearing in the pinned
bands; a loss on the forces' rigid-body part (the alternative to the energy weight; the diagnostic reads that part);
the raw dimers' uncovered orientations at O–O 3.2–5 Å (70 %, where the pinned rungs
already beat M_∞). Expected cost on one A100: ~8 h for the clusters part, ~40 min for the diagnostic (longer on a 3090).
