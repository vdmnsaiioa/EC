# E3-QP2 — the response field and its higher orders on q-AQUA-pol: pre-registration

*16 Oct 2026, written before any training. The follow-up of E3-QP (`e3qp-results.md`), whose read-out carried the
far-field many-body induction at half strength (51–62 % of the probes' non-additive energy left at 10–15 Å) and whose
order-2 read-out was no better than order 0. Harness v0.0.11; `configs/water_qaqua_pol2.yaml` holds every value of the
experiment. The numbers below are computed from the PES and with the harness itself, before training:
`scratch/e3qp2_preregistration.py` (§3a–c), `scratch/e3qp2_matched_response.py` (§3d, the matched responses — which
do not depend on E₀ or the seed), `scratch/e3qp_decomposition.py` and `scratch/e3qp_response_floor.py` (the
decomposition in `e3qp-results.md` §3). q-AQUA-pol at 3a38dc4 as built for E3-QP. The local master of this note is
`notes/` in the repository.*

## 1. What limited E3-QP, and the three changes

`e3qp-results.md` §3 found four limits; each change below removes one.

1. **The field** (the "response field"). The read-out answers the band field of the pinned charges, and the
   dipole-surface charges q3 have TTM3-F's dipole but half its quadrupole: the probe's electrostatics with q3 misses
   by 13–24 % of the probe's interaction — as much as the non-additive signal — and a per-species polarisability on
   the q3 field cannot beat 13–28 % even when fitted on the probes. **Change: the truth's permanent charges exactly**
   — TTM3-F's H charges on the atoms and its M-site as a virtual site, (1 − γ) r_O + γ/2 (r_H1 + r_H2) with γ = 0.46
   read back from the library (`Rung.msites`, `QAquaPolTruth.site_model`; rungs `*pm`).
2. **The training signal.** The probes entered the loss with their total energies (a hundred times their interaction,
   all of it the core's), and the read-out's coefficients — functions of E₀'s features, trained with E₀ — were set by
   the clusters' short-range residual. **Change: the response matched on the far field.** The read-out's coefficients
   come from the species alone (`Rung.bf_species`: fixed features, independent of E₀ and of the seed); before E₀ is
   trained they are solved on the training probes E₀ cannot see (no probe atom within r_c of the core), labelled with
   the probe's interaction energy (`loss.far_labels: interaction`), the ridge chosen on every fifth core
   (`readout.ridge_selection: groups`); then held while E₀ learns the rest with the response in place
   (`readout.stage: far_first`, `lbfgs.freeze_readout`). Energies only: the truth's atomic forces on a rigid monomer
   carry the flux of its geometry-dependent charges, which pinned charges cannot represent (fitting them wrecks the
   response: a test solve on 41 probes left 116 and 169 % (orders 0, first order and self-consistent) with forces, 8 and
   10 % without).
3. **Second order and beyond.** TTM3-F's induction is self-consistent: on the probes its first-order part leaves 9–11 %
   of the non-additive energy, through α²T 1.0–1.3 %; on the hexamers the mutual part is 25 % of the many-body
   induction. **Change: the C rungs** let the order-0 read-out's induced dipoles (μ = C E, C the read-out's local
   polarisability tensor) act on each other through the band kernel's dipole tensor between molecules,
   μ = (1 − C T)⁻¹ C E, solved exactly (`Rung.bf_mutual = −1`; Gauss–Newton steps in the solve, the ridge chosen once
   and kept).
4. **The read-out's orders 1–2** (its field-gradient terms: the M-site's offset from the atoms, the response to the
   field's variation) were left to L-BFGS from zero. **Change: solved with the rest** (`readout.orders: all`).

Also: training probes on 48 cores of 3, 4 and 5 molecules (16 each, 2 probes per distance: 384, 320 of them beyond r_c),
so that the matched response sees the environments of tetramers and pentamers; every reading applies the rung's pins
(E3-QP's M_16 2-body readings lacked their dispersion band); the truth's energy is q-AQUA-pol's gradient routine (its
energy-only routine is off in the 3-body blend band, `e3qp-results.md` §3.2).

## 2. Design (`configs/water_qaqua_pol2.yaml`)

**(a) The dimer**: E3-QP's ladder stands; the exact daggers `M_1pm`, `M_16pm` alone, 4 seeds, E3-QP's schedule — a
harness reading.

**(c) The clusters**: E3-QP's 300 training clusters, 24 test hexamers and dimers, 80 test probes (all bit for bit:
the Metropolis relaxation is unchanged), and 384 training probes (above). Rungs:

| rung | sources | read-out (species coefficients, matched on the far field) |
|---|---|---|
| M_∞ | — | — |
| M_16† (`M_16p`) | q3 + α, C₆ (E3-QP's) | — |
| M_16†ᵐ (`M_16pm`) | TTM3-F's charges + α, C₆ | — (the exact dagger) |
| M_16C0† (`M_16C0p`) | q3 | order 0, self-consistent — the control for the field |
| M_16S0†ᵐ (`M_16S0pm`) | TTM3-F's | order 0, first order |
| M_16C0†ᵐ (`M_16C0pm`) | TTM3-F's | order 0, self-consistent |
| M_16S2†ᵐ (`M_16S2pm`) | TTM3-F's | orders 0–2, first order |
| M_16C2†ᵐ (`M_16C2pm`) | TTM3-F's | orders 0–2, the order-0 part self-consistent |

r_c = 6 Å, ℓ₁ = 0.75 Å, K = 4. Schedule: the read-out's solve (energies, every order's columns, 4 Gauss–Newton steps,
ridge from 1e-12…1e-1 on held-out cores) → Adam 2000 (batch 32, read-out frozen) → L-BFGS 500 (read-out frozen). The
training probes carry their interaction energies (weight 1 in Adam and L-BFGS: E₀ sees only the 8 Å ones). Readings as
E3-QP, plus, for the C rungs, the probes read with the model's own response truncated at first and second order.

## 3. Numbers before training

**(a) The exact daggers on the dimer tail** (relative error at the reading rows 7.8 / 10.3 / 11.9 / 14.7 / 19.5 / 29.8
/ 48.8 / 98.7 / 305 Å; the harness's own rungs, E₀ blind beyond r_c):

| | |
|---|---|
| M_1†ᵐ | 0.0241 / 0.0112 / 0.00758 / 0.00416 / 0.00184 / 0.000516 / 9.72e-5 / 1.49e-5 / 3.0e-5 |
| M_16†ᵐ | 0.0062 / 0.00279 / 0.00186 / 0.000997 / 0.000422 / 1.0e-4 / 3.4e-7 / 2.69e-5 / 3.05e-5 |
| the truth's induction (share) | 0.00623 / 0.00282 / 0.00189 / 0.00103 / 0.000453 / 0.000131 / 3.0e-5 / 3.7e-6 / 1.3e-7 |
| its O–O vdW term | 0.0179 / 0.00843 / 0.00572 / 0.00317 / 0.00142 / 0.000416 / 9.8e-5 / 1.2e-5 / 4e-7 |

M_16†ᵐ misses exactly the induction, M_1†ᵐ the induction and the vdW term; beyond 50 Å what is left is TTM3-F's
Coulomb constant (CHARGECON² over the atomic unit in the potential's own units: 0.99996937, 3.06e-5). E3-QP's daggers
missed 14–16 % at 7.8 Å.

**(b) The truth's energy.** The energy-only routine against the gradient routine: relaxed trimers 2 of 12 differ (up
to 0.028 kcal/mol), hexamers 10 of 12 (0.040). The PES's own error on the raw 3-body set: 0–5 / 5–7 / 7–9 / ≥ 9 Å
0.0142 / **0.0084** / 0.0060 / 0.0023 kcal/mol (E3-QP read 0.0097 in the 5–7 Å bin: 627 frames in the blend band); all
4673: 0.0120 (E3-QP 0.0123).

**(c) The training probes** (by core size: beyond r_c at 8 / 10 / 12 / 15 Å; non-additive rms):

| cores | beyond r_c | non-additive rms (kcal/mol) |
|---|---|---|
| trimers | 56 / 100 / 100 / 100 % | 0.020 / 0.011 / 0.0049 / 0.0028 |
| tetramers | 28 / 91 / 100 / 100 % | 0.028 / 0.014 / 0.0082 / 0.0028 |
| pentamers | 31 / 94 / 100 / 100 % | 0.031 / 0.013 / 0.0066 / 0.0045 |

**(d) The matched responses** — the read-out solved exactly as `T.fit` does, then read with E₀ switched off: the pinned
bands and the response alone. Test probes: the non-additive energy's rmse as a share of its rms at 8 / 10 / 12 / 15 Å
(and on the 13 of 20 at 8 Å beyond r_c); the CCSD(T) trimers at largest O–O 7–9 and ≥ 9 Å (the response's 3-body energy
against the labels); the hexamers: the response's many-body energy against TTM3-F's many-body induction (0.772
kcal/mol rms), and its whole energy against TTM3-F's induction (−2.10 kcal/mol on average):

| rung | ridge | probes, non-additive | 8 Å beyond r_c | interaction 10 / 12 / 15 | trimers 7–9 / ≥ 9 Å | hexamers: many-body / whole (mean) |
|---|---|---|---|---|---|---|
| M_16† | — | 100 / 100 / 100 / 100 | 100 | 13 / 13 / 19 | 100 / 100 | — |
| M_16†ᵐ | — | 100 / 100 / 100 / 100 | 100 | 10 / 9 / 15 | 100 / 100 | — |
| M_16C0† | 1e-1 | 25 / 19 / 24 / 23 | 41 | 12 / 10 / 10 | 38 / 33 | 64 % / −2.84 |
| M_16S0†ᵐ | 1e-4 | 9 / 11 / 6 / 9 | 13 | 1 / 1 / 1 | 27 / 20 | 23 % / −2.48 |
| M_16C0†ᵐ | 1e-4 | 8 / 8 / 5 / 7 | 12 | 1 / 0 / 1 | 29 / 17 | 27 % / −2.65 |
| M_16S2†ᵐ | 1e-1 | 4 / 6 / 7 / 3 | 5 | 1 / 1 / 0 | 25 / 22 | 25 % / −1.41 |
| M_16C2†ᵐ | 1e-1 | 4 / 4 / 5 / 3 | 7 | 0 / 0 / 0 | 26 / 25 | 21 % / −1.95 |
| *E3-QP: S0†, 16S0†* | | *56 / 52 / 52 / 50; 61 / 58 / 56 / 55* | | | *58 / 57; 60 / 59* | |
| *the PES against the labels* | | | | | *22 / 21* | |

The C rungs' own expansion on the probes (first order / through second order / all orders): M_16C0†ᵐ 16 / 9 / 8 %,
M_16C2†ᵐ 14–15 / 4–5 / 3–5 % — the truth's is 9–11 / 1.0–1.3 / 0. The test dimers (TTM3-F's induction −0.267
kcal/mol on average): the responses give −0.35 (order 0), −0.17 (S2), −0.22 (C2); E₀ trains next to that.

What the table says before the run: the field matters most (the same self-consistent response on q3 leaves 19–25 %, on
TTM3-F's charges 5–8 %); matching on the far field is what turns E3-QP's 51–62 % into 3–11 %; the read-out's orders 1–2
halve what order 0 leaves; the mutual fields help the order-0 and order-2 read-outs on the probes by 1–3 points and
the order-2 one on the hexamers (25 → 21 %), not the order-0 one (23 → 27 %). On the CCSD(T) trimers beyond 9 Å the
response alone is at the PES's own error (17–25 % against 21 %).

## 4. Predictions

**(a) Dimer**: M_1†ᵐ and M_16†ᵐ at their §3a values at every reading row to 1e-6 relative (or the double-precision
floor of E(all) − Σ E(f)); spread exactly 0.

**(c) Clusters** (means over 4 seeds; percentages as in §3d):

- **Probes at 10, 12 and 15 Å — computed**: every read-out rung's non-additive and interaction readings equal its §3d
  values to the printed digit on every seed (seed spread 0: the matched read-out does not depend on E₀ or the seed, and
  E₀ cancels in the probe's interaction beyond r_c — every test probe there is beyond it); likewise the C rungs' first-
  and second-order truncations; M_∞, M_16†, M_16†ᵐ 100 % (non-additive) and their §3d interactions.
- **Probes at 8 Å**: the 13 beyond r_c as computed; the 7 inside r_c carry the trained E₀ — the 8 Å non-additive reading
  within ± 6 points of §3d.
- **CCSD(T) trimers beyond 9 Å (108)**: E₀ is never wholly blind there (every frame has a pair within r_c), but its
  3-body leak was nil in E3-QP (M_∞ and the daggers at 100 %): every `*pm` read-out rung within ± 5 points of §3d
  (17–25 %) and **below 30 % on every seed** (E3-QP's read-outs 57–59 %); M_16C0† 28–38 %; M_∞, M_16†, M_16†ᵐ 95–105 %.
- **CCSD(T) trimers at 7–9 Å (311)**: the `*pm` read-out rungs 20–35 % (E3-QP 58–60 %; the PES 22 %).
- **Hexamers** (non-additive rms 0.997 kcal/mol): the response leaves 0.16–0.21 kcal/mol of the many-body induction and
  the CCSD(T) corrections' 0.274 are E₀'s: the `*pm` read-out rungs 12–25 %; M_∞ 18–30 % (E3-QP 23 %); M_16†ᵐ 25–45 %;
  M_16C0† 20–40 %. Open, with a stated expectation: M_16C2†ᵐ at or below M_∞ on at least 3 of 4 seeds. Hexamer
  energy rmse < 0.4 kcal/mol for every rung but M_∞; held-out dimers ≤ 0.2.
- **Raw 2-body**: every rung 0.05–0.4 kcal/mol (the M_16 rungs now read with their band); **4-body**: 50–130 %.

## 5. Surprises, and what is not tested

A surprise would be: a read-out rung's probe reading at 10–15 Å off its §3d value by more than a printed digit, or
differing between seeds (the matching or the cancellation is not what §1 says); M_16†ᵐ off its §3a values; a `*pm`
read-out rung at or above 40 % on the CCSD(T) trimers beyond 9 Å on any seed; a `*pm` read-out rung worse than M_16†ᵐ
on the hexamers (the response, held, hurting E₀).

Not tested: a response whose coefficients depend on the environment through E₀ (matched this way it would drift with
E₀'s training — the species read-out is what makes the far-field values computable); flexible monomers; the read-out's
own band edge (`model.response_edge_angstrom`, available: a fixed per-species response on TTM3-F's charges does a few
points better at 1.25 Å than at 0.75 Å, `e3qp-results.md` §3.4); training with rigid-body force projections (which
would keep the forces' information without the charge flux). Expected cost on one A100: ~0.5 h for the dimer part,
~4 h for the clusters.
