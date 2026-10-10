# q-AQUA as the E3 truth — assessment of `github.com/jmbowma/q-AQUA`

*10 Oct 2026. Everything below was read from the repository itself (cloned into the session; MIT licence;
Yu et al., J. Phys. Chem. Lett. 2022, DOI 10.1021/acs.jpclett.2c00966). The paper is not reachable from
this session, so statements about the paper are from memory and marked so. Survey scripts:
`scratch/qaqua/survey.py`, `scratch/qaqua/trimer_far.py` (kept in the repository under `scratch/`).*

## What the repository contains

Three raw CCSD(T) n-body data sets (xyz, coordinates in Å, energies in hartree to 1e-8) and the fitted
potential with analytic gradients (`qAQUA_v1.zip`: Fortran, 2-b/3-b/4-b permutationally invariant
polynomials of ~130k generated lines, the Partridge–Schwenke monomer, the LTP2011 dipole moment surface
for the monomer charges; Makefile for ifort).

| set | n-body energy, level | configurations | O–O coverage | monomer geometries |
|---|---|---|---|---|
| `2b_data.xyz` | 2-b, CCSD(T)/CBS | 71,892 | 1.94–8.07 Å; 99 % within 7.94 Å, 8 configurations beyond 8 Å | strongly distorted (r_OH 0.57–2.9 Å, HOH 21–180°): a PES training set, not a thermal sample |
| `3b_data.xyz` | 3-b, BSSE-corrected CCSD(T)-F12a/aVTZ | 45,332 | largest O–O 2.3–12.0 Å; 2,118 beyond 8 Å | distorted, less so |
| `4b_data.xyz` | 4-b, CCSD(T)-F12/haTZ | 3,692 | largest O–O 3.3–9.0 Å | near-equilibrium |

Energies: 2-b rms 4.8 kcal/mol (median |E| 1.2; 40,188 configurations with O–O in 2.5–3.5 Å); 3-b rms 0.74
(median 0.32); 4-b rms 0.20 (median 0.03).

**The long range of the fitted potential (`src/pes_shell.f90`), which is what matters for us:**

- 2-b: the PIP fit for O–O < 6.5 Å; a quintic switch over 6.5–7.8 Å to **bare point-charge Coulomb** with
  geometry-dependent monomer charges from the LTP2011 DMS; beyond 7.8 Å point-charge Coulomb only — no
  induction, no dispersion. The tail is therefore dipole–dipole exactly (p = 3, no R⁻⁶ corrections), with
  known sources.
- 3-b: a short-range PIP (largest O–O ≤ 5 Å) and a long-range PIP (≥ 6 Å, fitted to 9.5 Å), switched over
  5–6 Å, and **switched to zero over 7–9 Å**.
- 4-b: switched to zero over 5.8–7.5 Å.

So q-AQUA v1 has no many-body energy beyond 9 Å and no two-body R⁻⁶ beyond 7.8 Å. (A polarisable version,
q-AQUA-pol, exists in the literature as far as I recall — not in this repository; to be checked.)

## The far-field three-body signal in their own raw data

For trimers with a hydrogen-bonded pair (closest O–O < 3.2 Å) and the third molecule at distance d from the
pair's midpoint, the raw CCSD(T)-F12 3-body energies give (kcal/mol):

| d (Å) | n | rms \|E₃b\| | median | fraction > 0.01 |
|---|---|---|---|---|
| 5–6 | 1401 | 0.087 | 0.059 | 0.91 |
| 6–7 | 578 | 0.053 | 0.035 | 0.86 |
| 7–8 | 322 | 0.031 | 0.022 | 0.74 |
| 8–9 | 486 | 0.018 | 0.012 | 0.57 |
| 9–10 | 75 | 0.014 | 0.009 | 0.45 |
| 10–12 | 10 | 0.006 | 0.006 | 0.00 |

rms ∝ d⁻⁴·⁰ on 6–12 Å (between the −3 of "pre-existing induced dipole × far field" and the −6 of the mutual
response, orientation-averaged). These are the real-water numbers for the quantity our far-field test reads
— the synthetic PW truth gave 0.076 / 0.050 / 0.032 / 0.018 kcal/mol at 6 / 7 / 8 / 10 Å, the same magnitudes —
and q-AQUA v1's PES sets exactly this to zero from 7–9 Å outward: it drops ~0.02–0.03 kcal/mol per trimer at
7–9 Å, a known approximation of the fit, quantified here from its own training data. The ~900 raw trimers at
7–10 Å are a CCSD(T)-F12 far-field three-body test set, with a label precision of roughly 0.01 kcal/mol (the
F12/BSSE floor; the file's 1e-8 E_h is formatting, not precision).

## Fit to the three E3 readings

- **(a) Dimer ladder, p⋆ = 3.** Yes, as the truth: an analytic, CCSD(T)-fitted potential with gradients,
  evaluable at every window node and out to 500 Å, with a tail that is point-charge electrostatics of known
  sources (the DMS charges) — the water analogue of E1 on Jäger's Ar₂. The identifying nodes (every
  intermolecular pair beyond r_c) fall in the PIP region (< 6.5 Å) and the switch (6.5–7.8 Å), both
  CCSD(T)-fitted or blended; the branch tests do not care. The dagger's sources exist: the truth's own
  LTP2011 charges (geometry-dependent), to be ported (`ltp2011_chg.f90` is small; the DMS routine `DIPS`
  is the bulk). The learned-dipole target is the DMS dipole (1.855 D at equilibrium).
- **(b) Matching.** The DMS dipole as above; and the 71,892 raw CCSD(T)/CBS dimers as a held-out CCSD(T)-level
  check of the model's two-body energies in 2–8 Å, at the ~1e-5 E_h precision of CBS extrapolation (the raw
  data constrain nothing beyond 8 Å: the PES tail is the model's).
- **(c) Clusters.** Train on PES-labelled clusters with forces (any size; cheap), test the model's non-additive
  energies against the **raw** 3-b (45k) and 4-b (3.7k) CCSD(T)-F12 sets — a reading free of the PES's own
  fitting error, directly on the quantity the cluster part measures. The near-field part (the hexamer-type
  non-additive energy, which the rehearsal showed is a statement about ℓ₁) is fully served.
- **The far-field many-body test is not readable against the v1 PES** (its truth is zero beyond 9 Å and
  switched over 7–9 Å). Options: the raw far trimers above as the test set (held out, 0.01 kcal/mol floor,
  the training-residual diagnostic alongside); q-AQUA-pol as the truth if obtainable (its long range is a
  polarisable model, so embedded induced dipoles — the dagger's inputs — are computable from it); or keep the
  far-field reading on the polarisable synthetic truth and report the two together.

## Practicalities and caveats

- Flexible monomers: the rehearsal's `sseft.water` samplers and pinned charges are rigid; with q-AQUA the
  sources are geometry-dependent (DMS) and the embedded-dagger inputs (polarisation work, dispersion
  coefficients) do not exist in v1 at all (no polarisation; dispersion is inside the PIP below 6.5 Å and
  absent beyond 7.8 Å) — the dagger for a v1 truth is M_1† with DMS charges, which is exact beyond 7.8 Å by
  construction, and that is a feature of the test, not a flaw.
- Precision: the PES is smooth to machine precision (the σ = 1e-9 E_h design applies as for E1); the raw
  labels are ~1e-5 E_h (2-b CBS) and ~1e-5 (F12 n-body), which sets the floor of every held-out reading.
- Compute: Fortran with an ifort Makefile (gfortran should do; not available in this container without
  installing); the PIP libraries take a while to compile; evaluation is milliseconds per cluster. A small
  driver (f2py or a subprocess reading xyz, writing energies and gradients) labels any data set we generate.
- Provenance of the 2-b set (which configurations, how sampled) is in the paper, not the README; the strongly
  distorted monomers say it is a PES-fitting set, and thermal sampling for our clusters is ours to do.
- Licence MIT; cite Yu et al. 2022.

## Recommendation

Adopt q-AQUA v1 as the E3 truth for the dimer ladder, the matching and the cluster part — an "E3 on a
published potential", the exact analogue of E1 — with the raw 2-b / 3-b / 4-b sets as the CCSD(T)-level
held-out checks, and the pre-registration numbers computed from the PES before training (the class minimum
of the charge scale on the identifying nodes, the DMS dipole, the band-edge dependence of the cluster
reading). Keep the far-field many-body reading on a polarisable truth (q-AQUA-pol if we can get it, otherwise
the synthetic one), and use the raw far trimers as the one CCSD(T)-level far-field check. Phase 2's own
CCSD(T)/SAPT points remain what the paper's E3 needs beyond 8 Å.
