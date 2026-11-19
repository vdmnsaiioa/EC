# E4, job 7: the scan of a field-blind charge scale, and the warm starts — results

*Read 19 Nov 2026 against `e4-job7-preregistration.md`, registered at 00:30 CET on 19 Nov, before any run. The job ran on
Niflheim at v0.0.31 (c63da14 on GitHub). It ran as four tasks of 5–6 runs each, not 22 tasks; the runs and
seeds are the registered ones. It was read with `scripts/e4_job7.py`, `scripts/e4_sumrules.py read --job7_dir` and
`scripts/e4_water.py summary`. Rigid forces in kcal/mol/Å on the unseen frames (validation and test at 64 molecules,
every split at 128 and 256), seed means over seeds 0 and 1; M_∞'s reference is job 6's four seeds.*

## 1. The registered readings

| | band | Niflheim | verdict |
|---|---|---|---|
| **J7.1** | M_1pm within ±7 % of the dagger at every size | −22.3 % at every size (0.560 against 0.721 at 64) | **fails, low** |
| **J7.2** | the scan's lowest at c ≥ 1.0 at every size | c = 1.1 at every size | **holds** |
| **J7.3** | the scan's lowest at least 10 % below M_∞ at every size | −34.7, −33.9, −34.1 % | **holds** |
| **J7.4** | the six warm-started dipoles within ±15 % of their mean; every final loss at or below M_∞'s largest (74.7) | 1.21–2.56 D, −36 % to +35 % of the mean (1.90 D); every loss passes (24.8–48.8) | **fails** |
| **J7.5** | the warm starts' mean at most 7 % above the scan's lowest | +16.8 to +16.9 % | **fails** |
| **J7.6** | \|b_T\| in 1.22–1.40; read when J7.4 holds | — | **not read** |

- **J7.1 fails in the direction §6 did not list.** §6 had "fails high": M_1pm above the dagger, the c6 carrying part
  of its gain. Instead M_1pm is far below it. Without the c6 band, MB-pol's pinned charges buy 38 % against M_∞; with
  it, 20 % (§3).
- **J7.4 fails on the registered branch** "J7.4 fails, with J7.3 holding": the data set a scale, and the learned
  charges do not reach it even from nearby. The optimisation is the obstacle (§4).
- **J7.5 fails** as a consequence of J7.4: the runs started at 0.5 and 1.5 did not arrive. The run started at 1.0 matches
  the scan's lowest (§4).

## 2. The scan: where the data put a field-blind scale

| c | forces 64 / 128 / 256 | final training loss |
|---|---|---|
| 0 (M_∞, job 6) | 0.907 / 0.917 / 0.929 | 63.8 |
| 0.4 | 0.826 / 0.835 / 0.847 | 52.5 |
| 0.7 | 0.740 / 0.748 / 0.759 | 43.1 |
| 1.0 | 0.631 / 0.640 / 0.643 | 31.1 |
| **1.1** | **0.592 / 0.606 / 0.612** | **27.5** |
| 1.45 | 0.714 / 0.755 / 0.767 | 40.3 |

- **The data constrain the scale strongly.** The forces fall by a third from c = 0 to c = 1.1 and rise again by 1.45.
  Job 6's "data" reading — cells that barely constrain the scale — is ruled out. Job 6's learned charges failed through
  the optimisation.
- **The minimum sits at the lemma's scale.**
  - c = 1.1 is lowest at every size and in the training objective.
  - It is below c = 1.0 by 1.4–1.7 standard errors at each size: suggestive, not resolved, as §4 of the
    pre-registration expected.
  - A parabola through the four points from 0.7 to 1.45 puts the minimum of the forces at c = 1.07–1.10, and of the
    training loss at 1.10. The lemma's scale is 1.10–1.11 (|b_T| = √ε_∞(k) through b_T × μ = 2.70 D).
- **Energies follow:** M_∞'s bias at 128 molecules (test, −0.016 kcal/mol per molecule) shrinks to −0.006 at c = 1.1.

## 3. The c6 band costs a fifth of the forces

- **The dagger reproduces job 1:** −20.5 / −19.6 / −20.3 % against M_∞ (job 1: −20.5 / −19.8 / −20.6 %).
- **Without its c6 band it is twice as good:** M_1pm reads 0.560 / 0.572 / 0.575, −38 % against M_∞. The band costs
  0.161 / 0.165 / 0.165 kcal/mol/Å at the three sizes, and the dagger's final training loss is 40.2 against M_1pm's 24.0.
- **The likely cause, not tested: the band is undamped at contact.**
  - The pinned c6 band uses the run's band edge, 0.75 Å. There the kernel is the full C₆/r⁶ to 0.1 % from 2.5 Å out.
  - An O–O pair at 2.8 Å carries −0.49 kcal/mol of band energy, with steep r⁻⁷ forces, which MB-pol's own damped
    dispersion does not have and E₀ must cancel.
  - That is E3's lesson for the pinned charges at contact (`e3-summary.md` §2), now for the c6.
  - The charges' band is also undamped at contact, yet buys 38 %: the charges carry the long range E₀ cannot, the c6
    band adds little beyond 6 Å to the forces of a uniform liquid.
- **E4's model carries the same band.** D′'s frozen form reads 0.541 / 0.551 / 0.552 (`e4-job5-results.md`). M_1pm,
  with no response and no c6, is within 3.5–4 % of it, and 7 % below job 1's frozen form (0.605). Part of what the
  response buys in the bulk may be spent cancelling the c6 band.

## 4. The warm starts

| start | dipole, start → end (D) | forces 64 / 128 / 256 | final loss |
|---|---|---|---|
| 0.5 | 0.93 → 1.21, 1.32 | 0.780 / 0.786 / 0.794 | 46.1, 48.8 |
| 1.0 | 1.86 → 1.93, 1.94 | 0.594 / 0.609 / 0.612 | 24.8, 30.1 |
| 1.5 | 2.80 → 2.46, 2.56 | 0.703 / 0.730 / 0.741 | 38.2, 39.8 |

- **Every start moved toward the scan's minimum, from both sides, and stopped short.** Toward the scan's 2.05 D (c =
  1.1) they covered 25–45 % of the distance. The end still depends on the start, so J7.4 fails.
- **Real data are slower than the synthetic check.** There, the same unit took 0.5 and 1.5 to 0.95 and 1.10 of the
  teacher's dipole in 600 steps (pre-registration §8).
- **Started at the monomer's dipole, learned charges match the best pinned O/H charges.** M_1w100 reads 0.594 /
  0.609 / 0.612 against M_1q110's 0.592 / 0.606 / 0.612, with the same final loss (27.4 against 27.5). Its dipole,
  1.93 D, is c = 1.04.
- **Beside them:**
  - |b_T| is 1.40–1.41 for the 1.0 starts, at L1.2's upper edge; 2.05–2.23 and 1.06–1.10 for the others. b_T × μ is
    2.71 D, as in job 6.
  - r is 1.8–2.0, the field-blind class.
  - The sign stayed physical on every run.
  - The far-field probes' non-additive error is 92–99.7 %: environment-dependent charges carry a few percent of it.
    Pinned charges read 100 %.

## 5. The O/H form against MB-pol's sites

- At the permanent dipole the O/H charges are 12 % above MB-pol's site charges (M_1q100 against M_1pm). Even at their
  best scale, c = 1.1, they are 5–6 % above.
- So MB-pol's M-site, its quadrupole, is worth 5–12 % in forces. Learned charges on the atoms alone, ours or LES's, cannot
  carry it; learned atomic dipoles or a virtual site could.

## 6. Reproduced, and the cost

- **Job 7's M_∞** matches job 6's seeds 0 and 1 to the printed digit at every size.
- **Runs took 8–14 minutes** (487–827 s), tasks 52–64 minutes: 3–5 times faster than estimated. A 4-hour task holds
  about 15 such runs.

## 7. What this says

- **The data set a field-blind scale, at the lemma's.** Forces and the training objective are lowest at c ≈ 1.07–1.10,
  against the lemma's 1.10–1.11.
- **Pinned charges are worth far more than job 1 showed:** −38 % against M_∞ without c6. The c6 band as pinned costs
  0.16 kcal/mol/Å, and E4's model carries it.
- **Learned charges reach the pinned O/H accuracy when started at the monomer's dipole,** but do not converge from a
  factor of two away within 3000 steps. The obstacle is the optimisation, not the data.
- **For the comparison:** a no-pins version exists in practice — learned charges warm-started at the monomer's dipole,
  −35 % against M_∞, with only a one-molecule property beyond energies and forces. Its scale is not shown to be the
  data's own.

## 8. Proposed next, not registered

- **Job 8, the c6 band at contact.** It bears on E4's model.
  - The dagger with its c6 band at its own edge of 2.0 Å: 31 % of C₆/r⁶ at an O–O contact of 2.8 Å, 99.4 % at 6 Å,
    100 % from 8 Å. The tail beyond E₀'s reach stays MB-pol's; the contact goes to E₀.
  - Beside it M_1pm and the dagger as they are.
  - D′ without its c6 band and with the damped one, against D′.
  - If the damped band recovers M_1pm's forces, E4's model takes it, and keeps the c6 tail for the pressure and the
    density.
- **The learned charges' optimisation** (J7.4's registered branch): the charge head's own learning rate, registered
  first. Less urgent: the start at the monomer's dipole already reaches the pinned accuracy.
