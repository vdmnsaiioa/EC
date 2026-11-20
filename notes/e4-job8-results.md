# E4, job 8: the pinned c6 band at contact — results

*Read 20 Nov 2026 against `e4-job8-preregistration.md`, registered at 17:17 CET on 19 Nov (f99ff87 on GitHub), before any
run. The job ran on Niflheim at v0.0.32. It ran as three tasks on 14-hour allocations, not eleven tasks: D′'s six
variants in two tasks of three runs, the dagger family's 32 runs in the third. The runs and seeds are the registered
ones. It was read with `scripts/e4_job8.py` and `scripts/e4_water.py summary`. Rigid forces in kcal/mol/Å on the unseen
frames (validation and test at 64 molecules, every split at 128 and 256), seed means over seeds 0–3 (0–1 for D′ without
the band). D′'s reference is its four seeds in job 5; M_∞'s is job 6's four seeds, 0.907 / 0.917 / 0.929.*

## 1. The registered readings

| | band | Niflheim | verdict |
|---|---|---|---|
| **K8.1** | the dagger at 2.0 Å within ±7 % of M_1pm | +5.4 / +5.5 / +5.1 % (0.584 against 0.554 at 64) | **holds** |
| **K8.2** | M_1pm_us within ±5 % of M_16pm_us | −14.3 / −13.9 / −14.0 % (0.481 against 0.561) | **fails, low** |
| **K8.3** | M_1pm_us within ±5 % of M_1pm | −13.2 / −12.7 / −12.8 % (0.481 against 0.554) | **fails, low** |
| **K8.4** | D′ at 2.0 Å at most +2 % above D′ | −11.6 / −12.0 / −11.8 % (0.478 against 0.541) | **holds** |
| **K8.5** | its ε_∞ within ±0.002; the far probes and T5 within ±0.5 points | ε_∞ 1.6893 against 1.6893; probes 2.23 / 1.98 / 2.66 %, T5 4.31 %, as D′'s | **holds** |

**The model rule fires.** K8.4 and K8.5 hold, so D′ with its c6 band at 2.0 Å becomes E4's model:
M_16C2pm_us_d20 on `configs/e4_job8d.yaml`, with D′'s read-out held from `results/e4_job5`.

- **K8.1 holds:** the band's cost was its contact.
- **K8.2 and K8.3 both fail low,** on a branch §6 did not list exactly. §6 had "both fail low: H2" and "only one holds:
  the two overlap in part". Both fail, but M_1pm_us lies between the hypotheses: 0.481, against H1's 0.56 and H2's
  0.43–0.44. The two causes overlap in part (§3).
- **K8.4 holds with a wide margin,** as §4 expected if K8.2 and K8.3 failed low.

## 2. The band's edge

The dagger (MB-pol's site charges and c6, no field term, no response) by the c6 band's edge:

| edge | forces 64 / 128 / 256 | over no band | against M_∞ |
|---|---|---|---|
| 0.75 Å (M_16pm, as before) | 0.722 / 0.736 / 0.739 | +30 % | −20 % |
| 1.5 Å | 0.657 / 0.670 / 0.671 | +18 % | −27 % |
| **2.0 Å** | **0.584 / 0.597 / 0.598** | **+5.4 %** | **−36 %** |
| 2.5 Å | 0.568 / 0.580 / 0.583 | +2.5 % | −37 % |
| no band (M_1pm) | 0.554 / 0.566 / 0.569 | — | −39 % |

- **The cost falls with the band's weight at contact.** The pre-registration's §8 gives the hydrogen bond's band force:
  4.1, 0.37, 0.06 and 0.01 kcal/mol/Å at 0.75, 1.5, 2.0 and 2.5 Å. The cost runs 30, 18, 5.4 and 2.5 %.
- **The residual at 2.0 Å is the O–O contact,** where the band keeps 31 % of C₆/r⁶ at 2.8 Å (0.14 kcal/mol/Å). At
  2.5 Å it keeps 13 % there, and the residual halves.
- **No edge reaches the band's absence.** Even at 2.5 Å, where the band keeps 93 % of the tail at 6 Å, the dagger is
  2.5 % above M_1pm. Without the field term, E₀ does not cancel what remains at contact. No gain from the tail shows in
  the forces.

## 3. One cause or two: they overlap in part

| | with the band at 0.75 Å | with it at 2.0 Å | without it |
|---|---|---|---|
| the dagger | 0.722 | 0.584 | 0.554 (M_1pm) |
| the dagger with the field term | 0.561 (M_16pm_us) | 0.483 (M_16pm_us_d20) | 0.481 (M_1pm_us) |
| the field term's gain | −22 % | −17 % | −13 % |

(forces at 64 molecules; 128 and 256 read alike)

- **The field term's gain shrinks as the band goes.** It is −22 % on the dagger with the band at 0.75 Å and −13 % with
  no band. So 55 % of what the field term bought in job 2 was repairing the band. The rest, −13 %, is its own.
- **The band's cost halves with the field term:** 0.168–0.170 kcal/mol/Å without it, 0.080–0.081 with it.
- **H1 and H2 both fail.** One cause predicted M_1pm_us near 0.56; two independent gains, 0.430 / 0.441 / 0.444. It
  reads 0.481 / 0.494 / 0.496.
- **With the field term, the damped band costs nothing.** M_16pm_us_d20 is +0.4 / +0.2 / +0.2 % above M_1pm_us. Without
  the field term the same band costs 5.4 %. So the field term lets E₀ cancel the 31 % left at the O–O contact.
- **The reading of H1's mechanism stands in part.** An H-bonded H sits in its partner's strong field, so the field term
  tells E₀ where the band over-binds. That accounts for half its gain. The other half is the field term's own: E₀
  reading the charges' field, which job 2 registered it for.

## 4. E4's model

| | forces 64 / 128 / 256 | against M_∞ | energies (kcal/mol per molecule) |
|---|---|---|---|
| D′ (job 5, band at 0.75 Å) | 0.541 / 0.551 / 0.552 | −40 % | 0.024 / 0.019 / 0.010 |
| **D′ with the band at 2.0 Å** | **0.478 / 0.485 / 0.487** | **−47 %** | **0.0215 / 0.0160 / 0.0081** |
| D′ without the band (seeds 0, 1) | 0.472 / 0.478 / 0.480 | −48 % | — |

- **The damped band takes 12 % off D′'s forces** at every size, and 10–19 % off its energies. Against job 1's frozen
  form (0.605 / 0.619 / 0.618) it is −21 %.
- **Its seeds agree far better:** spread 0.004–0.007 kcal/mol/Å, against D′'s 0.024–0.026. Its final training loss is
  18.85.
- **Its infrared is D′'s exactly.** The read-out is held, so ε_∞ before training, 1.689252397900301, is D′'s to every
  digit, and training does not move it. The far probes (2.23 / 1.98 / 2.66 % at 10 / 12 / 15 Å) and T5 (4.31 %) are
  D′'s.
- **Without the band, D′ reads 1.3–1.4 % lower,** on two seeds: within the noise, not resolved. It is −14.8 % against
  D′ on the same seeds.
- **Its far probes are the same without the band.** They read the non-additive error, and the c6 is pairwise additive.
  The tail's value is not in the bulk forces or the probes. It would be in the pressure, at another density, or in the
  energy's density dependence (the tail beyond 6 Å carries −0.19 kcal/mol per molecule, `e4-pressure-preregistration.md`
  §3). The model keeps it for those.

## 5. What the response buys in the bulk, now

| pair | with the band at 0.75 Å | at 2.0 Å | without the band |
|---|---|---|---|
| E4's model against the field-term dagger | −3.6 / −4.0 / −4.3 % (D′ against M_16pm_us) | **−1.0 / −2.0 / −2.0 %** (against M_16pm_us_d20) | −1.9 / −3.2 / −3.2 % (two seeds) |

- **With the band at contact, part of what the response bought was also repairing the band.** With it damped, the
  response buys 1–2 % in bulk forces.
- **The response's case now rests on the infrared:** ε_∞ (1.6893 against MB-pol's 1.6885, which a field-blind model
  cannot get right), T5 and the far probes. Bulk forces at one density do not probe those.

## 6. Reproduced

- **M_16pm_us matches job 2** to the printed digit: 0.561 / 0.574 / 0.577.
- **The dagger and M_1pm agree with job 7:** 0.722 against 0.721 at 64, and 0.554 against 0.560 (job 7 had two seeds;
  job 8 has four).
- **The held read-out's check passes:** ε_∞ on the read-out before training is D′'s 1.6893 to every printed digit.
- The run times were not read.

## 7. What this says

- **The c6 band as pinned since E3 over-bound the contact,** and that cost the dagger a fifth of its forces. Damped at
  2.0 Å it keeps MB-pol's tail beyond E₀'s reach (99.4 % at 6 Å) and costs 5 % without the field term, nothing with it.
- **E4's model improves by 12 %,** to 0.478 / 0.485 / 0.487, 47 % below M_∞, with the same infrared.
- **The field term's gain was half band repair.** Its own gain, −13 %, stands.
- **The response's bulk gain is now 1–2 %.** Its case rests on the infrared.

## 8. What it touches, not read

- **S1's MD** (`e4-s1-preregistration.md`) ran D′ and the dagger with the band at 0.75 Å. Their results stand for those
  models as registered; E4's model is now a different one. Whether S1 is run again with it is a decision, not made here.
- **E3's contact deficits.** E3's daggers carry the same band at 0.75 Å (`configs/water.yaml`). Part of what E3 read
  at contact may be this band. Not checked.
- **The comparison's second panel** (the pinned models, `e4-plan.md`): its dagger should carry the damped band
  (M_16pm_d20) or none (M_1pm), not M_16pm's. To be fixed when that panel is registered.
- **The pressure reader** (`e4-pressure-preregistration.md`) was never written. It is where the c6 tail should show.
