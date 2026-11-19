# E4, job 8: the pinned c6 band at contact — pre-registration

*19 Nov 2026, registered at 17:17 CET (this commit's time), before any run of job 8 and before any of its
numbers. Vadic's go of 19 Nov. Read against:*

- *`e4-job7-results.md` §3 (the band's cost; this job was proposed in §8 there);*
- *`e4-job2-results.md` (the field-term dagger M_16†ᵐ_us);*
- *`e4-job5-results.md` §2 (D′, E4's model).*

## 1. Why

**Job 7's finding.** MB-pol's site charges without the c6 band (M_1pm) read 0.560 kcal/mol/Å at 64 molecules. The
dagger, the same rung with the band, reads 0.721. The band costs 0.16, a fifth of the forces.

**The likely cause, untested: the band is undamped at contact.**

- At the run's band edge, 0.75 Å, the band is the full C₆/r⁶ from 2.5 Å out (§8).
- An O···H hydrogen bond at 1.9 Å carries −1.40 kcal/mol of band energy and 4.1 kcal/mol/Å of attraction. An O–O pair
  at 2.8 Å carries −0.49 kcal/mol and 1.05 kcal/mol/Å.
- MB-pol damps its own dispersion at contact, so E₀ must cancel the difference.

**A coincidence raises a second question.**

- Job 2's field-term dagger, M_16†ᵐ_us, is the dagger whose E₀ reads the pinned charges' field. It read 0.561 / 0.574 /
  0.577. M_1pm reads 0.560 / 0.572 / 0.575.
- So the field term and the band's removal each took the dagger from 0.721 to 0.56.
- **H1: one cause.** The field term gave E₀ what it needed to cancel the band at contact. An H-bonded H sits in its
  partner's strong field, which marks exactly the pairs the band over-binds.
- **H2: two causes, which add.** If each gain is 22 % and they are independent, the field-term dagger without the band
  would read about 0.721 × 0.78 × 0.78 ≈ 0.44.

**Why it matters for E4's model.** D′ has both the field term and the band. Under H1 its band costs it little. Under H2
it leaves accuracy on the table.

## 2. What changes (v0.0.32), fixed before any run

- **`Rung.disp_l`, the dispersion bands' own edge.** 0 is the band edge l₁, as before. The reciprocal grid is l₁'s, so
  disp_l is no narrower than l₁.
- **The rungs.**
  - The dagger with its band at 1.5, 2.0 and 2.5 Å: M_16pm_d15, M_16pm_d20, M_16pm_d25. Beside them are M_16pm
    (0.75 Å) and M_1pm (no band).
  - The field-term dagger M_16pm_us with its band at 2.0 Å (M_16pm_us_d20) and without it (M_1pm_us).
  - D′ with its band at 2.0 Å (M_16C2pm_us_d20) and without it (M_1C2pm_us).
- **A held read-out** (`training.readout.held_from`, `e4_water.held_readout`).
  - D′'s variants load the read-out matched for D′ in job 5 (`results/e4_job5`) and hold it. Nothing is matched, so a
    variant changes the band and nothing else.
  - The cache is checked against D′'s rung. It was keyed before the Rung fields added since v0.0.29; `rung_key`
    reproduces that key, which was checked against the v0.0.29 code.
  - **Why held, not matched again.** The read-out was matched on probes at 8–15 Å, where the band at 2.0 Å keeps at
    least 99.4 % of C₆/r⁶. Without the band, a fresh match would let the read-out absorb the missing tail.
- **The reader:** `scripts/e4_job8.py` reads K8.1–K8.5 and the model rule, without JAX.
- **Everything else as before.**
  - The dagger family runs on job 7's settings (`configs/e4_job8.yaml`).
  - D′'s variants run on job 5's settings (`configs/e4_job8d.yaml`), so the cache's tags are D′'s.
  - Four seeds, 0–3; two for D′ without the band.

## 3. The runs

`run_e4_job8.sbatch` has eleven tasks, each a list of runs:

- 0–1: the dagger at 0.75, 1.5, 2.0 and 2.5 Å, and M_1pm, four seeds each: about 2.1 h a task;
- 2–4: M_16pm_us, M_16pm_us_d20 and M_1pm_us, four seeds each, one rung a task: about 2 h;
- 5–8: D′ with its band at 2.0 Å, one seed a task: about 3.3 h;
- 9–10: D′ without the band, seeds 0 and 1: about 3 h.

Then:

```sh
python scripts/e4_job8.py --out_dir results/e4_job8 --reference_dir results/e4_job5
python scripts/e4_water.py summary --out_dir results/e4_job8
```

## 4. Registered

**The readings.** Rigid forces on the unseen frames (job 7's pooling), seed means over four seeds. D′'s reference is
its four seeds in job 5.

| | reading | band | prediction |
|---|---|---|---|
| **K8.1** | the dagger with its band at 2.0 Å against M_1pm | within ±7 % at every size | holds: the band's cost is its contact |
| **K8.2** | the field-term dagger without the band (M_1pm_us) against it with the band (M_16pm_us) | within ±5 % at every size | holds (H1) |
| **K8.3** | M_1pm_us against M_1pm | within ±5 % at every size | holds (H1) |
| **K8.4** | D′ with its band at 2.0 Å against D′ | at most +2 % at every size | holds |
| **K8.5** | its infrared against D′'s: ε_∞ on the held-out liquid; the far probes' non-additive error at 10, 12 and 15 Å; T5 | ε_∞ within ±0.002; the probes and T5 within ±0.5 points | holds |

**The model rule, registered with them.** If K8.4 and K8.5 hold, D′ with its band at 2.0 Å becomes E4's model. It keeps
MB-pol's c6 tail beyond E₀'s reach without asking E₀ to cancel the band at contact.

- **Why these bands.**
  - K8.1's 7 % is about a quarter of the band's cost (0.04 of 0.16). It is about two standard errors of a difference of
    four-seed means at M_1pm's seed spread (0.029).
  - K8.2's and K8.3's 5 % put the band between the hypotheses. H1 expects M_1pm_us near 0.56, and H2 near 0.44. 5 % is
    about two standard errors at the field-term dagger's spread (0.018 in job 2).
  - K8.4's +2 % lets noise pass under H1, where D′ with the damped band reads about D′. Under H2 it reads below.
  - K8.5's bands: the read-out is held, so ε_∞ and T5 are fixed by it, and the band beyond 6 Å changes by at most
    0.6 %. The far probes at 10–15 Å lie beyond E₀'s reach.
- **The predictions for K8.2 and K8.3 are the least certain.** They rest on the coincidence of §1 alone. If both fail
  low, H2 holds and K8.4 should hold with a margin.

## 5. Read beside, not registered

- **The dagger's forces against the band's edge,** 0.75 to 2.5 Å, and with no band: where the contact's cost lies, and
  where the tail's loss begins (2.5 Å keeps 92.6 % at 6 Å).
- **M_16pm_us against job 2's** 0.561 / 0.574 / 0.577: whether the rung reproduces.
- **D′ without the band** (two seeds) against D′ on the same seeds: its forces, and its far probes, which lose the c6
  tail.
- **D′'s variants paired by seed with job 5's D′.** They share E₀'s initialisation and batches.
- **The held read-out's check:** ε_∞ on the read-out before training, D′'s 1.6893 to the printed digit.
- **The energies and their bias.** The band's tail beyond 6 Å carries −0.19 kcal/mol per molecule
  (`e4-pressure-preregistration.md` §3).
- **The final training losses.**

## 6. What would change the reading

- **K8.1 fails high:** the damped dagger stays near 0.72. Then the band's cost is not its contact. Read the edge curve:
  - if 2.5 Å recovers and 2.0 Å does not, the cost lies at 2–3 Å;
  - if no edge recovers, the band itself is wrong for these data, for example counted twice with MB-pol's short-range
    terms.
- **K8.1 fails low:** the damped dagger is more than 7 % below M_1pm. The c6 tail helps once the contact is gone.
- **K8.2 and K8.3 fail low:** M_1pm_us well below 0.56. That is H2: the field term and the band's removal add, and E4's
  model should gain from the damped band.
- **Only one of K8.2 and K8.3 holds:** the two overlap in part. Read the numbers.
- **K8.4 fails:** the damped band costs D′. D′ stays E4's model with its band at 0.75 Å.
- **K8.5 fails:** the damped band moves the infrared. That is not expected with the read-out held. It would point at
  E₀ in the probes at 10 Å, or at the band's mid-range.

## 7. Cost

- **Tasks 0–4:** about 2 h each (job 7: the dagger 12–13 minutes a run; job 2: M_16pm_us 0.5 h).
- **Tasks 5–10:** about 3–3.5 h each (job 5: D′ 2:57–3:35).
- **In all:** about 30 GPU hours.

## 8. Numbers before any run

From the kernel, `kernels.f6_long`, and MB-pol's long-range c6 (C₆ = 17.22, 5.01 and 1.46 au for O–O, O–H and H–H); no
training. The pair energy and its force at a typical contact, by the band's edge:

| edge | O–O at 2.8 Å | O···H at 1.9 Å | H···H at 2.4 Å | share of C₆/r⁶ at 4.5 / 6 / 8 Å |
|---|---|---|---|---|
| 0.75 Å (now) | −0.49 kcal/mol, 1.05 kcal/mol/Å | −1.40, 4.09 | −0.11, 0.26 | 1.000 / 1.000 / 1.000 |
| 1.5 Å | −0.33, 0.49 | −0.32, 0.37 | −0.05, 0.07 | 0.994 / 1.000 / 1.000 |
| 2.0 Å | −0.15, 0.14 | −0.09, 0.06 | −0.02, 0.02 | 0.881 / 0.994 / 1.000 |
| 2.5 Å | −0.07, 0.04 | −0.03, 0.01 | −0.01, 0.00 | 0.628 / 0.926 / 0.998 |

- **At 2.0 Å** the hydrogen bond's band force falls from 4.1 to 0.06 kcal/mol/Å, and the tail beyond E₀'s reach stays
  at 99.4–100 %.
- **The forces in the table are attractive,** on each pair. A molecule makes about two such hydrogen bonds as a donor,
  against the dagger's rigid-force error of 0.72 kcal/mol/Å.
