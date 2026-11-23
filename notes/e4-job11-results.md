# E4, job 11: the long polish — the profile under it, the low side continued, and E4's model — results

*Read 23 Nov 2026 against `e4-job11-preregistration.md`, registered at 14:23 CET on 22 Nov, before any run. The job
ran on Niflheim at v0.0.35 as the registered five tasks. It was read with `scripts/e4_job11.py` and
`scripts/e4_water.py summary`.*

- *The learned dipole is read on the test frames of 64 molecules. Rigid forces in kcal/mol/Å are read on the unseen
  frames, seed means over seeds 0 and 1.*
- *The E4-class models are read against job 8's runs of the same seeds.*
- *Run times:*
  - *the M_1 polishes, 1.7–2.1 h a run, as in job 10;*
  - *the field-term dagger, 3.2–3.4 h for 1000 steps (estimated 4–5 h);*
  - *D′, 10.3 h each: its 600-minute budget stopped the polish after 277 and 290 steps (estimated about 350), at
    124–130 s a step.*

## 1. The registered readings

| | band | Niflheim | verdict |
|---|---|---|---|
| **K11.1** | the polished profile's minimum within ±0.05 of 1.098 | c*_lb = 1.070 (1.995 D) | **holds** |
| **K11.2** | job 10's two M_1w050 runs move up at least +0.06 D | +0.105 and −0.059 D | **fails** (predicted) |
| **K11.3** | the field-term dagger at 2.0 Å, polished, at least 5 % lower | −8.2 / −8.0 / −7.8 % (1000 steps) | **holds** |
| **K11.4** | D′ at 2.0 Å, polished, at least 3 % lower | −2.7 / −3.1 / −3.1 % (277 and 290 steps) | **fails**, at 64 molecules by 0.3 points |
| **K11.5** | D′'s infrared: ε_∞ ±0.002; the far probes and T5 ±0.5 points | ε_∞ 1.6893; probes 2.23 / 1.98 / 2.66 %; T5 4.31 %: job 8's to the printed digit | **holds** |

**The model rule does not fire.** K11.4 fails, narrowly. E4's model stays D′ at the 2.0 Å band as job 8 trained it.

## 2. The profile under the long polish

| c | 0.7 | 1.0 | 1.1 (job 10) | 1.45 |
|---|---|---|---|---|
| final loss, job 7 → polished | 43.1 → 30.75 | 31.1 → 23.63 | 27.5 → 21.45 | 40.3 → 31.44 |
| forces at 64, job 7 → polished | 0.740 → 0.622 | 0.631 → 0.546 | 0.592 → 0.520 | 0.714 → 0.624 |

- **The minimum moved a little lower: c*_lb = 1.070** (1.995 D), against job 7's 1.098 (2.047 D).
  - The polished forces' parabolas put it at 1.04–1.08 at the three sizes.
  - The lemma's scale is 1.10–1.11, so the polished minimum sits 3–4 % below it.
- **The polish lowers every point by 22–29 % in the objective and 12–16 % in forces.** The profile keeps its shape.
- **Job 10 re-read against c*_lb:**
  - its warm starts at c* end −3.4 % and +0.8 % from the new target;
  - from above, +5.7 % and +9.5 %;
  - from below, −24.6 % and −17.4 %;
  - the six's mean 1.897 D is −4.9 % (−7.3 % against job 7's target).

## 3. The low side has stopped; so has the high side

The learned dipole on the traced frames (D) through job 11's polish, from job 10's ends:

| run | 0 | 250 | 500 | 750 | 1000 | against c*_lb (1.995 D) |
|---|---|---|---|---|---|---|
| M_1w050 seed 0 | 1.503 | 1.544 | 1.581 | 1.596 | 1.608 | −19.3 % |
| M_1w050 seed 1 | 1.648 | 1.628 | 1.617 | 1.603 | 1.588 | −20.3 % |
| M_1w150 seed 0 | 2.108 | 2.091 | 2.087 | 2.091 | 2.113 | +5.9 % |
| M_1w150 seed 1 | 2.184 | 2.183 | 2.186 | 2.185 | 2.180 | +9.3 % |

- **The two runs from below now sit at the same scale, about 20 % short.** Seed 0 rose by 0.105 D and seed 1 fell by
  0.059 D, to 1.59–1.61 D (c ≈ 0.86).
- **The two runs from above did not move:** 2.11 and 2.18 D (c ≈ 1.13–1.17), 6–9 % above c*_lb.
- **The objective kept falling** in all four (−13 to −27 %). The gains came from E₀, not from the charges' scale.
- **The learned charges' ends depend on the start.** Across jobs 10 and 11 they end at 1.59–1.61 D from 0.5, 1.93–2.01
  D from 1.0 and 2.11–2.18 D from 1.5.
- **This is H_M (job 10's pre-registration), minima of their own, on both sides.**
- **The learned charges' environment dependence flattens their own profile.**
  - From below, polished twice, they read 0.552 / 0.558 / 0.564. The pinned O/H charges at that scale (c ≈ 0.86) would
    read about 0.58 by the polished scan's curve; at c = 1.0 they read 0.546.
  - From above they read 0.518 / 0.537 / 0.542, level with the pinned charges at c = 1.1 (0.520 / 0.536 / 0.541).
  - Started at c*, polished once (job 10), they read 0.491 / 0.507 / 0.510.
- **So with E₀ trained out, bulk forces at one density no longer fix the learned charges' scale.** Ends from 1.59 to
  2.18 D read within 0.49–0.56 kcal/mol/Å, and the start decides where they end.

## 4. The long polish for the E4-class models

| | job 8 (same seeds) | polished | change | steps |
|---|---|---|---|---|
| the field-term dagger at 2.0 Å (M_16pm_us_d20) | 0.479 / 0.493 / 0.494 | **0.440 / 0.453 / 0.456** | −8.2 / −8.0 / −7.8 % | 1000 |
| D′ at 2.0 Å (E4's model) | 0.474 / 0.478 / 0.480 | **0.461 / 0.464 / 0.466** | −2.7 / −3.1 / −3.1 % | 277, 290 |

- **The polish helps the E4-class model too: −8 % on the field-term dagger** in 1000 steps, without over-fitting (its
  training frames at 64: 0.430, the unseen 0.440).
- **D′ got only 28 % of the steps, and its objective was falling at the dagger's pace.**
  - The objective fell 6.4 % and 5.9 % in D′'s first 250 steps, against the dagger's 5.9 % and 4.8 %.
  - The dagger's then reached −16 % at 1000 steps.
  - So D′ is not nearer convergence; it ran out of budget. That is the branch "K11.3 holds and K11.4 fails" with the
    objective path read: the steps were not enough.
- **D′'s infrared is untouched:** ε_∞, the far probes and T5 are job 8's to the printed digit.
- **The response's bulk gain now depends on the polish's depth.**
  - The polished dagger (1000 steps) reads 2–5 % below the polished D′ (277–290 steps): +4.8 / +2.4 / +2.2 %.
  - Before the polish, D′ read 1–2 % below the dagger (job 8).
  - The comparison needs equal steps.
- **Energies:** the polished D′ reads 0.020 / 0.015 / 0.008 kcal/mol per molecule, job 8's 0.022 / 0.016 / 0.008.

## 5. What this says

- **The data's field-blind scale under the long polish is c = 1.07 (2.00 D),** 3–4 % below the lemma's. Job 7's 1.10 was
  the short polish's.
- **Learned charges do not find it from far away.**
  - Adam stops them early. Full-batch L-BFGS moves them much further, then they stop: about 20 % short from below and
    6–9 % above from above.
  - Started at the monomer's dipole they stay within 3.4 % of c*_lb.
  - The thread's conclusion: with a flexible charge head and E₀ trained out, the learned scale is set by the start, not
    by bulk energies and forces at one density.
- **The lemma cannot be read through learned charges trained this way.** Their |b_T| is still 2.71 D over their dipole,
  so it reads the start.
- **E4's model is under-trained by about the dagger's 8 %.**
  - The polish leaves the infrared alone.
  - Its cost is the obstacle: D′'s steps carry the mutual response, 125–130 s each.
- **But D′'s response does not depend on what the polish trains.**
  - D′'s read-out reads the band field with species coefficients (Rung.bf_species): no E₀ features. It is held.
  - Its sources are pinned.
  - So the response's energy and forces on each frame are constants of the polish.
  - The polish trains E₀ and the field term alone, exactly the field-term dagger's parameters.
  - With the response precomputed per training frame, D′'s polish would cost the dagger's 12 s a step, not 127 s,
    and follow the same objective.

## 6. Notes to the pre-registration

- **D′'s budget bought 277 and 290 steps, not the 350 estimated:** its steps took 124–130 s. K11.4's verdict stands.
  §5 reads why it failed.
- **The field-term dagger's 1000 steps took 3.2–3.4 h,** under its 300-minute budget.

## 7. Proposed next, not registered

- **(a) D′'s polish at the dagger's cost.**
  - The response's energy and forces are precomputed on the training frames, and E₀ and the field term are polished
    against the remainder.
  - First, a test that the difference between D′ and the field-term dagger with shared parameters does not change when
    those parameters do.
  - Then 1000 steps from job 8, as the dagger had, with the model rule registered again: about 3.5 h a run.
- **(b) The protocol from here on:** job 8's training followed by 1000 full-batch L-BFGS steps, for every model that
  enters the comparison, the baselines included.
- **(c) The learned charges' thread:** close it as above, or try the shallower valley (the charges' band with its own
  edge, job 9's option (b)) as the last attempt.
