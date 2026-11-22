# E4, job 10: following the valley — job 7's end states polished with L-BFGS — results

*Read 22 Nov 2026 against `e4-job10-preregistration.md`, registered at 19:05 CET on 21 Nov, before any run. The job ran
on Niflheim at v0.0.34 as the registered two tasks of four runs.*

- *Runs took 6188–7459 s (1.7–2.1 h), against the estimated 1.5–2 h. Tasks took 7.4–7.6 h, inside the 14 h asked.*
- *It was read with `scripts/e4_job10.py`, `scripts/e4_sumrules.py read --job10_dir` and `scripts/e4_water.py
  summary`.*
- *The learned dipole is read on the test frames of 64 molecules. Rigid forces in kcal/mol/Å are read on the unseen
  frames, seed means over seeds 0 and 1.*
- *The target is job 7's profile minimum, c* = 1.098 (2.047 D).*

## 1. The registered readings

| | band | Niflheim | verdict |
|---|---|---|---|
| **K10.1** | each off-scale start covers at least 25 % of its way to 2.047 D | 35, 45 % (from 0.5); 85, 73 % (from 1.5) | **holds** |
| **K10.2** | the six polished warm starts within ±15 % of their mean; the gate | 1.504–2.184 D: −20.7, −13.1, +1.6, +6.0, +11.1, +15.1 %; every loss passes | **fails** |
| **K10.3** | their mean within ±10 % of 2.047 D | 1.897 D, −7.3 % | **not read** |
| **K10.4** | their mean forces at most +7 % above the polished control's | +6.6 / +5.8 / +5.7 % | **holds** |
| **K10.5** | \|b_T\| in 1.22–1.40 on 4 of 6 | 3 of 6 (M_1w100 seed 1, M_1w150 seeds 0 and 1) | **not read** |

- **The registered branch is "K10.1 holds and K10.2 fails: moving, not arrived".** §3 reads the pace from the traces,
  as that branch asks.
- **K10.2 fails narrowly, at both ends.** M_1w050 seed 0 is 20.7 % below the six runs' mean and M_1w150 seed 1 15.1 %
  above it.
- **The mean would sit within K10.3's band** (−7.3 %), but K10.3 is read only when K10.2 holds.

## 2. Where the polish took the charges

| run | dipole before → after (D) | share of the way to 2.047 D | final loss before → after |
|---|---|---|---|
| M_1w050 seed 0 | 1.214 → 1.504 | 35 % | 46.1 → 31.6 |
| M_1w050 seed 1 | 1.315 → 1.648 | 45 % | 48.8 → 29.1 |
| M_1w100 seed 0 | 1.927 → 1.927 | — | 24.8 → 18.0 |
| M_1w100 seed 1 | 1.941 → 2.010 | — | 30.1 → 19.7 |
| M_1w150 seed 0 | 2.461 → 2.108 | 85 % | 38.2 → 23.5 |
| M_1w150 seed 1 | 2.562 → 2.184 | 73 % | 39.8 → 26.3 |

- **Full-batch L-BFGS moved the charges far more than Adam did.**
  - In 1000 steps the off-scale starts covered 35–85 % of the way.
  - Adam's 3000 steps had covered 25–45 % (job 7), and job 9's 50 polish steps 2–9 %.
- **From above, the charges arrived.** M_1w150 ends 3 % and 7 % above c*, where the profile's slope is small (+7 and
  +16 per unit of scale).
- **From below, they did not.** M_1w050 ends 20–27 % below c*, where the profile still pulls with −43 and −35 per unit
  of scale.
- **The runs started at c* stay near it.** M_1w100 seed 0 does not move. Seed 1 rises to 2.046 D by step 500 and falls
  back to 2.010 D.

## 3. The pace: fast, then slowing short of c* from below

The learned dipole on the traced frames (D) every 250 L-BFGS steps:

| run | 0 | 250 | 500 | 750 | 1000 | in the last 250 steps, against what remains |
|---|---|---|---|---|---|---|
| M_1w050 seed 0 | 1.213 | 1.361 | 1.451 | 1.463 | 1.503 | +0.040 of 0.544 (7 %) |
| M_1w050 seed 1 | 1.315 | 1.469 | 1.581 | 1.633 | 1.648 | +0.015 of 0.399 (4 %) |
| M_1w150 seed 0 | 2.461 | 2.221 | 2.130 | 2.123 | 2.108 | −0.015 of 0.061 |
| M_1w150 seed 1 | 2.563 | 2.396 | 2.247 | 2.207 | 2.184 | −0.023 of 0.137 |

- **Most of the movement came early.** The second half of the polish carried 6–20 % of each run's change.
- **From below, the pace fell to 4–7 % of the remaining distance per 250 steps.** At that pace, a slowing one, the rest
  would take several thousand steps if it came at all.
- **The objective kept falling all the way.** It fell 5–7 % in the last 250 steps of each traced run, so the polish had
  not converged. Its late gains came from E₀, not from the charges' scale.
- **The valley is as steep as before.** With the rest held, the curvature along the charges' scale is 11,000–22,000
  times the profile's at the end, as at the start and in job 9.
- **So the obstacle was the optimiser in large part, but not in full.**
  - A curvature-aware, full-batch optimiser does what Adam could not.
  - From below it slows toward ends that still depend on the start.
  - The traces cannot tell a very slow approach from a stop short of c*: minima of the runs' own, H_M's reading, on the
    low side only.

## 4. The control: the long polish alone takes 12 % off the forces

| | forces 64 / 128 / 256, before → after | training frames at 64 | final loss |
|---|---|---|---|
| M_1q110 (pinned, c = 1.1) | 0.592 / 0.606 / 0.612 → 0.520 / 0.536 / 0.541 (−12 %) | 0.571 / 0.581 → 0.505 / 0.515 (−11 %) | 27.5 → 21.5 (−22 %) |
| M_1w100 (learned, near c*) | 0.594 / 0.609 / 0.612 → 0.491 / 0.507 / 0.510 (−17 %) | 0.547 / 0.603 → 0.467 / 0.489 | 27.4 → 18.9 |

- **Job 7's training left 12 % of the forces on the table,** with the charges pinned and nothing to find.
- **It does not over-fit.** The training frames and the unseen ones fell by the same 11–13 %.
- **Learned charges near c* now beat the pinned O/H charges at c = 1.1 by 5.4–5.7 %.** After job 7's training the two
  were level (0.594 against 0.592). Their environment dependence is worth about 6 % once E₀ is trained out.
- **Polished M_1w100 is 3–5 % above E4's model** (D′ at the 2.0 Å band: 0.478 / 0.485 / 0.487). M_1w100 has learned
  charges and no c6, field term or response. E4's model has not had the long polish.

## 5. Beside them

- **b_T times the learned dipole** is 2.70–2.71 D on every polished run, as in jobs 6, 7 and 9.
  - K10.5's band is a dipole of 1.93–2.21 D.
  - M_1w100 seed 0 reads |b_T| 1.405–1.407 at 1.927 D, just outside.
- **The far probes' non-additive error** is 88–96 % for the learned charges and 99–100 % for the control. T5 is 100 %
  everywhere.
- **Energies** follow the forces. The polished M_1w100 reads 0.021 / 0.016 / 0.012 kcal/mol per molecule, the control
  0.022 / 0.017 / 0.012.

## 6. What this says

- **The optimiser was the larger part of the obstacle.**
  - A full-batch L-BFGS polish moves learned charges along the valley that stopped Adam, from both sides.
  - From above it reaches c* within 3–7 %.
- **It is not the whole of it.**
  - From below the charges slow 20–27 % short, still pulled by the profile.
  - Their ends still depend on the start, so K10.2 fails.
  - Either the approach is very slow there, or the low side has minima of its own.
- **The target is itself a reading made under job 7's short polish.** The long polish lowers the control's objective
  by 22 %. Where the profile's minimum sits under a long polish is not known. The runs started at c* did not move
  toward 2.047 D (M_1w100 seed 0 stays at 1.93), as they would if c* had moved down a little.
- **The training protocol is under-converged for every model.** 1000 full-batch L-BFGS steps take 12 % off the pinned
  control's forces without over-fitting.
  - This bears on every comparison so far, including E4's model.
  - It bears on the baselines too: MACE and MACE-LES must be compared at convergence on both sides.

## 7. Notes to the pre-registration

- **The time estimate held** (1.7–2.1 h a run against 1.5–2 h).
- **K10.3's target was taken from job 7's scan**, made with job 7's 50-step polish (§6). The registered verdicts stand.

## 8. Proposed next, not registered

- **(a) The profile under the long polish, and the low side continued.**
  - Polish job 7's scan (M_1q070, M_1q100, M_1q145; M_1q110 is done) the same way, for the polished profile and its
    minimum.
  - Continue job 10's two M_1w050 runs for another 1000 steps: very slow approach, or a stop?
  - About 8 runs of 2 h.
- **(b) The long polish for E4's model.**
  - The control suggests about 12 %.
  - D′'s steps are far dearer (its training took 3–3.6 h for 3000 Adam steps), so a few hundred steps, first on the
    field-term dagger at the 2.0 Å band (M_16pm_us_d20), then on D′.
- **(c) The charges' band with its own edge** (job 9's option (b)): makes the valley shallower, if (a) shows the low
  side stops.
