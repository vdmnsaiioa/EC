# E4, job 13: why learned charges end where their start puts them — results

*Read 25 Nov 2026 against `e4-job13-preregistration.md`, registered on 24 Nov before any run (425a336 on GitHub). The
job ran on Niflheim at v0.0.37 as the registered five tasks, on RTX 3090s. It was read with `scripts/e4_job13.py` and
`scripts/e4_water.py summary` on `results/e4_job13` and `results/e4_job13s`.*

- *The end is the learned dipole on the test frames of 64 molecules over 1.864 D: c. The target is c*_lb = 1.070
  (1.994 D), job 11's polished profile, recomputed by the reader.*
- *Run times: S's first stages 1350–1445 s; the polishes 7317–8513 s (2.0–2.4 h, against the estimated 1.7–2.1 h); T's
  6054–6305 s. Tasks took 5.0–8.9 h, inside the 14 h asked.*

## 1. The registered readings

| | band | Niflheim | verdict | predicted |
|---|---|---|---|---|
| **K13.1** | R by species alone: every end within ±0.05 of c*_lb | 0.754, 0.745 (from 0.7); 1.379, 1.380 (from 1.45) | **fails** | fails |
| **K13.2** | R with the head: every end within ±0.05 | 0.767, 0.758; 1.403, 1.382 | **fails** | fails |
| **K13.3** | by species alone nearer c*_lb on 3 of 4 pairs | 2 of 4 | **fails** | holds |
| **K13.4** | T: every final loss at most 5 % above the polished control's | +29.8, +29.4 % (from 0.7); +805, +990 % (from 1.45) | **fails** | holds |
| **K13.5** | S: every end within ±0.05 | 0.632, 0.625 (from 0.5); 1.347, 1.342 (from 1.5) | **fails** | fails |

**By the pre-registration's table, this is E₀'s memory (H_E):** K13.1 and K13.2 fail, K13.4 fails, K13.3 either.

- **The head (H_S) is refuted twice.** By species alone, the charges do not reach c* (K13.1, K13.5). And through job
  7's training, the head moves the scale four to seven times further than the species offsets alone do (§4).
- **The valley (H_V) predicted K13.4 to hold, and it failed decisively.** Moved to 1.1 for it, E₀ does not re-fit in
  1000 steps.

## 2. Released from the polished profile, the charges barely move (R)

| run | end c (from c0) | share of the way to c*_lb | the four 250-step moves on the traced frames (D) | final loss (start) |
|---|---|---|---|---|
| M_1zr070 seed 0 | 0.754 (0.7) | 15 % | +0.018, +0.022, +0.028, +0.033 | 26.22 (31.21) |
| M_1zr070 seed 1 | 0.745 | 12 % | +0.008, +0.027, +0.029, +0.020 | 25.33 (30.28) |
| M_1r070 seed 0 | 0.767 | 18 % | +0.029, +0.037, +0.014, +0.044 | 26.72 (31.21) |
| M_1r070 seed 1 | 0.758 | 16 % | +0.013, +0.011, +0.034, +0.049 | 25.09 (30.28) |
| M_1zr145 seed 0 | 1.379 (1.45) | 19 % | −0.016, −0.033, −0.073, −0.011 | 27.68 (32.87) |
| M_1zr145 seed 1 | 1.380 | 18 % | −0.017, −0.044, −0.027, −0.043 | 24.88 (30.01) |
| M_1r145 seed 0 | 1.403 | 12 % | −0.010, −0.013, −0.021, −0.044 | 29.51 (32.87) |
| M_1r145 seed 1 | 1.382 | 18 % | −0.028, −0.045, −0.025, −0.029 | 25.29 (30.01) |

- **They moved 0.045–0.071 in c in 1000 steps, 12–19 % of the way,** against the profile's pull of −46 and +47 per unit of
  c at the starts. Job 10's flexible warm starts had covered 35–85 % in the same steps.
- **They have not stopped.** They move 0.01–0.07 D every 250 steps, from below without slowing. At that pace c* is
  several thousand steps away.
- **The objective fell 10–17 % while the scale barely moved:** E₀ kept improving at its own scale. Seven of the eight
  ends read 2–12 % below the polished profile's value at their c (one 0.9 % above).
- **The head makes no difference here.** Each pair ends within 0.002–0.024 of each other in c: the head a little further
  from below, a little shorter from above.

## 3. Moved for it, E₀ does not follow (T)

| run | objective at the start | at 250 / 500 / 750 / 975 | final against the control | forces against the control's |
|---|---|---|---|---|
| M_1t070 seed 0 | 70,373 | 39.6 / 32.9 / 29.6 / 27.5 | 27.38 against 21.09 | 0.585 / 0.596 / 0.605 against 0.519 / 0.534 / 0.538 |
| M_1t070 seed 1 | 70,362 | 44.0 / 34.1 / 30.6 / 28.5 | 28.23 against 21.82 | 0.586 / 0.609 / 0.613 against 0.522 / 0.539 / 0.545 |
| M_1t145 seed 0 | 108,123 | 339 / 258 / 228 / 195 | 190.9 against 21.09 | 1.494 / 1.536 / 1.541 |
| M_1t145 seed 1 | 108,085 | 361 / 284 / 261 / 239 | 237.7 against 21.82 | 1.688 / 1.727 / 1.732 |

- **From 0.7, the re-fit costs about a training.** After 1000 steps the objective reads 27.4–28.2, still falling 7 % in
  the last 225 steps. That is where M_1q110 stood after job 7's whole training from scratch (27.05 / 28.00).
- **From 1.45, E₀ does not recover.** It stays about ten times above the control, with forces three times worse,
  falling 8–14 % in the last 225 steps.
- **The asymmetry is not explained here.** E₀ trained at 1.45 cancels a band contribution 4.3 times larger in amplitude
  than at 0.7 (c²), and the changes to 1.1 are of similar size (c² by −0.89 and +0.72). Undoing its cancellation is far
  harder than extending it.
- **The pre-registration's argument was wrong:** the move E₀ must make is not "the cancellation it already makes,
  rescaled", found in tens of steps.

## 4. By species alone from the start: the head moves the scale, it does not hold it (S)

| | from 0.5, seeds 0 / 1 | from 1.5, seeds 0 / 1 |
|---|---|---|
| start (D) | 0.932 | 2.796 |
| after job 7's training, by species | 0.973 / 1.011 (45.9 / 47.1) | 2.742 / 2.736 (39.4 / 37.1) |
| after job 7's training, with the head (job 9) | 1.214 / 1.315 (46.1 / 48.8) | 2.461 / 2.562 (38.2 / 39.8) |
| after the polish, by species | 1.179 / 1.165 (31.76 / 31.84) | 2.512 / 2.501 (29.56 / 27.32) |
| after the polish, with the head (job 10) | 1.504 / 1.648 (31.61 / 29.06) | 2.108 / 2.184 (23.53 / 26.31) |

(the final loss in brackets)

- **Through Adam the species offsets alone moved the scale 4–7 times less than the head did** (+0.04 / +0.08 D against
  +0.28 / +0.38 from below; −0.05 / −0.06 against −0.34 / −0.23 from above), at the same final loss.
  - Adam steps every parameter by about its learning rate, whatever the size of its gradient.
  - The head gives the scale its last layer's 33 parameters to move along besides the two species offsets.
- **Through the polish they moved half to two thirds of the head's distance** (46–71 %).
- **So the head is not what holds the scale; it is what moves it.** Without it the charges end further from c* (0.63
  against 0.81–0.88 from below, 1.34–1.35 against 1.13–1.17 from above).

## 5. The valley's walls are the band's alone

**With E₀ held, the curvature along the charges' scale is 1.087×10⁶ c⁴ for every state by species alone, to 0.3 %:**
R's starts and ends, S's starts and ends, E₀ trained from scratch or polished: sixteen states from c = 0.52 to 1.47.

- **Why.** With E₀ held, the residual along the scale is r + (s² − 1) v_C(c), so the curvature at s = 1 is 8 |v_C(c)|²
  less a term in the residual. v_C(c) = c² v_C(1) is the band's own contribution to every energy and force. It depends on
  the charges and the frames, not on E₀.
- **The head's variation between environments adds −5 to +43 %.**
- **At c*_lb the walls' curvature is 1.42×10⁶; the profile's, c² L*″, is 143.** The ratio is 10,000. The part of the
  band's contribution that E₀ cannot absorb, the part that identifies the scale, is 1.0 % of it in amplitude: Theorem 3's
  angle, from one constant and the profile.

## 6. The scale's mobility falls as E₀ trains

| E₀'s training at the scale, before the next 1000 L-BFGS steps | runs | the move in c |
|---|---|---|
| job 7's training (3000 Adam steps, 50 L-BFGS) | job 10 (with the head), job 13's S (by species) | 0.16–0.20; 0.08–0.13 |
| the same and a polish of 1000 steps | job 13's R (pinned, polished, released) | 0.05–0.07 |
| the same and two polishes | job 11 (with the head) | 0.00–0.06, toward each other from below |

- **The data's pull is the same throughout,** ±30–50 per unit of c (the profile's slope).
- **The scale's mobility is not.** The better E₀ is trained at a scale, the less the scale moves.
- **T says why:** re-fitting E₀ at another scale costs about a training (from 0.7) or fails (from 1.45).

## 7. Notes to the pre-registration

- **The predicted ends for R (0.83–0.87 and 1.13–1.18) assumed job 10's pace.** The releases moved a third as far: how
  long E₀ had been trained at the start mattered more than assumed.
- **K13.3's prediction was wrong:** from below, the head moved the scale further, not shorter.
- **K13.4's prediction was wrong, decisively** (§3). The registered verdicts stand.
- **The polishes took 2.0–2.4 h,** against the estimated 1.7–2.1 h.

## 8. What this says

- **Learned charges trained with E₀ do not find the data's scale, because the scale can only move with E₀, and E₀
  commits.**
  - The walls of the valley are the band's own contribution, E₀-independent (c⁴, §5). The floor is E₀'s re-fit.
  - Every step of E₀'s training at a scale makes leaving it dearer. Re-fitting at another costs about a training (§3).
  - So the scale ends where it stood when E₀ settled: where the start put it, moved by what the early, mobile phase
    allowed (§6).
- **It is not the head.** Learned charges by species alone, one free scale, stop the same way, and move less under Adam.
- **For learned-charge models (LES-type):** the learned scale is a fact about the training's history, not the data, at
  least where the training cell is one the local model spans (64 molecules here; larger cells were not tested). This is the measured form of `les-vs-eft-what-is-provable.md` §5:
  representable, identifiable at 1 %, and not found.
- **For E4's model:** its sources are matched and pinned, not fitted. This is the reason in mechanism.
- **The lemma's scale cannot be read through learned charges.** The pinned profile (job 7's scan, polished in jobs
  10–11) reads it.

## 9. Proposed next, not registered

- **(a) The thread closes here as a result.** The scale freezes as E₀ trains; the walls are the band's (c⁴); the floor
  is E₀'s re-fit; the head moves the scale rather than holding it.
- **(b) The same test on the baseline it is about.** MACE-LES trained from two latent-charge scales (its default
  initialisation times 0.5 and times 1.5), with its learned dipole read as here. If it ends where it starts, the claim is
  shown on the model it concerns.
- **(c) The remedies left inside learning,** if wanted: the band at its own edge (walls lower by the contact's share) or
  the window (128 molecules). Both cost a new profile.
