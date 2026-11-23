# E4, job 13: why learned charges end where their start puts them — pre-registration

*24 Nov 2026, registered at this commit's time, before any run of job 13 and before any of its numbers. Vadic's request
of 23 Nov: "a deeper investigation into the learned-charges issue". Read against:*

- *`e4-job11-results.md` §2, §3 and §5: the polished profile and where the learned charges stopped;*
- *`e4-job10-results.md` §2–§4 and `e4-job9-results.md` §2–§5: the valley, the pull, the paths;*
- *`les-vs-eft-what-is-provable.md` §5 (project): representability is not identifiability.*

## 1. What is known

**The data identify a field-blind scale.** Pinned O/H charges at c times MB-pol's dipole, with E₀ trained around each,
give a profile with its minimum at c*_lb = 1.070 (1.995 D) under the long polish (job 11).

| c | 0.7 | 1.0 | 1.1 | 1.45 |
|---|---|---|---|---|
| final loss, polished (jobs 10–11) | 30.75 | 23.63 | 21.45 | 31.44 |

The parabola through them pulls with −46 per unit of c at 0.7, −26 at 0.86, +10 at 1.15 and +47 at 1.45.

**Learned charges end where their start puts them** (jobs 7, 10, 11). With their environment dependence (the charge
head) and job 7's unit, 3000 Adam steps, then 2050 full-batch L-BFGS steps:

| start c0 | end (D) | end c | final loss |
|---|---|---|---|
| 0.5 | 1.608 / 1.588 | 0.86 / 0.85 | 23.15 / 24.51 |
| 1.0 (1050 L-BFGS steps) | 1.927 / 2.010 | 1.03 / 1.08 | 18.0 / 19.7 |
| 1.5 | 2.113 / 2.180 | 1.13 / 1.17 | 20.40 / 22.99 |

- The objective prefers c*: the runs near it are 20–25 % lower than those from 0.5, with half the polish.
- In job 11's 1000 steps, the two runs from 0.5 converged on c ≈ 0.86 from both sides (+0.105 D and −0.059 D), while
  the objective fell 13–27 %.

**The valley.** With E₀ held, the objective along the charges' scale is 10⁴ times steeper than the profile (jobs 9, 10).

- E₀ cancels a Coulomb contribution 40–100 times the residual, exactly only at the scale it was trained with.
- In Theorem 3's terms, the part of the Coulomb band's contribution that E₀ cannot absorb is about 1 % of it in
  amplitude: (profile curvature / curvature with E₀ held)^½.
- That 1 % is all that identifies the scale.

## 2. Three explanations, and what separates them

- **H_S, the shape.** The charge head's environment dependence substitutes for the scale. Charges that are wrong in
  scale but right in their variation between environments lower the objective by a path that does not move the scale,
  and the scale stops where that path ends.
  - In job 11 the charges stopped from below read 4–5 % better forces than pinned charges at their scale (0.552–0.564
    against about 0.58).
  - H_S predicts that charges by species alone, with one free scale, reach c*.
- **H_V, the valley.** The scale can move only together with E₀, along a valley floor 10⁴ times flatter than its walls.
  - The optimiser finds that coupled move slowly, and less as E₀'s own improvements fill its memory.
  - H_V predicts that charges by species alone stop as the head's do, and that E₀ re-fits at c* when the scale is moved
    there for it.
- **H_E, E₀'s memory.** An E₀ trained at the wrong scale is held in a basin that keeps that scale.
  - H_E predicts that E₀ does not re-fit when the scale is moved for it.

## 3. The arms (v0.0.37), fixed before any run

Every run is on job 7's settings, seeds 0 and 1, and ends with 1000 full-batch L-BFGS steps (job 11's polish), traced
every 25 steps.

- **R, released from the polished profile.** Job 11's polished pinned runs at c0 = 0.7 and 1.45 (M_1q070, M_1q145).
  - Their pinned O/H charges become learned charges at the same scale (`e4_water.released`): the species offsets at
    the pins, in job 7's unit 0.15 e, and the charge head's last layer at zero.
  - The model is the same at the start: energies and forces agree to 10⁻¹⁵ on fake cells.
  - Two kinds:
    - **by species alone** (M_1zr070, M_1zr145): the charge head held (`train.fit(freeze_charge_head=True)`), so every
      atom carries its species' offset in every environment. For water the neutrality shift ties O to H: one free
      scale;
    - **with the head** (M_1r070, M_1r145): the head learns, as in jobs 7–11.
  - Same start, same E₀, same steps; the head's freedom is the only difference.
- **T, the scale moved for E₀** (M_1t070, M_1t145). The same polished states with their pinned charges moved to c = 1.1
  (job 7's M_1q110 under other names), polished. Its reference is job 10's polished M_1q110 (21.09 / 21.82).
- **S, by species alone from the start of training** (M_1z050, M_1z150). Learned charges by species, started at job 7's
  0.5 and 1.5:
  - job 7's training (3000 Adam steps, 50 L-BFGS), traced as job 9 traced M_1w050 and M_1w150
    (`configs/e4_job13s.yaml`);
  - then job 10's polish (`configs/e4_job13.yaml`).
  - The like-for-like of jobs 7 and 10's flexible warm starts, with the head held.

**What changes in the code:**

- **`train.fit(freeze_charge_head=...)`** and `polish_lbfgs(freeze_charge_head=...)`: the charge head's layers held
  through Adam and the polish (`train.held_charge_head`), its species offsets trained. It needs learned charges, and the
  options record it.
- **`training.start_from.from_rung: {rung: another rung}`** (`scripts/e4_water.py`).
  - A run starts from that rung's run of the same seed, and its json's record must be that rung's.
  - Pinned charges are released into learned ones (`released`) when the rung learns what the source pins.
  - The json records `from_rung` and `released`.
- **The rungs:** M_1r070, M_1r145, M_1zr070, M_1zr145, M_1z050, M_1z150 (M_1 at job 7's unit, warm-started at their c0:
  `RELEASE_C`, `SPECIES_C`) and M_1t070, M_1t145 (M_1q110). The runs by species are `SPECIES_ONLY`.
- **`scripts/e4_job13.py`** reads K13.1–K13.5, without JAX.

## 4. The runs

`run_e4_job13.sbatch` has five tasks, one GPU each:

- **0 and 1:** S, seeds 0 and 1: the first stage of M_1z050 and M_1z150 (`results/e4_job13s`), then their polish;
- **2 and 3:** R, seeds 0 and 1: M_1zr070, M_1zr145, M_1r070, M_1r145;
- **4:** T: M_1t070, M_1t145, seeds 0 and 1.

Then:

```sh
python scripts/e4_job13.py --out_dir results/e4_job13 --stage_dir results/e4_job13s --job9_dir results/e4_job9 \
    --job10_dir results/e4_job10 --job11_dir results/e4_job11
python scripts/e4_water.py summary --out_dir results/e4_job13
```

## 5. Registered

**The readings.**

- The end is the learned dipole on the test frames of 64 molecules, over 1.864 D: c.
- The target is c*_lb = 1.070, job 11's polished profile.
- The final training loss is on every training frame. Rigid forces are read on the unseen frames.

| | reading | band | prediction |
|---|---|---|---|
| **K13.1** | R by species alone (M_1zr070, M_1zr145): every run's end | within ±0.05 of c*_lb (1.902–2.088 D) | **fails** on both sides |
| **K13.2** | R with the head (M_1r070, M_1r145): every run's end | within ±0.05 of c*_lb | **fails** |
| **K13.3** | the head's part: the run by species alone ends nearer c*_lb than the run with the head, pair by pair (start, seed) | on at least 3 of the 4 pairs | holds |
| **K13.4** | T (M_1t070, M_1t145): every run's final training loss against job 10's polished M_1q110 of the same seed | at most 5 % above | holds |
| **K13.5** | S (M_1z050, M_1z150), after the polish: every run's end | within ±0.05 of c*_lb | **fails** |

**How each explanation reads them.**

| | K13.1 | K13.2 | K13.3 | K13.4 | K13.5 |
|---|---|---|---|---|---|
| H_S, the shape | holds | fails | holds | holds | holds |
| H_V, the valley | fails | fails | either | holds | fails |
| H_E, E₀'s memory | fails | fails | either | fails | — |

**The prediction is H_V, with the shape's help at the margin.**

- **Where the runs should end.**
  - Released at 0.7 and 1.45, the charges start near where the flexible warm starts stood after job 7: c ≈ 0.65–0.71
    from below, 1.32–1.37 from above.
  - In job 10's 1000 steps those runs covered 35–45 % of the way from below and 70–85 % from above.
  - At those shares a release from 0.7 ends near 0.83–0.87 and one from 1.45 near 1.13–1.18. Both are outside ±0.05.
- **Why K13.1 is predicted to fail.**
  - The valley's steepness has nothing to do with the head. It comes from E₀ cancelling the Coulomb band, which pinned
    charges and charges by species have alike (job 9's quartic: the curvature with E₀ held is set by the band's
    contribution over the residual).
  - So removing the head removes the path H_S names, but not the valley.
- **Why K13.3 is predicted to hold.** The head gives the objective a way down that leaves the scale where it is: the
  charges stopped from below read 4–5 % better forces than pinned ones at their scale. With it the scale's pull is
  partly spent, so the runs with the head should stop a little shorter.
- **Why K13.4 is predicted to hold.**
  - Moving the scale from 0.7 to 1.1 with E₀ held multiplies the objective by about 10³ (the quartic with job 9's
    curvature).
  - But the move E₀ must then make is the cancellation it already makes, rescaled. L-BFGS should find it within tens of
    steps and spend the rest polishing.
  - M_1q110 itself was polished for 1000 steps from job 7's end (27.05 / 28.00 → 21.09 / 21.82).
- **Why K13.5 is predicted to fail.** S is job 7's and job 10's flexible warm starts without the head. Under H_V it ends
  where they did: about 0.81–0.88 from below and 1.13–1.17 from above.

**Why these bands.**

- **±0.05 in c** is job 11's band for the profile's minimum: half the scan's step, and the spread of the forces' minima
  (1.04–1.08) about c*_lb. A run inside it has arrived as far as the data can say.
- **3 of 4 pairs** keeps one pair free for the seed's noise: job 10's two seeds from 0.5 ended 0.08 apart in c.
- **5 % on T** is about one and a half times the polished control's range over its two seeds (21.09 against 21.82, 3.4
  %). A re-fit that leaves more than that above the control has not re-fitted.

## 6. Read beside, not registered

- **The paths:** the learned dipole on the traced frames every 250 steps, and the last 250 steps' move against the
  distance left (job 10's reading of a stop).
- **The end's final loss against the polished profile at the end's c.** A run by species alone that stops on the
  profile has an E₀ as good as the pinned scan's at its scale, with the pull unspent.
- **The valley's curvature with the rest held** at the start and the end of every learned run.
- **T's objective path:** how fast the re-fit is, and whether it is still falling at the end.
- **S's first stage against job 9's M_1w050 and M_1w150 at the same steps:** does the scale move faster through Adam
  without the head?
- **S's polish against job 10's polish of M_1w050 and M_1w150**, and job 11's continuation.
- **The starts' objective and forces** (job 11's polished M_1q070 and M_1q145), every run's forces, the energies.

## 7. What would change the reading

- **K13.1 and K13.5 hold, K13.2 fails (H_S).** The head holds the scale.
  - Learned charges then find the data's scale by species alone.
  - The recipe to test next is staged: the scale by species first, then the head released at it, as job 10's M_1w100,
    started at 1.0, stayed near it.
- **K13.1, K13.2 and K13.5 fail, K13.4 holds (H_V, predicted).** E₀ can re-fit at c* when the scale is moved there for
  it, but the optimiser does not find that move from either side, with or without the head.
  - The obstacle is the valley.
  - The next tests are the shallower valley (the charges' band with its own edge, as job 8 gave c6 one: less for E₀ to
    cancel) and the scale set from outside the gradient, by the profile or a sum rule.
  - For LES-type models this says that the learned scale is set by the start.
- **K13.4 fails (H_E).** E₀ trained at the wrong scale does not re-fit at c* in 1000 steps. Read T's path:
  - still falling fast at the end: a slow re-fit, and the band was too tight for the steps;
  - flat above the control: a basin.
  - Either way the learned scale depends on E₀'s history, and a release cannot test the head.
- **K13.2 holds.** The head arrives from the polished profile's states. The stops of jobs 10 and 11 then came from
  training E₀ and the head together from the start, not from the head.
- **K13.1 holds from above only,** as job 10 found for the head: the valley is asymmetric. Read the paths and the
  end's loss against the profile on the low side.

## 8. Not tested here

- **The window.** Training at 128 molecules (15.65 Å), where E₀'s two 6 Å layers no longer reach across the cell, should
  raise the part of the Coulomb band E₀ cannot absorb above job 7's 1 %. This is Theorem 3's angle, in system size.
  It is the next step if H_V holds and the shallower valley does not suffice.
- **The charges' band with its own edge** (job 9's option (b)): it needs a new profile at that edge.
- **Far-field structures in the training set**, where E₀ sees nothing and the band is the whole interaction: matching,
  in the EFT's sense, of a learned scale.

## 9. Cost

- **S's first stage:** job 9's runs took 20–23 minutes with the diagnostics.
- **The polishes:** job 10's and job 11's M_1 polishes took 1.7–2.1 h a run.
- **Tasks 0 and 1:** two first stages and two polishes, about 4.5 h. **Tasks 2–4:** four polishes, about 7–8.5 h.
- **In all:** about 34 GPU hours. The sbatch asks 14 h a task, as job 11 did.

## 10. Numbers before any run

**The starts of R and T** (job 11's polished profile, seeds 0 / 1):

- M_1q070: final loss 31.21 / 30.28, still falling 4 % in its last 250 steps;
- M_1q145: 32.87 / 30.01.

**T's reference** (job 10's polished M_1q110, seeds 0 / 1):

- final loss 21.09 / 21.82;
- forces 0.519 / 0.534 / 0.538 and 0.522 / 0.539 / 0.545.

**The flexible warm starts** (seeds 0 / 1):

| | job 7 (3050 steps) | job 10 (+1000) | job 11 (+1000) |
|---|---|---|---|
| M_1w050, D | 1.214 / 1.315 | 1.504 / 1.648 | 1.608 / 1.588 |
| M_1w050, final loss | 46.1 / 48.8 | 31.6 / 29.1 | 23.15 / 24.51 |
| M_1w150, D | 2.461 / 2.563 | 2.108 / 2.184 | 2.113 / 2.180 |
| M_1w150, final loss | 38.2 / 39.8 | 23.5 / 26.3 | 20.40 / 22.99 |

**On fake cells (`tests/test_job13.py`, CPU):**

- a release reproduces the pinned model's energies to 10⁻¹⁵ and forces to 10⁻¹⁷;
- with the head held, every O and every H carries one charge across all frames (spread 0), and the species offsets
  move.
