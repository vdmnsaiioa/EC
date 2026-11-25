# E4, job 14: the scale and E₀'s reach — results

*Read 25 Nov 2026 against `e4-job14-preregistration.md`, registered at 02:34 CET on 25 Nov before any run (a107b1d on
GitHub). The job ran on Niflheim at v0.0.38 as the registered six tasks, on RTX 3090s. It was read with
`scripts/e4_job14.py` and `scripts/e4_water.py summary` on `results/e4_job14_r45` and `results/e4_job14_r6`, and read
again with the reader's v0.0.39 additions for the paths (§7).*

- *The end of a warm start is its learned dipole on the test frames of 64 molecules over 1.864 D: c. Forces are rigid
  forces in kcal/mol/Å on the unseen frames, seed means over seeds 0 and 1.*
- *The 12 Å reference is job 11's polished profile (c* = 1.070, c*² L*″ = 143) and job 10's polished warm starts.*
- *Run times: first stages 460–690 s at 4.5 Å and 812–1183 s at 6 Å; polishes 1061–1645 s at 4.5 Å and 4295–5850 s at
  6 Å. Tasks took 1.7–2.4 h (4.5 Å) and 5.9–7.3 h (6 Å): 24.9 GPU hours, against the estimated 57. No run failed;
  every polish ran its 1000 steps.*

## 1. The registered readings

| | band | Niflheim | verdict | predicted |
|---|---|---|---|---|
| **K14.1** | the profile's curvature at 4.5 Å at least 5 times 12 Å's | 304.0 against 142.8: 2.1 times | **fails** | holds |
| **K14.2** | at 6 Å at least 2 times | 173.8: 1.2 times | **fails** | holds |
| **K14.3** | at 4.5 Å every polished warm start within ±0.05 of the reach's c* (1.006) | −0.032, −0.018 (from 0.5); −0.031, −0.036 (from 1.5) | **holds** | holds |
| **K14.4** | the gap, seed by seed, 4.5 Å < 6 Å < 12 Å | 0.001 < 0.012 < 0.324; −0.018 < 0.104 < 0.288 | **holds** | holds |

**The registered branch is "K14.3 holds and K14.1 fails"** (pre-registration §6): the scale becomes mobile without a
sharper profile; read the first stage's ends and the paths. K14.1's failure carries its own branch: E₀'s reach is not
what sets the 1 %, and the next test would be the cell (training at 128 molecules). K14.3 makes that less pressing: the
1 % was not what stopped the scale (§3).

- **H_R as stated is refuted.** The part of the band's contribution that identifies the scale barely grows with a
  shorter reach.
- **Its consequence holds anyway.** At 4.5 Å the learned charges end at one scale from both sides; at 6 Å they are
  still converging on one (§7).
- **K14.3's verdict depends on where the profile's minimum is put** (§4). What does not depend on it is K14.4: the start
  no longer decides the scale.

## 2. The profile barely sharpens

| reach | c* | c*² L*″ | against 12 Å | walls at c* over the profile | the identifying part, in amplitude |
|---|---|---|---|---|---|
| 12 Å (two layers of 6 Å) | 1.070 | 142.8 | 1 | 9,990 | 1.00 % |
| 6 Å (one layer) | 1.054 | 173.8 | 1.2 | 7,720 | 1.14 % |
| 4.5 Å (one layer) | 1.006 | 304.0 | 2.1 | 3,660 | 1.65 % |

(the walls are job 13's 1.087×10⁶ c⁴ at each c*)

- **The estimate said 19 and 8 times; the profile gives 2.1 and 1.2.**
  - The estimate summed uncorrelated dipoles beyond the reach. Water's are correlated and screen one another, which is
    the likely reason the part beyond 4.5 Å is about ten times smaller than assumed.
  - Even a one-layer E₀ of 4.5 Å absorbs 98.3 % of the band's dependence on the scale. That dependence is mostly
    near-field.
- **The failure does not depend on the construction or the seed.** A parabola in c² (the band's contribution goes as
  c²) gives 1.8 and 1.2 times; seed by seed the ratios are 1.8–2.7 and 1.0–1.6.
- **The sharpening is on the high side.** From c = 1.1 to 1.45 the profile rises 30.1 at 4.5 Å, against 14.3 at 6 Å and
  10.0 at 12 Å. From 1.0 to 0.7 it rises 12.1, 7.3 and 7.1. A short E₀ cannot absorb charges that are too strong.
- **c* moves down with the reach:** 1.070, 1.054, 1.006 on the registered construction. The scale the pinned charges
  pick depends a little on how much E₀ can absorb.

## 3. The scale moves, two to three times further

The ends in c, seeds 0 / 1:

| | 12 Å | 6 Å | 4.5 Å |
|---|---|---|---|
| from 0.5, after job 7's training | 0.651 / 0.705 | 0.840 / 0.844 | 0.964 / 0.946 |
| from 1.5, after job 7's training | 1.320 / 1.375 | 1.203 / 1.262 | 1.034 / 1.093 |
| the gap after job 7's training | 0.669 / 0.670 | 0.363 / 0.418 | 0.070 / 0.147 |
| from 0.5, polished | 0.807 / 0.884 | 0.963 / 0.932 | 0.974 / 0.987 |
| from 1.5, polished | 1.131 / 1.172 | 0.975 / 1.035 | 0.975 / 0.969 |
| the gap, polished (K14.4) | 0.324 / 0.288 | 0.012 / 0.104 | 0.001 / −0.018 |

- **Adam does most of it at short reach.** Job 7's training closes a third of the starting gap of 1.0 at 12 Å, 58–64 %
  at 6 Å and 85–93 % at 4.5 Å.
- **The polish moves the scale about as far at 6 Å as at 12 Å** (0.09–0.23 in c against 0.16–0.20), from starts that are
  nearer.
- **At 4.5 Å the runs from 1.5 cross the registered c*** (1.034 → 0.975, 1.093 → 0.969). All four ends lie 0.018–0.036
  below it, within 0.018 of one another.
- **The valley is not what changed.** Its walls over its floor fell from 10,000 to 3,700, a factor of 2.7. The mobility
  changed far more.
- **6 Å against 12 Å is depth alone** (and the capacity that comes with it): the same cutoff, one layer or two. The
  profile is 1.2 times sharper; Adam's move doubled, and the polished gap fell from 0.29–0.32 to 0.01–0.10, still
  closing (§7).
- **So the scale's freezing at 12 Å (jobs 9–13) was the dynamics, not the identification.** A two-layer E₀ commits to
  the scale it trains at; a one-layer E₀ lets the scale move while it trains.
- **The paths (§7) say how.** The data's pull on the scale is the same at every reach, in strength and in consistency
  from batch to batch, and the move keeps the same timing. At short reach the scale goes two to three times further for
  the same pull, early and late in Adam; in the polish it keeps moving where at 12 Å it stopped.

## 4. Where the profile's minimum is

**The scan cannot place it better than about ±0.05.**

- At every reach L(1.1) < L(1.0): by 1.9 at 4.5 Å, 3.9 at 6 Å and 2.2 at 12 Å, with seed spreads of at most 0.5. If the
  profile is convex, its minimum lies above 1.0.
- The registered parabola in c misfits the points at 1.0 and 1.1 by ±1.6 at 4.5 Å (±1.9 at 6 Å, ±1.0 at 12 Å). The
  steep point at 1.45 pulls its minimum down to 1.006.
- A parabola in c² misfits by ±0.7 and puts c* at 1.07 at 4.5 Å (1.12 at 6 Å, 1.13 at 12 Å). The cubic through the four
  points puts it at 1.11.
- Against those, the four ends at 4.5 Å sit 0.08–0.14 below the minimum, and K14.3 would fail.
- By seed the constructions agree at 4.5 Å and 6 Å (c* within 0.003, curvature within 3 %). At 12 Å they do not:
  seed 0 reads c* 1.061 and curvature 174, seed 1 1.085 and 112 (job 10's M_1q110 differs by 0.7 between seeds). K14.1's
  ratio by seed is 1.8 and 2.7, K14.2's 1.0 and 1.6: both fail on either seed.

**What is robust:**

- the four ends agree to within 0.018 (0.976 ± 0.007): the start no longer decides the scale;
- they lie below the pinned profile's minimum: by 0.02–0.04 on the registered construction, by up to 0.14 on others.

**The learned charges have a minimum of their own, below the pinned one.**

- At 4.5 Å the polish approaches it from both sides and overshoots it (§7). From 0.5, seed 1 rises to 1.004 by step 500
  and comes back to 0.986. From 1.5, seed 1 falls to 0.962 by step 500 and comes back to 0.969.
- Three of the four have stopped at 0.969–0.986 (at most 0.007 in the last 250 steps). The fourth, from 1.5, is at 0.975
  and still falling 0.017 per 250 steps.
- At 6 Å the runs from 1.5 are still falling through 1.0 (to 0.975 and 1.035), and one from 0.5 has stopped at 0.963.
- Their head varies the charges between environments; pinned charges by species are a different model. The pinned
  minimum lies above 1.0, the learned one at about 0.97–0.99.
- Whether the gap is the head's (by species alone the minimum would be the pinned one) needs the profile resolved and
  the warm starts run by species alone (§10(a)). The registered verdict stands.

The same caveat applies to job 11's c*_lb = 1.070 (1.13 in c²): job 11's polished forces put it at 1.04–1.08.

## 5. The start still marks the fit

| | 4.5 Å, seeds 0 / 1 | 6 Å, seeds 0 / 1 |
|---|---|---|
| final loss from 0.5 | 39.46 / 33.68 | 34.23 / 40.40 |
| final loss from 1.5 | 31.04 / 29.21 | 23.88 / 28.53 |
| forces at 64 / 128 / 256, from 0.5 | 0.668 / 0.671 / 0.675 | 0.679 / 0.680 / 0.683 |
| forces from 1.5 | 0.602 / 0.611 / 0.614 | 0.565 / 0.573 / 0.577 |
| the pinned scan's best (M_1q110) | 0.636 / 0.649 / 0.656 | 0.565 / 0.582 / 0.584 |

- **At the same scale and nearly the same charges, the runs from 0.5 fit worse.**
  - At 4.5 Å seed 0, from 0.5 and from 1.5 read q_O −0.645 and −0.646 e, c 0.974 and 0.975, and final losses 39.5 and
    31.0.
  - Final losses from 0.5 are 15–43 % above those from 1.5; forces are 11 % worse at 4.5 Å and 20 % worse at 6 Å.
- **The start's mark moved from the scale into E₀.** This is job 13's E₀ memory, at short reach.
- **It is read at 1000 steps, with every objective still falling 5–8 % in the last 250** (§7). The gap did not close in
  the polish. The runs from 0.5 started it 8–19 % higher and ended 15–43 % higher: the runs from 1.5 fell faster.
- **From 1.5 the learned charges beat the pinned scan's best at 4.5 Å by 5–6 %;** at 6 Å they are level with it.
- **Seed spreads are large** (±0.01–0.03 in forces, against at most ±0.005 for the pinned runs).

## 6. The cost of the short reach

Forces at 64 molecules against job 11's polished scan at the same c (job 10's M_1q110):

| c | 12 Å | 6 Å | 4.5 Å |
|---|---|---|---|
| 0.7 | 0.622 | 0.681 (+9.5 %) | 0.773 (+24.3 %) |
| 1.0 | 0.546 | 0.608 (+11.4 %) | 0.659 (+20.7 %) |
| 1.1 | 0.520 | 0.565 (+8.7 %) | 0.636 (+22.3 %) |
| 1.45 | 0.624 | 0.702 (+12.5 %) | 0.882 (+41.3 %) |

- **One layer costs 9–13 % at 6 Å and 21–41 % at 4.5 Å.** At 1.1 the same holds at 128 and 256 molecules (+8.6 / +7.9 %
  and +21.1 / +21.3 %).
- **The short reach transfers in size a little better:** from 64 to 256 molecules M_1q110 grows 3.1 % at 4.5 Å, 3.4 % at
  6 Å and 4.0 % at 12 Å.
- **Energies** are 0.01–0.04 kcal/mol per molecule at every reach, as at 12 Å.
- **Learned charges at their scale pay the same.** Against job 10's M_1w100 (learned, started at c*, polished once:
  0.491 / 0.507 / 0.510), the polished warm starts from 1.5 read +23 / +21 / +20 % at 4.5 Å and +15 / +13 / +13 % at 6 Å.
  - The reader compares them with job 10's M_1w050 and M_1w150 instead (+7–9 % at 4.5 Å, 0–10 % at 6 Å).
  - Those sat off their scale at 12 Å (c 0.81–0.88 and 1.13–1.17), so that comparison flatters the short reach.
- **For scale, E4's model (D′, long polish) reads 0.444 / 0.449 / 0.451.** The best 4.5 Å model is 36 % above it.
- **The walls at the warm starts' ends** read 1.0–3.4 times job 13's law by species (1.03, 3.42, 2.60, 2.69 at 4.5 Å;
  1.07, 2.39, 2.73, 2.92 at 6 Å), against −5 to +43 % at 12 Å.
  - They did not fall: the head's variation between environments adds more at short reach.
  - The walls therefore cannot be why the scale moves.

## 7. The paths

Read with the reader's v0.0.39 additions. The scale c is the learned dipole on the four traced training frames over
1.864 D.

**Through job 7's training (3000 Adam steps, then 50 L-BFGS):**

| | 12 Å (job 9) | 6 Å | 4.5 Å |
|---|---|---|---|
| from 0.5: the move in Adam steps 0–500 | +0.083 / +0.074 | +0.178 / +0.147 | +0.242 / +0.208 |
| in steps 500–3000 | +0.060 / +0.107 | +0.127 / +0.168 | +0.198 / +0.204 |
| from 1.5: in steps 0–500 | −0.097 / −0.072 | −0.149 / −0.108 | −0.326 / −0.225 |
| in steps 500–3000 | −0.061 / −0.047 | −0.119 / −0.106 | −0.114 / −0.150 |
| half of Adam's move made by step | 300–700 | 400–600 | 300–500 |
| the pull's mean / sd, steps 0–500 (from 0.5; from 1.5) | −0.06, −0.06; +0.11, +0.09 | −0.07, −0.08; +0.10, +0.08 | −0.09, −0.09; +0.11, +0.09 |
| the pull's mean / sd, later blocks of 500 | 0.04 at most | 0.02 at most | 0.02 at most |

(seeds 0 / 1; the pull's sign is the objective's slope along the scale, so negative pulls it up)

- **The timing is the same at every reach:** half of Adam's move within 300–700 steps, as the learning rate decays.
- **The size is not.** At 4.5 Å the scale moves about three times as far as at 12 Å in the first 500 steps (2.8–3.4
  times), and two to three times as far after (1.9–3.3). At 6 Å it moves 1.5–2.3 times as far.
- **The pull is the same at every reach.** It points to the middle from both sides, at 0.06–0.11 of its batch-to-batch
  spread in the first 500 steps and at most 0.04 after. The data push the scale no harder and no more consistently at
  short reach.
- **So what changed is the scale's response to the pull, not the pull.** With two layers of E₀ the same pull moves it a
  third as far. This job does not show the coupling that damps it.
- **The stage's 50 L-BFGS steps** move the scale 0.024–0.035 at both short reaches, more than Adam's last 1000 steps
  (0.012–0.025).

**Through the polish (1000 full-batch L-BFGS steps):**

| run | 0 | 250 | 500 | 750 | 1000 | last 250 | objective, last 250 |
|---|---|---|---|---|---|---|---|
| 4.5 Å from 0.5, seed 0 | 0.965 | 0.952 | 0.964 | 0.978 | 0.974 | −0.004 | −5.3 % |
| 4.5 Å from 0.5, seed 1 | 0.946 | 1.000 | 1.004 | 0.986 | 0.986 | +0.001 | −6.6 % |
| 4.5 Å from 1.5, seed 0 | 1.034 | 1.021 | 0.999 | 0.991 | 0.975 | −0.017 | −6.3 % |
| 4.5 Å from 1.5, seed 1 | 1.093 | 0.984 | 0.962 | 0.976 | 0.969 | −0.007 | −7.9 % |
| 6 Å from 0.5, seed 0 | 0.840 | 0.881 | 0.934 | 0.962 | 0.963 | +0.000 | −6.6 % |
| 6 Å from 0.5, seed 1 | 0.844 | 0.846 | 0.894 | 0.914 | 0.931 | +0.017 | −6.2 % |
| 6 Å from 1.5, seed 0 | 1.204 | 1.076 | 1.062 | 1.006 | 0.975 | −0.032 | −8.2 % |
| 6 Å from 1.5, seed 1 | 1.262 | 1.132 | 1.076 | 1.074 | 1.035 | −0.039 | −6.0 % |

(12 Å, job 10: the last 250 steps moved +0.022 / +0.008 from 0.5 and −0.008 / −0.012 from 1.5; job 11's further 1000
steps left the gap at about 0.3)

- **At 4.5 Å the scale has settled.**
  - Three of the four have stopped, at 0.969–0.986.
  - Two overshot and came back: from 0.5 seed 1 to 1.004, from 1.5 seed 1 to 0.962.
  - The fourth (from 1.5, seed 0) is at 0.975 and still falling 0.017 per 250 steps.
  - This is a minimum approached from both sides, at about 0.97–0.99 (§4).
- **At 6 Å it has not settled, and it is still moving toward one scale.**
  - Three of the four moved 0.017–0.039 in the last 250 steps, toward each other. That is faster than 12 Å's last moves
    (0.008–0.022), between runs that are much nearer.
  - Seed 0's pair is at 0.963 (stopped) and 0.975 (falling). Seed 1's gap of 0.104 is closing by about 0.06 per 250
    steps.
  - So K14.4's gap at 6 Å is a snapshot of a pair still in motion. At 12 Å the pairs had stopped apart.
- **E₀ has not converged at any reach.** The objective falls 5–8 % in the last 250 steps, as at 12 Å (job 10).

## 8. Notes to the pre-registration

- **The estimate behind K14.1 and K14.2 was wrong by an order of magnitude** (§2). It assumed uncorrelated dipoles
  beyond the reach; "screening and correlations change the numbers, not the order" was wrong.
- **K14.3's registered construction places c* at 1.006,** at the low edge of what the points allow (§4). The verdict
  stands as registered; its reading is in §4.
- **"One layer is not much cheaper" was wrong on the GPU.**
  - It came from a CPU timing of 10–13 %.
  - On the RTX 3090 a polish took 1.2–1.6 h at 6 Å and 0.3–0.5 h at 4.5 Å, against 1.7–2.4 h at 12 Å.
  - The job took 24.9 GPU hours, not 57.
- **The paths and the profile by seed were read with the reader's v0.0.39 additions,** made after the first read. They
  are beside the registered readings, as §5 of the pre-registration lists them, and change none of them.

## 9. What this says

- **The obstacle to learned charges at 12 Å was the dynamics, not the identification.**
  - The bulk data identify the scale about as weakly at 4.5 Å as at 12 Å: 1.65 % of the band's contribution against
    1.0 %.
  - The data's pull on the scale is the same at every reach (§7).
  - With a one-layer E₀ the scale responds two to three times more to that pull. It ends independent of the start at
    4.5 Å, and is still converging at 6 Å.
  - With two layers it freezes where its start and the early phase put it (jobs 9–13).
- **Depth is the lever, at least in part.** 6 Å against 12 Å changes only the layers. Whether reach or depth does the
  rest at 4.5 Å is not separated here.
- **The scale learned charges find is their own model's.** At 4.5 Å it is 0.97–0.99, approached from both sides. The
  pinned charges' minimum lies above 1.0, at 1.006–1.11 depending on the curve through the scan.
- **This reconciles the LES results with jobs 6–13.**
  - The LES water model has no message passing (4.5 Å, one layer). Its charges are mobile, and they come out
    physical.
  - The model at risk is the one with message passing over the cell: MACE-LES. Job 13's proposed test (b) now has a
    prediction: MACE-LES's latent charges end where their start puts them.
- **The start still marks the fit** (§5). Learned charges at short reach find the scale but carry their history in E₀,
  at least at 1000 steps.
- **For E4's model nothing changes:** its sources are matched and pinned. Job 14 gives the reason in a second form:
  learned charges can be mobile or accurate, here not both. At 4.5 Å the scale is found, at a cost of about 20 % in
  forces.
- **A route that may give both** is the short model to fix the scale, then the deep model with the charges held.

## 10. Proposed next, not registered

- **(a) The two minima, cheaply.** At 4.5 Å a run takes 0.5 h, stage and polish.
  - Pin c at 0.9, 0.95, 1.05, 1.2 and 1.3 to resolve the profile between 1.0 and 1.45.
  - Run the warm starts by species alone (the head held, job 13's S at 4.5 Å).
  - If species-only charges end at the resolved pinned minimum, the gap below it is the head's. About 14 runs, 7 GPU
    hours.
- **(b) Depth alone, finished.** Continue the four 6 Å warm starts for another 1000 steps (about 5 GPU hours).
  - Do they meet, as at 4.5 Å? This is the cleanest comparison with 12 Å, where the pairs stopped apart.
- **(c) Depth against reach.**
  - Two layers of 3 Å (a reach of 6 Å, depth 2) against job 14's one layer of 6 Å.
  - If it freezes like 12 Å, depth is the lever; if it moves like 6 Å, reach is.
- **(d) The route.** Transplant the 4.5 Å warm starts' charges into the 12 Å model, held or released, and polish.
  - Does the scale hold, and do the forces recover to 12 Å's?
  - About 2 h a run.
- **(e) MACE-LES from two starts** (job 13's (b)), with the prediction above.
