# E4, job 15: the learned charges' scale after job 14 — results, arms A–D

*Read 26 Nov 2026 against `e4-job15-preregistration.md`, registered at 17:47 CET on 25 Nov before any run (ad61717 on
GitHub). Arms A–D ran on Niflheim at v0.0.40 as `run_e4_job15.sbatch`'s eleven tasks, on RTX 3090s. They were read with
`scripts/e4_job15.py` and `scripts/e4_water.py summary` on `results/e4_job15_{a,b,c,d}`. Arm E (MACE-LES,
`run_e4_job15_les.sbatch`) has not run: K15.9 and K15.10 are not read here.*

- *The end of a run with learned charges is c, its learned dipole on the test frames of 64 molecules over 1.864 D.
  Forces are rigid forces in kcal/mol/Å on the unseen frames, seed means over seeds 0 and 1.*
- *Run times: first stages 366–1452 s and polishes 628–7626 s, about 46 GPU hours in all (estimated 35). No run
  failed and no polish hit its budget.*
  - *The same model's time varies widely between tasks. A's 4.5 Å polishes took 3501–4654 s, where job 14's took
    1061–1104 s; C's pinned polishes took 628–676 s and its learned ones 6199–6744 s; D's grown M_1q110 took 683–702 s
    for 1000 steps at 12 Å.*
  - *The polish runs every step unless its budget line prints, and none did. The spread is the tasks' nodes', as far as
    the logs say.*

## 1. The registered readings

| | arm | band | Niflheim | verdict | predicted |
|---|---|---|---|---|---|
| **K15.1** | A | c*_res at least 0.03 above the flexible ends (largest 0.987) | c*_res = 1.123: 0.136 above | **holds** | holds |
| **K15.2** | A | every species-only end within ±0.05 of c*_res | 0.773, 0.842 (from 0.5); 1.053, 1.082 (from 1.5): −0.350, −0.281, −0.070, −0.041 | **fails** | fails |
| **K15.3** | A | the species-only gap at most half of job 13's (0.715, 0.717) | 0.280, 0.240 | **holds** | holds |
| **K15.4** | B | \|the gap\| at most 0.05 after 2000 steps | −0.004, +0.040 | **holds** | holds |
| **K15.5** | C | ≤ 0.15 on both seeds: reach; ≥ 0.25: depth | +0.000, +0.021 | **reach** | reach |
| **K15.6** | D | every grown warm start within ±0.05 of its start | +0.017, +0.016, −0.002, −0.022 | **holds** | holds |
| **K15.7** | D | the grown warm starts' forces at least 5 % below B's | from 0.5 −7.7 / −7.3 / −7.2 %; from 1.5 −3.8 / −3.9 / −4.1 % | **fails** | holds |
| **K15.8** | D | the grown M_1q110 within ±3 % of job 10's | +2.9 / +2.4 / +2.3 % | **holds** | holds |
| K15.9 | E1 | MACE-LES at units 0.05, 0.15 within ±10 % of the default's | — | not read | fails |
| K15.10 | E2 | released at 0.05, 0.15 within ±10 % of the control's | — | not read | fails |

Seven of the eight read hold or fail as predicted. K15.7 fails where it was predicted to hold, by about one point, for
the warm starts from 1.5 (§5).

## 2. A: the pinned minimum at 4.5 Å is a flat bottom at 1.05–1.2

| c | 0.7 | 0.9 | 0.95 | 1.0 | 1.05 | 1.1 | 1.2 | 1.3 | 1.45 |
|---|---|---|---|---|---|---|---|---|---|
| final loss (seed mean) | 47.47 | 39.13 | 37.32 | 35.34 | 33.92 | 33.42 | **33.31** | 42.43 | 63.55 |
| forces at 64 | 0.773 | 0.696 | 0.674 | 0.659 | 0.643 | 0.636 | 0.635 | 0.719 | 0.882 |

(0.7, 1.0, 1.1 and 1.45 are job 14's runs)

- **The bottom is flat from 1.05 to 1.2,** within 0.6 of 33.3, about the seeds' spread there (±0.4–0.75). The profile
  rises gently below it and steeply above: by 9 from 1.2 to 1.3.
- **The registered construction puts c*_res at 1.123** (the five points around the lowest, 1.05–1.45; curvature 732).
  A parabola through all nine gives 1.029 in c and 1.082 in c².
- **The pinned minimum lies well above the flexible learned charges' common end.**
  - Every point from 1.05 to 1.2 lies below the points at 0.95 and 1.0 by 1.4–4.0.
  - The four flexible ends at 4.5 Å (0.969–0.987, job 14) sit 0.06–0.08 below the bottom's lower edge and 0.14–0.15
    below c*_res.
- **The forces agree:** lowest at 1.1–1.2 (0.635–0.636).
  - The flexible warm starts from 1.5 read 0.602 at their own scale, 5 % below any pinned point.
  - The learned charges with their head are a different model with a different optimum. Their environment dependence
    lowers the mean dipole that the data want, by about 0.1 in c here.

## 3. A: by species alone the scale does not move under Adam, at any reach

| | from 0.5, seeds 0 / 1 | from 1.5, seeds 0 / 1 |
|---|---|---|
| after job 7's Adam, 4.5 Å (traced frames) | 0.524 / 0.524 | 1.473 / 1.473 |
| the same at 12 Å (job 13's S) | 0.522 / 0.542 | 1.471 / 1.468 |
| with the head at 4.5 Å (job 14) | 0.940 / 0.912 | 1.060 / 1.125 |
| polished, 4.5 Å | 0.773 / 0.842 | 1.053 / 1.082 |
| polished, 12 Å (job 13's S) | 0.632 / 0.625 | 1.348 / 1.342 |
| the last 250 steps' move, 4.5 Å | +0.060 / +0.043 | −0.021 / −0.059 |

- **Through Adam the species offsets alone move the scale 0.02–0.04, at 4.5 Å as at 12 Å.** With the head, the same
  training at 4.5 Å moved it 0.38–0.44.
  - The pull on the scale is as at every reach (−0.10, +0.12 in the first block of 500 steps).
  - So the short reach's mobility under Adam passes entirely through the charge head, as job 13 found at 12 Å.
- **The polish moves them 2.2–3.5 times further at 4.5 Å than at 12 Å:** +0.25 / +0.29 from below, −0.41 / −0.38 from
  above.
  - They are still moving (0.02–0.06 per 250 steps), with the objective falling 6–10 % per 250.
  - From above they reach the flat bottom (1.05–1.08); from below they stop 0.2–0.3 short.
- **Their fit carries the start.** The final losses read 6–12 % above the pinned profile at their own c, though the
  model by species is the pinned model with one free scale.
- **K15.2 fails from below, as predicted; K15.3 holds:** the gap is 0.24–0.28 against 0.72 at 12 Å.

## 4. B and C: reach, not depth

**B, one layer of 6 Å, polished another 1000 steps:**

| | from 0.5, seeds 0 / 1 | from 1.5, seeds 0 / 1 |
|---|---|---|
| after 1000 steps (job 14) | 0.963 / 0.932 | 0.975 / 1.035 |
| after 2000 steps (B) | 0.937 / 0.912 | 0.933 / 0.952 |
| the last 250 steps' move | +0.009 / −0.001 | −0.019 / −0.012 |
| forces at 64 / 128 / 256 | 0.613 / 0.617 / 0.619 (±0.049) | 0.522 / 0.532 / 0.535 (±0.014) |

- **The pairs meet:** gaps −0.004 and +0.040. All four moved down, the ones from 0.5 too, to 0.91–0.95. They meet
  0.1–0.2 below the 6 Å pinned profile's minimum (1.054 in c, 1.118 in c²), as at 4.5 Å.
- **The forces fell 7–10 % in the second 1000 steps.** From 1.5 they now read 0.522 / 0.532 / 0.535, level with job 7's
  E₀ (two layers of 6 Å) with pinned charges at c = 1.1 (job 10: 0.520 / 0.536 / 0.541), and 8 % below one layer of
  6 Å with those charges (0.565 / 0.582 / 0.584).

**C, two layers of 3 Å (a reach of 6 Å at job 7's depth):**

| | from 0.5, seeds 0 / 1 | from 1.5, seeds 0 / 1 |
|---|---|---|
| Adam, half the move made by step | 200 / 100 | 100 / 100 |
| after job 7's training (traced frames) | 1.145 / 1.149 | 1.138 / 1.159 |
| polished | 1.101 / 1.093 | 1.102 / 1.114 |

- **The scale is the most mobile yet.** Adam closes the whole gap of 1.0 (to −0.007 and +0.010); the polish ends at
  1.09–1.11 from both sides. The verdict is "reach".
- **C's model is weak.** Its forces are 1.04–1.47 kcal/mol/Å, twice the 12 Å model's at the same c. Its profile is 9.3
  times sharper than 12 Å's (c* 1.080, c*² L*″ 1329; in c², 1.142). The 3 Å cutoff leaves the first shell's O···O
  nearly unseen.
- **Here the learned ends sit at the pinned minimum** (1.09–1.11 against 1.08–1.14), unlike at one layer of 4.5 or
  6 Å, where they sit 0.1–0.2 below it.

**Together:**

- Every model that reaches 6 Å or less finds one scale from both starts: one layer of 4.5 Å, one layer of 6 Å, two
  layers of 3 Å.
- Job 7's E₀, reaching 12 Å in 12.4 Å cells, does not.
- Job 14 called depth the lever. Its comparison of 6 Å against 12 Å changed the reach as well as the layers, and C
  takes depth out: two layers do not freeze the scale at 6 Å.
- What freezes it is E₀'s reach against the cell. The caveat is C's weakness: its sharper profile also helps.

## 5. D: the route keeps the scale and nearly pays

| | from 0.5, seeds 0 / 1 | from 1.5, seeds 0 / 1 | M_1q110 |
|---|---|---|---|
| start: B's end (c) | 0.937 / 0.912 | 0.933 / 0.952 | (job 14's 6 Å run) |
| grown and polished (c) | 0.954 / 0.928 | 0.931 / 0.930 | — |
| the gap | −0.023 / +0.002 | | |
| forces, grown | 0.565 / 0.572 / 0.574 | 0.502 / 0.511 / 0.513 | 0.535 / 0.550 / 0.554 |
| against B's (or the 6 Å start's) | −7.7 / −7.3 / −7.2 % | −3.8 / −3.9 / −4.1 % | −5.3 / −5.5 / −5.1 % |
| the objective, last 250 steps | −3.3 / −5.0 % | −1.5 / −3.1 % | |

- **The scale survives the second layer.** It moves at most 0.022 in 1000 steps, and the gap stays within 0.023.
  - At 12 Å the polish moved a frozen scale 0.05–0.07 in the same steps (job 13).
  - The deep E₀ grown from a trained shallow one does not pull the scale toward a 12 Å minimum of its own.
- **From 1.5 the grown runs reach the best 12 Å learned charges.** 0.502 / 0.511 / 0.513 is:
  - within 0.6–2.2 % of job 10's M_1w100, started at c* and polished once (0.491 / 0.507 / 0.510);
  - 3.5–5.2 % below the 12 Å pinned control.
  - Their scale (0.93) is the one the shallow model found from both starts, not one set by a start.
- **Growth itself costs 2–3 %.** The grown M_1q110 is +2.9 / +2.4 / +2.3 % on job 10's M_1q110, trained deep from the
  start, with a final loss 8.7 % higher.
- **K15.7 fails for the warm starts from 1.5,** at −3.8 to −4.1 % against the band's 5 %.
  - The second layer had less to add than the pinned scan suggested (8–13 %).
  - B's second 1000 steps had already brought the one-layer learned model to the 12 Å pinned level (0.522).
- **The start marks the fit through every stage.** From 0.5 the forces stay 12–17 % worse than from 1.5, at the same
  scale: B 0.613 against 0.522, D 0.565 against 0.502. The seed spreads are large (±0.04 from 0.5).

## 6. Notes to the pre-registration

- **K15.7's prediction was wrong, narrowly** (§5). The registered verdict stands.
- **c*_res's window takes in the steep points at 1.3 and 1.45.** The bottom is flat over 1.05–1.2 (§2). K15.1 holds by
  0.136 and would hold against any point of that bottom.
- **K15.5's "reach" comes from a weak model** (§4): two layers of 3 Å have forces twice the 12 Å model's and a profile
  nine times sharper.
- **The cost was 46 GPU hours, not 35.**
  - C's learned runs polished in 1.7–1.9 h, not the 0.5–1 h guessed.
  - Times vary between tasks by up to ten times for the same model.

## 7. What this says

- **What freezes learned charges is E₀'s reach over the cell, not its depth.**
  - One layer of 4.5 Å, one of 6 Å and two of 3 Å all find one scale from starts at 0.5 and 1.5.
  - Two layers of 6 Å, reaching 12 Å in 12.4 Å cells, do not.
  - This sharpens jobs 13–14: E₀ commits to the scale when it can represent what the band does across the whole cell.
  - The test is the cell: the 12 Å model trained on 128-molecule cells (15.6 Å).
- **Learned charges with a head have their own minimum, 0.1–0.2 below the pinned one** at one layer of 4.5 and 6 Å
  (0.97–0.99 and 0.91–0.95 against 1.05–1.2). With the head held the minimum is the pinned one by construction, but the
  scale then barely moves under Adam at any reach (§3).
- **The route works.** Find the scale with a shallow E₀, then grow it.
  - The scale survives the second layer.
  - From 1.5 the result is as accurate as the best 12 Å learned charges, with a scale that no longer depends on the
    start.
- **The start still marks E₀.** From 0.5 every stage is 12–17 % worse in forces at the same scale. A learned-charge
  model's history shows in its fit even when its scale is identified.
- **For E4's model nothing changes:** its sources are matched and pinned. The best learned-charge model here (0.502) is
  13 % above E4's model (0.444).
- **For MACE-LES the prediction sharpens.** It reaches 12 Å in the same 12.4 Å cells. By §4 its scale should freeze,
  which K15.9–K15.10 read when arm E runs.

## 8. Next

- **Arm E, MACE-LES, as registered:** the smoke run on the devel partition, then `run_e4_job15_les.sbatch`. K15.9 and
  K15.10 complete this note.
- **Proposed, not registered:**
  - (a) **The cell.** Job 7's model (two layers of 6 Å) trained on 128-molecule cells, warm starts from 0.5 and 1.5. If
    reach over the cell is the lever, the scale moves there.
  - (b) **The start's mark.** B's runs from 0.5 continued, or a short Adam restart from the polished state: is the 12–17
    % a slow approach or a separate minimum of E₀?
