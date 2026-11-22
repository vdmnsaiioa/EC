# E4, job 11: the long polish — the profile under it, the low side continued, and E4's model — pre-registration

*22 Nov 2026, registered at 14:23 CET (this commit's time), before any run of job 11 and before any of its numbers.
Vadic's go of 22 Nov ("a and b"). Read against:*

- *`e4-job10-results.md` §3, §4, §6 and §8: this job is (a) and (b) there;*
- *`e4-job8-results.md` §4: E4's model, D′ at the 2.0 Å c6 band;*
- *`e4-job7-results.md` §2: the scan.*

## 1. Why

**Job 10's findings.**

- **Full-batch L-BFGS moved the learned charges most of the way to job 7's c*** (1.098, 2.047 D).
  - From above they arrived, within 3–7 %.
  - From below they slowed 20–27 % short, at 4–7 % of the remaining distance per 250 steps, while the objective kept
    falling.
- **The target is itself a short-polish reading.** Job 7's scan was trained with 50 L-BFGS steps. The long polish
  lowers the control's objective by 22 %, and where the profile's minimum sits under it is not known.
- **The long polish alone takes 12 % off the pinned control's forces,** without over-fitting. Job 7's training, and
  every job's since job 1, is under-converged for E₀.

**Three questions.**

- **(a1)** Where is the profile's minimum under the long polish?
- **(a2)** On the low side, is the approach very slow, or have the charges stopped?
- **(b)** Does the long polish take a comparable share off E4's model? If it does, with the infrared unchanged, the
  polish becomes part of E4's model's training.

## 2. What changes (v0.0.35), fixed before any run

- **A wall-clock budget for the polish** (`train.fit(lbfgs_max_seconds=...)`, `polish_lbfgs(max_seconds=...)`,
  `training.lbfgs.max_minutes`).
  - The polish stops after the step that spends its budget, and the json records the steps done.
  - Needed because D′'s L-BFGS steps carry the mutual response and cost about 15 times an M_1 step.
- **The start by rung** (`training.start_from.by_rung`): one config polishes runs from several directories.
- **The objective traced through the polish for any rung** (`training.lbfgs_trace` in the json), every 25 steps. Before
  this only learned charges were traced.
- **`configs/e4_job11.yaml`, on job 7's settings:** no Adam, 1000 L-BFGS steps within 300 minutes.
  - M_1q070, M_1q100 and M_1q145 from job 7. M_1q110 was polished in job 10.
  - M_1w050 and M_1w150 from job 10: another 1000 steps.
  - The field-term dagger at the 2.0 Å band, M_16pm_us_d20, from job 8.
- **`configs/e4_job11d.yaml`, on job 8's D′ settings:** D′ at the 2.0 Å band (M_16C2pm_us_d20, E4's model) from job 8,
  with job 5's read-out held. Up to 1000 steps within 600 minutes; about 350 steps are expected (§7).
- **Seeds 0 and 1,** paired with the starts' seeds.
- **The reader:** `scripts/e4_job11.py` reads K11.1–K11.5 and the model rule, without JAX.

## 3. The runs

`run_e4_job11.sbatch` has five tasks, one GPU each:

- **0 and 1:** the polished scan and the continued warm starts, seeds 0 and 1, five runs each;
- **2:** the field-term dagger, seeds 0 and 1;
- **3 and 4:** D′, seeds 0 and 1.

Then:

```sh
python scripts/e4_job11.py --out_dir results/e4_job11 --job7_dir results/e4_job7 --job8_dir results/e4_job8 \
    --job10_dir results/e4_job10
python scripts/e4_water.py summary --out_dir results/e4_job11
```

## 4. Registered

**The readings.**

- Rigid forces on the unseen frames (job 7's pooling).
- The learned dipole on the test frames of 64 molecules.
- Seed means over seeds 0 and 1.
- **The references are the starts:** job 8's runs of the same seeds for the E4-class models, job 10's for the warm
  starts.

| | reading | band | prediction |
|---|---|---|---|
| **K11.1** | the polished profile's minimum c*_lb: the parabola through the polished scan's final losses at c = 0.7, 1.0, 1.1 (job 10) and 1.45 | within ±0.05 of job 7's c* = 1.098 | holds |
| **K11.2** | job 10's two M_1w050 runs, another 1000 steps: the learned dipole's move up | at least +0.06 D on each | **fails**: they have stopped |
| **K11.3** | the field-term dagger at the 2.0 Å band (M_16pm_us_d20), polished, against job 8's runs of the same seeds | at least 5 % lower at every size | holds |
| **K11.4** | D′ at the 2.0 Å band (M_16C2pm_us_d20, E4's model), polished, against job 8's runs of the same seeds | at least 3 % lower at every size | holds |
| **K11.5** | the polished D′'s infrared against job 8's: ε_∞ on the held-out liquid; the far probes' non-additive error at 10, 12 and 15 Å; T5 | ε_∞ within ±0.002; the probes and T5 within ±0.5 points | holds |

**The model rule, registered with them.** If K11.4 and K11.5 hold, E4's model is D′ at the 2.0 Å band with the long
polish. Its training is then job 8's, followed by L-BFGS on every training frame as in job 11.

**Why these bands, and these predictions.**

- **K11.1's ±0.05.**
  - The scan's points lie 0.1 apart in c. Job 7's forces put the minimum at c = 1.07–1.10, and its objective at
    1.10.
  - A shift of 0.05 is half the step from c = 1.0 to 1.1. It would show in the warm starts: the run started at c* (job
    10's M_1w100 seed 0) did not move toward 2.047 D, and stayed at c = 1.03.
  - **Prediction: holds.** The polish lowered the control's objective by 22 % without changing what the charges do at
    long range, which is what sets the scale.
- **K11.2's +0.06 D.**
  - In the last 250 steps of job 10, M_1w050 moved +0.040 and +0.015 D. At the slower of those paces, held for 1000
    steps, that is +0.06 D. A run still approaching at job 10's final pace passes.
  - **Prediction: fails.** The pace fell by a factor of 3–10 over job 10's 1000 steps (§8). The restarted polish
    rebuilds its curvature memory, and its first steps may move a little more, which is why the band is not set lower.
- **K11.3's 5 %.** The pinned control lost 12 % to the long polish. The field-term dagger's E₀ also reads the pinned
  charges' field, so it has more to train. 5 % allows for a smaller share and is about two standard errors at the
  field-term dagger's seed spread in job 2 (0.018).
- **K11.4's 3 %.**
  - D′ gets about 350 L-BFGS steps within its 600 minutes, not 1000 (§7).
  - In job 10 the objective fell 4–20 % in the first 250 steps, and forces fall by about half the objective's relative
    change (the control: −22 % against −12 %).
  - 3 % is about three times D′'s seed spread (0.004–0.007 at 0.48).
- **K11.5's bands are K8.5's.** The read-out is held, so ε_∞ and T5 are fixed by it. The far probes at 10–15 Å are mostly
  beyond E₀'s reach, but its two layers reach 12 Å, so the band allows a little change.

## 5. Read beside, not registered

- **The polished scan's forces** at every size, and their minimum.
- **The from-above runs continued** (M_1w150): whether they settle at c*_lb.
- **Every polish's objective path** every 25 steps (printed every 250), the steps done, and the training frames'
  forces against the unseen ones (over-fitting).
- **The response's bulk gain after the polish:** the polished D′ against the polished field-term dagger, against job 8's
  1–2 %.
- **The learned dipole's path** for the continued warm starts.
- **The energies, the probes, T5.**

## 6. What would change the reading

- **K11.1 fails, with the minimum lower.** The target was a short-polish artefact.
  - Job 10's verdicts are re-read against c*_lb.
  - If the minimum moved to about 1.03, job 10's warm starts from above and at c* had already arrived, and only the low
    side is short.
- **K11.1 fails, with the minimum higher.** The low side is further short than job 10 said.
- **K11.2 holds.** The low side is still approaching, slowly. Full-batch L-BFGS is the recipe, at a cost in steps.
- **K11.2 fails** (predicted). The low side has stopped short of the minimum: the runs started small have minima of
  their own. Learned charges then need a start within about 20 % of the data's scale, or a shallower valley: the
  charges' band with its own edge (job 9's option (b)).
- **K11.3 holds and K11.4 fails.** The polish helps the E4-class model without the response, and D′'s 350 steps were not
  enough, or the response's model is already nearer convergence. Read the objective path.
- **K11.4 holds and K11.5 fails.** The polish moves D′'s infrared, which is not expected with the read-out held. That
  points at E₀ in the probes at 10 Å. The model rule does not fire.
- **Both K11.3 and K11.4 fail.** The 12 % was the M_1 family's. E4's model is near convergence as trained.

## 7. Cost

- **The M_1 polishes:** job 10's took 1.7–2.1 h a run. Tasks 0 and 1 have five runs each: about 9–10 h.
- **The field-term dagger's steps** cost about 2.7 times an M_1 step (job 2: 0.5 h a training against job 7's 11
  minutes). 1000 steps take about 4–5 h, within the 300-minute budget. Task 2: at most 11 h.
- **D′'s steps** cost about 17 times an M_1 step (job 5: 3–3.6 h a training). Its 600 minutes buy about 350 steps.
  Tasks 3 and 4: at most 11.5 h each, with the readings.
- **In all:** about 50 GPU hours. The sbatch asks 14 h a task, as jobs 8 and 10 did.

## 8. Numbers before any run

**Job 10's low side** (the learned dipole on the traced frames, D):

| run | 0 | 250 | 500 | 750 | 1000 | the last 250 steps' move | the first 250 steps' move |
|---|---|---|---|---|---|---|---|
| M_1w050 seed 0 | 1.213 | 1.361 | 1.451 | 1.463 | 1.503 | +0.040 | +0.148 |
| M_1w050 seed 1 | 1.315 | 1.469 | 1.581 | 1.633 | 1.648 | +0.015 | +0.154 |

The pace fell by a factor of 3.7 and 10 over the 1000 steps.

**The starts of (b),** job 8's runs of seeds 0 and 1, read from job 8's jsons by the reader:

- the field-term dagger at the 2.0 Å band: four-seed mean 0.483 / 0.495 / 0.497;
- D′ at it: four-seed mean 0.478 / 0.485 / 0.487, seed spread 0.004–0.007, ε_∞ 1.6893, the far probes 2.23 / 1.98 /
  2.66 %, T5 4.31 %.

**The control's polish (job 10):**

- forces 0.592 / 0.606 / 0.612 → 0.520 / 0.536 / 0.541 (−12 %);
- the objective 27.5 → 21.5 (−22 %);
- training frames and unseen frames alike.
