# E4, job 12: E4's model polished, its response precomputed — pre-registration

*23 Nov 2026, registered at this commit's time, before any run of job 12 and before any of its numbers. Vadic's go of 23
Nov ("yes you can do a now"). Read against:*

- *`e4-job11-results.md` §4, §5 and §7(a): this job is (a) there;*
- *`e4-job8-results.md` §4: E4's model, D′ at the 2.0 Å c6 band;*
- *`e4-job10-results.md` §4 and §6: the training protocol is under-converged for every model.*

## 1. Why

**Job 11's findings.**

- **The long polish takes 8 % off the field-term dagger at the 2.0 Å band** (M_16pm_us_d20) in 1000 full-batch L-BFGS
  steps, without over-fitting.
- **D′ got only 277 and 290 steps in its 600 minutes** (124–130 s a step against the dagger's 12 s). It took 2.7–3.1 %
  off, and K11.4 failed by 0.3 points at 64 molecules. Its objective was falling at the dagger's pace.
- **D′'s infrared did not move:** ε_∞, the far probes and T5 were job 8's to the printed digit.

**Why D′'s steps cost ten times the dagger's.** Each carries the mutual response: a GMRES solve with implicit derivatives,
through the forces and then the parameters.

**Why that cost is not needed.** D′ is the dagger plus its response (`e4-job11-results.md` §5).

- Its read-out reads the pinned sources' band field with coefficients from the species alone (`Rung.bf_species`). It
  reads nothing E₀ or the field term learns, and it is held through training.
- So the response's energy and forces on each training frame are constants of the polish.
- The polish trains E₀ and the field term alone: exactly the dagger's parameters, which D′ and the dagger draw from the
  same keys.

**So D′'s polish is the dagger's polish against the remainder.** The response is computed once per frame, as D′'s energy
and forces less the dagger's at the same parameters. It is taken off the targets, and the dagger is trained against
the remainder. Every residual, and so the objective and its gradient, are D′'s exactly, at every step.

**The question.** With the steps it needs, does the long polish take a comparable share off E4's model, its infrared
untouched? If it does, the polish becomes part of E4's model's training.

## 2. What changes (v0.0.36), fixed before any run

- **`training.response_precomputed: {dagger: M_16pm_us_d20}`** (`scripts/e4_water.py`).
  - It requires the rung's response to be a constant of the rest (`model.response_constant`): a read-out from the
    species alone on pinned sources, held.
  - It requires the dagger to be the rung less its response (`model.is_response_free`): every field of the `Rung` but
    the read-out's own the same.
  - It computes the response's energy and forces on every training frame (`response_constants`).
  - It checks them: recomputed on 2 frames with other shared parameters (another seed's initialisation), they must
    agree within 10⁻⁶ of their rms.
  - It checks the objective: D′'s and the dagger's against the remainder must agree within 10⁻⁶ at the start and at the
    end. The run stops otherwise.
  - It polishes the dagger against the remainder from D′'s parameters less the read-out, and puts D′ back together with
    the held read-out before any reading.
  - The json records the checks (`training.response_precomputed`) and the fit's own time (`training.fit_time`).
- **The tests (`tests/test_job12.py`), on fake cells.**
  - The response does not change when every other parameter does: 10⁻¹⁴ of its rms.
  - The two objectives agree to 10⁻¹².
  - Two L-BFGS steps against the remainder give D′'s own two steps with the read-out held: the same objective at every
    step and the same parameters to 10⁻¹².
  - A read-out from E₀'s features fails the check.
- **`configs/e4_job12.yaml`** is `configs/e4_job11d.yaml` (job 8's D′ settings, job 5's read-out held) with three
  changes:
  - seeds 0–3;
  - the polish's budget 360 minutes, a safety bound;
  - the response precomputed.
  - Still no Adam, 1000 L-BFGS steps from job 8's runs, the objective traced every 25 steps.
- **The reader:** `scripts/e4_job12.py` reads K12.0–K12.3 and the model rule, without JAX.

## 3. The runs

`run_e4_job12.sbatch` has four tasks, one GPU each: D′ seeds 0, 1, 2 and 3, one run each. Then:

```sh
python scripts/e4_job12.py --out_dir results/e4_job12 --job8_dir results/e4_job8 --job11_dir results/e4_job11
python scripts/e4_water.py summary --out_dir results/e4_job12
```

## 4. Registered

**The readings.**

- Rigid forces on the unseen frames (job 7's pooling).
- Seed means over the four seeds against job 8's same four seeds (K12.1, K12.2), and over seeds 0 and 1 against job
  11's polished field-term dagger (K12.3).

| | reading | band | prediction |
|---|---|---|---|
| **K12.0** | exactness, in every run: the response's change under other shared parameters (energy and forces); D′'s objective against the dagger's on the remainder, at the start and the end; and, for seeds 0 and 1, the objective at the start against job 11's D′ at its step 0 | each within 10⁻⁶ (relative) | holds |
| **K12.1** | the polished D′ (M_16C2pm_us_d20, E4's model), against job 8's runs of the same seeds | at least 3 % lower at every size | holds |
| **K12.2** | its infrared against job 8's: ε_∞ on the held-out liquid; the far probes' non-additive error at 10, 12 and 15 Å; T5 | ε_∞ within ±0.002; the probes and T5 within ±0.5 points | holds |
| **K12.3** | the response's bulk gain at equal steps: the polished D′ against job 11's polished field-term dagger (1000 steps each), seeds 0 and 1 | at least 1.5 % lower at 128 and 256 molecules | holds |

**The model rule, registered with them.** If K12.0, K12.1 and K12.2 hold, E4's model is D′ at the 2.0 Å band with the
long polish. Its training is then job 8's, followed by 1000 L-BFGS steps on every training frame with the response
precomputed. This is K11.4's rule with the steps it needed.

**Why these bands, and these predictions.**

- **K12.0's 10⁻⁶.**
  - Only the GMRES solve's tolerance (10⁻¹⁰ of its residual) and rounding separate the two objectives. On fake cells
    they agree to 2×10⁻¹⁶.
  - A response that read the shared parameters would change by order one: the test's read-out from E₀'s features
    changes by more than 10⁻⁶.
  - Job 11's D′ started from the same parameters (job 8's runs, the same held read-out), so its step 0 is the same
    objective.
- **K12.1's 3 %, and the prediction.**
  - The dagger lost 8 % in 1000 steps. D′'s objective fell at the dagger's pace while it was measured (6.4 % and 5.9 %
    in its first 250 steps, against 5.9 % and 4.8 %).
  - Forces fall by about half the objective's relative change (the dagger: −16 % against −8 %; D′ at 277–290 steps:
    −6 % against −3 %).
  - So the prediction is a fall of about 7–8 % at every size. 3 % is K11.4's band, about three times D′'s seed spread
    (0.004–0.007 at 0.48).
- **K12.2's bands are K11.5's.** The read-out is held, so ε_∞ and T5 are fixed by it, and job 11's 277–290 steps left
  the far probes unchanged to the printed digit.
- **K12.3's 1.5 %, and the prediction.**
  - Before the polish, D′ read 1.0 / 3.0 / 2.8 % below the field-term dagger at 64 / 128 / 256 molecules (job 8, seeds
    0 and 1).
  - At equal steps both should lose about the same share, so the gain stays at about 3 % at 128 and 256.
  - 1.5 % is about one standard error of the difference of two seed means of two seeds.
  - At 64 molecules the gain was 1 % before the polish, within the seed spread, so 64 is read beside.

## 5. Read beside, not registered

- **The polished D′ against the polished dagger at 64 molecules.**
- **The objective path** every 25 steps (printed every 250), against job 11's D′ over its 277–290 steps and against the
  dagger's 1000.
  - In exact arithmetic the paths are job 11's, step by step: the same objective, gradient and L-BFGS state.
  - In floating point they agree until rounding sends a line search another way.
- **The cost of a step,** against job 11's 124–130 s, and the precomputation's time.
- **The training frames' forces against the unseen ones** (over-fitting).
- **The energies.**

## 6. What would change the reading

- **K12.0 fails.** The job is void: the precomputed polish was not D′'s. Nothing else is read until the cause is found.
- **K12.1 holds** (predicted). With K12.2 the model rule fires.
- **K12.1 fails, with the objective falling as the dagger's.** D′'s forces gain less from the same fall in the
  objective: its response already carries what E₀ learns in the polish. The model stays as job 8 trained it.
- **K12.2 fails.** The polish moves D′'s infrared, which a held read-out should not allow. That points at E₀ in the
  probes at 10 Å. The model rule does not fire.
- **K12.3 fails.** The response's bulk gain was E₀'s under-training: an E₀ trained out does without it in bulk. The
  response then earns its place by the infrared (ε_∞, the far probes, T5) alone.

## 7. Cost

- **The precomputation:** D′'s energy and forces and the dagger's on the 180 training frames, one frame at a time, and
  the two checks. A few minutes.
- **The polish:** 1000 steps at the dagger's 11.5–12.2 s a step (job 11), about 3.3 h.
- **The readings:** job 11's D′ spent about 0.3 h outside its polish.
- **So about 4 h a run.** The sbatch asks 8 h a task, four tasks: about 16 GPU hours.

## 8. Numbers before any run

**Job 8's runs of E4's model** (D′ at the 2.0 Å band), the starts and K12.1's reference:

- four-seed mean forces 0.478 / 0.485 / 0.487, seed spread 0.004–0.007;
- seeds 0 and 1: 0.474 / 0.478 / 0.480;
- ε_∞ 1.6893; the far probes 2.23 / 1.98 / 2.66 %; T5 4.31 %.

**Job 11's field-term dagger, polished** (seeds 0 and 1, 1000 steps), K12.3's reference:

- 0.440 / 0.453 / 0.456, from job 8's 0.479 / 0.493 / 0.494 (−8.2 / −8.0 / −7.8 %).

**Job 11's D′, polished within its budget** (seeds 0 and 1, 277 and 290 steps):

- 0.461 / 0.464 / 0.466 (−2.7 / −3.1 / −3.1 %).

**On fake cells (`tests/test_job12.py`, CPU):**

- the response's change under other shared parameters: 10⁻¹⁴ of its rms (forces), 0 (energy);
- D′'s objective against the dagger's on the remainder: 2×10⁻¹⁶ apart;
- three L-BFGS steps: the parameters 10⁻¹⁶ apart.
