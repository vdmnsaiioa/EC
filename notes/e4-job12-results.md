# E4, job 12: E4's model polished, its response precomputed — results

*Read 25 Nov 2026 against `e4-job12-preregistration.md`, registered on 23 Nov before any run (def0222 on GitHub). The
job ran on Niflheim at v0.0.37 (425a336) as the registered four tasks, on RTX 3090s. It was read with
`scripts/e4_job12.py` and `scripts/e4_water.py summary`.*

- *Rigid forces in kcal/mol/Å are read on the unseen frames, seed means over the four seeds against job 8's same four
  seeds, and over seeds 0 and 1 against job 11's polished field-term dagger.*
- *Run times: 12,990–13,149 s (3.6 h) a run, against the estimated 4 h. The response's precomputation took 153–157 s, the
  polish 11.8–11.9 s a step, against job 11's 124–130 s for D′.*

## 1. The registered readings

| | band | Niflheim | verdict |
|---|---|---|---|
| **K12.0** | exactness within 10⁻⁶ | the response under other shared parameters: ≤ 3.7×10⁻¹⁴; the two objectives at the start ≤ 2.7×10⁻¹⁴ and at the end ≤ 2.8×10⁻¹⁴; the start against job 11's step 0 ≤ 2.7×10⁻¹⁴ | **holds** |
| **K12.1** | at least 3 % below job 8, at every size | 0.444 / 0.449 / 0.451 against 0.478 / 0.485 / 0.487: −7.0 / −7.5 / −7.4 % | **holds** |
| **K12.2** | the infrared unchanged | ε_∞ 1.6893; probes 2.23 / 1.98 / 2.66 %; T5 4.31 %: job 8's to the printed digit | **holds** |
| **K12.3** | at least 1.5 % below the polished field-term dagger at 128 and 256 | −1.8 / −1.8 % | **holds**, narrowly |

**The model rule fires. E4's model is D′ at the 2.0 Å band with the long polish:** job 8's training followed by 1000
full-batch L-BFGS steps with the response precomputed. Its forces are 0.444 / 0.449 / 0.451 kcal/mol/Å (seed spread
0.004), its energies 0.020 / 0.014 / 0.008 kcal/mol per molecule, and its infrared is job 8's.

## 2. The precomputed polish

- **It is D′'s polish, to rounding.**
  - The response's energy and forces are the same in every seed (rms 239.2823 kcal/mol a cell and 2.8423 kcal/mol/Å),
    as they must be for one held read-out on the same frames.
  - Recomputed with another seed's initialisation of everything else, they change by at most 3.7×10⁻¹⁴ of their rms.
  - D′'s objective and the dagger's against the remainder agree to 11 printed digits at the start and the end of every
    run, and the start agrees with job 11's step 0.
- **It is ten times cheaper:** 11.8–11.9 s a step against 124–130 s, so the 1000 steps took 3.3 h instead of the 36 h they
  would have taken in job 11's way.
- **The path parts from job 11's by rounding, not by design** (the objective, job 12 against job 11):

| step | 0 | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|
| seed 0 | −1×10⁻¹⁴ | +4×10⁻⁷ | +0.5 % | +1.5 % | +1.4 % | +0.6 % |
| seed 1 | −3×10⁻¹⁴ | −3×10⁻⁵ | +0.5 % | +0.7 % | +1.4 % | +0.7 % |

  - The two paths agree to 10⁻⁷–10⁻⁵ at step 50 and part by about 1 % by step 150, then draw closer.
  - Job 11 solved the response with GMRES (tolerance 10⁻¹⁰) at every evaluation; job 12 does not. A difference at that
    level, carried through the line search's decisions, grows like this.
  - The end check is what matters: the model saved is D′, with its own objective.

## 3. What the polish bought

| | job 8 | polished | change |
|---|---|---|---|
| forces, four seeds | 0.478 / 0.485 / 0.487 | 0.444 / 0.449 / 0.451 | −7.0 / −7.5 / −7.4 % |
| final training loss | 18.50 / 18.61 / 19.17 / 19.12 | 15.82 / 16.26 / 16.34 / 16.00 | −13 to −16 % |
| energy per molecule | 0.022 / 0.016 / 0.008 | 0.020 / 0.014 / 0.008 | |

- **As predicted:** about 7–8 %, with the forces falling by half the objective's relative change. The field-term
  dagger lost 8 % to the same polish (job 11).
- **No over-fitting:** the training frames at 64 molecules read 0.438–0.443, the unseen 0.441–0.448.
- **The polish had not converged.** The objective fell 2–4 % in each run's last 225 steps.

## 4. The response's bulk gain, at equal steps

D′ against the field-term dagger, both polished for 1000 steps, seeds 0 and 1:

| | 64 | 128 | 256 | 64 → 256 |
|---|---|---|---|---|
| before the polish (job 8) | −1.0 % | −3.0 % | −2.8 % | D′ +1.3 %, dagger +3.1 % |
| after | **+0.5 %** | **−1.8 %** | **−1.8 %** | D′ +1.1 %, dagger +3.6 % |

- **K12.3 holds by 0.3 points; the prediction (about 3 %) was too high.**
- **At the training size the gain is gone.** +0.5 % is within the seed spread (0.004, 0.9 %). The dagger's training
  objective is lower than D′'s through the polish (job 11's path: 15.84 against 16.16 at step 750 for seed 0).
- **What remains is in going to larger cells.** From 64 to 256 molecules the dagger's error grows by 3.6 %, D′'s by 1.1 %,
  before the polish and after it alike.
- **The reading.** The dagger's E₀ and field term learn a substitute for the induction inside the 12.4 Å training cell,
  which their two 6 Å layers span. A polished E₀ does it as well as the response at that size. The substitute does not
  carry to larger cells; the response, a long-range term with coefficients matched on the far field, does.
- **So the response's case in bulk is its transfer in size,** beside the infrared (ε_∞, the far probes, T5), which no
  field-blind model gets right.

## 5. Notes to the pre-registration

- **The runs took 3.6 h,** against the estimated 4 h.
- **K12.3's prediction of a 3 % gain at 128 and 256 was too high;** the registered verdict (holds) stands, by 0.3
  points.
- **The path against job 11's** was read beside, as registered: it agrees at the start and to 10⁻⁵ at step 50, then
  parts by rounding.

## 6. What this says

- **E4's model takes the long polish.** It is 7 % better in bulk forces, with its infrared untouched.
- **Precomputing a held response is exact and ten times cheaper.** Any model whose response reads nothing it trains
  can be polished at its dagger's cost.
- **The training protocol from here on** is the polished one: 1000 full-batch L-BFGS steps after the standard training,
  for every model that enters the comparison (job 10, `e4-job10-results.md` §6).
- **The response's bulk gain is in size transfer.** At the training size, an E₀ trained out matches it; beyond the
  cell its E₀ cannot span, it does not. This is the bulk form of the infrared argument: what a local model learns in
  place of long-range physics holds where it was learned.
