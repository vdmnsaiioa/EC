# E4, job 14: the scale and E₀'s reach — pre-registration

*25 Nov 2026, registered at this commit's time, before any run of job 14 and before any of its numbers. Vadic's go of 25
Nov. Read against:*

- *`e4-job13-results.md` §5, §6 and §8: the walls, the scale's mobility, what it says;*
- *`learned-charges-literature.md` (project): the LES water model and Ibrahim's pLES;*
- *`les-vs-eft-what-is-provable.md` §5 (project): representability is not identifiability.*

## 1. Why

**What job 13 found.** Learned charges trained with E₀ do not find the data's scale.

- With E₀ held, the curvature along the charges' scale is 1.087×10⁶ c⁴ for every state, E₀-independent: the band's own
  contribution, |v_C(c)|².
- With E₀ re-fitted, the profile's curvature at c*_lb = 1.07 is c*² L*″ = 143.
- So the part of the band's contribution that identifies the scale, the part E₀ cannot absorb, is 1 % of it in amplitude
  (a ratio of 10⁴ in curvature).
- The scale moves only together with E₀, and moves less the longer E₀ has trained.

**What the literature adds.**

- The LES papers report learned charges that are physically meaningful: Born effective charges after one rescaling by
  √ε∞. Their water model has a 4.5 Å cutoff and no message passing: a local model far shorter than its cells.
- Ibrahim (arXiv:2611.06575) finds the scale only weakly constrained in 64-molecule cells. The compensation that fixes it
  acts over about 1 nm, which those cells do not probe.
- Our E₀ has two layers of 6 Å: a reach of 12 Å in 12.4 Å cells. It can represent almost the whole band.

**The hypothesis (H_R).** The identifying part is the part of the band's contribution beyond E₀'s reach. Shorten the
reach and that part grows:

- the profile sharpens against the walls, which do not change (§9);
- the scale's pull is no longer one E₀ can cancel, so learned charges become mobile and find the data's scale.

**A rough estimate.**

- The rms field from uncorrelated molecular dipoles beyond a radius R scales as μ (4πn/3)^½ R^−3/2.
- Relative to R = 12 Å, that is 4.4 times larger in amplitude at 4.5 Å and 2.8 times at 6 Å: 19 and 8 times in
  curvature.
- Screening and correlations change the numbers, not the order.
- If E₀'s capacity, not its reach, sets the 12 Å value, the gains are larger.

## 2. What changes (v0.0.38), fixed before any run

- **`model.layers`:** E₀'s message-passing layers.
  - It sets `Rung.e0["T"]` (`e3_water.configure`) and is recorded in the model record when it is not 2.
  - The default is 2, so every earlier rung, record and cached read-out key is unchanged.
- **Two reaches:**
  - **4.5 Å:** one layer of r_cut = 4.5 Å, as the LES water model;
  - **6 Å:** one layer of 6 Å, job 7's cutoff without message passing.
- **At each reach, job 7's scan and warm starts:**
  - the pinned O/H charges at c = 0.7, 1.0, 1.1 and 1.45 (M_1q070, M_1q100, M_1q110, M_1q145);
  - learned charges from 0.5 and 1.5 (M_1w050, M_1w150).
- **Trained as at 12 Å:**
  - the first stage is job 7's training traced as job 9 traced it (`configs/e4_job14s_r45.yaml`, `e4_job14s_r6.yaml`);
  - then jobs 10 and 11's polish of 1000 full-batch L-BFGS steps (`configs/e4_job14_r45.yaml`, `e4_job14_r6.yaml`).
- **Seeds 0 and 1.**
- **The 12 Å reference:** job 11's polished scan with job 10's M_1q110, and job 10's polished warm starts.
- **The reader:** `scripts/e4_job14.py` reads K14.1–K14.4, without JAX.

## 3. The runs

`run_e4_job14.sbatch` has six tasks, one GPU each. Each runs its first stages, then their polishes.

- **0 and 1:** the scan at 4.5 Å, seeds 0 and 1 (four runs each);
- **2:** the warm starts at 4.5 Å, seeds 0 and 1;
- **3, 4, 5:** the same at 6 Å.

Then:

```sh
python scripts/e4_job14.py --job9_dir results/e4_job9 --job10_dir results/e4_job10 --job11_dir results/e4_job11
python scripts/e4_water.py summary --out_dir results/e4_job14_r45
python scripts/e4_water.py summary --out_dir results/e4_job14_r6
```

## 4. Registered

**The readings.**

- The profile at a reach is the parabola through its polished scan's final training losses, seed means, at c = 0.7,
  1.0, 1.1 and 1.45 (job 11's construction). Its curvature is c*² L*″ at its minimum c*.
- The end of a warm start is its learned dipole on the test frames of 64 molecules over 1.864 D.
- The gap is a seed's end from 1.5 less its end from 0.5.

| | reading | band | prediction |
|---|---|---|---|
| **K14.1** | the profile's curvature at 4.5 Å against 12 Å's (143) | at least 5 times | holds |
| **K14.2** | the profile's curvature at 6 Å against 12 Å's | at least 2 times | holds |
| **K14.3** | at 4.5 Å, every polished warm start's end | within ±0.05 of the reach's own c*, which must lie in 0.7–1.45 (else not read) | holds |
| **K14.4** | the gap, seed by seed | smaller at 6 Å than at 12 Å (0.324, 0.288), and smaller at 4.5 Å than at 6 Å | holds |

**Why these bands, and these predictions.**

- **K14.1 and K14.2's 5 and 2 times** are about a quarter of the estimate's 19 and 8 (§1). A reach that leaves the
  profile less than that has not changed what identifies the scale.
- **K14.3's ±0.05** is jobs 11–13's band: half the scan's step.
  - At 4.5 Å the estimate leaves the walls about 500 times the profile, not 10⁴.
  - The prediction is that the warm starts then arrive from both sides, as the LES charges appear to.
- **K14.4** reads the start-dependence without a band: the gap was 0.29–0.32 at 12 Å after the same training. The
  prediction is that it shrinks with the reach.

## 5. Read beside, not registered

- **c* at each reach.** Whether the scale the data pick depends on how much E₀ can absorb.
- **The walls at the warm starts' ends** against job 13's law, 1.087×10⁶ c⁴. They should not depend on the reach; the
  head's variation adds to them.
- **The first stage's ends against job 9's at 12 Å:** the scale's mobility through Adam.
- **The paths through the polish**, the last 250 steps' move.
- **The cost of the short reach:** forces and energies at every size against 12 Å's, scan and warm starts alike.

## 6. What would change the reading

- **All four hold (H_R).** The reach is the lever.
  - When E₀ cannot span the training cell, bulk energies and forces identify the charges' scale, and learned charges
    find it.
  - This reconciles the LES results with jobs 6–13.
  - It is Theorem 3's angle measured as a function of reach over cell, with the short reach's cost in forces beside it.
- **K14.1 holds and K14.3 fails.** The identifying part grows, but the scale still stays where it starts. The dynamics
  (E₀'s commitment) dominate even with a sharper profile, and the scale must be set from outside the gradient: the
  matching.
- **K14.1 fails.** E₀'s reach is not what sets the 1 %.
  - Either its capacity does, or the scale is fixed by compensation over distances the 12.4 Å cells hold at any reach
    (Ibrahim's 1 nm).
  - The test is then the cell: training at 128 molecules.
- **K14.3 holds and K14.1 fails.** The scale becomes mobile without a sharper profile: a one-layer E₀ commits less. Read
  the first stage's ends and the paths.
- **K14.4 fails with K14.1 holding.** The ends still depend on the start where the profile is sharper; read the paths
  for a slow approach.

## 7. Not tested here

- **The cell:** training at 128 molecules, where the two-layer E₀ no longer spans the cell.
- **MACE-LES:** its learned scale from two starts (`e4-job13-results.md` §9(b)).

## 8. Cost

- **The first stages:** job 7's scan ran 8–14 minutes, job 9's traced warm starts 20–23 minutes.
- **The polishes:** jobs 10, 11 and 13 ran 1.7–2.4 h a run.
- **One layer is not much cheaper.** On CPU an objective and gradient on 4 cells costs 10–13 % less than with two (E₀
  is built over every pair; the band and the forces dominate).
- **Tasks:** about 9 h (scan) and 11 h (warm starts). About 57 GPU hours in all. The sbatch asks 14 h a task.
- **The 6 Å tasks (3–5) can run after the 4.5 Å ones.** K14.2 and K14.4 need them.

## 9. Numbers before any run

**The 12 Å reference:**

- the polished profile: 30.75 / 23.63 / 21.45 / 31.44 at c = 0.7 / 1.0 / 1.1 / 1.45; c* = 1.070; c*² L*″ = 143;
- the walls at c*: 1.087×10⁶ × 1.07⁴ = 1.42×10⁶, 10⁴ times the profile;
- the warm starts after job 7's training and job 10's polish:
  - from 0.5: 0.807 / 0.884;
  - from 1.5: 1.131 / 1.172;
  - the gaps 0.324 / 0.288;
- after job 7's training alone (job 9): 0.651 / 0.706 and 1.320 / 1.375.

**On CPU:**

- E₀ of one layer has 20,032 parameters, against two layers' 38,272;
- the L-BFGS objective and gradient on 4 cells of 64 molecules take 10.9–11.2 s, against 12.5 s;
- the memory per chunk of 4 is 1.46 GB, against 1.88 GB.
