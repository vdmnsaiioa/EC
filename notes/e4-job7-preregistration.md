# E4, job 7: where the data put a field-blind charge scale, and whether learned charges reach it — pre-registration

*19 Nov 2026, registered at 00:30 CET (this commit's time), before any run of job 7 on MB-pol's data and before
any of its numbers. Vadic's go of 18 Nov. Read against:*

- *`e4-job6-results.md` §4 and §5 (the learned charges did not converge; this job was proposed there);*
- *`e4-job1-results.md` §4b (the dagger's gain, its attribution not checked);*
- *the project's `path-integral-ir-sector.md` §5 (the lemma).*

## 1. Why

**Job 6's outcome.**

- The learned charges stopped where the seed and the unit left them: 0.37–1.10 D, a fifth to three fifths of MB-pol's
  permanent 1.864 D.
- They bought nothing resolved in forces.
- The dagger, with MB-pol's site charges and c6 pinned at the permanent scale, bought 20 % in job 1.

**Two readings are open** (`e4-job6-results.md` §4).

- **The optimisation.** The data favour a scale near the pinned charges', and the learned charges do not get there.
- **The data.** The 64-molecule cells barely constrain the scale. E₀ reaches the whole training cell, so it can absorb
  what the charges leave.
- The evidence was split. M_1uc's larger charges did no better, and the scale followed the seed. Both point to the
  data. The dagger's 20 % points to the optimisation, unless most of it is the c6.

**Job 7 separates them.**

- **(a)** Pinned charges at five scales map the loss and forces against the scale, without c6.
- **(b)** Learned charges started at three scales test convergence from both sides.
- **Beside them**, MB-pol's own site charges without c6 split the dagger's gain between its charges and its c6.

**A departure from job 6's registered branch.** Job 6 registered "a longer schedule" for "not converged". It is not
taken.

- The scale followed the seed and the unit, and M_1uc stopped well inside its step budget. So a longer run at the same
  units would likely still leave the scale to the path.
- Starting from both sides tests convergence directly, and the scan gives the target.
- This is stated here, before any run.

## 2. What changes (v0.0.31), fixed before any run

- **O/H charges with MB-pol's dipole** (`water.oh_equivalent_charges`).
  - MB-pol's site model with its M-site's charge moved onto the atoms by the site's weights: 1 − γ on O and γ/2 on each
    H.
  - That gives O −0.6623 and H +0.3312 e.
  - The molecule's charge and dipole (1.864 D) are the site model's at every geometry. The quadrupole is not.
  - `pin_rigid_site_model` pins them as the atoms' charges on every E4 cell and probe. They enter only the rungs that pin
    atom charges without a site model.
- **The scan** (`Rung.q_scale`): M_1q040, M_1q070, M_1q100, M_1q110, M_1q145.
  - Those charges times c, pinned. The Coulomb band only: no c6, no response. E₀ is trained around them as for the
    dagger.
  - c = 0.4 and 0.7: where job 6's learned charges stopped (0.20–0.59 of the permanent dipole).
  - c = 1.0: the permanent dipole.
  - c = 1.1: the lemma's scale. Job 6 found |b_T| × the dipole = 2.70 D for every field-blind source, so |b_T| = √ε_∞(k)
    = 1.31–1.32 means 2.05–2.06 D.
  - c = 1.45: the liquid's whole dipole, 2.70 D, where b_T = 1.
  - c = 0 is M_∞.
- **MB-pol's site charges without c6:** M_1pm, E3's rung. It is the dagger, M_16pm, less its dispersion band.
- **The warm start** (`Rung.q_ref_start`): M_1w050, M_1w100, M_1w150.
  - Learned charges (job 1's head) with their species offsets at c₀ = 0.5, 1.0 and 1.5 times the O/H charges.
  - The head's last layer starts at zero, so every atom starts at exactly that charge, with the physical sign.
  - The charges are in units of 0.15 e (M_1uc's), fixed by §8.
- **The readers:**
  - `scripts/e4_job7.py` reads J7.1–J7.5 from the training jsons, without JAX;
  - `scripts/e4_sumrules.py read --job7_dir` reads J7.6;
  - the training jsons record the pinned scale and the start (`pinned_charge_scale`, `charge_start_e`).
- **Everything else is job 6's:**
  - the 64-molecule cells;
  - Adam for 3000 steps, then L-BFGS for 50;
  - the loss weights and the rigid forces;
  - the readings on every frame of 64–256 molecules, the probes and T5.
- **Two seeds,** 0 and 1, against job 6's four.

## 3. The runs

`run_e4_job7.sbatch` has 22 tasks, eleven rungs times seeds 0 and 1:

- 0–1: M_∞, a check against job 6's;
- 2–3: M_16pm, the dagger;
- 4–5: M_1pm;
- 6–15: the scan, c = 0.4, 0.7, 1.0, 1.1, 1.45;
- 16–21: the warm starts, c₀ = 0.5, 1.0, 1.5. Each then writes its charges on S0's frames of 64–256 molecules, for J7.6.

Then:

```sh
python scripts/e4_job7.py --out_dir results/e4_job7 --reference_dir results/e4_job6
python scripts/e4_sumrules.py read --gate_dir results/e4_job6 --job7_dir results/e4_job7
python scripts/e4_water.py summary --out_dir results/e4_job7
```

## 4. Registered

**The readings.** Rigid forces on the unseen frames, job 1's convention.

- The frames: validation and test at 64 molecules, every split at 128 and 256. Each split's rmse is pooled by its frames.
- Seed means over seeds 0 and 1.
- M_∞'s reference is its four seeds in job 6.

| | reading | band | prediction |
|---|---|---|---|
| **J7.1** | M_1pm's forces against the dagger's | within ±7 % at 64, 128 and 256 molecules | holds: the dagger's gain is its charges' |
| **J7.2** | the c of the scan's lowest forces | c ≥ 1.0 at every size | holds |
| **J7.3** | the scan's lowest forces against M_∞'s | at least 10 % below at every size | holds: about the dagger's 20 % |
| **J7.4** | the six warm-started runs: the learned dipole at the end (the test frames of 64 molecules) and the final training loss | every dipole within ±15 % of the six runs' mean, and every final loss at or below the largest of M_∞'s (job 6's four runs and job 7's two) | holds |
| **J7.5** | the warm-started runs' mean forces against the scan's lowest | at most 7 % above at every size; read when J7.3 holds | holds |
| **J7.6** | \|b_T\| of the warm-started charges on S0's first shells of 128 and 256 molecules | job 6's L1.2 band, 1.22–1.40, at both shells on at least 4 of the 6 runs; read when J7.4 holds | holds: √ε_∞(k) = 1.32, 1.31, a learned dipole of 1.93–2.21 D |

- **Why these bands:**
  - J7.1's 7 % is 0.05 of the dagger's 0.72 kcal/mol/Å, about a quarter of its gain of 0.18. It is also about two
    standard errors of a difference of two-seed means at the dagger's seed spread (0.022).
  - J7.3's 10 % is half the dagger's gain. M_∞'s four seeds spread by 6.6 %.
  - J7.4's 15 %: the starts are a factor of 3 apart, and job 6's learned dipoles spread by a factor of 2.2 over seeds at
    one unit.
  - J7.5's 7 % is J7.1's.
  - J7.6's band is job 6's L1.2: the lemma's value, −7 % to +6.5 %.
- **J7.2 cannot tell 1.0 from 1.1.** Their forces should differ by less than the noise. The lemma is read through J7.6.
- **The gate in J7.4** guards against a run that did not train, like job 1's M_1, which ended at 3–5 times M_∞'s loss.
  It is not job 6's seed-by-seed gate, which was as noisy as the effect.
- **The sign.** The warm starts start with the physical sign. A run that flips is reported.
- **The verdicts** come from `scripts/e4_job7.py` (J7.1–J7.5) and `scripts/e4_sumrules.py read` (J7.6).

## 5. Read beside, not registered

- **The curve** of forces against c at every size, with c = 0 as M_∞; the energies; the probes; T5, which is 100 % for
  every rung here.
- **The c6's share** of the dagger's gain: (F_M_1pm − F_dagger) / (F_M∞ − F_dagger).
- **The O/H form** against MB-pol's sites at the permanent dipole: M_1q100 against M_1pm.
- **The final training loss against c:** the objective's own curve.
- **Job 7's M_∞** against job 6's seeds 0 and 1.
- **The warm-started charges:**
  - q_O and q_H, their spread over the molecules, and their sign;
  - r, against L1.3's band;
  - S0.5's ρ.

## 6. What would change the reading

- **J7.1 fails high:** M_1pm well above the dagger. The c6 carries part of the dagger's gain, so the comparison's pins
  include it for a reason. The scan reads the charges' part alone.
- **J7.2 fails:** the lowest at c = 0.4 or 0.7.
  - The training cells reward a scale below the permanent dipole.
  - Job 6's learned charges then stopped nearer the data's scale than the lemma says, and the lemma's scale is not what
    E4's training sets.
- **J7.3 fails:** no field-blind scale buys 10 %.
  - With J7.1 holding, the O/H form loses what MB-pol's sites carry: read M_1q100 against M_1pm.
  - With J7.1 failing, the dagger's gain is mostly c6.
  - Either way, learned O/H charges cannot be expected to buy much on these data.
- **J7.4 fails, with J7.3 holding.**
  - The data set a scale, and learned charges do not reach it even from nearby. The optimisation is the obstacle.
  - Next would be the charge head's own learning rate, registered first.
- **J7.4 holds.**
  - Learned charges work once started at the order of the monomer's dipole.
  - The like-for-like comparison with MACE-LES gives it the same start.
  - Stage 2, learned charges with the response, builds on it.
- **J7.6:** as job 6's §6.
  - It fails high near 1.455 if the charges are the permanent ones, unscreened.
  - It fails low near 1 if they carry the liquid's whole polarisation.

## 7. Cost

- **Training per seed in job 1:** 0.4–0.5 h for the dagger, 0.2–0.3 h for M_∞ and M_1.
- **A task,** with the readings and (for the warm starts) the model step, should take about an hour: 22 tasks in all.

## 8. The check on synthetic cells, before registering

**The setup** (`scratch/e4_job7_teacher_student.py . 600 12 32 <runs> 16`, CPU, about 40 minutes a run):

- **The teacher and the cells are job 6's** (§8 there): MB-pol's rigid site charges alone, through the Coulomb band, on
  12 random cells of 32 molecules. Its charges carry 1.864 D, and the O/H charges with its dipole are O −0.6623 and
  H +0.3312 e.
- **The students** are trained with E4's settings and 16 features: Adam for 600 steps, no polish.

| run | final training loss | against M_∞ (6.95, job 6's §8) | learned dipole, start → end (of the teacher's) |
|---|---|---|---|
| learned, 0.15 e, started at 0.5 | 1.80 | −74 % | 0.93 → 1.77 D (0.95) |
| learned, 0.15 e, started at 1.5 | 1.65 | −76 % | 2.80 → 2.05 D (1.10) |
| the O/H charges pinned at c = 1 (the scan's M_1q100) | 1.09 | −84 % | 1.864 D (1.00) |
| job 6's M_1u, 0.05 e, started near zero | 5.53 | −20 % | → 0.70 D (0.38) |

- **Both starts moved to the teacher's scale, from both sides, within 600 steps.**
  - They ended at 0.95 and 1.10 of it, 7 % either side of their mean.
  - That is inside J7.4's 15 %, on a fifth of job 7's schedule.
- **They trained far better than job 6's start near zero:** a final loss of 1.6–1.8, against M_1u's 5.53 and M_∞'s
  6.95.
- **The pinned O/H charges at c = 1 did better still: 1.09.**
  - The learned charges realised 88–91 % of the pinned charges' gain over M_∞ in the loss, and 81–85 % in rigid-force
    rmse (the square root of the force term).
  - They were still improving when the schedule ended.
  - On E4's data, J7.5's band allows about 28 % of the gain unrealised: 7 % of the dagger's 0.72 kcal/mol/Å, against its
    gain of 0.18. So J7.5 is predicted to hold.
- **The unit, 0.15 e, moves the charges fast enough.**
  - Adam's step budget over 600 steps is about 0.9 in the parameters, 0.14 e of charge in the species offsets.
  - The 0.5 start's H charge moved 0.15 e, the head's last layer carrying the rest.
  - Over job 7's 3000 steps the budget is five times larger.
  - Job 6's 0.05 e would give the 0.5 start 0.23 e over the whole schedule, against the 0.17–0.20 e it has to travel.
- **The sign stayed physical** on both runs.
