# E4, job 6: the learned charges, started near zero — results

*Read 17 Nov 2026 against `e4-job6-preregistration.md`, which was registered on 16 Nov at 03:43 CET, before any run.
The job ran on Niflheim at 585d655 as sixteen tasks: M_∞, M_1u, M_1uc and M_16u, seeds 0–3. It was read with
`scripts/e4_water.py summary --out_dir results/e4_job6` and `scripts/e4_sumrules.py read --gate_dir results/e4_job6`.
Spreads are standard deviations over the seeds.*

## 1. The registered readings

| | band | Niflheim | verdict |
|---|---|---|---|
| **L1.1**, the gate | M_1u's final training loss at or below M_∞'s on at least 3 of 4 seeds | seeds 1–3 pass (−19, −20, −0.3 %); seed 0 fails (+20 %) | **holds** |
| **L1.2**, \|b_T\| | 1.22–1.40 at both shells, with the scale converged | M_1u 3.32–7.40 at 128 molecules, gated mean 4.90; the control M_1uc 2.46–3.25, gated mean 2.77. The two must agree within 5 % | **not converged** |
| **L1.3**, r | 1.45–2.30 at both shells | 1.81–2.04 at 128 and 1.79–2.05 at 256, every seed | **holds** |

- **L1.1 holds narrowly.**
  - Seed 3 passes by 0.3 %.
  - M_∞'s own final loss spreads from 54 to 75 over the seeds, as wide as the effect.
- **L1.2 takes the registered branch "not converged"**, neither holding nor failing. M_1u's mean |b_T| is 1.77 times
  M_1uc's at both shells.
- **L1.3 holds,** as it must for a field-blind source.

## 2. What the charges learned

**The charges are small and nearly the same on every molecule.** q_O ≈ −2q_H, and the molecular dipole's spread over
the liquid is 2–4 %.

| rung (unit) | seed 0 | seed 1 | seed 2 | seed 3 | of MB-pol's permanent 1.864 D |
|---|---|---|---|---|---|
| M_1u (0.05 e) | 0.66 D | 0.68 D | 0.37 D | 0.81 D | 0.20–0.44 |
| M_1uc (0.15 e) | 0.95 D | 0.83 D | 1.04 D | 1.10 D | 0.45–0.59 |

The lemma's scale is about 2.07 D.

- **The sign is free.** Seeds 1 and 2 came out flipped in every rung, with O positive. Energies and forces of neutral
  molecules cannot fix the sign.
- **b_T reads only the scale.**
  - |b_T| times the learned dipole is 2.68–2.71 D on every seed of both rungs.
  - It is the same for MB-pol's permanent charges (1.456 × 1.864 D = 2.71 D), and about MB-pol's mean liquid dipole,
    2.67 D.
  - So L1.2's band is a learned dipole of 1.93–2.21 D.
- **The scale follows the training path, not the data.**
  - At one unit, the seed moves the scale by a factor of 2.2 (0.37–0.81 D).
  - The outcome is reproducible for a given seed and unit. M_16u, whose α never moved (§3), learned M_1u's charges seed
    by seed, within about 3 %.
  - Three times the unit gave 1.2–2.8 times the charges, seed by seed.
  - **The scale did not stop where Adam's step budget ran out.**
    - Adam moves a parameter by at most about its learning rate per step: 4.5 over the schedule.
    - For the species offsets, that is 0.23 e of charge in M_1u and 0.68 e in M_1uc.
    - M_1u's q_H reached 0.07–0.14 e, and M_1uc's 0.15–0.20 e.
- **S0.5's account, read beside.** At the first shells of 128 and 256 molecules, ρ = 0.33–0.75, against P_perm's 0.55
  and 0.58. The learned charges sit in the field-blind class with the permanent ones.

## 3. What they buy

**Rigid forces** on the test frames, in kcal/mol/Å:

| rung | 64 | 128 | 256 | against M_∞ |
|---|---|---|---|---|
| M_∞ | 0.898 ± 0.059 | 0.927 ± 0.064 | 0.926 ± 0.062 | — |
| M_1u | 0.865 ± 0.038 | 0.892 ± 0.036 | 0.897 ± 0.035 | −3 to −4 % |
| M_1uc | 0.903 ± 0.062 | 0.929 ± 0.074 | 0.932 ± 0.068 | +0.2 to +0.7 % |
| M_16u | 0.863 ± 0.039 | 0.891 ± 0.037 | 0.896 ± 0.038 | −3 to −4 % |

- **M_1u's gain is within the seed spreads:** 0.7–0.8 standard errors of the difference.
- **M_1uc's larger charges bought nothing.**
- **In job 1 the dagger was 20 % below M_∞**, 4.4 standard errors (0.721 ± 0.022 against 0.907 ± 0.069 kcal/mol/Å,
  unseen frames at 64).
  - The dagger is the same E₀ with MB-pol's site charges and c6 pinned.
  - How much of its gain is the charges and how much the c6 has not been checked (job 1's results, §4b).
- **Energies show no consistent change.** Every rung reads 0.018–0.036 kcal/mol per molecule, and the differences from
  M_∞ change sign between sizes.
- **The far-field probes' non-additive error** is 95–97 % for M_1u and 91–95 % for M_1uc, against M_∞'s 100 %.
- **T5's far-trimer three-body error** is 100 % for every rung, as it must be without a response.
- **M_16u's polarisabilities never left their start.**
  - C₆ prints as 0.00 au on every pair, and at most 0.01 au per molecule against MB-pol's 43.1.
  - Reaching MB-pol's C₆ needed 4.4–5.9 units of drift in the softplus argument: about the schedule's whole budget.
  - C₆ is bilinear in α, as the Coulomb band is in the charges. At a start of 10⁻⁵ of MB-pol's C₆, its gradient was
    likely too small to move it.
  - So M_16u is M_1u in every reading.

## 4. What this says

- **Stage 1's answer: with E4's data and job 1's training, our architecture does not learn its charges.**
  - The small unit removed job 1's stall. Job 1's M_1 ended at 1.70 kcal/mol/Å in rigid forces; M_1u ends at 0.87.
  - But the seed and the unit set the scale the charges reach, at a fifth to three fifths of the permanent dipole.
  - What the charges buy is not resolved from zero.
- **The lemma is not tested.** L1.2 read where the training stopped, not where the data put the scale.
- **Two readings are open.**
  - **The optimisation.** The data favour a scale near the pinned charges', which the dagger had by construction, and
    the learned charges do not get there.
  - **The data.** The 64-molecule cells barely constrain the scale.
    - E₀'s two message-passing layers reach 12 Å: the whole training cell (L₀ = 12.4 Å) through its images (job 1's
      results, §4b).
    - So E₀ can absorb whatever the charges leave, and the charges stop where E₀ leaves them no gradient.
  - **The evidence is split.**
    - M_1uc's larger charges did no better, and the scale follows the seed. Both lean to the data.
    - The dagger's 20 % leans to the optimisation, unless most of it is c6.
- **For the comparison.** A no-pins version of our model, trained this way, keeps none of the pinned version's 20 %.
  MACE-LES, trained on the same cells, faces the same question.

## 5. Proposed next, not registered: job 7

**(a) A scan of pinned charges,** to see whether and where the data put a field-blind scale.

- **The charges** are fixed O/H charges carrying c × MB-pol's permanent 1.864 D (q_H = 0.331c e). That is the learned
  charges' own form.
  - E₀ is trained around them as in job 1.
  - There is no c6 and no response.
- **The scale c:**
  - 0.4 and 0.7, where the learned charges stopped;
  - 1.0, the permanent scale;
  - 1.1, the lemma's;
  - 1.45, the whole liquid dipole's (b_T = 1).
  - c = 0 is job 6's M_∞.
- **Plus MB-pol's own site charges at c = 1:** the dagger without its c6.
- **Size:** two seeds each, 12 tasks.
- **It reads three things:**
  - whether the loss and forces depend on c at all (the data reading predicts flat);
  - where they are lowest;
  - how much of the dagger's 20 % is its charges.

**(b) Learned charges started at 0.5, 1 and 1.5 times the monomer's dipole.**

- **Size:** two seeds each, 6 tasks.
- **The start.** The species offsets are set so the start carries that dipole. The unit is fixed by a synthetic check
  first, as in §8 of the pre-registration.
- **What it tells.** If the three starts end at one scale, learned charges work once started at the right order, and
  that scale is the data's. If each stays near its start, these data do not determine the scale.

**Why not the registered branch, a longer schedule.**

- The scale followed the seed and the unit, and M_1uc stopped well inside its step budget. A longer run at the same
  units would likely still leave the scale to the path.
- Starting from both sides tests convergence directly, and the scan gives the target.
- The deviation would be stated in job 7's pre-registration.

**Fairness.** The warm start uses a property of one molecule, which any truth provides. MACE-LES would get the same
start.

**Parked: learned dispersion.** MACE-LES has none, and M_16u's start needs its own fix.
