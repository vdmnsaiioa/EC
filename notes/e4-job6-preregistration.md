# E4, job 6: the learned charges, started near zero — pre-registration

*16 Nov 2026, registered at 03:43 CET (this commit's time), before any run of job 6 on MB-pol's data and before any of
its numbers. Vadic's go of 16 Nov. The question came from the comparison with competing models: E4's model pins
MB-pol's charges and c6, while MACE-LES learns its charges from energies and forces alone. The like-for-like comparison
is our architecture with learned charges, and job 1's learned-charge rungs did not train. This job is stage 1: those
rungs, fixed, without the response. Stage 2 adds the response (§7). Read against:*

- *`e4-job1-results.md` §4d (M_1 and M_16 did not train);*
- *the project's `path-integral-ir-sector.md` §5 (the lemma, with its addition of 15 Nov);*
- *`e4-s0-results.md` §1 and §3 (S0.3, S0.5).*

## 1. Why, and what the theory predicts

**Job 1's failure.** Both learned-source rungs ended above M_∞, which they contain:

- **M_1, learned charges,** started where M_∞ did and stalled. Its final loss was 3–5 times M_∞'s, and its rigid
  forces 1.7 kcal/mol/Å against 0.91.
- **M_16** also learned α(iω). That started flat at softplus(1) = 1.31 bohr³: a C₆ 4 times MB-pol's for O–O and 49
  times for H–H. It ended at 11–16 times M_∞'s loss.
- **The diagnosis then, untested:** at E4's energy scale (the training frames' spread, 0.06 kcal/mol per atom), one Adam
  step of the charge head moved the energy by several times that spread. §8's synthetic check now reproduces the
  failure.

**The lemma's prediction for learned charges.** A field-blind source, including learned charges, gets the
long-wavelength energetics right only as Y = ±Π/√ε_∞, where Π is the short-circuit polarisation (lemma (ii)). Regressing
MB-pol's polarisation on such a source gives:

- b_T = √ε_∞(k) and b_L = 1/√ε_∞(k);
- r = ε_∞(k), plus the transverse residual v_η when the source misses part of Π (S0.5's account).

On S0's first shells of 128 and 256 molecules, √ε_∞(k) is 1.32 and 1.31. Two alternatives sit well apart:

- MB-pol's own permanent charges read b_T = 1.455 there (S0.3);
- a source carrying the liquid's whole polarisation would read 1.0.

Through b_T, the learned molecular dipole would be about 1.455/1.31 × 1.864 D ≈ 2.07 D in the liquid. That is the
electronic-continuum value μ_liquid/√ε_∞. The latent-charge papers supply the same √ε_∞ by hand: their periodic
Born-charge formula takes ε_∞ as an input.

## 2. What changes (v0.0.30), fixed before any run

- **The charges start small.**
  - The charge head's output is in units of 0.05 e (`Rung.q_unit`).
  - With the same parameters, M_1u's charges are exactly 0.05 times M_1's, so its Coulomb band starts at 2.5×10⁻³ of
    M_1's. The rung starts close to E₀ alone.
  - One Adam step moves the head's parameters by about the learning rate, so the charges now move 20 times more slowly
    than in job 1.
  - The start is not zero. The Coulomb band is bilinear in the charges, so zero charges are a stationary point of any
    loss, and the head would never move (`tests/test_job6.py`).
  - The unit was set by §8's check: 0.05 e did better there than 0.02 e, and job 1's 1 e reproduced job 1's failure.
- **A control for the learned scale.** M_1uc is M_1u at three times the unit, 0.15 e. If the two learn the same scale,
  the scale has converged within the training schedule. §8 found the charges still growing after 600 steps.
- **The polarisabilities start small.** α = softplus(−6 + x) ≈ 2.5×10⁻³ bohr³ at initialisation (`Rung.alpha_ref0`),
  instead of softplus(1) = 1.31. Where softplus ≈ eˣ, every step changes α by a fixed fraction. M_16u's charges use
  M_1u's unit.
- **The end-of-training loss** of every run is recorded: the full-batch objective after the polish (`final_loss`),
  which the gate compares.
- **The learned sources are read** on every split:
  - per species, the charges;
  - the molecular dipole, in D;
  - its overall sign, as the share of molecules whose dipole points to their H side;
  - with learned α, the band's C₆ per pair against MB-pol's.
- **S0's model step for learned charges** (`scripts/e4_sumrules.py model --params`): each frame's learned charges at
  the atoms, regressed by S0's read like any candidate.
- **Everything else is job 1's:**
  - the 64-molecule cells;
  - Adam for 3000 steps, then L-BFGS for 50;
  - the loss weights and the rigid forces;
  - four seeds;
  - the readings on every frame of 64–256 molecules, the probes and T5.

## 3. The runs

`run_e4_job6.sbatch` has sixteen tasks:

- 0–3: M_∞, seeds 0–3, the gate's reference;
- 4–7: M_1u;
- 8–11: M_1uc;
- 12–15: M_16u.

Each learned-charge task then writes its charges on S0's frames of 64–256 molecules. Then:

```sh
python scripts/e4_water.py summary --out_dir results/e4_job6
python scripts/e4_sumrules.py read --gate_dir results/e4_job6
```

## 4. Registered

| | reading | band | prediction |
|---|---|---|---|
| **L1.1** | the gate: M_1u's final training loss at or below M_∞'s, seed by seed | at least 3 of 4 seeds | passes |
| **L1.2** | \|b_T\| for M_1u's learned charges on S0's frames, the first shells of 128 and 256 molecules | 1.22–1.40 at both shells, on at least 3 seeds that pass the gate; read only if the scale converged | √ε_∞(k) = 1.32, 1.31 |
| **L1.3** | r there | 1.45–2.30 at both shells, on at least 3 seeds that pass the gate | ε_∞(k) + v_η, 1.7–2.0 |

- **Convergence (for L1.2).** At both shells, the control M_1uc's mean |b_T| over its gated seeds must lie within 5 %
  of M_1u's over its own, with at least 3 gated seeds each.
  - Otherwise L1.2 is reported as not converged, neither holding nor failing, and the next step is a longer schedule.
  - L1.3 needs no such check: r does not depend on the source's scale.
- **The sign.** Energies and forces of neutral molecules cannot fix the charges' overall sign: q → −q leaves the Coulomb
  band unchanged. So L1.2 reads b_T's magnitude, and the sign is reported.
- **L1.2's band** is the lemma's value, −7 % to +6.5 %. The two alternatives of §1, 1.455 and 1.0, lie outside it.
- **L1.3** is the field-blind class: S0.3's lower edge. A source carrying the response reads 1.
- **The verdicts** come from `scripts/e4_sumrules.py read`, which also reads the gates from job 6's training jsons, and
  from `scripts/e4_water.py summary` for L1.1.

## 5. Read beside, not registered

- **Bulk errors.** M_1u's, M_1uc's and M_16u's forces and energies at 64–256 molecules, against M_∞, the dagger (0.721
  kcal/mol/Å at 64) and D′ (0.541).
- **The learned charges:** q_O and q_H, the molecular dipole and its spread over the liquid (about 2.07 D expected), and
  its sign.
- **S0.5's account for them:** v_η and ρ. At these shells P_perm read ρ = 0.55–0.58, so the account is near its limit
  there.
- **M_16u:** its gate, and its C₆ per pair against MB-pol's long-range 17.2, 5.01 and 1.46 au (43.1 per molecule).
- **The far-field probes.**
- **Structural, not read:** for every learned-source rung ε_∞ = 1 exactly and T5 = 0, as in job 1.

## 6. What would change the reading

- **L1.1 fails.** The learned charges still do not train with this unit. Their readings are not read. The next step is
  the optimiser, for example a learning rate of the charge head's own, registered before it runs.
- **L1.1 holds and L1.2 holds, converged.** Learned charges in our architecture are screened by √ε_∞, as the lemma says:
  the latent-charge class, reproduced on our own short-range part. That is stage 2's baseline.
- **L1.2 is not converged.** The two units learn different scales within 3000 steps, so the schedule is too short to
  tell. The run is repeated with a longer one, registered first.
- **L1.2 fails high, converged,** near 1.455. The learned charges are the permanent charges, unscreened. Then the
  training cells' long-range energetics do not set the scale the lemma assumes: the 64-molecule cell's k ≥ 0.51 Å⁻¹,
  and 6–9 Å in real space.
- **L1.2 fails low, converged,** near 1. The charges carry the whole liquid polarisation, and the long-range energy is
  too strong by about ε_∞. That would be a fit E₀ compensates.
- **L1.3 reads near 1.** That is impossible for a field-blind source (lemma (iii)), so it would point at the reader.

## 7. Stage 2, not registered here

Learned charges with the response is the rung that matters for the comparison. It needs one design choice: the
response was matched on probes with the charges pinned. The choice is to learn the charges and the response together,
or in turn. It will be registered after job 6's read.

## 8. The check on synthetic cells, before registering

**The setup** (`scratch/e4_job6_teacher_student.py . 600 12 32 M_inf,M_1@1,M_1@0.02,M_1@0.05 16`, CPU, about 40 minutes a rung):

- **The teacher** is MB-pol's rigid site charges alone: O 0, H +0.578, M −1.155 e, through the Coulomb band in the cell,
  with no E₀ and no dispersion. It has no electrons, so the charges that reproduce it carry its 1.864 D.
- **The data:** it labels 12 random cells of 32 molecules (12.5 Å).
- **The students** are trained with E4's settings and 16 features: Adam for 600 steps, no polish.

| rung | charge unit | final training loss | against M_∞ | learned dipole |
|---|---|---|---|---|
| M_∞ | — | 6.95 | — | — |
| M_1, job 1's | 1 e | 8.47 (step 0: 1322) | +22 %, fails the gate | 1.07 D |
| M_1u at 0.02 e | 0.02 e | 6.49 | −7 % | 0.50 D |
| M_1u at 0.05 e | 0.05 e | 5.53 | −20 % | 0.70 D |

- **Job 1's failure reproduces with its unit.** M_1 starts far above M_∞ (its random charges' Coulomb energy) and ends
  above it.
- **Both small units pass,** and 0.05 e does better than 0.02 e. That is why 0.05 e is registered.
- **All the learned charges are still growing after 600 steps:** 0.5–1.1 D, against the teacher's 1.864 D. Job 6 trains
  for 3000 steps and polishes. That remainder of the schedule is what the control M_1uc checks.
- **The learned charges came out sign-flipped** on these seeds (O positive), as §4 allows.
