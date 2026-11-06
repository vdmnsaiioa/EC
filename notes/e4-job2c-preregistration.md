# E4, job 2, arm C: the calibration on the order-0 columns alone — pre-registration

*6 Nov 2026, v0.0.26. This was written after job 2's read (`e4-job2-results.md`) and its matching-level check (§4f
there), and before any arm-C training. It registers one more arm of job 2, run beside arms A and B and read against
them. The local master of this note is `notes/` in the repository.*

## 1. The question

Arm B calibrated the read-out on MB-pol's response in liquid cells.

- **What it did.** ε_∞ came out at MB-pol's value, and the frozen form's bulk forces were unchanged. But the ablation
  got 2–5 % worse on every seed, which undid the regulator's gain.
- **Why, read without training** (`e4-job2-results.md` §4b). The calibration moved every read-out order:
  - the orders ≥ 1 (the field-gradient couplings) took −0.79 kcal/mol per molecule of contact energy in the liquid,
    with rigid-force changes of 0.70 kcal/mol/Å rms;
  - the order-0 columns took +0.38 kcal/mol and 0.52 rms;
  - net, −0.41 kcal/mol and 0.47 rms.

  The field term reads the field gradient and absorbs such changes; E₀ alone does not.
- **What the derivative counting says** (`eft-of-lss-workflow.md` §5). Uniform-field (k = 0) data identify the
  order-0 coefficient alone.
- **The check at the matching level** (§4f). Calibrating the order-0 columns and the node term alone does the
  calibration's job:
  - ε_∞ 1.6895;
  - the molecules' dipoles at 0.991 of MB-pol's;
  - the far probes 2.3 / 2.2 / 2.7 % and T5 1.040;
  - the contact forces moved by 0.33 rms instead of 0.47, and nothing into the orders ≥ 1.

**Arm C asks whether that calibration also stops costing the ablation.** Two readings of arm B's cost predict
differently:

- **(i) The cost was the orders ≥ 1: what moved.** Arm C's order-0 change then moves the read-out the way the
  regulator did. The molecules' dipoles go from 6 % too large to MB-pol's scale, and the regulator's reduction of the
  over-polarisation (17 % → 6 %) bought the ablation 4 %. So arm C's ablation is at or below arm A's.
- **(ii) The cost is any change at contact E₀ cannot absorb: how much moved.** An error added in quadrature scales as
  the square of the change, so arm C pays about (0.33 / 0.47)² of arm B's +3.6 %: about +1.8 % over arm A.

I expect (i), with low confidence: about even odds.

## 2. Design

- **Config.** `configs/e4_job2c.yaml` is arm B's config with `training.readout.liquid.orders: zero` (v0.0.25), and
  nothing else changed (tested).
  - The two C rungs, the frozen form M_16C2†ᵐ_us and the ablation M_16C2†ᵐ, at the 2.0 Å mutual edge.
  - 4 seeds, the same training as job 1 and job 2.
  - Output in `results/e4_job2c`.
- **Response data.** The liquid response is job 2's file (`data/e4/response_mbpol.npz`, hash 44997141): 30
  calibration cells, 10 held out. Nothing is recomputed.
- **Matching.** The far-field stage is arm A's, recomputed in the new directory: its objective should read 9.696×10⁻⁶
  again. The calibration stage moves only the order-0 columns and the node term, with the probes' ridge.
- **Runs.** `run_e4_job2c.sbatch`: one job per seed, the frozen form then the ablation. That is about 6.5 h a job and
  26 GPU hours in all.
- **Readings.**
  - Job 2's readings.
  - `scripts/e4_report.py --job2 results/e4_job2a,results/e4_job2b,results/e4_job2c --job1 results/e4_job1` reads arm
    C against arms A and B, job 1 and this note.

## 3. Fixed before training

**On the local frames** (`e4-job2-results.md` §4f; 10 calibration frames, 3 held out, MB-pol's ε_∞ 1.6870 there):

| reading | arm C, local |
|---|---|
| ε_∞ | 1.6895 |
| cells' induced dipole | 3.1 % |
| molecules' induced dipole | 5.0 % |
| molecules' field response (iso) | 3.1 (2.3) % |
| \|μ\| ÷ MB-pol's (r) | 0.991 (0.999) |
| iso response spread ÷ MB-pol's (r) | 1.14 (0.04) |
| far probes, non-additive, 10 / 12 / 15 Å | 2.3 / 2.2 / 2.7 % |
| T5 slope (rmse) | 1.040 (4.1 %) |
| the probes' objective in the calibration | ×1.73 |

**C0: on Niflheim**, with 30 calibration cells, the log shows before training:

- ε_∞ within 0.010 of MB-pol's 1.6885 on the 10 held-out frames;
- \|μ\| ÷ MB-pol's within 0.98–1.02;
- the far probes at 10–15 Å within +1 point of arm A's (2 / 2 / 2 %);
- T5's slope within 0.95–1.10;
- the probes' objective rising ×1.3–2.5;
- every one of these the same after training and across seeds and rungs, as J0.

A C0 failure is an implementation or data matter, read before the bulk.

## 4. Predictions for training

Rigid-body forces on the unseen frames pooled, at 64, 128 and 256 molecules. Arms A and B are job 2's runs; the dagger
is job 1's. Arms A and C share every seed's E₀ initialisation and batches, so the paired arm-to-arm changes resolve about
1 % (`e4-job2-results.md` §5).

| | prediction | rule | arm B for comparison |
|---|---|---|---|
| **C1** | Arm C's ablation within +1 % of arm A's. Reading (ii) predicts +1.8 %, (i) 0 or below | seed means, every size | +3.6, +3.4, +3.3 % |
| **C2** | Arm C's ablation at or below arm A's: reading (i) | ≥ 3 of 4 seeds paired, every size | 0 of 4 |
| **C3** | J1 again: arm C's ablation at or below the dagger | ≥ 3 of 4 seeds paired, every size | 1 of 4; arm A 3 of 4 |
| **C4** | Arm C's frozen form within ±1 % of arm A's | seed means, every size | +0.7, +0.2, +0.4 % |

Readings beside them, not registered:

- arm C's ablation against arm B's;
- arm C's frozen form against M_16†ᵐ_us;
- the ablation's net forces;
- the energies;
- the field term's gain in arm C;
- the ablation's final training loss against arm A's (arm B was 4–10 % higher on every seed).

## 5. What would change the reading

- **C1 and C2 hold.** Reading (i): the calibration follows the derivative counting.
  - If C4 holds too, arm C is the production model: the infrared exactly, and the bulk at arm A's level or better.
  - The order-2 coefficients then wait for the finite-k response, as the workflow note orders.
- **C1 holds and C2 fails.** Arm C's ablation is within 1 % of arm A's but not below it on 3 seeds. The calibration then
  neither helps nor costs in bulk once restricted. Arm C is still the production candidate, on the infrared.
- **C1 fails.** Reading (ii): any calibration at contact costs the ablation, whatever its direction.
  - The ablation's bulk is then no argument against calibrating: the field term absorbs the change (C4, and arm B's
    frozen form).
  - The calibration belongs with the field term.
- **C4 fails by an improvement.** The order-0 calibration helps the frozen form too: the dipoles' scale was a bulk
  error that the field term had not absorbed, as the regulator found at 0.75 Å.
- **C0 fails.** The calibration on 30 cells does not reproduce the local one. That is read before any bulk reading.
