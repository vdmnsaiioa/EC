# E4, job 9: why the learned charges stop short, and the charge head's own learning rate — results

*Read 21 Nov 2026 against `e4-job9-preregistration.md`, registered at 02:01 CET on 21 Nov (9285074 on GitHub), before
any run. The job ran on Niflheim at v0.0.33 as the registered two tasks of five runs.*

*It was read with:*

- *`scripts/e4_job9.py`;*
- *`scripts/e4_sumrules.py read --job9_dir`;*
- *`scripts/e4_water.py summary`;*
- *the runs' diagnostics from their jsons: the pull's mean and sd over Adam's second half, and the curvature of the
  objective along the charges' scale with the rest held, from s = 0.99–1.01 and 0.9–1.1.*

*Rigid forces in kcal/mol/Å on the unseen frames, seed means over seeds 0 and 1. The profile is job 7's scan: its
parabola has its minimum at c* = 1.098, 2.047 D.*

## 1. The registered readings

| | band | Niflheim | verdict |
|---|---|---|---|
| **K9.1** | R in [0.5, 2] on 3 of 4 (\|R\| ≤ 0.25 on 3 of 4: locked) | +11.8, −0.74, −12.1, −9.0 | **fails**, in neither band |
| **K9.2** | the pull's \|mean\| / sd at most 0.3 on 3 of 4 | 0.009, 0.006, 0.021, 0.018 | **holds** |
| **K9.3** | the four warm starts at ten times within ±15 % of their mean; the gate | 1.615, 1.312, 1.972, 2.228 D: −9, −26, +11, +25 %; every loss passes | **fails** |
| **K9.4** | their mean within ±10 % of 2.047 D | 1.782 D, −13 % | **not read** |
| **K9.5** | their mean forces at most +7 % above the scan's lowest | 0.852 / 0.859 / 0.865 against 0.592 / 0.606 / 0.612: +44 / +42 / +41 % | **fails** |
| **K9.6** | \|b_T\| in 1.22–1.40 on 3 of 4 | 1 of 4 (M_1x150 seed 0, 1.37–1.38) | **not read** |

- **K9.1 fails on the branch the pre-registration's §4 lists last:** "fails in any other way: neither; read the paths".
- **§2 shows that K9.1 measured something other than what the pre-registration assumed.** §3 reads the pull the way it
  should have been read.
- **K9.3 and K9.5 fail:** ten times the charge head's learning rate is not the fix (§5).

## 2. What K9.1 measured: a valley ten thousand times steeper than the profile

**The objective along the charges' scale with the rest held,** for job 7's warm starts run again, at the end of Adam:

| run | c | curvature with the rest held | the profile's, c² L*″ | ratio | L(0.9) / L(1) |
|---|---|---|---|---|---|
| M_1w050 seed 0 | 0.643 | 5.7×10⁵ | 75 | 7,500 | 57 |
| M_1w050 seed 1 | 0.681 | 1.6×10⁶ | 85 | 19,000 | 143 |
| M_1w150 seed 0 | 1.342 | 2.4×10⁶ | 329 | 7,400 | 281 |
| M_1w150 seed 1 | 1.381 | 3.0×10⁶ | 349 | 8,600 | 340 |

- **Every run sits at the bottom of this valley.**
  - With the rest held, the minimum is at s = 1.000–1.002 in every reading printed: the four runs here at the end of
    Adam, and all ten at the end of training.
  - The ten runs' curvatures are 0.4–3.0×10⁶, except M_1x000 seed 0, whose charges stayed near zero (7×10²).
- **The curve is the even quartic of the pre-registration,** L = L_min + γ (s² − 1)².
  - With the rest held, the predicted energies and forces move by (s² − 1) times the Coulomb band's own contribution.
  - The curvatures from ±1 % and ±10 % agree in every run to the ratio such a quartic requires, 0.9975.
- **What sets the steepness.** The curvature over the objective, k / 8L, is the Coulomb band's contribution over the
  residual, squared.
  - It reads 1,500–9,300, so the charges contribute 40–100 times the residual, rms over the objective's terms.
  - E₀ cancels most of that contribution, and the cancellation is exact only at the scale it was trained with.
  - A change of the charges by 10 % with E₀ held multiplies the objective by 57–340. Along the profile, where E₀
    re-fits, the same change moves it by a few units in 30–50.
- **So K9.1 measured the end state's offset from the valley's bottom,** not the profile's pull.
  - The slope at s = 1 over the curvature is an offset of 2×10⁻⁵ to 1×10⁻³ in s, left by Adam's last steps.
  - The pre-registration assumed a gentle curve, on which that offset would not matter. At this curvature, an offset
    of 10⁻³ gives a slope ten times the profile's.
  - The registered verdict stands. K9.1 was the wrong measure of the pull.

## 3. The pull is there on average, buried 50–100 times under noise

The pull on the charges' scale over the batches of Adam's second half (steps 1500–2999):

| run | the profile's slope p | mean pull ± its standard error | sd per batch | mean / p |
|---|---|---|---|---|
| M_1w050 seed 0 | −53.5 | −20.7 ± 60 | 2,320 | 0.39 |
| M_1w050 seed 1 | −51.9 | −24.6 ± 100 | 3,860 | 0.47 |
| M_1w150 seed 0 | +59.8 | +79.6 ± 97 | 3,740 | 1.33 |
| M_1w150 seed 1 | +71.5 | +85.0 ± 120 | 4,630 | 1.19 |

The standard error treats the 1500 batches as independent.

- **All four have the profile's sign, at 0.4–1.3 of its size.** That is what the envelope theorem expects: on average
  Adam receives the profile's pull. But each is within about one standard error of zero, so it is consistent, not
  resolved run by run.
- **The noise has the valley's cause.**
  - The sd over the curvature is 1.5–4×10⁻³. A batch of six frames prefers a charge scale 0.15–0.4 % away from the
    full set's.
  - The steep valley turns that scatter into a pull 50–100 times the profile's.
- **K9.2's small ratio (0.006–0.021) is this stiffness, seen batch by batch.**

## 4. The paths: crawling, not locked

The learned dipole on the traced frames (D), by Adam step, then after the polish:

| run | 0 | 1500 | 3000 | the second half's share of the change | after L-BFGS |
|---|---|---|---|---|---|
| M_1w050 seed 0 | 0.932 | 1.153 | 1.198 | 0.17 | 1.213 |
| M_1w050 seed 1 | 0.932 | 1.212 | 1.270 | 0.17 | 1.315 |
| M_1w150 seed 0 | 2.796 | 2.549 | 2.502 | 0.16 | 2.461 |
| M_1w150 seed 1 | 2.796 | 2.611 | 2.575 | 0.16 | 2.563 |

- **The charges moved toward the data's scale all the way through, from both sides.**
  - 16–17 % of the change came in Adam's second half, against the 19 % of Adam's budget that lies there.
  - They moved at the pace of the learning rate, as a rate-limited motion does. They did not stop at a stationary point
    of their own.
- **The polish's 50 full-batch L-BFGS steps moved all four further the right way.** They moved +0.015, +0.045, −0.041
  and −0.012 D: between a third and nine tenths of what Adam's whole second half moved.
- **Job 7's warm starts reproduce job 7 to the printed digit:** dipoles 1.214, 1.315, 2.461, 2.562 D, and the forces. The
  diagnostics change nothing in the training.

## 5. Ten times the charge head's learning rate

| | dipole at the end (D), seeds 0 / 1 | forces 64 / 128 / 256 | final loss, seeds 0 / 1 |
|---|---|---|---|
| M_1w050 (job 7's rate) | 1.214 / 1.315 | 0.780 / 0.786 / 0.794 | 46.1 / 48.8 |
| **M_1x050** | **1.615 / 1.312** | **0.920 / 0.924 / 0.929** | **65.3 / 68.4** |
| M_1w150 (job 7's rate) | 2.461 / 2.562 | 0.703 / 0.730 / 0.741 | 38.2 / 39.8 |
| **M_1x150** | **1.972 / 2.228** | **0.784 / 0.795 / 0.800** | **57.7 / 40.0** |
| M_1x000 (from near zero) | 0.203 / 1.104, sign flipped | 1.260 / 1.281 / 1.283 | 142.9 / 114.3 |
| M_1q110 (job 7's scan, pinned) | 2.05 | 0.592 / 0.606 / 0.612 | 27.5 |

- **The faster head moves the charges further along the valley.**
  - From 1.5 they end at 1.97 and 2.23 D, against 2.46 and 2.56.
  - From 0.5, seed 0 reaches 1.62 D against 1.21. Seed 1 ends where job 7's did.
- **It costs the fit.**
  - Forces are 12–18 % worse than at job 7's rate.
  - M_1x150 seed 0 ends at 1.97 D (c = 1.06), next to the scan's best scale, yet its final loss is 57.7, against the
    profile's 29 there.
- **Why, by §2.**
  - At the schedule's start, the head's step on its species offsets is 0.0045 e, 1–2 % of an H charge.
  - At a curvature of about 2×10⁶, that costs 100–400 in the objective per step, several times the objective itself.
  - Each step throws the model up the valley's wall, and E₀ chases it.
- **From near zero both runs fail the gate** (143 and 114, against M_∞'s largest, 74.7), with forces worse than M_∞'s.
  Seed 0 never left zero (0.14 → 0.20 D).
- **The synthetic check could not see this.** Its teacher is the Coulomb band itself, so the student's E₀ has nothing
  to cancel: there is no valley.

## 6. Beside them

- **b_T times the learned dipole** is 2.71 D on every run, as in jobs 6 and 7: |b_T| reads the scale only.
- **The far probes' non-additive error** is 90–99.7 % for the learned charges. T5 is 100 %, as it must be without a
  response.
- **Run times** were 1214–1379 s (20–23 minutes), against the estimated 12–20 and job 7's 8–14, and tasks took 1.8 h.
  The diagnostics account for most of the difference.

## 7. What this says

- **Neither registered hypothesis, and both in part.**
  - The pull exists, as H_N assumed, and is buried in batch noise, as H_N said.
  - But the noise and the slowness have the cause H_L named: E₀ cancels most of what the charges contribute.
  - So the charges' scale is about ten thousand times stiffer with E₀ held than with E₀ following. Adam on batches of
    six crawls along that valley at the pace of its learning rate, and stops when the schedule does.
- **The charge head's own learning rate is not the fix.** Faster charges climb the walls.
- **By this reading, two things should help.**
  - **Make the valley shallower.** Give E₀ less of the charges' contribution to cancel: the Coulomb band damped at
    contact, as job 8 damped c6. This has a known cost to measure. E3's rehearsal found the band's contact part
    essential for pinned charges with the response on clusters (hexamers 0.07 kcal/mol at ℓ₁ = 0.75 Å, against
    0.55–0.88 at 1.5 Å). A damped charge band needs its own scan.
  - **Follow the valley.** A curvature-aware, full-batch optimiser: 50 L-BFGS steps already moved every run the right
    way.
- **For the comparison.** MACE-LES's charges are Gaussians of σ ≈ 1 Å in a reciprocal sum: 80–95 % of the full
  Coulomb at a contact of 1.9 Å. They likely face the same valley. Their learned scale should be read the same way.

## 8. Notes to the pre-registration

- **K9.1's measure assumed a gentle curve along the charges' scale.** At the curvature found, the slope at the end state
  reads the offset from the valley's bottom. The registered verdict stands. The pull is read in §3 from the batch
  means, which the pre-registration listed beside (`training.charges.pull`).
- **The diagnostics took longer than estimated:** 20–23 minutes a run, inside the 3 h asked.

## 9. Proposed next, not registered

- **(a) Follow the valley.** A long full-batch L-BFGS (about 500 steps) from job 9's and job 7's saved end states.
  - Does the valley lead to c* and to the scan's objective?
  - No new Adam; about an hour a run.
- **(b) Make it shallower.** The charges' band with its own edge, as `Rung.disp_l` gave c6.
  - The fall in curvature can be computed from the kernel before any run.
  - Then job 7's scan at the new edge, for its profile and its cost, and the warm starts at job 7's learning rate,
    traced.
