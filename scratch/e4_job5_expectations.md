# E4, job 5: expectations before any D′ number

*Written 10 Nov 2026, 14:06 CET (this commit's time). That is after job 4's training read
(`notes/e4-job4-results.md` §2) and before arm D′'s anchored third stage was solved on any liquid frame.*

*Seen before them:*

- *the anchored solve's tests on synthetic clusters and cells (`tests/test_arm_da.py`), none on a liquid frame;*
- *job 4's local and Niflheim readings of arms C, D and Dx, and the local change at contact
  (`scratch/e4_job4_contact.log`).*

*No anchored stage was solved on the local or Niflheim frames before this file, so no D′ number existed.*

## The design these expectations are for

- **Arm D′ (label Da).** It is arm C's read-out, then arm D's third stage anchored at arm C.
  - It keeps arm D's stage otherwise: the orders ≥ 1 columns alone, the same far probes' rows, the same finite-k rows
    (12 χ_mode a calibration cell, weight 0.1) and the same ridge.
  - The ridge pulls toward arm C's orders ≥ 1 instead of zero.
  - The far probes' rows are held at arm C's own predictions instead of MB-pol's energies. They resist any change of
    what arm C predicts for them.
  - So the orders ≥ 1 move from arm C's values only where the finite-k rows need it.
- **Read beside it** are arm C (the reference), and arm D and the control Dx from job 4's caches.
- **A new reading, the change at contact** (`kread.contact`). On the training size's test frames, it gives each
  read-out's own energy and rigid forces, and their change against arm C's. E₀ and the field term at initialisation
  cancel, as in `scratch/e4_job2_calibration_orders.py`. It also gives the orders ≥ 1 coefficients' change, relative
  to arm C's norm.

## Expectations

These are for the local frames: 10 calibration cells and 3 held out, and 3 test frames for the change at contact.
Held-out cells unless said.

1. **D′'s χ reaches the frozen form's ceiling, as arm D's does.** χ(k₁) is within ±0.5 points of arm D's local
   +1.40 %, and χ(k₂) within ±0.5 points of −0.46 %.
   - The ceiling is γ's: one k² coefficient against the order-0 sector's fixed k⁴ term (job 3 §6e).
   - The far probes see γ only weakly (Lemma 3), so holding them costs the fit little.
2. **The held-out cells read within 0.5 points of the calibration cells**, at both k.
3. **ε₀ is arm C's to every printed digit.** The cells' dipole at k = 0 moves from arm C's 3.1 % by no more than arm
   D's +1.0 point.
4. **The molecules at finite k come out between arm C's and arm D's.**
   - At k₁ they are within +0.9 points of arm C's 3.7 % (arm D's 4.6 %).
   - At k₂ they are within +2.3 points of arm C's 4.7 % (arm D's 7.1 %).
   - Arm D lost part of the per-molecule pattern in its re-fit; D′ keeps arm C's coefficients but along what χ needs.
5. **The far probes stay at arm C's.**
   - The non-additive energy at 10, 12 and 15 Å is within ±0.1 point of arm C's 2.3 / 2.2 / 2.7 %. Arm D was +0.2 to
     +0.5 points; Dx, which re-fitted them, was +0.2 to +0.6.
   - T5's slope is within ±0.005 of arm C's 1.040.
   - The probes' objective against MB-pol after the stage is 0.9–1.3 times stage 2's (arm D's 0.53).
6. **The change at contact is well below arm D's.**
   - D′'s rigid forces move by 0.05–0.25 kcal/mol/Å rms against arm C's: at most 0.6 of arm D's 0.42.
   - Its orders ≥ 1 move less than arm D's, in norm relative to arm C's.
   - The basis: Dx, the re-fit alone, moved them 0.49, and arm D 0.42. D′ has no re-fit, and what the finite-k rows
     need is mostly one direction, γ. But arm C's γ is three times the ceiling's (job 3), so that direction must move
     a long way. It is not known how much of it the liquid's forces see.
7. **Arms C, D and Dx reproduce job 4's local readings to the printed digit** from their caches. Arm D's change at
   contact against arm C is the contact log's 0.422 rms, and Dx's 0.485, on the same 3 frames.

*Not registered here: if 6 holds, I expect D′ to train within ±1.5 % of arm C on both rungs. The training predictions
are registered after the matching level, as job 4's were.*
