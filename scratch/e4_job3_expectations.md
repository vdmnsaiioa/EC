# E4 job 3: expectations, written before the numbers

*7 Nov 2026, 22:30 CET. Written before MB-pol's finite-k response was computed on any frame, and before the locality test
was run. Already seen, from a timing check of `model.mode_response` on one local training frame with arm C's local
read-out (no MB-pol numbers):*

- *arm C's ε_L at the first harmonic is 1.743, and 1.801 with the mode's gradients left out (the order-0 sector alone);*
- *the gradient terms change the mode's susceptibility by −4.4 % at k₁ = 0.51 Å⁻¹ and −12.5 % at k₂ = 1.01 Å⁻¹;*
- *they change the molecules' dipole response by 3.5 % and 10 % (rms).*

*Notation as in the pre-registration. χ(k) is the cells' longitudinal susceptibility to the external mode,
(χ_cos + χ_sin)/V averaged over the axes, and ε_L(k) = 1/(1 − 4πχ(k)).*

1. **MB-pol's χ(k₁) exceeds its screened k → 0 limit**, (1 − 1/ε₀)/4π, by 2–10 %, and χ(k₂) exceeds it by more.
2. **Arm C's χ against MB-pol's:** within ±3 % at k₁ and ±6 % at k₂, on the held-out frames.
3. **The orders ≥ 1 at finite k** (arm C against its order-0 sector alone): they come from the far probes, which do not
   identify them. At about even odds, they move χ(k₁) away from MB-pol's rather than toward it.
4. **The molecules' dipole response at finite k:** arm C's rmse ÷ rms is about 5 % at k₁ (2.9 % at k = 0 on Niflheim)
   and about 10 % at k₂.
5. **Arm A** is 3–5 % above MB-pol's χ(k₁), as at k → 0⁺: its ε₀ is 1.743.
6. **Arm B** moved its orders ≥ 1 on k = 0 data, which do not identify them either. Its χ(k₁) is no closer to MB-pol's
   than arm C's, and I expect it to be farther, by at least 1 point.
7. **The locality test.** The order-0 read-out calibrated on every liquid molecule's induced dipole and field response
   alone, from zero, with no probe in the objective.
   - Its far probes (non-additive, 10–15 Å) come within 1.5× of the probe-matched order-0 read-out's own errors
     (4.0–7.0 %, local).
   - T5's slope is within 0.95–1.10.
   - That would make the order-0 coefficient local at this cutoff, to the order-0 class's own floor.
   - Less sure than 1–6: about 60 %.
