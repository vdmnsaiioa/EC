# E4, job 4: expectations before any arm D number

*Written 8 Nov 2026, 22:49 CET (this commit's time), after job 3's read (`notes/e4-job3-results.md`) and before arm D's third stage was
solved on any frame.*

*Seen before them, on the local frames:*

- *a check of the finite-k rows on one calibration cell with arm C's read-out. The rows equal `model.mode_response`'s
  χ_mode exactly, and are linear in the orders ≥ 1 coefficients (to 1×10⁻¹³). Their Jacobian takes 9 s a cell;*
- *job 3's local readings of arms A, C and C0, and §6's diagnostic over the mutual edge.*

*No stage-3 solve, and so no arm D, Dx or H3 number, was computed before this file.*

## The design these expectations are for

- **Arm D.** Arm C's read-out, then a third stage: the orders ≥ 1 columns (3168 coefficients) solved on the far probes'
  rows and on every calibration cell's χ_mode in the 12 modes, the probes' ridge kept.
  - The rows are relative to MB-pol's mean χ_mode at the harmonic, at weight 0.1: a 1 % error costs about the probes'
    matched objective.
  - The liquid's k = 0 rows are left out: they reach the orders ≥ 1 only through the cells' dipole at zero field.
- **Dx, the control.** The same stage with no finite-k rows: the orders ≥ 1 re-solved on the probes alone, given arm C's
  calibrated order-0 sector.
- **H3.** Arms C and D at the 1.5 and 2.5 Å mutual edges, and their order-0 sectors.

## Expectations (local frames: 10 calibration cells, 3 held out; held-out cells unless said)

1. **Arm D's χ at the 2.0 Å edge sits near the γ-only least-squares point, about +1.3 % at k₁ and −0.4 % at k₂, each
   within ±1.0 point.**
   - The basis is job 3's §6: the orders ≥ 1 add exactly γk² to χ(k), and the order-0 sector's k⁴ term is fixed.
   - With equal weights on the modes, a single γ puts the k₁ error at about 3.3 times the k₂ error, of opposite sign.
   - The many coefficients can fit each mode's cell-to-cell fluctuations, but not the phase-summed k⁴ term.
2. **The held-out cells read within 0.5 points of the calibration cells,** at both k.
3. **ε₀ is arm C's to every printed digit,** since the orders ≥ 1 do not enter a uniform field's response.
   - The cells' dipole at k = 0 moves by less than 2 points from arm C's 3.1 %.
4. **The molecules at finite k come out at 3–7 % at both k,** not worse than arm C's by more than 2 points (3.7 / 4.7 %).
   The orders ≥ 1 fitted to the cells' sums may lose some of the per-molecule pattern.
5. **The far probes pay something.**
   - The non-additive energy comes out at 2.5–5 % at 10–15 Å (arm C 2.2–2.7 %, C0 6.0–7.7 %).
   - The stage's probe objective rises 1.2–3 times above the start.
   - T5's slope stays within 0.95–1.10.
6. **Dx moves χ(k₂) by less than 3 points from arm C's −11.5 %:** the probes do not know the finite-k response. Its
   probe objective falls back toward arm A's (within ×1.2 of 9.90×10⁻⁶).
7. **H3: arm D's χ at the other edges follows §6e's ceilings.**
   - At 1.5 Å, about +0.4 / −0.1 % (each within ±1 point).
   - At 2.5 Å, about +3.2 / −1.0 % (each within ±1.5 points).
   - The edge dependence of χ(k₁) and χ(k₂) from 1.5 to 2.5 Å falls at least twofold from the order-0 sectors to arm D.
     This is by construction at the fitted k.
   - The readings not in the rows (the molecules, the probes, T5, the cells' dipole at k = 0) are H3's out-of-sample
     part. No direction is expected for them.
