# E4: the response's regulator, placed by the EFT-of-LSS hierarchy — a scan before training

*4 Nov 2026. Matching only, no training: each read-out matched on MB-pol's far probes exactly as for job 1, then ε_∞
on the pre-registration's 8 liquid frames (MB-pol 1.6876 ± 0.0016), the far probes and T5, as
`e4-x3-preregistration.md` §4 and §6 computed them (`scratch/e4_x3_preregistration.py --parts sweep --sweep_mode
...`, v0.0.23). Each form's expectation was written down before its numbers were read. The local master of this note is
`notes/` in the repository.*

## 1. Why: the paper's hierarchy

Baumann, Nicolis, Senatore and Zaldarriaga, *Cosmological non-linearities as an effective fluid* (arXiv:1004.2488),
the founding paper of the EFT of large-scale structure, builds the long-wavelength theory by smoothing at a scale Λ
with

- k ≪ Λ ≪ q⋆: "an effective long-wavelength theory by 'integrating out' short-wavelength modes below a scale Λ ≪ q⋆"
  (§4.1), q⋆ the non-linear scale; terms mixing long and short modes are suppressed by k/Λ, higher derivatives by
  1/Λ²;
- a few parameters that "can be calibrated by computing the small-scale dynamics with numerical N-body simulations.
  Alternatively, the fluid parameters may simply be retained as free parameters to be measured by fitting predictions
  of the effective theory to observations" (after eq. 1.6).

Our long-range sector smooths the induced dipoles' mutual coupling at the response edge, 0.75 Å. MB-pol damps that
coupling (Thole, a_DD = 0.055, `electrostatics.cpp`) out to hydrogen-bond distances: it keeps 12–46 % of the bare
tensor at O···H 1.9 Å and 24–60 % at O–O 2.8 Å. The coupling's non-linear scale is 2–3 Å, and at 0.75 Å our tensor
is 97.5–100 % bare there: the smoothing length sits below the non-linear scale, on the wrong side of the hierarchy.

The erf-smeared dipole tensor's two radial parts, f₃ = erf x − (2x/√π) e^(−x²) and
f₅ = erf x − (2x/√π)(1 + 2x²/3) e^(−x²), x = r/ℓ:

| ℓ (Å) | at O···H 1.9 Å | at O–O 2.8 Å | at r_c = 6 Å |
|---|---|---|---|
| 0.75 | 97.5–99.5 % | 100 % | 100 % |
| 1.5 | 33–64 % | 78–93 % | 100 % |
| **2.0** | **12–39 %** | **44–73 %** | **99.7–100 %** |
| 2.5 | 5–24 % | 23–53 % | 96–99 % |
| MB-pol's Thole (a_DD 0.055) | 12–46 % | 24–60 % | 99.9 % |

So **2.0 Å was fixed in advance**: the edge that damps the contact coupling as MB-pol does and leaves it bare at r_c.

## 2. Three forms, in the order run

| form | edge (Å) | read-out order | ε_∞ (MB-pol 1.6876) | far probes, non-additive, 8 / 10 / 12 / 15 Å | T5 slope (rmse / rms) | ridge |
|---|---|---|---|---|---|---|
| job 1: everything at 0.75 Å | 0.75 | 0 | 1.8214 | 5.6 / 7.0 / 4.7 / 4.2 % | 0.943 (7.1 %) | 10⁻¹ |
| | 0.75 | 2 | 1.8003 | 1.9 / 2.5 / 2.6 / 2.0 % | 1.026 (3.7 %) | 10⁻¹ |
| (i) bands and response together | 2.0 | 0 | 1.5378 | 29.9 / 32.3 / 25.4 / 18.7 % | 1.742 (77 %) | 10⁻⁵ |
| | 2.0 | 2 | 1.2365 | 5.1 / 5.8 / 8.1 / 5.6 % | 1.242 (30 %) | 10⁻² |
| | 1.5 | 0 | 1.8040 | 9.2 / 13.7 / 14.5 / 13.1 % | 1.284 (29 %) | 10⁻⁴ |
| (ii) the response edge alone | 2.0 | 0 | not read | 30 / 32 / 25 / 19 % | 1.740 | — |
| **(iii) the mutual tensor alone** | 1.5 | 0 | 1.7682 | 5.2 / 6.8 / 4.3 / 4.2 % | 0.956 (5.9 %) | 10⁻¹ |
| | 1.5 | 2 | 1.7740 | 1.8 / 2.2 / 2.1 / 1.8 % | 1.034 (4.1 %) | 10⁻¹ |
| | **2.0** | **0** | **1.7225** | 4.9 / 7.0 / 4.0 / 4.5 % | 0.980 (4.0 %) | 10⁻¹ |
| | **2.0** | **2** | **1.7440** | **1.9 / 2.3 / 2.1 / 2.1 %** | **1.042 (4.7 %)** | 10⁻¹ |
| | 2.5 | 0 | 1.6919 | 4.9 / 7.3 / 4.0 / 4.8 % | 0.997 (3.2 %) | 10⁻¹ |
| | 2.5 | 2 | 1.7208 | 2.3 / 2.9 / 2.3 / 2.4 % | 1.048 (5.0 %) | 10⁻¹ |

ε_∞'s frame-to-frame spread is 0.003–0.005 for every row of (iii).

**(i) Job 2's registered form — the bands' edge and the response's together — failed.** Expected at 2.0 Å, order 0:
ε_∞ 1.64–1.74, the far probes ≤ 5 % at 10–15 Å, T5's slope 0.9–1.1. It gave 1.538, 19–32 % and 1.74. The far probes'
non-additive energy is mostly the cores' own induced dipoles, induced by their permanent charges at contact (E3's
rehearsal: 91–100 % of the probes' induction). Smoothing the field the read-out reads removes them, so the far field cannot be matched any
more. MB-pol does not smooth that field: its charge–dipole damping, a_CD = 0.4, leaves it about 99 % bare at O···H
1.9 Å (a u³ ≈ 4.4).

**(ii) The response edge alone failed the same way.** `Rung.bf_l` sets the field the read-out reads as well as the
mutual tensor. Stopped after 2.0 Å at order 0.

**(iii) The mutual tensor alone** (a new `Rung.bf_l_mutual`; the field the read-out reads stays at 0.75 Å), the
expectation of (i) kept:

- ε_∞ at 2.0 Å, order 0: **1.7225**, inside the expected 1.64–1.74. **Holds.**
- The far field: T5's slope 0.980, inside 0.9–1.1. The probes 4.0–7.0 %: 7.0 % at 10 Å misses the "≤ 5 %" as written,
  and equals job 1's order-0 value there, so nothing degraded. At order 2 the probes are 1.9–2.3 %, as in job 1.
- H3 (the change of ε_∞ from 1.5 to 2.5 Å smaller at order 2 by ≥ 2×): 0.076 at order 0, 0.053 at order 2, a fall
  of 1.4×. **Fails.**

## 3. Reading

- **The non-linear scale is the channel's.** The charge–dipole field is bare at contact in MB-pol and must stay so in
  the model, because the far probes measure exactly that. The mutual coupling is damped at contact in MB-pol and was
  not in the model. The probes carry little of it, so far-field matching could never correct it.
- **The regulator placed by the hierarchy halves the ε_∞ failure before training, with no liquid data and the far
  field untouched.** At the a-priori 2.0 Å the susceptibility is 8.2 % too high at the frozen form's order 2 and 5.1 %
  at order 0, against 16.4 % and 19.5 % with job 1's edge.
- **ε_∞ still runs with the edge.** Order 0 1.821 → 1.768 → 1.723 → 1.692 and order 2 1.800 → 1.774 → 1.744 → 1.721
  at 0.75, 1.5, 2.0 and 2.5 Å: about 0.05 per Å at order 2. The read-out matched on the far field cannot absorb this
  dependence. In EFT terms the mutual coupling at contact carries a counterterm that the probes do not see, and H3's
  fall-off is missing for that reason (the pre-registration's §6: "a counterterm not matched on data that feel it").
- **2.5 Å reads closest to MB-pol (1.692 at order 0) and is not the edge to take.** Choosing the edge by the
  observable is fitting the observable, and at 2.5 Å the tensor is already 1–4 % below bare at r_c.

## 4. What follows

- **Job 2 at the a-priori edge.** The mutual edge at 2.0 Å (`model.mutual_edge_angstrom: 2.0`), read-out orders 0 and
  2, trained as job 1. Its ε_∞ (1.7225, 1.7440), probes and T5 are the numbers above, fixed before training. A bulk
  prediction to register: with the contact over-polarisation halved, the response without the field term should no
  longer trail the dagger on the rigid forces (job 1: +3–4 %).
- **The paper's second move for the remaining 5–8 %.** Calibrate the contact mutual coupling in the full theory where
  it acts: MB-pol's per-molecule induced dipoles in a uniform field on liquid frames, computed with MBX, as matching
  data beside the far probes (64 molecular environments × 3 field directions per frame). ε_∞ is then matched rather
  than predicted. The predictions move to the finite-k response (X2-C), the bulk forces and the interfaces, and H3 is
  read there.
- *Both registered 5 Nov* (`e4-job2-preregistration.md`, v0.0.24): job 2's arm A is the first, arm B the second, the
  calibration on 30 liquid cells (12 numbers each, M and R) beside the far probes. Calibrated before training on 10
  local frames, ε_∞ is 1.688–1.690 at every edge from 1.5 to 2.5 Å and both orders, the far field kept.
