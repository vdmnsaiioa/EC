# E4, X3: the pre-registration of job 1 (and of job 2's band-edge sweep)

*2 Nov 2026, before any periodic training. Set up after Vadic's decisions of 2 Nov: job 1 first; the field-term
ablation in job 1 and the band-edge sweep as job 2; the MACE baselines on the boxes only. Every number here comes from
`scratch/e4_x3_preregistration.py`:*

- *MBX v1.4.0 built here at −O0, which changes no number (e4-plan.md §2);*
- *MB-pol's probes as job 1 makes them (`scripts/e4_water.py probes`, hash 72b73e3aa6c7b53f);*
- *for ε_∞, 8 frames of 64 molecules from two validation trajectories run here on 29 Oct (rigid MB-pol MD, labelled
  with E4's truth; not the data sets' frames).*

*The local master is `notes/` in the repository.*

## 1. Job 1

**Data and rungs.**

- Training: the 180 frames of 64 molecules, L₀ = 12.43 Å.
- Rungs, 4 seeds each:
  - M_∞, the short-range control;
  - M_1 and M_16, learned charges (neutral per molecule) and learned α(iω);
  - M_16†ᵐ, the dagger: MB-pol's charges, M-sites and c₆ pinned, no response;
  - M_16C2†ᵐ_us, E3's frozen form;
  - M_16C2†ᵐ, the same without the field term (the ablation);
  - M_G, the global pair function.

**Training** (`configs/e4_water.yaml`).

- E3-QP8's model settings in a cell, with the GMRES solve.
- 3000 Adam steps on 6 cells, then 50 L-BFGS steps.
- The energy weight is 100 against 1 for the forces, as in E3-QP2 on.
- The forces enter through their rigid-body part (§2a).

**Readings.**

- Energy per molecule and forces on every frame of 64, 128 and 256 molecules. At 128 and 256 every split is unseen.
- The far-field test probes and the far trimers (T5).
- ε_∞ on 10 test frames (C rungs).
- Later, once the readers come: Form I, the near field, and the 512- and 1024-molecule frames.

**Cost.** `run_e4_job1.sbatch` runs it as 4 jobs of the same length, one per seed, each training the 7 rungs one after
another on its GPU (the C rungs first). On the RTX 3090 a C rung takes about 5 h, M_16 3 h, M_1 and M_16†ᵐ 1.5 h each,
M_G and M_∞ under 1 h each: about 16 h per job, 65 GPU hours in all (2 Nov: every rung's training step compiled and
run in a cell, its cost relative to the C rung's measured on CPU).

## 2. Two things fixed before training

**(a) The forces' rigid-body part.**

- MBX's atomic forces on rigid molecules carry the flux of MB-pol's geometry-dependent charges, Σ_a φ(s_a) ∂q_a/∂r_i.
  This term has no net force and no torque on any molecule: the charges depend on the internal coordinates only.
- Rigid pinned charges cannot represent it. Its long-range part, the potential of the whole cell, no local model can.
- So the loss reads each molecule's net force and torque: the Euclidean projection of the residual on its rigid-body
  motions (`train.rigid_force_sums`, checked in `tests/test_e4_train.py`).
- That is all a rigid frame's dynamics uses, and it is the same data for every rung.
- The readings report the atomic forces too.

**(b) The matching fixes the infrared.** Three facts of this model make the IR readings computable before training:

- a uniform field reaches only the read-out (E₀ and the field term never see it);
- a cluster's images in a cell beyond r_c reach only the bands and the read-out;
- the read-out is matched on the open probes and then held, unchanged, through training (`train.fit(...,
  readout_init=...)`).

So ε_∞, the far field, T5 and the Form I tail of contents whose images lie beyond r_c are computed here, exactly, for
the dagger and the C rungs. Training cannot move them. Job 1 reads them again; any difference beyond round-off is an
implementation error.

*Consequence for the ablation.*

- M_16C2†ᵐ and M_16C2†ᵐ_us share the matched read-out exactly: the species read-out's initialisation is fixed, and the
  field term is blind to probes beyond r_c.
- Their IR readings are therefore identical by construction. In my reply to Vadic of 2 Nov I offered the ablation as a
  test for leakage from the short range into the IR; with the read-out held there is none to find.
- What the ablation measures is what the field term buys at contact: energies and forces per molecule at 64–256
  molecules, and later the near-field tetramers.

## 3. The truth, MB-pol

**Far-field probes** (80 test probes: a relaxed trimer and one molecule at d; rms in kcal/mol):

| d (Å) | 8 | 10 | 12 | 15 |
|---|---|---|---|---|
| probe interaction | 0.1573 | 0.0722 | 0.0320 | 0.0156 |
| its non-additive part | 0.0317 | 0.0133 | 0.0071 | 0.0039 |

**Far trimers (T5).** 40 trimers, every intermolecular atom pair beyond 6.5 Å, O–O at most 12 Å. MB-pol's 3-body
polynomial is off there, so their three-body energy is the many-body induction alone: rms 2.9×10⁻⁴ kcal/mol, largest
1.1×10⁻³.

**Form I**, Δ⋆(L) = E(L) − E(open) in kcal/mol (cubic cells of side L; tetragonal (L, L, 2L)):

| content | c15 | c20 | c25 | c30 | c40 | t15 | t20 | t25 |
|---|---|---|---|---|---|---|---|---|
| hexamer, μ_perm 2.89 D, μ_ind 0.64 D | −0.1113 | −0.0445 | −0.0210 | −0.0111 | −0.0036 | −0.0396 | −0.0100 | −0.0028 |
| tetramer (Cᵢ, μ = 0) | +0.0280 | +0.0071 | +0.0024 | +0.0010 | +0.0002 | +0.0198 | +0.0047 | +0.0016 |
| one molecule, 1.864 D | −0.0292 | −0.0126 | −0.0066 | −0.0038 | −0.0016 | +0.0219 | +0.0090 | +0.0045 |

The contents:

- *The hexamer*: E3's recipe on MB-pol (300 Metropolis steps at 300 K). Its largest atom–atom distance is 7.65 Å.
  MBX's open and periodic dispersion differ for pairs beyond 5.5 Å, so its Δ⋆ carries a constant offset. A fit
  c + a/L³ + b/L⁵ + e/L⁶ on the cubic cells gives c = +1.4×10⁻³ kcal/mol, subtracted below.
- *The tetramer*: a relaxed dimer and its inversion image, the most compact bound one of 400 tries (5.13 Å across,
  E −7.19 kcal/mol). Its total dipole is zero by symmetry, permanent and induced, so its tail starts with the
  quadrupole images (L⁻⁵).
- *One molecule*: the check of the boundary condition.

**ε_∞** (clamped nuclei, tin-foil):

- MBX's induced dipoles in a uniform field on 8 frames of 64 molecules: **1.6876 ± 0.0016**.
- The monomer's polarisability is 1.4075 Å³ (eigenvalues 1.313, 1.414, 1.496).
- 1 + 4πnα = 1.590 and Clausius–Mossotti 1.734 bracket the truth: the liquid's local-field enhancement over first order
  is 1.166.

## 4. Computed predictions (fixed by the matching)

**Matching.**

- E3-QP2's method: every order of the species read-out, 4 Gauss–Newton steps, the ridge chosen on held-out cores.
- It runs on the 328 of 384 training probes whose probe lies beyond r_c of every core atom.
- The objective falls from 2.44×10⁻² to 9.39×10⁻⁶; the ridge chosen is 10⁻¹, as on q-AQUA-pol.

| reading | dagger M_16†ᵐ | C rungs M_16C2†ᵐ(_us) | every rung without a response¹ |
|---|---|---|---|
| far probes, non-additive, 8 / 10 / 12 / 15 Å | 100 / 100 / 100 / 100 % | **1.9 / 2.5 / 2.6 / 2.0 %** | 100 % (M_∞)² |
| far probes, interaction | 21 / 19 / 22 / 25 % | 1.4 / 0.5 / 0.7 / 0.5 % | — |
| T5, three-body rmse / rms; slope | 100 %; 0 | **3.7 %; 1.026** | 100 %; 0 exactly |
| Form I, hexamer, ratio to Δ⋆ − c, c15 → c40 | 0.60, 0.68, 0.72, 0.74, 0.78 | **1.006, 1.007, 1.013, 1.022, 1.054** | M_∞: 0; M_G: 0 from L = 16 Å |
| … tetragonal t15 → t25 | 0.51, 0.60, 0.68 | 1.017, 1.039, 1.085 | |
| Form I, tetramer, c15 → c25 (t15 → t25) | 0.64, 0.67, 0.69 (0.56, 0.58, 0.60) | **1.008, 1.004, 0.996 (1.012, 1.005, 0.993)** | |
| Form I, one molecule | 0.998–1.001 | 0.999–1.001 | |
| ε_∞ | 1 exactly | **1.8003 ± 0.0021** | 1 exactly |

¹ M_∞, M_1, M_16, M_G and MACE-LES: no source answers a far field.

² M_1 and M_16 can carry a core's internally induced dipoles (§5), so their probe reading is measured, not fixed.

*Reading the Form I rows.*

- The dagger's leading L⁻³ coefficient is 0.80 of the truth's. That is |M_perm|²/|M_perm + M_ind|² = 0.78, the
  induced share of the hexamer's dipole, as e4-plan.md §6 said.
- The C rung is within 1–2 % up to 25 Å for both contents.
- At 30–40 Å, where the tails are 10⁻³–10⁻⁴ kcal/mol, its ratios drift: the hexamer's up to 1.05 (within the fitted
  offset's uncertainty, about 3×10⁻⁴ kcal/mol), the tetramer's down to 0.89 (3×10⁻⁵ kcal/mol at 40 Å, unexplained).
  The readers' 40–60 Å cells settle it.
- One molecule's own image polarisation, the difference between the two rungs, is 0.1–0.2 % of its tail, as T6 says
  it should be: second order.

**ε_∞: the e4-plan's prediction fails before training.**

e4-plan.md §6 said the C rung's ε_∞ would come within a few per cent. It comes out at 1.800 against 1.688: the
susceptibility is 16 % too high.

| | the model | MB-pol | ratio |
|---|---|---|---|
| the monomer's polarisability (Å³; the model's in a 30–40 Å cell) | 1.4436 | 1.4075 | 1.026 |
| first order in the liquid, 1 + 4πnα_eff | 1.6155 (α_eff 1.470 Å³) | 1.590 | χ 1.044 |
| self-consistent, ε_∞ | 1.8003 | 1.6876 | χ 1.164 |
| local-field enhancement, χ / χ₁ | 1.300 | 1.166 | 1.115 |

- Most of the excess is the local field, the response's dipole coupling between neighbours. The polarisability itself
  is matched to 2.6 %.
- The response's dipole tensor at the 0.75 Å edge is all but bare at a hydrogen bond: 97.5–99.5 % of the bare tensor's
  radial parts at O···H 1.9 Å, 100 % at O–O 2.8 Å.
- MB-pol damps the same coupling by Thole's form with the intermolecular a_DD = 0.055 (MBX, `sys_tools.cpp`): it keeps
  12–46 % of it at O···H 1.9 Å and 24–60 % at O–O 2.8 Å.
- The matching cannot correct this. It sets the read-out's polarisabilities; the coupling's short-range form is set by
  ℓ₁ alone.
- In EFT terms (proposal §3.5) the short-range coupling below the band is integrated out into screened, environment-
  dependent polarisabilities. The read-out's order-0 tensors are that counterterm. The far probes fix them on the
  cores' environments (3–5 molecules); the liquid's hydrogen-bonded environments are not among them.
- So ε_∞ here is an analytic low-energy constant, set by short-distance physics that the matching data do not reach.
  At order 0 the band-edge sweep (§6) confirms it: 1.821 at 0.75 Å, 1.833 at 1.0 Å, the 1.0 Å tensor still 99 % bare
  at 2.8 Å. At order 2 the counterterm turns out not to be identified at all (§6).
- The remedy the EFT points to is matching data that reach the counterterm: MB-pol's response to a uniform field on
  clusters cut from the liquid (a molecule and its first shell), a fragment-level matching condition (proposal §2).
  It is proposed as an arm of a later job, not run in job 1. Its ε_∞ is then computed before training, as here.
- X4b's "few per cent" pass is therefore not expected from far-field matching alone.

## 5. Predictions for training, with their criteria

**Bulk errors, 64–256 molecules.** These separate no classes (Form II, les-vs-eft caveat 2); they are recorded so that
nothing is read into them afterwards.

1. The C rungs lie at or below the dagger's per-molecule energy rmse and rigid-force rmse at every size (paired by
   seed, ≥ 3 of 4 seeds). The dagger lacks the induction, which E₀ can only partly absorb inside 6 Å.
2. M_16C2†ᵐ_us lies at or below M_16C2†ᵐ on both, at every size, on ≥ 3 of 4 seeds: the field term's gain at contact
   (E3-QP8: raw 2-body −25 %, raw 3-body −35 %). No magnitude is predicted.
3. No rung's per-molecule energy bias moves from 64 to 256 molecules by more than twice its frame-to-frame standard
   error, except M_G's.
4. M_G's pair function sees pairs only up to L/2: 6.2 Å in training, 7.8 and 9.9 Å at 128 and 256 molecules. It
   extrapolates past its training range there, and its bulk error grows with size, as on the dimers (E1: linear
   extrapolation). This is Form III, measured.

**The learned-charge rungs M_1, M_16.**

- A probe's core lies inside r_c, so environment-dependent charges can carry the core's internally induced dipoles,
  the leading R⁻³ part of the probe's non-additive energy, in the molecular plane only (three charges per molecule).
- Their far-probe reading is therefore predicted between the C rungs' and 100 %. This corrects e4-plan.md §6, which
  gave MACE-LES 100 % there.
- On T5 they give exactly zero, and their ε_∞ is exactly 1.

**MACE and MACE-LES** (Vadic's baselines, boxes only): T5 exactly zero and ε_∞ = 1 for MACE-LES (Theorem 2); the probes
and Form I are measured. `scripts/e4_far.py` gives the probes and the trimers as extxyz and reads any model's energies
of them the way the driver reads the harness's rungs.

**Gate C** cannot be read in job 1.

- M_G's pair function runs over minimum-image pairs, so a cluster in a cell wider than twice its size has no image term:
  its Form I relative error is 1 by construction.
- Gate C as an open question needs a learned global function that sees the images, such as learned filters on the
  k-grid. None is in the harness. This corrects e4-plan.md §4.

**The near field.** Carried from E3: the C rung's 4-body error on the liquid's compact tetramers (closest O–O below
2.7 Å) exceeds the labels' rms (E3: 0.57 against 0.34). It is read by the readers.

## 6. Job 2: the band-edge sweep (registered now, run after job 1)

**Design.**

- ℓ₁ = 0.75 and 1.0 Å, the bands' and the read-out's edges together.
- Read-out order 0 (M_16C0†ᵐ_us) and order 2 (M_16C2†ᵐ_us).
- Each matched as in job 1, trained as in job 1, 4 seeds.

**The matched part, computed now:**

| ℓ₁, read-out order | ridge chosen | ε_∞ (MB-pol 1.688) | far probes, non-additive 8 / 10 / 12 / 15 Å | T5 rmse; slope | hexamer Form I c15 / c25 / t15 |
|---|---|---|---|---|---|
| 0.75 Å, order 0 | 10⁻¹ | 1.8214 ± 0.0051 | 5.6 / 7.0 / 4.7 / 4.2 % | 7.1 %; 0.943 | 0.993 / 0.993 / 1.024 |
| 0.75 Å, order 2 (job 1's) | 10⁻¹ | 1.8003 ± 0.0021 | 1.9 / 2.5 / 2.6 / 2.0 % | 3.7 %; 1.026 | 1.006 / 1.013 / 1.017 |
| 1.0 Å, order 0 | 10⁻¹ | 1.8332 ± 0.0031 | 5.4 / 7.0 / 5.5 / 5.0 % | 5.2 %; 0.966 | 1.007 / 1.005 / 1.039 |
| 1.0 Å, order 2 | 10⁻³ | **1.3895 ± 0.0046** | 2.7 / 3.3 / 2.3 / 2.4 % | 13.6 %; 1.133 | 1.009 / 1.015 / 1.011 |

The regulator dependence |O(1.0 Å) − O(0.75 Å)| at order 0 and at order 2, and its fall from order 0 to 2:

| observable | order 0 | order 2 | ratio (H3: ≥ 2) |
|---|---|---|---|
| ε_∞ | 0.012 | 0.411 | **0.03** |
| T5 slope | 0.023 | 0.107 | **0.2** |
| far probes, non-additive (10–15 Å, mean) | 0.5 points | 0.5 points | ≈ 1 |
| hexamer Form I, c15 / c25 / t15 | 0.014 / 0.012 / 0.015 | 0.003 / 0.002 / 0.006 | 4.7 / 6 / 2.5 |

**Criterion** (H3; proposal §3.4): the regulator dependence of each IR observable falls with the read-out's order,
by a factor ≥ 2 from order 0 to order 2 once the polarisation channel is on. It is read here on ε_∞ and T5, and on the
modulation response (X2-C) of the trained models.

- A ratio near 1 is H3's failure signature: the response's range not inside the band (proposal risk 8), or a counterterm
  not matched on data that feel it.
- Either way it is reported as the model's, not as noise.

**What the sweep's matched part already says.**

- On the observables the probes see directly (the far field and the Form I tails of small clusters, whose dipoles are
  the cores'), the matched response hardly depends on ℓ₁, and on Form I the dependence falls with order as H3 asks.
- On the liquid's ε_∞ it depends on ℓ₁ more at order 2 than at order 0, thirty-fold. The fall-off H3 asks for is
  inverted.
- The cause is in the matching, not the kernel:
  - at 1.0 Å the held-out cores choose a ridge of 10⁻³ instead of 10⁻¹;
  - the read-out's environment-dependent terms then grow;
  - in the liquid, where every molecule has far more neighbours than in a 3–5-molecule core, they carry the response to
    1.39.
- The probes do not identify the read-out's environment dependence: two matchings fit them equally well (2–3 %) and
  differ by 0.41 in the liquid's ε_∞. This looks like Lemma 3's problem again (proposal §3.3), now on the response: every
  probe shows the core a far field of one kind, so the response to the liquid's near fields is left to the ridge.
- So ε_∞ is decided by regularisation, between 1.39 and 1.83 across these four settings, with MB-pol at 1.69. It is
  not predicted by the far-field matching.

**What follows.**

- Job 1 is unchanged. It is E3's frozen form on a new truth, and its ε_∞ (1.800) is a known, recorded failure.
- Job 2 as registered (the four settings trained, then read on the modulation response) is still runnable. But its ε_∞
  and T5 readings are already computed here, and the trained models would carry the same unidentified counterterm.
- Before job 2 runs I will propose to Vadic to add the matching data the EFT calls for: MB-pol's response to a uniform
  field on clusters cut from the liquid (a molecule and its first shell). Then:
  - the read-out's environment dependence is identified on the liquid's own environments;
  - ε_∞ becomes a prediction from fragment-level matching, computed before training as here;
  - H3 is read on that.
- Proposed, not run.

## 7. What would change the reading

- **Job 1's IR readings differ from §4 beyond round-off**: an implementation error, found and fixed before anything else
  is read.
- **The bulk predictions of §5 fail**: that is information about E₀ and the training, not about the EFT.
- **The probes' hash on Niflheim differs from 72b73e3aa6c7b53f**: the probes are not these. The matching and §4 are then
  computed again there before job 1's results are read.
