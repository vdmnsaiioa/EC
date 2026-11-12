# One field for induction and dispersion: v2's long-range sector — a proposal

*12 Nov 2026. Written after Vadic's question of 12 Nov — induction and dispersion come from the same path integral at
the electronic level — and his "yes" to a note. Read against:*

- *the project's `path-integral-ir-sector.md` §2 (the saddle and the loop);*
- *`variational-structure-ir-sector.md` §4 (the one-oscillator Lagrangian, its (9)–(11), §4.3 (a));*
- *`effective-action-ir-sector.md` §2.2–2.3 (S_disp; R1, R10);*
- *`nuclear-eft-ir-sector.md` §5.2;*
- *`e4-plan.md` §2 and `e4-s0-results.md`.*

*Nothing here was run on data. The tie reading of §5.1 is in the S1 patch (`scripts/e4_sumrules.py`). Everything else is
proposed, not registered.*

## 1. The statement

At clamped nuclei the electronic path integral over the induced dipoles is Gaussian.

- **The saddle is the induction solve**, at O(ħ⁰).
- **The determinant is the Casimir–Polder Tr ln**, at O(ħ). That is dispersion, many-body included (MBD).
- **There is nothing else.** The two separate exactly, with no cross term (the path-integral note's (3)–(4), §2.3 (a)).

**One field, one α.** One α_i(iξ) per site and one dipole tensor T serve both channels. R10's tie, C_i(0) = α_i(0), is
not an extra constraint: it is what one fluctuating field per site means.

**Already checked:**

- The variational note's normal-mode zero-point energy, its (11), equals the Tr ln to 10 digits on four oscillators. The
  pairwise London sum is 0.2 % off: that is the three-body part.
- E3-R measured a_eff/α = 1.05 and 1.00 on the isolated monomer. That is the induction side of the tie.

## 2. What E4's model does now

**It has two fields.**

- **Induction.** The order-0 sector's local polarisabilities C_i(0), from learned features, solved mutually above
  ℓ_m = 2.0 Å. They are matched on MB-pol's induced dipoles in liquid cells (arm C) and on its finite-k response (D′).
- **Dispersion.** The pair band above ℓ₁ = 0.75 Å, with α_i(iω) pinned by `water.mbpol_dispersion_pins`. It carries
  MB-pol's long-range per-site c6 on a fixed oscillator shape (ω = 0.5 E_h). Its static value is not MB-pol's
  polarisability, by construction.

**The loop's many-body remainder is absent from the model and from MB-pol alike.** MB-pol's long-range dispersion is
pairwise, and its explicit 3-body term is short-ranged.

**MB-pol is not one field either.** It uses Thole polarisabilities for induction and fitted c6 for dispersion. Its O–H
c6 also differs by 17 % between ranges: 83.50 kcal/mol Å⁶ in the short-range pair term, 69.06 in the long-range per-site
product (`e4-plan.md` §2).

## 3. What one field buys

**(a) Dispersion becomes a prediction.**

- Take one oscillator per site: α_i(iξ) = C_i(0)/(1 + ξ²/ω_Z²).
- The only new numbers are the frequencies ω_Z, one per species, matched on one gas-phase C₆ each.
- Everything that depends on the environment then follows from the response already matched (ε_∞ and χ(k)), including
  how C₆ changes from the gas to the liquid.
- This is the EFT-of-LSS logic: the loop is computed from the tree-level couplings, and its UV parts go to the
  counterterms — here the contact sector, by R1.

**(b) Many-body dispersion and the medium's screening, from the Tr ln.**

- In the liquid, the London coefficient between two molecules is set by ∂ln ε(iξ)/∂n, not by the bare α. This is
  McLachlan's in-medium result, the n ≥ 1 counterpart of the nuclear EFT note's (8).
- A Clausius–Mossotti estimate (mine) gives the in-medium α_eff/α = (ε + 2)²/9ε.
  - At ξ → 0, with ε_∞ = 1.69, that is 0.895, so C₆ is screened by 20 %.
  - At ξ ≈ ω, where ε(iξ) ≈ 1.34, the screening is 15 %.
  - Over the Casimir–Polder integral it comes to roughly 15–20 %.
- A pairwise c6 cannot carry this.

**(c) One learned field instead of two.** That is an inductive bias: the dipole data inform the dispersion, and the
dispersion data inform the response.

## 4. The design

- **The polarisabilities.** α_i(iξ) = C_i(0)/(1 + ξ²/ω_Z²). C_i(0) is the read-out's order-0 polarisability, its
  isotropic part or the full tensor. ω_Z is learned, or fixed from gas-phase C₆.
- **The pair band** stays as now, above ℓ₁ = 0.75 Å. It is built from these α_i(iξ) on the K = 8 Casimir–Polder grid,
  so C₆_ij = (3/π) Σ_k w_k α_i α_j now depends on the environment.
- **The many-body remainder** is (1/2π)∫dξ Tr ln[1 − α(iξ)T_m] minus its own pair part.
  - T_m is the induction's mutual tensor, at ℓ_m = 2.0 Å. At ξ = 0 the operator is the induction solve's, so it needs
    the same regulator, or the loop meets a polarisation catastrophe.
  - The difference between the pair term at ℓ₁ and at ℓ_m is analytic (R1), and E₀ takes it.
- **Cost and forces.**
  - Direct log det at 8 frequencies costs O(N³) each. That is fine at 64–128 molecules, where the cell's T is already
    built.
  - Beyond that: stochastic Lanczos quadrature, with the existing matrix-free dipole product
    (`periodic.dipole_matvec_pbc`). The forces come through Tr[(1 − αT)⁻¹ ∂(αT)] with the same probes.
  - Stochastic forces are noisy, so MD with v2 uses the direct form at the sizes it runs.
- **Cold electrons.** The loop is the zero-point integral, as the grid already is. Classical Drude MD would add
  London's 0.6 %, with the wrong temperature dependence (the path-integral note §2.3 (c)).
- **Hyperpolarisabilities** break the exact separation only at third order in T. For water that is 1–2 % at contact
  (the action note §3).

## 5. The tests, in order

### 5.1 Now: the tie, read on the frozen form

This is in the S1 patch, and not registered.

- **What is stored.** S0's model step (`scripts/e4_sumrules.py model`) stores tr C_i/3 for every atom on the liquid
  frames.
- **What the read step reports**, per species and per molecule:
  - α(0)'s mean and spread over the liquid;
  - the pins' α(0) beside it: the present gap between the two fields;
  - the ω that MB-pol's long-range c6 needs from that α(0), with one oscillator: ω = 4C₆/3α². The gas-phase monomer
    needs 0.65–0.67 E_h;
  - the cross term C₆_OH this tie then implies, against MB-pol's;
  - the implied C₆'s spread over the liquid at fixed ω. A tied model would carry that spread where MB-pol's c6 has
    none.
- **How to read it.**
  - ω per molecule near 0.65 E_h and a spread of a few % mean that tying costs little against MB-pol.
  - ω far off, or a large spread, mean that this truth's induction and dispersion are not one field. E4 is then not
    the place to test v2's tie, and §5.2 is.

### 5.2 A truth built as one field

- **The truth, MB-pol₁.** MB-pol with its long-range pairwise dispersion replaced by the MBD of its own Thole
  polarisabilities, using one ω per species fitted to its c6.
  - It is computed on E4's frames from MBX's sites and polarisabilities with the harness's Tr ln, which v2 needs
    anyway.
  - The MD driver can give it trajectories of its own if the frames' ensemble matters.
- **The experiment.** Train tied and untied models on 64-molecule cells with 25, 50 and 100 % of the frames. Then read:
  - the long-range dispersion on clusters and on the 512–1024 cells;
  - the many-body dispersion on trimers;
  - the bulk errors.
- **The question:** when the truth has the tie, does the model's tie buy data efficiency and transfer?

### 5.3 The DFT phase

The physically meaningful test needs a truth whose induction and dispersion come from one electronic response: RPA, or
MBD-corrected DFT with the same polarisabilities (`e4-plan.md` §8).

## 6. What would change the plan

- **§5.1 shows MB-pol far from one field.** Then §5.2 is the only test at E4's scale, and v2's dispersion is checked
  against a truth built for it.
- **§5.2 shows no gain in data efficiency or transfer.** Then the tie stays a statement of the theory, and the pinned
  pair band stays.

## 7. Order

- None of this displaces S1, the main line (`e4-s1-preregistration.md`).
- §5.1 runs with S0's model step, in minutes.
- §5.2 comes after S1. It needs the Tr ln, which is v2's first build item.
- §5.3 belongs to the DFT phase.
