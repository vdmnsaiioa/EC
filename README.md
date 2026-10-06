# EC — the scale-separated EFT model (harness v0)

`sseft` is a compact JAX implementation of the model studied in the project notes:

    E = E_0[local equivariant network]  +  Σ_c  IR band of channel c on learned (or pinned) sources
                                        [+ band-field inputs read at the atom, degree 2]

The incumbent cutoff model, the single-band EFT rungs, the analytic-kernel rung and the v1.5 band-field
read-out are all *configurations* of one code path (`sseft.model.Rung`), so that every comparison in the
programme is between models that differ only in the piece under test.  Everything is float64, forces come
from autodiff, and the measurement utilities of the Phase 1c protocol (ensemble spread, local exponent,
closure, regulator level) ship with it.

Internal units are atomic (bohr, hartree); `sseft.units` converts from/to Å and eV at the interfaces.

## Install

    pip install -e .            # jax, optax, numpy, scipy (tested with jax 0.10, optax 0.2.8, numpy 2.4, scipy 1.17)
    pytest -q tests/            # ~2.5 min on a CPU

For GPUs install the matching `jax[cuda]` wheel first.

## The rungs

| rung    | bands                | sources                | band fields | stands for |
|---------|----------------------|------------------------|-------------|------------|
| `M_inf` | none                 | –                      | –           | the incumbent (cutoff model); interaction energy exactly 0 beyond `r_cut` |
| `M_A`   | analytic pair kernel | scalar `s_i` learned   | –           | learned kernel analytic in `q²` (branch 3 of Theorem 3, `x1-prediction.md`) |
| `M_1`   | Coulomb              | `q_i` learned (+`μ_i` optional) | –  | LES / 4G-type: Coulomb added, no dispersion band |
| `M_6`   | dispersion           | `α_i(iω)` learned      | –           | EFT leading order for neutral fragments (L6a) |
| `M_16`  | Coulomb + dispersion | `q_i`, `α_i` learned   | –           | both bands, sources free |
| `M_16p` | Coulomb + dispersion | pinned                 | –           | same architecture, sources from an independent calculation (the dagger models) |
| `M_S0`  | as `M_16p`           | pinned                 | order 0     | v1.5: `E_1`, `|E_1|²`-type invariants at the atom |
| `M_S2`  | as `M_16p`           | pinned                 | order 2     | v1.5: field, gradient and Hessian tensors at the atom, degree 2 |

Every energy the model returns is an interaction (atomisation) energy: the model's own isolated-atom
energies are subtracted inside `energy_single`, so non-interacting fragments give exactly zero and the
asymptotic measurements are never contaminated by a constant.

Pieces (one module each):

* `e0.py` — PaiNN-style message passing with scalar, vector and traceless rank-2 tensor channels; Bessel
  radial basis with a cosine cutoff; messages are exactly zero beyond the cutoff (no bias in the radial
  weights).
* `heads.py` — site energies (data-derived output scale), charges with an exact neutrality shift, gated
  dipoles, dynamic polarisabilities at K = 8 Gauss–Legendre imaginary frequencies (softplus or
  one-oscillator), the scalar source of `M_A`.  Each source can be *pinned* to values carried by the structure.
* `kernels.py` — the split kernels: `erf(r/l)/r` and its derivatives (p = 1, dipoles), the `p = 6`
  long part `[1 − e^{−x}(1 + x + x²/2)]/r⁶` with its small-`x` series, the Gaussian Laplacian family
  `(−l²∇²)ⁿ e^{−r²/l²}` (exact rational recursion) for `M_A`, the Casimir–Polder quadrature.
* `bands.py` — the band energies for open systems (direct pair sums; Ewald versions are the week-2 item).
* `bandfields.py` — band-field tensors at the atom by nested forward-mode differentiation of the band
  potential (own charge excluded), and the degree-2 equivariant read-out with environment-dependent
  coefficients from `E_0`.
* `model.py`, `train.py`, `measure.py`, `data.py`, `toys.py` — assembly, training (Adam + optional
  L-BFGS polish, targets scaled never shifted), the Phase 1c / X2 measurements, the Tang–Toennies dimers and
  windows of `phase2-reference-data.md`, and the X2-B dielectric toy.

## Scripts

    python scripts/dimer_ladder.py --system Ar2 --rungs M_inf,M_6,M_A --seeds 4 --steps 1500
    python scripts/x2b_toy_regression.py --steps 1500 --lbfgs 200

`dimer_ladder.py` trains an ensemble of each rung on the Phase 2 window (40 log-uniform nodes on
`[R₋, 1.8 R₋]`, label noise `σ = 1e-9 E_h`) and prints the Phase 1c table on the evaluation grid out to
500 Å: ensemble mean, relative error, spread `W`, `W/|f*|`, local exponent `q`, and the plateau slope of
`log(W/|f*|)` on [30, 300] Å (branch 2 predicts 0).  `x2b_toy_regression.py` trains the band-field path
(orders 0 and 2) on the X2-B toy of `x2-protocol.md` §3 (truth T3, `a/l₁ = 1/4`).

## What v0 reproduces

* Forces of every rung agree with four-point finite differences to the float64 floor (`tests/`), the
  Coulomb band with learned/pinned charges and dipoles reduces to the closed-form two-body energy and to the
  classical multipole limit, the dispersion band to `−C₆ f₆(R)` and to London's formula, and energies are
  invariant (forces covariant) under rigid motions for the vector- and tensor-channel rungs.
* Ar₂, Tang–Toennies truth, window [5.15, 9.26] Å, `r_cut = 6` Å, 4 seeds (`scripts/dimer_ladder.py`):
  `M_inf` has relative error exactly 1 beyond the cutoff (branch 3); `M_6` has local exponent `q = 6.00`
  at every evaluation point with a flat relative spread `W/|f*| ≈ 1.2e-3` (the branch-2 plateau) and a
  16–25 % relative error from the `C₈, C₁₀` truncation (learned `C₆ = 80.4` vs 64.3 `E_h a₀⁶`, the L6a
  bias of the proof document).  A `C₈` channel (quadrupole polarisability) is the L6b item of the build plan.
* The X2-B toy: see `RESULTS.md`.

## Layout

    sseft/        the package
    scripts/      the experiments (each prints its table and writes a json)
    tests/        pytest (fast; the ladder and the toy are scripts)
    RESULTS.md    running log of the regression numbers per commit

The design, the predictions and the measurement protocols live in the project notes
(`multiscale-eft-proposal.md`, `phase-d-build-plan.md`, `x1-prediction.md`, `x2-protocol.md`,
`phase1c-pmin-measurement.md`); this repository is the executable part.

License: Apache 2.0 (see `LICENSE`).
