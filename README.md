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
    pytest -q tests/            # 29 tests, ~4 min on a CPU

For GPUs install the matching `jax[cuda]` wheel first.

## The rungs

| rung    | bands                | sources                | band fields | stands for |
|---------|----------------------|------------------------|-------------|------------|
| `M_inf` | none                 | –                      | –           | the incumbent (cutoff model); interaction energy exactly 0 beyond `r_cut` |
| `M_A`   | analytic pair kernel | scalar `s_i` learned   | –           | learned kernel analytic in `q²` (branch 3 of Theorem 3, `x1-prediction.md`) |
| `M_1`   | Coulomb              | `q_i` learned (+`μ_i` optional) | –  | LES / 4G-type: Coulomb added, no dispersion band |
| `M_6`   | dispersion           | `α_i(iω)` learned      | –           | EFT leading order for neutral fragments (L6a) |
| `M_68`  | dispersion, C₆ + C₈  | `α_i`, `α₂,i` learned  | –           | next order: the `p = 8` band on the quadrupole polarisability (L6b) |
| `M_16`  | Coulomb + dispersion | `q_i`, `α_i` learned   | –           | both bands, sources free |
| `M_16p` | Coulomb + dispersion | pinned                 | –           | same architecture, sources from an independent calculation (the dagger models) |
| `M_S0`  | as `M_16p`           | pinned                 | order 0     | v1.5: `E_1`, `|E_1|²`-type invariants at the atom |
| `M_S2`  | as `M_16p`           | pinned                 | order 2     | v1.5: field, gradient and Hessian tensors at the atom, degree 2 |
| `M_G`   | global pair block    | –                      | –           | the null: `Σ_{i≠j} MLP(s_i, s_j, r_ij)` with no cutoff and no envelope (an MLP in `r`) |

Every energy the model returns is an interaction (atomisation) energy: the model's own isolated-atom
energies are subtracted inside `energy_single`, so non-interacting fragments give exactly zero and the
asymptotic measurements are never contaminated by a constant.  Structures that carry fragments (the
molecules of a cluster, `Structure.frag`) subtract instead the model's own energy of each fragment alone at
its own geometry, `E_int = E(all) − Σ_f E(f)` — the atomic case is one atom per fragment — so that a
learned monomer energy never leaves a constant at infinity either.  The E3 script adds learned-source
variants of the band-field rungs (`M_S0mu`, `M_S2mu`: learned `q_i`, `μ_i` and the read-out) and the
*embedded daggers* (`M_1pemb`, `M_16pemb`): charges, atomic dipoles and `α(iω)` pinned to the monomer's values
*in its cluster*, plus a pinned per-atom self energy (`Structure.pinned["eself"]`, the polarisation work
`½|μ_ind|²/α`), without which a Coulomb band double counts the induction of pinned induced dipoles (the
identity band + self = E_es + E_ind is a test).

Pieces (one module each):

* `e0.py` — PaiNN-style message passing with scalar, vector and traceless rank-2 tensor channels; Bessel
  radial basis with a cosine cutoff; messages are exactly zero beyond the cutoff (no bias in the radial
  weights).
* `heads.py` — site energies (data-derived output scale), charges with an exact neutrality shift, gated
  dipoles, dipole and quadrupole dynamic polarisabilities at K = 8 Gauss–Legendre imaginary frequencies
  (softplus or one-oscillator), the scalar source of `M_A`.  Each source can be *pinned* to values carried
  by the structure.
* `kernels.py` — the split kernels: `erf(r/l)/r` and its derivatives (p = 1, dipoles), the `p = 6` and
  `p = 8` long parts `[1 − e^{−x}P(x)]/r^p` with their small-`x` series, the Gaussian Laplacian family
  `(−l²∇²)ⁿ e^{−r²/l²}` (exact rational recursion; used column-normalised, since the raw family's
  window Gram matrix has condition number ~10¹⁰) for `M_A`, the Casimir–Polder quadrature.
* `bands.py` — the band energies for open systems (direct pair sums).
* `ewald.py` — the same bands under periodic boundary conditions: the long parts of the p = 1 (charges and
  dipoles), 6 and 8 kernels summed over all images in reciprocal space with the band edge as the Ewald width
  (no real-space sum: the short remainders are E₀'s); tin-foil boundary conditions.  Checked against the
  Madelung constant of NaCl (1e-9), the simple-cubic lattice sums of r⁻⁶ and r⁻⁸ (1e-8), the point-charge
  limit of the dipole terms, and finite-difference forces.  Switched on per rung with `periodic=True`.  The
  band-field tensors of the v1.5 rungs come from the same Ewald potential under periodic boundary conditions
  (own charge removed analytically; checked against explicit image sums, with the depolarisation field of a
  spherical sum accounted for).  `M_A` under periodic boundary conditions is the polynomial in `k²` times the
  Gaussian envelope in reciprocal space (checked against explicit images to 1e-10).  Every rung now runs on
  open clusters and on periodic cells.
* `bandfields.py` — band-field tensors at the atom by nested forward-mode differentiation of the band
  potential (own charge excluded; optionally the near sources excluded by a smooth switch
  `1 − exp(−(r/r_s)¹⁰)`, `Rung.bf_rs`, because `|E_near + E_far|²` contains a term linear in the far field
  that is degenerate with a dipole source), and two read-outs of degree 2: the *pair* read-out (default), a
  Taylor class on the structure — `ε_i = Σ_j Σ_m w_m(r_ij; s_i, s_j) I_m(n̂_ij; E_i, ℓ∇E_i, ℓ²∇∇E_i)` over the
  14 invariants through order 2, higher orders zero-initialised — and the node read-out (`L ≤ 2`, which cannot
  carry the `L = 4` environment tensors of the order-2 operators; kept as `Rung.readout = "node"`).
* `model.py`, `train.py`, `measure.py`, `data.py`, `toys.py` — assembly, training (Adam + optional
  L-BFGS polish on an exact, memory-bounded full-data loss — a rematerialised `lax.map` over padded
  chunks — targets scaled never shifted), the Phase 1c / X2 measurements, the dimer truths
  (Tang–Toennies, and the published ab initio Ar₂ / Ne₂ potentials in the Rostock form with their
  provenance check), the windows of `phase2-reference-data.md`, the sealed E2 targets, and the X2-B
  dielectric toy (with its exact Taylor-class ceiling).
* `water.py` — the synthetic polarisable water truth of the E3 rehearsal: rigid gas-phase monomers with
  point charges (μ = 1.855 D), intermolecular Coulomb smeared at 0.5 Å, a self-consistent isotropic
  polarisable site on each O (α = 9.72 a.u.), exponential repulsion and damped C₆; dimer scans, relaxed
  random clusters (rigid-body Metropolis on the truth), probe configurations for the far-field induction
  test, and an extxyz loader for the real data.

A note on optimisation: the analytic rung `M_A` and the band-field read-out are ill-conditioned linear
problems inside a non-linear model; Adam alone leaves them far from the floor (`M_A` window rmse 3e-5 E_h
after 1500 steps), the L-BFGS polish (`--lbfgs 300`) brings them within reach (3e-8 E_h).  Always polish.

## Scripts

    python scripts/dimer_ladder.py --system Ar2 --truth tt --rungs M_inf,M_6,M_A,M_G --seeds 4 --steps 1500 --lbfgs 300
    python scripts/dimer_ladder.py --system Ne2,Ar2 --truth published --rungs M_6,M_16 --seeds 8 --lbfgs 300
    python scripts/x2b_toy_regression.py --steps 1500 --lbfgs 200
    python scripts/e3_water.py --part dimer --rungs M_inf,M_G,M_A,M_1,M_1mu,M_1p --seeds 4 --steps 1500 --lbfgs 300
    python scripts/e3_water.py --part clusters --rungs M_1p,M_S0p,M_S2p --seeds 2 --steps 1000 --lbfgs 150
    python scripts/e3_water.py --part far --train_far 1 --train_far_cores 8 --rungs M_1p,M_1mu,M_S0mu --seeds 2 --steps 1000 --lbfgs 150
    sh scripts/run_e1_k16.sh      # the K = 16 batch of E1 / E2 (GPU-sized)

`dimer_ladder.py` trains an ensemble of each rung on the Phase 2 window (40 log-uniform nodes on
`[R₋, 1.8 R₋]`, label noise `σ = 1e-9 E_h`) and prints the Phase 1c table on the evaluation grid out to
500 Å: ensemble mean, relative error, spread `W`, `W/|f*|`, local exponent `q`, and the plateau slope of
`log(W/|f*|)` on [30, 300] Å (branch 2 predicts 0).  With `--truth published` the truths are the ab initio
potentials; with several systems one model is trained on the union, and for the dispersion rungs the
learned free-atom `α(iω)` and the Casimir–Polder `C₆` of every pair of species — including the mixed pair
the model never saw — are compared with the sealed targets (E2).  `x2b_toy_regression.py` trains the
band-field path (orders 0 and 2) on the X2-B toy of `x2-protocol.md` §3 (truth T3, `a/l₁ = 1/4`).
`e3_water.py` is the E3 protocol — the fixed-orientation dimer ladder (p⋆ = 3, learned monomer dipole),
the clusters (n = 3–5 train, hexamers held out, non-additive energy against the model's own dimers) and
the far-field induction test (a probe monomer at 6–10 Å from a relaxed trimer; the reading is the error as
a share of the induction) — run here on the synthetic truth and, with `sseft.water.read_extxyz`, on the
CCSD(T) data when they exist.  Every part saves its json after each rung.

## What v0 reproduces

* Forces of every rung agree with four-point finite differences to the float64 floor (`tests/`), the
  Coulomb band with learned/pinned charges and dipoles reduces to the closed-form two-body energy and to the
  classical multipole limit, the dispersion band to `−C₆ f₆(R)` and to London's formula, and energies are
  invariant (forces covariant) under rigid motions for the vector- and tensor-channel rungs.
* Ar₂, Tang–Toennies truth, window [5.15, 9.26] Å, `r_cut = 6` Å, 4 seeds (`scripts/dimer_ladder.py`):
  `M_inf` has relative error exactly 1 beyond the cutoff (branch 3); `M_6` has local exponent `q = 6.00`
  at every evaluation point with a flat relative spread `W/|f*| ≈ 1.2e-3` (the branch-2 plateau) and a
  16–25 % relative error from the `C₈, C₁₀` truncation (learned `C₆ = 80.4` vs 64.3 `E_h a₀⁶`, the L6a
  bias of the proof document); `M_A(ℓ = 4 Å, N = 4)` shows the three regimes of `x1-prediction.md` — a
  burst beyond `R₊` with negative `q`, `q` rising through 6 near `ℓ√(N+3)` and then growing linearly in
  `R′²` with the slope `2/ℓ²` (measured 0.132 Å⁻² against 0.125), the mean's relative error settling at 1
  and the spread collapsing to the float64 floor by 30 Å.  A `C₈` channel (quadrupole polarisability) is
  the L6b item of the build plan.
* E1 / E2 on the published Ar₂ and Ne₂ potentials (every branch assignment as pre-registered; the unseen
  Ne–Ar C₆ from the learned α(iω); the C₆ bias computed under the training objective), the L6b rung M_68,
  and the E3 rehearsal on the synthetic water truth — the dimer ladder with p⋆ = 3 and the learned monomer
  dipole (1.861 ± 0.003 D against 1.855 once the identifying nodes are defined by the smallest intermolecular
  pair), the clusters, and the far-field induction test with eight pre-registered runs, ending with the
  embedded dagger `M_16pemb` reproducing the dimer tail to 6e-8 at 20 Å, the far field to the mutual response
  (5 % at 10 Å) and, at a band edge of 0.75 Å, the hexamers' non-additive energy to 4–9 %: `RESULTS.md`, with
  the pre-registration and result notes in `notes/`.
* The X2-B toy and the numbers behind the lines above: `RESULTS.md`.

## Layout

    sseft/        the package
    scripts/      the experiments (each prints its table and writes a json)
    tests/        pytest (fast; the ladder and the toy are scripts)
    notes/        the pre-registration and result notes of the experiments run from this repository
    scratch/      one-off computations quoted in the notes (not part of the package)
    results/      the logs and jsons of every run quoted in RESULTS.md
    RESULTS.md    running log of the regression numbers per commit

The design, the predictions and the measurement protocols live in the project notes
(`multiscale-eft-proposal.md`, `phase-d-build-plan.md`, `x1-prediction.md`, `x2-protocol.md`,
`phase1c-pmin-measurement.md`); this repository is the executable part.

License: Apache 2.0 (see `LICENSE`).
