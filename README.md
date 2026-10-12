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

    pip install -e .            # jax, optax, numpy, scipy, PyYAML (tested with jax 0.10, optax 0.2.8, numpy 2.4, scipy 1.17)
    pytest -q tests/            # 235 tests, ~6 min on a CPU (the 34 physics tests take most of it)

For GPUs install the matching `jax[cuda]` wheel first.

## Experiment selection with YAML

Select an experiment using `configs/dimer.yaml`, `configs/water.yaml`, or `configs/toy.yaml`:

```sh
python scripts/run_experiment.py --config configs/dimer.yaml --lbfgs 300
python scripts/run_experiment.py --config configs/water.yaml --cache results/cache_clusters_v2.pkl --out results/e3r_clusters_yaml.json
python scripts/run_experiment.py --config configs/toy.yaml
```

The config contains an `experiment` mapping. `type` is required:

| Type | Optional selection fields |
| --- | --- |
| `dimer_ladder` | `systems` (list), `truth` (`tt` or `published`), `rungs` (list) |
| `water` | `part` (`dimer`, `clusters`, `far`, or `ion`), `rungs` (list) |
| `x2b_toy` | `orders` (list containing `0` and/or `2`) |

Omitted fields retain the selected script's defaults. Explicit CLI flags override YAML values. Unknown fields and unsupported selections are rejected. YAML also supports the water model (architecture) settings, the water cluster datasets and the training settings below; the paths of a run (`--cache`, `--out`) stay on the command line. The examples run real experiments and may take substantial time.

Each original script also accepts `--config`, provided its experiment type matches. Existing commands without a config continue to work.

`configs/water.yaml` is the current protocol: the E3-R cluster run (R1 of `notes/e3r-bandfield-response-preregistration.md`, final schedule of `notes/e3r-results.md` §1) — the permanent-source daggers `M_16p`, `M_16S0p`, `M_16S2p` at the band edge ℓ₁ = 0.75 Å without the radial switch, the pre-registered cluster data, K = 2, Adam 1000 with the read-out frozen, its order-0 linear solve, L-BFGS 300, batch 12, chunk 6 — i.e. `scripts/e3_water.py --part clusters --rungs M_16p,M_16S0p,M_16S2p --seeds 2 --steps 1000 --lbfgs 300 --batch 12 --lbfgs_chunk 6 --l1 0.75 --rs_over_l1 0`.

### Water model settings

The optional `model` mapping (water only) sets the architecture of every selected rung; each field maps to an existing flag of `e3_water.py`, and explicit flags override YAML:

```yaml
model:
  charges: null                    # pinned | learned | null (keep each rung's own source);  --charges
  charge_neutrality: fragment      # fragment | structure;                                  --charge_neutrality
  band_edge_angstrom: 0.75         # l_1; null = the rung's own band edge (1.5 A);            --l1
  rs_over_l1: 0.0                  # near-source switch radius of the band-field input / l_1; --rs_over_l1
  r_cut_angstrom: 6.0              # E_0's cutoff;                                            --rcut
  features: 16                     # E_0's feature width F;                                   --F
  analytic_envelope_angstrom: 4.0  # M_A's envelope;                                          --lA
```

`charges: pinned` uses the synthetic water's fixed atomic charges; `learned` uses the model's learned charge head even for presets whose names end in `p`; omission or `null` preserves each preset's own charge source, allowing a mixed learned/pinned comparison (the daggers of `configs/water.yaml` pin theirs). The setting applies to both the Coulomb band and the band-field inputs in every water part; charge-free models such as `M_inf` stay charge-free, and dipole and polarisability pinning keep their preset settings.

`charge_neutrality: fragment` (the default since v0.0.8) takes the neutrality shift of the *learned* charges fragment by fragment: every molecule is neutral and an ion carries its own charge (`Structure.frag_charges`) whatever its environment, so the charges of a fragment beyond r_c of everything else are exactly the fragment's own, the fragment reference `E_int = E(all) − Σ_f E(f)` cancels the intramolecular band energy exactly, and no molecule acquires a monopole from another molecule's environment-dependent `dq`. `structure` is one shift over the whole structure — the head before v0.0.8, which spread an ion's +1 over every atom (`notes/e3r-results.md` §4(7)) — kept for controlled comparisons only. Structures without fragments (the atomic dimers) always use the one shift to their total charge.

`band_edge_angstrom` is ℓ₁ of every band (`--l1 0` on the command line means the rung's own 1.5 Å), `rs_over_l1` the radius of the near-source switch of the band-field input as a multiple of the band edge actually used (0 disables the switch; the own fragment is excluded from the input regardless, `Rung.bf_fragments`), `r_cut_angstrom` and `features` E_0's cutoff and width, `analytic_envelope_angstrom` the envelope of `M_A`.

Rung names remain the result labels; each rung's result records the architecture as run under `model` — the sources (`charges`, `dipoles`, `polarisabilities`: pinned / learned / off), `charge_neutrality`, `band_fields`, `band_edge_angstrom`, `near_source_switch_angstrom`, `r_cut_angstrom`, `features`, `analytic_envelope_angstrom`, `periodic` — so that two runs differing only in a flag are told apart in the json. Response-readout freezing is a separate training setting and does not select the charge source. Select response rungs such as `M_S0`/`M_S2`, `M_S0p`/`M_S2p` or `M_16S0p`/`M_16S2p` when you want band-field read-outs.

### Water cluster datasets

`configs/water.yaml` selects `part: clusters` and makes the dataset defaults (the pre-registered design) explicit:

```yaml
experiment:
  type: water
  part: clusters
  rungs: [M_16p, M_16S0p, M_16S2p]

data:
  train_sizes: [3, 4, 5]
  test_sizes: [6, 2]
  n_train: 12
  n_test: 8
  train_seed: 0
  test_seed: 100
  sigma: 1.0e-9
  monomer_references: 4
  mc_steps: 300
  temperature_kelvin: 300.0
  radius_angstrom: null
  min_oo_angstrom: 2.7
  min_atom_distance_angstrom: 1.7
```

Sizes are numbers of water molecules, and sampling counts are **per size**. These defaults produce 36 training clusters plus four monomer references, and eight test clusters for each of sizes 6 and 2. Sizes must be distinct integers of at least two, and both lists must be nonempty. Change either list to evaluate other cluster sizes; evaluation computes pair-subtracted non-additive energy using the actual size.

`train_seed` controls dataset generation independently of model initialization. Test sizes use seeds `test_seed + 100 * index` in the configured list order (defaults: size 6 uses 100; size 2 uses 200). `sigma` adds training energy noise in hartree. Test labels remain noiseless. `monomer_references` controls the number of isolated monomer references added to training.

`mc_steps` sets the number of rigid-body Metropolis relaxation attempts at `temperature_kelvin`. `radius_angstrom: null` chooses a generation radius of `1.9 * size**(1/3) + 0.8` Å for each size. `min_oo_angstrom` sets the initial oxygen separation, and `min_atom_distance_angstrom` sets the intermolecular atom distance floor during generation and relaxation. Use a radius large enough to place the requested molecules. Impossible diameter/separation combinations are rejected, and placement retries are bounded with an informative error.

CLI overrides remain available, for example `--n_train 20 --test_sizes 4,6 --mc_steps 500`. The `data` mapping currently applies only to `water` with `part: clusters`; omit it for other parts. Existing selection-only configs keep the previous defaults.

Cluster results record resolved dataset settings and per-size energy and non-additive RMSE under `clusters_by_size` (hartree). The original `hexamer`, `dimer`, `nonadditive`, and `nadd6_truth` fields remain when the corresponding sizes are evaluated. Use a separate output file for each experiment. Cluster runs reject an existing output with different or unknown dataset metadata, preventing results from different datasets from being merged.

Cluster caches record generation settings and reject a mismatch. Older cluster caches lack this metadata; use a new cache path to regenerate them.

### Training configuration

All experiment types accept an optional `training` mapping. `configs/water.yaml` lists every supported field (with the values of the E3-R final schedule):

```yaml
training:
  seeds: [0, 1]
  adam:
    steps: 1000
    learning_rate: 0.003
    final_learning_rate_ratio: 0.01
    beta1: 0.9
    beta2: 0.999
    epsilon: 1.0e-8
    epsilon_root: 0.0
    gradient_clip_norm: null
    batch_size: 12
  readout:
    freeze_during_adam: true
    linear_solve: true
    ridge: 1.0e-8
  lbfgs:
    steps: 300
    chunk_size: 6
    learning_rate: null
    memory_size: 10
    scale_init_precond: true
    max_linesearch_steps: 20
    readout_only: false
  loss:
    energy_weight: 1.0
    force_weight: 1.0
    far_weight: 1.0
  energy_scale: null
  verbose: false
```

`adam.steps` controls the initial training stage. With `readout.freeze_during_adam: true`, the entire response readout is frozen and its energy contribution remains zero; the GNN and active energy/source heads train. Thus this value controls the number of response-frozen training steps. Adam uses cosine learning-rate decay ending at `learning_rate * final_learning_rate_ratio`. `beta1`, `beta2`, `epsilon`, and `epsilon_root` configure Adam's moment estimates and numerical stabilizers. `gradient_clip_norm` optionally limits the norm of the trainable gradient; `null` disables clipping. `batch_size: null` uses all training structures each step.

After Adam, `readout.linear_solve: true` fits the order-0 response output coefficients by weighted least squares while holding the rest fixed. `ridge` sets regularization relative to the design matrix scale; it is not a general weight-decay penalty. Higher-order response coefficients are optimized in L-BFGS. Readout settings apply only when the selected rung has a response readout; for water, select rungs such as `M_16S0p` and `M_16S2p` (or `M_S0p`, `M_S2p`) to use them. The baseline models `M_inf`, `M_1`, `M_1p` and `M_16p` have no response readout and ignore them.

`lbfgs.steps` controls subsequent full-data optimization. With the default `readout_only: false`, the GNN, active heads, and response train jointly. Set `readout_only: true` to hold all non-readout parameters fixed in this stage; every selected model must have a response readout. `chunk_size` changes memory usage while preserving the full-data loss. `memory_size` controls the retained curvature history. `learning_rate: null` preserves automatic step scaling with the zoom line search; `max_linesearch_steps` bounds its internal trials, and `scale_init_precond` controls initial inverse-Hessian scaling. These are optimizer iterations, not epochs or line-search trial counts. Zero Adam steps allow an L-BFGS-only run; at least one optimizer step must be requested. Models with a frozen response readout require positive L-BFGS steps.

`loss.energy_weight` and `loss.force_weight` combine scaled energy and force residuals. Set `force_weight: 0.0` for energy-only training. They also determine the readout linear solve's objective. `far_weight` sets per-structure weights for water's `far` and `ion` parts. `energy_scale: null` derives the normalization from training energies; positive values supply a manual scale in hartree. Targets are never mean-centered. `verbose` controls progress output. Dataset caches remain valid when only training options change.

`seeds` explicitly selects model initialization/minibatch seeds, independently of dataset seeds. Each selected rung is trained with each seed. The toy supports multiple model seeds too and stores results under `runs`, keeping its original scalar metrics for a single-seed run. Results record the requested configuration under `training` and the effective per-seed options under `training_runs` (water/dimers) or each toy run's `training_options`, including the actual normalization scale, effective batch size, and whether a response solve ran.

Explicit CLI flags override YAML. Existing `--steps`, `--lbfgs`, `--batch`, and `--lbfgs_chunk` work; additional flags are listed by `--help`. For example:

```sh
python scripts/run_experiment.py --config configs/water.yaml --steps 100 --lbfgs 30 --learning_rate 0.001
```

For water/dimers, explicit `--seeds 2` overrides a YAML seed list with `[0, 1]`; for the toy, `--seed 7` overrides it with `[7]`. `--model_seeds 7,11` supplies an explicit list. Combining that list flag with a count/single-seed flag is rejected as ambiguous. Boolean overrides use positive/negative flags such as `--freeze_readout` / `--no-freeze_readout` and `--lbfgs_readout_only` / `--no-lbfgs_readout_only`.

Omitted training fields retain each workflow's previous defaults: water/dimer Adam learning rate 0.003 and force weight 1; water cluster/far/ion batch 12 and chunk 6; water dimer/noble-gas dimer full-batch Adam and chunk 48; toy batch 48, force weight 0, readout trained by Adam, no linear solve, and verbose output. Default step counts remain those of the original scripts (`e3_water.py` without a config: 4 seeds, Adam 1500, L-BFGS 300 — the dimer design; the cluster, far-field and ion runs of the notes used 2 seeds, Adam 1000 and L-BFGS 150 or 300, which `configs/water.yaml` encodes).

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
* `heads.py` — site energies (data-derived output scale), charges with an exact neutrality shift taken per
  fragment (every molecule neutral, an ion at its own charge, `Structure.frag_charges`; one shift over the
  structure where no fragments are given), gated dipoles, dipole and quadrupole dynamic polarisabilities at
  K = 8 Gauss–Legendre imaginary frequencies (softplus or one-oscillator), the scalar source of `M_A`.  Each
  source can be *pinned* to values carried by the structure.
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
  polarisable site on each O (α = 9.72 a.u.), exponential repulsion and damped C₆, and a sodium-like
  non-polarisable ion (`ION`) for the band-field response test; dimer scans, relaxed random clusters
  (rigid-body Metropolis on the truth), probe and ion configurations for the far-field tests, the exact
  decomposition of an ion's induction, and an extxyz loader for the real data.
* `linear.py` — the weighted linear least-squares solve of the band-field read-out's order-0 output layers
  under the training objective (energies, forces, per-structure weights), the rest of the model fixed: the
  read-out is a linear-in-coefficients term inside a non-linear model, and like `M_A`'s analytic family it is
  initialised by its solve and polished by L-BFGS, never trained by Adam (`train.fit` freezes it during the
  Adam stage; the five rules are in `notes/e3r-results.md` §5).

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
    python scripts/e3_water.py --part ion --rungs M_16p,M_16S0p --l1 0.75 --rs_over_l1 0 --train_far 1 --train_far_cores 8 \
        --test_cores 4 --extra_far 2 --seeds 2 --steps 1000 --lbfgs 300      # E3-R: the band-field response on an ion probe
    python scripts/run_experiment.py --config configs/water.yaml --cache results/cache_clusters_v2.pkl \
        --out results/e3r_clusters_yaml.json                                  # E3-R R1, the cluster protocol from YAML
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
* E3-R, the band-field response tested explicitly (`notes/e3r-*.md`): with permanent sources, the own fragment
  excluded from the band-field input and the band edge at 0.75 Å, the order-0 read-out carries the first-order
  induction — the hexamers' non-additive error from 90–106 % to 15–51 %, an ion's induction at 6–12 Å from
  87–95 % to 32–34 %, and the response coefficient of an isolated monomer read at 1.05 and 1.00 of α — once
  the training set holds configurations with the source beyond r_c of every atom and the read-out is trained
  by its linear solve rather than by Adam.
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
