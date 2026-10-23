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
    sh external/qaqua/build.sh  # only for E3-Q: the q-AQUA truth, needs gfortran (or FC=ifort); ~10 min
    sh external/qaqua/build_pol.sh  # only for E3-QP: the q-AQUA-pol truth, likewise
    pytest -q tests/            # 306 tests, ~6 min on a CPU; the 18 q-AQUA and q-AQUA-pol tests are skipped without the builds

For GPUs install the matching `jax[cuda]` wheel first.

## Experiment selection with YAML

Select an experiment using `configs/dimer.yaml`, `configs/water.yaml`, `configs/water_qaqua.yaml`, `configs/water_qaqua_pol.yaml`, or `configs/toy.yaml`:

```sh
python scripts/run_experiment.py --config configs/dimer.yaml --lbfgs 300
python scripts/run_experiment.py --config configs/water.yaml --cache results/cache_clusters_v2.pkl --out results/e3r_clusters_yaml.json
python scripts/run_experiment.py --config configs/water_qaqua.yaml --out_dir results/e3q      # both parts of E3-Q
python scripts/run_experiment.py --config configs/water_qaqua_pol.yaml --out_dir results/e3qp # both parts of E3-QP
python scripts/run_experiment.py --config configs/toy.yaml
```

The config contains an `experiment` mapping. `type` is required:

| Type | Optional selection fields |
| --- | --- |
| `dimer_ladder` | `systems` (list), `truth` (`tt` or `published`), `rungs` (list) |
| `water` | `part` (`dimer`, `clusters`, `far`, or `ion`, or a list of parts), `rungs` (list) |
| `x2b_toy` | `orders` (list containing `0` and/or `2`) |

Omitted fields retain the selected script's defaults. Explicit CLI flags override YAML values. Unknown fields and unsupported selections are rejected. YAML also supports the water truth, the water model (architecture) settings, the water dimer design and cluster datasets, the training settings below, and several water parts in one file; the paths of a run (`--cache`, `--out`, or `--out_dir` for several parts) stay on the command line. The examples run real experiments and may take substantial time.

Each original script also accepts `--config`, provided its experiment type matches, and `--validate_only`, which checks the configuration and the resolved arguments and exits without running. Existing commands without a config continue to work.

### Several parts in one file; the water truth

`experiment.part` may list several water parts. Shared `model`, `training` and `truth` mappings apply to all of them, and a `parts` mapping gives each part its own `rungs`, `model`, `data` and `training`, merged over the shared ones key by key (`data` exists per part only: the dimer design or the cluster data sets). `scripts/run_experiment.py` first checks every part (`--validate_only`), then runs them in sequence, each writing `<out_dir>/<part>.json` and `<part>_cache.pkl` (default `--out_dir results/<config name>`); a failed part is reported and the next one still runs. `--part dimer` runs one part alone, and explicit flags apply to every part they are given to.

The `truth` mapping (water only) selects the truth and its settings — `name: pw` (the synthetic polarisable water of the rehearsal, the default), `name: qaqua` or `name: qaqua_pol` with `root` (the build directory of `external/qaqua/build.sh` or `build_pol.sh`; omitted: `external/qaqua/build` or `external/qaqua/build_pol`), and `heldout`: the raw CCSD(T) n-body sets read in the clusters part (`sets`, `rigid_tolerance`, `strict_tolerance`, `max_pair_energy_kcal`, `max_frames`; the flags `--truth`, `--qaqua_root`, `--heldout*`). The dimer part's `data` mapping is its design: `re_angstrom` (null: the truth's own dimer minimum at φ = 0), `window_lower_over_re`, `window_ratio`, `window_nodes`, `phi_degrees`, `sigma`, `train_seed` (of the label noise), `monomer_references`, `eval_max_angstrom`, `eval_points` (the same names as flags).

### E3-Q: water on q-AQUA v1

`configs/water_qaqua.yaml` is E3 on the published CCSD(T) potential q-AQUA v1 (Q. Yu et al., J. Phys. Chem. Lett. 13, 5068 (2022); github.com/jmbowma/q-AQUA, MIT licence), pre-registered in `notes/e3q-preregistration.md` with the numbers computed from the PES before training (`scratch/e3q_preregistration.py`). Build the truth once, where there is internet access:

```sh
sh external/qaqua/build.sh            # clones q-AQUA, compiles its Fortran with a C interface into external/qaqua/build/libqaqua.so,
                                      # checks the authors' trimer and hexamer benchmarks; FC=ifort for the Intel compiler
python scripts/run_experiment.py --config configs/water_qaqua.yaml --out_dir results/e3q
```

The build copies the compiler's Fortran runtime (libgfortran, and libquadmath where that compiler uses it) to `external/qaqua/build/runtime`, which `libqaqua.so` finds through its RPATH, so the library loads in a batch job or another conda env without the compiler's module or env. A build made before this (`OSError: libgfortran.so.5: cannot open shared object file`) is repaired in seconds by `sh external/qaqua/build.sh --relink`, run in a shell where the compiler it was built with is available.

The truth is q-AQUA on rigid monomers at the harness geometry, minus its 1-body term (`sseft.qaqua`, `sseft.water.QAquaTruth`): its 2-body term is the bare point-charge Coulomb of the monomers' LTP2011 charges beyond 7.8 Å, its 3- and 4-body terms vanish beyond 9 and 7.5 Å, so the pinned-charge dagger `M_1p` is exact on its dimer tail, and the rungs pinning sources of the PW truth's polarisable model (`M_16*`, `*emb`) and the far and ion parts are refused. Two parts: (a) the dimer ladder at φ = 0 (`M_inf`, `M_G`, `M_A`, `M_1`, `M_1mu`, `M_1p`; r_c = 5 Å, K = 8) with the learned monomer dipole against the LTP2011 dipole; (c) the clusters (`M_inf`, `M_1p`, `M_S0p`, `M_S2p`, `M_1mu`; ℓ₁ = 0.75 Å, K = 4, the E3-R schedule), every trained model also read on the raw CCSD(T) 2-, 3- and 4-body sets of the q-AQUA repository (frames with near-rigid monomers inside the training domain), with the PES's own error on the same frames as the reference and the 3-body set by largest O–O distance — beyond 7–9 Å the PES's 3-body term is switched off, the CCSD(T) labels are not. Every trained model's parameters are saved next to the json (`<out>_<rung>_seed<k>.params.pkl`).

### E3-QP: water on q-AQUA-pol, hundreds of training clusters

`configs/water_qaqua_pol.yaml` is the controlled follow-up of E3-Q (`notes/e3q-results.md`): the same rungs and held-out CCSD(T) sets on q-AQUA-pol (Q. Yu et al., J. Chem. Theory Comput. 19, 3446 (2023); github.com/jmbowma/q-AQUA-pol, CC0), whose long range is TTM3-F's — geometry-dependent site charges, one Thole-smeared polarisable site per molecule with self-consistent induced dipoles and no cutoff, an O–O C₆ term — under CCSD(T) corrections that vanish beyond 7 Å; pre-registered in `notes/e3qp-preregistration.md` (`scratch/e3qp_preregistration.py`).

```sh
sh external/qaqua/build_pol.sh        # clones q-AQUA-pol (and q-AQUA for its raw CCSD(T) sets), builds external/qaqua/build_pol/libqaqua_pol.so,
                                      # checks the authors' hexamer and TTM3-F's pieces at 12 A; --relink as for build.sh
python scripts/run_experiment.py --config configs/water_qaqua_pol.yaml --out_dir results/e3qp
```

The pinned charges are the monomers' dipole-surface charges on O, H, H (TTM3-F's sites have their dipole, not their quadrupole: the daggers' tail errors are computed in the note), the `M_16` daggers also pin TTM3-F's polarisability and O–O C₆ in the dispersion band, and the embedded daggers are refused. (a) The dimer ladder as E3-Q plus `M_16p`, L-BFGS 1000; (c) the clusters with 300 training clusters (100 per size) and 160 probe configurations, 24 test hexamers and dimers, 80 test probes — `M_inf`, `M_1p`, `M_S0p`, `M_S2p`, `M_1mu`, `M_16p`, `M_16S0p` — every model read on the probes (the probe's interaction and its non-additive part, the many-body induction of a distant molecule) and on the raw CCSD(T) sets.

### E3-QP2: the response field and its higher orders on q-AQUA-pol

`configs/water_qaqua_pol2.yaml` follows E3-QP (`notes/e3qp-results.md`) on the same truth, clusters, test probes and
raw sets; pre-registered in `notes/e3qp2-preregistration.md` (`scratch/e3qp2_preregistration.py`,
`scratch/e3qp2_matched_response.py`). Three changes, each one of E3-QP's diagnosed limits:

- **The field**: the `*pm` rungs pin the truth's permanent charges exactly — TTM3-F's H charges on the atoms and its
  M-site as a virtual site placed from the molecule's atoms (`Rung.msites`, `truth.site_model`, `W.pin_site_model`) —
  so the band field the read-out answers is the truth's permanent field and the daggers `M_1pm`, `M_16pm` miss the
  dimer tail by TTM3-F's induction and C₆ term only.
- **The response, matched on the far field**: the read-out's coefficients come from the species alone
  (`Rung.bf_species`: fixed, seed-independent features instead of E₀'s), and `training.readout.stage: far_first` solves
  them before E₀ is trained, on the training probes E₀ cannot see (no probe atom within r_c of the core), labelled
  with the probe's interaction energy (`training.loss.far_labels: interaction`) and fitted on energies only, the ridge
  chosen on held-out cores; the read-out is then held through Adam and the polish (`training.lbfgs.freeze_readout`).
  Beyond r_c the trained models' probe readings are therefore computed before the run.
- **Second order and beyond**: the C rungs let the order-0 read-out's induced dipoles act on each other through the
  band kernel's dipole tensor between molecules, self-consistently (`Rung.bf_mutual = -1`; the series through order
  k with `bf_mutual = k`), against the first-order S rungs; the order-2 rungs add the read-out's field-gradient
  terms, now solved with the rest (`training.readout.orders: all`).

```sh
python scripts/run_experiment.py --config configs/water_qaqua_pol2.yaml --out_dir results/e3qp2
```

Rungs: (a) the dimer, `M_1pm`, `M_16pm`; (c) the clusters, `M_inf`, `M_16p`, `M_16pm`, `M_16C0p` (the self-consistent
response on the dipole-surface charges: the control for the field), `M_16S0pm`, `M_16C0pm`, `M_16S2pm`, `M_16C2pm`, with
384 training probes on 48 cores of 3, 4 and 5 molecules (`data.probe_train_core_sizes`) and E3-QP's test probes.
Every reading applies each rung's pins (v0.0.10 read the M_16 rungs' raw 2-body frames without their dispersion band).

### E3-QP3: the short range on q-AQUA-pol

`configs/water_qaqua_pol3.yaml` follows E3-QP2 (`notes/e3qp2-results.md`) on the same truth, training clusters,
probes, test sets and far-field matching; pre-registered in `notes/e3qp3-preregistration.md`
(`scratch/e3qp3_preregistration.py`). E3-QP2's short-range diagnostic (`scratch/e3qp3_shortrange_diagnostic.py`,
§3.5 of its results) found the pinned rungs better than M_∞ on rigid, covered dimers and worse on the compressed raw
dimers (O–O < 2.7 Å) the relaxed clusters hardly visit, E₀ fitting its own training energies no better than the test
ones, and a double count of the truth's charge flux in the raw sets' readings. The changes:

- **Data at the contact**: 600 rigid dimers and 200 rigid trimers at O–O 2.3–3.0 Å, every pair under the raw sets'
  2 kcal/mol cap (`data.contact_dimers`, `contact_trimers`, `contact_oo_angstrom`, `contact_max_pair_kcal`,
  `contact_min_atom_distance_angstrom`, `contact_seed`; `W.contact_configurations`): the raw compressed dimers beyond
  the training pairs' reach fall from 57 to 11 % (O–O 2.3–2.7 Å).
- **E₀ fitted to the energies**: `training.loss.energy_weight: 100` (the per-atom energy term was two orders of
  magnitude below the force term), Adam 4000, L-BFGS 1000.
- **The pins as trained**: `truth.heldout.pin_charges: rigid` reads the raw near-rigid frames with the rigid monomer's
  charges, as the models are trained (`frame`, the default, keeps the charges at each frame's own geometry, as up to
  v0.0.11); every held-out reading is also binned by the frame's closest contact (`by_smallest_oo_angstrom`).
- **The pinned bands' contact as a factor**: `M_16pm_e1`, `M_16C2pm_e1` are `M_16pm`, `M_16C2pm` with their own band
  edge ℓ₁ = 1.0 Å (`e3_water.RUNG_BAND_EDGE`), the read-out's at the run's (`model.response_edge_angstrom: 0.75`), so
  that the matched response and every far-field reading stay E3-QP2's.

```sh
python scripts/run_experiment.py --config configs/water_qaqua_pol3.yaml --out_dir results/e3qp3
python scratch/e3qp3_shortrange_diagnostic.py results/e3qp3 external/qaqua/build_pol --response_edge 0.75 --seeds all
```

### E3-QP4: E₀'s width, the energy term's balance, the read-out's near field

E3-QP3 (`notes/e3qp3-results.md`) fixed the 2-body short range and left the 3-body at contact, the hexamers and E₀'s own
training fit short of their registered values. Three limits, three separable changes, as two jobs that run in
parallel (pre-registered in `notes/e3qp4-preregistration.md`):

- **Job A** (`configs/water_qaqua_pol4a.yaml`): E3-QP3 at E₀'s width 32 (`model.features`; E3-QP3's energy weight
  traded training forces for training energies — E₀ at its width). The species read-out's random features share the
  width, so its matched far field is computed again before training.
- **Job B** (`configs/water_qaqua_pol4b.yaml`): job A with the contact configurations' energies at weight 0.125
  (`training.loss.contact_energy_weight`; their forces keep weight 1): the loss weighs a structure's energy by
  1 / n_atoms², so at weight 1 the 800 contacts outweigh the 300 relaxed clusters eightfold at equal errors.
- **The read-out's near field as a factor in both**: `M_16C2pm_r1` is `M_16C2pm` with the read-out's own band edge
  1.0 Å (`e3_water.RUNG_RESPONSE_EDGE`; the pinned bands at the run's 0.75 Å) — the far field the same to a point, the
  response closer to TTM3-F's induction at contact and on the hexamers (`notes/e3qp3-results.md` §3.4). The `_e1` rungs
  are dropped.

```sh
python scripts/run_experiment.py --config configs/water_qaqua_pol4a.yaml --out_dir results/e3qp4a      # job A
python scripts/run_experiment.py --config configs/water_qaqua_pol4b.yaml --out_dir results/e3qp4b      # job B
python scratch/e3qp3_shortrange_diagnostic.py results/e3qp4a external/qaqua/build_pol --F 32 --response_edge 0.75 --seeds all
```

### E3-QP5: E₀'s view of the pinned field

E3-QP4 (`notes/e3qp4-results.md`) left one reading where M_∞ wins: the pinned rungs' 3-body at contact (1.3–1.5× M_∞'s on
the raw trimers, 1.5× on the near-rigid ones, in the rigid part) — not E₀'s width, nor the energy term's balance, nor
coverage, nor the pinned bands' edge. E3-QP5 gives E₀ the field-like features it otherwise builds itself
(pre-registered in `notes/e3qp5-preregistration.md`):

- **The field term** (`Rung.e0_field`): E₀'s site energies read, at every atom, the potential, field and field gradient
  of the other molecules' pinned charges and M-sites (TTM3-F's site charges through the band kernel at ℓ₁;
  `bandfields.local_field_tensors`), every source's share tapered to zero between r_c − 1.5 and r_c − 0.5 Å
  (`e3_water.E0_FIELD_TAPER`), through an MLP of E₀'s invariants, its vector and tensor channels contracted with the
  field and its gradient, and the two squares, less the same at zero field (`heads.field_energy`; MACE-POLAR's
  non-local read-out, here local). The input never reaches beyond E₀'s own graph, so the far field is the base rung's;
  the term starts at zero, so a field rung starts as its base.
- **Rungs** (`configs/water_qaqua_pol5.yaml`: E3-QP4's job B otherwise): `M_16pm_f` and `M_16C2pm_f` beside `M_16pm` and
  `M_16C2pm` (run again as the controls), and `M_inf_f` — M_∞ with the same input and no bands: whether the field is the
  pinned rungs' missing piece or a better input for any E₀.

```sh
python scripts/run_experiment.py --config configs/water_qaqua_pol5.yaml --out_dir results/e3qp5
python scratch/e3qp3_shortrange_diagnostic.py results/e3qp5 external/qaqua/build_pol --F 32 --response_edge 0.75 --seeds all --save_frames
```

`configs/water.yaml` is the E3-R cluster protocol: the E3-R cluster run (R1 of `notes/e3r-bandfield-response-preregistration.md`, final schedule of `notes/e3r-results.md` §1) — the permanent-source daggers `M_16p`, `M_16S0p`, `M_16S2p` at the band edge ℓ₁ = 0.75 Å without the radial switch, the pre-registered cluster data, K = 2, Adam 1000 with the read-out frozen, its order-0 linear solve, L-BFGS 300, batch 12, chunk 6 — i.e. `scripts/e3_water.py --part clusters --rungs M_16p,M_16S0p,M_16S2p --seeds 2 --steps 1000 --lbfgs 300 --batch 12 --lbfgs_chunk 6 --l1 0.75 --rs_over_l1 0`.

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

Contact configurations (v0.0.12, any truth): `contact_dimers` rigid dimers and `contact_trimers` rigid trimers (a third
molecule at such a distance from one of the first two) at a hydrogen-bond contact or closer — O–O drawn uniformly from
`contact_oo_angstrom` ([lower, upper], default [2.3, 3.0]), every pair's truth 2-body energy at most
`contact_max_pair_kcal` (0: no cap), no two atoms of different molecules nearer than `contact_min_atom_distance_angstrom`
— seeded by `contact_seed` (+1 for the trimers) and added to the training set after the monomer references. Both
counts 0 (the default): none, and the dataset settings of before.

Probe configurations (any truth): `probe_distances_angstrom` (a list), `probe_train` and `probe_test` (per distance and core; 0, the default, means none), `probe_train_cores`, `probe_test_cores` and `probe_seed` (the test probes use +100) add a relaxed trimer with one monomer at each distance — the training probes weighted by `training.loss.far_weight`, the test probes read with every model (the probe's interaction with the trimer and its non-additive part, by distance). Without probes the dataset settings are those of before, so older caches stay valid.

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
    contact_energy_weight: 1.0
  energy_scale: null
  verbose: false
```

`adam.steps` controls the initial training stage. With `readout.freeze_during_adam: true`, the entire response readout is frozen and its energy contribution remains zero; the GNN and active energy/source heads train. Thus this value controls the number of response-frozen training steps. Adam uses cosine learning-rate decay ending at `learning_rate * final_learning_rate_ratio`. `beta1`, `beta2`, `epsilon`, and `epsilon_root` configure Adam's moment estimates and numerical stabilizers. `gradient_clip_norm` optionally limits the norm of the trainable gradient; `null` disables clipping. `batch_size: null` uses all training structures each step.

After Adam, `readout.linear_solve: true` fits the order-0 response output coefficients by weighted least squares while holding the rest fixed. `ridge` sets regularization relative to the design matrix scale; it is not a general weight-decay penalty. Higher-order response coefficients are optimized in L-BFGS. Readout settings apply only when the selected rung has a response readout; for water, select rungs such as `M_16S0p` and `M_16S2p` (or `M_S0p`, `M_S2p`) to use them. The baseline models `M_inf`, `M_1`, `M_1p` and `M_16p` have no response readout and ignore them.

The solve's further fields (v0.0.11; defaults keep the behaviour above): `readout.orders: all` solves every order's
output columns (with a ridge), `readout.solve_iterations` takes Gauss-Newton steps for a read-out that is not linear in
its coefficients (the self-consistent response), `readout.ridge_selection: groups` picks the ridge from 1e-12…1e-1 on
held-out groups (each cluster, and the probes by core; chosen at the first step and kept), `readout.far_weight` weights
the training probes in the solve, and `readout.stage: far_first` matches the read-out before Adam on the training
probes beyond r_c of their cores, energies only (it needs `loss.far_labels: interaction`, `lbfgs.freeze_readout: true`
and species read-outs). `lbfgs.freeze_readout: true` keeps the read-out through the polish. `loss.far_labels:
interaction` labels the clusters part's training probes with the probe's interaction energy and its forces (subtraction
groups core and probe, `Structure.sub`) instead of their totals. `model.response_edge_angstrom` gives the read-out its
own band edge (its band field and dipole tensor smeared there; null: the band edge).

`lbfgs.steps` controls subsequent full-data optimization. With the default `readout_only: false`, the GNN, active heads, and response train jointly. Set `readout_only: true` to hold all non-readout parameters fixed in this stage; every selected model must have a response readout. `chunk_size` changes memory usage while preserving the full-data loss. `memory_size` controls the retained curvature history. `learning_rate: null` preserves automatic step scaling with the zoom line search; `max_linesearch_steps` bounds its internal trials, and `scale_init_precond` controls initial inverse-Hessian scaling. These are optimizer iterations, not epochs or line-search trial counts. Zero Adam steps allow an L-BFGS-only run; at least one optimizer step must be requested. Models with a frozen response readout require positive L-BFGS steps.

`loss.energy_weight` and `loss.force_weight` combine scaled energy and force residuals. Set `force_weight: 0.0` for energy-only training. They also determine the readout linear solve's objective. `far_weight` sets per-structure weights for water's `far` and `ion` parts. `contact_energy_weight` (v0.0.13; the clusters part with contact configurations only) weights the contact configurations' energies alone: the energy term is a weighted mean over structures (`Structure.energy_weight` on top of `Structure.weight`), their forces keep their weight. `energy_scale: null` derives the normalization from training energies; positive values supply a manual scale in hartree. Targets are never mean-centered. `verbose` controls progress output. Dataset caches remain valid when only training options change.

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
* `qaqua.py` — q-AQUA v1 as a truth (E3-Q), through the C interface `external/qaqua/qaqua_capi.f90` of the library
  `external/qaqua/build.sh` builds: interaction energies and forces of water clusters in the harness's atom order
  (q-AQUA minus its 1-body term; the gradient analytic except q-AQUA's own switching and charge derivatives, exact
  to ~1e-7, and the energy carrying ~1e-10 E_h of rounding noise), the LTP2011 charges and dipole of a monomer, the
  many-body decomposition by subset energies (exact for this 4-body-truncated potential), the raw CCSD(T) n-body
  sets of the q-AQUA repository with the near-rigid filter, and the authors' benchmarks (`python sseft/qaqua.py
  --check external/qaqua/build`).  Only numpy and ctypes: it runs without JAX.
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
    python scripts/run_experiment.py --config configs/water_qaqua.yaml --out_dir results/e3q   # E3-Q (after external/qaqua/build.sh)
    python scripts/run_experiment.py --config configs/water_qaqua_pol.yaml --out_dir results/e3qp  # E3-QP (after build_pol.sh)
    python scripts/run_experiment.py --config configs/water_qaqua_pol2.yaml --out_dir results/e3qp2  # E3-QP2
    python scripts/run_experiment.py --config configs/water_qaqua_pol3.yaml --out_dir results/e3qp3  # E3-QP3
    python scripts/run_experiment.py --config configs/water_qaqua_pol4a.yaml --out_dir results/e3qp4a  # E3-QP4, job A
    python scripts/run_experiment.py --config configs/water_qaqua_pol4b.yaml --out_dir results/e3qp4b  # E3-QP4, job B
    python scripts/run_experiment.py --config configs/water_qaqua_pol5.yaml --out_dir results/e3qp5    # E3-QP5
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
    configs/      the YAML configurations (configs/water_qaqua.yaml: E3-Q; configs/water_qaqua_pol.yaml: E3-QP;
                  configs/water_qaqua_pol2.yaml: E3-QP2; configs/water_qaqua_pol3.yaml: E3-QP3;
                  configs/water_qaqua_pol4a.yaml, pol4b.yaml: E3-QP4's two jobs; configs/water_qaqua_pol5.yaml: E3-QP5)
    external/     the q-AQUA and q-AQUA-pol build scripts and their C interfaces (the builds, external/qaqua/build*/, are not committed)
    scratch/      one-off computations quoted in the notes (not part of the package)
    results/      the logs and jsons of every run quoted in RESULTS.md
    RESULTS.md    running log of the regression numbers per commit

The design, the predictions and the measurement protocols live in the project notes
(`multiscale-eft-proposal.md`, `phase-d-build-plan.md`, `x1-prediction.md`, `x2-protocol.md`,
`phase1c-pmin-measurement.md`); this repository is the executable part.

License: Apache 2.0 (see `LICENSE`).
