# E4 baselines: MACE and MACE-LES

The external baselines of `notes/e4-plan.md` §7:
- **MACE**: short range only (r_c 6 Å, 2 layers). It sits on the same rung as M_∞.
- **MACE-LES**: MACE plus Latent Ewald Summation, i.e. learned per-atom charges summed by Ewald. This is Theorem 2's
  class: no sum rule, no field response.

Both use the same hyper-parameters and train on the same E4 splits (`scripts/e4_data.py assemble`'s extxyz). The only
difference between them is `--model`.

## Environment

A separate conda env, `mace-les`. It's kept apart from `ec` because it uses PyTorch rather than JAX. Built on the login
node on 31 Oct 2026:

```bash
conda create -n mace-les python=3.11 pip
conda activate mace-les
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/ACEsuit/mace.git@7215fadd3b591444ab2c07634e84419685a48e87" \
            "git+https://github.com/ChengUCB/les.git@35b971a650cf30adb84fef7138761e81aa47b3a2"
```

The versions are mace-torch 0.3.18 and les 0.2.0. MACE-LES is `--model MACELES` in upstream MACE, which imports `les`.
Set `PYTHONNOUSERSITE=1` (the scripts do this) so that packages in `~/.local` don't leak into the env.

## Running

```bash
python scripts/e4_data.py assemble                                   # once: data/e4/{train,valid,test}_n*.extxyz
MODEL=MACELES N0=64 SEED=0 sbatch baselines/mace/train.sbatch        # one model, one L0, one seed
MODEL=MACE    N0=64 SEED=0 sbatch baselines/mace/train.sbatch
python baselines/mace/eval.py results/e4_baselines/<run>/<run>_stagetwo.model  # also run by train.sbatch
```

- `N0` is 64, 128 or 256 (the training size, i.e. L₀). The plan trains every rung with 4 seeds.
- `SMOKE=1` runs 2 epochs and evaluates 2 frames per set. Use it to check the pipeline on `sm3090el9_devel`.
- Output goes to `results/e4_baselines/<model>_n<N0>_s<SEED>/`. The final model is `<run>_stagetwo.model` (after SWA/stage two); `<run>.model` is only stage one's best checkpoint. `eval.py` writes `<run>_eval.json` with the following
  for every valid/test set (64 to 1024 molecules):
  - the per-molecule energy error: signed mean, MAE and RMSE;
  - the force RMSE;
  - for MACE-LES, the net latent charge per molecule.

## Choices, and why

- **E0s = 0 for H and O.** The labels are MB-pol interaction energies, so an isolated rigid monomer is zero.
- **LES is monopole only** (`les.yaml`), with the published defaults: σ 1 Å, k_max = π Å⁻¹, self-interaction removed.
  - LES now has options for latent dipoles/quadrupoles and for induced charges and dipoles (`use_induced_dipole`
    etc.). Those give it a field response, so ε_∞ ≠ 1. That puts it outside Theorem 2's class.
  - If one of those variants is wanted later, it's a separate baseline with its own yaml.
- **The legacy Ewald path** (`is_periodic` unset). It sizes the k-grid from each cell.
  - The vectorized path (`is_periodic: True`) has a fixed grid, N_max = 10. It silently truncates cells longer than
    20 Å, which are the 512- and 1024-molecule test cells.
- **Hyper-parameters**: r_max 6 Å, 2 interactions, 64 channels, max_L 1, correlation 3.
  - Loss weights: energy 10, forces 1000. Stage two (SWA) runs from epoch 350 of 500, with EMA. The model is float64.
  - These are a reasonable default and haven't been tuned. Tune them on validation, at one L₀, the same way for both
    models, before the L₀ sweep.

## Not yet here

- **ε_∞ and Z\*** (X4b, T4). Plain MACE has no field coupling. For MACE-LES, ε_∞ = 1 holds by construction, and its
  Z\* comes from `mace_eval_configs --compute_bec` or `MACECalculator(..., compute_bec=True)`.
- **Form I** (fixed clusters in growing cells) and the open far-field probes. Run them through `MACECalculator` once
  those structures exist.
- **The E3-QP8 clusters** (§7's first step, no new data). This needs an extxyz export of E3's clusters.
