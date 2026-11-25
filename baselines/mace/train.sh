#!/bin/bash
# E4's MACE and MACE-LES training (notes/e4-plan.md section 7), with the hyper-parameters fixed here for every run:
# baselines/mace/train.sbatch (the baselines) and run_e4_job15_les.sbatch (job 15's MACE-LES arm) call it.  Run from
# ~/EC in the mace-les env (baselines/mace/README.md).
#
#   MODEL=MACELES NAME=maceles_n64_s0 OUT=results/e4_baselines/maceles_n64_s0 N0=64 SEED=0 EPOCHS=500 SWA=350 \
#       bash baselines/mace/train.sh
#
# MODEL   MACE | MACELES
# NAME    the run's name, which tags its checkpoints (<OUT>/checkpoints/<NAME>_run-<SEED>_epoch-<n>[_swa].pt) and its
#         final models (<OUT>/<NAME>.model, <OUT>/<NAME>_stagetwo.model)
# OUT     the run's directory
# N0      the training size: data/e4/{train,valid}_n<N0>.extxyz
# SEED    the training seed
# EPOCHS  the last epoch; SWA: the first epoch of stage two
# LES     MACELES only: the LES options (default baselines/mace/les.yaml)
#
# --restart_latest: a run continues from its latest checkpoint in <OUT>/checkpoints, if any (a resubmission; job 15's
# releases put a trained run's checkpoints there first).
#
# Hyper-parameters, the same for both models:
#   r_max 6 A and 2 interaction layers (the plan's MACE: r_c 6 A, 2 layers);
#   64 channels, max_L 1, correlation 3 (MACE's water defaults);
#   energy weight 10, forces 1000 (eV, eV/A), then stage two (SWA) from epoch SWA with energy weighted up.
set -euo pipefail
: "${MODEL:?}" "${NAME:?}" "${OUT:?}" "${N0:?}" "${SEED:?}" "${EPOCHS:?}" "${SWA:?}"

case "$MODEL" in
    MACE)    EXTRA=() ;;
    MACELES) EXTRA=(--les_arguments="${LES:-baselines/mace/les.yaml}") ;;
    *) echo "MODEL must be MACE or MACELES, got $MODEL" >&2; exit 1 ;;
esac
mkdir -p "$OUT"

mace_run_train \
    --name="$NAME" \
    --work_dir="$OUT" \
    --model="$MODEL" "${EXTRA[@]}" \
    --train_file="data/e4/train_n${N0}.extxyz" \
    --valid_file="data/e4/valid_n${N0}.extxyz" \
    --energy_key=REF_energy \
    --forces_key=REF_forces \
    --E0s='{1: 0.0, 8: 0.0}' \
    --r_max=6.0 \
    --num_interactions=2 \
    --num_channels=64 \
    --max_L=1 \
    --correlation=3 \
    --energy_weight=10 \
    --forces_weight=1000 \
    --stress_weight=0 \
    --batch_size=2 \
    --valid_batch_size=2 \
    --max_num_epochs="$EPOCHS" \
    --swa --start_swa="$SWA" \
    --ema --ema_decay=0.99 \
    --scheduler_patience=15 \
    --patience=100 \
    --eval_interval=1 \
    --error_table=PerAtomRMSE \
    --default_dtype=float64 \
    --device=cuda \
    --seed="$SEED" \
    --restart_latest \
    --save_cpu
