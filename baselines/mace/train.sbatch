#!/bin/bash
#SBATCH --job-name=e4_mace
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s234633@dtu.dk
#SBATCH --partition=sm3090el9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00
#SBATCH --output=logs/e4_mace_%j.out
#SBATCH --error=logs_error/e4_mace_%j.err

# E4's baselines (notes/e4-plan.md section 7): MACE (short range) and MACE-LES (latent Ewald charges), trained on one
# training size L0 of E4's data with identical hyper-parameters; the only difference is --model.
#
#   MODEL=MACELES N0=64 SEED=0 sbatch baselines/mace/train.sbatch
#   MODEL=MACE    N0=64 SEED=0 sbatch baselines/mace/train.sbatch
#   SMOKE=1 MODEL=MACELES N0=64 sbatch -p sm3090el9_devel -t 00:20:00 baselines/mace/train.sbatch   # 2 epochs
#
# MODEL  MACE | MACELES       N0  64 | 128 | 256 (the training size; data/e4/{train,valid}_n<N0>.extxyz)
# SEED   training seed (the plan trains every rung with 4 seeds)
# Output: results/e4_baselines/<model>_n<N0>_s<SEED>/ (model, logs, checkpoints). Then evaluate on every size:
#   python baselines/mace/eval.py results/e4_baselines/<run>/<run>_stagetwo.model   (the final model; <run>.model is
#   stage one's best checkpoint)
#
# Data: energies are MB-pol interaction energies (an isolated rigid monomer is 0), so E0s are 0 for H and O.
# Needs `python scripts/e4_data.py assemble` first, and the mace-les env (baselines/mace/README.md).

set -euo pipefail

MODEL=${MODEL:-MACELES}
N0=${N0:-64}
SEED=${SEED:-0}
SMOKE=${SMOKE:-0}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mace-les
export PYTHONNOUSERSITE=1          # keep ~/.local packages out of the env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation on the 1024-molecule cells

cd ~/EC

case "$MODEL" in
    MACE)    NAME=mace;   EXTRA=() ;;
    MACELES) NAME=maceles; EXTRA=(--les_arguments=baselines/mace/les.yaml) ;;
    *) echo "MODEL must be MACE or MACELES, got $MODEL" >&2; exit 1 ;;
esac

RUN=${NAME}_n${N0}_s${SEED}
EPOCHS=500
SWA=350
if [ "$SMOKE" = 1 ]; then RUN=smoke_${RUN}; EPOCHS=2; SWA=1; fi
OUT=results/e4_baselines/$RUN
mkdir -p "$OUT"

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Run $RUN: model $MODEL, L0 = $N0 molecules, seed $SEED, $EPOCHS epochs"
python -c "import torch, mace, les; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mace', mace.__version__, '| les', les.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Hyper-parameters, the same for both models:
#   r_max 6 A and 2 interaction layers (the plan's MACE: r_c 6 A, 2 layers);
#   64 channels, max_L 1, correlation 3 (MACE's water defaults);
#   energy weight 10, forces 1000 (eV, eV/A), then stage two (SWA) from epoch $SWA with energy weighted up.
mace_run_train \
    --name="$RUN" \
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
    --max_num_epochs=$EPOCHS \
    --swa --start_swa=$SWA \
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

echo
echo "=== evaluation on every size ==="
EVAL_EXTRA=()
if [ "$SMOKE" = 1 ]; then EVAL_EXTRA=(--max_frames 2); fi
python -u baselines/mace/eval.py "$OUT/${RUN}_stagetwo.model" "${EVAL_EXTRA[@]}"

echo "Completed at $(date)"
