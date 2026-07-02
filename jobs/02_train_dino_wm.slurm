#!/bin/bash
#SBATCH --job-name=planbench-dinowm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/02_dino_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/02_dino_%j.err

# Train DINO-WM heads on each environment dataset.
# Encoder (DINOv2-small) is frozen — only dynamics/reward/done/value/policy heads are trained.
# Pre-encodes all observations once before the training loop (fast).

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA_DIR=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
CKPT_DIR=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Job     : $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started : $(date)"
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd $PROJECT

for ENV_NAME in minigrid_empty_8x8 minigrid_fourrooms; do
    DATA="$DATA_DIR/${ENV_NAME}.npz"
    OUT="$CKPT_DIR/dino_wm_${ENV_NAME}.pt"

    echo ""
    echo "=== DINO-WM  $ENV_NAME ==="
    $PYTHON training/train_dino_wm.py \
        --data      "$DATA" \
        --epochs    50 \
        --batch-size 256 \
        --lr        3e-4 \
        --device    cuda \
        --output    "$OUT"
done

echo ""
echo "Done: $(date)"
ls -lh $CKPT_DIR/dino_wm_*.pt
