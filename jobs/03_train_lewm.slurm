#!/bin/bash
#SBATCH --job-name=planbench-lewm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/03_lewm_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/03_lewm_%j.err

# Train LeWM (JEPA) end-to-end on each environment dataset.
# Encoder + predictor + reward/done/value heads all trained jointly.
# Loss: JEPA next-embedding MSE + SIGReg (Gaussian regulariser) + aux heads.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA_DIR=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
CKPT_DIR=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints

export PYTHONUNBUFFERED=1

echo "Job     : $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started : $(date)"
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd $PROJECT

for ENV_NAME in minigrid_empty_8x8 minigrid_fourrooms; do
    DATA="$DATA_DIR/${ENV_NAME}.npz"
    OUT="$CKPT_DIR/lewm_${ENV_NAME}.pt"

    echo ""
    echo "=== LeWM  $ENV_NAME ==="
    $PYTHON training/train_lewm.py \
        --data       "$DATA" \
        --epochs     100 \
        --batch-size 128 \
        --lr         5e-5 \
        --wd         1e-3 \
        --device     cuda \
        --w-jepa     1.0 \
        --w-sig      0.09 \
        --w-rew      1.0 \
        --w-done     1.0 \
        --w-val      0.5 \
        --output     "$OUT"
done

echo ""
echo "Done: $(date)"
ls -lh $CKPT_DIR/lewm_*.pt
