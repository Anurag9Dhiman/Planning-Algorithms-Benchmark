#!/bin/bash
#SBATCH --job-name=train-dino-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/08b_train_dino_ft_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/08b_train_dino_ft_%j.err

# Fine-tune DINO-WM: unfreeze last 2 of 12 DINOv2 transformer blocks.
# Warm-starts from existing heads checkpoint. Online encoding (slower than
# pre-encoding but necessary since encoder parameters change).
# Enc LR = 1e-5 (10x smaller than head LR) to preserve pretrained features.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
CKPT_IN=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
CKPT_OUT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT

echo ""
echo "════ DINO-WM fine-tune  ·  MiniGrid Empty-8x8 ════"
$PYTHON training/train_dino_wm_finetune.py \
    --data             $DATA/minigrid_empty_8x8.npz \
    --checkpoint       $CKPT_IN/dino_wm_minigrid_empty_8x8.pt \
    --unfreeze-blocks  2 \
    --epochs           30 \
    --batch-size       64 \
    --lr               3e-4 \
    --lr-enc           1e-5 \
    --device           cuda \
    --output           $CKPT_OUT/dino_wm_ft_minigrid_empty_8x8.pt

echo ""
echo "════ DINO-WM fine-tune  ·  MiniGrid FourRooms ════"
$PYTHON training/train_dino_wm_finetune.py \
    --data             $DATA/minigrid_fourrooms.npz \
    --checkpoint       $CKPT_IN/dino_wm_minigrid_fourrooms.pt \
    --unfreeze-blocks  2 \
    --epochs           30 \
    --batch-size       64 \
    --lr               3e-4 \
    --lr-enc           1e-5 \
    --device           cuda \
    --output           $CKPT_OUT/dino_wm_ft_minigrid_fourrooms.pt

echo ""
echo "Done: $(date)"
