#!/bin/bash
#SBATCH --job-name=train-cnn-wm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/08a_train_cnn_wm_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/08a_train_cnn_wm_%j.err

# Train CNNWorldModel on both envs simultaneously (sequential within job).
# CNN encoder trains from scratch jointly with all heads.
# No pretrained weights, no upscaling — works natively on 64x64 MiniGrid obs.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT

echo ""
echo "════ CNN-WM  ·  MiniGrid Empty-8x8 ════"
$PYTHON training/train_cnn_wm.py \
    --data       $DATA/minigrid_empty_8x8.npz \
    --epochs     100 \
    --batch-size 256 \
    --lr         1e-4 \
    --device     cuda \
    --output     $CKPT/cnn_wm_minigrid_empty_8x8.pt

echo ""
echo "════ CNN-WM  ·  MiniGrid FourRooms ════"
$PYTHON training/train_cnn_wm.py \
    --data       $DATA/minigrid_fourrooms.npz \
    --epochs     100 \
    --batch-size 256 \
    --lr         1e-4 \
    --device     cuda \
    --output     $CKPT/cnn_wm_minigrid_fourrooms.pt

echo ""
echo "Done: $(date)"
