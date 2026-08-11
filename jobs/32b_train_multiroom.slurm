#!/bin/bash
#SBATCH --job-name=train-multiroom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/32b_train_multiroom_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/32b_train_multiroom_%j.err

# Train base DINO-WM and base LeWM on MultiRoom-N4-S5.
# NOTE: goalft skipped — collected data has 0 successful episodes (env too hard for BFS).
# Depends on job 32a (minigrid_multiroom_n4.npz).

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $CKPT

ENV=minigrid_multiroom_n4
DATA_FILE=$DATA/${ENV}.npz

echo ""
echo "=== Base DINO-WM · MultiRoom ==="
$PYTHON training/train_base_dino_wm.py \
    --data       $DATA_FILE \
    --epochs     30 \
    --batch-size 256 \
    --lr         3e-4 \
    --device     cuda \
    --output     $CKPT/base_dino_wm_${ENV}.pt

echo ""
echo "=== Base LeWM · MultiRoom ==="
$PYTHON training/train_base_lewm.py \
    --data       $DATA_FILE \
    --epochs     50 \
    --batch-size 128 \
    --lr         1e-4 \
    --device     cuda \
    --output     $CKPT/base_lewm_${ENV}.pt

echo ""
echo "Done: $(date)"
ls -lh $CKPT/*${ENV}*.pt
