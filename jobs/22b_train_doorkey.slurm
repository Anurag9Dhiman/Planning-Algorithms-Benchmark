#!/bin/bash
#SBATCH --job-name=train-doorkey
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/22b_train_doorkey_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/22b_train_doorkey_%j.err

# Train CNN-WM and DINO-WM on DoorKey-8x8 data.
# CNN-WM: trained from scratch, jointly with all heads.
# DINO-WM: frozen DINOv2 encoder, trained heads only.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $CKPT

ENV=minigrid_doorkey_8x8
DATA_FILE=$DATA/${ENV}.npz

echo ""
echo "=== CNN-WM · DoorKey ==="
$PYTHON training/train_cnn_wm.py \
    --data    $DATA_FILE \
    --epochs  50 \
    --batch-size 128 \
    --lr      3e-4 \
    --device  cuda \
    --output  $CKPT/cnn_wm_fixed_${ENV}.pt

echo ""
echo "=== DINO-WM · DoorKey ==="
$PYTHON training/train_dino_wm.py \
    --data    $DATA_FILE \
    --epochs  30 \
    --batch-size 64 \
    --lr      1e-4 \
    --device  cuda \
    --output  $CKPT/dino_wm_fixed_${ENV}.pt

echo ""
echo "Done: $(date)"
ls -lh $CKPT/*${ENV}*.pt
