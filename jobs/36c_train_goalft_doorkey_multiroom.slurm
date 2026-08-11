#!/bin/bash
#SBATCH --job-name=train-goalft-dk-mr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36c_train_goalft_dk_mr_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36c_train_goalft_dk_mr_%j.err

# Goal-conditioned fine-tuning of base_lewm for DoorKey-8x8 and MultiRoom-N4-S5.
# Requires:
#   - jobs 36a and 36b to have completed successfully (goalft data collected)
#   - base_lewm checkpoints from jobs 24b (doorkey) and 32b (multiroom) to exist
#
# Fine-tuning trains the same JEPA objective but only on transitions from
# successful goal-reaching episodes, biasing the latent space so that L2
# distance becomes a meaningful planning signal (close latents = close to goal).

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT

# ── DoorKey-8x8 ──────────────────────────────────────────────────────────────
echo ""
echo "════ goalft fine-tune · DoorKey-8x8 ════"
$PYTHON training/finetune_lewm_goal.py \
    --checkpoint $CKPT/base_lewm_minigrid_doorkey_8x8.pt \
    --data       $DATA/minigrid_doorkey_8x8_goalft_data.npz \
    --epochs     20 \
    --lr         1e-5 \
    --device     cuda \
    --output     $CKPT/base_lewm_minigrid_doorkey_8x8_goalft.pt

# ── MultiRoom-N4-S5 ──────────────────────────────────────────────────────────
echo ""
echo "════ goalft fine-tune · MultiRoom-N4-S5 ════"
$PYTHON training/finetune_lewm_goal.py \
    --checkpoint $CKPT/base_lewm_minigrid_multiroom_n4.pt \
    --data       $DATA/minigrid_multiroom_n4_goalft_data.npz \
    --epochs     20 \
    --lr         1e-5 \
    --device     cuda \
    --output     $CKPT/base_lewm_minigrid_multiroom_n4_goalft.pt

echo ""
echo "Done: $(date)"
ls -lh $CKPT/base_lewm_minigrid_doorkey_8x8_goalft.pt
ls -lh $CKPT/base_lewm_minigrid_multiroom_n4_goalft.pt
