#!/bin/bash
#SBATCH --job-name=bench-multiroom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/32c_bench_multiroom_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/32c_bench_multiroom_%j.err

# Benchmark base_dino_wm and base_lewm on minigrid_multiroom_n4.
# NOTE: goalft omitted — no successful episodes in training data for this env.
# Depends on job 32b.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results/multiroom_bench

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $RESULTS

ENV=minigrid_multiroom_n4
EPISODES=20
BUDGET=500
WALL=60

echo ""
echo "════ base_dino_wm · $ENV ════"
$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         $ENV \
    --world-models base_dino_wm \
    --checkpoints  $CKPT/base_dino_wm_${ENV}.pt \
    --planners     track1 track2 \
    --episodes     $EPISODES \
    --budget       $BUDGET \
    --wall-time    $WALL \
    --device       cuda \
    --output       $RESULTS/base_dino_wm_${ENV}

echo ""
echo "════ base_lewm · $ENV ════"
$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         $ENV \
    --world-models base_lewm \
    --checkpoints  $CKPT/base_lewm_${ENV}.pt \
    --goal-data    $DATA/${ENV}.npz \
    --planners     lewm_cem track1 track2 \
    --episodes     $EPISODES \
    --budget       $BUDGET \
    --wall-time    $WALL \
    --device       cuda \
    --output       $RESULTS/base_lewm_${ENV}

echo ""
echo "Done: $(date)"
