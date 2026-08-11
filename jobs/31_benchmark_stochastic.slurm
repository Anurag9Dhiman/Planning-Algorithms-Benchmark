#!/bin/bash
#SBATCH --job-name=bench-stochastic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/31_bench_stochastic_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/31_bench_stochastic_%j.err

# Benchmark on minigrid_empty_8x8_stochastic (slip_prob=0.1).
# World models trained on DETERMINISTIC data — tests model-reality mismatch.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results/stochastic_bench

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $RESULTS

ENV=minigrid_empty_8x8_stochastic
BASE_ENV=minigrid_empty_8x8
EPISODES=20
BUDGET=500
WALL=60

echo ""
echo "════ base_dino_wm · $ENV ════"
$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         $ENV \
    --world-models base_dino_wm \
    --checkpoints  $CKPT/base_dino_wm_${BASE_ENV}.pt \
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
    --checkpoints  $CKPT/base_lewm_${BASE_ENV}.pt \
    --goal-data    $DATA/${BASE_ENV}.npz \
    --planners     lewm_cem track1 track2 \
    --episodes     $EPISODES \
    --budget       $BUDGET \
    --wall-time    $WALL \
    --device       cuda \
    --output       $RESULTS/base_lewm_${ENV}

# goalft checkpoint from job 29; skip gracefully if not present
GOALFT_CKPT=$CKPT/base_lewm_${BASE_ENV}_goalft.pt
if [ -f "$GOALFT_CKPT" ]; then
    echo ""
    echo "════ base_lewm_goalft · $ENV ════"
    $PYTHON experiments/run_main.py \
        --config       configs/default.yaml \
        --envs         $ENV \
        --world-models base_lewm \
        --checkpoints  $GOALFT_CKPT \
        --goal-data    $DATA/${BASE_ENV}.npz \
        --planners     lewm_cem track1 track2 \
        --episodes     $EPISODES \
        --budget       $BUDGET \
        --wall-time    $WALL \
        --device       cuda \
        --output       $RESULTS/base_lewm_goalft_${ENV}
else
    echo "SKIP goalft — $GOALFT_CKPT not found (run job 29 first)"
fi

echo ""
echo "Done: $(date)"
