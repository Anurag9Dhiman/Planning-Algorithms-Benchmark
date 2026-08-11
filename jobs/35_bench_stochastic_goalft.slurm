#!/bin/bash
#SBATCH --job-name=bench-stocha-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=22:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/35_bench_stochastic_goalft_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/35_bench_stochastic_goalft_%j.err

# Resume bench-stochastic (job 580590) — only the goalft section.
# dino_wm and base_lewm results already saved; only goalft was cut by the 24h limit.
# base_lewm took ~20h → goalft needs similar time → wall=22h.

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
echo "════ base_lewm_goalft · $ENV ════"
$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         $ENV \
    --world-models base_lewm \
    --checkpoints  $CKPT/base_lewm_${BASE_ENV}_goalft.pt \
    --goal-data    $DATA/${BASE_ENV}.npz \
    --planners     lewm_cem track1 track2 \
    --episodes     $EPISODES \
    --budget       $BUDGET \
    --wall-time    $WALL \
    --device       cuda \
    --output       $RESULTS/base_lewm_goalft_${ENV}

echo ""
echo "Done: $(date)"
