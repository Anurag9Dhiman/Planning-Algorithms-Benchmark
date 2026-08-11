#!/bin/bash
#SBATCH --job-name=bench-goalft-dk-mr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36d_bench_goalft_dk_mr_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36d_bench_goalft_dk_mr_%j.err

# Benchmark LeWM (goal-ft) on DoorKey-8x8 and MultiRoom-N4-S5.
# DINO-WM results for these envs already exist (jobs 22c, 32c).
# This job adds the missing LeWM goalft column for the Compositional and
# Visual Complexity groups in the benchmark analysis.
#
# Requires: job 36c (goalft checkpoints) to have completed.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT

EPISODES=20
BUDGET=500
WALL=60

# ── DoorKey-8x8 — LeWM goalft ────────────────────────────────────────────────
mkdir -p $RESULTS/doorkey_goalft_bench
echo ""
echo "════ base_lewm_goalft · DoorKey-8x8 ════"
$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         minigrid_doorkey_8x8 \
    --world-models base_lewm \
    --checkpoints  $CKPT/base_lewm_minigrid_doorkey_8x8_goalft.pt \
    --goal-data    $DATA/minigrid_doorkey_8x8_goalft_data.npz \
    --planners     lewm_cem track1 track2 \
    --episodes     $EPISODES \
    --budget       $BUDGET \
    --wall-time    $WALL \
    --device       cuda \
    --output       $RESULTS/doorkey_goalft_bench/base_lewm_goalft_doorkey

# ── MultiRoom-N4-S5 — LeWM goalft ────────────────────────────────────────────
mkdir -p $RESULTS/multiroom_goalft_bench
echo ""
echo "════ base_lewm_goalft · MultiRoom-N4-S5 ════"
$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         minigrid_multiroom_n4 \
    --world-models base_lewm \
    --checkpoints  $CKPT/base_lewm_minigrid_multiroom_n4_goalft.pt \
    --goal-data    $DATA/minigrid_multiroom_n4_goalft_data.npz \
    --planners     lewm_cem track1 track2 \
    --episodes     $EPISODES \
    --budget       $BUDGET \
    --wall-time    $WALL \
    --device       cuda \
    --output       $RESULTS/multiroom_goalft_bench/base_lewm_goalft_multiroom

echo ""
echo "Done: $(date)"
