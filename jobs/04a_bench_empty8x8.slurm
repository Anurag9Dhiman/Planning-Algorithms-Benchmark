#!/bin/bash
#SBATCH --job-name=bench-empty8x8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/04a_bench_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/04a_bench_%j.err

# Benchmark: LeWM only on empty_8x8
# (DINO-WM results already saved from previous run)

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd $PROJECT

$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         minigrid_empty_8x8 \
    --world-models lewm \
    --checkpoints  $CKPT/lewm_minigrid_empty_8x8.pt \
    --planners     all \
    --episodes     20 \
    --budget       200 \
    --device       cuda \
    --output       $RESULTS/minigrid_empty_8x8

echo "Done: $(date)"
