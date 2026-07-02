#!/bin/bash
#SBATCH --job-name=bench-fourrooms
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/04b_bench_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/04b_bench_%j.err

# Benchmark: both world models on fourrooms

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
    --envs         minigrid_fourrooms \
    --world-models dino_wm lewm \
    --checkpoints  $CKPT/dino_wm_minigrid_fourrooms.pt $CKPT/lewm_minigrid_fourrooms.pt \
    --planners     all \
    --episodes     20 \
    --budget       200 \
    --device       cuda \
    --output       $RESULTS/minigrid_fourrooms

echo "Done: $(date)"
