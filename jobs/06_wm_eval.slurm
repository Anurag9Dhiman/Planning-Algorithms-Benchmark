#!/bin/bash
#SBATCH --job-name=wm-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/06_wm_eval_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/06_wm_eval_%j.err

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

cd $PROJECT

# Each env has its own trained checkpoints — run separately so paths are unambiguous.

echo "=== Empty-8x8 ==="
$PYTHON experiments/eval_world_models.py \
    --envs         minigrid_empty_8x8 \
    --world-models dino_wm lewm \
    --checkpoints  $CKPT/dino_wm_minigrid_empty_8x8.pt $CKPT/lewm_minigrid_empty_8x8.pt \
    --device       cuda \
    --episodes     20 \
    --max-steps    100 \
    --output       $RESULTS/wm_eval

echo "=== FourRooms ==="
$PYTHON experiments/eval_world_models.py \
    --envs         minigrid_fourrooms \
    --world-models dino_wm lewm \
    --checkpoints  $CKPT/dino_wm_minigrid_fourrooms.pt $CKPT/lewm_minigrid_fourrooms.pt \
    --device       cuda \
    --episodes     20 \
    --max-steps    100 \
    --output       $RESULTS/wm_eval

echo "Done: $(date)"
