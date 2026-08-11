#!/bin/bash
#SBATCH --job-name=viz-doorkey
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/37_viz_doorkey_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/37_viz_doorkey_%j.err

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data
VIZ=/scratch/ma25m004/Planning-Algorithms-Benchmark/viz/doorkey_failure

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $VIZ

$PYTHON experiments/visualize_doorkey_failure.py \
    --checkpoint $CKPT/base_lewm_minigrid_doorkey_8x8_goalft.pt \
    --goal-data  $DATA/minigrid_doorkey_8x8_goalft_data.npz \
    --output-dir $VIZ \
    --seed       3

echo "Done: $(date)"
ls -lh $VIZ/
