#!/bin/bash
#SBATCH --job-name=collect-doorkey
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/22a_collect_doorkey_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/22a_collect_doorkey_%j.err

# DoorKey data collection — 96G to hold ~2.5M transitions (64x64x3 uint8 x2).
# Uses 30% BFS-guided episodes so sparse reward signal is present.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $DATA

$PYTHON training/collect_data.py \
    --env            MiniGrid-DoorKey-8x8-v0 \
    --episodes       2000 \
    --epsilon-greedy 0.3 \
    --max-steps      200 \
    --output         $DATA/minigrid_doorkey_8x8.npz

echo "Done: $(date)"
ls -lh $DATA/minigrid_doorkey_8x8.npz
$PYTHON -c "
import numpy as np
d = np.load('$DATA/minigrid_doorkey_8x8.npz')
print(f'obs shape    : {d[\"obs\"].shape}')
print(f'transitions  : {len(d[\"obs\"]):,}')
print(f'reward>0     : {(d[\"reward\"]>0).sum()} ({100*(d[\"reward\"]>0).mean():.3f}%)')
print(f'done=True    : {d[\"done\"].sum()} ({100*d[\"done\"].mean():.3f}%)')
"
