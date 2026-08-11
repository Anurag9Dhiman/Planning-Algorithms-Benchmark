#!/bin/bash
#SBATCH --job-name=collect-multiroom
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/32a_collect_multiroom_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/32a_collect_multiroom_%j.err

# Collect 2000 episodes from MultiRoom-N4-S5 (visual complexity category).
# MultiRoom has random room sizes and positions — high visual diversity.
# 40% BFS-guided (epsilon=0.4) because the env is harder than DoorKey:
# longer corridors, harder to reach goal by accident.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $DATA

$PYTHON training/collect_data.py \
    --env            MiniGrid-MultiRoom-N4-S5-v0 \
    --episodes       3000 \
    --epsilon-greedy 0.4 \
    --output         $DATA/minigrid_multiroom_n4.npz

echo "Done: $(date)"
ls -lh $DATA/minigrid_multiroom_n4.npz
$PYTHON -c "
import numpy as np
d = np.load('$DATA/minigrid_multiroom_n4.npz')
print(f'obs shape    : {d[\"obs\"].shape}')
print(f'transitions  : {len(d[\"obs\"]):,}')
print(f'reward>0     : {(d[\"reward\"]>0).sum()} ({100*(d[\"reward\"]>0).mean():.3f}%)')
print(f'done=True    : {d[\"done\"].sum()} ({100*d[\"done\"].mean():.3f}%)')
"
