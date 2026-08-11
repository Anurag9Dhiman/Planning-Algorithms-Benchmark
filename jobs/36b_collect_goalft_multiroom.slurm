#!/bin/bash
#SBATCH --job-name=collect-goalft-multiroom
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36b_collect_goalft_multiroom_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36b_collect_goalft_multiroom_%j.err

# Collect MultiRoom-N4-S5 data with deep BFS to get successful trajectories
# needed for LeWM goal-conditioned fine-tuning.
#
# Root cause of 0 successes in job 32a: BFS depth capped at min(env.max_steps,200)
# and _key() was missing door states — MultiRoom has 3 unlocked doors between rooms.
# Fix: bfs-depth=400 (4 rooms × ~100 steps each), door states now in key.
#
# The no-render BFS fix (job 36a) eliminates PIL resize per BFS expansion, making
# MultiRoom's larger state space (~4800 states) feasible within 12h wall time.
#
# epsilon=0.99: almost all BFS-guided so we maximise successful episode count.
# memory: 600 ep × 400 steps × 224×224×3 ≈ 45 GiB; 64G is fine.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $DATA

$PYTHON training/collect_data.py \
    --env            MiniGrid-MultiRoom-N4-S5-v0 \
    --episodes       600 \
    --epsilon-greedy 0.99 \
    --max-steps      400 \
    --bfs-depth      400 \
    --output         $DATA/minigrid_multiroom_n4_goalft_data.npz

echo "Done: $(date)"
ls -lh $DATA/minigrid_multiroom_n4_goalft_data.npz
$PYTHON -c "
import numpy as np
d = np.load('$DATA/minigrid_multiroom_n4_goalft_data.npz')
print(f'obs shape    : {d[\"obs\"].shape}')
print(f'transitions  : {len(d[\"obs\"]):,}')
print(f'reward>0     : {(d[\"reward\"]>0).sum()} ({100*(d[\"reward\"]>0).mean():.3f}%)')
print(f'done=True    : {d[\"done\"].sum()} ({100*d[\"done\"].mean():.3f}%)')
n_success = (d[\"done\"] & (d[\"reward\"]>0)).sum()
print(f'successful terminal transitions: {n_success}')
"
