#!/bin/bash
#SBATCH --job-name=collect-goalft-doorkey
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36a_collect_goalft_doorkey_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/36a_collect_goalft_doorkey_%j.err

# Collect DoorKey-8x8 data with BFS-guided episodes to get successful trajectories
# needed for LeWM goal-conditioned fine-tuning.
#
# Root cause of 0 successes in job 22a: BFS _key() was missing door open/closed
# state — now fixed. With the fix BFS correctly distinguishes closed vs open door
# and finds pick-up-key → open-door → reach-goal paths.
#
# epsilon=0.95: 95% BFS-guided episodes — almost every episode is successful.
# bfs-depth=250: DoorKey-8x8 optimal paths are ~50-150 steps.
# memory: 500 ep × 250 steps × 224×224×3 ≈ 18 GiB; 32G is fine.
# wall: BFS state space ~1024 states × 7 actions; fast with no-render BFS fix.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA=/scratch/ma25m004/Planning-Algorithms-Benchmark/data

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $DATA

$PYTHON training/collect_data.py \
    --env            MiniGrid-DoorKey-8x8-v0 \
    --episodes       500 \
    --epsilon-greedy 0.95 \
    --max-steps      250 \
    --bfs-depth      250 \
    --output         $DATA/minigrid_doorkey_8x8_goalft_data.npz

echo "Done: $(date)"
ls -lh $DATA/minigrid_doorkey_8x8_goalft_data.npz
$PYTHON -c "
import numpy as np
d = np.load('$DATA/minigrid_doorkey_8x8_goalft_data.npz')
print(f'obs shape    : {d[\"obs\"].shape}')
print(f'transitions  : {len(d[\"obs\"]):,}')
print(f'reward>0     : {(d[\"reward\"]>0).sum()} ({100*(d[\"reward\"]>0).mean():.3f}%)')
print(f'done=True    : {d[\"done\"].sum()} ({100*d[\"done\"].mean():.3f}%)')
n_success = (d[\"done\"] & (d[\"reward\"]>0)).sum()
print(f'successful terminal transitions: {n_success}')
"
