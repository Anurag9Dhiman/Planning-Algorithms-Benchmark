#!/bin/bash
#SBATCH --job-name=oracle-fourrooms
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/05b_oracle_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/05b_oracle_%j.err

# Oracle FourRooms continuation — remaining 8 planners not completed in job 527405.
# bfs/ucs/random_shooting/beam_search already done; skip them.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

cd $PROJECT

$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         minigrid_fourrooms \
    --world-models perfect \
    --planners     mcts astar greedy_bfs cem mcts_value latent_rollout value_guided_astar muzero_style \
    --episodes     20 \
    --budget       200 \
    --device       cpu \
    --output       $RESULTS/oracle_fourrooms

echo "Done: $(date)"
