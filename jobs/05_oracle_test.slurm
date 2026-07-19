#!/bin/bash
#SBATCH --job-name=oracle-bench
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/05_oracle_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/05_oracle_%j.err

# Oracle (PerfectModel) benchmark — ground-truth ceiling for all planners.
# No GPU needed; uses env deepcopy to simulate steps exactly.
# Purpose: determine if low success rates in jobs 04a/04b are due to
# world model quality or planner/budget limitations.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

cd $PROJECT

$PYTHON experiments/run_main.py \
    --config       configs/default.yaml \
    --envs         minigrid_empty_8x8 minigrid_fourrooms \
    --world-models perfect \
    --planners     all \
    --episodes     20 \
    --budget       200 \
    --device       cpu \
    --output       $RESULTS/oracle

echo "Done: $(date)"
