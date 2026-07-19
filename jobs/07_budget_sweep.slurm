#!/bin/bash
#SBATCH --job-name=budget-sweep
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/07_budget_sweep_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/07_budget_sweep_%j.err

# Budget sweep: oracle (PerfectModel) × {empty_8x8, fourrooms} × B in {200,500,1000,2000,5000}
# Uses 3 representative planners: one uninformed (bfs), one informed (astar),
# one value-guided (value_guided_astar).
# Goal: find the B at which each env becomes reliably solvable — the budget-success curve.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

cd $PROJECT

for B in 200 500 1000 2000 5000; do
    echo ""
    echo "════ Budget B=${B} ════"

    $PYTHON experiments/run_main.py \
        --config       configs/default.yaml \
        --envs         minigrid_empty_8x8 minigrid_fourrooms \
        --world-models perfect \
        --planners     bfs astar value_guided_astar \
        --episodes     20 \
        --budget       $B \
        --device       cpu \
        --output       $RESULTS/budget_sweep/B${B}
done

echo ""
echo "Done: $(date)"
