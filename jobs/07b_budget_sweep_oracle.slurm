#!/bin/bash
#SBATCH --job-name=budget-oracle
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/07b_budget_oracle_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/07b_budget_oracle_%j.err

# Proper oracle budget sweep: wall_time scales with B so model_calls is always
# the binding constraint (not the 30s cap).
# Oracle deepcopy ~0.10s/call → wall_time = B * 0.12  (20% headroom)
# Planners: bfs (uninformed), value_guided_astar (informed + value)
# Envs: both, to get two budget-success curves.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT

# B=200  wall=24s  (baseline — matches original oracle results)
echo ""; echo "════ B=200 ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models perfect --planners bfs value_guided_astar \
    --episodes 20 --budget 200 --wall-time 24 --device cpu \
    --output $RESULTS/budget_sweep_oracle/B200

# B=500  wall=60s
echo ""; echo "════ B=500 ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models perfect --planners bfs value_guided_astar \
    --episodes 20 --budget 500 --wall-time 60 --device cpu \
    --output $RESULTS/budget_sweep_oracle/B500

# B=1000  wall=120s
echo ""; echo "════ B=1000 ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models perfect --planners bfs value_guided_astar \
    --episodes 20 --budget 1000 --wall-time 120 --device cpu \
    --output $RESULTS/budget_sweep_oracle/B1000

# B=2000  wall=240s
echo ""; echo "════ B=2000 ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models perfect --planners bfs value_guided_astar \
    --episodes 20 --budget 2000 --wall-time 240 --device cpu \
    --output $RESULTS/budget_sweep_oracle/B2000

# B=5000  wall=600s
echo ""; echo "════ B=5000 ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models perfect --planners bfs value_guided_astar \
    --episodes 20 --budget 5000 --wall-time 600 --device cpu \
    --output $RESULTS/budget_sweep_oracle/B5000

echo ""; echo "Done: $(date)"
