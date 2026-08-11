#!/bin/bash
#SBATCH --job-name=bench-doorkey
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/22c_bench_doorkey_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/22c_bench_doorkey_%j.err

# Benchmark all planners on DoorKey-8x8 with trained CNN-WM and DINO-WM.
# Answers Q4: does a new environment expose different bottlenecks?
# DoorKey requires key pickup before goal — tests multi-step prerequisite reasoning.

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results/doorkey

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT
mkdir -p $RESULTS

ENV=minigrid_doorkey_8x8
EPISODES=20
BUDGET=200

for MODEL in cnn_wm dino_wm; do
    CKPT_FILE=$CKPT/${MODEL}_fixed_${ENV}.pt
    if [ ! -f "$CKPT_FILE" ]; then
        echo "SKIP $MODEL (not found: $CKPT_FILE)"
        continue
    fi
    echo ""
    echo "=== $MODEL · DoorKey ==="
    $PYTHON experiments/run_main.py \
        --config       configs/default.yaml \
        --planners     track1 track2 \
        --world-models $MODEL \
        --checkpoints  $CKPT_FILE \
        --envs         $ENV \
        --episodes     $EPISODES \
        --budget       $BUDGET \
        --device       cuda \
        --output       $RESULTS/${MODEL}_${ENV}
done

echo ""
echo "Done: $(date)"
echo ""
echo "=== DoorKey Results ==="
$PYTHON -c "
import json, glob
base = '$RESULTS'
models = ['cnn_wm','dino_wm']
planners = ['mcts','greedy_bfs','cem','mcts_value','muzero_style','value_beam_search']
print(f'planner                cnn_wm      dino_wm')
print('-' * 45)
data = {}
for m in models:
    f = glob.glob(f'{base}/{m}_minigrid_doorkey_8x8/*/summary.json')
    if not f:
        data[m] = {}
        continue
    rows = json.load(open(f[0]))
    data[m] = {r['planner']: r['success_rate'] for r in rows}
for p in planners:
    vals = '  '.join(f'{data.get(m,{}).get(p,0):>10.0%}' for m in models)
    print(f'{p:<22}  {vals}')
" 2>/dev/null || true
