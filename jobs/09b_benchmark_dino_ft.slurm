#!/bin/bash
#SBATCH --job-name=bench-dino-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/09b_bench_dino_ft_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/09b_bench_dino_ft_%j.err
#SBATCH --dependency=afterok:545307

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT

echo ""
echo "════ World model eval — DINO-WM fine-tuned ════"
$PYTHON experiments/eval_world_models.py \
    --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models dino_wm_ft dino_wm_ft \
    --checkpoints $CKPT/dino_wm_ft_minigrid_empty_8x8.pt $CKPT/dino_wm_ft_minigrid_fourrooms.pt \
    --device cuda --episodes 20 \
    --output $RESULTS/wm_eval/dino_wm_ft.json

echo ""
echo "════ Benchmark — DINO-WM-FT · Empty-8x8 ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml \
    --envs minigrid_empty_8x8 \
    --world-models dino_wm_ft \
    --checkpoints $CKPT/dino_wm_ft_minigrid_empty_8x8.pt \
    --planners all --episodes 20 --budget 200 --device cuda \
    --output $RESULTS/new_models/dino_wm_ft_empty8x8

echo ""
echo "════ Benchmark — DINO-WM-FT · FourRooms ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml \
    --envs minigrid_fourrooms \
    --world-models dino_wm_ft \
    --checkpoints $CKPT/dino_wm_ft_minigrid_fourrooms.pt \
    --planners all --episodes 20 --budget 200 --device cuda \
    --output $RESULTS/new_models/dino_wm_ft_fourrooms

echo ""
echo "Done: $(date)"
