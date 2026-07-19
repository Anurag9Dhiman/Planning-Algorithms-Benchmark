#!/bin/bash
#SBATCH --job-name=benchmark-new
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/09_benchmark_new_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/09_benchmark_new_%j.err
#SBATCH --dependency=afterok:REPLACE_WITH_08A_JOB_ID:REPLACE_WITH_08B_JOB_ID

# Run full benchmark on all 4 new model checkpoints.
# Depends on jobs 08a and 08b completing successfully.
# Replace REPLACE_WITH_08A_JOB_ID and REPLACE_WITH_08B_JOB_ID with actual job IDs
# from: sbatch jobs/08a_train_cnn_wm.slurm && sbatch jobs/08b_train_dino_ft.slurm

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
CKPT=/scratch/ma25m004/Planning-Algorithms-Benchmark/checkpoints
RESULTS=/scratch/ma25m004/Planning-Algorithms-Benchmark/results

export PYTHONUNBUFFERED=1
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd $PROJECT

# ── World model evaluation (rollout quality) ─────────────────────────────────
echo ""
echo "════ World model eval — CNN-WM ════"
$PYTHON experiments/eval_world_models.py \
    --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models cnn_wm cnn_wm \
    --checkpoints $CKPT/cnn_wm_minigrid_empty_8x8.pt $CKPT/cnn_wm_minigrid_fourrooms.pt \
    --device cuda --episodes 20 \
    --output $RESULTS/wm_eval/cnn_wm.json

echo ""
echo "════ World model eval — DINO-WM fine-tuned ════"
$PYTHON experiments/eval_world_models.py \
    --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models dino_wm_ft dino_wm_ft \
    --checkpoints $CKPT/dino_wm_ft_minigrid_empty_8x8.pt $CKPT/dino_wm_ft_minigrid_fourrooms.pt \
    --device cuda --episodes 20 \
    --output $RESULTS/wm_eval/dino_wm_ft.json

# ── Planning benchmark — CNN-WM ───────────────────────────────────────────────
echo ""
echo "════ Benchmark — CNN-WM · Empty-8x8 ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml \
    --envs minigrid_empty_8x8 \
    --world-models cnn_wm \
    --checkpoints $CKPT/cnn_wm_minigrid_empty_8x8.pt \
    --planners all --episodes 20 --budget 200 --device cuda \
    --output $RESULTS/new_models/cnn_wm_empty8x8

echo ""
echo "════ Benchmark — CNN-WM · FourRooms ════"
$PYTHON experiments/run_main.py \
    --config configs/default.yaml \
    --envs minigrid_fourrooms \
    --world-models cnn_wm \
    --checkpoints $CKPT/cnn_wm_minigrid_fourrooms.pt \
    --planners all --episodes 20 --budget 200 --device cuda \
    --output $RESULTS/new_models/cnn_wm_fourrooms

# ── Planning benchmark — DINO-WM fine-tuned ──────────────────────────────────
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
