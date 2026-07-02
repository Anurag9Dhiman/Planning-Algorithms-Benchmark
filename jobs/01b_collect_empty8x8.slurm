#!/bin/bash
#SBATCH --job-name=planbench-data2
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/01b_collect_%j.log
#SBATCH --error=/scratch/ma25m004/Planning-Algorithms-Benchmark/logs/01b_collect_%j.err

PYTHON=/home/ma25m004/.conda/envs/llm_stack/bin/python3
PROJECT=/home/ma25m004/Planning-Algorithms-Benchmark
DATA_DIR=/scratch/ma25m004/Planning-Algorithms-Benchmark/data

echo "Job     : $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started : $(date)"

cd $PROJECT

echo "=== MiniGrid-Empty-8x8-v0 ==="
$PYTHON training/collect_data.py \
    --env          MiniGrid-Empty-8x8-v0 \
    --episodes     1000 \
    --epsilon-greedy 0.0 \
    --render-size  64 \
    --seed         42 \
    --output       $DATA_DIR/minigrid_empty_8x8.npz

echo "Done: $(date)"
ls -lh $DATA_DIR/
