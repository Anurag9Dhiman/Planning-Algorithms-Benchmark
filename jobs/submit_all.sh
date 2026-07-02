#!/bin/bash
# Submit the full benchmark pipeline with SLURM dependencies.
#
# Usage:
#   cd /home/ma25m004/Planning-Algorithms-Benchmark
#   bash jobs/submit_all.sh
#
# Job graph:
#   01_collect_data
#       └─► 02_train_dino_wm  ─┐
#       └─► 03_train_lewm     ─┴─► 04_run_benchmark

set -e
cd /home/ma25m004/Planning-Algorithms-Benchmark

# ── Job 1: data collection (CPU) ──────────────────────────────────────────────
JOB1=$(sbatch --parsable jobs/01_collect_data.slurm)
echo "Submitted  01_collect_data    : $JOB1"

# ── Jobs 2 & 3: training (GPU, both depend on job 1) ─────────────────────────
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/02_train_dino_wm.slurm)
echo "Submitted  02_train_dino_wm   : $JOB2  (after $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/03_train_lewm.slurm)
echo "Submitted  03_train_lewm      : $JOB3  (after $JOB1)"

# ── Job 4: benchmark (GPU, depends on both training jobs) ────────────────────
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 jobs/04_run_benchmark.slurm)
echo "Submitted  04_run_benchmark   : $JOB4  (after $JOB2 and $JOB3)"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue --me"
echo "  tail -f /scratch/ma25m004/Planning-Algorithms-Benchmark/logs/01_collect_${JOB1}.log"
