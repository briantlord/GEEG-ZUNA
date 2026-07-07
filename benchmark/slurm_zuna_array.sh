#!/bin/bash
# =============================================================================
# GEEG-ZUNA benchmark — SLURM job-array template for University of Arizona HPC
# =============================================================================
# Verified against UA HPC docs (June 2026):
#   Puma   : 9 GPU nodes x 4 NVIDIA V100S (32 GB)  + 1 node 8x A100 (40 GB MIG)
#   Ocelote: 36 GPU nodes x 2 NVIDIA P100 (16 GB)  (group cap: 10 GPUs at once)
#   "new cat" (2026): 5 GPU nodes x 8 NVIDIA H200 (141 GB)  <- fastest if available
#   Scheduler SLURM | max wall-time 240 h | OS Rocky Linux 9 (Puma) | Apptainer for containers
#   Free monthly: 100k CPU-h (Puma) / 70k (Ocelote). Windfall = preemptible, unlimited.
# ZUNA (380M params) fits comfortably in a V100S (32 GB) or one A100 MIG slice (40 GB).
#
# FILL IN the three <...> placeholders below (group, partition, gres) then:
#   sbatch slurm_zuna_array.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=zuna_bench
#SBATCH --account=<your_PI_group>           # `va` / sponsor group name
#SBATCH --partition=windfall                # windfall = preemptible (free, idempotent queue tolerates it); or 'standard'
#SBATCH --gres=gpu:1                         # Puma V100S; for A100 MIG use gpu:1 + constraint; Ocelote also gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --array=0-49%50                      # 50 array tasks, 50 concurrent -> tune to allocation
#SBATCH --output=logs/zuna_%A_%a.out

set -euo pipefail
mkdir -p logs

# --- environment (choose ONE: module+conda OR apptainer) ---------------------
# module load cuda11/11.8
# source ~/.bashrc && conda activate zuna        # env with torch==2.5.1, mne, scipy
# --- or container ---
# APPTAINER="apptainer exec --nv /groups/<grp>/containers/zuna.sif"

DATA_DIR=/xdisk/<your_PI_group>/geeg/GEEG_Raw   # staged raw .cnt (high-capacity scratch)
OUT_DIR=/xdisk/<your_PI_group>/geeg/bench_out   # metrics + manifest (durable)
export HF_HOME=/xdisk/<your_PI_group>/geeg/HF_cache

# Each array task owns a shard of subjects; the idempotent manifest (benchmark/pilot.py
# Manifest class) means a preempted/failed task just leaves its units 'pending' for the
# next run — re-submit the same array to resume, nothing is recomputed.
python3 benchmark/run_units.py \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --array_id "${SLURM_ARRAY_TASK_ID}" --array_size "${SLURM_ARRAY_TASK_COUNT}" \
    --n_drop 2 4 8 16 32 --patterns scattered contiguous --trials 8 --epochs 64 \
    --methods zero mean nearest linear spline zuna \
    --zuna_data_norm 10.0 --zuna_steps 50 --gpu 0

# NOTES
# - run_units.py = thin wrapper over pilot.run_pilot that shards the recording list by
#   (array_id, array_size) and enables the ZUNA rung (reconstruct_zuna -> zuna.inference()).
# - One ZUNA model load per array task; stream many units through it (amortize load cost).
# - Do per-unit scratch I/O on $TMPDIR (node-local NVMe ~1.9 TB on Puma); sync only
#   metrics.csv + manifest.jsonl back to $OUT_DIR.
# - To target a V100 explicitly on Puma: #SBATCH --gres=gpu:volta:1
