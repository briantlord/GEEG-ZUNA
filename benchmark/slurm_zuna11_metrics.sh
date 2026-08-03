#!/bin/bash
# =============================================================================
# GEEG-ZUNA — ZUNA 1.1 metric-battery run, SLURM job array (Linux HPC)
# =============================================================================
# Runs the modular metric harness (benchmark/metrics/run.py) with the ZUNA 1.1
# (.fif / reconstruct_fif) reconstructor over all 5 subjects, sharded one
# recording per array task. Linux is the RIGHT venue: torch.compile works, so
# 1.1 runs at full speed (on Windows it is ~8x slower — eager only — which makes
# this pass infeasible; see ../HPC_RUNBOOK_zuna11.md).
#
# Corpus: 5 subjects x (G001:10 + G002-005:8 each) = 42 recordings.
# Each task reconstructs its recording's 5 drop sets (faa, theta_beta,
# frontal_midline_theta, mu_asymmetry, specparam_peaks) with linear + spline +
# zuna(1.1), writing a per-shard CSV (no cross-task races). Merge + aggregate
# after (see runbook).
#
# FILL IN the <...> placeholders, then:  sbatch slurm_zuna11_metrics.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=zuna11_metrics
#SBATCH --account=<your_PI_group>            # PI/sponsor group. NOT needed if you use gpu_windfall.
#SBATCH --partition=gpu_standard             # GPU partition (consumable hrs). Free+preemptible: gpu_windfall (drop --account). NB: plain 'standard' is CPU-ONLY.
##SBATCH --qos=user_qos_<your_PI_group>      # ONLY for buy-in gpu_high_priority; omit for gpu_standard/gpu_windfall
#SBATCH --gres=gpu:1                          # 1 GPU/task. Puma-specific: gpu:volta:1 (V100S 32 GB, recommended). Generic gpu:1 also works.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=06:00:00                       # ample: ~5 zuna-1.1 reconstructions/task, compiled
#SBATCH --array=0-41%20                       # 42 recordings -> 42 tasks; 20 concurrent (tune to allocation)
#SBATCH --output=logs/zuna11_%A_%a.out

set -euo pipefail
mkdir -p logs

# --- paths (edit) ------------------------------------------------------------
REPO=/xdisk/<your_PI_group>/geeg/GEEG-ZUNA            # this repo, on scratch
export HF_HOME=/xdisk/<your_PI_group>/geeg/HF_cache   # Zyphra/ZUNA1.1 weights cache (auto-downloads)
OUT=/xdisk/<your_PI_group>/geeg/zuna11_out
mkdir -p "$OUT"
cd "$REPO"                                            # GEEG_Raw/*.cnt must be under $REPO/GEEG_Raw

# --- environment (choose ONE) ------------------------------------------------
# module load cuda12/12.4
# source ~/.bashrc && conda activate zuna11            # python 3.10; torch>=2.5 +CUDA +triton; mne; scipy
#   one-time in that env:  pip install zuna            # pulls its deps (incl. transformers, lm-eval) on Linux
# --- or apptainer ---
# APPTAINER="apptainer exec --nv /groups/<grp>/containers/zuna11.sif"

# --- run this task's shard ---------------------------------------------------
# NOTE: Linux needs NONE of the Windows compat patches (SIGUSR2/gloo/libuv/torchdynamo) — the wrapper
# gates those to os.name=='nt', so torch.compile runs here. Each task preprocesses its own recording.
python benchmark/metrics/run.py \
    --subjects G001 G002 G003 G004 G005 \
    --methods linear spline zuna --zuna-version 1.1 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "${SLURM_ARRAY_TASK_COUNT}" \
    --out "${OUT}/shard_${SLURM_ARRAY_TASK_ID}.csv"

# After ALL tasks finish, merge shards + aggregate (see ../HPC_RUNBOOK_zuna11.md):
#   awk 'FNR==1 && NR!=1 {next} {print}' "${OUT}"/shard_*.csv > "${OUT}/metric_eval_5subj_zuna11.csv"
#   python benchmark/metrics/aggregate.py --csv "${OUT}/metric_eval_5subj_zuna11.csv"
