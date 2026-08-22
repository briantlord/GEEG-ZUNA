#!/bin/bash
# Full-cohort launcher. It remains fail-closed until the scientific contract is
# production_ready and APPROVED_RUN_ID exactly matches the immutable manifest.
#SBATCH --job-name=zuna11_metrics
#SBATCH --account=<slurm-allocation-name>
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
#SBATCH --array=0-41%20
#SBATCH --output=/path/to/GEEG_ZUNA/logs/zuna11_%A_%a.out

set -euo pipefail

BASE=/path/to/GEEG_ZUNA
REPO="$BASE/GEEG-ZUNA-share"
OUT="$BASE/zuna11_out"
RUN_MANIFEST="$OUT/run_manifest.json"
APPROVAL="$OUT/APPROVED_RUN_ID"
PYTHON="$BASE/zuna11_env/bin/python"

for path in "$REPO" "$RUN_MANIFEST" "$PYTHON"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path is missing: $path" >&2
        exit 2
    fi
done

cd "$REPO"
RUN_ID=$(
    "$PYTHON" -c 'import sys; sys.path.insert(0,"benchmark"); import run_manifest; print(run_manifest.load_verified(sys.argv[1])["run_id"])' "$RUN_MANIFEST"
)
if [[ ! -f "$APPROVAL" ]] || [[ "$(tr -d '[:space:]' < "$APPROVAL")" != "$RUN_ID" ]]; then
    echo "Run is not approved for this exact run ID: $RUN_ID" >&2
    exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export HF_HOME="$BASE/HF_cache"
export HF_HUB_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export ZUNA11_RECON_CACHE_DIR_V3="$BASE/zuna11_reconstructions_v3"
STAGE0_CACHE="$BASE/stage0_cache_v4"
mkdir -p "$OUT" "$ZUNA11_RECON_CACHE_DIR_V3" "$STAGE0_CACHE"

"$PYTHON" benchmark/preflight.py \
    --run-manifest "$RUN_MANIFEST" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUT"

PREFIX="${RUN_ID:0:16}"
RESULT="$OUT/shard_${PREFIX}_${SLURM_ARRAY_TASK_ID}.csv"
QC="$OUT/shard_${PREFIX}_${SLURM_ARRAY_TASK_ID}.qc.jsonl"
STATUS="$OUT/shard_${PREFIX}_${SLURM_ARRAY_TASK_ID}.status.jsonl"

"$PYTHON" benchmark/metrics/run.py \
    --methods spline zuna \
    --zuna-version 1.1 \
    --zuna-calibration median_survivor_std_zero_mean_carrier \
    --allow-phase2-zuna \
    --run-manifest "$RUN_MANIFEST" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --stage0-cache-dir "$STAGE0_CACHE" \
    --out "$RESULT" \
    --qc-out "$QC" \
    --status-out "$STATUS"
