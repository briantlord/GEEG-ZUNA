# Gated ZUNA 1.1 HPC runbook

Choose the installation root for your HPC and export it once per shell session:

```bash
export GEEG_ZUNA_BASE="/path/to/GEEG_ZUNA"
export GEEG_ZUNA_ACCOUNT="<slurm-allocation-name>"
```

The examples below use `$GEEG_ZUNA_BASE`; no username, allocation, or
site-specific filesystem path is built into the project.

The full array must not be submitted yet. `benchmark/preflight.py` intentionally
fails while the scientific contract has unresolved blockers. The SLURM launcher
also requires an approval file containing the exact immutable run ID; approval
cannot be transferred to changed code, inputs, weights, or configuration.

## 1. Install the source-only release

Build `dist/GEEG-ZUNA-share` locally with `scripts/build_hpc_share.py`, then
upload that generated directory. Do not upload the old hand-maintained share.

The HPC layout is:

```text
$GEEG_ZUNA_BASE/
  GEEG-ZUNA-share/
  GEEG_Raw/
  HF_cache/
  zuna11_env/
  zuna11_out/
  logs/
```

Create `logs/` before `sbatch`; SLURM opens the output path before the script
body executes.

## 2. Verify the environment and weights

Use Python 3.11 at
`$GEEG_ZUNA_BASE/zuna11_env/bin/python`. Install
`requirements.lock.txt` with a CUDA-enabled Torch 2.6.0 build appropriate for
the HPC driver. Model download is a separate online setup step. Execution uses
`HF_HUB_OFFLINE=1` and the exact revision, weight hash, and config hash frozen in
the run manifest.

## 3. Freeze the 42 inputs

After upload completes:

```bash
BASE="$GEEG_ZUNA_BASE"
PYTHON="$BASE/zuna11_env/bin/python"
cd "$BASE/GEEG-ZUNA-share"
mkdir -p "$BASE/logs" "$BASE/zuna11_out"
export HF_HOME="$BASE/HF_cache"
export HF_HUB_CACHE="$HF_HOME"
"$PYTHON" benchmark/run_manifest.py \
  --data-dir "$BASE/GEEG_Raw" --expected-recordings 42 \
  --methods spline zuna --out "$BASE/zuna11_out/run_manifest.json"
```

This hashes all 42 recordings and assigns array task `N` to one explicit
recording. Adding, removing, or renaming a file cannot shift the mapping.

## 4. Required validation ladder

Before the array, complete and record:

- CPU/static contract tests;
- saved one-step artifact replay, expected to fail completeness;
- one epoch at 50 steps, repeated for determinism and calibration sensitivity;
- one full 64-epoch recording across all five drop sets;
- two concurrent HPC tasks proving GPU binding and port isolation;
- a paired Rest1/Rest2 mini-cohort with finite, drop-frame-consistent floors.

Only then change the contract to `production_ready`, remove every listed
blocker, rebuild the release, regenerate the run manifest, and put its full run
ID (alone) in `$BASE/zuna11_out/APPROVED_RUN_ID`.

## 5. Submit and collect

```bash
cd "$BASE/GEEG-ZUNA-share"
sbatch --account="$GEEG_ZUNA_ACCOUNT" \
  --output="$BASE/logs/zuna11_%A_%a.out" \
  benchmark/slurm_zuna11_metrics.sh
```

After every exact shard exists:

```bash
"$PYTHON" benchmark/collect_run.py \
  --run-manifest "$BASE/zuna11_out/run_manifest.json" \
  --shard-dir "$BASE/zuna11_out" \
  --out-prefix "$BASE/zuna11_out/metric_eval_zuna11_v4"
```

The collector refuses missing, stale, unknown, mixed-run, incomplete, non-50-step,
epoch-empty, over-requested-epoch, or model-mismatched shards and writes a
checksummed bundle manifest. A readable recording may contribute fewer epochs
than the requested maximum; its actual Stage-0 and reconstruction shapes must
match.
