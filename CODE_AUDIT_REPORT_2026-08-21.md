# GEEG-ZUNA Code Audit Report — 2026-08-21

## Verdict

The corrected ZUNA 1.1 benchmark is **not ready to run on the full HPC array and
cannot currently support a biomarker-preservation conclusion**.

The full corrected experiment has not been run. The only corrected real-model
artifact is one epoch, F3/F4 only, at one diffusion step. The only v3 metric CSV
contains spline results and no ZUNA rows. More importantly, the scripts that
previously reported validation success do not require a complete experiment:
the named ZUNA audit passes a spline-only CSV, and the aggregator accepts one
recording with zero test-retest pairs and `nan` reliability floors.

This is not a conclusion that every implementation detail is wrong. The current
adapter does prevent direct held-out waveform leakage, preserves real epoch
boundaries, disables duplicate filtering/reference, validates masks, and saves
model inputs/outputs/reconstructions. Stage-0 caching also has strong content and
checksum controls. Those strengths are real. They are surrounded by experiment-
level, provenance, spatial-coordinate, scaling, aggregation, and HPC-launch
defects that block trustworthy execution.

## Audit scope and method

This audit ignored prior phase claims and reconstructed behavior from the code,
installed ZUNA 1.1.3 source, current manifests, current result files, and targeted
tests. It did not modify pipeline code or run new ZUNA inference.

Evidence collected included:

- inventory and entry-point/import tracing across the root, `benchmark/`, the
  uploadable share, legacy scripts, archives, environments, caches, and results;
- byte-for-byte comparison of local and share benchmark sources;
- direct inspection of preprocessing, reference, reconstruction, masking,
  normalization, cache, metric, QC, schema, aggregation, and launch code;
- direct inspection of installed pip ZUNA 1.1.3 normalization, masking, position
  discretization, and FIF output code;
- numerical checks of the saved Stage-0 tensor and corrected one-epoch model
  input/full/hybrid/reconstruction artifacts;
- 13 targeted CPU regression tests;
- direct execution of the current aggregator and ZUNA audit against the current
  spline-only smoke file;
- import-resolution checks from four working directories;
- syntax and line-ending checks.

The detailed chronological evidence record is in
`CODE_AUDIT_NOTEBOOK_2026-08-21.md`.

## What is actually present

### Corrected artifacts

- One Stage-0 cache entry for `G001Day1Rest1.cnt`: 64 epochs x 62 channels x
  1280 samples, float32 microvolts.
- One corrected ZUNA cache entry: epoch 0 only, F3/F4 only, one diffusion step.
- One v3 metric smoke CSV: one recording, truth plus spline, all five metrics,
  no ZUNA rows.

### Results that do not answer the corrected question

- Full old results exist in root, `results/`, `benchmark/`, and the share.
- The recent full-recording ZUNA 1.1 broadband run and its five reconstructions
  are under `archive/invalid_broadband_v1` and explicitly marked invalid.
- The share's headline report and reproduction commands describe ZUNA 1.0 and
  legacy pipelines, not a corrected ZUNA 1.1 result.

### Source trees

- `benchmark/` and `GEEG-ZUNA-share/benchmark/` are byte-identical for every
  common non-archive file at audit time.
- The project root also contains runnable legacy ZUNA 1.0 code and a vendored
  `zuna/` package.
- The uploadable share contains both the intended corrected benchmark and an
  older runnable `pipeline/` tree, plus stale reproduction instructions.
- The directory is not a Git repository and has no `.gitignore`.

## Actual intended data flow

The identifiable corrected path is:

1. `metrics/run.py` discovers recordings and metric plug-ins.
2. `stage0_cache.py` verifies or creates a Stage-0 tensor using
   `pilot.preprocess`.
3. For each metric drop set, `pilot.surviving_average_reference` subtracts the
   mean over non-dropped channels.
4. Spline reconstructs with MNE. ZUNA calls `zuna_method_v11.py`.
5. The ZUNA adapter replaces held-out target waveforms with deterministic
   calibration carriers, writes one FIF per real epoch, and calls `_recon11.py`.
6. The helper invokes pip ZUNA 1.1's direct-FIF evaluator with no extra filter or
   average reference, a whole-channel mask, and 50 steps by default.
7. The adapter reads model output on masked channels, restores observed channels
   exactly, and persists input/output/mask/reconstruction artifacts.
8. Broad physiological/integrity QC runs before metric computation.
9. Metric rows and reconstruction QC rows are appended to CSV/JSONL.
10. A separate aggregator attempts to compute same-day Rest1/Rest2 floors and
    compare method errors.

Five metrics currently have five distinct drop sets. Therefore a recording
requires five separate ZUNA reconstructions, not one reconstruction per
submetric. The current subprocess design also reloads the model five times.

## Blocking findings

### 1. There is no full corrected ZUNA result

The one saved corrected model run is an integration smoke, not a validation
run: one epoch, two missing channels, one diffusion step. Published/default
ZUNA 1.1 inference uses 50 steps. No 64-epoch corrected run exists for any one
of the five drop sets, let alone all five.

The HPC launcher is also deliberately stopped by an unconditional `exit 2`.

### 2. The validation scripts certify incomplete data

`audit_zuna11_pipeline.py` returns `status: pass` on the current spline-only
smoke CSV. It never requires method `zuna`, a ZUNA cache manifest, the expected
five ZUNA reconstruction units, 64 epochs, or a complete cohort.

`metrics/aggregate.py` labels the one-recording smoke `Validated Phase 3` even
though every floor is `nan` and there are zero paired days. It validates only
relative completeness among rows already present. If a method, shard, subject,
recording, or both members of a pair are absent together, it can still pass.

These two behaviors invalidate the earlier validation status.

### 3. Floor truth and reconstruction truth use different reference frames

Floor rows are calculated after a full-montage average reference. Reconstruction
error is calculated after a dropout-safe average reference over surviving
channels. The latter prevents held-target leakage, but it changes the metric
being estimated for reference-dependent measures.

In the real spline smoke, the two frontal-midline-theta floor truths are
`0.500472` and `-0.109089`; the reconstruction-row truths in the dropout-safe
frame are `0.620900` and `0.090450`. The reference-frame shifts (`0.120` and
`0.200`) are much larger than the reported spline reconstruction errors
(`0.015` each).

The aggregator compares those dropout-frame reconstruction errors against
full-reference test-retest floors. They are not the same estimand. The current
`ok`/`OVER` decision is therefore invalid for reference-dependent metrics.

### 4. ZUNA spatial coordinates are clipped

The Stage-0 standard-1005 coordinates reach z=`0.141549 m`. Installed ZUNA
1.1.3 uses a `[-0.12, 0.12] m` spatial range for each axis and clamps out-of-
range values after warning. Nine channels exceed the z range: FCZ, C1, CZ, C2,
CP1, CPZ, CP2, PZ, and P2.

All nine are clamped to the maximum discrete z bin, erasing their vertical
distinctions in ZUNA's positional encoding. These include direct targets and
nearby context for the central/midline metrics. The adapter does not validate,
transform, or record this event.

### 5. ZUNA cache identity omits positions

The reconstruction cache hashes blind waveform data, channel names, dropped
indices, model/settings/code, and packages, but not the electrode coordinate
array. A coordinate fix can therefore reuse output generated with the old
spatial tokens if the signal and channel names are unchanged.

### 6. Physical-unit output is confounded by a custom scale estimator

ZUNA performs per-channel, per-segment normalization and inverse-transforms its
output using the pre-mask channel mean and standard deviation. In a benchmark,
using the real held-out waveform for those statistics would leak target
information.

The adapter prevents leakage by replacing each target with a zero-mean carrier
whose standard deviation is the epoch median of surviving-channel standard
deviations. That is a deployable heuristic, but it is not model output and it
has not been validated. It supplies the mean and scale used to express the
generated target in physical units.

Absolute power, frontal/mu asymmetry, specparam offset/peak power, and regional
power ratios may therefore reflect this calibration prior as well as ZUNA. No
sensitivity analysis or deployable calibration comparison exists. Until that is
resolved, amplitude-dependent biomarker claims cannot be attributed to ZUNA
alone.

### 7. HPC execution can select the wrong environment, GPU, port, or model revision

The batch script has no active environment setup and runs plain `python`. It can
therefore inherit the wrong Python/user-site installation—the same failure mode
already observed with Python 3.9 and an older ZUNA package.

The helper overwrites SLURM's `CUDA_VISIBLE_DEVICES` with `0`, risking loss of
the scheduler's GPU binding. Every helper also uses `MASTER_PORT=29500`, so two
array tasks on the same node can collide. It deletes all `SLURM_*` variables and
records no job/GPU provenance.

Hugging Face offline mode is enabled locally but not on HPC. The adapter records
the current local `refs/main`, then the helper loads the repository name without
passing that revision. An upstream refresh could make the executed weights
differ from the pre-run manifest.

### 8. Sharding and collection are based on mutable directory contents

The array assumes 42 recordings but does not verify the exact filenames or
hashes. Each task strides over whichever sorted files exist at job start. An
incomplete upload or later filename change can shift the mapping between task ID
and recording.

Collection concatenates whatever `shard_*_v3.csv` files exist. It does not
require exactly 42 run-matching shards, reject stale shards, or verify all
expected recordings/methods/pairs. This compounds the aggregator's incomplete-
experiment acceptance.

## Other high-severity findings

- The adapter accepts arbitrary same-shaped arrays and unconditionally labels
  them corrected-v2. It does not require or verify a Stage-0 manifest. A stale
  ZUNA 1.1 validator actually supplies legacy `emg=False` data to it.
- Reconstruction failures are not first-class result rows. QC uses held truth
  as an inclusion gate and a ZUNA failure leaves a partial CSV. Without an
  immutable expected-unit manifest, failed units can disappear from analysis,
  creating selection bias.
- ZUNA commits a drop-set cache only after the entire 64-epoch helper succeeds.
  A mid-unit failure deletes all completed temporary epoch outputs. Recovery is
  drop-set-level, not epoch-level, and there is no cache lock for concurrent
  retries.
- Result rows do not hash metric code, common spectral code, runner code, QC
  code, schema code, or aggregator code. Four metrics record only
  `project-native`; formula changes can remain under the same v3 schema label.
- The share README advertises a blocked legacy command and states that a
  nonexistent `.gitignore` protects raw data. Multiple runnable legacy entry
  points and the vendored old ZUNA package remain beside the corrected path.
- Import behavior changes with the working directory. From project root,
  `import zuna` loads the vendored old package; from `benchmark/` and the share,
  it loads pip ZUNA 1.1.3. Distribution metadata still says 1.1.3 even when the
  old module is imported, so a metadata-only check is insufficient.

## Preprocessing findings

### What is strong

- Stage-0 is keyed by raw SHA-256, raw size, frozen preprocessing specification,
  source hashes, package versions, target/minimum epoch counts, and the EMG flag.
- Cached tensor checksum, dtype, shape, channels, positions, and embedded key are
  checked before reuse.
- Filtering happens once on continuous data; resampling precedes epoching; the
  edge crop occurs before ICA/epoch selection; average reference is deferred
  until after dropout; epoch candidates are real marker-locked, nonoverlapping
  five-second windows.
- The current Stage-0 tensor is finite and physiological in the retained band.

### What remains unresolved

- `int32` is frozen and prior project notes say it was confirmed, while MNE
  `auto` fails and `int16` yields different event parsing. The directory does not
  contain independent hardware/file-format evidence establishing that choice.
- The int32 continuous read contains a severe corrupted-looking tail near the
  final seconds (up to roughly 64 million microvolts). The ten-second edge crop
  removes it before ICA/epochs, but session QC runs before cropping and does not
  flag unreasonable scale. Manifest channel standard deviations around 0.84
  million microvolts therefore describe the tail, not retained data.
- Selected event sample/onset/code and accepted/rejected epoch indices are not
  stored, only counts.
- EOG channels are removed before ICA; muscle components are addressed but
  ocular component detection/removal is not.
- M1/M2 are called non-cortical and forbidden as targets, yet remain in the data
  and surviving-channel average reference.

These points require explicit protocol decisions and improved provenance; they
should not be silently inherited from the existing comments.

## Reconstruction and output findings

### Direct leakage check

No direct held-out waveform/statistic leak was found in the corrected adapter
for finite Stage-0 input. It overwrites targets before FIF serialization,
derives scale from surviving channels only, and restores only observed channels.
The regression test that radically changes held-out truth produces the same
blind key and reconstruction.

### Full versus hybrid output

The adapter validates the mask in ZUNA's `hybrid` directory but reads target
waveforms from `full_reconstruction`, then hard-restores good channels.
Installed ZUNA defines full as model everywhere and hybrid as model only on the
mask. With whole-channel masks and seam correction disabled, masked values are
expected to match.

They do match in the saved smoke: exact equality on masked F3/F4; only observed
samples differ. This is confusing provenance but not the source of the observed
bad old results. The safer implementation is still to read the hybrid artifact
or assert masked equality explicitly.

### QC interpretation

The QC gate permits aggregate 1-45 Hz target power between 5% and 1000% of truth
and maximum target amplitude up to 1000 microvolts. RMS ratio is logged but not
gated; correlation, per-channel power, offsets, flatness, clipping, spectral
shape, and epoch discontinuities are not gated.

A pass means finite shape, exact observed-channel preservation, and very broad
scale plausibility. It is not evidence that ZUNA output or a biomarker is good.

## Metric implementation findings

The five current plug-ins implement their stated formulas and enforce exact,
finite output keys. The specparam plug-in uses pinned `specparam==2.0.0rc7`,
passes linear frequency/power values as required, and extracts fixed aperiodic
offset/exponent and Gaussian peak parameters in the documented order.

Remaining issues are experimental rather than a single obvious algebraic bug:

- floor/reference mismatch described above;
- no specparam fit-quality threshold or saved fit diagnostics;
- generic metric provenance with no source/config hashes;
- protocol choices such as integrated versus mean band power, exact bands,
  CSD parameters, and drop-set stress regions are frozen only in code/comments,
  not in an independently reviewed analysis specification;
- the aggregator does not verify each metric's declared drop set, expected
  submetrics, implementation consistency, or common reconstruction diagnostics.

The identical numbers observed in the old specparam run came from the old
invalid execution path. The corrected metric runner groups by exact drop set and
does not reuse one reconstruction across different masks. However, there is no
corrected ZUNA metric run yet to demonstrate the new behavior end to end.

## Entry-point classification

### Intended corrected components, not production-ready

- `benchmark/metrics/run.py`
- `benchmark/stage0_cache.py`
- corrected functions inside `benchmark/pilot.py`
- `benchmark/zuna_method_v11.py`
- `benchmark/_recon11.py`
- `benchmark/reconstruction_qc.py`
- `benchmark/metrics/m_*.py`, `schema_v3.py`, `aggregate.py`
- `run_zuna11_local_one_record.ps1`
- `benchmark/slurm_zuna11_metrics.sh` (currently hard-blocked)

### Explicitly blocked legacy paths

- `benchmark/biomarker_eval.py`
- `benchmark/slurm_zuna_array.sh`

### Legacy/stale paths that remain runnable or misleading

- root `main_pipeline.py`, `load_data.py`, and root experimental scripts;
- root vendored `zuna/`;
- share `pipeline/`;
- `benchmark/zuna_method.py`, `_validate_zuna.py`, `_diag_zuna.py`;
- stale `_validate_zuna11.py`;
- legacy `benchmark/aggregate.py` and the legacy CLI inside `pilot.py`;
- obsolete local repair/merge launcher;
- README/report reproduction commands for the old path.

These should not coexist as peer entry points in a production upload.

## Tests performed

- 13 selected CPU regression tests: all passed.
- Saved corrected model I/O checks: passed for the single one-step epoch.
- Local/share common-source SHA-256 comparison: no differences.
- Stage-0 and corrected smoke code hashes: current source still matches saved
  manifests.
- Import resolution: confirmed working-directory-dependent old/new ZUNA import.
- Coordinate range: nine model-range violations confirmed.
- Aggregator incomplete-input test: accepted and printed `nan` floors.
- ZUNA audit spline-only test: returned `status: pass` without ZUNA.
- Python/PowerShell syntax and SLURM LF line endings: passed.

The tests that pass are useful regression evidence for narrow properties. They
do not cover the blockers above.

## Bottom line

Do not remove the HPC block and do not interpret any current artifact as a
corrected ZUNA 1.1 biomarker result. The next work should be the remediation
sequence in `CODE_REMEDIATION_PLAN_2026-08-21.md`, followed by a new one-recording
50-step gate. A 42-recording array is justified only after that gate produces a
complete, independently validated bundle with consistent truth frames,
coordinate handling, calibration sensitivity, exact environment/model identity,
and failure accounting.

## External primary references checked

- Official ZUNA repository and ZUNA 1.1 usage/training description:
  https://github.com/Zyphra/ZUNA
- ZUNA 1.1 paper: https://arxiv.org/abs/2607.27308
- Official specparam 2.0.0rc7 `SpectralModel` documentation:
  https://specparam-tools.github.io/generated/specparam.SpectralModel.html

