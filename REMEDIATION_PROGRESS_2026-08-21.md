# GEEG-ZUNA remediation progress — 2026-08-21

This is the execution ledger for `CODE_REMEDIATION_PLAN_2026-08-21.md`. It is
derived from changes and tests in the actual working tree. It does not reuse the
abandoned phase labels.

## Current verdict

Remediation is in progress. The project is substantially safer, but ZUNA model
execution and the full HPC array remain blocked. No new ZUNA inference was run
during this remediation work.

## Completed

### Authoritative source/release boundary

- Initialized a root Git repository and committed the audited source baseline as
  `5e4c6a7120ccc44b48e2f5e417c93cc21effac4b` using the authenticated GitHub
  no-reply identity for `briantlord`.
- Added ignore rules for raw EEG, model weights, environments, caches, results,
  generated releases, and archives.
- Declared `benchmark/` authoritative and made `GEEG-ZUNA-share/` generated.
- Preserved the old hand-maintained share under
  `archive/handmaintained_share_2026-08-21`.
- Preserved the vendored ZUNA 1.0 package and stale executables under
  `archive/legacy_active_tree_2026-08-21`.
- Removed the project-root `zuna` import collision. The active-tree check now
  resolves ZUNA to the pinned virtual-environment package.
- Hard-blocked the legacy `pilot.py` CLI and removed the permissive old ZUNA
  audit from the active release.
- Added a deterministic release builder and per-file SHA-256 release manifest.
  The current generated share contains 66 files and no legacy `pipeline/` or
  vendored ZUNA package. An independent exact verifier rejects missing, extra,
  size-mismatched, hash-mismatched, or non-canonical release content. The current
  verified source-content hash is
  `d8856bfa964e44f87fb4a016377e15b8e131eeb80ed782ebbc35cf55e35c5726`.

### Scientific/result identity

- Added `config/scientific_contract_v1.json` and a hashed experiment ID.
- Truth, reconstruction error, and test-retest floors are now defined in the
  same drop-set-specific surviving-channel reference frame.
- M1/M2 remain observed model context but are excluded from reference
  contributors, drop targets, and metric channels.
- Every reconstruction result points to a deterministic matching truth-unit ID.
- Rows carry the experiment, contract, run, Stage-0, metric/config, common
  spectral code, runner, QC, schema, and aggregator hashes.
- The aggregator rejects mixed runs/contracts/provenance, mismatched truth
  frames, missing truth links, no reconstruction methods, and zero paired floors.

### Stage-0

- QC now describes the cropped analysis interval rather than the corrupted raw
  footer/tail.
- Raw-tail QC is stored separately with an explicit warning.
- HEOG/VEOG remain available through ICA; ocular and muscle component indices
  and scores are recorded before auxiliary channels are removed.
- A complete event ledger records raw-sample estimate, resampled sample, onset,
  code, non-overlap selection, annotation acceptance, amplitude decision,
  peak-to-peak range, and final selected order.
- Stage-0 cache v3 hashes the data array, positions, and channel list and has a
  standalone verifier plus inspectable lock ownership.
- A real current-source non-model build on `G001Day1Rest1.cnt` completed and
  independently verifies: 64 x 62 x 1280, cache key
  `0251e6c90ed7a28ba2f88e34df460ca0d886be3a07f7821dfc44ef80f1071a53`.
  It records full component scores/topographies and warns on the abnormal raw
  footer/tail. The earlier `bb301f...` smoke is source-stale by design.
- The ICA review generator produces a 20-row reviewer table with detector
  labels/scores, strongest topography channels, PCA variance, and blank review
  fields. The current automatic policy still excludes 14/20 components and
  remains a release blocker rather than silently passing.

### ZUNA adapter and execution identity

- The adapter requires a typed verified Stage-0 object; naked arrays are rejected.
- Position arrays and discrete bins are part of cache identity.
- Coordinate handling now exactly documents and replicates the pinned official
  tokenizer. Original `standard_1005` head coordinates, componentwise-clipped
  model coordinates, and 100-bin XYZ tokens are all stored and hashed. Nine
  channels saturate the current montage's Z token, but all 62 full XYZ triplets
  remain unique; any future token collision fails before inference.
- Hybrid output is canonical and masked full/hybrid values must match.
- Normalized dropped output is stored separately from physical calibration.
- Three explicit blind calibration inputs are implemented and content-addressed:
  whole-montage median survivor SD, four-nearest-survivor median SD, and a
  fixed-penalty observed-position log-SD ridge model. All three are invariant to
  arbitrary changes in held-target waveforms in regression tests. The selected
  strategy is frozen in the run manifest and verified against every successful
  ZUNA reconstruction manifest.
- The helper runs Hugging Face offline, verifies revision/weight/config before
  and after inference, preserves scheduler GPU/SLURM state, and uses a
  per-process rendezvous port.
- Direct hidden-waveform invariance, exact observed-channel restoration, mask,
  boundary, cache-tamper, coordinate rejection, and position-key tests pass.
- Model inputs and outputs now live in the content-addressed unit directory
  before inference starts. Every epoch's full FIF, hybrid FIF, and mask are
  independently read and verified.
- A failed helper attempt preserves verified epochs, records completed and
  missing/invalid epoch indices, and retries only missing/invalid epochs.
- An atomic cache-unit lock records key, host, PID, and acquisition time and
  refuses concurrent ownership without deleting an unknown lock.
- An injected interruption after epoch 1 of a two-epoch test resumes by
  submitting only epoch 2 and then completes successfully.

### Expected-unit validation and HPC structure

- The immutable run manifest hashes exact recording paths/sizes/content,
  methods, model, sources, pair structure, result units, reconstruction units,
  and array mapping.
- The frozen 42-recording spline+ZUNA design requires exactly 1,764 result units
  and 420 reconstruction/QC units.
- The bundle validator rejects missing/extra/duplicate/mixed units and verifies
  successful ZUNA units used the exact model, 50 diffusion steps, and 64 epochs.
- HPC tasks map one-to-one to immutable manifest entries. The launcher uses an
  explicit interpreter and exact run-ID approval. The collector requires every
  exact shard and replaces wildcard concatenation with validated atomic output.
- Before computation, each selected expected unit is atomically registered as
  `preempted_incomplete`. Durable successes remove that state; caught model,
  QC, metric, and input failures replace it with the specific terminal state.
- Failure ledgers are retry-safe: a later failure replaces the prior state and
  a durable success clears it. Mixed run IDs and success/failure overlaps fail.
- The collector now requires and merges the exact status shard for every task.
  Complete bundles may report failures without dropping them from denominators;
  the validator still supports a stricter all-success gate.
- Machine result rows retain full Python floating-point representations instead
  of rounding to six decimals.
- Reconstruction QC v2 persists per-epoch/per-dropped-channel mean, SD, RMS,
  1-45 Hz power, power ratio, maximum amplitude, flatness, clipping fraction,
  waveform correlation, and log-spectral RMSE.
- Successful specparam rows persist fit status, posterior-channel count,
  R-squared, mean absolute error, detected-peak count, and alpha-peak count.

## Verification completed

- 23 CPU regression tests pass.
- The same 23 tests pass from the generated HPC share with bytecode writes
  disabled, and exact release verification still passes afterward.
- All edited Python files compile.
- Active-tree/import check passes.
- The current generated release excludes known legacy executables.
- A deliberately incomplete result bundle is rejected.
- The real Stage-0 smoke described above completed without invoking ZUNA.
- Comparing the new Stage-0 tensor with the prior tensor showed 0.809 retained
  1-45 Hz power, 0.975 retained alpha power, and samplewise correlation 0.847.

## Blocking decisions/evidence still required

1. Independent CNT/int32 acquisition-format evidence.
2. Independent review of the aggressive ICA decision. The prior pipeline marked
   13/20 components as muscle; ocular detection adds one unique component, for
   14/20 excluded in the current recording.
3. Calibration sensitivity results for all predeclared
   physical-scale strategies.
4. A frozen specparam fit-quality rule. Fit diagnostics are now persisted.

## Implementation still incomplete

- The physical-scale calibration alternatives are implemented but have not been
  evaluated with production 50-step output. The current median-survivor carrier
  remains only a development primary, not a released choice.
- Specparam fit diagnostics are persisted, but the minimum acceptable R-squared
  threshold remains deliberately unfrozen and therefore blocks release.
- Determinism at the production 50-step setting has not been measured.
- The rebuilt HPC path has not had the required two-task same-node systems test.

## Next implementation order

1. Resolve or externally escalate the CNT-format, ICA, and specparam-threshold
   decisions.
2. Only then run one epoch at 50 steps and execute the frozen calibration and
   determinism sensitivities; do not start the full recording or HPC
   array until each subsequent validation gate passes.
