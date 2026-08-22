# GEEG-ZUNA Code Audit Notebook — 2026-08-21

Status: **complete for the current directory snapshot**

## Audit rule

The repository is treated as untrusted. This audit derives behavior from the
code and saved artifacts as they exist now. Previous phase labels and previous
claims of completion are not accepted as evidence. No pipeline implementation,
HPC upload, or ZUNA inference is authorized during this audit.

## Evidence conventions

- `OBSERVED`: directly read from source, configuration, manifest, or artifact.
- `TESTED`: reproduced by a command or regression test during this audit.
- `INFERENCE`: conclusion supported by observed evidence but not executed end to end.
- `UNKNOWN`: cannot be established from the current directory.
- Severity: `BLOCKER`, `HIGH`, `MEDIUM`, `LOW`, or `INFORMATIONAL`.

## Initial inventory

`OBSERVED` — The workspace contains multiple generations of the project:

- root-level legacy pipeline scripts and large arrays;
- an active-looking `benchmark/` implementation;
- a second uploadable code tree at `GEEG-ZUNA-share/`;
- a vendored top-level `zuna/` package that can shadow pip-installed ZUNA;
- multiple Python environments (`.zuna11_local_env`, `zuna_env`, and a Linux backup);
- active `results/`, archived invalid results, corrected Stage-0 cache, and one
  corrected ZUNA reconstruction cache;
- multiple launchers and SLURM scripts from different generations.

`BLOCKER` — No single safe, production-ready entry point exists. The intended
corrected path can be identified, but its HPC launcher is deliberately blocked
and the validation/aggregation gates do not establish what their names claim.

## Evidence log

### E001 — Repository multiplicity

- Status: `OBSERVED`
- Evidence: root, `benchmark/`, `GEEG-ZUNA-share/`, and vendored `zuna/` all
  contain executable Python.
- Risk: imports and launch commands can resolve different implementations based
  on current working directory and `sys.path`.
- Disposition: resolved in E004, E015, E016, and the final report.

### E002 — Results multiplicity

- Status: `OBSERVED`
- Evidence: legacy CSVs remain in root, `benchmark/`, `results/`, and the share;
  invalid results are also quarantined under `archive/invalid_broadband_v1`.
- Risk: filenames alone do not reliably distinguish valid, invalid, diagnostic,
  and corrected results.
- Disposition: resolved in E005-E008, E018, E026, and E032.

### E003 — Source-copy comparison

- Status: `TESTED`
- Evidence: SHA-256 comparison of every non-archive file under `benchmark/`
  and `GEEG-ZUNA-share/benchmark/` found no differing files. The share omits
  several local-only artifacts and two local-only source files
  (`archive_invalid_broadband_v1.py`, `mne_source_method.py`), but the files
  present on both sides are byte-identical.
- Consequence: defects identified in the local corrected benchmark also exist
  in the already-uploaded share.

### E004 — Intended corrected execution path

- Status: `OBSERVED`
- Local path: `run_zuna11_local_one_record.ps1` ->
  `benchmark/_check_zuna11_local.py` -> `benchmark/metrics/run.py` ->
  `stage0_cache.load_or_create` / `pilot.preprocess` ->
  `pilot.surviving_average_reference` -> spline or
  `zuna_method_v11.zuna_reconstruct` -> `_recon11.py` -> pip ZUNA 1.1 ->
  reconstruction QC -> metric plug-ins -> v3 CSV and QC JSONL.
- HPC path: `benchmark/slurm_zuna11_metrics.sh` would call the same metric
  runner, but lines 35-36 unconditionally print `BLOCKED` and exit 2.
- Consequence: the full corrected ZUNA 1.1 metric run has not been run by the
  present HPC launcher.

### E005 — No corrected full-recording ZUNA result exists

- Severity: `BLOCKER`
- Status: `OBSERVED`
- Evidence: `results/zuna11_reconstructions_v2` contains one cache entry only:
  one epoch, two dropped channels (F3/F4), one diffusion step. Its manifest
  reports shape `[1, 62, 1280]`, `sample_steps: 1`, and `target_packed_seqlen:
  4000`.
- Evidence: the only v3 metric result is
  `results/phase3_spline_smoke_v3.csv`, containing spline and truth rows but no
  ZUNA rows.
- Consequence: neither the five-drop-set corrected run nor a complete
  64-epoch corrected ZUNA recording has been validated. Old full-recording ZUNA
  outputs are in the explicitly invalid broadband-v1 archive and cannot answer
  whether the corrected adapter works.

### E006 — The named ZUNA audit passes without ZUNA

- Severity: `BLOCKER`
- Status: `TESTED`
- Command: `.zuna11_local_env/Scripts/python.exe
  benchmark/audit_zuna11_pipeline.py --csv
  results/phase3_spline_smoke_v3.csv`
- Result: JSON `status: "pass"`, `methods: ["spline"]`, zero ZUNA requirement.
- Source cause: `audit_zuna11_pipeline.py` validates whichever reconstruction
  rows happen to exist. It never requires method `zuna`, a ZUNA manifest, a
  model revision, five ZUNA reconstruction units, a complete recording, or a
  complete cohort.
- Consequence: the prior `pass` was not evidence of ZUNA correctness.

### E007 — Aggregation accepts an incomplete experiment with no reliability floor

- Severity: `BLOCKER`
- Status: `TESTED`
- Command: `benchmark/metrics/aggregate.py --csv
  results/phase3_spline_smoke_v3.csv`.
- Result: it prints `Validated Phase 3` for one recording, reports `n=0 paired
  days`, and emits `nan` for all 14 floors without failing.
- Source cause: completeness is relative only to rows already present. There is
  no expected recording manifest, expected subject/session count, required
  method set, required ZUNA method, minimum paired-day count, or finite-floor
  gate. A wholly missing shard, subject, method, or cohort can therefore escape
  validation if its truth and reconstruction rows are absent together.
- Consequence: the current collector/aggregator cannot certify a production
  result.

### E008 — Reliability floors and reconstruction errors use different reference frames

- Severity: `BLOCKER`
- Status: `OBSERVED` and `TESTED`
- Source: `metrics/run.py` lines 218-252 writes floor truth from a full-montage
  average reference. Lines 263-269 compute reconstruction truth after an
  average reference over surviving channels only. The latter value is written
  only inside reconstruction rows. `metrics/aggregate.py` lines 138-147 builds
  floors exclusively from the full-reference truth rows.
- Real-artifact check: in the spline smoke, frontal-midline-theta full-reference
  truth is `0.500472` / `-0.109089`, while the reconstruction-row truth in the
  dropout-safe frame is `0.620900` / `0.090450`. Frame shifts are approximately
  `0.120` and `0.200`, respectively.
- Consequence: reconstruction error for reference-dependent metrics is compared
  against a test-retest floor for a different estimand. This invalidates the
  advertised `ok`/`OVER` comparison even when every waveform and metric
  calculation is internally correct.

### E009 — ZUNA coordinate clamping is real and unguarded

- Severity: `HIGH`
- Status: `TESTED`
- Evidence: cached Stage-0 standard-1005 coordinates range to z=`0.141549 m`,
  while the installed ZUNA 1.1 FIF config uses `chan_pos_xyz_extremes_type:
  twelves` (`[-0.12, 0.12]` on each axis). Nine coordinate elements exceed the
  model range: FCZ, C1, CZ, C2, CP1, CPZ, CP2, PZ, and P2.
- Installed ZUNA source warns and then clamps every out-of-bounds discrete
  coordinate to bin 0 or 99. The affected nine channels therefore lose vertical
  coordinate distinctions at the upper boundary. Several are targets or close
  context channels for the theta/beta, frontal-midline-theta, and mu drop sets.
- The adapter neither validates nor records this clipping. `metrics/run.py`
  globally suppresses warnings in its own process; helper warnings may reach a
  launcher log, but no coordinate audit is stored in the reconstruction
  manifest.

### E010 — Reconstruction cache identity omits electrode positions

- Severity: `HIGH`
- Status: `OBSERVED`
- Source: `zuna_method_v11._cache_location` hashes blind waveform bytes,
  channel names, dropped indices, settings, model identity, code hashes, and
  package versions. It receives no `pos` argument and cannot hash positions.
- Consequence: changing coordinate normalization, montage geometry, or any
  channel position while leaving waveforms/names unchanged reuses an old ZUNA
  reconstruction generated with different spatial tokens. The cached manifest
  also does not store the input coordinate array.

### E011 — Canonical ZUNA output selection is confusing but not the observed waveform bug

- Severity: `MEDIUM` (clarity/provenance), not presently a demonstrated numeric error
- Status: `OBSERVED` and `TESTED`
- Source: the adapter validates the mask in `model_output/hybrid` but loads
  waveforms from `model_output/full_reconstruction`, then hard-restores observed
  channels itself.
- Installed ZUNA 1.1 defines `full_reconstruction` as model output everywhere and
  `hybrid` as original on kept cells/model output on masked cells. Because this
  project masks complete channels and disables seam correction, the masked
  channel samples should be identical in both.
- Saved-artifact check: full and hybrid differ on observed samples by at most
  `0.398 uV`, but differ by exactly `0` on F3/F4, the masked channels. The saved
  reconstruction matches full-output F3/F4 to FIF precision and restores every
  observed channel bit-exactly.
- Disposition: use/read `hybrid` directly or assert full-vs-hybrid equality on
  the mask so the selected artifact and validated artifact are unambiguous.

### E012 — The only real corrected model smoke is integration-only

- Severity: `HIGH`
- Status: `TESTED`
- The one-epoch, one-step F3/F4 artifact is finite and internally consistent:
  hard inpainting passes; model input good channels match the dropout-safe
  reference to FIF precision; full and hybrid masked samples match; F3/F4
  correlations are about `0.697` and `0.611`; power ratios are about `0.68` and
  `0.62` by raw mean-square in this check.
- It does not test 50-step output, 64 epochs, four-channel/six-channel masks,
  all five drop sets, repeated determinism, or metric preservation. The report
  itself correctly says it is not a quality estimate.

### E013 — HPC GPU and rendezvous environment handling is unsafe

- Severity: `BLOCKER` for array release
- Status: `OBSERVED`; requires HPC execution to confirm site behavior
- `_recon11.py` overwrites `CUDA_VISIBLE_DEVICES` with the wrapper's default
  string `0`. Under SLURM this can discard the scheduler-provided GPU binding and
  select a GPU not assigned to the task.
- `_recon11.py` assigns every helper `MASTER_PORT=29500`. Multiple array tasks
  placed on the same node can collide on the same loopback rendezvous port.
- It deletes all `SLURM_*` variables, so job/task provenance is lost inside the
  model process and any package behavior relying on scheduler context is
  bypassed.
- No code records assigned GPU UUID/index, driver, CUDA runtime, host, SLURM job
  ID, or task ID in the reconstruction manifest.

### E014 — HPC sharding and collection have no corpus manifest

- Severity: `BLOCKER` for array release
- Status: `OBSERVED`
- The SLURM script hard-codes array `0-41` and passes the array task count to a
  stride over whatever sorted files exist when each task starts. It does not
  assert the exact 42 filenames or their hashes.
- If upload is incomplete, a file is renamed, or the corpus changes between
  retries, shard numbers can map to different recordings and append into stale
  shard CSVs.
- Collection is an `awk` concatenation over whatever `shard_*_v3.csv` files
  happen to exist. It does not require exactly 42 shards, unique shard IDs,
  matching run identities, or a complete expected recording set.
- The script's `mkdir -p logs` executes after SLURM has already tried to open the
  relative `#SBATCH --output=logs/...` path; the runbook does not itself create
  that directory.

### E015 — Package/import resolution is directory-dependent

- Severity: `HIGH`
- Status: `TESTED`
- From project root, `import zuna` resolves the vendored top-level package
  (`zuna/__init__.py`), which lacks `reconstruct_fif`. From `benchmark/`, the
  share root, and the share's `benchmark/`, it resolves pip ZUNA 1.1.3.
- `importlib.metadata.version("zuna")` reports `1.1.3` even when Python has
  actually imported the vendored module; metadata version checks alone cannot
  prove module identity.
- `_recon11.py` relies on script-directory ordering and retains any inherited
  `PYTHONPATH`. The HPC path has no equivalent of the local environment check
  and only asserts that the imported module has `reconstruct_fif`; it does not
  enforce distribution 1.1.3, module path under the active environment, model
  revision, or absence of the vendored package.

### E016 — Active documentation and executable files contradict the corrected path

- Severity: `HIGH`
- Status: `OBSERVED`
- `GEEG-ZUNA-share/README.md` says the shared results are ZUNA 1.0, advertises
  `benchmark/biomarker_eval.py` as the headline reproduction command, and says
  raw data are protected by a `.gitignore` that does not exist in this directory.
- `biomarker_eval.py` is now hard-blocked at its entry point. Its advertised
  command therefore cannot run.
- Root `main_pipeline.py`, root experimental scripts, the share's `pipeline/`,
  `_validate_zuna.py`, and `_diag_zuna.py` remain runnable legacy ZUNA 1.0 paths.
  `_validate_zuna11.py` is also stale: it loads the legacy EMG-disabled temp
  cache and expects debug keys (`Z`, `a`, `b`, `good`) no longer produced by the
  corrected adapter, so it can perform expensive inference and then fail.
- `slurm_zuna_array.sh` is blocked but contains a nonexistent
  `benchmark/run_units.py` path and obsolete 50-task instructions after its exit.
- Consequence: filenames and README instructions do not identify a safe path.

### E017 — The adapter can mislabel arbitrary arrays as corrected-v2

- Severity: `HIGH`
- Status: `OBSERVED`
- `zuna_method_v11._validate_inputs` checks shape, finiteness, positions, and
  dropped indices, but accepts no Stage-0 manifest or preprocessing identity.
  `_settings` then unconditionally tags the reconstruction with the corrected-v2
  protocol ID and preprocessing SHA.
- The stale `_validate_zuna11.py` demonstrates a real caller that supplies
  legacy `emg=False` data while invoking this adapter.
- Consequence: a cache manifest's protocol claim proves what the adapter declared,
  not how its input was produced.

### E018 — Aggregator/schema validation is incomplete even for present rows

- Severity: `HIGH`
- Status: `OBSERVED`
- It does not validate a metric's declared drop set, that reconstruction-row
  truth matches the floor truth (or a declared drop-frame truth), consistent
  metric implementation across truth/reconstruction rows, consistent diagnostics
  across submetrics sharing a reconstruction, expected submetric sets, expected
  reconstruction count, or required methods.
- `audit_zuna11_pipeline.py` overwrites duplicate QC entries in a dictionary and
  does not reject extra/missing method-specific experiment units.
- Consequence: syntactically valid but scientifically mismatched rows can pass.

### E019 — Current QC gates are safety rails, not quality validation

- Severity: `HIGH` if interpreted as model validation
- Status: `OBSERVED`
- A reconstruction passes with total held-out 1-45 Hz power anywhere from 5% to
  1000% of truth and max absolute amplitude up to 1000 uV. RMS ratio is logged
  but never gated. Correlation, spectral similarity, per-channel scale, DC
  offset, clipping, flatness, and boundary artifacts are not gated.
- The spline smoke passes, as expected. The one-step ZUNA smoke was never routed
  through a full five-unit v3 validation.
- Consequence: `gate_status=pass` means finite, unchanged observed channels, and
  very broadly plausible aggregate scale. It does not mean reconstruction or
  biomarker quality is sane.

### E020 — Test suite result and coverage gap

- Status: `TESTED`
- The 13 selected CPU regression tests pass.
- Missing tests include: position changes invalidate cache; no coordinate
  clipping; imported ZUNA module identity/version; required ZUNA rows; complete
  42-recording corpus/shards; finite nonempty floors; common truth/reference
  frame; full-vs-hybrid masked equality; 50-step determinism; SLURM GPU binding
  and port isolation; stale entry-point blocking; and end-to-end full-recording
  behavior.
- Consequence: 13/13 means only that the currently asserted narrow behavior is
  stable.

### E021 — Stage-0 strengths and unresolved preprocessing issues

- Status: `OBSERVED` and `TESTED`
- Strengths: Stage-0 is keyed by raw SHA-256, protocol hash, source hashes,
  package versions, epoch target/minimum, and EMG flag; tensor checksum and
  metadata are verified on load. The cached tensor is 64 x 62 x 1280 float32 uV
  and current source hashes still match its manifest.
- `int32` is frozen by code and prior project documents; MNE `auto` fails and
  `int16` produces a different event interpretation. Independent ground-truth
  confirmation of the hardware storage format is not present in the directory.
- The int32 continuous read contains extreme values near the final seconds (up
  to about 64 million uV). The 10-second edge crop removes that tail before ICA
  and epoching, but session QC is run before the crop and does not flag
  unreasonable absolute scale; it only checks nonfinite, flat, and repeated
  rail values.
- Channel QC manifest standard deviations (~0.84 million uV) therefore describe
  the corrupted continuous tail, not the retained physiological epochs. That
  makes the recorded QC misleading even though the cached epochs have
  physiological scale.
- The manifest records counts but not selected event sample/onset/code or the
  accepted/rejected epoch index list, limiting forensic traceability.
- EOG channels are dropped before ICA, so the pipeline performs muscle-component
  detection but no explicit EOG component detection/removal. Whether ocular
  cleaning is desired for these resting biomarkers is an unresolved protocol
  decision.
- M1/M2 are declared non-cortical and forbidden as drop targets, yet they remain
  in the 62-channel tensor and are included in the surviving-channel average
  reference. This is an unresolved reference-definition inconsistency.

### E022 — Comparator behavior

- Status: `OBSERVED`
- The implementation named `linear` fits each held-out target waveform using
  the target's own samples and is therefore an oracle. The v3 schema correctly
  forbids the label `linear` and exposes it only as `linear_oracle`; the current
  local/HPC corrected launchers request only spline and ZUNA.
- Legacy runners still call `linear` without the v3 semantic guard and old
  reports/README compare against it as though it were an ordinary baseline.
- Spline uses standard-1005 positions reconstructed from channel names and
  ignores the `pos` array argument. This matches the current Stage-0 montage but
  would diverge from ZUNA if custom positions were introduced.

### E023 — Requirements/environment are not fully reproducible

- Severity: `MEDIUM`
- Status: `OBSERVED`
- Scientific packages and ZUNA are pinned, but Torch and Matplotlib use lower
  bounds. GPU wheel index, CUDA runtime, driver, Python patch version, and
  platform lock are not captured in a lock file.
- The local validator requires exactly Torch `2.6.0+cu124`, CUDA, ZUNA
  distribution 1.1.3, and specparam 2.0.0rc7. The HPC runbook installs
  `torch>=2.5` and then a requirements file requiring `torch>=2.6`, and its
  one-line check prints `zuna.__version__` (`0.1.1` in the installed package)
  rather than distribution version `1.1.3`.
- Local and HPC environment validation are therefore not equivalent.

### E024 — Physical-unit reconstruction depends on an unvalidated calibration heuristic

- Severity: `BLOCKER` for amplitude-dependent biomarker claims
- Status: `OBSERVED`; scientific validity remains `UNKNOWN`
- Installed ZUNA 1.1 normalizes each channel within each segment and later
  inverse-transforms model output using that channel's pre-mask mean and standard
  deviation. The official direct-FIF path computes those statistics before
  applying the bad-channel mask. In a held-out benchmark, leaving the true
  target waveform in the FIF would therefore leak its mean and scale.
- The adapter correctly prevents that leakage by replacing every target with a
  zero-mean deterministic carrier. It sets each carrier's standard deviation to
  the median standard deviation of surviving channels for that epoch. ZUNA then
  inverse-scales target output with this constructed mean/scale.
- This is a project-created estimator, not a demonstrated property of ZUNA. It
  forces a shared scale prior across all dropped channels in an epoch and a zero
  mean. Absolute power, asymmetry, specparam offset/peak power, and regional
  power ratios can therefore reflect the calibration heuristic as well as the
  model.
- No sensitivity analysis compares deployable alternatives (median/neighbor
  scale, observed-channel model calibration, scale-free scoring), and no 50-step
  full-recording result establishes robustness. The manifest records the chosen
  heuristic but the benchmark currently treats its output as if it were a
  model-only result.

### E025 — Failure recovery is group-level, not epoch-level

- Severity: `HIGH`
- Status: `OBSERVED`
- Each drop set launches one helper over all 64 epoch FIFs. Input/output are held
  in temporary directories and moved into the persistent cache only after the
  helper exits successfully.
- If the model process fails after producing some epochs, `finally` deletes the
  temporary input/output directories. Those completed epochs cannot be resumed.
- Once an entire drop-set helper succeeds, its model inputs, full outputs,
  hybrid outputs, masks, and reconstruction are persistent and recoverable.
  Thus a five-drop-set run can reuse completed drop sets, but cannot recover a
  partially completed drop set.
- There is no reconstruction-cache lock. Concurrent retries of the same unit can
  duplicate the expensive inference and one will fail when both attempt to
  create the same cache directory.

### E026 — Truth-based QC can create selection bias unless failures are first-class results

- Severity: `HIGH`
- Status: `OBSERVED`
- Reconstruction QC compares held-out reconstruction power with held-out truth
  and only allows metric rows after the gate passes. A ZUNA gate failure aborts
  the runner, leaving a partial CSV; no explicit failed reconstruction row enters
  the result schema.
- Using truth for post-run diagnostics is valid, but using it as an inclusion
  gate without counting the failed unit in aggregation can select only favorable
  model outputs. The incomplete-cohort weakness in E007 makes that possible.
- A production benchmark needs an immutable expected-unit manifest and explicit
  failure accounting; quality diagnostics should not make failed units vanish.

### E027 — Warning suppression hides scientific and numerical diagnostics

- Severity: `MEDIUM`
- Status: `OBSERVED`
- `metrics/run.py` executes `warnings.simplefilter("ignore")` globally before
  preprocessing, spline interpolation, metric computation, and parent-process
  adapter work. Legacy runners do the same.
- This can hide filter-length, montage, numerical, spectral-fit, deprecation,
  and data-quality warnings. The ZUNA helper is a separate process, so not every
  warning is necessarily suppressed there, but warnings are not promoted into
  manifests or structured QC.

### E028 — Local resumability contradicts runner resumability

- Severity: `MEDIUM`
- Status: `OBSERVED`
- `metrics/run.py` is written to append and resume missing result keys. The local
  PowerShell launcher refuses to start whenever its output CSV already exists,
  stating that a partial file is not safely resumable.
- Persistent corrected reconstruction caches still prevent rerunning a fully
  completed drop set, but a user must manually move/rename a partial result and
  loses straightforward row-level resume. This contradiction is a likely source
  of operational confusion.

### E029 — Five drop sets mean five separate reconstructions

- Status: `OBSERVED`
- The five implemented metrics currently declare five distinct drop sets, so the
  runner performs five spline and five ZUNA reconstruction units per recording.
  It does not reconstruct separately for every submetric; each reconstruction is
  reused by all submetrics belonging to its exact drop set.
- Separate model inference is scientifically necessary when masks and the
  leakage-safe surviving-channel reference differ. It is not necessary to load
  the model from scratch five times, but the subprocess design currently does
  so, adding overhead.

### E030 — Leakage boundary is mostly enforced in the current adapter

- Status: `OBSERVED` and `TESTED`
- The runner passes the full evaluation tensor into the adapter, but the adapter
  overwrites held-out waveforms before serialization, derives carrier scale only
  from surviving channels, hashes the blind tensor, and hard-restores only
  observed channels after inference.
- The unit test that changes only held-out truth by an extreme amount returns the
  same cached reconstruction, and the saved model-input FIF contains carrier
  waveforms rather than target truth. This supports no waveform/statistic leakage
  into the model path for finite Stage-0 inputs.
- The API boundary remains unnecessarily permissive (E017), and the
  physical-unit calibration problem remains (E024), but no direct held-out
  waveform leak was found in `zuna_method_v11.py`.

### E031 — Syntax checks

- Status: `TESTED`
- AST parsing succeeded for 71 project Python files in the inspected active
  trees. Both project PowerShell launchers parse without syntax errors. The four
  active/share SLURM shell files use LF line endings with no CRLF issue.
- This establishes parseability only, not runtime or scientific correctness.

### E032 — Metric/result provenance is not content-addressed

- Severity: `HIGH`
- Status: `OBSERVED`
- Stage-0 and ZUNA reconstruction caches hash their source code, but v3 result
  rows contain no hash of `metrics/run.py`, `metrics/common.py`, the metric
  plug-in, `reconstruction_qc.py`, `schema_v3.py`, or aggregation code.
- Four metrics write only the generic implementation label `project-native`.
  Changing a formula or default without manually changing the schema string can
  produce incompatible rows that the aggregator treats as the same experiment.
- The CSV/QC pair has no run manifest or checksums and is not linked to an
  immutable expected-unit manifest. Stage-0 cache keys are present, but the
  aggregator does not locate or verify their manifests.
- The specparam implementation does use the pinned official package with linear
  frequency/power inputs and returns the documented fixed aperiodic and Gaussian
  peak parameters. It does not save fit error/goodness-of-fit or a fitted-model
  artifact, so a finite but poor fit passes as a valid biomarker value.

### E033 — HPC model resolution is online and not pinned at execution

- Severity: `BLOCKER` for reproducibility
- Status: `OBSERVED`
- The local launcher sets Hugging Face offline mode. The HPC launcher does not.
- Before inference, the adapter records the current local `refs/main`, config,
  and weight hash. The helper then invokes ZUNA's hard-coded
  `Zyphra/ZUNA1.1` loader without passing that recorded revision or local weight
  path.
- If upstream `main` changes or the hub refreshes during execution, the helper
  can use a different revision from the one recorded before it ran. No
  post-inference model hash check detects that race.
- Production must pin the exact revision/hash in the actual load operation and
  run offline after a separate verified download step.

### E034 — The HPC batch job does not activate a known environment

- Severity: `BLOCKER`
- Status: `OBSERVED`
- Every environment/module/conda command in `slurm_zuna11_metrics.sh` is
  commented. The executable line is plain `python benchmark/metrics/run.py`.
- Therefore the batch job relies on whatever `python` happens to be inherited
  from the submission shell. The original failure in this project came from a
  Python 3.9/user-site ZUNA installation, and the script contains no preflight
  equivalent to the local validator.
- The script must resolve an explicit interpreter and verify Python, ZUNA module
  path/distribution, Torch/CUDA, specparam, model revision/hash, data corpus, and
  write locations inside the allocation before starting any model unit.

### E035 — Official-model compatibility that is established vs not established

- Status: `OBSERVED` against installed ZUNA 1.1.3 source and official release
  documentation.
- Supported: 256 Hz; 5-second windows are within the trained 0.5-30 second
  range; whole-channel masks are a trained reconstruction pattern; arbitrary
  montage coordinates are the intended spatial input; 50 Euler steps is the
  published inference setting; 0.5-45 Hz input is within the training filter
  variants.
- Correct adapter choices: no second filtering after continuous preprocessing;
  no second average reference; per-real-epoch serialization avoids fake temporal
  joins; exact whole-channel output masks are checked.
- Not established: handling of the nine clipped spatial coordinates; the
  carrier-based inverse-scale heuristic; equivalence of 5-second independent
  contexts to the desired metric use case; 50-step reproducibility on this GPU/
  HPC stack; or biomarker validity of generated/imputed channels. The official
  project explicitly describes outputs as plausible imputation rather than
  ground-truth measurement.

### E036 — No version-control boundary or automated release process

- Severity: `HIGH`
- Status: `TESTED`
- `git status` reports that this directory is not a Git repository. There is no
  `.gitignore`, despite the share README saying raw EEG is excluded by one.
- The local and uploadable benchmark copies happen to be byte-identical now, but
  synchronization is manual and there is no commit ID, tag, CI job, signed
  manifest, or build step defining a release.
- Consequence: code/results cannot cite an immutable project revision, source
  drift cannot be reviewed normally, and accidental inclusion of raw human EEG
  or large caches is not prevented by repository rules.

## Questions to resolve from code

1. Which command actually preprocesses raw CNT data for the intended benchmark?
2. Which ZUNA package is imported by every launcher and helper?
3. Does any reconstruction function receive or use held-out samples or statistics?
4. What exact reference, filtering, normalization, epoch boundary, and unit
   transformations occur, and in what order?
5. Which ZUNA output file is treated as canonical?
6. Which comparators are scientifically deployable versus oracles?
7. Are all metrics what their labels claim to be?
8. Can invalid, partial, or stale results be resumed or aggregated silently?
9. Are caches content-addressed by every behavior-changing input?
10. Can the current HPC share run safely without relying on shell state or
    resolving the wrong environment/package?

## Audit closure

All ten opening questions were traced to source and artifacts. Findings E001-E036
are consolidated in `CODE_AUDIT_REPORT_2026-08-21.md`. The replacement work
sequence and explicit release gates are in
`CODE_REMEDIATION_PLAN_2026-08-21.md`.

No project implementation, launcher, cached tensor, model output, result CSV,
archive, environment, or HPC file was changed by this audit. The only writes are
the three audit documents.
