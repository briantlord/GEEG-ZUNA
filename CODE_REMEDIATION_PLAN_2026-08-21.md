# GEEG-ZUNA Remediation Plan — Derived from the 2026-08-21 Code Audit

## Objective

Produce one authoritative, reproducible ZUNA 1.1 benchmark whose result bundle
can prove all expected recordings, masks, methods, metrics, failures, model
weights, code, inputs, reference frames, and reconstructions belong to the same
experiment.

This plan replaces the abandoned phase sequence. Its order comes from the
dependencies found in the scratch audit. No full HPC run should start until the
one-recording release gate near the end passes.

## Workstream 1 — Establish one authoritative project and stop unsafe entry points

### Changes

1. Initialize version control in a clean source-only tree and add a protective
   `.gitignore` for raw EEG, model weights, environments, caches, temporary FIFs,
   logs, and generated results.
2. Make one source tree authoritative. Generate the HPC share from that tree by
   a scripted release/export step; do not maintain two hand-edited copies.
3. Move legacy ZUNA 1.0 and exploratory code under a clearly named archive that
   is not importable from the project root. Remove the vendored `zuna/` directory
   from the active Python path.
4. Hard-block or remove stale executable validators, old aggregators, the legacy
   `pilot.py` CLI, obsolete repair scripts, and share `pipeline/` entry points.
5. Replace README/runbook commands with one local preflight command, one
   one-record gate command, one HPC submit command, and one verified collection
   command.
6. Assign a release ID containing the Git commit, configuration hash, and build
   timestamp. Put it in every manifest and output row.

### Exit criteria

- From any working directory, the resolved ZUNA module is the same pinned pip
  package under the declared environment.
- A command inventory test proves every non-authoritative executable either
  exits with a legacy error or lives outside the release.
- The exported share is reproducibly generated and its manifest hashes every
  shipped file.

## Workstream 2 — Freeze the scientific contract before changing implementation

### Decisions that must be written down

1. **Reference frame:** define truth and reconstruction values in the same frame.
   For each metric/drop set, compute both test-retest truth and reconstruction
   error using that drop set's leakage-safe surviving-channel reference. Do not
   compare dropout-frame error with a full-reference floor.
2. **M1/M2:** decide whether mastoids are model inputs, reference contributors,
   metric channels, or excluded channels. Encode one consistent choice.
3. **CNT format:** preserve the known `int32` choice only after attaching
   independent format evidence (acquisition specification, verified header/byte
   layout, or a documented comparison against the acquisition software).
4. **Artifact handling:** decide whether ocular ICA is required, forbidden, or a
   sensitivity analysis. Do the same for muscle ICA and rejected annotations.
5. **ZUNA physical scaling:** choose candidate deployable calibration strategies
   before looking at held-out target outcomes. At minimum compare:
   median-survivor carrier scale, spatial-neighbor scale, an observed-channel
   calibration learned without target samples, and scale-free outcomes.
6. **Coordinates:** decide a model-compatible transformation based on official
   ZUNA training coordinates or vendor guidance. Never silently clamp.
7. **Metric contract:** freeze exact bands, Welch parameters, CSD parameters,
   regional channels, log/ratio definitions, specparam settings, fit-quality
   requirements, and the statistical definition of the test-retest threshold.
8. **Failure policy:** define failed reconstruction, missing peak, missing
   recording, failed QC, and preempted job as explicit outcomes. They may not
   disappear from the cohort.

### Exit criteria

- A versioned protocol/config file contains every decision and hashes to a
  single experiment ID.
- An independent reviewer can determine the expected numeric estimand for each
  metric/drop set without reading Python source.
- Calibration and coordinate choices have written justification and planned
  sensitivity tests.

## Workstream 3 — Rebuild Stage-0 provenance and input validation

### Changes

1. Run channel/session QC on the actual cropped analysis interval, while also
   separately recording raw-file tail anomalies.
2. Add plausible absolute-scale and discontinuity checks so the int32 footer/
   tail problem is visible rather than embedded in misleading channel SDs.
3. Store the complete candidate-event table: raw and resampled sample, onset,
   event code, annotation overlap, non-overlap decision, amplitude/flat decision,
   and final selected epoch order.
4. Record channel type, inclusion role, montage source, original and transformed
   coordinates, and M1/M2 treatment.
5. Split corrected Stage-0 code from legacy pilot/reconstruction/metric code so
   its source hash represents only preprocessing behavior.
6. Add stale-lock recovery with explicit ownership/age checks and an inspection
   command; never delete unknown locks automatically.
7. Require Stage-0 manifests to pass a standalone verifier before any runner can
   use them.

### Tests

- Golden-recording test for the exact selected epoch IDs and tensor checksum.
- Raw-tail corruption test.
- CNT format/event-timing comparison test.
- Reference contributor test including the M1/M2 decision.
- Manifest tamper, stale lock, cache race, and package/source drift tests.

### Exit criteria

- A fresh Stage-0 build and a cache hit produce the same verified tensor and
  event/QC provenance.
- No manifest field describes data outside the actual retained analysis interval
  without being labeled as such.

## Workstream 4 — Make the ZUNA adapter spatially, numerically, and operationally sound

### API and identity

1. Require a verified Stage-0 object/manifest and exact experiment ID; do not
   accept a naked array and then assert corrected-v2 provenance.
2. Hash and store original positions, transformed model positions, coordinate
   bounds/type, discrete bins, mask, reference definition, calibration method,
   and Stage-0 cache key.
3. Reject any coordinate outside the model contract unless the frozen protocol
   defines and records an explicit transform. Add a test that every target and
   context channel preserves intended spatial distinctions.
4. Pin the model revision/hash in the actual load call, run model inference
   offline, and verify the loaded weight/config after inference.
5. Enforce distribution version, imported module path, package source hashes,
   Python, Torch, CUDA, driver, device identity, and platform in preflight and
   manifests.

### Reconstruction behavior

6. Read the hybrid reconstruction as the canonical output, or assert masked
   full/hybrid equality before accepting full output.
7. Keep direct held-target replacement and add an explicit test proving model
   input bytes and cache identity are invariant to held-target waveform changes.
8. Implement the frozen calibration alternatives behind explicit configuration.
   Store reconstructed normalized output separately from its physical-unit
   calibration so model behavior and scale estimation can be evaluated apart.
9. Add per-channel/epoch diagnostics for mean, SD, RMS, 1-45 Hz power, clipping,
   flatness, correlation, and spectral error. Diagnostics may label failures but
   must not remove expected units.
10. Make seed/determinism explicit. Repeat identical 50-step units and quantify
    exact/numerical repeatability; if stochastic variation remains, predeclare
    seeds or replicates.

### Recovery and concurrency

11. Persist outputs atomically per epoch or small batch with an expected-epoch
    manifest so a failure at epoch 63 can resume from completed verified epochs.
12. Add an atomic unit lock with run ID/host/PID/job metadata and safe stale-lock
    inspection.
13. Preserve all completed raw model artifacts; never delete verified partial
    output solely because the helper exits nonzero.

### Exit criteria

- Position changes necessarily change the cache key.
- No target waveform/statistic can affect serialized model input or calibration.
- Calibration sensitivity is quantified without target-informed fitting.
- Repeated 50-step inference has a documented reproducibility result.
- An injected failure resumes without recomputing verified epochs.

## Workstream 5 — Redesign result schema and validation around an expected-unit manifest

### Run manifest

Create an immutable manifest before execution containing:

- experiment/release/config/code hashes;
- exact 42 recording paths, sizes, and SHA-256 values;
- subject/day/rest identity and required pair structure;
- exact methods, metric versions, drop sets, submetrics, and expected row counts;
- expected reconstruction units and epoch counts;
- model revision/weight/config hash;
- calibration and coordinate strategy;
- required Stage-0 identities;
- permitted failure states.

For the current design, expected successful counts are:

- 42 recordings;
- 5 drop-set reconstruction units per method per recording;
- 14 truth submetric rows per recording in their declared drop-safe frames;
- 14 reconstruction submetric rows per method per recording;
- with spline and ZUNA: 1,764 result rows and 420 reconstruction-unit QC records.

### Schema changes

1. Truth rows must carry their metric's drop set and reference-frame ID.
2. Reconstruction rows must reference the exact matching truth-unit ID.
3. Add metric/common/runner/QC/schema source hashes and full metric configuration.
4. Add explicit unit status: success, model failure, QC failure, metric failure,
   missing input, preempted/incomplete.
5. Preserve unrounded values in machine results; round only display tables.
6. Give the CSV, QC data, run manifest, and all reconstruction manifests a
   checksummed bundle manifest.

### Validator requirements

The validator must fail unless:

- the run manifest and bundle checksums match;
- every expected recording, pair, method, reconstruction unit, metric, and
  submetric is present exactly once with an explicit terminal status;
- every successful row uses the declared drop set/reference frame and matches
  its truth unit;
- diagnostics are identical across rows sharing a reconstruction key;
- metric implementation/config hashes agree everywhere;
- all required test-retest floors are finite and have the predeclared number of
  pairs;
- no unknown/stale shard or mixed run ID is present;
- required ZUNA units are actual ZUNA units with verified model manifests.

### Exit criteria

- The validator rejects the current spline-only smoke, any missing shard, a
  missing ZUNA method, a wrong drop set, mismatched truth frames, duplicate QC,
  zero paired floors, and mixed code/model revisions.
- It reports `pass` only for a complete expected experiment bundle.

## Workstream 6 — Rebuild HPC execution around explicit resources and identities

### Changes

1. Create the exact 42-recording input manifest after upload completes and verify
   every raw hash on the HPC.
2. Map each array task to one explicit manifest row, not a stride over a mutable
   directory listing.
3. Use an explicit environment interpreter path. Inside the allocation, run a
   fail-closed preflight for Python/package/module paths, CUDA, assigned GPU,
   model hash/revision, writable directories, disk space, and input identity.
4. Preserve scheduler-provided `CUDA_VISIBLE_DEVICES`. Address the assigned GPU
   as local device 0 only inside the preserved scheduler mask.
5. Allocate a collision-free rendezvous method/port per job step; do not use a
   fixed port shared by all tasks on a node.
6. Preserve SLURM identifiers and record job ID, array task, host, GPU UUID,
   driver/runtime, interpreter, and environment lock hash.
7. Set Hugging Face offline mode for execution and make model download/verification
   a separate setup command.
8. Precreate the scheduler log directory before `sbatch` and use an absolute log
   path.
9. Give every shard the experiment ID and its expected recording ID. Write
   atomically and never append data from another run ID.
10. Replace wildcard `awk` collection with a validator-driven collector that
    requires exactly the manifest's shards and checks each bundle before merge.

### Exit criteria

- A two-task same-node smoke demonstrates distinct GPU assignment and no port
  collision.
- Re-submission of a completed task is an audited cache hit for the same explicit
  unit.
- Adding/removing/renaming a raw file cannot change any existing task mapping.

## Workstream 7 — Validation ladder before the full cohort

Run these gates in order. Stop on the first failure.

### Gate A — CPU/static contract

- all existing narrow tests plus new tests from Workstreams 1-5;
- code/import inventory, schema completeness, source hashes, and environment
  lock checks;
- synthetic metrics with known spectra/asymmetries and reference-frame tests.

### Gate B — Saved-artifact replay

- verify the existing one-step smoke without model execution;
- prove full/hybrid masked equality, blind-input invariance, units, masks,
  coordinates, and manifest hashes;
- confirm that the new validator correctly rejects it as incomplete.

### Gate C — One epoch, 50 steps

- one declared drop set, production settings, exact pinned model;
- repeat the identical unit to measure determinism;
- run calibration alternatives and store normalized plus physical outputs.

### Gate D — One complete 64-epoch recording, all five drop sets

- spline and ZUNA, all five metrics, all expected truth/reconstruction/QC units;
- inspect waveform, per-channel scale, spectra, metric fits, coordinates, masks,
  cache/resume, and failures;
- compare calibration sensitivity and do not choose a strategy using held-target
  outcomes unless it was predeclared as a sensitivity analysis.

Expected for one successful recording: 42 result rows and 10 reconstruction QC
units (five spline, five ZUNA).

### Gate E — Two-task HPC systems smoke

- two recordings concurrently on one eligible node if possible;
- prove interpreter, GPU binding, port isolation, offline weight identity,
  logging, task mapping, and partial recovery.

### Gate F — Paired mini-cohort

- at least one complete Rest1/Rest2 subject-day pair, preferably several pairs;
- require finite drop-frame-consistent floors and exercise the final aggregator.

### Release gate

Only after Gates A-F pass may the full 42-recording array be unblocked. Record
the approval by experiment ID and code commit; do not remove a generic `exit 2`
without binding the launcher to that approved release.

## Workstream 8 — Full run, collection, and reporting

1. Submit the immutable 42-unit array.
2. Monitor explicit terminal states; preemption/failure remains visible and is
   retried by unit ID.
3. Collect only after the expected-unit validator passes.
4. Generate aggregate tables with sample counts, failure counts, paired-floor
   counts, uncertainty, calibration sensitivity, and method comparisons.
5. Keep old ZUNA 1.0 and invalid broadband-v1 results clearly separated. Do not
   overwrite them or relabel them as corrected.
6. Update the report with the exact experiment ID, model revision/hash, code
   commit, protocol/config hash, environment lock, input manifest, and bundle
   checksum.
7. State the official limitation plainly: reconstructed channels are generated
   imputations, not recovered measurements or a basis for clinical decisions.

## Immediate next action

Do not run ZUNA yet. Start with Workstreams 1 and 2: establish one authoritative
source/release boundary and resolve the reference-frame, coordinate, scaling,
M1/M2, CNT-format evidence, artifact-cleaning, metric, and failure-policy
decisions. Those decisions determine the cache keys and result schema; coding
around them first would create another incompatible generation.

