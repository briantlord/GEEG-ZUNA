# Primary Stage 0 — minimal preprocessing contract

Status: **FROZEN for implementation and CPU validation**

Protocol ID: `geeg-zuna-minimal-stage0-v1`

Adopted: 2026-08-21

This amendment supersedes every earlier requirement for ICA, ocular-component
removal, muscle-component removal, or amplitude-selected “clean” truth epochs.
Those operations are not required by ZUNA and must not define the primary
benchmark ground truth.

## Primary preprocessing contract

1. Read Neuroscan CNT with `data_format="int32"`; never silently fall back to
   `int16` or `auto`.
2. Assign HEOG/VEOG/EKG as auxiliary channels. Remove unpositioned CB1/CB2 and
   require exactly 62 positioned EEG channels in `standard_1005`.
3. Apply exactly one zero-phase 0.5–45 Hz FIR bandpass. No separate notch is
   applied because 60 Hz is outside the retained band.
4. Resample to exactly 256 Hz and crop 10 seconds from each continuous edge.
5. Perform **no ICA, no component fitting, and no component subtraction**.
   Store descriptive auxiliary-channel QC, then exclude those channels before
   forming the EEG truth tensor. Auxiliary values never select epochs.
6. Greedily identify non-overlapping, marker-locked 5-second epochs. Intervals
   explicitly marked bad by the source annotations are excluded.
7. Record flatness and >300 µV peak-to-peak measurements, but do not use them to
   classify channels, reject epochs, or reject recordings. Use the available
   source-valid epochs up to the maximum requested by the run, and record the
   requested, available, and selected counts.
8. Keep Stage 0 unreferenced. After a drop set is declared, truth and every
   reconstruction are placed in the same reference frame by subtracting the
   average of the observed non-mastoid channels. Held-out channels never
   contribute to that reference.
9. Score the same 1–45 Hz signal supplied to every method.

## What counts as QC rather than cleaning

The pipeline may report a genuine processing error for failed load, wrong
channel count, unresolved positions, non-finite samples, or no usable marker
epochs. Continuous-channel standard deviation and railing fraction are recorded
without excluding a finite readable recording. Large continuous
excursions and sample jumps are recorded as warnings because they may occur
outside the selected marker epochs; they do not reject an otherwise structurally
valid recording. The pipeline may not improve a passing waveform by
automatically removing estimated physiological or artifactual components.

Per-channel and per-epoch amplitude measurements are descriptive observations.
The primary path contains no numerical amplitude threshold that classifies a
channel or excludes an otherwise readable recording. No epoch is deleted and no
channel is repaired based on these measurements.

EOG and amplitude diagnostics may later support transparent stratified reports.
An ICA-cleaned sensitivity analysis, if ever wanted, must use a separate
protocol ID, experiment ID, and Stage-0 cache. It cannot replace or contaminate
the primary no-ICA benchmark.

## Identity and acceptance gates

- Stage-0 cache schema is `geeg-zuna-stage0-cache-v4` and its default directory
  is `stage0_cache_v4`; v3 ICA-cleaned entries are incompatible by design.
- The manifest records `component_removal="none"`, `ica_applied=false`, every
  amplitude flag, and the exact epoch-selection policy.
- Raw content, scientific contract, preprocessing specification, source files,
  package versions and requested maximum epoch count all
  participate in cache identity.
- A real-recording CPU build must produce finite float32 data at 256 Hz with 62
  channels and 1280 samples per epoch, and must pass the standalone verifier.
- This phase runs no ZUNA inference.
