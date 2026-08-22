# Phase 1 — corrected benchmark contract

Status: **FROZEN for implementation and smoke validation**  
Protocol ID: `geeg-zuna-corrected-v2`

This amendment supersedes the broadband/no-lowpass decisions in
`BENCHMARK_PROTOCOL.md`. The first local ZUNA 1.1 run demonstrated that the
124–128 Hz component in this corpus dominates ZUNA's per-channel scale and makes
the reconstruction scientifically uninterpretable.

## Corrected preprocessing contract

1. Read Neuroscan CNT with `data_format="int32"`; never silently fall back to
   `int16`.
2. Remove HEOG, VEOG, EKG, and the unpositioned CB1/CB2 channels; require the
   remaining 62 EEG channels to resolve in the `standard_1005` montage.
3. Apply exactly one zero-phase 0.5–45 Hz FIR bandpass before resampling. No
   separate notch is needed because 60 Hz and its harmonics are outside the
   retained band.
4. Resample to exactly 256 Hz and crop 10 seconds from each continuous edge.
5. Run deterministic ICA muscle cleaning. Production preprocessing fails closed
   if this step cannot run; it may not silently continue without EMG cleaning.
6. Select non-overlapping, marker-locked 5-second epochs. Reject annotated,
   flat, or >300 µV peak-to-peak epochs. Retain up to 64 and exclude a recording
   if fewer than 48 clean epochs remain.
7. Stage-0 truth remains unreferenced. Once a drop set is known, subtract the
   average of surviving channels from every channel. Held-out channels never
   contribute to this reference.
8. Score the same 1–45 Hz signal supplied to every method. High-gamma above
   45 Hz is removed from this benchmark.

## Evaluation guardrails

- The in-sample linear ridge method is an oracle/ceiling, not a baseline. It
  cannot appear as an ordinary production comparator.
- ZUNA may not calculate a held-out channel's normalization from its true data.
- Artificially concatenated epochs may not be filtered again as continuous data.
- Corrected caches and results use the v2 protocol ID and content hashes; v1
  artifacts are permanently marked invalid and cannot be resumed.
- The custom spectral peak heuristic must be labeled `specparam-style` until it
  is replaced by the actual `specparam` package.

## Phase 1 acceptance gates

- The preprocessing specification and digest are written to every Stage-0
  manifest.
- Raw-file content, preprocessing specification, source code, and package
  versions participate in the cache key.
- A cache load verifies tensor and manifest integrity before returning data.
- A real-recording CPU smoke test must show 256 Hz, 62 channels, at least 48
  clean epochs, finite float32 data, and no pathological 124–128 Hz content.
- Phase 1 does not run ZUNA. Masking-aware ZUNA integration is a later phase.
