# GEEG-ZUNA remediation progress — 2026-08-21

This ledger describes the current uncommitted working tree. No ZUNA inference
was run during this remediation.

## Current verdict

The active Stage-0 path is minimally processed, no-ICA EEG. Epoch count,
amplitude burden, and specparam fit statistics are descriptive observations;
they do not classify channels or exclude otherwise readable recordings.

The earlier 31/42 “eligible cohort” conclusion is withdrawn. It resulted from
two unjustified exclusions: requiring all recordings to supply 64 epochs and
rejecting a complete recording when one channel crossed a post-hoc amplitude
frequency threshold. The earlier cohort output remains historical diagnostic
data only and must not be used as an eligibility list.

## Active Stage-0 behavior

- Protocol ID: `geeg-zuna-minimal-stage0-v1`.
- Read CNT as int32, map the positioned 62-channel montage, apply one 0.5–45 Hz
  zero-phase FIR, resample to 256 Hz, crop continuous edges, and construct
  non-overlapping marker-locked 5-second epochs.
- Perform no ICA, component fitting, component subtraction, channel
  interpolation, or amplitude-based waveform cleaning.
- HEOG, VEOG, and EKG are descriptive auxiliary channels only and are excluded
  before the EEG tensor is formed.
- Use available source-annotation-accepted epochs up to the maximum requested by
  the run. Record requested, available, and selected counts.
- Record per-epoch and per-channel flatness and >300 µV peak-to-peak
  measurements without classifying channels, rejecting epochs, or rejecting
  recordings.
- Record continuous-channel standard deviation, railing fraction, maximum
  amplitude, and maximum sample jump without quality-based exclusion; nonfinite
  samples remain a genuine processing error.
- Persist specparam fit status, R², mean absolute error, posterior-channel count,
  detected-peak count, and alpha-peak count without an acceptance threshold.
- Keep Stage 0 unreferenced. Apply the surviving non-mastoid average reference
  only after a drop set is declared, identically to truth and reconstruction.

## Fresh G001 verification

The current release-identity Stage-0 entry is:

`results/stage0_cache_v4/G001Day1Rest1__b5c248899ca549004a01`

- cache key:
  `b5c248899ca549004a011d62d556241f166e804e62ce50f5a08f0b5f069fb668`
- 76 source-valid epochs available;
- requested maximum 64;
- selected 64;
- tensor shape 64 × 62 × 1280, float32, at 256 Hz;
- `ica_applied=false`;
- standalone cache verification passed;
- all full-recording and 8-epoch-block truth metrics finite; and
- ZUNA inference run: false.

Truth-only evidence is under:

`results/truth_qc_v4_G001Day1Rest1_no_exclusion_gates_final`

## ZUNA adapter safeguards retained

- The adapter requires a typed, verified Stage-0 object.
- Held-target waveforms, means, and standard deviations do not enter model input
  construction or physical-scale strategy selection.
- Original coordinates, official clipped coordinates, discrete tokens, masks,
  normalized outputs, and physical outputs are recorded and content-addressed.
- Verified per-epoch reconstructions are preserved; retries submit only missing
  or invalid epochs.
- Observed-channel restoration, masking, boundary checks, cache-tamper
  rejection, hidden-waveform invariance, coordinate identity, and interruption
  recovery have regression coverage.
- Every expected result unit must receive an explicit terminal state; failures
  cannot silently disappear during collection.

## Verification status

- 27/27 authoritative CPU regression tests pass.
- The same 27/27 tests pass from the generated HPC share.
- JSON configuration validation passes.
- Edited Python source compiles.
- Fresh G001 Stage-0 standalone verification passes.
- Fresh G001 truth-only metric QC passes.
- Both generated shares pass exact verification: 68 files, content hash
  `6ff12a634e5b4577eeba925b21d6e30a492ba3bc9ec47928e1c2c046cc84d1dd`,
  source identity `UNCOMMITTED@a55931837c968b56fe0349e98b48f35b1dcf43ce`.
- No ZUNA inference was run.

The current `GEEG-ZUNA-share` and `dist/GEEG-ZUNA-share` directories are
byte-identical corrected builds. Previous shares containing withdrawn gates
have been superseded.

## Remaining controlled work

1. Inspect the final diff and confirm no withdrawn gate remains active.
2. Do not commit or push without explicit authorization.
3. Do not run ZUNA until explicitly requested after this correction is reviewed.

Independent CNT acquisition/export evidence and production 50-step ZUNA
physical-scale sensitivity/determinism evidence remain unresolved. They are not
replaced by automatic recording exclusions.
