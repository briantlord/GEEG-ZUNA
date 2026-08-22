# Phase 2 — masking-aware ZUNA 1.1 integration

Status: **implemented and real-model smoke validated**  
Adapter ID: `zuna11-masking-aware-per-epoch-v2`

Phase 2 replaces the invalid broadband/concatenated FIF wrapper. It consumes only
the corrected `geeg-zuna-corrected-v2` Stage-0 tensor created in Phase 1.

## Blind-channel contract

For each drop set and each real 5-second epoch:

1. Compute the evaluation reference over surviving channels before calling ZUNA.
2. Preserve surviving data exactly.
3. Discard every held-out sample before serialization.
4. Estimate one blind epoch scale as the median temporal standard deviation of
   surviving channels.
5. Replace each held-out waveform with a deterministic zero-mean, unit-SD
   calibration carrier multiplied by that blind scale.
6. Mark exactly the requested held-out channels bad for the entire epoch.

The carrier is masked by ZUNA and does not condition reconstruction. Its only
purpose is to give the released loader a blind inverse-z-score scale. Neither the
held-out mean nor held-out standard deviation is available to the loader, model,
cache key, or inverse transform. Changing only held-out truth therefore leaves the
model input and result unchanged.

## Boundary and preprocessing contract

- One real epoch is written to one FIF; epochs are never concatenated.
- ZUNA receives `v4_segment_sec=5.0`.
- ZUNA highpass, lowpass, notch, and average-reference operations are disabled.
- The corrected 0.5–45 Hz Phase 1 signal is not filtered again.
- ZUNA's official per-segment/channel normalization and `data_norm=10` remain in
  place.
- Full output and official masks are read per epoch. The mask must be true for
  every held-out sample and false for every surviving sample.
- Surviving channels are hard-inpainted from the Phase 1 tensor bit-for-bit.
- The former observed-channel affine self-calibration has been removed.

## Cache and provenance

Corrected reconstruction caches use `results/zuna11_reconstructions_v2` and are
keyed by:

- blind-input bytes (held-out truth cannot affect the key);
- channels and exact drop set;
- preprocessing protocol digest;
- adapter settings, diffusion steps, and seed;
- ZUNA package, Torch, MNE, NumPy, and SciPy versions;
- adapter/helper/ZUNA loader/config source hashes;
- Hugging Face revision, model-config hash, weight SHA-256, and weight size.

Each entry retains the per-epoch blind input FIFs, full/hybrid model output,
official masks, reconstruction NPZ, checksums, and a machine-readable manifest.
Cache hits verify the reconstruction checksum and hard-inpainting integrity.

## Acceptance evidence

- Four no-GPU model-adapter tests pass.
- A 100,000× change to held-out truth produces a verified cache hit and an
  identical reconstruction.
- A damaged reconstruction cache is rejected.
- A real RTX 3080 Ti smoke ran one epoch, F3/F4, one diffusion step using the
  genuine ZUNA1.1 weights. It completed in 27 seconds with finite output and
  observed channels exactly preserved.
- Real-smoke reconstructed SD was 2.705/3.020 µV versus a blind scale of
  3.039 µV; 1–45 Hz power ratios to truth were 0.761/0.567. The catastrophic
  10–71× power inflation from the invalid run was absent.

## Remaining limitations and gates

- The blind median-survivor scale is an explicit benchmark estimator because an
  entirely missing channel has no observable absolute amplitude. It requires a
  later sensitivity analysis against other truth-free estimators.
- ZUNA's fixed ±0.12 m position cube still clamps nine standard_1005 electrodes;
  this is an upstream model/config issue, not changed here.
- The one-step smoke validates integration and scale, not reconstruction quality.
  Scientific runs retain 50 diffusion steps.
- The production SLURM array remains blocked until later phases finish metric,
  comparator, output-gate, and HPC-runner corrections.

