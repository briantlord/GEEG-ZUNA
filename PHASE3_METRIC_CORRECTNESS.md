# Phase 3 — Metric and comparator correctness

Status: **complete locally (2026-08-21)**. The production SLURM array remains
blocked for the Phase 4 execution/HPC gate.

## Corrections made

1. `specparam_peaks` now uses the official `specparam==2.0.0rc7`
   `SpectralModel`. The former dependency-free approximation was removed from
   the active code. Spectra are passed in linear units, fit from 2–40 Hz with a
   fixed aperiodic component and Gaussian periodic peaks, and the strongest
   detected 7–14 Hz peak is reported.
2. `linear` is forbidden. The in-sample ridge implementation can be invoked
   only as `linear_oracle`, is written with `comparator_class=oracle`, and is
   excluded from the main aggregate table unless `--include-oracles` is given.
3. Metric plug-ins must return exactly their declared submetrics and every value
   must be finite. Declared drop sets must be complete; M1/M2 are forbidden.
4. Every reconstruction is checked before biomarker scoring:
   - tensor shape and all values finite;
   - observed channels preserved bit-exactly;
   - total held-out 1–45 Hz power ratio in `[0.05, 10.0]`;
   - held-out maximum absolute amplitude no greater than `1000 µV`.

   These are validation gates only. Truth-derived scale is never fed back into
   reconstruction or used for amplitude calibration. Full diagnostics are
   appended to `<metrics.csv>.reconstruction_qc.jsonl`.
5. Result rows use the exact `geeg-zuna-metrics-v3` schema. They include the
   corrected-v2 preprocessing hash, Stage-0 cache identity, reconstruction
   identity, comparator class, gate diagnostics, and metric implementation.
   The aggregator rejects legacy headers/rows, duplicate rows, partial methods,
   failed gates, incorrect error arithmetic, and any protocol/hash mismatch.

## Validation

- Regression suite: 13/13 Phase 1–3 tests passed.
- Real corrected-v2 spline smoke: `G001Day1Rest1.cnt`, all five metric drop
  sets, 28 rows (14 truth + 14 reconstruction), all reconstruction gates pass.
- Spline 1–45 Hz power ratios by drop set were 0.627–1.287; held-out maximum
  amplitudes were 17.4–48.7 µV.
- Real Phase 2 ZUNA 1.1 one-epoch F3/F4 cache passes the new gate: power ratio
  0.646, RMS ratio 0.802, maximum amplitude 9.26 µV.
- Synthetic 10 Hz data verify that the official package detects alpha at
  10.003 Hz and emits finite aperiodic/periodic parameters.

Smoke artifacts:

- `results/phase3_spline_smoke_v3.csv`
- `results/phase3_spline_smoke_v3.csv.reconstruction_qc.jsonl`

## Phase 4 gate

Do not remove the intentional stop at the top of
`benchmark/slurm_zuna11_metrics.sh` yet. Phase 4 must validate the configured
HPC environment, one full 64-epoch ZUNA recording across all five drop sets,
shard merging, and restart behavior before the production array is enabled.
