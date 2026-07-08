# benchmark/ — Preprocessing Method B (surviving-channel average reference, 0.5 Hz high-pass)

The **current evaluation harness** and the second of the two preprocessing methods. Implements
`../BENCHMARK_PROTOCOL.md`. Runs the full ladder on CPU (every method except ZUNA) and drives the
ZUNA rung on GPU. **The GPU run is complete** — see `../REPORT.md` and `../results/`.

## Preprocessing choices (Method B)
- **Filter:** 0.5 Hz high-pass, **no low-pass** (feed ZUNA broadband, matching training), notch
  60/120/180 Hz, optional ICA muscle removal.
- **Reference:** **surviving-channel average reference** — mean over *surviving* channels only, applied
  **after** dropout (so the dropped channel never enters its own reference; avoids the
  average-reference leakage that makes reconstruction trivially easy). All methods, incl. ZUNA,
  reconstruct and are scored in this frame.
- **Baseline to beat:** **K=8 nearest-neighbour linear ridge regression** (not spline — spline
  underperforms linear on this dense 62-ch montage). Scoring is in-band ≤45 Hz.

## Files
- `pilot.py` — Stage 0 preprocess, seeded dropout (scattered/contiguous), the reconstruction
  ladder (zero/mean/nearest/linear/spline), fidelity + biomarker scoring, idempotent JSONL queue.
- `biomarker_eval.py` — **Evaluation A**: drop each biomarker's own channels (FAA→F3/F4/F7/F8;
  IAF→O1/O2/Oz/POz/PO3/PO4), reconstruct, recompute IAF / FAA(CSD) / posterior-α / 1-f-slope,
  log `|recon − truth|`. Resumable by recording.
- `zuna_method.py` — ZUNA wrapper: builds the `.pt` (z-score preserved channels, scale positions
  into ±0.12, filename the dataloader parses), calls `zuna.inference(data_norm=10, 50 steps)`,
  then **self-calibrates** the output back to microvolts against the observed channels.
- `aggregate.py` — computes the **same-day (Rest1↔Rest2) test-retest floor** per biomarker and the
  per-method error, and prints the pass/fail table.
- `_validate_zuna.py`, `_diag_zuna.py` — single-recording validation & diagnostics of the wrapper.
- `slurm_zuna_array.sh` — SLURM job-array template for scale-out on HPC.

## Reproduce the headline result
```bash
# GPU + zuna package + HF weights required for the 'zuna' method; linear/spline run on CPU.
python biomarker_eval.py --subjects G001 G002 G003 G004 G005 \
                         --methods linear spline zuna --out ../results/zuna_eval_5subj.csv
python aggregate.py --csv ../results/zuna_eval_5subj.csv
```

## Wrapper validation (before trusting numbers)
`_validate_zuna.py` / `_diag_zuna.py` confirmed the wrapper is correct: on the observed
(non-dropped) channels the raw ZUNA output correlates with truth at mean r ≈ +0.92 (median
+1.00), and the self-calibration to µV is a clean linear fit (R² ≈ 0.955). This rules out a
channel-permutation/scale bug and means the reconstruction quality reported is ZUNA's own.
