# pipeline/ — Preprocessing Method A (average-reference, 1–100 Hz)

This is the **original proof-of-concept pipeline** and the first of the **two preprocessing
methods** in this repository. It was written to match ZUNA's stated training distribution as
closely as we understood it.

## Preprocessing choices (Method A)
- **Filter:** 1–100 Hz band-pass (`load_data.py`).
- **Reference:** **average reference applied in Stage 0**, before any channel is dropped
  (ZUNA was trained on average-referenced data).
- **Epochs:** marker-locked 5 s (event codes 1–8), 16 test epochs by default.
- **Degradation:** dense cap → 19-channel clinical 10–20 montage (a spatial-upsampling test).
- **Baseline:** MNE spherical-spline interpolation.

## Files
- `load_data.py` — ingest `.cnt` → epoch → degrade to 19ch → spline baseline → export the
  ZUNA `.pt` tensors (z-score preserved channels, scale electrode positions into ZUNA's
  ±0.12 box, write the filename the dataloader parses).
- `main_pipeline.py` — 4-stage orchestrator: prepare tensors → mask → `zuna.inference()`
  (`data_norm=10.0`, 50 diffusion steps) → z-score reversal + hard-inpaint → alpha grading
  (IAF, FAA) with a PSD comparison plot.

## Run
```bash
# expects raw data under ../GEEG_Raw/ and the zuna package importable; HF_HOME optional
python main_pipeline.py
```

## Status
Superseded as the *evaluation* pipeline by `../benchmark/` (Method B), which added a proper
5-subject design, the K=8-linear fidelity baseline, and the reliability-floor biomarker test.
**Method A is retained deliberately** because we are unsure which preprocessing is the correct
one for ZUNA — see the top-level `REPORT.md` §"Two preprocessing methods" and the open
questions for Zyphra. The two methods are **not directly comparable** (different filter and
reference frame).
