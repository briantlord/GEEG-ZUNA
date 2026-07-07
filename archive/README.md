# archive/ — superseded exploration scripts

These predate the `benchmark/` harness and are kept for provenance only. They are **single-subject,
single-session**, hardcode input filenames, have no resumable queue, and were used to generate the
early figures in `../results/figures/`. They are **not** the current evaluation pipeline and their
conclusions are superseded by the 5-subject results in `../REPORT.md`.

| File | What it did |
|---|---|
| `Project_Overview_phase1.docx` | The original phase-1 write-up. Reports "ZUNA beats spline" on a **single subject** with **Method-A** preprocessing — a conclusion later overturned once K=8 linear (not spline) was used as the fidelity baseline across 5 subjects. Read alongside `../REPORT.md`, which supersedes it. |
| `compare_outputs.py` | 13-figure per-channel fidelity battery (temporal/spectral r, RMSE, SDR, topography). |
| `advanced_neuro_tests.py` | Connectivity (wPLI/PLV), source-space ROI power (eLORETA/dSPM), ICA topography correlation. |
| `test_2channel_dropout.py` | Focused 2-channel dropout with publication figures. |
| `test_n_channel_dropout.py` | N∈{2,4,8} random-dropout sweep (8 trials/N) → `SWEEP_SUMMARY.png`. |
| `regen_sweep_summary.py` | Regenerates the sweep summary plot from hardcoded metrics. |
| `run_stage4_only.py` | Re-runs only the alpha-grading stage from saved `.npy`. |
| `inspect_pt.py` | Debug utility: dumps `.pt` input/output shapes and stats. |

Stale `D:\` HuggingFace-cache paths in these scripts have been replaced with an `HF_HOME`-aware
snippet, but they still assume single-subject inputs and are not maintained.
