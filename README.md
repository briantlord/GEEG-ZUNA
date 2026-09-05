# GEEG-ZUNA

This project tests whether the ZUNA 1.1 EEG model can reconstruct missing EEG
channels better than spherical spline interpolation.

The benchmark starts with a real EEG recording, deliberately hides selected
channels, reconstructs those channels with each method, and compares the
reconstructed EEG with the original data. The main question is not merely
whether the waveform looks similar, but whether reconstruction preserves EEG
measurements that researchers actually use.

## Measurements tested

Five channel sets are removed and reconstructed separately:

| Measurement | Hidden channels |
| --- | --- |
| Frontal alpha asymmetry (FAA) | F3, F4, F7, F8 |
| Theta/beta ratio | Cz, FCz, Fz, CPz |
| Frontal-midline theta | Fz, FCz, F1, F2 |
| Mu asymmetry | C3, C4, C1, C2 |
| Specparam posterior peaks | O1, O2, Oz, POz, PO3, PO4 |

Each channel set requires its own reconstruction because the model must be run
with those particular channels hidden. Completed reconstructions are retained
and can be resumed without repeating verified epochs.

## Processing pipeline

For each recording, the active pipeline:

1. Reads the Neuroscan CNT file as `int32`.
2. Keeps the 62 positioned EEG channels.
3. Applies one 0.5–45 Hz zero-phase FIR filter.
4. Resamples the recording to 256 Hz.
5. Creates non-overlapping, marker-locked five-second epochs.
6. Uses available source-valid epochs up to the requested maximum.
7. Applies no ICA or component subtraction.
8. Records amplitude and signal-quality measurements descriptively without
   using them to reject otherwise readable recordings or channels.
9. Applies the surviving-channel average reference only after the channels to
   hide have been declared.
10. Reconstructs the hidden channels with ZUNA and spherical spline
    interpolation, while leaving every observed channel unchanged.
11. Compares each reconstructed measurement with the same measurement computed
    from the original hidden-channel data.

The authoritative implementation is in [`benchmark/`](benchmark/). The
scientific settings are recorded in
[`config/scientific_contract_v1.json`](config/scientific_contract_v1.json).

## Current status

A complete corrected pilot has been run locally on `G001Day1Rest1`:

- 64 five-second epochs per channel set;
- all five measurement-specific channel sets;
- ZUNA 1.1 with 50 diffusion steps;
- spherical spline comparison;
- all reconstruction and result units completed;
- both result bundles passed the strict completeness validator; and
- no failed or missing units.

The pilot produced mixed results. ZUNA preserved frontal-midline theta, Fz
theta/beta ratio, mu asymmetry, and most specparam outputs better than spline on
this recording. Spline performed better for several other outputs, including
primary FAA and Cz theta/beta ratio. One recording is not sufficient to make a
general claim about either method.

The full multi-recording benchmark has not yet been run. Formal production/HPC
execution remains pending the validation items listed in the scientific
contract, including independent confirmation of the CNT integer format and the
predeclared ZUNA physical-scale sensitivity analyses.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`benchmark/`](benchmark/) | Active benchmark implementation |
| [`config/`](config/) | Versioned scientific and execution settings |
| [`scripts/`](scripts/) | Repository checks and HPC release builder |
| `GEEG-ZUNA-share/` | Generated upload bundle for the HPC; do not edit it manually |
| `GEEG_Raw/` | Local private EEG recordings; not stored on GitHub |
| `HF_cache/` | Local ZUNA model weights; not stored on GitHub |
| `results/` | Local generated outputs and reconstruction caches; not stored on GitHub |
| `archive/` | Obsolete or invalid historical material; never used by the active pipeline |

`GEEG-ZUNA-share/` is generated from the authoritative source tree. Changes
should be made to `benchmark/`, `config/`, or `scripts/`, then the share should
be rebuilt.

## Verification and HPC use

The active regression tests can be run with:

```bash
python -m unittest discover -s benchmark -p "test_*.py"
```

Instructions for building and running the HPC bundle are in
[`benchmark/HPC_RUNBOOK_zuna11.md`](benchmark/HPC_RUNBOOK_zuna11.md).

The raw recordings, model weights, and generated results are deliberately
excluded from GitHub. GitHub contains the code and scientific specification,
not the private dataset or large model artifacts.
