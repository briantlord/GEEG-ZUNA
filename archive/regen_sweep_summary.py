"""Regenerate the sweep summary figure from already-computed data."""
import os, sys, glob
# ZUNA weights load from HuggingFace (Zyphra/ZUNA) and auto-download to the default HF cache.
# To use a pre-downloaded snapshot, set HF_HOME before running (e.g. export HF_HOME=/path/to/HF_cache).
if os.environ.get("HF_HOME"):
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from test_n_channel_dropout import (
    load_shared_data, select_channels, run_spline,
    compute_metrics, save_sweep_summary, RESULTS_ROOT
)
import mne

truth, ch_names, positions = load_shared_data()

all_metrics = {}
for n, seed in [(2, 44), (4, 46), (8, 50)]:
    drop_names, drop_indices = select_channels(ch_names, n, seed)

    # Reload spline (fast, no GPU needed)
    X_spline = run_spline(truth, ch_names, positions, drop_indices)

    # Load saved ZUNA output
    zuna_path = f"G001_Day1Rest1_X_zuna_test.npy"  # not saved per-N — recompute from saved pt
    # The per-N .npy weren't saved; load from the tmp outputs if they still exist,
    # otherwise we skip and just use the numbers already printed to log.
    pass

# Since we already have all the numbers from the run output, hard-code them
# so we can immediately generate the figure.
all_metrics = {
    2: {
        "r_temporal":  (0.6738, 0.7585),
        "r_spectral":  (0.9703, 0.8945),
        "rmse":        (3.1323, 2.6887),
        "sdr":         (3.4019, 4.0399),
        "band_power": {
            "Delta": (1.8136, 1.1603),
            "Theta": (0.5605, 0.4843),
            "Alpha": (0.5112, 0.0955),
            "Beta":  (1.4190, 0.1235),
            "Gamma": (1.6736, 0.3855),
        },
    },
    4: {
        "r_temporal":  (0.6534, 0.7321),
        "r_spectral":  (0.9391, 0.8818),
        "rmse":        (4.1629, 3.1721),
        "sdr":         (1.7936, 3.2724),
        "band_power": {
            "Delta": (1.8396, 0.7413),
            "Theta": (0.4697, 0.4849),
            "Alpha": (0.5236, 0.6323),
            "Beta":  (1.5831, 1.2297),
            "Gamma": (1.8453, 1.5525),
        },
    },
    8: {
        "r_temporal":  (0.6849, 0.7334),
        "r_spectral":  (0.9721, 0.9238),
        "rmse":        (4.2885, 4.2276),
        "sdr":         (4.2921, 3.4225),
        "band_power": {
            "Delta": (1.1641, 0.9702),
            "Theta": (0.5653, 0.4165),
            "Alpha": (0.6695, 0.2278),
            "Beta":  (1.2750, 0.7499),
            "Gamma": (1.7858, 0.6311),
        },
    },
}

save_sweep_summary([2, 4, 8], all_metrics, RESULTS_ROOT)
print("Done.")
