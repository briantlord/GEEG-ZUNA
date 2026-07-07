"""
test_2channel_dropout.py — Soft 2-Channel Dropout Test
=======================================================
Drops 2 maximally distant channels from the full electrode array, then
compares ZUNA inference vs spherical spline interpolation.

This is a 'softer' test than the 19→64 channel test: the model retains
nearly complete spatial context (~97% of channels), so both methods should
perform well. The interesting question is whether ZUNA's learned temporal
priors add anything over a purely spatial interpolator.

Stages
------
  0  Load truth array + channel info from existing .pt / .npy files
  1  Select 2 most spatially distant channels (or specify via --channels)
  2  Build degraded input (zero out 2 channels) + spline baseline via MNE
  3  Build ZUNA .pt tensor, run inference
  4  Reverse z-score, hard-inpaint the 60 preserved channels
  5  Compare: focused metrics + publication figures for the 2 dropped channels

Usage
-----
  python test_2channel_dropout.py
  python test_2channel_dropout.py --channels FP1 O2
  python test_2channel_dropout.py --epochs 32 --steps 50
"""

import os
# ZUNA weights load from HuggingFace (Zyphra/ZUNA) and auto-download to the default HF cache.
# To use a pre-downloaded snapshot, set HF_HOME before running (e.g. export HF_HOME=/path/to/HF_cache).
if os.environ.get("HF_HOME"):
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import glob
import shutil
import logging
import time

import numpy as np
import torch
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

from zuna import inference as zuna_inference


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SUBJECT_ID    = "G001"
SESSION_NAME  = "Day1Rest1"
SFREQ         = 256
NUM_EPOCHS    = 16        # set to None for all available epochs
DIFFUSION_STEPS = 50
GPU_DEVICE    = "0"
DATA_NORM     = 10.0      # MUST match ZUNA training: brings std from ~1 → 0.1

PT_INPUT_DIR  = "test_2ch_pt_in"
PT_OUTPUT_DIR = "test_2ch_pt_out"
RESULTS_DIR   = "results_2ch_dropout"

BANDS = {
    "Delta": (1,  4),
    "Theta": (4,  8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 50),
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("2CH_TEST")
log.setLevel(logging.DEBUG)
_fh = logging.FileHandler("test_2channel_dropout.log", mode="w", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_fh)
log.addHandler(_ch)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — Load existing truth data + channel metadata
# ══════════════════════════════════════════════════════════════════════════════
def stage0_load_data():
    log.info("=" * 72)
    log.info("STAGE 0: LOADING EXISTING TRUTH DATA")
    log.info("=" * 72)

    truth_path = f"{SUBJECT_ID}_{SESSION_NAME}_y_truth.npy"
    if not os.path.exists(truth_path):
        log.error(f"Missing: {truth_path}")
        log.error("Run main_pipeline.py first to generate reference arrays.")
        sys.exit(1)

    truth = np.load(truth_path)  # (epochs, channels, time)
    n_epochs_avail = truth.shape[0]
    n_epochs = min(NUM_EPOCHS, n_epochs_avail) if NUM_EPOCHS else n_epochs_avail
    truth = truth[:n_epochs]
    log.info(f"  Truth shape : {truth.shape}  ({truth.shape[2]/SFREQ:.1f}s epochs @ {SFREQ}Hz)")

    # Recover channel names + 3D positions from the staged .pt file
    pt_files = sorted(glob.glob("test_pt_out/*.pt"))
    if not pt_files:
        log.error("No staged .pt files found in test_pt_out/. Run main_pipeline.py first.")
        sys.exit(1)

    pt_data    = torch.load(pt_files[0], weights_only=False)
    meta       = pt_data.get("metadata", {})
    ch_names   = sorted(meta.get("ch_names", [f"Ch{i}" for i in range(truth.shape[1])]))
    pos_tensor = pt_data["channel_positions"]  # (n_channels, 3)

    # pos_tensor may have been scaled to ZUNA bounding box; recover from MNE montage
    # for accurate spatial distance calculations.
    montage = mne.channels.make_standard_montage("standard_1005")
    raw_pos = montage.get_positions()["ch_pos"]
    positions = np.zeros((len(ch_names), 3), dtype=np.float32)
    for i, ch in enumerate(ch_names):
        key = next((k for k in raw_pos if k.upper() == ch.upper()), None)
        if key:
            positions[i] = raw_pos[key]
        else:
            # Fall back to scaled ZUNA positions if no montage match
            positions[i] = pos_tensor[i].numpy() if hasattr(pos_tensor, "numpy") else pos_tensor[i]

    log.info(f"  Channels    : {len(ch_names)}")
    log.info(f"  Epochs used : {n_epochs} / {n_epochs_avail}")
    return truth, ch_names, positions


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Select 2 most spatially distant channels
# ══════════════════════════════════════════════════════════════════════════════
def stage1_select_channels(ch_names, positions, manual_channels=None):
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 1: CHANNEL SELECTION")
    log.info("=" * 72)

    if manual_channels:
        # Validate manually specified channels
        ch_upper = [c.upper() for c in ch_names]
        resolved = []
        for mc in manual_channels:
            if mc.upper() in ch_upper:
                resolved.append(ch_names[ch_upper.index(mc.upper())])
            else:
                log.error(f"Channel '{mc}' not found. Available: {ch_names}")
                sys.exit(1)
        drop_channels = resolved[:2]
        idx_a = ch_names.index(drop_channels[0])
        idx_b = ch_names.index(drop_channels[1])
        dist_m = float(np.linalg.norm(positions[idx_a] - positions[idx_b]))
        log.info(f"  Manual selection: {drop_channels[0]} and {drop_channels[1]}")
    else:
        # Auto-select: find the maximally distant pair
        dist_matrix = squareform(pdist(positions))  # (n_ch, n_ch)
        np.fill_diagonal(dist_matrix, 0.0)
        flat_idx = np.argmax(dist_matrix)
        idx_a, idx_b = np.unravel_index(flat_idx, dist_matrix.shape)
        drop_channels = [ch_names[idx_a], ch_names[idx_b]]
        dist_m = dist_matrix[idx_a, idx_b]
        log.info(f"  Auto-selected maximally distant pair: {drop_channels[0]} and {drop_channels[1]}")

    log.info(f"  Scalp distance  : {dist_m * 100:.1f} cm")
    log.info(f"  Position A ({drop_channels[0]}): {positions[ch_names.index(drop_channels[0])]}")
    log.info(f"  Position B ({drop_channels[1]}): {positions[ch_names.index(drop_channels[1])]}")

    drop_indices = [ch_names.index(drop_channels[0]), ch_names.index(drop_channels[1])]
    return drop_channels, drop_indices


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Build degraded input + spline baseline
# ══════════════════════════════════════════════════════════════════════════════
def stage2_degrade_and_spline(truth, ch_names, positions, drop_indices):
    """
    Creates:
      X_broken  — truth with the 2 dropped channels zeroed (µV)
      X_spline  — MNE spherical spline reconstruction (µV)
    """
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 2: DEGRADATION + SPLINE BASELINE")
    log.info("=" * 72)

    X_broken = truth.copy()
    X_broken[:, drop_indices, :] = 0.0
    log.info(f"  Zeroed channels at indices {drop_indices}")

    # ── Build MNE Epochs for spline interpolation ────────────────────────────
    # Create a minimal MNE Info + Epochs from the truth array
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)

    # Reconstruct the head montage from positions
    ch_pos_dict = {ch: pos for ch, pos in zip(ch_names, positions)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos_dict, coord_frame="head")

    # MNE EpochsArray expects data in Volts
    epochs_V = truth * 1e-6                 # µV → V, shape (n_epochs, n_ch, n_time)
    epochs_mne = mne.EpochsArray(epochs_V, info, verbose=False)
    epochs_mne.set_montage(montage, verbose=False)

    # Mark dropped channels as bad so MNE interpolates them
    epochs_mne.info["bads"] = [ch_names[i] for i in drop_indices]

    log.info("  Running MNE spherical spline interpolation ...")
    epochs_spline = epochs_mne.copy()
    epochs_spline.interpolate_bads(reset_bads=True, verbose=False)
    X_spline = epochs_spline.get_data(copy=True) * 1e6  # V → µV

    log.info(f"  [OK] Spline done. Shape: {X_spline.shape}")
    return X_broken, X_spline


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Build ZUNA .pt tensor + run inference
# ══════════════════════════════════════════════════════════════════════════════
def stage3_zuna_inference(truth, X_broken, ch_names, positions, drop_indices):
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 3: ZUNA INFERENCE")
    log.info("=" * 72)

    # ── Clear old directories ─────────────────────────────────────────────────
    for d in [PT_INPUT_DIR, PT_OUTPUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    n_epochs, n_ch, n_time = X_broken.shape

    # ── Z-score normalisation (ZUNA expects std ≈ 1.0 before data_norm /10) ──
    # Compute stats from the 60 non-zero (preserved) channels only
    nonzero_mask = np.any(X_broken != 0.0, axis=-1)   # (epochs, channels)
    nonzero_vals = X_broken[nonzero_mask]
    zscore_mean  = float(nonzero_vals.mean())
    zscore_std   = float(nonzero_vals.std())

    X_normed = X_broken.copy()
    for ep in range(n_epochs):
        for ch in range(n_ch):
            if nonzero_mask[ep, ch]:
                X_normed[ep, ch, :] = (X_normed[ep, ch, :] - zscore_mean) / zscore_std
    # Dropped channels stay at exactly 0.0

    log.info(f"  Z-score: mean={zscore_mean:.4f} µV, std={zscore_std:.4f} µV")
    log.info(f"  Post-norm std of preserved channels: {X_normed[nonzero_mask].std():.4f}")
    log.info(f"  After data_norm={DATA_NORM}: will be ~{X_normed[nonzero_mask].std()/DATA_NORM:.4f} (target 0.1)")

    # ── Scale positions to ZUNA's [-0.12, 0.12] bounding box ─────────────────
    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    max_val    = torch.abs(pos_tensor).max()
    if max_val > 0.119:
        scale = 0.119 / max_val
        pos_tensor = pos_tensor * scale
        log.info(f"  Rescaled positions by {scale:.4f} to fit ZUNA bounding box.")

    # ── Build .pt dict ────────────────────────────────────────────────────────
    pt_dict = {
        "data":              torch.tensor(X_normed, dtype=torch.float32),
        "channel_positions": pos_tensor,
        "metadata": {
            "sfreq":        SFREQ,
            "ch_names":     ch_names,
            "zscore_mean":  zscore_mean,
            "zscore_std":   zscore_std,
            "drop_channels": [ch_names[i] for i in drop_indices],
            "drop_indices":  drop_indices,
        },
    }

    # ZUNA parses filename for tensor dimensions: prefix_epochs_channels_time.pt
    fname = f"ds000000_000000_000000_d00_{n_epochs:05d}_{n_ch}_{n_time}.pt"
    torch.save(pt_dict, os.path.join(PT_INPUT_DIR, fname))
    log.info(f"  Saved ZUNA input: {fname}")

    # ── HPC network isolation (same as main_pipeline.py) ─────────────────────
    for k in [k for k in os.environ if k.startswith("SLURM")]:
        del os.environ[k]
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
        "WORLD_SIZE":  "1",         "RANK":         "0",
        "LOCAL_RANK":  "0",
        "NCCL_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1",
    })

    # ── Dynamic token count ───────────────────────────────────────────────────
    # 1280 samples / 128 fine_pts = 10 coarse steps; pack ~8 epochs per batch
    coarse_steps   = n_time // 128          # 10
    tokens_per_epoch = coarse_steps * n_ch  # 10 × 62 = 620
    tokens_per_batch = tokens_per_epoch * 8 # pack 8 epochs  (~5 000 tokens)
    log.info(f"  Coarse steps/epoch : {coarse_steps}")
    log.info(f"  Tokens/epoch       : {tokens_per_epoch}")
    log.info(f"  Target batch tokens: {tokens_per_batch}")

    t0 = time.time()
    zuna_inference(
        input_dir             = PT_INPUT_DIR,
        output_dir            = PT_OUTPUT_DIR,
        gpu_device            = GPU_DEVICE,
        diffusion_sample_steps= DIFFUSION_STEPS,
        tokens_per_batch      = tokens_per_batch,
        data_norm             = DATA_NORM,      # CRITICAL: ensures std 1.0 → 0.1
    )
    elapsed = time.time() - t0
    log.info(f"  [TIMER] Inference: {int(elapsed)//60}m {int(elapsed)%60}s")

    recon_files = sorted(glob.glob(f"{PT_OUTPUT_DIR}/*.pt"))
    log.info(f"  Output files : {len(recon_files)}")
    if not recon_files:
        log.error("Inference produced no output files.")
        sys.exit(1)

    return zscore_mean, zscore_std


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Reverse z-score + hard-inpaint preserved channels
# ══════════════════════════════════════════════════════════════════════════════
def stage4_export(truth, ch_names, drop_indices, zscore_mean, zscore_std):
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 4: RESCALING + HARD INPAINTING")
    log.info("=" * 72)

    n_epochs = truth.shape[0]

    # ── Collect all reconstructed epochs ─────────────────────────────────────
    all_epochs = []
    for ptf in sorted(glob.glob(f"{PT_OUTPUT_DIR}/*.pt")):
        data = torch.load(ptf, weights_only=False)
        key  = "data_reconstructed" if "data_reconstructed" in data else "data"
        for ep in data.get(key, []):
            arr = ep.cpu().numpy() if isinstance(ep, torch.Tensor) else np.asarray(ep)
            all_epochs.append(arr)

    if not all_epochs:
        log.error("No reconstructed epochs found in output .pt files.")
        sys.exit(1)

    Z = np.stack(all_epochs[:n_epochs])           # (n_epochs, n_ch, n_time)

    # ── Reverse z-score ───────────────────────────────────────────────────────
    log.info(f"  Reversing z-score: mean={zscore_mean:.4f}, std={zscore_std:.4f}")
    Z = Z * zscore_std + zscore_mean

    # Align shape to truth
    n_ch   = truth.shape[1]
    n_time = truth.shape[2]
    Z = Z[:, :n_ch, :n_time]

    # ── Hard inpaint: overwrite the 60 preserved channels with truth ─────────
    # ZUNA only needs to reconstruct the 2 dropped channels.
    # The preserved channels are replaced with ground truth to prevent
    # the model's output from contaminating channels it never needed to infer.
    preserve_mask = np.ones((n_epochs, n_ch), dtype=bool)
    preserve_mask[:, drop_indices] = False
    Z = np.where(preserve_mask[:, :, np.newaxis], truth, Z)

    log.info(f"  [OK] Hard inpainting applied. Preserved {preserve_mask.sum() // n_epochs} channels.")
    log.info(f"  Final shape: {Z.shape}")

    out_path = f"{SUBJECT_ID}_{SESSION_NAME}_X_zuna_2ch.npy"
    np.save(out_path, Z)
    log.info(f"  Saved: {out_path}")
    return Z


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Comparison: metrics + figures for the 2 dropped channels
# ══════════════════════════════════════════════════════════════════════════════

def _pearsonr_safe(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r, _ = pearsonr(a.ravel(), b.ravel())
    return float(r)


def _band_power(signal_flat, sfreq, lo, hi):
    """Mean band power (µV²/Hz) from a 1D signal."""
    f, p = welch(signal_flat, fs=sfreq, nperseg=min(sfreq * 2, len(signal_flat)))
    mask = (f >= lo) & (f <= hi)
    return float(p[mask].mean()) if mask.any() else 0.0


def _sdr_db(truth_flat, pred_flat):
    sig   = np.sum(truth_flat ** 2)
    noise = np.sum((truth_flat - pred_flat) ** 2)
    if noise < 1e-30:
        return np.inf
    return 10 * np.log10(sig / noise)


def stage5_compare(truth, X_spline, X_zuna, ch_names, drop_indices, drop_channels):
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 5: COMPARISON — 2 DROPPED CHANNELS")
    log.info("=" * 72)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_epochs, n_ch, n_time = truth.shape

    results = {}
    for idx, ch in zip(drop_indices, drop_channels):
        t_flat = truth[:, idx, :].ravel()
        s_flat = X_spline[:, idx, :].ravel()
        z_flat = X_zuna[:, idx, :].ravel()

        # ── Temporal correlation ──────────────────────────────────────────────
        r_s = _pearsonr_safe(t_flat, s_flat)
        r_z = _pearsonr_safe(t_flat, z_flat)

        # ── RMSE ─────────────────────────────────────────────────────────────
        rmse_s = float(np.sqrt(np.mean((t_flat - s_flat) ** 2)))
        rmse_z = float(np.sqrt(np.mean((t_flat - z_flat) ** 2)))

        # ── SDR ───────────────────────────────────────────────────────────────
        sdr_s = _sdr_db(t_flat, s_flat)
        sdr_z = _sdr_db(t_flat, z_flat)

        # ── Spectral correlation ──────────────────────────────────────────────
        _, p_t = welch(t_flat, fs=SFREQ, nperseg=SFREQ * 2)
        _, p_s = welch(s_flat, fs=SFREQ, nperseg=SFREQ * 2)
        _, p_z = welch(z_flat, fs=SFREQ, nperseg=SFREQ * 2)
        sc_s = _pearsonr_safe(np.log10(p_t + 1e-30), np.log10(p_s + 1e-30))
        sc_z = _pearsonr_safe(np.log10(p_t + 1e-30), np.log10(p_z + 1e-30))

        # ── Band power errors (dB) ────────────────────────────────────────────
        bp_errors = {}
        for band, (lo, hi) in BANDS.items():
            bp_t = _band_power(t_flat, SFREQ, lo, hi)
            bp_s = _band_power(s_flat, SFREQ, lo, hi)
            bp_z = _band_power(z_flat, SFREQ, lo, hi)
            err_s = abs(10 * np.log10(bp_t + 1e-30) - 10 * np.log10(bp_s + 1e-30))
            err_z = abs(10 * np.log10(bp_t + 1e-30) - 10 * np.log10(bp_z + 1e-30))
            bp_errors[band] = (err_s, err_z)

        results[ch] = dict(
            r_temporal_spline=r_s,   r_temporal_zuna=r_z,
            rmse_spline=rmse_s,       rmse_zuna=rmse_z,
            sdr_spline=sdr_s,         sdr_zuna=sdr_z,
            r_spectral_spline=sc_s,   r_spectral_zuna=sc_z,
            band_power_errors=bp_errors,
        )

    # ── Print summary table ───────────────────────────────────────────────────
    hdr = f"  {'Metric':<26}  {'Spline':>10}  {'ZUNA':>10}  {'Δ (ZUNA−Spline)':>16}"
    sep = "  " + "-" * 68
    log.info("")
    for ch in drop_channels:
        r = results[ch]
        log.info(f"  ┌─ Channel: {ch} {'─'*50}")
        log.info(hdr)
        log.info(sep)

        def row(label, sv, zv, higher_better=True):
            delta = zv - sv
            marker = "▲" if (delta > 0) == higher_better else "▼"
            log.info(f"  {label:<26}  {sv:>10.4f}  {zv:>10.4f}  {marker} {abs(delta):>12.4f}")

        row("Temporal corr (r)",    r["r_temporal_spline"],   r["r_temporal_zuna"])
        row("Spectral corr (r)",    r["r_spectral_spline"],   r["r_spectral_zuna"])
        row("SDR (dB)",             r["sdr_spline"],           r["sdr_zuna"])
        row("RMSE (µV)",            r["rmse_spline"],          r["rmse_zuna"], higher_better=False)
        log.info(sep)
        for band, (es, ez) in r["band_power_errors"].items():
            row(f"  {band} BP error (dB)", es, ez, higher_better=False)
        log.info("")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Time series for both dropped channels (first epoch, 2s)
    # ══════════════════════════════════════════════════════════════════════════
    dur_s   = 2.0
    n_samp  = int(dur_s * SFREQ)
    t_axis  = np.arange(n_samp) / SFREQ

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, idx, ch in zip(axes, drop_indices, drop_channels):
        r = results[ch]
        sig_t = truth[0, idx, :n_samp]
        sig_s = X_spline[0, idx, :n_samp]
        sig_z = X_zuna[0, idx, :n_samp]

        ax.plot(t_axis, sig_t, "k-",   lw=2.0, label="Truth",  zorder=3)
        ax.plot(t_axis, sig_s, "r--",  lw=1.5, alpha=0.85, label=f"Spline  r={r['r_temporal_spline']:.3f}", zorder=2)
        ax.plot(t_axis, sig_z, "b-",   lw=1.5, alpha=0.85, label=f"ZUNA    r={r['r_temporal_zuna']:.3f}",   zorder=2)
        ax.set_title(f"Channel {ch} — First 2s of Epoch 1  |  RMSE: Spline={r['rmse_spline']:.2f}µV  ZUNA={r['rmse_zuna']:.2f}µV")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"2-Channel Dropout Test — {SUBJECT_ID}  |  {drop_channels[0]} & {drop_channels[1]} reconstructed",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig1_timeseries.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  [OK] fig1_timeseries.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — PSD comparison (averaged across all epochs)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, idx, ch in zip(axes, drop_indices, drop_channels):
        r = results[ch]
        f_ax, p_t = welch(truth[:, idx, :].ravel(),    fs=SFREQ, nperseg=SFREQ * 2)
        _,    p_s = welch(X_spline[:, idx, :].ravel(), fs=SFREQ, nperseg=SFREQ * 2)
        _,    p_z = welch(X_zuna[:, idx, :].ravel(),   fs=SFREQ, nperseg=SFREQ * 2)

        ax.plot(f_ax, 10 * np.log10(p_t + 1e-30), "k-",  lw=2.0, label="Truth")
        ax.plot(f_ax, 10 * np.log10(p_s + 1e-30), "r--", lw=1.5, alpha=0.85,
                label=f"Spline  r_spec={r['r_spectral_spline']:.3f}")
        ax.plot(f_ax, 10 * np.log10(p_z + 1e-30), "b-",  lw=1.5, alpha=0.85,
                label=f"ZUNA    r_spec={r['r_spectral_zuna']:.3f}")
        ax.axvspan(8, 13, color="gray", alpha=0.1, label="Alpha (8–13 Hz)")
        ax.set_xlim(1, 50)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title(f"PSD — Channel {ch}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.suptitle("Spectral Fidelity — 2 Dropped Channels", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig2_psd.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  [OK] fig2_psd.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Per-epoch temporal correlation
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, idx, ch in zip(axes, drop_indices, drop_channels):
        rs_per_ep = [_pearsonr_safe(truth[ep, idx, :], X_spline[ep, idx, :]) for ep in range(n_epochs)]
        rz_per_ep = [_pearsonr_safe(truth[ep, idx, :], X_zuna[ep, idx, :])   for ep in range(n_epochs)]
        ep_ax = np.arange(1, n_epochs + 1)

        ax.plot(ep_ax, rs_per_ep, "rs--", ms=7,  lw=1.5, label=f"Spline (mean={np.mean(rs_per_ep):.3f})")
        ax.plot(ep_ax, rz_per_ep, "b^-",  ms=7,  lw=1.5, label=f"ZUNA   (mean={np.mean(rz_per_ep):.3f})")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Per-Epoch Temporal Correlation — {ch}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.suptitle("Epoch-wise Reconstruction Consistency", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig3_epoch_correlations.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  [OK] fig3_epoch_correlations.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 4 — Band power error comparison (bar chart)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ch in zip(axes, drop_channels):
        r = results[ch]
        bands_list = list(BANDS.keys())
        es = [r["band_power_errors"][b][0] for b in bands_list]
        ez = [r["band_power_errors"][b][1] for b in bands_list]
        x  = np.arange(len(bands_list))
        w  = 0.35

        ax.bar(x - w/2, es, w, color="red",  alpha=0.7, label="Spline")
        ax.bar(x + w/2, ez, w, color="blue", alpha=0.7, label="ZUNA")
        ax.set_xticks(x)
        ax.set_xticklabels(bands_list)
        ax.set_ylabel("Absolute Band Power Error (dB)")
        ax.set_title(f"Band Power Error — Channel {ch}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("Band Power Reconstruction Error", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig4_band_power.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  [OK] fig4_band_power.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 5 — Full-epoch waveform scatter (truth vs reconstruction)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, (idx, ch) in enumerate(zip(drop_indices, drop_channels)):
        t_flat = truth[:, idx, :].ravel()
        s_flat = X_spline[:, idx, :].ravel()
        z_flat = X_zuna[:, idx, :].ravel()

        # Subsample for plotting (every 16th point)
        step = 16
        vmin = min(t_flat[::step].min(), s_flat[::step].min(), z_flat[::step].min())
        vmax = max(t_flat[::step].max(), s_flat[::step].max(), z_flat[::step].max())
        diag = [vmin, vmax]

        for row, (pred, label, color) in enumerate([
            (s_flat, "Spline", "red"),
            (z_flat, "ZUNA",   "blue"),
        ]):
            ax = axes[row, col]
            ax.scatter(t_flat[::step], pred[::step], s=2, alpha=0.3, color=color)
            ax.plot(diag, diag, "k-", lw=1.5, label="y = x")
            r = _pearsonr_safe(t_flat, pred)
            ax.set_title(f"{label} vs Truth — {ch}  (r={r:.3f})")
            ax.set_xlabel("Truth (µV)")
            ax.set_ylabel(f"{label} (µV)")
            ax.legend(fontsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)

    plt.suptitle("Amplitude Scatter: Reconstruction vs Truth", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig5_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  [OK] fig5_scatter.png")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global NUM_EPOCHS, DIFFUSION_STEPS   # module-level knobs overridden by CLI below
    parser = argparse.ArgumentParser(description="2-channel dropout test for ZUNA")
    parser.add_argument("--channels",  nargs=2, default=None, metavar=("CH1", "CH2"),
                        help="Manually specify 2 channel names to drop (default: auto-select most distant)")
    parser.add_argument("--epochs",    type=int,   default=None,
                        help="Number of epochs to process (default: use NUM_EPOCHS constant)")
    parser.add_argument("--steps",     type=int,   default=DIFFUSION_STEPS,
                        help=f"Diffusion sample steps (default: {DIFFUSION_STEPS})")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.epochs:
        NUM_EPOCHS      = args.epochs
    if args.steps:
        DIFFUSION_STEPS = args.steps

    t_total = time.time()
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════════╗")
    log.info("║          ZUNA 2-CHANNEL DROPOUT TEST                            ║")
    log.info("╚══════════════════════════════════════════════════════════════════╝")
    log.info("")

    truth, ch_names, positions = stage0_load_data()
    drop_channels, drop_indices = stage1_select_channels(ch_names, positions, args.channels)

    log.info("")
    log.info(f"  Dropping: {drop_channels[0]} (idx {drop_indices[0]}) and {drop_channels[1]} (idx {drop_indices[1]})")
    log.info(f"  Keeping : {len(ch_names) - 2} / {len(ch_names)} channels as context")

    X_broken, X_spline = stage2_degrade_and_spline(truth, ch_names, positions, drop_indices)
    zscore_mean, zscore_std = stage3_zuna_inference(
        truth, X_broken, ch_names, positions, drop_indices
    )
    X_zuna = stage4_export(truth, ch_names, drop_indices, zscore_mean, zscore_std)
    results = stage5_compare(truth, X_spline, X_zuna, ch_names, drop_indices, drop_channels)

    elapsed = time.time() - t_total
    log.info("")
    log.info("=" * 72)
    log.info(f"COMPLETE — Total time: {int(elapsed)//60}m {int(elapsed)%60}s")
    log.info(f"Figures   → {os.path.abspath(RESULTS_DIR)}/")
    log.info(f"Audit log → test_2channel_dropout.log")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
