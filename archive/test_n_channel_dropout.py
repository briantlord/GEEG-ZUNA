"""
test_n_channel_dropout.py — N-Channel Random Dropout Sweep
===========================================================
Randomly drops N channels from the full electrode array, compares
ZUNA inference vs spherical spline interpolation, then sweeps across
multiple values of N to show how both methods degrade with increasing dropout.

Channel selection is random but reproducible (fixed seed per N).

Usage
-----
  python test_n_channel_dropout.py --n_drop 2 4 8
  python test_n_channel_dropout.py --n_drop 2 4 8 16 32 --seed 42
  python test_n_channel_dropout.py --n_drop 4 --channels FC1 FC2 CP1 CP2  (manual)
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
from scipy.signal import welch
from scipy.stats import pearsonr

from zuna import inference as zuna_inference


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SUBJECT_ID      = "G001"
SESSION_NAME    = "Day1Rest1"
SFREQ           = 256
NUM_EPOCHS      = 16
DIFFUSION_STEPS = 50
GPU_DEVICE      = "0"
DATA_NORM       = 10.0

RESULTS_ROOT    = "results_n_channel_dropout"

BANDS = {
    "Delta": (1,  4),
    "Theta": (4,  8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 50),
}

# Posterior channels for IAF center-of-gravity
POSTERIOR = {"O1","O2","OZ","P3","P4","PZ","P7","P8","PO3","PO4","POZ"}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("N_CH_TEST")
log.setLevel(logging.DEBUG)
_fh = logging.FileHandler("test_n_channel_dropout.log", mode="w", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_fh)
log.addHandler(_ch)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (once, shared across all N)
# ══════════════════════════════════════════════════════════════════════════════
def load_shared_data(n_epochs=NUM_EPOCHS):
    truth_path = f"{SUBJECT_ID}_{SESSION_NAME}_y_truth.npy"
    if not os.path.exists(truth_path):
        log.error(f"Missing {truth_path} — run main_pipeline.py first.")
        sys.exit(1)

    truth    = np.load(truth_path)[:n_epochs]
    pt_files = sorted(glob.glob("test_pt_out/*.pt"))
    if not pt_files:
        log.error("No .pt files in test_pt_out/ — run main_pipeline.py first.")
        sys.exit(1)

    pt_data   = torch.load(pt_files[0], weights_only=False)
    meta      = pt_data.get("metadata", {})
    ch_names  = sorted(meta.get("ch_names", [f"Ch{i}" for i in range(truth.shape[1])]))

    # Recover true (unscaled) 3D positions from the standard montage
    montage  = mne.channels.make_standard_montage("standard_1005")
    raw_pos  = montage.get_positions()["ch_pos"]
    positions = np.zeros((len(ch_names), 3), dtype=np.float32)
    for i, ch in enumerate(ch_names):
        key = next((k for k in raw_pos if k.upper() == ch.upper()), None)
        positions[i] = raw_pos[key] if key else pt_data["channel_positions"][i].numpy()

    log.info(f"  Loaded: {truth.shape}  ({len(ch_names)} channels, {NUM_EPOCHS} epochs)")
    return truth, ch_names, positions


# ══════════════════════════════════════════════════════════════════════════════
# Channel selection
# ══════════════════════════════════════════════════════════════════════════════
def select_channels(ch_names, n_drop, seed, manual=None):
    """
    Returns (drop_names, drop_indices).
    If manual is given it overrides random selection (must match n_drop in length).
    Random selection uses a fixed seed so results are reproducible.
    """
    if manual:
        ch_upper = [c.upper() for c in ch_names]
        resolved = []
        for mc in manual:
            if mc.upper() in ch_upper:
                resolved.append(ch_names[ch_upper.index(mc.upper())])
            else:
                log.error(f"Channel '{mc}' not found.")
                sys.exit(1)
        drop_names = resolved[:n_drop]
    else:
        rng        = np.random.default_rng(seed)
        drop_idx   = rng.choice(len(ch_names), size=n_drop, replace=False)
        drop_names = [ch_names[i] for i in sorted(drop_idx)]

    drop_indices = [ch_names.index(ch) for ch in drop_names]
    return drop_names, drop_indices


# ══════════════════════════════════════════════════════════════════════════════
# Spline baseline via MNE
# ══════════════════════════════════════════════════════════════════════════════
def run_spline(truth, ch_names, positions, drop_indices):
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    ch_pos_dict = {ch: pos for ch, pos in zip(ch_names, positions)}
    montage     = mne.channels.make_dig_montage(ch_pos=ch_pos_dict, coord_frame="head")
    epochs_mne  = mne.EpochsArray(truth * 1e-6, info, verbose=False)
    epochs_mne.set_montage(montage, verbose=False)
    epochs_mne.info["bads"] = [ch_names[i] for i in drop_indices]
    epochs_spline = epochs_mne.copy()
    epochs_spline.interpolate_bads(reset_bads=True, verbose=False)
    return epochs_spline.get_data(copy=True) * 1e6   # V → µV


# ══════════════════════════════════════════════════════════════════════════════
# ZUNA inference for one dropout set
# ══════════════════════════════════════════════════════════════════════════════
def run_zuna(truth, ch_names, positions, drop_indices, run_tag, n_steps=DIFFUSION_STEPS):
    pt_in  = f"_tmp_pt_in_{run_tag}"
    pt_out = f"_tmp_pt_out_{run_tag}"
    for d in [pt_in, pt_out]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    n_epochs, n_ch, n_time = truth.shape

    X_broken = truth.copy()
    X_broken[:, drop_indices, :] = 0.0

    # Z-score on preserved channels only
    nonzero_mask = np.any(X_broken != 0.0, axis=-1)
    nz_vals      = X_broken[nonzero_mask]
    zmean, zstd  = float(nz_vals.mean()), float(nz_vals.std())

    X_normed = X_broken.copy()
    for ep in range(n_epochs):
        for ch in range(n_ch):
            if nonzero_mask[ep, ch]:
                X_normed[ep, ch, :] = (X_normed[ep, ch, :] - zmean) / zstd

    # Scale positions to ZUNA bounding box
    pos_t   = torch.tensor(positions, dtype=torch.float32)
    max_val = pos_t.abs().max()
    if max_val > 0.119:
        pos_t = pos_t * (0.119 / max_val)

    pt_dict = {
        "data":              torch.tensor(X_normed, dtype=torch.float32),
        "channel_positions": pos_t,
        "metadata": {
            "sfreq":       SFREQ,
            "ch_names":    ch_names,
            "zscore_mean": zmean,
            "zscore_std":  zstd,
        },
    }
    fname = f"ds000000_000000_000000_d00_{n_epochs:05d}_{n_ch}_{n_time}.pt"
    torch.save(pt_dict, os.path.join(pt_in, fname))

    # HPC isolation
    for k in [k for k in os.environ if k.startswith("SLURM")]:
        del os.environ[k]
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
        "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
        "NCCL_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1",
    })

    coarse_steps     = n_time // 128
    tokens_per_batch = coarse_steps * n_ch * 8

    t0 = time.time()
    zuna_inference(
        input_dir              = pt_in,
        output_dir             = pt_out,
        gpu_device             = GPU_DEVICE,
        diffusion_sample_steps = n_steps,
        tokens_per_batch       = tokens_per_batch,
        data_norm              = DATA_NORM,
    )
    elapsed = time.time() - t0
    log.info(f"    Inference: {int(elapsed)//60}m {int(elapsed)%60}s")

    # Collect reconstructed epochs
    all_epochs = []
    for ptf in sorted(glob.glob(f"{pt_out}/*.pt")):
        data = torch.load(ptf, weights_only=False)
        key  = "data_reconstructed" if "data_reconstructed" in data else "data"
        for ep in data.get(key, []):
            arr = ep.cpu().numpy() if isinstance(ep, torch.Tensor) else np.asarray(ep)
            all_epochs.append(arr)

    if not all_epochs:
        log.error(f"No reconstructed epochs in {pt_out}")
        sys.exit(1)

    Z = np.stack(all_epochs[:n_epochs])
    Z = Z * zstd + zmean                       # reverse z-score
    Z = Z[:, :n_ch, :n_time]

    # Hard-inpaint preserved channels
    preserve = np.ones((n_epochs, n_ch), dtype=bool)
    preserve[:, drop_indices] = False
    Z = np.where(preserve[:, :, np.newaxis], truth, Z)

    # Cleanup temp dirs
    shutil.rmtree(pt_in)
    shutil.rmtree(pt_out)

    return Z


# ══════════════════════════════════════════════════════════════════════════════
# Metrics for the dropped channels only
# ══════════════════════════════════════════════════════════════════════════════
def _r(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(pearsonr(a.ravel(), b.ravel())[0])

def _sdr(truth_f, pred_f):
    noise = np.sum((truth_f - pred_f) ** 2)
    if noise < 1e-30: return np.inf
    return 10 * np.log10(np.sum(truth_f ** 2) / noise)

def _band_power(sig, sfreq, lo, hi):
    f, p = welch(sig, fs=sfreq, nperseg=min(sfreq * 2, len(sig)))
    m = (f >= lo) & (f <= hi)
    return float(p[m].mean()) if m.any() else 0.0

def compute_metrics(truth, X_spline, X_zuna, drop_indices):
    """Returns dict of scalar metrics averaged over the dropped channels."""
    rs_tcorr, rz_tcorr = [], []
    rs_scorr, rz_scorr = [], []
    rs_rmse,  rz_rmse  = [], []
    rs_sdr,   rz_sdr   = [], []
    rs_bp, rz_bp = {b: [] for b in BANDS}, {b: [] for b in BANDS}

    for idx in drop_indices:
        t = truth[:, idx, :].ravel()
        s = X_spline[:, idx, :].ravel()
        z = X_zuna[:, idx, :].ravel()

        rs_tcorr.append(_r(t, s))
        rz_tcorr.append(_r(t, z))
        rs_rmse.append(float(np.sqrt(np.mean((t - s) ** 2))))
        rz_rmse.append(float(np.sqrt(np.mean((t - z) ** 2))))
        rs_sdr.append(_sdr(t, s))
        rz_sdr.append(_sdr(t, z))

        _, pt = welch(t, fs=SFREQ, nperseg=SFREQ * 2)
        _, ps = welch(s, fs=SFREQ, nperseg=SFREQ * 2)
        _, pz = welch(z, fs=SFREQ, nperseg=SFREQ * 2)
        rs_scorr.append(_r(np.log10(pt + 1e-30), np.log10(ps + 1e-30)))
        rz_scorr.append(_r(np.log10(pt + 1e-30), np.log10(pz + 1e-30)))

        for band, (lo, hi) in BANDS.items():
            bp_t = _band_power(t, SFREQ, lo, hi)
            bp_s = _band_power(s, SFREQ, lo, hi)
            bp_z = _band_power(z, SFREQ, lo, hi)
            rs_bp[band].append(abs(10*np.log10(bp_t+1e-30) - 10*np.log10(bp_s+1e-30)))
            rz_bp[band].append(abs(10*np.log10(bp_t+1e-30) - 10*np.log10(bp_z+1e-30)))

    def _m(lst):
        finite = [x for x in lst if np.isfinite(x)]
        return float(np.mean(finite)) if finite else 0.0

    return {
        "r_temporal":  (_m(rs_tcorr), _m(rz_tcorr)),
        "r_spectral":  (_m(rs_scorr), _m(rz_scorr)),
        "rmse":        (_m(rs_rmse),  _m(rz_rmse)),
        "sdr":         (_m(rs_sdr),   _m(rz_sdr)),
        "band_power":  {b: (_m(rs_bp[b]), _m(rz_bp[b])) for b in BANDS},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-N figures
# ══════════════════════════════════════════════════════════════════════════════
def save_per_n_figures(truth, X_spline, X_zuna, ch_names, drop_names,
                       drop_indices, n_drop, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_epochs = truth.shape[0]

    # ── Fig A: Time series for first 2 dropped channels (2s, epoch 0) ────────
    n_show = min(4, n_drop)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    t_axis = np.arange(int(2.0 * SFREQ)) / SFREQ
    for ax, idx, ch in zip(axes, drop_indices[:n_show], drop_names[:n_show]):
        t_sig = truth[0, idx, :len(t_axis)]
        s_sig = X_spline[0, idx, :len(t_axis)]
        z_sig = X_zuna[0, idx, :len(t_axis)]
        r_s = _r(truth[:, idx, :].ravel(), X_spline[:, idx, :].ravel())
        r_z = _r(truth[:, idx, :].ravel(), X_zuna[:, idx, :].ravel())
        ax.plot(t_axis, t_sig, "k-",  lw=2.0, label="Truth",  zorder=3)
        ax.plot(t_axis, s_sig, "r--", lw=1.5, alpha=0.85, label=f"Spline r={r_s:.3f}")
        ax.plot(t_axis, z_sig, "b-",  lw=1.5, alpha=0.85, label=f"ZUNA   r={r_z:.3f}")
        ax.set_ylabel("µV")
        ax.set_title(f"Channel {ch}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"N={n_drop} dropout — Time Series (first {n_show} dropped channels, epoch 1)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A_timeseries.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Fig B: PSD for first 4 dropped channels ───────────────────────────────
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4), sharey=False)
    if n_show == 1:
        axes = [axes]
    for ax, idx, ch in zip(axes, drop_indices[:n_show], drop_names[:n_show]):
        fq, pt = welch(truth[:, idx, :].ravel(),    fs=SFREQ, nperseg=SFREQ * 2)
        _,  ps = welch(X_spline[:, idx, :].ravel(), fs=SFREQ, nperseg=SFREQ * 2)
        _,  pz = welch(X_zuna[:, idx, :].ravel(),   fs=SFREQ, nperseg=SFREQ * 2)
        ax.plot(fq, 10*np.log10(pt+1e-30), "k-",  lw=2.0, label="Truth")
        ax.plot(fq, 10*np.log10(ps+1e-30), "r--", lw=1.5, alpha=0.85, label="Spline")
        ax.plot(fq, 10*np.log10(pz+1e-30), "b-",  lw=1.5, alpha=0.85, label="ZUNA")
        ax.axvspan(8, 13, color="gray", alpha=0.10)
        ax.set_xlim(1, 50)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title(ch)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle(f"N={n_drop} dropout — PSD", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "B_psd.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Fig C: Per-epoch temporal correlation for all dropped channels ────────
    fig, axes = plt.subplots(1, min(n_drop, 4), figsize=(5 * min(n_drop, 4), 4))
    if n_drop == 1:
        axes = [axes]
    axes = np.array(axes).ravel()
    ep_ax = np.arange(1, n_epochs + 1)
    for ax, idx, ch in zip(axes, drop_indices[:4], drop_names[:4]):
        rs = [_r(truth[ep, idx, :], X_spline[ep, idx, :]) for ep in range(n_epochs)]
        rz = [_r(truth[ep, idx, :], X_zuna[ep, idx, :])   for ep in range(n_epochs)]
        ax.plot(ep_ax, rs, "rs--", ms=6, lw=1.2, label=f"Spline ({np.mean(rs):.3f})")
        ax.plot(ep_ax, rz, "b^-",  ms=6, lw=1.2, label=f"ZUNA   ({np.mean(rz):.3f})")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_ylim(-0.15, 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r")
        ax.set_title(ch)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle(f"N={n_drop} dropout — Per-Epoch Correlation", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_epoch_corr.png"), dpi=200, bbox_inches="tight")
    plt.close()

    log.info(f"    Saved figures → {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate metrics across trials  →  mean ± std per scalar key
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_trials(trial_metrics_list):
    """
    Given a list of metrics dicts (one per trial), return a single dict with
    (mean, std) tuples instead of (spline_val, zuna_val) tuples, shaped as:
      agg[key] = ((spline_mean, spline_std), (zuna_mean, zuna_std))
      agg["band_power"][band] = ((s_mean, s_std), (z_mean, z_std))
    """
    scalar_keys = ["r_temporal", "r_spectral", "rmse", "sdr"]
    agg = {}
    for key in scalar_keys:
        s_vals = [m[key][0] for m in trial_metrics_list]
        z_vals = [m[key][1] for m in trial_metrics_list]
        agg[key] = (
            (float(np.mean(s_vals)), float(np.std(s_vals))),
            (float(np.mean(z_vals)), float(np.std(z_vals))),
        )
    agg["band_power"] = {}
    for band in BANDS:
        s_vals = [m["band_power"][band][0] for m in trial_metrics_list]
        z_vals = [m["band_power"][band][1] for m in trial_metrics_list]
        agg["band_power"][band] = (
            (float(np.mean(s_vals)), float(np.std(s_vals))),
            (float(np.mean(z_vals)), float(np.std(z_vals))),
        )
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Summary sweep figure with error bars (mean ± std across trials)
# ══════════════════════════════════════════════════════════════════════════════
def save_sweep_summary(all_n, agg_metrics, out_dir, n_trials=1):
    """
    agg_metrics[n][key] = ((s_mean, s_std), (z_mean, z_std))
    When n_trials == 1 the std values are 0 and no error bars are drawn.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_vals = all_n

    def means(key, idx):
        return [agg_metrics[n][key][idx][0] for n in n_vals]

    def stds(key, idx):
        return [agg_metrics[n][key][idx][1] for n in n_vals]

    def errbar(ax, key, idx, color, marker, label):
        mu  = np.array(means(key, idx))
        sig = np.array(stds(key, idx))
        ax.errorbar(n_vals, mu, yerr=sig if n_trials > 1 else None,
                    color=color, marker=marker, lw=2, ms=8,
                    capsize=5, capthick=1.5, label=label)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    errbar(axes[0], "r_temporal", 0, "red",  "o", "Spline")
    errbar(axes[0], "r_temporal", 1, "blue", "^", "ZUNA")
    axes[0].set_title("Temporal Correlation (r)")
    axes[0].set_ylabel("Pearson r")
    axes[0].set_ylim(0, 1.05)

    errbar(axes[1], "r_spectral", 0, "red",  "o", "Spline")
    errbar(axes[1], "r_spectral", 1, "blue", "^", "ZUNA")
    axes[1].set_title("Spectral Correlation (r)")
    axes[1].set_ylabel("Pearson r")
    axes[1].set_ylim(0, 1.05)

    errbar(axes[2], "rmse", 0, "red",  "o", "Spline")
    errbar(axes[2], "rmse", 1, "blue", "^", "ZUNA")
    axes[2].set_title("RMSE (µV)  ↓ better")
    axes[2].set_ylabel("µV")

    errbar(axes[3], "sdr", 0, "red",  "o", "Spline")
    errbar(axes[3], "sdr", 1, "blue", "^", "ZUNA")
    axes[3].set_title("SDR (dB)  ↑ better")
    axes[3].set_ylabel("dB")

    # Alpha band power
    alpha_s_mu  = [agg_metrics[n]["band_power"]["Alpha"][0][0] for n in n_vals]
    alpha_s_sig = [agg_metrics[n]["band_power"]["Alpha"][0][1] for n in n_vals]
    alpha_z_mu  = [agg_metrics[n]["band_power"]["Alpha"][1][0] for n in n_vals]
    alpha_z_sig = [agg_metrics[n]["band_power"]["Alpha"][1][1] for n in n_vals]
    axes[4].errorbar(n_vals, alpha_s_mu, yerr=alpha_s_sig if n_trials > 1 else None,
                     color="red",  marker="o", lw=2, ms=8, capsize=5, capthick=1.5, label="Spline")
    axes[4].errorbar(n_vals, alpha_z_mu, yerr=alpha_z_sig if n_trials > 1 else None,
                     color="blue", marker="^", lw=2, ms=8, capsize=5, capthick=1.5, label="ZUNA")
    axes[4].set_title("Alpha Band Power Error (dB)  ↓ better")
    axes[4].set_ylabel("dB")

    for ax in axes[:5]:
        ax.set_xlabel("Channels dropped (N)")
        ax.set_xticks(n_vals)
        ax.legend()
        ax.grid(alpha=0.25)

    # Text summary table in 6th panel
    axes[5].axis("off")
    trial_note = f"  (mean ± std, {n_trials} trials per N)" if n_trials > 1 else ""
    lines = [f"Summary — reconstructed channels only{trial_note}\n"]
    hdr = f"{'N':>4}  {'Tcorr-S':>13}  {'Tcorr-Z':>13}  {'RMSE-S':>12}  {'RMSE-Z':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for n in n_vals:
        m = agg_metrics[n]
        sm, ss = m["r_temporal"][0]
        zm, zs = m["r_temporal"][1]
        rm_s, rs_s = m["rmse"][0]
        rm_z, rs_z = m["rmse"][1]
        if n_trials > 1:
            lines.append(
                f"{n:>4}  {sm:>6.3f}±{ss:<5.3f}  {zm:>6.3f}±{zs:<5.3f}"
                f"  {rm_s:>5.2f}±{rs_s:<4.2f}  {rm_z:>5.2f}±{rs_z:<4.2f}"
            )
        else:
            lines.append(f"{n:>4}  {sm:>13.4f}  {zm:>13.4f}  {rm_s:>12.3f}  {rm_z:>12.3f}")
    axes[5].text(0.02, 0.98, "\n".join(lines), transform=axes[5].transAxes,
                 fontsize=8.5, verticalalignment="top", fontfamily="monospace")

    title_suffix = f"  [{n_trials} trials/N]" if n_trials > 1 else ""
    plt.suptitle(
        f"ZUNA vs Spline — Random Channel Dropout Sweep  ({SUBJECT_ID} {SESSION_NAME}){title_suffix}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "SWEEP_SUMMARY.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  Sweep summary → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_drop",   nargs="+", type=int, default=[2, 4, 8],
                        help="N values to sweep, e.g. --n_drop 2 4 8 16")
    parser.add_argument("--n_trials", type=int, default=1,
                        help="How many independent random draws per N (default 1)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Base random seed; trial t of N channels uses seed + N + t*1000")
    parser.add_argument("--steps",    type=int, default=DIFFUSION_STEPS)
    parser.add_argument("--epochs",   type=int, default=NUM_EPOCHS)
    parser.add_argument("--channels", nargs="+", default=None,
                        help="Manual channel list for n_drop[0], trial 0 only")
    args = parser.parse_args()

    n_steps   = args.steps
    n_epochs  = args.epochs
    n_trials  = args.n_trials

    t_total = time.time()
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════════╗")
    log.info("║        ZUNA N-CHANNEL RANDOM DROPOUT SWEEP                      ║")
    log.info("╚══════════════════════════════════════════════════════════════════╝")
    log.info(f"  N values    : {args.n_drop}")
    log.info(f"  Trials/N    : {n_trials}")
    log.info(f"  Base seed   : {args.seed}")
    log.info(f"  Diff. steps : {n_steps}")
    log.info(f"  Epochs      : {n_epochs}")
    log.info(f"  Total runs  : {len(args.n_drop) * n_trials}")
    log.info("")

    truth, ch_names, positions = load_shared_data(n_epochs)

    # agg_metrics[n] = aggregated metrics (mean±std across trials)
    agg_metrics = {}

    for i, n in enumerate(args.n_drop):
        log.info("")
        log.info("━" * 72)
        log.info(f"  N = {n}  ({n_trials} trial{'s' if n_trials > 1 else ''})")
        log.info("━" * 72)

        trial_metrics = []

        for t in range(n_trials):
            seed   = args.seed + n + t * 1000
            manual = args.channels if (i == 0 and t == 0 and args.channels) else None
            drop_names, drop_indices = select_channels(ch_names, n, seed, manual)

            log.info(f"  Trial {t+1}/{n_trials}  seed={seed}  dropping: {drop_names}")

            X_spline = run_spline(truth, ch_names, positions, drop_indices)
            X_zuna   = run_zuna(truth, ch_names, positions, drop_indices,
                                run_tag=f"n{n}_t{t}", n_steps=n_steps)

            m = compute_metrics(truth, X_spline, X_zuna, drop_indices)
            trial_metrics.append(m)

            # Save per-trial figures into subdirectory
            trial_dir = os.path.join(RESULTS_ROOT, f"N{n:02d}_drop", f"trial_{t+1:02d}")
            save_per_n_figures(truth, X_spline, X_zuna, ch_names,
                               drop_names, drop_indices, n, trial_dir)

            log.info(f"    Tcorr  Spline={m['r_temporal'][0]:.4f}  ZUNA={m['r_temporal'][1]:.4f}"
                     f"  |  RMSE  Spline={m['rmse'][0]:.3f}  ZUNA={m['rmse'][1]:.3f}")

        # Aggregate across trials
        agg = aggregate_trials(trial_metrics)
        agg_metrics[n] = agg

        # Print aggregated summary for this N
        log.info(f"")
        log.info(f"  ── N={n} aggregate (mean ± std, {n_trials} trials) ──")
        log.info(f"  {'Metric':<22}  {'Spline mean±std':>18}  {'ZUNA mean±std':>18}")
        log.info(f"  {'-'*62}")
        for key, label in [("r_temporal","Temporal corr (r)"), ("r_spectral","Spectral corr (r)"),
                            ("rmse","RMSE (µV)"), ("sdr","SDR (dB)")]:
            sm, ss = agg[key][0]
            zm, zs = agg[key][1]
            log.info(f"  {label:<22}  {sm:>8.4f} ± {ss:<7.4f}  {zm:>8.4f} ± {zs:<7.4f}")
        for band in BANDS:
            sm, ss = agg["band_power"][band][0]
            zm, zs = agg["band_power"][band][1]
            log.info(f"  {'  '+band+' BP err (dB)':<22}  {sm:>8.4f} ± {ss:<7.4f}  {zm:>8.4f} ± {zs:<7.4f}")

    # Sweep summary figure
    if len(args.n_drop) > 1:
        save_sweep_summary(args.n_drop, agg_metrics, RESULTS_ROOT, n_trials=n_trials)

    elapsed = time.time() - t_total
    log.info("")
    log.info("=" * 72)
    log.info(f"SWEEP COMPLETE — Total time: {int(elapsed)//60}m {int(elapsed)%60}s")
    log.info(f"Results → {os.path.abspath(RESULTS_ROOT)}/")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
