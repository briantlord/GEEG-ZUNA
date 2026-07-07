"""
compare_outputs.py — Comprehensive ZUNA vs Truth vs Spline Comparison
======================================================================
Generates per-channel and aggregate metrics across preserved and
reconstructed channels, with publication-quality figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr
import os, sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ────────────────────────────────────────────────────────────
SUBJECT_ID   = "G001"
SESSION_NAME = "Day1Rest1"
SFREQ        = 256
NUM_EPOCHS   = 16

# Standard EEG frequency bands
BANDS = {
    'Delta':  (1,  4),
    'Theta':  (4,  8),
    'Alpha':  (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 50),
}

# The 19 channels preserved from the 10-20 system (uppercased for matching)
KEEP_19 = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7',  'C3',  'CZ', 'C4', 'T8',
    'P7',  'P3',  'PZ', 'P4', 'P8',
    'O1',  'O2',
]

OUT_DIR = "comparison_results"


def load_data():
    """Load truth, spline, and ZUNA arrays + channel names."""
    truth  = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_y_truth.npy")[:NUM_EPOCHS]
    spline = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_X_spline.npy")[:NUM_EPOCHS]
    zuna   = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_X_zuna_test.npy")[:NUM_EPOCHS]

    # Recover channel names from the staged .pt file
    import torch, glob
    pt_file = sorted(glob.glob("test_pt_out/*.pt"))[0]
    meta = torch.load(pt_file, weights_only=False).get('metadata', {})
    ch_names = meta.get('ch_names', [f"Ch{i}" for i in range(truth.shape[1])])
    # Channel names in the tensor are alphabetically sorted
    ch_names_sorted = sorted(ch_names)

    return truth, spline, zuna, ch_names_sorted


def classify_channels(ch_names):
    """Return boolean mask: True = preserved (kept), False = reconstructed."""
    return np.array([ch.upper() in KEEP_19 for ch in ch_names])


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 1: Per-Channel Temporal Correlation
# ═══════════════════════════════════════════════════════════════════════════════
def temporal_correlation(truth, compare, ch_names):
    """
    Pearson r between truth and compare time series, per channel.
    Concatenates all epochs into one long vector per channel.
    """
    n_ch = truth.shape[1]
    correlations = np.zeros(n_ch)
    for c in range(n_ch):
        t_flat = truth[:, c, :].ravel()
        c_flat = compare[:, c, :].ravel()
        correlations[c], _ = pearsonr(t_flat, c_flat)
    return correlations


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 2: Per-Channel RMSE (µV)
# ═══════════════════════════════════════════════════════════════════════════════
def per_channel_rmse(truth, compare):
    """RMSE in µV between truth and compare, per channel."""
    diff = truth - compare
    return np.sqrt(np.mean(diff ** 2, axis=(0, 2)))  # mean over epochs and time


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 3: Per-Channel Relative RMSE (% of truth RMS)
# ═══════════════════════════════════════════════════════════════════════════════
def per_channel_relative_rmse(truth, compare):
    """RMSE as a percentage of truth channel RMS."""
    rmse = per_channel_rmse(truth, compare)
    truth_rms = np.sqrt(np.mean(truth ** 2, axis=(0, 2)))
    return (rmse / truth_rms) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 4: Per-Channel Spectral (PSD) Correlation
# ═══════════════════════════════════════════════════════════════════════════════
def spectral_correlation(truth, compare, sfreq):
    """
    Pearson r between the log-PSD curves of truth and compare, per channel.
    Measures whether the spectral *shape* is preserved.
    """
    n_ch = truth.shape[1]
    correlations = np.zeros(n_ch)
    for c in range(n_ch):
        _, psd_t = welch(truth[:, c, :].ravel(), fs=sfreq, nperseg=sfreq * 2)
        _, psd_c = welch(compare[:, c, :].ravel(), fs=sfreq, nperseg=sfreq * 2)
        correlations[c], _ = pearsonr(np.log10(psd_t + 1e-30), np.log10(psd_c + 1e-30))
    return correlations


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 5: Band Power Error (dB)
# ═══════════════════════════════════════════════════════════════════════════════
def band_power_error(truth, compare, sfreq, bands):
    """
    Absolute difference in log band power (dB) between truth and compare,
    averaged across channels. Returns dict of {band_name: (mean_err, std_err)}.
    """
    n_ch = truth.shape[1]
    freqs, psd_t = welch(truth, fs=sfreq, nperseg=sfreq * 2, axis=-1)
    _,     psd_c = welch(compare, fs=sfreq, nperseg=sfreq * 2, axis=-1)

    # Average over epochs: shape becomes (channels, freqs)
    psd_t_avg = psd_t.mean(axis=0)
    psd_c_avg = psd_c.mean(axis=0)

    results = {}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bp_t = 10 * np.log10(psd_t_avg[:, mask].mean(axis=-1) + 1e-30)
        bp_c = 10 * np.log10(psd_c_avg[:, mask].mean(axis=-1) + 1e-30)
        err = np.abs(bp_t - bp_c)
        results[band_name] = (err.mean(), err.std())
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 6: Signal-to-Distortion Ratio (SDR, dB)
# ═══════════════════════════════════════════════════════════════════════════════
def signal_to_distortion(truth, compare):
    """
    SDR per channel in dB: 10 * log10( ||truth||^2 / ||truth - compare||^2 ).
    Higher is better.
    """
    n_ch = truth.shape[1]
    sdr = np.zeros(n_ch)
    for c in range(n_ch):
        sig_power = np.sum(truth[:, c, :] ** 2)
        noise_power = np.sum((truth[:, c, :] - compare[:, c, :]) ** 2)
        if noise_power < 1e-30:
            sdr[c] = np.inf
        else:
            sdr[c] = 10 * np.log10(sig_power / noise_power)
    return sdr


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 7: Epoch-wise Spatial Correlation (Topographic Similarity)
# ═══════════════════════════════════════════════════════════════════════════════
def spatial_correlation_per_timepoint(truth, compare, n_samples=500):
    """
    At random time points, compute Pearson r across channels (spatial pattern).
    Returns array of correlations. Measures topographic fidelity.
    """
    n_epochs, n_ch, n_time = truth.shape
    rng = np.random.default_rng(42)
    corrs = []
    for _ in range(n_samples):
        ep = rng.integers(0, n_epochs)
        tp = rng.integers(0, n_time)
        r, _ = pearsonr(truth[ep, :, tp], compare[ep, :, tp])
        corrs.append(r)
    return np.array(corrs)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_channel_bars(vals_spline, vals_zuna, ch_names, preserved_mask,
                          ylabel, title, filename, higher_is_better=True):
    """Side-by-side bar chart for a per-channel metric, colored by preserved/reconstructed."""
    n = len(ch_names)
    x = np.arange(n)
    w = 0.38

    fig, ax = plt.subplots(figsize=(18, 5))

    colors_s = ['#2d8a4e' if p else '#c0392b' for p in preserved_mask]
    colors_z = ['#27ae60' if p else '#e74c3c' for p in preserved_mask]

    ax.bar(x - w/2, vals_spline, w, color=colors_s, alpha=0.7, label='Spline')
    ax.bar(x + w/2, vals_zuna,   w, color=colors_z, alpha=0.7, label='ZUNA')

    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Add divider annotation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.7, label='Preserved channel'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Reconstructed channel'),
    ]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


def plot_spatial_correlation_hist(corrs_spline, corrs_zuna, filename):
    """Histogram of spatial (topographic) correlations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(corrs_spline, bins=50, alpha=0.6, color='red', label=f'Spline (median={np.median(corrs_spline):.4f})')
    ax.hist(corrs_zuna, bins=50, alpha=0.6, color='blue', label=f'ZUNA (median={np.median(corrs_zuna):.4f})')
    ax.set_xlabel('Spatial Pearson r')
    ax.set_ylabel('Count')
    ax.set_title('Topographic Fidelity — Spatial Correlation at Random Time Points')
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


def plot_band_power_table(bp_spline, bp_zuna, filename):
    """Bar chart of band power error by frequency band."""
    bands = list(bp_spline.keys())
    means_s = [bp_spline[b][0] for b in bands]
    stds_s  = [bp_spline[b][1] for b in bands]
    means_z = [bp_zuna[b][0] for b in bands]
    stds_z  = [bp_zuna[b][1] for b in bands]

    x = np.arange(len(bands))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, means_s, w, yerr=stds_s, color='red', alpha=0.7, capsize=4, label='Spline')
    ax.bar(x + w/2, means_z, w, yerr=stds_z, color='blue', alpha=0.7, capsize=4, label='ZUNA')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel('Absolute Band Power Error (dB)')
    ax.set_title('Band Power Reconstruction Error by Frequency Band')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()


def plot_psd_comparison(truth, spline, zuna, ch_names, preserved_mask, sfreq, filename):
    """
    4-panel PSD: preserved-avg, reconstructed-avg, best recon channel, worst recon channel.
    """
    freqs, psd_t = welch(truth,  fs=sfreq, nperseg=sfreq * 2, axis=-1)
    _,     psd_s = welch(spline, fs=sfreq, nperseg=sfreq * 2, axis=-1)
    _,     psd_z = welch(zuna,   fs=sfreq, nperseg=sfreq * 2, axis=-1)

    # Average over epochs
    psd_t = psd_t.mean(axis=0)  # (channels, freqs)
    psd_s = psd_s.mean(axis=0)
    psd_z = psd_z.mean(axis=0)

    recon_mask = ~preserved_mask
    recon_idx = np.where(recon_mask)[0]

    # Compute spectral correlation for reconstructed channels to find best/worst
    spec_corr_z = np.zeros(len(ch_names))
    for c in recon_idx:
        spec_corr_z[c], _ = pearsonr(np.log10(psd_t[c] + 1e-30), np.log10(psd_z[c] + 1e-30))

    best_recon  = recon_idx[np.argmax(spec_corr_z[recon_idx])]
    worst_recon = recon_idx[np.argmin(spec_corr_z[recon_idx])]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    def _plot_psd(ax, t, s, z, title):
        ax.plot(freqs, 10 * np.log10(t + 1e-30), 'k-', lw=1.5, label='Truth')
        ax.plot(freqs, 10 * np.log10(s + 1e-30), 'r--', lw=1, label='Spline')
        ax.plot(freqs, 10 * np.log10(z + 1e-30), 'b-', alpha=0.7, lw=1, label='ZUNA')
        ax.axvspan(8, 13, color='gray', alpha=0.08)
        ax.set_xlim(1, 50)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # Panel 1: Preserved channels average
    pres_idx = np.where(preserved_mask)[0]
    _plot_psd(axes[0, 0],
              psd_t[pres_idx].mean(axis=0),
              psd_s[pres_idx].mean(axis=0),
              psd_z[pres_idx].mean(axis=0),
              f'Preserved Channels Average (n={len(pres_idx)})')

    # Panel 2: Reconstructed channels average
    _plot_psd(axes[0, 1],
              psd_t[recon_idx].mean(axis=0),
              psd_s[recon_idx].mean(axis=0),
              psd_z[recon_idx].mean(axis=0),
              f'Reconstructed Channels Average (n={len(recon_idx)})')

    # Panel 3: Best reconstructed channel
    _plot_psd(axes[1, 0],
              psd_t[best_recon], psd_s[best_recon], psd_z[best_recon],
              f'Best Reconstructed: {ch_names[best_recon]} (r={spec_corr_z[best_recon]:.4f})')

    # Panel 4: Worst reconstructed channel
    _plot_psd(axes[1, 1],
              psd_t[worst_recon], psd_s[worst_recon], psd_z[worst_recon],
              f'Worst Reconstructed: {ch_names[worst_recon]} (r={spec_corr_z[worst_recon]:.4f})')

    plt.suptitle('Spectral Fidelity — Preserved vs Reconstructed Channels', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
def plot_all_channels_timeseries(truth, spline, zuna, ch_names, sfreq, prefix_filename, epoch_idx=0, duration_s=1.5):
    """Plot time series for epoch_idx of all channels in batches of 16."""
    samples = int(duration_s * sfreq)
    t = np.arange(samples) / sfreq
    
    n_ch = len(ch_names)
    channels_per_fig = 16
    n_figs = int(np.ceil(n_ch / channels_per_fig))
    
    for f in range(n_figs):
        fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
        axes = axes.flatten()
        
        start_idx = f * channels_per_fig
        end_idx = min(start_idx + channels_per_fig, n_ch)
        
        for i, c in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            ch_name = ch_names[c]
            
            sig_t = truth[epoch_idx, c, :samples]
            sig_s = spline[epoch_idx, c, :samples]
            sig_z = zuna[epoch_idx, c, :samples]
            
            if np.std(sig_t) > 0 and np.std(sig_s) > 0:
                r_s, _ = pearsonr(sig_t, sig_s)
            else:
                r_s = 1.0 if np.allclose(sig_t, sig_s) else 0.0
                
            if np.std(sig_t) > 0 and np.std(sig_z) > 0:
                r_z, _ = pearsonr(sig_t, sig_z)
            else:
                r_z = 1.0 if np.allclose(sig_t, sig_z) else 0.0
            
            ax.plot(t, sig_t, 'k-', lw=1.5, label='Truth')
            ax.plot(t, sig_s, 'b-', lw=1.0, alpha=0.8, label='Spline')
            ax.plot(t, sig_z, 'r-', lw=1.0, alpha=0.8, label='ZUNA')
            
            ax.set_title(f'{ch_name}\nSpline r: {r_s:.3f} | ZUNA r: {r_z:.3f}', fontsize=9)
            ax.set_ylabel('µV')
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
                
        # Turn off empty subplots
        for i in range(end_idx - start_idx, 16):
            axes[i].axis('off')
            
        plt.suptitle(f'First {duration_s}s of Epoch {epoch_idx+1} - Part {f+1}', y=0.95, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{prefix_filename}_part{f+1}.png"), dpi=200, bbox_inches='tight')
        plt.close()


def plot_all_channels_psd(truth, spline, zuna, ch_names, sfreq, prefix_filename, epoch_idx=0):
    """Plot PSD for epoch_idx for all channels in batches of 16."""
    n_ch = len(ch_names)
    channels_per_fig = 16
    n_figs = int(np.ceil(n_ch / channels_per_fig))
    
    for f in range(n_figs):
        fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
        axes = axes.flatten()
        
        start_idx = f * channels_per_fig
        end_idx = min(start_idx + channels_per_fig, n_ch)
        
        for i, c in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            ch_name = ch_names[c]
            
            # PSD for the single epoch
            f_t, p_t = welch(truth[epoch_idx, c, :], fs=sfreq, nperseg=sfreq * 2)
            f_s, p_s = welch(spline[epoch_idx, c, :], fs=sfreq, nperseg=sfreq * 2)
            f_z, p_z = welch(zuna[epoch_idx, c, :], fs=sfreq, nperseg=sfreq * 2)
            
            p_t_log = 10 * np.log10(p_t + 1e-30)
            p_s_log = 10 * np.log10(p_s + 1e-30)
            p_z_log = 10 * np.log10(p_z + 1e-30)
            
            ax.plot(f_t, p_t_log, 'k-', lw=1.5, label='Truth')
            ax.plot(f_s, p_s_log, 'b-', lw=1.0, alpha=0.8, label='Spline')
            ax.plot(f_z, p_z_log, 'r-', lw=1.0, alpha=0.8, label='ZUNA')
            
            ax.axvspan(8, 13, color='gray', alpha=0.08)
            ax.set_xlim(1, 50)
            
            # Reduce y-axis size: find min/max in the 1-50Hz range
            mask = (f_t >= 1) & (f_t <= 50)
            min_y = min(p_t_log[mask].min(), p_s_log[mask].min(), p_z_log[mask].min())
            max_y = max(p_t_log[mask].max(), p_s_log[mask].max(), p_z_log[mask].max())
            ax.set_ylim(min_y - 2, max_y + 5)
            
            ax.set_title(f'{ch_name}')
            ax.set_ylabel('Power (dB)')
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
                
        # Turn off empty subplots
        for i in range(end_idx - start_idx, 16):
            axes[i].axis('off')
            
        plt.suptitle(f'Frequency Spectra of Epoch {epoch_idx+1} - Part {f+1}', y=0.95, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{prefix_filename}_part{f+1}.png"), dpi=200, bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data ...")
    truth, spline, zuna, ch_names = load_data()
    preserved = classify_channels(ch_names)
    n_pres = preserved.sum()
    n_recon = (~preserved).sum()

    print(f"  Channels: {len(ch_names)} total, {n_pres} preserved, {n_recon} reconstructed")
    print(f"  Epochs:   {truth.shape[0]}")
    print(f"  Samples:  {truth.shape[2]} ({truth.shape[2]/SFREQ:.1f}s @ {SFREQ}Hz)")
    print()

    # ── 1. Temporal Correlation ──────────────────────────────────────────────
    print("Computing temporal correlations ...")
    tcorr_s = temporal_correlation(truth, spline, ch_names)
    tcorr_z = temporal_correlation(truth, zuna, ch_names)

    plot_per_channel_bars(tcorr_s, tcorr_z, ch_names, preserved,
                          'Pearson r', 'Per-Channel Temporal Correlation with Truth',
                          '1_temporal_correlation.png')

    # ── 2. RMSE ──────────────────────────────────────────────────────────────
    print("Computing RMSE ...")
    rmse_s = per_channel_rmse(truth, spline)
    rmse_z = per_channel_rmse(truth, zuna)

    plot_per_channel_bars(rmse_s, rmse_z, ch_names, preserved,
                          'RMSE (µV)', 'Per-Channel RMSE vs Truth',
                          '2_rmse.png', higher_is_better=False)

    # ── 3. Relative RMSE ─────────────────────────────────────────────────────
    print("Computing relative RMSE ...")
    rrmse_s = per_channel_relative_rmse(truth, spline)
    rrmse_z = per_channel_relative_rmse(truth, zuna)

    plot_per_channel_bars(rrmse_s, rrmse_z, ch_names, preserved,
                          'Relative RMSE (%)', 'Per-Channel Relative RMSE vs Truth',
                          '3_relative_rmse.png', higher_is_better=False)

    # ── 4. Spectral Correlation ──────────────────────────────────────────────
    print("Computing spectral correlations ...")
    scorr_s = spectral_correlation(truth, spline, SFREQ)
    scorr_z = spectral_correlation(truth, zuna, SFREQ)

    plot_per_channel_bars(scorr_s, scorr_z, ch_names, preserved,
                          'Spectral Pearson r', 'Per-Channel Spectral Correlation with Truth',
                          '4_spectral_correlation.png')

    # ── 5. Band Power Error ──────────────────────────────────────────────────
    print("Computing band power errors ...")
    bp_s = band_power_error(truth, spline, SFREQ, BANDS)
    bp_z = band_power_error(truth, zuna, SFREQ, BANDS)

    plot_band_power_table(bp_s, bp_z, '5_band_power_error.png')

    # ── 6. Signal-to-Distortion Ratio ────────────────────────────────────────
    print("Computing SDR ...")
    sdr_s = signal_to_distortion(truth, spline)
    sdr_z = signal_to_distortion(truth, zuna)

    # Cap infinite SDR for plotting (preserved channels = perfect match)
    sdr_s_plot = np.clip(sdr_s, -20, 60)
    sdr_z_plot = np.clip(sdr_z, -20, 60)

    plot_per_channel_bars(sdr_s_plot, sdr_z_plot, ch_names, preserved,
                          'SDR (dB)', 'Per-Channel Signal-to-Distortion Ratio',
                          '6_sdr.png')

    # ── 7. Spatial (Topographic) Correlation ─────────────────────────────────
    print("Computing spatial correlations ...")
    scor_s = spatial_correlation_per_timepoint(truth, spline)
    scor_z = spatial_correlation_per_timepoint(truth, zuna)

    plot_spatial_correlation_hist(scor_s, scor_z, '7_spatial_correlation.png')

    # ── 8. PSD Comparison Panel ──────────────────────────────────────────────
    print("Generating PSD comparison panels ...")
    plot_psd_comparison(truth, spline, zuna, ch_names, preserved, SFREQ,
                        '8_psd_panels.png')

    # ── 9. All Channels 1.5s Time Series ─────────────────────────────────────
    print("Generating all channels 1.5s time series (Epoch 1) ...")
    plot_all_channels_timeseries(truth, spline, zuna, ch_names, SFREQ,
                                 '9_all_channels_1.5s_timeseries', epoch_idx=0, duration_s=1.5)

    # ── 10. All Channels First Epoch PSD ─────────────────────────────────────
    print("Generating all channels first epoch PSD (Epoch 1) ...")
    plot_all_channels_psd(truth, spline, zuna, ch_names, SFREQ,
                          '10_all_channels_epoch1_psd', epoch_idx=0)

    # ── 11. All Channels 1s Time Series (Epoch 2) ────────────────────────────
    print("Generating all channels 1s time series (Epoch 2) ...")
    plot_all_channels_timeseries(truth, spline, zuna, ch_names, SFREQ,
                                 '11_all_channels_1s_timeseries_epoch2', epoch_idx=1, duration_s=1.0)

    # ── 12. All Channels Second Epoch PSD (Epoch 2) ──────────────────────────
    print("Generating all channels second epoch PSD (Epoch 2) ...")
    plot_all_channels_psd(truth, spline, zuna, ch_names, SFREQ,
                          '12_all_channels_epoch2_psd', epoch_idx=1)

    # ── 13. All Channels 512ms Time Series (Epoch 2) ─────────────────────────
    print("Generating all channels 512ms time series (Epoch 2) ...")
    plot_all_channels_timeseries(truth, spline, zuna, ch_names, SFREQ,
                                 '13_all_channels_512ms_timeseries_epoch2', epoch_idx=1, duration_s=0.512)

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 80)
    print()

    def _fmt(label, val_s_pres, val_z_pres, val_s_rec, val_z_rec, fmt=".4f"):
        ps = f"{val_s_pres:{fmt}}"
        pz = f"{val_z_pres:{fmt}}"
        rs = f"{val_s_rec:{fmt}}"
        rz = f"{val_z_rec:{fmt}}"
        print(f"  {label:<30s}  {ps:>10s}  {pz:>10s}  |  {rs:>10s}  {rz:>10s}")

    header = f"  {'Metric':<30s}  {'Spline':>10s}  {'ZUNA':>10s}  |  {'Spline':>10s}  {'ZUNA':>10s}"
    subhdr = f"  {'':<30s}  {'PRESERVED':^22s}  |  {'RECONSTRUCTED':^22s}"
    sep = "  " + "-" * 78

    print(subhdr)
    print(header)
    print(sep)

    _fmt("Temporal Corr (mean r)",
         tcorr_s[preserved].mean(), tcorr_z[preserved].mean(),
         tcorr_s[~preserved].mean(), tcorr_z[~preserved].mean())

    _fmt("RMSE (µV)",
         rmse_s[preserved].mean(), rmse_z[preserved].mean(),
         rmse_s[~preserved].mean(), rmse_z[~preserved].mean())

    _fmt("Relative RMSE (%)",
         rrmse_s[preserved].mean(), rrmse_z[preserved].mean(),
         rrmse_s[~preserved].mean(), rrmse_z[~preserved].mean())

    _fmt("Spectral Corr (mean r)",
         scorr_s[preserved].mean(), scorr_z[preserved].mean(),
         scorr_s[~preserved].mean(), scorr_z[~preserved].mean())

    finite_sdr_s_p = sdr_s[preserved & np.isfinite(sdr_s)]
    finite_sdr_z_p = sdr_z[preserved & np.isfinite(sdr_z)]
    finite_sdr_s_r = sdr_s[~preserved & np.isfinite(sdr_s)]
    finite_sdr_z_r = sdr_z[~preserved & np.isfinite(sdr_z)]

    _fmt("SDR (dB, mean)",
         finite_sdr_s_p.mean() if len(finite_sdr_s_p) else float('inf'),
         finite_sdr_z_p.mean() if len(finite_sdr_z_p) else float('inf'),
         finite_sdr_s_r.mean() if len(finite_sdr_s_r) else 0,
         finite_sdr_z_r.mean() if len(finite_sdr_z_r) else 0, fmt=".2f")

    print(sep)
    print()

    # Band power summary
    print("  Band Power Error (dB, all channels):")
    print(f"  {'Band':<10s}  {'Spline':>14s}  {'ZUNA':>14s}")
    print("  " + "-" * 42)
    for band in BANDS:
        ms, ss = bp_s[band]
        mz, sz = bp_z[band]
        print(f"  {band:<10s}  {ms:>6.3f} ± {ss:<5.3f}  {mz:>6.3f} ± {sz:<5.3f}")
    print()

    # Spatial correlation summary
    print(f"  Spatial Correlation (topographic):")
    print(f"    Spline:  median = {np.median(scor_s):.4f},  mean = {np.mean(scor_s):.4f}")
    print(f"    ZUNA:    median = {np.median(scor_z):.4f},  mean = {np.mean(scor_z):.4f}")
    print()

    print("=" * 80)
    print(f"All figures saved to: {os.path.abspath(OUT_DIR)}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
