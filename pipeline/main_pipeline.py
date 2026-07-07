"""
main_pipeline.py — ZUNA EEG Reconstruction Master Script
=========================================================
Consolidates the full proof-of-concept pipeline into four numbered stages
with integrated health checks logged to pipeline_audit.log.

Usage (from activated venv):
    python main_pipeline.py
"""

import os
# ZUNA weights load from HuggingFace (Zyphra/ZUNA) and auto-download to the default HF cache.
# To use a pre-downloaded snapshot, set HF_HOME before running (e.g. export HF_HOME=/path/to/HF_cache).
if os.environ.get("HF_HOME"):
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import sys
import time
import shutil
import glob
import logging
import textwrap

import numpy as np
import torch
import mne

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

from zuna import preprocessing, inference, pt_to_fif

from load_data import (
    load_cnt_data, 
    create_epochs, 
    degrade_channels, 
    interpolate_baseline, 
    export_zuna_tensors
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SUBJECT_ID       = "G001"
SESSION_NAME     = "Day1Rest1"
CNT_FILE         = "GEEG_Raw/G001Day1Rest1.cnt"

PT_DIR           = "test_pt_out"
RECON_PT_DIR     = "test_recon_pt"
FIF_OUT_DIR      = "test_fif_out"

DIFFUSION_STEPS  = 50        # Publication quality run
TOKENS_PER_BATCH = 2680      # fallback; overridden dynamically in stage2 (10 coarse steps × n_channels per epoch)
NUM_TEST_EPOCHS  = 16        # Set to None to process ALL epochs
GPU_DEVICE       = "0"       # "0" for 1st GPU, "" for CPU

KEEP_19 = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7',  'C3',  'CZ', 'C4', 'T8',
    'P7',  'P3',  'PZ', 'P4', 'P8',
    'O1',  'O2',
]

F3_IDX = 18    # Index of F3 in the 65-channel truth array
F4_IDX = 19   # Index of F4 in the 65-channel truth array

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup — dual output: file + terminal
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("ZUNA_PIPELINE")
log.setLevel(logging.DEBUG)

fh = logging.FileHandler("pipeline_audit.log", mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(message)s"))

log.addHandler(fh)
log.addHandler(ch)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — Reference Tensor Preparation
# ══════════════════════════════════════════════════════════════════════════════
def stage0_prepare_references():
    log.info("=" * 72)
    log.info("STAGE 0: REFERENCE TENSOR PREPARATION")
    log.info("=" * 72)
    
    log.info(f"Ingesting raw file: {CNT_FILE}")
    raw_data = load_cnt_data(CNT_FILE)
    if raw_data is None:
        log.error("Failed to load CNT file. Aborting.")
        sys.exit(1)
        
    log.info("Creating epochs (5.0s, marker-locked) ...")
    epoched_data = create_epochs(raw_data, duration=5.0)
    if epoched_data is None:
        log.error("Epoching failed. Aborting.")
        sys.exit(1)
        
    log.info("Degrading to 19-channel standard montage ...")
    degraded_data = degrade_channels(epoched_data, target_montage='19_channel')
    
    log.info("Interpolating spline baseline ...")
    reconstructed_data = interpolate_baseline(degraded_data)
    
    log.info("Exporting ground truth and spline tensors ...")
    export_zuna_tensors(
        pristine_epochs=epoched_data,
        degraded_epochs=degraded_data,
        spline_epochs=reconstructed_data,
        subject_id=SUBJECT_ID,
        session_name=SESSION_NAME,
        out_dir=".",
        num_test_epochs=NUM_TEST_EPOCHS
    )
    log.info("[OK] Reference tensors exported successfully.")
    log.info("")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Dynamic 10-20 Masking
# ══════════════════════════════════════════════════════════════════════════════
def stage1_masking():
    log.info("=" * 72)
    log.info("STAGE 1: DYNAMIC 10-20 MASKING")
    log.info("=" * 72)
    
    # We rely on load_data.py injecting the perfectly aligned .pt file directly 
    # into test_pt_out/. We just need to clear the old reconstruction directory.
    for d in [FIF_OUT_DIR, RECON_PT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        
    pt_files = sorted(glob.glob(f"{PT_DIR}/*.pt"))
    if not pt_files:
        log.error("Direct injection failed: No .pt files found in test_pt_out/.")
        sys.exit(1)

    log.info(f"[OK] Preprocessing bypassed. Found {len(pt_files)} perfectly aligned staged tensor(s).")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — High-Fidelity Inference
# ══════════════════════════════════════════════════════════════════════════════
def stage2_inference():
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 2: HIGH-FIDELITY INFERENCE (HPC GPU)")
    log.info("=" * 72)
    log.info(f"Diffusion steps : {DIFFUSION_STEPS}")
    log.info(f"Tokens/batch    : {TOKENS_PER_BATCH}")
    log.info("Running on GPU — HPC mode activated ...")
    log.info("")
    
    # --- THE ABSOLUTE NUCLEAR OVERRIDE ---
    # 1. Blind Lingua to the HPC cluster
    slurm_keys = [k for k in os.environ.keys() if k.startswith('SLURM')]
    for k in slurm_keys:
        del os.environ[k]
    
    # 2. Force PyTorch Distributed to initialize locally
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    
    # 3. Choke the NVIDIA NCCL backend
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"   # Force local loopback only (127.0.0.1)
    os.environ["NCCL_IB_DISABLE"] = "1"       # Completely disable Infiniband networking
    os.environ["NCCL_DEBUG"] = "INFO"         # Force NCCL to print exactly what it is doing
    # -------------------------------------

    log.info("HPC network interfaces disabled. Forcing local GPU execution.")
    
    # THE FIX: Dynamically calculate sequence length based on actual surviving channels
    try:
        staged_file = sorted(glob.glob(f"{PT_DIR}/*.pt"))[0]
        actual_chans = torch.load(staged_file, weights_only=False)['data'].shape[1]
        # 1280 samples / 128 fine-time pts = 10 coarse steps per epoch.
        # tokens_per_batch is the sequence packing target across multiple epochs.
        # Packing ~8 epochs per batch: 10 × channels × 8 = a reasonable GPU fill.
        coarse_steps   = 1280 // 128          # = 10
        tokens_per_epoch = coarse_steps * actual_chans
        dynamic_tokens   = tokens_per_epoch * 8
        log.info(f"Dynamically aligned sequence: {coarse_steps} coarse steps x {actual_chans} channels = {tokens_per_epoch} tokens/epoch")
        log.info(f"Packing ~8 epochs per batch: {dynamic_tokens} tokens/batch")
    except IndexError:
        log.error("No staged PT files found for inference.")
        sys.exit(1)

    t0 = time.time()

    inference(
        input_dir=PT_DIR,
        output_dir=RECON_PT_DIR,
        gpu_device=GPU_DEVICE,
        diffusion_sample_steps=DIFFUSION_STEPS,
        tokens_per_batch=dynamic_tokens,
        data_norm=10.0,  # CRITICAL: ZUNA expects std=0.1; our z-scored input is std=1.0
    )

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    # ── Health Check ──────────────────────────────────────────────────────
    recon_files = sorted(glob.glob(f"{RECON_PT_DIR}/*.pt"))
    expected_files = sorted(glob.glob(f"{PT_DIR}/*.pt"))

    log.info("")
    log.info(f"[TIMER]  Inference completed in {mins}m {secs}s")
    log.info(f"[CHECK]  Expected .pt files: {len(expected_files)}")
    log.info(f"[CHECK]  Produced .pt files: {len(recon_files)}")

    if len(recon_files) == 0:
        log.error("Inference produced ZERO output files — aborting.")
        sys.exit(1)

    if len(recon_files) != len(expected_files):
        log.warning(
            f"File count mismatch: expected {len(expected_files)}, "
            f"got {len(recon_files)}."
        )

    log.info("[OK] Inference complete.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Statistical Rescaling & NPY Export
# ══════════════════════════════════════════════════════════════════════════════
def stage3_rescale_and_export():
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 3: FORMATTING + Z-SCORE REVERSAL")
    log.info("=" * 72)

    # ── Collect all reconstructed epochs from .pt files ───────────────────
    pt_files = sorted(glob.glob(f"{RECON_PT_DIR}/*.pt"))
    all_epochs = []
    for ptf in pt_files:
        data = torch.load(ptf, weights_only=False)
        key = 'data_reconstructed' if 'data_reconstructed' in data else 'data'
        for epoch_data in data.get(key, []):
            if isinstance(epoch_data, torch.Tensor):
                all_epochs.append(epoch_data.cpu().numpy())
            else:
                all_epochs.append(np.asarray(epoch_data))

    num_epochs = NUM_TEST_EPOCHS if NUM_TEST_EPOCHS else len(all_epochs)
    Z = np.stack(all_epochs[:num_epochs])

    # ── Reverse Z-score normalization ────────────────────────────────────
    # The input .pt metadata stores the zscore_mean and zscore_std used
    # during preprocessing. ZUNA's output is in z-score space, so we
    # reverse it: output_uV = output_zscore * std + mean
    input_pt = sorted(glob.glob(f"{PT_DIR}/*.pt"))[0]
    input_meta = torch.load(input_pt, weights_only=False).get('metadata', {})
    zscore_mean = input_meta.get('zscore_mean', 0.0)
    zscore_std  = input_meta.get('zscore_std', 1.0)

    log.info(f" -> Reversing z-score: mean={zscore_mean:.4f}, std={zscore_std:.4f}")
    Z = Z * zscore_std + zscore_mean
    log.info(f" -> Post-reversal std: {np.std(Z):.4f} uV")
    
    # ── Dimensional Alignment to (Epochs, 64, Time) ──────────────────────
    truth_path  = f"{SUBJECT_ID}_{SESSION_NAME}_y_truth.npy"
    truth  = np.load(truth_path)
    target_chans = truth.shape[1]   
    target_time  = truth.shape[2]   

    Z = Z[:, :target_chans, :target_time]
    truth = truth[:num_epochs]  # Align epoch count with Z

    # --- HARD INPAINTING OVERRIDE ---
    # The user explicitly requested that the diffusion model only reconstruct the missing channels,
    # and that we preserve the original physical signals for the unmasked (kept) channels.
    broken_path = f"{SUBJECT_ID}_{SESSION_NAME}_X_broken.npy"
    broken = np.load(broken_path)[:num_epochs]
    
    # Identify which channels were preserved (not zeroed out)
    # broken shape is (Epochs, Channels, Time)
    # A channel is considered "dropped" if its entire time series is exactly 0.0
    channel_is_dropped = np.all(broken == 0.0, axis=-1)  # shape: (Epochs, Channels)
    
    # We replace Z's generated signal with the Truth signal where the channel was NOT dropped
    # Expand dims to broadcast across time
    preserve_mask = ~channel_is_dropped
    preserve_mask_expanded = np.expand_dims(preserve_mask, axis=-1)
    
    Z = np.where(preserve_mask_expanded, truth, Z)
    log.info(" -> Applied hard inpainting: Restored physical recordings for preserved channels.")

    final_std = np.std(Z)
    log.info(f"[SHAPE]  Final shape     : {Z.shape}")
    log.info(f"[SCALE]  ZUNA final std  : {final_std:.6e} uV")

    out_path = f"{SUBJECT_ID}_{SESSION_NAME}_X_zuna_test.npy"
    np.save(out_path, Z)
    return Z

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Automated Alpha Grading
# ══════════════════════════════════════════════════════════════════════════════
def get_alpha_metrics(epochs_data, sfreq=256, left_idx=F3_IDX, right_idx=F4_IDX,
                      ch_names=None):
    """
    Compute PSD, IAF (center-of-gravity), and FAA for an (Epochs, Channels, Time) array.

    IAF uses the center-of-gravity method on posterior channels only (O1, O2, P3, P4,
    Pz, Oz, P7, P8 — wherever present).  This gives sub-bin frequency resolution and
    avoids dilution from frontal/temporal channels where alpha is weak.

    nperseg = sfreq * 8 → 0.125 Hz resolution (vs 0.5 Hz with sfreq*2).
    """
    freqs, psd = welch(epochs_data, fs=sfreq, nperseg=sfreq * 8, axis=-1)

    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_freqs = freqs[alpha_mask]

    # ── Select posterior channels for IAF ────────────────────────────────────
    POSTERIOR = {'O1', 'O2', 'OZ', 'P3', 'P4', 'PZ', 'P7', 'P8', 'PO3', 'PO4', 'POZ'}
    if ch_names is not None:
        post_idx = [i for i, ch in enumerate(ch_names) if ch.upper() in POSTERIOR]
    else:
        post_idx = list(range(epochs_data.shape[1]))   # fall back to all channels

    # Average PSD over epochs and posterior channels: shape → (freqs,)
    psd_post = psd[:, post_idx, :].mean(axis=(0, 1))

    # Center-of-gravity IAF in the alpha band
    alpha_power = psd_post[alpha_mask]
    total_power = alpha_power.sum()
    if total_power > 0:
        iaf = float(np.sum(alpha_freqs * alpha_power) / total_power)
    else:
        iaf = float(alpha_freqs[np.argmax(alpha_power)])   # fallback to peak

    # Grand-average PSD across all channels (for plotting)
    avg_psd_all = psd.mean(axis=(0, 1))

    # ── FAA (uses F3/F4 as before) ────────────────────────────────────────────
    alpha_power_spatial = psd[:, :, alpha_mask].mean(axis=-1)
    mean_left  = alpha_power_spatial[:, left_idx].mean()
    mean_right = alpha_power_spatial[:, right_idx].mean()
    faa = np.log(mean_right) - np.log(mean_left)

    return freqs, avg_psd_all, iaf, faa


def stage4_alpha_grading(zuna_data):
    log.info("")
    log.info("=" * 72)
    log.info("STAGE 4: AUTOMATED ALPHA GRADING")
    log.info("=" * 72)

    truth_path  = f"{SUBJECT_ID}_{SESSION_NAME}_y_truth.npy"
    spline_path = f"{SUBJECT_ID}_{SESSION_NAME}_X_spline.npy"

    truth  = np.load(truth_path)
    spline = np.load(spline_path)

    n = zuna_data.shape[0]
    truth  = truth[:n]
    spline = spline[:n]

    # Recover channel names (alphabetically sorted, same order as arrays)
    import torch, glob
    pt_file   = sorted(glob.glob("test_pt_out/*.pt"))[0]
    ch_names  = sorted(torch.load(pt_file, weights_only=False)['metadata']['ch_names'])

    log.info(f"Comparing first {n} epochs across conditions ...")
    log.info(f"F3 index = {F3_IDX}, F4 index = {F4_IDX}")
    log.info(f"Posterior channels for IAF: {[c for c in ch_names if c.upper() in {'O1','O2','OZ','P3','P4','PZ','P7','P8','PO3','PO4','POZ'}]}")
    log.info("")

    f, psd_t, iaf_t, faa_t = get_alpha_metrics(truth,      sfreq=256, ch_names=ch_names)
    f, psd_s, iaf_s, faa_s = get_alpha_metrics(spline,     sfreq=256, ch_names=ch_names)
    f, psd_z, iaf_z, faa_z = get_alpha_metrics(zuna_data,  sfreq=256, ch_names=ch_names)

    # ── Summary Table ─────────────────────────────────────────────────────
    hdr  = f"{'Condition':<12} {'IAF (Hz)':>10} {'FAA':>12}"
    sep  = "-" * 36
    row_t = f"{'TRUTH':<12} {iaf_t:>10.2f} {faa_t:>12.4f}"
    row_s = f"{'SPLINE':<12} {iaf_s:>10.2f} {faa_s:>12.4f}"
    row_z = f"{'ZUNA':<12} {iaf_z:>10.2f} {faa_z:>12.4f}"

    for line in [sep, hdr, sep, row_t, row_s, row_z, sep]:
        log.info(line)

    # ── PSD Comparison Plot ───────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(f, 10 * np.log10(psd_t), color='black', lw=2,
             label=f'Truth (IAF={iaf_t:.1f}Hz, FAA={faa_t:.3f})')
    plt.plot(f, 10 * np.log10(psd_s), color='red', ls='--',
             label=f'Spline (IAF={iaf_s:.1f}Hz, FAA={faa_s:.3f})')
    plt.plot(f, 10 * np.log10(psd_z), color='blue', alpha=0.8,
             label=f'ZUNA (IAF={iaf_z:.1f}Hz, FAA={faa_z:.3f})')
    plt.axvspan(8, 13, color='gray', alpha=0.1, label='Alpha Band')
    plt.title(f"Spectral Fidelity — {SUBJECT_ID} (First {n} Epochs, {DIFFUSION_STEPS} Steps)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.xlim(2, 30)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("alpha_comparison.png", dpi=300)
    log.info("[OK] Saved alpha_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()

    log.info("")
    log.info("╔════════════════════════════════════════════════════════════════╗")
    log.info("║       ZUNA EEG RECONSTRUCTION — MASTER PIPELINE                ║")
    log.info("╚════════════════════════════════════════════════════════════════╝")
    log.info("")

    stage0_prepare_references()
    stage1_masking()
    stage2_inference()
    zuna_data = stage3_rescale_and_export()
    stage4_alpha_grading(zuna_data)

    elapsed = time.time() - t_total
    mins, secs = divmod(int(elapsed), 60)
    log.info("")
    log.info(f"{'=' * 72}")
    log.info(f"PIPELINE COMPLETE — Total wall time: {mins}m {secs}s")
    log.info(f"Full audit trail written to: pipeline_audit.log")
    log.info(f"{'=' * 72}")


if __name__ == "__main__":
    main()
