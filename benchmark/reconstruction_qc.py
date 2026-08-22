"""Pre-score physiological and integrity gates for reconstructed EEG tensors."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import welch


QC_SCHEMA = "geeg-zuna-reconstruction-qc-v2"
SFREQ = 256.0
BAND_HZ = (1.0, 45.0)
POWER_RATIO_LIMITS = (0.05, 10.0)
MAX_ABS_UV = 1000.0
FLAT_SD_UV = 0.1


def array_sha256(value: np.ndarray) -> str:
    """Content identity that includes dtype, shape, and C-order bytes."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _bandpower(data: np.ndarray) -> np.ndarray:
    frequencies, psd = welch(
        data, fs=SFREQ, nperseg=min(1024, data.shape[-1]), axis=-1
    )
    selected = (frequencies >= BAND_HZ[0]) & (frequencies < BAND_HZ[1])
    if selected.sum() < 2:
        raise ValueError("not enough frequency bins to evaluate the 1-45 Hz output gate")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(psd[..., selected], frequencies[selected], axis=-1)


def _safe_correlation(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def evaluate_reconstruction(truth, reconstruction, dropped, ch_names=None):
    """Return diagnostics and fail closed when an output is not scoreable.

    The power gate rejects reconstructions whose total held-out 1-45 Hz power is
    below 1/20 or above 10 times truth. It is validation only: no diagnostic is
    fed back into reconstruction or used for post-hoc amplitude calibration.
    """
    truth = np.asarray(truth)
    reconstruction = np.asarray(reconstruction)
    dropped = sorted({int(index) for index in dropped})
    if truth.ndim != 3 or reconstruction.shape != truth.shape:
        raise ValueError(
            f"reconstruction shape mismatch: truth={truth.shape}, reconstruction={reconstruction.shape}"
        )
    if not dropped or dropped[0] < 0 or dropped[-1] >= truth.shape[1]:
        raise ValueError(f"invalid dropped-channel indices: {dropped}")
    if not np.isfinite(truth).all() or not np.isfinite(reconstruction).all():
        raise ValueError("truth and reconstruction must contain only finite values")

    good = [index for index in range(truth.shape[1]) if index not in dropped]
    if not np.array_equal(reconstruction[:, good, :], truth[:, good, :]):
        raise ValueError("hard-inpainting integrity failed: an observed channel changed")

    held_truth = truth[:, dropped, :].astype(np.float64, copy=False)
    held_reconstruction = reconstruction[:, dropped, :].astype(np.float64, copy=False)
    truth_power = _bandpower(held_truth)
    reconstruction_power = _bandpower(held_reconstruction)
    frequencies, truth_psd = welch(
        held_truth, fs=SFREQ, nperseg=min(1024, truth.shape[-1]), axis=-1
    )
    _, reconstruction_psd = welch(
        held_reconstruction, fs=SFREQ,
        nperseg=min(1024, reconstruction.shape[-1]), axis=-1,
    )
    spectral_bins = (frequencies >= BAND_HZ[0]) & (frequencies < BAND_HZ[1])
    total_truth_power = float(np.sum(truth_power))
    total_reconstruction_power = float(np.sum(reconstruction_power))
    if not np.isfinite(total_truth_power) or total_truth_power <= 0:
        raise ValueError("held-out truth has no finite positive 1-45 Hz power")

    power_ratio = total_reconstruction_power / total_truth_power
    truth_rms = float(np.sqrt(np.mean(held_truth ** 2)))
    reconstruction_rms = float(np.sqrt(np.mean(held_reconstruction ** 2)))
    rms_ratio = reconstruction_rms / truth_rms if truth_rms > 0 else float("inf")
    max_abs_uv = float(np.max(np.abs(held_reconstruction)))
    low, high = POWER_RATIO_LIMITS
    reasons = []
    if not low <= power_ratio <= high:
        reasons.append(
            f"1-45 Hz power ratio {power_ratio:.6g} outside frozen [{low}, {high}]"
        )
    if max_abs_uv > MAX_ABS_UV:
        reasons.append(f"held-out max |amplitude| {max_abs_uv:.6g} uV exceeds {MAX_ABS_UV:g} uV")

    names = list(ch_names) if ch_names is not None else None
    if names is not None and len(names) != truth.shape[1]:
        raise ValueError("channel-name count does not match reconstruction tensor")
    per_epoch_channel = []
    for epoch_index in range(held_truth.shape[0]):
        for local_index, channel_index in enumerate(dropped):
            truth_epoch = held_truth[epoch_index, local_index]
            recon_epoch = held_reconstruction[epoch_index, local_index]
            truth_epoch_power = float(truth_power[epoch_index, local_index])
            recon_epoch_power = float(reconstruction_power[epoch_index, local_index])
            log_truth = np.log10(np.maximum(
                truth_psd[epoch_index, local_index, spectral_bins], np.finfo(float).tiny
            ))
            log_recon = np.log10(np.maximum(
                reconstruction_psd[epoch_index, local_index, spectral_bins], np.finfo(float).tiny
            ))
            per_epoch_channel.append({
                "epoch_index": epoch_index,
                "channel_index": channel_index,
                "channel": names[channel_index] if names is not None else str(channel_index),
                "truth_mean_uv": float(np.mean(truth_epoch)),
                "reconstruction_mean_uv": float(np.mean(recon_epoch)),
                "truth_sd_uv": float(np.std(truth_epoch)),
                "reconstruction_sd_uv": float(np.std(recon_epoch)),
                "truth_rms_uv": float(np.sqrt(np.mean(truth_epoch ** 2))),
                "reconstruction_rms_uv": float(np.sqrt(np.mean(recon_epoch ** 2))),
                "truth_power_1_45": truth_epoch_power,
                "reconstruction_power_1_45": recon_epoch_power,
                "power_ratio_1_45": (
                    recon_epoch_power / truth_epoch_power if truth_epoch_power > 0 else None
                ),
                "max_abs_uv": float(np.max(np.abs(recon_epoch))),
                "flat": bool(np.std(recon_epoch) < FLAT_SD_UV),
                "clipped_sample_fraction": float(np.mean(np.abs(recon_epoch) > MAX_ABS_UV)),
                "waveform_correlation": _safe_correlation(truth_epoch, recon_epoch),
                "log10_psd_rmse_1_45": float(np.sqrt(np.mean((log_recon - log_truth) ** 2))),
            })

    diagnostics = {
        "qc_schema": QC_SCHEMA,
        "status": "fail" if reasons else "pass",
        "power_ratio_1_45": float(power_ratio),
        "rms_ratio": float(rms_ratio),
        "max_abs_uv": max_abs_uv,
        "truth_power_1_45": total_truth_power,
        "reconstruction_power_1_45": total_reconstruction_power,
        "reconstruction_sha256": array_sha256(reconstruction),
        "diagnostic_thresholds": {
            "flat_sd_uv": FLAT_SD_UV,
            "clipping_abs_uv": MAX_ABS_UV,
        },
        "per_epoch_channel": per_epoch_channel,
        "reasons": reasons,
    }
    return diagnostics


def require_reconstruction(truth, reconstruction, dropped):
    """Evaluate and raise with diagnostics when a reconstruction fails."""
    diagnostics = evaluate_reconstruction(truth, reconstruction, dropped)
    if diagnostics["status"] != "pass":
        raise ValueError("; ".join(diagnostics["reasons"]))
    return diagnostics
