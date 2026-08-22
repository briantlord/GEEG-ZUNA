"""Truth-only QC report for a verified minimal Stage-0 cache entry.

This command never imports or invokes ZUNA. It summarizes the recorded EEG,
computes every metric in its contract-defined dropout reference frame, and
writes inspectable tables/plots tied to the Stage-0 cache identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

HERE = Path(__file__).resolve().parent
METRICS_DIR = HERE / "metrics"
for path in (HERE, METRICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pilot
import stage0_cache
import common as C
from base import REGISTRY
import m_faa  # noqa: F401
import m_frontal_midline_theta  # noqa: F401
import m_mu_asymmetry  # noqa: F401
import m_specparam_peaks  # noqa: F401
import m_theta_beta  # noqa: F401


BANDS = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "low_gamma_30_45": (30.0, 45.0),
}


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _quantiles(values) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: None for key in ("min", "q05", "q25", "median", "q75", "q95", "max")}
    points = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"), map(float, points)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(data: np.ndarray, ch_names: list[str]) -> tuple[list[dict], list[dict]]:
    full_rows, block_rows = [], []
    upper = [name.upper() for name in ch_names]
    for metric in REGISTRY.values():
        dropped = [upper.index(name.upper()) for name in metric.drop_channels]
        referenced = pilot.surviving_average_reference(data, dropped, ch_names)
        if metric.evaluate is not None:
            values, diagnostics = metric.evaluate(referenced, ch_names)
        else:
            values, diagnostics = metric.compute(referenced, ch_names), {}
        for submetric, value in values.items():
            full_rows.append({
                "metric": metric.key,
                "submetric": submetric,
                "value": _finite(value),
                "drop_set": "+".join(sorted(name.upper() for name in metric.drop_channels)),
                **diagnostics,
            })

        for block_start in range(0, data.shape[0], 8):
            block = data[block_start:block_start + 8]
            block_ref = pilot.surviving_average_reference(block, dropped, ch_names)
            if metric.evaluate is not None:
                block_values, block_diagnostics = metric.evaluate(block_ref, ch_names)
            else:
                block_values, block_diagnostics = metric.compute(block_ref, ch_names), {}
            for submetric, value in block_values.items():
                block_rows.append({
                    "metric": metric.key,
                    "submetric": submetric,
                    "block": block_start // 8 + 1,
                    "epoch_start": block_start,
                    "epoch_stop_exclusive": min(block_start + 8, data.shape[0]),
                    "value": _finite(value),
                    **block_diagnostics,
                })
    return full_rows, block_rows


def _plot_spectra(out: Path, frequencies: np.ndarray, psd: np.ndarray,
                  ch_names: list[str]) -> None:
    upper = [name.upper() for name in ch_names]
    regions = {
        "Frontal": ["FP1", "FP2", "AF3", "AF4", "F3", "F4", "FZ"],
        "Central": ["C3", "C4", "CZ", "FCZ", "CPZ"],
        "Posterior": ["O1", "O2", "OZ", "PO3", "PO4", "POZ", "PZ"],
    }
    keep = (frequencies >= 1) & (frequencies <= 45)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for label, channels in regions.items():
        indices = [upper.index(name) for name in channels if name in upper]
        regional = np.mean(psd[indices], axis=0)
        ax.plot(frequencies[keep], 10 * np.log10(regional[keep] + 1e-20), label=label)
    ax.set(title="No-ICA Stage-0 regional spectra", xlabel="Frequency (Hz)",
           ylabel="Power spectral density (dB µV²/Hz)")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(out / "regional_spectra.png", dpi=180)
    plt.close(fig)


def _plot_topographies(out: Path, bandpowers: dict[str, np.ndarray], ch_names: list[str]) -> None:
    info = mne.create_info(ch_names, 256.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"),
                     match_case=False, on_missing="raise")
    selected = ["delta_1_4", "theta_4_8", "alpha_8_13", "beta_13_30"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    for axis, key in zip(axes, selected):
        values = 10 * np.log10(bandpowers[key] + 1e-20)
        image, _ = mne.viz.plot_topomap(values, info, axes=axis, show=False, contours=4)
        axis.set_title(key.replace("_", " "))
        fig.colorbar(image, ax=axis, shrink=0.65, label="dB µV²")
    fig.suptitle("Average-reference band-power topographies")
    fig.savefig(out / "bandpower_topographies.png", dpi=180)
    plt.close(fig)


def _plot_distributions(out: Path, epoch_rms: np.ndarray, channel_rms: np.ndarray,
                        ch_names: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].plot(np.arange(1, len(epoch_rms) + 1), epoch_rms, marker=".", linewidth=1)
    axes[0].set(title="Epoch RMS", xlabel="Epoch", ylabel="Median channel RMS (µV)")
    order = np.argsort(channel_rms)
    axes[1].bar(np.arange(len(order)), channel_rms[order], width=0.85)
    axes[1].set(title="Channel RMS", xlabel="Channels sorted by RMS", ylabel="Median epoch RMS (µV)")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.2)
    fig.savefig(out / "rms_distributions.png", dpi=180)
    plt.close(fig)


def _plot_metric_blocks(out: Path, rows: list[dict]) -> None:
    series = {}
    for row in rows:
        if row["value"] is not None:
            series.setdefault((row["metric"], row["submetric"]), []).append(row)
    keys = list(series)
    ncols = 3
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.0 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, key in zip(axes, keys):
        values = series[key]
        axis.plot([row["block"] for row in values], [row["value"] for row in values], marker="o")
        axis.set(title=f"{key[0]}: {key[1]}", xlabel="8-epoch block", ylabel="Metric value")
        axis.grid(True, alpha=0.2)
    for axis in axes[len(keys):]:
        axis.remove()
    fig.suptitle("Truth-metric stability across eight consecutive blocks")
    fig.savefig(out / "metric_block_variation.png", dpi=180)
    plt.close(fig)


def build_report(stage0_entry: Path, output_dir: Path, verify_raw: bool = True) -> dict:
    data, ch_names, positions, manifest = stage0_cache.verify_entry(
        stage0_entry, verify_raw=verify_raw)
    data = np.asarray(data, dtype=np.float32)
    ch_names = list(ch_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    qc_frame = pilot.surviving_average_reference(data, [], ch_names)
    frequencies, epoch_psd = C.welch(qc_frame)
    mean_psd = epoch_psd.mean(axis=0)
    bandpowers = {key: C.bandpower(frequencies, mean_psd, lo, hi)
                  for key, (lo, hi) in BANDS.items()}

    rms = np.sqrt(np.mean(data ** 2, axis=-1))
    std = np.std(data, axis=-1)
    peak_to_peak = np.ptp(data, axis=-1)
    epoch_rows = [{
        "epoch": index + 1,
        "median_rms_uv": float(np.median(rms[index])),
        "maximum_rms_uv": float(np.max(rms[index])),
        "median_peak_to_peak_uv": float(np.median(peak_to_peak[index])),
        "maximum_peak_to_peak_uv": float(np.max(peak_to_peak[index])),
    } for index in range(data.shape[0])]
    channel_rows = []
    for index, name in enumerate(ch_names):
        row = {
            "channel": name,
            "median_rms_uv": float(np.median(rms[:, index])),
            "median_std_uv": float(np.median(std[:, index])),
            "median_peak_to_peak_uv": float(np.median(peak_to_peak[:, index])),
        }
        row.update({key: float(values[index]) for key, values in bandpowers.items()})
        channel_rows.append(row)

    metric_rows, metric_block_rows = _metric_rows(data, ch_names)
    _write_csv(output_dir / "epoch_qc.csv", epoch_rows)
    _write_csv(output_dir / "channel_qc.csv", channel_rows)
    _write_csv(output_dir / "metric_values.csv", metric_rows)
    _write_csv(output_dir / "metric_blocks.csv", metric_block_rows)

    _plot_spectra(output_dir, frequencies, mean_psd, ch_names)
    _plot_topographies(output_dir, bandpowers, ch_names)
    _plot_distributions(
        output_dir, np.median(rms, axis=1), np.median(rms, axis=0), ch_names)
    _plot_metric_blocks(output_dir, metric_block_rows)

    specparam_blocks = [row for row in metric_block_rows
                        if row["metric"] == "specparam_peaks" and row["submetric"] == "aperiodic_exponent"]
    specparam_r2 = [row.get("r_squared") for row in specparam_blocks
                    if row.get("r_squared") is not None]
    metric_variation = {}
    for key in {(row["metric"], row["submetric"]) for row in metric_block_rows}:
        values = [row["value"] for row in metric_block_rows
                  if (row["metric"], row["submetric"]) == key and row["value"] is not None]
        metric_variation[f"{key[0]}:{key[1]}"] = {
            "finite_blocks": len(values),
            "standard_deviation": float(np.std(values)) if values else None,
            "range": float(np.ptp(values)) if values else None,
        }

    summary = {
        "schema": "geeg-zuna-truth-qc-v1",
        "stage0_entry": str(stage0_entry.resolve()),
        "stage0_cache_key": manifest["identity"]["cache_key_sha256"],
        "protocol_id": manifest["identity"]["protocol_id"],
        "preprocessing_sha256": manifest["identity"]["preprocessing_sha256"],
        "shape": list(data.shape),
        "sfreq_hz": manifest["output"]["sfreq_hz"],
        "component_removal": manifest["preprocessing_meta"]["component_removal"],
        "ica_applied": manifest["preprocessing_meta"]["ica_applied"],
        "epochs_after_annotations": manifest["preprocessing_meta"]["epochs_after_annotations"],
        "selected_epochs": int(data.shape[0]),
        "requested_maximum_epochs": manifest["preprocessing_meta"]["requested_maximum_epochs"],
        "candidate_epochs_flagged_amplitude_or_flat": manifest["preprocessing_meta"]["candidate_epochs_flagged_amplitude_or_flat"],
        "selected_epochs_flagged_amplitude_or_flat": manifest["preprocessing_meta"]["selected_epochs_flagged_amplitude_or_flat"],
        "rms_uv": _quantiles(rms),
        "peak_to_peak_uv": _quantiles(peak_to_peak),
        "auxiliary_channel_qc": manifest["preprocessing_meta"]["auxiliary_channel_qc"],
        "specparam_block_r_squared": _quantiles(specparam_r2),
        "metric_block_variation": metric_variation,
        "all_metric_values_finite": all(row["value"] is not None for row in metric_rows),
        "all_block_metric_values_finite": all(row["value"] is not None for row in metric_block_rows),
        "artifacts": [
            "epoch_qc.csv", "channel_qc.csv", "metric_values.csv", "metric_blocks.csv",
            "regional_spectra.png", "bandpower_topographies.png",
            "rms_distributions.png", "metric_block_variation.png",
        ],
    }
    (output_dir / "truth_qc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    aux_lines = []
    for row in summary["auxiliary_channel_qc"]:
        strongest = row.get("strongest_eeg_correlations", [])
        top = strongest[0] if strongest else None
        aux_lines.append(
            f"- {row['channel']}: max |EEG correlation| "
            f"{row.get('maximum_absolute_eeg_correlation')!r}"
            + (f" ({top['eeg_channel']}, r={top['pearson_r']:.3f})" if top else "")
        )
    metric_lines = [
        f"- {row['metric']} / {row['submetric']}: {row['value']!r}"
        + (f"; fit status={row.get('fit_status')}, R²={row.get('r_squared'):.4f}"
           if row.get("r_squared") is not None else "")
        for row in metric_rows
    ]
    report = f"""# Truth-only QC — {stage0_entry.name}

Stage-0 cache key: `{summary['stage0_cache_key']}`
Protocol: `{summary['protocol_id']}`
Shape: `{data.shape[0]} x {data.shape[1]} x {data.shape[2]}` at `{summary['sfreq_hz']} Hz`
Component removal: `{summary['component_removal']}`; ICA applied: `{summary['ica_applied']}`

## Structural result

- {summary['epochs_after_annotations']} source-annotation-valid epochs were available; {summary['selected_epochs']} were analyzed, up to the requested maximum of {summary['requested_maximum_epochs']}.
- {summary['selected_epochs_flagged_amplitude_or_flat']} selected epochs and {summary['candidate_epochs_flagged_amplitude_or_flat']} total candidates were flagged by the record-only amplitude rule.
- All full-recording metric values finite: `{summary['all_metric_values_finite']}`.
- All 8-epoch-block metric values finite: `{summary['all_block_metric_values_finite']}`.
- Channel/epoch RMS and peak-to-peak distributions are in `channel_qc.csv` and `epoch_qc.csv`.

## Auxiliary diagnostics (never used for selection)

{chr(10).join(aux_lines)}

## Full-recording truth metrics

{chr(10).join(metric_lines)}

## Specparam fit-quality evidence

Eight consecutive 8-epoch truth blocks produced R² distribution:
`{json.dumps(summary['specparam_block_r_squared'], sort_keys=True)}`.
This report supplies evidence only; it does not choose or freeze an acceptance threshold.

## Plots

- `regional_spectra.png`
- `bandpower_topographies.png`
- `rms_distributions.png`
- `metric_block_variation.png`
"""
    (output_dir / "truth_qc_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage0_entry", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_report(args.stage0_entry, args.out_dir)
    print(json.dumps({
        "status": "pass",
        "stage0_cache_key": summary["stage0_cache_key"],
        "all_metric_values_finite": summary["all_metric_values_finite"],
        "all_block_metric_values_finite": summary["all_block_metric_values_finite"],
        "out_dir": str(args.out_dir.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
