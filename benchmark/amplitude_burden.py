"""Quantify record-only epoch/channel amplitude flags in verified Stage-0 truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from protocol_v2 import PREPROCESSING_SPEC


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(cohort_csv: Path, cache_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_rows, channel_rows, matrices = [], [], {}
    high_limit = float(PREPROCESSING_SPEC["epoch_peak_to_peak_max_uv"])
    flat_limit = float(PREPROCESSING_SPEC["epoch_peak_to_peak_flat_uv"])
    for cohort in _read(cohort_csv):
        stem = Path(cohort["recording"]).stem
        key = cohort["stage0_cache_key"]
        entry = cache_root / f"{stem}__{key[:20]}"
        with np.load(entry / "truth.npz", allow_pickle=False) as saved:
            data = np.asarray(saved["data"], dtype=np.float32)
            names = [str(name) for name in saved["ch_names"]]
        peak_to_peak = np.ptp(data, axis=-1)
        high = peak_to_peak > high_limit
        flat = peak_to_peak < flat_limit
        flagged = high | flat
        epoch_flagged = np.any(flagged, axis=1)
        high_counts = np.sum(high, axis=0)
        flat_counts = np.sum(flat, axis=0)
        order = np.argsort(high_counts)[::-1]
        top = [f"{names[index]}:{int(high_counts[index])}" for index in order if high_counts[index]][:10]
        recording_rows.append({
            "recording": cohort["recording"],
            "stage0_cache_key": key,
            "epochs": int(data.shape[0]),
            "epochs_with_any_flag": int(np.sum(epoch_flagged)),
            "epochs_with_high_amplitude": int(np.sum(np.any(high, axis=1))),
            "epochs_with_flat_channel": int(np.sum(np.any(flat, axis=1))),
            "channels_with_any_high_amplitude": int(np.sum(high_counts > 0)),
            "channels_with_any_flat_epoch": int(np.sum(flat_counts > 0)),
            "maximum_peak_to_peak_uv": float(np.max(peak_to_peak)),
            "top_high_amplitude_channels": ";".join(top),
        })
        for index, name in enumerate(names):
            channel_rows.append({
                "recording": cohort["recording"],
                "channel": name,
                "high_amplitude_epochs": int(high_counts[index]),
                "flat_epochs": int(flat_counts[index]),
                "maximum_peak_to_peak_uv": float(np.max(peak_to_peak[:, index])),
                "median_peak_to_peak_uv": float(np.median(peak_to_peak[:, index])),
            })
        matrices[cohort["recording"]] = high

    _write(output_dir / "recording_amplitude_burden.csv", recording_rows)
    _write(output_dir / "channel_amplitude_burden.csv", channel_rows)
    worst = sorted(recording_rows, key=lambda row: row["epochs_with_any_flag"], reverse=True)[:8]
    fig, axes = plt.subplots(len(worst), 1, figsize=(12, 1.8 * len(worst)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, worst):
        matrix = matrices[row["recording"]].T
        axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
        axis.set(title=row["recording"], ylabel="Channel", xlabel="Epoch")
    fig.suptitle(f"High-amplitude flags (> {high_limit:g} µV peak-to-peak)")
    fig.savefig(output_dir / "worst_recording_amplitude_flags.png", dpi=180)
    plt.close(fig)

    fully_flagged = [row for row in recording_rows if row["epochs_with_any_flag"] == row["epochs"]]
    clean = [row for row in recording_rows if row["epochs_with_any_flag"] == 0]
    summary = {
        "schema": "geeg-zuna-amplitude-burden-v1",
        "recordings": len(recording_rows),
        "high_amplitude_threshold_uv": high_limit,
        "flat_threshold_uv": flat_limit,
        "fully_flagged_recordings": [row["recording"] for row in fully_flagged],
        "zero_flag_recordings": [row["recording"] for row in clean],
        "median_flagged_epochs": float(np.median([
            row["epochs_with_any_flag"] for row in recording_rows
        ])),
        "maximum_flagged_epochs": max(
            row["epochs_with_any_flag"] for row in recording_rows
        ) if recording_rows else None,
        "zuna_inference_run": False,
    }
    (output_dir / "amplitude_burden_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.cohort_csv, args.cache_root, args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
