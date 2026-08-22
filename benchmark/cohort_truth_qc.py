"""Run the frozen truth-only QC sweep over the G001-G005 pilot cohort.

This module never imports or invokes ZUNA. Every requested recording is either
reported as a success or retained in the failure ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stage0_cache
import truth_qc


CONFIG_PATH = ROOT / "config" / "truth_qc_cohort_v1.json"


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _event_spacing(manifest: dict) -> float | None:
    onsets = sorted(
        float(row["onset_seconds"])
        for row in manifest["preprocessing_meta"]["event_qc"]
    )
    return float(np.median(np.diff(onsets))) if len(onsets) > 1 else None


def _specparam_result(report_dir: Path) -> tuple[dict, list[float]]:
    full_rows = [
        row for row in _read_csv(report_dir / "metric_values.csv")
        if row["metric"] == "specparam_peaks"
        and row["submetric"] == "aperiodic_exponent"
    ]
    if len(full_rows) != 1:
        raise RuntimeError(f"Expected one full specparam diagnostic row in {report_dir}")
    row = full_rows[0]
    block_rows = [
        item for item in _read_csv(report_dir / "metric_blocks.csv")
        if item["metric"] == "specparam_peaks"
        and item["submetric"] == "aperiodic_exponent"
    ]
    block_r_squared = [float(item["r_squared"]) for item in block_rows]
    return {
        "specparam_fit_status": row["fit_status"],
        "specparam_r_squared": float(row["r_squared"]),
        "specparam_mae": float(row["mean_absolute_error"]),
        "specparam_posterior_channel_count": int(row["posterior_channel_count"]),
        "specparam_detected_peak_count": int(row["detected_peak_count"]),
        "specparam_alpha_peak_count": int(row["alpha_peak_count"]),
    }, block_r_squared


def _cohort_plots(output_dir: Path, rows: list[dict]) -> None:
    x = np.arange(len(rows))
    labels = [row["recording"].replace(".cnt", "") for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].scatter(x, [row["specparam_r_squared"] for row in rows], s=22)
    axes[0].set(ylabel="Full-fit R²", title="Truth-only specparam diagnostics")
    axes[1].scatter(x, [row["median_event_spacing_seconds"] for row in rows], s=22)
    axes[1].set(ylabel="Median spacing (s)", title="CNT marker timing")
    axes[2].scatter(x, [row["median_rms_uv"] for row in rows], s=22)
    axes[2].set(ylabel="Median RMS (µV)", title="No-ICA Stage-0 signal scale")
    axes[2].set_xticks(x, labels, rotation=90, fontsize=7)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.2)
    fig.savefig(output_dir / "cohort_qc_overview.png", dpi=180)
    plt.close(fig)


def run(data_dir: Path, cache_root: Path, output_dir: Path) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = int(config["recording_set"]["expected_recordings"])
    files = sorted(
        path for subject in config["recording_set"]["subjects"]
        for path in data_dir.glob(f"{subject}Day*Rest*.cnt")
    )
    if len(files) != expected:
        raise RuntimeError(f"Frozen cohort requires {expected} recordings; found {len(files)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    recording_root = output_dir / "recordings"
    recording_root.mkdir(exist_ok=True)
    successes, failures = [], []
    cnt_checks = config["cnt_descriptive_checks"]
    all_block_r_squared = []

    for index, raw_path in enumerate(files, start=1):
        print(f"[cohort truth QC {index}/{expected}] {raw_path.name}", flush=True)
        try:
            stage0 = stage0_cache.load_or_create_object(raw_path, cache_root=cache_root)
            identity = stage0.manifest["identity"]
            entry = stage0_cache._entry_path(
                cache_root.resolve(), raw_path.resolve(), identity["cache_key_sha256"])
            report_dir = recording_root / raw_path.stem
            summary = truth_qc.build_report(entry, report_dir, verify_raw=False)
            specparam, block_r_squared = _specparam_result(report_dir)
            all_block_r_squared.extend(block_r_squared)
            meta = stage0.manifest["preprocessing_meta"]
            spacing = _event_spacing(stage0.manifest)
            aux = {row["channel"]: row for row in meta["auxiliary_channel_qc"]}
            cnt_structure_matches = (
                float(meta["original_sfreq_hz"]) == float(cnt_checks["required_original_sfreq_hz"])
                and int(stage0.data.shape[1]) == int(cnt_checks["required_stage0_channels"])
                and int(stage0.data.shape[2]) == int(cnt_checks["required_samples_per_epoch"])
                and float(stage0.manifest["output"]["sfreq_hz"]) == float(
                    cnt_checks["required_stage0_sfreq_hz"])
            )
            successes.append({
                "recording": raw_path.name,
                "raw_bytes": identity["raw_bytes"],
                "raw_sha256": identity["raw_sha256"],
                "stage0_cache_key": identity["cache_key_sha256"],
                "original_sfreq_hz": meta["original_sfreq_hz"],
                "original_duration_seconds": meta["original_duration_seconds"],
                "available_epochs": meta["epochs_after_annotations"],
                "selected_epochs": int(stage0.data.shape[0]),
                "candidate_amplitude_flagged_epochs": meta["candidate_epochs_flagged_amplitude_or_flat"],
                "selected_amplitude_flagged_epochs": meta["selected_epochs_flagged_amplitude_or_flat"],
                "median_event_spacing_seconds": spacing,
                "median_rms_uv": summary["rms_uv"]["median"],
                "median_peak_to_peak_uv": summary["peak_to_peak_uv"]["median"],
                "heog_max_abs_eeg_r": aux.get("HEOG", {}).get("maximum_absolute_eeg_correlation"),
                "veog_max_abs_eeg_r": aux.get("VEOG", {}).get("maximum_absolute_eeg_correlation"),
                "raw_tail_warning": bool(meta["raw_tail_qc"]["warning"]),
                "continuous_warning_channels": sum(
                    bool(row.get("warnings")) for row in meta["analysis_interval_channel_qc"]),
                "cnt_structure_matches_declared_reader": bool(cnt_structure_matches),
                "specparam_blocks": len(block_r_squared),
                "specparam_block_r_squared_minimum": min(block_r_squared) if block_r_squared else None,
                "specparam_block_r_squared_maximum": max(block_r_squared) if block_r_squared else None,
                **specparam,
            })
        except Exception as error:
            failures.append({
                "recording": raw_path.name,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            print(f"[cohort truth QC failure] {raw_path.name}: {type(error).__name__}: {error}", flush=True)

    _write_csv(output_dir / "cohort_recordings.csv", successes)
    _write_csv(output_dir / "cohort_failures.csv", failures)
    if successes:
        _cohort_plots(output_dir, successes)

    r2_values = [row["specparam_r_squared"] for row in successes]
    rms_values = [row["median_rms_uv"] for row in successes]
    spacing_values = [row["median_event_spacing_seconds"] for row in successes]
    selected_epoch_values = [row["selected_epochs"] for row in successes]
    result = {
        "schema": "geeg-zuna-truth-qc-cohort-result-v2",
        "config_path": str(CONFIG_PATH.resolve()),
        "config_sha256": stage0_cache.sha256_file(CONFIG_PATH),
        "expected_recordings": expected,
        "readable_processed_recordings": len(successes),
        "processing_errors": len(failures),
        "all_expected_recordings_accounted_for": len(successes) + len(failures) == expected,
        "all_processed_cnt_structures_match_declared_reader": bool(successes) and all(
            row["cnt_structure_matches_declared_reader"] for row in successes),
        "selected_epochs": {
            "minimum": min(selected_epoch_values) if selected_epoch_values else None,
            "median": statistics.median(selected_epoch_values) if selected_epoch_values else None,
            "maximum": max(selected_epoch_values) if selected_epoch_values else None,
        },
        "specparam_full_r_squared": {
            "minimum": min(r2_values) if r2_values else None,
            "median": statistics.median(r2_values) if r2_values else None,
            "maximum": max(r2_values) if r2_values else None,
        },
        "specparam_block_r_squared": {
            "minimum": min(all_block_r_squared) if all_block_r_squared else None,
            "median": statistics.median(all_block_r_squared) if all_block_r_squared else None,
            "maximum": max(all_block_r_squared) if all_block_r_squared else None,
            "total_blocks": len(all_block_r_squared),
        },
        "median_rms_uv_across_recordings": {
            "minimum": min(rms_values) if rms_values else None,
            "median": statistics.median(rms_values) if rms_values else None,
            "maximum": max(rms_values) if rms_values else None,
        },
        "median_event_spacing_seconds": {
            "minimum": min(spacing_values) if spacing_values else None,
            "median": statistics.median(spacing_values) if spacing_values else None,
            "maximum": max(spacing_values) if spacing_values else None,
        },
        "zuna_inference_run": False,
    }
    (output_dir / "cohort_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# G001–G005 truth-only cohort QC

Frozen configuration: `{result['config_sha256']}`
Recordings: `{result['readable_processed_recordings']}/{expected}` processed; `{result['processing_errors']}` processing errors
ZUNA inference run: `False`

## Descriptive recording summary

No epoch-count, amplitude, or specparam threshold is used to exclude a readable recording.
Selected epoch-count distribution: `{result['selected_epochs']}`
All processed CNT structures match the declared reader: `{result['all_processed_cnt_structures_match_declared_reader']}`
Median marker spacing range: `{result['median_event_spacing_seconds']}` seconds
Median recording RMS range: `{result['median_rms_uv_across_recordings']}` µV

This is internal consistency evidence for the forced int32 interpretation. It
does not replace independent acquisition/export documentation.

## Descriptive specparam diagnostics

Full-fit R² distribution: `{result['specparam_full_r_squared']}`
8-epoch block R² distribution: `{result['specparam_block_r_squared']}`

These values are reported without an acceptance threshold.

No recording was silently omitted. See `cohort_recordings.csv`,
`cohort_failures.csv`, and `recordings/` for exact per-recording evidence.
"""
    (output_dir / "cohort_report.md").write_text(report, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_dir, args.cache_root, args.out_dir)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["processing_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
