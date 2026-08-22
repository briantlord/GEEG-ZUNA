"""Run the corrected-v2 biomarker benchmark and emit Phase 3 validated rows.

Each metric drop set is reconstructed once per method and reused by every metric
with that exact drop set. A reconstruction must pass integrity and physiological
scale gates before any biomarker is computed. The production SLURM array remains
blocked pending the later execution gate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
for path in (HERE, BENCH):
    if path not in sys.path:
        sys.path.insert(0, path)

import base
import pilot
import reconstruction_qc
import stage0_cache
import run_manifest
from contract import CONTRACT_SHA256, EXPERIMENT_ID, metric_contract
from protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID
from schema_v3 import (
    COLUMNS, KEY_COLUMNS, RESULT_SCHEMA, classify_method, make_truth_unit_id,
    reference_frame_id,
)


class StatusLedger:
    """Atomic, retry-safe terminal failure states for expected benchmark units."""

    def __init__(self, path, run_id, terminal_states):
        self.path = Path(path)
        self.run_id = run_id
        self.terminal_states = set(terminal_states) - {"success"}
        self.rows = {}
        if self.path.is_file():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("run_id") != run_id:
                    raise RuntimeError(
                        f"status ledger has a mixed run ID at line {line_number}: {self.path}"
                    )
                if row.get("status") not in self.terminal_states:
                    raise RuntimeError(
                        f"invalid status at line {line_number}: {row.get('status')!r}"
                    )
                key = self._row_key(row)
                if key in self.rows:
                    raise RuntimeError(f"duplicate status unit at line {line_number}: {key}")
                self.rows[key] = row

    @staticmethod
    def _row_key(row):
        if row.get("unit_type") == "result":
            return ("result",) + tuple(str(row[name]) for name in KEY_COLUMNS)
        if row.get("unit_type") == "reconstruction":
            return (
                "reconstruction", str(row["recording"]),
                str(row["method"]), str(row["drop_set"]),
            )
        raise RuntimeError(f"invalid status-ledger unit type: {row.get('unit_type')!r}")

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for key in sorted(self.rows):
                stream.write(json.dumps(self.rows[key], sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, self.path)

    def set_result(self, key, status, error, stage):
        if status not in self.terminal_states:
            raise RuntimeError(f"invalid terminal failure status: {status}")
        row = {
            "unit_type": "result", "run_id": self.run_id, "status": status,
            **dict(zip(KEY_COLUMNS, key)), "stage": stage,
            "error_type": type(error).__name__, "error": str(error),
        }
        self.rows[self._row_key(row)] = row
        self._flush()

    def set_reconstruction(self, recording, method, drop_set, status, error, stage):
        if status not in self.terminal_states:
            raise RuntimeError(f"invalid terminal failure status: {status}")
        row = {
            "unit_type": "reconstruction", "run_id": self.run_id,
            "recording": recording, "method": method, "drop_set": drop_set,
            "status": status, "stage": stage,
            "error_type": type(error).__name__, "error": str(error),
        }
        self.rows[self._row_key(row)] = row
        self._flush()

    def clear_result(self, key):
        if self.rows.pop(("result",) + tuple(str(value) for value in key), None) is not None:
            self._flush()

    def clear_reconstruction(self, recording, method, drop_set):
        key = ("reconstruction", str(recording), str(method), str(drop_set))
        if self.rows.pop(key, None) is not None:
            self._flush()

    def initialize_pending(self, result_units, reconstruction_units,
                           successful_results, successful_reconstructions):
        changed = False
        for unit in result_units:
            key_values = tuple(str(unit[name]) for name in KEY_COLUMNS)
            key = ("result",) + key_values
            if key_values in successful_results or key in self.rows:
                continue
            self.rows[key] = {
                "unit_type": "result", "run_id": self.run_id,
                **dict(zip(KEY_COLUMNS, key_values)),
                "status": "preempted_incomplete", "stage": "runner_pending",
                "error_type": "PendingUnit",
                "error": "unit did not reach a later terminal state",
            }
            changed = True
        for unit in reconstruction_units:
            values = (
                str(unit["recording"]), str(unit["method"]), str(unit["drop_set"])
            )
            key = ("reconstruction",) + values
            if values in successful_reconstructions or key in self.rows:
                continue
            self.rows[key] = {
                "unit_type": "reconstruction", "run_id": self.run_id,
                "recording": values[0], "method": values[1], "drop_set": values[2],
                "status": "preempted_incomplete", "stage": "runner_pending",
                "error_type": "PendingUnit",
                "error": "unit did not reach a later terminal state",
            }
            changed = True
        if changed or not self.path.exists():
            self._flush()

    def pending_rows(self, recording):
        return [
            row for row in self.rows.values()
            if row.get("recording") == recording
            and row.get("status") == "preempted_incomplete"
        ]


def discover(keys=None):
    """Import requested metric plug-ins and validate the registry contract."""
    if keys:
        missing_modules = []
        for key in keys:
            module = os.path.join(HERE, f"m_{key}.py")
            if not os.path.exists(module):
                missing_modules.append(key)
            else:
                importlib.import_module(f"m_{key}")
        if missing_modules:
            raise ValueError(f"unknown metric module(s): {missing_modules}")
    else:
        for filename in sorted(glob.glob(os.path.join(HERE, "m_*.py"))):
            importlib.import_module(os.path.splitext(os.path.basename(filename))[0])
    selected = {key: value for key, value in base.REGISTRY.items() if keys is None or key in keys}
    if keys and set(selected) != set(keys):
        raise ValueError(f"requested metric(s) did not register: {sorted(set(keys) - set(selected))}")
    return selected


def load_truth(filename, cache_root=None):
    """Load the typed, verified Stage-0 object required by reconstruction adapters."""
    return stage0_cache.load_or_create_object(filename, cache_root=cache_root)


def drop_indices(ch_names, names):
    """Resolve a complete declared drop set; never silently shrink a condition."""
    upper = [channel.upper() for channel in ch_names]
    missing = [name for name in names if name.upper() not in upper]
    if missing:
        raise ValueError(f"recording is missing declared drop channels: {missing}")
    indices = sorted(upper.index(name.upper()) for name in names)
    if any(upper[index] in {"M1", "M2"} for index in indices):
        raise ValueError("M1/M2 are non-cortical and forbidden from metric drop sets")
    return indices


def metric_evaluation(metric, data, ch_names):
    """Enforce exact finite values and JSON-safe persisted metric diagnostics."""
    if metric.evaluate is None:
        values, diagnostics = metric.compute(data, ch_names), {}
    else:
        values, diagnostics = metric.evaluate(data, ch_names)
    if not isinstance(values, dict):
        raise TypeError(f"{metric.key} returned {type(values).__name__}, expected dict")
    expected = set(metric.submetrics)
    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"{metric.key} output keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    converted = {key: float(values[key]) for key in metric.submetrics}
    nonfinite = [key for key, value in converted.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError(f"{metric.key} produced non-finite submetrics: {nonfinite}")
    if not isinstance(diagnostics, dict):
        raise TypeError(f"{metric.key} diagnostics must be a dict")
    try:
        diagnostic_json = json.dumps(
            diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{metric.key} diagnostics are not finite JSON") from error
    return converted, diagnostic_json


def metric_values(metric, data, ch_names):
    """Compatibility wrapper returning only validated scalar outputs."""
    return metric_evaluation(metric, data, ch_names)[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_provenance(metric):
    module_path = Path(sys.modules[metric.compute.__module__].__file__).resolve()
    config = metric_contract(metric.key)
    return {
        "metric_source_sha256": sha256_file(module_path),
        "metric_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "metrics_common_sha256": sha256_file(Path(HERE) / "common.py"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "qc_sha256": sha256_file(Path(reconstruction_qc.__file__).resolve()),
        "schema_sha256": sha256_file(Path(HERE) / "schema_v3.py"),
        "aggregator_sha256": sha256_file(Path(HERE) / "aggregate.py"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["G001"])
    parser.add_argument("--methods", nargs="+", default=["spline"])
    parser.add_argument("--metrics", nargs="+", default=None, help="metric keys (default: all)")
    parser.add_argument("--out", default="results/metric_eval_v3.csv")
    parser.add_argument(
        "--qc-out", default=None,
        help="reconstruction QC JSONL (default: <out>.reconstruction_qc.jsonl)",
    )
    parser.add_argument(
        "--status-out", default=None,
        help="atomic terminal-failure JSONL (default: <out>.status.jsonl)",
    )
    parser.add_argument(
        "--data-dir", default="GEEG_Raw",
        help="directory containing CNT recordings (absolute paths supported)",
    )
    parser.add_argument(
        "--stage0-cache-dir", default=None,
        help="corrected-v2 content-addressed truth cache root",
    )
    parser.add_argument(
        "--allow-phase2-zuna", action="store_true",
        help="explicitly permit the validated Phase 2 ZUNA adapter (GPU required)",
    )
    parser.add_argument(
        "--zuna-version", default="1.1", choices=["1.1"],
        help="frozen production model version for the 'zuna' method",
    )
    parser.add_argument(
        "--zuna-calibration", choices=run_manifest.CALIBRATION_STRATEGIES,
        default=run_manifest.DEFAULT_CALIBRATION,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument(
        "--task-index", type=int, default=None,
        help="run exactly one immutable task-mapping entry; omit to run all manifest recordings",
    )
    args = parser.parse_args()

    frozen_run = run_manifest.load_verified(args.run_manifest)
    os.environ["GEEG_ZUNA_RUN_ID"] = frozen_run["run_id"]
    if frozen_run["source_sha256"] != run_manifest.source_hashes():
        parser.error("active source differs from --run-manifest; create a new immutable manifest")
    if set(args.methods) != set(frozen_run["methods"]):
        parser.error("--methods must exactly match the immutable run manifest")
    expected_calibration = (
        args.zuna_calibration if "zuna" in args.methods else None
    )
    if frozen_run["calibration_strategy"] != expected_calibration:
        parser.error("--zuna-calibration must exactly match the immutable run manifest")

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    try:
        method_classes = {method: classify_method(method) for method in args.methods}
    except ValueError as error:
        parser.error(str(error))
    if "zuna" in args.methods and not args.allow_phase2_zuna:
        parser.error("ZUNA requires explicit --allow-phase2-zuna; production SLURM remains gated")

    zuna_module = None
    if "zuna" in args.methods:
        import zuna_method_v11 as zuna_module

    registry = discover(args.metrics)
    if not registry:
        raise RuntimeError("no metrics registered/selected")
    if set(registry) != set(frozen_run["metrics"]):
        parser.error("--metrics must exactly match the immutable run manifest")
    print(
        f"[metrics v4] {list(registry)} | methods={args.methods} | run={frozen_run['run_id'][:16]}",
        flush=True,
    )

    groups = {}
    metric_provenance_by_key = {}
    for metric in registry.values():
        declared = metric_contract(metric.key)
        if [name.upper() for name in metric.drop_channels] != [name.upper() for name in declared["drop_set"]]:
            raise RuntimeError(f"{metric.key} code drop set differs from scientific contract")
        if list(metric.submetrics) != list(declared["submetrics"]):
            raise RuntimeError(f"{metric.key} code submetrics differ from scientific contract")
        metric_provenance_by_key[metric.key] = metric_provenance(metric)
        drop_set = frozenset(channel.upper() for channel in metric.drop_channels)
        groups.setdefault(drop_set, ("+".join(sorted(drop_set)), []))[1].append(metric)

    output_dir = os.path.dirname(args.out) or "."
    os.makedirs(output_dir, exist_ok=True)
    qc_output = args.qc_out or f"{args.out}.reconstruction_qc.jsonl"
    status_output = args.status_out or f"{args.out}.status.jsonl"
    os.makedirs(os.path.dirname(qc_output) or ".", exist_ok=True)

    written = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out, newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            if reader.fieldnames != COLUMNS:
                raise RuntimeError(
                    f"refusing to append Phase 3 rows to an incompatible CSV: {args.out}"
                )
            for row in reader:
                if row["result_schema"] != RESULT_SCHEMA:
                    raise RuntimeError(f"non-v3 row found in {args.out}")
                written.add(tuple(row[key] for key in KEY_COLUMNS))

    new_output = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    output_handle = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(output_handle, fieldnames=COLUMNS)
    if new_output:
        writer.writeheader()
        output_handle.flush()
    qc_written = set()
    if os.path.exists(qc_output):
        for line_number, line in enumerate(
            Path(qc_output).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != frozen_run["run_id"] or row.get("unit_status") != "success":
                raise RuntimeError(f"incompatible QC row at {qc_output}:{line_number}")
            key = (str(row["recording"]), str(row["method"]), str(row["drop_set"]))
            if key in qc_written:
                raise RuntimeError(f"duplicate QC unit at {qc_output}:{line_number}: {key}")
            qc_written.add(key)
    qc_handle = open(qc_output, "a", encoding="utf-8")
    status_ledger = StatusLedger(
        status_output, frozen_run["run_id"], frozen_run["terminal_states"]
    )

    def emit(row):
        key = tuple(str(row[column]) for column in KEY_COLUMNS)
        if key not in written:
            writer.writerow(row)
            written.add(key)
        status_ledger.clear_result(key)

    def log_qc(row):
        key = (str(row["recording"]), str(row["method"]), str(row["drop_set"]))
        if key not in qc_written:
            qc_handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            qc_handle.flush()
            qc_written.add(key)
        if row.get("unit_status") == "success":
            status_ledger.clear_reconstruction(
                row["recording"], row["method"], row["drop_set"]
            )

    def fail_result(key, status, error, stage):
        if tuple(str(value) for value in key) not in written:
            status_ledger.set_result(key, status, error, stage)

    def fail_recording(recording, status, error, stage):
        for unit in frozen_run["expected_result_units"]:
            if unit["recording"] == recording:
                fail_result(tuple(unit[name] for name in KEY_COLUMNS), status, error, stage)
        for unit in frozen_run["expected_reconstruction_units"]:
            if unit["recording"] == recording:
                status_ledger.set_reconstruction(
                    recording, unit["method"], unit["drop_set"], status, error, stage
                )

    failures = []
    if args.shard_count != 1 or args.shard_index != 0:
        parser.error("mutable directory-stride sharding is forbidden; use --task-index")
    frozen_recordings = list(frozen_run["recordings"])
    if args.task_index is not None:
        if not 0 <= args.task_index < len(frozen_recordings):
            parser.error("--task-index is outside the run manifest task mapping")
        frozen_recordings = [frozen_recordings[args.task_index]]
    selected_recording_names = {item["recording"] for item in frozen_recordings}
    status_ledger.initialize_pending(
        [
            unit for unit in frozen_run["expected_result_units"]
            if unit["recording"] in selected_recording_names
        ],
        [
            unit for unit in frozen_run["expected_reconstruction_units"]
            if unit["recording"] in selected_recording_names
        ],
        written,
        qc_written,
    )
    files = []
    for expected in frozen_recordings:
        path = os.path.abspath(expected["path"])
        files.append((path, expected))
    if not files:
        output_handle.close()
        qc_handle.close()
        raise RuntimeError("run manifest contains no selected recordings")

    try:
        for filename, expected_recording in files:
            recording = expected_recording["recording"]
            try:
                if not os.path.isfile(filename):
                    raise FileNotFoundError(f"manifest recording is missing: {filename}")
                if (
                    os.path.getsize(filename) != expected_recording["bytes"]
                    or stage0_cache.sha256_file(Path(filename)) != expected_recording["sha256"]
                ):
                    raise RuntimeError(f"manifest recording identity mismatch: {filename}")
                metadata = pilot.parse_meta(filename)
                stage0 = load_truth(filename, cache_root=args.stage0_cache_dir)
            except Exception as error:
                message = f"[missing-input] {recording}: {error}"
                print(f"  {message}")
                failures.append(message)
                fail_recording(recording, "missing_input", error, "stage0")
                continue
            truth = stage0.data
            ch_names = list(stage0.ch_names)
            positions = stage0.pos
            stage0_manifest = stage0.manifest
            identity = stage0_manifest["identity"]
            if identity["preprocessing_sha256"] != PREPROCESSING_SHA256:
                raise RuntimeError(f"unexpected preprocessing digest for {recording}")
            provenance = {
                "result_schema": RESULT_SCHEMA,
                "run_id": frozen_run["run_id"],
                "protocol_id": PROTOCOL_ID,
                "experiment_id": EXPERIMENT_ID,
                "scientific_contract_sha256": CONTRACT_SHA256,
                "preprocessing_sha256": identity["preprocessing_sha256"],
                "stage0_cache_key": identity["cache_key_sha256"],
            }
            expected_rows = set()

            for drop_set, (label, metrics) in groups.items():
                try:
                    dropped = drop_indices(ch_names, list(drop_set))
                except Exception as error:
                    message = f"[drop-set] {recording} {label}: {error}"
                    failures.append(message)
                    print(f"  {message}")
                    for metric in metrics:
                        for submetric in metric.submetrics:
                            fail_result(
                                (recording, "truth", label, "-", metric.key, submetric),
                                "missing_input", error, "drop_set_resolution",
                            )
                            for method in args.methods:
                                fail_result(
                                    (recording, "recon", label, method, metric.key, submetric),
                                    "missing_input", error, "drop_set_resolution",
                                )
                    for method in args.methods:
                        status_ledger.set_reconstruction(
                            recording, method, label, "missing_input", error,
                            "drop_set_resolution",
                        )
                    continue
                frame_id = reference_frame_id(label)
                reference_truth = pilot.surviving_average_reference(truth, dropped, ch_names)
                benchmark_truth = {}
                truth_metric_diagnostics = {}
                for metric in metrics:
                    expected_rows.update(
                        (recording, "truth", label, "-", metric.key, submetric)
                        for submetric in metric.submetrics
                    )
                    try:
                        benchmark_truth[metric.key], truth_metric_diagnostics[metric.key] = metric_evaluation(
                            metric, reference_truth, ch_names
                        )
                    except Exception as error:
                        message = f"[truthframe:{metric.key}] {recording}: {error}"
                        print(f"  {message}")
                        failures.append(message)
                        for submetric in metric.submetrics:
                            fail_result(
                                (recording, "truth", label, "-", metric.key, submetric),
                                "metric_failure", error, "truth_metric",
                            )
                            for method in args.methods:
                                fail_result(
                                    (recording, "recon", label, method, metric.key, submetric),
                                    "metric_failure", error, "truth_metric",
                                )
                        continue
                    for submetric, value in benchmark_truth[metric.key].items():
                        truth_id = make_truth_unit_id(
                            recording, identity["cache_key_sha256"], label,
                            metric.key, submetric,
                        )
                        emit({
                            "recording": recording,
                            "subject": metadata["subject"],
                            **provenance,
                            "kind": "truth",
                            "unit_status": "success",
                            "drop_set": label,
                            "reference_frame_id": frame_id,
                            "truth_unit_id": truth_id,
                            "method": "-",
                            "comparator_class": "truth",
                            "reconstruction_key": "-",
                            "gate_status": "not_applicable",
                            "power_ratio_1_45": "-",
                            "rms_ratio": "-",
                            "max_abs_uv": "-",
                            "metric": metric.key,
                            "metric_implementation": metric.implementation,
                            "metric_diagnostics_json": truth_metric_diagnostics[metric.key],
                            **metric_provenance_by_key[metric.key],
                            "submetric": submetric,
                            "truth": value,
                            "value": value,
                            "abs_err": 0,
                        })
                output_handle.flush()

                for method in args.methods:
                    method_expected = {
                        (recording, "recon", label, method, metric.key, submetric)
                        for metric in metrics for submetric in metric.submetrics
                    }
                    expected_rows.update(method_expected)
                    if (
                        method_expected.issubset(written)
                        and (recording, method, label) in qc_written
                    ):
                        print(f"  [resume] {recording} {label} {method} already scored", flush=True)
                        if (recording, method, label) in qc_written:
                            status_ledger.clear_reconstruction(recording, method, label)
                        continue

                    zuna_debug = {}
                    operation_stage = "model"
                    try:
                        if method == "zuna":
                            reconstruction = zuna_module.zuna_reconstruct(
                                stage0, dropped,
                                cache_label=f"{os.path.splitext(recording)[0]}__{label}",
                                debug=zuna_debug,
                                calibration_strategy=args.zuna_calibration,
                            )
                        else:
                            implementation_method = "linear" if method == "linear_oracle" else method
                            reconstruction = pilot.reconstruct(
                                implementation_method, reference_truth, ch_names, positions, dropped
                            )
                        operation_stage = "qc"
                        diagnostics = reconstruction_qc.evaluate_reconstruction(
                            reference_truth, reconstruction, dropped, ch_names=ch_names
                        )
                        reconstruction_key = (
                            zuna_debug["manifest"]["cache_key_sha256"]
                            if method == "zuna"
                            else diagnostics["reconstruction_sha256"]
                        )
                        qc_row = {
                            **diagnostics,
                            "result_schema": RESULT_SCHEMA,
                            "run_id": frozen_run["run_id"],
                            "unit_status": "success" if diagnostics["status"] == "pass" else "qc_failure",
                            "recording": recording,
                            "subject": metadata["subject"],
                            "protocol_id": PROTOCOL_ID,
                            "experiment_id": EXPERIMENT_ID,
                            "scientific_contract_sha256": CONTRACT_SHA256,
                            "stage0_cache_key": identity["cache_key_sha256"],
                            "drop_set": label,
                            "method": method,
                            "comparator_class": method_classes[method],
                            "reconstruction_key": reconstruction_key,
                            "zuna_cache_dir": zuna_debug.get("cache_dir", "-"),
                        }
                        if diagnostics["status"] != "pass":
                            raise ValueError("; ".join(diagnostics["reasons"]))
                        log_qc(qc_row)
                    except Exception as error:
                        message = f"[{method}] {recording} {label}: {error}"
                        print(f"  {message}")
                        failures.append(message)
                        failure_status = "model_failure" if operation_stage == "model" else "qc_failure"
                        status_ledger.set_reconstruction(
                            recording, method, label, failure_status, error, operation_stage
                        )
                        for key in method_expected:
                            fail_result(key, failure_status, error, operation_stage)
                        output_handle.flush()
                        continue

                    for metric in metrics:
                        if metric.key not in benchmark_truth:
                            continue
                        try:
                            reconstructed_values, reconstructed_diagnostics = metric_evaluation(
                                metric, reconstruction, ch_names
                            )
                        except Exception as error:
                            message = f"[{method}:{metric.key}] {recording}: {error}"
                            print(f"  {message}")
                            failures.append(message)
                            for submetric in metric.submetrics:
                                fail_result(
                                    (recording, "recon", label, method, metric.key, submetric),
                                    "metric_failure", error, "reconstruction_metric",
                                )
                            continue
                        for submetric in metric.submetrics:
                            truth_value = benchmark_truth[metric.key][submetric]
                            reconstructed_value = reconstructed_values[submetric]
                            emit({
                                "recording": recording,
                                "subject": metadata["subject"],
                                **provenance,
                                "kind": "recon",
                                "unit_status": "success",
                                "drop_set": label,
                                "reference_frame_id": frame_id,
                                "truth_unit_id": make_truth_unit_id(
                                    recording, identity["cache_key_sha256"], label,
                                    metric.key, submetric,
                                ),
                                "method": method,
                                "comparator_class": method_classes[method],
                                "reconstruction_key": reconstruction_key,
                                "gate_status": diagnostics["status"],
                                "power_ratio_1_45": diagnostics["power_ratio_1_45"],
                                "rms_ratio": diagnostics["rms_ratio"],
                                "max_abs_uv": diagnostics["max_abs_uv"],
                                "metric": metric.key,
                                "metric_implementation": metric.implementation,
                                "metric_diagnostics_json": reconstructed_diagnostics,
                                **metric_provenance_by_key[metric.key],
                                "submetric": submetric,
                                "truth": truth_value,
                                "value": reconstructed_value,
                                "abs_err": abs(reconstructed_value - truth_value),
                            })
                    output_handle.flush()

            pending = status_ledger.pending_rows(recording)
            if pending:
                failures.append(
                    f"{recording}: {len(pending)} expected units never reached an explicit outcome"
                )
                error = RuntimeError("expected unit was not reached by the runner")
                for row in pending:
                    if row["unit_type"] == "result":
                        status_ledger.set_result(
                            tuple(row[name] for name in KEY_COLUMNS),
                            "metric_failure", error, "runner_completeness",
                        )
                    else:
                        status_ledger.set_reconstruction(
                            row["recording"], row["method"], row["drop_set"],
                            "metric_failure", error, "runner_completeness",
                        )
            output_handle.flush()
            print(f"done {recording}", flush=True)
    finally:
        output_handle.close()
        qc_handle.close()

    if failures:
        raise RuntimeError("metric run incomplete:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
