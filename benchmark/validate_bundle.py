"""Fail-closed validation against an immutable expected-unit manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

try:
    from . import run_manifest
    from .metrics import aggregate
    from .metrics.schema_v3 import COLUMNS, KEY_COLUMNS, RESULT_SCHEMA
except ImportError:
    import run_manifest
    from metrics import aggregate
    from metrics.schema_v3 import COLUMNS, KEY_COLUMNS, RESULT_SCHEMA


def result_key(row: dict) -> tuple:
    return tuple(str(row[key]) for key in KEY_COLUMNS)


def reconstruction_key(row: dict) -> tuple:
    return (str(row["recording"]), str(row["method"]), str(row["drop_set"]))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing required JSONL: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def load_statuses(path: Path | None, terminal_states: set[str]):
    results, reconstructions = {}, {}
    if path is None:
        return results, reconstructions
    for row in read_jsonl(path):
        if not isinstance(row.get("run_id"), str) or not row["run_id"]:
            raise ValueError("Failure-ledger row lacks a run ID")
        status = row.get("status")
        if status not in terminal_states or status == "success":
            raise ValueError(f"Invalid failure-ledger status: {status!r}")
        unit_type = row.get("unit_type")
        if unit_type == "result":
            key = tuple(str(row[name]) for name in KEY_COLUMNS)
            target = results
        elif unit_type == "reconstruction":
            key = reconstruction_key(row)
            target = reconstructions
        else:
            raise ValueError(f"Invalid failure-ledger unit_type: {unit_type!r}")
        if key in target:
            raise ValueError(f"Duplicate failure-ledger unit: {key}")
        target[key] = status
    return results, reconstructions


def validate(run_path: Path, result_path: Path, qc_path: Path,
             status_path: Path | None = None, require_all_success: bool = False) -> dict:
    frozen = run_manifest.load_verified(run_path)
    if frozen["source_sha256"] != run_manifest.source_hashes():
        raise ValueError("Active source differs from the frozen run manifest")

    with result_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Result CSV has the wrong schema/columns")
        result_rows = list(reader)
    successful_results = {}
    for row in result_rows:
        if row["result_schema"] != RESULT_SCHEMA or row["run_id"] != frozen["run_id"]:
            raise ValueError("Result row has a mixed schema or run ID")
        if row["unit_status"] != "success":
            raise ValueError("Non-success rows belong in the explicit failure ledger")
        key = result_key(row)
        if key in successful_results:
            raise ValueError(f"Duplicate successful result unit: {key}")
        successful_results[key] = row

    qc_rows = read_jsonl(qc_path)
    successful_reconstructions = {}
    for row in qc_rows:
        if row.get("run_id") != frozen["run_id"] or row.get("unit_status") != "success":
            raise ValueError("QC row has a mixed run ID or non-success state")
        key = reconstruction_key(row)
        if key in successful_reconstructions:
            raise ValueError(f"Duplicate reconstruction QC unit: {key}")
        successful_reconstructions[key] = row

    terminal_states = set(frozen["terminal_states"])
    failed_results, failed_reconstructions = load_statuses(status_path, terminal_states)
    for row in read_jsonl(status_path) if status_path is not None else []:
        if row.get("run_id") != frozen["run_id"]:
            raise ValueError("Failure-ledger row has a mixed run ID")
    overlap = set(successful_results) & set(failed_results)
    if overlap:
        raise ValueError(f"Result units are both success and failure: {sorted(overlap)[:3]}")
    overlap = set(successful_reconstructions) & set(failed_reconstructions)
    if overlap:
        raise ValueError(f"Reconstruction units are both success and failure: {sorted(overlap)[:3]}")

    expected_results = {
        (
            row["recording"], row["kind"], row["drop_set"], row["method"],
            row["metric"], row["submetric"],
        )
        for row in frozen["expected_result_units"]
    }
    actual_results = set(successful_results) | set(failed_results)
    missing_results = expected_results - actual_results
    extra_results = actual_results - expected_results
    if missing_results or extra_results:
        raise ValueError(
            f"Result-unit completeness failure: missing={len(missing_results)}, extra={len(extra_results)}"
        )

    expected_reconstructions = {
        (row["recording"], row["method"], row["drop_set"])
        for row in frozen["expected_reconstruction_units"]
    }
    actual_reconstructions = set(successful_reconstructions) | set(failed_reconstructions)
    missing_reconstructions = expected_reconstructions - actual_reconstructions
    extra_reconstructions = actual_reconstructions - expected_reconstructions
    if missing_reconstructions or extra_reconstructions:
        raise ValueError(
            "Reconstruction-unit completeness failure: "
            f"missing={len(missing_reconstructions)}, extra={len(extra_reconstructions)}"
        )

    for key, row in successful_reconstructions.items():
        if key[1] != "zuna":
            continue
        cache_dir = Path(row.get("zuna_cache_dir", ""))
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Successful ZUNA unit lacks a reconstruction manifest: {key}")
        reconstruction = json.loads(manifest_path.read_text(encoding="utf-8"))
        if reconstruction.get("schema") != "geeg-zuna-reconstruction-cache-v3":
            raise ValueError(f"ZUNA unit uses the wrong cache schema: {key}")
        if reconstruction.get("status") != "complete" or reconstruction.get("settings", {}).get("sample_steps") != 50:
            raise ValueError(f"ZUNA unit is incomplete or did not use 50 steps: {key}")
        if reconstruction.get("shape", [0])[0] != 64:
            raise ValueError(f"ZUNA unit did not reconstruct 64 epochs: {key}")
        if reconstruction.get("model", {}).get("identity") != frozen.get("model"):
            raise ValueError(f"ZUNA model identity differs from run manifest: {key}")
        if (
            reconstruction.get("settings", {}).get("calibration_strategy")
            != frozen.get("calibration_strategy")
        ):
            raise ValueError(f"ZUNA calibration differs from run manifest: {key}")

    failures = Counter(failed_results.values()) + Counter(failed_reconstructions.values())
    if require_all_success and failures:
        raise ValueError(f"Bundle contains terminal failures: {dict(failures)}")

    # The strict numeric validator is meaningful only when all expected result rows succeeded.
    if not failed_results:
        aggregate.load_validated_rows(result_path)

    return {
        "status": "pass",
        "run_id": frozen["run_id"],
        "result_units": len(actual_results),
        "reconstruction_units": len(actual_reconstructions),
        "failures": dict(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--status-ledger", type=Path, default=None)
    parser.add_argument("--require-all-success", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(
        args.run_manifest, args.results, args.qc, args.status_ledger,
        require_all_success=args.require_all_success,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
