"""Collect only the exact shards declared by a run manifest, then validate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import run_manifest
import validate_bundle
from metrics.schema_v3 import COLUMNS
from stage0_cache import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    frozen = run_manifest.load_verified(args.run_manifest)
    prefix = frozen["run_id"][:16]
    expected = list(range(len(frozen["task_mapping"])))
    csv_paths = [args.shard_dir / f"shard_{prefix}_{index}.csv" for index in expected]
    qc_paths = [args.shard_dir / f"shard_{prefix}_{index}.qc.jsonl" for index in expected]
    status_paths = [args.shard_dir / f"shard_{prefix}_{index}.status.jsonl" for index in expected]
    allowed = set(csv_paths + qc_paths + status_paths)
    missing = [str(path) for path in allowed if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required run-matching shards: {missing[:3]}")
    unknown = sorted(
        path.name for path in args.shard_dir.glob(f"shard_{prefix}_*") if path not in allowed
    )
    if unknown:
        raise RuntimeError(f"Unknown/stale run-matching shards exist: {unknown[:3]}")

    result_path = args.out_prefix.with_suffix(".csv")
    qc_path = args.out_prefix.with_suffix(".qc.jsonl")
    status_path = args.out_prefix.with_suffix(".status.jsonl")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_result = result_path.with_suffix(".csv.tmp")
    temporary_qc = qc_path.with_suffix(".jsonl.tmp")
    temporary_status = status_path.with_suffix(".jsonl.tmp")
    with temporary_result.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        for path in csv_paths:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != COLUMNS:
                    raise RuntimeError(f"Shard schema mismatch: {path}")
                writer.writerows(reader)
    with temporary_qc.open("w", encoding="utf-8") as output:
        for path in qc_paths:
            content = path.read_text(encoding="utf-8")
            output.write(content)
            if content and not content.endswith("\n"):
                output.write("\n")
    with temporary_status.open("w", encoding="utf-8") as output:
        for path in status_paths:
            content = path.read_text(encoding="utf-8")
            output.write(content)
            if content and not content.endswith("\n"):
                output.write("\n")
    temporary_result.replace(result_path)
    temporary_qc.replace(qc_path)
    temporary_status.replace(status_path)

    report = validate_bundle.validate(
        args.run_manifest, result_path, qc_path, status_path=status_path,
        require_all_success=False,
    )
    bundle = {
        "schema": "geeg-zuna-result-bundle-v1",
        "run_id": frozen["run_id"],
        "run_manifest": {"path": str(args.run_manifest.resolve()), "sha256": sha256_file(args.run_manifest)},
        "results": {"path": str(result_path.resolve()), "sha256": sha256_file(result_path)},
        "qc": {"path": str(qc_path.resolve()), "sha256": sha256_file(qc_path)},
        "status_ledger": {
            "path": str(status_path.resolve()), "sha256": sha256_file(status_path)
        },
        "validation": report,
    }
    bundle_path = args.out_prefix.with_suffix(".bundle.json")
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
