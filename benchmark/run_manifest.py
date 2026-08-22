"""Create and verify the immutable expected-unit manifest for an experiment."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from .contract import CONTRACT, CONTRACT_SHA256, EXPERIMENT_ID
    from .protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID, canonical_json
    from .stage0_cache import sha256_file
except ImportError:
    from contract import CONTRACT, CONTRACT_SHA256, EXPERIMENT_ID
    from protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID, canonical_json
    from stage0_cache import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RECORDING_RE = re.compile(r"G(?P<subject>\d+)Day(?P<day>\d+)Rest(?P<rest>\d+)", re.I)
RUN_MANIFEST_SCHEMA = "geeg-zuna-run-manifest-v1"
DEFAULT_CALIBRATION = CONTRACT["zuna_scaling"]["primary_deployable_strategy"]
CALIBRATION_STRATEGIES = [
    DEFAULT_CALIBRATION,
    *[
        name for name in CONTRACT["zuna_scaling"]["predeclared_sensitivities"]
        if name != "normalized_scale_free_output"
    ],
]


def git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return f"UNCOMMITTED@{commit}" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED"


def source_hashes() -> dict[str, str]:
    paths = []
    for directory in (ROOT / "benchmark", ROOT / "config", ROOT / "scripts"):
        if directory.is_dir():
            paths.extend(
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in {".py", ".json", ".sh", ".ps1"}
            )
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(paths)
        if "__pycache__" not in path.parts
    }


def parse_recording(path: Path) -> dict:
    match = RECORDING_RE.search(path.name)
    if not match:
        raise ValueError(f"Recording filename does not encode subject/day/rest: {path.name}")
    return {
        "recording": path.name,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "subject": "G" + match.group("subject"),
        "day": int(match.group("day")),
        "rest": int(match.group("rest")),
    }


def expected_units(recordings: list[dict], methods: list[str]):
    rows = []
    reconstructions = []
    metric_items = [
        (key, value) for key, value in CONTRACT["metrics"].items()
        if isinstance(value, dict) and "drop_set" in value and "submetrics" in value
    ]
    for recording in recordings:
        for metric, definition in metric_items:
            drop_set = "+".join(sorted(name.upper() for name in definition["drop_set"]))
            for submetric in definition["submetrics"]:
                rows.append({
                    "recording": recording["recording"], "kind": "truth",
                    "drop_set": drop_set, "method": "-", "metric": metric,
                    "submetric": submetric,
                })
                for method in methods:
                    rows.append({
                        "recording": recording["recording"], "kind": "recon",
                        "drop_set": drop_set, "method": method, "metric": metric,
                        "submetric": submetric,
                    })
        drop_sets = sorted({
            "+".join(sorted(name.upper() for name in definition["drop_set"]))
            for _metric, definition in metric_items
        })
        for method in methods:
            for drop_set in drop_sets:
                reconstructions.append({
                    "recording": recording["recording"],
                    "method": method,
                    "drop_set": drop_set,
                    "requested_maximum_epochs": 64,
                })
    return rows, reconstructions


def create_manifest(files: list[Path], methods: list[str], model: dict | None = None,
                    calibration_strategy: str | None = None) -> dict:
    if not files:
        raise ValueError("Run manifest requires at least one recording")
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError("Methods must be a non-empty unique list")
    if "zuna" in methods:
        calibration_strategy = calibration_strategy or DEFAULT_CALIBRATION
        if calibration_strategy not in CALIBRATION_STRATEGIES:
            raise ValueError(f"Unsupported ZUNA calibration strategy: {calibration_strategy}")
    elif calibration_strategy is not None:
        raise ValueError("A ZUNA calibration strategy requires the zuna method")
    recordings = [parse_recording(path.resolve()) for path in sorted(files)]
    names = [item["recording"] for item in recordings]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate recording filenames are forbidden")
    rows, reconstructions = expected_units(recordings, methods)
    pairs = {}
    for item in recordings:
        key = f"{item['subject']}:Day{item['day']}"
        pairs.setdefault(key, {})[str(item["rest"])] = item["recording"]
    source = source_hashes()
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "protocol_id": PROTOCOL_ID,
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "git_commit": git_commit(),
        "source_sha256": source,
        "recordings": recordings,
        "methods": methods,
        "model": model,
        "calibration_strategy": calibration_strategy,
        "coordinate_strategy": {
            "input": CONTRACT["coordinates"]["input"],
            "transform": CONTRACT["coordinates"]["transform"],
            "out_of_bounds_policy": CONTRACT["coordinates"]["out_of_bounds_policy"],
        },
        "expected_result_units": rows,
        "expected_reconstruction_units": reconstructions,
        "pair_structure": pairs,
    }
    run_id = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        **identity,
        "expected_counts": {
            "recordings": len(recordings),
            "result_units": len(rows),
            "reconstruction_units": len(reconstructions),
            "truth_units": sum(row["kind"] == "truth" for row in rows),
            "recon_result_units": sum(row["kind"] == "recon" for row in rows),
        },
        "task_mapping": [
            {"array_index": index, "recording": item["recording"], "sha256": item["sha256"]}
            for index, item in enumerate(recordings)
        ],
        "terminal_states": CONTRACT["failure_policy"]["terminal_states"],
    }


def load_verified(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError("Unsupported run-manifest schema")
    identity_keys = (
        "experiment_id", "scientific_contract_sha256", "protocol_id",
        "preprocessing_sha256", "git_commit", "source_sha256", "recordings",
        "methods", "model", "expected_result_units",
        "calibration_strategy", "coordinate_strategy",
        "expected_reconstruction_units", "pair_structure",
    )
    identity = {key: manifest[key] for key in identity_keys}
    expected_id = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    if manifest.get("run_id") != expected_id:
        raise ValueError("Run manifest identity hash mismatch")
    if manifest["experiment_id"] != EXPERIMENT_ID or manifest["scientific_contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("Run manifest does not match the active scientific contract")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path)
    source.add_argument("--recordings", nargs="+", type=Path)
    parser.add_argument("--methods", nargs="+", default=["spline", "zuna"])
    parser.add_argument("--zuna-calibration", choices=CALIBRATION_STRATEGIES,
                        default=DEFAULT_CALIBRATION)
    parser.add_argument("--expected-recordings", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = list(args.recordings) if args.recordings else [
        Path(path) for path in glob.glob(str(args.data_dir / "*.cnt"))
    ]
    if args.expected_recordings is not None and len(files) != args.expected_recordings:
        raise RuntimeError(
            f"Expected exactly {args.expected_recordings} recordings, found {len(files)}"
        )
    model = None
    if "zuna" in args.methods:
        try:
            import zuna_method_v11
            model = zuna_method_v11._model_provenance()["identity"]
        except Exception as error:
            raise RuntimeError(f"Cannot freeze ZUNA model identity: {error}") from error
    calibration = args.zuna_calibration if "zuna" in args.methods else None
    manifest = create_manifest(
        files, args.methods, model=model, calibration_strategy=calibration
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": manifest["run_id"], **manifest["expected_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
