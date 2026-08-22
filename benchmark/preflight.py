"""Fail-closed environment, input, model, GPU, and run-identity preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch
import zuna

import contract
import run_manifest
import zuna_method_v11


EXACT_PACKAGES = {
    "mne": "1.12.1", "numpy": "2.4.6", "scipy": "1.17.1",
    "specparam": "2.0.0rc7", "scikit-learn": "1.9.0",
    "zuna": "1.1.3", "matplotlib": "3.11.1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument("--allow-development-contract", action="store_true")
    args = parser.parse_args()

    frozen = run_manifest.load_verified(args.run_manifest)
    if frozen["source_sha256"] != run_manifest.source_hashes():
        raise RuntimeError("Active source differs from the immutable run manifest")
    if not args.allow_development_contract:
        contract.assert_production_ready()
    if not 0 <= args.task_index < len(frozen["task_mapping"]):
        raise RuntimeError("Task index is outside the immutable task mapping")

    versions = {name: importlib.metadata.version(name) for name in EXACT_PACKAGES}
    mismatches = {
        name: {"expected": expected, "actual": versions[name]}
        for name, expected in EXACT_PACKAGES.items() if versions[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"Package lock mismatch: {mismatches}")
    if not torch.__version__.startswith("2.6.0+cu"):
        raise RuntimeError(f"Expected CUDA-enabled torch 2.6.0, found {torch.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Task must see exactly one scheduler-assigned CUDA device")
    if not hasattr(zuna, "reconstruct_fif"):
        raise RuntimeError(f"Wrong zuna module imported: {zuna.__file__}")
    if Path(zuna.__file__).resolve().is_relative_to(Path.cwd().resolve()):
        raise RuntimeError(f"Project-local zuna import is forbidden: {zuna.__file__}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required")

    expected_model = frozen.get("model")
    actual_model = zuna_method_v11._model_provenance()["identity"]
    if expected_model != actual_model:
        raise RuntimeError("Pinned local model differs from the run manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.output_dir)
    if usage.free < args.minimum_free_gb * 1024 ** 3:
        raise RuntimeError(f"Insufficient free disk: {usage.free / 1024 ** 3:.1f} GiB")
    probe = args.output_dir / f".preflight-write-{os.getpid()}"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()

    report = {
        "status": "pass",
        "run_id": frozen["run_id"],
        "task_index": args.task_index,
        "recording": frozen["task_mapping"][args.task_index]["recording"],
        "python": sys.version,
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "model": actual_model,
        "slurm": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
        "free_disk_gb": usage.free / 1024 ** 3,
    }
    destination = args.output_dir / f"preflight_{frozen['run_id'][:16]}_{args.task_index}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

