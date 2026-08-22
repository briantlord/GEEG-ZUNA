"""Build a deterministic, source-only HPC upload bundle.

The destination must be outside the authoritative source set. Generated files
receive a SHA-256 manifest; raw data, model weights, environments, caches, prior
results, and archived legacy code are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_FILES = (
    "README.md",
    "requirements.txt",
    "requirements.lock.txt",
    "BENCHMARK_PROTOCOL.md",
    "PHASE1_CORRECTED_PROTOCOL.md",
    "PHASE2_ZUNA_INTEGRATION.md",
    "PHASE3_METRIC_CORRECTNESS.md",
    "CODE_AUDIT_REPORT_2026-08-21.md",
    "CODE_REMEDIATION_PLAN_2026-08-21.md",
    "COORDINATE_EVIDENCE_2026-08-21.md",
    "CNT_FORMAT_EVIDENCE_2026-08-21.md",
    "run_zuna11_local_one_record.ps1",
)
INCLUDE_DIRS = ("benchmark", "config")
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".npy", ".npz", ".png"}
EXCLUDED_GLOBS = ("*_smoke.csv",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def include_source(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_NAMES for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return not any(path.match(pattern) for pattern in EXCLUDED_GLOBS)


def copy_sources(staging: Path) -> None:
    for relative in INCLUDE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required release file is missing: {source}")
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for relative in INCLUDE_DIRS:
        source_dir = ROOT / relative
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Required release directory is missing: {source_dir}")
        for source in sorted(source_dir.rglob("*")):
            if source.is_file() and include_source(source):
                target = staging / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def write_manifest(staging: Path) -> None:
    files = []
    for path in sorted(
        staging.rglob("*"), key=lambda item: item.relative_to(staging).as_posix()
    ):
        if path.is_file() and path.name != "RELEASE_MANIFEST.json":
            files.append({
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    content_hash = hashlib.sha256(
        "\n".join(f"{item['sha256']}  {item['path']}" for item in files).encode()
    ).hexdigest()
    manifest = {
        "schema": "geeg-zuna-release-v1",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "content_hash": content_hash,
        "files": files,
    }
    (staging / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def build(destination: Path) -> None:
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        # A generated child directory is allowed, but never overwrite source/data dirs.
        forbidden = {ROOT / name for name in ("benchmark", "config", "GEEG_Raw", "HF_cache", "results", "archive")}
        if destination in forbidden:
            raise ValueError(f"Refusing to overwrite protected directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geeg_zuna_release_", dir=destination.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        copy_sources(staging)
        write_manifest(staging)
        if destination.exists():
            marker = destination / "RELEASE_MANIFEST.json"
            if not marker.is_file():
                raise RuntimeError(
                    f"Refusing to replace unmarked directory {destination}; "
                    "move it aside once, then rebuild."
                )
            shutil.rmtree(destination)
        staging.replace(destination)

    # Import locally so the builder remains usable as a standalone script.
    from verify_hpc_share import verify
    verify(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.destination)
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
