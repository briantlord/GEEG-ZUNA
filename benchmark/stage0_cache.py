"""Content-addressed Stage-0 truth cache for the remediated protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

try:
    from . import pilot
    from .protocol_v2 import (
        PREPROCESSING_SHA256,
        PREPROCESSING_SPEC,
        PROTOCOL_ID,
        canonical_json,
    )
except ImportError:  # Script execution from benchmark/
    import pilot
    from protocol_v2 import (
        PREPROCESSING_SHA256,
        PREPROCESSING_SPEC,
        PROTOCOL_ID,
        canonical_json,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "results" / "stage0_cache_v3"


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


@dataclass(frozen=True)
class VerifiedStage0:
    data: np.ndarray
    ch_names: tuple[str, ...]
    pos: np.ndarray
    manifest: dict


def as_verified_stage0(loaded) -> VerifiedStage0:
    data, ch_names, pos, manifest = loaded
    output = manifest.get("output", {})
    if output.get("data_sha256") != sha256_array(data):
        raise RuntimeError("Stage-0 data array hash mismatch")
    if output.get("positions_sha256") != sha256_array(pos):
        raise RuntimeError("Stage-0 position array hash mismatch")
    if output.get("channels_sha256") != hashlib.sha256(
        canonical_json(list(ch_names)).encode("utf-8")
    ).hexdigest():
        raise RuntimeError("Stage-0 channel list hash mismatch")
    return VerifiedStage0(data=data, ch_names=tuple(ch_names), pos=pos, manifest=manifest)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("mne", "numpy", "scipy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _source_hashes() -> dict[str, str]:
    paths = {
        "contract.py": Path(__file__).with_name("contract.py").resolve(),
        "pilot.py": Path(pilot.__file__).resolve(),
        "protocol_v2.py": Path(__file__).with_name("protocol_v2.py").resolve(),
        "stage0_cache.py": Path(__file__).resolve(),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def build_identity(cnt_path: str | os.PathLike[str], n_epochs: int,
                   minimum_clean_epochs: int, emg: bool) -> dict[str, object]:
    raw_path = Path(cnt_path).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw_stat = raw_path.stat()
    identity = {
        "protocol_id": PROTOCOL_ID,
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "raw_sha256": sha256_file(raw_path),
        "raw_bytes": raw_stat.st_size,
        "target_epochs": int(n_epochs),
        "minimum_clean_epochs": int(minimum_clean_epochs),
        "emg_cleaning": bool(emg),
        "source_sha256": _source_hashes(),
        "package_versions": _package_versions(),
    }
    identity["cache_key_sha256"] = hashlib.sha256(
        canonical_json(identity).encode("utf-8")).hexdigest()
    return identity


def _entry_path(cache_root: Path, raw_path: Path, key: str) -> Path:
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "-" for c in raw_path.stem)
    return cache_root / f"{safe_stem}__{key[:20]}"


def _load_verified(entry: Path, expected_identity: dict[str, object]):
    manifest_path = entry / "manifest.json"
    tensor_path = entry / "truth.npz"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise RuntimeError(f"Incomplete Stage-0 cache entry requires inspection: {entry}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "geeg-zuna-stage0-cache-v3":
        raise RuntimeError(f"Unsupported Stage-0 cache schema: {entry}")
    if manifest.get("status") != "complete" or manifest.get("identity") != expected_identity:
        raise RuntimeError(f"Stage-0 cache identity/status mismatch: {entry}")
    expected_tensor_sha = manifest.get("artifacts", {}).get("truth_npz_sha256")
    actual_tensor_sha = sha256_file(tensor_path)
    if actual_tensor_sha != expected_tensor_sha:
        raise RuntimeError(f"Stage-0 tensor checksum mismatch: {entry}")
    with np.load(tensor_path, allow_pickle=False) as saved:
        data = saved["data"].astype(np.float32, copy=False)
        ch_names = [str(name) for name in saved["ch_names"]]
        pos = saved["pos"].astype(np.float32, copy=False)
        saved_key = str(saved["cache_key_sha256"].item())
    if saved_key != expected_identity["cache_key_sha256"]:
        raise RuntimeError(f"Stage-0 embedded cache key mismatch: {entry}")
    expected_shape = tuple(manifest["output"]["shape"])
    if data.shape != expected_shape or data.dtype != np.float32:
        raise RuntimeError(f"Stage-0 tensor metadata mismatch: {entry}")
    if data.ndim != 3 or len(ch_names) != data.shape[1] or pos.shape != (data.shape[1], 3):
        raise RuntimeError(f"Stage-0 tensor/channel/position mismatch: {entry}")
    if not np.isfinite(data).all() or not np.isfinite(pos).all():
        raise RuntimeError(f"Stage-0 cache contains non-finite values: {entry}")
    return data, ch_names, pos, manifest


def verify_entry(entry: str | os.PathLike[str], verify_raw: bool = True):
    """Standalone integrity/provenance verification for an existing cache entry."""
    entry_path = Path(entry).resolve()
    manifest_path = entry_path / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing Stage-0 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"Invalid Stage-0 identity: {entry_path}")
    if identity.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"Stage-0 protocol does not match active source: {entry_path}")
    if identity.get("preprocessing_sha256") != PREPROCESSING_SHA256:
        raise RuntimeError(f"Stage-0 preprocessing contract does not match active source: {entry_path}")
    if identity.get("source_sha256") != _source_hashes():
        raise RuntimeError(f"Stage-0 source hashes do not match active source: {entry_path}")
    if identity.get("package_versions") != _package_versions():
        raise RuntimeError(f"Stage-0 package versions do not match active environment: {entry_path}")
    loaded = _load_verified(entry_path, identity)
    if verify_raw:
        raw_path = Path(manifest.get("raw_path", ""))
        if not raw_path.is_file():
            raise RuntimeError(f"Stage-0 raw source is unavailable: {raw_path}")
        if raw_path.stat().st_size != identity.get("raw_bytes") or sha256_file(raw_path) != identity.get("raw_sha256"):
            raise RuntimeError(f"Stage-0 raw source identity mismatch: {raw_path}")
    return loaded


def load_or_create(cnt_path: str | os.PathLike[str], cache_root: str | os.PathLike[str] | None = None,
                   n_epochs: int = PREPROCESSING_SPEC["target_epochs"],
                   minimum_clean_epochs: int = PREPROCESSING_SPEC["minimum_clean_epochs"],
                   emg: bool = True):
    """Return verified truth data and manifest, creating a v3 cache on a miss."""
    if PREPROCESSING_SPEC["emg_required"] and not emg:
        raise ValueError("Remediated production Stage-0 requires ocular/muscle ICA")
    raw_path = Path(cnt_path).resolve()
    root = Path(cache_root or os.environ.get("GEEG_STAGE0_CACHE_DIR", DEFAULT_CACHE_ROOT)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = build_identity(raw_path, n_epochs, minimum_clean_epochs, emg)
    key = str(identity["cache_key_sha256"])
    entry = _entry_path(root, raw_path, key)

    if entry.exists():
        loaded = _load_verified(entry, identity)
        print(f"[stage0 v3 cache] hit -> {entry}", flush=True)
        return loaded

    lock = root / f".{key}.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another process is creating this Stage-0 entry; retry after it finishes: {lock}") from exc

    lock_owner = lock / "owner.json"
    lock_owner.write_text(json.dumps({
        "pid": os.getpid(),
        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "UNKNOWN",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "cache_key_sha256": key,
    }, indent=2) + "\n", encoding="utf-8")

    staging = root / f".{key}.tmp-{os.getpid()}"
    try:
        if staging.exists():
            raise RuntimeError(f"Refusing occupied Stage-0 staging path: {staging}")
        staging.mkdir()
        result = pilot.preprocess(
            str(raw_path), n_epochs=n_epochs, emg=emg,
            minimum_clean_epochs=minimum_clean_epochs)
        data = np.ascontiguousarray(result["data"], dtype=np.float32)
        ch_names = [str(name) for name in result["ch_names"]]
        pos = np.ascontiguousarray(result["pos"], dtype=np.float32)
        tensor_path = staging / "truth.npz"
        temporary_tensor = staging / "truth.tmp.npz"
        np.savez(
            temporary_tensor,
            data=data,
            ch_names=np.asarray(ch_names),
            pos=pos,
            cache_key_sha256=np.asarray(key),
        )
        os.replace(temporary_tensor, tensor_path)
        manifest = {
            "schema": "geeg-zuna-stage0-cache-v3",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "raw_path": str(raw_path),
            "identity": identity,
            "preprocessing_spec": PREPROCESSING_SPEC,
            "preprocessing_meta": result["meta"],
            "output": {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "channels": ch_names,
                "data_sha256": sha256_array(data),
                "positions_sha256": sha256_array(pos),
                "channels_sha256": hashlib.sha256(
                    canonical_json(ch_names).encode("utf-8")
                ).hexdigest(),
                "sfreq_hz": PREPROCESSING_SPEC["target_sfreq_hz"],
            },
            "artifacts": {
                "truth_npz": "truth.npz",
                "truth_npz_bytes": tensor_path.stat().st_size,
                "truth_npz_sha256": sha256_file(tensor_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.replace(staging, entry)
        except FileExistsError:
            # A race can only be accepted after independently verifying the winner.
            shutil.rmtree(staging)
        loaded = _load_verified(entry, identity)
        print(f"[stage0 v3 cache] created -> {entry}", flush=True)
        return loaded
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if lock_owner.exists():
            lock_owner.unlink()
        lock.rmdir()


def load_or_create_object(*args, **kwargs) -> VerifiedStage0:
    """Return the typed object required by reconstruction adapters."""
    return as_verified_stage0(load_or_create(*args, **kwargs))
