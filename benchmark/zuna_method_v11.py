"""Leakage-free, boundary-safe ZUNA 1.1 adapter for the remediated protocol.

The released FIF loader is retained for model tokenization and inference, but
its unsafe inputs are removed before it runs:

* each real 5-second epoch is a separate FIF (no artificial joins);
* no second filtering or average reference is allowed;
* held-out waveforms are replaced before serialization;
* deterministic calibration carriers derive their scale only from surviving
  channels, so inverse z-scoring cannot use held-out mean/std;
* no post-hoc fit to truth or observed model output is performed.

The complete blind inputs, official model outputs, masks, reconstruction, and
provenance are stored in a corrected-v2 content-addressed cache.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

try:
    from . import pilot, stage0_cache
    from .contract import CONTRACT, CONTRACT_SHA256, EXPERIMENT_ID
    from .protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID
except ImportError:
    import pilot, stage0_cache
    from contract import CONTRACT, CONTRACT_SHA256, EXPERIMENT_ID
    from protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID


_run_subprocess = subprocess.run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HELPER = HERE / "_recon11.py"
SFREQ = 256
EPOCH_SAMPLES = 1280
ADAPTER_VERSION = "zuna11-masking-aware-per-epoch-v3"
MODEL_REPOSITORY = "Zyphra/ZUNA1.1"
POSITION_MIN = np.asarray(CONTRACT["coordinates"]["zuna_expected_bounds_m"][0], dtype=np.float32)
POSITION_MAX = np.asarray(CONTRACT["coordinates"]["zuna_expected_bounds_m"][1], dtype=np.float32)
POSITION_BINS = 100
CALIBRATION_STRATEGIES = (
    "median_survivor_std_zero_mean_carrier",
    "spatial_neighbor_robust_std_zero_mean_carrier",
    "observed_position_logstd_ridge_zero_mean_carrier",
)

os.environ.setdefault("HF_HOME", str(ROOT / "HF_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "HF_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("zuna", "torch", "mne", "numpy", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _zuna_package_root() -> Path:
    distribution = importlib.metadata.distribution("zuna")
    path = Path(distribution.locate_file("zuna")).resolve()
    if not path.is_dir():
        raise RuntimeError(f"Cannot locate pip-installed ZUNA package: {path}")
    return path


def _code_provenance() -> dict[str, str]:
    zuna_root = _zuna_package_root()
    paths = {
        "adapter": Path(__file__).resolve(),
        "helper": HELPER,
        "zuna_pipeline": zuna_root / "pipeline.py",
        "zuna_eeg_data": (
            zuna_root / "inference" / "AY2l" / "lingua" / "apps"
            / "AY2latent_bci" / "eeg_data.py"
        ),
        "zuna_fif_config": (
            zuna_root / "inference" / "AY2l" / "lingua" / "apps"
            / "AY2latent_bci" / "configs" / "config_infer_fif.yaml"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing code provenance files: " + ", ".join(missing))
    return {name: _sha256_file(path) for name, path in paths.items()}


def _model_provenance() -> dict[str, object]:
    cache = Path(os.environ["HF_HOME"]).resolve() / "models--Zyphra--ZUNA1.1"
    ref = cache / "refs" / "main"
    if not ref.is_file():
        raise RuntimeError(f"Missing local ZUNA1.1 Hugging Face revision: {ref}")
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = cache / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError(f"Missing local ZUNA1.1 snapshot: {snapshot}")

    weight_candidates = sorted(snapshot.glob("*.safetensors"))
    if not weight_candidates:
        weight_candidates = sorted(
            path for path in (cache / "blobs").glob("*")
            if path.is_file() and path.stat().st_size > 100_000_000
        )
    if len(weight_candidates) != 1:
        raise RuntimeError(f"Expected one ZUNA1.1 weight file, found {weight_candidates}")
    weight = weight_candidates[0]
    weight_sha = _sha256_file(weight)
    config = snapshot / "config.json"
    if not config.is_file():
        raise RuntimeError(f"Missing model config: {config}")
    return {
        "identity": {
            "repository": MODEL_REPOSITORY,
            "revision": revision,
            "weight_sha256": weight_sha,
            "weight_bytes": weight.stat().st_size,
            "config_sha256": _sha256_file(config),
        },
        "locations": {
            "cache": str(cache),
            "snapshot": str(snapshot),
            "weight": str(weight.resolve()),
            "config": str(config.resolve()),
        },
    }


def _validate_inputs(stage0, dropped):
    if not isinstance(stage0, stage0_cache.VerifiedStage0):
        raise TypeError("ZUNA requires a stage0_cache.VerifiedStage0 object, not a naked array")
    data = np.asarray(stage0.data)
    ch_names = list(stage0.ch_names)
    pos = np.asarray(stage0.pos)
    manifest = stage0.manifest
    if manifest.get("schema") != "geeg-zuna-stage0-cache-v4":
        raise RuntimeError("ZUNA requires a verified Stage-0 v4 manifest")
    identity = manifest.get("identity", {})
    if identity.get("protocol_id") != PROTOCOL_ID or identity.get("preprocessing_sha256") != PREPROCESSING_SHA256:
        raise RuntimeError("Stage-0 identity does not match the active remediated protocol")
    stage0_cache.as_verified_stage0((data, ch_names, pos, manifest))
    if data.ndim != 3 or data.shape[1] != len(ch_names):
        raise ValueError(f"Expected (epochs, channels, time), got {data.shape}")
    if data.shape[2] != EPOCH_SAMPLES:
        raise ValueError(f"Corrected-v2 requires {EPOCH_SAMPLES}-sample epochs, got {data.shape[2]}")
    if pos.shape != (len(ch_names), 3) or not np.isfinite(pos).all():
        raise ValueError(f"Invalid channel positions: {pos.shape}")
    if not np.isfinite(data).all():
        raise ValueError("ZUNA input contains non-finite samples")
    dropped = [int(index) for index in dropped]
    if not dropped or len(set(dropped)) != len(dropped):
        raise ValueError("Dropped indices must be non-empty and unique")
    if min(dropped) < 0 or max(dropped) >= len(ch_names):
        raise ValueError(f"Dropped index out of range: {dropped}")
    good = [index for index in range(len(ch_names)) if index not in dropped]
    if len(good) < 3:
        raise ValueError("ZUNA requires at least three surviving channels")
    discrete = (((pos - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)) * POSITION_BINS)
    discrete = np.clip(discrete.astype(np.int64), 0, POSITION_BINS - 1)
    unique, counts = np.unique(discrete, axis=0, return_counts=True)
    if np.any(counts > 1):
        collisions = [
            [ch_names[index] for index in np.flatnonzero(np.all(discrete == token, axis=1))]
            for token in unique[counts > 1]
        ]
        raise ValueError(f"ZUNA coordinate token collision after official discretization: {collisions}")
    referenced = pilot.surviving_average_reference(data, dropped, ch_names)
    return (
        np.ascontiguousarray(referenced, dtype=np.float32), list(ch_names),
        np.asarray(pos, np.float32), discrete, dropped, good, manifest,
    )


def _calibration_scales(data, pos, dropped, good, strategy):
    good_std_uv = np.std(data[:, good, :], axis=-1).astype(np.float64)
    if not np.isfinite(good_std_uv).all() or np.any(good_std_uv <= 0):
        raise RuntimeError("Invalid per-channel scale among surviving channels")
    ne, n_drop = data.shape[0], len(dropped)
    if strategy == "median_survivor_std_zero_mean_carrier":
        scale_uv = np.repeat(
            np.median(good_std_uv, axis=1)[:, None], n_drop, axis=1
        )
    elif strategy == "spatial_neighbor_robust_std_zero_mean_carrier":
        distance = np.linalg.norm(
            np.asarray(pos)[dropped, None, :] - np.asarray(pos)[None, good, :], axis=-1
        )
        neighbor_count = min(4, len(good))
        neighbors = np.argsort(distance, axis=1)[:, :neighbor_count]
        scale_uv = np.empty((ne, n_drop), dtype=np.float64)
        for drop_order in range(n_drop):
            scale_uv[:, drop_order] = np.median(
                good_std_uv[:, neighbors[drop_order]], axis=1
            )
    elif strategy == "observed_position_logstd_ridge_zero_mean_carrier":
        # Per epoch, fit log(SD) from only observed channel coordinates. The
        # fixed ridge penalty is predeclared and cannot inspect target samples.
        good_pos = np.asarray(pos, dtype=np.float64)[good]
        dropped_pos = np.asarray(pos, dtype=np.float64)[dropped]
        center = good_pos.mean(axis=0)
        spread = good_pos.std(axis=0)
        spread[spread == 0] = 1.0
        design = np.column_stack((np.ones(len(good)), (good_pos - center) / spread))
        target_design = np.column_stack((
            np.ones(n_drop), (dropped_pos - center) / spread
        ))
        penalty = np.diag([0.0, 0.1, 0.1, 0.1])
        projector = np.linalg.solve(design.T @ design + penalty, design.T)
        scale_uv = np.exp((target_design @ projector @ np.log(good_std_uv).T).T)
    else:
        raise ValueError(
            f"Unknown ZUNA calibration strategy {strategy!r}; expected {CALIBRATION_STRATEGIES}"
        )
    if scale_uv.shape != (ne, n_drop) or not np.isfinite(scale_uv).all() or np.any(scale_uv <= 0):
        raise RuntimeError(f"Invalid blind calibration scale for strategy {strategy}")
    return np.asarray(scale_uv, dtype=np.float32)


def _blind_calibration_input(data: np.ndarray, pos, dropped: list[int], good: list[int],
                             strategy="median_survivor_std_zero_mean_carrier"):
    """Replace every hidden waveform using only surviving-channel scale.

    The official V4 loader adds 1e-6 V to every channel's segment std. Each
    strategy derives a carrier scale exclusively from surviving-channel samples
    and fixed channel positions, never from a held target.
    """
    ne, _, nt = data.shape
    blind = data.copy()
    scale_uv = _calibration_scales(data, pos, dropped, good, strategy)

    sample = np.arange(nt, dtype=np.float64)
    n_drop = len(dropped)
    carriers = []
    for order in range(n_drop):
        phase = 2.0 * np.pi * order / n_drop
        carrier = (
            np.sin(2.0 * np.pi * 7.0 * sample / nt + phase)
            + 0.25 * np.sin(2.0 * np.pi * 13.0 * sample / nt + 2.0 * phase)
        )
        carrier -= carrier.mean()
        carrier /= carrier.std()
        carriers.append(carrier.astype(np.float32))
    carriers = np.asarray(carriers, dtype=np.float32)
    blind[:, dropped, :] = scale_uv[:, :, None] * carriers[None, :, :]
    return np.ascontiguousarray(blind), scale_uv, carriers


def _settings(sample_steps, segment_sec, highpass_hz, seqlen, seed,
              calibration_strategy="median_survivor_std_zero_mean_carrier"):
    if float(segment_sec) != 5.0:
        raise ValueError("Phase 2 requires one real 5-second epoch per ZUNA segment")
    if highpass_hz is not None:
        raise ValueError("Phase 2 forbids a second ZUNA highpass; use highpass_hz=None")
    if calibration_strategy not in CALIBRATION_STRATEGIES:
        raise ValueError(f"Unsupported calibration strategy: {calibration_strategy}")
    return {
        "adapter_version": ADAPTER_VERSION,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "zuna_model": "1.1",
        "sfreq_hz": SFREQ,
        "epoch_samples": EPOCH_SAMPLES,
        "segment_sec": 5.0,
        "sample_steps": int(sample_steps),
        "diffusion_cfg": 1.0,
        "seed": int(seed),
        "target_packed_seqlen": int(seqlen),
        "zuna_highpass_hz": None,
        "zuna_lowpass_hz": None,
        "zuna_notch_hz": None,
        "zuna_average_reference": False,
        "zuna_zscore": "per-segment per-channel",
        "calibration_strategy": calibration_strategy,
        "inverse_scale": "per-epoch/per-target blind calibration carrier scale",
        "inverse_mean_uv": 0.0,
        "self_calibration": False,
        "epoch_serialization": "one FIF per real epoch",
    }


def _cache_location(blind, ch_names, pos, discrete_pos, dropped, cache_label, settings,
                    model_identity, code_sha, stage0_key):
    payload = {
        "blind_input_sha256": _sha256_array(blind),
        "channels": list(ch_names),
        "positions_sha256": _sha256_array(pos),
        "model_positions_sha256": _sha256_array(np.clip(pos, POSITION_MIN, POSITION_MAX)),
        "discrete_positions_sha256": _sha256_array(discrete_pos),
        "dropped": list(dropped),
        "stage0_cache_key": stage0_key,
        "settings": settings,
        "model": model_identity,
        "code_sha256": code_sha,
        "packages": _package_versions(),
    }
    key = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    label = cache_label or ("drop-" + "-".join(ch_names[index] for index in dropped))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")[:120] or "reconstruction"
    root = Path(os.environ.get(
        "ZUNA11_RECON_CACHE_DIR_V3", ROOT / "results" / "zuna11_reconstructions_v3"
    )).resolve()
    return root, root / f"{slug}__{key[:20]}", key, payload


def _write_manifest(path: Path, payload: dict[str, object]):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_epoch_fifs(input_dir: Path, blind, ch_names, pos, dropped):
    import mne

    mne.set_log_level("ERROR")
    montage = mne.channels.make_dig_montage(
        ch_pos={name: np.asarray(pos[index], float) for index, name in enumerate(ch_names)},
        coord_frame="head",
    )
    bad_names = [ch_names[index] for index in dropped]
    for epoch_index, epoch in enumerate(blind):
        info = mne.create_info(list(ch_names), SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(epoch.astype(np.float64) * 1e-6, info, verbose="ERROR")
        raw.set_montage(montage, match_case=True, on_missing="raise")
        raw.info["bads"] = list(bad_names)
        raw.save(input_dir / f"epoch_{epoch_index:04d}_raw.fif", overwrite=True, verbose="ERROR")


def _expanded_mask(mask_path: Path, ch_names, nt):
    with np.load(mask_path, allow_pickle=False) as saved:
        mask = np.asarray(saved["mask"], dtype=bool)
        names = [str(name) for name in saved["ch_names"]]
        tf = int(saved["num_fine_time_pts"]) if "num_fine_time_pts" in saved.files else None
    if mask.ndim != 2:
        raise RuntimeError(f"Invalid ZUNA mask shape in {mask_path}: {mask.shape}")
    if mask.shape[1] == nt:
        expanded = mask
    elif tf and mask.shape[1] == (nt + tf - 1) // tf:
        expanded = np.repeat(mask, tf, axis=1)[:, :nt]
    else:
        raise RuntimeError(f"Unsupported ZUNA mask resolution in {mask_path}: {mask.shape}")
    row = {name.upper(): index for index, name in enumerate(names)}
    try:
        return np.asarray([expanded[row[name.upper()]] for name in ch_names], dtype=bool)
    except KeyError as exc:
        raise RuntimeError(f"Mask channel mismatch in {mask_path}: {exc}") from exc


def _read_one_model_output(model_output: Path, epoch_index, ch_names, dropped, good, nt):
    import mne

    base = f"epoch_{epoch_index:04d}"
    full_path = model_output / "full_reconstruction" / f"{base}_raw.fif"
    hybrid_path = model_output / "hybrid" / f"{base}_raw.fif"
    mask_path = model_output / "hybrid" / f"{base}_mask.npz"
    if not full_path.is_file() or not hybrid_path.is_file() or not mask_path.is_file():
        raise RuntimeError(f"Missing ZUNA epoch output/mask for {base}")
    raw = mne.io.read_raw_fif(hybrid_path, preload=True, verbose="ERROR")
    name_to_index = {name.upper(): index for index, name in enumerate(raw.ch_names)}
    if any(name.upper() not in name_to_index for name in ch_names):
        raise RuntimeError(f"ZUNA hybrid output channel mismatch: {hybrid_path}")
    picks = [name_to_index[name.upper()] for name in ch_names]
    epoch = raw.get_data(picks=picks) * 1e6
    if epoch.shape != (len(ch_names), nt) or not np.isfinite(epoch).all():
        raise RuntimeError(f"Invalid ZUNA output shape/values in {hybrid_path}: {epoch.shape}")
    mask = _expanded_mask(mask_path, ch_names, nt)
    if not mask[dropped].all() or mask[good].any():
        raise RuntimeError(f"ZUNA mask is not exactly the requested whole-channel mask: {mask_path}")
    full_raw = mne.io.read_raw_fif(full_path, preload=True, verbose="ERROR")
    full_map = {name.upper(): index for index, name in enumerate(full_raw.ch_names)}
    if any(name.upper() not in full_map for name in ch_names):
        raise RuntimeError(f"ZUNA full output channel mismatch: {full_path}")
    full_epoch = full_raw.get_data(
        picks=[full_map[name.upper()] for name in ch_names]
    ) * 1e6
    if full_epoch.shape != epoch.shape or not np.isfinite(full_epoch).all():
        raise RuntimeError(f"Invalid ZUNA full output shape/values in {full_path}: {full_epoch.shape}")
    if not np.array_equal(full_epoch[dropped], epoch[dropped]):
        raise RuntimeError(f"ZUNA full/hybrid masked values differ: {base}")
    return epoch.astype(np.float32)


def _read_model_outputs(model_output: Path, ne, ch_names, dropped, good, nt):
    reconstructed = np.empty((ne, len(ch_names), nt), dtype=np.float32)
    for epoch_index in range(ne):
        reconstructed[epoch_index] = _read_one_model_output(
            model_output, epoch_index, ch_names, dropped, good, nt
        )
    return reconstructed


def _epoch_output_status(model_output: Path, ne, ch_names, dropped, good, nt):
    """Return independently verified completed epochs and failure reasons."""
    completed = []
    invalid = {}
    for epoch_index in range(ne):
        try:
            _read_one_model_output(model_output, epoch_index, ch_names, dropped, good, nt)
        except Exception as exc:  # An interrupted FIF/NPZ is a missing unit on retry.
            invalid[str(epoch_index)] = f"{type(exc).__name__}: {exc}"
        else:
            completed.append(epoch_index)
    return completed, invalid


def _acquire_unit_lock(cache_root: Path, cache_key: str):
    lock_dir = cache_root / f".{cache_key}.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        owner_path = lock_dir / "owner.json"
        owner = owner_path.read_text(encoding="utf-8") if owner_path.is_file() else "unknown"
        raise RuntimeError(
            f"ZUNA reconstruction unit is locked: {lock_dir}; owner={owner}"
        ) from exc
    try:
        _write_manifest(lock_dir / "owner.json", {
            "cache_key_sha256": cache_key,
            "run_id": os.environ.get("GEEG_ZUNA_RUN_ID"),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm": {
                key: value for key, value in os.environ.items() if key.startswith("SLURM_")
            },
            "acquired_at": datetime.now(timezone.utc).astimezone().isoformat(),
        })
    except Exception:
        shutil.rmtree(lock_dir, ignore_errors=True)
        raise
    return lock_dir


def _release_unit_lock(lock_dir: Path | None):
    if lock_dir is None:
        return
    owner_path = lock_dir / "owner.json"
    if owner_path.is_file():
        owner_path.unlink()
    lock_dir.rmdir()


def _inventory(*directories: Path):
    rows = []
    for directory in directories:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            rows.append({
                "path": path.relative_to(directory.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
    return rows


def _load_cached(rec_path, manifest_path, cache_key, data, ch_names, dropped, good):
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing remediated reconstruction manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "geeg-zuna-reconstruction-cache-v3"
        or manifest.get("status") != "complete"
        or manifest.get("cache_key_sha256") != cache_key
    ):
        raise RuntimeError(f"Remediated cache status/key mismatch: {manifest_path.parent}")
    if _sha256_file(rec_path) != manifest["artifacts"]["reconstruction_npz_sha256"]:
        raise RuntimeError(f"Remediated reconstruction checksum mismatch: {rec_path}")
    with np.load(rec_path, allow_pickle=False) as saved:
        rec = saved["reconstruction"].astype(np.float32, copy=False)
        saved_channels = [str(name) for name in saved["ch_names"]]
        saved_dropped = saved["dropped"].astype(int).tolist()
        saved_key = str(saved["cache_key_sha256"].item())
    if (
        rec.shape != data.shape or saved_channels != list(ch_names)
        or saved_dropped != list(dropped) or saved_key != cache_key
    ):
        raise RuntimeError(f"Remediated reconstruction metadata mismatch: {rec_path}")
    if not np.isfinite(rec).all() or not np.array_equal(rec[:, good, :], data[:, good, :]):
        raise RuntimeError(f"Remediated reconstruction failed hard-inpainting integrity: {rec_path}")
    return rec, manifest


def _initial_manifest(cache_key, cache_label, identity, settings, model, code_sha,
                      ch_names, stage0_manifest, pos, discrete_pos, dropped):
    return {
        "schema": "geeg-zuna-reconstruction-cache-v3",
        "status": "initializing",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "cache_key_sha256": cache_key,
        "cache_label": cache_label,
        "identity": identity,
        "settings": settings,
        "model": model,
        "code_sha256": code_sha,
        "channels": list(ch_names),
        "stage0_cache_key": stage0_manifest["identity"]["cache_key_sha256"],
        "original_positions_m": np.asarray(pos, dtype=float).tolist(),
        "model_positions_m": np.clip(pos, POSITION_MIN, POSITION_MAX).astype(float).tolist(),
        "position_bounds_m": [POSITION_MIN.astype(float).tolist(), POSITION_MAX.astype(float).tolist()],
        "position_bins": int(POSITION_BINS),
        "discrete_positions": discrete_pos.tolist(),
        "coordinate_transform": "official_zuna_componentwise_discrete_bin_clamp",
        "coordinate_clamping": True,
        "dropped_indices": list(dropped),
        "dropped_channels": [ch_names[index] for index in dropped],
        "expected_epochs": int(len(stage0_manifest.get("epochs", []))),
        "scientific_guards": {
            "held_out_samples_serialized": False,
            "held_out_stats_used": False,
            "reference_uses_held_out": False,
            "artificial_epoch_concatenation": False,
            "second_filter_pass": False,
            "posthoc_self_calibration": False,
        },
    }


def _validate_resume_manifest(manifest, expected, cache_dir):
    for field in (
        "schema", "cache_key_sha256", "identity", "settings", "model",
        "code_sha256", "channels", "stage0_cache_key", "dropped_indices",
    ):
        if manifest.get(field) != expected.get(field):
            raise RuntimeError(
                f"Incomplete ZUNA cache identity mismatch for {field}: {cache_dir}"
            )


def _save_blind_metadata(path, cache_key, blind, blind_scale_uv, carriers, dropped):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        temporary,
        cache_key_sha256=np.asarray(cache_key),
        blind_input_sha256=np.asarray(_sha256_array(blind)),
        scale_uv=blind_scale_uv,
        carrier_templates=carriers,
        dropped=np.asarray(dropped, dtype=np.int64),
    )
    os.replace(temporary, path)


def zuna_reconstruct(stage0, dropped, gpu_device=0,
                     sample_steps=50, segment_sec=5.0, highpass_hz=None, seqlen=8000,
                     seed=333, debug=None, cache_label=None,
                     calibration_strategy="median_survivor_std_zero_mean_carrier"):
    """Reconstruct held-out channels without exposing their samples or scale to ZUNA."""
    data, ch_names, pos, discrete_pos, dropped, good, stage0_manifest = _validate_inputs(
        stage0, dropped
    )
    settings = _settings(
        sample_steps, segment_sec, highpass_hz, seqlen, seed, calibration_strategy
    )
    blind, blind_scale_uv, carriers = _blind_calibration_input(
        data, pos, dropped, good, calibration_strategy
    )
    model = _model_provenance()
    code_sha = _code_provenance()
    cache_root, cache_dir, cache_key, identity = _cache_location(
        blind, ch_names, pos, discrete_pos, dropped, cache_label, settings,
        model["identity"], code_sha,
        stage0_manifest["identity"]["cache_key_sha256"],
    )
    rec_path = cache_dir / "reconstruction.npz"
    model_input = cache_dir / "model_input"
    model_output = cache_dir / "model_output"
    blind_meta_path = cache_dir / "blind_input_metadata.npz"
    manifest_path = cache_dir / "manifest.json"
    cache_root.mkdir(parents=True, exist_ok=True)

    if rec_path.is_file():
        rec, manifest = _load_cached(
            rec_path, manifest_path, cache_key, data, ch_names, dropped, good)
        print(f"[zuna11 v3 cache] hit -> {cache_dir}", flush=True)
        if debug is not None:
            debug.update(cache_hit=True, cache_dir=str(cache_dir), manifest=manifest,
                         blind_input=blind, blind_scale_uv=blind_scale_uv)
        return rec

    lock_dir = subset_input = work_temp = None
    try:
        lock_dir = _acquire_unit_lock(cache_root, cache_key)
        # A second process may have completed the unit between the first cache
        # check and our lock acquisition.
        if rec_path.is_file():
            rec, manifest = _load_cached(
                rec_path, manifest_path, cache_key, data, ch_names, dropped, good)
            print(f"[zuna11 v3 cache] hit after lock -> {cache_dir}", flush=True)
            if debug is not None:
                debug.update(cache_hit=True, cache_dir=str(cache_dir), manifest=manifest,
                             blind_input=blind, blind_scale_uv=blind_scale_uv)
            return rec

        expected_manifest = _initial_manifest(
            cache_key, cache_label, identity, settings, model, code_sha, ch_names,
            stage0_manifest, pos, discrete_pos, dropped,
        )
        expected_manifest["expected_epochs"] = int(data.shape[0])
        if cache_dir.exists():
            if not manifest_path.is_file():
                raise RuntimeError(f"Incomplete ZUNA cache has no manifest: {cache_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_resume_manifest(manifest, expected_manifest, cache_dir)
        else:
            cache_dir.mkdir()
            manifest = expected_manifest
            _write_manifest(manifest_path, manifest)

        # Inputs are deterministic from the content-addressed blind tensor. Rewriting
        # all of them makes an interruption during serialization safely recoverable.
        model_input.mkdir(exist_ok=True)
        _write_epoch_fifs(model_input, blind, ch_names, pos, dropped)
        expected_input_names = {f"epoch_{index:04d}_raw.fif" for index in range(data.shape[0])}
        actual_input_names = {path.name for path in model_input.glob("epoch_*_raw.fif")}
        if actual_input_names != expected_input_names:
            raise RuntimeError(f"ZUNA model input epoch set mismatch: {cache_dir}")
        _save_blind_metadata(
            blind_meta_path, cache_key, blind, blind_scale_uv, carriers, dropped
        )
        model_output.mkdir(exist_ok=True)

        completed, invalid = _epoch_output_status(
            model_output, data.shape[0], ch_names, dropped, good, data.shape[2]
        )
        missing = [index for index in range(data.shape[0]) if index not in completed]
        manifest.update({
            "status": "model_pending" if missing else "model_output_saved",
            "expected_epochs": int(data.shape[0]),
            "verified_completed_epochs": completed,
            "missing_or_invalid_epochs": missing,
            "invalid_epoch_reasons": invalid,
            "last_checked_at": datetime.now(timezone.utc).astimezone().isoformat(),
        })
        _write_manifest(manifest_path, manifest)

        if missing:
            submitted_epochs = list(missing)
            subset_input = Path(tempfile.mkdtemp(prefix="z11v3_missing_in_"))
            work_temp = Path(tempfile.mkdtemp(prefix="z11v3_tmp_"))
            for epoch_index in missing:
                shutil.copy2(
                    model_input / f"epoch_{epoch_index:04d}_raw.fif",
                    subset_input / f"epoch_{epoch_index:04d}_raw.fif",
                )
            repair = ",".join(ch_names[index] for index in dropped)
            cmd = [
                sys.executable, str(HELPER),
                "--input_dir", str(subset_input), "--output_dir", str(model_output),
                "--tmp_dir", str(work_temp), "--repair", repair,
                "--montage", "standard_1005", "--highpass", "none",
                "--segment_sec", "5.0", "--sample_steps", str(sample_steps),
                "--seqlen", str(seqlen), "--gpu", str(gpu_device), "--seed", str(seed),
                "--model_revision", str(model["identity"]["revision"]),
                "--weight_sha256", str(model["identity"]["weight_sha256"]),
                "--config_sha256", str(model["identity"]["config_sha256"]),
            ]
            attempts = manifest.setdefault("attempts", [])
            attempt = {
                "attempt": len(attempts) + 1,
                "status": "running",
                "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "submitted_epochs": submitted_epochs,
                "run_id": os.environ.get("GEEG_ZUNA_RUN_ID"),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "slurm": {
                    key: value for key, value in os.environ.items() if key.startswith("SLURM_")
                },
            }
            attempts.append(attempt)
            _write_manifest(manifest_path, manifest)
            try:
                _run_subprocess(cmd, check=True)
                completed, invalid = _epoch_output_status(
                    model_output, data.shape[0], ch_names, dropped, good, data.shape[2]
                )
                missing = [index for index in range(data.shape[0]) if index not in completed]
                if missing:
                    raise RuntimeError(
                        f"ZUNA helper returned without valid outputs for epochs: {missing}"
                    )
            except Exception as exc:
                completed, invalid = _epoch_output_status(
                    model_output, data.shape[0], ch_names, dropped, good, data.shape[2]
                )
                missing = [index for index in range(data.shape[0]) if index not in completed]
                manifest.update({
                    "status": "model_failure",
                    "failed_at": datetime.now(timezone.utc).astimezone().isoformat(),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "verified_completed_epochs": completed,
                    "missing_or_invalid_epochs": missing,
                    "invalid_epoch_reasons": invalid,
                })
                attempt.update({
                    "status": "failure",
                    "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "verified_completed_epochs_after_attempt": completed,
                    "missing_or_invalid_epochs_after_attempt": missing,
                })
                _write_manifest(manifest_path, manifest)
                raise

            helper_provenance = model_output / "helper_provenance.json"
            attempt.update({
                "status": "success",
                "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "verified_completed_epochs_after_attempt": completed,
                "helper_provenance_sha256": (
                    _sha256_file(helper_provenance) if helper_provenance.is_file() else None
                ),
            })
            manifest.update({
                "status": "model_output_saved",
                "model_output_saved_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "verified_completed_epochs": completed,
                "missing_or_invalid_epochs": [],
                "invalid_epoch_reasons": {},
                "recovery": {
                    "epochs_submitted_this_attempt": submitted_epochs,
                    "preserved_completed_epochs_before_attempt": [
                        index for index in range(data.shape[0])
                        if index not in submitted_epochs
                    ],
                },
            })
            _write_manifest(manifest_path, manifest)
            print(
                f"[zuna11 v3 cache] verified {len(completed)}/{data.shape[0]} model epochs -> {cache_dir}",
                flush=True,
            )
        else:
            print(f"[zuna11 v3 cache] all model epochs already verified -> {model_output}", flush=True)

        model_reconstruction = _read_model_outputs(
            model_output, data.shape[0], ch_names, dropped, good, data.shape[2])
        rec = data.copy()
        rec[:, dropped, :] = model_reconstruction[:, dropped, :]
        rec[:, good, :] = data[:, good, :]
        if not np.isfinite(rec).all() or not np.array_equal(rec[:, good, :], data[:, good, :]):
            raise RuntimeError("Remediated reconstruction failed output integrity gates")

        temporary_rec = cache_dir / "reconstruction.tmp.npz"
        np.savez(
            temporary_rec,
            reconstruction=rec,
            normalized_dropped_reconstruction=(
                model_reconstruction[:, dropped, :] / blind_scale_uv[:, :, None]
            ).astype(np.float32),
            ch_names=np.asarray(ch_names),
            dropped=np.asarray(dropped, dtype=np.int64),
            cache_key_sha256=np.asarray(cache_key),
        )
        os.replace(temporary_rec, rec_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory = _inventory(model_input, model_output)
        manifest.update({
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "shape": list(rec.shape),
            "blind_scale_uv": {
                "minimum": float(np.min(blind_scale_uv)),
                "median": float(np.median(blind_scale_uv)),
                "maximum": float(np.max(blind_scale_uv)),
            },
            "artifacts": {
                "reconstruction_npz": "reconstruction.npz",
                "reconstruction_npz_bytes": rec_path.stat().st_size,
                "reconstruction_npz_sha256": _sha256_file(rec_path),
                "blind_input_metadata": "blind_input_metadata.npz",
                "model_input": "model_input",
                "model_output": "model_output",
                "model_io_inventory": inventory,
            },
        })
        _write_manifest(manifest_path, manifest)
        print(f"[zuna11 v3 cache] reconstruction saved -> {rec_path}", flush=True)
        if debug is not None:
            debug.update(cache_hit=False, cache_dir=str(cache_dir), manifest=manifest,
                         blind_input=blind, blind_scale_uv=blind_scale_uv,
                         model_reconstruction=model_reconstruction)
        return rec
    finally:
        for directory in (subset_input, work_temp):
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
        _release_unit_lock(lock_dir)
