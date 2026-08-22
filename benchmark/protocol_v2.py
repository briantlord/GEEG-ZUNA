"""Frozen constants for the corrected ZUNA 1.1 benchmark (protocol v2).

Keep this module dependency-free: both the local runner and the HPC share import
it, and its canonical JSON digest is part of every Stage-0 cache identity.
"""

from __future__ import annotations

import hashlib
import json

try:
    from .contract import CONTRACT_SHA256, EXPERIMENT_ID
except ImportError:
    from contract import CONTRACT_SHA256, EXPERIMENT_ID


PROTOCOL_ID = "geeg-zuna-remediated-v1"

PREPROCESSING_SPEC = {
    "protocol_id": PROTOCOL_ID,
    "cnt_data_format": "int32",
    "montage": "standard_1005",
    "expected_eeg_channels": 62,
    "aux_channels": ["HEOG", "VEOG", "EKG"],
    "non_montage_channels": ["CB1", "CB2"],
    "bandpass_hz": [0.5, 45.0],
    "filter_design": "firwin",
    "filter_phase": "zero",
    "notch_hz": [],
    "target_sfreq_hz": 256,
    "edge_crop_seconds": 10.0,
    "emg_cleaning": "MNE ICA fastica + find_bads_muscle",
    "emg_required": True,
    "muscle_threshold": 0.5,
    "muscle_band_hz": [7.0, 45.0],
    "ocular_cleaning": "MNE ICA + find_bads_eog using HEOG and VEOG",
    "ocular_required": True,
    "ocular_threshold": 3.0,
    "ocular_measure": "zscore",
    "ica_components": 20,
    "ica_random_state": 0,
    "epoch_seconds": 5.0,
    "target_epochs": 64,
    "minimum_clean_epochs": 48,
    "event_codes": [str(value) for value in range(1, 9)],
    "epoch_peak_to_peak_max_uv": 300.0,
    "epoch_peak_to_peak_flat_uv": 0.1,
    "session_flat_std_uv": 0.1,
    "session_rail_fraction_max": 0.01,
    "analysis_abs_max_uv": 1000.0,
    "analysis_max_sample_jump_uv": 500.0,
    "raw_tail_abs_warn_uv": 100000.0,
    "reference_stage0": "none",
    "reference_evaluation": "average over surviving non-mastoid channels after dropout",
    "reference_excluded_channels": ["M1", "M2"],
    "scientific_contract_sha256": CONTRACT_SHA256,
    "experiment_id": EXPERIMENT_ID,
    "score_band_hz": [1.0, 45.0],
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def preprocessing_hash() -> str:
    return hashlib.sha256(canonical_json(PREPROCESSING_SPEC).encode("utf-8")).hexdigest()


PREPROCESSING_SHA256 = preprocessing_hash()
