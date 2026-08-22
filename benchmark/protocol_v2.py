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


PROTOCOL_ID = "geeg-zuna-minimal-stage0-v1"

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
    "component_removal": "none",
    "ica_policy": "forbidden_in_primary_stage0",
    "auxiliary_policy": "retain through filtering/resampling/cropping, then exclude from EEG tensor",
    "epoch_seconds": 5.0,
    "target_epochs": 64,
    "epoch_count_policy": "use_available_source_valid_epochs_up_to_requested_maximum",
    "event_codes": [str(value) for value in range(1, 9)],
    "epoch_peak_to_peak_max_uv": 300.0,
    "epoch_peak_to_peak_flat_uv": 0.1,
    "epoch_amplitude_policy": "record_only_no_rejection",
    "channel_amplitude_policy": "record_per_channel_counts_and_fractions_without_classification_or_exclusion",
    "bad_annotation_policy": "exclude_only_source_intervals_explicitly_marked_bad",
    "continuous_channel_statistics": "record_standard_deviation_railing_fraction_amplitude_and_sample_jumps_without_quality_exclusion",
    "analysis_abs_warn_uv": 1000.0,
    "analysis_max_sample_jump_warn_uv": 500.0,
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
