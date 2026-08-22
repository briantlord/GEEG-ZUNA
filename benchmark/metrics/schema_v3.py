"""Result-row identity and comparator semantics for the remediated benchmark."""

from __future__ import annotations

import hashlib
import json


RESULT_SCHEMA = "geeg-zuna-metrics-v5"

METHOD_CLASSES = {
    "zuna": "model",
    "spline": "baseline",
    "nearest": "baseline",
    "mean": "baseline",
    "mne": "baseline",
    "zero": "negative_control",
    "linear_oracle": "oracle",
}

COLUMNS = [
    "result_schema",
    "run_id",
    "recording",
    "subject",
    "protocol_id",
    "experiment_id",
    "scientific_contract_sha256",
    "preprocessing_sha256",
    "stage0_cache_key",
    "kind",
    "unit_status",
    "drop_set",
    "reference_frame_id",
    "truth_unit_id",
    "method",
    "comparator_class",
    "reconstruction_key",
    "gate_status",
    "power_ratio_1_45",
    "rms_ratio",
    "max_abs_uv",
    "metric",
    "metric_implementation",
    "metric_diagnostics_json",
    "metric_source_sha256",
    "metric_config_sha256",
    "metrics_common_sha256",
    "runner_sha256",
    "qc_sha256",
    "schema_sha256",
    "aggregator_sha256",
    "submetric",
    "truth",
    "value",
    "abs_err",
]

KEY_COLUMNS = ("recording", "kind", "drop_set", "method", "metric", "submetric")


def reference_frame_id(drop_set: str) -> str:
    if not drop_set or drop_set == "-":
        raise ValueError("drop-set-specific reference frame requires a concrete drop set")
    return f"survivor-average-excluding-M1-M2:{drop_set}"


def make_truth_unit_id(recording: str, stage0_cache_key: str, drop_set: str,
                       metric: str, submetric: str) -> str:
    identity = {
        "recording": recording,
        "stage0_cache_key": stage0_cache_key,
        "drop_set": drop_set,
        "reference_frame_id": reference_frame_id(drop_set),
        "metric": metric,
        "submetric": submetric,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classify_method(method: str) -> str:
    """Return the frozen comparator class, rejecting ambiguous or unknown labels."""
    if method == "linear":
        raise ValueError(
            "'linear' used held-out target samples during fitting and is forbidden; "
            "use the explicit 'linear_oracle' label"
        )
    try:
        return METHOD_CLASSES[method]
    except KeyError as error:
        raise ValueError(f"unknown reconstruction method: {method!r}") from error
