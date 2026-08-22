"""Metric plug-in contract for the modular biomarker-preservation harness.

A metric is a SELF-REGISTERING PLUG-IN. To add one, drop a file `m_<key>.py` in this directory
that builds a `Metric(...)` and calls `register(...)` at import time. The runner (`run.py`) and
aggregator (`aggregate.py`) discover it automatically — no shared file is edited. That is the whole
point: a new metric is a new module, not a new script.

Contract for `compute`:
    compute(data_uV, ch_names) -> {submetric_name: float}
      data_uV  : np.ndarray (n_epochs, n_channels, n_times), microvolts, in the benchmark's
                 surviving-channel average-reference frame (the runner passes this in).
      ch_names : list[str] channel labels aligned to axis 1 (standard_1005 names).
    Must return a dict whose keys are exactly `submetrics` and whose values are finite scalars.
    Phase 3 fails closed on missing or non-finite outputs instead of silently producing partial rows.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

REGISTRY: "Dict[str, Metric]" = {}


@dataclass
class Metric:
    key: str                  # unique short id, e.g. 'faa'
    name: str                 # human-readable name
    drop_channels: List[str]  # channels dropped (case-insensitive) to test preservation
    submetrics: List[str]     # names of the scalar outputs compute() returns
    compute: Callable         # compute(data_uV, ch_names) -> {submetric: float}
    reference: str = ""       # citation / provenance
    notes: str = ""           # one-line description
    implementation: str = "project-native"  # package/algorithm provenance written to every row
    evaluate: Callable | None = None  # optional single-pass (values, diagnostics) evaluator


def register(metric: "Metric") -> "Metric":
    if not metric.key or not metric.key.replace('_', '').isalnum():
        raise ValueError(f"invalid metric key: {metric.key!r}")
    if metric.key in REGISTRY:
        raise ValueError(f"duplicate metric key: {metric.key!r}")
    if not metric.drop_channels or len(set(c.upper() for c in metric.drop_channels)) != len(metric.drop_channels):
        raise ValueError(f"metric {metric.key!r} has an empty or duplicate drop set")
    if not metric.submetrics or len(set(metric.submetrics)) != len(metric.submetrics):
        raise ValueError(f"metric {metric.key!r} has empty or duplicate submetrics")
    if not callable(metric.compute):
        raise ValueError(f"metric {metric.key!r} compute is not callable")
    REGISTRY[metric.key] = metric
    return metric
