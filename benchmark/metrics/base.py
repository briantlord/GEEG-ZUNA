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
    Must return a dict whose keys are exactly `submetrics` (missing keys are simply skipped for
    that recording). Values are finite scalars; return float('nan') if not computable.
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


def register(metric: "Metric") -> "Metric":
    if metric.key in REGISTRY:
        raise ValueError(f"duplicate metric key: {metric.key!r}")
    REGISTRY[metric.key] = metric
    return metric
