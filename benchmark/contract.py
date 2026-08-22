"""Load and hash the independently readable scientific contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONTRACT_NAME = "scientific_contract_v1.json"


def _find_contract() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1] / "config" / CONTRACT_NAME,
        Path(__file__).resolve().parent.parent / "config" / CONTRACT_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate config/{CONTRACT_NAME}")


CONTRACT_PATH = _find_contract()
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
if CONTRACT.get("schema") != "geeg-zuna-scientific-contract-v1":
    raise RuntimeError(f"Unexpected scientific contract schema: {CONTRACT.get('schema')!r}")

CANONICAL_CONTRACT_JSON = json.dumps(
    CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=True
)
CONTRACT_SHA256 = hashlib.sha256(CANONICAL_CONTRACT_JSON.encode("utf-8")).hexdigest()
EXPERIMENT_ID = f"geeg-zuna-{CONTRACT_SHA256[:16]}"


def assert_production_ready() -> None:
    blockers = CONTRACT.get("blocking_decisions", [])
    if CONTRACT.get("contract_status") != "production_ready" or blockers:
        raise RuntimeError(
            "Scientific contract is not production-ready; blockers: " + ", ".join(blockers)
        )


def metric_contract(metric_key: str) -> dict:
    try:
        return CONTRACT["metrics"][metric_key]
    except KeyError as error:
        raise KeyError(f"Metric {metric_key!r} is absent from the scientific contract") from error

