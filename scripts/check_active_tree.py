"""Fail if active source can import or ship known legacy benchmark paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ACTIVE_PATHS = (
    "zuna",
    "main_pipeline.py",
    "load_data.py",
    "advanced_neuro_tests.py",
    "compare_outputs.py",
    "inspect_pt.py",
    "run_stage4_only.py",
    "test_2channel_dropout.py",
    "test_n_channel_dropout.py",
    "merge_zuna11_local_repair.ps1",
    "benchmark/zuna_method.py",
    "benchmark/_validate_zuna.py",
    "benchmark/_diag_zuna.py",
    "benchmark/_validate_zuna11.py",
    "benchmark/aggregate.py",
    "benchmark/biomarker_eval.py",
    "benchmark/mne_source_method.py",
    "benchmark/slurm_zuna_array.sh",
    "benchmark/audit_zuna11_pipeline.py",
    "benchmark/archive_invalid_broadband_v1.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    violations = [item for item in FORBIDDEN_ACTIVE_PATHS if (ROOT / item).exists()]
    local_zuna = importlib.util.find_spec("zuna")
    zuna_origin = None if local_zuna is None else local_zuna.origin
    if zuna_origin:
        resolved_origin = Path(zuna_origin).resolve()
        allowed_site_packages = (ROOT / ".zuna11_local_env" / "Lib" / "site-packages").resolve()
        if resolved_origin.is_relative_to(ROOT) and not resolved_origin.is_relative_to(allowed_site_packages):
            violations.append(f"project-local zuna import: {zuna_origin}")

    report = {"status": "pass" if not violations else "fail", "violations": violations,
              "zuna_origin": zuna_origin}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"active-tree check: {report['status']}")
        if zuna_origin:
            print(f"zuna origin: {zuna_origin}")
        for violation in violations:
            print(f"forbidden: {violation}")
    raise SystemExit(0 if not violations else 2)


if __name__ == "__main__":
    main()
