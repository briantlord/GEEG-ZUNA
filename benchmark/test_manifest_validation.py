"""Expected-unit manifest and fail-closed bundle validation tests."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest
import validate_bundle
from metrics.run import StatusLedger
from metrics.schema_v3 import COLUMNS


class ManifestValidationTest(unittest.TestCase):
    def test_current_design_has_exact_declared_counts(self):
        with tempfile.TemporaryDirectory(prefix="run_manifest_counts_") as temporary:
            root = Path(temporary)
            files = []
            for subject in range(1, 22):
                for rest in (1, 2):
                    path = root / f"G{subject:03d}Day1Rest{rest}.cnt"
                    path.write_bytes(f"{subject}:{rest}".encode())
                    files.append(path)
            manifest = run_manifest.create_manifest(
                files, ["spline", "zuna"],
                model={"revision": "test", "weight_sha256": "a" * 64},
            )
            self.assertEqual(manifest["expected_counts"]["recordings"], 42)
            self.assertEqual(manifest["expected_counts"]["truth_units"], 588)
            self.assertEqual(manifest["expected_counts"]["recon_result_units"], 1176)
            self.assertEqual(manifest["expected_counts"]["result_units"], 1764)
            self.assertEqual(manifest["expected_counts"]["reconstruction_units"], 420)

    def test_manifest_tamper_and_incomplete_bundle_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="run_manifest_validation_") as temporary:
            root = Path(temporary)
            recording = root / "G001Day1Rest1.cnt"
            recording.write_bytes(b"recording")
            manifest = run_manifest.create_manifest([recording], ["spline"], model=None)
            manifest_path = root / "run.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(run_manifest.load_verified(manifest_path)["run_id"], manifest["run_id"])

            tampered = dict(manifest)
            tampered["methods"] = ["spline", "zuna"]
            tampered_path = root / "tampered.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity hash mismatch"):
                run_manifest.load_verified(tampered_path)

            result_path = root / "results.csv"
            with result_path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=COLUMNS).writeheader()
            qc_path = root / "qc.jsonl"
            qc_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completeness failure"):
                validate_bundle.validate(manifest_path, result_path, qc_path)

    def test_explicit_terminal_failures_satisfy_completeness_but_not_all_success(self):
        with tempfile.TemporaryDirectory(prefix="run_manifest_failures_") as temporary:
            root = Path(temporary)
            recording = root / "G001Day1Rest1.cnt"
            recording.write_bytes(b"recording")
            manifest = run_manifest.create_manifest([recording], ["spline"], model=None)
            manifest_path = root / "run.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result_path = root / "results.csv"
            with result_path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=COLUMNS).writeheader()
            qc_path = root / "qc.jsonl"
            qc_path.write_text("", encoding="utf-8")
            status_path = root / "status.jsonl"
            ledger = StatusLedger(
                status_path, manifest["run_id"], manifest["terminal_states"]
            )
            error = FileNotFoundError("injected absent recording")
            for unit in manifest["expected_result_units"]:
                key = tuple(unit[name] for name in (
                    "recording", "kind", "drop_set", "method", "metric", "submetric"
                ))
                ledger.set_result(key, "missing_input", error, "stage0")
            for unit in manifest["expected_reconstruction_units"]:
                ledger.set_reconstruction(
                    unit["recording"], unit["method"], unit["drop_set"],
                    "missing_input", error, "stage0",
                )

            report = validate_bundle.validate(
                manifest_path, result_path, qc_path, status_path=status_path
            )
            self.assertEqual(report["result_units"], manifest["expected_counts"]["result_units"])
            self.assertEqual(
                report["reconstruction_units"],
                manifest["expected_counts"]["reconstruction_units"],
            )
            self.assertGreater(report["failures"]["missing_input"], 0)
            with self.assertRaisesRegex(ValueError, "terminal failures"):
                validate_bundle.validate(
                    manifest_path, result_path, qc_path, status_path=status_path,
                    require_all_success=True,
                )

    def test_status_ledger_retry_replaces_failure_and_success_clears_it(self):
        with tempfile.TemporaryDirectory(prefix="status_ledger_retry_") as temporary:
            path = Path(temporary) / "status.jsonl"
            terminal = ["success", "model_failure", "qc_failure"]
            ledger = StatusLedger(path, "run-1", terminal)
            key = ("r.cnt", "recon", "F3+F4", "zuna", "faa", "alpha")
            ledger.set_result(key, "model_failure", RuntimeError("first"), "model")
            ledger.set_result(key, "qc_failure", RuntimeError("second"), "qc")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "qc_failure")
            ledger.clear_result(key)
            self.assertEqual(path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
