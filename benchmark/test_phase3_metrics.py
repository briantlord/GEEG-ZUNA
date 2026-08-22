"""Phase 3 metric, comparator, schema, and physiological-gate regressions."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BENCH = Path(__file__).resolve().parent
METRICS = BENCH / "metrics"
sys.path[:0] = [str(METRICS), str(BENCH)]

import common
import m_specparam_peaks
import reconstruction_qc
import run
import aggregate
from contract import CONTRACT_SHA256, EXPERIMENT_ID
from protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID
from schema_v3 import (
    COLUMNS, RESULT_SCHEMA, classify_method, make_truth_unit_id, reference_frame_id,
)


class Phase3MetricsTest(unittest.TestCase):
    def test_official_specparam_detects_synthetic_alpha(self):
        rng = np.random.default_rng(3)
        time = np.arange(1280) / 256.0
        data = rng.normal(0, 1, size=(8, 6, 1280))
        data += 4 * np.sin(2 * np.pi * 10 * time)[None, None, :]
        values = m_specparam_peaks.compute(data, m_specparam_peaks.POSTERIOR)
        evaluated, diagnostics = m_specparam_peaks.evaluate(
            data, m_specparam_peaks.POSTERIOR
        )
        self.assertEqual(values, evaluated)
        self.assertEqual(diagnostics["fit_status"], "success")
        self.assertTrue(np.isfinite(diagnostics["r_squared"]))
        self.assertTrue(np.isfinite(diagnostics["mean_absolute_error"]))
        self.assertEqual(set(values), set(m_specparam_peaks.SUBMETRICS))
        self.assertTrue(all(np.isfinite(list(values.values()))))
        self.assertAlmostEqual(values["alpha_cf"], 10.0, delta=0.3)
        self.assertEqual(m_specparam_peaks.SPECPARAM_VERSION, "2.0.0rc7")
        self.assertNotIn("aperiodic_and_peak", dir(common))
        self.assertIn("specparam==2.0.0rc7", m_specparam_peaks.METRIC.implementation)

    def test_reconstruction_integrity_and_physiological_scale_gate(self):
        rng = np.random.default_rng(4)
        truth = rng.normal(0, 5, size=(2, 6, 1280)).astype(np.float32)
        passing = truth.copy()
        passing[:, :2] *= 0.9
        diagnostics = reconstruction_qc.evaluate_reconstruction(
            truth, passing, [0, 1], ch_names=["F3", "F4", "C3", "C4", "P3", "P4"]
        )
        self.assertEqual(diagnostics["status"], "pass")
        self.assertAlmostEqual(diagnostics["power_ratio_1_45"], 0.81, places=5)
        self.assertEqual(len(diagnostics["per_epoch_channel"]), 4)
        self.assertEqual(diagnostics["per_epoch_channel"][0]["channel"], "F3")
        self.assertIn("log10_psd_rmse_1_45", diagnostics["per_epoch_channel"][0])

        amplified = truth.copy()
        amplified[:, :2] *= 4.0
        diagnostics = reconstruction_qc.evaluate_reconstruction(truth, amplified, [0, 1])
        self.assertEqual(diagnostics["status"], "fail")
        self.assertAlmostEqual(diagnostics["power_ratio_1_45"], 16.0, places=5)

        changed_observed = passing.copy()
        changed_observed[:, 3] += 1.0
        with self.assertRaisesRegex(ValueError, "observed channel changed"):
            reconstruction_qc.evaluate_reconstruction(truth, changed_observed, [0, 1])

    def test_complete_drop_sets_and_metric_outputs_are_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing declared drop channels"):
            run.drop_indices(["F3", "F4"], ["F3", "F4", "F7"])
        metric = m_specparam_peaks.METRIC
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run.metric_values(metric, np.zeros((2, 6, 1280)), metric.drop_channels)

    def test_oracle_label_is_unambiguous(self):
        self.assertEqual(classify_method("linear_oracle"), "oracle")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            classify_method("linear")

    @staticmethod
    def _row(kind="truth", method="-"):
        drop_set = "CPZ+CZ+FCZ+FZ"
        stage0_key = "a" * 64
        truth_id = make_truth_unit_id(
            "G001Day1Rest1.cnt", stage0_key, drop_set, "theta_beta", "tbr_cz"
        )
        truth = {
            "result_schema": RESULT_SCHEMA,
            "run_id": "c" * 64,
            "recording": "G001Day1Rest1.cnt",
            "subject": "G001",
            "protocol_id": PROTOCOL_ID,
            "experiment_id": EXPERIMENT_ID,
            "scientific_contract_sha256": CONTRACT_SHA256,
            "preprocessing_sha256": PREPROCESSING_SHA256,
            "stage0_cache_key": stage0_key,
            "kind": "truth",
            "unit_status": "success",
            "drop_set": drop_set,
            "reference_frame_id": reference_frame_id(drop_set),
            "truth_unit_id": truth_id,
            "method": "-",
            "comparator_class": "truth",
            "reconstruction_key": "-",
            "gate_status": "not_applicable",
            "power_ratio_1_45": "-",
            "rms_ratio": "-",
            "max_abs_uv": "-",
            "metric": "theta_beta",
            "metric_implementation": "project-native",
            "metric_diagnostics_json": "{}",
            "metric_source_sha256": "1" * 64,
            "metric_config_sha256": "2" * 64,
            "metrics_common_sha256": "3" * 64,
            "runner_sha256": "4" * 64,
            "qc_sha256": "5" * 64,
            "schema_sha256": "6" * 64,
            "aggregator_sha256": "7" * 64,
            "submetric": "tbr_cz",
            "truth": "1.0",
            "value": "1.0",
            "abs_err": "0",
        }
        if kind == "recon":
            truth.update({
                "kind": "recon", "method": method,
                "comparator_class": classify_method(method),
                "reconstruction_key": "b" * 64, "gate_status": "pass",
                "power_ratio_1_45": "0.9", "rms_ratio": "0.95", "max_abs_uv": "50",
                "value": "1.2", "abs_err": "0.2",
            })
        return truth

    def test_v4_aggregation_rejects_legacy_and_accepts_matched_truth_rows(self):
        with tempfile.TemporaryDirectory(prefix="phase3_schema_") as temporary:
            valid = Path(temporary) / "valid.csv"
            with valid.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerow(self._row())
                writer.writerow(self._row("recon", "spline"))
            rows = aggregate.load_validated_rows(valid)
            self.assertEqual(len(rows), 2)

            legacy = Path(temporary) / "legacy.csv"
            legacy.write_text("recording,kind,method,value\nx,truth,-,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible result schema"):
                aggregate.load_validated_rows(legacy)


if __name__ == "__main__":
    unittest.main()
