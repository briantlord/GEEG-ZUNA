"""Regression tests for the remediated Stage-0 foundation (no GPU required)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot
import protocol_v2
import stage0_cache


class CorrectedProtocolTest(unittest.TestCase):
    def test_frozen_preprocessing_contract(self):
        spec = protocol_v2.PREPROCESSING_SPEC
        self.assertEqual(spec["protocol_id"], "geeg-zuna-minimal-stage0-v1")
        self.assertEqual(spec["cnt_data_format"], "int32")
        self.assertEqual(spec["bandpass_hz"], [0.5, 45.0])
        self.assertEqual(spec["notch_hz"], [])
        self.assertEqual(spec["target_sfreq_hz"], 256)
        self.assertEqual(
            spec["epoch_count_policy"],
            "use_available_source_valid_epochs_up_to_requested_maximum",
        )
        self.assertEqual(spec["component_removal"], "none")
        self.assertEqual(spec["ica_policy"], "forbidden_in_primary_stage0")
        self.assertEqual(spec["epoch_amplitude_policy"], "record_only_no_rejection")
        self.assertEqual(
            spec["channel_amplitude_policy"],
            "record_per_channel_counts_and_fractions_without_classification_or_exclusion",
        )
        self.assertEqual(
            spec["continuous_channel_statistics"],
            "record_standard_deviation_railing_fraction_amplitude_and_sample_jumps_without_quality_exclusion",
        )
        self.assertFalse(hasattr(pilot, "remove_artifacts"))
        self.assertEqual(spec["reference_stage0"], "none")
        self.assertEqual(spec["reference_excluded_channels"], ["M1", "M2"])
        self.assertEqual(len(protocol_v2.PREPROCESSING_SHA256), 64)
        self.assertEqual(
            protocol_v2.PREPROCESSING_SHA256,
            protocol_v2.preprocessing_hash(),
        )

    def test_surviving_reference_excludes_dropped_channel(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(2, 4, 16)).astype(np.float32)
        changed = data.copy()
        changed[:, 0, :] += 1_000_000.0
        referenced = pilot.surviving_average_reference(data, [0])
        changed_referenced = pilot.surviving_average_reference(changed, [0])
        np.testing.assert_array_equal(referenced[:, 1:, :], changed_referenced[:, 1:, :])
        np.testing.assert_allclose(referenced[:, 1:, :].mean(axis=1), 0.0, atol=1e-6)

    def test_surviving_reference_excludes_mastoids(self):
        data = np.zeros((1, 4, 8), dtype=np.float32)
        data[:, 0] = 2.0
        data[:, 1] = 4.0
        data[:, 2] = 1_000_000.0
        data[:, 3] = -1_000_000.0
        names = ["F3", "F4", "M1", "M2"]
        referenced = pilot.surviving_average_reference(data, [], names)
        np.testing.assert_allclose(referenced[:, :2, :].mean(axis=1), 0.0)
        np.testing.assert_allclose(referenced[:, 0, :], -1.0)

    def test_stage0_cache_is_content_addressed_and_verified(self):
        with tempfile.TemporaryDirectory(prefix="phase1_cache_test_") as temporary:
            temporary_path = Path(temporary)
            raw = temporary_path / "same-name.cnt"
            raw.write_bytes(b"raw-version-one")
            cache_root = temporary_path / "cache"
            fake = {
                "data": np.ones((2, 3, 8), dtype=np.float32),
                "ch_names": ["F3", "F4", "Cz"],
                "pos": np.ones((3, 3), dtype=np.float32),
                "meta": {
                    "protocol_id": protocol_v2.PROTOCOL_ID,
                    "preprocessing_sha256": protocol_v2.PREPROCESSING_SHA256,
                },
            }
            with mock.patch.object(stage0_cache.pilot, "preprocess", return_value=fake) as preprocess:
                first = stage0_cache.load_or_create(raw, cache_root=cache_root, n_epochs=2)
                second = stage0_cache.load_or_create(raw, cache_root=cache_root, n_epochs=2)
                self.assertEqual(preprocess.call_count, 1)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[3]["identity"], second[3]["identity"])

            entries_v1 = [path for path in cache_root.iterdir() if not path.name.startswith(".")]
            self.assertEqual(len(entries_v1), 1)
            manifest = json.loads((entries_v1[0] / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "geeg-zuna-stage0-cache-v4")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["identity"]["raw_sha256"], stage0_cache.sha256_file(raw))
            self.assertNotIn("emg_cleaning", manifest["identity"])

            raw.write_bytes(b"raw-version-two")
            with mock.patch.object(stage0_cache.pilot, "preprocess", return_value=fake) as preprocess:
                third = stage0_cache.load_or_create(raw, cache_root=cache_root, n_epochs=2)
                self.assertEqual(preprocess.call_count, 1)
            self.assertNotEqual(
                first[3]["identity"]["cache_key_sha256"],
                third[3]["identity"]["cache_key_sha256"],
            )
            entries_v2 = [path for path in cache_root.iterdir() if not path.name.startswith(".")]
            self.assertEqual(len(entries_v2), 2)

    def test_amplitude_qc_is_record_only(self):
        candidate = np.ones((3, 2, 8), dtype=np.float32)
        candidate[1, 0, 0] = -1000.0
        candidate[1, 0, 1] = 1000.0
        selected, _peak_to_peak, passed = pilot._select_primary_epochs(candidate, n_epochs=3)
        np.testing.assert_array_equal(selected, candidate)
        self.assertFalse(bool(passed[0]))  # flat, but retained
        self.assertFalse(bool(passed[1]))  # high amplitude, but retained
        self.assertFalse(bool(passed[2]))  # flat, but retained

    def test_shorter_recording_returns_available_epochs(self):
        candidate = np.ones((2, 3, 8), dtype=np.float32)
        selected, _peak_to_peak, _passed = pilot._select_primary_epochs(
            candidate, n_epochs=64)
        np.testing.assert_array_equal(selected, candidate)

    def test_continuous_excursion_is_warning_not_recording_failure(self):
        class FakeRaw:
            ch_names = ["F3"]

            def get_data(self, picks):
                values = np.linspace(-1e-6, 1e-6, 1000, dtype=float)
                values[500] = 2e-3
                return values[None, :]

        rows, failures = pilot._session_channel_qc(FakeRaw())
        self.assertEqual(failures, [])
        self.assertTrue(rows[0]["passed"])
        self.assertIn("large_continuous_excursion", rows[0]["warnings"])
        self.assertIn("large_continuous_sample_jump", rows[0]["warnings"])

    def test_continuous_flatness_and_railing_are_descriptive(self):
        class FakeRaw:
            ch_names = ["F3"]

            def get_data(self, picks):
                return np.zeros((1, 1000), dtype=float)

        rows, failures = pilot._session_channel_qc(FakeRaw())
        self.assertEqual(failures, [])
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[0]["std_uv"], 0.0)
        self.assertEqual(rows[0]["rail_fraction"], 1.0)

    def test_channel_amplitude_burden_is_descriptive_only(self):
        data = np.zeros((10, 2, 8), dtype=np.float32)
        data[:, 0, 0] = -200.0
        data[:, 0, 1] = 200.0
        data[:7, 1, 0] = -200.0
        data[:7, 1, 1] = 200.0
        rows = pilot._channel_amplitude_qc(data, ["F3", "F4"])
        self.assertEqual([row["channel"] for row in rows], ["F3", "F4"])
        self.assertEqual(rows[0]["high_amplitude_epoch_count"], 10)
        self.assertEqual(rows[1]["high_amplitude_epoch_count"], 7)
        self.assertTrue(all(
            row["role"] == "descriptive_only_no_classification_or_exclusion"
            for row in rows
        ))


if __name__ == "__main__":
    unittest.main()
