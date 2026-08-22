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
import ica_review


class CorrectedProtocolTest(unittest.TestCase):
    def test_frozen_preprocessing_contract(self):
        spec = protocol_v2.PREPROCESSING_SPEC
        self.assertEqual(spec["protocol_id"], "geeg-zuna-remediated-v1")
        self.assertEqual(spec["cnt_data_format"], "int32")
        self.assertEqual(spec["bandpass_hz"], [0.5, 45.0])
        self.assertEqual(spec["notch_hz"], [])
        self.assertEqual(spec["target_sfreq_hz"], 256)
        self.assertEqual(spec["minimum_clean_epochs"], 48)
        self.assertTrue(spec["emg_required"])
        self.assertTrue(spec["ocular_required"])
        self.assertEqual(spec["muscle_threshold"], 0.5)
        self.assertEqual(spec["muscle_band_hz"], [7.0, 45.0])
        self.assertEqual(spec["ocular_threshold"], 3.0)
        self.assertEqual(spec["ocular_measure"], "zscore")
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
                first = stage0_cache.load_or_create(
                    raw, cache_root=cache_root, n_epochs=2, minimum_clean_epochs=1)
                second = stage0_cache.load_or_create(
                    raw, cache_root=cache_root, n_epochs=2, minimum_clean_epochs=1)
                self.assertEqual(preprocess.call_count, 1)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[3]["identity"], second[3]["identity"])

            entries_v1 = [path for path in cache_root.iterdir() if not path.name.startswith(".")]
            self.assertEqual(len(entries_v1), 1)
            manifest = json.loads((entries_v1[0] / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["identity"]["raw_sha256"], stage0_cache.sha256_file(raw))

            raw.write_bytes(b"raw-version-two")
            with mock.patch.object(stage0_cache.pilot, "preprocess", return_value=fake) as preprocess:
                third = stage0_cache.load_or_create(
                    raw, cache_root=cache_root, n_epochs=2, minimum_clean_epochs=1)
                self.assertEqual(preprocess.call_count, 1)
            self.assertNotEqual(
                first[3]["identity"]["cache_key_sha256"],
                third[3]["identity"]["cache_key_sha256"],
            )
            entries_v2 = [path for path in cache_root.iterdir() if not path.name.startswith(".")]
            self.assertEqual(len(entries_v2), 2)

    def test_production_cache_rejects_disabled_emg(self):
        with self.assertRaisesRegex(ValueError, "requires ocular/muscle ICA"):
            stage0_cache.load_or_create("does-not-matter.cnt", emg=False)

    def test_ica_review_requires_complete_component_evidence(self):
        artifact = {
            "n_components": 2,
            "ica_channel_names": ["F3", "F4", "Cz"],
            "component_topographies": [[1.0, 0.2, 0.1], [0.1, -2.0, 0.3]],
            "muscle_scores": [0.2, 0.8],
            "pca_explained_variance": [4.0, 2.0],
            "ocular_scores": {"VEOG": [-0.7, 0.1]},
            "excluded_components": [0, 1],
            "ocular_components": [0],
            "muscle_components": [1],
        }
        rows = ica_review.component_rows({
            "preprocessing_meta": {"artifact_components": artifact}
        })
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["strongest_topography_channels"], "F3+F4+Cz")
        self.assertEqual(rows[1]["muscle_score"], 0.8)
        broken = json.loads(json.dumps(artifact))
        broken["component_topographies"] = []
        with self.assertRaisesRegex(ValueError, "topography dimensions"):
            ica_review.component_rows({
                "preprocessing_meta": {"artifact_components": broken}
            })


if __name__ == "__main__":
    unittest.main()
