"""Phase 2 regression tests for the ZUNA 1.1 adapter (no GPU/model required)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import mne
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zuna_method_v11
import stage0_cache
from protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID


class Zuna11CorrectedV2Test(unittest.TestCase):
    def setUp(self):
        self.cache_root = tempfile.mkdtemp(prefix="test_zuna11_v2_cache_")
        self.old_cache = os.environ.get("ZUNA11_RECON_CACHE_DIR_V3")
        os.environ["ZUNA11_RECON_CACHE_DIR_V3"] = self.cache_root
        self.fake_model = {
            "identity": {
                "repository": "Zyphra/ZUNA1.1",
                "revision": "test-revision",
                "weight_sha256": "a" * 64,
                "weight_bytes": 123,
                "config_sha256": "b" * 64,
            },
            "locations": {"cache": "test", "snapshot": "test", "weight": "test", "config": "test"},
        }
        self.fake_code = {
            "adapter": "1" * 64,
            "helper": "2" * 64,
            "zuna_pipeline": "3" * 64,
            "zuna_eeg_data": "4" * 64,
            "zuna_fif_config": "5" * 64,
        }
        self.model_calls = []
        self.fail_after_first_epoch_once = False

    def tearDown(self):
        if self.old_cache is None:
            os.environ.pop("ZUNA11_RECON_CACHE_DIR_V3", None)
        else:
            os.environ["ZUNA11_RECON_CACHE_DIR_V3"] = self.old_cache
        shutil.rmtree(self.cache_root, ignore_errors=True)

    @staticmethod
    def _positions(ch_names):
        positions = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        return np.asarray([positions[name] for name in ch_names], dtype=np.float32)

    @staticmethod
    def _data():
        rng = np.random.default_rng(7)
        data = rng.normal(0, 5, size=(2, 6, 1280)).astype(np.float32)
        # Put data in the surviving-channel-average-reference frame for dropped [0, 1].
        data -= data[:, 2:, :].mean(axis=1, keepdims=True)
        return data

    def _fake_model_run(self, cmd, check):
        self.assertTrue(check)
        self.assertEqual(cmd[cmd.index("--highpass") + 1], "none")
        self.assertEqual(cmd[cmd.index("--segment_sec") + 1], "5.0")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "333")
        input_dir = Path(cmd[cmd.index("--input_dir") + 1])
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        repair = cmd[cmd.index("--repair") + 1].split(",")
        inputs = sorted(input_dir.glob("epoch_*_raw.fif"))
        self.assertGreater(len(inputs), 0)
        full = output_dir / "full_reconstruction"
        hybrid = output_dir / "hybrid"
        full.mkdir(parents=True, exist_ok=True)
        hybrid.mkdir(parents=True, exist_ok=True)
        call_inputs = []
        for input_path in inputs:
            raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")
            self.assertEqual(raw.n_times, 1280)
            self.assertEqual(set(raw.info["bads"]), set(repair))
            call_inputs.append(raw.get_data().copy())
            output = raw.get_data().copy()
            index = {name: i for i, name in enumerate(raw.ch_names)}
            dropped = [index[name] for name in repair]
            output[dropped] = output[dropped] * 2.0 + 3e-6
            out_raw = mne.io.RawArray(output, raw.info.copy(), verbose="ERROR")
            base = input_path.stem.replace("_raw", "")
            out_raw.save(full / f"{base}_raw.fif", overwrite=True, verbose="ERROR")
            out_raw.save(hybrid / f"{base}_raw.fif", overwrite=True, verbose="ERROR")
            mask = np.zeros((len(raw.ch_names), raw.n_times), dtype=bool)
            mask[dropped] = True
            np.savez(
                hybrid / f"{base}_mask.npz",
                mask=mask,
                ch_names=np.asarray(raw.ch_names),
                sfreq=np.float32(raw.info["sfreq"]),
            )
            if self.fail_after_first_epoch_once:
                self.fail_after_first_epoch_once = False
                self.model_calls.append([path.name for path in inputs])
                raise RuntimeError("injected model interruption after one epoch")
        self.model_calls.append([path.name for path in inputs])

    def _run(self, data, debug=None):
        ch_names = ["F3", "F4", "C3", "C4", "P3", "P4"]
        positions = self._positions(ch_names)
        stage0_key = "9" * 64
        manifest = {
            "schema": "geeg-zuna-stage0-cache-v4",
            "identity": {
                "protocol_id": PROTOCOL_ID,
                "preprocessing_sha256": PREPROCESSING_SHA256,
                "cache_key_sha256": stage0_key,
            },
            "output": {
                "data_sha256": stage0_cache.sha256_array(data),
                "positions_sha256": stage0_cache.sha256_array(positions),
                "channels_sha256": __import__("hashlib").sha256(
                    __import__("json").dumps(
                        ch_names, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                    ).encode("utf-8")
                ).hexdigest(),
            },
        }
        stage0 = stage0_cache.VerifiedStage0(
            data=data, ch_names=tuple(ch_names), pos=positions, manifest=manifest
        )
        with (
            mock.patch.object(zuna_method_v11, "_model_provenance", return_value=self.fake_model),
            mock.patch.object(zuna_method_v11, "_code_provenance", return_value=self.fake_code),
            mock.patch.object(zuna_method_v11, "_run_subprocess", side_effect=self._fake_model_run),
        ):
            return zuna_method_v11.zuna_reconstruct(
                stage0, [0, 1],
                sample_steps=1, cache_label="recording__F3-F4", debug=debug)

    def test_held_out_truth_never_enters_input_cache_or_output(self):
        data = self._data()
        reference_data = data - data[:, 2:, :].mean(axis=1, keepdims=True)
        debug = {}
        first = self._run(data, debug=debug)
        self.assertEqual(len(self.model_calls), 1)
        blind = debug["blind_input"]
        np.testing.assert_array_equal(blind[:, 2:, :], reference_data[:, 2:, :])
        self.assertFalse(np.array_equal(blind[:, :2, :], data[:, :2, :]))
        expected_scale = np.median(np.std(reference_data[:, 2:, :], axis=-1), axis=1)
        expected_scale = np.repeat(expected_scale[:, None], 2, axis=1)
        np.testing.assert_allclose(debug["blind_scale_uv"], expected_scale, rtol=1e-6)
        np.testing.assert_allclose(
            np.std(blind[:, :2, :], axis=-1),
            expected_scale, rtol=1e-5)

        # No self-calibration: fake model's dropped output is used directly; good channels
        # are restored bit-exactly by hard inpainting.
        expected_dropped = blind[:, :2, :] * 2.0 + 3.0
        np.testing.assert_allclose(first[:, :2, :], expected_dropped, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(first[:, 2:, :], reference_data[:, 2:, :])

        # Change only the held-out truth by an absurd amount. The blind fingerprint is the
        # same, so this must be a verified cache hit and produce the identical result.
        changed_truth = data.copy()
        changed_truth[:, :2, :] = changed_truth[:, :2, :] * 100_000 + 999_999
        second_debug = {}
        second = self._run(changed_truth, debug=second_debug)
        self.assertTrue(second_debug["cache_hit"])
        self.assertEqual(len(self.model_calls), 1)
        np.testing.assert_array_equal(second, first)

    def test_boundaries_masks_manifest_and_cache_integrity(self):
        data = self._data()
        debug = {}
        reconstruction = self._run(data, debug=debug)
        entry = Path(debug["cache_dir"])
        self.assertEqual(len(list((entry / "model_input").glob("epoch_*_raw.fif"))), 2)
        self.assertEqual(len(list((entry / "model_output" / "full_reconstruction").glob("*.fif"))), 2)
        self.assertTrue((entry / "blind_input_metadata.npz").is_file())
        self.assertTrue((entry / "reconstruction.npz").is_file())

        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        guards = manifest["scientific_guards"]
        self.assertTrue(all(value is False for value in guards.values()))
        self.assertFalse(manifest["settings"]["self_calibration"])
        self.assertIsNone(manifest["settings"]["zuna_highpass_hz"])
        self.assertFalse(manifest["settings"]["zuna_average_reference"])
        self.assertEqual(manifest["settings"]["epoch_serialization"], "one FIF per real epoch")
        self.assertEqual(manifest["model"]["identity"]["weight_sha256"], "a" * 64)
        self.assertTrue(manifest["coordinate_clamping"])
        self.assertEqual(
            manifest["coordinate_transform"],
            "official_zuna_componentwise_discrete_bin_clamp",
        )
        self.assertIn("positions_sha256", manifest["identity"])
        self.assertGreater(len(manifest["artifacts"]["model_io_inventory"]), 0)
        expected = data - data[:, 2:, :].mean(axis=1, keepdims=True)
        np.testing.assert_array_equal(reconstruction[:, 2:, :], expected[:, 2:, :])

        # A damaged tensor must never be silently reused.
        with (entry / "reconstruction.npz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self._run(data)

    def test_interrupted_model_run_preserves_and_resumes_only_missing_epoch(self):
        data = self._data()
        self.fail_after_first_epoch_once = True
        with self.assertRaisesRegex(RuntimeError, "injected model interruption"):
            self._run(data)

        entries = [path for path in Path(self.cache_root).iterdir() if path.is_dir()]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "model_failure")
        self.assertEqual(manifest["verified_completed_epochs"], [0])
        self.assertEqual(manifest["missing_or_invalid_epochs"], [1])
        self.assertEqual(manifest["attempts"][0]["status"], "failure")
        self.assertEqual(manifest["attempts"][0]["submitted_epochs"], [0, 1])
        self.assertTrue((entry / "model_output" / "hybrid" / "epoch_0000_raw.fif").is_file())
        self.assertFalse((entry / "model_output" / "hybrid" / "epoch_0001_raw.fif").is_file())
        self.assertFalse((entry / "reconstruction.npz").exists())
        self.assertEqual(self.model_calls[0], ["epoch_0000_raw.fif", "epoch_0001_raw.fif"])

        debug = {}
        recovered = self._run(data, debug=debug)
        self.assertEqual(recovered.shape, data.shape)
        self.assertEqual(self.model_calls[1], ["epoch_0001_raw.fif"])
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["recovery"]["epochs_submitted_this_attempt"], [1])
        self.assertEqual(manifest["recovery"]["preserved_completed_epochs_before_attempt"], [0])
        self.assertEqual([row["status"] for row in manifest["attempts"]], ["failure", "success"])

    def test_existing_unit_lock_fails_with_owner_information(self):
        data = self._data()
        self.fail_after_first_epoch_once = True
        with self.assertRaises(RuntimeError):
            self._run(data)
        entry = next(path for path in Path(self.cache_root).iterdir() if path.is_dir())
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        lock = Path(self.cache_root) / f'.{manifest["cache_key_sha256"]}.lock'
        lock.mkdir()
        (lock / "owner.json").write_text('{"pid": 4242}', encoding="utf-8")
        try:
            with self.assertRaisesRegex(RuntimeError, "4242"):
                self._run(data)
        finally:
            (lock / "owner.json").unlink()
            lock.rmdir()

    def test_helper_disables_second_preprocessing(self):
        helper = (Path(__file__).with_name("_recon11.py")).read_text(encoding="utf-8")
        for required in (
            '"data.do_avg_ref": "false"',
            '"data.v4_highpass_hz": "null" if highpass is None else highpass',
            '"data.v4_lowpass_hz": "null"',
            '"data.v4_notch_hz": "null"',
            '"data.v4_recon_seam_correct": "false"',
            'env["HF_HUB_OFFLINE"] = "1"',
        ):
            self.assertIn(required, helper)

    def test_phase2_rejects_double_filter_and_wrong_epoch_length(self):
        data = self._data()
        ch_names = ["F3", "F4", "C3", "C4", "P3", "P4"]
        positions = self._positions(ch_names)
        with self.assertRaisesRegex(ValueError, "forbids a second ZUNA highpass"):
            zuna_method_v11._settings(50, 5.0, 0.5, 8000, 333)
        with self.assertRaisesRegex(ValueError, "1280-sample epochs"):
            bad = self._run_stage0(data[..., :-1], ch_names, positions)
            zuna_method_v11._validate_inputs(bad, [0, 1])

    @staticmethod
    def _run_stage0(data, ch_names, positions):
        manifest = {
            "schema": "geeg-zuna-stage0-cache-v4",
            "identity": {
                "protocol_id": PROTOCOL_ID,
                "preprocessing_sha256": PREPROCESSING_SHA256,
                "cache_key_sha256": "8" * 64,
            },
            "output": {
                "data_sha256": stage0_cache.sha256_array(data),
                "positions_sha256": stage0_cache.sha256_array(positions),
                "channels_sha256": __import__("hashlib").sha256(
                    __import__("json").dumps(
                        list(ch_names), sort_keys=True, separators=(",", ":"), ensure_ascii=True
                    ).encode("utf-8")
                ).hexdigest(),
            },
        }
        return stage0_cache.VerifiedStage0(
            data=data, ch_names=tuple(ch_names), pos=positions, manifest=manifest
        )

    def test_official_coordinate_clamp_preserves_unique_tokens_and_position_identity(self):
        data = self._data()
        names = ["F3", "F4", "C3", "C4", "P3", "P4"]
        positions = self._positions(names)
        outside = positions.copy()
        outside[2, 2] = 0.141
        outside_validated = zuna_method_v11._validate_inputs(
            self._run_stage0(data, names, outside), [0, 1]
        )
        self.assertEqual(outside_validated[3][2, 2], 99)
        self.assertEqual(len(np.unique(outside_validated[3], axis=0)), len(names))
        collision = positions.copy()
        collision[3] = collision[2]
        with self.assertRaisesRegex(ValueError, "token collision"):
            zuna_method_v11._validate_inputs(
                self._run_stage0(data, names, collision), [0, 1]
            )

        validated = zuna_method_v11._validate_inputs(
            self._run_stage0(data, names, positions), [0, 1]
        )
        referenced, channels, pos, discrete, dropped, _good, manifest = validated
        settings = zuna_method_v11._settings(1, 5.0, None, 8000, 333)
        blind, _scale, _carrier = zuna_method_v11._blind_calibration_input(
            referenced, pos, dropped, [2, 3, 4, 5]
        )
        first = zuna_method_v11._cache_location(
            blind, channels, pos, discrete, dropped, "x", settings,
            self.fake_model["identity"], self.fake_code,
            manifest["identity"]["cache_key_sha256"],
        )[2]
        changed = pos.copy()
        changed[2, 0] += 0.001
        changed_discrete = (((changed - zuna_method_v11.POSITION_MIN) /
                             (zuna_method_v11.POSITION_MAX - zuna_method_v11.POSITION_MIN)) *
                            zuna_method_v11.POSITION_BINS).astype(np.int64)
        second = zuna_method_v11._cache_location(
            blind, channels, changed, changed_discrete, dropped, "x", settings,
            self.fake_model["identity"], self.fake_code,
            manifest["identity"]["cache_key_sha256"],
        )[2]
        self.assertNotEqual(first, second)

    def test_all_calibration_sensitivities_are_blind_positive_and_content_addressed(self):
        data = self._data()
        names = ["F3", "F4", "C3", "C4", "P3", "P4"]
        pos = self._positions(names)
        referenced, channels, pos, discrete, dropped, good, manifest = (
            zuna_method_v11._validate_inputs(
                self._run_stage0(data, names, pos), [0, 1]
            )
        )
        changed = referenced.copy()
        changed[:, dropped, :] += 1_000_000
        keys = set()
        for strategy in zuna_method_v11.CALIBRATION_STRATEGIES:
            blind, scale, _ = zuna_method_v11._blind_calibration_input(
                referenced, pos, dropped, good, strategy
            )
            changed_blind, changed_scale, _ = zuna_method_v11._blind_calibration_input(
                changed, pos, dropped, good, strategy
            )
            self.assertEqual(scale.shape, (referenced.shape[0], len(dropped)))
            self.assertTrue(np.all(scale > 0))
            np.testing.assert_array_equal(scale, changed_scale)
            np.testing.assert_array_equal(blind, changed_blind)
            settings = zuna_method_v11._settings(
                1, 5.0, None, 8000, 333, strategy
            )
            keys.add(zuna_method_v11._cache_location(
                blind, channels, pos, discrete, dropped, "x", settings,
                self.fake_model["identity"], self.fake_code,
                manifest["identity"]["cache_key_sha256"],
            )[2])
        self.assertEqual(len(keys), len(zuna_method_v11.CALIBRATION_STRATEGIES))


if __name__ == "__main__":
    unittest.main()
