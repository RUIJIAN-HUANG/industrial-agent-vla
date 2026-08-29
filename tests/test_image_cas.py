from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import numpy as np

from industrial_agent.errors import FailureCode, ImageCasError
from industrial_agent.image_cas import (
    CAS_ROOT_ENV,
    ImageCas,
    ImageCasConfig,
)
from industrial_agent.perception import ImageReference
from simulation.rgb_cas_bridge import IsaacRgbCasPublisher


class ImageCasTests(unittest.TestCase):
    @staticmethod
    def _config(root: Path, **overrides: int) -> ImageCasConfig:
        values = {
            "max_blob_bytes": 1024 * 1024,
            "max_pixels": 1024 * 1024,
            "cache_max_bytes": 0,
            "missing_retry_count": 0,
            "missing_retry_delay_ms": 0,
        }
        values.update(overrides)
        return ImageCasConfig(root=root, **values)

    @staticmethod
    def _rgb(width: int = 5, height: int = 4) -> np.ndarray:
        values = np.arange(width * height * 3, dtype=np.uint8)
        return values.reshape(height, width, 3)

    def test_round_trip_restores_exact_immutable_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_cas = ImageCas(self._config(Path(temporary)))
            expected = self._rgb()

            reference = image_cas.write_rgb(expected, camera_id="CAM_A_TOP")
            resolved = image_cas.resolve_rgb(
                reference,
                expected_camera_id="CAM_A_TOP",
                expected_size=(5, 4),
            )

            np.testing.assert_array_equal(resolved.rgb, expected)
            self.assertEqual(resolved.rgb.dtype, np.uint8)
            self.assertEqual(resolved.rgb.shape, (4, 5, 3))
            self.assertFalse(resolved.rgb.flags.writeable)
            self.assertEqual(resolved.image_sha256, reference.image_sha256)
            digest = reference.image_sha256.removeprefix("sha256:")
            self.assertTrue(
                (Path(temporary) / "sha256" / digest[:2] / digest).is_file()
            )

    def test_same_pixels_produce_the_same_content_address(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_cas = ImageCas(self._config(Path(temporary)))
            first = image_cas.write_rgb(self._rgb(), camera_id="CAM_A_TOP")
            second = image_cas.write_rgb(self._rgb(), camera_id="CAM_A_TOP")
            self.assertEqual(first.uri, second.uri)
            self.assertEqual(first.image_sha256, second.image_sha256)

    def test_tampered_blob_fails_digest_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = ImageCas(self._config(root))
            reference = writer.write_rgb(self._rgb(), camera_id="CAM_A_TOP")
            digest = reference.image_sha256.removeprefix("sha256:")
            target = root / "sha256" / digest[:2] / digest
            content = bytearray(target.read_bytes())
            content[-1] ^= 0x01
            target.write_bytes(content)

            with self.assertRaises(ImageCasError) as caught:
                ImageCas(self._config(root)).resolve_rgb(reference)

            self.assertEqual(caught.exception.code, FailureCode.CAS_DIGEST_MISMATCH)
            self.assertFalse(caught.exception.retryable)

    def test_missing_blob_fails_closed_and_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            digest = "a" * 64
            reference = ImageReference(
                uri=f"cas://sha256/{digest}",
                image_sha256=f"sha256:{digest}",
                camera_id="CAM_A_TOP",
                width=5,
                height=4,
            )

            with self.assertRaises(ImageCasError) as caught:
                ImageCas(self._config(Path(temporary))).resolve_rgb(reference)

            self.assertEqual(caught.exception.code, FailureCode.CAS_NOT_FOUND)
            self.assertTrue(caught.exception.retryable)

    def test_digest_valid_non_png_blob_fails_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encoded = b"this is not a png"
            digest = sha256(encoded).hexdigest()
            target = root / "sha256" / digest[:2] / digest
            target.parent.mkdir(parents=True)
            target.write_bytes(encoded)
            reference = ImageReference(
                uri=f"cas://sha256/{digest}",
                image_sha256=f"sha256:{digest}",
                camera_id="CAM_A_TOP",
                width=5,
                height=4,
            )

            with self.assertRaises(ImageCasError) as caught:
                ImageCas(self._config(root)).resolve_rgb(reference)

            self.assertEqual(caught.exception.code, FailureCode.CAS_DECODE_FAILED)

    def test_blob_limit_is_checked_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encoded = b"x" * 128
            digest = sha256(encoded).hexdigest()
            target = root / "sha256" / digest[:2] / digest
            target.parent.mkdir(parents=True)
            target.write_bytes(encoded)
            reference = ImageReference(
                uri=f"cas://sha256/{digest}",
                image_sha256=f"sha256:{digest}",
                camera_id="CAM_A_TOP",
                width=5,
                height=4,
            )

            with self.assertRaises(ImageCasError) as caught:
                ImageCas(self._config(root, max_blob_bytes=64)).resolve_rgb(reference)

            self.assertEqual(caught.exception.code, FailureCode.CAS_LIMIT_EXCEEDED)

    def test_invalid_reference_is_normalized_to_cas_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid = {
                "uri": "file:///tmp/frame.png",
                "image_sha256": f"sha256:{'a' * 64}",
                "camera_id": "CAM_A_TOP",
                "width": 5,
                "height": 4,
            }
            with self.assertRaises(ImageCasError) as caught:
                ImageCas(self._config(Path(temporary))).resolve_rgb(invalid)
            self.assertEqual(caught.exception.code, FailureCode.CAS_METADATA_MISMATCH)

    def test_declared_dimensions_must_match_decoded_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_cas = ImageCas(self._config(Path(temporary)))
            reference = image_cas.write_rgb(self._rgb(), camera_id="CAM_A_TOP")
            invalid = ImageReference(
                uri=reference.uri,
                image_sha256=reference.image_sha256,
                camera_id=reference.camera_id,
                width=6,
                height=4,
            )

            with self.assertRaises(ImageCasError) as caught:
                image_cas.resolve_rgb(invalid)

            self.assertEqual(caught.exception.code, FailureCode.CAS_METADATA_MISMATCH)

    def test_expected_camera_is_enforced_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_cas = ImageCas(self._config(Path(temporary)))
            reference = image_cas.write_rgb(self._rgb(), camera_id="CAM_A_TOP")

            with self.assertRaises(ImageCasError) as caught:
                image_cas.resolve_rgb(
                    reference,
                    expected_camera_id="CAM_B_TOP",
                )

            self.assertEqual(caught.exception.code, FailureCode.CAS_METADATA_MISMATCH)

    def test_writer_rejects_non_uint8_or_non_rgb_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_cas = ImageCas(self._config(Path(temporary)))
            invalid_frames = (
                np.zeros((4, 5, 3), dtype=np.float32),
                np.zeros((4, 5, 4), dtype=np.uint8),
                np.zeros((4, 5), dtype=np.uint8),
            )
            for frame in invalid_frames:
                with self.subTest(dtype=frame.dtype, shape=frame.shape):
                    with self.assertRaises(ImageCasError) as caught:
                        image_cas.write_rgb(frame, camera_id="CAM_A_TOP")
                    self.assertEqual(
                        caught.exception.code,
                        FailureCode.CAS_METADATA_MISMATCH,
                    )

    def test_config_uses_explicit_environment_override(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = json.loads(
            (root / "configs" / "agent.v2.default.json").read_text(encoding="utf-8")
        )["image_cas"]
        configured = ImageCasConfig.from_mapping(
            raw,
            environ={CAS_ROOT_ENV: "D:/container/cas"},
        )
        self.assertEqual(configured.root, Path("D:/container/cas"))
        self.assertEqual(configured.max_pixels, 4_194_304)
        self.assertEqual(configured.LAYOUT, "sha256-v1")
        self.assertEqual(configured.ENCODING, "png")
        self.assertEqual(configured.DIGEST_SCOPE, "encoded_bytes")

    def test_protocol_constants_cannot_be_reconfigured(self) -> None:
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads(
            (root / "configs" / "agent.v2.default.json").read_text(encoding="utf-8")
        )["image_cas"]
        for field, value in (
            ("layout", "flat"),
            ("encoding", "jpeg"),
            ("digest_scope", "decoded_pixels"),
        ):
            with self.subTest(field=field):
                raw = dict(baseline)
                raw[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    ImageCasConfig.from_mapping(raw, environ={})

    def test_isaac_bridge_removes_alpha_without_changing_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(__file__).resolve().parents[1]
            scene = json.loads(
                (
                    root / "simulation" / "configs" / "single_bin_scene_v1.json"
                ).read_text(encoding="utf-8")
            )
            image_cas = ImageCas(
                self._config(
                    Path(temporary),
                    max_pixels=1280 * 720,
                    max_blob_bytes=4 * 1024 * 1024,
                )
            )
            publisher = IsaacRgbCasPublisher.from_scene_config(image_cas, scene)
            rgba = np.zeros((720, 1280, 4), dtype=np.uint8)
            rgba[:, :, 0] = 23
            rgba[:, :, 1] = 42
            rgba[:, :, 2] = 99
            rgba[:, :, 3] = 7

            reference = publisher.publish("CAM_B_TOP", rgba)
            resolved = image_cas.resolve_rgb(
                reference,
                expected_camera_id="CAM_B_TOP",
                expected_size=(1280, 720),
            )

            self.assertTrue(np.all(resolved.rgb[:, :, 0] == 23))
            self.assertTrue(np.all(resolved.rgb[:, :, 1] == 42))
            self.assertTrue(np.all(resolved.rgb[:, :, 2] == 99))


if __name__ == "__main__":
    unittest.main()
