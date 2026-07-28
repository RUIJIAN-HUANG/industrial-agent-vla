from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from industrial_agent.errors import FailureCode, ImageCasError
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from industrial_agent.service_images import (
    FROZEN_RGB_SIZE,
    CasRequestImageResolver,
)


class ServiceImageResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image_cas = ImageCas(ImageCasConfig(root=Path(self.temporary.name)))
        self.resolver = CasRequestImageResolver(self.image_cas)
        width, height = FROZEN_RGB_SIZE
        self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.rgb[:, :, 1] = 127

    def _reference(self, camera_id: str) -> dict[str, object]:
        return self.image_cas.write_rgb(
            self.rgb,
            camera_id=camera_id,
        ).to_dict()

    def test_resolves_pi05_openvla_and_yolo_requests_to_verified_pixels(self) -> None:
        pi05 = self.resolver.resolve_vla_request(
            {
                "executor": "pi05",
                "model_input": {
                    "observation": {
                        "camera": {
                            "full_image": self._reference("CAM_A_TOP"),
                            "wrist_image": None,
                        }
                    }
                },
            }
        )
        openvla = self.resolver.resolve_vla_request(
            {
                "executor": "openvla_oft",
                "model_input": {
                    "full_image": self._reference("CAM_B_TOP"),
                    "wrist_image": None,
                },
            }
        )
        yolo = self.resolver.resolve_yolo_request(
            {
                "detector": "yolo",
                "image": self._reference("CAM_HANDOFF"),
            }
        )

        for resolved in (pi05.full_image, openvla.full_image, yolo.image):
            self.assertEqual(
                (resolved.width, resolved.height),
                FROZEN_RGB_SIZE,
            )
            self.assertFalse(resolved.rgb.flags.writeable)
            np.testing.assert_array_equal(resolved.rgb, self.rgb)

    def test_rejects_non_null_wrist_wrong_size_and_unknown_camera(self) -> None:
        arm_a = self._reference("CAM_A_TOP")
        with self.assertRaises(ImageCasError) as caught:
            self.resolver.resolve_vla_request(
                {
                    "executor": "pi05",
                    "model_input": {
                        "observation": {
                            "camera": {
                                "full_image": arm_a,
                                "wrist_image": arm_a,
                            }
                        }
                    },
                }
            )
        self.assertEqual(caught.exception.code, FailureCode.CAS_METADATA_MISMATCH)

        wrong_size = dict(arm_a)
        wrong_size["width"] = 640
        with self.assertRaises(ImageCasError):
            self.resolver.resolve_vla_request(
                {
                    "executor": "pi05",
                    "model_input": {
                        "observation": {
                            "camera": {
                                "full_image": wrong_size,
                                "wrist_image": None,
                            }
                        }
                    },
                }
            )

        unknown = dict(arm_a)
        unknown["camera_id"] = "CAM_WRIST"
        with self.assertRaises(ImageCasError):
            self.resolver.resolve_yolo_request({"detector": "yolo", "image": unknown})

    def test_missing_blob_never_becomes_a_placeholder(self) -> None:
        missing = {
            "uri": f"cas://sha256/{'f' * 64}",
            "image_sha256": f"sha256:{'f' * 64}",
            "camera_id": "CAM_B_TOP",
            "width": 1280,
            "height": 720,
        }
        with self.assertRaises(ImageCasError) as caught:
            self.resolver.resolve_vla_request(
                {
                    "executor": "openvla_oft",
                    "model_input": {
                        "full_image": missing,
                        "wrist_image": None,
                    },
                }
            )
        self.assertEqual(caught.exception.code, FailureCode.CAS_NOT_FOUND)

    def test_rejects_non_scalar_executor_and_camera_metadata(self) -> None:
        with self.assertRaises(ImageCasError):
            self.resolver.resolve_vla_request({"executor": [], "model_input": {}})
        with self.assertRaises(ImageCasError):
            self.resolver.resolve_yolo_request(
                {
                    "detector": "yolo",
                    "image": {"camera_id": []},
                }
            )


if __name__ == "__main__":
    unittest.main()
