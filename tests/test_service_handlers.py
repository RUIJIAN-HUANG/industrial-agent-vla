from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from industrial_agent.errors import FailureCode, ImageCasError
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from industrial_agent.service_images import CasRequestImageResolver
from services.pi05 import build_v1_infer_handler as build_pi05_handler
from services.yolo import build_v1_detect_handler as build_yolo_handler


class RecordingBackend:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(request)
        return {"status": "ok"}


class ProductionServiceHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        image_cas = ImageCas(ImageCasConfig(root=Path(self.temporary.name)))
        self.resolver = CasRequestImageResolver(image_cas)
        self.rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 31
        self.rgb[:, :, 2] = 227
        self.image_cas = image_cas

    def reference(self, camera_id: str) -> dict[str, object]:
        return self.image_cas.write_rgb(
            self.rgb,
            camera_id=camera_id,
        ).to_dict()

    def test_all_three_route_cores_materialize_verified_pixels(self) -> None:
        cases = (
            (
                build_pi05_handler,
                {
                    "executor": "pi05",
                    "model_input": {
                        "prompt": "pack",
                        "observation": {
                            "camera": {
                                "full_image": self.reference("CAM_A_TOP"),
                                "wrist_image": None,
                            }
                        },
                    },
                },
                lambda request: request["model_input"]["observation"]["camera"][
                    "full_image"
                ],
            ),
            (
                build_yolo_handler,
                {
                    "detector": "yolo",
                    "image": self.reference("CAM_HANDOFF"),
                },
                lambda request: request["image"],
            ),
        )

        for builder, request, image_from_request in cases:
            backend = RecordingBackend()
            handler = builder(resolver=self.resolver, backend=backend)
            with self.subTest(builder=builder.__module__):
                self.assertEqual(handler.handle(request), {"status": "ok"})
                self.assertEqual(len(backend.requests), 1)
                pixels = image_from_request(backend.requests[0])
                self.assertIsInstance(pixels, np.ndarray)
                self.assertFalse(pixels.flags.writeable)
                np.testing.assert_array_equal(pixels, self.rgb)

    def test_missing_cas_blob_never_reaches_any_backend(self) -> None:
        def missing(camera_id: str) -> dict[str, object]:
            return {
                "uri": f"cas://sha256/{'f' * 64}",
                "image_sha256": f"sha256:{'f' * 64}",
                "camera_id": camera_id,
                "width": 1280,
                "height": 720,
            }

        cases = (
            (
                build_pi05_handler,
                {
                    "executor": "pi05",
                    "model_input": {
                        "prompt": "pack",
                        "observation": {
                            "camera": {
                                "full_image": missing("CAM_A_TOP"),
                                "wrist_image": None,
                            }
                        },
                    },
                },
            ),
            (
                build_yolo_handler,
                {
                    "detector": "yolo",
                    "image": missing("CAM_HANDOFF"),
                },
            ),
        )

        for builder, request in cases:
            backend = RecordingBackend()
            handler = builder(resolver=self.resolver, backend=backend)
            with self.subTest(builder=builder.__module__):
                with self.assertRaises(ImageCasError) as caught:
                    handler.handle(request)
                self.assertEqual(caught.exception.code, FailureCode.CAS_NOT_FOUND)
                self.assertEqual(backend.requests, [])

    def test_vla_route_rejects_wrong_executor_before_backend(self) -> None:
        backend = RecordingBackend()
        handler = build_pi05_handler(resolver=self.resolver, backend=backend)

        with self.assertRaises(ImageCasError) as caught:
            handler.handle(
                {
                    "executor": "retired",
                    "model_input": {},
                }
            )

        self.assertEqual(caught.exception.code, FailureCode.CAS_METADATA_MISMATCH)
        self.assertEqual(backend.requests, [])


if __name__ == "__main__":
    unittest.main()
