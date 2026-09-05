from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

import numpy as np

from simulation.v2_competition_controller import CompetitionController, UiYoloDetection
from simulation.v2_competition_window import (
    _annotate_yolo_frame,
    _yolo_detection_text,
)
from simulation.v2_competition_window import V2CompetitionWindow


class V2CompetitionWindowTests(unittest.TestCase):
    def test_annotate_yolo_frame_draws_red_bbox(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        detection = UiYoloDetection(
            detection_id="det-1",
            class_name="shaft_upright",
            confidence=0.875,
            bbox_xyxy=(3.0, 4.0, 15.0, 12.0),
            camera_id="CAM_A_TOP",
            image_width=30,
            image_height=20,
        )

        annotated = _annotate_yolo_frame(frame, (detection,))

        self.assertEqual(annotated.shape, (20, 30, 4))
        self.assertEqual(annotated[4, 3].tolist(), [255, 32, 32, 255])
        self.assertEqual(annotated[11, 14].tolist(), [255, 32, 32, 255])
        self.assertEqual(annotated[8, 8].tolist(), [0, 0, 0, 255])

    def test_yolo_detection_text_includes_class_confidence_and_bbox(self) -> None:
        controller = CompetitionController(max_steps=8, verifier_configured=True)
        controller.update_yolo_detections(
            {
                "observation_id": "obs-1",
                "camera_id": "CAM_A_TOP",
                "image_width": 1280,
                "image_height": 720,
                "detections": [
                    {
                        "detection_id": "det-1",
                        "class_name": "shaft_upright",
                        "confidence": 0.875,
                        "bbox_xyxy": [100.0, 120.0, 240.0, 360.0],
                        "camera_id": "CAM_A_TOP",
                        "image_width": 1280,
                        "image_height": 720,
                    }
                ],
            }
        )

        text = _yolo_detection_text(controller.snapshot())

        self.assertIn("shaft_upright", text)
        self.assertIn("0.875", text)
        self.assertIn("(100.0, 120.0, 240.0, 360.0)", text)

    def test_window_does_not_override_kit_global_font_atlas(self) -> None:
        source = textwrap.dedent(inspect.getsource(V2CompetitionWindow.__init__))
        tree = ast.parse(source)

        styled_vstacks = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "VStack":
                continue
            if any(keyword.arg == "style" for keyword in node.keywords):
                styled_vstacks.append(node)

        self.assertEqual(styled_vstacks, [])
        self.assertNotIn("_resolve_cjk_font", source)
        self.assertNotIn("_ui_styles", source)


if __name__ == "__main__":
    unittest.main()
