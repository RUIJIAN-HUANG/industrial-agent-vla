from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_detection_map import (
    EvaluationInputError,
    bbox_iou_xywh,
    evaluate_files,
    percentile,
    write_evidence_bundle,
)


class DetectionMapEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.ground_truth_path = self.root / "ground_truth.json"
        self.predictions_path = self.root / "online_raw_predictions.json"
        self.ground_truth = {
            "images": [
                {"id": 1, "width": 100, "height": 100},
                {"id": 2, "width": 100, "height": 100},
            ],
            "categories": [{"id": 7, "name": "red_part"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 2, "category_id": 7, "bbox": [50, 50, 20, 20]},
            ],
        }
        self.ground_truth_path.write_text(
            json.dumps(self.ground_truth),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _linked_prediction(
        image_id: int,
        bbox: list[int],
        score: float = 0.9,
    ) -> dict[str, object]:
        return {
            "image_id": image_id,
            "category_id": 7,
            "bbox": bbox,
            "score": score,
            "trace_id": f"trace-{image_id}",
            "observation_id": f"obs-{image_id}",
            "image_sha256": f"sha256:{image_id:064x}",
        }

    def _write_predictions(self, payload: object) -> bytes:
        content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
        self.predictions_path.write_bytes(content)
        return content

    def test_perfect_predictions_have_full_ap_and_latency_percentiles(self) -> None:
        self._write_predictions(
            {
                "schema_version": "1.0",
                "predictions": [
                    self._linked_prediction(1, [10, 10, 20, 20]),
                    self._linked_prediction(2, [50, 50, 20, 20]),
                ],
                "frame_latencies": [
                    {"image_id": 1, "latency_ms": 10.0},
                    {"image_id": 2, "latency_ms": 30.0},
                    {"image_id": 3, "latency_ms": 20.0},
                ],
            }
        )

        report = evaluate_files(
            self.ground_truth_path,
            self.predictions_path,
            require_trace_linkage=True,
        )

        self.assertAlmostEqual(report["metrics"]["ap50"], 1.0)
        self.assertAlmostEqual(report["metrics"]["ap75"], 1.0)
        self.assertAlmostEqual(report["metrics"]["map_50_95"], 1.0)
        operating_point = report["metrics"]["operating_point_iou_0_50"]
        self.assertEqual(operating_point["precision"], 1.0)
        self.assertEqual(operating_point["recall"], 1.0)
        self.assertEqual(
            operating_point["per_category"]["7"]["precision"],
            1.0,
        )
        self.assertEqual(report["latency_ms"]["p50"], 20.0)
        self.assertEqual(report["latency_ms"]["p95"], 29.0)
        self.assertTrue(report["trace_linkage"]["complete"])

    def test_missing_detection_reduces_101_point_average_precision(self) -> None:
        self._write_predictions([self._linked_prediction(1, [10, 10, 20, 20])])

        report = evaluate_files(self.ground_truth_path, self.predictions_path)

        expected = 51 / 101
        self.assertAlmostEqual(report["metrics"]["ap50"], expected)
        self.assertAlmostEqual(report["metrics"]["map_50_95"], expected)
        self.assertEqual(
            report["metrics"]["operating_point_iou_0_50"]["recall"],
            0.5,
        )
        self.assertIsNone(report["latency_ms"]["p95"])

    def test_ground_truth_like_keys_in_prediction_artifact_are_rejected(self) -> None:
        self._write_predictions(
            {
                "predictions": [
                    {
                        **self._linked_prediction(1, [10, 10, 20, 20]),
                        "ground_truth": {"bbox": [10, 10, 20, 20]},
                    }
                ]
            }
        )

        with self.assertRaisesRegex(EvaluationInputError, "forbidden GT-like"):
            evaluate_files(self.ground_truth_path, self.predictions_path)

    def test_trace_linkage_can_be_mandatory_for_final_evidence(self) -> None:
        self._write_predictions(
            [{"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "score": 1}]
        )

        with self.assertRaisesRegex(EvaluationInputError, "trace linkage"):
            evaluate_files(
                self.ground_truth_path,
                self.predictions_path,
                require_trace_linkage=True,
            )

    def test_evidence_bundle_preserves_raw_predictions_and_never_copies_gt(
        self,
    ) -> None:
        original = self._write_predictions(
            [self._linked_prediction(1, [10, 10, 20, 20])]
        )
        report = evaluate_files(self.ground_truth_path, self.predictions_path)
        output_dir = self.root / "evidence"

        metrics_path, raw_path = write_evidence_bundle(
            report,
            self.predictions_path,
            output_dir,
        )

        self.assertEqual(raw_path.read_bytes(), original)
        self.assertTrue(metrics_path.is_file())
        self.assertFalse((output_dir / "ground_truth.json").exists())
        written_report = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertFalse(written_report["artifacts"]["ground_truth_copied_to_output"])

    def test_invalid_bbox_and_numeric_helpers_fail_safely(self) -> None:
        self._write_predictions(
            [{"image_id": 1, "category_id": 7, "bbox": [1, 2, 0, 4], "score": 1}]
        )
        with self.assertRaisesRegex(EvaluationInputError, "must be positive"):
            evaluate_files(self.ground_truth_path, self.predictions_path)

        self.assertAlmostEqual(
            bbox_iou_xywh((0, 0, 10, 10), (5, 5, 10, 10)),
            25 / 175,
        )
        self.assertEqual(percentile([10.0, 20.0, 30.0], 0.95), 29.0)

    def test_minimal_engine_rejects_crowd_semantics(self) -> None:
        self.ground_truth["annotations"][0]["iscrowd"] = 1
        self.ground_truth_path.write_text(
            json.dumps(self.ground_truth),
            encoding="utf-8",
        )
        self._write_predictions([self._linked_prediction(1, [10, 10, 20, 20])])

        with self.assertRaisesRegex(EvaluationInputError, "iscrowd/ignore"):
            evaluate_files(
                self.ground_truth_path,
                self.predictions_path,
                engine="minimal",
            )


if __name__ == "__main__":
    unittest.main()
