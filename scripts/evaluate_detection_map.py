#!/usr/bin/env python3
"""Evaluate YOLO bbox predictions offline without exposing GT to online Agents.

The default ``minimal`` engine implements category-wise, 101-point interpolated
bbox AP at IoU 0.50:0.05:0.95 with at most 100 detections per image/category.
It is intentionally dependency-free and reports its limitations explicitly.
Use ``--engine pycocotools`` for canonical COCOeval competition evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

IOU_THRESHOLDS = tuple(round(0.50 + index * 0.05, 2) for index in range(10))
MAX_DETECTIONS_PER_IMAGE = 100
TRACE_FIELDS = ("trace_id", "observation_id", "image_sha256")
FORBIDDEN_PREDICTION_KEYS = {
    "annotations",
    "ground_truth",
    "groundtruth",
    "gt",
    "oracle",
    "privileged_state",
}


class EvaluationInputError(ValueError):
    """Raised when an evidence artifact cannot be evaluated safely."""


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvaluationInputError(f"{field} must be a JSON array")
    return value


def _require_id(value: object, field: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EvaluationInputError(f"{field} must be a string or integer ID")
    if isinstance(value, str) and not value:
        raise EvaluationInputError(f"{field} must not be empty")
    return value


def _bbox_xywh(value: object, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise EvaluationInputError(f"{field} must be [x, y, width, height]")
    if not all(_is_finite_number(item) for item in value):
        raise EvaluationInputError(f"{field} must contain four finite numbers")
    x, y, width, height = (float(item) for item in value)
    if width <= 0 or height <= 0:
        raise EvaluationInputError(f"{field} width and height must be positive")
    return x, y, width, height


def bbox_iou_xywh(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return IoU for two COCO ``[x, y, width, height]`` boxes."""

    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile using the inclusive endpoints."""

    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_ground_truth(
    payload: object,
) -> tuple[
    dict[tuple[str | int, str | int], list[tuple[float, float, float, float]]],
    dict[str | int, str],
    int,
]:
    if not isinstance(payload, dict):
        raise EvaluationInputError("ground truth must be a COCO object")

    images = _require_list(payload.get("images"), "ground_truth.images")
    annotations = _require_list(
        payload.get("annotations"),
        "ground_truth.annotations",
    )
    categories = _require_list(payload.get("categories"), "ground_truth.categories")

    image_ids: set[str | int] = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise EvaluationInputError(f"ground_truth.images[{index}] must be object")
        image_id = _require_id(image.get("id"), f"ground_truth.images[{index}].id")
        if image_id in image_ids:
            raise EvaluationInputError(f"duplicate ground-truth image id: {image_id}")
        image_ids.add(image_id)

    category_names: dict[str | int, str] = {}
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise EvaluationInputError(
                f"ground_truth.categories[{index}] must be object"
            )
        category_id = _require_id(
            category.get("id"),
            f"ground_truth.categories[{index}].id",
        )
        name = category.get("name")
        if not isinstance(name, str) or not name:
            raise EvaluationInputError(
                f"ground_truth.categories[{index}].name must be non-empty string"
            )
        if category_id in category_names:
            raise EvaluationInputError(
                f"duplicate ground-truth category id: {category_id}"
            )
        category_names[category_id] = name

    grouped: dict[
        tuple[str | int, str | int],
        list[tuple[float, float, float, float]],
    ] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise EvaluationInputError(
                f"ground_truth.annotations[{index}] must be object"
            )
        image_id = _require_id(
            annotation.get("image_id"),
            f"ground_truth.annotations[{index}].image_id",
        )
        category_id = _require_id(
            annotation.get("category_id"),
            f"ground_truth.annotations[{index}].category_id",
        )
        if image_id not in image_ids:
            raise EvaluationInputError(
                f"annotation {index} references unknown image_id {image_id}"
            )
        if category_id not in category_names:
            raise EvaluationInputError(
                f"annotation {index} references unknown category_id {category_id}"
            )
        grouped[(image_id, category_id)].append(
            _bbox_xywh(
                annotation.get("bbox"),
                f"ground_truth.annotations[{index}].bbox",
            )
        )

    return dict(grouped), category_names, len(image_ids)


def _ground_truth_requires_full_cocoeval(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        return False
    return any(
        isinstance(annotation, dict)
        and (
            annotation.get("iscrowd", 0) not in (0, False, None)
            or annotation.get("ignore", 0) not in (0, False, None)
        )
        for annotation in annotations
    )


def _prediction_list(payload: object) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise EvaluationInputError(
            "predictions must be a COCO result array or an object with predictions"
        )
    forbidden = FORBIDDEN_PREDICTION_KEYS.intersection(
        str(key).lower() for key in payload
    )
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise EvaluationInputError(
            f"raw predictions contain forbidden GT-like top-level keys: {names}"
        )
    return _require_list(payload.get("predictions"), "predictions.predictions")


def _validate_predictions(
    payload: object,
    valid_images: set[str | int],
    valid_categories: set[str | int],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, prediction in enumerate(_prediction_list(payload)):
        if not isinstance(prediction, dict):
            raise EvaluationInputError(f"predictions[{index}] must be object")
        forbidden = FORBIDDEN_PREDICTION_KEYS.intersection(
            str(key).lower() for key in prediction
        )
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise EvaluationInputError(
                f"predictions[{index}] contains forbidden GT-like keys: {names}"
            )
        image_id = _require_id(
            prediction.get("image_id"),
            f"predictions[{index}].image_id",
        )
        category_id = _require_id(
            prediction.get("category_id"),
            f"predictions[{index}].category_id",
        )
        if image_id not in valid_images:
            raise EvaluationInputError(
                f"prediction {index} references unknown image_id {image_id}"
            )
        if category_id not in valid_categories:
            raise EvaluationInputError(
                f"prediction {index} references unknown category_id {category_id}"
            )
        score = prediction.get("score")
        if not _is_finite_number(score) or not 0.0 <= float(score) <= 1.0:
            raise EvaluationInputError(
                f"predictions[{index}].score must be finite within [0, 1]"
            )
        normalized.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": list(
                    _bbox_xywh(
                        prediction.get("bbox"),
                        f"predictions[{index}].bbox",
                    )
                ),
                "score": float(score),
                "_source_index": index,
            }
        )
    return normalized


def _latency_values(payload: object, predictions: list[Any]) -> list[float]:
    samples: list[object] = []
    if isinstance(payload, dict):
        if "latency_samples_ms" in payload:
            samples = _require_list(
                payload["latency_samples_ms"],
                "predictions.latency_samples_ms",
            )
        elif "frame_latencies" in payload:
            frames = _require_list(
                payload["frame_latencies"],
                "predictions.frame_latencies",
            )
            for index, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    raise EvaluationInputError(
                        f"predictions.frame_latencies[{index}] must be object"
                    )
                samples.append(frame.get("latency_ms"))
    if not samples:
        seen: dict[tuple[object, object, object], float] = {}
        for index, prediction in enumerate(predictions):
            if not isinstance(prediction, dict) or "latency_ms" not in prediction:
                continue
            value = prediction["latency_ms"]
            if not _is_finite_number(value) or float(value) < 0:
                raise EvaluationInputError(
                    f"predictions[{index}].latency_ms must be finite and non-negative"
                )
            key = (
                prediction.get("trace_id"),
                prediction.get("observation_id"),
                prediction.get("image_id"),
            )
            if key == (None, None, prediction.get("image_id")):
                key = ("source_index", index, prediction.get("image_id"))
            previous = seen.get(key)
            if previous is not None and previous != float(value):
                raise EvaluationInputError(
                    f"conflicting latency values for frame key {key}"
                )
            seen[key] = float(value)
        samples = list(seen.values())

    result: list[float] = []
    for index, sample in enumerate(samples):
        if not _is_finite_number(sample) or float(sample) < 0:
            raise EvaluationInputError(
                f"latency sample {index} must be finite and non-negative"
            )
        result.append(float(sample))
    return result


def _trace_linkage(payload: object, predictions: list[Any]) -> dict[str, Any]:
    linked = 0
    for prediction in predictions:
        if isinstance(prediction, dict) and all(
            isinstance(prediction.get(field), str) and prediction[field]
            for field in TRACE_FIELDS
        ):
            linked += 1
    return {
        "required_fields": list(TRACE_FIELDS),
        "linked_predictions": linked,
        "prediction_count": len(predictions),
        "complete": linked == len(predictions),
        "payload_schema_version": (
            payload.get("schema_version") if isinstance(payload, dict) else None
        ),
    }


def _interpolated_ap(
    true_positives: Sequence[int],
    false_positives: Sequence[int],
    positive_count: int,
) -> float:
    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for true_positive, false_positive in zip(
        true_positives,
        false_positives,
        strict=True,
    ):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        recalls.append(cumulative_tp / positive_count)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))

    total = 0.0
    for index in range(101):
        recall_level = index / 100.0
        eligible = [
            precision
            for recall, precision in zip(recalls, precisions, strict=True)
            if recall >= recall_level
        ]
        total += max(eligible, default=0.0)
    return total / 101.0


def operating_point_at_iou(
    grouped_ground_truth: Mapping[
        tuple[str | int, str | int],
        Sequence[Sequence[float]],
    ],
    category_names: Mapping[str | int, str],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Return micro and per-class P/R after score filtering done by the detector.

    The raw prediction file therefore also freezes the operating confidence/NMS
    policy. AP still ranks all submitted predictions by score.
    """

    aggregate_tp = 0
    aggregate_fp = 0
    aggregate_fn = 0
    per_category: dict[str, dict[str, Any]] = {}
    for category_id, category_name in category_names.items():
        positive_count = sum(
            len(boxes)
            for (image_id, grouped_category), boxes in grouped_ground_truth.items()
            if grouped_category == category_id
        )
        grouped_predictions: dict[str | int, list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for prediction in predictions:
            if prediction["category_id"] == category_id:
                grouped_predictions[prediction["image_id"]].append(prediction)
        ranked: list[Mapping[str, Any]] = []
        for image_predictions in grouped_predictions.values():
            ranked.extend(
                sorted(
                    image_predictions,
                    key=lambda item: (-item["score"], item["_source_index"]),
                )[:MAX_DETECTIONS_PER_IMAGE]
            )
        ranked.sort(key=lambda item: (-item["score"], item["_source_index"]))

        matched: dict[str | int, list[bool]] = {
            image_id: [False] * len(boxes)
            for (image_id, grouped_category), boxes in grouped_ground_truth.items()
            if grouped_category == category_id
        }
        true_positives = 0
        false_positives = 0
        for prediction in ranked:
            image_id = prediction["image_id"]
            boxes = grouped_ground_truth.get((image_id, category_id), ())
            best_index = -1
            best_iou = iou_threshold
            for index, ground_truth_box in enumerate(boxes):
                if matched[image_id][index]:
                    continue
                overlap = bbox_iou_xywh(prediction["bbox"], ground_truth_box)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0:
                matched[image_id][best_index] = True
                true_positives += 1
            else:
                false_positives += 1

        false_negatives = positive_count - true_positives
        precision_denominator = true_positives + false_positives
        per_category[str(category_id)] = {
            "name": category_name,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": (
                true_positives / precision_denominator if precision_denominator else 0.0
            ),
            "recall": (true_positives / positive_count if positive_count else None),
        }
        aggregate_tp += true_positives
        aggregate_fp += false_positives
        aggregate_fn += false_negatives

    aggregate_precision_denominator = aggregate_tp + aggregate_fp
    aggregate_recall_denominator = aggregate_tp + aggregate_fn
    return {
        "iou_threshold": iou_threshold,
        "definition": (
            "micro precision/recall over submitted predictions after the "
            "detector's frozen confidence and NMS policy"
        ),
        "true_positives": aggregate_tp,
        "false_positives": aggregate_fp,
        "false_negatives": aggregate_fn,
        "precision": (
            aggregate_tp / aggregate_precision_denominator
            if aggregate_precision_denominator
            else 0.0
        ),
        "recall": (
            aggregate_tp / aggregate_recall_denominator
            if aggregate_recall_denominator
            else None
        ),
        "per_category": per_category,
    }


def evaluate_minimal_coco(
    grouped_ground_truth: Mapping[
        tuple[str | int, str | int],
        Sequence[Sequence[float]],
    ],
    category_names: Mapping[str | int, str],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute dependency-free, bbox-only, COCO-style 101-point AP."""

    by_category: dict[str | int, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_category[prediction["category_id"]].append(prediction)

    positive_counts: dict[str | int, int] = {
        category_id: sum(
            len(boxes)
            for (image_id, grouped_category), boxes in grouped_ground_truth.items()
            if grouped_category == category_id
        )
        for category_id in category_names
    }
    evaluated_categories = [
        category_id
        for category_id in category_names
        if positive_counts[category_id] > 0
    ]
    if not evaluated_categories:
        raise EvaluationInputError("ground truth has no positive bbox annotations")

    per_category_threshold: dict[str | int, dict[str, float]] = {}
    for category_id in evaluated_categories:
        grouped_predictions: dict[str | int, list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for prediction in by_category.get(category_id, []):
            grouped_predictions[prediction["image_id"]].append(prediction)

        limited: list[Mapping[str, Any]] = []
        for image_predictions in grouped_predictions.values():
            limited.extend(
                sorted(
                    image_predictions,
                    key=lambda item: (-item["score"], item["_source_index"]),
                )[:MAX_DETECTIONS_PER_IMAGE]
            )
        ranked = sorted(
            limited,
            key=lambda item: (-item["score"], item["_source_index"]),
        )

        threshold_scores: dict[str, float] = {}
        for threshold in IOU_THRESHOLDS:
            matched: dict[str | int, list[bool]] = {
                image_id: [False] * len(boxes)
                for (image_id, grouped_category), boxes in grouped_ground_truth.items()
                if grouped_category == category_id
            }
            true_positives: list[int] = []
            false_positives: list[int] = []
            for prediction in ranked:
                image_id = prediction["image_id"]
                boxes = grouped_ground_truth.get((image_id, category_id), ())
                best_index = -1
                best_iou = threshold
                for index, ground_truth_box in enumerate(boxes):
                    if matched[image_id][index]:
                        continue
                    overlap = bbox_iou_xywh(prediction["bbox"], ground_truth_box)
                    if overlap >= best_iou:
                        best_iou = overlap
                        best_index = index
                if best_index >= 0:
                    matched[image_id][best_index] = True
                    true_positives.append(1)
                    false_positives.append(0)
                else:
                    true_positives.append(0)
                    false_positives.append(1)

            threshold_scores[f"{threshold:.2f}"] = _interpolated_ap(
                true_positives,
                false_positives,
                positive_counts[category_id],
            )
        per_category_threshold[category_id] = threshold_scores

    per_iou = {
        f"{threshold:.2f}": sum(
            per_category_threshold[category_id][f"{threshold:.2f}"]
            for category_id in evaluated_categories
        )
        / len(evaluated_categories)
        for threshold in IOU_THRESHOLDS
    }
    per_category = {
        str(category_id): {
            "name": category_names[category_id],
            "positive_count": positive_counts[category_id],
            "ap50": per_category_threshold[category_id]["0.50"],
            "ap75": per_category_threshold[category_id]["0.75"],
            "map_50_95": sum(per_category_threshold[category_id].values())
            / len(IOU_THRESHOLDS),
        }
        for category_id in evaluated_categories
    }
    return {
        "map_50_95": sum(per_iou.values()) / len(IOU_THRESHOLDS),
        "ap50": per_iou["0.50"],
        "ap75": per_iou["0.75"],
        "per_iou_ap": per_iou,
        "per_category": per_category,
    }


def evaluate_pycocotools(
    ground_truth_path: Path,
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run canonical COCOeval when the optional dependency is installed."""

    try:
        from pycocotools.coco import COCO  # type: ignore[import-not-found]
        from pycocotools.cocoeval import COCOeval  # type: ignore[import-not-found]
    except ImportError as exc:
        raise EvaluationInputError(
            "pycocotools engine requested but unavailable; install pycocotools "
            "or rerun with --engine minimal (bbox-only 101-point fallback)"
        ) from exc

    coco_results = [
        {
            "image_id": item["image_id"],
            "category_id": item["category_id"],
            "bbox": item["bbox"],
            "score": item["score"],
        }
        for item in predictions
    ]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ground_truth = COCO(str(ground_truth_path))
            detections = ground_truth.loadRes(coco_results)
            evaluator = COCOeval(ground_truth, detections, "bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    except Exception as exc:
        raise EvaluationInputError(f"pycocotools COCOeval failed: {exc}") from exc

    precision = evaluator.eval["precision"]
    per_iou: dict[str, float] = {}
    for index, threshold in enumerate(evaluator.params.iouThrs):
        values = precision[index, :, :, 0, -1]
        valid = values[values > -1]
        per_iou[f"{float(threshold):.2f}"] = float(valid.mean()) if valid.size else 0.0

    per_category: dict[str, dict[str, Any]] = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, category_index, 0, -1]
        valid = values[values > -1]
        category = ground_truth.cats[category_id]
        per_category[str(category_id)] = {
            "name": category.get("name", str(category_id)),
            "map_50_95": float(valid.mean()) if valid.size else 0.0,
        }
    return {
        "map_50_95": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "per_iou_ap": per_iou,
        "per_category": per_category,
    }


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationInputError(f"{label} file not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"cannot read {label} JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def evaluate_files(
    ground_truth_path: Path,
    predictions_path: Path,
    *,
    engine: str = "minimal",
    require_trace_linkage: bool = False,
) -> dict[str, Any]:
    """Evaluate two separate files and return a JSON-serializable evidence report."""

    ground_truth_payload = _read_json(ground_truth_path, "ground truth")
    prediction_payload = _read_json(predictions_path, "raw predictions")
    grouped_ground_truth, category_names, image_count = _validate_ground_truth(
        ground_truth_payload
    )
    valid_images = {image_id for image_id, _ in grouped_ground_truth}
    if isinstance(ground_truth_payload, dict):
        valid_images.update(
            _require_id(image["id"], "ground_truth.images[].id")
            for image in ground_truth_payload["images"]
        )
    raw_prediction_list = _prediction_list(prediction_payload)
    normalized_predictions = _validate_predictions(
        prediction_payload,
        valid_images,
        set(category_names),
    )
    latencies = _latency_values(prediction_payload, raw_prediction_list)
    trace_linkage = _trace_linkage(prediction_payload, raw_prediction_list)
    if require_trace_linkage and not trace_linkage["complete"]:
        raise EvaluationInputError(
            "trace linkage is incomplete; every prediction must include "
            "trace_id, observation_id, and image_sha256"
        )

    warnings: list[str] = []
    selected_engine = engine
    requires_full_cocoeval = _ground_truth_requires_full_cocoeval(ground_truth_payload)
    if engine == "auto":
        try:
            metrics = evaluate_pycocotools(
                ground_truth_path,
                normalized_predictions,
            )
            selected_engine = "pycocotools"
        except EvaluationInputError as exc:
            if requires_full_cocoeval:
                raise EvaluationInputError(
                    "ground truth contains iscrowd/ignore annotations; "
                    "pycocotools is required and minimal fallback is unsafe"
                ) from exc
            warnings.append(f"{exc}; used minimal fallback")
            selected_engine = "minimal"
            metrics = evaluate_minimal_coco(
                grouped_ground_truth,
                category_names,
                normalized_predictions,
            )
    elif engine == "pycocotools":
        metrics = evaluate_pycocotools(ground_truth_path, normalized_predictions)
    elif engine == "minimal":
        if requires_full_cocoeval:
            raise EvaluationInputError(
                "minimal engine does not support COCO iscrowd/ignore semantics; "
                "use --engine pycocotools"
            )
        metrics = evaluate_minimal_coco(
            grouped_ground_truth,
            category_names,
            normalized_predictions,
        )
        warnings.append(
            "minimal engine is bbox-only 101-point AP for area=all and maxDet=100; "
            "use pycocotools for canonical COCOeval evidence"
        )
    else:
        raise EvaluationInputError(f"unsupported engine: {engine}")

    if not latencies:
        warnings.append(
            "no frame latency samples found; P50/P95 are null and speed is unproven"
        )
    if not trace_linkage["complete"]:
        warnings.append(
            "raw predictions lack complete same-frame trace linkage; final evidence "
            "should include trace_id, observation_id, and image_sha256"
        )

    metrics["operating_point_iou_0_50"] = operating_point_at_iou(
        grouped_ground_truth,
        category_names,
        normalized_predictions,
    )

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "offline_detection_evaluation_only",
        "engine": selected_engine,
        "metric_definition": {
            "task": "bbox",
            "score_range": "[0,1]",
            "iou_thresholds": list(IOU_THRESHOLDS),
            "ap_interpolation": "101-point",
            "max_detections_per_image": MAX_DETECTIONS_PER_IMAGE,
            "value_range": "[0,1]",
        },
        "dataset": {
            "image_count": image_count,
            "annotation_count": sum(
                len(boxes) for boxes in grouped_ground_truth.values()
            ),
            "category_count": len(category_names),
            "prediction_count": len(normalized_predictions),
        },
        "metrics": metrics,
        "latency_ms": {
            "sample_count": len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "minimum": min(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
        "trace_linkage": trace_linkage,
        "artifacts": {
            "ground_truth_sha256": _sha256(ground_truth_path),
            "raw_predictions_sha256": _sha256(predictions_path),
            "ground_truth_copied_to_output": False,
            "raw_predictions_output": "raw_predictions.json",
        },
        "warnings": warnings,
    }


def _write_bytes(path: Path, content: bytes, overwrite: bool) -> None:
    if path.exists() and path.read_bytes() != content and not overwrite:
        raise EvaluationInputError(
            f"refusing to overwrite existing evidence artifact: {path}; "
            "pass --overwrite only for an intentional rerun"
        )
    path.write_bytes(content)


def write_evidence_bundle(
    report: Mapping[str, Any],
    predictions_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write metrics and an exact raw-prediction copy; never copy offline GT."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / "raw_predictions.json"
    metrics_output = output_dir / "detection_metrics.json"
    prediction_bytes = predictions_path.read_bytes()
    if raw_output.resolve() != predictions_path.resolve():
        _write_bytes(raw_output, prediction_bytes, overwrite)
    metrics_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes(metrics_output, metrics_bytes, overwrite)
    return metrics_output, raw_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline YOLO bbox AP/latency evaluator. GT is read only by this "
            "process and is never copied into the online prediction artifact."
        )
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="COCO ground-truth JSON; offline evaluator access only",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="raw COCO results array or prediction envelope JSON",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for detection_metrics.json and exact raw_predictions.json",
    )
    parser.add_argument(
        "--engine",
        choices=("minimal", "pycocotools", "auto"),
        default="minimal",
        help="minimal is dependency-free; pycocotools is canonical COCOeval",
    )
    parser.add_argument(
        "--require-trace-linkage",
        action="store_true",
        help=(
            "fail unless every detection links trace_id + observation_id + "
            "image_sha256 to the frame sent to VLA"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing evidence bundle only for an intentional rerun",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = evaluate_files(
            arguments.ground_truth,
            arguments.predictions,
            engine=arguments.engine,
            require_trace_linkage=arguments.require_trace_linkage,
        )
        metrics_path, raw_path = write_evidence_bundle(
            report,
            arguments.predictions,
            arguments.output_dir,
            overwrite=arguments.overwrite,
        )
    except (EvaluationInputError, OSError) as exc:
        print(f"[FAIL] detection evaluation: {exc}", file=sys.stderr)
        return 2

    metrics = report["metrics"]
    latency = report["latency_ms"]
    print(
        "[PASS] offline detection evaluation: "
        f"engine={report['engine']} "
        f"AP50={metrics['ap50']:.6f} "
        f"AP75={metrics['ap75']:.6f} "
        f"mAP50:95={metrics['map_50_95']:.6f} "
        f"P50/P95={latency['p50']}/{latency['p95']} ms"
    )
    print(f"metrics: {metrics_path}")
    print(f"raw predictions: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
