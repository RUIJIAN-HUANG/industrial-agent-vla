"""YOLO model backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One model detection in pixel-space xyxy format."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ModelOutput:
    """Validated inference output and backend-measured NMS/postprocess time."""

    detections: tuple[Detection, ...]
    nms_ms: float


class YoloModel(Protocol):
    """Interface shared by mock and real YOLO implementations."""

    def detect(
        self,
        image: np.ndarray,
        *,
        allowed_class_names: Sequence[str],
        confidence: float,
        iou: float,
    ) -> ModelOutput:
        """Run inference on one immutable RGB image."""


class MockYoloModel:
    """Deterministic backend used before production weights are available."""

    def detect(
        self,
        image: np.ndarray,
        *,
        allowed_class_names: Sequence[str],
        confidence: float,
        iou: float,
    ) -> ModelOutput:
        del allowed_class_names, confidence, iou

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")

        return ModelOutput(detections=(), nms_ms=0.0)


class UltralyticsYoloModel:
    """Production adapter loaded only when real mode is enabled."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "install the 'model' extra to run YOLO in real mode"
            ) from exc
        self._model = YOLO(str(config["checkpoint_path"]))
        self._device = str(config["model"]["device"])
        expected_names = tuple(str(item) for item in config["class_names"])
        actual_names = _ordered_model_names(getattr(self._model, "names", None))
        if actual_names != expected_names:
            raise RuntimeError(
                "YOLO checkpoint class map does not match the frozen deployment "
                f"map: expected={expected_names!r}, actual={actual_names!r}"
            )
        self._class_ids_by_name = {
            name: class_id for class_id, name in enumerate(expected_names)
        }

    def detect(
        self,
        image: np.ndarray,
        *,
        allowed_class_names: Sequence[str],
        confidence: float,
        iou: float,
    ) -> ModelOutput:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have uint8 shape (height, width, 3)")
        unknown_classes = set(allowed_class_names) - set(self._class_ids_by_name)
        if unknown_classes:
            names = sorted(unknown_classes)
            raise ValueError(f"allowed_class_names contains unknown classes: {names}")
        allowed_ids = (
            [self._class_ids_by_name[name] for name in allowed_class_names]
            if allowed_class_names
            else None
        )
        # The shared CAS contract is RGB. Ultralytics documents that NumPy
        # inputs are OpenCV-compatible BGR, so convert explicitly at this one
        # model boundary and keep every upstream frame byte-identical.
        bgr_image = np.ascontiguousarray(image[..., ::-1])
        results = self._model.predict(
            source=bgr_image,
            conf=confidence,
            iou=iou,
            device=self._device,
            classes=allowed_ids,
            verbose=False,
        )
        if not results:
            raise RuntimeError("YOLO returned no per-image result or timing")
        nms_ms = 0.0
        detections: list[Detection] = []
        allowed = set(allowed_class_names)
        for result in results:
            speed = getattr(result, "speed", None)
            if not isinstance(speed, Mapping):
                raise RuntimeError("YOLO result exposes no timing map")
            postprocess_ms = speed.get("postprocess")
            if (
                isinstance(postprocess_ms, bool)
                or not isinstance(postprocess_ms, (int, float))
                or not isfinite(postprocess_ms)
                or postprocess_ms < 0.0
            ):
                raise RuntimeError("YOLO result has invalid postprocess timing")
            nms_ms += float(postprocess_ms)
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(names[class_id])
                if allowed and class_name not in allowed:
                    continue
                xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(box.conf.item()),
                        bbox_xyxy=xyxy,
                    )
                )
        return ModelOutput(detections=tuple(detections), nms_ms=nms_ms)


def build_model(config: Mapping[str, Any]) -> YoloModel:
    if config.get("mock_mode", True):
        return MockYoloModel()
    return UltralyticsYoloModel(config)


def _ordered_model_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        try:
            keys = sorted(value)
        except TypeError as exc:
            raise RuntimeError("YOLO checkpoint class IDs must be sortable") from exc
        if keys != list(range(len(keys))):
            raise RuntimeError("YOLO checkpoint class IDs must be contiguous from zero")
        return tuple(str(value[key]) for key in keys)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    raise RuntimeError("YOLO checkpoint exposes no valid class map")
