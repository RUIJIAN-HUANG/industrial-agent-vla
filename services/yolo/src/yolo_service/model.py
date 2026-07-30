"""YOLO model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One model detection in pixel-space xyxy format."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class YoloModel(Protocol):
    """Interface shared by mock and real YOLO implementations."""

    def detect(
        self,
        image: np.ndarray,
        *,
        allowed_class_names: Sequence[str],
        confidence: float,
        iou: float,
    ) -> list[Detection]:
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
    ) -> list[Detection]:
        del allowed_class_names, confidence, iou

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")

        return []


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

    def detect(
        self,
        image: np.ndarray,
        *,
        allowed_class_names: Sequence[str],
        confidence: float,
        iou: float,
    ) -> list[Detection]:
        results = self._model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        allowed = set(allowed_class_names)
        for result in results:
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
        return detections


def build_model(config: Mapping[str, Any]) -> YoloModel:
    if config.get("mock_mode", True):
        return MockYoloModel()
    return UltralyticsYoloModel(config)
