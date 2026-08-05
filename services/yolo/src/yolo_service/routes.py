"""Pure Python route handlers for the YOLO service."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from math import isfinite
from threading import RLock
from time import perf_counter
from typing import Any, Mapping

from industrial_agent.errors import ImageCasError
from industrial_agent.service_images import CasRequestImageResolver

from .exceptions import ServiceError
from .handler import build_v1_detect_handler
from .model import Detection, ModelOutput, YoloModel, build_model
from .schemas import (
    build_cancel_response,
    build_error_response,
    build_health_response,
    build_success_response,
    validate_cancel_request,
    validate_detect_request,
)


class YoloService:
    def __init__(
        self,
        config: Mapping[str, Any],
        model: YoloModel | None = None,
    ) -> None:
        self.config = config
        self.model = model or build_model(config)
        resolver = CasRequestImageResolver.from_agent_config(config)
        self._detect_handler = build_v1_detect_handler(
            resolver=resolver,
            backend=lambda request: request,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=int(config["api"]["max_concurrent_requests"])
        )
        self._active_by_task: dict[str, dict[str, Future[ModelOutput]]] = {}
        self._state_lock = RLock()
        self._quarantined = False
        self._quarantine_reason: str | None = None

    def health(self) -> tuple[int, dict[str, Any]]:
        with self._state_lock:
            quarantined = self._quarantined
            reason = self._quarantine_reason
        response = build_health_response(self.config)
        if quarantined:
            response["status"] = "degraded"
            response["device"] = {
                **response["device"],
                "quarantined": True,
                "quarantine_reason": reason,
            }
        return 200, response

    def detect(self, payload: Any) -> tuple[int, dict[str, Any]]:
        request_for_error = payload if isinstance(payload, Mapping) else {}
        with self._state_lock:
            quarantined = self._quarantined
        if quarantined:
            error = ServiceError(
                "PERC_2201_UNAVAILABLE",
                "YOLO instance is quarantined until service restart",
                retryable=False,
            )
            return 503, build_error_response(
                request_for_error,
                error,
                self.config,
            )
        try:
            request = validate_detect_request(payload, self.config)
            image_reference = dict(request["image"])
            start_total = perf_counter()
            start_preprocess = perf_counter()
            prepared = dict(self._detect_handler.handle(request))
            preprocess_ms = (perf_counter() - start_preprocess) * 1000
            image = prepared["image"]
            start_inference = perf_counter()
            future = self._submit(request, image)
            try:
                output = future.result(timeout=request["deadline_ms"] / 1000)
            except TimeoutError as exc:
                self._enter_quarantine("hard inference timeout")
                future.cancel()
                raise ServiceError(
                    "PERC_2202_TIMEOUT",
                    "YOLO inference exceeded deadline_ms",
                    retryable=False,
                ) from exc
            inference_ms = (perf_counter() - start_inference) * 1000
            prepared["image_reference"] = image_reference
            timing = {
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "nms_ms": output.nms_ms,
                "total_ms": (perf_counter() - start_total) * 1000,
            }
            return 200, build_success_response(
                prepared,
                [
                    _detection_to_dict(
                        item,
                        image_reference,
                        index,
                        class_names=self.config["class_names"],
                    )
                    for index, item in enumerate(output.detections)
                ],
                timing,
            )
        except (ImageCasError, ServiceError) as exc:
            error = _as_service_error(exc)
            return _http_status(error.code), build_error_response(
                request_for_error,
                error,
                self.config,
            )
        except Exception:
            error = ServiceError(
                "PERC_2201_UNAVAILABLE",
                "YOLO model inference failed",
                retryable=False,
            )
            return 503, build_error_response(
                request_for_error,
                error,
                self.config,
            )

    def cancel(self, payload: Any) -> tuple[int, dict[str, Any]]:
        try:
            request = validate_cancel_request(payload)
            with self._state_lock:
                active = self._active_by_task.pop(request["task_id"], {})
            for future in active.values():
                future.cancel()
            return 200, build_cancel_response(
                request,
                status="cancelled" if active else "not_found",
                cancelled_request_ids=sorted(active),
            )
        except ServiceError as exc:
            return 422, build_error_response(
                payload if isinstance(payload, Mapping) else {},
                exc,
                self.config,
            )

    def _submit(
        self,
        request: Mapping[str, Any],
        image: Any,
    ) -> Future[ModelOutput]:
        with self._state_lock:
            if self._quarantined:
                raise ServiceError(
                    "PERC_2201_UNAVAILABLE",
                    "YOLO instance is quarantined until service restart",
                    retryable=False,
                )
            active_total = sum(len(items) for items in self._active_by_task.values())
            maximum = int(self.config["api"]["max_concurrent_requests"])
            if active_total >= maximum:
                raise ServiceError(
                    "PERC_2201_UNAVAILABLE",
                    "YOLO service is busy",
                    retryable=True,
                )
            future = self._executor.submit(
                self.model.detect,
                image,
                allowed_class_names=request["allowed_class_names"],
                confidence=float(request["thresholds"]["confidence"]),
                iou=float(request["thresholds"]["iou"]),
            )
            self._active_by_task.setdefault(request["task_id"], {})[
                request["request_id"]
            ] = future
            future.add_done_callback(
                lambda completed: self._clear_active(
                    request["task_id"],
                    request["request_id"],
                    completed,
                )
            )
            return future

    def _clear_active(
        self,
        task_id: str,
        request_id: str,
        completed: Future[ModelOutput],
    ) -> None:
        with self._state_lock:
            active = self._active_by_task.get(task_id)
            if active is None:
                return
            if active.get(request_id) is completed:
                active.pop(request_id, None)
            if not active:
                self._active_by_task.pop(task_id, None)

    def _enter_quarantine(self, reason: str) -> None:
        with self._state_lock:
            self._quarantined = True
            self._quarantine_reason = reason

    def close(self) -> None:
        """Reject queued work and release the inference worker."""

        with self._state_lock:
            active = self._active_by_task
            self._active_by_task = {}
        for requests in active.values():
            for future in requests.values():
                future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


def _detection_to_dict(
    detection: Detection,
    image: Mapping[str, Any],
    index: int,
    *,
    class_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    if isinstance(detection.class_id, bool) or not isinstance(detection.class_id, int):
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "model returned a non-integer class_id",
        )
    if not isinstance(detection.class_name, str) or not detection.class_name:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "model returned an invalid class_name",
        )
    if detection.class_id >= len(class_names) or detection.class_id < 0:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "model returned an out-of-range class_id",
        )
    if class_names[detection.class_id] != detection.class_name:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "model returned a class_id/class_name mismatch",
        )
    if not isfinite(detection.confidence) or not 0.0 <= detection.confidence <= 1.0:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "model returned an invalid confidence",
        )
    x1, y1, x2, y2 = detection.bbox_xyxy
    width = int(image["width"])
    height = int(image["height"])
    if not all(isfinite(value) for value in (x1, y1, x2, y2)) or not (
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    ):
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "model returned an out-of-bounds detection",
        )
    return {
        "detection_id": f"det-{index}",
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_format": "xyxy_pixels",
        "camera_id": image["camera_id"],
        "image_width": width,
        "image_height": height,
        "attributes": {},
    }


def _as_service_error(error: ImageCasError | ServiceError) -> ServiceError:
    if isinstance(error, ServiceError):
        return error
    return ServiceError(error.code.value, str(error), retryable=error.retryable)


def _http_status(code: str) -> int:
    if code in {"PERC_2201_UNAVAILABLE", "CAS_1306_UNAVAILABLE"}:
        return 503
    if code == "PERC_2202_TIMEOUT":
        return 504
    return 422
