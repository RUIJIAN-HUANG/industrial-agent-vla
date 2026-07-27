"""Pure Python route handlers for the OpenVLA-OFT service."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from time import perf_counter
from typing import Any, Mapping

from .exceptions import ServiceError
from .model import OpenVLAOFTModel
from .schemas import (
    build_cancel_response,
    build_error_response,
    build_health_response,
    build_success_response,
    validate_cancel_request,
    validate_infer_request,
)


class OpenVLAOFTService:
    def __init__(
        self,
        config: Mapping[str, Any],
        model: OpenVLAOFTModel | None = None,
    ) -> None:
        self.config = config
        self.model = model or OpenVLAOFTModel(config)
        self.started_at = perf_counter()
        max_workers = int(config["api"]["max_concurrent_requests"])
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_by_task: dict[str, dict[str, Future[list[list[float]]]]] = {}
        self._completed_by_request: dict[str, dict[str, Any]] = {}
        self._completed_observation_by_task: dict[str, str] = {}

    def health(self) -> tuple[int, dict[str, Any]]:
        uptime_ms = int((perf_counter() - self.started_at) * 1000)
        return 200, build_health_response(self.config, uptime_ms=uptime_ms)

    def infer(self, payload: Any) -> tuple[int, dict[str, Any]]:
        artifacts = self.config["artifacts"]
        checkpoint_sha = artifacts["checkpoint_sha"]
        norm_stats_sha = artifacts["norm_stats_sha"]
        request_for_error = payload if isinstance(payload, Mapping) else {}
        request_id = payload.get("request_id") if isinstance(payload, Mapping) else None
        if isinstance(request_id, str) and request_id in self._completed_by_request:
            return 200, self._completed_by_request[request_id]

        try:
            request = validate_infer_request(payload, self.config)
            self._reject_stale_observation(request)
            self._register_active(request)
            start = perf_counter()
            future = self._active_by_task[request["task_id"]][request["request_id"]]
            try:
                actions = future.result(timeout=request["deadline_ms"] / 1000)
            except TimeoutError as exc:
                future.cancel()
                raise ServiceError(
                    "EXEC_2102_TIMEOUT",
                    "OpenVLA-OFT inference exceeded deadline_ms",
                    retryable=False,
                ) from exc
            inference_ms = (perf_counter() - start) * 1000
            response = build_success_response(
                request,
                actions,
                checkpoint_sha=checkpoint_sha,
                norm_stats_sha=norm_stats_sha,
                inference_ms=inference_ms,
            )
            self._completed_by_request[request["request_id"]] = response
            self._completed_observation_by_task[request["task_id"]] = request[
                "observation_id"
            ]
            return 200, response
        except ServiceError as exc:
            response = build_error_response(
                request_for_error,
                exc,
                checkpoint_sha=checkpoint_sha,
                norm_stats_sha=norm_stats_sha,
            )
            return _http_status_for_error(exc.code), response
        finally:
            if isinstance(payload, Mapping):
                self._clear_active(payload)

    def cancel(self, payload: Any) -> tuple[int, dict[str, Any]]:
        try:
            request = validate_cancel_request(payload)
            active = self._active_by_task.pop(request["task_id"], {})
            if active:
                for future in active.values():
                    future.cancel()
                response = build_cancel_response(
                    request,
                    status="cancelled",
                    cancelled_request_ids=sorted(active),
                )
            elif request["task_id"] in self._completed_observation_by_task:
                response = build_cancel_response(
                    request,
                    status="already_completed",
                    cancelled_request_ids=[],
                )
            else:
                response = build_cancel_response(
                    request,
                    status="not_found",
                    cancelled_request_ids=[],
                )
            return 200, response
        except ServiceError as exc:
            request = payload if isinstance(payload, Mapping) else {}
            artifacts = self.config["artifacts"]
            return 422, build_error_response(
                request,
                exc,
                checkpoint_sha=artifacts["checkpoint_sha"],
                norm_stats_sha=artifacts["norm_stats_sha"],
                status="cancelled",
            )

    def _register_active(self, request: Mapping[str, Any]) -> None:
        max_concurrent = int(self.config["api"]["max_concurrent_requests"])
        active_total = sum(len(items) for items in self._active_by_task.values())
        if active_total >= max_concurrent:
            raise ServiceError(
                "EXEC_2106_BACKPRESSURE",
                "OpenVLA-OFT service is busy",
                retryable=True,
                retry_after_ms=int(self.config["api"]["default_deadline_ms"]),
            )
        future = self._executor.submit(self.model.predict, request)
        self._active_by_task.setdefault(request["task_id"], {})[
            request["request_id"]
        ] = future

    def _clear_active(self, request: Mapping[str, Any]) -> None:
        task_id = request.get("task_id")
        request_id = request.get("request_id")
        if not isinstance(task_id, str) or not isinstance(request_id, str):
            return
        active = self._active_by_task.get(task_id)
        if active is None:
            return
        active.pop(request_id, None)
        if not active:
            self._active_by_task.pop(task_id, None)

    def _reject_stale_observation(self, request: Mapping[str, Any]) -> None:
        previous = self._completed_observation_by_task.get(request["task_id"])
        if previous == request["observation_id"]:
            raise ServiceError(
                "OBS_1101_INVALID",
                "fresh Arm_B observation is required for a new OpenVLA-OFT inference",
                retryable=True,
            )


def _http_status_for_error(code: str) -> int:
    if code == "EXEC_2106_BACKPRESSURE":
        return 429
    if code == "EXEC_2101_UNAVAILABLE":
        return 503
    if code == "EXEC_2102_TIMEOUT":
        return 504
    if code.startswith("SAFE_"):
        return 409
    return 422
