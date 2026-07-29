"""Pure Python route handlers for the OpenVLA-OFT service."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
)
from copy import deepcopy
from threading import Event, RLock
from time import perf_counter
from typing import Any, Mapping

from industrial_agent.errors import ImageCasError
from industrial_agent.service_images import CasRequestImageResolver

from .exceptions import ServiceError
from .handler import build_v1_infer_handler
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
        resolver = CasRequestImageResolver.from_agent_config(config)
        self._infer_handler = build_v1_infer_handler(
            resolver=resolver,
            backend=lambda request: request,
        )
        self.started_at = perf_counter()
        max_workers = int(config["api"]["max_concurrent_requests"])
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_by_task: dict[str, dict[str, Future[list[list[float]]]]] = {}
        self._cancel_event_by_request: dict[tuple[str, str], Event] = {}
        self._completed_by_request: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._completed_observation_by_task: OrderedDict[str, str] = OrderedDict()
        self._completed_cache_max_entries = int(
            config["api"]["completed_cache_max_entries"]
        )
        self._state_lock = RLock()

    def health(self) -> tuple[int, dict[str, Any]]:
        uptime_ms = int((perf_counter() - self.started_at) * 1000)
        status = "ready" if self.model.ready else "degraded"
        return 200, build_health_response(
            self.config,
            uptime_ms=uptime_ms,
            status=status,
        )

    def infer(self, payload: Any) -> tuple[int, dict[str, Any]]:
        artifacts = self.config["artifacts"]
        checkpoint_sha = artifacts["checkpoint_sha"]
        norm_stats_sha = artifacts["norm_stats_sha"]
        request_for_error = payload if isinstance(payload, Mapping) else {}
        request_id = payload.get("request_id") if isinstance(payload, Mapping) else None
        if isinstance(request_id, str):
            cached = self._get_completed_response(request_id)
            if cached is not None:
                return 200, cached

        try:
            request = validate_infer_request(payload, self.config)
            self._reject_stale_observation(request)
            if not self.model.ready:
                raise ServiceError(
                    "EXEC_2101_UNAVAILABLE",
                    "OpenVLA-OFT model is not ready",
                    retryable=False,
                )
            request = dict(self._infer_handler.handle(request))
            future, cancel_event = self._register_active(request)
            start = perf_counter()
            try:
                actions = future.result(timeout=request["deadline_ms"] / 1000)
            except TimeoutError as exc:
                self._cancel_request(request["task_id"], request["request_id"])
                future.cancel()
                raise ServiceError(
                    "EXEC_2102_TIMEOUT",
                    "OpenVLA-OFT inference exceeded deadline_ms",
                    retryable=False,
                ) from exc
            except CancelledError as exc:
                raise ServiceError(
                    "EXEC_2107_CANCELLED",
                    "OpenVLA-OFT inference was cancelled",
                    retryable=False,
                ) from exc
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError(
                    "EXEC_2104_RUNTIME",
                    "OpenVLA-OFT model inference failed",
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
            self._commit_completed(request, response, cancel_event)
            return 200, response
        except (ImageCasError, ServiceError) as exc:
            service_error = _as_service_error(exc)
            response = build_error_response(
                request_for_error,
                service_error,
                checkpoint_sha=checkpoint_sha,
                norm_stats_sha=norm_stats_sha,
            )
            return _http_status_for_error(service_error.code), response
        finally:
            if isinstance(payload, Mapping):
                self._clear_active(payload)

    def cancel(self, payload: Any) -> tuple[int, dict[str, Any]]:
        try:
            request = validate_cancel_request(payload)
            with self._state_lock:
                active = self._active_by_task.pop(request["task_id"], {})
                cancel_events = [
                    self._cancel_event_by_request.pop(
                        (request["task_id"], request_id),
                        None,
                    )
                    for request_id in active
                ]
                # Setting cancellation while holding the same lock used by
                # _commit_completed() is the linearization point: completion
                # can no longer commit success after cancel has won.
                for cancel_event in cancel_events:
                    if cancel_event is not None:
                        cancel_event.set()
                already_completed = (
                    request["task_id"] in self._completed_observation_by_task
                )
            if active:
                for future in active.values():
                    future.cancel()
                response = build_cancel_response(
                    request,
                    status="cancelled",
                    cancelled_request_ids=sorted(active),
                )
            elif already_completed:
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

    def _register_active(
        self,
        request: Mapping[str, Any],
    ) -> tuple[Future[list[list[float]]], Event]:
        max_concurrent = int(self.config["api"]["max_concurrent_requests"])
        with self._state_lock:
            active_total = sum(len(items) for items in self._active_by_task.values())
            if active_total >= max_concurrent:
                raise ServiceError(
                    "EXEC_2106_BACKPRESSURE",
                    "OpenVLA-OFT service is busy",
                    retryable=True,
                    retry_after_ms=int(self.config["api"]["default_deadline_ms"]),
                )
            cancel_event = Event()
            self._cancel_event_by_request[
                (request["task_id"], request["request_id"])
            ] = cancel_event
            future = self._executor.submit(
                self.model.predict,
                request,
                cancel_event=cancel_event,
            )
            self._active_by_task.setdefault(request["task_id"], {})[
                request["request_id"]
            ] = future
            return future, cancel_event

    def _cancel_request(self, task_id: str, request_id: str) -> None:
        with self._state_lock:
            event = self._cancel_event_by_request.pop((task_id, request_id), None)
        if event is not None:
            event.set()

    def _clear_active(self, request: Mapping[str, Any]) -> None:
        task_id = request.get("task_id")
        request_id = request.get("request_id")
        if not isinstance(task_id, str) or not isinstance(request_id, str):
            return
        with self._state_lock:
            active = self._active_by_task.get(task_id)
            if active is None:
                return
            active.pop(request_id, None)
            self._cancel_event_by_request.pop((task_id, request_id), None)
            if not active:
                self._active_by_task.pop(task_id, None)

    def _reject_stale_observation(self, request: Mapping[str, Any]) -> None:
        with self._state_lock:
            previous = self._completed_observation_by_task.get(request["task_id"])
        if previous == request["observation_id"]:
            raise ServiceError(
                "OBS_1101_INVALID",
                "fresh Arm_B observation is required for a new OpenVLA-OFT inference",
                retryable=True,
            )

    def _get_completed_response(self, request_id: str) -> dict[str, Any] | None:
        with self._state_lock:
            response = self._completed_by_request.get(request_id)
            if response is None:
                return None
            self._completed_by_request.move_to_end(request_id)
            return deepcopy(response)

    def _commit_completed(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        cancel_event: Event,
    ) -> None:
        request_id = str(request["request_id"])
        task_id = str(request["task_id"])
        observation_id = str(request["observation_id"])
        with self._state_lock:
            if cancel_event.is_set():
                raise ServiceError(
                    "EXEC_2107_CANCELLED",
                    "OpenVLA-OFT inference was cancelled before completion",
                    retryable=False,
                )
            self._completed_by_request[request_id] = deepcopy(dict(response))
            self._completed_by_request.move_to_end(request_id)
            self._completed_observation_by_task[task_id] = observation_id
            self._completed_observation_by_task.move_to_end(task_id)
            active = self._active_by_task.get(task_id)
            if active is not None:
                active.pop(request_id, None)
                if not active:
                    self._active_by_task.pop(task_id, None)
            self._cancel_event_by_request.pop((task_id, request_id), None)
            while len(self._completed_by_request) > self._completed_cache_max_entries:
                self._completed_by_request.popitem(last=False)
            while (
                len(self._completed_observation_by_task)
                > self._completed_cache_max_entries
            ):
                self._completed_observation_by_task.popitem(last=False)


def _as_service_error(error: ImageCasError | ServiceError) -> ServiceError:
    if isinstance(error, ServiceError):
        return error
    return ServiceError(
        error.code.value,
        str(error),
        retryable=error.retryable,
    )


def _http_status_for_error(code: str) -> int:
    if code == "EXEC_2106_BACKPRESSURE":
        return 429
    if code == "EXEC_2101_UNAVAILABLE":
        return 503
    if code == "CAS_1306_UNAVAILABLE":
        return 503
    if code == "EXEC_2102_TIMEOUT":
        return 504
    if code == "EXEC_2107_CANCELLED":
        return 409
    if code.startswith("SAFE_"):
        return 409
    return 422
