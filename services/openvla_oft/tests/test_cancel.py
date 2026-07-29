from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from time import sleep

from openvla_oft.exceptions import ServiceError


def _cancel_request(task_id: str = "task-1") -> dict[str, str]:
    return {
        "schema_version": "1.0",
        "request_id": "cancel-1",
        "trace_id": "run-1",
        "episode_id": "run-1",
        "task_id": task_id,
        "subtask_id": "S02_ARM_B_TRANSPORT",
        "reason": "operator requested cancellation",
    }


def test_cancel_no_active_request_returns_not_found(service):
    status, response = service.cancel(_cancel_request())

    assert status == 200
    assert response["status"] == "not_found"
    assert response["cancelled_request_ids"] == []
    assert response["server_context_cleared"] is True


def test_cancel_active_request_returns_cancelled(service):
    future: Future[list[list[float]]] = Future()
    service._active_by_task["task-1"] = {"req-1": future}

    status, response = service.cancel(_cancel_request())

    assert status == 200
    assert response["status"] == "cancelled"
    assert response["cancelled_request_ids"] == ["req-1"]
    assert "task-1" not in service._active_by_task


def test_cancel_completed_task_is_idempotent(
    service,
    valid_infer_request,
):
    service.infer(valid_infer_request)

    status, response = service.cancel(_cancel_request(valid_infer_request["task_id"]))

    assert status == 200
    assert response["status"] == "already_completed"
    assert response["cancelled_request_ids"] == []


def test_cancel_cooperative_infer_stops_running_request(
    config,
    valid_infer_request,
):
    class CooperativeModel:
        ready = True

        def predict(self, request, cancel_event=None):
            del request
            while cancel_event is not None and not cancel_event.is_set():
                sleep(0.01)
            raise ServiceError(
                "EXEC_2107_CANCELLED",
                "request cancelled cooperatively",
            )

    from openvla_oft.routes import OpenVLAOFTService

    service = OpenVLAOFTService(config, model=CooperativeModel())

    with ThreadPoolExecutor(max_workers=1) as executor:
        infer_future = executor.submit(service.infer, valid_infer_request)
        while not service._active_by_task:
            sleep(0.01)

        status, response = service.cancel(
            _cancel_request(valid_infer_request["task_id"])
        )
        assert status == 200
        assert response["status"] == "cancelled"

        infer_status, infer_response = infer_future.result(timeout=2)
        assert infer_status == 409
        assert infer_response["status"] == "error"
        assert infer_response["error"]["code"] == "EXEC_2107_CANCELLED"


def test_cancel_wins_when_model_ignores_cancel_event(
    config,
    valid_infer_request,
):
    class NonCooperativeModel:
        ready = True

        def __init__(self):
            self.entered = Event()
            self.release = Event()

        def predict(self, request, cancel_event=None):
            del request, cancel_event
            self.entered.set()
            assert self.release.wait(timeout=2)
            return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    from openvla_oft.routes import OpenVLAOFTService

    model = NonCooperativeModel()
    service = OpenVLAOFTService(config, model=model)

    with ThreadPoolExecutor(max_workers=1) as executor:
        infer_future = executor.submit(service.infer, valid_infer_request)
        assert model.entered.wait(timeout=2)

        status, response = service.cancel(
            _cancel_request(valid_infer_request["task_id"])
        )
        assert status == 200
        assert response["status"] == "cancelled"
        model.release.set()

        infer_status, infer_response = infer_future.result(timeout=2)

    assert infer_status == 409
    assert infer_response["error"]["code"] == "EXEC_2107_CANCELLED"
    assert valid_infer_request["request_id"] not in service._completed_by_request
