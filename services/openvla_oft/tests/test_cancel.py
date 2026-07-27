from __future__ import annotations

from concurrent.futures import Future


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


def test_cancel_no_active_request_returns_not_found(service, schema_validator):
    status, response = service.cancel(_cancel_request())

    assert status == 200
    schema_validator("executor-cancel.schema.json", "response").validate(response)
    assert response["status"] == "not_found"
    assert response["cancelled_request_ids"] == []
    assert response["server_context_cleared"] is True


def test_cancel_active_request_returns_cancelled(service, schema_validator):
    future: Future[list[list[float]]] = Future()
    service._active_by_task["task-1"] = {"req-1": future}

    status, response = service.cancel(_cancel_request())

    assert status == 200
    schema_validator("executor-cancel.schema.json", "response").validate(response)
    assert response["status"] == "cancelled"
    assert response["cancelled_request_ids"] == ["req-1"]
    assert "task-1" not in service._active_by_task


def test_cancel_completed_task_is_idempotent(
    service,
    valid_infer_request,
    schema_validator,
):
    service.infer(valid_infer_request)

    status, response = service.cancel(_cancel_request(valid_infer_request["task_id"]))

    assert status == 200
    schema_validator("executor-cancel.schema.json", "response").validate(response)
    assert response["status"] == "already_completed"
    assert response["cancelled_request_ids"] == []
