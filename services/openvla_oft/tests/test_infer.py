from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import numpy as np


def test_infer_valid_request_returns_schema_compliant_action_chunk(
    service,
    valid_infer_request,
):
    status, response = service.infer(valid_infer_request)

    assert status == 200
    assert response["status"] == "ok"
    assert response["executor"] == "openvla_oft"
    assert response["action_chunk"]["executor"] == "openvla_oft"
    assert response["action_chunk"]["task_id"] == valid_infer_request["task_id"]
    assert response["action_chunk"]["action_space"] == "ee_delta_pose_gripper"
    assert response["action_chunk"]["frame"] == "robot_base"
    for step in response["action_chunk"]["steps"]:
        assert len(step["values"]) == 7


def test_infer_null_wrist_image_is_accepted(
    service,
    valid_infer_request,
):
    status, response = service.infer(valid_infer_request)

    assert status == 200
    assert response["status"] == "ok"


def test_infer_non_null_wrist_image_is_rejected(
    service,
    valid_infer_request,
    copy_request,
):
    request = copy_request(valid_infer_request)
    request["model_input"]["wrist_image"] = dict(
        request["model_input"]["full_image"],
        camera_id="CAM_B_WRIST",
    )

    status, response = service.infer(request)

    assert status == 422
    assert response["status"] == "error"
    assert response["error"]["code"] == "CAS_1304_METADATA_MISMATCH"
    assert response["error"]["retryable"] is False


def test_infer_shared_handler_replaces_reference_with_verified_pixels(
    config,
    valid_infer_request,
):
    class RecordingModel:
        ready = True

        def __init__(self):
            self.request = None

        def predict(self, request, cancel_event=None):
            del cancel_event
            self.request = request
            return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    from openvla_oft.routes import OpenVLAOFTService

    model = RecordingModel()
    service = OpenVLAOFTService(config, model=model)
    status, response = service.infer(valid_infer_request)

    assert status == 200
    assert response["status"] == "ok"
    full_image = model.request["model_input"]["full_image"]
    assert isinstance(full_image, np.ndarray)
    assert full_image.shape == (720, 1280, 3)
    assert full_image.flags.writeable is False
    assert "full_image_rgb" not in model.request["model_input"]


def test_infer_missing_cas_blob_fails_before_model_call(
    config,
    valid_infer_request,
    copy_request,
):
    class RecordingModel:
        ready = True

        def __init__(self):
            self.calls = 0

        def predict(self, request, cancel_event=None):
            del request, cancel_event
            self.calls += 1
            return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    from openvla_oft.routes import OpenVLAOFTService

    request = copy_request(valid_infer_request)
    missing_digest = "a" * 64
    request["model_input"]["full_image"]["uri"] = f"cas://sha256/{missing_digest}"
    request["model_input"]["full_image"]["image_sha256"] = f"sha256:{missing_digest}"
    model = RecordingModel()
    service = OpenVLAOFTService(config, model=model)

    status, response = service.infer(request)

    assert status == 422
    assert response["error"]["code"] == "CAS_1301_NOT_FOUND"
    assert response["error"]["retryable"] is True
    assert model.calls == 0


def test_model_runtime_failure_returns_stable_executor_error(
    config,
    valid_infer_request,
):
    class FailingModel:
        ready = True

        def predict(self, request, cancel_event=None):
            del request, cancel_event
            raise RuntimeError("CUDA out of memory")

    from openvla_oft.routes import OpenVLAOFTService

    service = OpenVLAOFTService(config, model=FailingModel())

    status, response = service.infer(valid_infer_request)

    assert status == 422
    assert response["error"]["code"] == "EXEC_2104_RUNTIME"
    assert response["error"]["retryable"] is False


def test_cas_dependency_unavailable_maps_to_http_503():
    from openvla_oft.routes import _http_status_for_error

    assert _http_status_for_error("CAS_1306_UNAVAILABLE") == 503


def test_infer_duplicate_request_id_returns_cached_response(
    service,
    valid_infer_request,
):
    _, first = service.infer(valid_infer_request)
    _, second = service.infer(valid_infer_request)

    assert second == first


def test_infer_reused_observation_with_new_request_is_rejected(
    service,
    valid_infer_request,
    copy_request,
):
    service.infer(valid_infer_request)
    second = copy_request(valid_infer_request)
    second["request_id"] = "req-2"

    status, response = service.infer(second)

    assert status == 422
    assert response["status"] == "error"
    assert response["error"]["code"] == "OBS_1101_INVALID"
    assert response["error"]["retryable"] is True


def test_infer_wrong_camera_is_rejected(service, valid_infer_request, copy_request):
    request = copy_request(valid_infer_request)
    request["model_input"]["full_image"]["camera_id"] = "CAM_A_TOP"

    status, response = service.infer(request)

    assert status == 422
    assert response["status"] == "error"
    assert response["error"]["code"] == "CAS_1304_METADATA_MISMATCH"
    assert "action_chunk" not in response


def test_infer_wrong_phase_is_rejected(service, valid_infer_request, copy_request):
    request = copy_request(valid_infer_request)
    request["subtask_id"] = "S01_ARM_A_PACK_HANDOFF"

    status, response = service.infer(request)

    assert status == 409
    assert response["status"] == "error"
    assert response["error"]["code"] == "SAFE_9004_ACTION_REJECTED"


def test_infer_mismatched_checkpoint_is_rejected(
    service,
    valid_infer_request,
    copy_request,
):
    request = copy_request(valid_infer_request)
    request["checkpoint_sha"] = f"sha256:{'f' * 64}"

    status, response = service.infer(request)

    assert status == 422
    assert response["error"]["code"] == "EXEC_2105_MODEL_REVISION_MISMATCH"


def test_completed_response_caches_are_bounded(
    config,
    valid_infer_request,
    copy_request,
):
    from openvla_oft.routes import OpenVLAOFTService

    bounded_config = deepcopy(config)
    bounded_config["api"]["completed_cache_max_entries"] = 2
    service = OpenVLAOFTService(bounded_config)

    for index in range(3):
        request = copy_request(valid_infer_request)
        request["request_id"] = f"req-{index}"
        request["task_id"] = f"task-{index}:S02_ARM_B_TRANSPORT"
        request["observation_id"] = f"obs-{index}"
        status, _ = service.infer(request)
        assert status == 200

    assert list(service._completed_by_request) == ["req-1", "req-2"]
    assert list(service._completed_observation_by_task) == [
        "task-1:S02_ARM_B_TRANSPORT",
        "task-2:S02_ARM_B_TRANSPORT",
    ]


def test_concurrent_request_is_rejected_by_atomic_backpressure_gate(
    config,
    valid_infer_request,
    copy_request,
):
    class BlockingModel:
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

    model = BlockingModel()
    service = OpenVLAOFTService(config, model=model)
    second = copy_request(valid_infer_request)
    second["request_id"] = "req-concurrent"
    second["task_id"] = "task-concurrent:S02_ARM_B_TRANSPORT"
    second["observation_id"] = "obs-concurrent"

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(service.infer, valid_infer_request)
        assert model.entered.wait(timeout=2)
        try:
            status, response = service.infer(second)
            assert status == 429
            assert response["error"]["code"] == "EXEC_2106_BACKPRESSURE"
        finally:
            model.release.set()
        assert first.result(timeout=2)[0] == 200
