from __future__ import annotations


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


def test_infer_optional_wrist_image_is_resolved(
    service,
    valid_infer_request,
    valid_wrist_image,
    copy_request,
):
    request = copy_request(valid_infer_request)
    request["model_input"]["wrist_image"] = valid_wrist_image

    status, response = service.infer(request)

    assert status == 200
    assert response["status"] == "ok"


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
    assert response["error"]["code"] == "OBS_1101_INVALID"
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
