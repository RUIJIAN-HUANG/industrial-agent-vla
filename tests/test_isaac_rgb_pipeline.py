import pytest

from simulation.isaac_rgb_pipeline import build_camera_payload


def _references() -> dict[str, dict[str, object]]:
    return {
        camera_id: {
            "image_id": camera_id,
            "camera_id": camera_id,
            "width": 1280,
            "height": 720,
            "encoding": "rgb8",
            "uri": f"cas://sha256/{camera_id}",
            "sha256": "0" * 64,
        }
        for camera_id in ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
    }


def test_camera_payload_contains_frozen_streams_and_no_wrist_camera() -> None:
    payload = build_camera_payload(_references(), "Arm_A")

    assert payload["full_image"]["camera_id"] == "CAM_A_TOP"
    assert payload["arm_a_rgb"]["camera_id"] == "CAM_A_TOP"
    assert payload["handoff_rgb"]["camera_id"] == "CAM_HANDOFF"
    assert payload["arm_b_rgb"]["camera_id"] == "CAM_B_TOP"
    assert payload["wrist_image"] is None


def test_camera_payload_selects_active_arm_and_rejects_missing_stream() -> None:
    assert build_camera_payload(_references(), "Arm_B")["full_image"][
        "camera_id"
    ] == "CAM_B_TOP"
    incomplete = _references()
    incomplete.pop("CAM_HANDOFF")
    with pytest.raises(ValueError, match="CAM_HANDOFF"):
        build_camera_payload(incomplete, "Arm_A")
