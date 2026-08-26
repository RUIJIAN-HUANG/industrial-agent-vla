"""Dependency-light preflight for Member B's keyboard/RGB/CAS boundary.

Use this when the Isaac workstation cannot download pytest. It exercises the
same production contracts directly and exits non-zero on the first failure.
It does not launch Isaac Sim and does not replace the GUI smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from industrial_agent.image_cas import ImageCas, ImageCasConfig
from industrial_agent.observation import ObservationGateway
from industrial_agent.sync_contract import FROZEN_MULTI_RATE, STATE_7D_ORDER
from simulation.isaac_rgb_pipeline import build_camera_payload
from simulation.keyboard_teleop import KeyboardTeleopMapper
from simulation.rgb_cas_bridge import IsaacRgbCasPublisher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run keyboard/RGB/CAS checks without pytest or Isaac GUI."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "keyboard-teleop-preflight",
    )
    return parser.parse_args()


def _arm_state() -> dict[str, object]:
    return {
        "tcp_pose_m_rad": [0.0] * 6,
        "state": [0.0] * 6 + [1.0],
        "retreated": True,
        "gripper_open": True,
        "stationary": True,
    }


def main() -> int:
    args = _parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    result_path = artifact_dir / "result.json"
    result: dict[str, object]
    try:
        expected_order = (
            "x_m",
            "y_m",
            "z_m",
            "ax_rad",
            "ay_rad",
            "az_rad",
            "gripper_norm",
        )
        if STATE_7D_ORDER != expected_order:
            raise AssertionError("frozen state/action order drifted")
        frozen_counts = (
            FROZEN_MULTI_RATE.physics_ticks_per_model_step,
            FROZEN_MULTI_RATE.control_ticks_per_model_step,
            FROZEN_MULTI_RATE.render_frames_per_model_step,
        )
        if frozen_counts != (12, 6, 3):
            raise AssertionError("120/60/30/10Hz synchronization contract drifted")

        mapper = KeyboardTeleopMapper()
        expected_axes = {
            "w": (0, 0.005),
            "s": (0, -0.005),
            "a": (1, 0.005),
            "d": (1, -0.005),
            "q": (2, 0.005),
            "e": (2, -0.005),
        }
        for key, (axis, value) in expected_axes.items():
            action = mapper.parse(key).action
            if action is None or action.values[axis] != value:
                raise AssertionError(f"incorrect keyboard mapping for {key}")
            if action.duration_ms != 100 or len(action.values) != 7:
                raise AssertionError(f"invalid canonical action for {key}")
        gripper_close = mapper.parse("g").action
        gripper_open = mapper.parse("g").action
        if gripper_close is None or gripper_close.values[-1] != 0.0:
            raise AssertionError("gripper close must use endpoint 0.0")
        if gripper_open is None or gripper_open.values[-1] != 1.0:
            raise AssertionError("gripper open must use endpoint 1.0")

        camera_ids = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
        scene_config = {
            "cameras": [
                {"id": camera_id, "resolution_px": [1280, 720]}
                for camera_id in camera_ids
            ]
        }
        image_cas = ImageCas(ImageCasConfig(root=artifact_dir / "cas"))
        image_cas.assert_ready(writable=True)
        publisher = IsaacRgbCasPublisher.from_scene_config(image_cas, scene_config)
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        references = {
            camera_id: publisher.publish(camera_id, black_frame).to_dict()
            for camera_id in camera_ids
        }
        camera = build_camera_payload(references, "Arm_A")
        if camera["full_image"]["camera_id"] != "CAM_A_TOP":
            raise AssertionError("full_image did not select the active arm")
        if camera["wrist_image"] is not None:
            raise AssertionError("frozen scene must not invent a wrist camera")

        observation = {
            "observation_version": "1.0",
            "observation_id": "keyboard-preflight-000001",
            "timestamp_ms": int(time.time() * 1000),
            "camera": camera,
            "objects": [],
            "robot": {
                "active_arm": "Arm_A",
                "arm_a": _arm_state(),
                "arm_b": _arm_state(),
            },
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": None,
            },
            "task": {
                "packed_part_count": 0,
                "bin_at_handoff": False,
                "bin_at_finished": False,
                "bin_speed_m_s": 0.0,
                "status": "keyboard_teleop_preflight_v2",
                "v2_task_id": "P01_TO_S11",
                "v2_target_object_id": "P01",
                "v2_target_slot_id": "S11",
                "v2_terminal": False,
            },
            "quality": {"confidence": 1.0},
        }
        ObservationGateway().ingest_online(observation)
        result = {
            "status": "PASS",
            "pytest_required": False,
            "isaac_gui_launched": False,
            "keyboard_7d_contract": True,
            "multi_rate_contract": "120/60/30/10Hz",
            "three_rgb_cas_streams": True,
            "online_observation_validated": True,
            "next_step": "run one-action Isaac GUI smoke",
        }
    except BaseException as exc:
        result = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
