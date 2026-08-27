from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from simulation.run_pi05_isaac_closed_loop import (
    _capture_stable_observation_inputs,
    _parse_args,
    _pause_physics_world,
    _safety_policy_from_config,
    _update_ui_without_advancing_physics,
    build_observation,
    build_task_state,
)
from industrial_agent.v2_observation import V2ObservationGateway
from industrial_agent.v2_task_profile import require_formal_v2_task


def test_runtime_mode_defaults_to_direct_and_accepts_supervisor() -> None:
    assert _parse_args([]).runtime_mode == "direct"
    assert _parse_args(["--runtime-mode", "supervisor"]).runtime_mode == "supervisor"


def test_runtime_mode_rejects_unknown_value() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--runtime-mode", "unknown"])


def test_isaac_runtime_imports_are_deferred_until_after_kit_startup() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "simulation"
        / "run_pi05_isaac_closed_loop.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_closed_loop"
    )

    launch_line = next(
        node.lineno
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch_simulation_app"
    )
    runtime_modules = {
        "simulation.isaac_rgb_pipeline",
        "simulation.rgb_cas_bridge",
        "simulation.run_isaac_adapter_smoke",
        "simulation.run_v2_keyboard_collection",
        "simulation.single_bin_scene_v2_builder",
        "isaac_franka_controller",
        "isaacsim.core.api",
        "isaacsim.core.prims",
    }

    runtime_imports = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.ImportFrom) and node.module in runtime_modules
    ]

    assert {node.module for node in runtime_imports} == runtime_modules
    assert all(node.lineno > launch_line for node in runtime_imports)

    try_block = next(node for node in ast.walk(runner) if isinstance(node, ast.Try))
    try_body_lines = {node.lineno for node in try_block.body if hasattr(node, "lineno")}
    assert all(node.lineno in try_body_lines for node in runtime_imports)

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "simulation_app"
        for final_node in try_block.finalbody
        for node in ast.walk(final_node)
    )


def test_idle_update_pauses_playing_world_before_update() -> None:
    events: list[str] = []
    world = Mock()
    world.is_playing.return_value = True
    world.pause.side_effect = lambda: events.append("pause")
    simulation_app = Mock()
    simulation_app.update.side_effect = lambda: events.append("update")

    _update_ui_without_advancing_physics(
        world=world,
        simulation_app=simulation_app,
    )

    assert events == ["pause", "update"]


def test_idle_update_does_not_repause_paused_world() -> None:
    world = Mock()
    world.is_playing.return_value = False
    simulation_app = Mock()

    _update_ui_without_advancing_physics(
        world=world,
        simulation_app=simulation_app,
    )

    world.pause.assert_not_called()
    simulation_app.update.assert_called_once_with()


def test_idle_update_propagates_pause_failure_without_update() -> None:
    world = Mock()
    world.is_playing.return_value = True
    world.pause.side_effect = RuntimeError("pause failed")
    simulation_app = Mock()

    with pytest.raises(RuntimeError, match="pause failed"):
        _update_ui_without_advancing_physics(
            world=world,
            simulation_app=simulation_app,
        )

    simulation_app.update.assert_not_called()


def test_pause_physics_pauses_playing_world() -> None:
    events: list[str] = []
    world = Mock()
    world.is_playing.return_value = True
    world.pause.side_effect = lambda: events.append("pause")

    _pause_physics_world(world=world)

    assert events == ["pause"]
    world.pause.assert_called_once_with()


def test_pause_physics_does_not_repause_paused_world() -> None:
    world = Mock()
    world.is_playing.return_value = False

    _pause_physics_world(world=world)

    world.pause.assert_not_called()


def test_stable_observation_pause_failure_stops_capture() -> None:
    world = Mock()
    world.is_playing.return_value = True
    world.pause.side_effect = RuntimeError("pause failed")
    capture_camera = Mock()
    capture_state = Mock()

    with pytest.raises(RuntimeError, match="pause failed"):
        _capture_stable_observation_inputs(
            world=world,
            capture_camera=capture_camera,
            capture_state=capture_state,
        )

    capture_camera.assert_not_called()
    capture_state.assert_not_called()


def test_stable_observation_captures_camera_then_state() -> None:
    events: list[str] = []
    world = Mock()
    world.is_playing.side_effect = (True, False, False)
    world.pause.side_effect = lambda: events.append("pause")
    capture_camera = Mock(side_effect=lambda: events.append("camera") or {"rgb": 1})
    capture_state = Mock(side_effect=lambda: events.append("state") or {"robot": 2})

    camera, state = _capture_stable_observation_inputs(
        world=world,
        capture_camera=capture_camera,
        capture_state=capture_state,
    )

    assert events == ["pause", "camera", "state"]
    assert camera == {"rgb": 1}
    assert state == {"robot": 2}


def test_stable_observation_pauses_again_after_camera_restarts_world() -> None:
    events: list[str] = []
    world = Mock()
    playing = iter((True, True, False))
    world.is_playing.side_effect = lambda: next(playing)
    world.pause.side_effect = lambda: events.append("pause")
    capture_camera = Mock(side_effect=lambda: events.append("camera") or {"rgb": 1})
    capture_state = Mock(side_effect=lambda: events.append("state") or {"robot": 2})

    _capture_stable_observation_inputs(
        world=world,
        capture_camera=capture_camera,
        capture_state=capture_state,
    )

    assert events == ["pause", "camera", "pause", "state"]
    assert world.pause.call_count == 2


def test_stable_observation_camera_failure_propagates_without_state() -> None:
    world = Mock()
    world.is_playing.return_value = False
    capture_camera = Mock(side_effect=OSError("camera failed"))
    capture_state = Mock()

    with pytest.raises(OSError, match="camera failed"):
        _capture_stable_observation_inputs(
            world=world,
            capture_camera=capture_camera,
            capture_state=capture_state,
        )

    capture_state.assert_not_called()


def test_stable_observation_state_failure_does_not_return_partial_observation() -> None:
    world = Mock()
    world.is_playing.return_value = False
    capture_camera = Mock(return_value={"rgb": 1})
    capture_state = Mock(side_effect=TimeoutError("state failed"))

    with pytest.raises(TimeoutError, match="state failed"):
        _capture_stable_observation_inputs(
            world=world,
            capture_camera=capture_camera,
            capture_state=capture_state,
        )


def test_stable_observation_rejects_world_playing_after_state_capture() -> None:
    world = Mock()
    world.is_playing.side_effect = (False, True, True)
    capture_camera = Mock(return_value={"rgb": 1})
    capture_state = Mock(return_value={"robot": 2})

    with pytest.raises(RuntimeError, match="physics resumed"):
        _capture_stable_observation_inputs(
            world=world,
            capture_camera=capture_camera,
            capture_state=capture_state,
        )

    capture_state.assert_called_once_with()


def test_build_task_state_contains_no_ground_truth_fields() -> None:
    task = build_task_state(require_formal_v2_task("P01_TO_S11"))
    assert task == {
        "task_id": "P01_TO_S11",
        "target_object_id": "P01",
        "target_slot_id": "S11",
        "status": "ACTIVE",
        "terminal": False,
        "terminal_confidence": 0.0,
        "verification_votes": 0,
    }


def test_build_observation_is_accepted_by_online_gateway() -> None:
    observation = build_observation(
        camera={
            "full_image": {
                "uri": f"cas://sha256/{'a' * 64}",
                "image_sha256": f"sha256:{'a' * 64}",
                "camera_id": "CAM_A_TOP",
                "width": 1280,
                "height": 720,
            },
            "arm_a_rgb": {
                "uri": f"cas://sha256/{'a' * 64}",
                "image_sha256": f"sha256:{'a' * 64}",
                "camera_id": "CAM_A_TOP",
                "width": 1280,
                "height": 720,
            },
            "handoff_rgb": {
                "uri": f"cas://sha256/{'b' * 64}",
                "image_sha256": f"sha256:{'b' * 64}",
                "camera_id": "CAM_HANDOFF",
                "width": 1280,
                "height": 720,
            },
            "arm_b_rgb": {
                "uri": f"cas://sha256/{'c' * 64}",
                "image_sha256": f"sha256:{'c' * 64}",
                "camera_id": "CAM_B_TOP",
                "width": 1280,
                "height": 720,
            },
            "wrist_image": None,
        },
        robot={
            "active_arm": "Arm_A",
            "arm_a": {
                "state": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "tcp_pose_m_rad": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
                "retreated": False,
                "gripper_open": True,
                "stationary": True,
            },
            "arm_b": {
                "state": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0],
                "tcp_pose_m_rad": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0],
                "retreated": True,
                "gripper_open": True,
                "stationary": True,
            },
        },
        task=build_task_state(require_formal_v2_task("P01_TO_S11")),
        observation_id="closed-loop-obs-000001",
        timestamp_ms=1,
    )
    result = V2ObservationGateway().ingest_online(observation)
    assert result.observation_id == "closed-loop-obs-000001"
    assert result.data["camera"]["wrist_image"] is None


def test_safety_policy_is_loaded_from_agent_config() -> None:
    policy = _safety_policy_from_config(
        {
            "safety": {
                "axis_abs_limits": [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0],
                "workspace_by_arm": {
                    "Arm_A": {"min_m": [0.0, -0.6, 0.0], "max_m": [0.7, 0.45, 1.0]},
                    "Arm_B": {"min_m": [0.0, -0.25, 0.0], "max_m": [0.7, 0.6, 1.0]},
                },
                "max_chunk_steps": 32,
            }
        }
    )
    assert policy.axis_abs_limits == (0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0)
    assert policy.max_chunk_steps == 32
