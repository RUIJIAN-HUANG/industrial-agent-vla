"""Visible V2 dual-arm 5 mm up/return micro-motion acceptance."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"
ARM_IDS = ("Arm_A", "Arm_B")
DELTA_Z_M = 0.005
TCP_DELTA_TOLERANCE_M = 0.0015
TCP_RETURN_TOLERANCE_M = 0.0015
OTHER_ARM_JOINT_TOLERANCE_RAD = 0.002
FINGER_TOLERANCE_M = 0.001
MOTION_SETTLE_PHYSICS_STEPS = 60


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run visible V2 dual-arm micro-motion acceptance."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--review-seconds", type=int, default=15)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _micro_action_values(delta_z_m: float) -> list[float]:
    if abs(delta_z_m) > DELTA_Z_M + 1e-12:
        raise ValueError("V2 micro-motion cannot exceed 5 mm")
    return [0.0, 0.0, float(delta_z_m), 0.0, 0.0, 0.0, 1.0]


def _bounded_return_delta_z_m(measured_up_delta_base_m: Sequence[float]) -> float:
    """Return toward the recorded start without exceeding the 5 mm safety cap."""

    if len(measured_up_delta_base_m) != 3:
        raise ValueError("measured TCP delta must contain exactly three values")
    measured_z = float(measured_up_delta_base_m[2])
    if not math.isfinite(measured_z):
        raise ValueError("measured TCP z delta must be finite")
    return max(-DELTA_Z_M, min(DELTA_Z_M, -measured_z))


def _settle_motion(world: Any, *, steps: int = MOTION_SETTLE_PHYSICS_STEPS) -> None:
    """Let the articulation converge to the controller's final IK target."""

    if steps < 1:
        raise ValueError("motion settle steps must be positive")
    for _ in range(steps):
        world.step(render=True)


def main() -> int:
    args = _parse_args()
    if not 10 <= args.review_seconds <= 120:
        raise ValueError("--review-seconds must be in [10, 120]")
    for path in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    evidence_dir = args.evidence_dir.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = evidence_dir / "run_result.json"
    result: dict[str, Any] = {
        "status": "ERROR",
        "gate": "V2_VISIBLE_DUAL_ARM_5MM_MICRO_MOTION",
        "headless": False,
        "command_delta_z_m": DELTA_Z_M,
        "grasp_performed": False,
        "objects_contacted": False,
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    simulation_app = None
    controller = None
    try:
        import isaac_compat
        from run_g0_acceptance import (
            _capture_cameras,
            _robot_state,
            _write_explicit_home,
        )
        from v2_scene_contract import load_config, require_valid_config

        config = load_config(args.config)
        require_valid_config(config)
        simulation_app = isaac_compat.launch_simulation_app(headless=False)

        import numpy as np
        import single_bin_scene_v2_builder
        from isaac_franka_controller import (
            IsaacSimFrankaController,
            _quat_inverse,
            _rotate_vector,
        )
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from industrial_agent.contracts import ActionStep

        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(None)
        single_bin_scene_v2_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        isaac_compat.wait_for_stage_loading(simulation_app, timeout_seconds=180.0)
        scene_file = isaac_compat.save_stage_checked(args.output_scene)

        physics = config["physics"]
        if World.instance():
            World.instance().clear_instance()
        world = World(
            physics_dt=float(physics["physics_dt_s"]),
            rendering_dt=float(physics["rendering_dt_s"]),
            stage_units_in_meters=1.0,
        )
        arms = {
            arm_id: world.scene.add(
                SingleArticulation(
                    prim_path=f"/World/Robots/{arm_id}",
                    name=f"v2_micro_{arm_id.lower()}",
                )
            )
            for arm_id in ARM_IDS
        }
        world.reset()
        home_targets = {
            arm_id: _write_explicit_home(config, arms[arm_id], arm_id)
            for arm_id in ARM_IDS
        }
        for arm_id in ARM_IDS:
            arms[arm_id].get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=home_targets[arm_id])
            )
        for _ in range(120):
            world.step(render=True)

        controller = IsaacSimFrankaController(
            world=world,
            arms=arms,
            physics_dt_s=float(physics["physics_dt_s"]),
            virtual_tcp_fingertip_frame_names=(
                "panda_leftfingertip",
                "panda_rightfingertip",
            ),
        )
        errors: list[str] = []
        motion_records: list[dict[str, Any]] = []

        for arm_id in ARM_IDS:
            other_arm = "Arm_B" if arm_id == "Arm_A" else "Arm_A"
            controller.validate_ready(arm_id)
            start_tcp, _ = controller.end_effector_pose(arm_id)
            _, base_orientation = arms[arm_id].get_world_pose()
            start_active_joints = np.asarray(
                arms[arm_id].get_joint_positions(), dtype=float
            ).copy()
            start_other_joints = np.asarray(
                arms[other_arm].get_joint_positions(), dtype=float
            ).copy()
            start_fingers = controller.gripper_joint_positions(arm_id)

            up_action = ActionStep.from_sequence(
                _micro_action_values(DELTA_Z_M), duration_ms=100
            )
            controller.execute_action(up_action, arm_id=arm_id)
            # execute_action emits the final IK target after 100 ms, but the
            # articulation still needs time to physically converge to it.  Read
            # the TCP only after convergence; otherwise the following -5 mm
            # command starts from a partially completed upward move.
            _settle_motion(world)
            up_tcp, _ = controller.end_effector_pose(arm_id)
            up_delta_world = up_tcp - start_tcp
            up_delta_base = _rotate_vector(
                _quat_inverse(np.asarray(base_orientation, dtype=float)),
                up_delta_world,
            )
            expected_up = np.asarray([0.0, 0.0, DELTA_Z_M], dtype=float)
            up_ok = bool(
                np.allclose(
                    up_delta_base,
                    expected_up,
                    atol=TCP_DELTA_TOLERANCE_M,
                    rtol=0.0,
                )
            )
            if not up_ok:
                errors.append(
                    f"{arm_id}: 5 mm up delta mismatch: {up_delta_base.tolist()}"
                )

            return_delta_z_m = _bounded_return_delta_z_m(up_delta_base)
            down_action = ActionStep.from_sequence(
                _micro_action_values(return_delta_z_m), duration_ms=100
            )
            controller.execute_action(down_action, arm_id=arm_id)
            _settle_motion(world)
            final_tcp, _ = controller.end_effector_pose(arm_id)
            return_error_m = float(np.linalg.norm(final_tcp - start_tcp))
            if return_error_m > TCP_RETURN_TOLERANCE_M:
                errors.append(
                    f"{arm_id}: return error {return_error_m:.6f} m exceeds "
                    f"{TCP_RETURN_TOLERANCE_M:.6f} m"
                )

            final_other_joints = np.asarray(
                arms[other_arm].get_joint_positions(), dtype=float
            )
            other_joint_change = float(
                np.max(np.abs(final_other_joints - start_other_joints))
            )
            if other_joint_change > OTHER_ARM_JOINT_TOLERANCE_RAD:
                errors.append(
                    f"{other_arm}: changed {other_joint_change:.6f} rad while "
                    f"{arm_id} was active"
                )

            final_fingers = controller.gripper_joint_positions(arm_id)
            finger_change_m = float(np.max(np.abs(final_fingers - start_fingers)))
            if finger_change_m > FINGER_TOLERANCE_M:
                errors.append(
                    f"{arm_id}: finger change {finger_change_m:.6f} m exceeds "
                    f"{FINGER_TOLERANCE_M:.6f} m"
                )
            final_active_joints = np.asarray(
                arms[arm_id].get_joint_positions(), dtype=float
            )
            motion_records.append(
                {
                    "arm_id": arm_id,
                    "other_arm": other_arm,
                    "start_tcp_world_m": start_tcp,
                    "up_tcp_world_m": up_tcp,
                    "up_delta_world_m": up_delta_world,
                    "up_delta_base_m": up_delta_base,
                    "expected_up_delta_base_m": expected_up,
                    "up_delta_within_tolerance": up_ok,
                    "commanded_return_delta_z_m": return_delta_z_m,
                    "final_tcp_world_m": final_tcp,
                    "return_error_m": return_error_m,
                    "other_arm_max_joint_change_rad": other_joint_change,
                    "finger_max_change_m": finger_change_m,
                    "active_arm_joint_change_norm_rad": float(
                        np.linalg.norm(final_active_joints - start_active_joints)
                    ),
                    "gripper_command": "open",
                }
            )

        deadline = time.monotonic() + args.review_seconds
        while simulation_app.is_running() and time.monotonic() < deadline:
            world.step(render=True)
        final_robot_states = [_robot_state(arms[item], item) for item in ARM_IDS]
        stop_receipt = controller.safe_stop("V2 micro-motion acceptance completed")
        if not stop_receipt.confirmed:
            errors.append(f"safe-stop was not confirmed: {stop_receipt!r}")

        _write_json(
            evidence_dir / "motion_report.json",
            {
                "records": motion_records,
                "safe_stop": asdict(stop_receipt),
                "errors": errors,
            },
        )
        camera_captures = _capture_cameras(simulation_app, config, evidence_dir)
        if len(camera_captures) != 3:
            errors.append(
                f"expected exactly three camera captures, got {len(camera_captures)}"
            )
        result.update(
            {
                "status": "PASS" if not errors else "FAIL",
                "scene_id": config["scene_id"],
                "scene_file": scene_file,
                "franka_asset": franka_asset,
                "arm_sequence": list(ARM_IDS),
                "completed_motion_cycles": len(motion_records),
                "motion_records": motion_records,
                "safe_stop": asdict(stop_receipt),
                "final_robot_states": final_robot_states,
                "camera_captures": camera_captures,
                "errors": errors,
                "online_gt_included": False,
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(
            json.dumps(_jsonable(result), indent=2, ensure_ascii=False), file=sys.stderr
        )
        if controller is not None:
            try:
                receipt = controller.safe_stop("V2 micro-motion acceptance failed")
                result["failure_safe_stop"] = asdict(receipt)
            except BaseException as stop_exc:
                result["failure_safe_stop_error"] = repr(stop_exc)
    finally:
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            _write_json(result_path, result)
        finally:
            if simulation_app is not None:
                simulation_app.close()

    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
