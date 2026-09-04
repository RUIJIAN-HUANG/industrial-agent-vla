"""Headless V2 stability, reset, HOME, and three-camera acceptance.

Launch this file with Isaac Sim's ``python.sh``.  It deliberately records
stability evidence only; it does not claim collision, grasp, or transport
acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"
MAX_POSITION_DRIFT_M = 0.10
MAX_ORIENTATION_DRIFT_DEG = 10.0
MAX_LINEAR_SPEED_M_S = 0.02
MAX_ANGULAR_SPEED_RAD_S = 0.20


def _effective_reset_count(requested_resets: int) -> int:
    """Return resets actually executed, including the required initial reset."""

    if requested_resets < 0:
        raise ValueError("--resets cannot be negative")
    return max(1, requested_resets)


def _reset_metadata(
    requested_resets: int,
    completed_resets: int,
) -> dict[str, int | bool]:
    """Build the shared reset accounting written to every evidence file."""

    return {
        "resets_requested": requested_resets,
        "resets_completed": completed_resets,
        "implicit_initial_reset": requested_resets == 0 and completed_resets > 0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run headless V2 stability acceptance."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--resets", type=int, default=20)
    parser.add_argument("--reset-settle-steps", type=int, default=120)
    parser.add_argument(
        "--capture-cameras", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _expected_positions(config: dict[str, Any]) -> dict[str, list[float]]:
    return {
        path: state["position_m"] for path, state in _expected_states(config).items()
    }


def _rpy_deg_to_quaternion_wxyz(rpy_deg: list[float]) -> list[float]:
    if len(rpy_deg) != 3 or not all(math.isfinite(value) for value in rpy_deg):
        raise ValueError("rpy_deg must contain three finite values")
    roll, pitch, yaw = (math.radians(float(value)) / 2.0 for value in rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _expected_states(config: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    bodies = [(f"/World/Parts/{part['id']}", part["pose"]) for part in config["parts"]]
    bodies.append(("/World/Bins/Bin_01", config["bin"]["pose"]))
    return {
        path: {
            "position_m": [float(value) for value in pose["position_m"]],
            "orientation_wxyz": _rpy_deg_to_quaternion_wxyz(
                [float(value) for value in pose.get("rpy_deg", [0.0, 0.0, 0.0])]
            ),
        }
        for path, pose in bodies
    }


def _quaternion_error_rad(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("quaternions must contain four values")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("quaternions must be finite")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("quaternions cannot have zero norm")
    cosine = abs(
        sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, cosine)))


def _snapshot(stage: Any, paths: list[str]) -> dict[str, dict[str, list[float]]]:
    from pxr import Usd, UsdGeom

    snapshot: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Required prim is missing: {path}")
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        snapshot[path] = {
            "position_m": [
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
            ],
            "orientation_wxyz": [
                float(rotation.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
        }
    return snapshot


def _motion_between(
    previous: dict[str, dict[str, list[float]]],
    current: dict[str, dict[str, list[float]]],
    dt_s: float,
) -> dict[str, dict[str, float]]:
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    return {
        path: {
            "linear_speed_m_s": math.dist(
                previous[path]["position_m"], state["position_m"]
            )
            / dt_s,
            "angular_speed_rad_s": _quaternion_error_rad(
                previous[path]["orientation_wxyz"], state["orientation_wxyz"]
            )
            / dt_s,
        }
        for path, state in current.items()
    }


def _peak_motion(
    peak: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        path: {
            key: max(peak.get(path, {}).get(key, 0.0), value)
            for key, value in metrics.items()
        }
        for path, metrics in current.items()
    }


def _snapshot_errors(
    snapshot: dict[str, dict[str, list[float]]],
    expected: dict[str, dict[str, list[float]]],
    motion: dict[str, dict[str, float]],
) -> list[str]:
    errors: list[str] = []
    for path, state in snapshot.items():
        position = state["position_m"]
        orientation = state["orientation_wxyz"]
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            errors.append(f"{path} contains invalid coordinates: {position}")
            continue
        if len(orientation) != 4 or not all(
            math.isfinite(value) for value in orientation
        ):
            errors.append(f"{path} contains an invalid orientation: {orientation}")
            continue
        drift = math.dist(position, expected[path]["position_m"])
        if drift > MAX_POSITION_DRIFT_M:
            errors.append(f"{path} drifted {drift:.4f} m from its reset pose")
        orientation_error_deg = math.degrees(
            _quaternion_error_rad(
                orientation,
                expected[path]["orientation_wxyz"],
            )
        )
        if orientation_error_deg > MAX_ORIENTATION_DRIFT_DEG:
            errors.append(
                f"{path} rotated {orientation_error_deg:.3f} deg from its reset pose"
            )
        linear_speed = motion[path]["linear_speed_m_s"]
        angular_speed = motion[path]["angular_speed_rad_s"]
        if not math.isfinite(linear_speed) or linear_speed > MAX_LINEAR_SPEED_M_S:
            errors.append(
                f"{path} linear speed {linear_speed:.6f} m/s exceeds "
                f"{MAX_LINEAR_SPEED_M_S:.6f} m/s"
            )
        if not math.isfinite(angular_speed) or angular_speed > MAX_ANGULAR_SPEED_RAD_S:
            errors.append(
                f"{path} angular speed {angular_speed:.6f} rad/s exceeds "
                f"{MAX_ANGULAR_SPEED_RAD_S:.6f} rad/s"
            )
        x, y, z = position
        if not (-1.20 <= x <= 1.20 and -0.70 <= y <= 0.70 and 0.65 <= z <= 1.40):
            errors.append(f"{path} left the V2 workcell bounds: {position}")
    return errors


def _run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    reset_iterations = _effective_reset_count(args.resets)
    if args.reset_settle_steps < 1:
        raise ValueError("--reset-settle-steps must be at least 1")

    for path in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import isaac_compat
    from v2_scene_contract import load_config, require_valid_config

    config = load_config(args.config)
    require_valid_config(config)
    simulation_app = isaac_compat.launch_simulation_app(headless=True)
    result["simulation_app_started"] = True
    try:
        result["isaac_sim_version"] = isaac_compat.require_isaac_sim_51()
        import single_bin_scene_v2_builder
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from run_g0_acceptance import (
            _capture_cameras,
            _home_readback_errors,
            _robot_state,
            _write_explicit_home,
        )

        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(None)
        single_bin_scene_v2_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        isaac_compat.wait_for_stage_loading(simulation_app, timeout_seconds=180.0)

        expected = _expected_states(config)
        required_paths = [
            "/World/Robots/Arm_A",
            "/World/Robots/Arm_B",
            *expected.keys(),
            "/World/Stations/PACK_STATION",
            "/World/Stations/HANDOFF_CENTER",
            "/World/Stations/FINISHED_01",
            *[f"/World/Cameras/{camera['id']}" for camera in config["cameras"]],
        ]
        missing = [
            path for path in required_paths if not stage.GetPrimAtPath(path).IsValid()
        ]
        if missing:
            raise RuntimeError(f"required V2 prims are missing: {missing}")

        result["scene_file"] = isaac_compat.save_stage_checked(args.output_scene)
        result["franka_asset"] = franka_asset
        result["required_prim_count"] = len(required_paths)

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
                    name=f"v2_stability_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }

        reset_records: list[dict[str, Any]] = []
        reset_errors: list[str] = []
        home_targets: dict[str, list[float]] = {}
        for reset_index in range(1, reset_iterations + 1):
            world.reset()
            home_targets = {
                arm_id: _write_explicit_home(config, arms[arm_id], arm_id)
                for arm_id in ("Arm_A", "Arm_B")
            }
            for _ in range(args.reset_settle_steps - 1):
                world.step(render=False)
            before_final_settle = _snapshot(
                isaac_compat.get_current_stage(), list(expected)
            )
            world.step(render=False)
            current = _snapshot(isaac_compat.get_current_stage(), list(expected))
            final_motion = _motion_between(
                before_final_settle,
                current,
                float(physics["physics_dt_s"]),
            )
            errors = _snapshot_errors(current, expected, final_motion)
            for arm_id in ("Arm_A", "Arm_B"):
                errors.extend(
                    _home_readback_errors(arms[arm_id], arm_id, home_targets[arm_id])
                )
            reset_records.append(
                {
                    "reset_index": reset_index,
                    "dynamic_positions_m": {
                        path: state["position_m"] for path, state in current.items()
                    },
                    "dynamic_pose_states": current,
                    "final_motion": final_motion,
                    "robot_states": [
                        _robot_state(arms[arm_id], arm_id)
                        for arm_id in ("Arm_A", "Arm_B")
                    ],
                    "explicit_home_written": True,
                    "errors": errors,
                }
            )
            reset_errors.extend(f"reset {reset_index}: {message}" for message in errors)
        reset_metadata = _reset_metadata(args.resets, len(reset_records))
        result.update(reset_metadata)
        _write_json(
            args.evidence_dir / "reset_report.json",
            {
                **reset_metadata,
                "resets": reset_records,
            },
        )
        if reset_errors:
            raise RuntimeError("; ".join(reset_errors))

        started = time.monotonic()
        step_checks: list[dict[str, Any]] = []
        previous = _snapshot(isaac_compat.get_current_stage(), list(expected))
        interval_peak_motion: dict[str, dict[str, float]] = {}
        for step_index in range(1, args.steps + 1):
            world.step(render=(step_index % 30 == 0))
            current = _snapshot(isaac_compat.get_current_stage(), list(expected))
            step_motion = _motion_between(
                previous,
                current,
                float(physics["physics_dt_s"]),
            )
            interval_peak_motion = _peak_motion(interval_peak_motion, step_motion)
            previous = current
            robot_errors: list[str] = []
            for arm_id in ("Arm_A", "Arm_B"):
                robot_errors.extend(
                    _home_readback_errors(arms[arm_id], arm_id, home_targets[arm_id])
                )
            checkpoint = step_index % 100 == 0 or step_index == args.steps
            if robot_errors or checkpoint:
                errors = robot_errors
                if checkpoint:
                    errors.extend(
                        _snapshot_errors(current, expected, interval_peak_motion)
                    )
                step_checks.append(
                    {
                        "step": step_index,
                        "dynamic_positions_m": {
                            path: state["position_m"] for path, state in current.items()
                        },
                        "dynamic_pose_states": current,
                        "peak_motion_since_previous_check": interval_peak_motion,
                        "robot_states": [
                            _robot_state(arms[arm_id], arm_id)
                            for arm_id in ("Arm_A", "Arm_B")
                        ],
                        "errors": errors,
                    }
                )
                if errors:
                    _write_json(
                        args.evidence_dir / "step_checks.json",
                        {"checks": step_checks},
                    )
                    raise RuntimeError(f"step {step_index}: " + "; ".join(errors))
                interval_peak_motion = {}
        elapsed = time.monotonic() - started
        _write_json(args.evidence_dir / "step_checks.json", {"checks": step_checks})

        cameras: list[dict[str, Any]] = []
        if args.capture_cameras:
            cameras = _capture_cameras(simulation_app, config, args.evidence_dir)
            if len(cameras) != 3:
                raise RuntimeError(f"expected 3 camera captures, got {len(cameras)}")
            _write_json(
                args.evidence_dir / "camera_manifest.json",
                {"cameras": cameras, "online_gt_included": False},
            )

        result.update(
            {
                "status": "PASS",
                "gate": "V2_HEADLESS_STABILITY_ONLY",
                "scene_id": config["scene_id"],
                "headless_steps_requested": args.steps,
                "headless_steps_completed": args.steps,
                "reset_settle_steps": args.reset_settle_steps,
                "headless_elapsed_seconds": elapsed,
                "camera_capture_count": len(cameras),
                "stability_thresholds": {
                    "max_position_drift_m": MAX_POSITION_DRIFT_M,
                    "max_orientation_drift_deg": MAX_ORIENTATION_DRIFT_DEG,
                    "max_linear_speed_m_s": MAX_LINEAR_SPEED_M_S,
                    "max_angular_speed_rad_s": MAX_ANGULAR_SPEED_RAD_S,
                },
                "collision_acceptance_performed": False,
                "grasp_acceptance_performed": False,
                "loaded_transport_acceptance_performed": False,
                "online_gt_included": False,
                "errors": [],
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        # Some Kit builds terminate the interpreter from close() and never
        # return to main().  Persist the authoritative result first.
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(args.evidence_dir / "run_result.json", result)
        simulation_app.close()


def main() -> int:
    args = _parse_args()
    args.evidence_dir = args.evidence_dir.expanduser().resolve()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "FAIL",
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "simulation_app_started": False,
    }
    try:
        _run(args, result)
    except BaseException as exc:
        result.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(result["traceback"], file=sys.stderr)
    finally:
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(args.evidence_dir / "run_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
