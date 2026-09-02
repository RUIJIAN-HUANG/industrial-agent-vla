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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless V2 stability acceptance.")
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
    positions = {
        f"/World/Parts/{part['id']}": [
            float(value) for value in part["pose"]["position_m"]
        ]
        for part in config["parts"]
    }
    positions["/World/Bins/Bin_01"] = [
        float(value) for value in config["bin"]["pose"]["position_m"]
    ]
    return positions


def _snapshot(stage: Any, paths: list[str]) -> dict[str, list[float]]:
    from run_g0_acceptance import _world_position

    return {path: _world_position(stage, path) for path in paths}


def _snapshot_errors(
    snapshot: dict[str, list[float]], expected: dict[str, list[float]]
) -> list[str]:
    errors: list[str] = []
    for path, position in snapshot.items():
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            errors.append(f"{path} contains invalid coordinates: {position}")
            continue
        drift = math.dist(position, expected[path])
        if drift > 0.10:
            errors.append(f"{path} drifted {drift:.4f} m from its reset pose")
        x, y, z = position
        if not (-1.20 <= x <= 1.20 and -0.70 <= y <= 0.70 and 0.65 <= z <= 1.40):
            errors.append(f"{path} left the V2 workcell bounds: {position}")
    return errors


def _run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.resets < 0:
        raise ValueError("--resets cannot be negative")
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

        expected = _expected_positions(config)
        required_paths = [
            "/World/Robots/Arm_A",
            "/World/Robots/Arm_B",
            *expected.keys(),
            "/World/Stations/PACK_STATION",
            "/World/Stations/HANDOFF_CENTER",
            "/World/Stations/FINISHED_01",
            *[f"/World/Cameras/{camera['id']}" for camera in config["cameras"]],
        ]
        missing = [path for path in required_paths if not stage.GetPrimAtPath(path).IsValid()]
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
        reset_iterations = args.resets if args.resets else 1
        for reset_index in range(1, reset_iterations + 1):
            world.reset()
            targets = {
                arm_id: _write_explicit_home(config, arms[arm_id], arm_id)
                for arm_id in ("Arm_A", "Arm_B")
            }
            for _ in range(args.reset_settle_steps):
                world.step(render=False)
            current = _snapshot(isaac_compat.get_current_stage(), list(expected))
            errors = _snapshot_errors(current, expected)
            for arm_id in ("Arm_A", "Arm_B"):
                errors.extend(
                    _home_readback_errors(arms[arm_id], arm_id, targets[arm_id])
                )
            reset_records.append(
                {
                    "reset_index": reset_index,
                    "dynamic_positions_m": current,
                    "robot_states": [
                        _robot_state(arms[arm_id], arm_id)
                        for arm_id in ("Arm_A", "Arm_B")
                    ],
                    "explicit_home_written": True,
                    "errors": errors,
                }
            )
            reset_errors.extend(
                f"reset {reset_index}: {message}" for message in errors
            )
        _write_json(args.evidence_dir / "reset_report.json", {"resets": reset_records})
        if reset_errors:
            raise RuntimeError("; ".join(reset_errors))

        started = time.monotonic()
        step_checks: list[dict[str, Any]] = []
        for step_index in range(1, args.steps + 1):
            world.step(render=(step_index % 30 == 0))
            if step_index % 100 == 0 or step_index == args.steps:
                current = _snapshot(isaac_compat.get_current_stage(), list(expected))
                errors = _snapshot_errors(current, expected)
                step_checks.append(
                    {"step": step_index, "dynamic_positions_m": current, "errors": errors}
                )
                if errors:
                    raise RuntimeError(f"step {step_index}: " + "; ".join(errors))
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
                "resets_requested": args.resets,
                "resets_completed": args.resets,
                "reset_settle_steps": args.reset_settle_steps,
                "headless_elapsed_seconds": elapsed,
                "camera_capture_count": len(cameras),
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
