"""Visible, read-only V2 IK reachability screen at safe approach heights."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"
ARM_IDS = ("Arm_A", "Arm_B")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run visible read-only V2 IK reachability acceptance."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--review-seconds", type=int, default=20)
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


def _ik_targets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    zones = {item["id"]: item for item in config["zones"]}
    stations = {item["id"]: item for item in config["stations"]}
    handle = config["bin"]["carry_handle"]
    handle_z = float(handle["position_local_m"][2])
    approach_z = float(handle["approach_offset_m"][2])

    targets: list[dict[str, Any]] = []
    for zone_id in ("A", "B", "C", "D"):
        position = [float(value) for value in zones[zone_id]["pose"]["position_m"]]
        position[2] = float(config["table"]["surface_z_m"]) + 0.23
        targets.append(
            {
                "target_id": f"ARM_A_ZONE_{zone_id}_SAFE_APPROACH",
                "arm_id": "Arm_A",
                "position_world_m": position,
                "purpose": f"position-only IK above zone {zone_id}",
            }
        )

    for station_id in ("PACK_STATION", "HANDOFF_CENTER"):
        station = stations[station_id]
        position = [float(value) for value in station["pose"]["position_m"]]
        position[2] += handle_z + approach_z
        targets.append(
            {
                "target_id": f"ARM_A_{station_id}_HANDLE_APPROACH",
                "arm_id": "Arm_A",
                "position_world_m": position,
                "purpose": f"position-only IK above BIN_CARRY_TCP at {station_id}",
            }
        )

    for station_id in ("HANDOFF_CENTER", "FINISHED_01"):
        station = stations[station_id]
        position = [float(value) for value in station["pose"]["position_m"]]
        position[2] += handle_z + approach_z
        targets.append(
            {
                "target_id": f"ARM_B_{station_id}_HANDLE_APPROACH",
                "arm_id": "Arm_B",
                "position_world_m": position,
                "purpose": f"position-only IK above BIN_CARRY_TCP at {station_id}",
            }
        )

    if len(targets) != 8 or len({item["target_id"] for item in targets}) != 8:
        raise RuntimeError(
            "V2 IK target construction must produce eight unique targets"
        )
    for item in targets:
        if item["arm_id"] not in ARM_IDS:
            raise RuntimeError(f"invalid IK arm: {item['arm_id']}")
        if len(item["position_world_m"]) != 3 or not all(
            math.isfinite(value) for value in item["position_world_m"]
        ):
            raise RuntimeError(f"invalid IK target: {item}")
    return targets


def main() -> int:
    args = _parse_args()
    if not 10 <= args.review_seconds <= 300:
        raise ValueError("--review-seconds must be in [10, 300]")
    for path in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    evidence_dir = args.evidence_dir.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = evidence_dir / "run_result.json"
    result: dict[str, Any] = {
        "status": "ERROR",
        "gate": "V2_VISIBLE_READ_ONLY_IK_REACHABILITY",
        "headless": False,
        "position_only_ik": True,
        "ik_solution_applied": False,
        "cartesian_motion_performed": False,
        "grasp_performed": False,
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    simulation_app = None
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
        targets = _ik_targets(config)
        simulation_app = isaac_compat.launch_simulation_app(headless=False)

        import numpy as np
        import single_bin_scene_v2_builder
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
            interface_config_loader,
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
                    name=f"v2_ik_{arm_id.lower()}",
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

        joint_positions_before = {
            arm_id: np.asarray(arms[arm_id].get_joint_positions(), dtype=float).copy()
            for arm_id in ARM_IDS
        }
        solvers: dict[str, Any] = {}
        for arm_id in ARM_IDS:
            solver_config = (
                interface_config_loader.load_supported_lula_kinematics_solver_config(
                    "Franka"
                )
            )
            lula = LulaKinematicsSolver(**solver_config)
            base_position, base_orientation = arms[arm_id].get_world_pose()
            lula.set_robot_base_pose(
                np.asarray(base_position, dtype=float),
                np.asarray(base_orientation, dtype=float),
            )
            solvers[arm_id] = ArticulationKinematicsSolver(
                arms[arm_id], lula, "right_gripper"
            )

        ik_records: list[dict[str, Any]] = []
        errors: list[str] = []
        for target in targets:
            solver = solvers[target["arm_id"]]
            ik_action, success = solver.compute_inverse_kinematics(
                np.asarray(target["position_world_m"], dtype=float)
            )
            solution = getattr(ik_action, "joint_positions", None)
            solution_array = (
                np.asarray(solution, dtype=float)
                if solution is not None
                else np.asarray([])
            )
            finite_solution = bool(
                solution_array.size and np.all(np.isfinite(solution_array))
            )
            passed = bool(success and finite_solution)
            if not passed:
                errors.append(f"{target['target_id']}: Lula position-only IK failed")
            ik_records.append(
                {
                    **target,
                    "success": bool(success),
                    "finite_solution": finite_solution,
                    "solution_joint_positions_rad": solution_array,
                    "solution_applied": False,
                }
            )

        joint_positions_after = {
            arm_id: np.asarray(arms[arm_id].get_joint_positions(), dtype=float).copy()
            for arm_id in ARM_IDS
        }
        max_joint_change = {
            arm_id: float(
                np.max(
                    np.abs(
                        joint_positions_after[arm_id] - joint_positions_before[arm_id]
                    )
                )
            )
            for arm_id in ARM_IDS
        }
        for arm_id, change in max_joint_change.items():
            if change > 1e-9:
                errors.append(
                    f"{arm_id}: read-only IK changed a live joint by {change:.12f} rad"
                )

        _write_json(
            evidence_dir / "ik_report.json",
            {
                "targets": ik_records,
                "max_live_joint_change_rad": max_joint_change,
                "errors": errors,
            },
        )
        deadline = time.monotonic() + args.review_seconds
        while simulation_app.is_running() and time.monotonic() < deadline:
            world.step(render=True)
        final_robot_states = [_robot_state(arms[item], item) for item in ARM_IDS]

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
                "target_count": len(targets),
                "successful_target_count": sum(
                    1
                    for item in ik_records
                    if item["success"] and item["finite_solution"]
                ),
                "ik_records": ik_records,
                "max_live_joint_change_rad": max_joint_change,
                "final_robot_states": final_robot_states,
                "camera_captures": camera_captures,
                "review_seconds": args.review_seconds,
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
