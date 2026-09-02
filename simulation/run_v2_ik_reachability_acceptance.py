"""Visible, read-only V2 IK reachability screen at safe approach heights."""

from __future__ import annotations

import argparse
import importlib
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


def _preload_pink_runtime(ik_backend: str) -> None:
    """Mirror the proven Arm_A startup order before Kit loads plugins."""

    if ik_backend != "pink":
        return
    importlib.import_module("eigenpy")
    importlib.import_module("pinocchio")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run visible read-only V2 IK reachability acceptance."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--review-seconds", type=int, default=20)
    parser.add_argument("--ik-backend", choices=("lula", "pink"), default="lula")
    parser.add_argument(
        "--arm-b-bin-transport-only",
        action="store_true",
        help="Probe only Arm_B at the frozen bin start and FINISHED_01.",
    )
    parser.add_argument("--pink-max-virtual-actions", type=int, default=64)
    parser.add_argument("--pink-position-tolerance-m", type=float, default=0.01)
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
    initial_station_id = str(config["bin"]["initial_station_id"])
    frozen_bin_position = [
        float(value) for value in config["bin"]["pose"]["position_m"]
    ]
    initial_station_position = [
        float(value) for value in stations[initial_station_id]["pose"]["position_m"]
    ]
    bin_center_offset = [
        frozen_bin_position[index] - initial_station_position[index]
        for index in range(3)
    ]

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
        position = [
            value + bin_center_offset[index] for index, value in enumerate(position)
        ]
        position[2] += handle_z + approach_z
        targets.append(
            {
                "target_id": f"ARM_A_{station_id}_HANDLE_APPROACH",
                "arm_id": "Arm_A",
                "position_world_m": position,
                "purpose": f"position-only IK above BIN_CARRY_TCP at {station_id}",
            }
        )

    # Arm_B only enters the approved relay after Arm_A has placed Bin_01 at
    # HANDOFF_CENTER and retreated.  Probing Arm_B at PACK_STATION contradicts
    # the frozen BIN01_TO_FINISHED01 task contract and asks it to reach into
    # Arm_A's workspace.
    for station_id in ("HANDOFF_CENTER", "FINISHED_01"):
        station = stations[station_id]
        position = [float(value) for value in station["pose"]["position_m"]]
        position = [
            value + bin_center_offset[index] for index, value in enumerate(position)
        ]
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
        raise RuntimeError("V2 IK target construction must produce eight unique targets")
    for item in targets:
        if item["arm_id"] not in ARM_IDS:
            raise RuntimeError(f"invalid IK arm: {item['arm_id']}")
        if len(item["position_world_m"]) != 3 or not all(
            math.isfinite(value) for value in item["position_world_m"]
        ):
            raise RuntimeError(f"invalid IK target: {item}")
    return targets


def _pink_top_down_orientation_candidates(
    current_world_rotation: Any,
) -> list[dict[str, Any]]:
    """Return four yaw-equivalent, task-valid top-down tool orientations.

    The safe-approach targets constrain tool Z to world -Z, while wrist yaw is
    deliberately left free.  Searching the four quarter-turn equivalents
    avoids rejecting a reachable target solely because HOME has an unsuitable
    redundant-wrist yaw.
    """

    import numpy as np
    from simulation.scripted_expert_plan import yaw_preserving_top_down_rotation

    base = yaw_preserving_top_down_rotation(current_world_rotation)
    candidates: list[dict[str, Any]] = []
    for yaw_offset_deg in (0, 90, 180, -90):
        angle = math.radians(yaw_offset_deg)
        world_yaw = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        candidates.append(
            {
                "yaw_offset_deg": yaw_offset_deg,
                "rotation_world": world_yaw @ base,
            }
        )
    return candidates


def main() -> int:
    args = _parse_args()
    if not 10 <= args.review_seconds <= 300:
        raise ValueError("--review-seconds must be in [10, 300]")
    if not 1 <= args.pink_max_virtual_actions <= 200:
        raise ValueError("--pink-max-virtual-actions must be in [1, 200]")
    if not 0.001 <= args.pink_position_tolerance_m <= 0.05:
        raise ValueError("--pink-position-tolerance-m must be in [0.001, 0.05]")
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
        "position_only_ik": args.ik_backend == "lula",
        "pink_orientation_constraint": (
            "tool_z_world_down_yaw_free" if args.ik_backend == "pink" else None
        ),
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
        if args.arm_b_bin_transport_only:
            critical_ids = {
                "ARM_B_HANDOFF_CENTER_HANDLE_APPROACH",
                "ARM_B_FINISHED_01_HANDLE_APPROACH",
            }
            targets = [item for item in targets if item["target_id"] in critical_ids]
            if [item["target_id"] for item in targets] != [
                "ARM_B_HANDOFF_CENTER_HANDLE_APPROACH",
                "ARM_B_FINISHED_01_HANDLE_APPROACH",
            ]:
                raise RuntimeError("Arm_B bin-transport target selection drifted")
        _preload_pink_runtime(args.ik_backend)
        simulation_app = isaac_compat.launch_simulation_app(headless=False)

        import numpy as np
        import single_bin_scene_v2_builder
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

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
        ik_records: list[dict[str, Any]] = []
        errors: list[str] = []
        if args.ik_backend == "lula":
            from isaacsim.robot_motion.motion_generation import (
                ArticulationKinematicsSolver,
                LulaKinematicsSolver,
                interface_config_loader,
            )

            solvers: dict[str, Any] = {}
            for arm_id in ARM_IDS:
                solver_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
                    "Franka"
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
                    errors.append(
                        f"{target['target_id']}: Lula position-only IK failed"
                    )
                ik_records.append(
                    {
                        **target,
                        "backend": "lula",
                        "success": bool(success),
                        "finite_solution": finite_solution,
                        "solution_joint_positions_rad": solution_array,
                        "solution_applied": False,
                    }
                )
        else:
            from isaac_franka_controller import (
                IsaacSimFrankaController,
                _rotation_matrix_to_quaternion,
            )

            controller = IsaacSimFrankaController(
                world=world,
                arms=arms,
                physics_dt_s=float(physics["physics_dt_s"]),
                virtual_tcp_fingertip_frame_names=(
                    "panda_leftfingertip",
                    "panda_rightfingertip",
                ),
                ik_backend="pink",
            )
            for target in targets:
                arm_id = target["arm_id"]
                _, initial_rotation_world = controller.end_effector_pose(arm_id)
                target_position_world = np.asarray(
                    target["position_world_m"], dtype=float
                )
                candidate_records: list[dict[str, Any]] = []
                selected: dict[str, Any] | None = None
                for orientation in _pink_top_down_orientation_candidates(
                    initial_rotation_world
                ):
                    virtual_joints = joint_positions_before[arm_id].copy()
                    predicted_tcp_world = np.full(3, np.nan, dtype=float)
                    position_error_m = float("inf")
                    virtual_action_count = 0
                    target_orientation_world = _rotation_matrix_to_quaternion(
                        orientation["rotation_world"]
                    )
                    for virtual_action_count in range(
                        1, args.pink_max_virtual_actions + 1
                    ):
                        virtual_joints, predicted_tcp_world, _ = (
                            controller.predict_pink_tcp_pose_read_only(
                                arm_id=arm_id,
                                current_joint_positions=virtual_joints,
                                target_tcp_position_world_m=target_position_world,
                                target_tcp_orientation_world_wxyz=(
                                    target_orientation_world
                                ),
                                dt_s=0.1,
                            )
                        )
                        position_error_m = float(
                            np.linalg.norm(
                                predicted_tcp_world - target_position_world
                            )
                        )
                        if position_error_m <= args.pink_position_tolerance_m:
                            break
                    finite_candidate = bool(
                        np.all(np.isfinite(virtual_joints))
                        and np.all(np.isfinite(predicted_tcp_world))
                        and math.isfinite(position_error_m)
                    )
                    candidate_record = {
                        "yaw_offset_deg": orientation["yaw_offset_deg"],
                        "target_orientation_world_wxyz": target_orientation_world,
                        "finite_solution": finite_candidate,
                        "virtual_action_count": virtual_action_count,
                        "final_position_error_m": position_error_m,
                        "predicted_tcp_position_world_m": predicted_tcp_world,
                        "solution_joint_positions_rad": virtual_joints,
                    }
                    candidate_records.append(candidate_record)
                    if (
                        finite_candidate
                        and position_error_m <= args.pink_position_tolerance_m
                    ):
                        selected = candidate_record
                        break
                if selected is None:
                    finite_candidates = [
                        item for item in candidate_records if item["finite_solution"]
                    ]
                    selected = min(
                        finite_candidates or candidate_records,
                        key=lambda item: item["final_position_error_m"],
                    )
                virtual_joints = selected["solution_joint_positions_rad"]
                predicted_tcp_world = selected["predicted_tcp_position_world_m"]
                position_error_m = float(selected["final_position_error_m"])
                virtual_action_count = int(selected["virtual_action_count"])
                finite_solution = bool(
                    np.all(np.isfinite(virtual_joints))
                    and np.all(np.isfinite(predicted_tcp_world))
                    and math.isfinite(position_error_m)
                )
                success = bool(
                    finite_solution
                    and position_error_m <= args.pink_position_tolerance_m
                )
                if not success:
                    errors.append(
                        f"{target['target_id']}: Pink virtual TCP error "
                        f"{position_error_m:.6f} m exceeds "
                        f"{args.pink_position_tolerance_m:.6f} m"
                    )
                ik_records.append(
                    {
                        **target,
                        "backend": "pink",
                        "orientation_constraint": "tool_z_world_down_yaw_free",
                        "selected_yaw_offset_deg": selected["yaw_offset_deg"],
                        "orientation_candidates": candidate_records,
                        "success": success,
                        "finite_solution": finite_solution,
                        "virtual_action_count": virtual_action_count,
                        "position_tolerance_m": args.pink_position_tolerance_m,
                        "final_position_error_m": position_error_m,
                        "predicted_tcp_position_world_m": predicted_tcp_world,
                        "solution_joint_positions_rad": virtual_joints,
                        "solution_applied": False,
                        "pink_diagnostics": controller.diagnostics(arm_id),
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
                "ik_backend": args.ik_backend,
                "arm_b_bin_transport_only": args.arm_b_bin_transport_only,
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
                "ik_backend": args.ik_backend,
                "arm_b_bin_transport_only": args.arm_b_bin_transport_only,
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
