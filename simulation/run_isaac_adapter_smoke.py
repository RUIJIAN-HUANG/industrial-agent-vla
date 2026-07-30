"""Exercise the real Isaac execution adapter on the frozen dual-Franka scene.

This is a controller smoke test, not a production observation pipeline.  Its
observation source contains live articulation telemetry but deliberately no
camera evidence.  A successful run proves that an ``ActionStep`` crosses the
production adapter and reaches the Isaac Sim 5.1 Franka controller.
"""

from __future__ import annotations

import argparse
import json
from math import atan2, isfinite
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v1.json"
DEFAULT_SCENE = SCRIPT_DIR / "generated" / "single_bin_scene_v1.usda"
DEFAULT_COMMAND_LEDGER = (
    REPOSITORY_ROOT / "artifacts" / "isaac-adapter" / "command-ids.jsonl"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one small action through the real Isaac adapter."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--franka-usd")
    parser.add_argument("--arm-id", choices=("Arm_A", "Arm_B"), default="Arm_A")
    parser.add_argument(
        "--delta-z-m",
        type=float,
        default=0.005,
        help="Small base-frame Z translation used for the smoke action.",
    )
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument(
        "--command-ledger",
        type=Path,
        default=DEFAULT_COMMAND_LEDGER,
        help="Fsync-backed exactly-once command ledger.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Optional JSON path for preserving the smoke-test result.",
    )
    return parser.parse_args()


def _joint_names(arm: Any) -> list[str]:
    names = getattr(arm, "dof_names", None)
    if names is None:
        names = getattr(arm, "joint_names", None)
    if names is None:
        raise RuntimeError("Franka articulation exposes no joint names")
    return [str(item) for item in names]


def _quaternion_to_rotvec(quaternion: np.ndarray) -> list[float]:
    """Convert normalized wxyz quaternion to the shortest rotation vector."""

    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("TCP quaternion is invalid")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise RuntimeError("TCP quaternion has zero norm")
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * atan2(vector_norm, float(quaternion[0]))
    axis = quaternion[1:] / vector_norm
    return [float(item) for item in axis * angle]


def _is_retreated(position: np.ndarray, config: dict[str, Any]) -> bool:
    handoff = config["safety"]["handoff_zone"]
    return not all(
        float(bounds[0]) <= float(value) <= float(bounds[1])
        for value, bounds in zip(
            position,
            (handoff["x_m"], handoff["y_m"], handoff["z_m"]),
        )
    )


def _arm_state(
    controller: Any,
    arm_id: str,
    arm: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    world_position, _ = controller.end_effector_pose(arm_id)
    base_position, base_orientation = controller.end_effector_pose_in_base(arm_id)
    positions = np.asarray(arm.get_joint_positions(), dtype=float)
    velocities = np.asarray(arm.get_joint_velocities(), dtype=float)
    names = _joint_names(arm)
    finger_indices = [
        names.index("panda_finger_joint1"),
        names.index("panda_finger_joint2"),
    ]
    return {
        "tcp_pose_m_rad": [
            *(round(float(item), 8) for item in base_position),
            *(round(item, 8) for item in _quaternion_to_rotvec(base_orientation)),
        ],
        "state": [round(float(item), 8) for item in positions],
        "retreated": _is_retreated(world_position, config),
        "gripper_open": bool(np.mean(positions[finger_indices]) >= 0.02),
        "stationary": bool(
            velocities.size
            and np.all(np.isfinite(velocities))
            and np.max(np.abs(velocities)) <= 1e-3
        ),
    }


def _write_result(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    result_path = path.expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    if (
        not isfinite(args.delta_z_m)
        or abs(args.delta_z_m) < 1e-4
        or abs(args.delta_z_m) > 0.01
    ):
        raise ValueError("--delta-z-m must be finite and within 0.0001..0.01 m")
    if not 1 <= args.duration_ms <= 2_000:
        raise ValueError("--duration-ms must be in [1, 2000]")
    for path in (SOURCE_DIR, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import isaac_compat
    import scene_layout

    config = scene_layout.load_config(args.config)
    errors = scene_layout.validate_scene_config(config)
    if errors:
        raise ValueError("Frozen scene contract failed: " + "; ".join(errors))

    phase = "launch_simulation_app"
    environment = None
    runtime_gate = None
    simulation_app = isaac_compat.launch_simulation_app(headless=args.headless)
    try:
        phase = "verify_isaac_version"
        isaac_version = isaac_compat.require_isaac_sim_51()
        import single_bin_scene_builder
        from isaac_franka_controller import (
            IsaacSimFrankaController,
            _quat_inverse,
            _rotate_vector,
        )
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        from industrial_agent.contracts import ActionStep
        from industrial_agent.environment import execution_guard_digest
        from industrial_agent.isaac_environment import IsaacExecutionEnvironment
        from industrial_agent.isaac_runtime import IsaacMainThreadGate

        phase = "build_scene"
        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(args.franka_usd)
        single_bin_scene_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        isaac_compat.wait_for_stage_loading(simulation_app, timeout_seconds=180.0)
        isaac_compat.save_stage_checked(args.output_scene)

        physics = config["physics"]
        if World.instance():
            World.instance().clear_instance()
        phase = "initialize_world"
        world = World(
            physics_dt=float(physics["physics_dt_s"]),
            rendering_dt=float(physics["rendering_dt_s"]),
            stage_units_in_meters=1.0,
        )
        arms = {
            arm_id: world.scene.add(
                SingleArticulation(
                    prim_path=f"/World/Robots/{arm_id}",
                    name=f"adapter_smoke_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }
        world.reset()
        for _ in range(120):
            world.step(render=not args.headless)

        observation_counter = 0

        phase = "initialize_controller"
        controller = IsaacSimFrankaController(
            world=world,
            arms=arms,
            physics_dt_s=float(physics["physics_dt_s"]),
        )
        runtime_gate = IsaacMainThreadGate()

        def guarded_state() -> dict[str, Any]:
            return {
                "objects": [],
                "robot": {
                    "active_arm": args.arm_id,
                    "arm_a": _arm_state(
                        controller,
                        "Arm_A",
                        arms["Arm_A"],
                        config,
                    ),
                    "arm_b": _arm_state(
                        controller,
                        "Arm_B",
                        arms["Arm_B"],
                        config,
                    ),
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
                    "status": "controller_smoke",
                },
                "quality": {"confidence": 1.0},
            }

        def observation_source() -> dict[str, Any]:
            nonlocal observation_counter
            observation_counter += 1
            return {
                "observation_version": "1.0",
                "observation_id": f"isaac-adapter-smoke-{time.time_ns()}",
                "timestamp_ms": int(time.time() * 1000),
                "camera": {},
                **guarded_state(),
            }

        environment = IsaacExecutionEnvironment(
            observation_source=observation_source,
            state_guard_source=guarded_state,
            control_lease_source=lambda: (
                "A_ONLY" if args.arm_id == "Arm_A" else "B_ONLY"
            ),
            controller=controller,
            runtime_gate=runtime_gate,
            command_ledger_path=args.command_ledger,
            runtime_observe_timeout_s=2.0,
            runtime_action_timeout_s=max(
                5.0,
                args.duration_ms / 1_000.0 + 2.0,
            ),
            runtime_stop_timeout_s=2.0,
        )

        phase = "capture_pre_action_state"
        names = _joint_names(arms[args.arm_id])
        arm_joint_indices = [
            index
            for index, name in enumerate(names)
            if name not in {"panda_finger_joint1", "panda_finger_joint2"}
        ]
        finger_indices = [
            names.index("panda_finger_joint1"),
            names.index("panda_finger_joint2"),
        ]
        before_positions = np.asarray(
            arms[args.arm_id].get_joint_positions(), dtype=float
        )
        before_tcp_position, _ = controller.end_effector_pose(args.arm_id)
        _, base_orientation = arms[args.arm_id].get_world_pose()
        gripper_command = (
            1.0 if np.mean(before_positions[finger_indices]) >= 0.02 else 0.0
        )
        action = ActionStep.from_sequence(
            [0.0, 0.0, args.delta_z_m, 0.0, 0.0, 0.0, gripper_command],
            duration_ms=args.duration_ms,
        )

        def gated_workflow() -> dict[str, Any]:
            nonlocal phase
            phase = "capture_pre_action_observation"
            observation = environment.observe()
            phase = "execute_action"
            after_observation = environment.step(
                action,
                arm_id=args.arm_id,
                control_token=("A_ONLY" if args.arm_id == "Arm_A" else "B_ONLY"),
                command_id=f"isaac-adapter-smoke-{time.time_ns()}",
                expected_observation_id=str(observation["observation_id"]),
                expected_state_digest=execution_guard_digest(observation),
            )
            phase = "safe_stop"
            receipt = environment.safe_stop("Isaac adapter smoke completed")
            return {
                "observation": observation,
                "after_observation": after_observation,
                "receipt": receipt,
            }

        gated_result = runtime_gate.run_worker_until_complete(
            gated_workflow,
            idle_callback=simulation_app.update,
        )
        observation = gated_result["observation"]
        after_observation = gated_result["after_observation"]
        receipt = gated_result["receipt"]
        after_positions = np.asarray(
            arms[args.arm_id].get_joint_positions(), dtype=float
        )
        after_tcp_position, _ = controller.end_effector_pose(args.arm_id)
        arm_joint_delta_norm = float(
            np.linalg.norm(
                after_positions[arm_joint_indices] - before_positions[arm_joint_indices]
            )
        )
        finger_delta_norm = float(
            np.linalg.norm(
                after_positions[finger_indices] - before_positions[finger_indices]
            )
        )
        tcp_delta_world = after_tcp_position - before_tcp_position
        tcp_delta_base = _rotate_vector(
            _quat_inverse(np.asarray(base_orientation, dtype=float)),
            tcp_delta_world,
        )
        tcp_delta_norm_m = float(np.linalg.norm(tcp_delta_world))
        if arm_joint_delta_norm <= 1e-7:
            raise RuntimeError(
                "Adapter returned without a measurable seven-arm-joint change"
            )
        if tcp_delta_norm_m <= 1e-5:
            raise RuntimeError("Adapter returned without a measurable TCP translation")
        expected_tcp_delta = np.asarray([0.0, 0.0, args.delta_z_m], dtype=float)
        tcp_tolerance_m = max(0.001, abs(args.delta_z_m) * 0.2)
        if not np.allclose(
            tcp_delta_base,
            expected_tcp_delta,
            atol=tcp_tolerance_m,
            rtol=0.0,
        ):
            raise RuntimeError(
                "TCP translation did not match the commanded base-frame delta "
                f"within {tcp_tolerance_m:.6f} m"
            )
        if not receipt.confirmed:
            raise RuntimeError(f"Safe-stop readback was not confirmed: {receipt!r}")

        result = {
            "status": "PASS",
            "isaac_sim_version": isaac_version,
            "arm_id": args.arm_id,
            "delta_z_m": args.delta_z_m,
            "duration_ms": args.duration_ms,
            "arm_joint_delta_norm": arm_joint_delta_norm,
            "finger_delta_norm": finger_delta_norm,
            "tcp_delta_norm_m": tcp_delta_norm_m,
            "tcp_delta_base_m": [float(item) for item in tcp_delta_base],
            "before_observation_id": observation["observation_id"],
            "after_observation_id": after_observation["observation_id"],
            "safe_stop_confirmed": receipt.confirmed,
        }
        _write_result(args.result_file, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        stop_result: dict[str, Any] | None = None
        if environment is not None:
            try:
                emergency_receipt = environment.safe_stop(
                    f"Isaac adapter smoke failed during {phase}"
                )
                stop_result = {
                    "confirmed": emergency_receipt.confirmed,
                    "stop_epoch": emergency_receipt.stop_epoch,
                }
            except BaseException as stop_exc:
                stop_result = {
                    "confirmed": False,
                    "error_type": type(stop_exc).__name__,
                    "error": str(stop_exc),
                }
        result = {
            "status": "FAIL",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "emergency_stop": stop_result,
        }
        _write_result(args.result_file, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if runtime_gate is not None:
            runtime_gate.close("Isaac adapter smoke is shutting down")
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
