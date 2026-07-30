"""Exercise the real Isaac execution adapter on the frozen dual-Franka scene.

This is a controller smoke test, not a production observation pipeline.  Its
observation source contains live articulation telemetry but deliberately no
camera evidence.  A successful run proves that an ``ActionStep`` crosses the
production adapter and reaches the Isaac Sim 5.1 Franka controller.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Optional JSON path for preserving the smoke-test result.",
    )
    return parser.parse_args()


def _joint_state(arm: Any) -> dict[str, Any]:
    names = getattr(arm, "dof_names", None)
    if names is None:
        names = getattr(arm, "joint_names", None)
    return {
        "joint_names": [str(item) for item in names],
        "joint_positions_rad": [
            float(item) for item in np.asarray(arm.get_joint_positions())
        ],
        "joint_velocities_rad_s": [
            float(item) for item in np.asarray(arm.get_joint_velocities())
        ],
        "retreated": True,
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
    simulation_app = isaac_compat.launch_simulation_app(headless=args.headless)
    try:
        phase = "verify_isaac_version"
        isaac_version = isaac_compat.require_isaac_sim_51()
        import single_bin_scene_builder
        from isaac_franka_controller import IsaacSimFrankaController
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        from industrial_agent.contracts import ActionStep
        from industrial_agent.environment import execution_guard_digest
        from industrial_agent.isaac_environment import IsaacExecutionEnvironment

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

        def observation_source() -> dict[str, Any]:
            nonlocal observation_counter
            observation_counter += 1
            return {
                "observation_id": f"isaac-adapter-smoke-{time.time_ns()}",
                "timestamp_ms": int(time.time() * 1000),
                "camera": {},
                "objects": [],
                "robot": {
                    "active_arm": args.arm_id,
                    "arm_a": _joint_state(arms["Arm_A"]),
                    "arm_b": _joint_state(arms["Arm_B"]),
                },
                "safety": {
                    "emergency_stop": False,
                    "protective_stop": False,
                    "system_fault": None,
                },
                "task": {"kind": "controller_smoke"},
                "quality": {"live_robot_telemetry": True},
                "smoke_observation_sequence": observation_counter,
            }

        phase = "initialize_controller"
        controller = IsaacSimFrankaController(
            world=world,
            arms=arms,
            physics_dt_s=float(physics["physics_dt_s"]),
        )
        environment = IsaacExecutionEnvironment(
            observation_source=observation_source,
            controller=controller,
        )

        phase = "capture_pre_action_state"
        before_positions = np.asarray(
            arms[args.arm_id].get_joint_positions(), dtype=float
        )
        observation = environment.observe()
        action = ActionStep.from_sequence(
            [0.0, 0.0, args.delta_z_m, 0.0, 0.0, 0.0, 1.0],
            duration_ms=args.duration_ms,
        )
        phase = "execute_action"
        after_observation = environment.step(
            action,
            arm_id=args.arm_id,
            control_token="A_ONLY" if args.arm_id == "Arm_A" else "B_ONLY",
            command_id=f"isaac-adapter-smoke-{time.time_ns()}",
            expected_observation_id=str(observation["observation_id"]),
            expected_state_digest=execution_guard_digest(observation),
        )
        after_positions = np.asarray(
            arms[args.arm_id].get_joint_positions(), dtype=float
        )
        joint_delta_norm = float(np.linalg.norm(after_positions - before_positions))
        phase = "safe_stop"
        receipt = environment.safe_stop("Isaac adapter smoke completed")
        if joint_delta_norm <= 1e-7:
            raise RuntimeError(
                "Adapter returned without a measurable Franka joint-state change"
            )
        if not receipt.confirmed:
            raise RuntimeError(f"Safe-stop readback was not confirmed: {receipt!r}")

        result = {
            "status": "PASS",
            "isaac_sim_version": isaac_version,
            "arm_id": args.arm_id,
            "delta_z_m": args.delta_z_m,
            "duration_ms": args.duration_ms,
            "joint_delta_norm": joint_delta_norm,
            "before_observation_id": observation["observation_id"],
            "after_observation_id": after_observation["observation_id"],
            "safe_stop_confirmed": receipt.confirmed,
        }
        _write_result(args.result_file, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        result = {
            "status": "FAIL",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_result(args.result_file, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
