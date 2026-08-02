"""Interactive, step-wise keyboard smoke for the frozen Isaac Sim scene.

This proves Member B's control and three-camera observation boundary.  The
JSONL written here is explicitly smoke evidence, not a Canonical Episode and
must not be used as the Member C production Recorder output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread
import time
import traceback
from typing import Any
from uuid import uuid4

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v1.json"
DEFAULT_SCENE = SCRIPT_DIR / "generated" / "single_bin_scene_v1.usda"
DEFAULT_ARTIFACT_DIR = REPOSITORY_ROOT / "artifacts" / "keyboard-teleop-smoke"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step-wise keyboard smoke with three RGB CAS streams."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--franka-usd")
    parser.add_argument("--arm-id", choices=("Arm_A", "Arm_B"), default="Arm_A")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--translation-step-m", type=float, default=0.005)
    parser.add_argument("--rotation-step-deg", type=float, default=2.0)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=50,
        help="Hard safety cap for one smoke session.",
    )
    return parser.parse_args()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _start_terminal_reader(output: Queue[str]) -> Thread:
    def read_commands() -> None:
        while True:
            try:
                value = input("teleop> ")
            except EOFError:
                output.put("x")
                return
            output.put(value)
            if value.strip().lower() in {"x", "esc", "quit", "exit"}:
                return

    thread = Thread(target=read_commands, name="teleop-terminal", daemon=True)
    thread.start()
    return thread


def main() -> int:
    args = _parse_args()
    if not 1 <= args.max_actions <= 500:
        raise ValueError("--max-actions must be in [1, 500]")
    if args.translation_step_m <= 0.0 or args.translation_step_m > 0.01:
        raise ValueError("--translation-step-m must be in (0, 0.01]")
    if args.rotation_step_deg <= 0.0 or args.rotation_step_deg > 5.0:
        raise ValueError("--rotation-step-deg must be in (0, 5]")
    for path in (REPOSITORY_ROOT, SOURCE_DIR, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import isaac_compat
    import scene_layout

    config = scene_layout.load_config(args.config)
    errors = scene_layout.validate_scene_config(config)
    if errors:
        raise ValueError("Frozen scene contract failed: " + "; ".join(errors))

    artifact_dir = args.artifact_dir.expanduser().resolve()
    trace_path = artifact_dir / "teleop-smoke-actions.jsonl"
    result_path = artifact_dir / "result.json"
    command_ledger = artifact_dir / "command-ids.jsonl"
    cas_root = artifact_dir / "cas"
    session_id = f"keyboard-smoke-{uuid4()}"
    phase = "launch_simulation_app"
    environment = None
    runtime_gate = None
    rgb_pipeline = None
    simulation_app = isaac_compat.launch_simulation_app(headless=False)
    action_count = 0
    checkpoint_count = 0
    try:
        phase = "verify_isaac_version"
        isaac_version = isaac_compat.require_isaac_sim_51()
        import single_bin_scene_builder
        from isaac_franka_controller import IsaacSimFrankaController
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        from industrial_agent.environment import execution_guard_digest
        from industrial_agent.image_cas import ImageCas, ImageCasConfig
        from industrial_agent.isaac_environment import IsaacExecutionEnvironment
        from industrial_agent.isaac_runtime import IsaacMainThreadGate
        from industrial_agent.observation import ObservationGateway
        from simulation.isaac_rgb_pipeline import IsaacRgbObservationPipeline
        from simulation.keyboard_teleop import KeyboardTeleopMapper
        from simulation.rgb_cas_bridge import IsaacRgbCasPublisher
        from simulation.run_isaac_adapter_smoke import _arm_state

        phase = "build_scene"
        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(args.franka_usd)
        single_bin_scene_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        default_light = stage.GetPrimAtPath("/Environment/defaultLight")
        if default_light and default_light.IsValid():
            raise RuntimeError(
                "unexpected New Stage /Environment/defaultLight; frozen scene "
                "must use only /World/Lighting"
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
                    name=f"keyboard_smoke_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }
        world.reset()
        for _ in range(120):
            world.step(render=True)

        phase = "initialize_controller_and_rgb"
        controller = IsaacSimFrankaController(
            world=world,
            arms=arms,
            physics_dt_s=float(physics["physics_dt_s"]),
        )
        runtime_gate = IsaacMainThreadGate()
        image_cas = ImageCas(ImageCasConfig(root=cas_root))
        image_cas.assert_ready(writable=True)
        publisher = IsaacRgbCasPublisher.from_scene_config(image_cas, config)
        rgb_pipeline = IsaacRgbObservationPipeline(
            simulation_app=simulation_app,
            scene_config=config,
            publisher=publisher,
        )
        observation_gateway = ObservationGateway()
        observation_counter = 0
        last_timestamp_ms = -1

        def guarded_state() -> dict[str, Any]:
            return {
                "objects": [],
                "robot": {
                    "active_arm": args.arm_id,
                    "arm_a": _arm_state(controller, "Arm_A", arms["Arm_A"], config),
                    "arm_b": _arm_state(controller, "Arm_B", arms["Arm_B"], config),
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
                    "status": "keyboard_teleop_smoke",
                },
                "quality": {"confidence": 1.0},
            }

        def observation_source() -> dict[str, Any]:
            nonlocal observation_counter, last_timestamp_ms
            observation_counter += 1
            timestamp_ms = max(int(time.time() * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            raw = {
                "observation_version": "1.0",
                "observation_id": f"{session_id}-obs-{observation_counter:06d}",
                "timestamp_ms": timestamp_ms,
                "camera": rgb_pipeline.capture(args.arm_id),
                **guarded_state(),
            }
            observation_gateway.ingest_online(raw)
            return raw

        environment = IsaacExecutionEnvironment(
            observation_source=observation_source,
            state_guard_source=guarded_state,
            control_lease_source=lambda: (
                "A_ONLY" if args.arm_id == "Arm_A" else "B_ONLY"
            ),
            controller=controller,
            runtime_gate=runtime_gate,
            command_ledger_path=command_ledger,
            runtime_observe_timeout_s=5.0,
            runtime_action_timeout_s=10.0,
            runtime_stop_timeout_s=2.0,
        )
        active_state = guarded_state()["robot"][args.arm_id.lower()]
        mapper = KeyboardTeleopMapper(
            translation_step_m=args.translation_step_m,
            rotation_step_rad=np.deg2rad(args.rotation_step_deg),
            duration_ms=100,
            gripper_open=bool(active_state["gripper_open"]),
        )
        _append_jsonl(
            trace_path,
            {
                "record_type": "smoke_session_start",
                "smoke_only": True,
                "not_canonical_episode": True,
                "session_id": session_id,
                "scene_id": config["scene_id"],
                "arm_id": args.arm_id,
                "action_order": [
                    "dx_m",
                    "dy_m",
                    "dz_m",
                    "dax_rad",
                    "day_rad",
                    "daz_rad",
                    "gripper_norm",
                ],
                "timestamp_ms": int(time.time() * 1000),
            },
        )

        phase = "interactive_teleop"
        print("\n键采冒烟已就绪。每次输入一个键后按 Enter。")
        print(mapper.help_text())
        print("先用 W/A/Q 等小步移动；确认画面正常后按 P 留检查点，按 X 退出。")
        command_queue: Queue[str] = Queue()
        _start_terminal_reader(command_queue)
        running = True
        while running and simulation_app.is_running():
            try:
                raw_key = command_queue.get_nowait()
            except Empty:
                world.step(render=True)
                continue
            try:
                command = mapper.parse(raw_key)
            except ValueError as exc:
                print(f"无效按键：{exc}")
                continue
            if command.kind == "help":
                print(mapper.help_text())
                continue
            if command.kind == "quit":
                running = False
                continue
            if command.kind == "reset":
                world.reset()
                for _ in range(120):
                    world.step(render=True)
                reset_state = guarded_state()["robot"][args.arm_id.lower()]
                mapper.set_gripper_open(bool(reset_state["gripper_open"]))
                _append_jsonl(
                    trace_path,
                    {
                        "record_type": "scene_reset",
                        "smoke_only": True,
                        "session_id": session_id,
                        "timestamp_ms": int(time.time() * 1000),
                    },
                )
                print("场景已重置。")
                continue
            if command.kind == "checkpoint":
                checkpoint_count += 1

                def capture_checkpoint() -> dict[str, Any]:
                    return dict(environment.observe())

                observation = runtime_gate.run_worker_until_complete(
                    capture_checkpoint,
                    idle_callback=simulation_app.update,
                )
                _append_jsonl(
                    trace_path,
                    {
                        "record_type": "smoke_checkpoint",
                        "smoke_only": True,
                        "session_id": session_id,
                        "checkpoint_index": checkpoint_count,
                        "observation": observation,
                    },
                )
                print(f"检查点 {checkpoint_count} 已写入。")
                continue
            if action_count >= args.max_actions:
                print("达到 max-actions 安全上限，正在安全退出。")
                running = False
                continue
            if command.action is None:
                raise RuntimeError("action command contains no ActionStep")

            def execute_one_action() -> tuple[dict[str, Any], dict[str, Any]]:
                before = dict(environment.observe())
                after = dict(
                    environment.step(
                        command.action,
                        arm_id=args.arm_id,
                        control_token=(
                            "A_ONLY" if args.arm_id == "Arm_A" else "B_ONLY"
                        ),
                        command_id=f"{session_id}-cmd-{uuid4()}",
                        expected_observation_id=str(before["observation_id"]),
                        expected_state_digest=execution_guard_digest(before),
                    )
                )
                return before, after

            before, after = runtime_gate.run_worker_until_complete(
                execute_one_action,
                idle_callback=simulation_app.update,
            )
            action_count += 1
            _append_jsonl(
                trace_path,
                {
                    "record_type": "teleop_smoke_action",
                    "smoke_only": True,
                    "session_id": session_id,
                    "action_index": action_count,
                    "key": command.key,
                    "description": command.description,
                    "action": command.action.to_dict(),
                    "before_observation": before,
                    "after_observation": after,
                },
            )
            print(
                f"动作 {action_count}: {command.description}; "
                f"after={after['observation_id']}"
            )

        phase = "safe_stop"

        def stop_workflow() -> Any:
            return environment.safe_stop("keyboard teleop smoke completed")

        receipt = runtime_gate.run_worker_until_complete(
            stop_workflow,
            idle_callback=simulation_app.update,
        )
        if not receipt.confirmed:
            raise RuntimeError("safe-stop readback was not confirmed")
        result = {
            "status": "PASS",
            "smoke_only": True,
            "not_canonical_episode": True,
            "session_id": session_id,
            "isaac_sim_version": isaac_version,
            "scene_id": config["scene_id"],
            "arm_id": args.arm_id,
            "action_count": action_count,
            "checkpoint_count": checkpoint_count,
            "three_rgb_cas_streams": True,
            "online_observation_validated": True,
            "safe_stop_confirmed": True,
            "trace_path": str(trace_path),
            "cas_root": str(cas_root),
        }
        _write_result(result_path, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        stop_result: dict[str, Any] | None = None
        if environment is not None:
            try:
                receipt = environment.safe_stop(
                    f"keyboard teleop smoke failed during {phase}"
                )
                stop_result = {"confirmed": receipt.confirmed}
            except BaseException as stop_exc:
                stop_result = {
                    "confirmed": False,
                    "error_type": type(stop_exc).__name__,
                    "error": str(stop_exc),
                }
        result = {
            "status": "FAIL",
            "smoke_only": True,
            "session_id": session_id,
            "phase": phase,
            "action_count": action_count,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "emergency_stop": stop_result,
        }
        _write_result(result_path, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if rgb_pipeline is not None:
            try:
                rgb_pipeline.close()
            except BaseException:
                traceback.print_exc()
        if runtime_gate is not None:
            runtime_gate.close("keyboard teleop smoke is shutting down")
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
