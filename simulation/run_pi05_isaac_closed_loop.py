"""Run the π0.5 Isaac closed-loop evaluation entry point.

This role-E runner owns the Isaac main thread and connects the existing
components without duplicating their contracts:

    Isaac RGB/CAS observation -> Pi05Adapter -> POST /v1/infer
    -> validated 7-D ActionChunk -> IsaacExecutionEnvironment -> next observation

It is deliberately an evaluation runner, not the production Supervisor lifecycle
runner.  It reports a closed-loop pass after the requested action budget is
completed; task-terminal success is reported only when a detector-backed V2
task state says ``terminal`` with sufficient evidence. No ground-truth state is added to the online
observation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import importlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence
from uuid import uuid4


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
DEFAULT_SCENE_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"
DEFAULT_AGENT_CONFIG = REPOSITORY_ROOT / "configs" / "agent.default.json"
DEFAULT_TASK = REPOSITORY_ROOT / "configs" / "task.v2.p01-to-s11.example.json"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "pi05-isaac-closed-loop"
logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the π0.5 Isaac camera -> inference -> 7-D action loop."
    )
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--agent-config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--franka-usd")
    parser.add_argument("--output-scene", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--command-ledger", type=Path)
    parser.add_argument("--arm-id", choices=("Arm_A",), default="Arm_A")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--runtime-mode",
        choices=("direct", "supervisor"),
        default="direct",
        help="Use the legacy direct adapter loop or the formal V2 Supervisor.",
    )
    parser.add_argument("--deadline-ms", type=int, default=15_000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--ik-backend", choices=("lula", "pink"), default="lula")
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help="Return failure unless detector-backed V2 task state reaches terminal.",
    )
    parser.add_argument(
        "--task-state-factory",
        help=(
            "Optional module:callable factory returning a zero-argument, sensor-only "
            "V2 task-state provider. Required with --require-terminal."
        ),
    )
    parser.add_argument("--result-file", type=Path)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_task_state_provider(
    reference: str,
    *,
    task_spec: Any,
    controller: Any,
    arms: Mapping[str, Any],
    scene_config: Mapping[str, Any],
):
    """Load the deployment-owned sensor verifier without importing it globally."""

    module_name, separator, attribute_name = reference.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute_name
        or ":" in attribute_name
    ):
        raise ValueError(
            "task-state factory must use exact form 'module.path:callable'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError("task-state factory reference is not callable")
    provider = factory(
        task_spec=task_spec,
        controller=controller,
        arms=arms,
        scene_config=scene_config,
    )
    if not callable(provider):
        raise TypeError("task-state factory must return a zero-argument callable")
    return provider


def build_task_state(task_spec: Any | None = None) -> dict[str, Any]:
    """Return the non-GT task fields required by the online contract.

    A deployment-specific online task-state provider may replace the static
    ``terminal=false`` value with detector-backed facts. It must never read
    Isaac GT coordinates.
    """

    task_id = getattr(task_spec, "task_id", "")
    target_object = getattr(task_spec, "target_object", "")
    target_slot = getattr(task_spec, "target_slot", None)

    return {
        "task_id": task_id,
        "target_object_id": target_object,
        "target_slot_id": target_slot,
        "status": "ACTIVE",
        "terminal": False,
        "terminal_confidence": 0.0,
        "verification_votes": 0,
    }


def build_observation(
    *,
    camera: Mapping[str, Any],
    robot: Mapping[str, Any],
    task: Mapping[str, Any],
    observation_id: str,
    timestamp_ms: int,
) -> dict[str, Any]:
    """Build the allow-listed online observation envelope."""

    return {
        "observation_version": "2.0",
        "observation_id": observation_id,
        "timestamp_ms": timestamp_ms,
        "camera": dict(camera),
        "objects": [],
        "robot": dict(robot),
        "safety": {
            "emergency_stop": False,
            "protective_stop": False,
            "system_fault": None,
        },
        "task": dict(task),
        "quality": {"confidence": 1.0},
    }


def _safety_policy_from_config(raw_config: Mapping[str, Any]):
    from industrial_agent.safety import SafetyPolicy

    raw = raw_config.get("safety")
    if not isinstance(raw, Mapping):
        raise ValueError("agent config must contain safety")
    workspace = raw.get("workspace_by_arm")
    if not isinstance(workspace, Mapping):
        raise ValueError("agent config safety.workspace_by_arm is required")

    def bounds(arm_id: str, key: str) -> tuple[float, float, float]:
        arm = workspace.get(arm_id)
        if not isinstance(arm, Mapping) or not isinstance(arm.get(key), list):
            raise ValueError(f"missing safety workspace {arm_id}.{key}")
        values = tuple(float(item) for item in arm[key])
        if len(values) != 3:
            raise ValueError(f"safety workspace {arm_id}.{key} must have 3 values")
        return values  # type: ignore[return-value]

    limits = raw.get("axis_abs_limits")
    if not isinstance(limits, list):
        raise ValueError("agent config safety.axis_abs_limits is required")
    return SafetyPolicy(
        axis_abs_limits=tuple(float(item) for item in limits),
        arm_a_workspace_min_m=bounds("Arm_A", "min_m"),
        arm_a_workspace_max_m=bounds("Arm_A", "max_m"),
        arm_b_workspace_min_m=bounds("Arm_B", "min_m"),
        arm_b_workspace_max_m=bounds("Arm_B", "max_m"),
        max_chunk_steps=int(raw.get("max_chunk_steps", 32)),
    )


def _pause_physics_world(*, world: Any) -> None:
    """Pause Isaac physics on the owner thread before stable observation work."""

    try:
        if world.is_playing():
            world.pause()
    except (AttributeError, RuntimeError):
        logger.exception("Failed to pause Isaac physics")
        raise


def _update_ui_without_advancing_physics(*, world: Any, simulation_app: Any) -> None:
    """Keep Kit responsive while holding the physics world during inference."""

    _pause_physics_world(world=world)
    simulation_app.update()


def _capture_stable_observation_inputs(
    *,
    world: Any,
    capture_camera: Callable[[], Mapping[str, Any]],
    capture_state: Callable[[], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Capture camera and guarded state without allowing physics to drift."""

    _pause_physics_world(world=world)

    try:
        camera = capture_camera()
    except (RuntimeError, OSError, TimeoutError):
        logger.exception("Failed to capture Isaac RGB observation")
        raise

    # Camera rendering may advance or otherwise change the Isaac timeline.
    _pause_physics_world(world=world)

    try:
        state = capture_state()
    except (RuntimeError, OSError, TimeoutError):
        logger.exception("Failed to capture Isaac guarded state")
        raise

    if world.is_playing():
        raise RuntimeError("Isaac physics resumed during stable observation capture")

    return camera, state


def _run_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    for path in (REPOSITORY_ROOT, SOURCE_DIR, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import isaac_compat
    import scene_layout
    from industrial_agent.contracts import TaskSchema
    from industrial_agent.executor import ExecutionContext, Pi05Adapter
    from industrial_agent.http_transport import BoundedHTTPTransport
    from industrial_agent.image_cas import ImageCas, ImageCasConfig
    from industrial_agent.v2_observation import V2ObservationGateway
    from industrial_agent.safety import ActionSafetyValidator
    from industrial_agent.v2_task_profile import require_formal_v2_task
    from industrial_agent.isaac_environment import IsaacExecutionEnvironment
    from industrial_agent.isaac_runtime import IsaacMainThreadGate
    from industrial_agent.environment import execution_guard_digest

    scene_config_path = args.scene_config.expanduser().resolve()
    scene_config = scene_layout.load_config(scene_config_path)
    from simulation.v2_scene_contract import require_valid_config

    require_valid_config(scene_config)
    agent_config = _read_json(args.agent_config)
    if agent_config.get("profile_id") != "single_bin_manual_industrial_v2":
        raise ValueError(
            "role-E closed-loop runner requires the V2 agent config; "
            "use configs/agent.default.json"
        )
    task_payload = _read_json(args.task)
    task = TaskSchema.from_dict(task_payload)
    task_spec = require_formal_v2_task(task.task_id)
    if task.instruction != task_spec.instruction:
        raise ValueError("task instruction does not match the frozen V2 task catalog")
    task = TaskSchema.from_dict(
        {
            **task.to_dict(),
            "target_object": task_spec.target_object,
            "target_location": task_spec.target_slot,
            "metadata": {
                **dict(task.metadata),
                "profile_id": "single_bin_manual_industrial_v2",
                "subtask_id": task_spec.task_id,
            },
        }
    )

    adapter = None
    if args.runtime_mode == "direct":
        raw_pi05 = agent_config.get("executors", {}).get("pi05")
        if not isinstance(raw_pi05, Mapping):
            raise ValueError("agent config executors.pi05 is required")
        transport = BoundedHTTPTransport(str(raw_pi05["base_url"]))
        adapter = Pi05Adapter(
            transport,
            checkpoint_sha=str(raw_pi05["checkpoint_sha"]),
            norm_stats_sha=str(raw_pi05["norm_stats_sha"]),
        )
        if not adapter.health():
            raise RuntimeError("π0.5 /health is not ready or its identity is invalid")

    artifact_root = args.artifact_root.expanduser().resolve()
    cas_root = artifact_root / "cas"
    ledger_path = (
        args.command_ledger.expanduser().resolve()
        if args.command_ledger is not None
        else artifact_root / "command-ids.jsonl"
    )
    output_scene = (
        args.output_scene.expanduser().resolve()
        if args.output_scene is not None
        else artifact_root / "single_bin_scene_v2.usda"
    )
    result_path = (
        args.result_file.expanduser().resolve()
        if args.result_file is not None
        else artifact_root / "result.json"
    )

    simulation_app = isaac_compat.launch_simulation_app(headless=args.headless)
    rgb_pipeline = None
    environment = None
    gate = None
    phase = "launch_simulation_app"
    episode_id = f"pi05-isaac-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    records: list[dict[str, Any]] = []
    terminal = False
    supervisor_stop_handled = False
    try:
        phase = "load_isaac_runtime"
        from simulation.isaac_rgb_pipeline import IsaacRgbObservationPipeline
        from simulation.rgb_cas_bridge import IsaacRgbCasPublisher
        from simulation.run_isaac_adapter_smoke import _arm_state
        from simulation.run_v2_keyboard_collection import _apply_home
        from simulation.single_bin_scene_v2_builder import build_scene
        from isaac_franka_controller import IsaacSimFrankaController
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        phase = "verify_isaac_version"
        isaac_version = isaac_compat.require_isaac_sim_51()
        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(args.franka_usd)
        build_scene(
            stage,
            scene_config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        isaac_compat.wait_for_stage_loading(simulation_app, timeout_seconds=180.0)
        isaac_compat.save_stage_checked(output_scene)

        physics = scene_config["physics"]
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
                    name=f"pi05_closed_loop_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }
        world.reset()
        _apply_home(world, arms, scene_config)

        controller = IsaacSimFrankaController(
            world=world,
            arms=arms,
            physics_dt_s=float(physics["physics_dt_s"]),
            virtual_tcp_fingertip_frame_names=(
                "panda_leftfingertip",
                "panda_rightfingertip",
            ),
            ik_backend=args.ik_backend,
        )
        image_cas = ImageCas(ImageCasConfig(root=cas_root))
        image_cas.assert_ready(writable=True)
        publisher = IsaacRgbCasPublisher.from_scene_config(image_cas, scene_config)
        rgb_pipeline = IsaacRgbObservationPipeline(
            simulation_app=simulation_app,
            scene_config=scene_config,
            publisher=publisher,
        )
        gate = IsaacMainThreadGate()
        if args.task_state_factory:
            task_state_provider = resolve_task_state_provider(
                args.task_state_factory,
                task_spec=task_spec,
                controller=controller,
                arms=arms,
                scene_config=scene_config,
            )
        else:

            def task_state_provider() -> dict[str, Any]:
                return build_task_state(task_spec)

        observation_counter = 0

        def guarded_state() -> dict[str, Any]:
            return {
                "objects": [],
                "robot": {
                    "active_arm": args.arm_id,
                    "arm_a": _arm_state(
                        controller,
                        "Arm_A",
                        arms["Arm_A"],
                        scene_config,
                        continuous_state=True,
                    ),
                    "arm_b": _arm_state(
                        controller,
                        "Arm_B",
                        arms["Arm_B"],
                        scene_config,
                        continuous_state=True,
                    ),
                },
                "safety": {
                    "emergency_stop": False,
                    "protective_stop": False,
                    "system_fault": None,
                },
                "task": dict(task_state_provider()),
                "quality": {"confidence": 1.0},
            }

        def observation_source() -> dict[str, Any]:
            nonlocal observation_counter
            camera, state = _capture_stable_observation_inputs(
                world=world,
                capture_camera=lambda: rgb_pipeline.capture(args.arm_id),
                capture_state=guarded_state,
            )
            observation_counter += 1
            return build_observation(
                camera=camera,
                robot=state["robot"],
                task=state["task"],
                observation_id=f"{episode_id}-obs-{observation_counter:06d}",
                timestamp_ms=int(time.time() * 1000),
            )

        environment = IsaacExecutionEnvironment(
            observation_source=observation_source,
            state_guard_source=guarded_state,
            control_lease_source=lambda: "A_ONLY",
            controller=controller,
            runtime_gate=gate,
            command_ledger_path=ledger_path,
            runtime_observe_timeout_s=2.0,
            runtime_action_timeout_s=max(10.0, args.deadline_ms / 1000 + 5.0),
            runtime_stop_timeout_s=2.0,
        )
        gateway = V2ObservationGateway()
        safety = ActionSafetyValidator(_safety_policy_from_config(agent_config))
        executor_task = TaskSchema.from_dict(
            {
                **task.to_dict(),
                "task_id": task.task_id,
            }
        )
        run_id = episode_id

        if args.runtime_mode == "supervisor":
            from industrial_agent.supervisor_main import run_result_to_dict
            from simulation.pi05_isaac_supervisor_runtime import (
                run_supervisor_runtime,
            )

            phase = "supervisor"
            try:
                runtime_report = run_supervisor_runtime(
                    config=agent_config,
                    task=executor_task,
                    environment=environment,
                    gate=gate,
                    max_steps=args.max_steps,
                    idle_callback=lambda: _update_ui_without_advancing_physics(
                        world=world,
                        simulation_app=simulation_app,
                    ),
                )
            finally:
                # The runtime owns the idempotent safe-stop boundary.  The
                # outer exception handler must not issue a second Isaac stop.
                supervisor_stop_handled = True

            supervisor_result = run_result_to_dict(runtime_report.run_result)
            terminal = bool(runtime_report.run_result.success)
            if not runtime_report.safe_stop_confirmed:
                status = "SAFE_STOP_UNCONFIRMED"
            elif runtime_report.run_result.success:
                status = "TASK_SUCCEEDED"
            else:
                status = "SAFE_STOPPED"
            if (
                args.require_terminal
                and not terminal
                and runtime_report.safe_stop_confirmed
            ):
                status = "TERMINAL_CONDITION_NOT_REACHED"
            result = {
                "status": status,
                "episode_id": episode_id,
                "isaac_sim_version": isaac_version,
                "task_id": executor_task.task_id,
                "subtask_id": task_spec.task_id,
                "arm_id": args.arm_id,
                "runtime_mode": args.runtime_mode,
                "requested_steps": args.max_steps,
                "executed_steps": len(runtime_report.actions),
                "terminal": terminal,
                "safe_stop_confirmed": runtime_report.safe_stop_confirmed,
                "failure_code": supervisor_result["failure_code"],
                "message": supervisor_result["message"],
                "supervisor": supervisor_result,
                "steps": runtime_report.action_dicts(),
            }
            _write_json(result_path, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return result

        def worker_loop() -> dict[str, Any]:
            nonlocal terminal
            if adapter is None:
                raise RuntimeError("direct runtime adapter was not initialized")
            raw = environment.observe()
            observation = gateway.ingest_online(raw)
            for step_id in range(args.max_steps):
                infer_started = time.perf_counter()
                context = ExecutionContext(
                    run_id=run_id,
                    strategy_attempt=0,
                    replan_index=0,
                    step_id=step_id,
                    timeout_ms=args.deadline_ms,
                    original_instruction=task.instruction,
                )
                chunk = adapter.plan(executor_task, observation, context)
                inference_ms = (time.perf_counter() - infer_started) * 1000.0
                decision = safety.validate_and_limit(
                    chunk,
                    observation,
                    arm_id=args.arm_id,
                    control_token="A_ONLY",
                )
                if not decision.accepted or decision.chunk is None:
                    raise RuntimeError(
                        f"unsafe π0.5 action rejected: {decision.code.value}: "
                        f"{decision.reason}"
                    )
                action = decision.chunk.steps[0]
                command_id = f"{episode_id}-action-{step_id:06d}"
                next_raw = environment.step(
                    action,
                    arm_id=args.arm_id,
                    control_token="A_ONLY",
                    command_id=command_id,
                    expected_observation_id=observation.observation_id,
                    expected_state_digest=execution_guard_digest(observation.data),
                )
                next_observation = gateway.ingest_online(next_raw)
                next_task = next_observation.data.get("task", {})
                terminal = (
                    isinstance(next_task, Mapping)
                    and next_task.get("terminal") is True
                    and next_task.get("status") == "SUCCEEDED"
                    and float(next_task.get("terminal_confidence", 0.0)) >= 0.6
                    and int(next_task.get("verification_votes", 0)) >= 2
                )
                records.append(
                    {
                        "step_id": step_id,
                        "observation_id": observation.observation_id,
                        "next_observation_id": next_observation.observation_id,
                        "command_id": command_id,
                        "chunk_id": decision.chunk.chunk_id,
                        "proposed_steps": len(decision.chunk.steps),
                        "executed_steps": 1,
                        "inference_ms": round(inference_ms, 3),
                        "limited_axes": list(decision.limited_axes),
                        "action_7d": list(action.values),
                    }
                )
                observation = next_observation
                if terminal:
                    break
            return {
                "terminal": terminal,
                "last_observation_id": observation.observation_id,
            }

        phase = "closed_loop"
        loop_result = gate.run_worker_until_complete(
            worker_loop,
            idle_callback=lambda: _update_ui_without_advancing_physics(
                world=world,
                simulation_app=simulation_app,
            ),
        )
        phase = "safe_stop"
        receipt = environment.safe_stop("π0.5 Isaac closed-loop evaluation completed")
        status = "TASK_SUCCEEDED" if loop_result["terminal"] else "CLOSED_LOOP_PASS"
        if args.require_terminal and not loop_result["terminal"]:
            status = "TERMINAL_CONDITION_NOT_REACHED"
        result = {
            "status": status,
            "episode_id": episode_id,
            "isaac_sim_version": isaac_version,
            "task_id": executor_task.task_id,
            "subtask_id": task_spec.task_id,
            "arm_id": args.arm_id,
            "runtime_mode": args.runtime_mode,
            "requested_steps": args.max_steps,
            "executed_steps": len(records),
            "terminal": bool(loop_result["terminal"]),
            "safe_stop_confirmed": bool(receipt.confirmed),
            "steps": records,
        }
        if not receipt.confirmed:
            result["status"] = "SAFE_STOP_UNCONFIRMED"
        _write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except BaseException as exc:
        failure_phase = phase
        safe_stop_confirmed: bool | None = None
        if environment is not None and not supervisor_stop_handled:
            try:
                phase = "safe_stop"
                receipt = environment.safe_stop(
                    f"π0.5 Isaac closed-loop failure during {failure_phase}"
                )
                safe_stop_confirmed = bool(receipt.confirmed)
            except BaseException:
                logger.exception(
                    "Failed to confirm Isaac safe-stop after closed-loop failure"
                )
                safe_stop_confirmed = False
        result = {
            "status": "FAIL",
            "episode_id": episode_id,
            "phase": failure_phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "executed_steps": len(records),
            "steps": records,
        }
        if safe_stop_confirmed is not None:
            result["safe_stop_confirmed"] = safe_stop_confirmed
        _write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
    finally:
        if rgb_pipeline is not None:
            try:
                rgb_pipeline.close()
            except BaseException:
                pass
        if gate is not None:
            gate.close("π0.5 Isaac closed-loop runner exiting")
        simulation_app.close()


def main() -> int:
    args = _parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.deadline_ms < 1:
        raise ValueError("--deadline-ms must be positive")
    if args.require_terminal and not args.task_state_factory:
        raise ValueError(
            "--require-terminal requires --task-state-factory; static task state "
            "cannot claim V2 task success"
        )
    result = _run_closed_loop(args)
    if result["status"] in {"FAIL", "SAFE_STOP_UNCONFIRMED"}:
        return 1
    if args.runtime_mode == "supervisor" and result["status"] != "TASK_SUCCEEDED":
        return 1
    if args.require_terminal and result["status"] != "TASK_SUCCEEDED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
