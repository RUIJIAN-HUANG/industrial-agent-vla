"""Run the visible V2 competition console inside one Isaac Sim process.

The entry owns the Isaac main thread.  UI callbacks only update the pure
``CompetitionController``; every scene, observation, action, reset, and stop
operation is executed through the existing owner-thread boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
DEFAULT_SCENE_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"
DEFAULT_AGENT_CONFIG = REPOSITORY_ROOT / "configs" / "agent.default.json"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "competition-ui"
logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the Isaac Sim V2 competition task window."
    )
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--agent-config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--franka-usd")
    parser.add_argument("--output-scene", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--deadline-ms", type=int, default=15_000)
    parser.add_argument("--ik-backend", choices=("lula", "pink"), default="lula")
    parser.add_argument("--pi05-url")
    parser.add_argument("--yolo-url", default="http://127.0.0.1:8103")
    parser.add_argument(
        "--task-state-factory",
        help=(
            "Optional module:callable sensor verifier. The callable receives "
            "task_spec, controller, arms and scene_config and returns a "
            "zero-argument task-state provider. For BIN01 it may also expose "
            "active_arm() returning Arm_A, Arm_B or NONE."
        ),
    )
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help="Refuse to run without a sensor-backed terminal-state provider.",
    )
    parser.add_argument("--health-interval-s", type=float, default=2.0)
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
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _service_ready(base_url: str, expected_service: str) -> bool:
    try:
        with urlopen(base_url.rstrip("/") + "/health", timeout=1.0) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    service = payload.get("service") or payload.get("name")
    return payload.get("status") == "ready" and service in {None, expected_service}


def _resolve_task_state_provider(
    reference: str,
    *,
    task_spec: Any,
    controller: Any,
    arms: Mapping[str, Any],
    scene_config: Mapping[str, Any],
) -> Callable[[], Mapping[str, Any]]:
    from simulation.run_pi05_isaac_closed_loop import resolve_task_state_provider

    return resolve_task_state_provider(
        reference,
        task_spec=task_spec,
        controller=controller,
        arms=arms,
        scene_config=scene_config,
    )


def _active_arm_from_provider(provider: Any, task_id: str) -> str:
    if task_id != "BIN01_TO_FINISHED01":
        return "Arm_A"
    accessor = getattr(provider, "active_arm", None)
    value = accessor() if callable(accessor) else "Arm_A"
    if value not in {"Arm_A", "Arm_B", "NONE"}:
        raise ValueError("task-state provider active_arm() returned an invalid arm")
    return str(value)


def _run_competition(args: argparse.Namespace) -> int:
    for path in (REPOSITORY_ROOT, SOURCE_DIR, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import isaac_compat
    import scene_layout
    from industrial_agent.image_cas import ImageCas, ImageCasConfig
    from industrial_agent.isaac_environment import IsaacExecutionEnvironment
    from industrial_agent.isaac_runtime import IsaacMainThreadGate
    from industrial_agent.v2_task_profile import require_formal_v2_task
    from simulation.pi05_isaac_supervisor_runtime import run_supervisor_runtime
    from simulation.run_pi05_isaac_closed_loop import (
        _capture_stable_observation_inputs,
        _update_ui_without_advancing_physics,
        build_observation,
        build_task_state,
    )
    from simulation.v2_competition_controller import (
        CompetitionCommandType,
        CompetitionController,
        load_competition_task,
    )
    from simulation.v2_scene_contract import require_valid_config

    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.deadline_ms < 1:
        raise ValueError("--deadline-ms must be positive")
    if args.health_interval_s <= 0:
        raise ValueError("--health-interval-s must be positive")
    if args.require_terminal and not args.task_state_factory:
        raise ValueError(
            "--require-terminal requires --task-state-factory; the static provider "
            "cannot claim task success"
        )

    scene_config = scene_layout.load_config(args.scene_config.expanduser().resolve())
    require_valid_config(scene_config)
    agent_config = _read_json(args.agent_config)
    if agent_config.get("profile_id") != "single_bin_manual_industrial_v2":
        raise ValueError("competition UI requires the formal V2 agent config")
    raw_pi05 = agent_config.get("executors", {}).get("pi05")
    if not isinstance(raw_pi05, dict):
        raise ValueError("agent config executors.pi05 is required")
    pi05_url = str(args.pi05_url or raw_pi05.get("base_url", "")).rstrip("/")
    yolo_url = str(args.yolo_url).rstrip("/")
    if not pi05_url or not yolo_url:
        raise ValueError("Π0.5 and YOLO service URLs must be non-empty")
    raw_pi05["base_url"] = pi05_url

    artifact_root = args.artifact_root.expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_scene = (
        args.output_scene.expanduser().resolve()
        if args.output_scene is not None
        else artifact_root / "single_bin_scene_v2.usda"
    )
    cas_root = artifact_root / "cas"
    simulation_app = isaac_compat.launch_simulation_app(headless=False)
    rgb_pipeline = None
    window = None
    active_environment = None
    active_gate = None
    controller_state = CompetitionController(
        max_steps=args.max_steps,
        verifier_configured=bool(args.task_state_factory),
    )
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from simulation.isaac_franka_controller import IsaacSimFrankaController
        from simulation.isaac_rgb_pipeline import IsaacRgbObservationPipeline
        from simulation.rgb_cas_bridge import IsaacRgbCasPublisher
        from simulation.run_isaac_adapter_smoke import _arm_state
        from simulation.run_v2_keyboard_collection import _apply_home
        from simulation.single_bin_scene_v2_builder import build_scene
        from simulation.v2_competition_window import V2CompetitionWindow

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
                    name=f"competition_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }
        world.reset()
        _apply_home(world, arms, scene_config)

        image_cas = ImageCas(ImageCasConfig(root=cas_root))
        image_cas.assert_ready(writable=True)
        publisher = IsaacRgbCasPublisher.from_scene_config(image_cas, scene_config)
        rgb_pipeline = IsaacRgbObservationPipeline(
            simulation_app=simulation_app,
            scene_config=scene_config,
            publisher=publisher,
        )
        window = V2CompetitionWindow(controller_state)
        next_health_check = 0.0

        def refresh_health(*, force: bool = False) -> None:
            nonlocal next_health_check
            now = time.monotonic()
            if not force and now < next_health_check:
                return
            next_health_check = now + float(args.health_interval_s)
            controller_state.update_service_health(
                pi05_ready=_service_ready(pi05_url, "pi05"),
                yolo_ready=_service_ready(yolo_url, "yolo"),
            )

        def reset_scene(*, notify_controller: bool = True) -> None:
            world.reset()
            _apply_home(world, arms, scene_config)
            for _ in range(10):
                world.step(render=True)
            world.pause()
            if notify_controller:
                controller_state.mark_reset_complete()

        def execute_task(task_id: str) -> None:
            nonlocal active_environment, active_gate
            task = load_competition_task(task_id)
            task_spec = require_formal_v2_task(task.task_id)
            run_stamp = time.strftime("%Y%m%d-%H%M%S")
            episode_id = f"competition-{run_stamp}-{uuid4().hex[:8]}"
            run_dir = artifact_root / episode_id
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / "run_result.json"
            reset_scene(notify_controller=False)
            robot_controller = IsaacSimFrankaController(
                world=world,
                arms=arms,
                physics_dt_s=float(physics["physics_dt_s"]),
                virtual_tcp_fingertip_frame_names=(
                    "panda_leftfingertip",
                    "panda_rightfingertip",
                ),
                ik_backend=args.ik_backend,
            )
            gate = IsaacMainThreadGate()
            active_gate = gate
            if args.task_state_factory:
                task_state_provider = _resolve_task_state_provider(
                    args.task_state_factory,
                    task_spec=task_spec,
                    controller=robot_controller,
                    arms=arms,
                    scene_config=scene_config,
                )
            else:
                task_state_provider = lambda: build_task_state(task_spec)

            observation_counter = 0

            def current_active_arm() -> str:
                return _active_arm_from_provider(task_state_provider, task.task_id)

            def guarded_state() -> dict[str, Any]:
                active_arm = current_active_arm()
                return {
                    "objects": [],
                    "robot": {
                        "active_arm": active_arm,
                        "arm_a": _arm_state(
                            robot_controller,
                            "Arm_A",
                            arms["Arm_A"],
                            scene_config,
                            continuous_state=True,
                        ),
                        "arm_b": _arm_state(
                            robot_controller,
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
                    capture_camera=lambda: rgb_pipeline.capture(current_active_arm()),
                    capture_state=guarded_state,
                )
                observation_counter += 1
                controller_state.update_progress(
                    max(0, observation_counter - 1),
                    f"正在执行 {task.task_id}",
                )
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
                control_lease_source=lambda: (
                    "A_ONLY"
                    if current_active_arm() == "Arm_A"
                    else "B_ONLY"
                    if current_active_arm() == "Arm_B"
                    else "NONE"
                ),
                controller=robot_controller,
                runtime_gate=gate,
                command_ledger_path=run_dir / "command-ids.jsonl",
                runtime_observe_timeout_s=2.0,
                runtime_action_timeout_s=max(10.0, args.deadline_ms / 1000 + 5.0),
                runtime_stop_timeout_s=2.0,
            )
            active_environment = environment
            stop_applied = False

            def idle() -> None:
                nonlocal stop_applied
                if not window.visible:
                    controller_state.request_exit()
                if controller_state.stop_event.is_set() and not stop_applied:
                    stop_applied = True
                    environment.safe_stop("operator requested safe-stop from UI")
                _update_ui_without_advancing_physics(
                    world=world,
                    simulation_app=simulation_app,
                )
                window.refresh()

            result: dict[str, Any]
            try:
                report = run_supervisor_runtime(
                    config=agent_config,
                    task=task,
                    environment=environment,
                    gate=gate,
                    max_steps=args.max_steps,
                    idle_callback=idle,
                    stop_event=controller_state.stop_event,
                )
                run_result = report.run_result
                result = {
                    "status": "TASK_SUCCEEDED" if run_result.success else "SAFE_STOPPED",
                    "episode_id": episode_id,
                    "task_id": task.task_id,
                    "instruction": task.instruction,
                    "isaac_sim_version": isaac_version,
                    "success": run_result.success,
                    "failure_code": run_result.failure_code.value,
                    "message": run_result.message,
                    "safe_stop_confirmed": report.safe_stop_confirmed,
                    "executed_steps": len(report.actions),
                    "steps": report.action_dicts(),
                }
                if controller_state.stop_event.is_set():
                    controller_state.mark_stopped("任务已由操作员安全停止")
                elif run_result.success:
                    controller_state.mark_succeeded("任务执行成功")
                else:
                    controller_state.mark_failed(
                        "任务未达到完成条件",
                        error=run_result.message,
                    )
            except BaseException as exc:
                receipt = environment.safe_stop(
                    f"competition runtime exception: {type(exc).__name__}"
                )
                result = {
                    "status": "FAIL",
                    "episode_id": episode_id,
                    "task_id": task.task_id,
                    "instruction": task.instruction,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "safe_stop_confirmed": receipt.confirmed,
                }
                controller_state.mark_failed("任务执行异常", error=str(exc))
            finally:
                _write_json(result_path, result)
                gate.close("competition task finished")
                active_gate = None
                active_environment = None
                window.refresh()

        refresh_health(force=True)
        window.refresh()
        while simulation_app.is_running():
            simulation_app.update()
            if not window.visible:
                controller_state.request_exit()
            refresh_health()
            window.refresh()
            for command in controller_state.drain_commands():
                if command.kind == CompetitionCommandType.START:
                    assert command.task_id is not None
                    execute_task(command.task_id)
                elif command.kind == CompetitionCommandType.RESET:
                    reset_scene()
                elif command.kind == CompetitionCommandType.EXIT:
                    return 0
            if controller_state.snapshot().exit_requested:
                return 0
        return 0
    finally:
        if active_environment is not None:
            try:
                active_environment.safe_stop("competition UI exiting")
            except BaseException:
                logger.exception("final competition safe-stop failed")
        if active_gate is not None:
            active_gate.close("competition UI exiting")
        if rgb_pipeline is not None:
            try:
                rgb_pipeline.close()
            except BaseException:
                logger.exception("RGB pipeline close failed")
        if window is not None:
            window.destroy()
        simulation_app.close()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    try:
        return _run_competition(args)
    except BaseException as exc:
        payload = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
