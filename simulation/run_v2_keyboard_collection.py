"""Visible V2 manual keyboard collection into one Canonical episode."""

from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from math import radians
from pathlib import Path
from queue import Empty, Queue
import sys
import time
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _collect_p01_terminal_success(
    *,
    controller: Any,
    probe: Any,
    config: dict[str, Any],
    artifact_dir: Path,
) -> tuple[Any, Path]:
    """Run ten real 100 ms hold actions and evaluate the frozen GT gates."""

    from industrial_agent.contracts import ActionStep
    from industrial_agent.sync_contract import FROZEN_MULTI_RATE
    from simulation.v2_terminal_success import (
        P01_MAX_VERTICAL_ERROR_RAD,
        evaluate_p01_terminal_success,
    )

    part_path = "/World/Parts/P01"
    bin_path = "/World/Bins/Bin_01"
    hold_action = ActionStep.from_sequence(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        duration_ms=100,
    )
    positions = [probe.world_position(part_path)]
    initial_orientation_error = probe.part_vertical_error_rad(
        part_path=part_path,
        bin_path=bin_path,
    )
    timestamps_s = [
        float(controller.physics_tick_index) / FROZEN_MULTI_RATE.physics_hz
    ]
    vote_reports: list[dict[str, Any]] = []
    vote_steps = {1, 5, 10}
    for step in range(1, 11):
        controller.execute_action(hold_action, arm_id="Arm_B")
        physics_tick = int(controller.physics_tick_index)
        positions.append(probe.world_position(part_path))
        timestamps_s.append(float(physics_tick) / FROZEN_MULTI_RATE.physics_hz)
        if step not in vote_steps:
            continue
        orientation_error = probe.part_vertical_error_rad(
            part_path=part_path,
            bin_path=bin_path,
        )
        containment = probe.part_fully_inside_bin(
            part_path=part_path,
            bin_path=bin_path,
            bin_config=config["bin"],
        )
        vote_reports.append(
            {
                "observation_id": f"physics-{physics_tick}",
                "timestamp_s": timestamps_s[-1],
                "physics_tick": physics_tick,
                "orientation_error_rad": orientation_error,
                "containment": containment,
                "pass": bool(
                    containment["pass"]
                    and orientation_error <= P01_MAX_VERTICAL_ERROR_RAD
                ),
            }
        )

    result = evaluate_p01_terminal_success(
        orientation_error_rad=initial_orientation_error,
        vote_reports=vote_reports,
        positions_world=positions,
        timestamps_s=timestamps_s,
    )
    payload = result.to_dict()
    payload.update(
        {
            "scene_id": config["scene_id"],
            "task_id": "P01_TO_S11",
            "part_path": part_path,
            "bin_path": bin_path,
            "position_samples_world": positions,
            "timestamp_samples_s": timestamps_s,
            "vote_reports": vote_reports,
            "isolation": "offline_gt_only",
        }
    )
    report_path = artifact_dir / "offline_gt" / "p01_terminal_success.json"
    _write_json_atomic(report_path, payload)
    return result, report_path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene config root must be an object")
    return payload


def _state_7d(controller: Any, arms: dict[str, Any], config: dict[str, Any]):
    from simulation.run_isaac_adapter_smoke import _arm_state

    return {
        arm_id: _arm_state(
            controller,
            arm_id,
            arms[arm_id],
            config,
            continuous_state=True,
        )["state"]
        for arm_id in ("Arm_A", "Arm_B")
    }


def _arm_readback(
    controller: Any,
    arms: dict[str, Any],
    config: dict[str, Any],
    arm_id: str,
) -> dict[str, Any]:
    from simulation.run_isaac_adapter_smoke import _arm_state

    return _arm_state(
        controller,
        arm_id,
        arms[arm_id],
        config,
        continuous_state=True,
    )


def _apply_home(world: Any, arms: dict[str, Any], config: dict[str, Any]) -> None:
    from isaacsim.core.utils.types import ArticulationAction
    from simulation.run_g0_acceptance import (
        _home_readback_errors,
        _write_explicit_home,
    )

    targets = {
        arm_id: _write_explicit_home(config, arms[arm_id], arm_id)
        for arm_id in ("Arm_A", "Arm_B")
    }
    for arm_id in ("Arm_A", "Arm_B"):
        arms[arm_id].get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=targets[arm_id])
        )
    for _ in range(120):
        world.step(render=True)
    errors = []
    for arm_id in ("Arm_A", "Arm_B"):
        errors.extend(_home_readback_errors(arms[arm_id], arm_id, targets[arm_id]))
    if errors:
        raise RuntimeError("explicit HOME failed: " + "; ".join(errors))


def _preload_pink_runtime(ik_backend: str) -> None:
    """Register Pinocchio C++ bindings before Kit loads its plugins."""
    if ik_backend != "pink":
        return
    importlib.import_module("eigenpy")
    importlib.import_module("pinocchio")


def main() -> int:
    for path in (REPOSITORY_ROOT, SOURCE_DIR, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from simulation.v2_collection_entry import (
        build_parser,
        create_recorder,
        git_identity,
        preflight_from_args,
        preflight_payload,
        write_result,
    )

    args = build_parser().parse_args()
    git_sha, worktree_clean = git_identity()
    preflight = preflight_from_args(
        args, git_sha=git_sha, worktree_clean=worktree_clean
    )
    artifact_dir = args.artifact_dir.expanduser().resolve()
    result_path = artifact_dir / "result.json"
    result: dict[str, Any] = {
        "status": "ERROR",
        "preflight": preflight_payload(preflight),
        "headless": False,
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    simulation_app = None
    rgb_pipeline = None
    gui_keyboard = None
    status_window = None
    controller = None
    bridge = None
    machine = None
    offline_gt_probe = None
    terminal_hold_requested = False
    terminal_success_report = None
    terminal_success_path: Path | None = None
    action_count = 0
    phase = "launch"
    episode_path: Path | None = None
    try:
        import isaac_compat
        from simulation.canonical_recorder_bridge import CanonicalRecorderBridge
        from simulation.isaac_gui_keyboard import IsaacGuiKeyboardSource
        from simulation.isaac_rgb_pipeline import IsaacRgbObservationPipeline
        from simulation.keyboard_teleop import KeyboardTeleopMapper
        from simulation.offline_gt import OfflineGtProbe
        from simulation.rgb_cas_bridge import IsaacRgbCasPublisher
        from simulation.v2_collection_state import (
            EpisodeOutcome,
            V2CollectionContract,
            V2ManualCollectionStateMachine,
        )
        from simulation.v2_scene_contract import require_valid_config

        config = _load_json(preflight.config_path)
        require_valid_config(config)
        contract = V2CollectionContract.from_config(config)
        machine = V2ManualCollectionStateMachine(contract)

        _preload_pink_runtime(args.ik_backend)
        simulation_app = isaac_compat.launch_simulation_app(headless=False)
        phase = "build_scene"
        import single_bin_scene_v2_builder
        from isaac_franka_controller import IsaacSimFrankaController
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(args.franka_usd)
        single_bin_scene_v2_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        offline_gt_probe = OfflineGtProbe(stage)
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
                    name=f"v2_collection_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }
        world.reset()
        phase = "home"
        _apply_home(world, arms, config)

        phase = "initialize_recorder"
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
        image_cas, recorder = create_recorder(preflight)
        publisher = IsaacRgbCasPublisher.from_scene_config(image_cas, config)
        rgb_pipeline = IsaacRgbObservationPipeline(
            simulation_app=simulation_app,
            scene_config=config,
            publisher=publisher,
        )
        bridge = CanonicalRecorderBridge(
            recorder=recorder,
            rgb_pipeline=rgb_pipeline,
            state_source=lambda: _state_7d(controller, arms, config),
        )
        bridge.record_initial(physics_tick=0)
        controller.set_tick_observer(bridge.observe_physics_tick)
        active_arm = "Arm_A"
        mapper = KeyboardTeleopMapper(
            translation_step_m=args.translation_step_m,
            fine_translation_step_m=args.fine_translation_step_m,
            rotation_step_rad=radians(args.rotation_step_deg),
            duration_ms=100,
            gripper_open=True,
        )
        command_queue: Queue[str] = Queue()
        gui_keyboard = IsaacGuiKeyboardSource.from_isaac(command_queue)
        gui_keyboard.start()

        from omni import ui  # type: ignore[import-not-found]

        status_window = ui.Window(
            "V2 Canonical Keyboard Collection", width=760, height=250
        )
        with status_window.frame:
            with ui.VStack(spacing=5):
                ui.Label("W/S X | A/D Y | Q/E Z | I/K J/L U/O rotation | G gripper")
                ui.Label(
                    f"F toggles COARSE/FINE translation "
                    f"({args.translation_step_m * 1000:.0f} mm / "
                    f"{args.fine_translation_step_m * 1000:.0f} mm)"
                )
                ui.Label("P01 target: S11 = bin back row, left-most slot")
                ui.Label(
                    f"IK backend: {controller.ik_backend.upper()} + null-space posture"
                )
                ui.Label("Z confirm next part | V verify handoff | B activate Arm_B")
                ui.Label("C complete full task | P checkpoint | X safe-stop")
                ui.Label("Tap keys once. Do not hold. Formal actions are recorded.")
                status_label = ui.Label("READY | A_ONLY | next=P01")

        print("V2 canonical keyboard collection READY")
        print(mapper.help_text())
        phase = "interactive"
        running = True
        while running and simulation_app.is_running():
            try:
                raw_key = command_queue.get_nowait()
            except Empty:
                simulation_app.update()
                continue
            try:
                command = mapper.parse(raw_key)
                if command.kind == "help":
                    print(mapper.help_text())
                    continue
                if command.kind == "quit":
                    running = False
                    continue
                if command.kind == "reset":
                    raise RuntimeError(
                        "reset is forbidden inside a formal episode; safe-stop instead"
                    )
                if command.kind == "checkpoint":
                    print(
                        f"CHECKPOINT token={machine.token.value} "
                        f"next={machine.next_part_id} actions={action_count}"
                    )
                    continue
                if command.kind == "toggle_precision":
                    mode = mapper.toggle_precision()
                    print(
                        f"PRECISION={mode} "
                        f"step_mm={mapper.translation_step_m * 1000:.1f}"
                    )
                    status_label.text = (
                        f"{machine.token.value} | {mode} "
                        f"{mapper.translation_step_m * 1000:.0f}mm | "
                        f"arm={active_arm} | actions={action_count}"
                    )
                    continue
                if command.kind == "part_placed":
                    part_id = machine.next_part_id
                    if part_id is None:
                        raise RuntimeError("all formal parts are already confirmed")
                    machine.record_part_placement(
                        part_id=part_id,
                        slot_id=contract.part_to_slot[part_id],
                        stable=True,
                    )
                    print(
                        f"HUMAN CONFIRMED {part_id}->{contract.part_to_slot[part_id]}"
                    )
                    continue
                if command.kind == "handoff_verify":
                    arm_a = _arm_readback(controller, arms, config, "Arm_A")
                    machine.enter_handoff_verify(
                        bin_at_handoff_center=True,
                        bin_stable=True,
                        arm_a_gripper_open=arm_a["gripper_open"],
                        arm_a_clear=arm_a["retreated"],
                    )
                    print("HUMAN CONFIRMED HANDOFF_VERIFY")
                    continue
                if command.kind == "activate_b":
                    machine.activate_b_only()
                    active_arm = "Arm_B"
                    mapper.set_gripper_open(True)
                    print("TOKEN=B_ONLY active_arm=Arm_B")
                    continue
                if command.kind == "complete":
                    arm_b = _arm_readback(controller, arms, config, "Arm_B")
                    machine.complete(
                        bin_at_finished=True,
                        bin_stable=True,
                        arm_b_gripper_open=arm_b["gripper_open"],
                        arm_b_clear=arm_b["retreated"],
                    )
                    print("HUMAN CONFIRMED FULL TASK COMPLETE")
                    terminal_hold_requested = True
                    running = False
                    continue
                if command.action is None:
                    raise RuntimeError("action command contains no ActionStep")
                if action_count >= args.max_actions:
                    raise RuntimeError("max-actions safety limit reached")
                machine.require_arm_action(active_arm)
                rejection = controller.action_rejection_reason(
                    command.action,
                    arm_id=active_arm,
                )
                if rejection is not None:
                    print(f"ACTION REJECTED: {rejection}; choose another direction")
                    status_label.text = (
                        f"REJECTED | {rejection} | actions={action_count}"
                    )
                    continue
                tick = controller.physics_tick_index
                bridge.record_action(
                    command.action,
                    arm_id=active_arm,
                    subtask_id=preflight.task_id,
                    chunk_id=f"{preflight.episode_id}-{action_count:06d}",
                    physics_tick=tick,
                )

                controller.execute_action(command.action, arm_id=active_arm)
                action_count += 1
                status_label.text = (
                    f"{machine.token.value} | arm={active_arm} | "
                    f"next={machine.next_part_id} | actions={action_count}"
                )
                print(f"ACTION {action_count} {active_arm}: {command.description}")
            except BaseException:
                running = False
                raise

        if terminal_hold_requested and machine.outcome is EpisodeOutcome.SUCCEEDED:
            phase = "terminal_hold_offline_gt"
            terminal_success_report, terminal_success_path = (
                _collect_p01_terminal_success(
                    controller=controller,
                    probe=offline_gt_probe,
                    config=config,
                    artifact_dir=artifact_dir,
                )
            )
            if not terminal_success_report.passed:
                from simulation.v2_collection_state import V2FailureCode

                code = terminal_success_report.failure_codes[0]
                machine.fail_offline_gt(V2FailureCode(code))

        phase = "safe_stop"
        receipt = controller.safe_stop("V2 keyboard collection finished")
        confirmed = bool(receipt.confirmed)
        if machine.outcome is None:
            machine.safe_stop(confirmed=confirmed)
        elif not confirmed:
            raise RuntimeError("final safe-stop readback was not confirmed")
        if action_count < 1:
            bridge.abort()
            raise RuntimeError(
                "collection produced no actions; unpublished episode aborted"
            )
        if (
            preflight.full_task_required
            and machine.outcome is not EpisodeOutcome.SUCCEEDED
        ):
            bridge.abort()
            raise RuntimeError("train/validation requires a complete successful task")
        episode_path = bridge.save(
            outcome=machine.outcome.value,
            failure_code=(machine.failure_code.value if machine.failure_code else None),
        )
        result.update(
            {
                "status": "PASS",
                "scene_file": str(scene_file),
                "episode_path": str(episode_path),
                "outcome": machine.outcome.value,
                "failure_code": (
                    machine.failure_code.value if machine.failure_code else None
                ),
                "action_count": action_count,
                "token": machine.token.value,
                "placed_parts": list(machine.placed_parts),
                "safe_stop": asdict(receipt),
                "training_allowed": preflight.training_allowed,
                "offline_gt_included": False,
                "offline_gt_path": (
                    str(terminal_success_path)
                    if terminal_success_path is not None
                    else None
                ),
                "terminal_success": (
                    terminal_success_report.to_dict()
                    if terminal_success_report is not None
                    else None
                ),
            }
        )
        return 0
    except BaseException as exc:
        if bridge is not None and episode_path is None:
            try:
                bridge.abort()
            except BaseException:
                pass
        if controller is not None:
            try:
                result["failure_safe_stop"] = asdict(
                    controller.safe_stop(f"collection failed during {phase}")
                )
            except BaseException as stop_exc:
                result["failure_safe_stop_error"] = repr(stop_exc)
        result.update(
            {
                "status": "FAIL",
                "phase": phase,
                "action_count": action_count,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        if terminal_success_path is not None:
            result.setdefault("offline_gt_path", str(terminal_success_path))
        if terminal_success_report is not None:
            result.setdefault(
                "terminal_success", terminal_success_report.to_dict()
            )
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_result(result_path, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if gui_keyboard is not None:
            gui_keyboard.close()
        if status_window is not None:
            status_window.visible = False
        if rgb_pipeline is not None:
            rgb_pipeline.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
