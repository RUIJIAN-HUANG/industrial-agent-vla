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
TERMINAL_HOLD_ACTION_COUNT = 10
APPROACH_CURVE_AMPLITUDE_M = 0.0005
APPROACH_CURVE_MAX_STEP_RATIO = 0.1
GRIPPER_SETTLE_ACTION_COUNT = 5


def _interactive_action_repeat_count(command: Any) -> int:
    """Give a gripper toggle enough recorded 100 ms actions to reach target."""

    return GRIPPER_SETTLE_ACTION_COUNT if command.key == "g" else 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _handoff_precondition_failures(
    placement: dict[str, Any],
    arm_a: dict[str, Any],
) -> tuple[str, ...]:
    """Return actionable reasons why a Bin_01 handoff cannot proceed yet."""

    failures: list[str] = []
    if not bool(placement.get("pass")):
        failures.append("PLACE BIN_01 INSIDE HANDOFF_CENTER")
    if not bool(arm_a.get("gripper_open")):
        failures.append("OPEN ARM_A GRIPPER WITH G")
    if not bool(arm_a.get("retreated")):
        failures.append("RETREAT ARM_A OUTSIDE GREEN ZONE")
    return tuple(failures)


def _replay_task_actions_from_rows(rows: list[Any]) -> list[Any]:
    """Convert validated rows while removing the frozen terminal hold suffix."""

    import numpy as np

    from industrial_agent.contracts import ActionStep

    if len(rows) <= TERMINAL_HOLD_ACTION_COUNT:
        raise ValueError(
            "--replay-episode contains no task actions before terminal hold"
        )
    expected_hold = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    terminal_rows = rows[-TERMINAL_HOLD_ACTION_COUNT:]
    if any(not np.array_equal(row, expected_hold) for row in terminal_rows):
        raise ValueError(
            "--replay-episode must end with exactly ten canonical open-gripper "
            "hold actions"
        )
    actions = [
        ActionStep.from_sequence(row.tolist(), duration_ms=100)
        for row in rows[:-TERMINAL_HOLD_ACTION_COUNT]
    ]
    return actions


def _validate_replay_source_metadata(
    metadata: dict[str, Any],
    *,
    expected_scene_config_sha256: str,
) -> None:
    """Reject replay data produced from a different frozen scene config."""

    if metadata["scene_config_sha256"] != expected_scene_config_sha256:
        raise ValueError(
            "--replay-episode scene config SHA-256 does not match the current "
            "collection config"
        )


def _load_replay_task_actions(
    episode_dir: Path,
    *,
    expected_scene_config_sha256: str,
    expected_task_id: str,
) -> tuple[str, list[Any], list[str]]:
    """Load validated task actions, excluding the frozen terminal hold suffix."""

    from scripts.pi05.canonical_v2 import CanonicalV2Reader

    with CanonicalV2Reader(episode_dir) as reader:
        metadata = reader.manifest["metadata"]
        if metadata["outcome"] != "SUCCEEDED":
            raise ValueError("--replay-episode must have outcome SUCCEEDED")
        if metadata["task_id"] != expected_task_id:
            raise ValueError("--replay-episode task_id does not match this collection")
        _validate_replay_source_metadata(
            metadata,
            expected_scene_config_sha256=expected_scene_config_sha256,
        )
        rows = list(reader.iter_action_7d())
        stored_arm_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in reader.h5["actions/arm_id"][:]
        ]
        source_episode_id = reader.episode_id
    actions = _replay_task_actions_from_rows(rows)
    arm_ids = stored_arm_ids[: len(actions)]
    if len(stored_arm_ids) != len(rows) or len(arm_ids) != len(actions):
        raise ValueError("--replay-episode action identity count is inconsistent")
    if expected_task_id == "BIN01_TO_FINISHED01":
        if "Arm_A" not in arm_ids or "Arm_B" not in arm_ids:
            raise ValueError(
                "--replay-episode BIN01_TO_FINISHED01 requires both Arm_A and Arm_B"
            )
        first_b = arm_ids.index("Arm_B")
        if any(arm_id != "Arm_A" for arm_id in arm_ids[:first_b]) or any(
            arm_id != "Arm_B" for arm_id in arm_ids[first_b:]
        ):
            raise ValueError(
                "--replay-episode dual-arm actions must be ordered Arm_A then Arm_B"
            )
    return source_episode_id, actions, arm_ids


def _diversify_replay_actions(
    actions: list[Any],
    *,
    profile: str,
    seed: int,
    variant: int = 0,
    lift_mm: float | None = None,
    final_y_offset_mm: float = 0.0,
    final_z_offset_mm: float = 0.0,
    arm_ids: list[str] | None = None,
) -> list[Any]:
    """Apply a bounded, smooth deterministic variation to replay actions.

    V2 actions are Cartesian deltas, so perturb cumulative translation and
    convert it back to deltas.  Grasp/release commands remain unchanged.
    """

    if profile == "baseline":
        return list(actions)
    if profile not in {"diverse_low", "approach_curve"}:
        raise ValueError(f"unsupported trajectory profile: {profile}")
    if arm_ids is not None:
        if len(arm_ids) != len(actions):
            raise ValueError("arm_ids must align one-to-one with replay actions")
        boundaries = [
            index
            for index in range(1, len(arm_ids))
            if arm_ids[index] != arm_ids[index - 1]
        ]
        if boundaries:
            if len(boundaries) != 1 or set(arm_ids) != {"Arm_A", "Arm_B"}:
                raise ValueError(
                    "dual-arm replay requires exactly one Arm_A-to-Arm_B transition"
                )
            split = boundaries[0]
            if arm_ids[0] != "Arm_A" or arm_ids[split] != "Arm_B":
                raise ValueError("dual-arm replay must be ordered Arm_A then Arm_B")
            arm_a_actions = _diversify_replay_actions(
                actions[:split],
                profile=profile,
                seed=seed,
                variant=variant,
                lift_mm=lift_mm,
            )
            arm_b_actions = _diversify_replay_actions(
                actions[split:],
                profile=profile,
                seed=seed + 1,
                variant=variant,
                lift_mm=lift_mm,
                final_y_offset_mm=final_y_offset_mm,
                final_z_offset_mm=final_z_offset_mm,
            )
            return arm_a_actions + arm_b_actions
    if len(actions) < 8:
        raise ValueError(f"{profile} requires at least eight task actions")

    import numpy as np

    from industrial_agent.contracts import ActionStep

    values = np.asarray([action.values for action in actions], dtype=np.float64)
    closed_indices = np.flatnonzero(values[:, 6] < 0.5)
    if closed_indices.size < 4:
        raise ValueError(
            f"{profile} could not identify a closed-gripper transfer segment"
        )

    start = int(closed_indices[0])
    end = int(closed_indices[-1])
    span = end - start
    if span < 4:
        raise ValueError(f"{profile} transfer segment is too short")

    first = start + max(1, span // 5)
    last = end - max(1, span // 5)
    if last <= first:
        raise ValueError(f"{profile} has no safe interior transfer segment")

    cumulative = np.zeros((len(actions) + 1, 3), dtype=np.float64)
    cumulative[1:] = np.cumsum(values[:, :3], axis=0)
    offsets = np.zeros_like(cumulative)
    if profile == "approach_curve":
        if not 1 <= variant <= 4:
            raise ValueError("approach_curve variant must be in [1, 4]")
        approach_end = start
        if approach_end < 6:
            raise ValueError("approach_curve requires at least six pre-grasp actions")
        approach_start = max(0, approach_end - max(10, approach_end // 2))
        axis = 1 if variant in (1, 2) else 0
        sign = 1.0 if variant in (1, 3) else -1.0
        moving_indices = [
            index
            for index in range(approach_start, approach_end)
            if np.linalg.norm(values[index, :3]) > np.finfo(np.float64).eps
        ]
        if len(moving_indices) < 2:
            raise ValueError(
                "approach_curve has fewer than two movable pre-grasp steps"
            )
        phase = np.linspace(0.0, np.pi, len(moving_indices) + 1)
        curve_positions = np.zeros((len(moving_indices) + 1, 3), dtype=np.float64)
        curve_positions[:, axis] = sign * APPROACH_CURVE_AMPLITUDE_M * np.sin(phase)
        curve_deltas = np.diff(curve_positions, axis=0)
        # Pink's tracking guard evaluates each Cartesian step, not only the
        # total path.  Bound the perturbation against the corresponding source
        # step so the curve cannot dominate forward progress on a short step.
        curve_scale = 1.0
        for index, perturbation in zip(moving_indices, curve_deltas):
            base_step = values[index, :3]
            base_norm = float(np.linalg.norm(base_step))
            perturbation_norm = float(np.linalg.norm(perturbation))
            if perturbation_norm == 0.0:
                continue
            curve_scale = min(
                curve_scale,
                APPROACH_CURVE_MAX_STEP_RATIO * base_norm / perturbation_norm,
            )
        offset_deltas = np.zeros_like(offsets)
        for index, perturbation in zip(moving_indices, curve_deltas):
            offset_deltas[index] = curve_scale * perturbation
        offsets[1:] = np.cumsum(offset_deltas[:-1], axis=0)
    else:
        phase = np.linspace(0.0, np.pi, last - first + 1)
        if lift_mm is None:
            rng = np.random.default_rng(int(seed))
            lift_m = float(rng.choice(np.asarray([0.0002, 0.0003, 0.0005])))
        else:
            if not 0.0 < float(lift_mm) <= 5.0:
                raise ValueError("lift_mm must be in (0, 5] millimetres")
            lift_m = float(lift_mm) / 1000.0
        offsets[first : last + 1, 2] = lift_m * np.sin(phase)

    final_offset = np.asarray(
        [0.0, float(final_y_offset_mm) / 1000.0, float(final_z_offset_mm) / 1000.0],
        dtype=np.float64,
    )
    if np.any(final_offset):
        release_candidates = np.flatnonzero(
            (np.arange(len(actions)) > end) & (values[:, 6] >= 0.5)
        )
        if release_candidates.size == 0:
            raise ValueError(
                "final placement correction requires an open-gripper release action"
            )
        release = int(release_candidates[0])
        ramp_start = min(last, release)
        ramp = np.linspace(0.0, 1.0, release - ramp_start + 1)
        offsets[ramp_start : release + 1] += ramp[:, None] * final_offset
        offsets[release + 1 :] += final_offset

    varied_positions = cumulative + offsets
    varied = values.copy()
    varied[:, :3] = varied_positions[1:] - varied_positions[:-1]
    return [
        ActionStep.from_sequence(row.tolist(), duration_ms=action.duration_ms)
        for row, action in zip(varied, actions)
    ]


def _collect_p01_terminal_success(
    *,
    bridge: Any,
    controller: Any,
    probe: Any,
    config: dict[str, Any],
    artifact_dir: Path,
    task_id: str,
    episode_id: str,
    action_count: int,
    max_actions: int | None = None,
) -> tuple[Any, Path, int]:
    """Run ten real 100 ms Arm_A hold actions and evaluate frozen GT gates."""

    if max_actions is not None and action_count + 10 > max_actions:
        raise RuntimeError(
            "terminal hold actions exceed the --max-actions safety limit"
        )

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
    timestamps_s = [float(controller.physics_tick_index) / FROZEN_MULTI_RATE.physics_hz]
    vote_reports: list[dict[str, Any]] = []
    vote_steps = {1, 5, 10}
    for step in range(1, 11):
        _record_and_execute_formal_action(
            bridge=bridge,
            controller=controller,
            action=hold_action,
            arm_id="Arm_A",
            task_id=task_id,
            episode_id=episode_id,
            action_index=action_count,
        )
        action_count += 1
        physics_tick = int(controller.physics_tick_index)
        positions.append(probe.world_position(part_path))
        timestamps_s.append(float(physics_tick) / FROZEN_MULTI_RATE.physics_hz)
        if step not in vote_steps:
            continue
        orientation_error = probe.part_vertical_error_rad(
            part_path=part_path,
            bin_path=bin_path,
        )
        placement = probe.p01_in_s11(
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
                "containment": placement["containment"],
                "pass": bool(
                    placement["pass"]
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
    return result, report_path, action_count


def _collect_w01_terminal_success(
    *,
    bridge: Any,
    controller: Any,
    probe: Any,
    config: dict[str, Any],
    artifact_dir: Path,
    task_id: str,
    episode_id: str,
    action_count: int,
    max_actions: int | None = None,
) -> tuple[Any, Path, int]:
    """Hold W01 for one real second, voting on S14 containment and orientation."""

    if max_actions is not None and action_count + 10 > max_actions:
        raise RuntimeError(
            "terminal hold actions exceed the --max-actions safety limit"
        )
    from industrial_agent.contracts import ActionStep
    from industrial_agent.sync_contract import FROZEN_MULTI_RATE
    from simulation.v2_terminal_success import evaluate_w01_terminal_success

    part_path = "/World/Parts/W01"
    bin_path = "/World/Bins/Bin_01"
    hold_action = ActionStep.from_sequence(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), duration_ms=100
    )
    positions = [probe.world_position(part_path)]
    flat_error, heading_error = probe.w01_orientation_errors(
        part_path=part_path, bin_path=bin_path
    )
    timestamps_s = [float(controller.physics_tick_index) / FROZEN_MULTI_RATE.physics_hz]
    vote_reports: list[dict[str, Any]] = []
    for step in range(1, 11):
        _record_and_execute_formal_action(
            bridge=bridge,
            controller=controller,
            action=hold_action,
            arm_id="Arm_A",
            task_id=task_id,
            episode_id=episode_id,
            action_index=action_count,
        )
        action_count += 1
        physics_tick = int(controller.physics_tick_index)
        positions.append(probe.world_position(part_path))
        timestamps_s.append(float(physics_tick) / FROZEN_MULTI_RATE.physics_hz)
        if step not in {1, 5, 10}:
            continue
        placement = probe.w01_in_s14(
            part_path=part_path, bin_path=bin_path, bin_config=config["bin"]
        )
        vote_reports.append(
            {
                "observation_id": f"physics-{physics_tick}",
                "timestamp_s": timestamps_s[-1],
                "physics_tick": physics_tick,
                "containment": placement["containment"],
                "flat_error_rad": placement["flat_error_rad"],
                "heading_error_rad": placement["heading_error_rad"],
                "pass": bool(placement["pass"]),
            }
        )

    result = evaluate_w01_terminal_success(
        flat_error_rad=flat_error,
        heading_error_rad=heading_error,
        vote_reports=vote_reports,
        positions_world=positions,
        timestamps_s=timestamps_s,
    )
    payload = result.to_dict()
    payload.update(
        {
            "scene_id": config["scene_id"],
            "task_id": task_id,
            "part_path": part_path,
            "bin_path": bin_path,
            "position_samples_world": positions,
            "timestamp_samples_s": timestamps_s,
            "vote_reports": vote_reports,
            "isolation": "offline_gt_only",
        }
    )
    report_path = artifact_dir / "offline_gt" / "w01_terminal_success.json"
    _write_json_atomic(report_path, payload)
    return result, report_path, action_count


def _collect_bin01_terminal_success(
    *,
    bridge: Any,
    controller: Any,
    probe: Any,
    config: dict[str, Any],
    artifact_dir: Path,
    task_id: str,
    episode_id: str,
    action_count: int,
    max_actions: int | None = None,
) -> tuple[Any, Path, int]:
    """Hold the released bin for one second and verify FINISHED_01."""

    if max_actions is not None and action_count + 10 > max_actions:
        raise RuntimeError(
            "terminal hold actions exceed the --max-actions safety limit"
        )
    from industrial_agent.contracts import ActionStep
    from industrial_agent.sync_contract import FROZEN_MULTI_RATE
    from simulation.v2_terminal_success import evaluate_bin01_terminal_success

    bin_path = "/World/Bins/Bin_01"
    hold_action = ActionStep.from_sequence(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), duration_ms=100
    )
    positions = [probe.world_position(bin_path)]
    timestamps_s = [float(controller.physics_tick_index) / FROZEN_MULTI_RATE.physics_hz]
    vote_reports: list[dict[str, Any]] = []
    for step in range(1, 11):
        _record_and_execute_formal_action(
            bridge=bridge,
            controller=controller,
            action=hold_action,
            arm_id="Arm_B",
            task_id=task_id,
            episode_id=episode_id,
            action_index=action_count,
        )
        action_count += 1
        physics_tick = int(controller.physics_tick_index)
        positions.append(probe.world_position(bin_path))
        timestamps_s.append(float(physics_tick) / FROZEN_MULTI_RATE.physics_hz)
        if step not in {1, 5, 10}:
            continue
        placement = probe.bin01_in_finished01(
            bin_path=bin_path,
            stations=config["stations"],
            bin_config=config["bin"],
        )
        vote_reports.append(
            {
                "observation_id": f"physics-{physics_tick}",
                "timestamp_s": timestamps_s[-1],
                "physics_tick": physics_tick,
                "pass": bool(placement["pass"]),
                "placement": placement,
            }
        )

    result = evaluate_bin01_terminal_success(
        vote_reports=vote_reports,
        positions_world=positions,
        timestamps_s=timestamps_s,
    )
    payload = result.to_dict()
    payload.update(
        {
            "scene_id": config["scene_id"],
            "task_id": task_id,
            "bin_path": bin_path,
            "station_id": "FINISHED_01",
            "position_samples_world": positions,
            "timestamp_samples_s": timestamps_s,
            "vote_reports": vote_reports,
            "isolation": "offline_gt_only",
        }
    )
    report_path = artifact_dir / "offline_gt" / "bin01_terminal_success.json"
    _write_json_atomic(report_path, payload)
    return result, report_path, action_count


def _record_and_execute_formal_action(
    *,
    bridge,
    controller,
    action,
    arm_id: str,
    task_id: str,
    episode_id: str,
    action_index: int,
    bin_grasp_manager: Any | None = None,
) -> None:
    """Record one 100 ms command and execute its real 12 physics ticks."""
    tick = controller.physics_tick_index
    bridge.record_action(
        action,
        arm_id=arm_id,
        subtask_id=task_id,
        chunk_id=f"{episode_id}-{action_index:06d}",
        physics_tick=tick,
    )
    if bin_grasp_manager is not None:
        bin_grasp_manager.before_action(action, arm_id=arm_id)
    controller.execute_action(action, arm_id=arm_id)
    if bin_grasp_manager is not None:
        bin_grasp_manager.after_action(action, arm_id=arm_id)


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
    bin_grasp_manager = None
    terminal_hold_requested = False
    terminal_success_report = None
    terminal_success_path: Path | None = None
    action_count = 0
    replay_source_episode_id: str | None = None
    replay_actions: list[Any] | None = None
    replay_arm_ids: list[str] | None = None
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
            Bin01ToFinished01CollectionStateMachine,
            EpisodeOutcome,
            P01ToS11CollectionStateMachine,
            V2CollectionContract,
            W01ToS14CollectionStateMachine,
        )
        from simulation.v2_scene_contract import require_valid_config

        config = _load_json(preflight.config_path)
        require_valid_config(config)
        contract = V2CollectionContract.from_config(config)
        task_profiles = {
            "P01_TO_S11": (
                P01ToS11CollectionStateMachine,
                "P01",
                "S11",
                "Arm_A",
            ),
            "W01_TO_S14": (
                W01ToS14CollectionStateMachine,
                "W01",
                "S14",
                "Arm_A",
            ),
            "BIN01_TO_FINISHED01": (
                Bin01ToFinished01CollectionStateMachine,
                None,
                "FINISHED_01",
                "Arm_A",
            ),
        }
        machine_type, task_part_id, task_slot_id, task_arm_id = task_profiles[
            preflight.task_id
        ]
        machine = machine_type(contract)

        if args.replay_episode is not None:
            phase = "load_replay_episode"
            replay_path = args.replay_episode.expanduser().resolve()
            (
                replay_source_episode_id,
                replay_actions,
                replay_arm_ids,
            ) = _load_replay_task_actions(
                replay_path,
                expected_scene_config_sha256=preflight.scene_config_sha256,
                expected_task_id=preflight.task_id,
            )
            replay_actions = _diversify_replay_actions(
                replay_actions,
                profile=args.trajectory_profile,
                seed=args.trajectory_seed,
                variant=args.trajectory_variant,
                lift_mm=args.lift_mm,
                final_y_offset_mm=args.final_y_offset_mm,
                final_z_offset_mm=args.final_z_offset_mm,
                arm_ids=replay_arm_ids,
            )
            if len(replay_actions) + TERMINAL_HOLD_ACTION_COUNT > args.max_actions:
                raise RuntimeError(
                    "replayed task actions plus terminal holds exceed "
                    "the --max-actions safety limit"
                )
            result.update(
                {
                    "replay_source_episode_id": replay_source_episode_id,
                    "replay_source_path": str(replay_path),
                    "replay_task_action_count": len(replay_actions),
                    "trajectory_profile": args.trajectory_profile,
                    "trajectory_seed": args.trajectory_seed,
                    "trajectory_variant": args.trajectory_variant,
                    "trajectory_lift_mm": args.lift_mm,
                    "final_y_offset_mm": args.final_y_offset_mm,
                    "final_z_offset_mm": args.final_z_offset_mm,
                }
            )
            write_result(
                artifact_dir / "trajectory_metadata.json",
                {
                    "reference_episode_id": replay_source_episode_id,
                    "reference_episode_path": str(replay_path),
                    "trajectory_profile": args.trajectory_profile,
                    "trajectory_seed": args.trajectory_seed,
                    "trajectory_variant": args.trajectory_variant,
                    "trajectory_lift_mm": args.lift_mm,
                    "final_y_offset_mm": args.final_y_offset_mm,
                    "final_z_offset_mm": args.final_z_offset_mm,
                    "scene_seed": preflight.scene_seed,
                    "task_id": preflight.task_id,
                },
            )

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
        if preflight.task_id == "BIN01_TO_FINISHED01":
            from simulation.bin_carry_grasp import (
                BinCarryGraspManager,
                UsdFixedJointBinCarryBackend,
            )

            bin_grasp_manager = BinCarryGraspManager(
                UsdFixedJointBinCarryBackend(stage=stage, controller=controller)
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
        active_arm = task_arm_id
        if replay_actions is not None:
            assert replay_arm_ids is not None
            phase = "replay_actions"
            print(
                f"V2 REPLAY READY source={replay_source_episode_id} "
                f"task_actions={len(replay_actions)}"
            )
            for action, replay_arm_id in zip(replay_actions, replay_arm_ids):
                if replay_arm_id != active_arm:
                    if not (
                        preflight.task_id == "BIN01_TO_FINISHED01"
                        and active_arm == "Arm_A"
                        and replay_arm_id == "Arm_B"
                    ):
                        raise RuntimeError(
                            "invalid replay arm transition "
                            f"{active_arm}->{replay_arm_id}"
                        )
                    placement = offline_gt_probe.bin01_in_handoff_center(
                        bin_path="/World/Bins/Bin_01",
                        stations=config["stations"],
                        bin_config=config["bin"],
                    )
                    arm_a = _arm_readback(controller, arms, config, "Arm_A")
                    machine.enter_handoff_verify(
                        bin_at_handoff_center=bool(placement["pass"]),
                        bin_stable=True,
                        arm_a_gripper_open=arm_a["gripper_open"],
                        arm_a_clear=arm_a["retreated"],
                    )
                    machine.activate_b_only()
                    active_arm = "Arm_B"
                machine.require_arm_action(active_arm)
                rejection = controller.action_rejection_reason(
                    action,
                    arm_id=active_arm,
                )
                if rejection is not None:
                    raise RuntimeError(
                        f"replay action {action_count} rejected: {rejection}"
                    )
                _record_and_execute_formal_action(
                    bridge=bridge,
                    controller=controller,
                    action=action,
                    arm_id=active_arm,
                    task_id=preflight.task_id,
                    episode_id=preflight.episode_id,
                    action_index=action_count,
                    bin_grasp_manager=bin_grasp_manager,
                )
                action_count += 1
                print(
                    f"REPLAY ACTION {action_count}/{len(replay_actions)} {active_arm}"
                )
            if preflight.task_id == "BIN01_TO_FINISHED01":
                placement = offline_gt_probe.bin01_in_finished01(
                    bin_path="/World/Bins/Bin_01",
                    stations=config["stations"],
                    bin_config=config["bin"],
                )
                arm_b = _arm_readback(controller, arms, config, "Arm_B")
                machine.complete(
                    bin_at_finished=bool(placement["pass"]),
                    bin_stable=True,
                    arm_b_gripper_open=arm_b["gripper_open"],
                    arm_b_clear=arm_b["retreated"],
                )
            else:
                assert task_part_id is not None
                machine.record_part_placement(
                    part_id=task_part_id,
                    slot_id=contract.part_to_slot[task_part_id],
                    stable=True,
                )
                arm_a = _arm_readback(controller, arms, config, "Arm_A")
                machine.complete(
                    arm_a_gripper_open=arm_a["gripper_open"],
                    arm_a_clear=arm_a["retreated"],
                )
            terminal_hold_requested = True
            print(f"REPLAY COMPLETED {preflight.task_id}; starting terminal validation")
        else:
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
            bin_transport_task = preflight.task_id == "BIN01_TO_FINISHED01"
            target_label = (
                "Bin_01 target: FINISHED_01"
                if bin_transport_task
                else f"{task_part_id} target: {task_slot_id}"
            )
            completion_label = (
                "V verify HANDOFF_CENTER | B activate Arm_B | "
                "C confirm FINISHED_01"
                if bin_transport_task
                else f"Z confirm {task_part_id} in {task_slot_id} | "
                f"C complete {task_part_id} task"
            )
            workflow_label = (
                "Start Arm_A | V verify handoff | B switch to Arm_B | "
                "P checkpoint | X safe-stop"
                if bin_transport_task
                else "P checkpoint | X safe-stop | V/B disabled for this task"
            )
            with status_window.frame:
                with ui.VStack(spacing=5):
                    ui.Label(
                        "W/S X | A/D Y | Q/E Z | I/K J/L U/O rotation | "
                        "G gripper (0.5 s settle)"
                    )
                    ui.Label(
                        f"F toggles COARSE/FINE translation "
                        f"({args.translation_step_m * 1000:.0f} mm / "
                        f"{args.fine_translation_step_m * 1000:.0f} mm)"
                    )
                    ui.Label(target_label)
                    ui.Label(
                        f"IK backend: {controller.ik_backend.upper()} + "
                        "null-space posture"
                    )
                    ui.Label(completion_label)
                    ui.Label(workflow_label)
                    ui.Label("Tap keys once. Do not hold. Formal actions are recorded.")
                    status_label = ui.Label(
                        f"READY | {machine.token.value} | arm={active_arm}"
                    )

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
                            "reset is forbidden inside a formal episode; "
                            "safe-stop instead"
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
                        if bin_transport_task:
                            raise RuntimeError(
                                "BIN01_TO_FINISHED01 does not use Z; use V then B "
                                "for handoff, and C only after Arm_B finishes"
                            )
                        part_id = machine.next_part_id
                        if part_id is None:
                            raise RuntimeError("all formal parts are already confirmed")
                        machine.record_part_placement(
                            part_id=part_id,
                            slot_id=contract.part_to_slot[part_id],
                            stable=True,
                        )
                        print(
                            f"HUMAN CONFIRMED {part_id}->"
                            f"{contract.part_to_slot[part_id]}"
                        )
                        continue
                    if command.kind == "handoff_verify":
                        if not bin_transport_task:
                            raise RuntimeError(
                                f"{preflight.task_id} ends before handoff; "
                                "V is not allowed"
                            )
                        placement = offline_gt_probe.bin01_in_handoff_center(
                            bin_path="/World/Bins/Bin_01",
                            stations=config["stations"],
                            bin_config=config["bin"],
                        )
                        arm_a = _arm_readback(controller, arms, config, "Arm_A")
                        handoff_report = {
                            "placement": placement,
                            "arm_a_gripper_open": arm_a["gripper_open"],
                            "arm_a_clear": arm_a["retreated"],
                        }
                        _write_json_atomic(
                            artifact_dir / "offline_gt" / "bin01_handoff.json",
                            handoff_report,
                        )
                        handoff_failures = _handoff_precondition_failures(
                            placement,
                            arm_a,
                        )
                        if handoff_failures:
                            message = "HANDOFF NOT READY | " + " | ".join(
                                handoff_failures
                            )
                            status_label.text = message + " | THEN PRESS V AGAIN"
                            print(message)
                            continue
                        machine.enter_handoff_verify(
                            bin_at_handoff_center=bool(placement["pass"]),
                            bin_stable=True,
                            arm_a_gripper_open=arm_a["gripper_open"],
                            arm_a_clear=arm_a["retreated"],
                        )
                        active_arm = "NONE"
                        status_label.text = (
                            "HANDOFF_VERIFY PASS | both arms locked | press B"
                        )
                        print(
                            "HANDOFF VERIFIED: Bin_01 stable in HANDOFF_CENTER; "
                            "press B to activate Arm_B"
                        )
                        continue
                    if command.kind == "activate_b":
                        if not bin_transport_task:
                            raise RuntimeError(
                                f"{preflight.task_id} permits Arm_A only; "
                                "B is not allowed"
                            )
                        machine.activate_b_only()
                        active_arm = "Arm_B"
                        status_label.text = (
                            "B_ONLY | arm=Arm_B | continue to FINISHED_01"
                        )
                        print("CONTROL SWITCHED: Arm_B is now active")
                        continue
                    if command.kind == "complete":
                        if bin_transport_task:
                            placement = offline_gt_probe.bin01_in_finished01(
                                bin_path="/World/Bins/Bin_01",
                                stations=config["stations"],
                                bin_config=config["bin"],
                            )
                            arm_b = _arm_readback(controller, arms, config, "Arm_B")
                            machine.complete(
                                bin_at_finished=bool(placement["pass"]),
                                bin_stable=True,
                                arm_b_gripper_open=arm_b["gripper_open"],
                                arm_b_clear=arm_b["retreated"],
                            )
                        else:
                            arm_a = _arm_readback(controller, arms, config, "Arm_A")
                            machine.complete(
                                arm_a_gripper_open=arm_a["gripper_open"],
                                arm_a_clear=arm_a["retreated"],
                            )
                        print(f"HUMAN CONFIRMED {preflight.task_id} COMPLETE")
                        terminal_hold_requested = True
                        running = False
                        continue
                    if command.action is None:
                        raise RuntimeError("action command contains no ActionStep")
                    repeat_count = _interactive_action_repeat_count(command)
                    if action_count + repeat_count > args.max_actions:
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
                    for repeat_index in range(repeat_count):
                        _record_and_execute_formal_action(
                            bridge=bridge,
                            controller=controller,
                            action=command.action,
                            arm_id=active_arm,
                            task_id=preflight.task_id,
                            episode_id=preflight.episode_id,
                            action_index=action_count,
                            bin_grasp_manager=bin_grasp_manager,
                        )
                        action_count += 1
                        if repeat_count > 1:
                            print(
                                f"GRIPPER SETTLE {repeat_index + 1}/{repeat_count} "
                                f"actions={action_count}"
                            )
                    grasp_status = ""
                    if bin_grasp_manager is not None and command.key == "g":
                        gripper_open = float(command.action.values[6]) >= 0.5
                        if gripper_open:
                            grasp_status = " | BIN GRASP RELEASED"
                            print("BIN GRASP RELEASED")
                        elif bin_grasp_manager.attached_arm == active_arm:
                            grasp_status = " | BIN GRASP LOCKED"
                            print(f"BIN GRASP LOCKED: {active_arm}")
                        else:
                            distance = bin_grasp_manager.diagnostics()[
                                "last_attach_distance_m"
                            ]
                            distance_text = (
                                "unknown"
                                if distance is None
                                else f"{float(distance) * 1000.0:.1f} mm"
                            )
                            grasp_status = " | GRASP NOT LOCKED"
                            print(
                                "GRASP NOT LOCKED: align the gripper with "
                                f"BIN_CARRY_TCP (distance={distance_text}), "
                                "open with G, reposition, then close with G"
                            )
                    status_label.text = (
                        f"{machine.token.value} | arm={active_arm} | "
                        f"next={machine.next_part_id} | actions={action_count}"
                        f"{grasp_status}"
                    )
                    print(f"ACTION {action_count} {active_arm}: {command.description}")
                except BaseException:
                    running = False
                    raise

        if terminal_hold_requested and machine.outcome is EpisodeOutcome.SUCCEEDED:
            phase = "terminal_hold_offline_gt"
            terminal_collectors = {
                "P01_TO_S11": _collect_p01_terminal_success,
                "W01_TO_S14": _collect_w01_terminal_success,
                "BIN01_TO_FINISHED01": _collect_bin01_terminal_success,
            }
            terminal_collector = terminal_collectors[preflight.task_id]
            terminal_success_report, terminal_success_path, action_count = (
                terminal_collector(
                    bridge=bridge,
                    controller=controller,
                    probe=offline_gt_probe,
                    config=config,
                    artifact_dir=artifact_dir,
                    task_id=preflight.task_id,
                    episode_id=preflight.episode_id,
                    action_count=action_count,
                    max_actions=args.max_actions,
                )
            )
            if not terminal_success_report.passed:
                from simulation.v2_collection_state import V2FailureCode

                code = terminal_success_report.failure_codes[0]
                machine.fail_offline_gt(V2FailureCode(code))

        phase = "safe_stop"
        receipt = controller.safe_stop("V2 collection finished")
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
                "bin_grasp": (
                    bin_grasp_manager.diagnostics()
                    if bin_grasp_manager is not None
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
        if bin_grasp_manager is not None:
            try:
                result.setdefault("bin_grasp", bin_grasp_manager.diagnostics())
                bin_grasp_manager.detach()
            except BaseException as grasp_exc:
                result.setdefault("bin_grasp_cleanup_error", repr(grasp_exc))
        if terminal_success_path is not None:
            result.setdefault("offline_gt_path", str(terminal_success_path))
        if terminal_success_report is not None:
            result.setdefault("terminal_success", terminal_success_report.to_dict())
        if controller is not None:
            try:
                result.setdefault(
                    "controller_diagnostics",
                    {
                        arm_id: controller.diagnostics(arm_id)
                        for arm_id in ("Arm_A", "Arm_B")
                    },
                )
            except BaseException as diagnostics_exc:
                result.setdefault("controller_diagnostics_error", repr(diagnostics_exc))
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
