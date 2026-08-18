"""Run one physical P01 scripted-expert TEST episode in Isaac Sim 5.1.

This is the first task-level gate after the Recorder integration smoke.  It is
always assigned to the TEST split and is never training-eligible.  Isaac ground
truth is read only through :mod:`offline_gt`; detailed values are written only
below the sibling ``offline_gt`` artifact directory.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
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
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "scripted-expert-p01-smoke"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic P01 pick/place TEST gate with Canonical recording."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--franka-usd")
    parser.add_argument("--approach-clearance-m", type=float, default=0.10)
    parser.add_argument("--max-cartesian-step-m", type=float, default=0.02)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _clean_git_sha() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("P01 collection requires a clean committed Git tree")
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise RuntimeError("git rev-parse HEAD did not return a full SHA")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    for path in (REPOSITORY_ROOT, SOURCE_DIR, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import isaac_compat
    import scene_layout
    from scripted_expert_plan import P01ExpertTuning

    tuning = P01ExpertTuning(
        approach_clearance_m=args.approach_clearance_m,
        max_cartesian_step_m=args.max_cartesian_step_m,
    )
    tuning.validate()
    config_path = args.config.expanduser().resolve()
    config = scene_layout.load_config(config_path)
    errors = scene_layout.validate_scene_config(config)
    if errors:
        raise ValueError("Frozen scene contract failed: " + "; ".join(errors))

    git_sha = _clean_git_sha()
    scene_config_sha256 = _file_digest(config_path)
    episode_id = f"scripted-expert-p01-smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    artifact_root = args.artifact_root.expanduser().resolve()
    episode_root = artifact_root / "episodes"
    cas_root = artifact_root / "cas"
    offline_gt_root = artifact_root / "offline_gt"
    result_path = artifact_root / f"{episode_id}-result.json"
    registry_path = artifact_root / "split_registry.json"
    phase = "launch_simulation_app"
    bridge = None
    rgb_pipeline = None
    controller = None
    simulation_app = isaac_compat.launch_simulation_app(headless=args.headless)
    try:
        phase = "verify_isaac_version"
        isaac_version = isaac_compat.require_isaac_sim_51()
        import single_bin_scene_builder
        from canonical_recorder_bridge import CanonicalRecorderBridge
        from isaac_franka_controller import (
            IsaacSimFrankaController,
            _quat_inverse,
            _rotate_vector,
        )
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        from industrial_agent.contracts import ActionStep
        from industrial_agent.data.recorder import CanonicalRecorder, EpisodeMetadata
        from industrial_agent.data.replay import (
            CanonicalEpisodeReader,
            OfflineEpisodeReplay,
        )
        from industrial_agent.data.split_registry import SplitRegistry
        from industrial_agent.image_cas import ImageCas, ImageCasConfig
        from offline_gt import OfflineGtProbe
        from scripted_expert_plan import (
            bin_slot_local_centers,
            bounded_world_delta,
            conservative_step_limit,
            frozen_success_vote,
            grasp_follow_report,
            minimum_symmetric_finger_contact_m,
            motion_sample_violation,
            orthogonal_transfer_waypoints,
            select_safest_slot_index,
            symmetric_finger_contact_report,
            top_down_tilt_error_rad,
            yaw_preserving_top_down_rotation,
        )
        from simulation.isaac_rgb_pipeline import IsaacRgbObservationPipeline
        from simulation.rgb_cas_bridge import IsaacRgbCasPublisher
        from simulation.run_g0_acceptance import _write_explicit_home
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
        isaac_compat.wait_for_stage_loading(simulation_app, timeout_seconds=180.0)
        isaac_compat.save_stage_checked(args.output_scene)

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
                    name=f"p01_smoke_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }
        world.reset()
        for arm_id, arm in arms.items():
            _write_explicit_home(config, arm, arm_id)
        for _ in range(120):
            world.step(render=True)

        phase = "initialize_recorder_and_offline_gt"
        controller = IsaacSimFrankaController(
            world=world,
            arms=arms,
            physics_dt_s=float(physics["physics_dt_s"]),
            virtual_tcp_fingertip_frame_names=(
                "panda_leftfingertip",
                "panda_rightfingertip",
            ),
        )
        gt = OfflineGtProbe(isaac_compat.get_current_stage())
        image_cas = ImageCas(ImageCasConfig(root=cas_root))
        image_cas.assert_ready(writable=True)
        publisher = IsaacRgbCasPublisher.from_scene_config(image_cas, config)
        rgb_pipeline = IsaacRgbObservationPipeline(
            simulation_app=simulation_app,
            scene_config=config,
            publisher=publisher,
        )
        metadata = EpisodeMetadata(
            episode_id=episode_id,
            task_id="B-SCRIPTED-EXPERT-P01-SMOKE-NOT-TRAINING",
            instruction=(
                "TEST only: Arm_A places P01 into a guarded Bin_01 slot; "
                "offline_gt is excluded from all Canonical fields"
            ),
            scene_seed=0,
            git_sha=git_sha,
            scene_config_sha256=scene_config_sha256,
        )
        recorder = CanonicalRecorder(episode_root, metadata, image_cas=image_cas)

        def state_source() -> dict[str, list[float]]:
            return {
                arm_id: list(
                    _arm_state(controller, arm_id, arms[arm_id], config)["state"]
                )
                for arm_id in ("Arm_A", "Arm_B")
            }

        bridge = CanonicalRecorderBridge(
            recorder=recorder,
            rgb_pipeline=rgb_pipeline,
            state_source=state_source,
        )
        bridge.record_initial(physics_tick=controller.physics_tick_index)

        vote_reports: list[dict[str, Any]] = []
        vote_mode = False
        bin_config = config["bin"]

        def observe(physics_tick: int, render_due: bool) -> None:
            bridge.observe_physics_tick(physics_tick, render_due)
            if vote_mode and render_due and len(vote_reports) < 3:
                report = gt.part_fully_inside_bin(
                    part_path="/World/Parts/P01",
                    bin_path="/World/Bins/Bin_01",
                    bin_config=bin_config,
                    numerical_tolerance_m=0.001,
                )
                report["vote_index"] = len(vote_reports) + 1
                report["physics_tick"] = physics_tick
                vote_reports.append(report)

        controller.set_tick_observer(observe)
        action_index = 0
        motion_diagnostics_path = offline_gt_root / "p01_motion_diagnostics.json"
        motion_diagnostics: dict[str, dict[str, Any]] = {}

        def execute(
            *,
            subtask_id: str,
            base_delta: np.ndarray,
            gripper_open: bool,
            base_rotation_delta: np.ndarray | None = None,
        ) -> None:
            nonlocal action_index
            rotation = (
                np.zeros(3, dtype=float)
                if base_rotation_delta is None
                else np.asarray(base_rotation_delta, dtype=float)
            )
            if rotation.shape != (3,) or not np.all(np.isfinite(rotation)):
                raise ValueError("base_rotation_delta must be a finite 3-D vector")
            values = [
                float(base_delta[0]),
                float(base_delta[1]),
                float(base_delta[2]),
                float(rotation[0]),
                float(rotation[1]),
                float(rotation[2]),
                1.0 if gripper_open else 0.0,
            ]
            action = ActionStep.from_sequence(values, duration_ms=100)
            bridge.record_action(
                action,
                arm_id="Arm_A",
                subtask_id=subtask_id,
                chunk_id=f"p01-smoke-{action_index:04d}",
                physics_tick=controller.physics_tick_index,
            )
            controller.execute_action(action, arm_id="Arm_A")
            action_index += 1

        def move_tcp_world(
            *,
            subtask_id: str,
            target_world: np.ndarray,
            gripper_open: bool,
        ) -> None:
            initial_position, _ = controller.end_effector_pose("Arm_A")
            samples: list[dict[str, Any]] = []
            limit = conservative_step_limit(
                float(np.linalg.norm(target_world - initial_position)),
                tuning.max_cartesian_step_m,
            )
            consecutive_divergent_steps = 0
            consecutive_stalled_steps = 0

            def persist_motion(status: str, error: str | None = None) -> None:
                motion_diagnostics[subtask_id] = {
                    "status": status,
                    "target_world_m": target_world.tolist(),
                    "position_tolerance_m": tuning.position_tolerance_m,
                    "minimum_progress_m": tuning.minimum_progress_m,
                    "step_limit": limit,
                    "samples": samples,
                    "error": error,
                }
                _write_json(
                    motion_diagnostics_path,
                    {
                        "isolation": "offline_gt_only",
                        "canonical_included": False,
                        "tcp_definition": controller.tcp_definition("Arm_A"),
                        "motions": motion_diagnostics,
                    },
                )

            for step_index in range(limit):
                current_position, _ = controller.end_effector_pose("Arm_A")
                error = target_world - current_position
                current_error_m = float(np.linalg.norm(error))
                if current_error_m <= tuning.position_tolerance_m:
                    persist_motion("completed")
                    return
                world_delta = bounded_world_delta(
                    current_position,
                    target_world,
                    max_step_m=tuning.max_cartesian_step_m,
                )
                _, base_orientation = arms["Arm_A"].get_world_pose()
                base_delta = _rotate_vector(
                    _quat_inverse(np.asarray(base_orientation, dtype=float)),
                    world_delta,
                )
                execute(
                    subtask_id=subtask_id,
                    base_delta=base_delta,
                    gripper_open=gripper_open,
                )
                after_position, _ = controller.end_effector_pose("Arm_A")
                after_error_m = float(np.linalg.norm(target_world - after_position))
                violation = motion_sample_violation(
                    current_position,
                    after_position,
                    target_world,
                    max_actual_step_m=tuning.max_actual_step_m,
                    divergence_tolerance_m=tuning.divergence_tolerance_m,
                )
                progress_m = current_error_m - after_error_m
                samples.append(
                    {
                        "step": step_index + 1,
                        "before_world_m": current_position.tolist(),
                        "after_world_m": after_position.tolist(),
                        "before_error_m": current_error_m,
                        "after_error_m": after_error_m,
                        "progress_m": progress_m,
                        "violation": violation,
                    }
                )
                persist_motion("moving")
                if violation is None:
                    consecutive_divergent_steps = 0
                elif not violation.startswith("TCP moved away"):
                    controller.safe_stop(f"{subtask_id}: {violation}")
                    persist_motion("safety_stop", violation)
                    raise RuntimeError(f"{subtask_id} safety stop: {violation}")
                else:
                    consecutive_divergent_steps += 1
                    if (
                        consecutive_divergent_steps
                        >= tuning.max_consecutive_divergent_steps
                    ):
                        controller.safe_stop(f"{subtask_id}: {violation}")
                        persist_motion("safety_stop", violation)
                        raise RuntimeError(f"{subtask_id} safety stop: {violation}")

                if after_error_m <= tuning.position_tolerance_m:
                    persist_motion("completed")
                    return

                if progress_m < tuning.minimum_progress_m:
                    consecutive_stalled_steps += 1
                else:
                    consecutive_stalled_steps = 0
                if (
                    after_error_m > tuning.position_tolerance_m
                    and consecutive_stalled_steps
                    >= tuning.max_consecutive_stalled_steps
                ):
                    message = (
                        f"virtual TCP stalled at {after_error_m:.6f} m after "
                        f"{consecutive_stalled_steps} low-progress steps"
                    )
                    controller.safe_stop(f"{subtask_id}: {message}")
                    persist_motion("stalled", message)
                    raise RuntimeError(f"{subtask_id} safety stop: {message}")
            final_position, _ = controller.end_effector_pose("Arm_A")
            remaining_error_m = float(np.linalg.norm(target_world - final_position))
            persist_motion(
                "step_limit",
                f"remaining_error_m={remaining_error_m:.6f}",
            )
            raise RuntimeError(
                f"Arm_A did not reach {subtask_id} within {limit} steps; initial_world_m={initial_position.tolist()}; target_world_m={target_world.tolist()}; final_world_m={final_position.tolist()}; remaining_error_m={remaining_error_m:.6f}"
            )

        def align_tcp_top_down(*, subtask_id: str) -> dict[str, Any]:
            """Align tool Z downward without constraining cylinder-irrelevant yaw."""

            _, initial_rotation = controller.end_effector_pose("Arm_A")
            initial_tilt_error = top_down_tilt_error_rad(initial_rotation)
            for step_index in range(tuning.max_rotation_steps):
                _, current_rotation = controller.end_effector_pose("Arm_A")
                tilt_error = top_down_tilt_error_rad(current_rotation)
                target_world_rotation = yaw_preserving_top_down_rotation(
                    current_rotation
                )
                if tilt_error <= tuning.rotation_tolerance_rad:
                    return {
                        "objective": "tool_z_to_world_minus_z_yaw_free",
                        "target_world_rotation": target_world_rotation.tolist(),
                        "initial_tilt_error_rad": initial_tilt_error,
                        "final_tilt_error_rad": tilt_error,
                        "steps": step_index,
                    }
                error = controller.world_orientation_error_in_base(
                    "Arm_A", target_world_rotation
                )
                error_norm = float(np.linalg.norm(error))
                rotation_delta = error
                if error_norm > tuning.max_rotation_step_rad:
                    rotation_delta = error * (tuning.max_rotation_step_rad / error_norm)
                execute(
                    subtask_id=subtask_id,
                    base_delta=np.zeros(3, dtype=float),
                    base_rotation_delta=rotation_delta,
                    gripper_open=True,
                )
            _, final_rotation = controller.end_effector_pose("Arm_A")
            final_tilt_error = top_down_tilt_error_rad(final_rotation)
            raise RuntimeError(
                "Arm_A did not align tool Z with world -Z; "
                f"remaining_tilt_error_rad={final_tilt_error:.6f}"
            )

        def hold(*, subtask_id: str, gripper_open: bool, steps: int) -> None:
            for _ in range(steps):
                execute(
                    subtask_id=subtask_id,
                    base_delta=np.zeros(3, dtype=float),
                    gripper_open=gripper_open,
                )

        part_path = "/World/Parts/P01"

        phase = "execute_p01_pick_place"
        part_config = next(part for part in config["parts"] if part["id"] == "P01")
        slot_locals = bin_slot_local_centers(
            size_m=bin_config["size_m"],
            wall_thickness_m=float(bin_config["wall_thickness_m"]),
            bottom_thickness_m=float(bin_config["bottom_thickness_m"]),
            part_height_m=float(part_config["geometry"]["height_m"]),
        )
        slot_part_centers = tuple(
            np.asarray(
                gt.local_point_to_world("/World/Bins/Bin_01", slot_local),
                dtype=float,
            )
            for slot_local in slot_locals
        )
        slot_tcp_candidates = tuple(
            center + np.asarray([0.0, 0.0, tuning.approach_clearance_m], dtype=float)
            for center in slot_part_centers
        )
        arm_config = next(robot for robot in config["robots"] if robot["id"] == "Arm_A")
        arm_base_world = np.asarray(arm_config["base_pose"]["position_m"], dtype=float)
        slot_index = select_safest_slot_index(
            slot_tcp_candidates,
            arm_base_world_m=arm_base_world,
            soft_work_radius_m=float(arm_config["soft_work_radius_m"]),
            work_radius_margin_m=tuning.slot_work_radius_margin_m,
        )
        slot_local = slot_locals[slot_index]
        verification_path = offline_gt_root / "p01_grasp_verification.json"
        calibration_path = offline_gt_root / "p01_pinch_calibration.json"
        live_part_center = np.asarray(gt.world_position(part_path), dtype=float)
        desired_approach_pinch = live_part_center + np.asarray(
            [0.0, 0.0, tuning.approach_clearance_m], dtype=float
        )
        move_tcp_world(
            subtask_id="p01-grasp-approach-position",
            target_world=desired_approach_pinch,
            gripper_open=True,
        )
        orientation_report = align_tcp_top_down(subtask_id="p01-grasp-align-top-down")
        live_part_center = np.asarray(gt.world_position(part_path), dtype=float)
        desired_approach_pinch = live_part_center + np.asarray(
            [0.0, 0.0, tuning.approach_clearance_m], dtype=float
        )
        move_tcp_world(
            subtask_id="p01-grasp-recenter-after-orientation",
            target_world=desired_approach_pinch,
            gripper_open=True,
        )
        live_part_center = np.asarray(gt.world_position(part_path), dtype=float)
        desired_grasp_pinch = live_part_center.copy()
        calibration_report = {
            "isolation": "offline_gt_only",
            "canonical_included": False,
            "method": "lula_two_fingertip_virtual_tcp",
            "tcp_definition": controller.tcp_definition("Arm_A"),
            "desired_grasp_center_world_m": desired_grasp_pinch.tolist(),
            "alignment_tolerance_m": tuning.position_tolerance_m,
            "status": "descending",
        }
        _write_json(calibration_path, calibration_report)
        move_tcp_world(
            subtask_id="p01-grasp-single-descend",
            target_world=desired_grasp_pinch,
            gripper_open=True,
        )
        aligned_pinch_world_m, _ = controller.end_effector_pose("Arm_A")
        alignment_residual_m = float(
            np.linalg.norm(desired_grasp_pinch - aligned_pinch_world_m)
        )
        calibration_report.update(
            {
                "status": "aligned",
                "measured_grasp_center_world_m": aligned_pinch_world_m.tolist(),
                "alignment_residual_m": alignment_residual_m,
            }
        )
        _write_json(calibration_path, calibration_report)
        if alignment_residual_m > tuning.position_tolerance_m:
            controller.safe_stop("virtual grasp center did not align with P01")
            raise RuntimeError(
                "GRASP_ALIGNMENT_FAILED: virtual grasp center missed P01; "
                "see offline_gt/p01_pinch_calibration.json"
            )
        hold(
            subtask_id="p01-grasp-single-close",
            gripper_open=False,
            steps=tuning.close_steps,
        )
        finger_positions_m = controller.gripper_joint_positions("Arm_A")
        minimum_contact_m = minimum_symmetric_finger_contact_m(
            part_radius_m=float(part_config["geometry"]["radius_m"]),
            minimum_contact_ratio=tuning.minimum_finger_contact_ratio,
        )
        finger_contact_report = symmetric_finger_contact_report(
            finger_positions_m,
            minimum_contact_m=minimum_contact_m,
        )
        probe_tcp_before, _ = controller.end_effector_pose("Arm_A")
        probe_part_before = np.asarray(gt.world_position(part_path), dtype=float)
        probe_target = probe_tcp_before + np.asarray(
            [0.0, 0.0, tuning.grasp_probe_lift_m], dtype=float
        )
        move_tcp_world(
            subtask_id="p01-grasp-single-probe-lift",
            target_world=probe_target,
            gripper_open=False,
        )
        probe_tcp_after, _ = controller.end_effector_pose("Arm_A")
        probe_part_after = np.asarray(gt.world_position(part_path), dtype=float)
        grasp_report = grasp_follow_report(
            tcp_before_world_m=probe_tcp_before,
            tcp_after_world_m=probe_tcp_after,
            part_before_world_m=probe_part_before,
            part_after_world_m=probe_part_after,
            minimum_follow_ratio=tuning.minimum_grasp_follow_ratio,
            maximum_follow_error_m=tuning.maximum_grasp_follow_error_m,
        )
        grasp_report.update(
            {
                "desired_virtual_tcp_world_m": desired_grasp_pinch.tolist(),
                "virtual_tcp_alignment_residual_m": alignment_residual_m,
                "tcp_before_world_m": probe_tcp_before.tolist(),
                "tcp_after_world_m": probe_tcp_after.tolist(),
                "part_before_world_m": probe_part_before.tolist(),
                "part_after_world_m": probe_part_after.tolist(),
                "finger_joint_positions_m": finger_positions_m.tolist(),
                "finger_contact": finger_contact_report,
                "orientation": orientation_report,
            }
        )
        grasp_passed = bool(grasp_report["pass"]) and bool(
            finger_contact_report["pass"]
        )
        _write_json(
            verification_path,
            {
                "isolation": "offline_gt_only",
                "canonical_included": False,
                "strategy": "lula_virtual_tcp_finger_and_lift_verified_grasp",
                "passed": grasp_passed,
                "attempt": grasp_report,
                "pinch_calibration_path": str(calibration_path),
            },
        )
        if not grasp_passed:
            controller.safe_stop("single verified P01 grasp failed")
            raise RuntimeError(
                "GRASP_FAILED: single top-down grasp failed finger/lift "
                "verification; see offline_gt/p01_grasp_verification.json"
            )

        carried_tcp_to_part_center = probe_tcp_after - probe_part_after
        remaining_clearance_m = max(
            0.02,
            tuning.approach_clearance_m - tuning.grasp_probe_lift_m,
        )
        grasp_approach = probe_tcp_after + np.asarray(
            [0.0, 0.0, remaining_clearance_m], dtype=float
        )
        move_tcp_world(
            subtask_id="p01-lift-after-verified-grasp",
            target_world=grasp_approach,
            gripper_open=False,
        )

        place_tcp = slot_part_centers[slot_index] + carried_tcp_to_part_center
        place_approach = place_tcp + np.asarray(
            [0.0, 0.0, tuning.approach_clearance_m], dtype=float
        )
        transfer_waypoints = orthogonal_transfer_waypoints(
            grasp_approach,
            place_approach,
            arm_base_world_m=arm_base_world,
            transit_clearance_m=tuning.transit_clearance_m,
        )
        _write_json(
            offline_gt_root / "p01_motion_plan.json",
            {
                "isolation": "offline_gt_only",
                "canonical_included": False,
                "slot_index": slot_index,
                "slot_local_m": slot_local.tolist(),
                "arm_base_world_m": arm_base_world.tolist(),
                "grasp_strategy": "lula_virtual_tcp_finger_and_lift_verified_grasp",
                "pinch_calibration_path": str(calibration_path),
                "verified_tcp_to_part_center_m": carried_tcp_to_part_center.tolist(),
                "grasp_approach_world_m": grasp_approach.tolist(),
                "place_tcp_world_m": place_tcp.tolist(),
                "place_approach_world_m": place_approach.tolist(),
                "transfer_waypoints_world_m": [
                    waypoint.tolist() for waypoint in transfer_waypoints
                ],
            },
        )
        for waypoint_index, waypoint in enumerate(transfer_waypoints, start=1):
            move_tcp_world(
                subtask_id=f"p01-transfer-{waypoint_index}",
                target_world=waypoint,
                gripper_open=False,
            )
        move_tcp_world(
            subtask_id="p01-place",
            target_world=place_tcp,
            gripper_open=False,
        )
        hold(subtask_id="p01-release", gripper_open=True, steps=tuning.release_steps)
        move_tcp_world(
            subtask_id="p01-retreat",
            target_world=place_approach,
            gripper_open=True,
        )

        phase = "three_fresh_frame_offline_gt_vote"
        vote_mode = True
        hold(subtask_id="p01-final-lock", gripper_open=True, steps=1)
        vote_mode = False
        if len(vote_reports) != 3:
            raise RuntimeError(
                f"expected exactly 3 fresh offline_gt frames, got {len(vote_reports)}"
            )
        task_succeeded = frozen_success_vote(
            [bool(report["pass"]) for report in vote_reports]
        )
        _write_json(
            offline_gt_root / "p01_containment_votes.json",
            {
                "isolation": "offline_gt_only",
                "canonical_included": False,
                "rule": "exactly 3 fresh frames; at least 2 whole-frame passes",
                "votes": vote_reports,
                "passed": task_succeeded,
            },
        )

        controller.set_tick_observer(None)
        phase = "publish_test_episode"
        episode_path = bridge.save(
            outcome="SUCCEEDED" if task_succeeded else "FAILED",
            failure_code=None if task_succeeded else "P01_NOT_FULLY_IN_BIN",
        )
        bridge = None
        registry = (
            SplitRegistry.load(registry_path)
            if registry_path.exists()
            else SplitRegistry()
        )
        registry.assign_episode(
            episode_id,
            "test",
            scenario_group_id="scripted-expert-p01-smoke",
            scene_seed=0,
            asset_variant="frozen-single-bin-v1",
            camera_seed=0,
            lighting_seed=0,
        )
        registry.save(registry_path)

        phase = "reader_replay_validation"
        with CanonicalEpisodeReader(
            episode_path,
            split_registry=registry,
            is_training=False,
        ) as reader:
            replay_actions = OfflineEpisodeReplay(reader).actions()
            camera_counts = {
                camera_id: int(reader.camera_frames(camera_id).shape[0])
                for camera_id in ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
            }
            state_counts = {
                arm_id: int(reader.state_stream(arm_id)["state_7d"].shape[0])
                for arm_id in ("Arm_A", "Arm_B")
            }
        if len(replay_actions) != action_index:
            raise RuntimeError("Reader replay action count does not match execution")

        result = {
            "status": "PASS" if task_succeeded else "TASK_FAILED",
            "smoke_only": True,
            "training_allowed": False,
            "split": "test",
            "task_scope": "P01_ONLY_NOT_FULL_FROZEN_TASK",
            "episode_id": episode_id,
            "episode_path": str(episode_path),
            "registry_path": str(registry_path),
            "offline_gt_report": str(offline_gt_root / "p01_containment_votes.json"),
            "offline_gt_in_canonical": False,
            "success_votes": sum(bool(report["pass"]) for report in vote_reports),
            "fresh_frame_count": len(vote_reports),
            "git_sha": git_sha,
            "scene_config_sha256": scene_config_sha256,
            "isaac_sim_version": isaac_version,
            "camera_counts": camera_counts,
            "state_counts": state_counts,
            "replay_action_count": len(replay_actions),
        }
        _write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if task_succeeded else 2
    except BaseException as exc:
        if controller is not None:
            try:
                controller.safe_stop(f"P01 smoke failed during {phase}: {exc}")
            except BaseException:
                pass
        if bridge is not None:
            try:
                bridge.abort()
            except BaseException:
                pass
        result = {
            "status": "ERROR",
            "smoke_only": True,
            "training_allowed": False,
            "task_scope": "P01_ONLY_NOT_FULL_FROZEN_TASK",
            "episode_id": episode_id,
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    finally:
        if rgb_pipeline is not None:
            try:
                rgb_pipeline.close()
            except BaseException:
                pass
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
