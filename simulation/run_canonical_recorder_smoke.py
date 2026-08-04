"""Run one automatic Isaac-to-Canonical integration episode.

This is a non-training TEST-split gate for the recorder wiring.  It deliberately
does not claim that the packing/handoff task succeeded.  The scripted expert
may only reuse this bridge after this gate passes and Reader verifies the saved
episode.
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v1.json"
DEFAULT_SCENE = SCRIPT_DIR / "generated" / "single_bin_scene_v1.usda"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "canonical-recorder-smoke"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic TEST-split Canonical Recorder integration smoke."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--franka-usd")
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
        raise RuntimeError(
            "Canonical collection requires a clean committed tree; commit the "
            "verified integration changes before running this smoke"
        )
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

    config_path = args.config.expanduser().resolve()
    config = scene_layout.load_config(config_path)
    errors = scene_layout.validate_scene_config(config)
    if errors:
        raise ValueError("Frozen scene contract failed: " + "; ".join(errors))

    git_sha = _clean_git_sha()
    scene_config_sha256 = _file_digest(config_path)
    episode_id = f"canonical-recorder-smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    artifact_root = args.artifact_root.expanduser().resolve()
    episode_root = artifact_root / "episodes"
    cas_root = artifact_root / "cas"
    result_path = artifact_root / f"{episode_id}-result.json"
    registry_path = artifact_root / "split_registry.json"
    phase = "launch_simulation_app"
    bridge = None
    rgb_pipeline = None
    simulation_app = isaac_compat.launch_simulation_app(headless=args.headless)
    try:
        phase = "verify_isaac_version"
        isaac_version = isaac_compat.require_isaac_sim_51()
        import single_bin_scene_builder
        from canonical_recorder_bridge import CanonicalRecorderBridge
        from isaac_franka_controller import IsaacSimFrankaController
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
                    name=f"canonical_smoke_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }
        world.reset()
        for arm_id, arm in arms.items():
            _write_explicit_home(config, arm, arm_id)
        for _ in range(120):
            world.step(render=True)

        phase = "initialize_recorder_bridge"
        controller = IsaacSimFrankaController(
            world=world,
            arms=arms,
            physics_dt_s=float(physics["physics_dt_s"]),
        )
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
            task_id="B-C-CANONICAL-RECORDER-INTEGRATION-SMOKE-NOT-TRAINING",
            instruction=(
                "TEST split only: verify synchronized Isaac RGB, dual-arm state, "
                "action recording, and offline Reader replay"
            ),
            scene_seed=0,
            git_sha=git_sha,
            scene_config_sha256=scene_config_sha256,
        )
        recorder = CanonicalRecorder(
            episode_root,
            metadata,
            image_cas=image_cas,
        )

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
        controller.set_tick_observer(bridge.observe_physics_tick)

        phase = "execute_automatic_motion"
        actions = (
            ("recorder-smoke-up", [0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 1.0]),
            ("recorder-smoke-down", [0.0, 0.0, -0.003, 0.0, 0.0, 0.0, 1.0]),
        )
        for index, (subtask_id, values) in enumerate(actions):
            action = ActionStep.from_sequence(values, duration_ms=100)
            bridge.record_action(
                action,
                arm_id="Arm_A",
                subtask_id=subtask_id,
                chunk_id=f"canonical-smoke-{index:03d}",
                physics_tick=controller.physics_tick_index,
            )
            controller.execute_action(action, arm_id="Arm_A")

        controller.set_tick_observer(None)
        phase = "publish_and_register_test_episode"
        episode_path = bridge.save(outcome="SUCCEEDED")
        bridge = None
        registry = (
            SplitRegistry.load(registry_path)
            if registry_path.exists()
            else SplitRegistry()
        )
        registry.assign_episode(
            episode_id,
            "test",
            scenario_group_id="canonical-recorder-integration-smoke",
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
        if len(replay_actions) != len(actions):
            raise RuntimeError("Reader replay action count does not match execution")

        result = {
            "status": "PASS",
            "smoke_only": True,
            "training_allowed": False,
            "split": "test",
            "episode_id": episode_id,
            "episode_path": str(episode_path),
            "registry_path": str(registry_path),
            "git_sha": git_sha,
            "scene_config_sha256": scene_config_sha256,
            "isaac_sim_version": isaac_version,
            "camera_counts": camera_counts,
            "state_counts": state_counts,
            "replay_action_count": len(replay_actions),
        }
        _write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        if bridge is not None:
            try:
                bridge.abort()
            except BaseException:
                pass
        result = {
            "status": "FAIL",
            "smoke_only": True,
            "training_allowed": False,
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
