from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from industrial_agent.contracts import ActionStep
from industrial_agent.data import (
    CanonicalV2EpisodeMetadata,
    CanonicalV2Recorder,
    SplitRegistry,
)
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from scripts.pi05.convert_openpi_v2 import preflight_canonical_v2_windows
from simulation.canonical_recorder_bridge import CanonicalRecorderBridge
from simulation.run_v2_keyboard_collection import (
    GRIPPER_SETTLE_ACTION_COUNT,
    _collect_p01_terminal_success,
    _interactive_action_repeat_count,
    _record_and_execute_formal_action,
    _replay_task_actions_from_rows,
    _validate_replay_source_metadata,
)


def test_gripper_toggle_gets_five_recorded_settle_actions() -> None:
    assert GRIPPER_SETTLE_ACTION_COUNT == 5
    assert _interactive_action_repeat_count(SimpleNamespace(key="g")) == 5
    assert _interactive_action_repeat_count(SimpleNamespace(key="q")) == 1


class _RgbPipeline:
    def __init__(self, image_cas: ImageCas) -> None:
        self.image_cas = image_cas
        self.calls = 0

    def capture_references(self):
        self.calls += 1
        frame = np.full((720, 1280, 3), self.calls, dtype=np.uint8)
        return {
            camera_id: self.image_cas.write_rgb(
                frame,
                camera_id=camera_id,
            ).to_dict()
            for camera_id in ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
        }


class _LiveController:
    def __init__(self) -> None:
        self.physics_tick_index = 0
        self._observer = None

    def set_tick_observer(self, observer) -> None:
        self._observer = observer

    def execute_action(self, action, *, arm_id: str) -> None:
        del action, arm_id
        assert self._observer is not None
        for _ in range(12):
            self.physics_tick_index += 1
            self._observer(
                self.physics_tick_index,
                self.physics_tick_index % 4 == 0,
            )


class _TerminalProbe:
    def world_position(self, path: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def part_vertical_error_rad(self, *, part_path: str, bin_path: str) -> float:
        return float(np.deg2rad(5.0))

    def p01_in_s11(
        self,
        *,
        part_path: str,
        bin_path: str,
        bin_config,
    ) -> dict[str, object]:
        containment = {"pass": True, "slot_id": "S11"}
        return {
            "pass": True,
            "slot_id": "S11",
            "containment": containment,
        }


def test_replay_strips_exactly_ten_canonical_terminal_holds() -> None:
    task_row = np.asarray([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hold_row = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    actions = _replay_task_actions_from_rows([task_row] + [hold_row] * 10)
    assert len(actions) == 1
    np.testing.assert_array_equal(actions[0].values, task_row)


def test_replay_rejects_noncanonical_terminal_suffix() -> None:
    task_row = np.asarray([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    bad_hold = np.asarray([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "exactly ten canonical"):
        _replay_task_actions_from_rows([task_row] + [bad_hold] * 10)


def test_replay_rejects_scene_config_mismatch() -> None:
    with np.testing.assert_raises_regex(ValueError, "scene config SHA-256"):
        _validate_replay_source_metadata(
            {"scene_config_sha256": "sha256:" + "a" * 64},
            expected_scene_config_sha256="sha256:" + "b" * 64,
        )


def test_formal_keyboard_actions_and_terminal_holds_remain_exactly_12_ticks_and_pass_v2_preflight(
    tmp_path: Path,
) -> None:
    episode_id = "v2-live-timing-000001"
    episode_root = tmp_path / "episodes"
    image_cas = ImageCas(ImageCasConfig(root=tmp_path / "cas"))
    metadata = CanonicalV2EpisodeMetadata(
        episode_id=episode_id,
        task_id="P01_TO_S11",
        instruction="请将螺母 P01 放置到料箱的 S11 格子中。",
        scene_seed=31,
        git_sha="a" * 40,
        scene_config_sha256=f"sha256:{'b' * 64}",
        scene_id="single_bin_manual_industrial_v2",
    )
    recorder = CanonicalV2Recorder(
        episode_root,
        metadata,
        image_cas=image_cas,
    )
    state = {
        "Arm_A": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        "Arm_B": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
    }
    bridge = CanonicalRecorderBridge(
        recorder=recorder,
        rgb_pipeline=_RgbPipeline(image_cas),
        state_source=lambda: state,
        timestamp_origin_ns=1_000_000_000,
    )
    controller = _LiveController()
    bridge.record_initial(physics_tick=0)
    controller.set_tick_observer(bridge.observe_physics_tick)

    action = ActionStep.from_sequence(
        [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        duration_ms=100,
    )
    for index in range(10):
        _record_and_execute_formal_action(
            bridge=bridge,
            controller=controller,
            action=action,
            arm_id="Arm_A",
            task_id="P01_TO_S11",
            episode_id=episode_id,
            action_index=index,
        )

    terminal_result, _, action_count = _collect_p01_terminal_success(
        bridge=bridge,
        controller=controller,
        probe=_TerminalProbe(),
        config={"scene_id": "single_bin_manual_industrial_v2", "bin": {}},
        artifact_dir=tmp_path,
        task_id="P01_TO_S11",
        episode_id=episode_id,
        action_count=10,
        max_actions=50,
    )
    assert terminal_result.passed is True
    assert action_count == 20

    episode_path = bridge.save(outcome="SUCCEEDED")

    with h5py.File(episode_path / "episode.h5", "r") as episode:
        physics_ticks = np.asarray(episode["actions/physics_tick"][:], dtype=np.int64)
        timestamps_ns = np.asarray(episode["actions/timestamp_ns"][:], dtype=np.int64)
        actions = np.asarray(episode["actions/action_7d"][:], dtype=np.float32)

    assert physics_ticks.tolist() == list(range(0, 240, 12))
    assert np.all(np.diff(physics_ticks) == 12)
    assert np.all(np.diff(timestamps_ns) == 100_000_000)
    np.testing.assert_array_equal(
        actions[-10:],
        np.tile(
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (10, 1),
        ),
    )

    registry = SplitRegistry()
    registry.assign_episode(
        episode_id,
        "train",
        scenario_group_id="group-v2-live-timing",
        scene_seed=31,
        asset_variant="v2-fixed",
        camera_seed=41,
        lighting_seed=51,
    )
    report = preflight_canonical_v2_windows(
        data_dir=episode_path.parent,
        split_registry=registry,
    )
    assert report["counts"]["actions"] == 20
    assert report["counts"]["windows"] == 11
