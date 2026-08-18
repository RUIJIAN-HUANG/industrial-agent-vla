from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from industrial_agent.contracts import ActionStep
from industrial_agent.data.recorder import CanonicalRecorder, EpisodeMetadata
from industrial_agent.data.replay import CanonicalEpisodeReader, OfflineEpisodeReplay
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from simulation.canonical_recorder_bridge import CanonicalRecorderBridge


class _RgbPipeline:
    def __init__(self, image_cas: ImageCas):
        self.image_cas = image_cas
        self.calls = 0

    def capture_references(self):
        self.calls += 1
        frame = np.full((720, 1280, 3), self.calls, dtype=np.uint8)
        return {
            camera_id: self.image_cas.write_rgb(frame, camera_id=camera_id).to_dict()
            for camera_id in ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
        }


class CanonicalRecorderBridgeTests(unittest.TestCase):
    def test_one_action_produces_reader_verified_multirate_episode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_cas = ImageCas(ImageCasConfig(root=root / "cas"))
            recorder = CanonicalRecorder(
                root / "episodes",
                EpisodeMetadata(
                    episode_id="bridge-test",
                    task_id="scripted-expert-integration",
                    instruction="automated integration test",
                    scene_seed=7,
                    git_sha="a" * 40,
                    scene_config_sha256="sha256:" + "b" * 64,
                ),
                image_cas=image_cas,
            )
            state = {
                "Arm_A": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                "Arm_B": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
            }
            bridge = CanonicalRecorderBridge(
                recorder=recorder,
                rgb_pipeline=_RgbPipeline(image_cas),
                state_source=lambda: state,
                timestamp_origin_ns=1_000_000_000,
            )
            bridge.record_initial()
            action = ActionStep.from_sequence(
                [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                duration_ms=100,
            )
            bridge.record_action(
                action,
                arm_id="Arm_A",
                subtask_id="integration",
                chunk_id="integration-000",
                physics_tick=0,
            )
            for tick in range(1, 13):
                bridge.observe_physics_tick(tick, tick % 4 == 0)
            episode_path = bridge.save(outcome="SUCCEEDED")

            with CanonicalEpisodeReader(episode_path) as reader:
                self.assertEqual(reader.camera_frames("CAM_A_TOP").shape[0], 4)
                self.assertEqual(reader.state_stream("Arm_A")["state_7d"].shape[0], 7)
                actions = OfflineEpisodeReplay(reader).actions()
                self.assertEqual(len(actions), 1)
                self.assertEqual(actions[0].arm_id, "Arm_A")

    def test_tick_gap_fails_closed(self):
        bridge = object.__new__(CanonicalRecorderBridge)
        bridge._closed = False
        bridge._initial_recorded = True
        bridge._last_tick = 0
        with self.assertRaisesRegex(ValueError, "contiguous"):
            bridge.observe_physics_tick(2, False)


if __name__ == "__main__":
    unittest.main()
