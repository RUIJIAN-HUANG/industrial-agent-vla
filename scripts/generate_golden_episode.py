#!/usr/bin/env python3
"""Generate the compact, contract-frozen Golden Episode test fixture."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_agent.data import (  # noqa: E402
    CanonicalEpisodeReader,
    CanonicalRecorder,
    EpisodeMetadata,
)
from industrial_agent.image_cas import ImageCas, ImageCasConfig  # noqa: E402
from industrial_agent.sync_contract import canonical_state_7d  # noqa: E402


EPISODE_ID = "golden_episode_v1"
CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
BASE_TIMESTAMP_NS = 1_800_000_000_000_000_000
RENDER_FRAME_COUNT = 3
STATE_FRAME_COUNT = 6


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _file_sha256(path: Path) -> str:
    return f"sha256:{sha256(path.read_bytes()).hexdigest()}"


def _synthetic_rgb(camera_id: str, sequence_id: int) -> np.ndarray:
    """Build a deterministic, scene-like 1280x720 RGB observation."""

    height, width = 720, 1280
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.asarray(38 + x // 32 + y // 48, dtype=np.uint8)
    frame[..., 1] = np.asarray(44 + x // 48 + y // 32, dtype=np.uint8)
    frame[..., 2] = np.asarray(52 + x // 40 + y // 40, dtype=np.uint8)

    motion = sequence_id * 4
    if camera_id == "CAM_A_TOP":
        # Four red parts and the blue source bin in the Arm_A top view.
        for left, top in ((180, 180), (360, 250), (540, 170), (690, 310)):
            frame[top : top + 72, left + motion : left + motion + 96] = (205, 38, 32)
        frame[430:650, 850:1160] = (35, 86, 150)
    elif camera_id == "CAM_HANDOFF":
        # Green handoff marker and a centered blue bin.
        frame[180:540, 420:860] = (42, 118, 68)
        frame[270:470, 520 + motion : 780 + motion] = (38, 82, 148)
    elif camera_id == "CAM_B_TOP":
        # Yellow finished-zone marker and the transported blue bin.
        frame[150:570, 760:1160] = (190, 158, 42)
        frame[300:520, 390 + motion : 690 + motion] = (36, 84, 152)
    else:  # pragma: no cover - guarded by the frozen constant above
        raise ValueError(f"unsupported camera_id: {camera_id}")
    return frame


def generate(output_root: Path) -> Path:
    """Record and verify one self-contained Golden Episode."""

    fixture_path = output_root / EPISODE_ID
    if fixture_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Golden Episode: {fixture_path}"
        )

    config_path = REPO_ROOT / "configs" / "v2-task-profile.json"
    task_profile = json.loads(config_path.read_text(encoding="utf-8"))
    arm_a_instruction = next(
        task["instruction"]
        for task in task_profile["tasks"]
        if task["active_arm"] == "Arm_A"
    )
    with TemporaryDirectory(prefix="golden-episode-cas-") as temporary_cas:
        image_cas = ImageCas(ImageCasConfig(root=Path(temporary_cas)))
        recorder = CanonicalRecorder(
            output_root,
            EpisodeMetadata(
                episode_id=EPISODE_ID,
                task_id="golden-task-v1",
                instruction=arm_a_instruction,
                scene_seed=20260803,
                git_sha=_git_sha(),
                scene_config_sha256=_file_sha256(config_path),
            ),
            image_cas=image_cas,
        )
        with recorder:
            for sequence_id in range(RENDER_FRAME_COUNT):
                physics_tick = sequence_id * 4
                timestamp_ns = BASE_TIMESTAMP_NS + round(
                    sequence_id * 1_000_000_000 / 30
                )
                for camera_id in CAMERA_IDS:
                    reference = image_cas.write_rgb(
                        _synthetic_rgb(camera_id, sequence_id),
                        camera_id=camera_id,
                    )
                    recorder.add_frame(
                        camera_id=camera_id,
                        timestamp_ns=timestamp_ns,
                        physics_tick=physics_tick,
                        sequence_id=sequence_id,
                        image_reference=reference,
                    )

            for sequence_id in range(STATE_FRAME_COUNT):
                physics_tick = sequence_id * 2
                timestamp_ns = BASE_TIMESTAMP_NS + round(
                    sequence_id * 1_000_000_000 / 60
                )
                recorder.add_state(
                    arm_id="Arm_A",
                    timestamp_ns=timestamp_ns,
                    physics_tick=physics_tick,
                    sequence_id=sequence_id,
                    state_7d=canonical_state_7d(
                        [
                            0.450 + sequence_id * 0.001,
                            -0.120,
                            0.420 - sequence_id * 0.0005,
                            0.0,
                            0.020,
                            -0.010,
                        ],
                        sequence_id < 3,
                    ),
                )
                recorder.add_state(
                    arm_id="Arm_B",
                    timestamp_ns=timestamp_ns,
                    physics_tick=physics_tick,
                    sequence_id=sequence_id,
                    state_7d=canonical_state_7d(
                        [0.400, 0.180, 0.430, 0.0, -0.015, 0.005],
                        True,
                    ),
                )

            recorder.add_action_chunk(
                arm_id="Arm_A",
                executor="pi05",
                subtask_id="S01_ARM_A_PACK_HANDOFF",
                chunk_id="golden-chunk-001",
                start_timestamp_ns=BASE_TIMESTAMP_NS,
                start_physics_tick=0,
                start_sequence_id=0,
                actions=[[0.006, -0.002, -0.004, 0.0, 0.010, -0.010, 0.0]],
            )
            episode_path = recorder.save_episode(outcome="SUCCEEDED")

    with CanonicalEpisodeReader(episode_path) as reader:
        if any(
            reader.camera_frames(camera_id).shape != (RENDER_FRAME_COUNT, 720, 1280, 3)
            for camera_id in CAMERA_IDS
        ):
            raise RuntimeError("generated Golden Episode has invalid RGB streams")
        if len(tuple(reader.iter_valid_actions())) != 1:
            raise RuntimeError("generated Golden Episode has invalid actions")
    return episode_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures",
        help="parent directory for golden_episode_v1",
    )
    args = parser.parse_args()
    episode_path = generate(args.output_root.resolve())
    print(f"Generated and verified {episode_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
