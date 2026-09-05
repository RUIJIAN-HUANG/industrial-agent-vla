from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from scripts.pi05.canonical_v2 import (
    CanonicalV2Error,
    CanonicalV2Reader,
    EXPECTED_INSTRUCTION,
    EXPECTED_SCENE_ID,
    EXPECTED_TASK_ID,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_V1 = REPO_ROOT / "tests" / "fixtures" / "golden_episode_v1"


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _refresh_sha(episode_path: Path) -> None:
    structure_path = episode_path / "structure.json"
    manifest = json.loads(structure_path.read_text(encoding="utf-8"))
    manifest["storage"]["sha256"] = _sha256(episode_path / "episode.h5")
    structure_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_v2_episode(tmp_path: Path) -> Path:
    episode_path = tmp_path / "episode-v2"
    shutil.copytree(GOLDEN_V1, episode_path)
    structure_path = episode_path / "structure.json"
    manifest = json.loads(structure_path.read_text(encoding="utf-8"))
    manifest["canonical_schema_version"] = "2.0"
    del manifest["schema_version"]
    manifest["metadata"].update(
        {
            "scene_id": EXPECTED_SCENE_ID,
            "task_id": EXPECTED_TASK_ID,
            "instruction": EXPECTED_INSTRUCTION,
            "padding_policy": {"strategy": "none", "target_length": None},
        }
    )
    structure_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with h5py.File(episode_path / "episode.h5", "r+") as h5:
        del h5.attrs["schema_version"]
        h5.attrs["canonical_schema_version"] = "2.0"
        h5.attrs["scene_id"] = EXPECTED_SCENE_ID
        h5.attrs["task_id"] = EXPECTED_TASK_ID
        h5.attrs["instruction"] = EXPECTED_INSTRUCTION
        h5.attrs["padding_strategy"] = "none"
        h5.attrs["padding_target_length"] = -1
        h5["actions/subtask_id"][0] = EXPECTED_TASK_ID
    _refresh_sha(episode_path)
    return episode_path


def test_canonical_v2_reader_accepts_frozen_episode(tmp_path: Path) -> None:
    episode_path = _build_v2_episode(tmp_path)

    with CanonicalV2Reader(episode_path) as reader:
        assert reader.manifest["canonical_schema_version"] == "2.0"
        assert reader.manifest["metadata"]["scene_id"] == EXPECTED_SCENE_ID
        assert reader.state_7d("Arm_A").shape == (6, 7)
        actions = tuple(reader.iter_action_7d())
        assert len(actions) == 1
        assert actions[0].dtype == np.float32
        assert actions[0].shape == (7,)


@pytest.mark.parametrize(
    ("dataset_path", "index", "value", "message"),
    [
        ("robot_state/Arm_A/state_7d", (0, 0), np.nan, "NaN or Infinity"),
        ("robot_state/Arm_B/state_7d", (0, 6), 1.1, "state gripper"),
        ("actions/action_7d", (0, 2), np.inf, "NaN or Infinity"),
        ("actions/action_7d", (0, 6), 0.5, "action gripper"),
        ("actions/valid_mask", 0, False, "padding/masked"),
    ],
)
def test_canonical_v2_reader_rejects_invalid_values_and_padding(
    tmp_path: Path,
    dataset_path: str,
    index: Any,
    value: Any,
    message: str,
) -> None:
    episode_path = _build_v2_episode(tmp_path)
    with h5py.File(episode_path / "episode.h5", "r+") as h5:
        h5[dataset_path][index] = value
    _refresh_sha(episode_path)

    with pytest.raises(CanonicalV2Error, match=message):
        CanonicalV2Reader(episode_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm_id", "Arm_B"),
        ("executor", "retired_executor"),
        ("subtask_id", "P02_TO_S21"),
    ],
)
def test_canonical_v2_reader_rejects_wrong_action_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    episode_path = _build_v2_episode(tmp_path)
    with h5py.File(episode_path / "episode.h5", "r+") as h5:
        h5[f"actions/{field}"][0] = value
    _refresh_sha(episode_path)

    with pytest.raises(CanonicalV2Error, match=f"actions.{field}"):
        CanonicalV2Reader(episode_path)


def test_canonical_v2_reader_rejects_non_frozen_manifest_identity(
    tmp_path: Path,
) -> None:
    episode_path = _build_v2_episode(tmp_path)
    structure_path = episode_path / "structure.json"
    manifest = json.loads(structure_path.read_text(encoding="utf-8"))
    manifest["metadata"]["instruction"] = "把P02放到S21中"
    structure_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(CanonicalV2Error, match="manifest validation failed"):
        CanonicalV2Reader(episode_path)


def test_canonical_v2_reader_rejects_hdf5_identity_mismatch(tmp_path: Path) -> None:
    episode_path = _build_v2_episode(tmp_path)
    with h5py.File(episode_path / "episode.h5", "r+") as h5:
        h5.attrs["scene_id"] = "single_bin_static_handoff_v1"
    _refresh_sha(episode_path)

    with pytest.raises(CanonicalV2Error, match="attrs.scene_id"):
        CanonicalV2Reader(episode_path)
