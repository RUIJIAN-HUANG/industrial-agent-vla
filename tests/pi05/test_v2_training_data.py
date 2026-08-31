from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from configs.pi05.train_config import action_sequence_keys_for_input
from industrial_agent.data import DatasetSplit, SplitAssignment, SplitRegistry
from scripts.pi05.compute_norm_stats import compute_stats, validate_dimensions
from scripts.pi05.convert_openpi_v2 import (
    MANIFEST_SHA256_FILENAME,
    verify_conversion_manifest,
)
from scripts.pi05.lerobot_v2_norm_source import (
    V2NormSourceError,
    load_lerobot_v2_norm_source,
)


class FakeDataset:
    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._frames[index]


def _registry(split: DatasetSplit = DatasetSplit.TRAIN) -> SplitRegistry:
    return SplitRegistry(
        [
            SplitAssignment(
                episode_id="episode-001",
                split=split,
                scenario_group_id="fixed-scene",
                scene_seed=0,
                asset_variant="v2-fixed",
                camera_seed=0,
                lighting_seed=0,
            )
        ]
    )


def _frames() -> list[dict[str, Any]]:
    return [
        {
            "state": np.full(7, index, dtype=np.float32),
            "actions": np.arange(70, dtype=np.float32).reshape(10, 7) + index,
            "task": "把P01放到S11中",
            "episode_index": np.asarray(index * 0, dtype=np.int64),
            "frame_index": np.asarray(index, dtype=np.int64),
        }
        for index in range(2)
    ]


def _write_manifest(
    root: Path,
    registry: SplitRegistry,
    *,
    canonical_split: str = "train",
) -> Path:
    manifest = {
        "manifest_version": "1.0",
        "source_format": "canonical_hdf5_v2",
        "repo_id": "industrial/pi05-test",
        "action_horizon": 10,
        "action_shape": [10, 7],
        "padding": "forbidden",
        "window_rule": "N-9",
        "source_split_registry_sha256": registry.registry_sha256,
        "counts": {"episodes": 1, "windows": 2},
        "roundtrip": {"episodes": 1, "windows": 2, "max_action_error": 0.0},
        "episodes": [
            {
                "lerobot_episode_index": 0,
                "canonical_episode_id": "episode-001",
                "canonical_split": canonical_split,
                "source_action_count": 11,
                "window_count": 2,
                "source_hdf5_sha256": "sha256:" + "1" * 64,
                "source_structure_sha256": "sha256:" + "2" * 64,
                "window_start_action_indices": [0, 1],
            }
        ],
    }
    path = root / "pi05_v2_conversion.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / MANIFEST_SHA256_FILENAME).write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return path


def _write_segment_manifest(root: Path, registry: SplitRegistry) -> Path:
    episodes = []
    for index, (arm_id, camera_id, source_executor) in enumerate(
        (
            ("Arm_A", "CAM_A_TOP", "pi05"),
            ("Arm_B", "CAM_B_TOP", "openvla_oft"),
        )
    ):
        episodes.append(
            {
                "lerobot_episode_index": index,
                "canonical_episode_id": "episode-001",
                "canonical_split": "train",
                "active_arm": arm_id,
                "camera_id": camera_id,
                "source_executor": source_executor,
                "target_executor": "pi05",
                "source_hdf5_sha256": "sha256:" + "1" * 64,
                "source_structure_sha256": "sha256:" + "2" * 64,
                "canonical_action_count": 20,
                "source_action_start_index": index * 10,
                "source_action_stop_index": (index + 1) * 10,
                "source_action_count": 10,
                "window_count": 1,
                "window_start_action_indices": [index * 10],
            }
        )
    manifest = {
        "manifest_version": "1.1",
        "source_format": "canonical_hdf5_v2",
        "repo_id": "industrial/pi05-test-segments",
        "action_horizon": 10,
        "action_shape": [10, 7],
        "padding": "forbidden",
        "window_rule": "per-arm-segment-N-9",
        "included_splits": ["train"],
        "source_split_registry_sha256": registry.registry_sha256,
        "counts": {"episodes": 2, "canonical_episodes": 1, "windows": 2},
        "roundtrip": {"episodes": 2, "windows": 2, "max_action_error": 0.0},
        "episodes": episodes,
    }
    path = root / "pi05_v2_conversion.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / MANIFEST_SHA256_FILENAME).write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    return path


def test_v2_source_loads_verified_train_windows(tmp_path: Path) -> None:
    registry = _registry()
    manifest_path = _write_manifest(tmp_path, registry)
    frames = _frames()

    loaded = load_lerobot_v2_norm_source(
        tmp_path,
        repo_id="industrial/pi05-test",
        split_registry=registry,
        dataset_opener=lambda _root, _repo_id: FakeDataset(frames),
        manifest_path=manifest_path,
    )

    assert loaded.state.shape == (2, 7)
    assert loaded.actions.shape == (2, 10, 7)
    assert loaded.state.dtype == np.float32
    assert loaded.actions.dtype == np.float32
    assert loaded.source_manifest["input_format"] == "lerobot_v2"
    assert loaded.source_manifest["counts"] == {
        "windows": 2,
        "action_vectors": 20,
    }


def test_v2_source_accepts_two_arm_segments_from_one_canonical_episode(
    tmp_path: Path,
) -> None:
    registry = _registry()
    manifest_path = _write_segment_manifest(tmp_path, registry)
    frames = [
        {
            "state": np.full(7, index, dtype=np.float32),
            "actions": np.full((10, 7), index, dtype=np.float32),
            "task": "把Bin_01搬到FINISHED_01",
            "episode_index": np.asarray(index, dtype=np.int64),
            "frame_index": np.asarray(0, dtype=np.int64),
        }
        for index in range(2)
    ]

    loaded = load_lerobot_v2_norm_source(
        tmp_path,
        repo_id="industrial/pi05-test-segments",
        split_registry=registry,
        dataset_opener=lambda _root, _repo_id: FakeDataset(frames),
        manifest_path=manifest_path,
    )

    assert loaded.state.shape == (2, 7)
    assert loaded.actions.shape == (2, 10, 7)
    assert len(loaded.source_manifest["sources"]) == 2
    assert [source["active_arm"] for source in loaded.source_manifest["sources"]] == [
        "Arm_A",
        "Arm_B",
    ]


@pytest.mark.parametrize(
    ("ranges", "canonical_count", "message"),
    [
        ([(0, 10), (20, 30)], 30, "not contiguous"),
        ([(0, 10), (9, 19)], 19, "not contiguous"),
    ],
)
def test_v2_manifest_rejects_non_contiguous_arm_segments(
    tmp_path: Path,
    ranges: list[tuple[int, int]],
    canonical_count: int,
    message: str,
) -> None:
    registry = _registry()
    manifest_path = _write_segment_manifest(tmp_path, registry)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item, (start, stop) in zip(manifest["episodes"], ranges, strict=True):
        item["canonical_action_count"] = canonical_count
        item["source_action_start_index"] = start
        item["source_action_stop_index"] = stop
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_path.with_name(MANIFEST_SHA256_FILENAME).write_text(
        f"{digest}  {manifest_path.name}\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match=message):
        verify_conversion_manifest(manifest_path)


def test_v2_source_rejects_manifest_checksum_tampering(tmp_path: Path) -> None:
    registry = _registry()
    manifest_path = _write_manifest(tmp_path, registry)
    manifest_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(V2NormSourceError, match="SHA-256 mismatch"):
        load_lerobot_v2_norm_source(
            tmp_path,
            repo_id="industrial/pi05-test",
            split_registry=registry,
            dataset_opener=lambda _root, _repo_id: FakeDataset(_frames()),
        )


def test_v2_source_rejects_split_mismatch(tmp_path: Path) -> None:
    registry = _registry()
    _write_manifest(tmp_path, registry, canonical_split="val")

    with pytest.raises(V2NormSourceError, match="split does not match"):
        load_lerobot_v2_norm_source(
            tmp_path,
            repo_id="industrial/pi05-test",
            split_registry=registry,
            dataset_opener=lambda _root, _repo_id: FakeDataset(_frames()),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("state", np.zeros(6, dtype=np.float32), "state must be"),
        ("actions", np.zeros((10, 6), dtype=np.float32), "actions must be"),
        (
            "actions",
            np.full((10, 7), np.nan, dtype=np.float32),
            "NaN or Infinity",
        ),
    ],
)
def test_v2_source_rejects_invalid_frame_arrays(
    tmp_path: Path,
    field: str,
    bad_value: np.ndarray,
    message: str,
) -> None:
    registry = _registry()
    _write_manifest(tmp_path, registry)
    frames = _frames()
    frames[0][field] = bad_value

    with pytest.raises(V2NormSourceError, match=message):
        load_lerobot_v2_norm_source(
            tmp_path,
            repo_id="industrial/pi05-test",
            split_registry=registry,
            dataset_opener=lambda _root, _repo_id: FakeDataset(frames),
        )


def test_v2_source_enforces_io_deadline(tmp_path: Path, monkeypatch: Any) -> None:
    registry = _registry()
    _write_manifest(tmp_path, registry)
    clock = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(
        "scripts.pi05.lerobot_v2_norm_source.time.monotonic",
        lambda: next(clock),
    )

    with pytest.raises(TimeoutError, match="dataset open"):
        load_lerobot_v2_norm_source(
            tmp_path,
            repo_id="industrial/pi05-test",
            split_registry=registry,
            dataset_opener=lambda _root, _repo_id: FakeDataset(_frames()),
            io_timeout_s=1.0,
        )


def test_v2_actions_keep_seven_dimensional_norm_stats() -> None:
    actions = np.arange(140, dtype=np.float32).reshape(2, 10, 7)
    stats = compute_stats(actions, key="actions")

    assert stats["mean"].shape == (7,)
    np.testing.assert_allclose(stats["mean"], actions.reshape(-1, 7).mean(axis=0))
    np.testing.assert_allclose(
        stats["q01"], np.quantile(actions.reshape(-1, 7), 0.01, axis=0)
    )
    np.testing.assert_allclose(
        stats["q99"], np.quantile(actions.reshape(-1, 7), 0.99, axis=0)
    )
    validate_dimensions(
        {
            "state": np.zeros((2, 7), dtype=np.float32),
            "actions": actions,
        },
        expected_state_dim=7,
    )


def test_sparse_action_quantiles_fall_back_to_observed_range(
    caplog: pytest.LogCaptureFixture,
) -> None:
    actions = np.zeros((1000, 1, 7), dtype=np.float32)
    actions[-2, 0, 3] = np.deg2rad(2.0)
    actions[-1, 0, 3] = np.deg2rad(5.0)

    stats = compute_stats(actions, key="actions")

    assert stats["q01"][3] == pytest.approx(0.0)
    assert stats["q99"][3] == pytest.approx(np.deg2rad(5.0))
    assert "sparse quantile fallback at dim=3" in caplog.text
    normalized = (actions[:, 0, 3] - stats["q01"][3]) / (
        stats["q99"][3] - stats["q01"][3] + 1e-6
    ) * 2.0 - 1.0
    assert normalized[-2] == pytest.approx(-0.2, abs=3e-5)
    assert normalized[-1] == pytest.approx(1.0, abs=3e-5)


def test_constant_action_dimension_keeps_constant_quantiles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    actions = np.zeros((100, 10, 7), dtype=np.float32)

    stats = compute_stats(actions, key="actions")

    np.testing.assert_array_equal(stats["q01"], np.zeros(7))
    np.testing.assert_array_equal(stats["q99"], np.zeros(7))
    assert "sparse quantile fallback" not in caplog.text


def test_dimension_validation_rejects_double_windowing() -> None:
    with pytest.raises(ValueError, match="actions must be"):
        validate_dimensions(
            {
                "state": np.zeros((2, 7), dtype=np.float32),
                "actions": np.zeros((2, 10, 10, 7), dtype=np.float32),
            },
            expected_state_dim=7,
        )


def test_training_config_selects_action_windowing_by_input_format() -> None:
    assert action_sequence_keys_for_input("lerobot") is None
    assert action_sequence_keys_for_input("lerobot-v2") == ()


def test_training_config_rejects_unknown_input_format() -> None:
    with pytest.raises(ValueError, match="PI05_INPUT_FORMAT"):
        action_sequence_keys_for_input("unknown")
