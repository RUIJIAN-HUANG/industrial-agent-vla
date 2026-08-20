from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from industrial_agent.data import SplitRegistry
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from scripts.pi05.canonical_v2 import CanonicalV2Error
from scripts.pi05.convert_openpi import main as convert_openpi_main
from scripts.pi05.convert_openpi_v2 import (
    ACTION_HORIZON,
    build_complete_action_windows,
    convert_canonical_v2_to_lerobot,
    main,
    preflight_canonical_v2_windows,
    verify_conversion_manifest,
)
from simulation.v2_collection_recorder import (
    ARM_IDS,
    CAMERA_IDS,
    V2CollectionIdentity,
    V2CollectionRecorder,
)


class FakeLeRobotDataset:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True)
        self._buffer: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.num_episodes = 0
        self.writer_closed = False

    def add_frame(self, frame: dict[str, Any]) -> None:
        self._buffer.append(frame)

    def save_episode(self) -> None:
        if not self._buffer:
            raise ValueError("cannot save an empty Episode")
        self.frames.extend(self._buffer)
        self._buffer.clear()
        self.num_episodes += 1

    def stop_image_writer(self) -> None:
        self.writer_closed = True

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.frames[index]


def _record_episode(
    tmp_path: Path,
    *,
    action_count: int = 10,
    outcome: str = "SUCCEEDED",
) -> Path:
    image_cas = ImageCas(ImageCasConfig(root=tmp_path / "cas"))
    identity = V2CollectionIdentity(
        episode_id="v2-convert-000001",
        scene_seed=31,
        git_sha="a" * 40,
        scene_config_sha256=f"sha256:{'b' * 64}",
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[0, 0] = (4, 5, 6)
    images = {
        camera_id: image_cas.write_rgb(frame, camera_id=camera_id)
        for camera_id in CAMERA_IDS
    }
    states = {arm_id: [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.375] for arm_id in ARM_IDS}
    writer = V2CollectionRecorder(
        tmp_path / "canonical",
        identity,
        image_cas=image_cas,
    )
    with writer:
        for index in range(action_count):
            timestamp_ns = 1_000_000_000 + index * 100_000_000
            physics_tick = index * 12
            writer.record_camera_bundle(
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
                sequence_id=index,
                images=images,
            )
            writer.record_state_bundle(
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
                sequence_id=index,
                states=states,
            )
            writer.record_action(
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
                sequence_id=index,
                chunk_id=f"manual-p01-{index}",
                action_7d=[index / 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        failure_code = None if outcome == "SUCCEEDED" else "TEST_FAILURE"
        return writer.finalize(outcome=outcome, failure_code=failure_code)


def _registry(episode_id: str) -> SplitRegistry:
    registry = SplitRegistry()
    registry.assign_episode(
        episode_id,
        "train",
        scenario_group_id=f"group-{episode_id}",
        scene_seed=31,
        asset_variant="v2-fixed",
        camera_seed=41,
        lighting_seed=51,
    )
    return registry


def test_complete_windows_are_exactly_n_minus_nine_and_lossless() -> None:
    actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)

    windows = build_complete_action_windows(actions)

    assert len(windows) == 3
    assert all(window.shape == (ACTION_HORIZON, 7) for window in windows)
    np.testing.assert_array_equal(windows[0], actions[0:10])
    np.testing.assert_array_equal(windows[1], actions[1:11])
    np.testing.assert_array_equal(windows[2], actions[2:12])


def test_complete_windows_reject_short_episode_without_padding() -> None:
    with pytest.raises(ValueError, match="padding is forbidden"):
        build_complete_action_windows(np.zeros((9, 7), dtype=np.float32))


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((10, 6), dtype=np.float32),
        np.zeros((10, 7), dtype=np.float64),
        np.full((10, 7), np.nan, dtype=np.float32),
    ],
)
def test_complete_windows_reject_invalid_action_contract(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="float32|NaN"):
        build_complete_action_windows(actions)


@pytest.mark.parametrize("outcome", ["FAILED", "SAFE_STOPPED", "SAFE_STOP_FAILED"])
def test_v2_preflight_rejects_non_succeeded_episode(
    tmp_path: Path,
    outcome: str,
) -> None:
    episode_path = _record_episode(tmp_path, outcome=outcome)

    with pytest.raises(CanonicalV2Error, match="metadata.outcome.*SUCCEEDED"):
        preflight_canonical_v2_windows(
            data_dir=episode_path.parent,
            split_registry=_registry("v2-convert-000001"),
        )


def test_v2_conversion_rejects_failed_episode_before_dataset_creation(
    tmp_path: Path,
) -> None:
    episode_path = _record_episode(tmp_path, outcome="FAILED")
    factory_called = False

    def factory(**_: Any) -> FakeLeRobotDataset:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("dataset must not be created for a failed Episode")

    with pytest.raises(CanonicalV2Error, match="metadata.outcome.*SUCCEEDED"):
        convert_canonical_v2_to_lerobot(
            data_dir=episode_path.parent,
            output_dir=tmp_path / "must-not-exist",
            repo_id="test/pi05-v2-reject-failed",
            split_registry=_registry("v2-convert-000001"),
            dataset_factory=factory,
        )

    assert factory_called is False
    assert not (tmp_path / "must-not-exist").exists()


def test_canonical_v2_to_lerobot_smoke_is_lossless(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    episode_path = _record_episode(tmp_path)
    registry = _registry("v2-convert-000001")
    report = preflight_canonical_v2_windows(
        data_dir=episode_path.parent,
        split_registry=registry,
    )
    assert report["counts"]["actions"] == 10
    assert report["counts"]["windows"] == 1
    assert report["counts"]["splits"] == {"train": 1, "val": 0, "test": 0}

    registry_path = registry.save(tmp_path / "split-registry.json")
    assert (
        main(
            [
                "--data-dir",
                str(episode_path.parent),
                "--split-registry",
                str(registry_path),
                "--preflight-only",
            ]
        )
        == 0
    )
    cli_report = capsys.readouterr().out
    assert '"window_rule": "N-9"' in cli_report

    holder: dict[str, FakeLeRobotDataset] = {}

    def factory(**kwargs: Any) -> FakeLeRobotDataset:
        dataset = FakeLeRobotDataset(kwargs["output_dir"])
        holder["dataset"] = dataset
        return dataset

    def opener(_: Path, __: str) -> FakeLeRobotDataset:
        return holder["dataset"]

    result = convert_canonical_v2_to_lerobot(
        data_dir=episode_path.parent,
        output_dir=tmp_path / "lerobot",
        repo_id="test/pi05-v2",
        split_registry=registry,
        dataset_factory=factory,
        dataset_opener=opener,
    )

    dataset = holder["dataset"]
    assert dataset.writer_closed is True
    assert dataset.num_episodes == 1
    assert len(dataset) == 1
    assert dataset[0]["actions"].shape == (10, 7)
    assert dataset[0]["actions"].dtype == np.float32
    assert dataset[0]["state"][6] == pytest.approx(0.375)
    assert result.manifest["action_horizon"] == 10
    assert result.manifest["padding"] == "forbidden"
    assert result.manifest["window_rule"] == "N-9"
    assert result.manifest["counts"] == {"episodes": 1, "windows": 1}
    assert result.manifest["roundtrip"]["max_action_error"] == 0.0
    assert result.manifest["episodes"][0]["source_action_count"] == 10
    assert result.manifest["episodes"][0]["window_count"] == 1
    assert result.manifest_path.is_file()
    assert result.manifest_checksum_path.is_file()
    assert len(result.manifest_sha256) == 64
    assert verify_conversion_manifest(result.manifest_path) == result.manifest

    result.manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_conversion_manifest(result.manifest_path)


def test_formal_converter_entry_dispatches_v2_reader_and_preflight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    episode_path = _record_episode(tmp_path)
    registry_path = _registry("v2-convert-000001").save(
        tmp_path / "split-registry-dispatch.json"
    )

    assert (
        convert_openpi_main(
            [
                "v2",
                "--data-dir",
                str(episode_path.parent),
                "--split-registry",
                str(registry_path),
                "--preflight-only",
            ]
        )
        == 0
    )
    report = capsys.readouterr().out
    assert '"source_format": "canonical_hdf5_v2"' in report
    assert '"windows": 1' in report
