from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.pi05.compute_norm_stats as norm_stats_module
import scripts.pi05.convert_openpi as convert_module
from scripts.pi05.canonical_v1 import (
    CanonicalEpisode,
    CanonicalStep,
    CanonicalV1Error,
    StateMapper,
    require_state_mapper,
)
from scripts.pi05.compute_norm_stats import (
    calculate_norm_stats,
    load_dataset,
    validate_dimensions,
    write_norm_stats_bundle,
)
from scripts.pi05.convert_openpi import convert_canonical_to_lerobot
from scripts.pi05.smoke_lerobot_loader import (
    load_provenance,
    validate_provenance_manifest,
    verify_provenance_checksum,
)
from tests.pi05.test_canonical_v1 import build_episode, mutate_meta


class TestOnlyJointStateMapper:
    """Test-only mapping; deliberately not approved for production."""

    __test__ = False
    name = "test-only-joint-position-v1"
    version = "1.0-test"
    state_dim = 2
    approved_for_production = False

    def map_state(self, episode: CanonicalEpisode, step: CanonicalStep) -> np.ndarray:
        del episode
        return np.asarray(step.joint_position, dtype=np.float32)


class FakeLeRobotDataset:
    def __init__(self) -> None:
        self._episode_buffer: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.episode_tasks: list[str] = []
        self.writer_closed = False

    @property
    def num_episodes(self) -> int:
        return len(self.episode_tasks)

    def add_frame(self, frame: dict[str, Any]) -> None:
        self._episode_buffer.append(frame)

    def save_episode(self) -> None:
        tasks = {frame.get("task") for frame in self._episode_buffer}
        if len(tasks) != 1 or not all(isinstance(task, str) for task in tasks):
            raise ValueError("each frame must contain one consistent task")
        task = next(iter(tasks))
        episode_index = len(self.episode_tasks)
        for frame_index, frame in enumerate(self._episode_buffer):
            frame["episode_index"] = np.int64(episode_index)
            frame["frame_index"] = np.int64(frame_index)
        self.frames.extend(self._episode_buffer)
        self._episode_buffer.clear()
        self.episode_tasks.append(task)

    def clear_episode_buffer(self) -> None:
        self._episode_buffer.clear()

    def stop_image_writer(self) -> None:
        self.writer_closed = True

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.frames[index]


class FailingWriterCloseDataset(FakeLeRobotDataset):
    def stop_image_writer(self) -> None:
        raise RuntimeError("injected writer close failure")


class FailingAddFrameDataset(FakeLeRobotDataset):
    def add_frame(self, frame: dict[str, Any]) -> None:
        del frame
        raise RuntimeError("injected add_frame failure")


class FailingSaveEpisodeDataset(FakeLeRobotDataset):
    def save_episode(self) -> None:
        raise RuntimeError("injected save_episode failure")


class FailingTraversalDataset(FakeLeRobotDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        raise RuntimeError(f"injected loader failure at frame {index}")


def fake_dataset_factory(**_: Any) -> FakeLeRobotDataset:
    global _LAST_FAKE_DATASET
    _LAST_FAKE_DATASET = FakeLeRobotDataset()
    return _LAST_FAKE_DATASET


_LAST_FAKE_DATASET: FakeLeRobotDataset | None = None


def fake_dataset_opener(_: Path, __: str) -> FakeLeRobotDataset:
    if _LAST_FAKE_DATASET is None:
        raise RuntimeError("fake dataset has not been created")
    return _LAST_FAKE_DATASET


def test_state_mapper_blocks_unapproved_production_semantics() -> None:
    mapper: StateMapper = TestOnlyJointStateMapper()
    with pytest.raises(RuntimeError, match="not approved for production"):
        require_state_mapper(mapper, production=True)
    assert require_state_mapper(mapper, production=False) is mapper
    with pytest.raises(RuntimeError, match="StateMapper is required"):
        require_state_mapper(None, production=True)


def test_unfrozen_action_horizon_is_a_production_runtime_sentinel() -> None:
    from configs.pi05.train_config import require_frozen_action_horizon

    with pytest.raises(RuntimeError, match="ARCH-2026-001 item 4"):
        require_frozen_action_horizon(10, production=True)
    assert require_frozen_action_horizon(10, production=False) == 10


def test_converter_contains_no_legacy_consolidate_call() -> None:
    source = Path(convert_module.__file__).read_text(encoding="utf-8")
    assert ".consolidate(" not in source


def test_conversion_preserves_counts_traceability_and_actions(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    source_episode = build_episode(canonical_root, step_count=12)
    output_dir = tmp_path / "lerobot"
    result = convert_canonical_to_lerobot(
        data_dir=canonical_root,
        output_dir=output_dir,
        output_repo_id="test/pi05",
        fps=10,
        timestamp_tolerance_ns=0,
        state_mapper=TestOnlyJointStateMapper(),
        production=False,
        dataset_factory=fake_dataset_factory,
        dataset_opener=fake_dataset_opener,
    )
    assert _LAST_FAKE_DATASET is not None
    assert _LAST_FAKE_DATASET.writer_closed is True
    assert result.manifest["fps"] == 10
    assert result.manifest["timestamp_tolerance_ns"] == 0
    assert result.manifest["image"]["shape"] == [720, 1280, 3]
    assert result.manifest["image"]["preprocessed"] is False
    assert result.manifest["counts"] == {
        "episodes": 1,
        "steps": 12,
        "images": 12,
        "instructions": 1,
        "language_frames": 12,
        "states": 12,
        "actions": 12,
    }
    episode_manifest = result.manifest["episodes"][0]
    assert episode_manifest["canonical_episode_id"] == source_episode.name
    assert episode_manifest["canonical_split"] == "train"
    assert episode_manifest["source_step_indices"] == list(range(12))
    assert len(episode_manifest["source_image_sha256"]) == 12
    assert result.manifest["roundtrip"]["roundtrip_samples"] == 10
    assert result.manifest["roundtrip"]["max_action_error"] < 1e-6
    assert result.manifest_path.is_file()
    assert result.manifest_checksum_path.is_file()
    assert len(result.manifest_sha256) == 64
    assert _LAST_FAKE_DATASET.episode_tasks == [
        f"instruction for {source_episode.name}"
    ]


def test_invalid_episode_prevents_any_partial_dataset_creation(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root, "train-a-000001")
    bad = build_episode(canonical_root, "train-a-000002")
    (bad / "rgb/CAM_A_TOP/000005.png").unlink()
    calls = 0

    def tracking_factory(**_: Any) -> FakeLeRobotDataset:
        nonlocal calls
        calls += 1
        return FakeLeRobotDataset()

    with pytest.raises(Exception, match="does not exist"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=tmp_path / "lerobot",
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=tracking_factory,
        )
    assert calls == 0
    assert not (tmp_path / "lerobot").exists()


@pytest.mark.parametrize(
    "dataset_type",
    [
        FailingAddFrameDataset,
        FailingSaveEpisodeDataset,
        FailingWriterCloseDataset,
        FailingTraversalDataset,
    ],
)
def test_conversion_runtime_failure_removes_staging_and_public_output(
    tmp_path: Path, dataset_type: type[FakeLeRobotDataset]
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    output_dir = tmp_path / "lerobot"
    dataset = dataset_type()

    with pytest.raises(Exception, match="injected"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=output_dir,
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=lambda **_: dataset,
            dataset_opener=lambda _root, _repo_id: dataset,
        )
    assert not output_dir.exists()
    assert list(tmp_path.glob(".lerobot.staging-*")) == []


def test_offline_reopen_failure_removes_staging(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    output_dir = tmp_path / "lerobot"

    def fail_reopen(_: Path, __: str) -> FakeLeRobotDataset:
        raise OSError("injected offline reopen failure")

    with pytest.raises(RuntimeError, match="offline reopen"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=output_dir,
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fail_reopen,
        )
    assert not output_dir.exists()
    assert list(tmp_path.glob(".lerobot.staging-*")) == []


def test_provenance_write_failure_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    output_dir = tmp_path / "lerobot"

    def fail_write(_: Path, __: dict[str, Any]) -> str:
        raise OSError("injected provenance write failure")

    monkeypatch.setattr(convert_module, "_write_json_atomic", fail_write)
    with pytest.raises(OSError, match="provenance write"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=output_dir,
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )
    assert not output_dir.exists()
    assert list(tmp_path.glob(".lerobot.staging-*")) == []


def test_provenance_sidecar_publish_failure_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    output_dir = tmp_path / "lerobot"

    def fail_publish(_: Path) -> tuple[Path, str]:
        raise OSError("injected provenance sidecar publish failure")

    monkeypatch.setattr(convert_module, "write_provenance_checksum", fail_publish)
    with pytest.raises(OSError, match="sidecar publish"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=output_dir,
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )
    assert not output_dir.exists()
    assert list(tmp_path.glob(".lerobot.staging-*")) == []


def test_staging_cleanup_failure_reports_original_and_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    output_dir = tmp_path / "lerobot"
    real_rmtree = convert_module.shutil.rmtree

    def fail_cleanup(_: Path) -> None:
        raise OSError("injected staging cleanup failure")

    monkeypatch.setattr(convert_module.shutil, "rmtree", fail_cleanup)
    with pytest.raises(RuntimeError) as captured:
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=output_dir,
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=lambda **_: FailingAddFrameDataset(),
        )
    message = str(captured.value)
    assert "injected add_frame failure" in message
    assert "injected staging cleanup failure" in message
    assert "conversion failed and staging cleanup failed" in caplog.text
    assert not output_dir.exists()
    staging = list(tmp_path.glob(".lerobot.staging-*"))
    assert len(staging) == 1
    monkeypatch.setattr(convert_module.shutil, "rmtree", real_rmtree)
    real_rmtree(staging[0])


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    output_dir = tmp_path / "lerobot"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=output_dir,
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_provenance_tampering_is_rejected(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    result = convert_canonical_to_lerobot(
        data_dir=canonical_root,
        output_dir=tmp_path / "lerobot",
        output_repo_id="test/pi05",
        fps=10,
        timestamp_tolerance_ns=0,
        state_mapper=TestOnlyJointStateMapper(),
        production=False,
        dataset_factory=fake_dataset_factory,
        dataset_opener=fake_dataset_opener,
    )
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["episodes"][0]["canonical_split"] = "val"
    result.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        load_provenance(result.manifest_path)


def test_provenance_checksum_sidecar_missing_and_malformed_are_rejected(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    result = convert_canonical_to_lerobot(
        data_dir=canonical_root,
        output_dir=tmp_path / "lerobot",
        output_repo_id="test/pi05",
        fps=10,
        timestamp_tolerance_ns=0,
        state_mapper=TestOnlyJointStateMapper(),
        production=False,
        dataset_factory=fake_dataset_factory,
        dataset_opener=fake_dataset_opener,
    )
    sidecar = result.manifest_checksum_path
    original = sidecar.read_text(encoding="ascii")
    sidecar.unlink()
    with pytest.raises(ValueError, match="cannot read provenance checksum"):
        verify_provenance_checksum(result.manifest_path)
    sidecar.write_text("malformed\n", encoding="ascii")
    with pytest.raises(ValueError, match="invalid format"):
        verify_provenance_checksum(result.manifest_path)
    sidecar.write_text(f"{'0' * 64}  {result.manifest_path.name}\n", encoding="ascii")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_provenance_checksum(result.manifest_path)
    sidecar.write_text(original, encoding="ascii")


def test_provenance_critical_fields_are_fail_closed(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    result = convert_canonical_to_lerobot(
        data_dir=canonical_root,
        output_dir=tmp_path / "lerobot",
        output_repo_id="test/pi05",
        fps=10,
        timestamp_tolerance_ns=0,
        state_mapper=TestOnlyJointStateMapper(),
        production=False,
        dataset_factory=fake_dataset_factory,
        dataset_opener=fake_dataset_opener,
    )

    def cloned() -> dict[str, Any]:
        return json.loads(json.dumps(result.manifest))

    invalid_manifests: list[dict[str, Any]] = []
    candidate = cloned()
    candidate.pop("source_root")
    invalid_manifests.append(candidate)
    candidate = cloned()
    candidate["image"]["shape"] = [224, 224, 3]
    invalid_manifests.append(candidate)
    candidate = cloned()
    candidate["state_mapper"]["version"] = ""
    invalid_manifests.append(candidate)
    candidate = cloned()
    candidate["roundtrip"]["roundtrip_samples"] = 9
    invalid_manifests.append(candidate)
    candidate = cloned()
    candidate["counts"]["actions"] -= 1
    invalid_manifests.append(candidate)
    candidate = cloned()
    candidate["episodes"][0].pop("source_image_paths")
    invalid_manifests.append(candidate)
    candidate = cloned()
    candidate["episodes"][0]["source_timestamp_ns"] = "invalid"
    invalid_manifests.append(candidate)
    candidate = cloned()
    candidate["episodes"][0]["source_action_duration_s"][0] = 0.0
    invalid_manifests.append(candidate)

    for candidate in invalid_manifests:
        with pytest.raises(ValueError):
            validate_provenance_manifest(candidate, expected_repo_id="test/pi05")


def test_explicit_fps_must_match_timestamps_before_dataset_creation(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    calls = 0

    def tracking_factory(**_: Any) -> FakeLeRobotDataset:
        nonlocal calls
        calls += 1
        return FakeLeRobotDataset()

    with pytest.raises(CanonicalV1Error, match="explicitly supplied FPS"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=tmp_path / "lerobot",
            output_repo_id="test/pi05",
            fps=20,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=tracking_factory,
        )
    assert calls == 0


def test_eligible_false_episode_rejects_conversion(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root, "train-a-000001", eligible=True)
    build_episode(canonical_root, "train-a-000002", eligible=False)
    with pytest.raises(CanonicalV1Error, match="not eligible"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=tmp_path / "lerobot",
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )
    assert not (tmp_path / "lerobot").exists()


def test_valid_false_step_rejects_entire_conversion(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(
        canonical_root,
        "train-a-000001",
        step_count=12,
        valid=lambda index: index != 5,
    )
    with pytest.raises(CanonicalV1Error, match="rejects the entire Episode"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=tmp_path / "lerobot",
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyJointStateMapper(),
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )
    assert not (tmp_path / "lerobot").exists()


def test_norm_stats_use_only_train_eligible_valid_steps(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(
        canonical_root,
        "train-a-000001",
        split="train",
        sentinel=1.0,
        valid=lambda index: index != 11,
    )
    build_episode(
        canonical_root,
        "val-a-000001",
        split="val",
        sentinel=100.0,
    )
    build_episode(
        canonical_root,
        "test-a-000001",
        split="test",
        sentinel=1000.0,
    )
    build_episode(
        canonical_root,
        "train-a-000002",
        split="train",
        eligible=False,
        sentinel=10_000.0,
    )
    mapper = TestOnlyJointStateMapper()
    loaded = load_dataset(
        canonical_root,
        input_format="canonical-v1",
        state_mapper=mapper,
        production=False,
    )
    assert loaded.state.shape == (11, 2)
    assert loaded.actions.shape == (11, 7)
    assert np.max(loaded.state) < 100.0
    assert loaded.source_manifest["split"] == "train"
    assert loaded.source_manifest["excluded"] == {
        "non_train_episodes": 2,
        "ineligible_episodes": 1,
        "invalid_steps": 1,
    }

    norm_stats, stats_by_key = calculate_norm_stats(loaded, state_dim=mapper.state_dim)
    assert np.max(norm_stats["state"].mean) < 100.0
    output = tmp_path / "norm_stats.json"
    stats_sha, manifest_path, manifest_sha = write_norm_stats_bundle(
        output_path=output,
        norm_stats=norm_stats,
        loaded=loaded,
        mapper=mapper,
    )
    assert output.is_file()
    assert manifest_path.is_file()
    assert len(stats_sha) == len(manifest_sha) == 64
    assert stats_by_key["actions"]["mean"][0] < 100.0
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert source["source"]["split"] == "train"
    assert source["counts"] == {"state_rows": 11, "action_rows": 11}


def test_lerobot_norm_stats_use_only_manifest_train_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root, "train-a-000001", split="train", sentinel=1.0)
    build_episode(canonical_root, "val-a-000001", split="val", sentinel=100.0)
    build_episode(canonical_root, "test-a-000001", split="test", sentinel=1000.0)
    output_dir = tmp_path / "lerobot"
    mapper = TestOnlyJointStateMapper()
    converted = convert_canonical_to_lerobot(
        data_dir=canonical_root,
        output_dir=output_dir,
        output_repo_id="test/pi05",
        fps=10,
        timestamp_tolerance_ns=0,
        state_mapper=mapper,
        production=False,
        dataset_factory=fake_dataset_factory,
        dataset_opener=fake_dataset_opener,
    )
    assert _LAST_FAKE_DATASET is not None
    monkeypatch.setattr(
        norm_stats_module,
        "open_offline_dataset",
        lambda dataset_root, repo_id: _LAST_FAKE_DATASET,
    )
    loaded = load_dataset(
        output_dir,
        input_format="lerobot",
        state_mapper=mapper,
        production=False,
        repo_id="test/pi05",
    )
    assert loaded.state.shape == (12, 2)
    assert loaded.actions.shape == (12, 7)
    assert np.max(loaded.state) < 100.0
    assert loaded.source_manifest["excluded"] == {"non_train_episodes": 2}
    norm_stats, _ = calculate_norm_stats(loaded, state_dim=mapper.state_dim)
    assert np.max(norm_stats["state"].mean) < 100.0


def test_dimension_and_mask_mismatches_fail_closed() -> None:
    with pytest.raises(ValueError, match="row count mismatch"):
        validate_dimensions(
            {
                "state": np.zeros((3, 2), dtype=np.float32),
                "actions": np.zeros((2, 7), dtype=np.float32),
            },
            expected_state_dim=2,
        )
    with pytest.raises(ValueError, match="mask length"):
        validate_dimensions(
            {
                "state": np.zeros((3, 2), dtype=np.float32),
                "actions": np.zeros((3, 7), dtype=np.float32),
                "mask": np.ones(2, dtype=bool),
            },
            expected_state_dim=2,
        )
    with pytest.raises(ValueError, match="state dimension"):
        validate_dimensions(
            {
                "state": np.zeros((3, 3), dtype=np.float32),
                "actions": np.zeros((3, 7), dtype=np.float32),
            },
            expected_state_dim=2,
        )


def test_nonfinite_stats_fail_before_output_is_written(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    episode = build_episode(canonical_root)
    mutate_meta(episode, "eligible_for_imitation", True)
    mapper = TestOnlyJointStateMapper()
    loaded = load_dataset(
        canonical_root,
        input_format="canonical-v1",
        state_mapper=mapper,
        production=False,
    )
    loaded.state[0, 0] = np.nan
    output = tmp_path / "norm_stats.json"
    with pytest.raises(ValueError, match="NaN or Infinity"):
        calculate_norm_stats(loaded, state_dim=mapper.state_dim)
    assert not output.exists()


def test_norm_stats_publication_failure_rolls_back_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    mapper = TestOnlyJointStateMapper()
    loaded = load_dataset(
        canonical_root,
        input_format="canonical-v1",
        state_mapper=mapper,
        production=False,
    )
    norm_stats, _ = calculate_norm_stats(loaded, state_dim=mapper.state_dim)
    output = tmp_path / "norm_stats.json"
    path_type = type(output)
    original_replace = path_type.replace

    def injected_replace(path: Path, target: Path) -> Path:
        if path.name.startswith(".norm_stats_source_manifest.json."):
            raise PermissionError("injected unwritable output directory")
        return original_replace(path, target)

    monkeypatch.setattr(path_type, "replace", injected_replace)
    with pytest.raises(PermissionError, match="injected unwritable"):
        write_norm_stats_bundle(
            output_path=output,
            norm_stats=norm_stats,
            loaded=loaded,
            mapper=mapper,
        )
    assert not output.exists()
    assert not (tmp_path / "norm_stats_source_manifest.json").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_norm_stats_temporary_write_failure_leaves_no_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    mapper = TestOnlyJointStateMapper()
    loaded = load_dataset(
        canonical_root,
        input_format="canonical-v1",
        state_mapper=mapper,
        production=False,
    )
    norm_stats, _ = calculate_norm_stats(loaded, state_dim=mapper.state_dim)
    output = tmp_path / "norm_stats.json"
    path_type = type(output)
    original_write_text = path_type.write_text

    def injected_write_text(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name.startswith(".norm_stats.json."):
            raise PermissionError("injected norm-stats temporary write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "write_text", injected_write_text)
    with pytest.raises(PermissionError, match="temporary write"):
        write_norm_stats_bundle(
            output_path=output,
            norm_stats=norm_stats,
            loaded=loaded,
            mapper=mapper,
        )
    assert not output.exists()
    assert not (tmp_path / "norm_stats_source_manifest.json").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.skip(
    reason="real LeRobot/OpenPI Gate is executed by the external Ubuntu/Docker owner"
)
def test_real_lerobot_runtime_is_required_for_release() -> None:
    """Prevent local control-flow tests from being reported as the real Gate."""
