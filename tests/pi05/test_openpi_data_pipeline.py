from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.pi05.compute_norm_stats as norm_stats_module
from configs.pi05.train_config import OPENPI_COMMIT, require_frozen_action_horizon
from industrial_agent.data import SplitRegistry
from scripts.pi05.canonical_v1 import (
    CanonicalEpisode,
    CanonicalPi05StateMapper,
    CanonicalStep,
    StateMapper,
    find_episode_dirs,
    require_state_mapper,
)
from scripts.pi05.compute_norm_stats import (
    LoadedDataset,
    calculate_norm_stats,
    load_dataset,
    write_norm_stats_bundle,
)
from scripts.pi05.convert_openpi import convert_canonical_to_lerobot
from scripts.pi05.provenance_context import (
    ProvenanceContext,
    resolve_provenance_context,
)
from scripts.pi05.smoke_lerobot_loader import (
    load_provenance,
    validate_provenance_manifest,
    verify_provenance_checksum,
)
from tests.pi05.test_canonical_v1 import build_episode, build_registry


TEST_PROVENANCE_CONTEXT = ProvenanceContext(
    project_git_sha="1" * 40,
    project_worktree_dirty=True,
    project_worktree_diff_sha256="2" * 64,
    openpi_commit="15a9616a00943ada6c20a0f158e3adb39df2ccac",
)


class TestOnlyStateMapper:
    __test__ = False
    name = "test-only-state-7d-v1"
    version = "1.0-test"
    state_dim = 7
    approved_for_production = False

    def map_state(self, episode: CanonicalEpisode, step: CanonicalStep) -> np.ndarray:
        del episode
        return step.state_7d.copy()


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
        if not self._episode_buffer:
            raise ValueError("cannot save an empty Episode")
        tasks = {frame.get("task") for frame in self._episode_buffer}
        if len(tasks) != 1:
            raise ValueError("Episode task must be stable")
        episode_index = len(self.episode_tasks)
        for frame_index, frame in enumerate(self._episode_buffer):
            frame["episode_index"] = np.int64(episode_index)
            frame["frame_index"] = np.int64(frame_index)
        self.frames.extend(self._episode_buffer)
        self._episode_buffer.clear()
        self.episode_tasks.append(str(next(iter(tasks))))

    def clear_episode_buffer(self) -> None:
        self._episode_buffer.clear()

    def stop_image_writer(self) -> None:
        self.writer_closed = True

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.frames[index]


class FailingAddFrameDataset(FakeLeRobotDataset):
    def add_frame(self, frame: dict[str, Any]) -> None:
        del frame
        raise RuntimeError("injected add_frame failure")


class FailingSaveEpisodeDataset(FakeLeRobotDataset):
    def save_episode(self) -> None:
        raise RuntimeError("injected save_episode failure")


class FailingWriterCloseDataset(FakeLeRobotDataset):
    def stop_image_writer(self) -> None:
        raise RuntimeError("injected writer close failure")


class FailingTraversalDataset(FakeLeRobotDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        raise RuntimeError(f"injected traversal failure at {index}")


_LAST_DATASET: FakeLeRobotDataset | None = None


def fake_dataset_factory(**_: Any) -> FakeLeRobotDataset:
    global _LAST_DATASET
    _LAST_DATASET = FakeLeRobotDataset()
    return _LAST_DATASET


def fake_dataset_opener(_: Path, __: str) -> FakeLeRobotDataset:
    if _LAST_DATASET is None:
        raise RuntimeError("fake dataset has not been created")
    return _LAST_DATASET


def _convert_one(
    tmp_path: Path,
    *,
    split: str = "train",
    dataset_factory: Any = fake_dataset_factory,
    dataset_opener: Any = fake_dataset_opener,
) -> tuple[Any, SplitRegistry]:
    canonical_root = tmp_path / "canonical"
    episode = build_episode(canonical_root)
    registry = build_registry([(episode, split)])
    result = convert_canonical_to_lerobot(
        data_dir=canonical_root,
        output_dir=tmp_path / "lerobot",
        output_repo_id="test/pi05",
        fps=10,
        timestamp_tolerance_ns=0,
        state_mapper=TestOnlyStateMapper(),
        split_registry=registry,
        provenance_context=TEST_PROVENANCE_CONTEXT,
        production=False,
        dataset_factory=dataset_factory,
        dataset_opener=dataset_opener,
    )
    return result, registry


def test_state_mapper_blocks_unapproved_production_semantics() -> None:
    mapper: StateMapper = TestOnlyStateMapper()
    with pytest.raises(RuntimeError, match="not approved for production"):
        require_state_mapper(mapper, production=True)
    assert require_state_mapper(mapper, production=False) is mapper


def test_unfrozen_action_horizon_remains_a_production_sentinel() -> None:
    with pytest.raises(RuntimeError, match="ARCH-2026-001 item 4"):
        require_frozen_action_horizon(10, production=True)
    assert require_frozen_action_horizon(10, production=False) == 10


def test_conversion_uses_hdf5_lineage_and_atomic_offline_reopen(
    tmp_path: Path,
) -> None:
    result, registry = _convert_one(tmp_path)
    assert _LAST_DATASET is not None
    assert _LAST_DATASET.writer_closed is True
    assert result.manifest["source_format"] == "canonical_hdf5_v1"
    assert (
        result.manifest["source_split_registry_sha256"]
        == (registry.registry_sha256.split(":", 1)[-1])
    )
    assert result.manifest["producer"] == TEST_PROVENANCE_CONTEXT.as_manifest()
    assert result.manifest["robot_type"] == "franka"
    assert result.manifest["image"]["wrist_image"] is None
    assert result.manifest["counts"] == {
        "episodes": 1,
        "steps": 1,
        "images": 1,
        "instructions": 1,
        "language_frames": 1,
        "states": 1,
        "actions": 1,
    }
    source = result.manifest["episodes"][0]
    assert source["robot_role"] == "arm_a_pi05"
    assert source["source_action_sequence_ids"] == [0]
    assert source["source_camera_sequence_ids"] == [0]
    assert source["source_state_sequence_ids"] == [0]
    assert source["source_physics_ticks"] == [0]
    assert source["source_image_datasets"] == ["/cameras/CAM_A_TOP/rgb[0]"]
    assert source["source_recorder_git_sha"] == "a" * 40
    assert result.manifest["roundtrip"]["roundtrip_samples"] == 1
    assert result.manifest_path.is_file()
    assert result.manifest_checksum_path.is_file()
    assert verify_provenance_checksum(result.manifest_path) == result.manifest_sha256
    assert validate_provenance_manifest(
        load_provenance(result.manifest_path),
        expected_repo_id="test/pi05",
        expected_provenance_context=TEST_PROVENANCE_CONTEXT,
    )


def test_converter_requires_verified_split_registry(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    build_episode(canonical_root)
    with pytest.raises(TypeError, match="split_registry"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=tmp_path / "lerobot",
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyStateMapper(),
            split_registry=None,  # type: ignore[arg-type]
            provenance_context=TEST_PROVENANCE_CONTEXT,
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )


def test_converter_requires_verified_provenance_context(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    episode = build_episode(canonical_root)
    registry = build_registry([(episode, "train")])
    with pytest.raises(TypeError, match="provenance_context"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=tmp_path / "lerobot",
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyStateMapper(),
            split_registry=registry,
            provenance_context=None,  # type: ignore[arg-type]
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )


@pytest.mark.parametrize(
    ("dataset_type", "opener"),
    [
        (FailingAddFrameDataset, None),
        (FailingSaveEpisodeDataset, None),
        (FailingWriterCloseDataset, None),
        (FailingTraversalDataset, "self"),
    ],
)
def test_conversion_failure_removes_staging(
    tmp_path: Path,
    dataset_type: type[FakeLeRobotDataset],
    opener: str | None,
) -> None:
    canonical_root = tmp_path / "canonical"
    episode = build_episode(canonical_root)
    registry = build_registry([(episode, "train")])
    dataset = dataset_type()

    def factory(**_: Any) -> FakeLeRobotDataset:
        return dataset

    def open_dataset(_: Path, __: str) -> FakeLeRobotDataset:
        return dataset

    with pytest.raises(Exception, match="injected"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=tmp_path / "lerobot",
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyStateMapper(),
            split_registry=registry,
            provenance_context=TEST_PROVENANCE_CONTEXT,
            production=False,
            dataset_factory=factory,
            dataset_opener=open_dataset if opener else fake_dataset_opener,
        )
    assert not (tmp_path / "lerobot").exists()
    assert list(tmp_path.glob(".lerobot.staging-*")) == []


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    episode = build_episode(canonical_root)
    registry = build_registry([(episode, "train")])
    output = tmp_path / "lerobot"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert_canonical_to_lerobot(
            data_dir=canonical_root,
            output_dir=output,
            output_repo_id="test/pi05",
            fps=10,
            timestamp_tolerance_ns=0,
            state_mapper=TestOnlyStateMapper(),
            split_registry=registry,
            provenance_context=TEST_PROVENANCE_CONTEXT,
            production=False,
            dataset_factory=fake_dataset_factory,
            dataset_opener=fake_dataset_opener,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_provenance_tampering_is_detected(tmp_path: Path) -> None:
    result, _ = _convert_one(tmp_path)
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["episodes"][0]["robot_role"] = "arm_b_openvla"
    result.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_provenance(result.manifest_path)


def test_provenance_rejects_forged_producer_and_recorder_git_sha(
    tmp_path: Path,
) -> None:
    result, _ = _convert_one(tmp_path)
    forged_producer = json.loads(json.dumps(result.manifest))
    forged_producer["producer"]["project_git_sha"] = "3" * 40
    with pytest.raises(ValueError, match="does not match"):
        validate_provenance_manifest(
            forged_producer,
            expected_repo_id="test/pi05",
            expected_provenance_context=TEST_PROVENANCE_CONTEXT,
        )

    missing_recorder_sha = json.loads(json.dumps(result.manifest))
    missing_recorder_sha["episodes"][0].pop("source_recorder_git_sha")
    with pytest.raises(ValueError, match="source_recorder_git_sha"):
        validate_provenance_manifest(
            missing_recorder_sha,
            expected_repo_id="test/pi05",
            expected_provenance_context=TEST_PROVENANCE_CONTEXT,
        )


def test_canonical_norm_stats_selects_only_registry_train_split(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    train = build_episode(
        canonical_root,
        "train-a-000001",
        scene_seed=101,
        sentinel=1.0,
    )
    val = build_episode(
        canonical_root,
        "val-a-000001",
        scene_seed=202,
        sentinel=100.0,
    )
    test = build_episode(
        canonical_root,
        "test-a-000001",
        scene_seed=303,
        sentinel=1000.0,
    )
    registry = build_registry([(train, "train"), (val, "val"), (test, "test")])
    mapper = TestOnlyStateMapper()
    loaded = load_dataset(
        canonical_root,
        input_format="canonical-v1",
        state_mapper=mapper,
        split_registry=registry,
        provenance_context=TEST_PROVENANCE_CONTEXT,
        production=False,
    )
    assert loaded.state.shape == (1, 7)
    assert loaded.state[0, 0] == pytest.approx(1.0)
    assert loaded.source_manifest["split_registry_sha256"] == registry.registry_sha256


def test_lerobot_norm_stats_revalidates_same_split_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, registry = _convert_one(tmp_path)
    assert _LAST_DATASET is not None
    monkeypatch.setattr(norm_stats_module, "open_offline_dataset", fake_dataset_opener)
    loaded = load_dataset(
        tmp_path / "lerobot",
        input_format="lerobot",
        state_mapper=TestOnlyStateMapper(),
        split_registry=registry,
        provenance_context=TEST_PROVENANCE_CONTEXT,
        production=False,
        repo_id="test/pi05",
        manifest_path=result.manifest_path,
    )
    assert loaded.state.shape == (1, 7)
    assert loaded.actions.shape == (1, 7)

    other_registry = SplitRegistry()
    other_registry.assign_episode(
        "train-a-000001",
        "train",
        scenario_group_id="different",
        scene_seed=999,
        asset_variant="different",
        camera_seed=999,
        lighting_seed=999,
    )
    with pytest.raises(ValueError, match="Registry SHA"):
        load_dataset(
            tmp_path / "lerobot",
            input_format="lerobot",
            state_mapper=TestOnlyStateMapper(),
            split_registry=other_registry,
            provenance_context=TEST_PROVENANCE_CONTEXT,
            production=False,
            repo_id="test/pi05",
            manifest_path=result.manifest_path,
        )

    forged_context = ProvenanceContext(
        project_git_sha="4" * 40,
        project_worktree_dirty=True,
        project_worktree_diff_sha256="2" * 64,
        openpi_commit=TEST_PROVENANCE_CONTEXT.openpi_commit,
    )
    with pytest.raises(ValueError, match="producer does not match"):
        load_dataset(
            tmp_path / "lerobot",
            input_format="lerobot",
            state_mapper=TestOnlyStateMapper(),
            split_registry=registry,
            provenance_context=forged_context,
            production=False,
            repo_id="test/pi05",
            manifest_path=result.manifest_path,
        )


def test_norm_stats_bundle_is_atomic_and_records_sources(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    episode = build_episode(canonical_root)
    registry = build_registry([(episode, "train")])
    mapper = TestOnlyStateMapper()
    loaded = load_dataset(
        canonical_root,
        input_format="canonical-v1",
        state_mapper=mapper,
        split_registry=registry,
        provenance_context=TEST_PROVENANCE_CONTEXT,
        production=False,
    )
    loaded_for_stats = LoadedDataset(
        state=np.concatenate((loaded.state, loaded.state + 0.01), axis=0),
        actions=np.concatenate((loaded.actions, loaded.actions + 0.01), axis=0),
        mask=None,
        source_manifest=loaded.source_manifest,
    )
    norm_stats, _ = calculate_norm_stats(
        loaded_for_stats,
        state_dim=mapper.state_dim,
    )
    output = tmp_path / "norm_stats.json"
    stats_sha, manifest_path, manifest_sha = write_norm_stats_bundle(
        output_path=output,
        norm_stats=norm_stats,
        loaded=loaded_for_stats,
        mapper=mapper,
        provenance_context=TEST_PROVENANCE_CONTEXT,
    )
    assert output.is_file()
    assert manifest_path.is_file()
    assert len(stats_sha) == 64
    assert len(manifest_sha) == 64
    source = json.loads(manifest_path.read_text(encoding="utf-8"))["source"]
    assert source["split"] == "train"
    assert source["split_registry_sha256"] == registry.registry_sha256
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["producer"] == TEST_PROVENANCE_CONTEXT.as_manifest()


REAL_GATE_VARS = (
    "PI05_REAL_CANONICAL_ROOT",
    "PI05_REAL_SPLIT_REGISTRY",
    "PI05_PINNED_OPENPI_COMMIT",
    "PI05_PROJECT_ROOT",
)


@pytest.mark.skipif(
    not all(os.environ.get(name) for name in REAL_GATE_VARS),
    reason=(
        "real 5-Episode LeRobot/OpenPI Gate requires the pinned Ubuntu/Docker "
        "environment and external Isaac artifacts"
    ),
)
def test_real_five_episode_lerobot_openpi_release_gate(tmp_path: Path) -> None:
    """Mandatory release Gate; ordinary local CI is not release evidence."""

    expected_commit = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    assert os.environ["PI05_PINNED_OPENPI_COMMIT"] == expected_commit
    provenance_context = resolve_provenance_context(
        repo_root=os.environ["PI05_PROJECT_ROOT"],
        openpi_commit=os.environ["PI05_PINNED_OPENPI_COMMIT"],
        expected_openpi_commit=OPENPI_COMMIT,
    )
    canonical_root = Path(os.environ["PI05_REAL_CANONICAL_ROOT"])
    registry = SplitRegistry.load(os.environ["PI05_REAL_SPLIT_REGISTRY"])
    assert len(find_episode_dirs(canonical_root)) == 5
    output = tmp_path / "real-lerobot"
    result = convert_canonical_to_lerobot(
        data_dir=canonical_root,
        output_dir=output,
        output_repo_id="release/pi05",
        fps=10,
        timestamp_tolerance_ns=0,
        state_mapper=CanonicalPi05StateMapper(),
        split_registry=registry,
        provenance_context=provenance_context,
        production=True,
    )
    assert result.manifest["counts"]["episodes"] == 5
    assert result.manifest["roundtrip"]["roundtrip_samples"] == 10
    loaded = load_dataset(
        output,
        input_format="lerobot",
        state_mapper=CanonicalPi05StateMapper(),
        split_registry=registry,
        provenance_context=provenance_context,
        production=True,
        repo_id="release/pi05",
        manifest_path=result.manifest_path,
    )
    norm_stats, _ = calculate_norm_stats(loaded, state_dim=7)
    assert set(norm_stats) == {"state", "actions"}
