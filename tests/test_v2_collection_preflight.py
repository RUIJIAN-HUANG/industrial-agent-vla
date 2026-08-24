import json
from hashlib import sha256
from pathlib import Path

import pytest

from simulation.v2_collection_preflight import (
    CollectionPreflightError,
    CollectionSplit,
    build_collection_preflight,
)
from configs.pi05.constants import OPENPI_COMMIT


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "simulation/configs/single_bin_scene_v2.json"
GIT_SHA = "a" * 40
SCENE_SHA = f"sha256:{sha256(CONFIG.read_bytes()).hexdigest()}"


def _build(tmp_path: Path, **overrides):
    values = {
        "config_path": CONFIG,
        "episode_root": tmp_path / "episodes",
        "cas_root": tmp_path / "cas",
        "episode_id": "v2-manual-20260812-130000-seed007-run001",
        "task_id": "P01_TO_S11",
        "instruction": "把P01放到S11中",
        "scene_seed": 7,
        "split": CollectionSplit.PRACTICE,
        "headless": False,
        "git_sha": GIT_SHA,
        "worktree_clean": True,
        "frozen_collection_sha": GIT_SHA,
        "expected_scene_config_sha256": SCENE_SHA,
        "openpi_git_sha": OPENPI_COMMIT,
        "openpi_worktree_clean": True,
    }
    values.update(overrides)
    return build_collection_preflight(**values)


def test_valid_practice_preflight_uses_real_v2_identity(tmp_path: Path) -> None:
    result = _build(tmp_path)

    assert result.scene_id == "single_bin_manual_industrial_v2"
    assert result.canonical_schema_version == "2.0"
    assert result.task_id == "P01_TO_S11"
    assert result.instruction == "把P01放到S11中"


def test_w01_s14_practice_preflight_uses_second_frozen_identity(
    tmp_path: Path,
) -> None:
    result = _build(
        tmp_path,
        task_id="W01_TO_S14",
        instruction="把W01放到S14中",
    )
    assert result.task_id == "W01_TO_S14"
    assert result.instruction == "把W01放到S14中"
    assert result.split is CollectionSplit.PRACTICE
    assert result.training_allowed is False
    assert result.full_task_required is False
    assert result.git_sha == GIT_SHA
    assert result.scene_config_sha256.startswith("sha256:")
    assert len(result.scene_config_sha256) == 71
    assert (
        result.episode_dir
        == (
            tmp_path / "episodes" / "v2-manual-20260812-130000-seed007-run001"
        ).resolve()
    )


def test_bin01_finished01_practice_preflight_uses_arm_b_identity(
    tmp_path: Path,
) -> None:
    result = _build(
        tmp_path,
        task_id="BIN01_TO_FINISHED01",
        instruction="把Bin_01搬到FINISHED_01",
    )
    assert result.task_id == "BIN01_TO_FINISHED01"
    assert result.instruction == "把Bin_01搬到FINISHED_01"
    assert result.split is CollectionSplit.PRACTICE


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_id", "v2-practice-grasp-upright-shaft", "task_id must equal"),
        ("instruction", "使用 Arm_A 将 P01 放入 S11", "instruction must equal"),
    ],
)
def test_non_frozen_v2_identity_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(CollectionPreflightError, match=message):
        _build(tmp_path, **{field: value})


@pytest.mark.parametrize(
    ("split", "training_allowed", "full_task_required"),
    [
        (CollectionSplit.PRACTICE, False, False),
        (CollectionSplit.TEST, False, False),
        (CollectionSplit.TRAIN, True, True),
        (CollectionSplit.VALIDATION, False, True),
    ],
)
def test_split_semantics(
    tmp_path: Path,
    split: CollectionSplit,
    training_allowed: bool,
    full_task_required: bool,
) -> None:
    result = _build(tmp_path, split=split)

    assert result.training_allowed is training_allowed
    assert result.full_task_required is full_task_required


def test_headless_manual_collection_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CollectionPreflightError, match="visible Isaac GUI"):
        _build(tmp_path, headless=True)


def test_dirty_worktree_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CollectionPreflightError, match="worktree"):
        _build(tmp_path, worktree_clean=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "p01_to_s11"),
        ("instruction", "把 P01 放到 S11 中"),
        ("instruction", "把P01放到S11中。"),
    ],
)
def test_atomic_identity_must_match_exactly(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(CollectionPreflightError, match=field):
        _build(tmp_path, **{field: value})


def test_formal_collection_rejects_unfrozen_provenance(tmp_path: Path) -> None:
    with pytest.raises(CollectionPreflightError, match="frozen collection"):
        _build(
            tmp_path,
            split=CollectionSplit.TRAIN,
            frozen_collection_sha="b" * 40,
        )
    with pytest.raises(CollectionPreflightError, match="scene config SHA"):
        _build(
            tmp_path,
            split=CollectionSplit.TRAIN,
            expected_scene_config_sha256=f"sha256:{'0' * 64}",
        )
    with pytest.raises(CollectionPreflightError, match="OpenPI HEAD"):
        _build(
            tmp_path,
            split=CollectionSplit.TRAIN,
            openpi_git_sha="b" * 40,
        )


@pytest.mark.parametrize(
    "git_sha",
    ["", "abc", "g" * 40, "a" * 39, "a" * 41],
)
def test_invalid_git_sha_is_rejected(tmp_path: Path, git_sha: str) -> None:
    with pytest.raises(CollectionPreflightError, match="git_sha"):
        _build(tmp_path, git_sha=git_sha)


@pytest.mark.parametrize(
    "episode_id",
    ["", "../escape", "has space", "bad/slash", "x" * 129],
)
def test_unsafe_episode_id_is_rejected(
    tmp_path: Path,
    episode_id: str,
) -> None:
    with pytest.raises(CollectionPreflightError, match="episode_id"):
        _build(tmp_path, episode_id=episode_id)


def test_existing_episode_is_never_overwritten(tmp_path: Path) -> None:
    episode_root = tmp_path / "episodes"
    episode_dir = episode_root / "duplicate"
    episode_dir.mkdir(parents=True)

    with pytest.raises(CollectionPreflightError, match="already exists"):
        _build(
            tmp_path,
            episode_root=episode_root,
            episode_id="duplicate",
        )


def test_symlink_output_root_is_rejected(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-episodes"
    real_root.mkdir()
    linked_root = tmp_path / "linked-episodes"

    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(CollectionPreflightError, match="symbolic link"):
        _build(tmp_path, episode_root=linked_root)


def test_config_hash_uses_exact_file_bytes(tmp_path: Path) -> None:
    copied = tmp_path / "scene.json"
    copied.write_bytes(CONFIG.read_bytes())

    first = _build(tmp_path, config_path=copied)
    payload = json.loads(copied.read_text(encoding="utf-8"))
    payload["description"] = payload["description"] + " "
    copied.write_text(json.dumps(payload), encoding="utf-8")
    second = _build(tmp_path, config_path=copied)

    assert first.scene_config_sha256 != second.scene_config_sha256
