import argparse
from pathlib import Path

import pytest

from simulation.v2_collection_entry import build_parser, preflight_from_args


def _args(tmp_path: Path) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--episode-root", str(tmp_path / "episodes"),
            "--cas-root", str(tmp_path / "cas"),
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--output-scene", str(tmp_path / "scene.usda"),
            "--episode-id", "v2-practice-001",
            "--task-id", "v2-practice-grasp-upright-shaft",
            "--instruction", "使用 Arm_A 抓取 P01 并放入 S11",
            "--scene-seed", "7",
            "--split", "practice",
        ]
    )


def test_cli_builds_visible_practice_preflight(tmp_path: Path) -> None:
    result = preflight_from_args(
        _args(tmp_path), git_sha="a" * 40, worktree_clean=True
    )
    assert result.scene_id == "single_bin_manual_industrial_v2"
    assert result.training_allowed is False
    assert result.full_task_required is False


def test_cli_rejects_oversized_motion_step(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.translation_step_m = 0.02
    with pytest.raises(ValueError, match="translation-step"):
        preflight_from_args(
            args, git_sha="a" * 40, worktree_clean=True
        )
