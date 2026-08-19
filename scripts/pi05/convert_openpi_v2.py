"""Convert validated Canonical V2 Episodes to complete LeRobot action windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from industrial_agent.data import SplitRegistry
from scripts.pi05.canonical_v2 import CanonicalV2Error, CanonicalV2Reader


ACTION_HORIZON = 10
ACTION_DIM = 7
STATE_DIM = 7
FPS = 10
STEP_NS = 100_000_000
ACTION_TICK_STRIDE = 12
MANIFEST_FILENAME = "pi05_v2_conversion.json"
MANIFEST_SHA256_FILENAME = "pi05_v2_conversion.sha256"

LeRobotDataset: Any = None
LEROBOT_IMPORT_ERROR: str | None = None
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except Exception as exc:  # pragma: no cover - optional local dependency
    LEROBOT_IMPORT_ERROR = str(exc)


@dataclass(frozen=True)
class ConversionResult:
    output_dir: Path
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    manifest_checksum_path: Path


def build_complete_action_windows(
    actions: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return exactly ``N-9`` lossless float32 windows; never pad or truncate."""

    array = np.asarray(actions)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError("actions must be native float32[N,7]")
    if not np.all(np.isfinite(array)):
        raise ValueError("actions contain NaN or Infinity")
    count = int(array.shape[0])
    if count < ACTION_HORIZON:
        raise ValueError(
            f"Episode has {count} actions; at least {ACTION_HORIZON} are required "
            "and padding is forbidden"
        )
    windows = tuple(
        np.ascontiguousarray(array[start : start + ACTION_HORIZON].copy())
        for start in range(count - ACTION_HORIZON + 1)
    )
    if len(windows) != count - 9:
        raise AssertionError("complete-window count invariant failed")
    return windows


def _unique_tick_index(ticks: np.ndarray, *, field: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for index, raw_tick in enumerate(ticks):
        tick = int(raw_tick)
        if tick in result:
            raise ValueError(f"{field} contains duplicate physics_tick {tick}")
        result[tick] = index
    return result


def _find_episode_dirs(data_dir: str | Path) -> list[Path]:
    source_root = Path(data_dir)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Canonical V2 root does not exist: {source_root}")
    episode_dirs = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir()
        and (path / "structure.json").is_file()
        and (path / "episode.h5").is_file()
    )
    if not episode_dirs:
        raise FileNotFoundError("no Canonical V2 Episodes found")
    return episode_dirs


def _validate_contiguous_action_timeline(reader: CanonicalV2Reader) -> None:
    ticks = [int(value) for value in reader.h5["actions/physics_tick"][:]]
    timestamps = [int(value) for value in reader.h5["actions/timestamp_ns"][:]]
    for previous, current in zip(ticks, ticks[1:]):
        if current - previous != ACTION_TICK_STRIDE:
            raise CanonicalV2Error(
                "action stream must be contiguous at exactly 10 Hz",
                episode_id=reader.episode_id,
                field="actions.physics_tick",
            )
    for previous, current in zip(timestamps, timestamps[1:]):
        if current - previous != STEP_NS:
            raise CanonicalV2Error(
                "action timestamps must be exactly 100 ms apart",
                episode_id=reader.episode_id,
                field="actions.timestamp_ns",
            )


def preflight_canonical_v2_windows(
    *,
    data_dir: str | Path,
    split_registry: SplitRegistry,
) -> dict[str, Any]:
    """Read-only validation and exact N-9 window count without LeRobot."""

    if not isinstance(split_registry, SplitRegistry):
        raise TypeError("split_registry must be a verified SplitRegistry")
    episodes: list[dict[str, Any]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    total_actions = 0
    total_windows = 0
    for episode_dir in _find_episode_dirs(data_dir):
        with CanonicalV2Reader(episode_dir) as reader:
            assignment = split_registry.assert_episode_allowed(
                reader.episode_id,
                is_training=False,
            )
            _validate_contiguous_action_timeline(reader)
            actions = np.asarray(reader.h5["actions/action_7d"][:])
            windows = build_complete_action_windows(actions)
            action_ticks = np.asarray(
                reader.h5["actions/physics_tick"][:],
                dtype=np.uint64,
            )
            camera_indices = _unique_tick_index(
                np.asarray(
                    reader.h5["cameras/CAM_A_TOP/physics_tick"][:],
                    dtype=np.uint64,
                ),
                field="CAM_A_TOP",
            )
            state_indices = _unique_tick_index(
                np.asarray(
                    reader.h5["robot_state/Arm_A/physics_tick"][:],
                    dtype=np.uint64,
                ),
                field="Arm_A state",
            )
            for start in range(len(windows)):
                tick = int(action_ticks[start])
                if tick not in camera_indices or tick not in state_indices:
                    raise CanonicalV2Error(
                        "window start lacks exact-tick CAM_A_TOP or Arm_A state",
                        episode_id=reader.episode_id,
                        field="window_alignment",
                    )
            action_count = int(actions.shape[0])
            window_count = len(windows)
            split = assignment.split.value
            split_counts[split] += 1
            total_actions += action_count
            total_windows += window_count
            episodes.append(
                {
                    "episode_id": reader.episode_id,
                    "split": split,
                    "action_count": action_count,
                    "window_count": window_count,
                }
            )
    return {
        "status": "ok",
        "source_format": "canonical_hdf5_v2",
        "action_horizon": ACTION_HORIZON,
        "window_rule": "N-9",
        "padding": "forbidden",
        "source_split_registry_sha256": split_registry.registry_sha256,
        "counts": {
            "episodes": len(episodes),
            "actions": total_actions,
            "windows": total_windows,
            "splits": split_counts,
        },
        "episodes": episodes,
    }


def _create_dataset(
    *,
    repo_id: str,
    output_dir: Path,
    robot_type: str,
) -> Any:
    if LeRobotDataset is None:
        raise RuntimeError(f"LeRobot is unavailable: {LEROBOT_IMPORT_ERROR}")
    features = {
        "image": {
            "dtype": "image",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_HORIZON, ACTION_DIM),
            "names": ["horizon", "action"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type=robot_type,
        fps=FPS,
        features=features,
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate_reopened_dataset(
    dataset: Any,
    *,
    expected_actions: Sequence[np.ndarray],
    expected_tasks: Sequence[str],
    expected_episodes: int,
) -> dict[str, Any]:
    if len(dataset) != len(expected_actions):
        raise ValueError(
            "LeRobot window count mismatch: "
            f"expected={len(expected_actions)} actual={len(dataset)}"
        )
    episode_count = getattr(dataset, "num_episodes", None)
    if episode_count is not None and int(episode_count) != expected_episodes:
        raise ValueError("LeRobot Episode count mismatch")
    max_action_error = 0.0
    for index, expected in enumerate(expected_actions):
        frame = dataset[index]
        if not isinstance(frame, Mapping):
            raise TypeError(f"LeRobot frame {index} is not a mapping")
        if set(("image", "state", "actions", "task")) - set(frame):
            raise ValueError(f"LeRobot frame {index} is missing required keys")
        image = _as_numpy(frame["image"])
        state = _as_numpy(frame["state"])
        actions = _as_numpy(frame["actions"])
        if image.shape not in ((720, 1280, 3), (3, 720, 1280)):
            raise ValueError(f"LeRobot frame {index} image shape is invalid")
        if state.dtype != np.float32 or state.shape != (STATE_DIM,):
            raise ValueError(f"LeRobot frame {index} state must be float32[7]")
        if actions.dtype != np.float32 or actions.shape != (
            ACTION_HORIZON,
            ACTION_DIM,
        ):
            raise ValueError(f"LeRobot frame {index} actions must be float32[10,7]")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"LeRobot frame {index} contains NaN or Infinity")
        if frame["task"] != expected_tasks[index]:
            raise ValueError(f"LeRobot frame {index} task changed during conversion")
        error = float(
            np.max(np.abs(actions.astype(np.float64) - expected.astype(np.float64)))
        )
        max_action_error = max(max_action_error, error)
    if max_action_error != 0.0:
        raise ValueError(
            f"action conversion must be lossless, max error={max_action_error}"
        )
    return {
        "episodes": expected_episodes,
        "windows": len(expected_actions),
        "max_action_error": max_action_error,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_conversion_manifest(path: str | Path) -> dict[str, Any]:
    """Verify the published conversion manifest and its SHA sidecar."""

    manifest_path = Path(path)
    checksum_path = manifest_path.with_name(MANIFEST_SHA256_FILENAME)
    parts = checksum_path.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[1].lstrip("*") != manifest_path.name:
        raise ValueError("conversion manifest checksum sidecar is malformed")
    expected = parts[0]
    actual = _sha256_file(manifest_path)
    if expected != actual:
        raise ValueError(
            f"conversion manifest SHA-256 mismatch: expected={expected} actual={actual}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("conversion manifest must be an object")
    if (
        manifest.get("source_format") != "canonical_hdf5_v2"
        or manifest.get("action_horizon") != ACTION_HORIZON
        or manifest.get("action_shape") != [ACTION_HORIZON, ACTION_DIM]
        or manifest.get("padding") != "forbidden"
        or manifest.get("window_rule") != "N-9"
    ):
        raise ValueError("conversion manifest contract is invalid")
    counts = manifest.get("counts")
    episodes = manifest.get("episodes")
    if (
        not isinstance(counts, dict)
        or not isinstance(episodes, list)
        or counts.get("episodes") != len(episodes)
        or not isinstance(counts.get("windows"), int)
        or counts["windows"] < 1
    ):
        raise ValueError("conversion manifest counts are invalid")
    expected_windows = 0
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("conversion manifest Episode entry is invalid")
        action_count = episode.get("source_action_count")
        window_count = episode.get("window_count")
        if (
            isinstance(action_count, bool)
            or not isinstance(action_count, int)
            or action_count < ACTION_HORIZON
            or window_count != action_count - 9
        ):
            raise ValueError("conversion manifest violates the N-9 rule")
        expected_windows += window_count
    if counts["windows"] != expected_windows:
        raise ValueError("conversion manifest total window count is inconsistent")
    roundtrip = manifest.get("roundtrip")
    if not isinstance(roundtrip, dict) or roundtrip.get("max_action_error") != 0.0:
        raise ValueError("conversion manifest lossless roundtrip proof is invalid")
    return manifest


def convert_canonical_v2_to_lerobot(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    repo_id: str,
    split_registry: SplitRegistry,
    robot_type: str = "franka",
    dataset_factory: Any | None = None,
    dataset_opener: Any | None = None,
) -> ConversionResult:
    """Convert every V2 Episode atomically and verify offline roundtrip."""

    if not isinstance(split_registry, SplitRegistry):
        raise TypeError("split_registry must be a verified SplitRegistry")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string")
    episode_dirs = _find_episode_dirs(data_dir)
    preflight_canonical_v2_windows(
        data_dir=data_dir,
        split_registry=split_registry,
    )
    final_dir = Path(output_dir).resolve()
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_dir.parent / f".{final_dir.name}.staging-{uuid.uuid4().hex}"
    create = dataset_factory or _create_dataset
    expected_actions: list[np.ndarray] = []
    expected_tasks: list[str] = []
    episode_manifest: list[dict[str, Any]] = []
    dataset: Any = None

    try:
        dataset = create(
            repo_id=repo_id,
            output_dir=staging_dir,
            robot_type=robot_type,
        )
        for output_episode_index, episode_dir in enumerate(episode_dirs):
            with CanonicalV2Reader(episode_dir) as reader:
                assignment = split_registry.assert_episode_allowed(
                    reader.episode_id,
                    is_training=False,
                )
                _validate_contiguous_action_timeline(reader)
                action_values = np.asarray(
                    reader.h5["actions/action_7d"][:],
                    dtype=np.float32,
                )
                windows = build_complete_action_windows(action_values)
                action_ticks = np.asarray(
                    reader.h5["actions/physics_tick"][:],
                    dtype=np.uint64,
                )
                camera_indices = _unique_tick_index(
                    np.asarray(
                        reader.h5["cameras/CAM_A_TOP/physics_tick"][:],
                        dtype=np.uint64,
                    ),
                    field="CAM_A_TOP",
                )
                state_indices = _unique_tick_index(
                    np.asarray(
                        reader.h5["robot_state/Arm_A/physics_tick"][:],
                        dtype=np.uint64,
                    ),
                    field="Arm_A state",
                )
                starts: list[int] = []
                for start, window in enumerate(windows):
                    tick = int(action_ticks[start])
                    camera_index = camera_indices.get(tick)
                    state_index = state_indices.get(tick)
                    if camera_index is None or state_index is None:
                        raise CanonicalV2Error(
                            "window start lacks exact-tick CAM_A_TOP or Arm_A state",
                            episode_id=reader.episode_id,
                            field="window_alignment",
                        )
                    image = np.asarray(
                        reader.h5["cameras/CAM_A_TOP/rgb"][camera_index],
                        dtype=np.uint8,
                    ).copy()
                    state = np.asarray(
                        reader.h5["robot_state/Arm_A/state_7d"][state_index],
                        dtype=np.float32,
                    ).copy()
                    dataset.add_frame(
                        {
                            "image": image,
                            "state": state,
                            "actions": window.copy(),
                            "task": reader.manifest["metadata"]["instruction"],
                        }
                    )
                    expected_actions.append(window.copy())
                    expected_tasks.append(reader.manifest["metadata"]["instruction"])
                    starts.append(start)
                dataset.save_episode()
                episode_manifest.append(
                    {
                        "lerobot_episode_index": output_episode_index,
                        "canonical_episode_id": reader.episode_id,
                        "canonical_split": assignment.split.value,
                        "source_hdf5_sha256": reader.manifest["storage"]["sha256"],
                        "source_structure_sha256": _sha256_file(
                            episode_dir / "structure.json"
                        ),
                        "source_action_count": int(action_values.shape[0]),
                        "window_count": len(windows),
                        "window_start_action_indices": starts,
                    }
                )
        stop_writer = getattr(dataset, "stop_image_writer", None)
        if not callable(stop_writer):
            raise TypeError("LeRobot dataset lacks stop_image_writer")
        stop_writer()
        if dataset_opener is None:
            if LeRobotDataset is None:
                raise RuntimeError(f"LeRobot is unavailable: {LEROBOT_IMPORT_ERROR}")

            def opener(root: Path, name: str) -> Any:
                return LeRobotDataset(repo_id=name, root=root)
        else:
            opener = dataset_opener
        reopened = opener(staging_dir, repo_id)
        smoke = _validate_reopened_dataset(
            reopened,
            expected_actions=expected_actions,
            expected_tasks=expected_tasks,
            expected_episodes=len(episode_manifest),
        )
        manifest = {
            "manifest_version": "1.0",
            "source_format": "canonical_hdf5_v2",
            "repo_id": repo_id,
            "robot_type": robot_type,
            "fps": FPS,
            "action_horizon": ACTION_HORIZON,
            "action_shape": [ACTION_HORIZON, ACTION_DIM],
            "padding": "forbidden",
            "window_rule": "N-9",
            "source_split_registry_sha256": split_registry.registry_sha256,
            "counts": {
                "episodes": len(episode_manifest),
                "windows": len(expected_actions),
            },
            "roundtrip": smoke,
            "episodes": episode_manifest,
        }
        manifest_path = staging_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha = _sha256_file(manifest_path)
        checksum_path = staging_dir / MANIFEST_SHA256_FILENAME
        checksum_path.write_text(
            f"{manifest_sha}  {manifest_path.name}\n",
            encoding="ascii",
        )
        verify_conversion_manifest(manifest_path)
        if final_dir.exists():
            raise FileExistsError(
                f"output appeared during conversion; refusing to overwrite: {final_dir}"
            )
        staging_dir.replace(final_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return ConversionResult(
        output_dir=final_dir,
        manifest=manifest,
        manifest_path=final_dir / MANIFEST_FILENAME,
        manifest_sha256=manifest_sha,
        manifest_checksum_path=final_dir / MANIFEST_SHA256_FILENAME,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or convert strict Canonical V2 PI05 Episodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-registry", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--repo-id", default="industrial/pi05-v2")
    parser.add_argument("--robot-type", default="franka")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        registry = SplitRegistry.load(args.split_registry)
        if args.preflight_only:
            report = preflight_canonical_v2_windows(
                data_dir=args.data_dir,
                split_registry=registry,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if not args.output_dir:
            raise ValueError("--output-dir is required unless --preflight-only is used")
        result = convert_canonical_v2_to_lerobot(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            split_registry=registry,
            robot_type=args.robot_type,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "manifest": str(result.manifest_path),
                    "manifest_sha256": result.manifest_sha256,
                    "counts": result.manifest["counts"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "ACTION_HORIZON",
    "ConversionResult",
    "build_complete_action_windows",
    "convert_canonical_v2_to_lerobot",
    "main",
    "preflight_canonical_v2_windows",
    "verify_conversion_manifest",
]


if __name__ == "__main__":
    sys.exit(main())
