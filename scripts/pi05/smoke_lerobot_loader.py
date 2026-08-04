"""Offline LeRobot loader smoke test for the PI05 data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.pi05.train_config import OPENPI_COMMIT
from scripts.pi05.provenance_context import (
    LEROBOT_PROVENANCE_MANIFEST_TYPE,
    ProvenanceContext,
    resolve_provenance_context,
    validate_provenance_context,
)

PROVENANCE_FILENAME = "pi05_provenance.json"
PROVENANCE_SHA256_FILENAME = "pi05_provenance.sha256"
REQUIRED_FRAME_KEYS = ("image", "state", "actions", "task")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _dataset_episode_count(dataset: Any) -> int | None:
    for owner in (dataset, getattr(dataset, "meta", None)):
        if owner is None:
            continue
        for attribute in ("num_episodes", "total_episodes"):
            value = getattr(owner, attribute, None)
            if value is not None:
                return int(value)
    tasks = getattr(dataset, "episode_tasks", None)
    if tasks is not None:
        return len(tasks)
    return None


def validate_dataset_instance(
    dataset: Any,
    *,
    expected_frames: int,
    expected_episodes: int,
    expected_actions: Sequence[np.ndarray] | None = None,
    expected_tasks: Sequence[str] | None = None,
    expected_state_dim: int | None = None,
    roundtrip_samples: int = 10,
) -> dict[str, int | float]:
    """Traverse an already-open dataset and enforce count/action invariants."""

    actual_frames = len(dataset)
    if actual_frames != expected_frames:
        raise ValueError(
            f"LeRobot frame count mismatch: expected={expected_frames} "
            f"actual={actual_frames}"
        )
    actual_episodes = _dataset_episode_count(dataset)
    if actual_episodes is not None and actual_episodes != expected_episodes:
        raise ValueError(
            f"LeRobot episode count mismatch: expected={expected_episodes} "
            f"actual={actual_episodes}"
        )

    observed_actions: list[np.ndarray] = []
    for index in range(actual_frames):
        frame = dataset[index]
        if not isinstance(frame, Mapping):
            raise TypeError(f"LeRobot frame {index} is not a mapping")
        missing = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
        if missing:
            raise ValueError(f"LeRobot frame {index} is missing keys {missing}")
        image = _as_numpy(frame["image"])
        state = _as_numpy(frame["state"])
        action = _as_numpy(frame["actions"])
        task = frame["task"]
        if image.shape not in ((720, 1280, 3), (3, 720, 1280)):
            raise ValueError(
                f"LeRobot frame {index} image must represent 1280x720 RGB, "
                f"got {image.shape}"
            )
        if image.shape == (720, 1280, 3):
            if image.dtype != np.uint8:
                raise ValueError(
                    f"LeRobot frame {index} HWC image must be uint8, got {image.dtype}"
                )
        elif (
            image.dtype != np.float32
            or not np.all(np.isfinite(image))
            or (image.size and (float(image.min()) < 0.0 or float(image.max()) > 1.0))
        ):
            raise ValueError(
                f"LeRobot frame {index} pinned CHW image must be finite float32 "
                f"in [0,1], got dtype={image.dtype}"
            )
        if (
            state.dtype != np.float32
            or state.ndim != 1
            or not np.all(np.isfinite(state))
        ):
            raise ValueError(
                f"LeRobot frame {index} state must be finite float32[D], "
                f"got dtype={state.dtype} shape={state.shape}"
            )
        if expected_state_dim is not None and state.shape != (expected_state_dim,):
            raise ValueError(
                f"LeRobot frame {index} state shape mismatch: "
                f"expected=({expected_state_dim},) actual={state.shape}"
            )
        if (
            action.dtype != np.float32
            or action.shape != (7,)
            or not np.all(np.isfinite(action))
        ):
            raise ValueError(
                f"LeRobot frame {index} action must be finite float32[7], "
                f"got dtype={action.dtype} shape={action.shape}"
            )
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"LeRobot frame {index} task must be a non-empty string")
        observed_actions.append(action)
        if expected_tasks is not None:
            if len(expected_tasks) != actual_frames:
                raise ValueError(
                    "expected task count does not match dataset frame count"
                )
            if task != expected_tasks[index]:
                raise ValueError(
                    f"LeRobot frame {index} task mismatch: "
                    f"expected={expected_tasks[index]!r} actual={task!r}"
                )

    max_error = 0.0
    if expected_actions is not None:
        if len(expected_actions) != actual_frames:
            raise ValueError(
                "expected action count does not match the traversed LeRobot dataset"
            )
        sample_count = min(roundtrip_samples, actual_frames)
        rng = np.random.default_rng(20260802)
        indices = rng.choice(actual_frames, size=sample_count, replace=False)
        for index in indices:
            expected = np.asarray(expected_actions[int(index)], dtype=np.float32)
            error = float(
                np.max(
                    np.abs(observed_actions[int(index)].astype(np.float64) - expected)
                )
            )
            max_error = max(max_error, error)
        if max_error >= 1e-6:
            raise ValueError(
                f"action roundtrip error must be <1e-6, got {max_error:.9g}"
            )

    return {
        "episodes": expected_episodes,
        "frames": actual_frames,
        "language_frames": actual_frames,
        "roundtrip_samples": (
            min(roundtrip_samples, actual_frames)
            if expected_actions is not None
            else 0
        ),
        "max_action_error": max_error,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance_checksum(path: Path) -> tuple[Path, str]:
    """Write the manifest checksum beside a staged provenance file."""

    digest = _sha256_file(path)
    checksum_path = path.with_name(PROVENANCE_SHA256_FILENAME)
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return checksum_path, digest


def verify_provenance_checksum(path: Path) -> str:
    """Fail closed if the provenance or its required sidecar changed."""

    checksum_path = path.with_name(PROVENANCE_SHA256_FILENAME)
    try:
        line = checksum_path.read_text(encoding="ascii").strip()
    except Exception as exc:
        raise ValueError(
            f"cannot read provenance checksum {checksum_path}: {exc}"
        ) from exc
    parts = line.split()
    if len(parts) != 2 or parts[1].lstrip("*") != path.name:
        raise ValueError("provenance checksum sidecar has invalid format")
    expected = parts[0].lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("provenance checksum sidecar has invalid SHA-256")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"provenance SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def load_provenance(path: Path) -> dict[str, Any]:
    verify_provenance_checksum(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read provenance manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("provenance manifest must be a JSON object")
    return payload


def validate_provenance_manifest(
    provenance: Mapping[str, Any],
    *,
    expected_repo_id: str | None = None,
    expected_provenance_context: ProvenanceContext | None = None,
) -> list[dict[str, Any]]:
    """Validate conversion traceability before any dataset frame is trusted."""

    if provenance.get("manifest_type") != LEROBOT_PROVENANCE_MANIFEST_TYPE:
        raise ValueError("LeRobot provenance manifest_type is invalid")
    if provenance.get("source_format") != "canonical_hdf5_v1":
        raise ValueError(
            "LeRobot provenance source_format must be canonical_hdf5_v1"
        )
    validate_provenance_context(
        provenance.get("producer"),
        expected=expected_provenance_context,
    )
    registry_sha = provenance.get("source_split_registry_sha256")
    if not isinstance(registry_sha, str) or _SHA256_HEX.fullmatch(registry_sha) is None:
        raise ValueError("LeRobot provenance Split Registry SHA-256 is invalid")
    repo_id = provenance.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("LeRobot provenance repo_id must be a non-empty string")
    robot_type = provenance.get("robot_type")
    if not isinstance(robot_type, str) or not robot_type.strip():
        raise ValueError("LeRobot provenance robot_type must be a non-empty string")
    if expected_repo_id is not None and provenance.get("repo_id") != expected_repo_id:
        raise ValueError(
            "LeRobot provenance repo_id does not match the requested dataset"
        )
    if provenance.get("schema_version") != "1.0":
        raise ValueError("LeRobot provenance schema_version is invalid")
    for key in ("fps", "timestamp_tolerance_ns"):
        value = provenance.get(key)
        minimum = 1 if key == "fps" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "positive" if minimum == 1 else "non-negative"
            raise ValueError(f"LeRobot provenance {key} must be a {qualifier} integer")
    image = provenance.get("image")
    if not isinstance(image, dict) or image != {
        "camera_id": "CAM_A_TOP",
        "dtype": "uint8",
        "shape": [720, 1280, 3],
        "preprocessed": False,
        "wrist_image": None,
    }:
        raise ValueError("LeRobot provenance image contract is invalid")
    mapper = provenance.get("state_mapper")
    if not isinstance(mapper, dict):
        raise ValueError("LeRobot provenance is missing state_mapper")
    if (
        not isinstance(mapper.get("name"), str)
        or not mapper["name"]
        or isinstance(mapper.get("state_dim"), bool)
        or not isinstance(mapper.get("state_dim"), int)
        or mapper["state_dim"] < 1
        or not isinstance(mapper.get("approved_for_production"), bool)
        or not isinstance(mapper.get("version"), str)
        or not mapper["version"]
    ):
        raise ValueError("LeRobot provenance state_mapper is invalid")
    counts = provenance.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("LeRobot provenance is missing counts")
    required_counts = {
        "episodes",
        "steps",
        "images",
        "instructions",
        "language_frames",
        "states",
        "actions",
    }
    if not required_counts.issubset(counts):
        raise ValueError("LeRobot provenance counts are incomplete")
    for key in required_counts:
        if isinstance(counts[key], bool) or not isinstance(counts[key], int):
            raise ValueError(f"LeRobot provenance count {key!r} must be an integer")
    frame_count = counts["steps"]
    if frame_count < 1 or any(
        counts[key] != frame_count
        for key in ("images", "language_frames", "states", "actions")
    ):
        raise ValueError("LeRobot provenance frame-level counts are inconsistent")
    raw_episodes = provenance.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != counts["episodes"]:
        raise ValueError("LeRobot provenance episode count is inconsistent")
    if counts["instructions"] != counts["episodes"]:
        raise ValueError("LeRobot provenance instruction count is inconsistent")
    if counts["episodes"] < 1:
        raise ValueError("LeRobot provenance must contain at least one Episode")

    roundtrip = provenance.get("roundtrip")
    if not isinstance(roundtrip, dict):
        raise ValueError("LeRobot provenance roundtrip result is missing")
    if (
        roundtrip.get("episodes") != counts["episodes"]
        or roundtrip.get("frames") != frame_count
        or roundtrip.get("language_frames") != frame_count
    ):
        raise ValueError("LeRobot provenance roundtrip counts are inconsistent")
    samples = roundtrip.get("roundtrip_samples")
    max_error = roundtrip.get("max_action_error")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples != min(10, frame_count)
        or samples > frame_count
        or isinstance(max_error, bool)
        or not isinstance(max_error, (int, float))
        or not math.isfinite(float(max_error))
        or float(max_error) < 0.0
        or float(max_error) >= 1e-6
    ):
        raise ValueError("LeRobot provenance action roundtrip result is invalid")

    episodes: list[dict[str, Any]] = []
    total_steps = 0
    for expected_index, item in enumerate(raw_episodes):
        if not isinstance(item, dict):
            raise ValueError("LeRobot provenance episode entry must be an object")
        if item.get("lerobot_episode_index") != expected_index:
            raise ValueError("LeRobot provenance episode indices must be contiguous")
        if item.get("canonical_split") not in {"train", "val", "test"}:
            raise ValueError("LeRobot provenance canonical_split is invalid")
        if item.get("robot_role") != "arm_a_pi05":
            raise ValueError("LeRobot provenance robot_role must be derived as arm_a_pi05")
        episode_id = item.get("canonical_episode_id")
        instruction = item.get("instruction")
        count = item.get("step_count")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("LeRobot provenance canonical_episode_id is invalid")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("LeRobot provenance instruction is invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("LeRobot provenance step_count must be a positive integer")
        if (
            item.get("instruction_sha256")
            != hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        ):
            raise ValueError("LeRobot provenance instruction SHA-256 is invalid")
        for key in (
            "source_structure_sha256",
            "source_hdf5_sha256",
            "source_split_registry_sha256",
        ):
            if (
                not isinstance(item.get(key), str)
                or _SHA256_HEX.fullmatch(item[key]) is None
            ):
                raise ValueError(f"LeRobot provenance {key} is invalid")
        recorder_git_sha = item.get("source_recorder_git_sha")
        if (
            not isinstance(recorder_git_sha, str)
            or re.fullmatch(r"[0-9a-f]{40}", recorder_git_sha) is None
        ):
            raise ValueError("LeRobot provenance source_recorder_git_sha is invalid")
        vectors = (
            ("source_action_sequence_ids", int),
            ("source_action_timestamp_ns", int),
            ("source_physics_ticks", int),
            ("source_camera_sequence_ids", int),
            ("source_camera_timestamp_ns", int),
            ("source_state_sequence_ids", int),
            ("source_state_timestamp_ns", int),
            ("source_image_datasets", str),
            ("source_image_sha256", str),
            ("source_action_duration_s", (int, float)),
        )
        for key, item_type in vectors:
            values = item.get(key)
            if (
                not isinstance(values, list)
                or len(values) != count
                or not all(
                    not isinstance(value, bool) and isinstance(value, item_type)
                    for value in values
                )
            ):
                raise ValueError(f"LeRobot provenance {key} is invalid")
        if any(
            _SHA256_HEX.fullmatch(value) is None
            for value in item["source_image_sha256"]
        ):
            raise ValueError("LeRobot provenance source_image_sha256 is invalid")
        indices = item["source_action_sequence_ids"]
        if any(value < 0 for value in indices) or any(
            current <= previous for previous, current in zip(indices, indices[1:])
        ):
            raise ValueError(
                "LeRobot provenance action sequence IDs must strictly increase"
            )
        timestamps = item["source_action_timestamp_ns"]
        if any(value < 0 for value in timestamps) or any(
            current <= previous for previous, current in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("LeRobot provenance action timestamps must increase")
        ticks = item["source_physics_ticks"]
        if any(value < 0 for value in ticks) or any(
            current <= previous for previous, current in zip(ticks, ticks[1:])
        ):
            raise ValueError("LeRobot provenance physics ticks must increase")
        if item["source_split_registry_sha256"] != registry_sha:
            raise ValueError("Episode Split Registry SHA does not match provenance")
        if any(
            not value.startswith("/cameras/CAM_A_TOP/rgb[")
            or not value.endswith("]")
            or "\\" in value
            for value in item["source_image_datasets"]
        ):
            raise ValueError("LeRobot provenance source_image_datasets is invalid")
        if any(
            not math.isfinite(float(value)) or value <= 0
            for value in item["source_action_duration_s"]
        ):
            raise ValueError("LeRobot provenance source_action_duration_s is invalid")
        total_steps += count
        episodes.append(item)
    if total_steps != frame_count:
        raise ValueError("LeRobot provenance total step count is inconsistent")
    return episodes


def open_offline_dataset(dataset_root: Path, repo_id: str) -> Any:
    """Open LeRobot from local files with network access disabled."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            f"LeRobot is required for offline loader smoke: {exc}"
        ) from exc
    return LeRobotDataset(repo_id=repo_id, root=dataset_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traverse a PI05 LeRobot dataset offline"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"Defaults to <dataset-root>/{PROVENANCE_FILENAME}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else dataset_root / PROVENANCE_FILENAME
    )
    try:
        provenance_context = resolve_provenance_context(
            repo_root=args.project_root,
            openpi_commit=args.openpi_commit,
            expected_openpi_commit=OPENPI_COMMIT,
        )
        manifest = load_provenance(manifest_path)
        episodes = validate_provenance_manifest(
            manifest,
            expected_repo_id=args.repo_id,
            expected_provenance_context=provenance_context,
        )
        counts = manifest.get("counts")
        if not isinstance(counts, dict):
            raise ValueError("provenance manifest is missing counts")
        dataset = open_offline_dataset(dataset_root, args.repo_id)
        expected_tasks = [
            item["instruction"]
            for item in episodes
            for _ in range(int(item["step_count"]))
        ]
        mapper = manifest.get("state_mapper")
        if not isinstance(mapper, dict):
            raise ValueError("provenance manifest is missing state_mapper")
        result = validate_dataset_instance(
            dataset,
            expected_frames=int(counts["steps"]),
            expected_episodes=int(counts["episodes"]),
            expected_tasks=expected_tasks,
            expected_state_dim=int(mapper["state_dim"]),
        )
    except Exception as exc:
        print(f"ERROR: offline LeRobot loader smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
