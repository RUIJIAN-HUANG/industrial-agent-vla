"""Validated Train-only source adapter for pre-windowed PI05 LeRobot V2 data."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from industrial_agent.data import SplitRegistry
from scripts.pi05.convert_openpi_v2 import (
    ACTION_DIM,
    ACTION_HORIZON,
    MANIFEST_FILENAME,
    STATE_DIM,
    verify_conversion_manifest,
)

logger = logging.getLogger(__name__)


class V2Dataset(Protocol):
    """Minimal random-access interface required from a LeRobot dataset."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


DatasetOpener = Callable[[Path, str], V2Dataset]


class V2NormSourceError(ValueError):
    """Raised when V2 data cannot be proven safe for norm-stat computation."""


@dataclass(frozen=True)
class ValidatedV2NormSource:
    """Validated arrays; state is [N,7], actions is [N,10,7], both float32."""

    state: np.ndarray
    actions: np.ndarray
    source_manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar_int(value: Any, *, field: str, dataset_index: int) -> int:
    scalar = _to_numpy(value)
    if scalar.size != 1:
        raise V2NormSourceError(
            f"LeRobot frame {dataset_index} {field} must be scalar, got {scalar.shape}"
        )
    item = scalar.reshape(-1)[0]
    if isinstance(item, (bool, np.bool_)):
        raise V2NormSourceError(
            f"LeRobot frame {dataset_index} {field} must be an integer"
        )
    try:
        result = int(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise V2NormSourceError(
            f"LeRobot frame {dataset_index} {field} must be an integer"
        ) from exc
    if float(item) != float(result):
        raise V2NormSourceError(
            f"LeRobot frame {dataset_index} {field} must be an integer"
        )
    return result


def _check_deadline(deadline: float, *, operation: str) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError(f"LeRobot V2 {operation} exceeded the configured timeout")


def load_lerobot_v2_norm_source(
    dataset_path: str | Path,
    *,
    repo_id: str,
    split_registry: SplitRegistry,
    dataset_opener: DatasetOpener,
    manifest_path: str | Path | None = None,
    io_timeout_s: float = 300.0,
) -> ValidatedV2NormSource:
    """Validate V2 provenance and load Train rows for norm statistics.

    The timeout is an overall deadline checked around external file/dataset I/O and
    during the frame scan. It prevents an indefinitely slow scan from publishing
    partial or unverified statistics.
    """

    if not isinstance(split_registry, SplitRegistry):
        raise TypeError("split_registry must be a verified SplitRegistry")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise V2NormSourceError("repo_id is required for LeRobot V2 input")
    if isinstance(io_timeout_s, bool) or not isinstance(io_timeout_s, (int, float)):
        raise TypeError("io_timeout_s must be a positive number")
    if not np.isfinite(io_timeout_s) or io_timeout_s <= 0:
        raise V2NormSourceError("io_timeout_s must be a positive finite number")

    root = Path(dataset_path).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise V2NormSourceError(
            f"LeRobot V2 dataset path must be a real directory: {root}"
        )
    conversion_path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else root / MANIFEST_FILENAME
    )
    deadline = time.monotonic() + float(io_timeout_s)

    try:
        _check_deadline(deadline, operation="manifest verification")
        manifest = verify_conversion_manifest(conversion_path)
        _check_deadline(deadline, operation="manifest verification")

        if manifest.get("repo_id") != repo_id:
            raise V2NormSourceError(
                "LeRobot V2 repo_id does not match conversion manifest: "
                f"requested={repo_id!r} manifest={manifest.get('repo_id')!r}"
            )
        if (
            manifest.get("source_split_registry_sha256")
            != split_registry.registry_sha256
        ):
            raise V2NormSourceError(
                "LeRobot V2 conversion manifest Split Registry SHA does not match "
                "the supplied registry"
            )

        episodes = manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise V2NormSourceError("LeRobot V2 manifest contains no episodes")

        seen_source_segments: set[tuple[str, str | None]] = set()
        sources: list[dict[str, Any]] = []
        train_ranges: list[tuple[int, int, int, dict[str, Any]]] = []
        offset = 0
        for expected_episode_index, raw_item in enumerate(episodes):
            if not isinstance(raw_item, dict):
                raise V2NormSourceError("LeRobot V2 episode entry must be an object")
            item = dict(raw_item)
            episode_id = item.get("canonical_episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise V2NormSourceError(
                    "LeRobot V2 episode is missing canonical_episode_id"
                )
            active_arm = item.get("active_arm")
            if active_arm is not None and active_arm not in ("Arm_A", "Arm_B"):
                raise V2NormSourceError(
                    f"LeRobot V2 active_arm is invalid for {episode_id!r}"
                )
            source_segment = (episode_id, active_arm)
            if source_segment in seen_source_segments:
                raise V2NormSourceError(
                    "LeRobot V2 manifest repeats Canonical arm segment "
                    f"{source_segment!r}"
                )
            seen_source_segments.add(source_segment)
            if item.get("lerobot_episode_index") != expected_episode_index:
                raise V2NormSourceError(
                    f"LeRobot V2 episode index is not contiguous for {episode_id!r}"
                )
            authoritative_split = split_registry.get_split(episode_id).value
            if item.get("canonical_split") != authoritative_split:
                raise V2NormSourceError(
                    "LeRobot V2 manifest split does not match Split Registry for "
                    f"{episode_id!r}: manifest={item.get('canonical_split')!r} "
                    f"registry={authoritative_split!r}"
                )
            window_count = item.get("window_count")
            if isinstance(window_count, bool) or not isinstance(window_count, int):
                raise V2NormSourceError(
                    f"LeRobot V2 window_count must be an integer for {episode_id!r}"
                )
            if authoritative_split == "train":
                split_registry.assert_episode_allowed(episode_id, is_training=True)
                train_ranges.append(
                    (offset, window_count, expected_episode_index, item)
                )
                sources.append(item)
            offset += window_count

        expected_windows = int(manifest["counts"]["windows"])
        if offset != expected_windows:
            raise V2NormSourceError(
                f"LeRobot V2 episode windows do not total {expected_windows}: {offset}"
            )
        if not train_ranges:
            raise V2NormSourceError("LeRobot V2 manifest contains no Train windows")

        _check_deadline(deadline, operation="dataset open")
        dataset = dataset_opener(root, repo_id)
        _check_deadline(deadline, operation="dataset open")
        if len(dataset) != expected_windows:
            raise V2NormSourceError(
                f"LeRobot V2 dataset/manifest window count mismatch: "
                f"{len(dataset)} != {expected_windows}"
            )

        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for start, count, episode_index, item in train_ranges:
            episode_task: str | None = None
            for local_frame_index in range(count):
                dataset_index = start + local_frame_index
                _check_deadline(
                    deadline, operation=f"frame scan at index {dataset_index}"
                )
                frame = dataset[dataset_index]
                if not isinstance(frame, Mapping):
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} is not a mapping"
                    )
                required = {"state", "actions", "task", "episode_index", "frame_index"}
                missing = sorted(required.difference(frame))
                if missing:
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} is missing keys: {missing}"
                    )
                if (
                    _scalar_int(
                        frame["episode_index"],
                        field="episode_index",
                        dataset_index=dataset_index,
                    )
                    != episode_index
                ):
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} episode_index mismatch"
                    )
                if (
                    _scalar_int(
                        frame["frame_index"],
                        field="frame_index",
                        dataset_index=dataset_index,
                    )
                    != local_frame_index
                ):
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} frame_index mismatch"
                    )
                task = frame["task"]
                if not isinstance(task, str) or not task.strip():
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} task must be non-empty"
                    )
                if episode_task is None:
                    episode_task = task
                elif task != episode_task:
                    raise V2NormSourceError(
                        "LeRobot V2 task changed within Canonical Episode "
                        f"{item['canonical_episode_id']!r}"
                    )

                state = _to_numpy(frame["state"])
                action = _to_numpy(frame["actions"])
                if state.dtype != np.float32 or state.shape != (STATE_DIM,):
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} state must be "
                        f"float32[{STATE_DIM}], got dtype={state.dtype} shape={state.shape}"
                    )
                if action.dtype != np.float32 or action.shape != (
                    ACTION_HORIZON,
                    ACTION_DIM,
                ):
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} actions must be "
                        f"float32[{ACTION_HORIZON},{ACTION_DIM}], "
                        f"got dtype={action.dtype} shape={action.shape}"
                    )
                if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
                    raise V2NormSourceError(
                        f"LeRobot V2 frame {dataset_index} contains NaN or Infinity"
                    )
                states.append(state)
                actions.append(action)

        _check_deadline(deadline, operation="frame scan")
        state_array = np.stack(states).astype(np.float32, copy=False)
        action_array = np.stack(actions).astype(np.float32, copy=False)
        return ValidatedV2NormSource(
            state=state_array,
            actions=action_array,
            source_manifest={
                "input_format": "lerobot_v2",
                "input_path": str(root),
                "repo_id": repo_id,
                "split": "train",
                "split_registry_sha256": split_registry.registry_sha256,
                "conversion_manifest_path": str(conversion_path),
                "conversion_manifest_sha256": _sha256(conversion_path),
                "sources": sources,
                "excluded": {
                    "non_train_episodes": len(episodes) - len(sources),
                },
                "counts": {
                    "windows": int(state_array.shape[0]),
                    "action_vectors": int(
                        action_array.shape[0] * action_array.shape[1]
                    ),
                },
            },
        )
    except TimeoutError:
        logger.exception("LeRobot V2 norm source timed out: %s", root)
        raise
    except V2NormSourceError:
        logger.exception("LeRobot V2 norm source validation failed: %s", root)
        raise
    except (
        FileNotFoundError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
    ) as exc:
        logger.exception("LeRobot V2 norm source I/O or schema failure: %s", root)
        raise V2NormSourceError(f"failed to validate LeRobot V2 source: {exc}") from exc
