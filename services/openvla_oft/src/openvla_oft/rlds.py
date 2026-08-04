"""Lightweight RLDS-style export for OpenVLA-OFT Arm_B data.

The production training stack may import this interchange into TFDS/RLDS
outside this core service.  This module stays dependency-light and writes only
metadata plus NumPy arrays so CI can validate the Canonical -> OpenVLA path.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .canonical import EXPECTED_INSTRUCTION, OpenVLACanonicalStep
from .dataset import ARM_B_ROLE
from .exceptions import ServiceError

RLDS_STYLE_SCHEMA_VERSION = "openvla_oft_rlds_style_v1"
_EXPORT_FILES = frozenset({"metadata.json", "steps.jsonl", "arrays.npz"})
_STEP_RECORD_FIELDS = frozenset(
    {
        "step_index",
        "robot_role",
        "image_array",
        "state_array",
        "action_array",
        "reward",
        "discount",
        "is_first",
        "is_last",
        "is_terminal",
        "metadata",
    }
)


def build_rlds_episode(
    steps: Iterable[OpenVLACanonicalStep],
) -> dict[str, Any]:
    """Build an in-memory RLDS-style episode from loaded OpenVLA steps."""

    step_list = tuple(steps)
    if not step_list:
        raise _bad_export("cannot export an empty OpenVLA episode")
    episode_id = step_list[0].episode_id
    task_id = step_list[0].task_id
    instruction = step_list[0].language_instruction
    split = step_list[0].source.split
    split_registry_sha256 = step_list[0].source.split_registry_sha256
    for index, step in enumerate(step_list):
        if step.episode_id != episode_id:
            raise _bad_export("all exported steps must come from one episode")
        if step.task_id != task_id:
            raise _bad_export("all exported steps must share task_id")
        if step.language_instruction != instruction:
            raise _bad_export("all exported steps must share language_instruction")
        if step.step_index != index:
            raise _bad_export("OpenVLA step_index must be contiguous from zero")
        if step.source.split != split or split != "train":
            raise _bad_export("all exported steps must belong to the Train split")
        if step.source.split_registry_sha256 != split_registry_sha256:
            raise _bad_export("all exported steps must share one Split Registry")
    final_index = len(step_list) - 1
    return {
        "schema_version": RLDS_STYLE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "task_id": task_id,
        "language_instruction": instruction,
        "robot_role": ARM_B_ROLE,
        "split": split,
        "split_registry_sha256": split_registry_sha256,
        "steps": [
            {
                "robot_role": ARM_B_ROLE,
                "observation": {
                    "image": step.image.copy(),
                    "state": list(step.state_7d),
                    "wrist_image": None,
                },
                "action": list(step.action_7d),
                "reward": 1.0 if index == final_index else 0.0,
                "discount": 1.0,
                "is_first": index == 0,
                "is_last": index == final_index,
                "is_terminal": index == final_index,
                "metadata": step.source.__dict__.copy(),
            }
            for index, step in enumerate(step_list)
        ],
    }


def write_rlds_style_episode(
    steps: Iterable[OpenVLACanonicalStep],
    output_dir: str | Path,
) -> Path:
    """Write one dependency-free RLDS-style episode export.

    Output files:
    - ``metadata.json``: episode-level fields and array shapes
    - ``steps.jsonl``: per-step flags and Canonical source lineage
    - ``arrays.npz``: ``images`` uint8 ``[T,720,1280,3]``, states ``[T,7]``,
      and actions ``[T,7]``
    """

    episode = build_rlds_episode(steps)
    root = Path(output_dir).expanduser()
    if root.is_symlink() or root.exists():
        raise _bad_export("output_dir must not exist")
    root.parent.mkdir(parents=True, exist_ok=True)

    episode_steps = episode["steps"]
    images = np.stack(
        [item["observation"]["image"] for item in episode_steps],
        axis=0,
    ).astype(np.uint8, copy=False)
    states = np.asarray(
        [item["observation"]["state"] for item in episode_steps],
        dtype=np.float32,
    )
    actions = np.asarray([item["action"] for item in episode_steps], dtype=np.float32)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.staging-",
            dir=root.parent,
        )
    )

    try:
        np.savez_compressed(
            staging / "arrays.npz",
            images=images,
            states=states,
            actions=actions,
        )
        step_records = []
        with (staging / "steps.jsonl").open("x", encoding="utf-8") as stream:
            for index, item in enumerate(episode_steps):
                record = {
                    "step_index": index,
                    "robot_role": item["robot_role"],
                    "image_array": f"arrays.npz:images[{index}]",
                    "state_array": f"arrays.npz:states[{index}]",
                    "action_array": f"arrays.npz:actions[{index}]",
                    "reward": item["reward"],
                    "discount": item["discount"],
                    "is_first": item["is_first"],
                    "is_last": item["is_last"],
                    "is_terminal": item["is_terminal"],
                    "metadata": item["metadata"],
                }
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
                stream.write("\n")
                step_records.append(record)

        metadata: dict[str, Any] = {
            "schema_version": episode["schema_version"],
            "episode_id": episode["episode_id"],
            "task_id": episode["task_id"],
            "language_instruction": episode["language_instruction"],
            "robot_role": episode["robot_role"],
            "split": episode["split"],
            "split_registry_sha256": episode["split_registry_sha256"],
            "step_count": len(episode_steps),
            "arrays": {
                "file": "arrays.npz",
                "images": {"dtype": "uint8", "shape": list(images.shape)},
                "states": {"dtype": "float32", "shape": list(states.shape)},
                "actions": {"dtype": "float32", "shape": list(actions.shape)},
            },
            "steps_file": "steps.jsonl",
            "canonical_sources": [record["metadata"] for record in step_records],
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if root.is_symlink() or root.exists():
            raise _bad_export("output_dir appeared during export")
        staging.replace(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return root


def load_rlds_style_episode(export_dir: str | Path) -> dict[str, Any]:
    """Fully traverse and validate one published RLDS-style smoke export."""

    root = Path(export_dir).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise _bad_export("export_dir must be a real directory")
    if {path.name for path in root.iterdir()} != _EXPORT_FILES:
        raise _bad_export("export_dir must contain exactly the three export files")

    metadata = summarize_rlds_style_export(root)
    required_metadata = {
        "schema_version",
        "episode_id",
        "task_id",
        "language_instruction",
        "robot_role",
        "split",
        "split_registry_sha256",
        "step_count",
        "arrays",
        "steps_file",
        "canonical_sources",
    }
    if set(metadata) != required_metadata:
        raise _bad_export("metadata.json has missing or unexpected fields")
    if metadata["schema_version"] != RLDS_STYLE_SCHEMA_VERSION:
        raise _bad_export("metadata.json has an unsupported schema_version")
    if metadata["language_instruction"] != EXPECTED_INSTRUCTION:
        raise _bad_export("metadata.json does not contain the frozen Arm_B instruction")
    if metadata["robot_role"] != ARM_B_ROLE or metadata["split"] != "train":
        raise _bad_export("metadata.json must identify Arm_B Train data")
    registry_sha = metadata["split_registry_sha256"]
    if (
        not isinstance(registry_sha, str)
        or len(registry_sha) != 71
        or not registry_sha.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in registry_sha[7:])
    ):
        raise _bad_export("metadata.json has an invalid Split Registry SHA-256")
    step_count = metadata["step_count"]
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 1
    ):
        raise _bad_export("metadata.json step_count must be a positive integer")
    if metadata["steps_file"] != "steps.jsonl":
        raise _bad_export("metadata.json steps_file is invalid")

    try:
        records = [
            json.loads(line)
            for line in (root / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _bad_export("steps.jsonl is unreadable") from exc
    if len(records) != step_count:
        raise _bad_export("steps.jsonl count does not match metadata.json")
    canonical_sources = metadata["canonical_sources"]
    if not isinstance(canonical_sources, list) or len(canonical_sources) != step_count:
        raise _bad_export("canonical_sources count does not match step_count")

    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != _STEP_RECORD_FIELDS:
            raise _bad_export(f"steps.jsonl record {index} has invalid fields")
        expected_flags = (index == 0, index == step_count - 1)
        if (
            record["step_index"] != index
            or record["robot_role"] != ARM_B_ROLE
            or record["image_array"] != f"arrays.npz:images[{index}]"
            or record["state_array"] != f"arrays.npz:states[{index}]"
            or record["action_array"] != f"arrays.npz:actions[{index}]"
            or record["is_first"] is not expected_flags[0]
            or record["is_last"] is not expected_flags[1]
            or record["is_terminal"] is not expected_flags[1]
        ):
            raise _bad_export(f"steps.jsonl record {index} violates step boundaries")
        source = record["metadata"]
        if not isinstance(source, Mapping) or source != canonical_sources[index]:
            raise _bad_export(f"steps.jsonl record {index} has inconsistent lineage")
        if (
            source.get("episode_id") != metadata["episode_id"]
            or source.get("task_id") != metadata["task_id"]
            or source.get("split") != "train"
            or source.get("split_registry_sha256") != registry_sha
            or source.get("camera_id") != "CAM_B_TOP"
            or source.get("state_arm_id") != "Arm_B"
            or source.get("camera_physics_tick") != source.get("action_physics_tick")
            or source.get("state_physics_tick") != source.get("action_physics_tick")
        ):
            raise _bad_export(f"steps.jsonl record {index} has invalid source lineage")

    try:
        with np.load(root / "arrays.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"images", "states", "actions"}:
                raise _bad_export("arrays.npz has missing or unexpected arrays")
            images = np.asarray(archive["images"]).copy()
            states = np.asarray(archive["states"]).copy()
            actions = np.asarray(archive["actions"]).copy()
    except (OSError, ValueError) as exc:
        raise _bad_export("arrays.npz is unreadable") from exc
    if images.dtype != np.uint8 or images.shape != (step_count, 720, 1280, 3):
        raise _bad_export("arrays.npz images must be uint8 [T,720,1280,3]")
    if states.dtype != np.float32 or states.shape != (step_count, 7):
        raise _bad_export("arrays.npz states must be float32 [T,7]")
    if actions.dtype != np.float32 or actions.shape != (step_count, 7):
        raise _bad_export("arrays.npz actions must be float32 [T,7]")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise _bad_export("arrays.npz contains non-finite state/action values")
    expected_arrays = {
        "file": "arrays.npz",
        "images": {"dtype": "uint8", "shape": list(images.shape)},
        "states": {"dtype": "float32", "shape": list(states.shape)},
        "actions": {"dtype": "float32", "shape": list(actions.shape)},
    }
    if metadata["arrays"] != expected_arrays:
        raise _bad_export("metadata.json array declarations do not match arrays.npz")
    return {
        "metadata": dict(metadata),
        "steps": tuple(dict(record) for record in records),
        "images": images,
        "states": states,
        "actions": actions,
    }


def summarize_rlds_style_export(export_dir: str | Path) -> Mapping[str, Any]:
    """Read export metadata without loading image arrays."""

    metadata_path = Path(export_dir) / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _bad_export("metadata.json is missing or unreadable") from exc
    if not isinstance(metadata, Mapping):
        raise _bad_export("metadata.json must contain an object")
    return metadata


def _bad_export(message: str) -> ServiceError:
    return ServiceError("DATA_3003_RLDS_EXPORT_INVALID", message, retryable=False)
