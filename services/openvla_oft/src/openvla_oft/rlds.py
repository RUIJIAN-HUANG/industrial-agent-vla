"""Lightweight RLDS-style export for OpenVLA-OFT Arm_B data.

The production training stack may import this interchange into TFDS/RLDS
outside this core service.  This module stays dependency-light and writes only
metadata plus NumPy arrays so CI can validate the Canonical -> OpenVLA path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .canonical import OpenVLACanonicalStep
from .exceptions import ServiceError

RLDS_STYLE_SCHEMA_VERSION = "openvla_oft_rlds_style_v1"


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
    for index, step in enumerate(step_list):
        if step.episode_id != episode_id:
            raise _bad_export("all exported steps must come from one episode")
        if step.task_id != task_id:
            raise _bad_export("all exported steps must share task_id")
        if step.language_instruction != instruction:
            raise _bad_export("all exported steps must share language_instruction")
        if step.step_index != index:
            raise _bad_export("OpenVLA step_index must be contiguous from zero")
    final_index = len(step_list) - 1
    return {
        "schema_version": RLDS_STYLE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "task_id": task_id,
        "language_instruction": instruction,
        "steps": [
            {
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
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise _bad_export("output_dir must be empty or absent")
    root.mkdir(parents=True, exist_ok=True)

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

    np.savez_compressed(
        root / "arrays.npz",
        images=images,
        states=states,
        actions=actions,
    )
    step_records = []
    with (root / "steps.jsonl").open("x", encoding="utf-8") as stream:
        for index, item in enumerate(episode_steps):
            record = {
                "step_index": index,
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
    (root / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root


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
