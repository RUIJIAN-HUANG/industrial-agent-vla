"""Auditable, deterministic input adapter for the frozen Isaac sensor profile."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def adapt_top_only_droid_input(data: Mapping[str, Any]) -> dict[str, Any]:
    """Map one top-camera observation to the pi0.5/DROID model contract.

    Missing wrist cameras are represented by zero arrays whose masks are false.
    This function performs no learning, calibration, or normalization.
    """

    base_image = np.asarray(data["observation/exterior_image_1_left"])
    joints = np.asarray(data["observation/joint_position"], dtype=np.float32)
    gripper = np.asarray(
        data["observation/gripper_position"], dtype=np.float32
    ).reshape(-1)
    if base_image.ndim != 3 or base_image.shape[2] != 3:
        raise ValueError("exterior image must have shape [height,width,3]")
    if base_image.dtype != np.uint8:
        raise ValueError("exterior image must be uint8 RGB")
    if joints.shape != (7,) or gripper.shape != (1,):
        raise ValueError("DROID state requires seven joints and one gripper value")

    padding = np.zeros_like(base_image)
    result: dict[str, Any] = {
        "state": np.concatenate([joints, gripper]),
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": padding,
            "right_wrist_0_rgb": padding.copy(),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.False_,
            "right_wrist_0_rgb": np.False_,
        },
    }
    if "prompt" in data:
        result["prompt"] = data["prompt"]
    return result
