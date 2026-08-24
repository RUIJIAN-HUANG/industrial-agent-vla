"""Deterministic initial states for atomic V2 collection tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


BIN01_TO_FINISHED01 = "BIN01_TO_FINISHED01"


def bin01_transport_initial_poses(
    config: Mapping[str, Any],
) -> dict[str, dict[str, list[float]]]:
    """Return a full Bin_01 at HANDOFF_CENTER for the Arm_B atomic task."""

    stations = {str(item["id"]): item for item in config["stations"]}
    handoff = stations["HANDOFF_CENTER"]["pose"]
    pack = stations["PACK_STATION"]["pose"]
    bin_config = config["bin"]
    original_bin_position = [float(value) for value in bin_config["pose"]["position_m"]]
    pack_position = [float(value) for value in pack["position_m"]]
    handoff_position = [float(value) for value in handoff["position_m"]]
    bin_position = [
        handoff_position[0],
        handoff_position[1],
        handoff_position[2] + original_bin_position[2] - pack_position[2],
    ]
    poses: dict[str, dict[str, list[float]]] = {
        "/World/Bins/Bin_01": {
            "position_m": bin_position,
            "rpy_deg": [float(value) for value in handoff["rpy_deg"]],
        }
    }

    parts = {str(item["id"]): item for item in config["parts"]}
    bin_bottom = (
        bin_position[2]
        - float(bin_config["size_m"][2]) / 2.0
        + float(bin_config["bottom_thickness_m"])
    )
    for slot in bin_config["slots"]:
        part_id = str(slot["part_id"])
        part = parts[part_id]
        geometry = part["geometry"]
        part_type = str(part["part_type"])
        if part_type == "shaft":
            height = float(geometry["height_m"])
            rpy = [0.0, 0.0, 0.0]
        elif part_type == "nut":
            height = float(geometry["height_m"])
            rpy = [0.0, 0.0, float(part["pose"]["rpy_deg"][2])]
        elif part_type == "wrench":
            height = float(geometry["thickness_m"])
            rpy = [0.0, 0.0, 90.0]
        else:
            raise ValueError(f"unsupported V2 part type: {part_type}")
        center = [float(value) for value in slot["center_local_m"]]
        poses[f"/World/Parts/{part_id}"] = {
            "position_m": [
                bin_position[0] + center[0],
                bin_position[1] + center[1],
                bin_bottom + height / 2.0 + 0.001,
            ],
            "rpy_deg": rpy,
        }
    return poses


__all__ = ["BIN01_TO_FINISHED01", "bin01_transport_initial_poses"]
