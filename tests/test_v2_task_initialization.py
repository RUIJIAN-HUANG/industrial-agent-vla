from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.v2_task_initialization import bin01_transport_initial_poses


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "simulation"
    / "configs"
    / "single_bin_scene_v2.json"
)


def test_bin01_transport_starts_full_at_handoff_center() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    poses = bin01_transport_initial_poses(config)

    assert poses["/World/Bins/Bin_01"]["position_m"] == pytest.approx(
        [0.0, 0.0, 0.795]
    )
    assert set(poses) == {
        "/World/Bins/Bin_01",
        *(
            f"/World/Parts/{part_id}"
            for part_id in config["collection"]["formal_part_order"]
        ),
    }
    assert poses["/World/Parts/P03"]["rpy_deg"] == [0.0, 0.0, 0.0]
    assert poses["/World/Parts/W01"]["rpy_deg"] == [0.0, 0.0, 90.0]
