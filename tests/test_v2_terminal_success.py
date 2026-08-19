import math

import pytest

from simulation.run_v2_keyboard_collection import _collect_p01_terminal_success
from simulation.offline_gt import slot_interior_bounds
from simulation.v2_terminal_success import (
    P01_MAX_VERTICAL_ERROR_RAD,
    evaluate_fresh_gt_votes,
    evaluate_p01_terminal_success,
    evaluate_terminal_hold_drift,
    vertical_error_rad,
)


class _FakeController:
    physics_tick_index = 0

    def __init__(self) -> None:
        self.actions = []

    def execute_action(self, action, *, arm_id: str) -> None:
        self.actions.append((action, arm_id))
        self.physics_tick_index += 12


class _FakeProbe:
    def world_position(self, path: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def part_vertical_error_rad(self, *, part_path: str, bin_path: str) -> float:
        return math.radians(5.0)

    def part_fully_inside_slot(
        self, *, part_path: str, bin_path: str, bin_config, slot_id: str
    ):
        return {
            "pass": True,
            "part_path": part_path,
            "bin_path": bin_path,
            "slot_id": slot_id,
        }


def _votes(*passed: bool) -> list[dict[str, object]]:
    return [
        {
            "observation_id": f"gt-{index}",
            "timestamp_s": 0.1 * (index + 1),
            "pass": value,
        }
        for index, value in enumerate(passed)
    ]


def test_vertical_error_uses_directed_axis_and_includes_boundary() -> None:
    angle = math.radians(15.0)
    measured = vertical_error_rad((math.sin(angle), 0.0, math.cos(angle)), (0, 0, 1))
    assert measured == pytest.approx(P01_MAX_VERTICAL_ERROR_RAD)
    assert vertical_error_rad((0, 0, -1), (0, 0, 1)) == pytest.approx(math.pi)


def test_vertical_error_rejects_zero_or_nonfinite_vectors() -> None:
    with pytest.raises(ValueError):
        vertical_error_rad((0, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError):
        vertical_error_rad((math.nan, 0, 1), (0, 0, 1))


def test_fresh_votes_require_exactly_three_unique_chronological_observations() -> None:
    result = evaluate_fresh_gt_votes(_votes(True, False, True))
    assert result["pass"] is True
    assert result["passed_count"] == 2

    duplicate = _votes(True, False, True)
    duplicate[2]["observation_id"] = duplicate[1]["observation_id"]
    assert evaluate_fresh_gt_votes(duplicate)["failure_codes"] == ["P01_GT_NOT_FRESH"]

    assert evaluate_fresh_gt_votes(_votes(True, True))["failure_codes"] == [
        "P01_GT_NOT_FRESH"
    ]
    assert evaluate_fresh_gt_votes(_votes(True, False, False))["failure_codes"] == [
        "P01_GT_VOTE_INSUFFICIENT"
    ]


def test_terminal_hold_requires_one_second_and_one_mm_drift() -> None:
    positions = [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]
    result = evaluate_terminal_hold_drift(positions, [0.0, 1.0])
    assert result["pass"] is True
    assert result["max_drift_m"] == pytest.approx(0.001)

    too_short = evaluate_terminal_hold_drift(positions, [0.0, 0.999])
    assert too_short["pass"] is False
    assert "P01_TERMINAL_HOLD_TOO_SHORT" in too_short["failure_codes"]

    too_far = evaluate_terminal_hold_drift([[0, 0, 0], [0.00101, 0, 0]], [0, 1])
    assert too_far["pass"] is False
    assert "P01_TERMINAL_DRIFT_EXCEEDED" in too_far["failure_codes"]


def test_all_three_gates_are_required_and_gt_is_not_canonical() -> None:
    result = evaluate_p01_terminal_success(
        orientation_error_rad=math.radians(10),
        vote_reports=_votes(True, False, True),
        positions_world=[[0, 0, 0], [0.0005, 0, 0]],
        timestamps_s=[0, 1],
    )
    assert result.passed is True
    assert result.to_dict()["canonical_included"] is False

    failed = evaluate_p01_terminal_success(
        orientation_error_rad=math.radians(16),
        vote_reports=_votes(True, True, True),
        positions_world=[[0, 0, 0], [0.0005, 0, 0]],
        timestamps_s=[0, 1],
    )
    assert failed.passed is False
    assert "P01_ORIENTATION_EXCEEDED" in failed.failure_codes


def test_s11_bounds_exclude_neighbor_slots_and_dividers() -> None:
    config = {
        "size_m": [0.3, 0.22, 0.09],
        "wall_thickness_m": 0.006,
        "divider_thickness_m": 0.004,
        "bottom_thickness_m": 0.006,
        "slots": [
            {"id": "S11", "center_local_m": [-0.1125, 0.055, 0.0]},
            {"id": "S12", "center_local_m": [-0.0375, 0.055, 0.0]},
            {"id": "S21", "center_local_m": [-0.1125, -0.055, 0.0]},
            {"id": "S22", "center_local_m": [-0.0375, -0.055, 0.0]},
        ],
    }

    bounds = slot_interior_bounds(config, "S11")

    assert bounds["min"] == pytest.approx([-0.144, 0.002, -0.039])
    assert bounds["max"] == pytest.approx([-0.077, 0.104, 0.045])


def test_terminal_collection_runs_ten_real_hold_actions_and_writes_sidecar(
    tmp_path,
) -> None:
    controller = _FakeController()
    result, report_path = _collect_p01_terminal_success(
        controller=controller,
        probe=_FakeProbe(),
        config={"scene_id": "single_bin_manual_industrial_v2", "bin": {}},
        artifact_dir=tmp_path,
    )

    assert result.passed is True
    assert len(controller.actions) == 10
    assert {arm_id for _, arm_id in controller.actions} == {"Arm_A"}
    assert (
        _FakeProbe().part_fully_inside_slot(
            part_path="/World/Parts/P01",
            bin_path="/World/Bins/Bin_01",
            bin_config={},
            slot_id="S11",
        )["slot_id"]
        == "S11"
    )
    assert report_path.is_file()
    payload = report_path.read_text(encoding="utf-8")
    assert '"canonical_included": false' in payload
