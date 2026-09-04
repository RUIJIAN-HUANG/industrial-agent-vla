from simulation.run_v2_stability_acceptance import (
    _effective_reset_count,
    _expected_positions,
    _motion_between,
    _reset_metadata,
    _rpy_deg_to_quaternion_wxyz,
    _snapshot_errors,
)


def test_effective_reset_count_records_implicit_initial_reset() -> None:
    assert _effective_reset_count(0) == 1
    assert _effective_reset_count(3) == 3
    assert _reset_metadata(0, 1) == {
        "resets_requested": 0,
        "resets_completed": 1,
        "implicit_initial_reset": True,
    }
    assert _reset_metadata(3, 3) == {
        "resets_requested": 3,
        "resets_completed": 3,
        "implicit_initial_reset": False,
    }


def test_effective_reset_count_rejects_negative_values() -> None:
    try:
        _effective_reset_count(-1)
    except ValueError as exc:
        assert str(exc) == "--resets cannot be negative"
    else:
        raise AssertionError("negative reset count was accepted")


def _state(
    position: list[float],
    rpy_deg: list[float] | None = None,
) -> dict[str, list[float]]:
    return {
        "position_m": position,
        "orientation_wxyz": _rpy_deg_to_quaternion_wxyz(rpy_deg or [0.0, 0.0, 0.0]),
    }


def _stationary(*paths: str) -> dict[str, dict[str, float]]:
    return {
        path: {"linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0} for path in paths
    }


def test_expected_positions_include_all_parts_and_bin() -> None:
    config = {
        "parts": [
            {"id": "P01", "pose": {"position_m": [0.1, 0.2, 0.3]}},
            {"id": "W02", "pose": {"position_m": [0.4, 0.5, 0.6]}},
        ],
        "bin": {"pose": {"position_m": [0.0, 0.0, 0.8]}},
    }
    assert _expected_positions(config) == {
        "/World/Parts/P01": [0.1, 0.2, 0.3],
        "/World/Parts/W02": [0.4, 0.5, 0.6],
        "/World/Bins/Bin_01": [0.0, 0.0, 0.8],
    }


def test_snapshot_gate_rejects_drift_and_non_finite_values() -> None:
    expected = {
        "/World/Parts/P01": _state([0.0, 0.0, 0.8]),
        "/World/Parts/P02": _state([0.0, 0.0, 0.8]),
    }
    errors = _snapshot_errors(
        {
            "/World/Parts/P01": _state([0.2, 0.0, 0.8]),
            "/World/Parts/P02": _state([float("nan"), 0.0, 0.8]),
        },
        expected,
        _stationary(*expected),
    )
    assert any("drifted" in error for error in errors)
    assert any("invalid coordinates" in error for error in errors)


def test_snapshot_gate_accepts_small_settling_motion() -> None:
    path = "/World/Bins/Bin_01"
    expected = {path: _state([0.0, 0.0, 0.8])}
    assert not _snapshot_errors(
        {path: _state([0.001, -0.001, 0.799], [0.0, 0.0, 1.0])},
        expected,
        _stationary(path),
    )


def test_snapshot_gate_rejects_tipped_or_still_moving_bodies() -> None:
    part_path = "/World/Parts/P01"
    bin_path = "/World/Bins/Bin_01"
    expected = {
        part_path: _state([0.0, 0.0, 0.8]),
        bin_path: _state([0.0, 0.0, 0.8]),
    }
    errors = _snapshot_errors(
        {
            part_path: _state([0.0, 0.0, 0.8], [25.0, 0.0, 0.0]),
            bin_path: _state([0.0, 0.0, 0.8]),
        },
        expected,
        {
            part_path: {"linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0},
            bin_path: {"linear_speed_m_s": 0.03, "angular_speed_rad_s": 0.3},
        },
    )
    assert any("rotated" in error for error in errors)
    assert any("linear speed" in error for error in errors)
    assert any("angular speed" in error for error in errors)


def test_motion_estimate_uses_pose_delta_per_physics_step() -> None:
    path = "/World/Parts/P01"
    previous = {path: _state([0.0, 0.0, 0.8])}
    current = {path: _state([0.001, 0.0, 0.8], [0.0, 0.0, 1.0])}
    motion = _motion_between(previous, current, 0.1)[path]
    assert motion["linear_speed_m_s"] == 0.01
    assert abs(motion["angular_speed_rad_s"] - 0.17453292519943295) < 1e-12
