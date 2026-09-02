from simulation.run_v2_stability_acceptance import (
    _expected_positions,
    _snapshot_errors,
)


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
        "/World/Parts/P01": [0.0, 0.0, 0.8],
        "/World/Parts/P02": [0.0, 0.0, 0.8],
    }
    errors = _snapshot_errors(
        {
            "/World/Parts/P01": [0.2, 0.0, 0.8],
            "/World/Parts/P02": [float("nan"), 0.0, 0.8],
        },
        expected,
    )
    assert any("drifted" in error for error in errors)
    assert any("invalid coordinates" in error for error in errors)


def test_snapshot_gate_accepts_small_settling_motion() -> None:
    expected = {"/World/Bins/Bin_01": [0.0, 0.0, 0.8]}
    assert not _snapshot_errors(
        {"/World/Bins/Bin_01": [0.001, -0.001, 0.799]}, expected
    )
