from dataclasses import dataclass

import pytest

from simulation.bin_carry_grasp import (
    BinCarryGraspManager,
    UsdFixedJointBinCarryBackend,
    distance_m,
    follow_error_m,
)


@dataclass(frozen=True)
class _Action:
    values: list[float]


class _Backend:
    def __init__(self) -> None:
        self.tcp = {"Arm_A": [0.0, 0.0, 0.0], "Arm_B": [1.0, 0.0, 0.0]}
        self.bin = [0.0, 0.0, 0.0]
        self.handle = [0.0, 0.0, 0.0]
        self.attached_arm: str | None = None
        self.detach_calls = 0

    def tcp_position(self, arm_id: str) -> list[float]:
        return list(self.tcp[arm_id])

    def bin_position(self) -> list[float]:
        return list(self.bin)

    def handle_position(self) -> list[float]:
        return list(self.handle)

    def attach(self, arm_id: str) -> None:
        self.attached_arm = arm_id

    def detach(self) -> None:
        self.attached_arm = None
        self.detach_calls += 1


_CLOSED = _Action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_OPEN = _Action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_distance_and_follow_error_are_measured_in_metres() -> None:
    assert distance_m([0.0, 0.0, 0.0], [0.003, 0.004, 0.0]) == pytest.approx(
        0.005
    )
    assert follow_error_m(
        tcp_before=[0.0, 0.0, 0.0],
        tcp_after=[0.0, 0.0, 0.01],
        bin_before=[1.0, 0.0, 0.0],
        bin_after=[1.0, 0.0, 0.01],
    ) == pytest.approx(0.0)


def test_close_only_attaches_when_tcp_is_near_frozen_handle() -> None:
    backend = _Backend()
    backend.handle = [0.04, 0.0, 0.0]
    manager = BinCarryGraspManager(backend)

    manager.before_action(_CLOSED, arm_id="Arm_A")
    manager.after_action(_CLOSED, arm_id="Arm_A")
    assert manager.attached_arm is None

    backend.handle = [0.03, 0.0, 0.0]
    manager.before_action(_CLOSED, arm_id="Arm_A")
    manager.after_action(_CLOSED, arm_id="Arm_A")
    assert manager.attached_arm == "Arm_A"
    assert backend.attached_arm == "Arm_A"
    assert manager.diagnostics()["attach_count"] == 1


def test_attached_bin_must_follow_tcp_and_opening_detaches() -> None:
    backend = _Backend()
    manager = BinCarryGraspManager(backend)
    manager.after_action(_CLOSED, arm_id="Arm_A")

    manager.before_action(_CLOSED, arm_id="Arm_A")
    backend.tcp["Arm_A"][2] += 0.01
    backend.bin[2] += 0.01
    manager.after_action(_CLOSED, arm_id="Arm_A")
    assert manager.diagnostics()["last_follow_error_m"] == pytest.approx(0.0)

    manager.before_action(_OPEN, arm_id="Arm_A")
    manager.after_action(_OPEN, arm_id="Arm_A")
    assert manager.attached_arm is None
    assert backend.detach_calls == 1
    assert manager.diagnostics()["detach_count"] == 1


def test_attached_bin_follow_failure_stops_collection() -> None:
    backend = _Backend()
    manager = BinCarryGraspManager(backend)
    manager.after_action(_CLOSED, arm_id="Arm_A")

    manager.before_action(_CLOSED, arm_id="Arm_A")
    backend.tcp["Arm_A"][2] += 0.02
    with pytest.raises(RuntimeError, match="grasp-follow gate"):
        manager.after_action(_CLOSED, arm_id="Arm_A")


def test_other_arm_cannot_move_an_attached_bin() -> None:
    backend = _Backend()
    manager = BinCarryGraspManager(backend)
    manager.after_action(_CLOSED, arm_id="Arm_A")

    with pytest.raises(RuntimeError, match="attached to Arm_A, not Arm_B"):
        manager.after_action(_CLOSED, arm_id="Arm_B")


def test_live_prim_resolution_uses_isaac_public_utility() -> None:
    class _Prim:
        def IsValid(self) -> bool:
            return True

    class _Stage:
        def GetPrimAtPath(self, _path: object) -> None:
            raise AssertionError("cross-ABI GetPrimAtPath must not be used")

        def Traverse(self) -> None:
            raise AssertionError("cross-ABI Traverse must not be used")

    calls: list[str] = []

    def get_prim_at_path(path: str) -> _Prim:
        calls.append(path)
        return _Prim()

    backend = UsdFixedJointBinCarryBackend(
        stage=_Stage(),
        controller=object(),
        get_prim_at_path=get_prim_at_path,
        define_prim=lambda *_args, **_kwargs: None,
        delete_prim=lambda *_args, **_kwargs: None,
        set_prim_property=lambda *_args, **_kwargs: None,
        set_targets=lambda *_args, **_kwargs: None,
    )
    prim = backend._prim_at("/World/Bins/Bin_01")
    assert prim.IsValid()
    assert calls == ["/World/Bins/Bin_01"]
