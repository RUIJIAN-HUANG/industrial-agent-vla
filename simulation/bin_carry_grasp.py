"""Runtime-only grasp constraint for the V2 bin carry handle.

The frozen scene remains unchanged.  A temporary fixed joint is authored only
in the live stage after a closed gripper is close enough to BIN_CARRY_TCP, and
is removed as soon as the gripper opens.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Protocol


BIN_PATH = "/World/Bins/Bin_01"
HANDLE_TCP_PATH = f"{BIN_PATH}/Carry_Handle/BIN_CARRY_TCP"
RUNTIME_SCOPE_PATH = "/World/RuntimeGrasps"
RUNTIME_JOINT_PATH = f"{RUNTIME_SCOPE_PATH}/Bin01CarryJoint"


def distance_m(left: list[float], right: list[float]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("grasp positions must be 3-D")
    return sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def follow_error_m(
    *,
    tcp_before: list[float],
    tcp_after: list[float],
    bin_before: list[float],
    bin_after: list[float],
) -> float:
    if any(len(value) != 3 for value in (tcp_before, tcp_after, bin_before, bin_after)):
        raise ValueError("follow positions must be 3-D")
    tcp_delta = [float(b) - float(a) for a, b in zip(tcp_before, tcp_after)]
    bin_delta = [float(b) - float(a) for a, b in zip(bin_before, bin_after)]
    return distance_m(tcp_delta, bin_delta)


class BinCarryBackend(Protocol):
    def tcp_position(self, arm_id: str) -> list[float]: ...

    def bin_position(self) -> list[float]: ...

    def handle_position(self) -> list[float]: ...

    def attach(self, arm_id: str) -> None: ...

    def detach(self) -> None: ...


@dataclass
class _MotionStart:
    arm_id: str
    tcp_position: list[float]
    bin_position: list[float]


class BinCarryGraspManager:
    """Gate a live fixed joint and verify that an attached bin follows the TCP."""

    def __init__(
        self,
        backend: BinCarryBackend,
        *,
        attach_tolerance_m: float = 0.035,
        follow_tolerance_m: float = 0.012,
        minimum_probe_motion_m: float = 0.003,
    ) -> None:
        if not 0.0 < attach_tolerance_m <= 0.05:
            raise ValueError("attach_tolerance_m must be in (0, 0.05]")
        if not 0.0 < follow_tolerance_m <= 0.02:
            raise ValueError("follow_tolerance_m must be in (0, 0.02]")
        if minimum_probe_motion_m <= 0.0:
            raise ValueError("minimum_probe_motion_m must be positive")
        self._backend = backend
        self._attach_tolerance_m = float(attach_tolerance_m)
        self._follow_tolerance_m = float(follow_tolerance_m)
        self._minimum_probe_motion_m = float(minimum_probe_motion_m)
        self._attached_arm: str | None = None
        self._motion_start: _MotionStart | None = None
        self._last_attach_distance_m: float | None = None
        self._last_follow_error_m: float | None = None
        self._attach_count = 0
        self._detach_count = 0

    @property
    def attached_arm(self) -> str | None:
        return self._attached_arm

    def before_action(self, action: Any, *, arm_id: str) -> None:
        gripper_open = float(action.values[6]) >= 0.5
        self._motion_start = None
        if self._attached_arm == arm_id and not gripper_open:
            self._motion_start = _MotionStart(
                arm_id=arm_id,
                tcp_position=self._backend.tcp_position(arm_id),
                bin_position=self._backend.bin_position(),
            )

    def after_action(self, action: Any, *, arm_id: str) -> None:
        gripper_open = float(action.values[6]) >= 0.5
        if gripper_open:
            if self._attached_arm is not None:
                self._backend.detach()
                self._attached_arm = None
                self._detach_count += 1
            self._motion_start = None
            return

        if self._attached_arm is None:
            attach_distance = distance_m(
                self._backend.tcp_position(arm_id),
                self._backend.handle_position(),
            )
            self._last_attach_distance_m = attach_distance
            if attach_distance <= self._attach_tolerance_m:
                self._backend.attach(arm_id)
                self._attached_arm = arm_id
                self._attach_count += 1
            self._motion_start = None
            return

        if self._attached_arm != arm_id:
            raise RuntimeError(
                f"Bin_01 is attached to {self._attached_arm}, not {arm_id}"
            )
        start = self._motion_start
        self._motion_start = None
        if start is None:
            return
        tcp_after = self._backend.tcp_position(arm_id)
        bin_after = self._backend.bin_position()
        tcp_motion = distance_m(start.tcp_position, tcp_after)
        if tcp_motion < self._minimum_probe_motion_m:
            return
        error = follow_error_m(
            tcp_before=start.tcp_position,
            tcp_after=tcp_after,
            bin_before=start.bin_position,
            bin_after=bin_after,
        )
        self._last_follow_error_m = error
        if error > self._follow_tolerance_m:
            raise RuntimeError(
                "Bin_01 failed the attached grasp-follow gate: "
                f"error {error:.6f} m exceeds {self._follow_tolerance_m:.6f} m"
            )

    def detach(self) -> None:
        if self._attached_arm is not None:
            self._backend.detach()
            self._attached_arm = None
            self._detach_count += 1
        self._motion_start = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "attached_arm": self._attached_arm,
            "attach_tolerance_m": self._attach_tolerance_m,
            "follow_tolerance_m": self._follow_tolerance_m,
            "last_attach_distance_m": self._last_attach_distance_m,
            "last_follow_error_m": self._last_follow_error_m,
            "attach_count": self._attach_count,
            "detach_count": self._detach_count,
        }


class UsdFixedJointBinCarryBackend:
    """Author a temporary fixed joint while preserving the current world poses."""

    def __init__(
        self,
        *,
        stage: Any,
        controller: Any,
        get_prim_at_path: Any | None = None,
        define_prim: Any | None = None,
        delete_prim: Any | None = None,
        set_prim_property: Any | None = None,
        set_targets: Any | None = None,
    ) -> None:
        if any(
            helper is None
            for helper in (
                get_prim_at_path,
                define_prim,
                delete_prim,
                set_prim_property,
                set_targets,
            )
        ):
            from isaacsim.core.utils.prims import (
                define_prim as isaac_define_prim,
                delete_prim as isaac_delete_prim,
                get_prim_at_path as isaac_get_prim_at_path,
                set_prim_property as isaac_set_prim_property,
                set_targets as isaac_set_targets,
            )

            get_prim_at_path = get_prim_at_path or isaac_get_prim_at_path
            define_prim = define_prim or isaac_define_prim
            delete_prim = delete_prim or isaac_delete_prim
            set_prim_property = set_prim_property or isaac_set_prim_property
            set_targets = set_targets or isaac_set_targets
        self._stage = stage
        self._controller = controller
        self._get_prim_at_path = get_prim_at_path
        self._define_prim = define_prim
        self._delete_prim = delete_prim
        self._set_prim_property = set_prim_property
        self._set_targets = set_targets
        self._hand_paths: dict[str, str] = {}

    def _prim_at(self, path: str) -> Any:
        """Resolve a live prim without calling ABI-sensitive UsdStage methods."""

        prim = self._get_prim_at_path(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"live grasp prim is missing: {path}")
        return prim

    def _prim_exists(self, path: str) -> bool:
        prim = self._get_prim_at_path(path)
        return bool(prim and prim.IsValid())

    def _world_matrix(self, path: Any) -> Any:
        from pxr import Usd, UsdGeom

        prim = self._prim_at(path)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(prim)

    def _world_position(self, path: Any) -> list[float]:
        value = self._world_matrix(path).ExtractTranslation()
        return [float(value[0]), float(value[1]), float(value[2])]

    def tcp_position(self, arm_id: str) -> list[float]:
        position, _ = self._controller.end_effector_pose(arm_id)
        return [float(value) for value in position]

    def bin_position(self) -> list[float]:
        return self._world_position(BIN_PATH)

    def handle_position(self) -> list[float]:
        return self._world_position(HANDLE_TCP_PATH)

    def _hand_path(self, arm_id: str) -> str:
        cached = self._hand_paths.get(arm_id)
        if cached is not None:
            return cached
        candidates = (
            f"/World/Robots/{arm_id}/panda_hand",
            f"/World/Robots/{arm_id}/franka/panda_hand",
            f"/World/Robots/{arm_id}/panda_link0/panda_hand",
        )
        for path in candidates:
            if self._prim_exists(path):
                self._hand_paths[arm_id] = path
                return path
        raise RuntimeError(
            f"could not resolve panda_hand for {arm_id}; checked {list(candidates)}"
        )

    @staticmethod
    def _quatf(value: Any) -> Any:
        from pxr import Gf

        imaginary = value.GetImaginary()
        return Gf.Quatf(
            float(value.GetReal()),
            Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
        )

    def attach(self, arm_id: str) -> None:
        from pxr import Gf

        if self._prim_exists(RUNTIME_JOINT_PATH):
            raise RuntimeError("Bin_01 runtime grasp joint already exists")
        hand_path = self._hand_path(arm_id)
        self._prim_at(BIN_PATH)
        hand_world = self._world_matrix(hand_path)
        bin_world = self._world_matrix(BIN_PATH)
        bin_in_hand = bin_world * hand_world.GetInverse()
        relative = Gf.Transform(bin_in_hand)
        translation = relative.GetTranslation()
        rotation = relative.GetRotation().GetQuat()

        if not self._prim_exists(RUNTIME_SCOPE_PATH):
            self._define_prim(RUNTIME_SCOPE_PATH, prim_type="Scope")
        joint_prim = self._define_prim(
            RUNTIME_JOINT_PATH, prim_type="PhysicsFixedJoint"
        )
        self._set_targets(joint_prim, "physics:body0", [hand_path])
        self._set_targets(joint_prim, "physics:body1", [BIN_PATH])
        self._set_prim_property(
            RUNTIME_JOINT_PATH,
            "physics:localPos0",
            Gf.Vec3f(
                float(translation[0]), float(translation[1]), float(translation[2])
            ),
        )
        self._set_prim_property(
            RUNTIME_JOINT_PATH, "physics:localRot0", self._quatf(rotation)
        )
        self._set_prim_property(
            RUNTIME_JOINT_PATH, "physics:localPos1", Gf.Vec3f(0.0, 0.0, 0.0)
        )
        self._set_prim_property(
            RUNTIME_JOINT_PATH,
            "physics:localRot1",
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)),
        )

    def detach(self) -> None:
        if self._prim_exists(RUNTIME_JOINT_PATH):
            self._delete_prim(RUNTIME_JOINT_PATH)
