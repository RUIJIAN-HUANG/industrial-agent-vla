"""Pure keyboard-to-action mapping for the Isaac Sim teleoperation smoke.

The module intentionally has no Isaac Sim imports so the frozen 7-D action
order can be unit-tested on Windows before the Linux simulator is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Literal

from industrial_agent.contracts import ActionStep

CommandKind = Literal["action", "reset", "checkpoint", "help", "quit"]


@dataclass(frozen=True)
class TeleopCommand:
    kind: CommandKind
    key: str
    description: str
    action: ActionStep | None = None


class KeyboardTeleopMapper:
    """Map one terminal key to one small canonical 100 ms action."""

    def __init__(
        self,
        *,
        translation_step_m: float = 0.005,
        rotation_step_rad: float = radians(2.0),
        duration_ms: int = 100,
        gripper_open: bool = True,
    ) -> None:
        if translation_step_m <= 0.0 or rotation_step_rad <= 0.0:
            raise ValueError("teleop steps must be positive")
        self.translation_step_m = float(translation_step_m)
        self.rotation_step_rad = float(rotation_step_rad)
        self.duration_ms = int(duration_ms)
        self.gripper_open = bool(gripper_open)

    def set_gripper_open(self, is_open: bool) -> None:
        self.gripper_open = bool(is_open)

    def _action(
        self,
        key: str,
        description: str,
        axis: int | None = None,
        sign: float = 0.0,
    ) -> TeleopCommand:
        values = [0.0] * 6
        if axis is not None:
            step = (
                self.translation_step_m
                if axis < 3
                else self.rotation_step_rad
            )
            values[axis] = sign * step
        values.append(1.0 if self.gripper_open else 0.0)
        return TeleopCommand(
            kind="action",
            key=key,
            description=description,
            action=ActionStep.from_sequence(values, duration_ms=self.duration_ms),
        )

    def parse(self, raw_key: str) -> TeleopCommand:
        key = raw_key.strip().lower()
        motion = {
            "w": (0, 1.0, "+X"),
            "s": (0, -1.0, "-X"),
            "a": (1, 1.0, "+Y"),
            "d": (1, -1.0, "-Y"),
            "q": (2, 1.0, "+Z"),
            "e": (2, -1.0, "-Z"),
            "i": (3, 1.0, "+axis-angle X"),
            "k": (3, -1.0, "-axis-angle X"),
            "j": (4, 1.0, "+axis-angle Y"),
            "l": (4, -1.0, "-axis-angle Y"),
            "u": (5, 1.0, "+axis-angle Z"),
            "o": (5, -1.0, "-axis-angle Z"),
        }
        if key in motion:
            axis, sign, description = motion[key]
            return self._action(key, description, axis, sign)
        if key == "g":
            self.gripper_open = not self.gripper_open
            state = "open" if self.gripper_open else "close"
            return self._action(key, f"gripper {state}")
        if key == "r":
            return TeleopCommand("reset", key, "reset the scene")
        if key in {"space", "p"}:
            return TeleopCommand("checkpoint", key, "write a smoke checkpoint")
        if key in {"h", "help", "?"}:
            return TeleopCommand("help", key, "show keyboard help")
        if key in {"x", "esc", "quit", "exit"}:
            return TeleopCommand("quit", key, "safe-stop and quit")
        raise ValueError(f"unknown teleop key: {raw_key!r}")

    def help_text(self) -> str:
        return (
            "W/S: +/-X, A/D: +/-Y, Q/E: +/-Z; "
            "I/K: +/-rotX, J/L: +/-rotY, U/O: +/-rotZ; "
            "G: gripper, R: reset, P or SPACE: checkpoint, X: quit"
        )
