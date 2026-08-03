"""Isaac app-window keyboard events for step-wise teleoperation.

Isaac modules are imported only by :meth:`IsaacGuiKeyboardSource.from_isaac`
after ``SimulationApp`` startup.  The callback only enqueues normalized keys;
all controller and stage work remains on the existing owner-thread loop.
"""

from __future__ import annotations

from queue import Queue
from typing import Any


GUI_KEY_TO_TELEOP_KEY = {
    "W": "w",
    "S": "s",
    "A": "a",
    "D": "d",
    "Q": "q",
    "E": "e",
    "I": "i",
    "K": "k",
    "J": "j",
    "L": "l",
    "U": "u",
    "O": "o",
    "G": "g",
    "R": "r",
    "P": "p",
    "H": "h",
    "SPACE": "space",
    "ESCAPE": "x",
    "ESC": "x",
    "X": "x",
}


def normalize_gui_key_name(value: Any) -> str | None:
    """Return one teleop command for an Isaac ``KeyboardInput`` value."""

    raw_name = getattr(value, "name", None)
    if not isinstance(raw_name, str):
        raw_name = str(value)
    name = raw_name.rsplit(".", 1)[-1].strip().upper()
    return GUI_KEY_TO_TELEOP_KEY.get(name)


class IsaacGuiKeyboardSource:
    """Subscribe once to KEY_PRESS and enqueue commands without auto-repeat."""

    def __init__(
        self,
        *,
        output: Queue[str],
        input_interface: Any,
        keyboard: Any,
        key_press_type: Any,
    ) -> None:
        self._output = output
        self._input_interface = input_interface
        self._keyboard = keyboard
        self._key_press_type = key_press_type
        self._subscription: Any | None = None

    @classmethod
    def from_isaac(cls, output: Queue[str]) -> IsaacGuiKeyboardSource:
        """Build from the active Isaac Sim application window."""

        import carb.input  # type: ignore[import-not-found]
        import omni.appwindow  # type: ignore[import-not-found]

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Isaac default app window is unavailable")
        keyboard = app_window.get_keyboard()
        if keyboard is None:
            raise RuntimeError("Isaac app-window keyboard is unavailable")
        return cls(
            output=output,
            input_interface=carb.input.acquire_input_interface(),
            keyboard=keyboard,
            key_press_type=carb.input.KeyboardEventType.KEY_PRESS,
        )

    def start(self) -> None:
        if self._subscription is not None:
            raise RuntimeError("GUI keyboard source is already started")
        self._subscription = self._input_interface.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        if self._subscription is None:
            raise RuntimeError("Isaac keyboard subscription failed")

    def _on_keyboard_event(self, event: Any) -> bool:
        if getattr(event, "type", None) != self._key_press_type:
            return True
        command_key = normalize_gui_key_name(getattr(event, "input", None))
        if command_key is not None:
            self._output.put(command_key)
        return True

    def close(self) -> None:
        subscription, self._subscription = self._subscription, None
        if subscription is not None:
            self._input_interface.unsubscribe_to_keyboard_events(
                self._keyboard,
                subscription,
            )
