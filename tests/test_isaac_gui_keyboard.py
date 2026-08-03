from queue import Empty, Queue
from types import SimpleNamespace

import pytest

from simulation.isaac_gui_keyboard import (
    IsaacGuiKeyboardSource,
    normalize_gui_key_name,
)


class _FakeInputInterface:
    def __init__(self) -> None:
        self.callback = None
        self.unsubscribed = None

    def subscribe_to_keyboard_events(self, keyboard, callback):
        self.callback = callback
        return "subscription-1"

    def unsubscribe_to_keyboard_events(self, keyboard, subscription):
        self.unsubscribed = (keyboard, subscription)


def test_normalize_gui_key_name_accepts_enum_name_and_string() -> None:
    assert normalize_gui_key_name(SimpleNamespace(name="W")) == "w"
    assert normalize_gui_key_name("KeyboardInput.SPACE") == "space"
    assert normalize_gui_key_name(SimpleNamespace(name="ESCAPE")) == "x"
    assert normalize_gui_key_name(SimpleNamespace(name="F1")) is None


def test_gui_source_only_enqueues_key_press_and_ignores_repeat() -> None:
    output: Queue[str] = Queue()
    interface = _FakeInputInterface()
    keyboard = object()
    source = IsaacGuiKeyboardSource(
        output=output,
        input_interface=interface,
        keyboard=keyboard,
        key_press_type="press",
    )
    source.start()

    assert interface.callback(SimpleNamespace(type="repeat", input="W")) is True
    with pytest.raises(Empty):
        output.get_nowait()
    assert (
        interface.callback(
            SimpleNamespace(type="press", input=SimpleNamespace(name="Q"))
        )
        is True
    )
    assert output.get_nowait() == "q"

    source.close()
    source.close()
    assert interface.unsubscribed == (keyboard, "subscription-1")


def test_gui_source_rejects_double_start() -> None:
    source = IsaacGuiKeyboardSource(
        output=Queue(),
        input_interface=_FakeInputInterface(),
        keyboard=object(),
        key_press_type="press",
    )
    source.start()
    with pytest.raises(RuntimeError, match="already started"):
        source.start()
    source.close()
