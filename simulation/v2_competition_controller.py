"""Thread-safe state and command boundary for the V2 competition window.

This module intentionally has no Isaac Sim imports.  Omni UI callbacks only
update this controller; the Isaac owner thread drains commands and performs all
scene, robot, and Supervisor operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from industrial_agent.contracts import TaskSchema
from industrial_agent.instructions import normalize_mvp_instruction
from industrial_agent.v2_task_profile import require_formal_v2_task


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CompetitionTaskOption:
    task_id: str
    display_instruction: str
    canonical_instruction: str
    task_file: Path


FORMAL_COMPETITION_TASKS = (
    CompetitionTaskOption(
        task_id="P01_TO_S11",
        display_instruction="把P01放到S11中",
        canonical_instruction="把P01放到S11中",
        task_file=REPOSITORY_ROOT / "configs" / "task.v2.p01-to-s11.example.json",
    ),
    CompetitionTaskOption(
        task_id="W01_TO_S14",
        display_instruction="把W01放到S14中",
        canonical_instruction="把W01放到S14中",
        task_file=REPOSITORY_ROOT / "configs" / "task.v2.w01-to-s14.example.json",
    ),
    CompetitionTaskOption(
        task_id="BIN01_TO_FINISHED01",
        display_instruction="请将料箱 Bin_01 搬运到成品区 FINISHED_01。",
        canonical_instruction="把Bin_01搬到FINISHED_01",
        task_file=REPOSITORY_ROOT
        / "configs"
        / "task.v2.bin01-to-finished01.example.json",
    ),
)


_TASK_BY_ID = {item.task_id: item for item in FORMAL_COMPETITION_TASKS}


class UiRunState(str, Enum):
    STARTING = "STARTING"
    WAITING_SERVICES = "WAITING_SERVICES"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    EXITING = "EXITING"


class CompetitionCommandType(str, Enum):
    START = "START"
    RESET = "RESET"
    EXIT = "EXIT"


@dataclass(frozen=True)
class CompetitionCommand:
    kind: CompetitionCommandType
    task_id: str | None = None


@dataclass(frozen=True)
class CompetitionSnapshot:
    state: UiRunState
    selected_task_id: str
    instruction: str
    pi05_ready: bool
    yolo_ready: bool
    verifier_configured: bool
    step: int
    max_steps: int
    message: str
    last_error: str
    exit_requested: bool


def task_option_for_instruction(text: str) -> CompetitionTaskOption:
    """Resolve one supported display/canonical instruction to a formal task."""

    option = normalize_mvp_instruction(text)
    require_formal_v2_task(option.task_id)
    try:
        return _TASK_BY_ID[option.task_id]
    except KeyError as exc:
        raise ValueError(
            f"task {option.task_id!r} is not enabled in the competition window"
        ) from exc


def load_competition_task(task_id: str) -> TaskSchema:
    """Load and cross-check one frozen competition task JSON."""

    import json

    try:
        option = _TASK_BY_ID[task_id]
    except KeyError as exc:
        raise ValueError(f"unsupported competition task_id: {task_id!r}") from exc
    payload = json.loads(option.task_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task JSON root must be an object: {option.task_file}")
    task = TaskSchema.from_dict(payload)
    spec = require_formal_v2_task(task.task_id)
    if task.task_id != option.task_id or task.instruction != spec.instruction:
        raise ValueError(
            f"task JSON disagrees with frozen V2 profile: {option.task_file}"
        )
    return task


class CompetitionController:
    """Own UI state while keeping callbacks independent from Isaac APIs."""

    def __init__(self, *, max_steps: int, verifier_configured: bool) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        default = FORMAL_COMPETITION_TASKS[0]
        self._lock = Lock()
        self._commands: Queue[CompetitionCommand] = Queue()
        self._stop_event = Event()
        self._state = UiRunState.STARTING
        self._selected_task_id = default.task_id
        self._instruction = default.display_instruction
        self._pi05_ready = False
        self._yolo_ready = False
        self._verifier_configured = bool(verifier_configured)
        self._step = 0
        self._max_steps = int(max_steps)
        self._message = "正在启动比赛运行环境"
        self._last_error = ""
        self._exit_requested = False

    @property
    def stop_event(self) -> Event:
        return self._stop_event

    def select_preset(self, task_id: str) -> None:
        option = _TASK_BY_ID[task_id]
        with self._lock:
            if self._state in {UiRunState.RUNNING, UiRunState.STOPPING}:
                return
            self._selected_task_id = option.task_id
            self._instruction = option.display_instruction
            self._last_error = ""

    def set_instruction(self, text: str) -> None:
        with self._lock:
            if self._state in {UiRunState.RUNNING, UiRunState.STOPPING}:
                return
            self._instruction = str(text)

    def update_service_health(self, *, pi05_ready: bool, yolo_ready: bool) -> None:
        with self._lock:
            self._pi05_ready = bool(pi05_ready)
            self._yolo_ready = bool(yolo_ready)
            if self._state in {
                UiRunState.STARTING,
                UiRunState.WAITING_SERVICES,
                UiRunState.READY,
            }:
                if self._pi05_ready and self._yolo_ready:
                    self._state = UiRunState.READY
                    self._message = "模型服务已就绪，请选择任务"
                else:
                    self._state = UiRunState.WAITING_SERVICES
                    self._message = "等待 Π0.5 和 YOLO 服务就绪"

    def request_start(self, instruction: str) -> bool:
        try:
            option = task_option_for_instruction(instruction)
        except (TypeError, ValueError) as exc:
            with self._lock:
                self._last_error = str(exc)
                self._message = "指令不在三个正式比赛任务中"
            return False
        with self._lock:
            if self._state not in {
                UiRunState.READY,
                UiRunState.SUCCEEDED,
                UiRunState.FAILED,
                UiRunState.STOPPED,
            }:
                self._last_error = f"当前状态 {self._state.value} 不允许开始任务"
                return False
            if not self._pi05_ready or not self._yolo_ready:
                self._state = UiRunState.WAITING_SERVICES
                self._last_error = "Π0.5 或 YOLO 服务尚未就绪"
                return False
            self._selected_task_id = option.task_id
            self._instruction = option.display_instruction
            self._state = UiRunState.RUNNING
            self._step = 0
            self._message = f"正在执行 {option.task_id}"
            self._last_error = ""
            self._stop_event.clear()
            self._commands.put(
                CompetitionCommand(CompetitionCommandType.START, option.task_id)
            )
            return True

    def request_safe_stop(self) -> bool:
        with self._lock:
            if self._state not in {UiRunState.RUNNING, UiRunState.STOPPING}:
                return False
            self._stop_event.set()
            self._state = UiRunState.STOPPING
            self._message = "正在执行安全停止"
            return True

    def request_reset(self) -> bool:
        with self._lock:
            if self._state in {UiRunState.RUNNING, UiRunState.STOPPING}:
                self._last_error = "任务运行中不能重置场景，请先安全停止"
                return False
            self._commands.put(CompetitionCommand(CompetitionCommandType.RESET))
            self._message = "正在重置场景"
            self._last_error = ""
            return True

    def request_exit(self) -> None:
        with self._lock:
            self._exit_requested = True
            if self._state in {UiRunState.RUNNING, UiRunState.STOPPING}:
                self._stop_event.set()
                self._state = UiRunState.STOPPING
                self._message = "正在安全停止后退出"
            else:
                self._state = UiRunState.EXITING
                self._commands.put(CompetitionCommand(CompetitionCommandType.EXIT))

    def update_progress(self, step: int, message: str | None = None) -> None:
        with self._lock:
            self._step = max(0, min(int(step), self._max_steps))
            if message:
                self._message = str(message)

    def mark_reset_complete(self) -> None:
        with self._lock:
            self._step = 0
            self._state = (
                UiRunState.READY
                if self._pi05_ready and self._yolo_ready
                else UiRunState.WAITING_SERVICES
            )
            self._message = "场景已重置"

    def mark_succeeded(self, message: str) -> None:
        with self._lock:
            self._state = UiRunState.SUCCEEDED
            self._message = message
            self._last_error = ""

    def mark_stopped(self, message: str) -> None:
        with self._lock:
            self._state = UiRunState.STOPPED
            self._message = message
            self._last_error = ""

    def mark_failed(self, message: str, *, error: str = "") -> None:
        with self._lock:
            self._state = UiRunState.FAILED
            self._message = message
            self._last_error = error or message

    def drain_commands(self) -> tuple[CompetitionCommand, ...]:
        commands: list[CompetitionCommand] = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except Empty:
                return tuple(commands)

    def snapshot(self) -> CompetitionSnapshot:
        with self._lock:
            return CompetitionSnapshot(
                state=self._state,
                selected_task_id=self._selected_task_id,
                instruction=self._instruction,
                pi05_ready=self._pi05_ready,
                yolo_ready=self._yolo_ready,
                verifier_configured=self._verifier_configured,
                step=self._step,
                max_steps=self._max_steps,
                message=self._message,
                last_error=self._last_error,
                exit_requested=self._exit_requested,
            )


__all__ = [
    "FORMAL_COMPETITION_TASKS",
    "CompetitionCommand",
    "CompetitionCommandType",
    "CompetitionController",
    "CompetitionSnapshot",
    "CompetitionTaskOption",
    "UiRunState",
    "load_competition_task",
    "task_option_for_instruction",
]
