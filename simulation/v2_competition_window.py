"""Isaac Sim ``omni.ui`` window for formal V2 competition tasks."""

from __future__ import annotations

from simulation.v2_competition_controller import (
    CompetitionController,
    FORMAL_COMPETITION_TASKS,
    UiRunState,
)


def _ready_text(ready: bool) -> str:
    return "● 已就绪" if ready else "● 未就绪"


class V2CompetitionWindow:
    """Render controls while delegating all state changes to the controller."""

    def __init__(self, controller: CompetitionController) -> None:
        # Import only after SimulationApp has started.
        from omni import ui  # type: ignore[import-not-found]

        self._ui = ui
        self._controller = controller
        self._window = ui.Window(
            "工业具身智能任务执行系统",
            width=720,
            height=430,
        )
        self._instruction_model = ui.SimpleStringModel(
            controller.snapshot().instruction
        )
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("V2 比赛任务控制台", height=28)
                with ui.HStack(height=24, spacing=12):
                    self._pi05_label = ui.Label("Π0.5：● 未就绪")
                    self._yolo_label = ui.Label("YOLO：● 未就绪")
                    self._verifier_label = ui.Label("完成判定：未配置")
                ui.Spacer(height=4)
                ui.Label("自然语言任务指令")
                self._instruction_field = ui.StringField(
                    self._instruction_model,
                    height=30,
                )
                ui.Label("正式任务快捷选择")
                with ui.HStack(height=34, spacing=6):
                    self._preset_buttons = []
                    for option in FORMAL_COMPETITION_TASKS:
                        button = ui.Button(
                            option.task_id,
                            clicked_fn=lambda task_id=option.task_id: self._preset(
                                task_id
                            ),
                        )
                        self._preset_buttons.append(button)
                with ui.HStack(height=38, spacing=8):
                    self._start_button = ui.Button("执行任务", clicked_fn=self._start)
                    self._stop_button = ui.Button(
                        "安全停止", clicked_fn=self._controller.request_safe_stop
                    )
                    self._reset_button = ui.Button(
                        "重置场景", clicked_fn=self._controller.request_reset
                    )
                    self._exit_button = ui.Button(
                        "退出", clicked_fn=self._controller.request_exit
                    )
                ui.Spacer(height=4)
                self._task_label = ui.Label("当前任务：P01_TO_S11")
                self._state_label = ui.Label("当前状态：STARTING")
                self._step_label = ui.Label("当前步骤：0 / 0")
                self._message_label = ui.Label("正在启动比赛运行环境", word_wrap=True)
                self._error_label = ui.Label("", word_wrap=True)
        self.refresh()

    @property
    def visible(self) -> bool:
        return bool(self._window.visible)

    def _preset(self, task_id: str) -> None:
        self._controller.select_preset(task_id)
        snapshot = self._controller.snapshot()
        self._instruction_model.set_value(snapshot.instruction)
        self.refresh()

    def _start(self) -> None:
        instruction = self._instruction_model.as_string
        self._controller.set_instruction(instruction)
        self._controller.request_start(instruction)
        snapshot = self._controller.snapshot()
        self._instruction_model.set_value(snapshot.instruction)
        self.refresh()

    def refresh(self) -> None:
        snapshot = self._controller.snapshot()
        self._pi05_label.text = f"Π0.5：{_ready_text(snapshot.pi05_ready)}"
        self._yolo_label.text = f"YOLO：{_ready_text(snapshot.yolo_ready)}"
        self._verifier_label.text = (
            "完成判定：已配置"
            if snapshot.verifier_configured
            else "完成判定：未配置（只能执行，不能判成功）"
        )
        self._task_label.text = f"当前任务：{snapshot.selected_task_id}"
        self._state_label.text = f"当前状态：{snapshot.state.value}"
        self._step_label.text = f"当前步骤：{snapshot.step} / {snapshot.max_steps}"
        self._message_label.text = snapshot.message
        self._error_label.text = (
            f"错误：{snapshot.last_error}" if snapshot.last_error else ""
        )
        busy = snapshot.state in {UiRunState.RUNNING, UiRunState.STOPPING}
        self._start_button.enabled = (
            not busy and snapshot.pi05_ready and snapshot.yolo_ready
        )
        self._stop_button.enabled = busy
        self._reset_button.enabled = not busy
        self._instruction_field.enabled = not busy
        for button in self._preset_buttons:
            button.enabled = not busy

    def destroy(self) -> None:
        self._window.visible = False
        self._window = None


__all__ = ["V2CompetitionWindow"]
