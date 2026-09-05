"""Isaac Sim ``omni.ui`` window for formal V2 competition tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from simulation.v2_competition_controller import (
    CompetitionController,
    FORMAL_COMPETITION_TASKS,
    CompetitionSnapshot,
    UiRunState,
    UiYoloDetection,
)


def _ready_text(ready: bool) -> str:
    return "● 已就绪" if ready else "● 未就绪"


def _yolo_detection_text(snapshot: CompetitionSnapshot) -> str:
    """Format the latest validated detector packet for the visible UI."""

    if not snapshot.yolo_observation_id:
        return "YOLO检测结果：尚未产生检测结果"
    header = (
        f"YOLO检测结果：{len(snapshot.yolo_detections)} 个目标"
        f"｜摄像头：{snapshot.yolo_camera_id}"
        f"｜图像：{snapshot.yolo_image_width}×{snapshot.yolo_image_height}"
    )
    if not snapshot.yolo_detections:
        return f"{header}\n当前帧未检测到目标"

    lines = [header]
    for index, detection in enumerate(snapshot.yolo_detections[:20], start=1):
        x_min, y_min, x_max, y_max = detection.bbox_xyxy
        lines.append(
            f"{index}. {detection.class_name}"
            f"｜置信度：{detection.confidence:.3f}"
            f"｜框：({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})"
        )
    remaining = len(snapshot.yolo_detections) - 20
    if remaining > 0:
        lines.append(f"……还有 {remaining} 个目标未展开")
    return "\n".join(lines)


def _annotate_yolo_frame(
    frame: np.ndarray,
    detections: Sequence[UiYoloDetection],
) -> np.ndarray:
    """Return an RGBA frame with every YOLO bbox outlined in red."""

    image = np.asarray(frame)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("YOLO preview frame must be uint8 RGB or RGBA")
    height, width = image.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = image[:, :, :3]
    rgba[:, :, 3] = 255

    color = np.array((255, 32, 32, 255), dtype=np.uint8)
    thickness = max(2, min(5, round(min(width, height) / 240)))
    for detection in detections:
        if detection.camera_id == "":
            continue
        scale_x = width / detection.image_width
        scale_y = height / detection.image_height
        x_min, y_min, x_max, y_max = detection.bbox_xyxy
        left = max(0, min(width - 1, round(x_min * scale_x)))
        top = max(0, min(height - 1, round(y_min * scale_y)))
        right = max(left, min(width - 1, round(x_max * scale_x) - 1))
        bottom = max(top, min(height - 1, round(y_max * scale_y) - 1))
        rgba[top : min(height, top + thickness), left : right + 1] = color
        rgba[max(0, bottom - thickness + 1) : bottom + 1, left : right + 1] = color
        rgba[top : bottom + 1, left : min(width, left + thickness)] = color
        rgba[top : bottom + 1, max(0, right - thickness + 1) : right + 1] = color
    return rgba


class V2CompetitionWindow:
    """Render controls while delegating all state changes to the controller."""

    def __init__(self, controller: CompetitionController) -> None:
        # Import only after SimulationApp has started.
        from omni import ui  # type: ignore[import-not-found]

        self._ui = ui
        self._controller = controller
        self._yolo_image_provider = ui.ByteImageProvider()
        self._latest_yolo_frame: np.ndarray | None = None
        self._yolo_frame_generation = 0
        self._rendered_yolo_frame_generation = -1
        self._yolo_image_provider.set_data_array(
            np.zeros((2, 2, 4), dtype=np.uint8),
            [2, 2],
        )
        self._window = ui.Window(
            "工业具身智能任务执行系统",
            width=720,
            height=760,
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
                ui.Label("YOLO检测画面（红框为检测结果）")
                with ui.HStack(height=220, spacing=8):
                    ui.ImageWithProvider(
                        self._yolo_image_provider,
                        width=360,
                        height=200,
                    )
                    self._yolo_detection_label = ui.Label(
                        "YOLO检测结果：尚未产生检测结果",
                        word_wrap=True,
                    )
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

    def update_yolo_frame(self, frame: np.ndarray | None) -> None:
        """Queue the newest camera frame for the next UI refresh."""

        self._yolo_frame_generation += 1
        self._latest_yolo_frame = None if frame is None else np.asarray(frame).copy()
        if frame is None:
            self._yolo_image_provider.set_data_array(
                np.zeros((2, 2, 4), dtype=np.uint8),
                [2, 2],
            )
            self._rendered_yolo_frame_generation = self._yolo_frame_generation

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
        self._yolo_detection_label.text = _yolo_detection_text(snapshot)
        if (
            self._latest_yolo_frame is not None
            and self._rendered_yolo_frame_generation != self._yolo_frame_generation
        ):
            annotated = _annotate_yolo_frame(
                self._latest_yolo_frame,
                snapshot.yolo_detections,
            )
            self._yolo_image_provider.set_data_array(
                annotated,
                [annotated.shape[1], annotated.shape[0]],
            )
            self._rendered_yolo_frame_generation = self._yolo_frame_generation
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
