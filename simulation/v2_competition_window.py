"""Isaac Sim ``omni.ui`` window for formal V2 competition tasks."""

from __future__ import annotations

import os
from pathlib import Path

from simulation.v2_competition_controller import (
    CompetitionController,
    FORMAL_COMPETITION_TASKS,
    UiRunState,
)


def _ready_text(ready: bool) -> str:
    return "● 已就绪" if ready else "● 未就绪"


def _extension_font_candidates(extension_root: Path) -> tuple[Path, ...]:
    """Find supported CJK font files below one Kit language extension."""

    if not extension_root.is_dir():
        return ()
    try:
        paths = [
            path
            for path in extension_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".ttf", ".otf"}
            and any(
                marker in path.name.lower()
                for marker in ("cjk", "chinese", "simplified", "sc", "han")
            )
        ]
        return tuple(
            sorted(
                paths,
                key=lambda path: (
                    "regular" not in path.name.lower(),
                    path.name.lower(),
                ),
            )
        )
    except OSError:
        return ()


def _cjk_font_candidates() -> tuple[Path, ...]:
    """Return portable, locally-installed CJK font candidates.

    ``omni.ui`` accepts TTF/OTF files for custom fonts.  Windows commonly
    installs Chinese fonts as TTC collections, while Ubuntu/Kit commonly uses
    Noto Sans CJK or the bundled Simplified Chinese language extension.
    """

    configured = os.environ.get("INDUSTRIAL_AGENT_UI_FONT", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(os.path.expandvars(configured)).expanduser())

    # Isaac Sim/Kit ships an optional Simplified Chinese extension.  When it
    # is enabled, use its own font so the UI does not depend on the host OS.
    try:
        import omni.kit.app  # type: ignore[import-not-found]

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        for module in ("omni.kit.language.simplified_chinese",):
            try:
                extension_path = extension_manager.get_extension_path_by_module(module)
            except (AttributeError, LookupError, RuntimeError):
                continue
            if not extension_path:
                continue
            candidates.extend(_extension_font_candidates(Path(extension_path)))
    except (ImportError, AttributeError, LookupError, OSError, RuntimeError):
        # Unit tests and non-Kit Python environments do not provide omni.
        pass

    # The language extension may be installed but not enabled yet.  Its
    # standard per-user Ubuntu location can still be inspected safely.
    extension_cache = Path.home() / ".local" / "share" / "ov" / "data" / "exts" / "v2"
    try:
        for extension_root in extension_cache.glob(
            "omni.kit.language.simplified_chinese-*"
        ):
            candidates.extend(_extension_font_candidates(extension_root))
    except OSError:
        pass

    windir = os.environ.get("WINDIR", r"C:\Windows")
    if os.name == "nt":
        candidates.extend(
            (
                Path(windir) / "Fonts" / "simhei.ttf",
                Path(windir) / "Fonts" / "simkai.ttf",
                Path(windir) / "Fonts" / "Deng.ttf",
            )
        )
    else:
        candidates.extend(
            (
                # Google/Noto standalone SC fonts are TTF/OTF and are
                # supported directly by omni.ui.
                Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Medium.otf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansMonoCJKsc-Regular.otf"),
                Path("/usr/share/fonts/OTF/NotoSansCJKsc-Regular.otf"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf"),
                Path("/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf"),
                Path("/usr/local/share/fonts/NotoSansCJKsc-Regular.otf"),
                Path.home() / ".local" / "share" / "fonts" / "NotoSansSC-Regular.ttf",
                Path.home() / ".fonts" / "NotoSansSC-Regular.ttf",
            )
        )
    return tuple(candidates)


def _resolve_cjk_font() -> str | None:
    """Resolve a font file that contains the Simplified Chinese glyphs."""

    for candidate in _cjk_font_candidates():
        if candidate.is_file() and candidate.suffix.lower() in {".ttf", ".otf"}:
            return str(candidate)
    return None


def _ui_styles(font_path: str | None) -> dict[str, dict[str, str]]:
    """Build a local style sheet for every text-bearing widget in the window."""

    if not font_path:
        return {}
    font_style = {"font": font_path}
    return {
        "Label": font_style,
        "Button": font_style,
        "StringField": font_style,
    }


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
        # Isaac Sim's default UI font does not necessarily contain CJK glyphs.
        # Keep the override local to this window instead of changing the
        # global Kit style used by other extensions.
        self._cjk_font = _resolve_cjk_font()
        self._styles = _ui_styles(self._cjk_font)
        self._instruction_model = ui.SimpleStringModel(
            controller.snapshot().instruction
        )
        with self._window.frame:
            # A Window frame's style is not propagated to its content.  Apply
            # the type-selector sheet to the root content container so every
            # text-bearing child receives the CJK-capable font.
            with ui.VStack(spacing=8, height=0, style=self._styles):
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
