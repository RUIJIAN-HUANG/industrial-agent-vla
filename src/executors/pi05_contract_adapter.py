"""π0.5 体系B 契约适配器（叠加层）。

把 src/industrial_agent/executor.py 的 Executor Protocol（体系B）翻译为
src/executors/pi05.py 的 Pi05Executor.infer()（体系A），再把 CanonicalActionChunk
包成体系B 的 ActionChunk。不修改任何冻结文件；结构子类型实现，不显式继承。

方案书出处：interface-contracts.md §4/§7；agent-framework.md §9 统一 7 维动作合同。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.contracts.observation import ObsPacket
from src.executors.pi05 import Pi05Executor
from src.industrial_agent.contracts import ActionChunk, Observation, TaskSchema
from src.industrial_agent.errors import ExecutorError, FailureCode
from src.industrial_agent.executor import ExecutionContext, ExecutorDescriptor


def _decode_image(raw: Any) -> np.ndarray | None:
    """容错解码图像为 numpy uint8[H,W,3]；无法识别返回 None。

    支持：numpy 数组直通；dict 内嵌 numpy 或 base64 data；bytes/str 走 base64+PIL。
    """
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw
    if isinstance(raw, Mapping):
        for key in ("data", "array", "image"):
            child = raw.get(key)
            if isinstance(child, np.ndarray):
                return child
        raw = raw.get("data") or raw.get("b64")
        if raw is None:
            return None
    if isinstance(raw, (bytes, bytearray, str)):
        try:
            import base64
            import io
            from PIL import Image  # type: ignore

            data = raw.encode("ascii") if isinstance(raw, str) else bytes(raw)
            try:
                data = base64.b64decode(data, validate=True)
            except Exception:
                pass  # 可能本身就是 raw image bytes
            return np.array(
                Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8
            )
        except Exception:
            return None
    return None


class Pi05ContractAdapter:
    """体系B Executor Protocol 的 π0.5 适配实现（结构子类型，不显式继承）。

    职责：把 (TaskSchema, Observation, ExecutionContext) 拆成 ObsPacket，
    调 Pi05Executor.infer()，再把 CanonicalActionChunk 包成 ActionChunk。
    """

    def __init__(self, executor: "Pi05Executor | None" = None) -> None:
        self._executor: Pi05Executor = executor or Pi05Executor()

    @property
    def descriptor(self) -> ExecutorDescriptor:
        return self._executor.descriptor

    def health(self) -> bool:
        try:
            info = self._executor.health_check()
            return info.get("mode") is not None
        except Exception:
            return False

    def plan(
        self, task: TaskSchema, observation: Observation, context: ExecutionContext
    ) -> ActionChunk:
        camera = observation.data.get("camera", {})
        robot = observation.data.get("robot", {})
        if not isinstance(camera, Mapping):
            camera = {}
        if not isinstance(robot, Mapping):
            robot = {}

        rgb_front = _decode_image(camera.get("full_image"))
        wrist_raw = camera.get("wrist_image")
        rgb_wrist = _decode_image(wrist_raw) if wrist_raw is not None else None

        robot_raw = robot.get("state", robot.get("tcp_pose_m_rad"))
        robot_state = (
            np.asarray(robot_raw, dtype=np.float32)
            if robot_raw is not None
            else np.zeros(0, dtype=np.float32)
        )

        obs_packet = ObsPacket(
            episode_id=context.run_id,
            step_id=context.step_id,
            timestamp_ns=observation.timestamp_ms * 1_000_000,
            rgb_front=rgb_front,  # type: ignore[arg-type]
            rgb_wrist=rgb_wrist,
            robot_state=robot_state,
            instruction=task.instruction,
            runtime_flags={},
        )
        try:
            canonical = self._executor.infer(obs_packet)
        except Exception as exc:
            # FailureCode 无 EXECUTOR_INFERENCE_FAILED；EXECUTOR_RUNTIME 对应模型运行错误
            raise ExecutorError(
                FailureCode.EXECUTOR_RUNTIME,
                f"π0.5 inference failed: {exc}",
                retryable=True,
            ) from exc
        return self._executor.to_action_chunk(canonical, task.task_id, "pi05")

    def cancel(self, task_id: str, reason: str) -> None:
        try:
            self._executor.cancel_pending_chunk()
        except Exception:
            pass
