"""π0.5 体系B 契约适配器（叠加层）。

把 src/industrial_agent/executor.py 的 Executor Protocol（体系B）翻译为
services/pi05/src/pi05.py 的 Pi05Executor.infer()（体系A），再把 CanonicalActionChunk
包成体系B 的 ActionChunk。不修改任何冻结文件；结构子类型实现，不显式继承。

方案书出处：interface-contracts.md §4/§7；agent-framework.md §9 统一 7 维动作合同。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from services.pi05.src.observation import (
    ObsPacket,
    image_reference_to_placeholder,
    is_image_reference,
)
from services.pi05.src.pi05 import Pi05Executor
from src.industrial_agent.contracts import ActionChunk, Observation, TaskSchema
from src.industrial_agent.errors import ExecutorError, FailureCode
from src.industrial_agent.executor import ExecutionContext, ExecutorDescriptor

logger = logging.getLogger("pi05.contract_adapter")


def _decode_image(raw: Any) -> np.ndarray | None:
    """容错解码图像为 numpy uint8[H,W,3]；无法识别返回 None。

    支持：numpy 数组直通；ImageReference dict（无像素→None，调用方应提前用
    image_reference_to_placeholder 创建占位）；dict 内嵌 numpy 或 base64 data；
    bytes/str 走 base64+PIL。
    """
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw
    if isinstance(raw, Mapping):
        # ImageReference 不含原始像素，返回 None
        if is_image_reference(raw):
            return None
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
            return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
    return None


class Pi05ContractAdapter:
    """体系B Executor Protocol 的 π0.5 适配实现（结构子类型，不显式继承）。

    职责：把 (TaskSchema, Observation, ExecutionContext) 拆成 ObsPacket，
    调 Pi05Executor.infer()，再把 CanonicalActionChunk 包成 ActionChunk。
    """

    def __init__(self, executor: Pi05Executor | None = None) -> None:
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
        # 方案书 interface-contracts.md §7.3 / executor.py Pi05Adapter.plan()：
        # Pi05Adapter 从 observation.data.robot.arm_a 提取状态，
        # 从 observation.data.camera.arm_a_rgb 提取图像（camera_id=CAM_A_TOP）。
        # 框架 _phase_vla_inputs 要求 camera.arm_a_rgb 必须存在，不做 fallback。
        camera = observation.data.get("camera", {})
        robot = observation.data.get("robot", {})
        if not isinstance(camera, Mapping):
            camera = {}
        if not isinstance(robot, Mapping):
            robot = {}

        # ---- arm_a_rgb（框架固定 camera_key，不使用 camera.full_image）----
        raw_front = camera.get("arm_a_rgb")
        if raw_front is None:
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "camera.arm_a_rgb is required (Pi05Adapter._phase_vla_inputs "
                "guarantees this field; observation may be corrupted)",
            )
        if is_image_reference(raw_front):
            # ImageReference 不含原始像素 → 零图占位（dummy 模式不依赖像素内容）
            rgb_front = image_reference_to_placeholder(raw_front)
        else:
            rgb_front = _decode_image(raw_front)
        if rgb_front is None:
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "camera.arm_a_rgb could not be decoded to uint8[H,W,3]",
            )

        # ---- wrist_image（ImageReference 或 null）----
        raw_wrist = camera.get("wrist_image")
        if raw_wrist is None:
            rgb_wrist = None
        elif is_image_reference(raw_wrist):
            rgb_wrist = image_reference_to_placeholder(raw_wrist)
        else:
            rgb_wrist = _decode_image(raw_wrist)

        # ---- arm_a.state（框架 _phase_vla_inputs arm_key="arm_a"）----
        arm_a = robot.get("arm_a", {})
        if not isinstance(arm_a, Mapping):
            arm_a = {}
        robot_raw = arm_a.get("state", arm_a.get("tcp_pose_m_rad"))
        if robot_raw is None:
            logger.warning(
                "robot.arm_a 缺失 state/tcp_pose_m_rad，使用零状态占位 "
                "(task_id=%s step=%d)",
                task.task_id,
                context.step_id,
            )
        robot_state = (
            np.asarray(robot_raw, dtype=np.float32)
            if robot_raw is not None
            else np.zeros(0, dtype=np.float32)
        )

        # 优先使用 context.original_instruction（FixedDualVLAPlanner 设定的冻结指令）
        # 回退 task.instruction（方案书 executor.py Pi05Adapter.plan() L771）
        instruction = context.original_instruction or task.instruction

        obs_packet = ObsPacket(
            episode_id=context.run_id,
            step_id=context.step_id,
            timestamp_ns=observation.timestamp_ms * 1_000_000,
            rgb_front=rgb_front,  # type: ignore[arg-type]
            rgb_wrist=rgb_wrist,
            robot_state=robot_state,
            instruction=instruction,
            runtime_flags={},
        )
        try:
            canonical = self._executor.infer(obs_packet)
        except (ValueError, TypeError, KeyError) as exc:
            # 输入/参数错误（数据非法、维度不匹配等），重试无效
            raise ExecutorError(
                FailureCode.EXECUTOR_RUNTIME,
                f"π0.5 inference failed (input error): {exc}",
                retryable=False,
            ) from exc
        except Exception as exc:
            # 运行时错误（网络超时、模型 OOM 等），可重试
            raise ExecutorError(
                FailureCode.EXECUTOR_RUNTIME,
                f"π0.5 inference failed (runtime): {exc}",
                retryable=True,
            ) from exc
        return self._executor.to_action_chunk(canonical, task.task_id, "pi05")

    def cancel(self, task_id: str, reason: str) -> None:
        try:
            self._executor.cancel_pending_chunk()
        except Exception:
            pass
